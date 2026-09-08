import { prisma } from "@/lib/db";
import { generateStoragePath, initializeBucket, uploadFile } from "@/lib/minio";
import { PYTHON_SERVICE_URL } from "@/lib/python-service";

/**
 * The subset of a Source needed to look up and store its PDF.
 */
export interface PdfRetrievalSource {
  id: string;
  studyId: string;
  title: string;
  authors: string[];
  doi: string | null;
  originalUrl: string | null;
  storagePath: string | null;
  hasPdf: boolean;
  publicationDate: Date | null;
  metadataExtension?: unknown;
}

export type PdfRetrievalStatus = "success" | "skipped" | "error";

/** Display names for the python service's provider ids. */
const PROVIDER_LABELS: Record<string, string> = {
  "arxiv-id": "arXiv",
  "arxiv-search": "arXiv",
  unpaywall: "Unpaywall",
  openalex: "OpenAlex",
  "semantic-scholar": "Semantic Scholar",
  elsevier: "Elsevier",
  ieee: "IEEE Xplore",
  acm: "ACM DL",
  "landing-page": "the publisher page",
};

export function providerLabel(provider: string): string {
  return PROVIDER_LABELS[provider] ?? provider;
}

export interface PdfRetrievalOutcome {
  status: PdfRetrievalStatus;
  message: string;
  provider?: string;
  sourceUrl?: string;
  license?: string;
  /** Open copies we found but could not download — offered to the user as links. */
  openAccessUrls?: string[];
}

/** Provenance we keep on the source so the search protocol can report it. */
interface PdfRetrievalRecord {
  status: "retrieved" | "not_found" | "error";
  provider?: string;
  url?: string;
  license?: string;
  reason?: string;
  providersTried?: string[];
  /** Where a human can still get the paper when the host blocked us. */
  openAccessUrls?: string[];
  attemptedAt: string;
}

function mergeRetrievalRecord(
  metadataExtension: unknown,
  record: PdfRetrievalRecord,
): Record<string, unknown> {
  const base =
    metadataExtension && typeof metadataExtension === "object" && !Array.isArray(metadataExtension)
      ? (metadataExtension as Record<string, unknown>)
      : {};
  return { ...base, pdfRetrieval: record };
}

/**
 * Look up a PDF for a source (open access, or via the configured publisher
 * credentials), store it in MinIO and flag the source.
 *
 * Never throws — batch callers rely on the outcome to keep going.
 */
export async function fetchAndStoreOpenAccessPdf(
  source: PdfRetrievalSource,
): Promise<PdfRetrievalOutcome> {
  if (source.hasPdf && source.storagePath) {
    return { status: "skipped", message: "PDF already stored" };
  }

  if (!source.doi && !source.title && !source.originalUrl) {
    return { status: "skipped", message: "No DOI, title or URL to search with" };
  }

  let response: Response;
  try {
    response = await fetch(`${PYTHON_SERVICE_URL}/api/retrieve-pdf`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        doi: source.doi,
        title: source.title,
        authors: source.authors,
        year: source.publicationDate ? source.publicationDate.getFullYear() : null,
        url: source.originalUrl,
      }),
    });
  } catch (error) {
    return {
      status: "error",
      message: `PDF service unreachable: ${error instanceof Error ? error.message : "unknown error"}`,
    };
  }

  // Most papers in a mapping study are paywalled, so "no open copy" is the
  // expected outcome rather than a failure — the source just stays in
  // "Needs PDF" for manual upload. Only real faults are reported as errors.
  if (response.status === 404) {
    const detail = (await response.json().catch(() => ({}))) as {
      reason?: string;
      providers_tried?: string[];
      open_access_urls?: string[];
    };
    const reason = detail.reason || "No downloadable copy found";
    const openAccessUrls = detail.open_access_urls || [];
    await recordAttempt(source, {
      status: "not_found",
      reason,
      providersTried: detail.providers_tried,
      openAccessUrls: openAccessUrls.length ? openAccessUrls : undefined,
      attemptedAt: new Date().toISOString(),
    });
    return { status: "skipped", message: reason, openAccessUrls };
  }

  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    return {
      status: "error",
      message: `PDF service failed (${response.status}): ${detail.slice(0, 200)}`,
    };
  }

  const provider = response.headers.get("x-pdf-provider") || "unknown";
  const sourceUrl = response.headers.get("x-pdf-source-url") || undefined;
  const license = response.headers.get("x-pdf-license") || undefined;

  const buffer = Buffer.from(await response.arrayBuffer());
  if (buffer.length === 0) {
    return { status: "error", message: "PDF service returned an empty file" };
  }

  try {
    await initializeBucket();
    const objectPath = generateStoragePath(source.studyId, source.id, ".pdf");
    await uploadFile(objectPath, buffer, {
      "Content-Type": "application/pdf",
      "Original-Filename": `${source.id}.pdf`,
      "Pdf-Provider": provider,
    });

    await prisma.source.update({
      where: { id: source.id },
      data: {
        storagePath: objectPath,
        hasPdf: true,
        needsPdf: false,
        metadataExtension: mergeRetrievalRecord(source.metadataExtension, {
          status: "retrieved",
          provider,
          url: sourceUrl,
          license,
          attemptedAt: new Date().toISOString(),
        }) as never,
      },
    });
  } catch (error) {
    return {
      status: "error",
      message: `Failed to store PDF: ${error instanceof Error ? error.message : "unknown error"}`,
    };
  }

  return {
    status: "success",
    message: `Retrieved via ${providerLabel(provider)}${license ? ` (${license})` : ""}`,
    provider,
    sourceUrl,
    license,
  };
}

/** Persist a failed lookup so repeated runs and the report can see what happened. */
async function recordAttempt(
  source: PdfRetrievalSource,
  record: PdfRetrievalRecord,
): Promise<void> {
  try {
    await prisma.source.update({
      where: { id: source.id },
      data: {
        metadataExtension: mergeRetrievalRecord(source.metadataExtension, record) as never,
      },
    });
  } catch (error) {
    console.error("[pdf-retrieval] failed to record attempt", source.id, error);
  }
}
