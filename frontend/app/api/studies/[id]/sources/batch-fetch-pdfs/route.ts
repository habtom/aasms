import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { fetchAndStoreOpenAccessPdf } from "@/lib/api-utils/pdf-retrieval";

export const dynamic = "force-dynamic";

const SOURCE_SELECT = {
  id: true,
  studyId: true,
  title: true,
  authors: true,
  doi: true,
  originalUrl: true,
  storagePath: true,
  hasPdf: true,
  publicationDate: true,
  metadataExtension: true,
} as const;

/**
 * POST /api/studies/[id]/sources/batch-fetch-pdfs
 *
 * Fetches PDFs for every source that still needs one and streams
 * per-source progress as SSE. An optional `{ sourceIds: string[] }` body
 * restricts the run to a single import batch.
 *
 * Sources already screened out are left alone — chasing full texts for papers
 * that will never be classified is most of the work for none of the value.
 */
export async function POST(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  try {
    const { id: studyId } = await params;

    const body = await request.json().catch(() => ({}));
    const sourceIds: string[] | undefined = Array.isArray(body?.sourceIds)
      ? body.sourceIds
      : undefined;

    const sources = await prisma.source.findMany({
      where: {
        studyId,
        hasPdf: false,
        // finalDecision is null until a source is screened, and SQL's
        // three-valued logic drops those rows from both `NOT: { OR: [...] }`
        // and the `{ not: "EXCLUDE" }` shorthand — which silently matched
        // nothing at all. The null case has to be spelled out.
        AND: [
          { OR: [{ finalDecision: null }, { finalDecision: { not: "EXCLUDE" } }] },
          { status: { not: "EXCLUDED" } },
        ],
        ...(sourceIds ? { id: { in: sourceIds } } : {}),
      },
      select: SOURCE_SELECT,
      orderBy: { uploadedAt: "desc" },
    });

    console.log("[batch-fetch-pdfs] start", { studyId, count: sources.length });

    const encoder = new TextEncoder();
    // Set when the browser goes away, so the run stops instead of grinding
    // through hundreds of network lookups nobody is waiting for any more.
    let clientGone = false;

    const stream = new ReadableStream({
      async start(controller) {
        const sendEvent = (data: Record<string, unknown>) => {
          if (clientGone) return;
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
        };

        let successCount = 0;
        let errorCount = 0;
        let skippedCount = 0;

        try {
          for (let i = 0; i < sources.length; i++) {
            if (clientGone || controller.desiredSize === null) {
              console.log("[batch-fetch-pdfs] client disconnected, stopping", {
                studyId,
                processed: i,
                total: sources.length,
              });
              return;
            }

            const source = sources[i];

            sendEvent({
              type: "progress",
              current: i,
              total: sources.length,
              sourceId: source.id,
              sourceTitle: source.title,
            });

            const outcome = await fetchAndStoreOpenAccessPdf(source);

            if (outcome.status === "success") successCount++;
            else if (outcome.status === "skipped") skippedCount++;
            else errorCount++;

            sendEvent({
              type: "progress",
              current: i + 1,
              total: sources.length,
              sourceId: source.id,
              sourceTitle: source.title,
              status: outcome.status,
              message: outcome.message,
            });
          }

          sendEvent({
            type: "complete",
            summary: {
              total: sources.length,
              success: successCount,
              errors: errorCount,
              skipped: skippedCount,
            },
          });

          controller.close();
        } catch (error) {
          console.error("[batch-fetch-pdfs] stream error", error);
          if (!clientGone) controller.error(error);
        }
      },
      cancel() {
        clientGone = true;
      },
    });

    return new NextResponse(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  } catch (error) {
    console.error("[batch-fetch-pdfs] error", error);
    return NextResponse.json(
      {
        error: "Batch PDF retrieval failed",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
}
