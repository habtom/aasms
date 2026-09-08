import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { fetchAndStoreOpenAccessPdf } from "@/lib/api-utils/pdf-retrieval";

/**
 * POST /api/studies/[id]/sources/[sourceId]/fetch-pdf
 *
 * Retries the PDF lookup for a single source.
 */
export async function POST(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string; sourceId: string }> },
) {
  try {
    const { id: studyId, sourceId } = await params;

    const source = await prisma.source.findFirst({
      where: { id: sourceId, studyId },
      select: {
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
      },
    });

    if (!source) {
      return NextResponse.json({ error: "Source not found" }, { status: 404 });
    }

    const outcome = await fetchAndStoreOpenAccessPdf(source);

    return NextResponse.json(outcome, { status: outcome.status === "error" ? 404 : 200 });
  } catch (error) {
    console.error("[fetch-pdf] error", error);
    return NextResponse.json(
      {
        error: "PDF retrieval failed",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
}
