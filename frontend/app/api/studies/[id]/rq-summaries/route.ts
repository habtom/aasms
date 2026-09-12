import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { PYTHON_SERVICE_URL } from "@/lib/python-service";
import { Decision, SourceStatus } from "@prisma/client";
import type { RQSummaryData, RQSummaryItem } from "@/types/analysis";

const GENERATION_FAILED = "Generating summaries failed. Nothing was changed.";

// True if research questions were added, removed, reworded or reordered since generating
function questionsChanged(summaries: RQSummaryItem[], questions: { question: string }[]) {
  return (
    summaries.length !== questions.length ||
    summaries.some((s, i) => s.question.trim() !== questions[i].question.trim())
  );
}

// Returns the last saved summaries exactly as generated, or null if none were saved.
export async function GET(_request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  try {
    const { id: studyId } = await params;

    const study = await prisma.study.findUnique({
      where: { id: studyId },
      select: {
        rqSummaries: true,
        rqSummariesGeneratedAt: true,
        researchQuestions: { select: { question: true }, orderBy: { order: "asc" } },
      },
    });

    if (!study) {
      return NextResponse.json({ error: "Study not found" }, { status: 404 });
    }

    const summaries = (study.rqSummaries as { summaries?: RQSummaryItem[] } | null)?.summaries;

    if (!summaries?.length) {
      return NextResponse.json({ data: null });
    }

    const data: RQSummaryData = {
      summaries,
      generatedAt: study.rqSummariesGeneratedAt?.toISOString() ?? null,
      questionsChanged: questionsChanged(summaries, study.researchQuestions),
    };
    return NextResponse.json({ data });
  } catch (error) {
    console.error("Error fetching RQ summaries:", error);
    return NextResponse.json({ error: "Failed to fetch RQ summaries" }, { status: 500 });
  }
}

export async function POST(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  try {
    const { id: studyId } = await params;

    // Fetch study with RQs and facets
    const study = await prisma.study.findUnique({
      where: { id: studyId },
      include: {
        researchQuestions: { orderBy: { order: "asc" } },
        facets: {
          include: {
            categories: { orderBy: { order: "asc" } },
            researchQuestions: true,
          },
          orderBy: { order: "asc" },
        },
      },
    });

    if (!study) {
      return NextResponse.json({ error: "Study not found" }, { status: 404 });
    }

    if (study.researchQuestions.length === 0) {
      return NextResponse.json(
        { error: "Add research questions before generating summaries." },
        { status: 400 },
      );
    }

    // Build facet data with classification counts
    const facetData = await Promise.all(
      study.facets.map(async (facet) => {
        const classifications = await prisma.classification.findMany({
          where: {
            facetId: facet.id,
            analysis: {
              source: {
                studyId,
                status: SourceStatus.CLASSIFIED,
                finalDecision: Decision.INCLUDE,
              },
            },
          },
          include: {
            category: true,
            analysis: { select: { sourceId: true } },
          },
        });

        // Count unique sources per category
        const categoryMap = new Map<string, { name: string; sourceIds: Set<string> }>();

        if (facet.type === "CLOSED" || facet.type === "OPEN_CODED") {
          for (const cat of facet.categories) {
            categoryMap.set(cat.id, { name: cat.name, sourceIds: new Set() });
          }
          for (const c of classifications) {
            if (c.categoryId) {
              const entry = categoryMap.get(c.categoryId);
              if (entry) entry.sourceIds.add(c.analysis.sourceId);
            }
          }
        } else {
          // OPEN facets: group by value
          for (const c of classifications) {
            const val = c.value || "Unknown";
            if (!categoryMap.has(val)) {
              categoryMap.set(val, { name: val, sourceIds: new Set() });
            }
            categoryMap.get(val)!.sourceIds.add(c.analysis.sourceId);
          }
        }

        const totalSources = new Set(classifications.map((c) => c.analysis.sourceId)).size;
        const categories = Array.from(categoryMap.entries()).map(([, { name, sourceIds }]) => ({
          name,
          count: sourceIds.size,
          percentage: totalSources > 0 ? (sourceIds.size / totalSources) * 100 : 0,
        }));

        return {
          id: facet.id,
          name: facet.name,
          type: facet.type,
          research_question_ids: facet.researchQuestions.map((frq) => frq.researchQuestionId),
          categories,
        };
      }),
    );

    // Fetch included sources with classifications for evidence
    const sources = await prisma.source.findMany({
      where: {
        studyId,
        status: SourceStatus.CLASSIFIED,
        finalDecision: Decision.INCLUDE,
      },
      include: {
        analysis: {
          include: {
            classifications: {
              include: {
                facet: true,
                category: true,
              },
            },
          },
        },
      },
      take: 20,
      orderBy: { updatedAt: "desc" },
    });

    const sourceEvidence = sources.map((source) => {
      const classifications: Record<string, string> = {};
      if (source.analysis) {
        for (const c of source.analysis.classifications) {
          const facetName = c.facet.name;
          classifications[facetName] = c.category?.name || c.value || "Unknown";
        }
      }
      return {
        title: source.title,
        abstract: source.abstract || "",
        classifications,
      };
    });

    // Call Python service
    const researchQuestions = study.researchQuestions.map((rq) => ({
      id: rq.id,
      question: rq.question,
    }));

    const pythonRes = await fetch(`${PYTHON_SERVICE_URL}/api/rq-summaries`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        research_questions: researchQuestions,
        facet_data: facetData,
        source_evidence: sourceEvidence,
      }),
    });

    if (!pythonRes.ok) {
      const errorText = await pythonRes.text();
      console.error("Python service rq-summaries error:", errorText);
      return NextResponse.json({ error: GENERATION_FAILED }, { status: pythonRes.status });
    }

    const result: { summaries?: RQSummaryItem[] } = await pythonRes.json();
    const summaries = result.summaries ?? [];
    const failedCount = summaries.filter((s) => s.failed).length;

    // Only complete runs are saved, so a failure never replaces a saved summary
    if (failedCount > 0 || summaries.length !== study.researchQuestions.length) {
      console.error(
        `rq-summaries: ${failedCount} failed, ${summaries.length} returned for ${study.researchQuestions.length} RQs`,
      );
      const error =
        failedCount > 0
          ? `Generating failed for ${failedCount} of ${summaries.length} research questions. Nothing was changed.`
          : GENERATION_FAILED;
      return NextResponse.json({ error }, { status: 502 });
    }

    const now = new Date();
    let saveFailed = false;
    try {
      await prisma.study.update({
        where: { id: studyId },
        data: { rqSummaries: { summaries }, rqSummariesGeneratedAt: now },
      });
    } catch (error) {
      // Still show the summaries; the LLM calls already ran
      console.error("Error saving RQ summaries:", error);
      saveFailed = true;
    }

    const data: RQSummaryData = {
      summaries,
      generatedAt: now.toISOString(),
      questionsChanged: false,
      saveFailed,
    };
    return NextResponse.json({ data });
  } catch (error) {
    console.error("Error generating RQ summaries:", error);
    return NextResponse.json({ error: GENERATION_FAILED }, { status: 500 });
  }
}
