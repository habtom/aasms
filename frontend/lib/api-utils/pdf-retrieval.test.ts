import { describe, it, expect, vi, beforeEach } from "vitest";

const mockPrisma = vi.hoisted(() => ({
  source: {
    update: vi.fn(),
  },
}));

const mockMinio = vi.hoisted(() => ({
  uploadFile: vi.fn(),
  initializeBucket: vi.fn(),
  generateStoragePath: vi.fn(
    (studyId: string, sourceId: string, ext: string) =>
      `studies/${studyId}/sources/${sourceId}${ext}`,
  ),
}));

vi.mock("@/lib/db", () => ({ prisma: mockPrisma }));
vi.mock("@/lib/minio", () => mockMinio);

import { fetchAndStoreOpenAccessPdf, type PdfRetrievalSource, providerLabel } from "./pdf-retrieval";

const baseSource: PdfRetrievalSource = {
  id: "source-1",
  studyId: "study-1",
  title: "Cooperative Perception for CAVs",
  authors: ["Doe, J."],
  doi: "10.1145/3597503",
  originalUrl: null,
  storagePath: null,
  hasPdf: false,
  publicationDate: new Date("2024-03-01"),
  metadataExtension: { database_source: "SCOPUS" },
};

function pdfResponse(headers: Record<string, string> = {}) {
  return new Response(new Uint8Array([0x25, 0x50, 0x44, 0x46]), {
    status: 200,
    headers: { "Content-Type": "application/pdf", ...headers },
  });
}

describe("fetchAndStoreOpenAccessPdf", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockPrisma.source.update.mockResolvedValue({});
  });

  it("stores the PDF and flags the source when one is found", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      pdfResponse({
        "X-Pdf-Provider": "unpaywall",
        "X-Pdf-Source-Url": "https://repo.example.org/oa.pdf",
        "X-Pdf-License": "cc-by",
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const outcome = await fetchAndStoreOpenAccessPdf(baseSource);

    expect(outcome.status).toBe("success");
    expect(outcome.provider).toBe("unpaywall");
    expect(mockMinio.uploadFile).toHaveBeenCalledWith(
      "studies/study-1/sources/source-1.pdf",
      expect.any(Buffer),
      expect.objectContaining({ "Content-Type": "application/pdf" }),
    );

    const update = mockPrisma.source.update.mock.calls[0][0];
    expect(update.where).toEqual({ id: "source-1" });
    expect(update.data.hasPdf).toBe(true);
    expect(update.data.needsPdf).toBe(false);
    expect(update.data.storagePath).toBe("studies/study-1/sources/source-1.pdf");
    // Existing metadata survives, provenance is added next to it
    expect(update.data.metadataExtension).toMatchObject({
      database_source: "SCOPUS",
      pdfRetrieval: { status: "retrieved", provider: "unpaywall", license: "cc-by" },
    });
  });

  it("sends the metadata the python service needs to search with", async () => {
    const fetchMock = vi.fn().mockResolvedValue(pdfResponse());
    vi.stubGlobal("fetch", fetchMock);

    await fetchAndStoreOpenAccessPdf(baseSource);

    const body = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(body).toEqual({
      doi: "10.1145/3597503",
      title: "Cooperative Perception for CAVs",
      authors: ["Doe, J."],
      year: 2024,
      url: null,
    });
  });

  it("treats a missing open-access copy as skipped, not an error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({ reason: "No open-access PDF found", providers_tried: ["unpaywall"] }),
          {
            status: 404,
            headers: { "Content-Type": "application/json" },
          },
        ),
      ),
    );

    const outcome = await fetchAndStoreOpenAccessPdf(baseSource);

    expect(outcome.status).toBe("skipped");
    expect(outcome.message).toBe("No open-access PDF found");
    expect(mockMinio.uploadFile).not.toHaveBeenCalled();
    expect(mockPrisma.source.update.mock.calls[0][0].data.metadataExtension).toMatchObject({
      pdfRetrieval: { status: "not_found", providersTried: ["unpaywall"] },
    });
  });

  it("keeps the open-access links so the user can download them by hand", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            reason: "Open-access copy listed at www.mdpi.com but the host refused the download",
            providers_tried: ["openalex"],
            open_access_urls: ["https://www.mdpi.com/1/pdf"],
          }),
          { status: 404, headers: { "Content-Type": "application/json" } },
        ),
      ),
    );

    const outcome = await fetchAndStoreOpenAccessPdf(baseSource);

    expect(outcome.status).toBe("skipped");
    expect(outcome.openAccessUrls).toEqual(["https://www.mdpi.com/1/pdf"]);
    expect(mockPrisma.source.update.mock.calls[0][0].data.metadataExtension).toMatchObject({
      pdfRetrieval: { openAccessUrls: ["https://www.mdpi.com/1/pdf"] },
    });
  });

  it("skips sources that already have a stored PDF", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const outcome = await fetchAndStoreOpenAccessPdf({
      ...baseSource,
      hasPdf: true,
      storagePath: "studies/study-1/sources/source-1.pdf",
    });

    expect(outcome).toEqual({ status: "skipped", message: "PDF already stored" });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("skips sources with nothing to search on", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const outcome = await fetchAndStoreOpenAccessPdf({
      ...baseSource,
      doi: null,
      title: "",
      originalUrl: null,
    });

    expect(outcome.status).toBe("skipped");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("reports an unreachable python service without throwing", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("ECONNREFUSED")));

    const outcome = await fetchAndStoreOpenAccessPdf(baseSource);

    expect(outcome.status).toBe("error");
    expect(outcome.message).toContain("ECONNREFUSED");
    expect(mockPrisma.source.update).not.toHaveBeenCalled();
  });

  it("reports a storage failure instead of marking the source as done", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(pdfResponse()));
    mockMinio.uploadFile.mockRejectedValueOnce(new Error("MinIO down"));

    const outcome = await fetchAndStoreOpenAccessPdf(baseSource);

    expect(outcome.status).toBe("error");
    expect(outcome.message).toContain("MinIO down");
    expect(mockPrisma.source.update).not.toHaveBeenCalled();
  });

  it("rejects an empty response body", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(new Uint8Array([]), {
          status: 200,
          headers: { "Content-Type": "application/pdf" },
        }),
      ),
    );

    const outcome = await fetchAndStoreOpenAccessPdf(baseSource);

    expect(outcome.status).toBe("error");
    expect(mockMinio.uploadFile).not.toHaveBeenCalled();
  });
});

describe("providerLabel", () => {
  it("maps provider ids to display names and leaves unknown ids alone", () => {
    expect(providerLabel("elsevier")).toBe("Elsevier");
    expect(providerLabel("ieee")).toBe("IEEE Xplore");
    expect(providerLabel("acm")).toBe("ACM DL");
    expect(providerLabel("unpaywall")).toBe("Unpaywall");
    expect(providerLabel("something-new")).toBe("something-new");
  });
});
