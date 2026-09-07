"""Endpoints for fetching open-access PDFs from bibliographic metadata."""

import logging

import httpx
from fastapi import APIRouter, Response
from pydantic import BaseModel

from src.core.config import settings
from src.services.pdf_retrieval_service import PdfQuery, PdfRetrievalService

logger = logging.getLogger(__name__)

router = APIRouter()


class PdfRetrievalRequest(BaseModel):
    """Everything the caller knows about the source."""

    doi: str | None = None
    title: str | None = None
    authors: list[str] = []
    year: int | None = None
    url: str | None = None


class PdfCandidateResponse(BaseModel):
    url: str
    provider: str
    license: str | None = None
    version: str | None = None


class ResolvePdfResponse(BaseModel):
    found: bool
    candidates: list[PdfCandidateResponse] = []
    providers_tried: list[str] = []


class PdfNotFoundResponse(BaseModel):
    found: bool = False
    reason: str
    providers_tried: list[str] = []
    #: Open copies we located but could not download (publisher bot walls).
    #: Offering them as links beats making the user search again by hand.
    open_access_urls: list[str] = []


def _build_query(request: PdfRetrievalRequest) -> PdfQuery:
    return PdfQuery(
        doi=request.doi,
        title=request.title,
        authors=tuple(request.authors or ()),
        year=request.year,
        url=request.url,
    )


@router.post(
    "/retrieve-pdf",
    responses={
        200: {"content": {"application/pdf": {}}, "description": "The retrieved PDF"},
        404: {"model": PdfNotFoundResponse},
    },
)
async def retrieve_pdf(request: PdfRetrievalRequest):
    """Resolve an open-access PDF for a record and stream it back.

    Returns the raw PDF with provenance in the ``X-Pdf-*`` headers, or 404 with
    a machine-readable reason when no open copy could be found.
    """
    async with httpx.AsyncClient(
        timeout=settings.pdf_retrieval_timeout_seconds, follow_redirects=True
    ) as client:
        service = PdfRetrievalService(client)
        result = await service.retrieve(_build_query(request))

    if not result.found or not result.content:
        logger.info(
            "No open-access PDF for doi=%s title=%s (%s)",
            request.doi,
            (request.title or "")[:80],
            result.reason,
        )
        return Response(
            content=PdfNotFoundResponse(
                reason=result.reason or "No open-access PDF found",
                providers_tried=result.providers_tried,
                open_access_urls=result.open_access_urls(),
            ).model_dump_json(),
            media_type="application/json",
            status_code=404,
        )

    return Response(
        content=result.content,
        media_type="application/pdf",
        headers={
            "X-Pdf-Provider": result.provider or "",
            "X-Pdf-Source-Url": result.source_url or "",
            "X-Pdf-License": result.license or "",
        },
    )


@router.post("/resolve-pdf-url", response_model=ResolvePdfResponse)
async def resolve_pdf_url(request: PdfRetrievalRequest):
    """List the open-access PDF links for a record without downloading them."""
    async with httpx.AsyncClient(
        timeout=settings.pdf_retrieval_timeout_seconds, follow_redirects=True
    ) as client:
        service = PdfRetrievalService(client)
        candidates, providers_tried = await service.resolve_candidates(_build_query(request))

    return ResolvePdfResponse(
        found=bool(candidates),
        candidates=[
            # ``link`` rather than ``url``: publisher candidates point at an API
            # endpoint that only works with our credentials attached.
            PdfCandidateResponse(
                url=c.link, provider=c.provider, license=c.license, version=c.version
            )
            for c in candidates
        ],
        providers_tried=providers_tried,
    )
