"""Tests for the RQ summary service."""

import pytest

from src.core.fake_provider import FakeProvider
from src.services.rq_summary_service import RQSummaryService


@pytest.fixture
def service(fake_provider):
    return RQSummaryService(fake_provider)


@pytest.fixture
def sample_facet_data():
    return [
        {
            "id": "f1",
            "name": "Research Type",
            "type": "CLOSED",
            "research_question_ids": ["rq1"],
            "categories": [
                {"name": "Validation Research", "count": 5, "percentage": 50.0},
                {"name": "Evaluation Research", "count": 3, "percentage": 30.0},
                {"name": "Solution Proposal", "count": 2, "percentage": 20.0},
            ],
        }
    ]


@pytest.fixture
def sample_sources():
    return [
        {
            "title": "A Study on Software Architecture",
            "abstract": "This paper evaluates software architecture patterns.",
            "classifications": {"Research Type": "Validation Research"},
        },
        {
            "title": "Testing Methodologies Review",
            "abstract": "A comprehensive review of testing approaches.",
            "classifications": {"Research Type": "Evaluation Research"},
        },
    ]


@pytest.mark.asyncio
async def test_generate_summary_returns_expected_fields(service, sample_facet_data, sample_sources):
    result = await service.generate_summary(
        question="What approaches exist in this area?",
        facet_data=sample_facet_data,
        source_evidence=sample_sources,
    )

    assert "summary" in result
    assert "key_findings" in result
    assert "evidence_quality" in result
    assert "gaps" in result
    assert isinstance(result["summary"], str)
    assert isinstance(result["key_findings"], list)
    assert result["evidence_quality"] in ("strong", "moderate", "limited")


@pytest.mark.asyncio
async def test_generate_summary_with_empty_facets(service, sample_sources):
    result = await service.generate_summary(
        question="What approaches exist?",
        facet_data=[],
        source_evidence=sample_sources,
    )

    assert "summary" in result
    assert len(result["summary"]) > 0


@pytest.mark.asyncio
async def test_generate_summary_with_empty_sources(service, sample_facet_data):
    result = await service.generate_summary(
        question="What approaches exist?",
        facet_data=sample_facet_data,
        source_evidence=[],
    )

    assert "summary" in result


@pytest.mark.asyncio
async def test_generate_all_summaries(service, sample_facet_data, sample_sources):
    rqs = [
        {"id": "rq1", "question": "What approaches exist?"},
        {"id": "rq2", "question": "How are they evaluated?"},
    ]

    results = await service.generate_all_summaries(rqs, sample_facet_data, sample_sources)

    assert len(results) == 2
    assert results[0]["rq_id"] == "rq1"
    assert results[0]["question"] == "What approaches exist?"
    assert results[1]["rq_id"] == "rq2"
    assert "summary" in results[0]
    assert "key_findings" in results[0]


@pytest.mark.asyncio
async def test_generate_all_summaries_handles_error():
    provider = FakeProvider()
    provider.set_error(RuntimeError("LLM unavailable"))
    service = RQSummaryService(provider)

    rqs = [{"id": "rq1", "question": "What approaches exist?"}]

    results = await service.generate_all_summaries(rqs, [], [])

    assert len(results) == 1
    assert results[0]["rq_id"] == "rq1"
    assert "Failed to generate summary" in results[0]["summary"]
    assert results[0]["evidence_quality"] == "limited"


@pytest.mark.asyncio
async def test_generate_summary_with_custom_responses():
    provider = FakeProvider()
    provider.set_responses(
        [
            {
                "summary": "Custom summary for this RQ.",
                "key_findings": ["Finding A", "Finding B"],
                "evidence_quality": "strong",
                "gaps": ["Gap X"],
            }
        ]
    )
    service = RQSummaryService(provider)

    result = await service.generate_summary(
        question="Custom question?",
        facet_data=[],
        source_evidence=[],
    )

    assert result["summary"] == "Custom summary for this RQ."
    assert result["key_findings"] == ["Finding A", "Finding B"]
    assert result["evidence_quality"] == "strong"
    assert result["gaps"] == ["Gap X"]


@pytest.mark.asyncio
async def test_prompt_includes_question_and_facet_data(sample_facet_data, sample_sources):
    provider = FakeProvider()
    service = RQSummaryService(provider)

    prompt = service._build_prompt(
        question="What approaches exist?",
        facet_data=sample_facet_data,
        source_evidence=sample_sources,
    )

    assert "What approaches exist?" in prompt
    assert "Research Type" in prompt
    assert "Validation Research" in prompt
    assert "5 sources" in prompt
    assert "Software Architecture" in prompt


@pytest.mark.asyncio
async def test_prompt_demands_bare_json(sample_facet_data, sample_sources):
    """The Claude path drops response_schema, so the prompt must ask for JSON itself.

    Without this the model replies in Markdown, the parse yields {}, and every RQ
    lands on the "Failed to generate summary" fallback.
    """
    service = RQSummaryService(FakeProvider())

    prompt = service._build_prompt(
        question="What approaches exist?",
        facet_data=sample_facet_data,
        source_evidence=sample_sources,
    )

    assert "JSON" in prompt
    assert "no Markdown headings" in prompt
    for field in ("summary", "key_findings", "evidence_quality", "gaps"):
        assert f'"{field}"' in prompt


@pytest.mark.asyncio
async def test_facet_filtering_by_rq_id(service, sample_facet_data, sample_sources):
    """Facets not linked to a given RQ should be filtered out."""
    rqs = [
        {"id": "rq1", "question": "What approaches exist?"},
        {"id": "rq999", "question": "Unlinked question?"},
    ]

    results = await service.generate_all_summaries(rqs, sample_facet_data, sample_sources)

    assert len(results) == 2
    # Both should succeed (rq999 just gets no facet data)
    assert results[0]["rq_id"] == "rq1"
    assert results[1]["rq_id"] == "rq999"


class _SequenceProvider:
    """Replays queued structured responses, recording the budget it was given."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.max_tokens = None

    async def generate_structured_output(self, prompt, response_schema, max_tokens, **kwargs):
        self.calls += 1
        self.max_tokens = max_tokens
        return self._responses.pop(0)


_GOOD = {
    "summary": "Coordination is dominated by centralized optimization.",
    "key_findings": ["one"],
    "evidence_quality": "strong",
    "gaps": ["two"],
}


@pytest.mark.asyncio
async def test_truncated_reply_is_retried():
    provider = _SequenceProvider([{}, {}, _GOOD])

    result = await RQSummaryService(provider).generate_summary("RQ?", [], [])

    assert provider.calls == 3
    assert result["evidence_quality"] == "strong"


@pytest.mark.asyncio
async def test_no_retry_once_a_summary_arrives():
    provider = _SequenceProvider([_GOOD, _GOOD, _GOOD])

    await RQSummaryService(provider).generate_summary("RQ?", [], [])

    assert provider.calls == 1


@pytest.mark.asyncio
async def test_failed_generation_is_not_reported_as_limited_evidence():
    """A blank answer labelled "limited" reads as a real finding. It must raise."""
    provider = _SequenceProvider([{}, {}, {}])

    with pytest.raises(RuntimeError):
        await RQSummaryService(provider).generate_summary("RQ?", [], [])


@pytest.mark.asyncio
async def test_budget_allows_a_full_narrative_answer():
    provider = _SequenceProvider([_GOOD])

    await RQSummaryService(provider).generate_summary("RQ?", [], [])

    assert provider.max_tokens >= 8000


class _ByQuestionProvider:
    """Answers per question, so concurrent RQs cannot race over a shared queue."""

    def __init__(self, responses):
        self._responses = responses

    async def generate_structured_output(self, prompt, response_schema, max_tokens, **kwargs):
        for marker, response in self._responses.items():
            if marker in prompt:
                return response
        return {}


@pytest.mark.asyncio
async def test_one_failing_rq_does_not_sink_the_others():
    # RQ2 never produces a summary; the other two must still come back.
    provider = _ByQuestionProvider({"RQ1?": _GOOD, "RQ2?": {}, "RQ3?": _GOOD})
    rqs = [
        {"id": "a", "question": "RQ1?"},
        {"id": "b", "question": "RQ2?"},
        {"id": "c", "question": "RQ3?"},
    ]

    summaries = await RQSummaryService(provider).generate_all_summaries(rqs, [], [])

    assert [s["rq_id"] for s in summaries] == ["a", "b", "c"]
    assert summaries[1]["summary"].startswith("Failed to generate")
    assert summaries[2]["summary"] == _GOOD["summary"]
