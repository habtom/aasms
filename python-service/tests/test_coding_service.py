"""Tests for the OPEN-facet coding service."""

import pytest

from src.services.coding_service import CodingService


class StubProvider:
    """Records the call it received and replays a canned structured response."""

    def __init__(self, response):
        self._response = response
        self.max_tokens = None

    async def generate_structured_output(self, prompt, response_schema, max_tokens, **kwargs):
        self.max_tokens = max_tokens
        return self._response


@pytest.mark.asyncio
async def test_token_budget_scales_with_value_count():
    """The model echoes every value back, so the budget has to grow with input."""
    provider = StubProvider({"categories": [], "uncategorized": [], "reasoning": ""})
    values = [f"value {i}" for i in range(938)]

    await CodingService(provider).suggest_categories(values, "Coordination mechanism")

    assert provider.max_tokens > 4000
    assert provider.max_tokens <= 32000


@pytest.mark.asyncio
async def test_token_budget_is_capped():
    provider = StubProvider({"categories": [], "uncategorized": [], "reasoning": ""})
    values = [f"value {i}" for i in range(10000)]

    await CodingService(provider).suggest_categories(values, "Coordination mechanism")

    assert provider.max_tokens == 32000


@pytest.mark.asyncio
async def test_empty_llm_reply_leaves_every_value_assignable():
    """A truncated response parses to {} — the values must still reach the wizard."""
    provider = StubProvider({})
    values = ["consensus", "MILP", "Stackelberg game"]

    result = await CodingService(provider).suggest_categories(values, "Coordination mechanism")

    assert result["categories"] == []
    assert sorted(result["uncategorized"]) == sorted(values)


@pytest.mark.asyncio
async def test_values_dropped_by_the_model_fall_back_to_uncategorized():
    provider = StubProvider(
        {
            "categories": [
                {"name": "Optimization", "description": "solver-based", "values": ["MILP"]}
            ],
            "uncategorized": ["consensus"],
            "reasoning": "grouped by solver family",
        }
    )
    values = ["MILP", "consensus", "Stackelberg game"]

    result = await CodingService(provider).suggest_categories(values, "Coordination mechanism")

    assert result["categories"][0]["values"] == ["MILP"]
    # "Stackelberg game" was never mentioned by the model; it must not vanish.
    assert sorted(result["uncategorized"]) == ["Stackelberg game", "consensus"]


@pytest.mark.asyncio
async def test_categorized_values_are_not_duplicated_into_uncategorized():
    provider = StubProvider(
        {
            "categories": [
                {"name": "Optimization", "description": "", "values": ["MILP", "consensus"]}
            ],
            "uncategorized": [],
            "reasoning": "",
        }
    )
    values = ["MILP", "consensus"]

    result = await CodingService(provider).suggest_categories(values, "Coordination mechanism")

    assert result["uncategorized"] == []


class SequenceProvider:
    """Replays a queued list of responses, one per call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.prompts = []

    async def generate_structured_output(self, prompt, response_schema, max_tokens, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        return self._responses.pop(0)


GOOD = {
    "categories": [{"name": "Optimization", "description": "", "values": ["MILP"]}],
    "uncategorized": [],
    "reasoning": "",
}


@pytest.mark.asyncio
async def test_empty_clustering_is_retried():
    """A truncated reply is {} — indistinguishable from 'no categories'. Retry it."""
    provider = SequenceProvider([{}, {}, GOOD])

    result = await CodingService(provider).suggest_categories(["MILP"], "Coordination mechanism")

    assert provider.calls == 3
    assert len(result["categories"]) == 1


@pytest.mark.asyncio
async def test_retries_stop_once_categories_come_back():
    provider = SequenceProvider([GOOD, GOOD, GOOD])

    await CodingService(provider).suggest_categories(["MILP"], "Coordination mechanism")

    assert provider.calls == 1


@pytest.mark.asyncio
async def test_all_attempts_empty_still_yields_every_value():
    provider = SequenceProvider([{}, {}, {}])
    values = ["MILP", "consensus"]

    result = await CodingService(provider).suggest_categories(values, "Coordination mechanism")

    assert provider.calls == 3
    assert result["categories"] == []
    assert sorted(result["uncategorized"]) == sorted(values)


@pytest.mark.asyncio
async def test_values_are_sorted_in_the_prompt():
    """Near-duplicates must sit adjacently for the model to group them."""
    provider = SequenceProvider([GOOD])

    await CodingService(provider).suggest_categories(
        ["consensus", "ADMM (full name)", "ADMM"], "Coordination mechanism"
    )

    prompt = provider.prompts[0]
    assert prompt.index("- ADMM\n") < prompt.index("- ADMM (full name)") < prompt.index("- consensus")


@pytest.mark.asyncio
async def test_a_value_repeated_across_categories_is_kept_once():
    """Otherwise a paper gets filed under both categories at apply time."""
    provider = SequenceProvider(
        [
            {
                "categories": [
                    {"name": "Optimization", "description": "", "values": ["MILP", "ADMM"]},
                    {"name": "Consensus", "description": "", "values": ["ADMM", "consensus"]},
                ],
                "uncategorized": [],
                "reasoning": "",
            }
        ]
    )
    values = ["MILP", "ADMM", "consensus"]

    result = await CodingService(provider).suggest_categories(values, "Coordination mechanism")

    assert result["categories"][0]["values"] == ["MILP", "ADMM"]
    assert result["categories"][1]["values"] == ["consensus"]
    assert result["uncategorized"] == []


@pytest.mark.asyncio
async def test_counts_always_reconcile_with_the_input():
    provider = SequenceProvider(
        [
            {
                "categories": [
                    {"name": "A", "description": "", "values": ["one", "two"]},
                    {"name": "B", "description": "", "values": ["two", "three"]},
                ],
                # "one" is both categorized and listed as uncategorized.
                "uncategorized": ["one", "four"],
                "reasoning": "",
            }
        ]
    )
    values = ["one", "two", "three", "four", "five"]

    result = await CodingService(provider).suggest_categories(values, "Coordination mechanism")

    assigned = sum(len(c["values"]) for c in result["categories"])
    assert assigned + len(result["uncategorized"]) == len(values)


@pytest.mark.asyncio
async def test_values_the_model_invented_are_discarded():
    """A hallucinated value has no keyword rows behind it and cannot classify anything."""
    provider = SequenceProvider(
        [
            {
                "categories": [
                    {"name": "A", "description": "", "values": ["MILP", "not a real value"]}
                ],
                "uncategorized": ["also invented"],
                "reasoning": "",
            }
        ]
    )
    values = ["MILP", "consensus"]

    result = await CodingService(provider).suggest_categories(values, "Coordination mechanism")

    assert result["categories"][0]["values"] == ["MILP"]
    assert result["uncategorized"] == ["consensus"]


@pytest.mark.asyncio
async def test_category_left_empty_after_dedupe_is_dropped():
    provider = SequenceProvider(
        [
            {
                "categories": [
                    {"name": "A", "description": "", "values": ["MILP"]},
                    {"name": "B", "description": "", "values": ["MILP"]},
                ],
                "uncategorized": [],
                "reasoning": "",
            }
        ]
    )

    result = await CodingService(provider).suggest_categories(["MILP"], "Coordination mechanism")

    assert [c["name"] for c in result["categories"]] == ["A"]


class ScriptedProvider:
    """Replays one canned response per call, in order, recording each prompt."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.prompts = []

    async def generate_structured_output(self, prompt, response_schema, max_tokens, **kwargs):
        self.prompts.append(prompt)
        return self._responses.pop(0) if self._responses else {}


def _cats(n, per=2):
    """n suggested categories, each holding `per` distinct values."""
    return [
        {
            "name": f"Cat {i}",
            "description": f"d{i}",
            "values": [f"v{i}-{j}" for j in range(per)],
        }
        for i in range(n)
    ]


def _values(n, per=2):
    return [f"v{i}-{j}" for i in range(n) for j in range(per)]


@pytest.mark.asyncio
async def test_cap_is_not_enforced_when_the_model_stays_within_it():
    provider = ScriptedProvider({"categories": _cats(5), "uncategorized": [], "reasoning": ""})

    result = await CodingService(provider).suggest_categories(
        _values(5), "Coordination mechanism", max_categories=12
    )

    assert len(result["categories"]) == 5
    assert len(provider.prompts) == 1  # no merge pass


@pytest.mark.asyncio
async def test_overshooting_model_is_merged_back_to_the_cap():
    """Regression: a large facet came back with far more categories than the cap."""
    merge = {
        "groups": [
            {"name": "A", "description": "", "members": [f"Cat {i}" for i in range(0, 11)]},
            {"name": "B", "description": "", "members": [f"Cat {i}" for i in range(11, 21)]},
        ]
    }
    provider = ScriptedProvider(
        {"categories": _cats(21), "uncategorized": [], "reasoning": ""}, merge
    )

    result = await CodingService(provider).suggest_categories(
        _values(21), "Coordination mechanism", max_categories=12
    )

    assert [c["name"] for c in result["categories"]] == ["A", "B"]
    assert sum(len(c["values"]) for c in result["categories"]) == 42


@pytest.mark.asyncio
async def test_merge_never_loses_or_duplicates_a_value():
    values = _values(21)
    # The merge model forgets three categories entirely.
    merge = {
        "groups": [
            {"name": "A", "description": "", "members": [f"Cat {i}" for i in range(0, 9)]},
            {"name": "B", "description": "", "members": [f"Cat {i}" for i in range(9, 18)]},
        ]
    }
    provider = ScriptedProvider(
        {"categories": _cats(21), "uncategorized": [], "reasoning": ""}, merge
    )

    result = await CodingService(provider).suggest_categories(
        values, "Coordination mechanism", max_categories=12
    )

    placed = [v for c in result["categories"] for v in c["values"]] + result["uncategorized"]
    assert sorted(placed) == sorted(values)
    assert len(placed) == len(set(placed))
    assert len(result["categories"]) <= 12


@pytest.mark.asyncio
async def test_cap_holds_when_the_merge_pass_fails():
    """A failed or empty merge must still not exceed the cap."""
    provider = ScriptedProvider({"categories": _cats(21), "uncategorized": [], "reasoning": ""}, {})

    result = await CodingService(provider).suggest_categories(
        _values(21), "Coordination mechanism", max_categories=12
    )

    assert len(result["categories"]) == 12
    assert result["categories"][-1]["name"] == "Other"
    placed = [v for c in result["categories"] for v in c["values"]]
    assert sorted(placed) == sorted(_values(21))


@pytest.mark.asyncio
async def test_merge_prompt_omits_the_values():
    """Re-sending every value is what overflows the first pass; names suffice."""
    provider = ScriptedProvider({"categories": _cats(21), "uncategorized": [], "reasoning": ""}, {})

    await CodingService(provider).suggest_categories(
        _values(21), "Coordination mechanism", max_categories=12
    )

    merge_prompt = provider.prompts[1]
    assert "Cat 0" in merge_prompt
    assert "v0-0" not in merge_prompt


@pytest.mark.asyncio
async def test_prompt_states_the_cap_as_a_hard_constraint():
    provider = ScriptedProvider({"categories": [], "uncategorized": [], "reasoning": ""})

    await CodingService(provider).suggest_categories(
        ["a", "b"], "Coordination mechanism", max_categories=12
    )

    assert "at most 12 categories" in provider.prompts[0]
    assert "HARD CONSTRAINT" in provider.prompts[0]
