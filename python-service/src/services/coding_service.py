"""Coding service for generating categories from OPEN facet values."""

import logging
from typing import Any

from src.core.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

# How many times to ask for a clustering before giving up and letting the
# caller hand every value to the user unassigned.
_SUGGEST_ATTEMPTS = 3


class CodingService:
    """Service for LLM-assisted coding of OPEN facet values into categories."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    async def suggest_categories(
        self,
        values: list[str],
        facet_name: str,
        facet_description: str | None = None,
        min_categories: int = 3,
        max_categories: int = 12,
    ) -> dict[str, Any]:
        """
        Generate category suggestions from a list of OPEN facet keywords.

        Uses LLM to cluster similar values and suggest meaningful category names.

        Args:
            values: List of unique keywords/phrases extracted from sources
            facet_name: Name of the facet being coded
            facet_description: Optional description of what this facet captures
            min_categories: Minimum number of categories to suggest
            max_categories: Maximum number of categories to suggest

        Returns:
            Dictionary with categories, uncategorized values, and reasoning
        """
        logger.info(f"Suggesting categories for facet '{facet_name}' with {len(values)} values")

        # Sorted so near-duplicates ("ADMM", "ADMM (Alternating Direction...)")
        # sit next to each other. The model groups a sorted list far more
        # reliably than the arbitrary insertion order the caller hands us.
        sorted_values = sorted(values, key=str.casefold)

        # Build the prompt
        values_list = "\n".join(f"- {v}" for v in sorted_values)
        description_text = f"\nDescription: {facet_description}" if facet_description else ""

        prompt = f"""You are helping a researcher organize data from a systematic mapping study.

The researcher has a facet called "{facet_name}" that collected keywords/phrases from research papers.{description_text}

Here are all the unique keywords/phrases extracted:
{values_list}

Your task is to group these values into at most {max_categories} categories (and at least {min_categories}).

HARD CONSTRAINT: the "categories" array must contain no more than {max_categories} entries. If you want more, merge the closest ones instead. A broader category is always preferable to exceeding the limit.

Guidelines:
1. Categories should be mutually exclusive and collectively exhaustive
2. Use clear, descriptive category names suitable for academic writing
3. Group semantically similar values together (synonyms, related concepts)
4. Create an "Other/Unclassified" category only for truly miscellaneous values
5. Each category should have at least 2 values if possible
6. Never return more than {max_categories} categories

Respond with a JSON object containing:
- categories: array of objects with name (string), description (string), and values (array of strings that belong to this category)
- uncategorized: array of values that don't fit any category
- reasoning: brief explanation of the categorization approach

Important: Include ALL values from the input in either a category or uncategorized. Do not miss any values."""

        try:
            # Use proper schema format for arrays - list with item schema as first element
            response_schema = {
                "categories": [
                    {
                        "name": "string",
                        "description": "string",
                        "values": ["string"],  # Array of strings
                    }
                ],
                "uncategorized": ["string"],  # Array of strings
                "reasoning": "string",
            }

            # The model has to echo every input value back inside the grouping,
            # so the output grows with the corpus. A fixed budget silently
            # truncates the JSON on large facets, and a truncated body parses
            # to {} — which looks like "the LLM found no categories" instead of
            # the overflow it actually is.
            max_tokens = min(32000, 2000 + len(values) * 16)

            # An overflowing or malformed reply reaches us as {}, indistinguishable
            # from "no categories found". On a large facet that happens often
            # enough to matter, and a bare retry usually lands, so try again
            # rather than handing the wizard an empty board.
            result: dict[str, Any] = {}
            for attempt in range(1, _SUGGEST_ATTEMPTS + 1):
                result = await self.llm_provider.generate_structured_output(
                    prompt=prompt,
                    response_schema=response_schema,
                    max_tokens=max_tokens,
                )
                if result.get("categories"):
                    break
                logger.warning(
                    "Category suggestion returned no categories for '%s' "
                    "(attempt %d/%d, %d values)",
                    facet_name,
                    attempt,
                    _SUGGEST_ATTEMPTS,
                    len(values),
                )

            # Every value must land in exactly one place, or the wizard's
            # "assigned + unassigned" no longer reconciles with the value count
            # and a value in two categories files its sources under both.
            # The model routinely repeats a value across groups, so first
            # placement wins and later ones are dropped.
            known = set(values)
            seen: set[str] = set()

            categories = result.get("categories", [])
            formatted_categories = []
            for cat in categories:
                if not isinstance(cat, dict):
                    continue
                unique_values = []
                for v in cat.get("values", []):
                    # Guard against the model inventing values that were never
                    # in the input — those have no keyword rows to classify.
                    if isinstance(v, str) and v in known and v not in seen:
                        seen.add(v)
                        unique_values.append(v)
                if not unique_values:
                    continue
                formatted_categories.append(
                    {
                        "name": cat.get("name", "Unknown"),
                        "description": cat.get("description", ""),
                        "values": unique_values,
                        "source_count": len(unique_values),
                    }
                )

            # Whatever the model dropped — or lost to a truncated response —
            # still has to reach the coding wizard, otherwise those values are
            # invisible and cannot be assigned by hand. Anything unaccounted
            # for falls back to uncategorized.
            uncategorized = []
            for v in result.get("uncategorized", []):
                if isinstance(v, str) and v in known and v not in seen:
                    seen.add(v)
                    uncategorized.append(v)
            uncategorized.extend(v for v in values if v not in seen)

            # The cap is a prompt instruction, and the model overshoots it on
            # large facets.
            # Merge back down rather than handing the wizard a schema the
            # researcher then has to collapse by hand.
            if len(formatted_categories) > max_categories:
                formatted_categories = await self._enforce_category_cap(
                    formatted_categories, facet_name, max_categories
                )

            return {
                "categories": formatted_categories,
                "uncategorized": uncategorized,
                "reasoning": result.get("reasoning", ""),
            }

        except Exception as e:
            logger.error(f"Error suggesting categories: {e}")
            raise

    async def _enforce_category_cap(
        self,
        categories: list[dict[str, Any]],
        facet_name: str,
        max_categories: int,
    ) -> list[dict[str, Any]]:
        """
        Merge an over-long category list down to ``max_categories``.

        Only the category names and descriptions go to the model, never the
        values themselves — the merge decision does not need them, and echoing
        a large facet's values a second time is what makes the first pass
        overflow. Values are re-attached locally, so no value can be lost or
        duplicated by the merge.
        """
        by_name = {c["name"]: c for c in categories}
        listing = "\n".join(
            f'- {c["name"]} ({len(c["values"])} values): {c.get("description", "")}'
            for c in categories
        )

        prompt = f"""A researcher coded the facet "{facet_name}" and ended up with \
{len(categories)} categories, which is more than the {max_categories} they can use.

Current categories:
{listing}

Merge them into at most {max_categories} groups. Every current category must appear in \
exactly one group. Merge the most closely related ones; keep the largest categories intact \
where you can.

Respond with a JSON object containing:
- groups: array of objects with name (string), description (string), and members (array of \
the current category names that belong to this group)"""

        response_schema = {
            "groups": [{"name": "string", "description": "string", "members": ["string"]}]
        }

        try:
            result = await self.llm_provider.generate_structured_output(
                prompt=prompt,
                response_schema=response_schema,
                max_tokens=4000,
            )
        except Exception as e:  # a failed merge must not lose the coding run
            logger.warning("Category cap merge failed for '%s': %s", facet_name, e)
            result = {}

        merged: list[dict[str, Any]] = []
        claimed: set[str] = set()
        for group in result.get("groups", []):
            if not isinstance(group, dict):
                continue
            members = [
                m for m in group.get("members", []) if m in by_name and m not in claimed
            ]
            if not members:
                continue
            claimed.update(members)
            values: list[str] = []
            for m in members:
                values.extend(by_name[m]["values"])
            merged.append(
                {
                    "name": group.get("name") or by_name[members[0]]["name"],
                    "description": group.get("description", ""),
                    "values": values,
                    "source_count": len(values),
                }
            )

        # Anything the model forgot to place keeps its own group for now; the
        # deterministic pass below is what actually guarantees the cap.
        merged.extend(c for c in categories if c["name"] not in claimed)

        if len(merged) > max_categories:
            merged.sort(key=lambda c: len(c["values"]), reverse=True)
            kept, overflow = merged[: max_categories - 1], merged[max_categories - 1 :]
            spill: list[str] = []
            for c in overflow:
                spill.extend(c["values"])
            logger.warning(
                "Category cap for '%s' still exceeded after merge; folding %d "
                "categories into 'Other'",
                facet_name,
                len(overflow),
            )
            kept.append(
                {
                    "name": "Other",
                    "description": "Categories that could not be merged elsewhere.",
                    "values": spill,
                    "source_count": len(spill),
                }
            )
            merged = kept

        return merged

    async def classify_value(
        self,
        value: str,
        categories: list[dict[str, Any]],
        facet_name: str,
    ) -> dict[str, Any]:
        """
        Classify a single value into one of the existing categories.

        Used for auto-categorizing new sources after coding is enabled.

        Args:
            value: The value to classify
            categories: Existing categories with id, name, description
            facet_name: Name of the facet

        Returns:
            Dictionary with category_id, category_name, confidence, and reasoning
        """
        logger.info(f"Classifying value '{value}' for facet '{facet_name}'")

        # Build categories description
        categories_desc = "\n".join(
            f"- {cat.get('name', 'Unknown')}: {cat.get('description', 'No description')}"
            for cat in categories
        )

        prompt = f"""You are classifying a value into predefined categories for a systematic mapping study.

Facet: "{facet_name}"

Available categories:
{categories_desc}

Value to classify: "{value}"

Determine which category this value best belongs to. If it doesn't clearly fit any category, classify it as null.

Respond with a JSON object containing:
- category_name: Name of the matching category (or null if no match)
- confidence: Number between 0.0-1.0
- reasoning: Brief explanation of why this classification was chosen

Confidence guidelines:
- 0.9-1.0: Very clear match (exact or near-exact terminology)
- 0.7-0.9: Good match (same concept, different wording)
- 0.5-0.7: Possible match (related but not certain)
- 0.0-0.5: Poor match (likely doesn't belong)

If confidence is below 0.5, set category_name to null."""

        try:
            response_schema = {
                "category_name": "string",
                "confidence": "number",
                "reasoning": "string",
            }

            result = await self.llm_provider.generate_structured_output(
                prompt=prompt,
                response_schema=response_schema,
                max_tokens=1000,
            )

            # Find the category ID from the name
            category_name = result.get("category_name")
            category_id = None

            if category_name:
                for cat in categories:
                    if cat.get("name", "").lower() == category_name.lower():
                        category_id = cat.get("id")
                        category_name = cat.get("name")  # Use exact casing
                        break

            confidence = float(result.get("confidence", 0.0))

            # If confidence is low, don't assign a category
            if confidence < 0.5:
                category_id = None
                category_name = None

            return {
                "category_id": category_id,
                "category_name": category_name,
                "confidence": confidence,
                "reasoning": result.get("reasoning", ""),
            }

        except Exception as e:
            logger.error(f"Error classifying value: {e}")
            raise

    async def map_keyword(
        self,
        keyword: str,
        categories: list[dict[str, Any]],
        facet_name: str,
        facet_description: str | None = None,
    ) -> dict[str, Any]:
        """
        Map a keyword/phrase to an existing category or propose a new category.

        Returns a suggested existing category or a proposed new category definition.
        """
        logger.info(f"Mapping keyword '{keyword}' for facet '{facet_name}'")

        categories_desc = "\n".join(
            f"- {cat.get('name', 'Unknown')}: {cat.get('description', 'No description')}"
            for cat in categories
        )
        description_text = f"\nFacet description: {facet_description}" if facet_description else ""

        prompt = f"""You are helping a researcher map keywords to coding categories.

Facet: "{facet_name}"{description_text}

Existing categories:
{categories_desc}

Keyword/phrase: "{keyword}"

Task:
1) If the keyword clearly fits one existing category, select that category.
2) If no category fits well, propose a new category name and description.

Respond with JSON:
- category_name: name of the best existing category (or null if none fits)
- confidence: 0.0-1.0
- reasoning: brief justification grounded in the keyword meaning
- propose_new: boolean
- proposed_category_name: string (required if propose_new=true)
- proposed_category_description: string (required if propose_new=true)
"""

        try:
            response_schema = {
                "category_name": "string",
                "confidence": "number",
                "reasoning": "string",
                "propose_new": "boolean",
                "proposed_category_name": "string",
                "proposed_category_description": "string",
            }

            result = await self.llm_provider.generate_structured_output(
                prompt=prompt,
                response_schema=response_schema,
                max_tokens=1000,
            )

            category_name = result.get("category_name")
            propose_new = bool(result.get("propose_new"))

            if propose_new:
                category_name = None

            return {
                "category_name": category_name,
                "confidence": float(result.get("confidence", 0.0)),
                "reasoning": result.get("reasoning", ""),
                "propose_new": propose_new,
                "proposed_category_name": result.get("proposed_category_name"),
                "proposed_category_description": result.get("proposed_category_description"),
            }
        except Exception as e:
            logger.error(f"Error mapping keyword: {e}")
            raise
