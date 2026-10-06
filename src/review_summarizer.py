# src/review_summarizer.py

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


# ==========================================================
# ENVIRONMENT
# ==========================================================

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")
AZURE_OPENAI_BASE_URL = os.getenv(
    "AZURE_OPENAI_BASE_URL",
    "https://ai-pgbiht.openai.azure.com/openai/v1/",
)

if not AZURE_OPENAI_API_KEY:
    raise ValueError("AZURE_OPENAI_API_KEY not found.")

if not AZURE_OPENAI_MODEL:
    raise ValueError("AZURE_OPENAI_MODEL not found.")

client = OpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    base_url=AZURE_OPENAI_BASE_URL,
)


# ==========================================================
# NULL / TEXT HELPERS
# ==========================================================

NULL_LIKE = {"", "nan", "none", "null", "nat", "n/a", "na"}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in NULL_LIKE:
        return ""

    return text


def _word_count(text: str) -> int:
    return len([word for word in text.strip().split() if word])


# ==========================================================
# NEUTRAL REVIEW SUMMARY
# ==========================================================

REVIEW_SUMMARY_SYSTEM_PROMPT = """
You create neutral summaries of hotel customer feedback for a blinded
managerial evaluation study.

Your task is ONLY to summarize the supplied customer reviews.

STRICT RULES:
1. Use only the supplied reviews.
2. Do not use hotel KPIs or any external information.
3. Do not recommend managerial actions.
4. Do not rank or prioritize issues.
5. Do not infer causes that are not explicitly supported.
6. Represent both positive and negative feedback when both are present.
7. Mention recurring themes when they are visible in the supplied reviews.
8. Do not exaggerate isolated comments into general conclusions.
9. Do not mention AI, agents, systems, experimental conditions, or methods.
10. Write one concise paragraph in English.
11. Target approximately 50-70 words.
12. Reviews without written liked/disliked text provide no textual evidence
    and should not be described as if they did.

The same summary will be shown when evaluating both blinded decisions.
"""


REVIEW_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
        },
    },
    "required": ["summary"],
    "additionalProperties": False,
}


def build_neutral_review_summary(
    case: dict[str, Any],
    max_reviews: int = 20,
) -> dict[str, Any]:
    """
    Build one neutral customer-feedback summary from the SAME neutral
    review_sample used by the Generic LLM baseline.

    Important:
    - the first `max_reviews` rows are preserved as the source sample;
    - no extra reviews are fetched to replace rows without written text;
    - rows with no written liked/disliked text are ignored only when
      constructing the textual summary.
    """

    review_sample = case.get("review_sample", [])[:max_reviews]

    reviews_for_prompt = []

    for review in review_sample:
        liked = _clean_text(review.get("liked"))
        disliked = _clean_text(review.get("disliked"))

        if not liked and not disliked:
            continue

        reviews_for_prompt.append(
            {
                "review_date": review.get("review_date"),
                "rating": review.get("rating"),
                "liked": liked or None,
                "disliked": disliked or None,
            }
        )

    if not reviews_for_prompt:
        return {
            "summary": (
                "No written customer feedback is available in the recent "
                "review sample for this case."
            ),
            "source_reviews_selected": len(review_sample),
            "source_reviews_with_written_feedback": 0,
        }

    prompt = f"""
Summarize the customer feedback below for a hotel manager.

Do not make recommendations and do not prioritize issues.
Produce a balanced factual summary of approximately 50-70 words.

CUSTOMER REVIEWS

{json.dumps(reviews_for_prompt, ensure_ascii=False, indent=2, default=str)}
"""

    response = client.responses.create(
        model=AZURE_OPENAI_MODEL,
        instructions=REVIEW_SUMMARY_SYSTEM_PROMPT,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "neutral_review_summary",
                "schema": REVIEW_SUMMARY_SCHEMA,
                "strict": True,
            }
        },
    )

    raw = response.output_text

    if not raw:
        raise ValueError("The review summarizer returned an empty response.")

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(
            "The review summarizer did not return valid JSON."
        ) from error

    summary = _clean_text(result.get("summary"))

    if not summary:
        raise ValueError("The review summarizer returned an empty summary.")

    return {
        "summary": summary,
        "source_reviews_selected": len(review_sample),
        "source_reviews_with_written_feedback": len(reviews_for_prompt),
    }


# ==========================================================
# SHORT DECISION PRESENTATION
# ==========================================================

SHORT_RECOMMENDATION_SYSTEM_PROMPT = """
You shorten hotel managerial recommendations for presentation in a blinded
human-evaluation questionnaire.

You are a FORMATTER, not a decision-maker.

STRICT RULES:
1. Preserve the managerial meaning of the original recommendation.
2. Do not add a new action, issue, cause, benefit, cost, or expected outcome.
3. Do not make the recommendation more specific or more aggressive.
4. Preserve cautious wording such as assess, review, inspect, monitor,
   evaluate, or consider when it appears in the original.
5. Do not convert assessment into implementation.
6. Do not add evidence that is not in the original recommendation.
7. Return one natural English sentence.
8. Target 10-15 words.
9. Never exceed 18 words.
10. Do not mention AI, agents, models, or experimental conditions.
"""


SHORT_RECOMMENDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "short_recommendation": {
            "type": "string",
        },
    },
    "required": ["short_recommendation"],
    "additionalProperties": False,
}


def shorten_recommendation(
    area: Any,
    recommended_action: Any,
) -> str:
    """
    Create a short presentation-only version of an existing recommendation.

    The same formatter and prompt are applied to both experimental systems.
    The original full decision outputs remain unchanged.
    """

    area_text = _clean_text(area)
    action_text = _clean_text(recommended_action)

    if not action_text:
        return area_text or "No recommendation provided."

    # Already concise enough: do not call the model unnecessarily.
    if 10 <= _word_count(action_text) <= 15:
        return action_text

    prompt = f"""
AREA
{area_text or "Not specified"}

ORIGINAL RECOMMENDATION
{action_text}

Rewrite only the ORIGINAL RECOMMENDATION as one concise 10-15 word sentence.
"""

    response = client.responses.create(
        model=AZURE_OPENAI_MODEL,
        instructions=SHORT_RECOMMENDATION_SYSTEM_PROMPT,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "short_managerial_recommendation",
                "schema": SHORT_RECOMMENDATION_SCHEMA,
                "strict": True,
            }
        },
    )

    raw = response.output_text

    if not raw:
        raise ValueError(
            "The recommendation formatter returned an empty response."
        )

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(
            "The recommendation formatter did not return valid JSON."
        ) from error

    short_text = _clean_text(result.get("short_recommendation"))

    if not short_text:
        raise ValueError(
            "The recommendation formatter returned an empty recommendation."
        )

    if _word_count(short_text) > 18:
        raise ValueError(
            "The shortened recommendation exceeds the 18-word safety limit: "
            f"{short_text}"
        )

    return short_text
