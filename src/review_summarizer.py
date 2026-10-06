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

AZURE_OPENAI_API_KEY = os.getenv(
    "AZURE_OPENAI_API_KEY"
)

AZURE_OPENAI_MODEL = os.getenv(
    "AZURE_OPENAI_MODEL"
)

AZURE_OPENAI_BASE_URL = os.getenv(
    "AZURE_OPENAI_BASE_URL",
    "https://ai-pgbiht.openai.azure.com/openai/v1/",
)


if not AZURE_OPENAI_API_KEY:
    raise ValueError(
        "AZURE_OPENAI_API_KEY not found."
    )

if not AZURE_OPENAI_MODEL:
    raise ValueError(
        "AZURE_OPENAI_MODEL not found."
    )


client = OpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    base_url=AZURE_OPENAI_BASE_URL,
)


# ==========================================================
# NULL / TEXT HELPERS
# ==========================================================

NULL_LIKE = {
    "",
    "nan",
    "none",
    "null",
    "nat",
    "n/a",
    "na",
}


def _clean_text(
    value: Any,
) -> str:
    """
    Convert a value to clean text.

    Null-like values are returned as an empty string.
    """

    if value is None:
        return ""

    text = str(
        value
    ).strip()

    if text.lower() in NULL_LIKE:
        return ""

    return text


def _word_count(
    text: str,
) -> int:
    """
    Count whitespace-separated words.
    """

    return len(
        [
            word
            for word in text.strip().split()
            if word
        ]
    )


# ==========================================================
# NEUTRAL REVIEW SUMMARY
# ==========================================================

REVIEW_SUMMARY_SYSTEM_PROMPT = """
You create neutral summaries of hotel customer feedback for a blinded
managerial evaluation study.

Your task is ONLY to summarize the supplied customer reviews.

STRICT RULES:

1. Use only the supplied reviews.

2. Do not use hotel KPIs, hotel performance indicators, or any external
   information.

3. Do not recommend managerial actions.

4. Do not rank or prioritize issues.

5. Do not infer causes that are not explicitly supported by the reviews.

6. Represent both positive and negative feedback when both are present.

7. Mention recurring themes when they are clearly visible across the
   supplied reviews.

8. Distinguish recurring feedback from isolated observations.
   Do not present a single isolated comment as a general pattern.

9. Do not exaggerate the severity or prevalence of any issue.

10. Preserve important nuance when feedback is mixed.

11. Do not mention AI, agents, models, systems, experimental conditions,
    or research methods.

12. Write one concise, natural paragraph in English.

13. Target approximately 50-70 words.

14. Reviews without written liked/disliked text provide no textual
    evidence and should not be described as if they did.

The summary will be shown identically when evaluating both blinded
managerial decisions. It must therefore remain neutral and must not
favour either decision.
"""


REVIEW_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
        },
    },
    "required": [
        "summary",
    ],
    "additionalProperties": False,
}


def build_neutral_review_summary(
    case: dict[str, Any],
    max_reviews: int = 20,
) -> dict[str, Any]:
    """
    Build one neutral customer-feedback summary from the SAME
    neutral review_sample used by the Generic LLM baseline.

    Important
    ---------
    - The first `max_reviews` rows are preserved as the source sample.
    - No additional reviews are fetched to replace rows without text.
    - Reviews without written liked/disliked feedback are ignored only
      when constructing the textual summary.
    - The resulting summary is common to Decision A and Decision B.
    """

    review_sample = (
        case.get(
            "review_sample",
            [],
        )[:max_reviews]
    )

    reviews_for_prompt = []

    for review in review_sample:

        liked = _clean_text(
            review.get(
                "liked"
            )
        )

        disliked = _clean_text(
            review.get(
                "disliked"
            )
        )

        # No written textual evidence in this review.
        if not liked and not disliked:
            continue

        reviews_for_prompt.append(
            {
                "review_date": (
                    review.get(
                        "review_date"
                    )
                ),
                "rating": (
                    review.get(
                        "rating"
                    )
                ),
                "liked": (
                    liked
                    if liked
                    else None
                ),
                "disliked": (
                    disliked
                    if disliked
                    else None
                ),
            }
        )

    # ------------------------------------------------------
    # NO WRITTEN FEEDBACK
    # ------------------------------------------------------

    if not reviews_for_prompt:

        return {
            "summary": (
                "No written customer feedback is available "
                "in the recent review sample for this case."
            ),
            "source_reviews_selected": (
                len(
                    review_sample
                )
            ),
            "source_reviews_with_written_feedback": 0,
        }

    # ------------------------------------------------------
    # PROMPT
    # ------------------------------------------------------

    prompt = f"""
Summarize the customer feedback below for a hotel manager.

Create a neutral factual overview of the feedback.

Do NOT:
- recommend actions;
- prioritize issues;
- interpret hotel performance;
- infer causes not contained in the reviews;
- exaggerate isolated comments.

Represent both positive and negative feedback when supported.

Target approximately 50-70 words.

CUSTOMER REVIEWS

{json.dumps(
    reviews_for_prompt,
    ensure_ascii=False,
    indent=2,
    default=str,
)}
"""

    # ------------------------------------------------------
    # LLM CALL
    # ------------------------------------------------------

    response = client.responses.create(

        model=AZURE_OPENAI_MODEL,

        instructions=(
            REVIEW_SUMMARY_SYSTEM_PROMPT
        ),

        input=prompt,

        text={
            "format": {
                "type": "json_schema",
                "name": (
                    "neutral_review_summary"
                ),
                "schema": (
                    REVIEW_SUMMARY_SCHEMA
                ),
                "strict": True,
            }
        },
    )

    # ------------------------------------------------------
    # PARSE
    # ------------------------------------------------------

    raw = response.output_text

    if not raw:
        raise ValueError(
            "The review summarizer returned "
            "an empty response."
        )

    try:

        result = json.loads(
            raw
        )

    except json.JSONDecodeError as error:

        raise ValueError(
            "The review summarizer did not "
            "return valid JSON."
        ) from error

    summary = _clean_text(
        result.get(
            "summary"
        )
    )

    if not summary:
        raise ValueError(
            "The review summarizer returned "
            "an empty summary."
        )

    # ------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------

    return {
        "summary": summary,

        "source_reviews_selected": (
            len(
                review_sample
            )
        ),

        "source_reviews_with_written_feedback": (
            len(
                reviews_for_prompt
            )
        ),
    }


# ==========================================================
# SHORT DECISION PRESENTATION
# ==========================================================

SHORT_RECOMMENDATION_SYSTEM_PROMPT = """
You create a concise presentation version of an existing hotel
managerial recommendation for a blinded human-evaluation study.

You are NOT making a new managerial decision.

Your task is to identify and express the SINGLE CORE MANAGERIAL ACTION
contained in the original recommendation.

STRICT RULES:

1. Preserve the central managerial meaning of the original recommendation.

2. Select the most important managerial action rather than attempting
   to compress every action contained in the original recommendation.

3. Do not add any new:
   - action;
   - issue;
   - cause;
   - evidence;
   - benefit;
   - cost;
   - expected outcome;
   - operational detail.

4. Do not make the recommendation more specific, stronger, or more
   aggressive than the original.

5. Preserve the original level of managerial caution.

6. If the original recommendation uses cautious language such as:
   - assess;
   - review;
   - inspect;
   - monitor;
   - evaluate;
   - consider;
   - verify;
   then preserve that cautious orientation.

7. NEVER convert an assessment-oriented recommendation into an
   implementation-oriented recommendation.

8. If the original contains several possible actions, identify the
   central action. Do NOT attempt to include all secondary actions.

9. Do not combine several actions using semicolons.

10. Avoid lists inside the sentence.

11. Write ONE clear and natural English sentence.

12. Target 10-15 words.

13. Never exceed 18 words.

14. Do not include the priority rank.

15. Do not repeat the managerial area as a heading inside the sentence.

16. Do not mention AI, agents, models, systems, experimental conditions,
    or research methods.

The shortened sentence is only a presentation layer. It must preserve
the substantive nature of the original recommendation without improving,
correcting, or extending it.
"""


SHORT_RECOMMENDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "short_recommendation": {
            "type": "string",
        },
    },
    "required": [
        "short_recommendation",
    ],
    "additionalProperties": False,
}


def shorten_recommendation(
    area: Any,
    recommended_action: Any,
) -> str:
    """
    Create a concise presentation-only version of an existing
    managerial recommendation.

    The same transformation is applied to recommendations from
    BOTH experimental systems.

    Important
    ---------
    This function does not change the original experimental output.
    It creates only the short representation shown to human evaluators.

    The goal is to preserve the SINGLE CORE MANAGERIAL ACTION rather
    than compressing every action from the original recommendation.
    """

    area_text = _clean_text(
        area
    )

    action_text = _clean_text(
        recommended_action
    )

    # ------------------------------------------------------
    # MISSING ACTION
    # ------------------------------------------------------

    if not action_text:

        return (
            area_text
            or "No recommendation provided."
        )

    # ------------------------------------------------------
    # IMPORTANT:
    #
    # We intentionally do NOT automatically return recommendations
    # that are already 10-15 words.
    #
    # Even a short original recommendation may contain multiple
    # compressed actions. Every recommendation therefore passes
    # through the SAME extraction step.
    # ------------------------------------------------------

    prompt = f"""
MANAGERIAL AREA

{area_text or "Not specified"}


ORIGINAL MANAGERIAL RECOMMENDATION

{action_text}


TASK

Extract the SINGLE CORE MANAGERIAL ACTION from the original
recommendation.

Do not summarize every action.

Do not add anything that is not present in the original recommendation.

Preserve the original degree of caution.

Return ONE natural English sentence of approximately 10-15 words,
with an absolute maximum of 18 words.
"""

    # ------------------------------------------------------
    # LLM CALL
    # ------------------------------------------------------

    response = client.responses.create(

        model=AZURE_OPENAI_MODEL,

        instructions=(
            SHORT_RECOMMENDATION_SYSTEM_PROMPT
        ),

        input=prompt,

        text={
            "format": {
                "type": "json_schema",
                "name": (
                    "short_managerial_recommendation"
                ),
                "schema": (
                    SHORT_RECOMMENDATION_SCHEMA
                ),
                "strict": True,
            }
        },
    )

    # ------------------------------------------------------
    # PARSE
    # ------------------------------------------------------

    raw = response.output_text

    if not raw:
        raise ValueError(
            "The recommendation formatter returned "
            "an empty response."
        )

    try:

        result = json.loads(
            raw
        )

    except json.JSONDecodeError as error:

        raise ValueError(
            "The recommendation formatter did not "
            "return valid JSON."
        ) from error

    short_text = _clean_text(
        result.get(
            "short_recommendation"
        )
    )

    if not short_text:
        raise ValueError(
            "The recommendation formatter returned "
            "an empty recommendation."
        )

    # ------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------

    word_count = _word_count(
        short_text
    )

    if word_count > 18:

        raise ValueError(
            "The shortened recommendation exceeds "
            "the 18-word maximum. "
            f"Word count: {word_count}. "
            f"Recommendation: {short_text}"
        )

    # Semicolons are a useful warning signal that the model
    # has attempted to compress multiple actions again.
    if ";" in short_text:

        raise ValueError(
            "The shortened recommendation contains a semicolon, "
            "which may indicate multiple compressed actions: "
            f"{short_text}"
        )

    return short_text