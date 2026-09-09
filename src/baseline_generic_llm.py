# O que faria um LLM normal se um gestor lhe desse as reviews recentes e alguns KPIs básicos, 
# sem o nosso framework evidence-informed?

# src/baseline_generic_llm.py

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
    "https://ai-pgbiht.openai.azure.com/openai/v1/"
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
# GENERIC SYSTEM PROMPT
# ==========================================================

GENERIC_SYSTEM_PROMPT = """
You are an AI assistant supporting hotel managers.

Your task is to review recent customer feedback and basic hotel
performance information and identify the most relevant managerial
priorities.

Use only the information supplied in the case.

Provide practical recommendations that could help hotel management
address the issues identified in customer feedback.

Do not invent facts that are not contained in the supplied case.

Do not claim that customer reviews caused changes in financial
performance.

Your role is to support managerial judgment, not replace it.
"""


# ==========================================================
# OUTPUT SCHEMA
# ==========================================================

GENERIC_DECISION_SCHEMA = {
    "type": "object",

    "properties": {

        "overall_risk": {
            "type": "string",
            "enum": [
                "low",
                "moderate",
                "high",
                "critical",
            ],
        },

        "overall_risk_rationale": {
            "type": "string",
        },

        "executive_summary": {
            "type": "string",
        },

        "priorities": {
            "type": "array",

            "items": {
                "type": "object",

                "properties": {

                    "rank": {
                        "type": "integer",
                    },

                    "area": {
                        "type": "string",
                    },

                    "managerial_priority": {
                        "type": "string",
                        "enum": [
                            "low",
                            "medium",
                            "high",
                            "urgent",
                        ],
                    },

                    "action_horizon": {
                        "type": "string",
                        "enum": [
                            "immediate",
                            "short_term",
                            "medium_term",
                            "monitor",
                        ],
                    },

                    "intervention_type": {
                        "type": "string",
                        "enum": [
                            "operational",
                            "maintenance",
                            "policy_process",
                            "capital_assessment",
                            "monitoring",
                            "mixed",
                        ],
                    },

                    "problem": {
                        "type": "string",
                    },

                    "recommended_action": {
                        "type": "string",
                    },

                    "rationale": {
                        "type": "string",
                    },

                    "supporting_evidence": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    },

                    "safety_relevance": {
                        "type": "boolean",
                    },

                    "requires_further_assessment": {
                        "type": "boolean",
                    },

                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },

                "required": [
                    "rank",
                    "area",
                    "managerial_priority",
                    "action_horizon",
                    "intervention_type",
                    "problem",
                    "recommended_action",
                    "rationale",
                    "supporting_evidence",
                    "safety_relevance",
                    "requires_further_assessment",
                    "confidence",
                ],

                "additionalProperties": False,
            },
        },

        "strengths_to_preserve": {
            "type": "array",

            "items": {
                "type": "object",

                "properties": {

                    "area": {
                        "type": "string",
                    },

                    "reason": {
                        "type": "string",
                    },

                    "management_implication": {
                        "type": "string",
                    },
                },

                "required": [
                    "area",
                    "reason",
                    "management_implication",
                ],

                "additionalProperties": False,
            },
        },

        "monitoring_indicators": {
            "type": "array",

            "items": {
                "type": "object",

                "properties": {

                    "indicator": {
                        "type": "string",
                    },

                    "reason": {
                        "type": "string",
                    },

                    "desired_direction": {
                        "type": "string",
                        "enum": [
                            "increase",
                            "decrease",
                            "stable",
                            "context_dependent",
                        ],
                    },
                },

                "required": [
                    "indicator",
                    "reason",
                    "desired_direction",
                ],

                "additionalProperties": False,
            },
        },

        "limitations": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
    },

    "required": [
        "overall_risk",
        "overall_risk_rationale",
        "executive_summary",
        "priorities",
        "strengths_to_preserve",
        "monitoring_indicators",
        "limitations",
    ],

    "additionalProperties": False,
}


# ==========================================================
# BUILD GENERIC CONTEXT
# ==========================================================

def build_generic_context(
    case: dict[str, Any],
    max_reviews: int = 20,
) -> dict[str, Any]:
    """
    Builds the information available to the Generic LLM baseline.

    The Generic LLM intentionally receives only:

    - basic case metadata;
    - basic hotel information;
    - basic performance indicators;
    - a neutral sample of the most recent customer reviews.

    It intentionally does NOT receive:

    - knowledge-base managerial categories;
    - managerial evidence;
    - diagnostic vulnerabilities;
    - diagnostic scores;
    - evidence patterns;
    - expectation violations;
    - expectation-experience gap;
    - semantic misalignment;
    - emotional complexity;
    - emerging-negative signals;
    - preliminary signals;
    - diagnostic complaint sampling.

    This creates a generic LLM baseline without the
    evidence-informed reasoning framework.
    """

    # ------------------------------------------------------
    # CASE METADATA
    # ------------------------------------------------------

    metadata = case.get(
        "case_metadata",
        {}
    )

    # ------------------------------------------------------
    # HOTEL PROFILE
    # ------------------------------------------------------

    hotel = case.get(
        "hotel_profile",
        {}
    )

    # ------------------------------------------------------
    # BASIC TRENDS
    # ------------------------------------------------------

    trends = case.get(
        "trends",
        {}
    )

    # ------------------------------------------------------
    # NEUTRAL REVIEW SAMPLE
    #
    # IMPORTANT:
    # Do NOT use case["complaints"] here.
    # ------------------------------------------------------

    review_sample = case.get(
        "review_sample",
        []
    )

    # ------------------------------------------------------
    # BASIC PERFORMANCE INFORMATION ONLY
    # ------------------------------------------------------

    allowed_metrics = [
        "REVPAR_WEEK",
        "REVPOR_WEEK",
        "TREVPAR_WEEK",
        "TAXA_OCUPACAO",
        "AVG_RATING",
        "N_REVIEWS",
    ]

    basic_performance = {}

    for metric in allowed_metrics:

        metric_data = trends.get(
            metric
        )

        if not metric_data:
            continue

        basic_performance[
            metric
        ] = {

            "first_value": (
                metric_data.get(
                    "first_value"
                )
            ),

            "last_value": (
                metric_data.get(
                    "last_value"
                )
            ),

            "percentage_change": (
                metric_data.get(
                    "percentage_change"
                )
            ),

            "n_observations": (
                metric_data.get(
                    "n_observations"
                )
            ),
        }

    # ------------------------------------------------------
    # RAW RECENT REVIEWS
    # ------------------------------------------------------

    recent_reviews = []

    for review in review_sample[
        :max_reviews
    ]:

        recent_reviews.append(
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
                    review.get(
                        "liked"
                    )
                ),

                "disliked": (
                    review.get(
                        "disliked"
                    )
                ),

                "country": (
                    review.get(
                        "country"
                    )
                ),

                "stay_type": (
                    review.get(
                        "stay_type"
                    )
                ),
            }
        )

    # ------------------------------------------------------
    # FINAL GENERIC CONTEXT
    # ------------------------------------------------------

    return {

        "case": {

            "case_id": (
                metadata.get(
                    "case_id"
                )
            ),

            "decision_week": (
                metadata.get(
                    "decision_week"
                )
            ),

            "decision_date": (
                metadata.get(
                    "decision_date"
                )
            ),

            "window_start": (
                metadata.get(
                    "window_start"
                )
            ),

            "window_end": (
                metadata.get(
                    "window_end"
                )
            ),

            "lookback_weeks": (
                metadata.get(
                    "lookback_weeks_requested"
                )
            ),
        },

        "hotel": hotel,

        "basic_performance": (
            basic_performance
        ),

        "recent_customer_reviews": (
            recent_reviews
        ),

        "review_sample_information": {

            "reviews_available_in_window": (
                case.get(
                    "review_sample_metadata",
                    {}
                ).get(
                    "total_reviews_in_window"
                )
            ),

            "reviews_provided_to_llm": (
                len(
                    recent_reviews
                )
            ),

            "sampling_method": (
                "Most recent reviews in the "
                "decision window. No selection "
                "based on rating, sentiment, "
                "complaint severity or text length."
            ),
        },
    }


# ==========================================================
# PROMPT BUILDER
# ==========================================================

def _build_generic_prompt(
    generic_context: dict[str, Any],
) -> str:
    """
    Builds a deliberately generic managerial prompt.

    The prompt does not contain the evidence-informed
    reasoning rules used by the treatment agent.
    """

    context_json = json.dumps(
        generic_context,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    return f"""
Review the following hotel case.

Based on the recent customer reviews and basic hotel performance
information, identify the most important managerial priorities.

INSTRUCTIONS

- Select a maximum of 5 priorities.
- Rank them from most important to least important.
- Explain the problem associated with each priority.
- Recommend an appropriate managerial action.
- Identify any potential safety relevance.
- Identify strengths that management should preserve.
- Suggest useful indicators for monitoring the situation.
- Use only the supplied case information.
- Do not invent facts that are not contained in the case.
- Do not claim that customer reviews caused changes in financial
  performance.

HOTEL CASE

{context_json}
"""


# ==========================================================
# VALIDATION
# ==========================================================

def _validate_generic_decision(
    decision: dict[str, Any],
) -> None:
    """
    Performs basic structural validation of the
    Generic LLM output.
    """

    priorities = decision.get(
        "priorities",
        []
    )

    if len(priorities) > 5:

        raise ValueError(
            "Generic LLM returned more "
            "than 5 priorities."
        )

    ranks = [
        priority.get(
            "rank"
        )
        for priority in priorities
    ]

    expected_ranks = list(
        range(
            1,
            len(priorities) + 1,
        )
    )

    if ranks != expected_ranks:

        raise ValueError(
            "Priority ranks are not sequential. "
            f"Received: {ranks}"
        )

    for priority in priorities:

        confidence = priority.get(
            "confidence"
        )

        if confidence is None:

            raise ValueError(
                "A priority is missing confidence."
            )

        if not (
            0 <= confidence <= 1
        ):

            raise ValueError(
                "Priority confidence must be "
                "between 0 and 1."
            )


# ==========================================================
# GENERIC LLM BASELINE
# ==========================================================

def run_generic_llm_baseline(
    case: dict[str, Any],
    max_reviews: int = 20,
) -> dict[str, Any]:
    """
    Runs the Generic LLM baseline.

    Experimental interpretation
    ---------------------------
    The same underlying LLM deployment used by the
    Evidence-Informed Agent is used here.

    The key difference is the information and reasoning
    architecture:

    Generic LLM:
        neutral recent reviews
        + basic KPIs
        + generic managerial reasoning

    Evidence-Informed Agent:
        structured managerial evidence
        + diagnostic complaint evidence
        + experiential indicators
        + temporal signals
        + evidence-informed reasoning rules
    """

    # ------------------------------------------------------
    # BUILD GENERIC CONTEXT
    # ------------------------------------------------------

    generic_context = (
        build_generic_context(
            case=case,
            max_reviews=max_reviews,
        )
    )

    # ------------------------------------------------------
    # BUILD PROMPT
    # ------------------------------------------------------

    prompt = (
        _build_generic_prompt(
            generic_context
        )
    )

    # ------------------------------------------------------
    # CALL LLM
    # ------------------------------------------------------

    response = client.responses.create(

        model=AZURE_OPENAI_MODEL,

        instructions=(
            GENERIC_SYSTEM_PROMPT
        ),

        input=prompt,

        text={
            "format": {
                "type": "json_schema",
                "name": (
                    "generic_hotel_decision"
                ),
                "schema": (
                    GENERIC_DECISION_SCHEMA
                ),
                "strict": True,
            }
        },
    )

    # ------------------------------------------------------
    # PARSE RESPONSE
    # ------------------------------------------------------

    raw = response.output_text

    if not raw:

        raise ValueError(
            "The Generic LLM returned "
            "an empty response."
        )

    try:

        decision = json.loads(
            raw
        )

    except json.JSONDecodeError as error:

        raise ValueError(
            "The Generic LLM did not "
            "return valid JSON."
        ) from error

    # ------------------------------------------------------
    # VALIDATE
    # ------------------------------------------------------

    _validate_generic_decision(
        decision
    )

    # ------------------------------------------------------
    # EXPERIMENT METADATA
    # ------------------------------------------------------

    decision[
        "system"
    ] = "generic_llm_baseline"

    decision[
        "system_version"
    ] = "generic_llm_v1"

    decision[
        "case_id"
    ] = (
        generic_context[
            "case"
        ].get(
            "case_id"
        )
    )

    decision[
        "model"
    ] = AZURE_OPENAI_MODEL

    decision[
        "input_review_count"
    ] = len(
        generic_context[
            "recent_customer_reviews"
        ]
    )

    decision[
        "input_type"
    ] = (
        "neutral_recent_reviews_plus_basic_kpis"
    )

    return decision