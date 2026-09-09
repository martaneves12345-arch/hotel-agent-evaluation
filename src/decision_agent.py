# src/decision_agent.py

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
# SYSTEM PROMPT
# ==========================================================

SYSTEM_PROMPT = """
You are an evidence-informed hotel managerial decision agent.

Your task is to transform structured customer-review evidence,
experience indicators, and hotel performance information into
managerially useful recommendations.

You are NOT a generic hotel consultant.

You must reason strictly from the evidence supplied in the case.


CORE PRINCIPLES

1. Do not invent facts, causes, trends, costs, operational constraints,
   technical specifications, customer preferences, or implementation
   details that are not supported by the evidence.

2. Distinguish clearly between:
   - net-negative areas;
   - positive areas with specific friction;
   - clear strengths;
   - mixed areas.

3. A positive-with-friction area must NOT be described as generally poor.
   Preserve the underlying strength while addressing the specific friction.

4. Do not interpret diagnostic_score as the final managerial priority.
   It is only a transparent evidence-selection heuristic.

5. Determine final priorities independently by considering:
   - negative evidence;
   - positive counterevidence;
   - expectation violations;
   - emerging negative evidence;
   - persistence across weeks;
   - severity of concrete complaints;
   - potential safety implications;
   - current business performance;
   - recent trend reversals;
   - likely nature of the intervention.

6. Distinguish between:
   - operational interventions;
   - maintenance interventions;
   - policy/process interventions;
   - capital assessment;
   - monitoring actions;
   - mixed interventions.

7. Do not recommend major capital investment solely because complaints exist.
   If the evidence suggests a structural issue but investment requirements
   cannot be determined from the data, recommend assessment or inspection
   before investment.

8. Do not prescribe specific technical solutions, products, discounts,
   packages, staffing levels, physical modifications, or operational
   mechanisms unless they are directly supported by the evidence.

9. When the evidence identifies a problem but does not determine the
   appropriate implementation, recommend:
   - assessment;
   - inspection;
   - process review;
   - policy review;
   - managerial evaluation;
   rather than inventing a specific solution.

10. Do not transform a single guest suggestion into a recommended intervention
    unless the broader case evidence supports that intervention.

11. Guest-proposed solutions contained in reviews are evidence of perceived
    friction, expectations, or preferences. They are NOT automatically
    validated managerial solutions.

12. When a review proposes a specific solution, separate:
    - the underlying problem supported by the review;
    - the guest's proposed solution.

    Use the problem as evidence.
    Treat the proposed solution only as contextual information unless broader
    evidence independently supports it.

13. When multiple reviews describe the same underlying friction but propose
    different solutions, recommend managerial review of the underlying process
    rather than selecting one guest-proposed solution.

14. Safety-related evidence should receive additional managerial attention,
    even when its frequency is limited.

15. Strong financial performance does not eliminate the relevance of
    customer-experience vulnerabilities.

16. Global trends and recent weekly movement must be interpreted separately.

17. If recent_reversal is true:
    - explicitly acknowledge the reversal;
    - do not describe the metric as continuously worsening or continuously
      improving.

18. Selected complaints are diagnostic examples and are not statistically
    representative of the entire customer population.

19. Recommendations must be actionable but must remain within the scope of
    the available evidence.

20. Every priority must be traceable to evidence contained in the case.

21. Do not claim causality between review evidence and RevPAR, revenue,
    occupancy, ratings, or any other performance metric.

22. Confidence represents confidence in the evidential support for the
    recommendation given the supplied case.

23. Confidence does NOT represent:
    - probability of managerial success;
    - expected ROI;
    - probability of improving RevPAR;
    - causal effect size.

24. Supporting evidence must refer only to information present in the case.

25. Positive counterevidence must be explicitly acknowledged when present.

26. Do not convert isolated negative comments into a general hotel-wide
    conclusion unless broader evidence supports that interpretation.

27. Where evidence is insufficient to determine scope, cause, cost, or
    solution, explicitly state that further assessment is required.

28. Recommendations about policies, access rules, pricing, packaging,
    amenities, or service design must be framed as review/evaluation unless
    the supplied evidence clearly supports the specific intervention.

29. Do not assume that a guest-requested policy is optimal for hotel
    operations, profitability, fairness, capacity management, or other guests.

Your role is to support managerial judgment, not replace it.
"""


# ==========================================================
# STRUCTURED OUTPUT SCHEMA
# ==========================================================

DECISION_SCHEMA = {
    "type": "object",
    "properties": {

        # --------------------------------------------------
        # OVERALL ASSESSMENT
        # --------------------------------------------------

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

        # --------------------------------------------------
        # PRIORITIES
        # --------------------------------------------------

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

                    "evidence_pattern": {
                        "type": "string",
                        "enum": [
                            "net_negative",
                            "positive_with_friction",
                            "predominantly_positive",
                            "mixed",
                        ],
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

                    # --------------------------------------
                    # TRACEABILITY
                    # --------------------------------------

                    "supporting_evidence": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    },

                    "positive_counterevidence": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    },

                    # --------------------------------------
                    # STRUCTURED EVIDENCE BASIS
                    # --------------------------------------

                    "evidence_basis": {
                        "type": "object",
                        "properties": {

                            "positive_mentions": {
                                "type": "integer",
                            },

                            "negative_mentions": {
                                "type": "integer",
                            },

                            "expectation_violation": {
                                "type": "boolean",
                            },

                            "emerging_negative": {
                                "type": "boolean",
                            },

                            "experience_weeks": {
                                "type": "integer",
                            },

                            "expectation_weeks": {
                                "type": "integer",
                            },

                            "violated_expectation_weeks": {
                                "type": "integer",
                            },

                            "emerging_negative_weeks": {
                                "type": "integer",
                            },

                            "review_examples_used": {
                                "type": "integer",
                            },

                            "relevant_trend_signals": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                },
                            },
                        },

                        "required": [
                            "positive_mentions",
                            "negative_mentions",
                            "expectation_violation",
                            "emerging_negative",
                            "experience_weeks",
                            "expectation_weeks",
                            "violated_expectation_weeks",
                            "emerging_negative_weeks",
                            "review_examples_used",
                            "relevant_trend_signals",
                        ],

                        "additionalProperties": False,
                    },

                    # --------------------------------------
                    # SAFETY / UNCERTAINTY
                    # --------------------------------------

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
                    "evidence_pattern",
                    "managerial_priority",
                    "action_horizon",
                    "intervention_type",
                    "problem",
                    "recommended_action",
                    "rationale",
                    "supporting_evidence",
                    "positive_counterevidence",
                    "evidence_basis",
                    "safety_relevance",
                    "requires_further_assessment",
                    "confidence",
                ],

                "additionalProperties": False,
            },
        },

        # --------------------------------------------------
        # STRENGTHS
        # --------------------------------------------------

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

        # --------------------------------------------------
        # MONITORING
        # --------------------------------------------------

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

                    "monitoring_horizon": {
                        "type": "string",
                        "enum": [
                            "immediate",
                            "short_term",
                            "medium_term",
                            "ongoing",
                        ],
                    },
                },

                "required": [
                    "indicator",
                    "reason",
                    "desired_direction",
                    "monitoring_horizon",
                ],

                "additionalProperties": False,
            },
        },

        # --------------------------------------------------
        # LIMITATIONS
        # --------------------------------------------------

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
# PROMPT BUILDER
# ==========================================================

def _build_user_prompt(
    agent_context: dict[str, Any],
) -> str:
    """
    Converts the structured agent context into the
    decision prompt.
    """

    context_json = json.dumps(
        agent_context,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    return f"""
Analyze the following hotel decision case.

Produce an evidence-informed managerial assessment based ONLY
on the supplied evidence.


IMPORTANT DECISION INSTRUCTIONS

1. Select a maximum of 5 managerial priorities.

2. Do not mechanically reproduce the diagnostic vulnerability ranking.

3. A lower-ranked diagnostic issue may become a higher managerial
   priority if:
   - severity is greater;
   - safety relevance is greater;
   - the issue appears structural;
   - the evidence indicates stronger expectation failure.

4. Conversely, an area with many negative mentions may receive a lower
   managerial priority if:
   - positive counterevidence is strong;
   - the issue is narrow;
   - the issue is operational rather than structural;
   - recent evidence suggests improvement.

5. Preserve strengths.

6. Treat positive-with-friction areas as strengths requiring targeted
   correction, not as globally negative areas.

7. Explicitly account for recent trend reversals.

8. Do not infer causal effects on RevPAR, occupancy, rating, sentiment,
   or any other performance metric.

9. Do not invent:
   - ROI;
   - intervention costs;
   - technical specifications;
   - staffing levels;
   - discounts;
   - packages;
   - physical modifications;
   - operational mechanisms;
   unless directly supported by the supplied evidence.

10. When evidence identifies a problem but not the appropriate solution,
    recommend assessment, inspection, process review, policy review, or
    managerial evaluation.

11. Do not transform a single guest suggestion into a recommended intervention
    unless the broader case evidence supports that intervention.

12. Guest-proposed solutions contained in reviews are evidence of perceived
    friction, expectations, or preferences. They are not automatically
    validated managerial solutions.

13. When a review proposes a specific solution, identify the underlying
    problem separately from the guest-proposed solution. Use the underlying
    problem as evidence. Do not adopt the guest's proposed solution unless
    broader evidence supports it independently.

14. If several guests describe the same friction but only one proposes a
    specific remedy, recommend review of the relevant process or policy rather
    than that specific remedy.

15. Recommendations concerning guest-priority policies, access allocation,
    pricing changes, complimentary amenities, discounts, packages, or service
    design should normally be framed as managerial review/evaluation unless
    the broader evidence clearly supports a specific intervention.

16. If capital expenditure may eventually be required but the evidence
    is insufficient to determine scope, recommend assessment first.

17. Safety evidence may justify a higher managerial priority even when
    complaint frequency is limited.

18. Supporting evidence must be traceable to this case.

19. For each priority, populate evidence_basis using the evidence supplied
    in managerial_evidence and complaint_evidence.

20. review_examples_used should reflect how many concrete complaint examples
    you actually relied on when reasoning about that priority.

21. relevant_trend_signals should contain only signals that are genuinely
    relevant to the priority.

22. Confidence must reflect evidential support for the recommendation,
    not expected financial success.

23. Use action_horizon as follows:

    immediate:
    safety, serious maintenance, or issues requiring prompt managerial review.

    short_term:
    operational or policy/process issues that should be reviewed soon.

    medium_term:
    structural issues requiring assessment, planning, or possible investment.

    monitor:
    issues where evidence is weak, improving, or insufficient for immediate
    intervention.

24. Do not describe the diagnostic_score as a managerial priority score.

25. Do not generalize isolated complaints beyond the evidence.

26. If a recommendation requires further technical, financial, or operational
    analysis, set requires_further_assessment = true.

27. Keep recommendations actionable but evidence-bounded.

28. If a guest says that hotel guests "should" receive preferential treatment,
    treat that statement as evidence of perceived access frustration, not as
    evidence that guest-priority allocation is the correct operational policy.

29. If a guest says that a specific amenity should be free or included,
    treat this as evidence of an expectation/value gap, not as proof that the
    hotel should provide that amenity for free.

30. Prefer wording such as:
    - "review the reservation/access policy";
    - "evaluate whether guest-priority mechanisms are appropriate";
    - "assess whether current inclusions align with guest expectations";
    - "review pricing and value positioning";
    rather than directly prescribing a guest-suggested solution.


CASE CONTEXT

{context_json}
"""


# ==========================================================
# OUTPUT VALIDATION
# ==========================================================

def _validate_decision(
    decision: dict[str, Any],
) -> None:
    """
    Performs lightweight logical validation after the
    structured output has been parsed.
    """

    priorities = decision.get(
        "priorities",
        []
    )

    if len(priorities) > 5:
        raise ValueError(
            "Decision agent returned more than "
            "5 priorities."
        )

    # ------------------------------------------------------
    # RANKS
    # ------------------------------------------------------

    ranks = [
        priority.get("rank")
        for priority in priorities
    ]

    expected_ranks = list(
        range(
            1,
            len(priorities) + 1
        )
    )

    if ranks != expected_ranks:
        raise ValueError(
            "Priority ranks are not sequential. "
            f"Received: {ranks}"
        )

    # ------------------------------------------------------
    # CONFIDENCE
    # ------------------------------------------------------

    for priority in priorities:

        confidence = priority.get(
            "confidence"
        )

        if confidence is None:
            raise ValueError(
                "A priority is missing confidence."
            )

        if not 0 <= confidence <= 1:
            raise ValueError(
                "Confidence must be between 0 and 1."
            )

    # ------------------------------------------------------
    # EVIDENCE BASIS
    # ------------------------------------------------------

    for priority in priorities:

        evidence_basis = priority.get(
            "evidence_basis"
        )

        if not evidence_basis:
            raise ValueError(
                "A priority is missing evidence_basis."
            )


# ==========================================================
# DECISION AGENT
# ==========================================================

def run_decision_agent(
    agent_context: dict[str, Any],
) -> dict[str, Any]:
    """
    Runs the evidence-informed hotel managerial
    decision agent.

    Returns
    -------
    dict
        Structured managerial decision.
    """

    prompt = _build_user_prompt(
        agent_context
    )

    response = client.responses.create(
        model=AZURE_OPENAI_MODEL,

        instructions=SYSTEM_PROMPT,

        input=prompt,

        text={
            "format": {
                "type": "json_schema",
                "name": (
                    "hotel_managerial_decision"
                ),
                "schema": DECISION_SCHEMA,
                "strict": True,
            }
        },
    )

    raw = response.output_text

    if not raw:
        raise ValueError(
            "The decision agent returned "
            "an empty response."
        )

    try:

        decision = json.loads(
            raw
        )

    except json.JSONDecodeError as error:

        raise ValueError(
            "The decision agent did not "
            "return valid JSON."
        ) from error

    _validate_decision(
        decision
    )

    return decision