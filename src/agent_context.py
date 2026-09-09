# src/agent_context.py

from __future__ import annotations

from typing import Any


# ==========================================================
# CONFIGURAÇÃO
# ==========================================================

PERFORMANCE_METRICS = [
    "REVPAR_WEEK",
    "REVPOR_WEEK",
    "TREVPAR_WEEK",
    "TAXA_OCUPACAO",
    "AVG_RATING",
    "SENTIMENT_SCORE_AVG",
]


EXPERIENCE_METRICS = [
    "EXPECTATION_EXPERIENCE_GAP_AVG",
    "SEMANTIC_MISALIGNMENT_SCORE_AVG",
    "EXPECTATION_VIOLATION_RATIO",
    "EMOTION_ENTROPY",
    "EXPECTATION_SHARE",
    "HIGH_MISALIGNMENT_SHARE",
    "PERC_REVIEWS_NEGATIVE",
    "PERC_OPERATIONAL_COMPLAINTS",
    "POLARIZATION_INDEX",
    "COMPLEXITY_INDEX",
]


# ==========================================================
# FUNÇÕES AUXILIARES
# ==========================================================

def _round_value(
    value: Any,
    decimals: int = 4
) -> Any:
    """
    Arredonda floats para reduzir ruído no contexto
    enviado ao LLM.
    """

    if isinstance(value, float):
        return round(
            value,
            decimals
        )

    return value


def _classify_evidence_pattern(
    positive: int,
    negative: int,
    has_expectation_violation: bool,
    is_emerging_negative: bool
) -> str:
    """
    Classifica a configuração da evidência de cada
    área gerencial.

    Possíveis valores:

    - net_negative
        Há mais menções negativas do que positivas.

    - positive_with_friction
        A área é predominantemente positiva, mas existe
        evidência de expectativa violada e/ou problema
        negativo emergente.

    - predominantly_positive
        A área é predominantemente positiva sem sinais
        negativos estruturais adicionais.

    - mixed
        Evidência positiva e negativa equilibrada.
    """

    if negative > positive:
        return "net_negative"

    if (
        positive > negative
        and (
            has_expectation_violation
            or is_emerging_negative
        )
    ):
        return "positive_with_friction"

    if positive > negative:
        return "predominantly_positive"

    return "mixed"


# ==========================================================
# CONTEXTO DAS MÉTRICAS
# ==========================================================

def _build_metric_context(
    case: dict[str, Any],
    metrics: list[str]
) -> list[dict[str, Any]]:
    """
    Extrai apenas a informação temporal relevante
    para interpretação pelo decision agent.
    """

    trends = case.get(
        "trends",
        {}
    )

    output = []

    for metric in metrics:

        if metric not in trends:
            continue

        trend = trends[
            metric
        ]

        output.append(
            {
                "metric": metric,

                "direction": trend.get(
                    "direction"
                ),

                "trend_strength": trend.get(
                    "trend_strength"
                ),

                "first_value": _round_value(
                    trend.get(
                        "first_value"
                    )
                ),

                "last_value": _round_value(
                    trend.get(
                        "last_value"
                    )
                ),

                "absolute_change": _round_value(
                    trend.get(
                        "absolute_change"
                    )
                ),

                "percentage_change": _round_value(
                    trend.get(
                        "percentage_change"
                    ),
                    decimals=2
                ),

                "direction_consistency": _round_value(
                    trend.get(
                        "direction_consistency"
                    ),
                    decimals=3
                ),

                "recent_movement": trend.get(
                    "recent_movement"
                ),

                "recent_absolute_change": _round_value(
                    trend.get(
                        "recent_absolute_change"
                    )
                ),

                "recent_reversal": trend.get(
                    "recent_reversal"
                ),

                "n_observations": trend.get(
                    "n_observations"
                ),
            }
        )

    return output


# ==========================================================
# EVIDÊNCIA GERENCIAL
# ==========================================================

def _build_managerial_evidence(
    case: dict[str, Any],
    top_n: int = 10
) -> list[dict[str, Any]]:
    """
    Converte as categorias provenientes do Knowledge Base
    numa representação orientada à decisão.
    """

    categories = (
        case
        .get(
            "topic_summary",
            {}
        )
        .get(
            "managerial_categories",
            []
        )
    )

    output = []

    for row in categories:

        positive = int(
            row.get(
                "positive_review_mentions",
                0
            )
        )

        negative = int(
            row.get(
                "negative_review_mentions",
                0
            )
        )

        total_mentions = (
            positive
            + negative
        )

        positive_share = (
            positive / total_mentions
            if total_mentions > 0
            else 0.0
        )

        negative_share = (
            negative / total_mentions
            if total_mentions > 0
            else 0.0
        )

        violated_expectation_weeks = int(
            row.get(
                "violated_expectation_weeks",
                0
            )
        )

        emerging_negative_weeks = int(
            row.get(
                "emerging_negative_weeks",
                0
            )
        )

        has_expectation_violation = (
            violated_expectation_weeks > 0
        )

        is_emerging_negative = (
            emerging_negative_weeks > 0
        )

        evidence_pattern = (
            _classify_evidence_pattern(
                positive=positive,
                negative=negative,
                has_expectation_violation=(
                    has_expectation_violation
                ),
                is_emerging_negative=(
                    is_emerging_negative
                ),
            )
        )

        output.append(
            {
                "area": row.get(
                    "label"
                ),

                "managerial_domain": row.get(
                    "managerial_domain"
                ),

                "default_priority": row.get(
                    "default_priority"
                ),

                # ------------------------------------------
                # Review evidence
                # ------------------------------------------

                "review_mentions_total": int(
                    row.get(
                        "review_mentions_total",
                        0
                    )
                ),

                "positive_mentions": positive,

                "negative_mentions": negative,

                "net_mentions": (
                    positive
                    - negative
                ),

                "positive_share": round(
                    positive_share,
                    3
                ),

                "negative_share": round(
                    negative_share,
                    3
                ),

                # ------------------------------------------
                # Temporal / experiential evidence
                # ------------------------------------------

                "experience_weeks": int(
                    row.get(
                        "experience_weeks",
                        0
                    )
                ),

                "expectation_weeks": int(
                    row.get(
                        "expectation_weeks",
                        0
                    )
                ),

                "violated_expectation_weeks": (
                    violated_expectation_weeks
                ),

                "emerging_negative_weeks": (
                    emerging_negative_weeks
                ),

                "has_expectation_violation": (
                    has_expectation_violation
                ),

                "is_emerging_negative": (
                    is_emerging_negative
                ),

                # ------------------------------------------
                # Padrão de evidência
                # ------------------------------------------

                "evidence_pattern": (
                    evidence_pattern
                ),

                # ------------------------------------------
                # Proveniência
                # ------------------------------------------

                "matched_topics": row.get(
                    "matched_raw_topics",
                    []
                ),

                "source_columns": row.get(
                    "source_columns",
                    []
                ),
            }
        )

    return output[
        :top_n
    ]


# ==========================================================
# STRENGTHS
# ==========================================================

def _identify_strengths(
    managerial_evidence: list[dict[str, Any]],
    max_strengths: int = 5
) -> list[dict[str, Any]]:
    """
    Identifica áreas predominantemente positivas.

    Uma área pode continuar a ser uma força mesmo que
    apresente friction points específicos.
    """

    candidates = []

    for row in managerial_evidence:

        positive = row[
            "positive_mentions"
        ]

        negative = row[
            "negative_mentions"
        ]

        total = (
            positive
            + negative
        )

        if total == 0:
            continue

        positive_share = (
            positive / total
        )

        if (
            positive > negative
            and positive >= 5
            and positive_share >= 0.65
        ):

            if (
                row[
                    "evidence_pattern"
                ]
                == "positive_with_friction"
            ):
                strength_type = (
                    "strength_with_friction"
                )

            else:
                strength_type = (
                    "clear_strength"
                )

            candidates.append(
                {
                    "area": row[
                        "area"
                    ],

                    "positive_mentions": (
                        positive
                    ),

                    "negative_mentions": (
                        negative
                    ),

                    "positive_share": round(
                        positive_share,
                        3
                    ),

                    "strength_type": (
                        strength_type
                    ),

                    "has_expectation_violation": (
                        row[
                            "has_expectation_violation"
                        ]
                    ),

                    "is_emerging_negative": (
                        row[
                            "is_emerging_negative"
                        ]
                    ),
                }
            )

    candidates.sort(
        key=lambda row: (
            row[
                "strength_type"
            ] != "clear_strength",
            -row[
                "positive_share"
            ],
            -row[
                "positive_mentions"
            ],
        )
    )

    return candidates[
        :max_strengths
    ]


# ==========================================================
# VULNERABILITIES
# ==========================================================

def _identify_vulnerabilities(
    managerial_evidence: list[dict[str, Any]],
    max_vulnerabilities: int = 6
) -> list[dict[str, Any]]:
    """
    Identifica áreas que merecem atenção do agente.

    IMPORTANTE:
    diagnostic_score serve apenas para selecionar e ordenar
    evidência que entra no contexto do LLM.

    NÃO representa a prioridade gerencial final.
    """

    candidates = []

    for row in managerial_evidence:

        negative = row[
            "negative_mentions"
        ]

        positive = row[
            "positive_mentions"
        ]

        negative_share = row[
            "negative_share"
        ]

        if negative == 0:
            continue

        score_components = {}

        # --------------------------------------------------
        # 1. Frequência negativa
        # --------------------------------------------------

        score_components[
            "negative_frequency"
        ] = min(
            negative,
            10
        )

        # --------------------------------------------------
        # 2. Evidência líquida negativa
        # --------------------------------------------------

        score_components[
            "net_negative"
        ] = (
            3
            if negative > positive
            else 0
        )

        # --------------------------------------------------
        # 3. Share de evidência negativa
        # --------------------------------------------------

        if negative_share >= 0.60:

            score_components[
                "negative_share"
            ] = 2

        elif negative_share >= 0.50:

            score_components[
                "negative_share"
            ] = 1

        else:

            score_components[
                "negative_share"
            ] = 0

        # --------------------------------------------------
        # 4. Expectation violation
        # --------------------------------------------------

        score_components[
            "expectation_violation"
        ] = (
            4
            if row[
                "has_expectation_violation"
            ]
            else 0
        )

        # --------------------------------------------------
        # 5. Emerging negative
        # --------------------------------------------------

        score_components[
            "emerging_negative"
        ] = (
            3
            if row[
                "is_emerging_negative"
            ]
            else 0
        )

        # --------------------------------------------------
        # 6. Prioridade estrutural da taxonomia
        # --------------------------------------------------

        score_components[
            "default_priority"
        ] = (
            2
            if row[
                "default_priority"
            ] == "high"
            else 0
        )

        diagnostic_score = sum(
            score_components.values()
        )

        candidates.append(
            {
                **row,

                "diagnostic_score": (
                    diagnostic_score
                ),

                "diagnostic_score_components": (
                    score_components
                ),
            }
        )

    candidates.sort(
        key=lambda row: (
            -row[
                "diagnostic_score"
            ],
            -row[
                "negative_mentions"
            ],
        )
    )

    return candidates[
        :max_vulnerabilities
    ]


# ==========================================================
# COMPLAINT EVIDENCE
# ==========================================================

def _build_complaint_evidence(
    case: dict[str, Any],
    max_complaints: int = 12
) -> list[dict[str, Any]]:
    """
    Prepara exemplos concretos de reviews para o agente.

    complaint e positive_context devem estar em inglês
    quando as traduções estiverem disponíveis.
    """

    complaints = case.get(
        "complaints",
        []
    )

    output = []

    for review in complaints[
        :max_complaints
    ]:

        output.append(
            {
                "review_date": review.get(
                    "review_date"
                ),

                "rating": review.get(
                    "rating"
                ),

                "original_language": review.get(
                    "language"
                ),

                "complaint": review.get(
                    "complaint_text"
                ),

                "positive_context": review.get(
                    "liked_context"
                ),

                "country": review.get(
                    "country"
                ),

                "stay_type": review.get(
                    "stay_type"
                ),
            }
        )

    return output


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def build_agent_context(
    case: dict[str, Any],
    top_n_categories: int = 10,
    max_vulnerabilities: int = 6,
    max_strengths: int = 5,
    max_complaints: int = 12,
) -> dict[str, Any]:
    """
    Converte o case completo numa representação compacta
    e explicitamente orientada à decisão.

    Esta função NÃO toma a decisão final.

    O objetivo é selecionar, estruturar e contextualizar
    evidência para posterior análise pelo decision agent.
    """

    managerial_evidence = (
        _build_managerial_evidence(
            case=case,
            top_n=top_n_categories,
        )
    )

    vulnerabilities = (
        _identify_vulnerabilities(
            managerial_evidence=(
                managerial_evidence
            ),
            max_vulnerabilities=(
                max_vulnerabilities
            ),
        )
    )

    strengths = (
        _identify_strengths(
            managerial_evidence=(
                managerial_evidence
            ),
            max_strengths=(
                max_strengths
            ),
        )
    )

    context = {

        # ==================================================
        # CONTEXT METADATA
        # ==================================================

        "context_metadata": {

            "purpose": (
                "Evidence preparation for managerial "
                "decision support."
            ),

            "diagnostic_score_role": (
                "The diagnostic score is a transparent "
                "heuristic used only to select evidence "
                "for the decision agent. It is not the "
                "final managerial priority score."
            ),

            "evidence_pattern_definitions": {

                "net_negative": (
                    "Negative review evidence exceeds "
                    "positive evidence."
                ),

                "positive_with_friction": (
                    "The area is predominantly positive "
                    "but shows expectation violations "
                    "and/or emerging negative evidence."
                ),

                "predominantly_positive": (
                    "Positive review evidence dominates "
                    "without additional structural "
                    "warning signals."
                ),

                "mixed": (
                    "Positive and negative evidence are "
                    "approximately balanced."
                ),
            },
        },

        # ==================================================
        # CASE
        # ==================================================

        "case": case.get(
            "case_metadata",
            {}
        ),

        # ==================================================
        # HOTEL
        # ==================================================

        "hotel": case.get(
            "hotel_profile",
            {}
        ),

        # ==================================================
        # LATEST CONTEXT
        # ==================================================

        "latest_context": case.get(
            "latest_context",
            {}
        ),

        # ==================================================
        # BUSINESS PERFORMANCE
        # ==================================================

        "business_performance": (
            _build_metric_context(
                case=case,
                metrics=(
                    PERFORMANCE_METRICS
                ),
            )
        ),

        # ==================================================
        # EXPERIENCE INDICATORS
        # ==================================================

        "experience_indicators": (
            _build_metric_context(
                case=case,
                metrics=(
                    EXPERIENCE_METRICS
                ),
            )
        ),

        # ==================================================
        # PRELIMINARY SIGNALS
        # ==================================================

        "preliminary_signals": (
            case.get(
                "preliminary_signals",
                []
            )
        ),

        # ==================================================
        # COMPLETE MANAGERIAL EVIDENCE
        # ==================================================

        "managerial_evidence": (
            managerial_evidence
        ),

        # ==================================================
        # DIAGNOSTIC VULNERABILITIES
        # ==================================================

        "diagnostic_vulnerabilities": (
            vulnerabilities
        ),

        # ==================================================
        # STRENGTHS
        # ==================================================

        "strengths": (
            strengths
        ),

        # ==================================================
        # REVIEW EVIDENCE
        # ==================================================

        "complaint_evidence": (
            _build_complaint_evidence(
                case=case,
                max_complaints=(
                    max_complaints
                ),
            )
        ),

        # ==================================================
        # REVIEW SAMPLE METADATA
        # ==================================================

        "complaint_sample_metadata": (
            case.get(
                "complaint_metadata",
                {}
            )
        ),

        # ==================================================
        # INTERPRETATION RULES FOR THE LLM
        # ==================================================

        "interpretation_rules": [

            (
                "Global trend direction and the most "
                "recent weekly movement must always be "
                "interpreted separately."
            ),

            (
                "If recent_reversal is true, do not "
                "describe the metric as continuously "
                "worsening or continuously improving."
            ),

            (
                "Strong current financial performance "
                "does not rule out emerging customer "
                "experience vulnerabilities."
            ),

            (
                "A net-negative evidence pattern indicates "
                "that negative lexical evidence exceeds "
                "positive lexical evidence, but it does "
                "not establish causality."
            ),

            (
                "A positive-with-friction pattern should "
                "not be described as a generally poor "
                "area. Preserve the underlying strength "
                "while addressing the specific friction."
            ),

            (
                "Positive and negative review mentions "
                "are diagnostic lexical evidence and "
                "must not be interpreted as causal effects."
            ),

            (
                "Expectation violations increase the "
                "diagnostic relevance of an issue but "
                "do not automatically imply that the "
                "issue requires capital investment."
            ),

            (
                "Emerging negative topics indicate "
                "recent or newly visible friction and "
                "should be distinguished from persistent "
                "structural weaknesses."
            ),

            (
                "Selected complaints are diagnostic "
                "examples and are not statistically "
                "representative of the complete review "
                "distribution."
            ),

            (
                "The diagnostic_score is only an evidence "
                "selection heuristic. The final managerial "
                "priority must be reasoned independently "
                "by the decision agent."
            ),

            (
                "Recommendations should distinguish "
                "operational actions from structural or "
                "capital-intensive interventions."
            ),

            (
                "Clearly identified strengths should be "
                "preserved and should not be converted "
                "into problems solely because isolated "
                "negative evidence exists."
            ),
        ],
    }

    return context