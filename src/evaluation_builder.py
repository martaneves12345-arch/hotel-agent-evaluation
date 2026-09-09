# src/evaluation_builder.py

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


# ==========================================================
# CONFIGURATION
# ==========================================================

EVALUATION_VERSION = "human_eval_v2"

DEFAULT_RANDOM_SEED = 20260909

SYSTEMS_FOR_HUMAN_EVALUATION = [
    "generic_llm",
    "evidence_informed",
]


# ==========================================================
# EVALUATOR-VISIBLE METRICS
# ==========================================================

# IMPORTANT:
# Only basic hotel-performance information is shown
# to the evaluator.
#
# Treatment-specific experiential indicators are
# intentionally excluded to preserve blinding.

EVALUATOR_METRICS = [
    "WEEK_YEAR",
    "week_date",
    "REVPAR_WEEK",
    "REVPOR_WEEK",
    "TREVPAR_WEEK",
    "TAXA_OCUPACAO",
    "N_REVIEWS",
    "AVG_RATING",
]


# ==========================================================
# HELPERS
# ==========================================================

def _load_json(
    path: Path,
) -> dict[str, Any]:
    """
    Loads a JSON file.
    """

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def _find_single_json(
    directory: Path,
    pilot_id: str,
) -> Path:
    """
    Finds exactly one JSON file for a given pilot case.
    """

    matches = list(
        directory.glob(
            f"{pilot_id}_*.json"
        )
    )

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No JSON found for {pilot_id} "
            f"in {directory}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple JSON files found for "
            f"{pilot_id} in {directory}: "
            f"{matches}"
        )

    return matches[0]


# ==========================================================
# COMMON CASE INFORMATION
# ==========================================================

def _build_common_case_information(
    case: dict[str, Any],
    max_reviews: int = 20,
) -> dict[str, Any]:
    """
    Builds a neutral case representation that can be shown
    to evaluators for BOTH systems.

    The evaluator receives:

    - evaluation window;
    - basic hotel information;
    - basic operational / financial KPIs;
    - recent customer reviews.

    The evaluator does NOT receive treatment-specific
    engineered evidence such as:

    - preliminary signals;
    - diagnostic scores;
    - evidence patterns;
    - expectation-experience gap;
    - semantic misalignment;
    - expectation violations;
    - emotional complexity;
    - emerging-negative signals;
    - agent-context evidence structures.
    """

    metadata = case.get(
        "case_metadata",
        {},
    )

    hotel = case.get(
        "hotel_profile",
        {},
    )

    weekly_history = case.get(
        "weekly_history",
        [],
    )

    review_sample = case.get(
        "review_sample",
        [],
    )[:max_reviews]

    # ------------------------------------------------------
    # HOTEL PROFILE
    # ------------------------------------------------------

    hotel_information = {
        "stars": hotel.get(
            "stars"
        ),
        "region": (
            hotel.get("nuts2")
            or hotel.get("region")
        ),
        "number_of_rooms": (
            hotel.get("n_rooms")
            or hotel.get("number_of_rooms")
        ),
    }

    # ------------------------------------------------------
    # CLEAN WEEKLY HISTORY
    # ------------------------------------------------------

    clean_weekly_history = []

    for week in weekly_history:

        clean_weekly_history.append(
            {
                metric: week.get(metric)
                for metric in EVALUATOR_METRICS
            }
        )

    # ------------------------------------------------------
    # RECENT REVIEWS
    # ------------------------------------------------------

    reviews = []

    for review in review_sample:

        reviews.append(
            {
                "review_date": review.get(
                    "review_date"
                ),
                "rating": review.get(
                    "rating"
                ),
                "liked": review.get(
                    "liked"
                ),
                "disliked": review.get(
                    "disliked"
                ),
                "country": review.get(
                    "country"
                ),
                "stay_type": review.get(
                    "stay_type"
                ),
            }
        )

    # ------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------

    return {
        "case_id": metadata.get(
            "case_id"
        ),

        "decision_week": metadata.get(
            "decision_week"
        ),

        "window_start": metadata.get(
            "window_start"
        ),

        "window_end": metadata.get(
            "window_end"
        ),

        "lookback_weeks": metadata.get(
            "lookback_weeks_requested"
        ),

        "hotel_information": (
            hotel_information
        ),

        "weekly_history": (
            clean_weekly_history
        ),

        "recent_reviews": (
            reviews
        ),
    }


# ==========================================================
# CLEAN DECISION FOR HUMAN EVALUATION
# ==========================================================

def _clean_decision_output(
    decision: dict[str, Any],
) -> dict[str, Any]:
    """
    Removes system-identifying or treatment-specific
    information while preserving the managerial content
    required for human evaluation.

    Fields intentionally excluded:

    - evidence_pattern;
    - evidence_basis;
    - positive_counterevidence;
    - supporting_evidence;
    - confidence;
    - safety_relevance;
    - requires_further_assessment;
    - monitoring_indicators;
    - system metadata;
    - experiment metadata.
    """

    cleaned = {
        "overall_risk": (
            decision.get(
                "overall_risk"
            )
        ),

        "overall_risk_rationale": (
            decision.get(
                "overall_risk_rationale"
            )
        ),

        "executive_summary": (
            decision.get(
                "executive_summary"
            )
        ),

        "priorities": [],

        "strengths_to_preserve": (
            decision.get(
                "strengths_to_preserve",
                [],
            )
        ),

        "limitations": (
            decision.get(
                "limitations",
                [],
            )
        ),
    }

    # ------------------------------------------------------
    # PRIORITIES
    # ------------------------------------------------------

    for priority in decision.get(
        "priorities",
        [],
    ):

        cleaned[
            "priorities"
        ].append(
            {
                "rank": priority.get(
                    "rank"
                ),

                "area": priority.get(
                    "area"
                ),

                "managerial_priority": (
                    priority.get(
                        "managerial_priority"
                    )
                ),

                "action_horizon": (
                    priority.get(
                        "action_horizon"
                    )
                ),

                "intervention_type": (
                    priority.get(
                        "intervention_type"
                    )
                ),

                "problem": (
                    priority.get(
                        "problem"
                    )
                ),

                "recommended_action": (
                    priority.get(
                        "recommended_action"
                    )
                ),

                "rationale": (
                    priority.get(
                        "rationale"
                    )
                ),
            }
        )

    return cleaned


# ==========================================================
# RANDOMIZATION
# ==========================================================

def _randomize_system_labels(
    pilot_ids: list[str],
    random_seed: int,
) -> pd.DataFrame:
    """
    Creates a balanced blinded assignment.

    For an even number of cases:
    - Generic appears as Decision A in exactly half;
    - Evidence-Informed appears as Decision A in exactly half.

    Assignment is randomized across cases using a fixed
    random seed for reproducibility.
    """

    rng = random.Random(
        random_seed
    )

    n_cases = len(
        pilot_ids
    )

    assignments = []

    # ------------------------------------------------------
    # BALANCED A/B ASSIGNMENTS
    # ------------------------------------------------------

    for i in range(
        n_cases
    ):

        if i % 2 == 0:

            assignments.append(
                (
                    "generic_llm",
                    "evidence_informed",
                )
            )

        else:

            assignments.append(
                (
                    "evidence_informed",
                    "generic_llm",
                )
            )

    # Randomize which pilot receives which assignment.
    rng.shuffle(
        assignments
    )

    records = []

    for pilot_id, assignment in zip(
        pilot_ids,
        assignments,
    ):

        records.append(
            {
                "pilot_id": (
                    pilot_id
                ),

                "decision_a_system": (
                    assignment[0]
                ),

                "decision_b_system": (
                    assignment[1]
                ),
            }
        )

    return pd.DataFrame(
        records
    )


# ==========================================================
# VALIDATION
# ==========================================================

def _validate_blinding_key(
    blinding_key: pd.DataFrame,
) -> None:
    """
    Performs basic validation of the randomized A/B mapping.
    """

    if blinding_key[
        "pilot_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate pilot IDs found in "
            "the blinding key."
        )

    for _, row in (
        blinding_key.iterrows()
    ):

        systems = {
            row[
                "decision_a_system"
            ],
            row[
                "decision_b_system"
            ],
        }

        if systems != {
            "generic_llm",
            "evidence_informed",
        }:

            raise ValueError(
                "Invalid system assignment "
                f"for {row['pilot_id']}."
            )


def _validate_clean_decision(
    decision: dict[str, Any],
) -> None:
    """
    Checks that treatment-specific output fields have not
    leaked into the human-evaluation representation.
    """

    forbidden_top_level = {
        "system",
        "system_version",
        "model",
        "experiment_metadata",
        "monitoring_indicators",
    }

    leaked_top_level = (
        forbidden_top_level
        & set(
            decision.keys()
        )
    )

    if leaked_top_level:

        raise ValueError(
            "System-identifying fields leaked into "
            f"evaluation output: "
            f"{sorted(leaked_top_level)}"
        )

    forbidden_priority_fields = {
        "evidence_pattern",
        "evidence_basis",
        "positive_counterevidence",
        "supporting_evidence",
        "confidence",
        "safety_relevance",
        "requires_further_assessment",
    }

    for priority in decision.get(
        "priorities",
        [],
    ):

        leaked_priority_fields = (
            forbidden_priority_fields
            & set(
                priority.keys()
            )
        )

        if leaked_priority_fields:

            raise ValueError(
                "Treatment-specific priority fields "
                "leaked into evaluation output: "
                f"{sorted(leaked_priority_fields)}"
            )


# ==========================================================
# BUILD HUMAN EVALUATION PACKAGE
# ==========================================================

def build_human_evaluation_package(
    experiment_dir: str | Path,
    output_dir: str | Path,
    random_seed: int = DEFAULT_RANDOM_SEED,
    max_reviews: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds the blinded human-evaluation package.

    Human evaluation compares:

        Generic LLM
        vs.
        Evidence-Informed Agent

    The Frequency Baseline is intentionally excluded from
    the qualitative A/B evaluation because its output
    structure is not directly comparable.

    Returns
    -------
    evaluation_manifest:
        Manifest containing the blinded evaluation cases.

    blinding_key:
        PRIVATE mapping between Decision A / Decision B and
        the underlying systems.
    """

    experiment_dir = Path(
        experiment_dir
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    blinded_cases_dir = (
        output_dir
        / "blinded_cases"
    )

    blinded_cases_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # LOAD FROZEN PILOT MANIFEST
    # ------------------------------------------------------

    pilot_manifest_path = (
        experiment_dir
        / "pilot_manifest.csv"
    )

    if not pilot_manifest_path.exists():

        raise FileNotFoundError(
            "pilot_manifest.csv not found at "
            f"{pilot_manifest_path}"
        )

    pilot_manifest = pd.read_csv(
        pilot_manifest_path
    )

    pilot_manifest = (
        pilot_manifest
        .sort_values(
            "pilot_id"
        )
        .reset_index(
            drop=True
        )
    )

    required_columns = {
        "pilot_id",
        "case_id",
    }

    missing_columns = (
        required_columns
        - set(
            pilot_manifest.columns
        )
    )

    if missing_columns:

        raise ValueError(
            "pilot_manifest is missing required "
            f"columns: {sorted(missing_columns)}"
        )

    pilot_ids = (
        pilot_manifest[
            "pilot_id"
        ]
        .astype(str)
        .tolist()
    )

    # ------------------------------------------------------
    # CREATE RANDOMIZED BLINDING KEY
    # ------------------------------------------------------

    blinding_key = (
        _randomize_system_labels(
            pilot_ids=pilot_ids,
            random_seed=random_seed,
        )
    )

    _validate_blinding_key(
        blinding_key
    )

    evaluation_records = []

    # ======================================================
    # BUILD EACH BLINDED CASE
    # ======================================================

    for _, pilot_row in (
        pilot_manifest.iterrows()
    ):

        pilot_id = str(
            pilot_row[
                "pilot_id"
            ]
        )

        expected_case_id = str(
            pilot_row[
                "case_id"
            ]
        )

        # --------------------------------------------------
        # LOCATE FILES
        # --------------------------------------------------

        case_path = (
            _find_single_json(
                experiment_dir
                / "cases",
                pilot_id,
            )
        )

        generic_path = (
            _find_single_json(
                experiment_dir
                / "generic_llm",
                pilot_id,
            )
        )

        evidence_path = (
            _find_single_json(
                experiment_dir
                / "evidence_informed",
                pilot_id,
            )
        )

        # --------------------------------------------------
        # LOAD ORIGINAL FROZEN OUTPUTS
        # --------------------------------------------------

        case = _load_json(
            case_path
        )

        generic = _load_json(
            generic_path
        )

        evidence = _load_json(
            evidence_path
        )

        # --------------------------------------------------
        # CASE-ID CONSISTENCY
        # --------------------------------------------------

        actual_case_id = (
            case
            .get(
                "case_metadata",
                {},
            )
            .get(
                "case_id"
            )
        )

        if (
            actual_case_id
            != expected_case_id
        ):

            raise ValueError(
                f"Case mismatch for {pilot_id}: "
                f"{actual_case_id} != "
                f"{expected_case_id}"
            )

        # --------------------------------------------------
        # BUILD COMMON NEUTRAL CASE INFORMATION
        # --------------------------------------------------

        common_case = (
            _build_common_case_information(
                case=case,
                max_reviews=max_reviews,
            )
        )

        # --------------------------------------------------
        # CLEAN ORIGINAL DECISION OUTPUTS
        # --------------------------------------------------

        generic_clean = (
            _clean_decision_output(
                generic
            )
        )

        evidence_clean = (
            _clean_decision_output(
                evidence
            )
        )

        _validate_clean_decision(
            generic_clean
        )

        _validate_clean_decision(
            evidence_clean
        )

        system_outputs = {
            "generic_llm": (
                generic_clean
            ),
            "evidence_informed": (
                evidence_clean
            ),
        }

        # --------------------------------------------------
        # GET A/B MAPPING
        # --------------------------------------------------

        mapping = (
            blinding_key[
                blinding_key[
                    "pilot_id"
                ]
                == pilot_id
            ]
            .iloc[0]
        )

        system_a = (
            mapping[
                "decision_a_system"
            ]
        )

        system_b = (
            mapping[
                "decision_b_system"
            ]
        )

        # --------------------------------------------------
        # BUILD BLINDED CASE
        # --------------------------------------------------

        blinded_case = {
            "evaluation_version": (
                EVALUATION_VERSION
            ),

            "evaluation_case_id": (
                pilot_id
            ),

            "case_information": (
                common_case
            ),

            "decision_a": (
                system_outputs[
                    system_a
                ]
            ),

            "decision_b": (
                system_outputs[
                    system_b
                ]
            ),
        }

        # --------------------------------------------------
        # SAVE BLINDED JSON
        # --------------------------------------------------

        output_path = (
            blinded_cases_dir
            / f"{pilot_id}.json"
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                blinded_case,
                file,
                ensure_ascii=False,
                indent=2,
            )

        # --------------------------------------------------
        # MANIFEST RECORD
        # --------------------------------------------------

        evaluation_records.append(
            {
                "evaluation_case_id": (
                    pilot_id
                ),

                "case_id": (
                    expected_case_id
                ),

                "stratum": (
                    pilot_row.get(
                        "stratum"
                    )
                ),

                "blinded_file": (
                    str(
                        output_path
                    )
                ),
            }
        )

    # ======================================================
    # SAVE EVALUATION MANIFEST
    # ======================================================

    evaluation_manifest = (
        pd.DataFrame(
            evaluation_records
        )
    )

    evaluation_manifest_path = (
        output_dir
        / "evaluation_manifest.csv"
    )

    evaluation_manifest.to_csv(
        evaluation_manifest_path,
        index=False,
    )

    # ======================================================
    # SAVE PRIVATE BLINDING KEY
    # ======================================================

    private_key_path = (
        output_dir
        / "PRIVATE_blinding_key.csv"
    )

    blinding_key.to_csv(
        private_key_path,
        index=False,
    )

    # ======================================================
    # FINAL CHECKS
    # ======================================================

    if len(
        evaluation_manifest
    ) != len(
        pilot_manifest
    ):

        raise ValueError(
            "Evaluation package does not contain "
            "the same number of cases as the "
            "pilot manifest."
        )

    print(
        "\n"
        "=================================================="
    )

    print(
        "HUMAN EVALUATION PACKAGE CREATED"
    )

    print(
        "=================================================="
    )

    print(
        f"Evaluation version: "
        f"{EVALUATION_VERSION}"
    )

    print(
        f"Cases: "
        f"{len(evaluation_manifest)}"
    )

    print(
        f"Reviews shown per case: "
        f"up to {max_reviews}"
    )

    print(
        "\nDecision A assignment:"
    )

    print(
        blinding_key[
            "decision_a_system"
        ]
        .value_counts()
    )

    print(
        "\nDecision B assignment:"
    )

    print(
        blinding_key[
            "decision_b_system"
        ]
        .value_counts()
    )

    print(
        "\nBlinded cases saved to:"
    )

    print(
        blinded_cases_dir
    )

    print(
        "\nPRIVATE key saved to:"
    )

    print(
        private_key_path
    )

    return (
        evaluation_manifest,
        blinding_key,
    )