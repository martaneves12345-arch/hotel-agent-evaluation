# src/evaluation_builder.py

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from src.review_summarizer import (
    build_neutral_review_summary,
    shorten_recommendation,
)

# ==========================================================
# CONFIGURATION
# ==========================================================

EVALUATION_VERSION = "human_eval_v3"

DEFAULT_RANDOM_SEED = 20260909

SYSTEMS_FOR_HUMAN_EVALUATION = [
    "generic_llm",
    "evidence_informed",
]

EVALUATOR_METRICS = [
    "WEEK_YEAR",
    "week_date",
    "REVPAR_WEEK",
    "TAXA_OCUPACAO",
    "N_REVIEWS",
    "AVG_RATING",
]


# ==========================================================
# HELPERS
# ==========================================================

def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _find_single_json(directory: Path, pilot_id: str) -> Path:
    matches = list(directory.glob(f"{pilot_id}_*.json"))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No JSON found for {pilot_id} in {directory}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple JSON files found for {pilot_id} in {directory}: "
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
    Build the neutral evaluator-visible case information.

    The evaluator sees:
    - evaluation window;
    - compact basic performance history;
    - one neutral summary of the SAME recent review sample.

    The evaluator does NOT see:
    - raw individual reviews;
    - hotel category / region / room count;
    - treatment-specific experiential indicators;
    - diagnostic scores or treatment-specific evidence structures.
    """

    metadata = case.get("case_metadata", {})
    weekly_history = case.get("weekly_history", [])

    clean_weekly_history = []

    for week in weekly_history:
        clean_weekly_history.append(
            {
                metric: week.get(metric)
                for metric in EVALUATOR_METRICS
            }
        )

    review_summary = build_neutral_review_summary(
        case=case,
        max_reviews=max_reviews,
    )

    return {
        "case_id": metadata.get("case_id"),
        "decision_week": metadata.get("decision_week"),
        "window_start": metadata.get("window_start"),
        "window_end": metadata.get("window_end"),
        "lookback_weeks": metadata.get("lookback_weeks_requested"),
        "weekly_history": clean_weekly_history,
        "customer_feedback_summary": review_summary["summary"],

        # Audit metadata; not displayed in the Streamlit interface.
        "review_summary_metadata": {
            "source_reviews_selected": (
                review_summary["source_reviews_selected"]
            ),
            "source_reviews_with_written_feedback": (
                review_summary[
                    "source_reviews_with_written_feedback"
                ]
            ),
            "summary_method": (
                "Neutral LLM summary of the frozen recent review sample; "
                "no additional reviews retrieved."
            ),
        },
    }


# ==========================================================
# SHORT DECISION FOR HUMAN EVALUATION
# ==========================================================

def _build_short_decision_output(
    decision: dict[str, Any],
    max_priorities: int = 3,
) -> dict[str, Any]:
    """
    Build the concise participant-facing representation.

    The original experimental output is NOT modified. This function only
    creates a short display layer for human evaluation.

    Visible:
    - overall risk;
    - top three ranked managerial recommendations.

    Hidden:
    - executive summary;
    - rationale;
    - problem text;
    - supporting evidence;
    - confidence;
    - safety flags;
    - treatment-specific fields;
    - strengths / limitations / monitoring details.
    """

    priorities = sorted(
        decision.get("priorities", []) or [],
        key=lambda item: item.get("rank", 999),
    )[:max_priorities]

    short_priorities = []

    for fallback_rank, priority in enumerate(priorities, start=1):
        rank = priority.get("rank") or fallback_rank
        area = priority.get("area")
        original_action = priority.get("recommended_action")

        short_action = shorten_recommendation(
            area=area,
            recommended_action=original_action,
        )

        short_priorities.append(
            {
                "rank": rank,
                "area": area,
                "recommendation": short_action,
            }
        )

    return {
        "overall_risk": decision.get("overall_risk"),
        "priorities": short_priorities,
    }


# ==========================================================
# RANDOMIZATION
# ==========================================================

def _randomize_system_labels(
    pilot_ids: list[str],
    random_seed: int,
) -> pd.DataFrame:
    """
    Create a balanced blinded A/B assignment.
    """

    rng = random.Random(random_seed)
    n_cases = len(pilot_ids)

    assignments = []

    for i in range(n_cases):
        if i % 2 == 0:
            assignments.append(
                ("generic_llm", "evidence_informed")
            )
        else:
            assignments.append(
                ("evidence_informed", "generic_llm")
            )

    rng.shuffle(assignments)

    records = []

    for pilot_id, assignment in zip(
        pilot_ids,
        assignments,
    ):
        records.append(
            {
                "pilot_id": pilot_id,
                "decision_a_system": assignment[0],
                "decision_b_system": assignment[1],
            }
        )

    return pd.DataFrame(records)


def _validate_blinding_key(
    blinding_key: pd.DataFrame,
) -> None:
    if blinding_key["pilot_id"].duplicated().any():
        raise ValueError("Duplicate pilot IDs found in the blinding key.")

    for _, row in blinding_key.iterrows():
        systems = {
            row["decision_a_system"],
            row["decision_b_system"],
        }

        if systems != {
            "generic_llm",
            "evidence_informed",
        }:
            raise ValueError(
                f"Invalid system assignment for {row['pilot_id']}."
            )


def _validate_short_decision(
    decision: dict[str, Any],
) -> None:
    allowed_top_level = {
        "overall_risk",
        "priorities",
    }

    extra_top_level = set(decision.keys()) - allowed_top_level

    if extra_top_level:
        raise ValueError(
            "Unexpected fields leaked into short evaluation output: "
            f"{sorted(extra_top_level)}"
        )

    for priority in decision.get("priorities", []):
        allowed_priority_fields = {
            "rank",
            "area",
            "recommendation",
        }

        extra_priority_fields = (
            set(priority.keys())
            - allowed_priority_fields
        )

        if extra_priority_fields:
            raise ValueError(
                "Unexpected priority fields leaked into short evaluation "
                f"output: {sorted(extra_priority_fields)}"
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
    Build the blinded v3 human-evaluation package.

    The original frozen Generic LLM and Evidence-Informed outputs are loaded
    from experiment_dir. They are not regenerated.

    A neutral customer-feedback summary is generated once per case from the
    same frozen recent-review sample and is shared by both blinded decisions.
    """

    experiment_dir = Path(experiment_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    blinded_cases_dir = output_dir / "blinded_cases"

    blinded_cases_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pilot_manifest_path = (
        experiment_dir
        / "pilot_manifest.csv"
    )

    if not pilot_manifest_path.exists():
        raise FileNotFoundError(
            "pilot_manifest.csv not found at "
            f"{pilot_manifest_path}"
        )

    pilot_manifest = (
        pd.read_csv(pilot_manifest_path)
        .sort_values("pilot_id")
        .reset_index(drop=True)
    )

    required_columns = {
        "pilot_id",
        "case_id",
    }

    missing_columns = (
        required_columns
        - set(pilot_manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            "pilot_manifest is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    pilot_ids = (
        pilot_manifest["pilot_id"]
        .astype(str)
        .tolist()
    )

    blinding_key = _randomize_system_labels(
        pilot_ids=pilot_ids,
        random_seed=random_seed,
    )

    _validate_blinding_key(blinding_key)

    evaluation_records = []

    for _, pilot_row in pilot_manifest.iterrows():
        pilot_id = str(pilot_row["pilot_id"])
        expected_case_id = str(pilot_row["case_id"])

        case_path = _find_single_json(
            experiment_dir / "cases",
            pilot_id,
        )

        generic_path = _find_single_json(
            experiment_dir / "generic_llm",
            pilot_id,
        )

        evidence_path = _find_single_json(
            experiment_dir / "evidence_informed",
            pilot_id,
        )

        case = _load_json(case_path)
        generic = _load_json(generic_path)
        evidence = _load_json(evidence_path)

        actual_case_id = (
            case
            .get("case_metadata", {})
            .get("case_id")
        )

        if actual_case_id != expected_case_id:
            raise ValueError(
                f"Case mismatch for {pilot_id}: "
                f"{actual_case_id} != {expected_case_id}"
            )

        # One neutral common summary per case.
        common_case = _build_common_case_information(
            case=case,
            max_reviews=max_reviews,
        )

        # Presentation-only shortening; original outputs stay frozen.
        generic_short = _build_short_decision_output(generic)
        evidence_short = _build_short_decision_output(evidence)

        _validate_short_decision(generic_short)
        _validate_short_decision(evidence_short)

        system_outputs = {
            "generic_llm": generic_short,
            "evidence_informed": evidence_short,
        }

        mapping = (
            blinding_key[
                blinding_key["pilot_id"] == pilot_id
            ]
            .iloc[0]
        )

        system_a = mapping["decision_a_system"]
        system_b = mapping["decision_b_system"]

        blinded_case = {
            "evaluation_version": EVALUATION_VERSION,
            "evaluation_case_id": pilot_id,
            "case_information": common_case,
            "decision_a": system_outputs[system_a],
            "decision_b": system_outputs[system_b],
        }

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

        evaluation_records.append(
            {
                "evaluation_case_id": pilot_id,
                "case_id": expected_case_id,
                "stratum": pilot_row.get("stratum"),
                "blinded_file": str(output_path),
                "written_reviews_in_summary_source": (
                    common_case[
                        "review_summary_metadata"
                    ][
                        "source_reviews_with_written_feedback"
                    ]
                ),
            }
        )

    evaluation_manifest = pd.DataFrame(
        evaluation_records
    )

    evaluation_manifest_path = (
        output_dir
        / "evaluation_manifest.csv"
    )

    evaluation_manifest.to_csv(
        evaluation_manifest_path,
        index=False,
    )

    private_key_path = (
        output_dir
        / "PRIVATE_blinding_key.csv"
    )

    blinding_key.to_csv(
        private_key_path,
        index=False,
    )

    if len(evaluation_manifest) != len(pilot_manifest):
        raise ValueError(
            "Evaluation package does not contain the same number of cases "
            "as the pilot manifest."
        )

    print(
        "\n"
        "==================================================\n"
        "HUMAN EVALUATION PACKAGE CREATED\n"
        "=================================================="
    )

    print(f"Evaluation version: {EVALUATION_VERSION}")
    print(f"Cases: {len(evaluation_manifest)}")
    print(
        "Review-summary source: up to "
        f"{max_reviews} frozen recent reviews per case."
    )

    print("\nDecision A assignment:")
    print(
        blinding_key[
            "decision_a_system"
        ].value_counts()
    )

    print("\nDecision B assignment:")
    print(
        blinding_key[
            "decision_b_system"
        ].value_counts()
    )

    print("\nBlinded cases saved to:")
    print(blinded_cases_dir)

    print("\nPRIVATE key saved to:")
    print(private_key_path)

    return (
        evaluation_manifest,
        blinding_key,
    )
