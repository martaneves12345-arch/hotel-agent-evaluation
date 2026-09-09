# src/experiment_runner.py

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.case_builder import CaseBuilder
from src.agent_context import build_agent_context
from src.baseline_frequency import run_frequency_baseline
from src.baseline_generic_llm import run_generic_llm_baseline
from src.decision_agent import run_decision_agent


# ==========================================================
# CONFIGURATION
# ==========================================================

EXPERIMENT_VERSION = "pilot_v1"

SYSTEMS = [
    "frequency_baseline",
    "generic_llm",
    "evidence_informed",
]


# ==========================================================
# JSON HELPERS
# ==========================================================

def _json_default(value: Any) -> Any:
    """
    Converts common pandas/numpy values into JSON-compatible
    representations.
    """

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            if np.isnan(value):
                return None
            return float(value)

        if isinstance(value, np.ndarray):
            return value.tolist()

    except ImportError:
        pass

    if pd.isna(value):
        return None

    return str(value)


def _save_json(
    data: dict[str, Any],
    path: Path,
) -> None:
    """
    Saves a dictionary as formatted UTF-8 JSON.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )


# ==========================================================
# OUTPUT STRUCTURE
# ==========================================================

def _prepare_output_directories(
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Creates the experiment output structure.
    """

    root = Path(
        output_dir
    )

    directories = {
        "root": root,
        "cases": root / "cases",
        "frequency_baseline": (
            root / "frequency_baseline"
        ),
        "generic_llm": (
            root / "generic_llm"
        ),
        "evidence_informed": (
            root / "evidence_informed"
        ),
        "logs": root / "logs",
    }

    for directory in directories.values():
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    return directories


# ==========================================================
# EXPERIMENT LOG
# ==========================================================

def _append_log(
    log_path: Path,
    record: dict[str, Any],
) -> None:
    """
    Appends one execution record to the experiment log.
    """

    row = pd.DataFrame(
        [record]
    )

    if log_path.exists():

        row.to_csv(
            log_path,
            mode="a",
            header=False,
            index=False,
        )

    else:

        row.to_csv(
            log_path,
            mode="w",
            header=True,
            index=False,
        )


# ==========================================================
# RUN ONE SYSTEM
# ==========================================================

def _run_and_save_system(
    system_name: str,
    runner,
    output_path: Path,
    log_path: Path,
    pilot_id: str,
    case_id: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Executes one system and immediately saves its output.

    If an output already exists and overwrite=False,
    the API/system is not executed again.
    """

    # ------------------------------------------------------
    # CHECKPOINT
    # ------------------------------------------------------

    if (
        output_path.exists()
        and not overwrite
    ):

        print(
            f"    {system_name}: already exists -> skipped"
        )

        return {
            "status": "skipped",
            "system": system_name,
            "output_path": str(
                output_path
            ),
        }

    started_at = datetime.now(
        timezone.utc
    )

    start_time = time.perf_counter()

    try:

        # --------------------------------------------------
        # RUN SYSTEM
        # --------------------------------------------------

        result = runner()

        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        finished_at = datetime.now(
            timezone.utc
        )

        # --------------------------------------------------
        # ADD EXPERIMENT METADATA
        # --------------------------------------------------

        result[
            "experiment_metadata"
        ] = {

            "experiment_version": (
                EXPERIMENT_VERSION
            ),

            "pilot_id": (
                pilot_id
            ),

            "case_id": (
                case_id
            ),

            "system": (
                system_name
            ),

            "started_at_utc": (
                started_at.isoformat()
            ),

            "finished_at_utc": (
                finished_at.isoformat()
            ),

            "elapsed_seconds": round(
                elapsed_seconds,
                3,
            ),
        }

        # --------------------------------------------------
        # SAVE IMMEDIATELY
        # --------------------------------------------------

        _save_json(
            result,
            output_path,
        )

        # --------------------------------------------------
        # LOG
        # --------------------------------------------------

        _append_log(
            log_path,
            {
                "experiment_version": (
                    EXPERIMENT_VERSION
                ),
                "pilot_id": pilot_id,
                "case_id": case_id,
                "system": system_name,
                "status": "success",
                "elapsed_seconds": round(
                    elapsed_seconds,
                    3,
                ),
                "started_at_utc": (
                    started_at.isoformat()
                ),
                "finished_at_utc": (
                    finished_at.isoformat()
                ),
                "error_type": None,
                "error_message": None,
                "output_path": str(
                    output_path
                ),
            },
        )

        print(
            f"    {system_name}: OK "
            f"({elapsed_seconds:.1f}s)"
        )

        return {
            "status": "success",
            "system": system_name,
            "elapsed_seconds": (
                elapsed_seconds
            ),
            "output_path": str(
                output_path
            ),
        }

    except Exception as error:

        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        finished_at = datetime.now(
            timezone.utc
        )

        # --------------------------------------------------
        # LOG ERROR
        # --------------------------------------------------

        _append_log(
            log_path,
            {
                "experiment_version": (
                    EXPERIMENT_VERSION
                ),
                "pilot_id": pilot_id,
                "case_id": case_id,
                "system": system_name,
                "status": "error",
                "elapsed_seconds": round(
                    elapsed_seconds,
                    3,
                ),
                "started_at_utc": (
                    started_at.isoformat()
                ),
                "finished_at_utc": (
                    finished_at.isoformat()
                ),
                "error_type": (
                    type(error).__name__
                ),
                "error_message": str(
                    error
                ),
                "output_path": str(
                    output_path
                ),
            },
        )

        print(
            f"    {system_name}: ERROR -> "
            f"{type(error).__name__}: {error}"
        )

        return {
            "status": "error",
            "system": system_name,
            "error_type": (
                type(error).__name__
            ),
            "error_message": str(
                error
            ),
        }


# ==========================================================
# BUILD ONE CASE
# ==========================================================

def _build_experimental_case(
    builder: CaseBuilder,
    pilot_row: pd.Series,
    lookback_weeks: int = 4,
    max_reviews: int = 20,
    max_complaints: int = 20,
    top_n_topics: int = 10,
) -> dict[str, Any]:
    """
    Reconstructs the exact CaseBuilder case corresponding
    to one frozen pilot row.
    """

    hotel_id = int(
        pilot_row[
            "hotel_id"
        ]
    )

    decision_week = (
        pilot_row[
            "decision_week"
        ]
    )

    case = builder.build_case(
        hotel_id=hotel_id,
        decision_week=decision_week,
        lookback_weeks=lookback_weeks,
        max_reviews=max_reviews,
        max_complaints=max_complaints,
        top_n_topics=top_n_topics,
    )

    # ------------------------------------------------------
    # SAFETY CHECK
    #
    # Ensures that CaseBuilder reconstructed the exact
    # frozen pilot case.
    # ------------------------------------------------------

    expected_case_id = str(
        pilot_row[
            "case_id"
        ]
    )

    actual_case_id = (
        case
        .get(
            "case_metadata",
            {}
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
            "Case ID mismatch. "
            f"Pilot sample expects '{expected_case_id}' "
            f"but CaseBuilder produced '{actual_case_id}'."
        )

    return case


# ==========================================================
# RUN ONE PILOT CASE
# ==========================================================

def run_one_experimental_case(
    builder: CaseBuilder,
    pilot_row: pd.Series,
    directories: dict[str, Path],
    log_path: Path,
    overwrite: bool = False,
    run_frequency: bool = True,
    run_generic: bool = True,
    run_evidence: bool = True,
    lookback_weeks: int = 4,
    max_reviews: int = 20,
    max_complaints: int = 20,
    top_n_topics: int = 10,
) -> dict[str, Any]:
    """
    Runs all requested systems for one frozen pilot case.
    """

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

    print(
        "\n"
        f"{pilot_id} | {expected_case_id}"
    )

    # ------------------------------------------------------
    # BUILD CASE
    # ------------------------------------------------------

    case = _build_experimental_case(
        builder=builder,
        pilot_row=pilot_row,
        lookback_weeks=lookback_weeks,
        max_reviews=max_reviews,
        max_complaints=max_complaints,
        top_n_topics=top_n_topics,
    )

    case_id = (
        case[
            "case_metadata"
        ][
            "case_id"
        ]
    )

    # ------------------------------------------------------
    # SAVE EXACT CASE INPUT
    # ------------------------------------------------------

    case_path = (
        directories[
            "cases"
        ]
        / f"{pilot_id}_{case_id}.json"
    )

    if (
        overwrite
        or not case_path.exists()
    ):

        _save_json(
            case,
            case_path,
        )

    # ------------------------------------------------------
    # BUILD AGENT CONTEXT
    #
    # Used by:
    # - Frequency Baseline
    # - Evidence-Informed Agent
    # ------------------------------------------------------

    agent_context = (
        build_agent_context(
            case
        )
    )

    results = {}

    # ======================================================
    # 1. FREQUENCY BASELINE
    # ======================================================

    if run_frequency:

        frequency_path = (
            directories[
                "frequency_baseline"
            ]
            / f"{pilot_id}_{case_id}.json"
        )

        results[
            "frequency_baseline"
        ] = _run_and_save_system(

            system_name=(
                "frequency_baseline"
            ),

            runner=lambda: (
                run_frequency_baseline(
                    agent_context=(
                        agent_context
                    ),
                    max_priorities=5,
                )
            ),

            output_path=(
                frequency_path
            ),

            log_path=(
                log_path
            ),

            pilot_id=(
                pilot_id
            ),

            case_id=(
                case_id
            ),

            overwrite=(
                overwrite
            ),
        )

    # ======================================================
    # 2. GENERIC LLM
    # ======================================================

    if run_generic:

        generic_path = (
            directories[
                "generic_llm"
            ]
            / f"{pilot_id}_{case_id}.json"
        )

        results[
            "generic_llm"
        ] = _run_and_save_system(

            system_name=(
                "generic_llm"
            ),

            runner=lambda: (
                run_generic_llm_baseline(
                    case=case,
                    max_reviews=20,
                )
            ),

            output_path=(
                generic_path
            ),

            log_path=(
                log_path
            ),

            pilot_id=(
                pilot_id
            ),

            case_id=(
                case_id
            ),

            overwrite=(
                overwrite
            ),
        )

    # ======================================================
    # 3. EVIDENCE-INFORMED AGENT
    # ======================================================

    if run_evidence:

        evidence_path = (
            directories[
                "evidence_informed"
            ]
            / f"{pilot_id}_{case_id}.json"
        )

        results[
            "evidence_informed"
        ] = _run_and_save_system(

            system_name=(
                "evidence_informed"
            ),

            runner=lambda: (
                run_decision_agent(
                    agent_context
                )
            ),

            output_path=(
                evidence_path
            ),

            log_path=(
                log_path
            ),

            pilot_id=(
                pilot_id
            ),

            case_id=(
                case_id
            ),

            overwrite=(
                overwrite
            ),
        )

    return results


# ==========================================================
# RUN COMPLETE PILOT
# ==========================================================

def run_pilot_experiment(
    pilot_cases: pd.DataFrame,
    builder: CaseBuilder,
    output_dir: str | Path = "../outputs/experiment_v1",
    overwrite: bool = False,
    run_frequency: bool = True,
    run_generic: bool = True,
    run_evidence: bool = True,
    lookback_weeks: int = 4,
    max_reviews: int = 20,
    max_complaints: int = 20,
    top_n_topics: int = 10,
) -> pd.DataFrame:
    """
    Runs the complete frozen pilot experiment.

    Each pilot case can be evaluated by:

    1. Frequency Baseline
    2. Generic LLM
    3. Evidence-Informed Agent

    Existing outputs are skipped unless overwrite=True.
    """

    required_columns = {
        "pilot_id",
        "case_id",
        "hotel_id",
        "decision_week",
    }

    missing = (
        required_columns
        - set(
            pilot_cases.columns
        )
    )

    if missing:

        raise ValueError(
            "pilot_cases is missing required columns: "
            f"{sorted(missing)}"
        )

    # ------------------------------------------------------
    # DUPLICATE CHECKS
    # ------------------------------------------------------

    if (
        pilot_cases[
            "pilot_id"
        ].duplicated().any()
    ):

        raise ValueError(
            "pilot_id contains duplicates."
        )

    if (
        pilot_cases[
            "case_id"
        ].duplicated().any()
    ):

        raise ValueError(
            "case_id contains duplicates."
        )

    # ------------------------------------------------------
    # OUTPUT DIRECTORIES
    # ------------------------------------------------------

    directories = (
        _prepare_output_directories(
            output_dir
        )
    )

    log_path = (
        directories[
            "logs"
        ]
        / "experiment_log.csv"
    )

    # ------------------------------------------------------
    # SAVE FROZEN PILOT MANIFEST
    # ------------------------------------------------------

    manifest_path = (
        directories[
            "root"
        ]
        / "pilot_manifest.csv"
    )

    if not manifest_path.exists():

        pilot_cases.to_csv(
            manifest_path,
            index=False,
        )

    # ------------------------------------------------------
    # RUN
    # ------------------------------------------------------

    total_cases = len(
        pilot_cases
    )

    experiment_summary = []

    for position, (_, row) in enumerate(
        pilot_cases.iterrows(),
        start=1,
    ):

        print(
            "\n"
            "=================================================="
        )

        print(
            f"CASE {position}/{total_cases}"
        )

        print(
            "=================================================="
        )

        try:

            results = (
                run_one_experimental_case(
                    builder=builder,
                    pilot_row=row,
                    directories=directories,
                    log_path=log_path,
                    overwrite=overwrite,
                    run_frequency=run_frequency,
                    run_generic=run_generic,
                    run_evidence=run_evidence,
                    lookback_weeks=lookback_weeks,
                    max_reviews=max_reviews,
                    max_complaints=max_complaints,
                    top_n_topics=top_n_topics,
                )
            )

            experiment_summary.append(
                {
                    "pilot_id": (
                        row[
                            "pilot_id"
                        ]
                    ),
                    "case_id": (
                        row[
                            "case_id"
                        ]
                    ),
                    "hotel_id": (
                        row[
                            "hotel_id"
                        ]
                    ),
                    "decision_week": (
                        row[
                            "decision_week"
                        ]
                    ),
                    "frequency_status": (
                        results
                        .get(
                            "frequency_baseline",
                            {}
                        )
                        .get(
                            "status"
                        )
                    ),
                    "generic_status": (
                        results
                        .get(
                            "generic_llm",
                            {}
                        )
                        .get(
                            "status"
                        )
                    ),
                    "evidence_status": (
                        results
                        .get(
                            "evidence_informed",
                            {}
                        )
                        .get(
                            "status"
                        )
                    ),
                }
            )

        except Exception as error:

            print(
                "CASE-LEVEL ERROR -> "
                f"{type(error).__name__}: {error}"
            )

            experiment_summary.append(
                {
                    "pilot_id": (
                        row[
                            "pilot_id"
                        ]
                    ),
                    "case_id": (
                        row[
                            "case_id"
                        ]
                    ),
                    "hotel_id": (
                        row[
                            "hotel_id"
                        ]
                    ),
                    "decision_week": (
                        row[
                            "decision_week"
                        ]
                    ),
                    "frequency_status": (
                        "case_error"
                    ),
                    "generic_status": (
                        "case_error"
                    ),
                    "evidence_status": (
                        "case_error"
                    ),
                }
            )

    # ------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------

    summary_df = pd.DataFrame(
        experiment_summary
    )

    summary_path = (
        directories[
            "root"
        ]
        / "experiment_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\n"
        "=================================================="
    )

    print(
        "PILOT EXPERIMENT FINISHED"
    )

    print(
        "=================================================="
    )

    print(
        f"Cases: {total_cases}"
    )

    print(
        f"Outputs: {directories['root']}"
    )

    return summary_df