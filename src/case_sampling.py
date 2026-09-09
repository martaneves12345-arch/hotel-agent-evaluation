# src/case_sampling.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.case_builder import calculate_trend


# ==========================================================
# CONFIGURATION
# ==========================================================

DEFAULT_LOOKBACK_WEEKS = 4

DEFAULT_MIN_REVIEWS_WINDOW = 20

DEFAULT_RANDOM_STATE = 42

DEFAULT_N_PER_STRATUM = 10


# ----------------------------------------------------------
# Metrics used only for experimental sampling
#
# IMPORTANT:
# These metrics are used to create heterogeneous decision
# situations. They do NOT determine the final managerial
# priority produced by any of the three systems.
# ----------------------------------------------------------

PERFORMANCE_METRICS = [
    "REVPAR_WEEK",
    "TREVPAR_WEEK",
    "TAXA_OCUPACAO",
]


EXPERIENCE_WARNING_RULES = {
    "SENTIMENT_SCORE_AVG": "decreasing",
    "EXPECTATION_EXPERIENCE_GAP_AVG": "increasing",
    "SEMANTIC_MISALIGNMENT_SCORE_AVG": "increasing",
    "EXPECTATION_VIOLATION_RATIO": "increasing",
    "EMOTION_ENTROPY": "increasing",
    "PERC_OPERATIONAL_COMPLAINTS": "increasing",
}


# ==========================================================
# HELPERS
# ==========================================================

def _prepare_panel(
    panel_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Basic preparation of the hotel-week panel.
    """

    panel = panel_df.copy()

    required_columns = {
        "ID_HOTEL",
        "WEEK_YEAR",
        "week_date",
        "N_REVIEWS",
    }

    missing = (
        required_columns
        - set(panel.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns for case sampling: "
            f"{sorted(missing)}"
        )

    panel["ID_HOTEL"] = pd.to_numeric(
        panel["ID_HOTEL"],
        errors="coerce",
    ).astype("Int64")

    panel["week_date"] = pd.to_datetime(
        panel["week_date"],
        errors="coerce",
    ).dt.normalize()

    panel["N_REVIEWS"] = pd.to_numeric(
        panel["N_REVIEWS"],
        errors="coerce",
    ).fillna(0)

    panel = (
        panel
        .dropna(
            subset=[
                "ID_HOTEL",
                "week_date",
            ]
        )
        .sort_values(
            [
                "ID_HOTEL",
                "week_date",
            ]
        )
        .reset_index(drop=True)
    )

    return panel


def _safe_get_latest(
    window_df: pd.DataFrame,
    column: str,
) -> Any:
    """
    Returns the latest value of a column if available.
    """

    if column not in window_df.columns:
        return None

    value = window_df.iloc[-1].get(
        column
    )

    if pd.isna(value):
        return None

    return value


def _window_is_complete(
    window_df: pd.DataFrame,
    lookback_weeks: int,
) -> bool:
    """
    Checks whether the decision window contains the requested
    number of consecutive weekly observations.
    """

    if len(window_df) != lookback_weeks:
        return False

    dates = (
        window_df["week_date"]
        .sort_values()
        .reset_index(drop=True)
    )

    differences = (
        dates.diff()
        .dropna()
        .dt.days
    )

    if differences.empty:
        return False

    return bool(
        (differences == 7).all()
    )


# ==========================================================
# TREND FEATURES
# ==========================================================

def _calculate_window_trends(
    window_df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """
    Calculates only the trends needed for the sampling design.
    """

    metrics = (
        PERFORMANCE_METRICS
        + list(
            EXPERIENCE_WARNING_RULES.keys()
        )
    )

    output = {}

    for metric in metrics:

        if metric not in window_df.columns:
            continue

        output[metric] = calculate_trend(
            window_df[metric]
        )

    return output


# ==========================================================
# EXPERIENCE WARNING SCORE
# ==========================================================

def _experience_warning_score(
    trends: dict[str, dict[str, Any]],
) -> tuple[int, list[str]]:
    """
    Counts how many experiential indicators show deterioration.

    A warning is counted only when:
    - the metric moves in the theoretically adverse direction;
    - direction consistency >= 0.50.

    This mirrors the logic used for identifying a meaningful
    trend rather than reacting to a single weekly fluctuation.
    """

    warning_count = 0
    warning_metrics = []

    for metric, adverse_direction in (
        EXPERIENCE_WARNING_RULES.items()
    ):

        trend = trends.get(
            metric
        )

        if not trend:
            continue

        direction = trend.get(
            "direction"
        )

        consistency = trend.get(
            "direction_consistency"
        )

        if consistency is None:
            consistency = 0

        if (
            direction == adverse_direction
            and consistency >= 0.50
        ):
            warning_count += 1

            warning_metrics.append(
                metric
            )

    return (
        warning_count,
        warning_metrics,
    )


# ==========================================================
# PERFORMANCE STATE
# ==========================================================

def _performance_state(
    trends: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Summarizes basic business-performance behaviour.

    deterioration_count:
        number of performance indicators decreasing.

    improvement_count:
        number of performance indicators increasing.
    """

    deterioration_count = 0
    improvement_count = 0
    stable_or_other_count = 0

    deterioration_metrics = []
    improvement_metrics = []

    for metric in PERFORMANCE_METRICS:

        trend = trends.get(
            metric
        )

        if not trend:
            continue

        direction = trend.get(
            "direction"
        )

        consistency = trend.get(
            "direction_consistency"
        )

        if consistency is None:
            consistency = 0

        # Only treat a trend as meaningful when it has at
        # least moderate directional consistency.
        if (
            direction == "decreasing"
            and consistency >= 0.50
        ):

            deterioration_count += 1

            deterioration_metrics.append(
                metric
            )

        elif (
            direction == "increasing"
            and consistency >= 0.50
        ):

            improvement_count += 1

            improvement_metrics.append(
                metric
            )

        else:
            stable_or_other_count += 1

    return {
        "performance_deterioration_count": (
            deterioration_count
        ),

        "performance_improvement_count": (
            improvement_count
        ),

        "performance_stable_or_mixed_count": (
            stable_or_other_count
        ),

        "performance_deterioration_metrics": (
            deterioration_metrics
        ),

        "performance_improvement_metrics": (
            improvement_metrics
        ),
    }


# ==========================================================
# STRATUM CLASSIFICATION
# ==========================================================

def _classify_stratum(
    experience_warning_count: int,
    performance_deterioration_count: int,
    performance_improvement_count: int,
) -> str:
    """
    Creates four predefined experimental strata.

    A. strong_performance_experience_warning
       Experience deterioration exists despite relatively
       resilient business performance.

    B. broad_deterioration
       Both experience and business performance show
       deterioration.

    C. stable_or_healthy
       Few experiential warning signals and no broad
       performance deterioration.

    D. mixed_or_weak_signals
       Everything that does not fall clearly into the
       previous groups.
    """

    # ------------------------------------------------------
    # A — Experience warning despite resilient performance
    # ------------------------------------------------------

    if (
        experience_warning_count >= 3
        and performance_deterioration_count <= 1
        and performance_improvement_count >= 1
    ):
        return (
            "strong_performance_experience_warning"
        )

    # ------------------------------------------------------
    # B — Broad deterioration
    # ------------------------------------------------------

    if (
        experience_warning_count >= 3
        and performance_deterioration_count >= 2
    ):
        return (
            "broad_deterioration"
        )

    # ------------------------------------------------------
    # C — Stable / relatively healthy
    # ------------------------------------------------------

    if (
        experience_warning_count <= 1
        and performance_deterioration_count <= 1
    ):
        return (
            "stable_or_healthy"
        )

    # ------------------------------------------------------
    # D — Mixed / weak signals
    # ------------------------------------------------------

    return (
        "mixed_or_weak_signals"
    )


# ==========================================================
# BUILD ONE CANDIDATE
# ==========================================================

def _build_candidate_record(
    hotel_id: int,
    window_df: pd.DataFrame,
    lookback_weeks: int,
) -> dict[str, Any]:
    """
    Builds one candidate hotel-week record.
    """

    latest_row = (
        window_df.iloc[-1]
    )

    trends = (
        _calculate_window_trends(
            window_df
        )
    )

    (
        experience_warning_count,
        experience_warning_metrics,
    ) = _experience_warning_score(
        trends
    )

    performance = (
        _performance_state(
            trends
        )
    )

    stratum = (
        _classify_stratum(
            experience_warning_count=(
                experience_warning_count
            ),
            performance_deterioration_count=(
                performance[
                    "performance_deterioration_count"
                ]
            ),
            performance_improvement_count=(
                performance[
                    "performance_improvement_count"
                ]
            ),
        )
    )

    decision_date = (
        latest_row["week_date"]
    )

    # ------------------------------------------------------
    # Basic hotel metadata
    # ------------------------------------------------------

    nuts2 = None

    if "NUTS2" in window_df.columns:
        nuts2 = latest_row.get(
            "NUTS2"
        )

    elif "NUTS2_x" in window_df.columns:
        nuts2 = latest_row.get(
            "NUTS2_x"
        )

    hotel_name = None

    if "NAME" in window_df.columns:
        hotel_name = latest_row.get(
            "NAME"
        )

    stars = None

    if "STARS" in window_df.columns:
        stars = latest_row.get(
            "STARS"
        )

    # ------------------------------------------------------
    # Review volume in entire 4-week window
    # ------------------------------------------------------

    total_reviews_window = float(
        pd.to_numeric(
            window_df["N_REVIEWS"],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    # ------------------------------------------------------
    # Main trend information for inspection
    # ------------------------------------------------------

    revpar_trend = trends.get(
        "REVPAR_WEEK",
        {}
    )

    trevpar_trend = trends.get(
        "TREVPAR_WEEK",
        {}
    )

    occupancy_trend = trends.get(
        "TAXA_OCUPACAO",
        {}
    )

    sentiment_trend = trends.get(
        "SENTIMENT_SCORE_AVG",
        {}
    )

    gap_trend = trends.get(
        "EXPECTATION_EXPERIENCE_GAP_AVG",
        {}
    )

    misalignment_trend = trends.get(
        "SEMANTIC_MISALIGNMENT_SCORE_AVG",
        {}
    )

    violation_trend = trends.get(
        "EXPECTATION_VIOLATION_RATIO",
        {}
    )

    entropy_trend = trends.get(
        "EMOTION_ENTROPY",
        {}
    )

    operational_trend = trends.get(
        "PERC_OPERATIONAL_COMPLAINTS",
        {}
    )

    return {

        # --------------------------------------------------
        # Case identity
        # --------------------------------------------------

        "case_id": (
            f"H{int(hotel_id)}_"
            f"{decision_date.strftime('%Y%m%d')}_"
            f"L{lookback_weeks}"
        ),

        "hotel_id": int(
            hotel_id
        ),

        "hotel_name": (
            hotel_name
        ),

        "nuts2": (
            nuts2
        ),

        "stars": (
            stars
        ),

        "decision_week": (
            latest_row.get(
                "WEEK_YEAR"
            )
        ),

        "decision_date": (
            decision_date
        ),

        "window_start": (
            window_df[
                "week_date"
            ].min()
        ),

        "window_end": (
            decision_date
            + pd.Timedelta(
                days=6
            )
        ),

        "lookback_weeks": (
            lookback_weeks
        ),

        # --------------------------------------------------
        # Review availability
        # --------------------------------------------------

        "reviews_in_window": int(
            total_reviews_window
        ),

        "avg_reviews_per_week": float(
            total_reviews_window
            / lookback_weeks
        ),

        "latest_week_reviews": int(
            pd.to_numeric(
                latest_row.get(
                    "N_REVIEWS"
                ),
                errors="coerce",
            )
            if pd.notna(
                latest_row.get(
                    "N_REVIEWS"
                )
            )
            else 0
        ),

        # --------------------------------------------------
        # Experimental stratum
        # --------------------------------------------------

        "stratum": (
            stratum
        ),

        "experience_warning_count": (
            experience_warning_count
        ),

        "experience_warning_metrics": (
            experience_warning_metrics
        ),

        **performance,

        # --------------------------------------------------
        # Performance trends
        # --------------------------------------------------

        "revpar_direction": (
            revpar_trend.get(
                "direction"
            )
        ),

        "revpar_trend_strength": (
            revpar_trend.get(
                "trend_strength"
            )
        ),

        "revpar_percentage_change": (
            revpar_trend.get(
                "percentage_change"
            )
        ),

        "trevpar_direction": (
            trevpar_trend.get(
                "direction"
            )
        ),

        "occupancy_direction": (
            occupancy_trend.get(
                "direction"
            )
        ),

        # --------------------------------------------------
        # Experience trends
        # --------------------------------------------------

        "sentiment_direction": (
            sentiment_trend.get(
                "direction"
            )
        ),

        "expectation_gap_direction": (
            gap_trend.get(
                "direction"
            )
        ),

        "misalignment_direction": (
            misalignment_trend.get(
                "direction"
            )
        ),

        "violation_ratio_direction": (
            violation_trend.get(
                "direction"
            )
        ),

        "emotion_entropy_direction": (
            entropy_trend.get(
                "direction"
            )
        ),

        "operational_complaints_direction": (
            operational_trend.get(
                "direction"
            )
        ),

        # --------------------------------------------------
        # Latest values for descriptive analysis
        # --------------------------------------------------

        "latest_revpar": (
            _safe_get_latest(
                window_df,
                "REVPAR_WEEK",
            )
        ),

        "latest_rating": (
            _safe_get_latest(
                window_df,
                "AVG_RATING",
            )
        ),

        "latest_sentiment": (
            _safe_get_latest(
                window_df,
                "SENTIMENT_SCORE_AVG",
            )
        ),

        "latest_expectation_gap": (
            _safe_get_latest(
                window_df,
                "EXPECTATION_EXPERIENCE_GAP_AVG",
            )
        ),

        "latest_misalignment": (
            _safe_get_latest(
                window_df,
                "SEMANTIC_MISALIGNMENT_SCORE_AVG",
            )
        ),

        "latest_violation_ratio": (
            _safe_get_latest(
                window_df,
                "EXPECTATION_VIOLATION_RATIO",
            )
        ),
    }


# ==========================================================
# BUILD ALL ELIGIBLE CANDIDATES
# ==========================================================

def build_candidate_cases(
    panel_df: pd.DataFrame,
    evaluation_start: str = "2024-01-01",
    evaluation_end: str = "2024-12-31",
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    min_reviews_window: int = DEFAULT_MIN_REVIEWS_WINDOW,
) -> pd.DataFrame:
    """
    Creates all eligible hotel-week decision cases.

    Eligibility criteria:
    - decision date within the evaluation period;
    - complete consecutive lookback window;
    - at least min_reviews_window reviews across the window.

    Note:
    The lookback may begin before evaluation_start.
    Only the decision date itself must fall inside the
    evaluation period.
    """

    panel = (
        _prepare_panel(
            panel_df
        )
    )

    evaluation_start = pd.Timestamp(
        evaluation_start
    ).normalize()

    evaluation_end = pd.Timestamp(
        evaluation_end
    ).normalize()

    candidate_records = []

    # ------------------------------------------------------
    # Work hotel by hotel
    # ------------------------------------------------------

    for hotel_id, hotel_df in (
        panel.groupby(
            "ID_HOTEL",
            sort=False,
        )
    ):

        hotel_df = (
            hotel_df
            .sort_values(
                "week_date"
            )
            .reset_index(
                drop=True
            )
        )

        if len(
            hotel_df
        ) < lookback_weeks:
            continue

        # --------------------------------------------------
        # Each eligible row can become a decision date
        # --------------------------------------------------

        for end_position in range(
            lookback_weeks - 1,
            len(hotel_df),
        ):

            window_df = (
                hotel_df.iloc[
                    end_position
                    - lookback_weeks
                    + 1:
                    end_position
                    + 1
                ]
                .copy()
            )

            decision_date = (
                window_df.iloc[-1][
                    "week_date"
                ]
            )

            # ----------------------------------------------
            # Evaluation year restriction
            # ----------------------------------------------

            if (
                decision_date
                < evaluation_start
            ):
                continue

            if (
                decision_date
                > evaluation_end
            ):
                continue

            # ----------------------------------------------
            # Require consecutive complete window
            # ----------------------------------------------

            if not _window_is_complete(
                window_df=window_df,
                lookback_weeks=lookback_weeks,
            ):
                continue

            # ----------------------------------------------
            # Require sufficient review volume
            # ----------------------------------------------

            reviews_in_window = (
                pd.to_numeric(
                    window_df[
                        "N_REVIEWS"
                    ],
                    errors="coerce",
                )
                .fillna(0)
                .sum()
            )

            if (
                reviews_in_window
                < min_reviews_window
            ):
                continue

            candidate = (
                _build_candidate_record(
                    hotel_id=int(
                        hotel_id
                    ),
                    window_df=window_df,
                    lookback_weeks=(
                        lookback_weeks
                    ),
                )
            )

            candidate_records.append(
                candidate
            )

    candidates = pd.DataFrame(
        candidate_records
    )

    if candidates.empty:
        raise ValueError(
            "No eligible candidate cases were found."
        )

    return (
        candidates
        .sort_values(
            [
                "decision_date",
                "hotel_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ==========================================================
# STRATIFIED PILOT SAMPLE
# ==========================================================

def sample_pilot_cases(
    candidates_df: pd.DataFrame,
    n_per_stratum: int = DEFAULT_N_PER_STRATUM,
    random_state: int = DEFAULT_RANDOM_STATE,
    unique_hotels: bool = True,
) -> pd.DataFrame:
    """
    Samples an equal number of cases from each experimental
    stratum.

    By default:
    - 10 cases per stratum;
    - 4 strata;
    - 40 total pilot cases;
    - one case per hotel whenever possible.

    Hotels already selected for one stratum are excluded from
    the subsequent strata.
    """

    required_columns = {
        "case_id",
        "hotel_id",
        "stratum",
    }

    missing = (
        required_columns
        - set(
            candidates_df.columns
        )
    )

    if missing:
        raise ValueError(
            "Candidates dataframe is missing: "
            f"{sorted(missing)}"
        )

    candidates = (
        candidates_df.copy()
    )

    strata_order = [
        "strong_performance_experience_warning",
        "broad_deterioration",
        "mixed_or_weak_signals",
        "stable_or_healthy",
    ]

    rng = np.random.default_rng(
        random_state
    )

    selected_parts = []

    used_hotels: set[int] = set()

    for stratum in strata_order:

        stratum_df = (
            candidates[
                candidates[
                    "stratum"
                ] == stratum
            ]
            .copy()
        )

        if stratum_df.empty:

            print(
                f"WARNING: no candidates "
                f"for stratum '{stratum}'."
            )

            continue

        # --------------------------------------------------
        # Avoid reusing hotels across strata
        # --------------------------------------------------

        if unique_hotels:

            stratum_df = stratum_df[
                ~stratum_df[
                    "hotel_id"
                ].isin(
                    used_hotels
                )
            ].copy()

            # ----------------------------------------------
            # One candidate week per hotel.
            #
            # Randomize first, then keep one row per hotel,
            # preventing systematic preference for early or
            # late weeks.
            # ----------------------------------------------

            random_order = rng.permutation(
                len(
                    stratum_df
                )
            )

            stratum_df = (
                stratum_df
                .iloc[
                    random_order
                ]
                .drop_duplicates(
                    subset=[
                        "hotel_id"
                    ],
                    keep="first",
                )
                .reset_index(
                    drop=True
                )
            )

        available = len(
            stratum_df
        )

        if available < n_per_stratum:

            print(
                f"WARNING: stratum '{stratum}' "
                f"has only {available} eligible "
                f"unique hotels; requested "
                f"{n_per_stratum}."
            )

            n_to_sample = available

        else:
            n_to_sample = (
                n_per_stratum
            )

        if n_to_sample == 0:
            continue

        sampled_indices = (
            rng.choice(
                stratum_df.index.to_numpy(),
                size=n_to_sample,
                replace=False,
            )
        )

        sampled = (
            stratum_df.loc[
                sampled_indices
            ]
            .copy()
        )

        selected_parts.append(
            sampled
        )

        if unique_hotels:

            used_hotels.update(
                sampled[
                    "hotel_id"
                ]
                .astype(int)
                .tolist()
            )

    if not selected_parts:

        raise ValueError(
            "No pilot cases could be sampled."
        )

    pilot = (
        pd.concat(
            selected_parts,
            ignore_index=True,
        )
    )

    # ------------------------------------------------------
    # Experimental identifier independent of agent outputs
    # ------------------------------------------------------

    pilot = (
        pilot
        .sort_values(
            [
                "stratum",
                "hotel_id",
                "decision_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    pilot.insert(
        0,
        "pilot_id",
        [
            f"P{i:03d}"
            for i in range(
                1,
                len(pilot) + 1
            )
        ],
    )

    return pilot


# ==========================================================
# SAMPLE SUMMARY
# ==========================================================

def summarize_sample(
    sample_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Creates simple tables for checking the pilot sample.
    """

    by_stratum = (
        sample_df
        .groupby(
            "stratum",
            dropna=False,
        )
        .agg(
            n_cases=(
                "case_id",
                "count",
            ),
            n_hotels=(
                "hotel_id",
                "nunique",
            ),
            avg_reviews=(
                "reviews_in_window",
                "mean",
            ),
            avg_experience_warnings=(
                "experience_warning_count",
                "mean",
            ),
            avg_performance_deterioration=(
                "performance_deterioration_count",
                "mean",
            ),
        )
        .reset_index()
    )

    if (
        "nuts2"
        in sample_df.columns
    ):

        by_region = (
            sample_df[
                "nuts2"
            ]
            .value_counts(
                dropna=False
            )
            .rename_axis(
                "nuts2"
            )
            .reset_index(
                name="n_cases"
            )
        )

    else:

        by_region = (
            pd.DataFrame()
        )

    if (
        "stars"
        in sample_df.columns
    ):

        by_stars = (
            sample_df[
                "stars"
            ]
            .value_counts(
                dropna=False
            )
            .sort_index()
            .rename_axis(
                "stars"
            )
            .reset_index(
                name="n_cases"
            )
        )

    else:

        by_stars = (
            pd.DataFrame()
        )

    return {
        "by_stratum": (
            by_stratum
        ),
        "by_region": (
            by_region
        ),
        "by_stars": (
            by_stars
        ),
    }


# ==========================================================
# SAVE
# ==========================================================

def save_sampling_outputs(
    candidates_df: pd.DataFrame,
    pilot_df: pd.DataFrame,
    output_dir: str | Path = "../outputs",
) -> None:
    """
    Saves both the complete candidate pool and pilot sample.
    """

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates_path = (
        output_dir
        / "candidate_cases_2024.parquet"
    )

    pilot_csv_path = (
        output_dir
        / "pilot_cases_40.csv"
    )

    pilot_parquet_path = (
        output_dir
        / "pilot_cases_40.parquet"
    )

    candidates_df.to_parquet(
        candidates_path,
        index=False,
    )

    pilot_df.to_csv(
        pilot_csv_path,
        index=False,
    )

    pilot_df.to_parquet(
        pilot_parquet_path,
        index=False,
    )

    print(
        f"Saved candidates to: "
        f"{candidates_path}"
    )

    print(
        f"Saved pilot sample to: "
        f"{pilot_csv_path}"
    )

    print(
        f"Saved pilot sample to: "
        f"{pilot_parquet_path}"
    )