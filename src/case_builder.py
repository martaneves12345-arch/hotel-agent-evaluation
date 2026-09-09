# src/case_builder.py

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.knowledge_base import (
    METRIC_DEFINITIONS,
    build_managerial_topic_summary,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

DEFAULT_METRICS = [
    # Performance
    "REVPAR_WEEK",
    "REVPOR_WEEK",
    "TREVPAR_WEEK",
    "TAXA_OCUPACAO",

    # Traditional review metrics
    "N_REVIEWS",
    "AVG_RATING",
    "SENTIMENT_SCORE_AVG",
    "SENTIMENT_SCORE_STD",

    # Experiential structures
    "EXPECTATION_EXPERIENCE_GAP_AVG",
    "SEMANTIC_MISALIGNMENT_SCORE_AVG",
    "EXPECTATION_VIOLATION_RATIO",
    "MAJOR_MISALIGNMENT_SHARE",
    "EMOTION_ENTROPY",
    "EXPECTATION_SHARE",
    "HIGH_EXPECTATION_SHARE",
    "HIGH_MISALIGNMENT_SHARE",

    # Complementary structures
    "PERC_REVIEWS_NEGATIVE",
    "PERC_OPERATIONAL_COMPLAINTS",
    "PERC_PRICE_VALUE_MENTIONS",
    "POLARIZATION_INDEX",
    "COMPLEXITY_INDEX",
    "SENTIMENT_TREND_SLOPE",
]


TOPIC_COLUMNS = [
    "EXPECTATION_TOPICS_UNIQUE",
    "EXPERIENCE_TOPICS_UNIQUE",
    "VIOLATED_EXPECTATION_TOPICS_UNIQUE",
    "LATENT_EXPECTATION_TOPICS_UNIQUE",
    "EMERGING_NEGATIVE_TOPIC",
]


HOTEL_COLUMNS = [
    "ID_HOTEL",
    "NAME",
    "NOME_HOTEL_URL",
    "ID_LOCAL",
    "NUTS2_x",
    "N_ROOMS",
    "STARS",
    "PRINCIPAIS_COMODIDADES",
]


# ==========================================================
# GENERAL HELPERS
# ==========================================================

def clean_scalar(value: Any) -> Any:
    """
    Converts pandas/numpy scalar values into JSON-serializable values.
    """

    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, float) and np.isnan(value):
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


def safe_numeric(value: Any) -> Optional[float]:
    """
    Converts a value to float.
    Returns None when conversion is not possible.
    """

    numeric_value = pd.to_numeric(
        value,
        errors="coerce",
    )

    if pd.isna(numeric_value):
        return None

    return float(numeric_value)


def parse_topic_list(value: Any) -> list[str]:
    """
    Converts string representations of Python lists
    into actual lists.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    if isinstance(value, tuple):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    if isinstance(value, np.ndarray):
        return [
            str(item).strip()
            for item in value.tolist()
            if str(item).strip()
        ]

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    value = str(value).strip()

    if value.lower() in {
        "",
        "[]",
        "nan",
        "none",
    }:
        return []

    try:
        parsed = ast.literal_eval(value)

        if isinstance(
            parsed,
            (list, tuple)
        ):
            return [
                str(item).strip()
                for item in parsed
                if str(item).strip()
            ]

    except (
        ValueError,
        SyntaxError,
    ):
        pass

    return [value]


def normalize_text(value: Any) -> Optional[str]:
    """
    Cleans empty text and NaN-like representations.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    value = str(value).strip()

    if value.lower() in {
        "",
        "nan",
        "none",
        "nothing",
        "n/a",
        "na",
    }:
        return None

    return value


def calculate_trend(
    values: pd.Series,
    stable_threshold: float = 0.15,
    recent_change_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """
    Calculates:
    - overall direction;
    - trend strength;
    - direction consistency;
    - latest movement;
    - recent reversal.
    """

    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    )

    valid_values = (
        numeric_values
        .dropna()
    )

    if len(valid_values) < 2:

        single_value = (
            clean_scalar(
                valid_values.iloc[0]
            )
            if len(valid_values) == 1
            else None
        )

        return {
            "direction": "insufficient_data",
            "trend_strength": "insufficient_data",
            "first_value": single_value,
            "last_value": single_value,
            "minimum_value": single_value,
            "maximum_value": single_value,
            "absolute_change": None,
            "percentage_change": None,
            "slope": None,
            "standardized_slope": None,
            "direction_consistency": None,
            "recent_movement": "insufficient_data",
            "recent_absolute_change": None,
            "recent_reversal": None,
            "n_observations": int(
                len(valid_values)
            ),
        }

    y = valid_values.to_numpy(
        dtype=float
    )

    x = np.arange(
        len(y),
        dtype=float
    )

    slope = float(
        np.polyfit(
            x,
            y,
            1,
        )[0]
    )

    standard_deviation = float(
        np.std(
            y,
            ddof=0,
        )
    )

    standardized_slope = (
        slope / standard_deviation
        if standard_deviation > 0
        else 0.0
    )

    first_value = float(y[0])
    last_value = float(y[-1])

    minimum_value = float(
        np.min(y)
    )

    maximum_value = float(
        np.max(y)
    )

    absolute_change = (
        last_value
        - first_value
    )

    percentage_change = (
        absolute_change
        / abs(first_value)
        * 100
        if abs(first_value) > 1e-12
        else None
    )

    value_range = (
        maximum_value
        - minimum_value
    )

    if value_range <= 1e-12:
        direction = "stable"

    elif abs(
        absolute_change
    ) <= 1e-12:
        direction = "fluctuating"

    elif (
        standardized_slope
        > stable_threshold
        and absolute_change > 0
    ):
        direction = "increasing"

    elif (
        standardized_slope
        < -stable_threshold
        and absolute_change < 0
    ):
        direction = "decreasing"

    else:
        direction = "fluctuating"

    changes = np.diff(y)

    if direction == "increasing":
        direction_consistency = float(
            np.mean(
                changes > 0
            )
        )

    elif direction == "decreasing":
        direction_consistency = float(
            np.mean(
                changes < 0
            )
        )

    elif direction == "stable":
        direction_consistency = float(
            np.mean(
                np.abs(changes)
                <= 1e-12
            )
        )

    else:

        positive_share = float(
            np.mean(
                changes > 0
            )
        )

        negative_share = float(
            np.mean(
                changes < 0
            )
        )

        direction_consistency = max(
            positive_share,
            negative_share,
        )

    if direction in {
        "increasing",
        "decreasing",
    }:

        if direction_consistency >= 0.75:
            trend_strength = "consistent"

        elif direction_consistency >= 0.50:
            trend_strength = "moderate"

        else:
            trend_strength = "weak"

    elif direction == "stable":
        trend_strength = "stable"

    elif direction == "fluctuating":
        trend_strength = "fluctuating"

    else:
        trend_strength = "insufficient_data"

    recent_absolute_change = float(
        y[-1]
        - y[-2]
    )

    if (
        recent_absolute_change
        > recent_change_tolerance
    ):
        recent_movement = "up"

    elif (
        recent_absolute_change
        < -recent_change_tolerance
    ):
        recent_movement = "down"

    else:
        recent_movement = "flat"

    recent_reversal = False

    if (
        direction == "increasing"
        and recent_movement == "down"
    ):
        recent_reversal = True

    elif (
        direction == "decreasing"
        and recent_movement == "up"
    ):
        recent_reversal = True

    return {
        "direction": direction,
        "trend_strength": trend_strength,
        "first_value": first_value,
        "last_value": last_value,
        "minimum_value": minimum_value,
        "maximum_value": maximum_value,
        "absolute_change": absolute_change,
        "percentage_change": percentage_change,
        "slope": slope,
        "standardized_slope": standardized_slope,
        "direction_consistency": direction_consistency,
        "recent_movement": recent_movement,
        "recent_absolute_change": recent_absolute_change,
        "recent_reversal": recent_reversal,
        "n_observations": int(
            len(y)
        ),
    }


# ==========================================================
# CASE BUILDER
# ==========================================================

@dataclass
class CaseBuilder:

    panel_df: pd.DataFrame
    reviews_df: pd.DataFrame

    def __post_init__(self) -> None:

        self.panel = self._prepare_panel(
            self.panel_df
        )

        self.reviews = self._prepare_reviews(
            self.reviews_df
        )

        self._validate_panel_uniqueness()


    # ======================================================
    # DATA PREPARATION
    # ======================================================

    @staticmethod
    def _prepare_panel(
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        panel = df.copy()

        required_columns = {
            "ID_HOTEL",
            "WEEK_YEAR",
            "week_date",
        }

        missing_columns = (
            required_columns
            - set(panel.columns)
        )

        if missing_columns:

            raise ValueError(
                "Missing required panel columns: "
                f"{sorted(missing_columns)}"
            )

        panel[
            "ID_HOTEL"
        ] = pd.to_numeric(
            panel["ID_HOTEL"],
            errors="coerce",
        ).astype(
            "Int64"
        )

        panel[
            "week_date"
        ] = pd.to_datetime(
            panel["week_date"],
            errors="coerce",
        ).dt.normalize()

        panel = panel.dropna(
            subset=[
                "ID_HOTEL",
                "week_date",
            ]
        ).copy()

        return (
            panel
            .sort_values(
                [
                    "ID_HOTEL",
                    "week_date",
                ]
            )
            .reset_index(
                drop=True
            )
        )


    @staticmethod
    def _prepare_reviews(
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        reviews = df.copy()

        required_columns = {
            "ID_HOTEL",
            "review_post_date",
        }

        missing_columns = (
            required_columns
            - set(reviews.columns)
        )

        if missing_columns:

            raise ValueError(
                "Missing required review columns: "
                f"{sorted(missing_columns)}"
            )

        reviews = reviews.drop(
            columns=[
                "Unnamed: 0"
            ],
            errors="ignore",
        )

        reviews[
            "ID_HOTEL"
        ] = pd.to_numeric(
            reviews[
                "ID_HOTEL"
            ],
            errors="coerce",
        ).astype(
            "Int64"
        )

        reviews[
            "review_date"
        ] = pd.to_datetime(
            reviews[
                "review_post_date"
            ],
            format="%m-%d-%Y %H:%M:%S",
            errors="coerce",
        )

        reviews[
            "week_date"
        ] = (
            reviews[
                "review_date"
            ]
            - pd.to_timedelta(
                reviews[
                    "review_date"
                ].dt.weekday,
                unit="D",
            )
        ).dt.normalize()

        reviews = reviews.dropna(
            subset=[
                "ID_HOTEL",
                "review_date",
                "week_date",
            ]
        ).copy()

        return (
            reviews
            .sort_values(
                [
                    "ID_HOTEL",
                    "review_date",
                ]
            )
            .reset_index(
                drop=True
            )
        )


    def _validate_panel_uniqueness(
        self,
    ) -> None:

        duplicate_mask = (
            self.panel.duplicated(
                [
                    "ID_HOTEL",
                    "week_date",
                ],
                keep=False,
            )
        )

        if duplicate_mask.any():

            duplicate_examples = (
                self.panel.loc[
                    duplicate_mask,
                    [
                        "ID_HOTEL",
                        "WEEK_YEAR",
                        "week_date",
                    ],
                ]
                .drop_duplicates()
                .head(10)
            )

            raise ValueError(
                "Panel contains more than one row "
                "per ID_HOTEL and week_date.\n"
                "Examples:\n"
                f"{duplicate_examples.to_string(index=False)}"
            )


    # ======================================================
    # DATE RESOLUTION
    # ======================================================

    def _resolve_decision_date(
        self,
        hotel_id: int,
        decision_week: str | pd.Timestamp,
    ) -> pd.Timestamp:

        hotel_rows = self.panel[
            self.panel[
                "ID_HOTEL"
            ] == hotel_id
        ]

        if hotel_rows.empty:

            raise ValueError(
                f"Hotel {hotel_id} does not "
                "exist in the panel."
            )

        if isinstance(
            decision_week,
            pd.Timestamp,
        ):

            decision_date = (
                decision_week.normalize()
            )

        elif isinstance(
            decision_week,
            str,
        ):

            decision_week = (
                decision_week.strip()
            )

            if "_W" in decision_week:

                matching_rows = hotel_rows[
                    hotel_rows[
                        "WEEK_YEAR"
                    ] == decision_week
                ]

                if matching_rows.empty:

                    raise ValueError(
                        f"Week {decision_week} "
                        f"does not exist for hotel "
                        f"{hotel_id}."
                    )

                decision_date = (
                    matching_rows[
                        "week_date"
                    ].iloc[0]
                )

            else:

                decision_date = (
                    pd.to_datetime(
                        decision_week,
                        errors="coerce",
                    )
                )

                if pd.isna(
                    decision_date
                ):

                    raise ValueError(
                        "decision_week must be "
                        "a valid date or use the "
                        "format '2024_W20'."
                    )

                decision_date = (
                    decision_date.normalize()
                )

        else:

            raise TypeError(
                "decision_week must be a "
                "string or pd.Timestamp."
            )

        return decision_date


    # ======================================================
    # HOTEL PROFILE
    # ======================================================

    @staticmethod
    def _build_hotel_profile(
        hotel_rows: pd.DataFrame,
    ) -> dict[str, Any]:

        latest_row = (
            hotel_rows.iloc[-1]
        )

        return {
            column: clean_scalar(
                latest_row[column]
            )
            for column in HOTEL_COLUMNS
            if column in hotel_rows.columns
        }


    # ======================================================
    # WEEKLY HISTORY
    # ======================================================

    @staticmethod
    def _build_weekly_history(
        window_df: pd.DataFrame,
        metrics: list[str],
    ) -> list[dict[str, Any]]:

        available_metrics = [
            metric
            for metric in metrics
            if metric in window_df.columns
        ]

        output_columns = [
            "WEEK_YEAR",
            "week_date",
            *available_metrics,
        ]

        return [
            {
                column: clean_scalar(
                    row[column]
                )
                for column
                in output_columns
            }
            for _, row
            in window_df[
                output_columns
            ].iterrows()
        ]


    # ======================================================
    # TRENDS
    # ======================================================

    @staticmethod
    def _build_trends(
        window_df: pd.DataFrame,
        metrics: list[str],
    ) -> dict[str, dict[str, Any]]:

        return {
            metric: calculate_trend(
                window_df[
                    metric
                ]
            )
            for metric
            in metrics
            if metric
            in window_df.columns
        }


    # ======================================================
    # TOPICS
    # ======================================================

    @staticmethod
    def _count_topics(
        window_df: pd.DataFrame,
        column: str,
        top_n: int = 10,
    ) -> list[dict[str, Any]]:

        topic_counter: dict[
            str,
            int,
        ] = {}

        if (
            column
            not in window_df.columns
        ):
            return []

        for value in window_df[
            column
        ]:

            topics_in_week = {
                topic.strip()
                for topic
                in parse_topic_list(
                    value
                )
                if topic.strip()
            }

            for topic in topics_in_week:

                topic_counter[
                    topic
                ] = (
                    topic_counter.get(
                        topic,
                        0,
                    )
                    + 1
                )

        ordered_topics = sorted(
            topic_counter.items(),
            key=lambda item: (
                -item[1],
                item[0].lower(),
            ),
        )

        return [
            {
                "topic": topic,
                "weeks_present": count,
            }
            for topic, count
            in ordered_topics[
                :top_n
            ]
        ]


    def _build_topic_summary(
        self,
        window_df: pd.DataFrame,
        top_n: int = 10,
    ) -> dict[
        str,
        list[
            dict[
                str,
                Any,
            ]
        ],
    ]:

        return {
            column: self._count_topics(
                window_df=window_df,
                column=column,
                top_n=top_n,
            )
            for column in TOPIC_COLUMNS
        }


    # ======================================================
    # REVIEW WINDOW
    # ======================================================

    def _get_review_window(
        self,
        hotel_id: int,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Returns all reviews falling inside the decision
        window.

        end_date represents the Monday of the decision
        week, therefore the review window includes the
        complete decision week until Sunday.
        """

        return self.reviews[
            (
                self.reviews[
                    "ID_HOTEL"
                ] == hotel_id
            )
            & (
                self.reviews[
                    "review_date"
                ] >= start_date
            )
            & (
                self.reviews[
                    "review_date"
                ]
                < end_date
                + pd.Timedelta(
                    days=7
                )
            )
        ].copy()


    # ======================================================
    # NEUTRAL REVIEW SAMPLE
    # Used by Generic LLM
    # ======================================================

    @staticmethod
    def _build_review_text(
        row: pd.Series,
    ) -> dict[str, Optional[str]]:
        """
        Retrieves translated review text when available,
        falling back to the original text.
        """

        liked = (
            normalize_text(
                row.get(
                    "en_review_text_liked"
                )
            )
            or normalize_text(
                row.get(
                    "review_text_liked"
                )
            )
        )

        disliked = (
            normalize_text(
                row.get(
                    "en_review_text_disliked"
                )
            )
            or normalize_text(
                row.get(
                    "review_text_disliked"
                )
            )
        )

        return {
            "liked": liked,
            "disliked": disliked,
        }


    def _select_review_sample(
        self,
        review_window: pd.DataFrame,
        max_reviews: int = 20,
    ) -> tuple[
        list[
            dict[
                str,
                Any,
            ]
        ],
        dict[
            str,
            Any,
        ],
    ]:
        """
        Creates a neutral recent-review sample for the
        Generic LLM baseline.

        Selection rule:
        - no filtering by rating;
        - no filtering by complaint presence;
        - no prioritization by complaint severity;
        - reviews ordered only by recency.

        This represents what a generic LLM could receive
        if a hotel manager supplied the most recent
        customer reviews.
        """

        review_window = (
            review_window.copy()
        )

        total_reviews_in_window = int(
            len(
                review_window
            )
        )

        if review_window.empty:

            return [], {
                "total_reviews_in_window": 0,
                "total_reviews_selected": 0,
                "selection_strategy": (
                    "Most recent reviews in the "
                    "decision window, without "
                    "selection by rating, sentiment "
                    "or complaint content."
                ),
            }

        sample = (
            review_window
            .sort_values(
                "review_date",
                ascending=False,
            )
            .head(
                max_reviews
            )
            .copy()
        )

        output = []

        for _, row in (
            sample.iterrows()
        ):

            text = (
                self._build_review_text(
                    row
                )
            )

            output.append(
                {
                    "review_date": clean_scalar(
                        row[
                            "review_date"
                        ]
                    ),

                    "rating": safe_numeric(
                        row.get(
                            "rating"
                        )
                    ),

                    "language": normalize_text(
                        row.get(
                            "original_lang"
                        )
                    ),

                    "title": normalize_text(
                        row.get(
                            "review_title"
                        )
                    ),

                    "liked": text[
                        "liked"
                    ],

                    "disliked": text[
                        "disliked"
                    ],

                    "country": normalize_text(
                        row.get(
                            "user_country"
                        )
                    ),

                    "stay_type": normalize_text(
                        row.get(
                            "stay_type"
                        )
                    ),
                }
            )

        metadata = {

            "total_reviews_in_window": (
                total_reviews_in_window
            ),

            "total_reviews_selected": int(
                len(output)
            ),

            "selection_strategy": (
                "Neutral recent-review sampling. "
                "Reviews are ordered by review_date "
                "descending and the most recent "
                "reviews are selected. No filtering "
                "or ranking based on rating, sentiment, "
                "complaint severity or text length is used."
            ),

            "interpretation_warning": (
                "The sample represents the most recent "
                "reviews in the decision window, not "
                "necessarily the full review distribution."
            ),
        }

        return (
            output,
            metadata,
        )


    # ======================================================
    # DIAGNOSTIC COMPLAINT SAMPLE
    # Used by Evidence-Informed Agent
    # ======================================================

    @staticmethod
    def _has_substantive_complaint(
        row: pd.Series,
    ) -> bool:

        disliked = normalize_text(
            row.get(
                "review_text_disliked"
            )
        )

        if not disliked:
            return False

        return (
            len(
                disliked.split()
            )
            >= 2
        )


    @staticmethod
    def _create_complaint_text(
        row: pd.Series,
    ) -> Optional[str]:

        disliked_en = normalize_text(
            row.get(
                "en_review_text_disliked"
            )
        )

        disliked_original = normalize_text(
            row.get(
                "review_text_disliked"
            )
        )

        disliked = (
            disliked_en
            or disliked_original
        )

        if not disliked:
            return None

        return (
            f"Disliked: {disliked}"
        )


    def _select_complaints(
        self,
        review_window: pd.DataFrame,
        max_complaints: int = 20,
    ) -> tuple[
        list[
            dict[
                str,
                Any,
            ]
        ],
        dict[
            str,
            Any,
        ],
    ]:
        """
        Diagnostic complaint sampling used by the
        Evidence-Informed Agent.

        Eligible reviews:
        - must contain substantive disliked text.

        Ranking:
        1. lower rating;
        2. longer disliked text;
        3. greater recency.

        These examples are diagnostic rather than
        representative.
        """

        review_window = (
            review_window.copy()
        )

        total_reviews_in_window = int(
            len(
                review_window
            )
        )

        review_window[
            "has_disliked"
        ] = (
            review_window[
                "review_text_disliked"
            ]
            .apply(
                lambda value:
                normalize_text(
                    value
                )
                is not None
            )
        )

        total_reviews_with_disliked = int(
            review_window[
                "has_disliked"
            ].sum()
        )

        complaint_window = (
            review_window[
                review_window.apply(
                    self._has_substantive_complaint,
                    axis=1,
                )
            ]
            .copy()
        )

        complaint_window[
            "selected_text"
        ] = complaint_window.apply(
            self._create_complaint_text,
            axis=1,
        )

        complaint_window[
            "disliked_word_count"
        ] = (
            complaint_window[
                "review_text_disliked"
            ]
            .fillna("")
            .astype(str)
            .str.split()
            .str.len()
        )

        complaint_window[
            "rating_numeric"
        ] = pd.to_numeric(
            complaint_window.get(
                "rating"
            ),
            errors="coerce",
        )

        complaint_window = (
            complaint_window
            .sort_values(
                [
                    "rating_numeric",
                    "disliked_word_count",
                    "review_date",
                ],
                ascending=[
                    True,
                    False,
                    False,
                ],
            )
        )

        selected_reviews = (
            complaint_window
            .head(
                max_complaints
            )
            .copy()
        )

        output = []

        for _, row in (
            selected_reviews.iterrows()
        ):

            output.append(
                {
                    "review_date": clean_scalar(
                        row[
                            "review_date"
                        ]
                    ),

                    "rating": safe_numeric(
                        row.get(
                            "rating"
                        )
                    ),

                    "language": normalize_text(
                        row.get(
                            "original_lang"
                        )
                    ),

                    "title": normalize_text(
                        row.get(
                            "review_title"
                        )
                    ),

                    "complaint_text": (
                        normalize_text(
                            row.get(
                                "selected_text"
                            )
                        )
                    ),

                    "disliked": (
                        normalize_text(
                            row.get(
                                "en_review_text_disliked"
                            )
                        )
                        or normalize_text(
                            row.get(
                                "review_text_disliked"
                            )
                        )
                    ),

                    "liked_context": (
                        normalize_text(
                            row.get(
                                "en_review_text_liked"
                            )
                        )
                        or normalize_text(
                            row.get(
                                "review_text_liked"
                            )
                        )
                    ),

                    "country": normalize_text(
                        row.get(
                            "user_country"
                        )
                    ),

                    "stay_type": normalize_text(
                        row.get(
                            "stay_type"
                        )
                    ),

                    "disliked_word_count": int(
                        row.get(
                            "disliked_word_count",
                            0,
                        )
                    ),
                }
            )

        metadata = {

            "total_reviews_in_window": int(
                total_reviews_in_window
            ),

            "total_reviews_with_disliked": int(
                total_reviews_with_disliked
            ),

            "share_reviews_with_disliked": (
                float(
                    total_reviews_with_disliked
                    / total_reviews_in_window
                )
                if total_reviews_in_window > 0
                else 0.0
            ),

            "total_substantive_complaints_available": int(
                len(
                    complaint_window
                )
            ),

            "total_complaints_selected": int(
                len(
                    output
                )
            ),

            "selection_strategy": (
                "Complaint-focused diagnostic sampling. "
                "Only reviews with substantive content "
                "in review_text_disliked are eligible. "
                "Complaints are ranked by lower rating, "
                "longer disliked text and recency."
            ),

            "interpretation_warning": (
                "Selected complaints are diagnostic "
                "evidence and are not representative "
                "of the complete review distribution. "
                "Prevalence should be assessed using "
                "the full review-window metadata and "
                "managerial evidence."
            ),
        }

        return (
            output,
            metadata,
        )


    # ======================================================
    # PRELIMINARY SIGNALS
    # ======================================================

    @staticmethod
    def _build_preliminary_signals(
        trends: dict[
            str,
            dict[
                str,
                Any,
            ],
        ],
        latest_row: pd.Series,
    ) -> list[
        dict[
            str,
            Any,
        ]
    ]:

        signals = []

        rules = [
            {
                "metric": "REVPAR_WEEK",
                "direction": "decreasing",
                "signal": "declining_revpar",
                "severity": "high",
                "description": (
                    "RevPAR shows a global "
                    "decreasing trend."
                ),
            },
            {
                "metric": "SENTIMENT_SCORE_AVG",
                "direction": "decreasing",
                "signal": "declining_sentiment",
                "severity": "medium",
                "description": (
                    "Average sentiment shows a "
                    "global decreasing trend."
                ),
            },
            {
                "metric": "EXPECTATION_EXPERIENCE_GAP_AVG",
                "direction": "increasing",
                "signal": "increasing_expectation_gap",
                "severity": "high",
                "description": (
                    "Expectation–experience discrepancy "
                    "shows a global increasing trend."
                ),
            },
            {
                "metric": "SEMANTIC_MISALIGNMENT_SCORE_AVG",
                "direction": "increasing",
                "signal": "increasing_misalignment",
                "severity": "high",
                "description": (
                    "Semantic misalignment shows a "
                    "global increasing trend."
                ),
            },
            {
                "metric": "EMOTION_ENTROPY",
                "direction": "increasing",
                "signal": (
                    "increasing_emotional_complexity"
                ),
                "severity": "medium",
                "description": (
                    "Emotional heterogeneity shows "
                    "a global increasing trend."
                ),
            },
            {
                "metric": "PERC_OPERATIONAL_COMPLAINTS",
                "direction": "increasing",
                "signal": (
                    "increasing_operational_complaints"
                ),
                "severity": "high",
                "description": (
                    "The share of operational complaints "
                    "shows a global increasing trend."
                ),
            },
        ]

        for rule in rules:

            metric_trend = trends.get(
                rule[
                    "metric"
                ]
            )

            if not metric_trend:
                continue

            consistency = (
                metric_trend.get(
                    "direction_consistency"
                )
            )

            if consistency is None:
                consistency = 0

            if (
                metric_trend[
                    "direction"
                ]
                == rule[
                    "direction"
                ]
                and consistency >= 0.50
            ):

                signals.append(
                    {
                        "signal": (
                            rule[
                                "signal"
                            ]
                        ),

                        "severity": (
                            rule[
                                "severity"
                            ]
                        ),

                        "metric": (
                            rule[
                                "metric"
                            ]
                        ),

                        "description": (
                            rule[
                                "description"
                            ]
                        ),

                        "trend_strength": (
                            metric_trend.get(
                                "trend_strength"
                            )
                        ),

                        "direction_consistency": (
                            metric_trend.get(
                                "direction_consistency"
                            )
                        ),

                        "recent_movement": (
                            metric_trend.get(
                                "recent_movement"
                            )
                        ),

                        "recent_reversal": (
                            metric_trend.get(
                                "recent_reversal"
                            )
                        ),
                    }
                )

        if (
            "PRIORITY_ALERT_FLAG"
            in latest_row.index
        ):

            priority_alert = clean_scalar(
                latest_row[
                    "PRIORITY_ALERT_FLAG"
                ]
            )

            if priority_alert not in {
                None,
                0,
                0.0,
                False,
                "0",
            }:

                signals.append(
                    {
                        "signal": (
                            "existing_priority_alert"
                        ),

                        "severity": "high",

                        "metric": (
                            "PRIORITY_ALERT_FLAG"
                        ),

                        "description": (
                            "The latest observation "
                            "contains an existing "
                            "priority alert."
                        ),

                        "trend_strength": None,
                        "direction_consistency": None,
                        "recent_movement": None,
                        "recent_reversal": None,
                    }
                )

        return signals


    # ======================================================
    # BUILD COMPLETE CASE
    # ======================================================

    def build_case(
        self,
        hotel_id: int,
        decision_week: str | pd.Timestamp,
        lookback_weeks: int = 4,
        max_reviews: int = 20,
        max_complaints: int = 20,
        top_n_topics: int = 10,
        metrics: Optional[
            list[str]
        ] = None,
    ) -> dict[str, Any]:
        """
        Builds a complete hotel decision case.

        Outputs two distinct review samples:

        review_sample:
            Neutral recent reviews for the Generic LLM.

        complaints:
            Diagnostic complaint sample for the
            Evidence-Informed Agent.
        """

        if lookback_weeks < 2:

            raise ValueError(
                "lookback_weeks must be "
                "at least 2."
            )

        if max_reviews < 1:

            raise ValueError(
                "max_reviews must be "
                "at least 1."
            )

        if max_complaints < 1:

            raise ValueError(
                "max_complaints must be "
                "at least 1."
            )

        hotel_id = int(
            hotel_id
        )

        selected_metrics = (
            metrics
            if metrics is not None
            else DEFAULT_METRICS
        )

        decision_date = (
            self._resolve_decision_date(
                hotel_id=hotel_id,
                decision_week=decision_week,
            )
        )

        start_date = (
            decision_date
            - pd.Timedelta(
                weeks=(
                    lookback_weeks
                    - 1
                )
            )
        )

        hotel_panel = self.panel[
            self.panel[
                "ID_HOTEL"
            ] == hotel_id
        ].copy()

        window_df = hotel_panel[
            (
                hotel_panel[
                    "week_date"
                ] >= start_date
            )
            & (
                hotel_panel[
                    "week_date"
                ] <= decision_date
            )
        ].copy()

        window_df = (
            window_df
            .sort_values(
                "week_date"
            )
        )

        if window_df.empty:

            raise ValueError(
                "No observations exist for "
                f"hotel {hotel_id} between "
                f"{start_date.date()} and "
                f"{decision_date.date()}."
            )

        latest_row = (
            window_df.iloc[-1]
        )

        # --------------------------------------------------
        # FULL REVIEW WINDOW
        # --------------------------------------------------

        full_review_window = (
            self._get_review_window(
                hotel_id=hotel_id,
                start_date=start_date,
                end_date=decision_date,
            )
        )

        # --------------------------------------------------
        # NEUTRAL REVIEW SAMPLE
        # Generic LLM
        # --------------------------------------------------

        (
            review_sample,
            review_sample_metadata,
        ) = self._select_review_sample(
            review_window=(
                full_review_window
            ),
            max_reviews=max_reviews,
        )

        # --------------------------------------------------
        # DIAGNOSTIC COMPLAINT SAMPLE
        # Evidence-Informed Agent
        # --------------------------------------------------

        (
            complaints,
            complaint_metadata,
        ) = self._select_complaints(
            review_window=(
                full_review_window
            ),
            max_complaints=(
                max_complaints
            ),
        )

        # --------------------------------------------------
        # WEEKLY HISTORY
        # --------------------------------------------------

        weekly_history = (
            self._build_weekly_history(
                window_df=window_df,
                metrics=selected_metrics,
            )
        )

        # --------------------------------------------------
        # TRENDS
        # --------------------------------------------------

        trends = (
            self._build_trends(
                window_df=window_df,
                metrics=selected_metrics,
            )
        )

        # --------------------------------------------------
        # TOPICS
        # --------------------------------------------------

        raw_topic_summary = (
            self._build_topic_summary(
                window_df=window_df,
                top_n=top_n_topics,
            )
        )

        managerial_topic_summary = (
            build_managerial_topic_summary(
                weekly_topic_summary=(
                    raw_topic_summary
                ),
                reviews_df=(
                    full_review_window
                ),
                top_n=(
                    top_n_topics
                ),
            )
        )

        # --------------------------------------------------
        # SIGNALS
        # --------------------------------------------------

        preliminary_signals = (
            self._build_preliminary_signals(
                trends=trends,
                latest_row=latest_row,
            )
        )

        # --------------------------------------------------
        # FINAL CASE
        # --------------------------------------------------

        case = {

            "case_metadata": {

                "case_id": (
                    f"H{hotel_id}_"
                    f"{decision_date.strftime('%Y%m%d')}_"
                    f"L{lookback_weeks}"
                ),

                "hotel_id": (
                    hotel_id
                ),

                "decision_week": (
                    clean_scalar(
                        latest_row.get(
                            "WEEK_YEAR"
                        )
                    )
                ),

                "decision_date": (
                    clean_scalar(
                        decision_date
                    )
                ),

                "window_start": (
                    clean_scalar(
                        start_date
                    )
                ),

                "window_end": (
                    clean_scalar(
                        decision_date
                        + pd.Timedelta(
                            days=6
                        )
                    )
                ),

                "lookback_weeks_requested": (
                    lookback_weeks
                ),

                "panel_weeks_available": (
                    len(
                        window_df
                    )
                ),

                "complete_window": (
                    len(
                        window_df
                    )
                    == lookback_weeks
                ),
            },

            # ----------------------------------------------
            # HOTEL
            # ----------------------------------------------

            "hotel_profile": (
                self._build_hotel_profile(
                    hotel_rows=window_df
                )
            ),

            # ----------------------------------------------
            # LATEST CONTEXT
            # ----------------------------------------------

            "latest_context": {

                "dominant_emotion": (
                    clean_scalar(
                        latest_row.get(
                            "DOMINANT_EMOTION"
                        )
                    )
                ),

                "dominant_expectation_experience_label": (
                    clean_scalar(
                        latest_row.get(
                            "DOMINANT_EXPECTATION_EXPERIENCE_LABEL"
                        )
                    )
                ),

                "season": (
                    clean_scalar(
                        latest_row.get(
                            "SEASON_FLAG"
                        )
                    )
                ),

                "priority_alert_flag": (
                    clean_scalar(
                        latest_row.get(
                            "PRIORITY_ALERT_FLAG"
                        )
                    )
                ),

                "emerging_negative_topic": (
                    parse_topic_list(
                        latest_row.get(
                            "EMERGING_NEGATIVE_TOPIC"
                        )
                    )
                ),
            },

            # ----------------------------------------------
            # METRIC DEFINITIONS
            # ----------------------------------------------

            "metric_definitions": {

                metric: (
                    METRIC_DEFINITIONS[
                        metric
                    ]
                )

                for metric
                in selected_metrics

                if metric
                in METRIC_DEFINITIONS
            },

            # ----------------------------------------------
            # PANEL HISTORY
            # ----------------------------------------------

            "weekly_history": (
                weekly_history
            ),

            "trends": (
                trends
            ),

            # ----------------------------------------------
            # TOPIC EVIDENCE
            # ----------------------------------------------

            "topic_summary": {

                "raw_weekly_topics": (
                    raw_topic_summary
                ),

                "managerial_categories": (
                    managerial_topic_summary
                ),

                "interpretation_note": (
                    "Topic-source metrics must be interpreted "
                    "separately. experience_weeks may include "
                    "positive or neutral experiences. "
                    "violated_expectation_weeks and "
                    "emerging_negative_weeks indicate negative "
                    "evidence. negative_review_mentions are based "
                    "only on review_text_disliked. Cross-structure "
                    "topic occurrences may refer to the same "
                    "underlying issue and are not independent "
                    "complaints."
                ),
            },

            # ----------------------------------------------
            # SIGNALS
            # ----------------------------------------------

            "preliminary_signals": (
                preliminary_signals
            ),

            # ----------------------------------------------
            # GENERIC LLM INPUT
            # ----------------------------------------------

            "review_sample": (
                review_sample
            ),

            "review_sample_metadata": (
                review_sample_metadata
            ),

            # ----------------------------------------------
            # EVIDENCE-INFORMED INPUT
            # ----------------------------------------------

            "complaints": (
                complaints
            ),

            "complaint_metadata": (
                complaint_metadata
            ),
        }

        return case