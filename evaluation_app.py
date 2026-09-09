# evaluation_app.py

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials


# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

EVALUATION_DIR = (
    BASE_DIR
    / "src"
    / "outputs"
    / "human_evaluation_v2"
)

CASES_DIR = (
    EVALUATION_DIR
    / "blinded_cases"
)

GOOGLE_SHEET_NAME = "hotel_agent_human_evaluation"
GOOGLE_WORKSHEET_NAME = "responses"


RATING_DIMENSIONS = {
    "prioritization_quality": (
        "Prioritization quality",
        "The decision appropriately identifies and ranks "
        "the most important managerial issues.",
    ),

    "evidence_alignment": (
        "Evidence alignment",
        "The conclusions and recommendations are well "
        "supported by the case information provided.",
    ),

    "actionability": (
        "Actionability",
        "The recommendations are sufficiently concrete "
        "and actionable for hotel management.",
    ),

    "proportionality": (
        "Proportionality",
        "The proposed actions are proportionate to the "
        "strength and severity of the available evidence.",
    ),

    "overall_usefulness": (
        "Overall managerial usefulness",
        "Overall, this decision would be useful as a basis "
        "for managerial decision-making.",
    ),
}


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Hotel Decision Evaluation",
    page_icon="🏨",
    layout="wide",
)


# ==========================================================
# JSON / CASE HELPERS
# ==========================================================

def load_json(path: Path) -> dict:
    """
    Load one blinded evaluation case.
    """

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def get_case_files() -> list[Path]:
    """
    Return all blinded pilot cases.
    """

    files = sorted(
        CASES_DIR.glob("P*.json")
    )

    if not files:
        st.error(
            f"No blinded cases found in {CASES_DIR}"
        )
        st.stop()

    return files


# ==========================================================
# GOOGLE SHEETS
# ==========================================================

@st.cache_resource
def get_google_worksheet():
    """
    Connect to the private Google Sheet used to persist
    evaluator responses.
    """

    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]

        credentials_dict = dict(
            st.secrets["gcp_service_account"]
        )

        credentials = (
            Credentials.from_service_account_info(
                credentials_dict,
                scopes=scopes,
            )
        )

        client = gspread.authorize(
            credentials
        )

        spreadsheet = client.open(
            GOOGLE_SHEET_NAME
        )

        worksheet = spreadsheet.worksheet(
            GOOGLE_WORKSHEET_NAME
        )

        return worksheet

    except Exception as exc:
        st.error(
            "Could not connect to the evaluation database."
        )

        st.caption(
            "Please contact the study administrator."
        )

        # Helpful locally, but avoids showing credentials.
        st.code(
            f"{type(exc).__name__}: {exc}"
        )

        st.stop()


def get_all_responses() -> pd.DataFrame:
    """
    Load all stored evaluation responses.
    """

    worksheet = get_google_worksheet()

    records = worksheet.get_all_records()

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(
        records
    )


def load_existing_responses(
    evaluator_id: str,
) -> pd.DataFrame:
    """
    Return responses belonging only to the current evaluator.
    """

    df = get_all_responses()

    if df.empty:
        return pd.DataFrame()

    if "evaluator_id" not in df.columns:
        return pd.DataFrame()

    return (
        df[
            df["evaluator_id"]
            .astype(str)
            .str.strip()
            == str(evaluator_id).strip()
        ]
        .copy()
    )


def save_response(
    response: dict,
    evaluator_id: str,
):
    """
    Save one evaluation to Google Sheets.

    If the sheet is empty, create the header automatically.

    If the same evaluator submits the same case again,
    replace the previous response instead of creating
    a duplicate.
    """

    worksheet = get_google_worksheet()

    expected_headers = list(response.keys())

    # ======================================================
    # READ CURRENT SHEET
    # ======================================================

    all_values = worksheet.get_all_values()

    # Remove completely empty rows, if Google returns them
    non_empty_rows = [
        row
        for row in all_values
        if any(
            str(value).strip()
            for value in row
        )
    ]

    # ======================================================
    # FIRST EVER RESPONSE
    # ======================================================

    if not non_empty_rows:

        worksheet.clear()

        # Write header
        worksheet.update(
            range_name="A1",
            values=[expected_headers],
            value_input_option="RAW",
        )

        # Write first response
        worksheet.update(
            range_name="A2",
            values=[
                [
                    response.get(header, "")
                    for header in expected_headers
                ]
            ],
            value_input_option="RAW",
        )

        return

    # ======================================================
    # EXISTING SHEET
    # ======================================================

    headers = [
        str(value).strip()
        for value in non_empty_rows[0]
    ]

    # Remove empty trailing cells
    while headers and not headers[-1]:
        headers.pop()

    if headers != expected_headers:

        raise ValueError(
            "The Google Sheet columns do not match "
            "the evaluation response schema.\n\n"
            f"Expected: {expected_headers}\n\n"
            f"Found: {headers}"
        )

    # ======================================================
    # LOOK FOR EXISTING EVALUATION
    # ======================================================

    records = worksheet.get_all_records()

    existing_row = None

    for row_number, record in enumerate(
        records,
        start=2,
    ):

        same_evaluator = (
            str(
                record.get(
                    "evaluator_id",
                    ""
                )
            ).strip()
            == str(evaluator_id).strip()
        )

        same_case = (
            str(
                record.get(
                    "evaluation_case_id",
                    ""
                )
            ).strip()
            == str(
                response[
                    "evaluation_case_id"
                ]
            ).strip()
        )

        if same_evaluator and same_case:

            existing_row = row_number
            break

    # ======================================================
    # BUILD ROW
    # ======================================================

    row_values = [
        response.get(
            header,
            ""
        )
        for header in headers
    ]

    # ======================================================
    # REPLACE EXISTING RESPONSE
    # ======================================================

    if existing_row is not None:

        last_column = gspread.utils.rowcol_to_a1(
            existing_row,
            len(headers),
        )

        worksheet.update(
            range_name=(
                f"A{existing_row}:{last_column}"
            ),
            values=[
                row_values
            ],
            value_input_option="RAW",
        )

    # ======================================================
    # NEW RESPONSE
    # ======================================================

    else:

        worksheet.append_row(
            row_values,
            value_input_option="RAW",
        )
# ==========================================================
# DISPLAY HELPERS
# ==========================================================

def display_hotel_information(
    info: dict,
):
    """
    Display basic hotel information.
    """

    st.subheader(
        "Hotel context"
    )

    col1, col2, col3 = st.columns(3)

    stars = info.get(
        "stars"
    )

    region = info.get(
        "region"
    )

    rooms = info.get(
        "number_of_rooms"
    )

    col1.metric(
        "Hotel category",
        (
            f"{stars} stars"
            if stars is not None
            else "Not available"
        ),
    )

    col2.metric(
        "Region",
        (
            region
            if region
            else "Not available"
        ),
    )

    col3.metric(
        "Number of rooms",
        (
            rooms
            if rooms is not None
            else "Not available"
        ),
    )


def display_performance(
    weekly_history: list[dict],
):
    """
    Display the four-week hotel performance table.
    """

    st.subheader(
        "4-week hotel performance"
    )

    df = pd.DataFrame(
        weekly_history
    )

    rename = {
        "WEEK_YEAR": "Week",
        "week_date": "Week starting",
        "REVPAR_WEEK": "RevPAR",
        "REVPOR_WEEK": "RevPOR",
        "TREVPAR_WEEK": "TRevPAR",
        "TAXA_OCUPACAO": "Occupancy (%)",
        "N_REVIEWS": "Reviews",
        "AVG_RATING": "Average rating",
    }

    df = df.rename(
        columns=rename
    )

    numeric_columns = [
        "RevPAR",
        "RevPOR",
        "TRevPAR",
        "Occupancy (%)",
        "Average rating",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = (
                pd.to_numeric(
                    df[column],
                    errors="coerce",
                )
                .round(2)
            )

    if "Reviews" in df.columns:
        df["Reviews"] = (
            pd.to_numeric(
                df["Reviews"],
                errors="coerce",
            )
            .round(0)
            .astype("Int64")
        )

    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "RevPAR = revenue per available room; "
        "RevPOR = revenue per occupied room; "
        "TRevPAR = total revenue per available room."
    )


def display_reviews(
    reviews: list[dict],
):
    """
    Display the neutral recent-review sample.
    """

    st.subheader(
        "Recent customer reviews"
    )

    st.caption(
        "The reviews below constitute the common customer "
        "evidence presented for this case."
    )

    for index, review in enumerate(
        reviews,
        start=1,
    ):
        rating = review.get(
            "rating"
        )

        date = review.get(
            "review_date"
        )

        country = review.get(
            "country"
        )

        stay_type = review.get(
            "stay_type"
        )

        title = (
            f"Review {index}"
            f"  ·  Rating: {rating}"
            f"  ·  {date}"
        )

        with st.expander(
            title,
            expanded=False,
        ):
            metadata = []

            if country:
                metadata.append(
                    f"Country: {country}"
                )

            if stay_type:
                metadata.append(
                    f"Stay type: {stay_type}"
                )

            if metadata:
                st.caption(
                    " | ".join(
                        metadata
                    )
                )

            liked = review.get(
                "liked"
            )

            disliked = review.get(
                "disliked"
            )

            if liked:
                st.markdown(
                    "**Liked**"
                )

                st.write(
                    liked
                )

            if disliked:
                st.markdown(
                    "**Disliked**"
                )

                st.write(
                    disliked
                )

            if not liked and not disliked:
                st.write(
                    "No written comment."
                )


def display_strengths(
    strengths: list,
):
    """
    Display strengths identified by the decision system.
    """

    if not strengths:
        return

    st.markdown(
        "#### Strengths to preserve"
    )

    for strength in strengths:

        if isinstance(
            strength,
            dict,
        ):
            area = strength.get(
                "area",
                "Strength",
            )

            reason = strength.get(
                "reason"
            )

            implication = strength.get(
                "management_implication"
            )

            st.markdown(
                f"**{area}**"
            )

            if reason:
                st.write(
                    reason
                )

            if implication:
                st.caption(
                    "Management implication"
                )

                st.write(
                    implication
                )

        else:
            st.write(
                f"• {strength}"
            )


def display_decision(
    decision: dict,
    label: str,
):
    """
    Display one blinded managerial decision.
    """

    st.subheader(
        f"Decision {label}"
    )

    risk = decision.get(
        "overall_risk"
    )

    if risk:
        st.markdown(
            f"**Overall risk: {str(risk).title()}**"
        )

    rationale = decision.get(
        "overall_risk_rationale"
    )

    if rationale:
        st.markdown(
            "**Overall risk rationale**"
        )

        st.write(
            rationale
        )

    summary = decision.get(
        "executive_summary"
    )

    if summary:
        st.markdown(
            "**Executive summary**"
        )

        st.write(
            summary
        )

    priorities = decision.get(
        "priorities",
        [],
    )

    st.markdown(
        "#### Managerial priorities"
    )

    for priority in priorities:

        rank = priority.get(
            "rank"
        )

        area = priority.get(
            "area",
            "Priority",
        )

        priority_level = priority.get(
            "managerial_priority"
        )

        horizon = priority.get(
            "action_horizon"
        )

        intervention = priority.get(
            "intervention_type"
        )

        heading = (
            f"Priority {rank}: {area}"
        )

        with st.expander(
            heading,
            expanded=True,
        ):
            metadata = []

            if priority_level:
                metadata.append(
                    "Priority level: "
                    f"{priority_level}"
                )

            if horizon:
                metadata.append(
                    "Horizon: "
                    f"{horizon}"
                )

            if intervention:
                metadata.append(
                    "Intervention: "
                    f"{intervention}"
                )

            if metadata:
                st.caption(
                    " | ".join(
                        metadata
                    )
                )

            problem = priority.get(
                "problem"
            )

            if problem:
                st.markdown(
                    "**Problem**"
                )

                st.write(
                    problem
                )

            action = priority.get(
                "recommended_action"
            )

            if action:
                st.markdown(
                    "**Recommended action**"
                )

                st.write(
                    action
                )

            rationale = priority.get(
                "rationale"
            )

            if rationale:
                st.markdown(
                    "**Rationale**"
                )

                st.write(
                    rationale
                )

    display_strengths(
        decision.get(
            "strengths_to_preserve",
            [],
        )
    )

    limitations = decision.get(
        "limitations",
        [],
    )

    if limitations:
        with st.expander(
            "Limitations",
            expanded=False,
        ):
            for limitation in limitations:
                st.write(
                    f"• {limitation}"
                )


# ==========================================================
# RATING FORM
# ==========================================================

def decision_rating_form(
    label: str,
    case_id: str,
) -> dict:
    """
    Display the five evaluation dimensions.

    No rating is pre-selected.
    """

    st.markdown(
        f"### Evaluate Decision {label}"
    )

    st.caption(
        "Rate each statement from 1 "
        "(Strongly disagree) to 7 "
        "(Strongly agree)."
    )

    ratings = {}

    for key, (
        title,
        description,
    ) in RATING_DIMENSIONS.items():

        st.markdown(
            f"**{title}**"
        )

        st.caption(
            description
        )

        ratings[key] = st.radio(
            label=(
                f"{title} — "
                f"Decision {label}"
            ),
            options=[
                1,
                2,
                3,
                4,
                5,
                6,
                7,
            ],
            index=None,
            horizontal=True,
            key=(
                f"{case_id}_"
                f"{label}_"
                f"{key}"
            ),
            label_visibility="collapsed",
        )

        st.caption(
            "1 = Strongly disagree · "
            "4 = Neither agree nor disagree · "
            "7 = Strongly agree"
        )

        st.write("")

    return ratings


# ==========================================================
# SESSION INITIALIZATION
# ==========================================================

case_files = get_case_files()

if (
    "current_case_index"
    not in st.session_state
):
    st.session_state[
        "current_case_index"
    ] = 0


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title(
    "Evaluation"
)

evaluator_id = (
    st.sidebar.text_input(
        "Evaluator ID",
        placeholder="e.g. E01",
    )
)

st.sidebar.caption(
    "Use the anonymous evaluator code provided "
    "by the study administrator."
)

if not evaluator_id:

    st.title(
        "Hotel Managerial Decision Evaluation"
    )

    st.info(
        "Enter your evaluator ID in the sidebar "
        "to begin."
    )

    st.stop()


# ==========================================================
# GOOGLE SHEETS CONNECTION
# ==========================================================

# Establish connection before evaluation begins.
get_google_worksheet()


# ==========================================================
# LOAD EXISTING RESPONSES
# ==========================================================

existing_responses = (
    load_existing_responses(
        evaluator_id
    )
)

completed_cases = set()

if not existing_responses.empty:

    completed_cases = set(
        existing_responses[
            "evaluation_case_id"
        ]
        .astype(str)
        .tolist()
    )


# ==========================================================
# SIDEBAR PROGRESS
# ==========================================================

st.sidebar.metric(
    "Completed cases",
    f"{len(completed_cases)} / {len(case_files)}",
)

case_labels = [
    path.stem
    for path in case_files
]

current_index = min(
    st.session_state[
        "current_case_index"
    ],
    len(case_files) - 1,
)

selected_case = (
    st.sidebar.selectbox(
        "Case",
        options=case_labels,
        index=current_index,
    )
)

selected_index = (
    case_labels.index(
        selected_case
    )
)

st.session_state[
    "current_case_index"
] = selected_index


# ==========================================================
# LOAD CURRENT CASE
# ==========================================================

case_path = (
    CASES_DIR
    / f"{selected_case}.json"
)

case = load_json(
    case_path
)

case_id = case[
    "evaluation_case_id"
]

case_info = case[
    "case_information"
]

decision_a = case[
    "decision_a"
]

decision_b = case[
    "decision_b"
]


# ==========================================================
# HEADER
# ==========================================================

st.title(
    "Hotel Managerial Decision Evaluation"
)

st.markdown(
    f"## Case {case_id}"
)

if case_id in completed_cases:

    st.success(
        "You have already evaluated this case. "
        "Submitting again will replace your previous response."
    )

window_start = (
    case_info.get(
        "window_start"
    )
)

window_end = (
    case_info.get(
        "window_end"
    )
)

decision_week = (
    case_info.get(
        "decision_week"
    )
)

st.write(
    f"**Evaluation window:** "
    f"{window_start} to {window_end}"
)

st.write(
    f"**Decision week:** "
    f"{decision_week}"
)


# ==========================================================
# INSTRUCTIONS
# ==========================================================

with st.expander(
    "Evaluation instructions",
    expanded=False,
):

    st.write(
        """
You will review a hotel case and two AI-generated managerial
decisions, labelled Decision A and Decision B.

Both decisions concern the same hotel case.

Please evaluate each decision independently based only on the
case information provided.

Focus on whether the decision identifies the relevant managerial
issues, remains grounded in the available evidence, proposes
proportionate and actionable responses, and would be useful for
managerial decision-making.

For each criterion, use the 1–7 scale provided.

There are no correct or incorrect answers. We are interested in
your professional judgment.

The identity of the systems generating Decision A and Decision B
is intentionally hidden.
        """
    )


# ==========================================================
# CASE INFORMATION
# ==========================================================

st.divider()

st.header(
    "1. Case information"
)

display_hotel_information(
    case_info.get(
        "hotel_information",
        {},
    )
)

display_performance(
    case_info.get(
        "weekly_history",
        [],
    )
)

display_reviews(
    case_info.get(
        "recent_reviews",
        [],
    )
)


# ==========================================================
# DECISION A
# ==========================================================

st.divider()

st.header(
    "2. Decision A"
)

display_decision(
    decision_a,
    "A",
)

ratings_a = (
    decision_rating_form(
        label="A",
        case_id=case_id,
    )
)


# ==========================================================
# DECISION B
# ==========================================================

st.divider()

st.header(
    "3. Decision B"
)

display_decision(
    decision_b,
    "B",
)

ratings_b = (
    decision_rating_form(
        label="B",
        case_id=case_id,
    )
)


# ==========================================================
# COMPARATIVE EVALUATION
# ==========================================================

st.divider()

st.header(
    "4. Comparative evaluation"
)

preference = st.radio(
    (
        "Considering the available evidence, which "
        "decision would you prefer to use as a basis "
        "for managerial decision-making?"
    ),
    options=[
        "Decision A",
        "No preference",
        "Decision B",
    ],
    index=None,
    key=(
        f"{case_id}_"
        "preference"
    ),
)

overprescription = st.radio(
    (
        "Which decision, if any, is more likely to "
        "recommend actions that go beyond what is "
        "justified by the available evidence?"
    ),
    options=[
        "Decision A",
        "Both equally",
        "Decision B",
        "Neither",
    ],
    index=None,
    key=(
        f"{case_id}_"
        "overprescription"
    ),
)

comments = st.text_area(
    "Optional comments or reasoning",
    placeholder=(
        "You may briefly explain your evaluation, "
        "especially when the two decisions differ "
        "substantially."
    ),
    key=(
        f"{case_id}_"
        "comments"
    ),
)


# ==========================================================
# SUBMISSION
# ==========================================================

st.divider()

if st.button(
    "Save evaluation",
    type="primary",
    width="stretch",
):

    # ------------------------------------------------------
    # VALIDATE ALL RATINGS
    # ------------------------------------------------------

    missing_a = [
        key
        for key, value in ratings_a.items()
        if value is None
    ]

    missing_b = [
        key
        for key, value in ratings_b.items()
        if value is None
    ]

    if missing_a or missing_b:

        st.error(
            "Please complete all five ratings for "
            "Decision A and all five ratings for "
            "Decision B."
        )

        st.stop()

    # ------------------------------------------------------
    # VALIDATE COMPARATIVE QUESTIONS
    # ------------------------------------------------------

    if preference is None:

        st.error(
            "Please answer the overall preference question."
        )

        st.stop()

    if overprescription is None:

        st.error(
            "Please answer the question about recommendations "
            "that go beyond the available evidence."
        )

        st.stop()

    # ------------------------------------------------------
    # BUILD RESPONSE
    # ------------------------------------------------------

    response = {
        "evaluation_version": (
            case.get(
                "evaluation_version"
            )
        ),

        "evaluator_id": (
            evaluator_id.strip()
        ),

        "evaluation_case_id": (
            case_id
        ),

        "submitted_at": (
            datetime.now(
                timezone.utc
            )
            .isoformat(
                timespec="seconds"
            )
        ),

        # Decision A
        "A_prioritization_quality": (
            ratings_a[
                "prioritization_quality"
            ]
        ),

        "A_evidence_alignment": (
            ratings_a[
                "evidence_alignment"
            ]
        ),

        "A_actionability": (
            ratings_a[
                "actionability"
            ]
        ),

        "A_proportionality": (
            ratings_a[
                "proportionality"
            ]
        ),

        "A_overall_usefulness": (
            ratings_a[
                "overall_usefulness"
            ]
        ),

        # Decision B
        "B_prioritization_quality": (
            ratings_b[
                "prioritization_quality"
            ]
        ),

        "B_evidence_alignment": (
            ratings_b[
                "evidence_alignment"
            ]
        ),

        "B_actionability": (
            ratings_b[
                "actionability"
            ]
        ),

        "B_proportionality": (
            ratings_b[
                "proportionality"
            ]
        ),

        "B_overall_usefulness": (
            ratings_b[
                "overall_usefulness"
            ]
        ),

        # Comparative judgments
        "preferred_decision": (
            preference
        ),

        "overprescription_decision": (
            overprescription
        ),

        "comments": (
            comments.strip()
        ),
    }

    # ------------------------------------------------------
    # SAVE TO GOOGLE SHEETS
    # ------------------------------------------------------

    try:

        save_response(
            response=response,
            evaluator_id=evaluator_id,
        )

    except Exception as exc:

        st.error(
            "The evaluation could not be saved. "
            "Please do not close this page and contact "
            "the study administrator."
        )

        st.code(
            f"{type(exc).__name__}: {exc}"
        )

        st.stop()

    # ------------------------------------------------------
    # SUCCESS
    # ------------------------------------------------------

    st.success(
        f"Evaluation for {case_id} saved successfully."
    )

    # ------------------------------------------------------
    # MOVE TO NEXT CASE
    # ------------------------------------------------------

    if (
        selected_index
        < len(case_files) - 1
    ):

        st.session_state[
            "current_case_index"
        ] = (
            selected_index + 1
        )

        st.rerun()

    else:

        st.balloons()

        st.success(
            "You have reached the final case. "
            "Thank you for completing the evaluation."
        )