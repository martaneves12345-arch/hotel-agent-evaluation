# evaluation_app.py

from __future__ import annotations
import random

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
GOOGLE_ASSIGNMENTS_WORKSHEET_NAME = "assignments"
GOOGLE_SCREENING_WORKSHEET_NAME = "screening"

# Prolific completion paths
PROLIFIC_COMPLETION_URL = (
    "https://app.prolific.com/submissions/complete?cc=C88K5Q1C"
)
PROLIFIC_SCREENED_OUT_URL = (
    "https://app.prolific.com/submissions/complete?cc=C1HKGMMH"
)


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


@st.cache_resource
def get_assignments_worksheet():
    """
    Connect to the worksheet used to reserve case assignments.

    The worksheet is created automatically if it does not yet exist.
    If it exists but is empty, the required headers are added.
    """

    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]

        credentials_dict = dict(
            st.secrets["gcp_service_account"]
        )

        credentials = Credentials.from_service_account_info(
            credentials_dict,
            scopes=scopes,
        )

        client = gspread.authorize(credentials)

        spreadsheet = client.open(
            GOOGLE_SHEET_NAME
        )

        try:
            worksheet = spreadsheet.worksheet(
                GOOGLE_ASSIGNMENTS_WORKSHEET_NAME
            )

        except gspread.WorksheetNotFound:

            worksheet = spreadsheet.add_worksheet(
                title=GOOGLE_ASSIGNMENTS_WORKSHEET_NAME,
                rows=1000,
                cols=5,
            )

        expected_headers = [
            "evaluator_id",
            "evaluation_case_id",
            "assigned_at",
            "status",
            "submitted_at",
        ]

        values = worksheet.get_all_values()

        # --------------------------------------------------
        # EMPTY WORKSHEET
        # --------------------------------------------------

        if (
            not values
            or not any(
                str(cell).strip()
                for row in values
                for cell in row
            )
        ):

            worksheet.clear()

            worksheet.update(
                range_name="A1",
                values=[expected_headers],
                value_input_option="RAW",
            )

            return worksheet

        # --------------------------------------------------
        # EXISTING WORKSHEET WITH CONTENT
        # --------------------------------------------------

        headers = [
            str(value).strip()
            for value in values[0]
        ]

        while headers and not headers[-1]:
            headers.pop()

        if headers != expected_headers:

            raise ValueError(
                "The assignments worksheet columns do not match "
                "the expected schema.\n\n"
                f"Expected: {expected_headers}\n\n"
                f"Found: {headers}"
            )

        return worksheet

    except Exception as exc:

        st.error(
            "Could not connect to the case-assignment database."
        )

        st.caption(
            "Please contact the study administrator."
        )

        st.code(
            f"{type(exc).__name__}: {exc}"
        )

        st.stop()



@st.cache_resource
def get_screening_worksheet():
    """
    Connect to the worksheet used to store screening answers.

    The worksheet is created automatically if it does not yet exist.
    If it exists but is empty, the required headers are added.
    """

    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]

        credentials_dict = dict(
            st.secrets["gcp_service_account"]
        )

        credentials = Credentials.from_service_account_info(
            credentials_dict,
            scopes=scopes,
        )

        client = gspread.authorize(credentials)

        spreadsheet = client.open(
            GOOGLE_SHEET_NAME
        )

        try:
            worksheet = spreadsheet.worksheet(
                GOOGLE_SCREENING_WORKSHEET_NAME
            )

        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(
                title=GOOGLE_SCREENING_WORKSHEET_NAME,
                rows=1000,
                cols=12,
            )

        expected_headers = [
            "evaluator_id",
            "prolific_pid",
            "study_id",
            "session_id",
            "recruitment_source",
            "hotel_experience",
            "role_category",
            "decision_responsibility",
            "years_experience",
            "eligible",
            "screened_at",
        ]

        values = worksheet.get_all_values()

        if (
            not values
            or not any(
                str(cell).strip()
                for row in values
                for cell in row
            )
        ):
            worksheet.clear()
            worksheet.update(
                range_name="A1",
                values=[expected_headers],
                value_input_option="RAW",
            )
            return worksheet

        headers = [
            str(value).strip()
            for value in values[0]
        ]

        while headers and not headers[-1]:
            headers.pop()

        if headers != expected_headers:
            raise ValueError(
                "The screening worksheet columns do not match "
                "the expected schema.\\n\\n"
                f"Expected: {expected_headers}\\n\\n"
                f"Found: {headers}"
            )

        return worksheet

    except Exception as exc:
        st.error(
            "Could not connect to the screening database."
        )
        st.caption(
            "Please contact the study administrator."
        )
        st.code(
            f"{type(exc).__name__}: {exc}"
        )
        st.stop()


def save_screening_result(
    evaluator_id: str,
    prolific_pid: str,
    study_id: str,
    session_id: str,
    recruitment_source: str,
    hotel_experience: str,
    role_category: str,
    decision_responsibility: str,
    years_experience: str,
    eligible: bool,
) -> None:
    """Save or update one participant's screening result."""

    worksheet = get_screening_worksheet()
    records = worksheet.get_all_records()

    screened_at = datetime.now(
        timezone.utc
    ).isoformat(
        timespec="seconds"
    )

    row_values = [
        evaluator_id,
        prolific_pid,
        study_id,
        session_id,
        recruitment_source,
        hotel_experience,
        role_category,
        decision_responsibility,
        years_experience,
        "yes" if eligible else "no",
        screened_at,
    ]

    existing_row = None

    for row_number, record in enumerate(
        records,
        start=2,
    ):
        if (
            str(record.get("evaluator_id", "")).strip()
            == str(evaluator_id).strip()
        ):
            existing_row = row_number
            break

    if existing_row is None:
        worksheet.append_row(
            row_values,
            value_input_option="RAW",
        )
    else:
        last_column = gspread.utils.rowcol_to_a1(
            existing_row,
            len(row_values),
        )
        worksheet.update(
            range_name=f"A{existing_row}:{last_column}",
            values=[row_values],
            value_input_option="RAW",
        )


def get_query_parameter(name: str) -> str:
    """Read one optional URL parameter from Streamlit."""
    value = st.query_params.get(name, "")

    if isinstance(value, list):
        value = value[0] if value else ""

    return str(value).strip()


def get_all_assignments() -> pd.DataFrame:
    """Load all case reservations."""

    worksheet = get_assignments_worksheet()
    records = worksheet.get_all_records()

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


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

def generate_unique_evaluator_id() -> str:
    """
    Generate a unique anonymous evaluator ID.

    IDs are checked against both completed responses and current
    case reservations.
    """

    responses = get_all_responses()
    assignments = get_all_assignments()

    existing_ids = set()

    if (
        not responses.empty
        and "evaluator_id" in responses.columns
    ):
        existing_ids.update(
            responses["evaluator_id"]
            .astype(str)
            .str.strip()
            .tolist()
        )

    if (
        not assignments.empty
        and "evaluator_id" in assignments.columns
    ):
        existing_ids.update(
            assignments["evaluator_id"]
            .astype(str)
            .str.strip()
            .tolist()
        )

    while True:
        evaluator_id = f"E{random.randint(1000, 9999)}"

        if evaluator_id not in existing_ids:
            return evaluator_id


def reserve_balanced_random_case(
    case_files: list[Path],
    evaluator_id: str,
) -> str:
    """
    Reserve exactly one case for the evaluator.

    The function counts existing reservations, identifies the cases
    with the fewest assignments, randomly chooses among those cases,
    and immediately records the reservation in Google Sheets.

    If this evaluator already has a reservation, that same case is
    returned instead of assigning a new one.
    """

    worksheet = get_assignments_worksheet()
    assignments = get_all_assignments()

    case_ids = [
        path.stem
        for path in case_files
    ]

    if (
        not assignments.empty
        and "evaluator_id" in assignments.columns
    ):
        previous = assignments[
            assignments["evaluator_id"]
            .astype(str)
            .str.strip()
            == str(evaluator_id).strip()
        ]

        if not previous.empty:
            previous_case = str(
                previous.iloc[-1]["evaluation_case_id"]
            ).strip()

            if previous_case in case_ids:
                return previous_case

    if (
        assignments.empty
        or "evaluation_case_id" not in assignments.columns
    ):
        case_counts = {
            case_id: 0
            for case_id in case_ids
        }
    else:
        counts = (
            assignments["evaluation_case_id"]
            .astype(str)
            .value_counts()
            .to_dict()
        )

        case_counts = {
            case_id: counts.get(case_id, 0)
            for case_id in case_ids
        }

    min_count = min(case_counts.values())

    least_assigned_cases = [
        case_id
        for case_id, count in case_counts.items()
        if count == min_count
    ]

    assigned_case = random.choice(
        least_assigned_cases
    )

    assigned_at = datetime.now(
        timezone.utc
    ).isoformat(
        timespec="seconds"
    )

    worksheet.append_row(
        [
            evaluator_id,
            assigned_case,
            assigned_at,
            "assigned",
            "",
        ],
        value_input_option="RAW",
    )

    return assigned_case


def mark_assignment_completed(
    evaluator_id: str,
    case_id: str,
) -> None:
    """Mark the evaluator's reserved case as completed."""

    worksheet = get_assignments_worksheet()
    records = worksheet.get_all_records()

    completed_at = datetime.now(
        timezone.utc
    ).isoformat(
        timespec="seconds"
    )

    for row_number, record in enumerate(
        records,
        start=2,
    ):
        same_evaluator = (
            str(record.get("evaluator_id", "")).strip()
            == str(evaluator_id).strip()
        )

        same_case = (
            str(record.get("evaluation_case_id", "")).strip()
            == str(case_id).strip()
        )

        if same_evaluator and same_case:
            worksheet.update(
                range_name=f"D{row_number}:E{row_number}",
                values=[["completed", completed_at]],
                value_input_option="RAW",
            )
            return


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


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title(
    "Evaluation"
)

# ==========================================================
# AUTOMATIC EVALUATOR ID
# ==========================================================

# Establish Google Sheets connection first
get_google_worksheet()

if "evaluator_id" not in st.session_state:

    st.session_state["evaluator_id"] = (
        generate_unique_evaluator_id()
    )

evaluator_id = (
    st.session_state["evaluator_id"]
)

st.sidebar.markdown(
    f"**Evaluator ID:** `{evaluator_id}`"
)

st.sidebar.caption(
    "This anonymous ID was generated automatically "
    "for your evaluation session."
)



# ==========================================================
# PROLIFIC / RECRUITMENT PARAMETERS
# ==========================================================

prolific_pid = get_query_parameter("PROLIFIC_PID")
study_id = get_query_parameter("STUDY_ID")
session_id = get_query_parameter("SESSION_ID")

recruitment_source = (
    "prolific"
    if prolific_pid
    else "direct"
)


# ==========================================================
# ELIGIBILITY SCREENING
# ==========================================================

if "screening_completed" not in st.session_state:
    st.session_state["screening_completed"] = False

if "screening_eligible" not in st.session_state:
    st.session_state["screening_eligible"] = None

if not st.session_state["screening_completed"]:

    st.title(
        "Hotel Management Study"
    )

    st.header(
        "Eligibility screening"
    )

    st.write(
        "Before starting the evaluation, please answer a few "
        "short questions about your professional experience."
    )

    st.caption(
        "These questions are used only to determine whether your "
        "professional background matches the requirements of this study."
    )

    hotel_experience = st.radio(
        (
            "Do you currently work, or have you previously worked, "
            "in a hotel or other accommodation establishment "
            "(e.g., hotel, resort, aparthotel)?"
        ),
        options=[
            "Yes, currently",
            "Yes, previously",
            "No",
        ],
        index=None,
        key="screen_hotel_experience",
    )

    role_category = st.selectbox(
        (
            "Which of the following best describes your current "
            "or previous role in the hotel/accommodation sector?"
        ),
        options=[
            "",
            "General Manager / Hotel Manager",
            "Department Manager",
            "Revenue / Commercial / Sales Manager",
            "Operations Manager",
            "Front Office / Guest Relations",
            "Food & Beverage Management",
            "Other supervisory or managerial role",
            "Non-managerial operational role",
            "Other hotel/accommodation role",
        ],
        index=0,
        key="screen_role_category",
    )

    decision_responsibility = st.radio(
        (
            "Did your role involve supervisory, managerial, "
            "operational decision-making, revenue/commercial decisions, "
            "or responsibility for guest experience?"
        ),
        options=[
            "Yes",
            "No",
        ],
        index=None,
        key="screen_decision_responsibility",
    )

    years_experience = st.selectbox(
        "How much professional experience do you have in hotels/accommodation?",
        options=[
            "",
            "Less than 1 year",
            "1–2 years",
            "3–5 years",
            "6–10 years",
            "More than 10 years",
        ],
        index=0,
        key="screen_years_experience",
    )

    if st.button(
        "Continue",
        type="primary",
        width="stretch",
        key="screening_continue",
    ):

        missing_screening = (
            hotel_experience is None
            or not role_category
            or decision_responsibility is None
            or not years_experience
        )

        if missing_screening:
            st.error(
                "Please answer all screening questions before continuing."
            )
            st.stop()

        eligible = (
            hotel_experience in {
                "Yes, currently",
                "Yes, previously",
            }
            and decision_responsibility == "Yes"
        )

        save_screening_result(
            evaluator_id=evaluator_id,
            prolific_pid=prolific_pid,
            study_id=study_id,
            session_id=session_id,
            recruitment_source=recruitment_source,
            hotel_experience=hotel_experience,
            role_category=role_category,
            decision_responsibility=decision_responsibility,
            years_experience=years_experience,
            eligible=eligible,
        )

        st.session_state["screening_completed"] = True
        st.session_state["screening_eligible"] = eligible

        if eligible:
            st.rerun()

        else:
            st.warning(
                "Thank you for your interest. Based on the eligibility "
                "criteria for this study, you do not qualify for the "
                "main evaluation task."
            )

            if prolific_pid:
                st.link_button(
                    "Return to Prolific",
                    PROLIFIC_SCREENED_OUT_URL,
                    type="primary",
                    width="stretch",
                )
            else:
                st.info(
                    "You may now close this page."
                )

            st.stop()


if st.session_state["screening_eligible"] is False:

    st.warning(
        "Thank you for your interest. Based on the eligibility "
        "criteria for this study, you do not qualify for the "
        "main evaluation task."
    )

    if prolific_pid:
        st.link_button(
            "Return to Prolific",
            PROLIFIC_SCREENED_OUT_URL,
            type="primary",
            width="stretch",
        )
    else:
        st.info(
            "You may now close this page."
        )

    st.stop()


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
# AUTOMATIC CASE ASSIGNMENT
# ==========================================================

if "assigned_case" not in st.session_state:

    st.session_state["assigned_case"] = (
        reserve_balanced_random_case(
            case_files=case_files,
            evaluator_id=evaluator_id,
        )
    )

selected_case = (
    st.session_state["assigned_case"]
)

st.sidebar.markdown(
    f"**Assigned case:** `{selected_case}`"
)

st.sidebar.caption(
    "One case has been automatically assigned "
    "to this evaluation session."
)

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

        mark_assignment_completed(
            evaluator_id=evaluator_id,
            case_id=case_id,
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
    # COMPLETION
    # ------------------------------------------------------
    st.balloons()

    st.success(
        "Thank you. Your evaluation has been completed successfully."
    )

    if prolific_pid:
        st.link_button(
            "Return to Prolific",
            PROLIFIC_COMPLETION_URL,
            type="primary",
            width="stretch",
        )
    else:
        st.info(
            "You may now close this page."
        )

    st.stop()