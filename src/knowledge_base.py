# src/knowledge_base.py

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Optional

import pandas as pd


# ==========================================================
# DEFINIÇÕES DAS MÉTRICAS
# ==========================================================

METRIC_DEFINITIONS: dict[str, dict[str, Any]] = {
    "REVPAR_WEEK": {
        "label": "Weekly RevPAR",
        "meaning": "Weekly revenue per available room.",
        "higher_is_generally_better": True,
    },
    "REVPOR_WEEK": {
        "label": "Weekly RevPOR",
        "meaning": "Weekly revenue per occupied room.",
        "higher_is_generally_better": True,
    },
    "TREVPAR_WEEK": {
        "label": "Weekly TRevPAR",
        "meaning": "Weekly total revenue per available room.",
        "higher_is_generally_better": True,
    },
    "TAXA_OCUPACAO": {
        "label": "Occupancy rate",
        "meaning": "Percentage of available room capacity occupied.",
        "higher_is_generally_better": True,
    },
    "N_REVIEWS": {
        "label": "Number of reviews",
        "meaning": "Number of reviews associated with the hotel-week.",
        "higher_is_generally_better": None,
    },
    "AVG_RATING": {
        "label": "Average rating",
        "meaning": "Average numerical review rating.",
        "higher_is_generally_better": True,
    },
    "SENTIMENT_SCORE_AVG": {
        "label": "Average sentiment",
        "meaning": "Average sentiment score extracted from review text.",
        "higher_is_generally_better": True,
    },
    "SENTIMENT_SCORE_STD": {
        "label": "Sentiment dispersion",
        "meaning": "Dispersion of sentiment across reviews.",
        "higher_is_generally_better": None,
    },
    "EXPECTATION_EXPERIENCE_GAP_AVG": {
        "label": "Expectation–experience gap",
        "meaning": (
            "Average discrepancy between expressed expectations and "
            "reported experience."
        ),
        "higher_is_generally_better": False,
    },
    "SEMANTIC_MISALIGNMENT_SCORE_AVG": {
        "label": "Semantic misalignment",
        "meaning": (
            "Average semantic divergence between expectations and "
            "reported experience."
        ),
        "higher_is_generally_better": False,
    },
    "EXPECTATION_VIOLATION_RATIO": {
        "label": "Expectation violation ratio",
        "meaning": (
            "Share or ratio of expectation-related observations "
            "classified as violated."
        ),
        "higher_is_generally_better": False,
    },
    "MAJOR_MISALIGNMENT_SHARE": {
        "label": "Major misalignment share",
        "meaning": (
            "Share of reviews or review structures showing major "
            "expectation–experience misalignment."
        ),
        "higher_is_generally_better": False,
    },
    "EMOTION_ENTROPY": {
        "label": "Emotional complexity",
        "meaning": (
            "Entropy-based heterogeneity of emotions expressed across "
            "reviews."
        ),
        "higher_is_generally_better": None,
    },
    "EXPECTATION_SHARE": {
        "label": "Expectation share",
        "meaning": "Share of reviews containing identifiable expectations.",
        "higher_is_generally_better": None,
    },
    "HIGH_EXPECTATION_SHARE": {
        "label": "High-expectation share",
        "meaning": "Share of reviews containing high-intensity expectations.",
        "higher_is_generally_better": None,
    },
    "HIGH_MISALIGNMENT_SHARE": {
        "label": "High-misalignment share",
        "meaning": "Share of observations with high semantic misalignment.",
        "higher_is_generally_better": False,
    },
    "PERC_REVIEWS_NEGATIVE": {
        "label": "Reviews containing negative/disliked content",
        "meaning": (
            "Percentage of reviews containing a negative or disliked "
            "text component. It does not mean that the full review or "
            "numerical rating is negative."
        ),
        "higher_is_generally_better": False,
        "critical_interpretation_warning": (
            "Do not interpret this variable as the percentage of guests "
            "giving a negative overall evaluation."
        ),
    },
    "PERC_OPERATIONAL_COMPLAINTS": {
        "label": "Operational complaint share",
        "meaning": (
            "Percentage of complaint content associated with operational "
            "or service-delivery issues."
        ),
        "higher_is_generally_better": False,
    },
    "PERC_PRICE_VALUE_MENTIONS": {
        "label": "Price–value mention share",
        "meaning": (
            "Percentage of review content mentioning price or perceived "
            "value."
        ),
        "higher_is_generally_better": None,
    },
    "POLARIZATION_INDEX": {
        "label": "Review polarization",
        "meaning": (
            "Degree to which review experiences are divided or polarized."
        ),
        "higher_is_generally_better": False,
    },
    "COMPLEXITY_INDEX": {
        "label": "Experience complexity index",
        "meaning": (
            "Composite indicator of heterogeneous and layered customer "
            "experience signals."
        ),
        "higher_is_generally_better": None,
    },
    "SENTIMENT_TREND_SLOPE": {
        "label": "Precomputed sentiment trend slope",
        "meaning": (
            "Trend slope calculated previously in the source panel. "
            "Do not confuse it with the case-window trend recalculated "
            "by the Case Builder."
        ),
        "higher_is_generally_better": None,
    },
}


# ==========================================================
# TAXONOMIA GERENCIAL
# ==========================================================

TOPIC_TAXONOMY: dict[str, dict[str, Any]] = {
    "parking": {
        "label": "Parking",
        "keywords": [
            "parking", "car park", "carpark", "garage",
            "park the car", "reserve parking", "parking reservation",
            "parking cost", "estacionamento", "garagem",
        ],
        "managerial_domain": "access_and_facilities",
        "default_priority": "medium",
    },
    "bathroom": {
        "label": "Bathroom condition",
        "keywords": [
            "bathroom", "toilet", "shower", "bathtub", "washroom",
            "toiletries", "hair conditioner", "body lotion",
            "casa de banho", "duche", "chuveiro", "sanita",
        ],
        "managerial_domain": "rooms_and_maintenance",
        "default_priority": "high",
    },
    "room_comfort": {
        "label": "Room comfort",
        "keywords": [
            "room comfort", "bed", "mattress", "pillow", "noise",
            "temperature", "air conditioning", "air-conditioning",
            "space", "room size", "small room", "soundproof",
            "conforto do quarto", "cama", "colchão", "barulho",
            "ar condicionado", "tamanho do quarto",
        ],
        "managerial_domain": "rooms_and_maintenance",
        "default_priority": "high",
    },
    "room_amenities": {
        "label": "In-room amenities",
        "keywords": [
            "amenities", "in-room amenities", "minibar", "mini bar",
            "coffee machine", "kettle", "iron", "steamer",
            "fridge", "refrigerator", "room facilities",
            "comodidades", "minibar", "chaleira", "ferro",
        ],
        "managerial_domain": "rooms_and_maintenance",
        "default_priority": "medium",
    },
    "cleanliness": {
        "label": "Cleanliness and housekeeping",
        "keywords": [
            "cleanliness", "clean", "dirty", "dust", "housekeeping",
            "room cleaning", "regular housekeeping",
            "limpeza", "limpo", "sujo",
        ],
        "managerial_domain": "housekeeping",
        "default_priority": "high",
    },
    "breakfast": {
        "label": "Breakfast",
        "keywords": [
            "breakfast", "buffet", "coffee", "food variety",
            "breakfast variety", "pequeno-almoço", "café da manhã",
        ],
        "managerial_domain": "food_and_beverage",
        "default_priority": "medium",
    },
    "restaurant_bar": {
        "label": "Restaurant and bar",
        "keywords": [
            "restaurant", "bar", "rooftop bar", "cocktail",
            "dinner", "lunch", "table reservation",
            "reserve tables", "restaurante", "bar", "jantar",
        ],
        "managerial_domain": "food_and_beverage",
        "default_priority": "medium",
    },
    "rooftop_access": {
        "label": "Rooftop access",
        "keywords": [
            "rooftop", "rooftop terrace", "terrace access",
            "access to rooftop", "terraço",
        ],
        "managerial_domain": "access_and_facilities",
        "default_priority": "medium",
    },
    "staff_service": {
        "label": "Staff and service",
        "keywords": [
            "staff", "employee", "reception", "front desk",
            "service", "professionalism", "friendly", "helpful",
            "training", "funcionário", "equipa", "receção",
            "atendimento",
        ],
        "managerial_domain": "service_delivery",
        "default_priority": "high",
    },
    "location_access": {
        "label": "Location and accessibility",
        "keywords": [
            "location", "central", "transport", "metro", "tram",
            "accessibility", "localização", "transportes",
        ],
        "managerial_domain": "location_and_access",
        "default_priority": "low",
    },
    "price_value": {
        "label": "Price–value",
        "keywords": [
            "overpriced", "price", "value", "expensive",
            "cost", "worth", "too much", "preço", "caro",
            "relação qualidade preço",
        ],
        "managerial_domain": "pricing_and_value",
        "default_priority": "high",
    },
    "booking_payment": {
        "label": "Booking and payment",
        "keywords": [
            "booking", "reservation", "payment", "charged",
            "double charge", "invoice", "refund",
            "reserva", "pagamento", "cobrado", "reembolso",
        ],
        "managerial_domain": "commercial_processes",
        "default_priority": "high",
    },
    "wifi_technology": {
        "label": "Wi-Fi and technology",
        "keywords": [
            "wifi", "wi-fi", "internet", "television", "tv",
            "socket", "charger", "technology",
        ],
        "managerial_domain": "technology",
        "default_priority": "medium",
    },
    "spa_pool": {
        "label": "Spa and pool",
        "keywords": [
            "spa", "pool", "swimming pool", "sauna", "steam room",
            "jacuzzi", "wellness", "piscina",
        ],
        "managerial_domain": "leisure_facilities",
        "default_priority": "medium",
    },
}

SOURCE_METRIC_MAPPING = {
    "EXPERIENCE_TOPICS_UNIQUE": "experience_weeks",
    "EXPECTATION_TOPICS_UNIQUE": "expectation_weeks",
    "VIOLATED_EXPECTATION_TOPICS_UNIQUE": "violated_expectation_weeks",
    "LATENT_EXPECTATION_TOPICS_UNIQUE": "latent_expectation_weeks",
    "EMERGING_NEGATIVE_TOPIC": "emerging_negative_weeks",
}



def _normalize_for_matching(text: Any) -> str:
    """Normaliza texto para correspondência lexical simples."""
    if text is None:
        return ""

    value = str(text).lower().strip()

    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        char for char in value
        if not unicodedata.combining(char)
    )

    value = re.sub(r"\s+", " ", value)
    return value


def _contains_keyword(text: str, keyword: str) -> bool:
    """
    Correspondência lexical com fronteiras razoáveis.

    Para expressões multi-palavra usa substring normalizada.
    Para palavras simples usa limites de palavra.
    """
    normalized_keyword = _normalize_for_matching(keyword)

    if not normalized_keyword:
        return False

    if " " in normalized_keyword or "-" in normalized_keyword:
        return normalized_keyword in text

    return bool(
        re.search(
            rf"\b{re.escape(normalized_keyword)}\b",
            text,
        )
    )


def map_text_to_categories(
    text: Any,
    taxonomy: Optional[dict[str, dict[str, Any]]] = None,
) -> list[str]:
    """
    Mapeia um tópico ou texto para zero ou mais categorias gerenciais.
    """
    taxonomy = taxonomy or TOPIC_TAXONOMY
    normalized_text = _normalize_for_matching(text)

    if not normalized_text:
        return []

    matches = []

    for category, config in taxonomy.items():
        keywords = config.get("keywords", [])

        if any(
            _contains_keyword(normalized_text, keyword)
            for keyword in keywords
        ):
            matches.append(category)

    return matches


def _get_text_value(
    row: pd.Series,
    columns: list[str],
) -> str:
    """Concatena apenas os campos textuais indicados."""
    parts = []

    for column in columns:
        if column not in row.index:
            continue

        value = row.get(column)

        if value is None:
            continue

        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass

        value = str(value).strip()

        if value and value.lower() not in {"nan", "none"}:
            parts.append(value)

    return " ".join(parts)


def _iter_weekly_topics(
    weekly_topic_summary: dict[str, list[dict[str, Any]]],
) -> Iterable[tuple[str, int, str]]:
    """
    Produz:
        topic, weeks_present, source_column
    """
    for source_column, topic_rows in weekly_topic_summary.items():
        for row in topic_rows:
            topic = str(row.get("topic", "")).strip()

            if not topic:
                continue

            weeks_present = int(
                row.get(
                    "weeks_present",
                    row.get("weeks_mentioned", 0),
                )
            )

            yield topic, weeks_present, source_column


def build_managerial_topic_summary(
    weekly_topic_summary: dict[str, list[dict[str, Any]]],
    reviews_df: Optional[pd.DataFrame] = None,
    top_n: int = 10,
    taxonomy: Optional[dict[str, dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """
    Agrega tópicos granulares em categorias gerenciais.

    positive_review_mentions usa preferencialmente en_review_text_liked,
    com fallback para review_text_liked.
    negative_review_mentions usa preferencialmente en_review_text_disliked,
    com fallback para review_text_disliked.
    As diferentes fontes semanais são mantidas separadas.
    """
    taxonomy = taxonomy or TOPIC_TAXONOMY

    category_data: dict[str, dict[str, Any]] = {
        category: {
            "category": category,
            "label": config["label"],
            "managerial_domain": config["managerial_domain"],
            "default_priority": config["default_priority"],

            "weeks_present": 0,
            "experience_weeks": 0,
            "expectation_weeks": 0,
            "violated_expectation_weeks": 0,
            "latent_expectation_weeks": 0,
            "emerging_negative_weeks": 0,

            "cross_structure_topic_occurrences": 0,

            "review_mentions_total": 0,
            "positive_review_mentions": 0,
            "negative_review_mentions": 0,

            "matched_raw_topics": set(),
            "source_columns": set(),
        }
        for category, config in taxonomy.items()
    }

    for topic, weeks_present, source_column in _iter_weekly_topics(
        weekly_topic_summary
    ):
        matched_categories = map_text_to_categories(
            topic,
            taxonomy=taxonomy,
        )

        for category in matched_categories:
            current = category_data[category]

            current["weeks_present"] = max(
                current["weeks_present"],
                weeks_present,
            )

            source_metric = SOURCE_METRIC_MAPPING.get(source_column)

            if source_metric:
                current[source_metric] = max(
                    current[source_metric],
                    weeks_present,
                )

            current["cross_structure_topic_occurrences"] += (
                weeks_present
            )
            current["matched_raw_topics"].add(topic)
            current["source_columns"].add(source_column)

    if reviews_df is not None and not reviews_df.empty:
        for _, row in reviews_df.iterrows():
            # ------------------------------------------------------
            # Preferir tradução inglesa quando disponível.
            # Fallback para o texto original se a tradução falhou.
            # ------------------------------------------------------
            liked_en = _get_text_value(
                row,
                ["en_review_text_liked"],
            )
            liked_original = _get_text_value(
                row,
                ["review_text_liked"],
            )
            positive_text = liked_en or liked_original

            disliked_en = _get_text_value(
                row,
                ["en_review_text_disliked"],
            )
            disliked_original = _get_text_value(
                row,
                ["review_text_disliked"],
            )
            negative_text = disliked_en or disliked_original

            title = _get_text_value(
                row,
                ["review_title"],
            )

            total_text = " ".join(
                part
                for part in [title, positive_text, negative_text]
                if part
            )

            total_categories = set(
                map_text_to_categories(
                    total_text,
                    taxonomy=taxonomy,
                )
            )

            positive_categories = set(
                map_text_to_categories(
                    positive_text,
                    taxonomy=taxonomy,
                )
            )

            negative_categories = set(
                map_text_to_categories(
                    negative_text,
                    taxonomy=taxonomy,
                )
            )

            for category in total_categories:
                category_data[category][
                    "review_mentions_total"
                ] += 1

            for category in positive_categories:
                category_data[category][
                    "positive_review_mentions"
                ] += 1

            for category in negative_categories:
                category_data[category][
                    "negative_review_mentions"
                ] += 1

    output = []

    for category, values in category_data.items():
        if (
            values["weeks_present"] == 0
            and values["review_mentions_total"] == 0
        ):
            continue

        positive_mentions = int(
            values["positive_review_mentions"]
        )
        negative_mentions = int(
            values["negative_review_mentions"]
        )

        output.append(
            {
                "category": category,
                "label": values["label"],
                "managerial_domain": values["managerial_domain"],
                "default_priority": values["default_priority"],

                "weeks_present": int(values["weeks_present"]),
                "experience_weeks": int(
                    values["experience_weeks"]
                ),
                "expectation_weeks": int(
                    values["expectation_weeks"]
                ),
                "violated_expectation_weeks": int(
                    values["violated_expectation_weeks"]
                ),
                "latent_expectation_weeks": int(
                    values["latent_expectation_weeks"]
                ),
                "emerging_negative_weeks": int(
                    values["emerging_negative_weeks"]
                ),

                "cross_structure_topic_occurrences": int(
                    values["cross_structure_topic_occurrences"]
                ),

                "review_mentions_total": int(
                    values["review_mentions_total"]
                ),
                "positive_review_mentions": positive_mentions,
                "negative_review_mentions": negative_mentions,
                "net_sentiment_mentions": (
                    positive_mentions - negative_mentions
                ),

                "matched_raw_topics": sorted(
                    values["matched_raw_topics"]
                ),
                "source_columns": sorted(
                    values["source_columns"]
                ),
            }
        )

    output.sort(
        key=lambda row: (
            -row["negative_review_mentions"],
            -row["violated_expectation_weeks"],
            -row["emerging_negative_weeks"],
            -row["expectation_weeks"],
            -row["weeks_present"],
            row["label"].lower(),
        )
    )

    return output[:top_n]