"""Classification of foreign country mentions along two axes:
- Reference type: policy vs situation vs mixed vs neutral
- Sentiment polarity: positive vs negative vs neutral
"""

from __future__ import annotations

import polars as pl


# === Lexical markers for the reference-type classifier ===

POLICY_MARKERS_FR = {
    "loi", "lois", "politique", "politiques", "modèle", "modèles",
    "approche", "système", "réglementation", "décret", "directive",
    "réforme", "réformes", "voté", "voter", "adopté", "adopter",
    "rejeté", "rejeter", "applique", "appliquer", "décide", "décider",
    "instaure", "instaurer", "accord", "accords", "traité", "traités",
    "convention", "conventions", "pacte", "pactes",
    "gouvernement", "gouvernementale", "ministre", "parlement",
    "quota", "quotas", "régularisation", "expulsion", "expulsions",
    "naturalisation", "asile",
}

SITUATION_MARKERS_FR = {
    "crise", "crises", "flux", "vague", "vagues", "nombre", "nombres",
    "arrivées", "arrivants", "morts", "noyades", "naufrage", "naufrages",
    "camps", "camp", "situation", "événements", "guerre", "conflit",
    "afflux", "exode", "drame", "tragédie", "tragédies",
    "frontière", "frontières", "migrants", "réfugiés", "demandeurs",
    "passeurs", "trafic", "Méditerranée", "Manche",
}


def _count_marker_hits(text: str, markers: set[str]) -> int:
    """Count how many distinct markers appear in the text (case-insensitive)."""
    if not isinstance(text, str) or not text:
        return 0
    lower = text.lower()
    return sum(1 for m in markers if m in lower)


def classify_reference_type(window: str) -> str:
    """Classify a context window as policy / situation / mixed / neutral.

    Heuristic for the pilot. Replace with a trained classifier
    once you have a manually-coded training sample.
    """
    if not isinstance(window, str) or not window.strip():
        return "unknown"

    policy_hits = _count_marker_hits(window, POLICY_MARKERS_FR)
    situation_hits = _count_marker_hits(window, SITUATION_MARKERS_FR)

    if policy_hits == 0 and situation_hits == 0:
        return "neutral_reference"
    if policy_hits >= 2 and situation_hits >= 2:
        return "mixed"
    if policy_hits > situation_hits:
        return "policy"
    if situation_hits > policy_hits:
        return "situation"
    return "mixed"


def bucket_sentiment(value: float | None, label: str | None) -> str:
    """Collapse continuous sentiment + categorical label into 3 buckets.

    ParlaMint provides both a numeric value and a categorical 'ana' tag
    (e.g. senti:positive, senti:negative, senti:neutral, senti:mixpos, senti:mixneg).
    We rely primarily on the value with the label as a tiebreaker.
    """
    if value is None:
        if label is None:
            return "neutral"
        if "positive" in label or label == "senti:pos":
            return "positive"
        if "negative" in label or label == "senti:neg":
            return "negative"
        return "neutral"

    if value >= 1.0:
        return "positive"
    if value <= -1.0:
        return "negative"
    return "neutral"


def apply_typology(df: pl.DataFrame) -> pl.DataFrame:
    """Apply both axes (reference_type, sentiment_bucket) to the dataframe."""
    return df.with_columns([
        pl.col("context_window")
        .map_elements(classify_reference_type, return_dtype=pl.Utf8)
        .alias("ref_type"),
        pl.struct(["sentence_sentiment_value", "sentence_sentiment_ana"])
        .map_elements(
            lambda s: bucket_sentiment(s["sentence_sentiment_value"],
                                       s["sentence_sentiment_ana"]),
            return_dtype=pl.Utf8,
        )
        .alias("sentiment_bucket"),
    ])


def build_2x3_matrix(df: pl.DataFrame) -> pl.DataFrame:
    """Build the final 2x3 cross-tabulation: ref_type x sentiment_bucket."""
    return (
        df
        .group_by(["ref_type", "sentiment_bucket"])
        .agg(pl.len().alias("n"))
        .pivot(
            values="n",
            index="ref_type",
            on="sentiment_bucket",
            aggregate_function="sum",
        )
        .fill_null(0)
    )


def matrix_by_country(df: pl.DataFrame, min_mentions: int = 5) -> pl.DataFrame:
    """Per-country breakdown of the 2x3 matrix. Useful for top mentioned countries."""
    return (
        df
        .group_by(["entity_content", "ref_type", "sentiment_bucket"])
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= min_mentions)
        .sort(["entity_content", "ref_type", "sentiment_bucket"])
    )
