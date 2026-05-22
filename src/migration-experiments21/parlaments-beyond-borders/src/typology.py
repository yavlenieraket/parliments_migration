"""Classify mentions: reference type x full 6-level ParlaMint sentiment.

All lexical markers are English because this pilot uses ParlaMint-en.ana.
"""

from __future__ import annotations

import polars as pl


# === Policy reference markers ===
# Explanation: These words signal that the mention is about what a country or
# institution does legally, administratively, or politically.
POLICY_MARKERS_EN = {
    "law", "laws", "legislation", "legislative",
    "policy", "policies", "regulation", "regulations",
    "decree", "directive", "directives", "reform", "reforms",
    "rule", "rules", "act", "bill", "framework",
    "voted", "vote", "adopted", "adopt", "approved", "approve",
    "rejected", "reject", "passed", "ratified", "ratify",
    "decided", "decide", "implemented", "implement", "applies", "apply",
    "introduced", "introduce", "enacted", "enact",
    "government", "governments", "minister", "ministry", "ministries",
    "parliament", "parliamentary", "chancellor", "president", "presidency",
    "administration", "cabinet", "coalition",
    "quota", "quotas", "regularization", "regularisation",
    "deportation", "deportations", "naturalization", "naturalisation",
    "asylum", "visa", "visas", "residence permit", "residency",
    "amnesty", "expulsion", "expulsions", "readmission",
    "agreement", "agreements", "treaty", "treaties",
    "convention", "conventions", "pact", "pacts", "accord", "accords",
    "protocol", "protocols",
    "model", "models", "approach", "approaches", "system", "systems",
    "example", "case", "experience",
}

# === Situation reference markers ===
# Explanation: These words signal that the mention is about events, conditions,
# routes, camps, humanitarian situations, or conflict around a place.
SITUATION_MARKERS_EN = {
    "crisis", "crises", "emergency", "emergencies",
    "wave", "waves", "flow", "flows", "influx", "exodus",
    "surge", "surges", "tide",
    "number", "numbers", "arrivals", "arriving",
    "departures", "departing", "newcomers",
    "dead", "deaths", "drowned", "drownings", "drowning",
    "shipwreck", "shipwrecks", "tragedy", "tragedies",
    "victims", "victim", "killed", "lost their lives",
    "camp", "camps", "border", "borders", "frontier", "frontiers",
    "reception centre", "reception center", "detention",
    "checkpoint", "crossing", "crossings",
    "migrants", "refugees", "asylum seekers", "asylum-seekers",
    "displaced", "displacement", "smugglers", "trafficking",
    "unaccompanied minors",
    "war", "conflict", "conflicts", "civil war",
    "famine", "drought", "persecution",
    "situation", "events", "happenings",
    "the Mediterranean", "Mediterranean", "Channel", "the Channel",
    "Lampedusa", "Lesbos", "Moria",
}

# === Sentiment: 6-level scale from ParlaMint ===
# Explanation: We keep the original ParlaMint categorical scale instead of
# collapsing it to positive/negative/neutral. This preserves intensity.
SENTIMENT_LEVELS = [
    "senti:negneg",  # strongly negative
    "senti:mixneg",  # mixed leaning negative
    "senti:neuneg",  # neutral with slight negative tilt
    "senti:neupos",  # neutral with slight positive tilt
    "senti:mixpos",  # mixed leaning positive
    "senti:pospos",  # strongly positive
]

# Explanation: Human-readable labels are used in tables and chart legends.
SENTIMENT_LABELS = {
    "senti:negneg": "strong neg",
    "senti:mixneg": "mixed neg",
    "senti:neuneg": "neutral neg",
    "senti:neupos": "neutral pos",
    "senti:mixpos": "mixed pos",
    "senti:pospos": "strong pos",
}

# Explanation: This optional 3-way grouping is useful for quick explanation,
# but charts now use the richer 6-level sentiment_level column.
SENTIMENT_POLARITY = {
    "senti:negneg": "negative",
    "senti:mixneg": "negative",
    "senti:neuneg": "neutral",
    "senti:neupos": "neutral",
    "senti:mixpos": "positive",
    "senti:pospos": "positive",
}


def _count_marker_hits(text: str, markers: set[str]) -> int:
    """Count distinct markers appearing in the text, case-insensitively."""
    # Explanation: Empty or non-text values cannot contribute marker evidence.
    if not isinstance(text, str) or not text:
        return 0
    # Explanation: Lowercase both sides so "Policy" and "policy" count the same.
    lower = text.lower()
    return sum(1 for marker in markers if marker.lower() in lower)


def classify_reference_type(window: str) -> str:
    """Classify a context window as policy / situation / mixed / neutral."""
    # Explanation: Missing context cannot be classified meaningfully.
    if not isinstance(window, str) or not window.strip():
        return "unknown"

    # Explanation: Count separate evidence for institutional policy and situation context.
    policy_hits = _count_marker_hits(window, POLICY_MARKERS_EN)
    situation_hits = _count_marker_hits(window, SITUATION_MARKERS_EN)

    # Explanation: No marker evidence means the country is mentioned without our target frame.
    if policy_hits == 0 and situation_hits == 0:
        return "neutral_reference"
    # Explanation: Strong evidence for both axes means the mention is analytically mixed.
    if policy_hits >= 2 and situation_hits >= 2:
        return "mixed"
    if policy_hits > situation_hits:
        return "policy"
    if situation_hits > policy_hits:
        return "situation"
    return "mixed"


def normalize_sentiment_label(label: str | None) -> str:
    """Return the original ParlaMint 6-level sentiment label when recognized."""
    # Explanation: Unknown or missing labels are kept explicit so colleagues can audit them.
    if label is None or not isinstance(label, str):
        return "unknown"
    if label in SENTIMENT_LEVELS:
        return label
    return "unknown"


def add_sentiment_label_readable(df: pl.DataFrame) -> pl.DataFrame:
    """Add readable sentiment labels and broad polarity labels."""
    # Explanation: sentiment_level is the analytical value; readable/polarity help presentation.
    return df.with_columns([
        pl.col("sentiment_level")
        .replace(SENTIMENT_LABELS, default="unknown")
        .alias("sentiment_readable"),
        pl.col("sentiment_level")
        .replace(SENTIMENT_POLARITY, default="unknown")
        .alias("sentiment_polarity"),
    ])


def apply_typology(df: pl.DataFrame) -> pl.DataFrame:
    """Annotate each mention with reference type and 6-level sentiment."""
    # Explanation: Reference type is our heuristic; sentiment comes directly from ParlaMint.
    return (
        df
        .with_columns([
            pl.col("context_window")
            .map_elements(classify_reference_type, return_dtype=pl.Utf8)
            .alias("ref_type"),
            pl.col("sentence_sentiment_ana")
            .map_elements(normalize_sentiment_label, return_dtype=pl.Utf8)
            .alias("sentiment_level"),
        ])
        .pipe(add_sentiment_label_readable)
    )


def build_matrix(df: pl.DataFrame) -> pl.DataFrame:
    """Build the headline matrix: reference type x 6-level sentiment."""
    # Explanation: Counts are pivoted so rows are reference types and columns are sentiment levels.
    matrix = (
        df
        .group_by(["ref_type", "sentiment_level"])
        .agg(pl.len().alias("n"))
        .pivot(
            values="n",
            index="ref_type",
            on="sentiment_level",
            aggregate_function="sum",
        )
        .fill_null(0)
    )
    # Explanation: Enforce the negative-to-positive column order when columns exist.
    ordered_cols = ["ref_type"] + [col for col in SENTIMENT_LEVELS if col in matrix.columns]
    return matrix.select(ordered_cols)


def build_2x3_matrix(df: pl.DataFrame) -> pl.DataFrame:
    """Backward-compatible alias for older notebook cells.

    Despite the legacy name, this now returns the 4 x 6 matrix.
    """
    # Explanation: Keeping this wrapper prevents old notebooks from failing abruptly.
    return build_matrix(df)


def matrix_by_country(df: pl.DataFrame, min_mentions: int = 5) -> pl.DataFrame:
    """Per-country breakdown of reference type x 6-level sentiment."""
    # Explanation: This table supports country-level comparison after filtering rare cells.
    return (
        df
        .group_by(["entity_content", "ref_type", "sentiment_level"])
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= min_mentions)
        .sort(["entity_content", "ref_type", "sentiment_level"])
    )


def sentiment_distribution_per_country(
    df: pl.DataFrame,
    countries: list[str],
) -> pl.DataFrame:
    """Return one row per country and one column per 6-level sentiment value."""
    # Explanation: This wide table is useful for heatmaps and manual checking.
    grouped = (
        df
        .filter(pl.col("entity_content").is_in(countries))
        .group_by(["entity_content", "sentiment_level"])
        .agg(pl.len().alias("n"))
    )
    pivot = (
        grouped
        .pivot(
            values="n",
            index="entity_content",
            on="sentiment_level",
            aggregate_function="sum",
        )
        .fill_null(0)
    )
    ordered_cols = ["entity_content"] + [col for col in SENTIMENT_LEVELS if col in pivot.columns]
    return pivot.select(ordered_cols)
