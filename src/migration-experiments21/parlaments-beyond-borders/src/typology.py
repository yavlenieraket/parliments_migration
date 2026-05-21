"""Classify mentions: reference type (policy/situation) x sentiment bucket.

All markers are English because we work on ParlaMint-en.ana.
"""

from __future__ import annotations

import polars as pl


# === Policy reference markers ===
# Signals that the mention is about what a country does legislatively,
# administratively, or institutionally.
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
# Signals that the mention is about what happens to or at a country:
# events, conditions, humanitarian situations.
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


def _count_marker_hits(text: str, markers: set[str]) -> int:
    """Count distinct markers appearing in the text (case-insensitive)."""
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


def bucket_sentiment(value: float | None, label: str | None) -> str:
    """Collapse numeric sentiment + categorical label into 3 buckets."""
    # Explanation: Prefer the numeric sentiment value; fall back to the label when missing.
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
    """Apply reference type and sentiment bucket to each mention."""
    # Explanation: Add one column for the reference frame and one for sentiment.
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
    # Explanation: Counts are pivoted so rows are reference types and columns are sentiment.
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
    """Per-country breakdown of the 2x3 matrix."""
    # Explanation: This table supports country-level comparison after filtering rare cells.
    return (
        df
        .group_by(["entity_content", "ref_type", "sentiment_bucket"])
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= min_mentions)
        .sort(["entity_content", "ref_type", "sentiment_bucket"])
    )
