"""Internal vs external migration-flow direction labels."""

from __future__ import annotations

import re

import polars as pl


SOURCE_PATTERNS = [
    re.compile(r"\bfrom\s+([A-Z][A-Za-zÀ-ÿ' -]{2,40})"),
    re.compile(r"\bleaving\s+([A-Z][A-Za-zÀ-ÿ' -]{2,40})"),
]

DESTINATION_PATTERNS = [
    re.compile(r"\bto\s+([A-Z][A-Za-zÀ-ÿ' -]{2,40})"),
    re.compile(r"\binto\s+([A-Z][A-Za-zÀ-ÿ' -]{2,40})"),
    re.compile(r"\btowards\s+([A-Z][A-Za-zÀ-ÿ' -]{2,40})"),
    re.compile(r"\bin\s+([A-Z][A-Za-zÀ-ÿ' -]{2,40})"),
]

FRANCE_TERMS = {"France", "French Republic", "Republic", "our country", "the Republic"}


def _extract_with_patterns(text: str | None, patterns: list[re.Pattern]) -> str:
    """Extract first source/destination candidate from surface patterns."""
    if not isinstance(text, str):
        return ""
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(1).strip(" .,;:")
    return ""


def classify_direction(text: str | None, speaker_country: str = "France") -> str:
    """Classify migration process direction relative to the speaker's country."""
    # Explanation: This is a rule-based SRL substitute until a full SRL model is added.
    if not isinstance(text, str):
        return "ambiguous"
    lower = text.lower()
    domestic_terms = {speaker_country.lower(), "france", "our country", "french territory"}
    inbound_markers = ["come to", "arrive in", "arrival in", "into france", "to france", "in france"]
    outbound_markers = ["from france", "leave france", "leaving france"]
    if any(marker in lower for marker in inbound_markers) or any(term in lower for term in domestic_terms):
        if any(word in lower for word in ["arrive", "arrival", "come", "enter", "into", "asylum in"]):
            return "inbound_internal"
    if any(marker in lower for marker in outbound_markers):
        return "outbound_from_domestic"
    if any(word in lower for word in ["from", "to", "into", "towards", "between"]):
        return "external_transnational"
    return "ambiguous"


def add_directional_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Add source/destination candidates and internal/external direction label."""
    # Explanation: flow_source_candidate and flow_destination_candidate are audit fields.
    return df.with_columns([
        pl.col("context_window")
        .map_elements(lambda text: _extract_with_patterns(text, SOURCE_PATTERNS), return_dtype=pl.Utf8)
        .alias("flow_source_candidate"),
        pl.col("context_window")
        .map_elements(lambda text: _extract_with_patterns(text, DESTINATION_PATTERNS), return_dtype=pl.Utf8)
        .alias("flow_destination_candidate"),
        pl.col("context_window")
        .map_elements(classify_direction, return_dtype=pl.Utf8)
        .alias("migration_direction"),
        pl.lit("surface_srl_rules").alias("direction_method"),
    ])
