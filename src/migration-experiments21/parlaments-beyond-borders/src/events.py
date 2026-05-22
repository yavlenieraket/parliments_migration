"""High-concreteness event extraction for map and timeline visualizations."""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl
from country_named_entity_recognition import find_countries


PROPER_NOUN_RE = re.compile(r"\b[A-Z][A-Za-zÀ-ÿ'’-]{2,}(?:\s+[A-Z][A-Za-zÀ-ÿ'’-]{2,}){0,3}\b")


def country_mentions_in_text(text: str | None) -> list[dict[str, str]]:
    """Extract country mentions with ISO3 using descending-length country matching."""
    # Explanation: country_named_entity_recognition avoids Niger/Nigeria-style partial matches.
    if not isinstance(text, str):
        return []
    mentions = []
    for country, match in find_countries(text, is_ignore_case=False):
        mentions.append({
            "country_name": country.name,
            "country_iso3": country.alpha_3,
            "surface": match.group(0),
        })
    return mentions


def proper_nouns(text: str | None, limit: int = 8) -> str:
    """Extract proper-noun-like event anchors from text."""
    # Explanation: These labels identify visible facts/events in timeline tooltips.
    if not isinstance(text, str):
        return ""
    seen = []
    for match in PROPER_NOUN_RE.findall(text.replace("||", " ")):
        cleaned = match.strip()
        if cleaned not in seen:
            seen.append(cleaned)
        if len(seen) >= limit:
            break
    return ", ".join(seen)


def build_high_concreteness_events(
    df: pl.DataFrame,
    min_score: float = 3.3,
) -> pl.DataFrame:
    """Return high-concreteness snippets with dates, countries, and proper nouns."""
    # Explanation: With the fallback scorer, 3.3+ means concrete_leaning.
    events = (
        df
        .filter(pl.col("concreteness_score") >= min_score)
        .with_columns([
            pl.col("context_window")
            .map_elements(proper_nouns, return_dtype=pl.Utf8)
            .alias("proper_noun_anchors"),
            pl.col("context_window")
            .map_elements(
                lambda text: ", ".join(
                    item["country_name"] for item in country_mentions_in_text(text)
                ),
                return_dtype=pl.Utf8,
            )
            .alias("countries_detected_in_context"),
            pl.col("context_window")
            .map_elements(
                lambda text: ", ".join(
                    item["country_iso3"] for item in country_mentions_in_text(text)
                ),
                return_dtype=pl.Utf8,
            )
            .alias("country_iso3_detected_in_context"),
        ])
        .select([
            "source_year",
            "session_date",
            "entity_content",
            "target_iso3",
            "weog_group",
            "concreteness_score",
            "concreteness_band",
            "proper_noun_anchors",
            "countries_detected_in_context",
            "country_iso3_detected_in_context",
            "narrative_frame",
            "policy_agency_type",
            "migrant_cohort",
            "policy_measure",
            "migration_direction",
            "context_window",
            "sentence_id",
        ])
        .sort(["concreteness_score", "session_date"], descending=[True, False])
    )
    return events


def visible_country_summary(events: pl.DataFrame) -> pl.DataFrame:
    """Summarize high-concreteness visibility by target country/entity."""
    # Explanation: This table powers choropleth/map-like views.
    return (
        events
        .group_by(["entity_content", "target_iso3", "weog_group"])
        .agg([
            pl.len().alias("visible_event_mentions"),
            pl.col("concreteness_score").mean().round(3).alias("mean_concreteness"),
            pl.col("proper_noun_anchors").drop_nulls().unique().str.join(", ").alias("proper_nouns_seen"),
            pl.col("narrative_frame").mode().first().alias("dominant_frame"),
            pl.col("policy_agency_type").mode().first().alias("dominant_agency"),
        ])
        .sort("visible_event_mentions", descending=True)
    )


def save_event_outputs(
    df: pl.DataFrame,
    processed_dir: Path,
    prefix: str = "FRA_2017_2022",
    min_score: float = 3.3,
) -> dict[str, Path]:
    """Save high-concreteness event table and country visibility summary."""
    # Explanation: These outputs are the data behind the fact timeline and map.
    processed_dir.mkdir(parents=True, exist_ok=True)
    events = build_high_concreteness_events(df, min_score=min_score)
    events_path = processed_dir / f"{prefix}_high_concreteness_events.csv"
    summary_path = processed_dir / f"{prefix}_visible_country_summary.csv"
    events.write_csv(events_path)
    visible_country_summary(events).write_csv(summary_path)
    return {
        "high_concreteness_events": events_path,
        "visible_country_summary": summary_path,
    }
