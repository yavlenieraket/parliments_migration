"""Filtering functions for migration discourse and foreign country mentions."""

import polars as pl

from src.config import EXCLUDE_FROM_FOREIGN, FRENCH_OVERSEAS, MIGRATION_TOPIC


def filter_migration(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Keep only rows from speeches tagged as immigration (CAP topic 9)."""
    return lf.filter(pl.col("debate_topic") == MIGRATION_TOPIC)


def filter_country_mentions(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Keep only LOC entities (countries, cities, geographic locations)."""
    return lf.filter(pl.col("entity_category") == "LOC")


def filter_foreign(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Exclude France itself, French cities/regions, and non-country geo terms.

    Overseas territories are kept but flagged separately (see add_geo_class).
    """
    return lf.filter(~pl.col("entity_content").is_in(EXCLUDE_FROM_FOREIGN))


def add_geo_class(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add a column distinguishing foreign countries from French overseas territories."""
    return lf.with_columns(
        pl.when(pl.col("entity_content").is_in(FRENCH_OVERSEAS))
        .then(pl.lit("french_overseas"))
        .otherwise(pl.lit("foreign"))
        .alias("geo_class")
    )


def build_context_window(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Concatenate previous/current/next sentences into a single context field.

    The separator '||' makes it easy to split back later if needed.
    """
    return lf.with_columns(
        pl.concat_str([
            pl.col("sentence_content_previous").fill_null(""),
            pl.lit(" || "),
            pl.col("sentence_content_current"),
            pl.lit(" || "),
            pl.col("sentence_content_next").fill_null(""),
        ]).alias("context_window")
    )


def build_migration_mentions(lf: pl.LazyFrame) -> pl.DataFrame:
    """Full pipeline: facts -> migration debates -> LOC entities ->
    foreign countries (with overseas flag) -> with context window.

    Returns an eagerly-collected DataFrame.
    """
    return (
        lf
        .pipe(filter_migration)
        .pipe(filter_country_mentions)
        .pipe(filter_foreign)
        .pipe(add_geo_class)
        .pipe(build_context_window)
        .select([
            "sentence_id",
            "speech_id",
            "session_date",
            "speaker_id",
            "speaker_ana",
            "entity_content",
            "geo_class",
            "context_window",
            "sentence_content_current",
            "sentence_sentiment_value",
            "sentence_sentiment_ana",
            "debate_topic",
            "country",
        ])
        .collect()
    )
