"""Sanity-check helpers for inspecting intermediate dataframes."""

import polars as pl


def show_topic_distribution(lf: pl.LazyFrame, top_n: int = 20) -> pl.DataFrame:
    """Distribution of entity mentions across CAP debate topics."""
    return (
        lf
        .group_by("debate_topic")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(top_n)
        .collect()
    )


def show_top_countries(
    df: pl.DataFrame,
    top_n: int = 30,
    min_mentions: int = 5,
) -> pl.DataFrame:
    """Top mentioned countries with aggregate sentiment statistics."""
    return (
        df
        .group_by("entity_content")
        .agg([
            pl.len().alias("n_mentions"),
            pl.col("sentence_sentiment_value").mean().round(3).alias("avg_sentiment"),
            pl.col("sentence_sentiment_value").std().round(3).alias("std_sentiment"),
            pl.col("sentence_sentiment_value").min().alias("min_sentiment"),
            pl.col("sentence_sentiment_value").max().alias("max_sentiment"),
        ])
        .filter(pl.col("n_mentions") >= min_mentions)
        .sort("n_mentions", descending=True)
        .head(top_n)
    )


def show_sample_contexts(
    df: pl.DataFrame,
    country: str,
    n_samples: int = 5,
) -> pl.DataFrame:
    """Random sample of context windows for a specific country.

    Used for manual validation of the heuristic classifier.
    """
    return (
        df
        .filter(pl.col("entity_content") == country)
        .select([
            "session_date",
            "entity_content",
            "ref_type",
            "sentiment_bucket",
            "sentence_sentiment_value",
            "context_window",
        ])
        .sample(n=min(n_samples, df.height), seed=42)
    )
