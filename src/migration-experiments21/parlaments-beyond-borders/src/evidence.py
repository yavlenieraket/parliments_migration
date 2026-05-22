"""Evidence tables and result notes for country-mention analysis."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def compact_context(text: str | None, max_chars: int = 360) -> str:
    """Return a compact one-line context excerpt."""
    # Explanation: Context exports must be readable in CSV, notebooks, and tooltips.
    if not isinstance(text, str):
        return ""
    clean = " ".join(text.replace("||", " ").split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def add_context_excerpt(df: pl.DataFrame, max_chars: int = 360) -> pl.DataFrame:
    """Add a short context excerpt column."""
    # Explanation: Keep full context_window for audit, but use excerpt for display.
    return df.with_columns(
        pl.col("context_window")
        .map_elements(lambda text: compact_context(text, max_chars=max_chars), return_dtype=pl.Utf8)
        .alias("context_excerpt")
    )


def country_mention_profile(df: pl.DataFrame, min_mentions: int = 10) -> pl.DataFrame:
    """Summarize how each mentioned country/entity appears in both studies."""
    # Explanation: This is the central table for "how other countries are mentioned".
    return (
        df
        .group_by(["entity_content", "target_iso3", "geo_class", "region_group", "weog_group"])
        .agg([
            pl.len().alias("n_mentions"),
            pl.col("source_year").min().alias("first_year"),
            pl.col("source_year").max().alias("last_year"),
            pl.col("concreteness_score").mean().round(3).alias("mean_concreteness"),
            pl.col("concreteness_band").mode().first().alias("dominant_concreteness_band"),
            pl.col("ref_type").mode().first().alias("dominant_ref_type"),
            pl.col("sentiment_readable").mode().first().alias("dominant_sentiment"),
            pl.col("migrant_cohort").mode().first().alias("dominant_cohort"),
            pl.col("policy_measure").mode().first().alias("dominant_policy_measure"),
            pl.col("policy_agency_type").mode().first().alias("dominant_policy_agency"),
            pl.col("narrative_frame").mode().first().alias("dominant_narrative_frame"),
            pl.col("narrative_polarity").mode().first().alias("dominant_narrative_polarity"),
            pl.col("migration_direction").mode().first().alias("dominant_migration_direction"),
            pl.col("concrete_marker_hits").drop_nulls().unique().str.join(", ").alias("concrete_markers_seen"),
            pl.col("abstract_marker_hits").drop_nulls().unique().str.join(", ").alias("abstract_markers_seen"),
        ])
        .filter(pl.col("n_mentions") >= min_mentions)
        .sort("n_mentions", descending=True)
    )


def yearly_country_profile(df: pl.DataFrame, min_mentions: int = 5) -> pl.DataFrame:
    """Summarize country/entity mentions by year."""
    # Explanation: This supports heatmaps and lets us see whether references shift over time.
    return (
        df
        .group_by(["source_year", "entity_content", "target_iso3", "weog_group", "region_group"])
        .agg([
            pl.len().alias("n_mentions"),
            pl.col("concreteness_score").mean().round(3).alias("mean_concreteness"),
            pl.col("ref_type").mode().first().alias("dominant_ref_type"),
            pl.col("migrant_cohort").mode().first().alias("dominant_cohort"),
            pl.col("policy_measure").mode().first().alias("dominant_policy_measure"),
            pl.col("policy_agency_type").mode().first().alias("dominant_policy_agency"),
            pl.col("narrative_frame").mode().first().alias("dominant_narrative_frame"),
            pl.col("migration_direction").mode().first().alias("dominant_migration_direction"),
        ])
        .filter(pl.col("n_mentions") >= min_mentions)
        .sort(["source_year", "n_mentions"], descending=[False, True])
    )


def country_context_examples(
    df: pl.DataFrame,
    countries: list[str] | None = None,
    examples_per_country: int = 4,
) -> pl.DataFrame:
    """Return concrete and abstract context examples for selected countries."""
    # Explanation: We take extremes so users can inspect what drives the score.
    working = add_context_excerpt(df)
    if countries:
        working = working.filter(pl.col("entity_content").is_in(countries))

    selected = []
    for country in working.get_column("entity_content").unique().to_list():
        sub = working.filter(pl.col("entity_content") == country)
        if sub.is_empty():
            continue
        abstract = (
            sub
            .sort(["concreteness_score", "source_year"])
            .head(examples_per_country)
            .with_columns(pl.lit("most_abstract").alias("example_type"))
        )
        concrete = (
            sub
            .sort(["concreteness_score", "source_year"], descending=[True, False])
            .head(examples_per_country)
            .with_columns(pl.lit("most_concrete").alias("example_type"))
        )
        selected.extend([abstract, concrete])

    if not selected:
        return pl.DataFrame()

    return (
        pl.concat(selected, how="vertical")
        .select([
            "entity_content",
            "source_year",
            "example_type",
            "concreteness_score",
            "concreteness_band",
            "ref_type",
            "sentiment_readable",
            "migrant_cohort",
            "migrant_cohort_marker",
            "policy_measure",
            "policy_measure_marker",
            "policy_agency_type",
            "policy_agency_marker",
            "narrative_frame",
            "narrative_frame_marker",
            "argument_scheme",
            "migration_direction",
            "flow_source_candidate",
            "flow_destination_candidate",
            "concrete_marker_hits",
            "abstract_marker_hits",
            "context_excerpt",
            "sentence_id",
        ])
        .sort(["entity_content", "example_type", "concreteness_score"])
    )


def cohort_policy_context_examples(
    df: pl.DataFrame,
    min_mentions: int = 10,
    examples_per_pair: int = 3,
) -> pl.DataFrame:
    """Return examples for frequent migrant-cohort x policy-measure pairs."""
    # Explanation: This documents how institutional categories are tied to policy measures.
    working = add_context_excerpt(df)
    frequent_pairs = (
        working
        .group_by(["migrant_cohort", "policy_measure"])
        .agg(pl.len().alias("n_mentions"))
        .filter(pl.col("n_mentions") >= min_mentions)
        .sort("n_mentions", descending=True)
    )

    selected = []
    for row in frequent_pairs.to_dicts():
        sub = (
            working
            .filter(
                (pl.col("migrant_cohort") == row["migrant_cohort"])
                & (pl.col("policy_measure") == row["policy_measure"])
            )
            .sort(["source_year", "entity_content"])
            .head(examples_per_pair)
            .with_columns(pl.lit(row["n_mentions"]).alias("pair_mentions"))
        )
        selected.append(sub)

    if not selected:
        return pl.DataFrame()

    return (
        pl.concat(selected, how="vertical")
        .select([
            "migrant_cohort",
            "policy_measure",
            "pair_mentions",
            "entity_content",
            "source_year",
            "ref_type",
            "sentiment_readable",
            "migrant_cohort_marker",
            "policy_measure_marker",
            "policy_agency_type",
            "policy_agency_marker",
            "narrative_frame",
            "argument_scheme",
            "migration_direction",
            "context_excerpt",
            "sentence_id",
        ])
        .sort(["pair_mentions", "migrant_cohort", "policy_measure"], descending=[True, False, False])
    )


def build_result_notes(df: pl.DataFrame, edges: pl.DataFrame) -> str:
    """Create a short markdown note describing received results."""
    # Explanation: This creates narrative notes next to the generated tables/figures.
    n_mentions = df.height
    year_min = df.get_column("source_year").min()
    year_max = df.get_column("source_year").max()
    top_entities = country_mention_profile(df, min_mentions=1).head(8)
    weog = (
        df
        .drop_nulls("concreteness_score")
        .filter(pl.col("weog_group").is_in(["weog", "non_weog"]))
        .group_by("weog_group")
        .agg([
            pl.len().alias("n"),
            pl.col("concreteness_score").mean().round(3).alias("mean"),
        ])
        .sort("weog_group")
    )
    top_edges = (
        edges
        .group_by("target_entity")
        .agg(pl.col("weight").sum().alias("weight"))
        .sort("weight", descending=True)
        .head(8)
    )

    lines = [
        f"# Result Notes: France {year_min}-{year_max}",
        "",
        f"- Retained migration mentions with country/entity references: **{n_mentions:,}**.",
        "- Main research question: when France mentions other countries in migration debates, are those references concrete events/places/instruments or abstract/speculative frames?",
        "- Second research question: when migrant cohorts are institutionally segmented, which countries are invoked as policy examples or warnings?",
        "",
        "## Concreteness by Region",
    ]
    for row in weog.to_dicts():
        lines.append(f"- `{row['weog_group']}`: {row['n']:,} mentions, mean concreteness {row['mean']}.")

    lines.extend(["", "## Most Mentioned Countries / Entities"])
    for row in top_entities.to_dicts():
        lines.append(
            f"- {row['entity_content']}: {row['n_mentions']:,} mentions, "
            f"mean concreteness {row['mean_concreteness']}, "
            f"dominant ref `{row['dominant_ref_type']}`, "
            f"dominant cohort `{row['dominant_cohort']}`, "
            f"dominant policy `{row['dominant_policy_measure']}`, "
            f"dominant agency `{row['dominant_policy_agency']}`, "
            f"dominant frame `{row['dominant_narrative_frame']}`."
        )

    lines.extend(["", "## Strongest Diffusion Targets"])
    for row in top_edges.to_dicts():
        lines.append(f"- {row['target_entity']}: {row['weight']:,} weighted edge mentions.")

    lines.extend([
        "",
        "## Reading Guidance",
        "- Treat the concreteness scores as pilot indicators unless a Brysbaert lexicon is supplied.",
        "- Use the context CSVs to validate whether country mentions are actual country references or residual NER noise.",
        "- For presentation, prioritize the interactive HTML files because tooltips show country/entity, year, cohort, policy measure, and counts.",
    ])
    return "\n".join(lines) + "\n"


def save_evidence_outputs(
    df: pl.DataFrame,
    edges: pl.DataFrame,
    processed_dir: Path,
    prefix: str = "FRA_2017_2022",
    top_n_context_countries: int = 12,
) -> dict[str, Path]:
    """Save country profiles, context examples, and markdown result notes."""
    # Explanation: These outputs are the audit trail behind the visualizations.
    processed_dir.mkdir(parents=True, exist_ok=True)
    profile_path = processed_dir / f"{prefix}_country_mention_profile.csv"
    yearly_path = processed_dir / f"{prefix}_country_year_profile.csv"
    contexts_path = processed_dir / f"{prefix}_country_context_examples.csv"
    cohort_contexts_path = processed_dir / f"{prefix}_cohort_policy_context_examples.csv"
    notes_path = processed_dir / f"{prefix}_result_notes.md"

    profile = country_mention_profile(df, min_mentions=1)
    profile.write_csv(profile_path)
    yearly_country_profile(df, min_mentions=1).write_csv(yearly_path)

    top_countries = profile.head(top_n_context_countries).get_column("entity_content").to_list()
    country_context_examples(df, countries=top_countries).write_csv(contexts_path)
    cohort_policy_context_examples(df).write_csv(cohort_contexts_path)
    notes_path.write_text(build_result_notes(df, edges), encoding="utf-8")

    return {
        "country_mention_profile": profile_path,
        "country_year_profile": yearly_path,
        "country_context_examples": contexts_path,
        "cohort_policy_context_examples": cohort_contexts_path,
        "result_notes": notes_path,
    }
