"""Vega-Altair visualization helpers for the France 2018 migration pilot."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import polars as pl

from src.typology import SENTIMENT_LABELS, SENTIMENT_LEVELS, build_matrix


# Explanation: Disable Altair's row limit because the pilot tables can exceed 5,000 rows.
alt.data_transformers.disable_max_rows()

# Explanation: Keep the 6-level sentiment colors stable across every figure.
SENTIMENT_COLORS = {
    "senti:negneg": "#7f1d1d",
    "senti:mixneg": "#c2410c",
    "senti:neuneg": "#9ca3af",
    "senti:neupos": "#a3a3a3",
    "senti:mixpos": "#15803d",
    "senti:pospos": "#14532d",
}

# Explanation: Reference-type colors separate policy talk from situation/context talk.
REF_TYPE_COLORS = {
    "policy": "#31688e",
    "situation": "#de8f05",
    "mixed": "#6a3d9a",
    "neutral_reference": "#9a9a9a",
    "unknown": "#d0d0d0",
}

REGION_GROUP_COLORS = {
    "european_country": "#4c78a8",
    "non_european_country": "#72b7b2",
    "european_union": "#b279a2",
    "french_overseas": "#f58518",
}

REGION_GROUP_LABELS = {
    "european_country": "European country",
    "non_european_country": "Non-European country/case",
    "european_union": "European Union",
    "french_overseas": "French overseas territory",
}

WEOG_GROUP_LABELS = {
    "weog": "WEOG / Western & others",
    "non_weog": "Non-WEOG",
    "european_union": "European Union",
    "french_overseas": "French overseas territory",
    "unknown": "Unknown",
}

# Explanation: Fixed category orders make comparisons stable across notebook reruns.
SENTIMENT_READABLE_ORDER = [SENTIMENT_LABELS[level] for level in SENTIMENT_LEVELS]
REF_TYPE_ORDER = ["policy", "situation", "mixed", "neutral_reference", "unknown"]
REGION_GROUP_ORDER = [
    "european_country",
    "non_european_country",
    "european_union",
    "french_overseas",
]


def ensure_figures_dir(processed_dir: Path) -> Path:
    """Create and return the directory where Altair figures are saved."""
    # Explanation: Altair outputs are kept separate from older matplotlib PNGs.
    figures_dir = processed_dir / "figures_altair"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _sentiment_scale() -> alt.Scale:
    """Return the shared sentiment color scale."""
    # Explanation: The domain is the readable label order from strongest negative to positive.
    return alt.Scale(
        domain=SENTIMENT_READABLE_ORDER,
        range=[SENTIMENT_COLORS[level] for level in SENTIMENT_LEVELS],
    )


def _ref_type_scale() -> alt.Scale:
    """Return the shared reference-type color scale."""
    return alt.Scale(
        domain=REF_TYPE_ORDER,
        range=[REF_TYPE_COLORS[level] for level in REF_TYPE_ORDER],
    )


def _region_group_scale() -> alt.Scale:
    """Return the shared region-group color scale."""
    return alt.Scale(
        domain=[REGION_GROUP_LABELS[group] for group in REGION_GROUP_ORDER],
        range=[REGION_GROUP_COLORS[group] for group in REGION_GROUP_ORDER],
    )


def _save_chart(chart: alt.Chart, output_path: Path) -> Path:
    """Save a chart as HTML, Vega-Lite JSON, and PNG when possible."""
    # Explanation: HTML is the safest interactive format; PNG is convenient for slides.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = output_path.with_suffix("")
    html_path = base.with_suffix(".html")
    json_path = base.with_suffix(".vl.json")
    png_path = base.with_suffix(".png")

    chart.save(html_path)
    chart.save(json_path)
    try:
        chart.save(png_path)
        return png_path
    except Exception as exc:
        # Explanation: PNG export requires vl-convert-python; keep a small note if it fails.
        note_path = base.with_suffix(".png_export_note.txt")
        note_path.write_text(
            f"PNG export failed. Open the interactive HTML instead.\n{exc}\n",
            encoding="utf-8",
        )
        return html_path


def top_entities(df: pl.DataFrame, top_n: int = 15) -> list[str]:
    """Return the most frequently mentioned entities."""
    # Explanation: Visualizations use the same top-entity universe for readability.
    return (
        df
        .group_by("entity_content")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(top_n)
        .get_column("entity_content")
        .to_list()
    )


def entity_display_labels(df: pl.DataFrame, entities: list[str]) -> dict[str, str]:
    """Return labels that mark French overseas territories and the EU explicitly."""
    # Explanation: Overseas territories are politically French, not foreign states.
    geo_lookup = {
        row["entity_content"]: row["geo_class"]
        for row in (
            df
            .filter(pl.col("entity_content").is_in(entities))
            .select(["entity_content", "geo_class", "region_group"])
            .unique()
            .to_dicts()
        )
    }
    return {
        entity: (
            f"{entity} (French overseas)"
            if geo_lookup.get(entity) == "french_overseas"
            else f"{entity} (EU)"
            if entity == "European Union"
            else entity
        )
        for entity in entities
    }


def entity_distribution_table(df: pl.DataFrame, min_mentions: int = 1) -> pl.DataFrame:
    """Return the complete distribution of mentioned entities."""
    # Explanation: This table is the source for distribution charts and CSV export.
    return (
        df
        .group_by(["entity_content", "geo_class", "region_group"])
        .agg([
            pl.len().alias("n_mentions"),
            (pl.len() / df.height * 100).round(2).alias("share_percent"),
        ])
        .filter(pl.col("n_mentions") >= min_mentions)
        .sort("n_mentions", descending=True)
    )


def save_entity_distribution_table(df: pl.DataFrame, output_path: Path) -> Path:
    """Save the full mentioned-entity distribution as CSV."""
    # Explanation: CSV is the easiest format for colleagues to audit every entity.
    entity_distribution_table(df).write_csv(output_path)
    return output_path


def save_significant_entity_distribution_table(
    df: pl.DataFrame,
    output_path: Path,
    min_mentions: int = 26,
) -> Path:
    """Save distribution for entities that meet the minimum mention threshold."""
    # Explanation: min_mentions=26 means the displayed chart includes entities with >25 mentions.
    entity_distribution_table(df, min_mentions=min_mentions).write_csv(output_path)
    return output_path


def _add_display_columns(
    pdf: pd.DataFrame,
    df: pl.DataFrame,
    entities: list[str] | None = None,
) -> pd.DataFrame:
    """Add display labels for entities, region groups, and sentiment levels."""
    # Explanation: The data keeps analytical codes, while display columns keep charts readable.
    pdf = pdf.copy()
    if "entity_content" in pdf.columns:
        chosen_entities = entities or pdf["entity_content"].drop_duplicates().tolist()
        labels = entity_display_labels(df, chosen_entities)
        pdf["display_entity"] = pdf["entity_content"].map(labels)
    if "region_group" in pdf.columns:
        pdf["region_group_readable"] = pdf["region_group"].map(REGION_GROUP_LABELS)
    if "sentiment_level" in pdf.columns and "sentiment_readable" not in pdf.columns:
        pdf["sentiment_readable"] = pdf["sentiment_level"].map(SENTIMENT_LABELS).fillna("unknown")
    return pdf


def _complete_count_long(
    df: pl.DataFrame,
    category_column: str,
    categories: list[str],
    entities: list[str],
) -> pd.DataFrame:
    """Build a complete entity x category count table with zero-filled cells."""
    # Explanation: Polars returns only observed combinations; charts need explicit zeroes.
    counts = (
        df
        .filter(pl.col("entity_content").is_in(entities))
        .group_by(["entity_content", category_column])
        .agg(pl.len().alias("n"))
        .to_pandas()
    )
    index = pd.MultiIndex.from_product(
        [entities, categories],
        names=["entity_content", category_column],
    )
    counts = (
        counts
        .set_index(["entity_content", category_column])
        .reindex(index, fill_value=0)
        .reset_index()
    )
    return counts


def plot_entity_distribution(
    df: pl.DataFrame,
    output_path: Path,
    top_n: int | None = 40,
    min_mentions: int = 1,
) -> Path:
    """Save an Altair bar chart of total migration mentions by entity."""
    # Explanation: min_mentions removes low-frequency entities from the displayed chart.
    distribution = entity_distribution_table(df, min_mentions=min_mentions)
    if top_n is not None:
        distribution = distribution.head(top_n)

    entities = distribution.get_column("entity_content").to_list()
    plot_df = _add_display_columns(distribution.to_pandas(), df, entities)
    display_order = plot_df["display_entity"].tolist()

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("n_mentions:Q", title="Number of mentions"),
            y=alt.Y("display_entity:N", title="Mentioned entity", sort=display_order),
            color=alt.Color(
                "region_group_readable:N",
                title="Analytical group",
                scale=_region_group_scale(),
            ),
            tooltip=[
                alt.Tooltip("display_entity:N", title="Entity"),
                alt.Tooltip("region_group_readable:N", title="Group"),
                alt.Tooltip("n_mentions:Q", title="Mentions"),
                alt.Tooltip("share_percent:Q", title="Share (%)"),
            ],
        )
        .properties(
            title="Distribution of mentioned entities in France 2018 migration debates",
            width=780,
            height=max(280, 22 * len(plot_df)),
        )
    )
    return _save_chart(chart, output_path)


def plot_region_group_distribution(df: pl.DataFrame, output_path: Path) -> Path:
    """Save an Altair chart of European / non-European / EU / overseas mentions."""
    # Explanation: The EU and French overseas territories are not foreign states, so they stay separate.
    counts = (
        df
        .group_by("region_group")
        .agg(pl.len().alias("n_mentions"))
        .to_pandas()
        .set_index("region_group")
        .reindex(REGION_GROUP_ORDER, fill_value=0)
        .reset_index()
    )
    counts["region_group_readable"] = counts["region_group"].map(REGION_GROUP_LABELS)

    chart = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("region_group_readable:N", title="Analytical group", sort=None),
            y=alt.Y("n_mentions:Q", title="Number of mentions"),
            color=alt.Color(
                "region_group_readable:N",
                title="Analytical group",
                scale=_region_group_scale(),
            ),
            tooltip=[
                alt.Tooltip("region_group_readable:N", title="Group"),
                alt.Tooltip("n_mentions:Q", title="Mentions"),
            ],
        )
        .properties(title="Mentions by European / non-European grouping", width=720, height=360)
    )
    return _save_chart(chart, output_path)


def plot_region_group_sentiment(df: pl.DataFrame, output_path: Path) -> Path:
    """Save an Altair stacked bar chart of 6-level sentiment by region group."""
    # Explanation: This shows whether sentiment differs by broad geography group.
    counts = (
        df
        .group_by(["region_group", "sentiment_level"])
        .agg(pl.len().alias("n"))
        .to_pandas()
    )
    index = pd.MultiIndex.from_product(
        [REGION_GROUP_ORDER, SENTIMENT_LEVELS],
        names=["region_group", "sentiment_level"],
    )
    plot_df = counts.set_index(["region_group", "sentiment_level"]).reindex(index, fill_value=0).reset_index()
    plot_df["region_group_readable"] = plot_df["region_group"].map(REGION_GROUP_LABELS)
    plot_df["sentiment_readable"] = plot_df["sentiment_level"].map(SENTIMENT_LABELS)
    plot_df["sentiment_order"] = plot_df["sentiment_level"].map(
        {level: idx for idx, level in enumerate(SENTIMENT_LEVELS)}
    )

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("n:Q", title="Number of mentions", stack="zero"),
            y=alt.Y("region_group_readable:N", title="Analytical group", sort=[
                REGION_GROUP_LABELS[group] for group in REGION_GROUP_ORDER
            ]),
            color=alt.Color(
                "sentiment_readable:N",
                title="Sentiment level",
                scale=_sentiment_scale(),
                sort=SENTIMENT_READABLE_ORDER,
            ),
            order=alt.Order("sentiment_order:Q"),
            tooltip=[
                alt.Tooltip("region_group_readable:N", title="Group"),
                alt.Tooltip("sentiment_readable:N", title="Sentiment"),
                alt.Tooltip("n:Q", title="Mentions"),
            ],
        )
        .properties(title="6-level sentiment by European / non-European grouping", width=760, height=320)
    )
    return _save_chart(chart, output_path)


def plot_country_sentiment_mentions(
    df: pl.DataFrame,
    output_path: Path,
    top_n: int = 15,
) -> Path:
    """Save stacked Altair bars for top entities by 6-level sentiment."""
    # Explanation: Pick entities before counting sentiment, so all charts align.
    entities = top_entities(df, top_n=top_n)
    plot_df = _complete_count_long(df, "sentiment_level", SENTIMENT_LEVELS, entities)
    plot_df = _add_display_columns(plot_df, df, entities)
    plot_df["sentiment_readable"] = plot_df["sentiment_level"].map(SENTIMENT_LABELS)
    plot_df["sentiment_order"] = plot_df["sentiment_level"].map(
        {level: idx for idx, level in enumerate(SENTIMENT_LEVELS)}
    )
    display_order = [entity_display_labels(df, entities)[entity] for entity in entities]

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("n:Q", title="Number of mentions", stack="zero"),
            y=alt.Y("display_entity:N", title="Mentioned entity", sort=display_order),
            color=alt.Color(
                "sentiment_readable:N",
                title="Sentiment level",
                scale=_sentiment_scale(),
                sort=SENTIMENT_READABLE_ORDER,
            ),
            order=alt.Order("sentiment_order:Q"),
            tooltip=[
                alt.Tooltip("display_entity:N", title="Entity"),
                alt.Tooltip("sentiment_readable:N", title="Sentiment"),
                alt.Tooltip("n:Q", title="Mentions"),
            ],
        )
        .properties(title="6-level sentiment composition for top mentioned entities", width=780, height=320)
    )
    return _save_chart(chart, output_path)


def plot_entity_sentiment_heatmap(
    df: pl.DataFrame,
    output_path: Path,
    min_mentions: int = 26,
) -> Path:
    """Save an Altair heatmap of 6-level sentiment by mentioned entity."""
    # Explanation: min_mentions=26 means the heatmap includes all entities with >25 mentions.
    entities = entity_distribution_table(df, min_mentions=min_mentions).get_column("entity_content").to_list()
    return plot_country_x_sentiment_heatmap(df, entities, output_path)


def plot_country_x_sentiment_heatmap(
    df: pl.DataFrame,
    countries: list[str],
    output_path: Path,
) -> Path:
    """Save an Altair heatmap: entity x 6-level sentiment."""
    # Explanation: This reveals which entities cluster at which negative/positive intensity.
    plot_df = _complete_count_long(df, "sentiment_level", SENTIMENT_LEVELS, countries)
    plot_df = _add_display_columns(plot_df, df, countries)
    plot_df["sentiment_readable"] = plot_df["sentiment_level"].map(SENTIMENT_LABELS)
    display_order = [entity_display_labels(df, countries)[entity] for entity in countries]

    base = alt.Chart(plot_df).encode(
        x=alt.X("sentiment_readable:N", title="Sentiment level", sort=SENTIMENT_READABLE_ORDER),
        y=alt.Y("display_entity:N", title="Mentioned entity", sort=display_order),
        tooltip=[
            alt.Tooltip("display_entity:N", title="Entity"),
            alt.Tooltip("sentiment_readable:N", title="Sentiment"),
            alt.Tooltip("n:Q", title="Mentions"),
        ],
    )
    heatmap = base.mark_rect().encode(color=alt.Color("n:Q", title="Mentions", scale=alt.Scale(scheme="redyellowgreen")))
    labels = base.mark_text(fontSize=11).encode(text=alt.Text("n:Q"), color=alt.value("#111111"))
    chart = (heatmap + labels).properties(
        title="Mentioned entity x 6-level sentiment heatmap",
        width=620,
        height=max(260, 25 * len(countries)),
    )
    return _save_chart(chart, output_path)


def plot_entity_distribution_heatmap(
    df: pl.DataFrame,
    output_path: Path,
    min_mentions: int = 26,
) -> Path:
    """Save a one-column Altair heatmap of mention volume by entity."""
    # Explanation: This is the distribution chart as a heatmap, using the same >25 threshold.
    distribution = entity_distribution_table(df, min_mentions=min_mentions)
    entities = distribution.get_column("entity_content").to_list()
    plot_df = _add_display_columns(distribution.to_pandas(), df, entities)
    plot_df["metric"] = "mentions"
    display_order = plot_df["display_entity"].tolist()

    base = alt.Chart(plot_df).encode(
        x=alt.X("metric:N", title=""),
        y=alt.Y("display_entity:N", title="Mentioned entity", sort=display_order),
        tooltip=[
            alt.Tooltip("display_entity:N", title="Entity"),
            alt.Tooltip("n_mentions:Q", title="Mentions"),
            alt.Tooltip("share_percent:Q", title="Share (%)"),
        ],
    )
    heatmap = base.mark_rect().encode(color=alt.Color("n_mentions:Q", title="Mentions", scale=alt.Scale(scheme="blues")))
    labels = base.mark_text(fontSize=11).encode(text=alt.Text("n_mentions:Q"), color=alt.value("#111111"))
    chart = (heatmap + labels).properties(
        title="Mention distribution heatmap (>25 mentions)",
        width=180,
        height=max(260, 25 * len(plot_df)),
    )
    return _save_chart(chart, output_path)


def plot_country_reference_mentions(
    df: pl.DataFrame,
    output_path: Path,
    top_n: int = 15,
) -> Path:
    """Save stacked Altair bars for top entities by reference type."""
    # Explanation: This shows whether a country is used as policy comparison or context.
    entities = top_entities(df, top_n=top_n)
    plot_df = _complete_count_long(df, "ref_type", REF_TYPE_ORDER, entities)
    plot_df = _add_display_columns(plot_df, df, entities)
    plot_df["ref_type_order"] = plot_df["ref_type"].map(
        {ref_type: idx for idx, ref_type in enumerate(REF_TYPE_ORDER)}
    )
    display_order = [entity_display_labels(df, entities)[entity] for entity in entities]

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("n:Q", title="Number of mentions", stack="zero"),
            y=alt.Y("display_entity:N", title="Mentioned entity", sort=display_order),
            color=alt.Color("ref_type:N", title="Reference type", scale=_ref_type_scale(), sort=REF_TYPE_ORDER),
            order=alt.Order("ref_type_order:Q"),
            tooltip=[
                alt.Tooltip("display_entity:N", title="Entity"),
                alt.Tooltip("ref_type:N", title="Reference type"),
                alt.Tooltip("n:Q", title="Mentions"),
            ],
        )
        .properties(title="Policy vs situation references by mentioned entity", width=780, height=320)
    )
    return _save_chart(chart, output_path)


def plot_policy_situation_sentiment(
    df: pl.DataFrame,
    output_path: Path,
    top_n: int = 12,
) -> Path:
    """Save Altair faceted bars for policy/situation mentions by sentiment."""
    # Explanation: "Situation" operationalizes international context: crisis, borders, camps,
    # war, refugees, routes, and other event/condition markers.
    entities = top_entities(df, top_n=top_n)
    filtered = df.filter(
        pl.col("entity_content").is_in(entities)
        & pl.col("ref_type").is_in(["policy", "situation"])
    )
    counts = (
        filtered
        .group_by(["entity_content", "ref_type", "sentiment_level"])
        .agg(pl.len().alias("n"))
        .to_pandas()
    )
    index = pd.MultiIndex.from_product(
        [entities, ["policy", "situation"], SENTIMENT_LEVELS],
        names=["entity_content", "ref_type", "sentiment_level"],
    )
    plot_df = counts.set_index(["entity_content", "ref_type", "sentiment_level"]).reindex(index, fill_value=0).reset_index()
    plot_df = _add_display_columns(plot_df, df, entities)
    plot_df["sentiment_readable"] = plot_df["sentiment_level"].map(SENTIMENT_LABELS)
    plot_df["sentiment_order"] = plot_df["sentiment_level"].map(
        {level: idx for idx, level in enumerate(SENTIMENT_LEVELS)}
    )
    display_order = [entity_display_labels(df, entities)[entity] for entity in entities]

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("n:Q", title="Number of mentions", stack="zero"),
            y=alt.Y("display_entity:N", title="Mentioned entity", sort=display_order),
            color=alt.Color(
                "sentiment_readable:N",
                title="Sentiment level",
                scale=_sentiment_scale(),
                sort=SENTIMENT_READABLE_ORDER,
            ),
            order=alt.Order("sentiment_order:Q"),
            tooltip=[
                alt.Tooltip("display_entity:N", title="Entity"),
                alt.Tooltip("ref_type:N", title="Reference type"),
                alt.Tooltip("sentiment_readable:N", title="Sentiment"),
                alt.Tooltip("n:Q", title="Mentions"),
            ],
        )
        .facet(column=alt.Column("ref_type:N", title="Reference type", sort=["policy", "situation"]))
        .properties(title="6-level sentiment by entity: policy vs international situation context")
        .resolve_scale(y="shared")
    )
    return _save_chart(chart, output_path)


def plot_policy_situation_sentiment_heatmap(df: pl.DataFrame, output_path: Path) -> Path:
    """Save an Altair heatmap for policy vs situation by 6-level sentiment."""
    # Explanation: This is the compact headline view for policy vs international context.
    filtered = df.filter(pl.col("ref_type").is_in(["policy", "situation"]))
    counts = (
        filtered
        .group_by(["ref_type", "sentiment_level"])
        .agg(pl.len().alias("n"))
        .to_pandas()
    )
    index = pd.MultiIndex.from_product(
        [["policy", "situation"], SENTIMENT_LEVELS],
        names=["ref_type", "sentiment_level"],
    )
    plot_df = counts.set_index(["ref_type", "sentiment_level"]).reindex(index, fill_value=0).reset_index()
    plot_df["sentiment_readable"] = plot_df["sentiment_level"].map(SENTIMENT_LABELS)

    base = alt.Chart(plot_df).encode(
        x=alt.X("sentiment_readable:N", title="Sentiment level", sort=SENTIMENT_READABLE_ORDER),
        y=alt.Y("ref_type:N", title="Reference type", sort=["policy", "situation"]),
        tooltip=[
            alt.Tooltip("ref_type:N", title="Reference type"),
            alt.Tooltip("sentiment_readable:N", title="Sentiment"),
            alt.Tooltip("n:Q", title="Mentions"),
        ],
    )
    heatmap = base.mark_rect().encode(color=alt.Color("n:Q", title="Mentions", scale=alt.Scale(scheme="reds")))
    labels = base.mark_text(fontSize=12).encode(text=alt.Text("n:Q"), color=alt.value("#111111"))
    chart = (heatmap + labels).properties(
        title="6-level sentiment heatmap: policy vs international situation references",
        width=620,
        height=160,
    )
    return _save_chart(chart, output_path)


def plot_4x6_heatmap(matrix: pl.DataFrame, output_path: Path) -> Path:
    """Save the headline Altair heatmap: reference type x 6-level sentiment."""
    # Explanation: The matrix is 4 reference types by 6 sentiment levels when all appear.
    pdf = matrix.to_pandas()
    plot_df = pdf.melt(id_vars="ref_type", var_name="sentiment_level", value_name="n")
    plot_df["sentiment_readable"] = plot_df["sentiment_level"].map(SENTIMENT_LABELS)

    base = alt.Chart(plot_df).encode(
        x=alt.X("sentiment_readable:N", title="Sentiment level", sort=SENTIMENT_READABLE_ORDER),
        y=alt.Y("ref_type:N", title="Reference type", sort=REF_TYPE_ORDER),
        tooltip=[
            alt.Tooltip("ref_type:N", title="Reference type"),
            alt.Tooltip("sentiment_readable:N", title="Sentiment"),
            alt.Tooltip("n:Q", title="Mentions"),
        ],
    )
    heatmap = base.mark_rect().encode(color=alt.Color("n:Q", title="Mentions", scale=alt.Scale(scheme="blues")))
    labels = base.mark_text(fontSize=12).encode(text=alt.Text("n:Q"), color=alt.value("#111111"))
    chart = (heatmap + labels).properties(
        title="Reference type x 6-level sentiment",
        width=620,
        height=240,
    )
    return _save_chart(chart, output_path)


def plot_reference_heatmap(
    df: pl.DataFrame,
    output_path: Path,
    top_n: int = 15,
) -> Path:
    """Save an Altair heatmap of mentioned entities by reference type."""
    # Explanation: The heatmap makes each entity's dominant mode of mention visible quickly.
    entities = top_entities(df, top_n=top_n)
    plot_df = _complete_count_long(df, "ref_type", REF_TYPE_ORDER, entities)
    plot_df = _add_display_columns(plot_df, df, entities)
    display_order = [entity_display_labels(df, entities)[entity] for entity in entities]

    base = alt.Chart(plot_df).encode(
        x=alt.X("ref_type:N", title="Reference type", sort=REF_TYPE_ORDER),
        y=alt.Y("display_entity:N", title="Mentioned entity", sort=display_order),
        tooltip=[
            alt.Tooltip("display_entity:N", title="Entity"),
            alt.Tooltip("ref_type:N", title="Reference type"),
            alt.Tooltip("n:Q", title="Mentions"),
        ],
    )
    heatmap = base.mark_rect().encode(color=alt.Color("n:Q", title="Mentions", scale=alt.Scale(scheme="yellowgreenblue")))
    labels = base.mark_text(fontSize=11).encode(text=alt.Text("n:Q"), color=alt.value("#111111"))
    chart = (heatmap + labels).properties(
        title="Reference-type intensity by mentioned entity",
        width=520,
        height=max(260, 25 * len(entities)),
    )
    return _save_chart(chart, output_path)


def save_all_figures(
    df: pl.DataFrame,
    processed_dir: Path,
    top_n: int = 10,
    min_mentions_for_all: int = 26,
) -> dict[str, Path]:
    """Create every pilot Altair visualization and return the saved file paths."""
    # Explanation: One function lets notebook users regenerate all figures consistently.
    figures_dir = ensure_figures_dir(processed_dir)
    output_paths = {
        "entity_distribution_top10": plot_entity_distribution(
            df,
            figures_dir / "entity_distribution_top10.png",
            top_n=10,
        ),
        "entity_distribution_min26": plot_entity_distribution(
            df,
            figures_dir / "entity_distribution_min26.png",
            top_n=None,
            min_mentions=min_mentions_for_all,
        ),
        "entity_distribution_min26_csv": save_significant_entity_distribution_table(
            df,
            processed_dir / "entity_distribution_min26.csv",
            min_mentions=min_mentions_for_all,
        ),
        "entity_distribution_all_csv": save_entity_distribution_table(
            df,
            processed_dir / "entity_distribution_all_for_audit.csv",
        ),
        "reference_type_sentiment_heatmap": plot_4x6_heatmap(
            build_matrix(df),
            figures_dir / "reference_type_sentiment_heatmap.png",
        ),
        "country_sentiment_top10": plot_country_sentiment_mentions(
            df,
            figures_dir / "country_sentiment_mentions_top10.png",
            top_n=top_n,
        ),
        "entity_sentiment_heatmap_min26": plot_entity_sentiment_heatmap(
            df,
            figures_dir / "entity_sentiment_heatmap_min26.png",
            min_mentions=min_mentions_for_all,
        ),
        "entity_distribution_heatmap_min26": plot_entity_distribution_heatmap(
            df,
            figures_dir / "entity_distribution_heatmap_min26.png",
            min_mentions=min_mentions_for_all,
        ),
        "country_reference_type_top10": plot_country_reference_mentions(
            df,
            figures_dir / "country_reference_type_mentions_top10.png",
            top_n=top_n,
        ),
        "policy_vs_situation_sentiment_top10": plot_policy_situation_sentiment(
            df,
            figures_dir / "policy_vs_situation_sentiment_top10.png",
            top_n=top_n,
        ),
        "policy_situation_sentiment_heatmap": plot_policy_situation_sentiment_heatmap(
            df,
            figures_dir / "policy_situation_sentiment_heatmap.png",
        ),
        "region_group_distribution": plot_region_group_distribution(
            df,
            figures_dir / "region_group_distribution.png",
        ),
        "region_group_sentiment": plot_region_group_sentiment(
            df,
            figures_dir / "region_group_sentiment.png",
        ),
        "reference_heatmap_top10": plot_reference_heatmap(
            df,
            figures_dir / "country_reference_heatmap_top10.png",
            top_n=top_n,
        ),
    }
    return output_paths


def plot_concreteness_density_by_weog(
    df: pl.DataFrame,
    output_path: Path,
) -> Path:
    """Save a density plot of concreteness score by WEOG/non-WEOG group."""
    # Explanation: This directly visualizes the abstraction/concreteness hypothesis.
    plot_df = (
        df
        .drop_nulls("concreteness_score")
        .filter(pl.col("weog_group").is_in(["weog", "non_weog"]))
        .select(["concreteness_score", "weog_group"])
        .to_pandas()
    )
    plot_df["weog_group_readable"] = plot_df["weog_group"].map(WEOG_GROUP_LABELS)

    chart = (
        alt.Chart(plot_df)
        .transform_density(
            "concreteness_score",
            as_=["concreteness_score", "density"],
            groupby=["weog_group_readable"],
            extent=[1, 5],
        )
        .mark_area(opacity=0.45)
        .encode(
            x=alt.X("concreteness_score:Q", title="Concreteness score (1 abstract - 5 concrete)"),
            y=alt.Y("density:Q", title="Density"),
            color=alt.Color("weog_group_readable:N", title="Mentioned-country group"),
            tooltip=[
                alt.Tooltip("weog_group_readable:N", title="Group"),
                alt.Tooltip("concreteness_score:Q", title="Concreteness", format=".2f"),
                alt.Tooltip("density:Q", title="Density", format=".3f"),
            ],
        )
        .properties(title="Concreteness distribution: WEOG vs non-WEOG mentions", width=760, height=360)
    )
    return _save_chart(chart, output_path)


def plot_concreteness_by_year_region(
    df: pl.DataFrame,
    output_path: Path,
) -> Path:
    """Save yearly mean concreteness by WEOG/non-WEOG group."""
    # Explanation: The line chart checks whether regional differences are stable over time.
    plot_df = (
        df
        .drop_nulls("concreteness_score")
        .filter(pl.col("weog_group").is_in(["weog", "non_weog"]))
        .group_by(["source_year", "weog_group"])
        .agg([
            pl.len().alias("n_mentions"),
            pl.col("concreteness_score").mean().round(3).alias("mean_concreteness"),
        ])
        .sort(["source_year", "weog_group"])
        .to_pandas()
    )
    plot_df["weog_group_readable"] = plot_df["weog_group"].map(WEOG_GROUP_LABELS)

    chart = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("source_year:O", title="Year"),
            y=alt.Y("mean_concreteness:Q", title="Mean concreteness", scale=alt.Scale(zero=False)),
            color=alt.Color("weog_group_readable:N", title="Mentioned-country group"),
            tooltip=[
                alt.Tooltip("source_year:O", title="Year"),
                alt.Tooltip("weog_group_readable:N", title="Group"),
                alt.Tooltip("mean_concreteness:Q", title="Mean concreteness"),
                alt.Tooltip("n_mentions:Q", title="Mentions"),
            ],
        )
        .properties(title="Mean concreteness by mentioned-country group and year", width=760, height=360)
    )
    return _save_chart(chart, output_path)


def plot_diffusion_top_targets(
    edges: pl.DataFrame,
    output_path: Path,
    top_n: int = 20,
) -> Path:
    """Save top target entities in the policy-diffusion network."""
    # Explanation: Edge weights count how often France references a target entity.
    plot_df = (
        edges
        .group_by(["target_entity", "region_group", "weog_group"])
        .agg(pl.col("weight").sum().alias("total_mentions"))
        .sort("total_mentions", descending=True)
        .head(top_n)
        .to_pandas()
    )
    plot_df["region_group_readable"] = plot_df["region_group"].map(REGION_GROUP_LABELS).fillna(plot_df["weog_group"])
    target_order = plot_df["target_entity"].tolist()

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("total_mentions:Q", title="Weighted mentions"),
            y=alt.Y("target_entity:N", title="Target entity", sort=target_order),
            color=alt.Color("region_group_readable:N", title="Analytical group"),
            tooltip=[
                alt.Tooltip("target_entity:N", title="Target"),
                alt.Tooltip("region_group_readable:N", title="Group"),
                alt.Tooltip("total_mentions:Q", title="Mentions"),
            ],
        )
        .properties(title="Top targets in France policy-diffusion network", width=760, height=max(320, 22 * len(plot_df)))
    )
    return _save_chart(chart, output_path)


def plot_cohort_policy_heatmap(
    edges: pl.DataFrame,
    output_path: Path,
) -> Path:
    """Save a heatmap of migrant cohort x policy measure edge weights."""
    # Explanation: This shows which migrant groups are linked to which policy instruments.
    plot_df = (
        edges
        .group_by(["migrant_cohort", "policy_measure"])
        .agg(pl.col("weight").sum().alias("weight"))
        .to_pandas()
    )

    base = alt.Chart(plot_df).encode(
        x=alt.X("policy_measure:N", title="Policy measure"),
        y=alt.Y("migrant_cohort:N", title="Migrant cohort"),
        tooltip=[
            alt.Tooltip("migrant_cohort:N", title="Cohort"),
            alt.Tooltip("policy_measure:N", title="Policy measure"),
            alt.Tooltip("weight:Q", title="Weighted mentions"),
        ],
    )
    heatmap = base.mark_rect().encode(color=alt.Color("weight:Q", title="Weighted mentions", scale=alt.Scale(scheme="blues")))
    labels = base.mark_text(fontSize=10).encode(text=alt.Text("weight:Q"), color=alt.value("#111111"))
    chart = (heatmap + labels).properties(
        title="Migrant cohort x policy measure in diffusion references",
        width=760,
        height=300,
    )
    return _save_chart(chart, output_path)


def save_extended_figures(
    df: pl.DataFrame,
    edges: pl.DataFrame,
    processed_dir: Path,
) -> dict[str, Path]:
    """Save Altair figures for the 2017-2022 extended hypotheses."""
    # Explanation: These figures are separate from the original 2018 pilot charts.
    figures_dir = processed_dir / "figures_altair_extended"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return {
        "concreteness_density_by_weog": plot_concreteness_density_by_weog(
            df,
            figures_dir / "concreteness_density_by_weog.png",
        ),
        "concreteness_by_year_region": plot_concreteness_by_year_region(
            df,
            figures_dir / "concreteness_by_year_region.png",
        ),
        "diffusion_top_targets": plot_diffusion_top_targets(
            edges,
            figures_dir / "diffusion_top_targets.png",
        ),
        "cohort_policy_heatmap": plot_cohort_policy_heatmap(
            edges,
            figures_dir / "cohort_policy_heatmap.png",
        ),
    }
