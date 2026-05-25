"""Vega-Altair visualization helpers for the France 2018 migration pilot."""

from __future__ import annotations

import html
import json
from pathlib import Path

import altair as alt
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go

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


def _save_plotly(fig: go.Figure, output_path: Path) -> Path:
    """Save a Plotly figure as interactive HTML."""
    # Explanation: Plotly HTML works well for ternary plots, maps, and dense timelines.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_path = output_path.with_suffix(".html")
    fig.write_html(html_path, include_plotlyjs="cdn")
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


def plot_bilateral_concreteness_matrix(
    matrix: pl.DataFrame,
    output_path: Path,
    min_mentions: int = 5,
) -> Path:
    """Save source-country x target-country matrix: color concreteness, dot size volume."""
    plot_df = (
        matrix
        .filter(pl.col("n_mentions") >= min_mentions)
        .with_columns(
            pl.when(pl.col("target_country_name").is_not_null())
            .then(pl.concat_str([pl.col("target_country_name"), pl.lit(" ("), pl.col("target_country"), pl.lit(")")]))
            .otherwise(pl.col("target_country"))
            .alias("target_label")
        )
        .sort(["source_country", "n_mentions"], descending=[False, True])
        .to_pandas()
    )
    if plot_df.empty:
        plot_df = pd.DataFrame(columns=[
            "source_country",
            "target_label",
            "mean_concreteness",
            "median_concreteness",
            "n_mentions",
            "n_speeches",
        ])

    base = alt.Chart(plot_df).encode(
        x=alt.X("target_label:N", title="Mentioned country", sort="-y"),
        y=alt.Y("source_country:N", title="Speaking parliament"),
        tooltip=[
            alt.Tooltip("source_country:N", title="Source"),
            alt.Tooltip("target_label:N", title="Target"),
            alt.Tooltip("n_mentions:Q", title="Mentions"),
            alt.Tooltip("n_speeches:Q", title="Speeches"),
            alt.Tooltip("mean_concreteness:Q", title="Mean concreteness", format=".2f"),
            alt.Tooltip("median_concreteness:Q", title="Median concreteness", format=".2f"),
        ],
    )
    heat = base.mark_rect().encode(
        color=alt.Color(
            "mean_concreteness:Q",
            title="Mean concreteness",
            scale=alt.Scale(scheme="viridis"),
        )
    )
    dots = base.mark_circle(color="#111827", opacity=0.55).encode(
        size=alt.Size("n_mentions:Q", title="Mentions", scale=alt.Scale(range=[20, 900])),
    )
    chart = (heat + dots).interactive().properties(
        title="Bilateral migration-reference matrix: concreteness and mention volume",
        width=max(760, 24 * plot_df["target_label"].nunique()) if not plot_df.empty else 760,
        height=max(260, 58 * plot_df["source_country"].nunique()) if not plot_df.empty else 260,
    )
    return _save_chart(chart, output_path)


def plot_attention_asymmetry_bars(
    asymmetry: pl.DataFrame,
    output_path: Path,
    top_n: int = 30,
) -> Path:
    """Save the most asymmetric reciprocal country-pair attention gaps."""
    if asymmetry.is_empty():
        plot_df = pd.DataFrame(columns=[
            "pair",
            "attention_log_ratio",
            "attention_share_AtoB",
            "n_mentions",
            "n_mentions_reverse",
            "total_traffic",
            "sentiment_gap",
            "direction",
        ])
    else:
        plot_df = (
            asymmetry
            .with_columns([
                pl.concat_str([
                    pl.col("source_country"),
                    pl.lit(" -> "),
                    pl.col("target_country"),
                    pl.lit(" vs reverse"),
                ]).alias("pair"),
                pl.when(pl.col("attention_log_ratio") >= 0)
                .then(pl.lit("A mentions B more"))
                .otherwise(pl.lit("B mentions A more"))
                .alias("direction"),
                pl.col("attention_log_ratio").abs().alias("abs_attention_log_ratio"),
            ])
            .sort("abs_attention_log_ratio", descending=True)
            .head(top_n)
            .sort("attention_log_ratio")
            .to_pandas()
        )
    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("attention_log_ratio:Q", title="Attention log-ratio"),
            y=alt.Y("pair:N", title="Country pair", sort=plot_df["pair"].tolist() if not plot_df.empty else []),
            color=alt.Color(
                "direction:N",
                title="Asymmetry direction",
                scale=alt.Scale(range=["#b42318", "#1d70b8"]),
            ),
            tooltip=[
                alt.Tooltip("pair:N", title="Pair"),
                alt.Tooltip("n_mentions:Q", title="A -> B mentions"),
                alt.Tooltip("n_mentions_reverse:Q", title="B -> A mentions"),
                alt.Tooltip("total_traffic:Q", title="Total traffic"),
                alt.Tooltip("attention_share_AtoB:Q", title="A -> B share", format=".1%"),
                alt.Tooltip("sentiment_gap:Q", title="Sentiment gap", format=".3f"),
            ],
        )
        .interactive()
        .properties(
            title="Most asymmetric reciprocal migration attention",
            width=780,
            height=max(320, 24 * len(plot_df)),
        )
    )
    return _save_chart(chart, output_path)


def plot_shock_delta_heatmap(
    shock_table: pl.DataFrame,
    output_path: Path,
    title: str,
    min_abs_delta: int = 1,
) -> Path:
    """Save a source-target heatmap of attention change after a shock event."""
    target_col = "target_country_iso3" if "target_country_iso3" in shock_table.columns else "target_country"
    plot_df = (
        shock_table
        .filter(pl.col("delta_mentions").abs() >= min_abs_delta)
        .with_columns(
            pl.when(pl.col("target_country_name").is_not_null())
            .then(pl.concat_str([pl.col("target_country_name"), pl.lit(" ("), pl.col(target_col), pl.lit(")")]))
            .otherwise(pl.col(target_col))
            .alias("target_label")
        )
        .sort("delta_mentions", descending=True)
        .to_pandas()
    )
    if plot_df.empty:
        plot_df = pd.DataFrame(columns=[
            "source_country",
            "target_label",
            "delta_mentions",
            "delta_sentiment",
            "n_mentions_pre",
            "n_mentions_post",
        ])
    chart = (
        alt.Chart(plot_df)
        .mark_rect()
        .encode(
            x=alt.X("target_label:N", title="Target country", sort="-color"),
            y=alt.Y("source_country:N", title="Speaking parliament"),
            color=alt.Color(
                "delta_mentions:Q",
                title="Post - pre mentions",
                scale=alt.Scale(scheme="redblue", reverse=True),
            ),
            tooltip=[
                alt.Tooltip("source_country:N", title="Source"),
                alt.Tooltip("target_label:N", title="Target"),
                alt.Tooltip("n_mentions_pre:Q", title="Pre mentions"),
                alt.Tooltip("n_mentions_post:Q", title="Post mentions"),
                alt.Tooltip("delta_mentions:Q", title="Delta mentions"),
                alt.Tooltip("delta_sentiment:Q", title="Delta sentiment", format=".3f"),
            ],
        )
        .interactive()
        .properties(
            title=title,
            width=max(720, 26 * plot_df["target_label"].nunique()) if not plot_df.empty else 720,
            height=max(260, 54 * plot_df["source_country"].nunique()) if not plot_df.empty else 260,
        )
    )
    return _save_chart(chart, output_path)


def save_data_model_figures(
    bilateral_concrete: pl.DataFrame,
    asymmetry: pl.DataFrame,
    processed_dir: Path,
    shock_tables: dict[str, pl.DataFrame] | None = None,
) -> dict[str, Path]:
    """Save the core dyadic-model visualizations."""
    figures_dir = processed_dir / "figures_data_model"
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "bilateral_concreteness_matrix": plot_bilateral_concreteness_matrix(
            bilateral_concrete,
            figures_dir / "bilateral_concreteness_matrix.png",
        ),
        "attention_asymmetry_bars": plot_attention_asymmetry_bars(
            asymmetry,
            figures_dir / "attention_asymmetry_bars.png",
        ),
    }
    for shock_name, shock_table in (shock_tables or {}).items():
        paths[f"shock_{shock_name}"] = plot_shock_delta_heatmap(
            shock_table,
            figures_dir / f"shock_{shock_name}_delta_heatmap.png",
            title=f"Attention shift around {shock_name.replace('_', ' ')}",
        )
    return paths


def plot_country_concreteness_year_lines(
    df: pl.DataFrame,
    output_path: Path,
    top_n: int = 12,
) -> Path:
    """Save country/entity-level concreteness trends by year."""
    entities = top_entities(df, top_n=top_n)
    plot_df = (
        df
        .drop_nulls("concreteness_score")
        .filter(pl.col("entity_content").is_in(entities))
        .group_by(["entity_content", "source_year"])
        .agg([
            pl.len().alias("n_mentions"),
            pl.col("concreteness_score").mean().round(3).alias("mean_concreteness"),
            pl.col("narrative_frame").mode().first().alias("dominant_frame"),
            pl.col("policy_agency_type").mode().first().alias("dominant_agency"),
            pl.col("migration_direction").mode().first().alias("dominant_direction"),
        ])
        .sort(["entity_content", "source_year"])
        .to_pandas()
    )
    chart = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("source_year:O", title="Year"),
            y=alt.Y("mean_concreteness:Q", title="Mean concreteness", scale=alt.Scale(zero=False)),
            color=alt.Color("entity_content:N", title="Country/entity"),
            size=alt.Size("n_mentions:Q", title="Mentions", scale=alt.Scale(range=[1, 5])),
            tooltip=[
                alt.Tooltip("entity_content:N", title="Entity"),
                alt.Tooltip("source_year:O", title="Year"),
                alt.Tooltip("n_mentions:Q", title="Mentions"),
                alt.Tooltip("mean_concreteness:Q", title="Mean concreteness"),
                alt.Tooltip("dominant_frame:N", title="Dominant frame"),
                alt.Tooltip("dominant_agency:N", title="Dominant agency"),
                alt.Tooltip("dominant_direction:N", title="Dominant direction"),
            ],
        )
        .interactive()
        .properties(
            title="Country/entity-level concreteness trends by year",
            width=820,
            height=430,
        )
    )
    return _save_chart(chart, output_path)


def plot_concreteness_feature_pattern_heatmap(
    df: pl.DataFrame,
    output_path: Path,
    top_frames: int = 16,
) -> Path:
    """Save a heatmap showing repeated narrative/agency patterns and concreteness."""
    frames = (
        df
        .group_by("narrative_frame")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(top_frames)
        .get_column("narrative_frame")
        .to_list()
    )
    plot_df = (
        df
        .drop_nulls("concreteness_score")
        .filter(pl.col("narrative_frame").is_in(frames))
        .group_by(["policy_agency_type", "narrative_frame"])
        .agg([
            pl.len().alias("n_mentions"),
            pl.col("concreteness_score").mean().round(3).alias("mean_concreteness"),
            pl.col("entity_content").mode().first().alias("example_entity"),
            pl.col("migration_direction").mode().first().alias("dominant_direction"),
        ])
        .to_pandas()
    )
    base = alt.Chart(plot_df).encode(
        x=alt.X("narrative_frame:N", title="Narrative frame", sort=frames),
        y=alt.Y("policy_agency_type:N", title="Policy agency"),
        tooltip=[
            alt.Tooltip("policy_agency_type:N", title="Agency"),
            alt.Tooltip("narrative_frame:N", title="Frame"),
            alt.Tooltip("n_mentions:Q", title="Mentions"),
            alt.Tooltip("mean_concreteness:Q", title="Mean concreteness"),
            alt.Tooltip("example_entity:N", title="Typical entity"),
            alt.Tooltip("dominant_direction:N", title="Dominant direction"),
        ],
    )
    heatmap = base.mark_rect().encode(
        color=alt.Color(
            "mean_concreteness:Q",
            title="Mean concreteness",
            scale=alt.Scale(scheme="viridis", domain=[2.8, 3.8]),
        )
    )
    labels = base.mark_text(fontSize=9).encode(
        text=alt.Text("n_mentions:Q"),
        color=alt.value("#111111"),
    )
    chart = (heatmap + labels).interactive().properties(
        title="Repeated agency x narrative patterns: count and mean concreteness",
        width=840,
        height=320,
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


def save_concreteness_quote_panels(
    df: pl.DataFrame,
    output_path: Path,
    top_n: int = 12,
    examples_per_country: int = 2,
) -> Path:
    """Save an HTML quote panel with concrete and abstract excerpts by country/entity."""
    entities = top_entities(df, top_n=top_n)
    working = (
        df
        .filter(pl.col("entity_content").is_in(entities))
        .with_columns(
            pl.col("context_window")
            .map_elements(lambda text: " ".join(str(text).replace("||", " ").split())[:520], return_dtype=pl.Utf8)
            .alias("context_excerpt")
        )
    )
    rows = []
    for entity in entities:
        sub = working.filter(pl.col("entity_content") == entity)
        if sub.is_empty():
            continue
        concrete = (
            sub
            .sort(["concreteness_score", "source_year"], descending=[True, False])
            .head(examples_per_country)
            .with_columns(pl.lit("most concrete").alias("example_type"))
        )
        abstract = (
            sub
            .sort(["concreteness_score", "source_year"], descending=[False, False])
            .head(examples_per_country)
            .with_columns(pl.lit("most abstract").alias("example_type"))
        )
        rows.extend((concrete.select([
            "entity_content",
            "example_type",
            "source_year",
            "session_date",
            "concreteness_score",
            "narrative_frame",
            "policy_agency_type",
            "migration_direction",
            "context_excerpt",
        ])).to_dicts())
        rows.extend((abstract.select([
            "entity_content",
            "example_type",
            "source_year",
            "session_date",
            "concreteness_score",
            "narrative_frame",
            "policy_agency_type",
            "migration_direction",
            "context_excerpt",
        ])).to_dicts())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for row in rows:
        cards.append(
            "<article class='quote-card'>"
            f"<h3>{html.escape(str(row['entity_content']))} <span>{html.escape(str(row['example_type']))}</span></h3>"
            "<p class='meta'>"
            f"{html.escape(str(row['session_date']))} | score {row['concreteness_score']:.3f} | "
            f"{html.escape(str(row['narrative_frame']))} | {html.escape(str(row['policy_agency_type']))} | "
            f"{html.escape(str(row['migration_direction']))}"
            "</p>"
            f"<p class='excerpt'>{html.escape(str(row['context_excerpt']))}</p>"
            "</article>"
        )
    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Concreteness Quote Panels</title>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:32px;background:#f8f9fa;color:#1f2933;}"
        "h1{font-size:24px;margin-bottom:6px;} .sub{color:#52606d;margin-bottom:24px;}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px;}"
        ".quote-card{background:#fff;border:1px solid #d9e2ec;border-radius:8px;padding:14px;box-shadow:0 1px 2px rgba(15,23,42,.05);}"
        "h3{font-size:16px;margin:0 0 8px 0;} h3 span{font-size:12px;color:#52606d;font-weight:500;}"
        ".meta{font-size:12px;color:#616e7c;margin:0 0 10px 0;} .excerpt{font-size:14px;line-height:1.45;margin:0;}"
        "</style></head><body>"
        "<h1>What People Actually Say: Concrete and Abstract Migration References</h1>"
        "<p class='sub'>Top mentioned countries/entities, with high- and low-concreteness excerpts for interpretation.</p>"
        f"<section class='grid'>{''.join(cards)}</section>"
        "</body></html>"
    )
    output_path.write_text(document, encoding="utf-8")
    return output_path


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
        "country_concreteness_year_lines": plot_country_concreteness_year_lines(
            df,
            figures_dir / "country_concreteness_year_lines.png",
        ),
        "concreteness_feature_pattern_heatmap": plot_concreteness_feature_pattern_heatmap(
            df,
            figures_dir / "concreteness_feature_pattern_heatmap.png",
        ),
        "concreteness_quote_panels": save_concreteness_quote_panels(
            df,
            figures_dir / "concreteness_quote_panels.html",
        ),
        "diffusion_top_targets": plot_diffusion_top_targets(
            edges,
            figures_dir / "diffusion_top_targets.png",
        ),
        "cohort_policy_heatmap": plot_cohort_policy_heatmap(
            edges,
            figures_dir / "cohort_policy_heatmap.png",
        ),
        "country_concreteness_bubble": plot_country_concreteness_bubble(
            df,
            figures_dir / "country_concreteness_bubble.png",
        ),
        "country_year_concreteness_heatmap": plot_country_year_concreteness_heatmap(
            df,
            figures_dir / "country_year_concreteness_heatmap.png",
        ),
        "country_cohort_heatmap": plot_country_cohort_heatmap(
            edges,
            figures_dir / "country_cohort_heatmap.png",
        ),
        "country_policy_heatmap": plot_country_policy_heatmap(
            edges,
            figures_dir / "country_policy_heatmap.png",
        ),
    }


def plot_policy_agency_network(
    agency_edges: pl.DataFrame,
    output_path: Path,
    top_n: int = 30,
) -> Path:
    """Save an interactive Altair node-link view of policy agency edges."""
    # Explanation: This compact network centers France and shows how targets are
    # mentioned as FROM models, TO pressure targets, competitors, partners, or neutral reports.
    import networkx as nx

    top_targets = (
        agency_edges
        .group_by("target_entity")
        .agg(pl.col("weight").sum().alias("total_weight"))
        .sort("total_weight", descending=True)
        .head(top_n)
        .get_column("target_entity")
        .to_list()
    )
    filtered = agency_edges.filter(pl.col("target_entity").is_in(top_targets))
    graph = nx.Graph()
    for row in filtered.to_dicts():
        graph.add_node(row["source_country"])
        graph.add_node(row["target_entity"])
        graph.add_edge(row["source_country"], row["target_entity"], weight=int(row["weight"]))
    positions = nx.spring_layout(graph, seed=42, k=0.8)

    node_weights = (
        filtered
        .group_by("target_entity")
        .agg(pl.col("weight").sum().alias("total_weight"))
        .to_pandas()
    )
    node_weight_lookup = dict(zip(node_weights["target_entity"], node_weights["total_weight"]))
    nodes = pd.DataFrame([
        {
            "node": node,
            "x": xy[0],
            "y": xy[1],
            "total_weight": node_weight_lookup.get(node, filtered.get_column("weight").sum()),
            "node_type": "source" if node in set(filtered.get_column("source_country").unique().to_list()) else "target",
        }
        for node, xy in positions.items()
    ])

    edges = filtered.to_pandas()
    edges["x"] = edges["source_country"].map(lambda node: positions[node][0])
    edges["y"] = edges["source_country"].map(lambda node: positions[node][1])
    edges["x2"] = edges["target_entity"].map(lambda node: positions[node][0])
    edges["y2"] = edges["target_entity"].map(lambda node: positions[node][1])

    edge_layer = (
        alt.Chart(edges)
        .mark_rule(opacity=0.55)
        .encode(
            x="x:Q",
            y="y:Q",
            x2="x2:Q",
            y2="y2:Q",
            color=alt.Color("policy_agency_type:N", title="Policy agency"),
            size=alt.Size("weight:Q", title="Weight", scale=alt.Scale(range=[0.5, 6])),
            tooltip=[
                alt.Tooltip("source_country:N", title="Source"),
                alt.Tooltip("target_entity:N", title="Target"),
                alt.Tooltip("policy_agency_type:N", title="Agency"),
                alt.Tooltip("migrant_cohort:N", title="Cohort"),
                alt.Tooltip("policy_measure:N", title="Policy measure"),
                alt.Tooltip("weight:Q", title="Mentions"),
                alt.Tooltip("agency_markers:N", title="Agency markers"),
            ],
        )
    )
    node_layer = (
        alt.Chart(nodes)
        .mark_circle(stroke="#222222", strokeWidth=0.7)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            size=alt.Size("total_weight:Q", title="Target mentions", scale=alt.Scale(range=[150, 1800])),
            color=alt.Color("node_type:N", title="Node type"),
            tooltip=[
                alt.Tooltip("node:N", title="Node"),
                alt.Tooltip("total_weight:Q", title="Mentions"),
            ],
        )
    )
    text_layer = (
        alt.Chart(nodes)
        .mark_text(dx=8, dy=-8, fontSize=11)
        .encode(x="x:Q", y="y:Q", text="node:N")
    )
    chart = (edge_layer + node_layer + text_layer).interactive().properties(
        title="Policy Agency Network: France -> mentioned countries/entities",
        width=820,
        height=620,
    )
    return _save_chart(chart, output_path)


def plot_policy_agency_country_heatmap(
    agency_edges: pl.DataFrame,
    output_path: Path,
    top_n: int = 30,
) -> Path:
    """Save target country/entity x policy agency heatmap."""
    targets = _top_targets_from_edges(
        agency_edges.rename({"target_entity": "target_entity"}),
        top_n=top_n,
    )
    plot_df = (
        agency_edges
        .filter(pl.col("target_entity").is_in(targets))
        .group_by(["target_entity", "policy_agency_type"])
        .agg(pl.col("weight").sum().alias("weight"))
        .to_pandas()
    )
    base = alt.Chart(plot_df).encode(
        x=alt.X("policy_agency_type:N", title="Policy agency"),
        y=alt.Y("target_entity:N", title="Mentioned country/entity", sort=targets),
        tooltip=[
            alt.Tooltip("target_entity:N", title="Target"),
            alt.Tooltip("policy_agency_type:N", title="Agency"),
            alt.Tooltip("weight:Q", title="Mentions"),
        ],
    )
    heatmap = base.mark_rect().encode(color=alt.Color("weight:Q", title="Mentions", scale=alt.Scale(scheme="purples")))
    labels = base.mark_text(fontSize=9).encode(text=alt.Text("weight:Q"), color=alt.value("#111111"))
    chart = (heatmap + labels).interactive().properties(
        title="Which countries/entities are mentioned through which policy agency mechanisms?",
        width=760,
        height=max(360, 22 * len(targets)),
    )
    return _save_chart(chart, output_path)


def plot_policy_hubs_pagerank(
    hubs: pl.DataFrame,
    output_path: Path,
    top_n: int = 20,
) -> Path:
    """Save a PageRank bar chart for countries/entities cited as policy hubs."""
    plot_df = (
        hubs
        .sort("pagerank", descending=True)
        .head(top_n)
        .to_pandas()
    )
    target_order = plot_df["target_entity"].tolist()
    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("pagerank:Q", title="PageRank score"),
            y=alt.Y("target_entity:N", title="Policy hub", sort=target_order),
            color=alt.Color("dominant_policy_agency:N", title="Dominant agency"),
            tooltip=[
                alt.Tooltip("target_entity:N", title="Target"),
                alt.Tooltip("pagerank:Q", title="PageRank", format=".4f"),
                alt.Tooltip("total_weight:Q", title="Weighted mentions"),
                alt.Tooltip("dominant_policy_agency:N", title="Dominant agency"),
            ],
        )
        .interactive()
        .properties(
            title="Policy Hubs: PageRank in the directed agency network",
            width=760,
            height=max(320, 24 * len(plot_df)),
        )
    )
    return _save_chart(chart, output_path)


def plot_narrative_ternary(
    df: pl.DataFrame,
    output_path: Path,
    min_mentions: int = 10,
) -> Path:
    """Save a Plotly ternary plot of positive/risk/administrative narrative flavor."""
    # Explanation: Each point is a country/entity; coordinates are narrative polarity shares.
    counts = (
        df
        .group_by(["entity_content", "narrative_polarity"])
        .agg(pl.len().alias("n"))
        .to_pandas()
    )
    pivot = counts.pivot(index="entity_content", columns="narrative_polarity", values="n").fillna(0)
    for col in ["positive_sympathy", "positive_benefit", "negative_risk", "neutral_administrative"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["positive"] = pivot["positive_sympathy"] + pivot["positive_benefit"]
    pivot["negative"] = pivot["negative_risk"]
    pivot["neutral"] = pivot["neutral_administrative"]
    pivot["total"] = pivot[["positive", "negative", "neutral"]].sum(axis=1)
    pivot = pivot[pivot["total"] >= min_mentions].reset_index()
    for col in ["positive", "negative", "neutral"]:
        pivot[f"{col}_share"] = pivot[col] / pivot["total"]

    fig = px.scatter_ternary(
        pivot,
        a="positive_share",
        b="negative_share",
        c="neutral_share",
        size="total",
        hover_name="entity_content",
        hover_data={"total": True, "positive_share": ":.2f", "negative_share": ":.2f", "neutral_share": ":.2f"},
        title="Narrative Mirror: positive / risk / administrative framing by country/entity",
    )
    fig.update_layout(ternary_sum=1)
    return _save_plotly(fig, output_path)


def plot_narrative_ternary_by_group(
    df: pl.DataFrame,
    output_path: Path,
    group_col: str,
    title: str,
    min_mentions: int = 10,
) -> Path:
    """Save a ternary narrative plot for a supplied grouping column."""
    counts = (
        df
        .drop_nulls(group_col)
        .group_by([group_col, "narrative_polarity"])
        .agg(pl.len().alias("n"))
        .to_pandas()
    )
    pivot = counts.pivot(index=group_col, columns="narrative_polarity", values="n").fillna(0)
    for col in ["positive_sympathy", "positive_benefit", "negative_risk", "neutral_administrative"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["positive"] = pivot["positive_sympathy"] + pivot["positive_benefit"]
    pivot["negative"] = pivot["negative_risk"]
    pivot["neutral"] = pivot["neutral_administrative"]
    pivot["total"] = pivot[["positive", "negative", "neutral"]].sum(axis=1)
    pivot = pivot[pivot["total"] >= min_mentions].reset_index()
    for col in ["positive", "negative", "neutral"]:
        pivot[f"{col}_share"] = pivot[col] / pivot["total"]

    fig = px.scatter_ternary(
        pivot,
        a="positive_share",
        b="negative_share",
        c="neutral_share",
        size="total",
        hover_name=group_col,
        hover_data={"total": True, "positive_share": ":.2f", "negative_share": ":.2f", "neutral_share": ":.2f"},
        title=title,
    )
    fig.update_layout(ternary_sum=1)
    return _save_plotly(fig, output_path)


def plot_narrative_mirror_bars(
    df: pl.DataFrame,
    output_path: Path,
    top_n: int = 20,
) -> Path:
    """Save bar charts comparing imagined risks, benefits, and neutral frames."""
    top_entities_for_plot = top_entities(df, top_n=top_n)
    plot_df = (
        df
        .filter(pl.col("entity_content").is_in(top_entities_for_plot))
        .with_columns(
            pl.when(pl.col("narrative_polarity").is_in(["positive_sympathy", "positive_benefit"]))
            .then(pl.lit("Imagined benefits / obligations"))
            .when(pl.col("narrative_polarity") == "negative_risk")
            .then(pl.lit("Imagined risks / threats"))
            .otherwise(pl.lit("Administrative / neutral"))
            .alias("narrative_bucket")
        )
        .group_by(["entity_content", "narrative_bucket"])
        .agg(pl.len().alias("n"))
        .to_pandas()
    )
    bucket_order = [
        "Imagined benefits / obligations",
        "Imagined risks / threats",
        "Administrative / neutral",
    ]
    index = pd.MultiIndex.from_product(
        [top_entities_for_plot, bucket_order],
        names=["entity_content", "narrative_bucket"],
    )
    plot_df = (
        plot_df
        .set_index(["entity_content", "narrative_bucket"])
        .reindex(index, fill_value=0)
        .reset_index()
    )
    totals = plot_df.groupby("entity_content")["n"].transform("sum")
    plot_df["share"] = (plot_df["n"] / totals).fillna(0)
    plot_df["bucket_order"] = plot_df["narrative_bucket"].map(
        {bucket: idx for idx, bucket in enumerate(bucket_order)}
    )
    plot_df = _add_display_columns(plot_df, df, top_entities_for_plot)
    display_order = [entity_display_labels(df, top_entities_for_plot)[entity] for entity in top_entities_for_plot]

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("n:Q", title="Share of narrative frames", stack="normalize"),
            y=alt.Y("display_entity:N", title="Mentioned country/entity", sort=display_order),
            color=alt.Color(
                "narrative_bucket:N",
                title="Narrative side",
                scale=alt.Scale(
                    domain=bucket_order,
                    range=["#2a9d8f", "#c44536", "#6c757d"],
                ),
            ),
            order=alt.Order("bucket_order:Q"),
            tooltip=[
                alt.Tooltip("display_entity:N", title="Entity"),
                alt.Tooltip("narrative_bucket:N", title="Narrative side"),
                alt.Tooltip("n:Q", title="Mentions"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        )
        .interactive()
        .properties(
            title="Narrative Mirror: imagined risks versus benefits by country/entity",
            width=780,
            height=max(360, 24 * len(top_entities_for_plot)),
        )
    )
    return _save_chart(chart, output_path)


def plot_narrative_mirror_bars_by_group(
    df: pl.DataFrame,
    output_path: Path,
    group_col: str,
    title: str,
    top_n: int = 20,
) -> Path:
    """Save risk/benefit narrative bars for a supplied grouping column."""
    groups = (
        df
        .drop_nulls(group_col)
        .group_by(group_col)
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(top_n)
        .get_column(group_col)
        .to_list()
    )
    plot_df = (
        df
        .filter(pl.col(group_col).is_in(groups))
        .with_columns(
            pl.when(pl.col("narrative_polarity").is_in(["positive_sympathy", "positive_benefit"]))
            .then(pl.lit("Imagined benefits / obligations"))
            .when(pl.col("narrative_polarity") == "negative_risk")
            .then(pl.lit("Imagined risks / threats"))
            .otherwise(pl.lit("Administrative / neutral"))
            .alias("narrative_bucket")
        )
        .group_by([group_col, "narrative_bucket"])
        .agg(pl.len().alias("n"))
        .to_pandas()
    )
    bucket_order = [
        "Imagined benefits / obligations",
        "Imagined risks / threats",
        "Administrative / neutral",
    ]
    index = pd.MultiIndex.from_product([groups, bucket_order], names=[group_col, "narrative_bucket"])
    plot_df = (
        plot_df
        .set_index([group_col, "narrative_bucket"])
        .reindex(index, fill_value=0)
        .reset_index()
    )
    totals = plot_df.groupby(group_col)["n"].transform("sum")
    plot_df["share"] = (plot_df["n"] / totals).fillna(0)
    plot_df["bucket_order"] = plot_df["narrative_bucket"].map(
        {bucket: idx for idx, bucket in enumerate(bucket_order)}
    )

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("n:Q", title="Share of narrative frames", stack="normalize"),
            y=alt.Y(f"{group_col}:N", title=group_col.replace("_", " ").title(), sort=groups),
            color=alt.Color(
                "narrative_bucket:N",
                title="Narrative side",
                scale=alt.Scale(domain=bucket_order, range=["#2a9d8f", "#c44536", "#6c757d"]),
            ),
            order=alt.Order("bucket_order:Q"),
            tooltip=[
                alt.Tooltip(f"{group_col}:N", title=group_col.replace("_", " ").title()),
                alt.Tooltip("narrative_bucket:N", title="Narrative side"),
                alt.Tooltip("n:Q", title="Mentions"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        )
        .interactive()
        .properties(title=title, width=780, height=max(320, 28 * len(groups)))
    )
    return _save_chart(chart, output_path)


def plot_evidence_visibility_map(
    visible_summary: pl.DataFrame,
    output_path: Path,
) -> Path:
    """Save a Plotly choropleth of high-concreteness visible country mentions."""
    # Explanation: This shows which real-world countries become concrete evidence points.
    plot_df = (
        visible_summary
        .drop_nulls("target_iso3")
        .filter(pl.col("target_iso3") != "")
        .to_pandas()
    )
    fig = px.choropleth(
        plot_df,
        locations="target_iso3",
        color="visible_event_mentions",
        hover_name="entity_content",
        hover_data={
            "visible_event_mentions": True,
            "mean_concreteness": True,
            "dominant_frame": True,
            "dominant_agency": True,
            "target_iso3": False,
        },
        color_continuous_scale="YlOrRd",
        title="Evidence-Visibility Map: high-concreteness country references",
    )
    fig.update_geos(showframe=False, showcoastlines=True, projection_type="natural earth")
    return _save_plotly(fig, output_path)


def plot_fact_density_timeline(
    events: pl.DataFrame,
    output_path: Path,
    max_points: int = 1000,
) -> Path:
    """Save Plotly timeline of high-concreteness fact/event snippets."""
    # Explanation: Points are concrete snippets; hover text shows named entities and frames.
    plot_df = (
        events
        .sort("concreteness_score", descending=True)
        .head(max_points)
        .with_columns(
            pl.col("context_window")
            .map_elements(lambda text: " ".join(str(text).replace("||", " ").split())[:420], return_dtype=pl.Utf8)
            .alias("context_excerpt")
        )
        .to_pandas()
    )
    fig = px.scatter(
        plot_df,
        x="session_date",
        y="concreteness_score",
        size="concreteness_score",
        color="narrative_frame",
        hover_name="entity_content",
        hover_data={
            "proper_noun_anchors": True,
            "countries_detected_in_context": True,
            "policy_agency_type": True,
            "migrant_cohort": True,
            "policy_measure": True,
            "migration_direction": True,
            "context_excerpt": True,
            "concreteness_score": ":.3f",
        },
        title="Fact-Density Timeline: high-concreteness migration references",
    )
    fig.update_layout(xaxis_title="Session date", yaxis_title="Concreteness score")
    return _save_plotly(fig, output_path)


def plot_europe_concrete_conversation_map(
    europe_summary: pl.DataFrame,
    output_path: Path,
) -> Path:
    """Save a Europe-focused map of concrete migration conversations."""
    plot_df = (
        europe_summary
        .drop_nulls("target_iso3")
        .filter(pl.col("target_iso3") != "")
        .to_pandas()
    )
    fig = px.choropleth(
        plot_df,
        locations="target_iso3",
        color="concrete_event_mentions",
        hover_name="entity_content",
        hover_data={
            "concrete_event_mentions": True,
            "mean_fact_concreteness": True,
            "dominant_frame": True,
            "dominant_agency": True,
            "top_event_labels": True,
            "sample_excerpt": True,
            "target_iso3": False,
        },
        color_continuous_scale="YlOrRd",
        title="Europe Map: concrete migration conversations mentioned in parliament",
    )
    fig.update_geos(
        scope="europe",
        showframe=False,
        showcoastlines=True,
        projection_type="natural earth",
    )
    return _save_plotly(fig, output_path)


def plot_europe_concrete_event_timeline(
    europe_events: pl.DataFrame,
    output_path: Path,
    max_points: int = 1500,
) -> Path:
    """Save a Europe-focused timeline of concrete event mentions."""
    plot_df = (
        europe_events
        .sort("session_date")
        .head(max_points)
        .with_columns([
            pl.when(pl.col("event_label").fill_null("").str.len_chars() > 0)
            .then(pl.col("event_label"))
            .otherwise(pl.col("entity_content"))
            .alias("display_event_label"),
            pl.col("context_window")
            .map_elements(lambda text: " ".join(str(text).replace("||", " ").split())[:520], return_dtype=pl.Utf8)
            .alias("context_excerpt"),
        ])
        .to_pandas()
    )
    fig = px.scatter(
        plot_df,
        x="session_date",
        y="entity_content",
        size="fact_concreteness_score",
        color="narrative_frame",
        hover_name="display_event_label",
        hover_data={
            "entity_content": True,
            "proper_noun_anchors": True,
            "countries_detected_in_context": True,
            "policy_agency_type": True,
            "migrant_cohort": True,
            "policy_measure": True,
            "migration_direction": True,
            "context_excerpt": True,
            "fact_concreteness_score": ":.2f",
        },
        title="Europe Timeline: concrete migration facts and event references",
    )
    fig.update_layout(xaxis_title="Session date", yaxis_title="Mentioned European country/entity")
    return _save_plotly(fig, output_path)


def plot_europe_country_frame_heatmap(
    europe_events: pl.DataFrame,
    output_path: Path,
    top_n: int = 25,
) -> Path:
    """Save a country x narrative-frame heatmap for concrete European event mentions."""
    countries = (
        europe_events
        .group_by("entity_content")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(top_n)
        .get_column("entity_content")
        .to_list()
    )
    frames = (
        europe_events
        .filter(pl.col("entity_content").is_in(countries))
        .group_by("narrative_frame")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(18)
        .get_column("narrative_frame")
        .to_list()
    )
    plot_df = (
        europe_events
        .filter(pl.col("entity_content").is_in(countries) & pl.col("narrative_frame").is_in(frames))
        .group_by(["entity_content", "narrative_frame"])
        .agg([
            pl.len().alias("n_mentions"),
            pl.col("fact_concreteness_score").mean().round(3).alias("mean_fact_concreteness"),
            pl.col("event_label").drop_nulls().unique().str.join(", ").alias("event_labels"),
        ])
        .to_pandas()
    )
    base = alt.Chart(plot_df).encode(
        x=alt.X("narrative_frame:N", title="Narrative / event frame", sort=frames),
        y=alt.Y("entity_content:N", title="European country/entity", sort=countries),
        tooltip=[
            alt.Tooltip("entity_content:N", title="Entity"),
            alt.Tooltip("narrative_frame:N", title="Frame"),
            alt.Tooltip("n_mentions:Q", title="Mentions"),
            alt.Tooltip("mean_fact_concreteness:Q", title="Mean fact score"),
            alt.Tooltip("event_labels:N", title="Event labels"),
        ],
    )
    heatmap = base.mark_rect().encode(
        color=alt.Color("n_mentions:Q", title="Concrete mentions", scale=alt.Scale(scheme="reds"))
    )
    labels = base.mark_text(fontSize=9).encode(text=alt.Text("n_mentions:Q"), color=alt.value("#111111"))
    chart = (heatmap + labels).interactive().properties(
        title="Europe: what concrete migration topics are attached to which countries?",
        width=880,
        height=max(380, 22 * len(countries)),
    )
    return _save_chart(chart, output_path)


def plot_europe_concrete_map_by_year(
    europe_year_summary: pl.DataFrame,
    output_path: Path,
) -> Path:
    """Save an animated Europe map of concrete migration conversations by year."""
    plot_df = (
        europe_year_summary
        .drop_nulls("target_iso3")
        .filter(pl.col("target_iso3") != "")
        .sort(["source_year", "entity_content"])
        .to_pandas()
    )
    fig = px.choropleth(
        plot_df,
        locations="target_iso3",
        color="concrete_event_mentions",
        animation_frame="source_year",
        hover_name="entity_content",
        hover_data={
            "concrete_event_mentions": True,
            "dominant_frame": True,
            "dominant_agency": True,
            "sample_event_label": True,
            "target_iso3": False,
        },
        color_continuous_scale="YlOrRd",
        title="Europe Map by Year: concrete migration conversations",
    )
    fig.update_geos(scope="europe", showframe=False, showcoastlines=True, projection_type="natural earth")
    return _save_plotly(fig, output_path)


def plot_europe_country_year_event_heatmap(
    europe_events: pl.DataFrame,
    output_path: Path,
    top_n: int = 25,
) -> Path:
    """Save a country x year heatmap of concrete European event mentions."""
    countries = (
        europe_events
        .group_by("entity_content")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(top_n)
        .get_column("entity_content")
        .to_list()
    )
    plot_df = (
        europe_events
        .filter(pl.col("entity_content").is_in(countries))
        .group_by(["entity_content", "source_year"])
        .agg([
            pl.len().alias("n_mentions"),
            pl.col("narrative_frame").mode().first().alias("dominant_frame"),
            pl.col("event_label").drop_nulls().unique().str.join(", ").alias("event_labels"),
        ])
        .to_pandas()
    )
    base = alt.Chart(plot_df).encode(
        x=alt.X("source_year:O", title="Year"),
        y=alt.Y("entity_content:N", title="European country/entity", sort=countries),
        tooltip=[
            alt.Tooltip("entity_content:N", title="Entity"),
            alt.Tooltip("source_year:O", title="Year"),
            alt.Tooltip("n_mentions:Q", title="Concrete mentions"),
            alt.Tooltip("dominant_frame:N", title="Dominant frame"),
            alt.Tooltip("event_labels:N", title="Event labels"),
        ],
    )
    heatmap = base.mark_rect().encode(
        color=alt.Color("n_mentions:Q", title="Concrete mentions", scale=alt.Scale(scheme="orangered"))
    )
    labels = base.mark_text(fontSize=10).encode(text=alt.Text("n_mentions:Q"), color=alt.value("#111111"))
    chart = (heatmap + labels).interactive().properties(
        title="Europe: concrete migration event mentions by country and year",
        width=680,
        height=max(360, 24 * len(countries)),
    )
    return _save_chart(chart, output_path)


def plot_europe_concrete_sankey(
    europe_events: pl.DataFrame,
    output_path: Path,
    top_n: int = 12,
) -> Path:
    """Save a Sankey diagram: European country -> narrative frame -> agency."""
    countries = (
        europe_events
        .group_by("entity_content")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(top_n)
        .get_column("entity_content")
        .to_list()
    )
    working = europe_events.filter(pl.col("entity_content").is_in(countries))
    left = (
        working
        .group_by(["entity_content", "narrative_frame"])
        .agg(pl.len().alias("value"))
        .to_dicts()
    )
    right = (
        working
        .group_by(["narrative_frame", "policy_agency_type"])
        .agg(pl.len().alias("value"))
        .to_dicts()
    )
    labels = []
    for row in left:
        labels.extend([row["entity_content"], row["narrative_frame"]])
    for row in right:
        labels.extend([row["narrative_frame"], row["policy_agency_type"]])
    labels = list(dict.fromkeys(labels))
    index = {label: idx for idx, label in enumerate(labels)}
    sources = []
    targets = []
    values = []
    for row in left:
        sources.append(index[row["entity_content"]])
        targets.append(index[row["narrative_frame"]])
        values.append(row["value"])
    for row in right:
        sources.append(index[row["narrative_frame"]])
        targets.append(index[row["policy_agency_type"]])
        values.append(row["value"])

    fig = go.Figure(data=[go.Sankey(
        node={"pad": 14, "thickness": 16, "line": {"color": "#52606d", "width": 0.4}, "label": labels},
        link={"source": sources, "target": targets, "value": values},
    )])
    fig.update_layout(
        title_text="Europe Concrete Conversations: country -> narrative frame -> policy agency",
        font_size=11,
        height=720,
    )
    return _save_plotly(fig, output_path)


def plot_europe_concrete_treemap(
    europe_events: pl.DataFrame,
    output_path: Path,
) -> Path:
    """Save a treemap of concrete European conversations by country/frame/event label."""
    plot_df = (
        europe_events
        .with_columns(
            pl.when(pl.col("event_label").fill_null("").str.len_chars() > 0)
            .then(pl.col("event_label"))
            .otherwise(pl.col("entity_content"))
            .alias("display_event_label")
        )
        .group_by(["entity_content", "narrative_frame", "display_event_label"])
        .agg([
            pl.len().alias("n_mentions"),
            pl.col("policy_agency_type").mode().first().alias("dominant_agency"),
        ])
        .sort("n_mentions", descending=True)
        .head(180)
        .to_pandas()
    )
    fig = px.treemap(
        plot_df,
        path=["entity_content", "narrative_frame", "display_event_label"],
        values="n_mentions",
        color="dominant_agency",
        hover_data={"n_mentions": True, "dominant_agency": True},
        title="Europe Concrete Conversations Treemap: country, frame, and event labels",
    )
    return _save_plotly(fig, output_path)


def plot_geolocated_concrete_event_map(
    points: pl.DataFrame,
    output_path: Path,
    title: str = "Geolocated concrete migration event mentions",
) -> Path:
    """Save a clickable point map where each marker opens the full speech context."""
    import folium
    from folium.plugins import MarkerCluster

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if points.is_empty():
        output_path.write_text(
            f"<!doctype html><html><body><h1>{html.escape(title)}</h1><p>No geolocated points.</p></body></html>",
            encoding="utf-8",
        )
        return output_path

    plot_df = points.to_dicts()
    mean_lat = sum(float(row["latitude"]) for row in plot_df) / len(plot_df)
    mean_lon = sum(float(row["longitude"]) for row in plot_df) / len(plot_df)
    fmap = folium.Map(location=[mean_lat, mean_lon], zoom_start=4, tiles="CartoDB positron")
    cluster = MarkerCluster(name="Concrete event mentions").add_to(fmap)

    for row in plot_df:
        context = html.escape(str(row.get("context_window") or row.get("context_excerpt") or ""))
        popup = (
            "<div style='width:440px;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif'>"
            f"<h3 style='margin:0 0 6px 0'>{html.escape(str(row.get('place_name') or ''))}</h3>"
            f"<p><b>Date:</b> {html.escape(str(row.get('session_date') or ''))}<br>"
            f"<b>Entity:</b> {html.escape(str(row.get('entity_content') or ''))}<br>"
            f"<b>Event label:</b> {html.escape(str(row.get('event_label') or ''))}<br>"
            f"<b>Frame:</b> {html.escape(str(row.get('narrative_frame') or ''))}<br>"
            f"<b>Agency:</b> {html.escape(str(row.get('policy_agency_type') or ''))}<br>"
            f"<b>Direction:</b> {html.escape(str(row.get('migration_direction') or ''))}</p>"
            f"<p style='line-height:1.35'><b>Full context:</b><br>{context}</p>"
            "</div>"
        )
        tooltip = (
            f"{row.get('place_name')} | {row.get('session_date')} | "
            f"{row.get('entity_content')}"
        )
        folium.CircleMarker(
            location=[float(row["latitude"]), float(row["longitude"])],
            radius=max(4, min(12, 3 + float(row.get("point_weight") or 1))),
            color="#b42318",
            fill=True,
            fill_color="#f97316",
            fill_opacity=0.72,
            tooltip=tooltip,
            popup=folium.Popup(popup, max_width=480),
        ).add_to(cluster)

    title_html = (
        f"<div style='position: fixed; top: 12px; left: 52px; z-index: 9999; "
        "background: white; padding: 8px 12px; border: 1px solid #d9e2ec; "
        "border-radius: 6px; font-family: sans-serif; font-size: 15px;'>"
        f"<b>{html.escape(title)}</b><br>{len(plot_df)} geolocated mentions"
        "</div>"
    )
    fmap.get_root().html.add_child(folium.Element(title_html))
    folium.LayerControl().add_to(fmap)
    fmap.save(output_path)
    return output_path


def plot_geolocated_event_timeline(
    points: pl.DataFrame,
    output_path: Path,
    max_points: int = 1500,
) -> Path:
    """Save a timeline of geolocated concrete events by specific place."""
    plot_df = (
        points
        .sort("session_date")
        .head(max_points)
        .with_columns(
            pl.col("context_window")
            .map_elements(lambda text: " ".join(str(text).replace("||", " ").split())[:520], return_dtype=pl.Utf8)
            .alias("context_excerpt")
        )
        .to_pandas()
    )
    fig = px.scatter(
        plot_df,
        x="session_date",
        y="place_name",
        size="point_weight",
        color="narrative_frame",
        hover_name="event_label",
        hover_data={
            "entity_content": True,
            "place_kind": True,
            "policy_agency_type": True,
            "migration_direction": True,
            "context_excerpt": True,
        },
        title="Timeline of geolocated concrete migration event mentions",
    )
    fig.update_layout(xaxis_title="Session date", yaxis_title="Specific city / region / border")
    return _save_plotly(fig, output_path)


def save_concrete_conversations_explorer(
    points: pl.DataFrame,
    output_path: Path,
    title: str = "Concrete Migration Conversations Explorer",
) -> Path:
    """Save a polished interactive explorer for geolocated concrete event mentions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, row in enumerate(points.sort(["session_date", "place_name"]).to_dicts()):
        rows.append({
            "id": idx,
            "date": str(row.get("session_date") or ""),
            "year": int(row.get("source_year") or 0),
            "entity": str(row.get("entity_content") or ""),
            "place": str(row.get("place_name") or ""),
            "placeCountry": str(row.get("place_country") or ""),
            "placeKind": str(row.get("place_kind") or ""),
            "lat": float(row.get("latitude") or 0),
            "lon": float(row.get("longitude") or 0),
            "event": str(row.get("event_label") or ""),
            "frame": str(row.get("narrative_frame") or ""),
            "agency": str(row.get("policy_agency_type") or ""),
            "direction": str(row.get("migration_direction") or ""),
            "cohort": str(row.get("migrant_cohort") or ""),
            "policy": str(row.get("policy_measure") or ""),
            "context": str(row.get("context_window") or ""),
            "sentence": str(row.get("sentence_id") or ""),
        })

    years = sorted({row["year"] for row in rows if row["year"]})
    entities = sorted({row["entity"] for row in rows if row["entity"]})
    places = sorted({row["place"] for row in rows if row["place"]})
    frames = sorted({row["frame"] for row in rows if row["frame"]})
    agencies = sorted({row["agency"] for row in rows if row["agency"]})
    data_json = json.dumps(rows, ensure_ascii=False)

    def options(values: list[object]) -> str:
        return "".join(f"<option value='{html.escape(str(value))}'>{html.escape(str(value))}</option>" for value in values)

    document = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    :root {{
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #607080;
      --line: #d8e0e8;
      --accent: #b42318;
      --blue: #275f8f;
      --green: #287d67;
      --orange: #c65f1a;
      --purple: #7353ba;
      --gray: #7a8694;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    header {{
      height: 58px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }}
    header h1 {{ font-size: 18px; margin: 0; font-weight: 680; }}
    .stats {{ display: flex; gap: 12px; align-items: center; }}
    .stat {{ min-width: 82px; }}
    .stat strong {{ display: block; font-size: 18px; line-height: 1.1; }}
    .stat span {{ color: var(--muted); font-size: 11px; }}
    .app {{
      height: calc(100vh - 58px);
      display: grid;
      grid-template-columns: 300px minmax(520px, 1fr) 390px;
      grid-template-rows: minmax(360px, 1fr) 230px;
      gap: 10px;
      padding: 10px;
    }}
    aside, .map-panel, .timeline-panel, .detail-panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    aside {{ grid-row: 1 / span 2; display: flex; flex-direction: column; }}
    .filters {{ padding: 12px; border-bottom: 1px solid var(--line); }}
    label {{ display: block; color: var(--muted); font-size: 11px; margin: 10px 0 4px; }}
    select, input {{
      width: 100%;
      height: 34px;
      border: 1px solid #c8d2dc;
      border-radius: 6px;
      background: #fff;
      padding: 0 8px;
      color: var(--ink);
      font-size: 13px;
    }}
    .event-list {{ overflow: auto; padding: 8px; }}
    .event-row {{
      width: 100%;
      text-align: left;
      border: 1px solid transparent;
      background: #fff;
      border-radius: 7px;
      padding: 9px;
      margin-bottom: 6px;
      cursor: pointer;
    }}
    .event-row:hover, .event-row.active {{ border-color: #8bb4d8; background: #eef6fd; }}
    .event-row .top {{ display: flex; justify-content: space-between; gap: 8px; font-size: 12px; color: var(--muted); }}
    .event-row .place {{ font-size: 14px; font-weight: 650; margin: 4px 0 2px; }}
    .event-row .event {{ font-size: 12px; color: #334e68; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .map-panel {{ grid-column: 2; grid-row: 1; position: relative; }}
    #map {{ height: 100%; width: 100%; }}
    .legend {{
      position: absolute;
      right: 12px;
      bottom: 12px;
      z-index: 600;
      background: rgba(255,255,255,.95);
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 8px 10px;
      font-size: 12px;
      box-shadow: 0 1px 6px rgba(15,23,42,.12);
    }}
    .legend div {{ margin: 3px 0; }}
    .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }}
    .detail-panel {{ grid-column: 3; grid-row: 1 / span 2; display: flex; flex-direction: column; }}
    .detail-head {{ padding: 14px; border-bottom: 1px solid var(--line); }}
    .detail-head h2 {{ margin: 0 0 5px; font-size: 18px; }}
    .detail-head p {{ margin: 0; color: var(--muted); font-size: 12px; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 6px; padding: 10px 14px; border-bottom: 1px solid var(--line); }}
    .chip {{ border: 1px solid #c8d2dc; border-radius: 999px; padding: 4px 8px; font-size: 12px; background: #f8fafc; }}
    .context {{ padding: 14px; overflow: auto; line-height: 1.48; font-size: 14px; }}
    .context h3 {{ margin: 0 0 8px; font-size: 13px; color: var(--muted); text-transform: uppercase; }}
    .timeline-panel {{ grid-column: 2; grid-row: 2; padding: 10px 12px; }}
    .timeline-title {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
    .timeline-title h2 {{ margin: 0; font-size: 15px; }}
    .timeline-title span {{ color: var(--muted); font-size: 12px; }}
    #timeline {{ width: 100%; height: 178px; }}
    .bar {{ fill: #9fc5e8; }}
    .bar:hover {{ fill: #477da8; }}
    .tick-label {{ fill: var(--muted); font-size: 11px; }}
    .marker {{ cursor: pointer; }}
    .leaflet-popup-content {{ margin: 10px 12px; }}
    .leaflet-popup-content h3 {{ margin: 0 0 5px; font-size: 15px; }}
    .leaflet-popup-content p {{ margin: 4px 0; }}
    @media (max-width: 1100px) {{
      .app {{ grid-template-columns: 280px 1fr; grid-template-rows: 400px 220px 420px; height: auto; min-height: calc(100vh - 58px); }}
      .detail-panel {{ grid-column: 1 / span 2; grid-row: 3; }}
      aside {{ grid-row: 1 / span 2; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="stats">
      <div class="stat"><strong id="statEvents">0</strong><span>visible mentions</span></div>
      <div class="stat"><strong id="statPlaces">0</strong><span>places</span></div>
      <div class="stat"><strong id="statCountries">0</strong><span>entities</span></div>
    </div>
  </header>
  <main class="app">
    <aside>
      <section class="filters">
        <label for="search">Search context</label>
        <input id="search" type="search" placeholder="Calais, border, asylum">
        <label for="year">Year</label>
        <select id="year"><option value="">All years</option>{options(years)}</select>
        <label for="entity">Mentioned country/entity</label>
        <select id="entity"><option value="">All entities</option>{options(entities)}</select>
        <label for="place">Specific place</label>
        <select id="place"><option value="">All places</option>{options(places)}</select>
        <label for="frame">Narrative frame</label>
        <select id="frame"><option value="">All frames</option>{options(frames)}</select>
        <label for="agency">Policy agency</label>
        <select id="agency"><option value="">All agency types</option>{options(agencies)}</select>
      </section>
      <section id="eventList" class="event-list"></section>
    </aside>
    <section class="map-panel">
      <div id="map"></div>
      <div class="legend" id="legend"></div>
    </section>
    <section class="timeline-panel">
      <div class="timeline-title"><h2>Timeline</h2><span id="timelineSubtitle"></span></div>
      <svg id="timeline" role="img"></svg>
    </section>
    <section class="detail-panel">
      <div class="detail-head">
        <h2 id="detailTitle">Select a point or event</h2>
        <p id="detailMeta">Map points are concrete migration references geolocated to a city, region, border, island, or route.</p>
      </div>
      <div class="chips" id="detailChips"></div>
      <div class="context">
        <h3>Full context</h3>
        <div id="detailContext">No event selected.</div>
      </div>
    </section>
  </main>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const DATA = {data_json};
    const palette = {{
      border_crisis: '#b42318',
      national_identity: '#7c3aed',
      neutral_reporting: '#667085',
      humanitarian_obligation: '#287d67',
      asylum_right: '#1d70b8',
      security_threat: '#c65f1a',
      war_conflict: '#8b5e34',
      international_law: '#3b7c9f'
    }};
    const fallbackColors = ['#b42318','#1d70b8','#287d67','#c65f1a','#7c3aed','#667085','#8b5e34'];
    const colorFor = frame => palette[frame] || fallbackColors[Math.abs(hash(frame)) % fallbackColors.length];
    function hash(text) {{ let h = 0; for (let i=0; i<text.length; i++) h = ((h << 5) - h) + text.charCodeAt(i); return h; }}
    const state = {{ selectedId: DATA[0]?.id ?? null, filtered: DATA.slice(), markers: new Map() }};

    const map = L.map('map', {{ zoomControl: true }}).setView([48.5, 8.5], 4);
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 18,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);
    const markerLayer = L.layerGroup().addTo(map);

    const controls = ['search','year','entity','place','frame','agency'].reduce((acc, id) => {{
      acc[id] = document.getElementById(id);
      acc[id].addEventListener(id === 'search' ? 'input' : 'change', render);
      return acc;
    }}, {{}});

    function currentFilters() {{
      return {{
        q: controls.search.value.trim().toLowerCase(),
        year: controls.year.value,
        entity: controls.entity.value,
        place: controls.place.value,
        frame: controls.frame.value,
        agency: controls.agency.value
      }};
    }}
    function applyFilters() {{
      const f = currentFilters();
      return DATA.filter(d => {{
        if (f.year && String(d.year) !== f.year) return false;
        if (f.entity && d.entity !== f.entity) return false;
        if (f.place && d.place !== f.place) return false;
        if (f.frame && d.frame !== f.frame) return false;
        if (f.agency && d.agency !== f.agency) return false;
        if (f.q) {{
          const hay = [d.place,d.entity,d.event,d.context,d.frame,d.agency,d.direction,d.policy,d.cohort].join(' ').toLowerCase();
          if (!hay.includes(f.q)) return false;
        }}
        return true;
      }});
    }}
    function escapeHtml(text) {{
      return String(text ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}}[ch]));
    }}
    function compact(text, n=120) {{ text = String(text || ''); return text.length > n ? text.slice(0, n - 1) + '...' : text; }}

    function renderStats(rows) {{
      document.getElementById('statEvents').textContent = rows.length;
      document.getElementById('statPlaces').textContent = new Set(rows.map(d => d.place)).size;
      document.getElementById('statCountries').textContent = new Set(rows.map(d => d.entity)).size;
    }}
    function renderLegend(rows) {{
      const frames = [...new Set(rows.map(d => d.frame).filter(Boolean))].slice(0, 8);
      document.getElementById('legend').innerHTML = frames.map(frame =>
        `<div><span class="dot" style="background:${{colorFor(frame)}}"></span>${{escapeHtml(frame)}}</div>`
      ).join('');
    }}
    function renderMap(rows) {{
      markerLayer.clearLayers();
      state.markers.clear();
      rows.forEach(d => {{
        const marker = L.circleMarker([d.lat, d.lon], {{
          radius: state.selectedId === d.id ? 10 : 7,
          color: '#293241',
          weight: state.selectedId === d.id ? 2 : 1,
          fillColor: colorFor(d.frame),
          fillOpacity: 0.82,
          className: 'marker'
        }});
        marker.bindTooltip(`${{d.place}} | ${{d.date}}`);
        marker.bindPopup(`<h3>${{escapeHtml(d.place)}}</h3><p><b>${{escapeHtml(d.date)}}</b><br>${{escapeHtml(d.entity)}}<br>${{escapeHtml(d.frame)}}</p>`);
        marker.on('click', () => selectEvent(d.id, true));
        marker.addTo(markerLayer);
        state.markers.set(d.id, marker);
      }});
      if (rows.length) {{
        const bounds = L.latLngBounds(rows.map(d => [d.lat, d.lon]));
        map.fitBounds(bounds.pad(0.18), {{ animate: false }});
      }}
    }}
    function renderList(rows) {{
      const list = document.getElementById('eventList');
      list.innerHTML = rows.map(d => `
        <button class="event-row ${{state.selectedId === d.id ? 'active' : ''}}" data-id="${{d.id}}">
          <div class="top"><span>${{escapeHtml(d.date)}}</span><span>${{escapeHtml(d.entity)}}</span></div>
          <div class="place">${{escapeHtml(d.place)}}</div>
          <div class="event">${{escapeHtml(compact(d.event || d.context, 125))}}</div>
        </button>
      `).join('');
      list.querySelectorAll('.event-row').forEach(btn => {{
        btn.addEventListener('click', () => selectEvent(Number(btn.dataset.id), true));
      }});
    }}
    function renderTimeline(rows) {{
      const svg = document.getElementById('timeline');
      const width = svg.clientWidth || 700;
      const height = 178;
      const margin = {{left: 34, right: 18, top: 10, bottom: 24}};
      const counts = new Map();
      rows.forEach(d => counts.set(d.year, (counts.get(d.year) || 0) + 1));
      const years = [...new Set(DATA.map(d => d.year))].sort();
      const maxCount = Math.max(1, ...counts.values());
      const barW = (width - margin.left - margin.right) / Math.max(1, years.length);
      const bars = years.map((year, i) => {{
        const count = counts.get(year) || 0;
        const barH = (height - margin.top - margin.bottom) * count / maxCount;
        const x = margin.left + i * barW + 4;
        const y = height - margin.bottom - barH;
        return `<g data-year="${{year}}" class="year-bin">
          <rect class="bar" x="${{x}}" y="${{y}}" width="${{Math.max(6, barW-8)}}" height="${{barH}}" rx="3"></rect>
          <text class="tick-label" x="${{x + barW/2 - 4}}" y="${{height - 6}}" text-anchor="middle">${{year}}</text>
          <text class="tick-label" x="${{x + barW/2 - 4}}" y="${{Math.max(14, y - 5)}}" text-anchor="middle">${{count || ''}}</text>
        </g>`;
      }}).join('');
      svg.setAttribute('viewBox', `0 0 ${{width}} ${{height}}`);
      svg.innerHTML = bars;
      svg.querySelectorAll('.year-bin').forEach(g => {{
        g.addEventListener('click', () => {{
          controls.year.value = g.dataset.year;
          render();
        }});
      }});
      document.getElementById('timelineSubtitle').textContent = `${{rows.length}} filtered mentions`;
    }}
    function selectEvent(id, pan=false) {{
      state.selectedId = id;
      const d = DATA.find(item => item.id === id);
      if (!d) return;
      document.getElementById('detailTitle').textContent = d.place;
      document.getElementById('detailMeta').textContent = `${{d.date}} | ${{d.entity}} | ${{d.placeKind}}`;
      document.getElementById('detailChips').innerHTML = [d.frame,d.agency,d.direction,d.cohort,d.policy]
        .filter(Boolean).map(v => `<span class="chip">${{escapeHtml(v)}}</span>`).join('');
      document.getElementById('detailContext').textContent = d.context || 'No context.';
      if (pan && state.markers.has(id)) {{
        const marker = state.markers.get(id);
        map.setView(marker.getLatLng(), Math.max(map.getZoom(), 7));
        marker.openPopup();
      }}
      renderList(state.filtered);
      renderMap(state.filtered);
    }}
    function render() {{
      const rows = applyFilters();
      state.filtered = rows;
      if (!rows.some(d => d.id === state.selectedId)) state.selectedId = rows[0]?.id ?? null;
      renderStats(rows);
      renderLegend(rows);
      renderMap(rows);
      renderList(rows);
      renderTimeline(rows);
      if (state.selectedId !== null) selectEvent(state.selectedId, false);
    }}
    render();
  </script>
</body>
</html>"""
    output_path.write_text(document, encoding="utf-8")
    return output_path


def save_concrete_event_mentions_table(
    events: pl.DataFrame,
    output_path: Path,
    title: str,
) -> Path:
    """Save an HTML table with concrete event mentions and context excerpts."""
    rows = (
        events
        .sort(["session_date", "entity_content"])
        .with_columns([
            pl.when(pl.col("event_label").fill_null("").str.len_chars() > 0)
            .then(pl.col("event_label"))
            .otherwise(pl.col("entity_content"))
            .alias("display_event_label"),
            pl.col("context_window")
            .map_elements(lambda text: " ".join(str(text).replace("||", " ").split())[:700], return_dtype=pl.Utf8)
            .alias("context_excerpt"),
        ])
        .select([
            "session_date",
            "entity_content",
            "display_event_label",
            "proper_noun_anchors",
            "countries_detected_in_context",
            "narrative_frame",
            "policy_agency_type",
            "migration_direction",
            "fact_concreteness_score",
            "context_excerpt",
        ])
        .to_dicts()
    )
    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['session_date']))}</td>"
            f"<td>{html.escape(str(row['entity_content']))}</td>"
            f"<td>{html.escape(str(row['display_event_label']))}</td>"
            f"<td>{html.escape(str(row['proper_noun_anchors'] or ''))}</td>"
            f"<td>{html.escape(str(row['countries_detected_in_context'] or ''))}</td>"
            f"<td>{html.escape(str(row['narrative_frame']))}</td>"
            f"<td>{html.escape(str(row['policy_agency_type']))}</td>"
            f"<td>{html.escape(str(row['migration_direction']))}</td>"
            f"<td>{float(row['fact_concreteness_score']):.2f}</td>"
            f"<td class='excerpt'>{html.escape(str(row['context_excerpt']))}</td>"
            "</tr>"
        )
    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;background:#f8f9fa;color:#1f2933;}"
        "h1{font-size:24px;margin:0 0 8px 0;} .sub{color:#52606d;margin-bottom:16px;}"
        "table{border-collapse:collapse;width:100%;background:#fff;font-size:13px;}"
        "th,td{border:1px solid #d9e2ec;padding:7px;vertical-align:top;} th{position:sticky;top:0;background:#e5e9f0;text-align:left;}"
        ".excerpt{min-width:420px;line-height:1.35;}"
        "</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f"<p class='sub'>{len(rows)} concrete event mentions. Use browser search to find countries, events, frames, or dates.</p>"
        "<table><thead><tr>"
        "<th>Date</th><th>Entity</th><th>Event label</th><th>Proper nouns</th><th>Countries in context</th>"
        "<th>Frame</th><th>Agency</th><th>Direction</th><th>Fact score</th><th>What was said</th>"
        "</tr></thead><tbody>"
        f"{''.join(table_rows)}"
        "</tbody></table></body></html>"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path


def plot_direction_agenda_split_bars(
    df: pl.DataFrame,
    output_path: Path,
    group_col: str = "entity_content",
    top_n: int = 20,
) -> Path:
    """Save split bars for internal/inbound versus external/transnational agenda."""
    groups = (
        df
        .drop_nulls(group_col)
        .group_by(group_col)
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(top_n)
        .get_column(group_col)
        .to_list()
    )
    plot_df = (
        df
        .filter(pl.col(group_col).is_in(groups))
        .with_columns(
            pl.when(pl.col("migration_direction") == "inbound_internal")
            .then(pl.lit("Internal / inbound"))
            .when(pl.col("migration_direction") == "outbound_from_domestic")
            .then(pl.lit("Domestic outbound"))
            .when(pl.col("migration_direction") == "external_transnational")
            .then(pl.lit("External / transnational"))
            .otherwise(pl.lit("Ambiguous"))
            .alias("agenda_direction")
        )
        .group_by([group_col, "agenda_direction"])
        .agg(pl.len().alias("n"))
        .to_pandas()
    )
    direction_order = [
        "Internal / inbound",
        "Domestic outbound",
        "External / transnational",
        "Ambiguous",
    ]
    index = pd.MultiIndex.from_product([groups, direction_order], names=[group_col, "agenda_direction"])
    plot_df = (
        plot_df
        .set_index([group_col, "agenda_direction"])
        .reindex(index, fill_value=0)
        .reset_index()
    )
    totals = plot_df.groupby(group_col)["n"].transform("sum")
    plot_df["share"] = (plot_df["n"] / totals).fillna(0)
    plot_df["direction_order"] = plot_df["agenda_direction"].map(
        {direction: idx for idx, direction in enumerate(direction_order)}
    )
    if group_col == "entity_content":
        plot_df = _add_display_columns(plot_df, df, groups)
        y_col = "display_entity"
        y_title = "Mentioned country/entity"
        y_order = [entity_display_labels(df, groups)[entity] for entity in groups]
    else:
        y_col = group_col
        y_title = group_col.replace("_", " ").title()
        y_order = groups

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("n:Q", title="Share of migration-direction agenda", stack="normalize"),
            y=alt.Y(f"{y_col}:N", title=y_title, sort=y_order),
            color=alt.Color(
                "agenda_direction:N",
                title="Direction",
                scale=alt.Scale(
                    domain=direction_order,
                    range=["#2a9d8f", "#8ab17d", "#457b9d", "#8d99ae"],
                ),
            ),
            order=alt.Order("direction_order:Q"),
            tooltip=[
                alt.Tooltip(f"{y_col}:N", title=y_title),
                alt.Tooltip("agenda_direction:N", title="Direction"),
                alt.Tooltip("n:Q", title="Mentions"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        )
        .interactive()
        .properties(
            title="Internal vs external migration agenda split",
            width=780,
            height=max(340, 26 * len(groups)),
        )
    )
    return _save_chart(chart, output_path)


def save_advanced_figures(
    df: pl.DataFrame,
    agency_edges: pl.DataFrame,
    events: pl.DataFrame,
    visible_summary: pl.DataFrame,
    processed_dir: Path,
) -> dict[str, Path]:
    """Save the deepest interactive figures for policy agency, narratives, and facts."""
    figures_dir = processed_dir / "figures_interactive_advanced"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return {
        "policy_agency_network": plot_policy_agency_network(
            agency_edges,
            figures_dir / "policy_agency_network.png",
        ),
        "policy_agency_country_heatmap": plot_policy_agency_country_heatmap(
            agency_edges,
            figures_dir / "policy_agency_country_heatmap.png",
        ),
        "narrative_ternary": plot_narrative_ternary(
            df,
            figures_dir / "narrative_ternary.html",
        ),
        "narrative_mirror_bars": plot_narrative_mirror_bars(
            df,
            figures_dir / "narrative_mirror_bars.png",
        ),
        "evidence_visibility_map": plot_evidence_visibility_map(
            visible_summary,
            figures_dir / "evidence_visibility_map.html",
        ),
        "fact_density_timeline": plot_fact_density_timeline(
            events,
            figures_dir / "fact_density_timeline.html",
        ),
        "direction_agenda_split_bars": plot_direction_agenda_split_bars(
            df,
            figures_dir / "direction_agenda_split_bars.png",
        ),
    }


def _top_targets_from_edges(edges: pl.DataFrame, top_n: int = 25) -> list[str]:
    """Return top target entities from a diffusion edge table."""
    # Explanation: Shared helper keeps country heatmaps focused and comparable.
    return (
        edges
        .group_by("target_entity")
        .agg(pl.col("weight").sum().alias("total_mentions"))
        .sort("total_mentions", descending=True)
        .head(top_n)
        .get_column("target_entity")
        .to_list()
    )


def plot_country_concreteness_bubble(
    df: pl.DataFrame,
    output_path: Path,
    min_mentions: int = 10,
) -> Path:
    """Save an interactive country bubble chart for concreteness and volume."""
    # Explanation: This chart puts "which countries are mentioned how" at the center.
    plot_df = (
        df
        .drop_nulls("concreteness_score")
        .group_by(["entity_content", "target_iso3", "weog_group", "region_group", "geo_class"])
        .agg([
            pl.len().alias("n_mentions"),
            pl.col("concreteness_score").mean().round(3).alias("mean_concreteness"),
            pl.col("ref_type").mode().first().alias("dominant_ref_type"),
            pl.col("migrant_cohort").mode().first().alias("dominant_cohort"),
            pl.col("policy_measure").mode().first().alias("dominant_policy_measure"),
            pl.col("sentiment_readable").mode().first().alias("dominant_sentiment"),
        ])
        .filter(pl.col("n_mentions") >= min_mentions)
        .sort("n_mentions", descending=True)
        .to_pandas()
    )
    plot_df["weog_group_readable"] = plot_df["weog_group"].map(WEOG_GROUP_LABELS).fillna(plot_df["weog_group"])

    selector = alt.selection_point(fields=["weog_group_readable"], bind="legend")
    chart = (
        alt.Chart(plot_df)
        .mark_circle(stroke="#333333", strokeWidth=0.5)
        .encode(
            x=alt.X("mean_concreteness:Q", title="Mean concreteness (1 abstract - 5 concrete)", scale=alt.Scale(zero=False)),
            y=alt.Y("n_mentions:Q", title="Number of mentions", scale=alt.Scale(type="sqrt")),
            size=alt.Size("n_mentions:Q", title="Mentions", scale=alt.Scale(range=[80, 1600])),
            color=alt.Color("weog_group_readable:N", title="Country/entity group"),
            opacity=alt.condition(selector, alt.value(0.9), alt.value(0.18)),
            tooltip=[
                alt.Tooltip("entity_content:N", title="Mentioned country/entity"),
                alt.Tooltip("target_iso3:N", title="ISO3"),
                alt.Tooltip("weog_group_readable:N", title="Group"),
                alt.Tooltip("n_mentions:Q", title="Mentions"),
                alt.Tooltip("mean_concreteness:Q", title="Mean concreteness"),
                alt.Tooltip("dominant_ref_type:N", title="Dominant reference"),
                alt.Tooltip("dominant_cohort:N", title="Dominant cohort"),
                alt.Tooltip("dominant_policy_measure:N", title="Dominant policy"),
                alt.Tooltip("dominant_sentiment:N", title="Dominant sentiment"),
            ],
        )
        .add_params(selector)
        .interactive()
        .properties(
            title="How other countries/entities are mentioned: volume x concreteness",
            width=820,
            height=460,
        )
    )
    return _save_chart(chart, output_path)


def plot_country_year_concreteness_heatmap(
    df: pl.DataFrame,
    output_path: Path,
    top_n: int = 25,
) -> Path:
    """Save an interactive country x year heatmap of mean concreteness."""
    # Explanation: This shows whether country references become more concrete/abstract over time.
    entities = top_entities(df, top_n=top_n)
    plot_df = (
        df
        .drop_nulls("concreteness_score")
        .filter(pl.col("entity_content").is_in(entities))
        .group_by(["entity_content", "source_year", "weog_group"])
        .agg([
            pl.len().alias("n_mentions"),
            pl.col("concreteness_score").mean().round(3).alias("mean_concreteness"),
            pl.col("ref_type").mode().first().alias("dominant_ref_type"),
        ])
        .sort(["entity_content", "source_year"])
        .to_pandas()
    )
    entity_order = entities
    base = alt.Chart(plot_df).encode(
        x=alt.X("source_year:O", title="Year"),
        y=alt.Y("entity_content:N", title="Mentioned country/entity", sort=entity_order),
        tooltip=[
            alt.Tooltip("entity_content:N", title="Entity"),
            alt.Tooltip("source_year:O", title="Year"),
            alt.Tooltip("n_mentions:Q", title="Mentions"),
            alt.Tooltip("mean_concreteness:Q", title="Mean concreteness"),
            alt.Tooltip("dominant_ref_type:N", title="Dominant reference"),
        ],
    )
    heatmap = base.mark_rect().encode(
        color=alt.Color(
            "mean_concreteness:Q",
            title="Mean concreteness",
            scale=alt.Scale(scheme="viridis", domain=[2.8, 3.6]),
        )
    )
    labels = base.mark_text(fontSize=9).encode(
        text=alt.Text("mean_concreteness:Q", format=".2f"),
        color=alt.value("#111111"),
    )
    chart = (heatmap + labels).interactive().properties(
        title="Country/entity x year concreteness heatmap",
        width=640,
        height=max(360, 22 * len(entities)),
    )
    return _save_chart(chart, output_path)


def plot_country_cohort_heatmap(
    edges: pl.DataFrame,
    output_path: Path,
    top_n: int = 25,
) -> Path:
    """Save target country/entity x migrant cohort heatmap."""
    # Explanation: This answers which countries are invoked for which migrant groups.
    targets = _top_targets_from_edges(edges, top_n=top_n)
    plot_df = (
        edges
        .filter(pl.col("target_entity").is_in(targets))
        .group_by(["target_entity", "migrant_cohort"])
        .agg(pl.col("weight").sum().alias("weight"))
        .to_pandas()
    )
    base = alt.Chart(plot_df).encode(
        x=alt.X("migrant_cohort:N", title="Migrant cohort"),
        y=alt.Y("target_entity:N", title="Mentioned country/entity", sort=targets),
        tooltip=[
            alt.Tooltip("target_entity:N", title="Target"),
            alt.Tooltip("migrant_cohort:N", title="Cohort"),
            alt.Tooltip("weight:Q", title="Mentions"),
        ],
    )
    heatmap = base.mark_rect().encode(color=alt.Color("weight:Q", title="Mentions", scale=alt.Scale(scheme="blues")))
    labels = base.mark_text(fontSize=9).encode(text=alt.Text("weight:Q"), color=alt.value("#111111"))
    chart = (heatmap + labels).interactive().properties(
        title="Which countries/entities are mentioned for which migrant cohorts?",
        width=700,
        height=max(360, 22 * len(targets)),
    )
    return _save_chart(chart, output_path)


def plot_country_policy_heatmap(
    edges: pl.DataFrame,
    output_path: Path,
    top_n: int = 25,
) -> Path:
    """Save target country/entity x policy measure heatmap."""
    # Explanation: This answers which countries are invoked for which policy instruments.
    targets = _top_targets_from_edges(edges, top_n=top_n)
    plot_df = (
        edges
        .filter(pl.col("target_entity").is_in(targets))
        .group_by(["target_entity", "policy_measure"])
        .agg(pl.col("weight").sum().alias("weight"))
        .to_pandas()
    )
    base = alt.Chart(plot_df).encode(
        x=alt.X("policy_measure:N", title="Policy measure"),
        y=alt.Y("target_entity:N", title="Mentioned country/entity", sort=targets),
        tooltip=[
            alt.Tooltip("target_entity:N", title="Target"),
            alt.Tooltip("policy_measure:N", title="Policy measure"),
            alt.Tooltip("weight:Q", title="Mentions"),
        ],
    )
    heatmap = base.mark_rect().encode(color=alt.Color("weight:Q", title="Mentions", scale=alt.Scale(scheme="oranges")))
    labels = base.mark_text(fontSize=9).encode(text=alt.Text("weight:Q"), color=alt.value("#111111"))
    chart = (heatmap + labels).interactive().properties(
        title="Which countries/entities are mentioned for which policy measures?",
        width=820,
        height=max(360, 22 * len(targets)),
    )
    return _save_chart(chart, output_path)
