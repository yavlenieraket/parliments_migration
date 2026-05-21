"""Visualization helpers for the France 2018 migration pilot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns


# Explanation: Keep sentiment colors stable across every figure.
SENTIMENT_COLORS = {
    "positive": "#1d9e75",
    "negative": "#a32d2d",
    "neutral": "#888780",
}

# Explanation: Reference-type colors separate policy talk from situation/context talk.
REF_TYPE_COLORS = {
    "policy": "#31688e",
    "situation": "#de8f05",
    "mixed": "#6a3d9a",
    "neutral_reference": "#9a9a9a",
    "unknown": "#d0d0d0",
}

# Explanation: Fixed category orders make comparisons stable across notebook reruns.
SENTIMENT_ORDER = ["positive", "negative", "neutral"]
REF_TYPE_ORDER = ["policy", "situation", "mixed", "neutral_reference", "unknown"]


def ensure_figures_dir(processed_dir: Path) -> Path:
    """Create and return the directory where plot images are saved."""
    # Explanation: All generated figures live beside the processed parquet output.
    figures_dir = processed_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def top_entities(df: pl.DataFrame, top_n: int = 15) -> list[str]:
    """Return the most frequently mentioned entities."""
    # Explanation: Visualizations use the same top-country universe for readability.
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
    """Return labels that mark French overseas territories explicitly."""
    # Explanation: Overseas territories are politically French, not foreign states.
    geo_lookup = {
        row["entity_content"]: row["geo_class"]
        for row in (
            df
            .filter(pl.col("entity_content").is_in(entities))
            .select(["entity_content", "geo_class"])
            .unique()
            .to_dicts()
        )
    }
    return {
        entity: (
            f"{entity} (French overseas)"
            if geo_lookup.get(entity) == "french_overseas"
            else entity
        )
        for entity in entities
    }


def entity_distribution_table(df: pl.DataFrame, min_mentions: int = 1) -> pl.DataFrame:
    """Return the complete distribution of mentioned entities."""
    # Explanation: This table is the source for the distribution chart and CSV export.
    return (
        df
        .group_by(["entity_content", "geo_class"])
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
    min_mentions: int = 50,
) -> Path:
    """Save the distribution for entities that meet the minimum mention threshold."""
    # Explanation: This matches the displayed "everything above threshold" chart.
    entity_distribution_table(df, min_mentions=min_mentions).write_csv(output_path)
    return output_path


def _complete_count_table(
    df: pl.DataFrame,
    category_column: str,
    categories: list[str],
    entities: list[str],
) -> pd.DataFrame:
    """Build a complete entity x category count table with zero-filled missing cells."""
    # Explanation: Polars gives only observed combinations; plotting needs zero columns too.
    counts = (
        df
        .filter(pl.col("entity_content").is_in(entities))
        .group_by(["entity_content", category_column])
        .agg(pl.len().alias("n"))
        .to_pandas()
    )

    # Explanation: A MultiIndex gives every country/category combination explicitly.
    index = pd.MultiIndex.from_product(
        [entities, categories],
        names=["entity_content", category_column],
    )
    counts = counts.set_index(["entity_content", category_column]).reindex(index, fill_value=0)

    # Explanation: The pivoted table is the direct input for stacked horizontal bars.
    table = counts["n"].unstack(category_column).reset_index()
    return table[["entity_content", *categories]]


def plot_entity_distribution(
    df: pl.DataFrame,
    output_path: Path,
    top_n: int | None = 40,
    min_mentions: int = 1,
) -> Path:
    """Plot total migration mentions by country/territory entity."""
    # Explanation: min_mentions removes low-frequency entities from the displayed chart.
    distribution = entity_distribution_table(df, min_mentions=min_mentions)
    if top_n is not None:
        distribution = distribution.head(top_n)

    entities = distribution.get_column("entity_content").to_list()
    labels = entity_display_labels(df, entities)
    plot_df = distribution.to_pandas()
    plot_df["display_entity"] = plot_df["entity_content"].map(labels)
    plot_df["bar_color"] = plot_df["geo_class"].map({
        "foreign": "#4c78a8",
        "french_overseas": "#f58518",
    })

    # Explanation: The figure height grows with the number of entities in the chart.
    fig_height = max(7, len(plot_df) * 0.22)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.barh(
        plot_df["display_entity"],
        plot_df["n_mentions"],
        color=plot_df["bar_color"],
    )
    ax.invert_yaxis()
    ax.set_title("Distribution of mentioned entities in France 2018 migration debates")
    ax.set_xlabel("Number of mentions")
    ax.set_ylabel("Mentioned entity")

    # Explanation: The legend clarifies that overseas territories are not foreign states.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#4c78a8", label="Foreign state/entity"),
        plt.Rectangle((0, 0), 1, 1, color="#f58518", label="French overseas territory"),
    ]
    ax.legend(handles=handles, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_country_sentiment_mentions(
    df: pl.DataFrame,
    output_path: Path,
    top_n: int = 15,
) -> Path:
    """Plot top countries by positive/negative/neutral migration mentions."""
    # Explanation: Pick the countries before counting sentiment, so all charts align.
    entities = top_entities(df, top_n=top_n)
    labels = entity_display_labels(df, entities)
    table = _complete_count_table(df, "sentiment_bucket", SENTIMENT_ORDER, entities)
    plot_df = table.set_index("entity_content").loc[entities]
    plot_df.index = [labels[entity] for entity in plot_df.index]

    # Explanation: Horizontal stacked bars keep country names readable.
    ax = plot_df.plot(
        kind="barh",
        stacked=True,
        figsize=(10, 7),
        color=[SENTIMENT_COLORS[col] for col in plot_df.columns],
    )
    ax.invert_yaxis()
    ax.set_title("Mentions in France 2018 migration debates")
    ax.set_xlabel("Number of mentions")
    ax.set_ylabel("Mentioned entity")
    ax.legend(title="Sentiment", loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_country_reference_mentions(
    df: pl.DataFrame,
    output_path: Path,
    top_n: int = 15,
) -> Path:
    """Plot top countries by policy/situation/mixed/neutral reference type."""
    # Explanation: This shows whether a country is used as a policy comparison or context.
    entities = top_entities(df, top_n=top_n)
    labels = entity_display_labels(df, entities)
    table = _complete_count_table(df, "ref_type", REF_TYPE_ORDER, entities)
    plot_df = table.set_index("entity_content").loc[entities]
    plot_df.index = [labels[entity] for entity in plot_df.index]

    ax = plot_df.plot(
        kind="barh",
        stacked=True,
        figsize=(10, 7),
        color=[REF_TYPE_COLORS[col] for col in plot_df.columns],
    )
    ax.invert_yaxis()
    ax.set_title("Policy vs situation references by mentioned entity")
    ax.set_xlabel("Number of mentions")
    ax.set_ylabel("Mentioned entity")
    ax.legend(title="Reference type", loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_policy_situation_sentiment(
    df: pl.DataFrame,
    output_path: Path,
    top_n: int = 12,
) -> Path:
    """Plot sentiment colors separately for policy and situation/context mentions."""
    # Explanation: This directly compares policy talk with international situation context.
    entities = top_entities(df, top_n=top_n)
    labels = entity_display_labels(df, entities)
    filtered = df.filter(
        pl.col("entity_content").is_in(entities)
        & pl.col("ref_type").is_in(["policy", "situation"])
    )

    # Explanation: Create a full grid so missing negative values still appear in the legend.
    counts = (
        filtered
        .group_by(["entity_content", "ref_type", "sentiment_bucket"])
        .agg(pl.len().alias("n"))
        .to_pandas()
    )
    index = pd.MultiIndex.from_product(
        [entities, ["policy", "situation"], SENTIMENT_ORDER],
        names=["entity_content", "ref_type", "sentiment_bucket"],
    )
    counts = counts.set_index(["entity_content", "ref_type", "sentiment_bucket"]).reindex(index, fill_value=0)
    plot_df = counts.reset_index()

    # Explanation: Two panels keep policy and situation readable without overloading one chart.
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 7), sharey=True)
    for ax, ref_type in zip(axes, ["policy", "situation"]):
        panel = (
            plot_df[plot_df["ref_type"] == ref_type]
            .pivot(index="entity_content", columns="sentiment_bucket", values="n")
            .loc[entities, SENTIMENT_ORDER]
        )
        panel.index = [labels[entity] for entity in panel.index]
        panel.plot(
            kind="barh",
            stacked=True,
            ax=ax,
            color=[SENTIMENT_COLORS[col] for col in SENTIMENT_ORDER],
            legend=(ref_type == "situation"),
        )
        ax.invert_yaxis()
        ax.set_title(f"{ref_type.title()} references")
        ax.set_xlabel("Number of mentions")
        ax.set_ylabel("Mentioned entity" if ref_type == "policy" else "")
    axes[1].legend(title="Sentiment", loc="lower right")
    fig.suptitle("Sentiment by country: policy vs international situation context")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_reference_heatmap(
    df: pl.DataFrame,
    output_path: Path,
    top_n: int = 15,
) -> Path:
    """Plot a heatmap of mentioned entities by reference type."""
    # Explanation: The heatmap makes each country's dominant mode of mention visible at a glance.
    entities = top_entities(df, top_n=top_n)
    labels = entity_display_labels(df, entities)
    table = _complete_count_table(df, "ref_type", REF_TYPE_ORDER, entities)
    heatmap_df = table.set_index("entity_content").loc[entities, REF_TYPE_ORDER]
    heatmap_df.index = [labels[entity] for entity in heatmap_df.index]

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Mentions"},
    )
    plt.title("Reference-type intensity by mentioned entity")
    plt.xlabel("Reference type")
    plt.ylabel("Mentioned entity")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_all_figures(
    df: pl.DataFrame,
    processed_dir: Path,
    top_n: int = 10,
    min_mentions_for_all: int = 50,
) -> dict[str, Path]:
    """Create every pilot visualization and return the saved file paths."""
    # Explanation: One function lets notebook users regenerate all figures consistently.
    figures_dir = ensure_figures_dir(processed_dir)
    return {
        "entity_distribution_top10": plot_entity_distribution(
            df,
            figures_dir / "entity_distribution_top10.png",
            top_n=10,
        ),
        "entity_distribution_min50": plot_entity_distribution(
            df,
            figures_dir / "entity_distribution_min50.png",
            top_n=None,
            min_mentions=min_mentions_for_all,
        ),
        "entity_distribution_min50_csv": save_significant_entity_distribution_table(
            df,
            processed_dir / "entity_distribution_min50.csv",
            min_mentions=min_mentions_for_all,
        ),
        "entity_distribution_all_csv": save_entity_distribution_table(
            df,
            processed_dir / "entity_distribution_all_for_audit.csv",
        ),
        "country_sentiment": plot_country_sentiment_mentions(
            df,
            figures_dir / "country_sentiment_mentions_top10.png",
            top_n=top_n,
        ),
        "country_reference_type": plot_country_reference_mentions(
            df,
            figures_dir / "country_reference_type_mentions_top10.png",
            top_n=top_n,
        ),
        "policy_vs_situation_sentiment": plot_policy_situation_sentiment(
            df,
            figures_dir / "policy_vs_situation_sentiment_top10.png",
            top_n=top_n,
        ),
        "reference_heatmap": plot_reference_heatmap(
            df,
            figures_dir / "country_reference_heatmap_top10.png",
            top_n=top_n,
        ),
    }
