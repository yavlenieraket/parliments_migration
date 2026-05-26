"""Build a comparative matrix atlas across all studied parliaments.

The atlas is meant to replace scattered one-off comparisons with a compact set
of matrices:

* country-to-country similarity across all encoded analytical levels
* integrated feature matrix with over/under-representation by country
* row-normalized matrices for targets, policy, cohort, agency, narrative,
  argument scheme, entity scope, and migration direction
* source-target sentiment and concreteness matrices

Outputs are written by default to:

    data/processed/ALL_AVAILABLE_COUNTRIES_comparisons/
        figures_cross_country/interactive/
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl


TOP_TARGETS = 35
TOP_FEATURE_TARGETS = 25
MIN_TARGET_CELL_COUNT = 20

ANALYTICAL_LEVELS = [
    ("entity_scope", "Entity scope"),
    ("policy_measure", "Policy measure"),
    ("migrant_cohort", "Migrant cohort"),
    ("policy_agency_type", "Policy agency"),
    ("narrative_polarity", "Narrative polarity"),
    ("narrative_frame", "Narrative frame"),
    ("argument_scheme", "Argument scheme"),
    ("migration_direction", "Migration direction"),
    ("sentiment_polarity", "Sentiment polarity"),
    ("concreteness_band", "Concreteness band"),
]

GENERIC_LAYER_VALUES = {
    "other",
    "missing",
    "general_policy",
    "general_migration",
    "neutral_reporting",
    "neutral_administrative",
    "external_transnational",
}


def find_project_root(start: Path | None = None) -> Path:
    if start is None:
        start = Path.cwd()
    start = start.resolve()
    candidates = [start, *start.parents]
    script_parent = Path(__file__).resolve().parents[1]
    candidates.extend([script_parent, *script_parent.parents])
    for path in candidates:
        if (path / "data" / "processed").exists() and (path / "scripts").exists():
            return path
    raise FileNotFoundError("Could not find parlaments-beyond-borders project root.")


def processed_dir(project_root: Path) -> Path:
    return project_root / "data" / "processed"


def default_output_dir(project_root: Path) -> Path:
    return (
        processed_dir(project_root)
        / "ALL_AVAILABLE_COUNTRIES_comparisons"
        / "figures_cross_country"
        / "interactive"
    )


def read_mentions(project_root: Path) -> pd.DataFrame:
    paths = sorted(processed_dir(project_root).glob("*_*_*/**/*_migration_mentions_extended.parquet"))
    if not paths:
        raise FileNotFoundError("No per-country *_migration_mentions_extended.parquet files found.")

    frames = []
    for path in paths:
        source_country = path.name.split("_", 1)[0]
        frame = pl.read_parquet(path).with_columns(pl.lit(source_country).alias("source_country"))
        frames.append(frame)
    df = pl.concat(frames, how="diagonal").to_pandas()
    df["source_country"] = df["source_country"].astype(str)
    return df


def ordered_sources(df: pd.DataFrame) -> list[str]:
    return (
        df.groupby("source_country")
        .size()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )


def safe_label(value: object) -> str:
    if pd.isna(value):
        return "missing"
    text = str(value).strip()
    return text if text else "missing"


def top_values(df: pd.DataFrame, column: str, n: int | None = None) -> list[str]:
    values = df[column].map(safe_label)
    counts = values.value_counts()
    if n is not None:
        counts = counts.head(n)
    return counts.index.astype(str).tolist()


def share_matrix(
    df: pd.DataFrame,
    column: str,
    *,
    rows: list[str],
    columns: list[str] | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    working = df[["source_country", column]].copy()
    working[column] = working[column].map(safe_label)
    if columns is None:
        columns = top_values(working, column, n=top_n)
    working = working[working[column].isin(columns)]
    counts = pd.crosstab(working["source_country"], working[column])
    totals = df.groupby("source_country").size()
    shares = counts.div(totals, axis=0).fillna(0)
    return shares.reindex(index=rows, columns=columns, fill_value=0)


def target_count_matrix(df: pd.DataFrame, *, rows: list[str], top_targets: int) -> pd.DataFrame:
    targets = top_values(df, "entity_content", n=top_targets)
    counts = pd.crosstab(df["source_country"], df["entity_content"].map(safe_label))
    return counts.reindex(index=rows, columns=targets, fill_value=0)


def target_share_matrix(df: pd.DataFrame, *, rows: list[str], top_targets: int) -> pd.DataFrame:
    counts = target_count_matrix(df, rows=rows, top_targets=top_targets)
    totals = df.groupby("source_country").size()
    return counts.div(totals, axis=0).fillna(0)


def target_metric_matrix(
    df: pd.DataFrame,
    metric: str,
    *,
    rows: list[str],
    targets: list[str],
    min_cell_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df[df["entity_content"].map(safe_label).isin(targets)].copy()
    working["target"] = working["entity_content"].map(safe_label)
    grouped = working.groupby(["source_country", "target"])
    values = grouped[metric].mean().unstack()
    counts = grouped.size().unstack()
    values = values.reindex(index=rows, columns=targets)
    counts = counts.reindex(index=rows, columns=targets).fillna(0)
    values = values.where(counts >= min_cell_count)
    return values, counts


def profile_features(df: pd.DataFrame, rows: list[str]) -> pd.DataFrame:
    grouped = df.groupby("source_country")
    features = pd.DataFrame(index=rows)
    features["log_mentions"] = np.log1p(grouped.size()).reindex(rows)
    features["log_speeches"] = np.log1p(grouped["speech_id"].nunique()).reindex(rows)
    features["log_speakers"] = np.log1p(grouped["speaker_id"].nunique()).reindex(rows)
    features["unique_targets"] = grouped["entity_content"].nunique().reindex(rows)
    if "concreteness_score" in df:
        features["mean_concreteness"] = grouped["concreteness_score"].mean().reindex(rows)
    if "sentence_sentiment_value" in df:
        features["mean_sentiment"] = grouped["sentence_sentiment_value"].mean().reindex(rows)
    return features.fillna(0)


def comparative_feature_matrix(df: pd.DataFrame, rows: list[str]) -> pd.DataFrame:
    pieces = [profile_features(df, rows)]
    for column, label in ANALYTICAL_LEVELS:
        if column not in df.columns:
            continue
        matrix = share_matrix(df, column, rows=rows)
        matrix = matrix.rename(columns={col: f"{label}: {col}" for col in matrix.columns})
        pieces.append(matrix)

    if "entity_content" in df.columns:
        target_shares = target_share_matrix(df, rows=rows, top_targets=TOP_FEATURE_TARGETS)
        target_shares = target_shares.rename(columns={col: f"Target share: {col}" for col in target_shares.columns})
        pieces.append(target_shares)

    return pd.concat(pieces, axis=1).fillna(0)


def zscore_columns(matrix: pd.DataFrame) -> pd.DataFrame:
    centered = matrix - matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=0).replace(0, np.nan)
    z = centered.div(std, axis=1).fillna(0)
    return z.clip(-2.5, 2.5)


def country_similarity(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    z = zscore_columns(feature_matrix)
    values = np.nan_to_num(
        z.to_numpy(dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    norms = np.linalg.norm(values, axis=1)
    norms[~np.isfinite(norms) | (norms == 0)] = 1
    normalized = np.nan_to_num(values / norms[:, None], nan=0.0, posinf=0.0, neginf=0.0)
    similarity = np.einsum("ik,jk->ij", normalized, normalized)
    similarity = np.nan_to_num(similarity, nan=0.0, posinf=1.0, neginf=-1.0)
    similarity = np.clip(similarity, -1, 1)
    return pd.DataFrame(similarity, index=feature_matrix.index, columns=feature_matrix.index)


def format_feature_name(name: str) -> str:
    return name.replace("_", " ")


def html_table(df: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    header = "".join(f"<th>{html.escape(str(col))}</th>" for col in df.columns)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(value))}</td>" for value in row.tolist())
            + "</tr>"
        )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def markdown_table(df: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    columns = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(value).replace("|", "\\|") for value in row.tolist()]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def similarity_pairs(similarity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = similarity.index.tolist()
    for i, left in enumerate(labels):
        for right in labels[i + 1:]:
            rows.append({
                "left": left,
                "right": right,
                "similarity": float(similarity.loc[left, right]),
            })
    return pd.DataFrame(rows)


def signature_table(z_features: pd.DataFrame, *, n_features: int = 4) -> pd.DataFrame:
    excluded = {"log_mentions", "log_speeches", "log_speakers", "unique_targets"}
    rows = []
    for source, values in z_features.drop(columns=[c for c in excluded if c in z_features.columns]).iterrows():
        top = values.sort_values(ascending=False).head(n_features)
        signatures = [
            f"{format_feature_name(feature)} ({score:.1f}z)"
            for feature, score in top.items()
            if score > 0
        ]
        rows.append({
            "Parliament": source,
            "Most distinctive positive features": "; ".join(signatures) if signatures else "No strong positive outlier",
        })
    return pd.DataFrame(rows)


def target_lift_table(
    target_shares: pd.DataFrame,
    target_counts: pd.DataFrame,
    *,
    min_count: int = MIN_TARGET_CELL_COUNT,
    top_n: int = 15,
) -> pd.DataFrame:
    global_share = target_counts.sum(axis=0) / target_counts.to_numpy().sum()
    rows = []
    for source in target_shares.index:
        for target in target_shares.columns:
            count = int(target_counts.loc[source, target])
            if count < min_count or global_share[target] <= 0:
                continue
            share = float(target_shares.loc[source, target])
            lift = share / float(global_share[target])
            rows.append({
                "Parliament": source,
                "Target": target,
                "Share": f"{share:.1%}",
                "Mentions": f"{count:,}",
                "Lift vs corpus": f"{lift:.1f}x",
                "_lift": lift,
            })
    result = pd.DataFrame(rows).sort_values("_lift", ascending=False).head(top_n)
    return result.drop(columns=["_lift"])


def metric_extreme_table(
    matrix: pd.DataFrame,
    counts: pd.DataFrame,
    *,
    metric_label: str,
    ascending: bool,
    top_n: int = 12,
) -> pd.DataFrame:
    rows = []
    for source in matrix.index:
        for target in matrix.columns:
            value = matrix.loc[source, target]
            if pd.isna(value):
                continue
            rows.append({
                "Parliament": source,
                "Target": target,
                metric_label: f"{float(value):.3f}",
                "Mentions": f"{int(counts.loc[source, target]):,}",
                "_value": float(value),
            })
    result = pd.DataFrame(rows).sort_values("_value", ascending=ascending).head(top_n)
    return result.drop(columns=["_value"])


def share_leaders_table(
    matrix: pd.DataFrame,
    *,
    level_label: str,
    top_n: int = 10,
    min_share: float = 0.05,
    excluded_values: set[str] | None = None,
) -> pd.DataFrame:
    excluded_values = excluded_values or set()
    rows = []
    for source in matrix.index:
        for category in matrix.columns:
            if str(category) in excluded_values:
                continue
            share = float(matrix.loc[source, category])
            if share < min_share:
                continue
            rows.append({
                "Level": level_label,
                "Parliament": source,
                "Category": category,
                "Share": f"{share:.1%}",
                "_share": share,
            })
    result = pd.DataFrame(rows).sort_values("_share", ascending=False).head(top_n)
    return result.drop(columns=["_share"])


def build_first_results(
    df: pd.DataFrame,
    similarity: pd.DataFrame,
    z_features: pd.DataFrame,
    target_shares: pd.DataFrame,
    target_counts: pd.DataFrame,
    target_sentiment: pd.DataFrame,
    target_sentiment_counts: pd.DataFrame,
    target_concreteness: pd.DataFrame,
    target_concreteness_counts: pd.DataFrame,
) -> tuple[str, str, dict[str, pd.DataFrame]]:
    pair_df = similarity_pairs(similarity)
    closest = pair_df.sort_values("similarity", ascending=False).head(8)
    divergent = pair_df.sort_values("similarity", ascending=True).head(8)
    signatures = signature_table(z_features)
    target_lifts = target_lift_table(target_shares, target_counts)
    positive_sentiment = metric_extreme_table(
        target_sentiment,
        target_sentiment_counts,
        metric_label="Mean sentiment",
        ascending=False,
    )
    negative_sentiment = metric_extreme_table(
        target_sentiment,
        target_sentiment_counts,
        metric_label="Mean sentiment",
        ascending=True,
    )
    concreteness = metric_extreme_table(
        target_concreteness,
        target_concreteness_counts,
        metric_label="Mean concreteness",
        ascending=False,
    )

    layer_leaders = []
    for column, label in ANALYTICAL_LEVELS:
        if column in df.columns:
            layer_leaders.append(
                share_leaders_table(
                    share_matrix(df, column, rows=similarity.index.tolist()),
                    level_label=label,
                    top_n=5,
                    excluded_values=GENERIC_LAYER_VALUES,
                )
            )
    layer_leaders_df = pd.concat(layer_leaders, ignore_index=True) if layer_leaders else pd.DataFrame()

    n_mentions = len(df)
    n_sources = df["source_country"].nunique()
    n_targets = df["entity_content"].map(safe_label).nunique()
    years = sorted(df["source_year"].dropna().astype(int).unique().tolist()) if "source_year" in df else []
    year_text = f"{years[0]}-{years[-1]}" if years else "unknown years"
    top_targets = df["entity_content"].map(safe_label).value_counts().head(6)
    top_target_text = ", ".join(f"{target} ({count:,})" for target, count in top_targets.items())

    closest_display = closest.assign(
        Pair=closest["left"] + " - " + closest["right"],
        Similarity=closest["similarity"].map(lambda value: f"{value:.2f}"),
    )[["Pair", "Similarity"]]
    divergent_display = divergent.assign(
        Pair=divergent["left"] + " - " + divergent["right"],
        Similarity=divergent["similarity"].map(lambda value: f"{value:.2f}"),
    )[["Pair", "Similarity"]]

    first_closest = closest_display.iloc[0]["Pair"] if not closest_display.empty else "none"
    first_divergent = divergent_display.iloc[0]["Pair"] if not divergent_display.empty else "none"
    first_lift = (
        f"{target_lifts.iloc[0]['Parliament']} -> {target_lifts.iloc[0]['Target']} "
        f"({target_lifts.iloc[0]['Lift vs corpus']})"
        if not target_lifts.empty
        else "none"
    )
    eu_europe_share = (
        df["entity_content"]
        .map(safe_label)
        .isin(["European Union", "Europe"])
        .mean()
    )
    route_targets = {"Mediterranean", "Libya", "Africa", "Turkey", "Syria", "Greece", "Morocco"}
    route_rows = target_lifts[target_lifts["Target"].isin(route_targets)].head(5)
    route_text = "; ".join(
        f"{row['Parliament']} -> {row['Target']} ({row['Lift vs corpus']})"
        for _, row in route_rows.iterrows()
    ) or "no route-specific overrepresentation among the top lifted cells"

    html_report = (
        "<section class='results'>"
        "<h2>First Results From The Matrix Atlas</h2>"
        "<p class='lead'>"
        f"The atlas compares {n_mentions:,} migration entity mentions from {n_sources} parliaments over {year_text}. "
        f"It covers {n_targets:,} distinct target labels. The most visible targets overall are {html.escape(top_target_text)}."
        "</p>"
        "<div class='result-grid'>"
        "<article><h3>Initial Reading</h3><ul>"
        f"<li><strong>EU/Europe is the common background:</strong> the two labels European Union and Europe account for {eu_europe_share:.1%} of all cleaned migration entity mentions. This means the most important comparative signal is not whether a parliament mentions Europe, but how it combines Europe with specific targets, policy tools, narratives, and argument schemes.</li>"
        f"<li>The closest all-level pair is <strong>{html.escape(first_closest)}</strong>; these parliaments look similar once target attention, policy labels, narratives, argument schemes, direction, sentiment, and concreteness are combined.</li>"
        f"<li>The strongest contrast is <strong>{html.escape(first_divergent)}</strong>, which should be treated as a first candidate for comparative close reading.</li>"
        f"<li>The most overrepresented source-target attention cell is <strong>{html.escape(first_lift)}</strong>, meaning this parliament discusses that target far more than the corpus baseline.</li>"
        f"<li><strong>Route and border geography appears immediately:</strong> {html.escape(route_text)}. These cells are good first places to inspect whether the discourse is about route management, crisis proximity, historical memory, or policy learning.</li>"
        "<li>Use the matrices below as evidence: the report gives leads, while the heatmaps show the underlying structure.</li>"
        "</ul></article>"
        "<article><h3>Closest Pairs</h3>" + html_table(closest_display) + "</article>"
        "<article><h3>Most Divergent Pairs</h3>" + html_table(divergent_display) + "</article>"
        "</div>"
        "<h3>Parliament Signatures</h3>"
        "<p class='note'>Positive z-scores identify what is unusually emphasized by each parliament compared with the others. Volume-only features are excluded from this signature table.</p>"
        f"{html_table(signatures)}"
        "<div class='result-grid'>"
        "<article><h3>Overrepresented Targets</h3>" + html_table(target_lifts) + "</article>"
        "<article><h3>Most Positive Target Cells</h3>" + html_table(positive_sentiment) + "</article>"
        "<article><h3>Most Negative Target Cells</h3>" + html_table(negative_sentiment) + "</article>"
        "<article><h3>Most Concrete Target Cells</h3>" + html_table(concreteness) + "</article>"
        "</div>"
        "<h3>Substantive Layer Leaders</h3>"
        "<p class='note'>These are the highest within-parliament shares after removing generic defaults such as other, general migration, general policy, neutral reporting, neutral administrative, and external transnational. This makes the table more useful for interpretation.</p>"
        f"{html_table(layer_leaders_df, max_rows=40)}"
        "</section>"
    )

    markdown = [
        "# First Results From The Comparative Matrix Atlas",
        "",
        f"- Corpus: {n_mentions:,} migration entity mentions, {n_sources} parliaments, {year_text}, {n_targets:,} target labels.",
        f"- Most visible targets overall: {top_target_text}.",
        f"- European Union + Europe share: {eu_europe_share:.1%}.",
        f"- Closest all-level pair: {first_closest}.",
        f"- Strongest contrast: {first_divergent}.",
        f"- Strongest overrepresented source-target attention: {first_lift}.",
        f"- Route/border geography lead: {route_text}.",
        "",
        "## Closest Pairs",
        markdown_table(closest_display),
        "",
        "## Most Divergent Pairs",
        markdown_table(divergent_display),
        "",
        "## Parliament Signatures",
        markdown_table(signatures),
        "",
        "## Overrepresented Targets",
        markdown_table(target_lifts),
        "",
        "## Most Positive Target Cells",
        markdown_table(positive_sentiment),
        "",
        "## Most Negative Target Cells",
        markdown_table(negative_sentiment),
        "",
        "## Most Concrete Target Cells",
        markdown_table(concreteness),
        "",
        "## Substantive Layer Leaders",
        markdown_table(layer_leaders_df, max_rows=40),
        "",
    ]
    tables = {
        "closest_pairs": closest_display,
        "divergent_pairs": divergent_display,
        "parliament_signatures": signatures,
        "overrepresented_targets": target_lifts,
        "positive_target_sentiment": positive_sentiment,
        "negative_target_sentiment": negative_sentiment,
        "concrete_target_cells": concreteness,
        "layer_leaders": layer_leaders_df,
    }
    return html_report, "\n".join(markdown), tables


def heatmap(
    matrix: pd.DataFrame,
    *,
    title: str,
    colorbar: str,
    colorscale: str | list,
    zmin: float | None = None,
    zmax: float | None = None,
    height: int = 720,
    percentage: bool = False,
    extra_counts: pd.DataFrame | None = None,
) -> go.Figure:
    z = matrix.to_numpy(dtype=float)
    if percentage:
        text = np.vectorize(lambda value: "" if np.isnan(value) else f"{value:.1%}")(z)
        hover = "%{y}<br>%{x}<br>Share: %{z:.1%}<extra></extra>"
    else:
        text = np.vectorize(lambda value: "" if np.isnan(value) else f"{value:.2f}")(z)
        hover = "%{y}<br>%{x}<br>Value: %{z:.3f}<extra></extra>"
    customdata = None
    if extra_counts is not None:
        customdata = extra_counts.reindex(index=matrix.index, columns=matrix.columns).to_numpy()
        hover = "%{y}<br>%{x}<br>Value: %{z:.3f}<br>Mentions: %{customdata:,}<extra></extra>"

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=matrix.columns,
            y=matrix.index,
            text=text,
            customdata=customdata,
            hovertemplate=hover,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar={"title": colorbar},
        )
    )
    fig.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        height=height,
        margin={"l": 90, "r": 30, "t": 90, "b": 130},
        xaxis={"tickangle": -45, "side": "bottom"},
        yaxis={"autorange": "reversed"},
    )
    return fig


def write_html(
    figures: list[tuple[str, str, go.Figure]],
    output_path: Path,
    *,
    first_results_html: str = "",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for idx, (anchor, note, fig) in enumerate(figures):
        include_js = "cdn" if idx == 0 else False
        parts.append(
            f"<section id='{html.escape(anchor)}'>"
            f"<p class='note'>{html.escape(note)}</p>"
            f"{pio.to_html(fig, include_plotlyjs=include_js, full_html=False)}"
            "</section>"
        )
    nav = "".join(
        f"<a href='#{html.escape(anchor)}'>{html.escape(anchor.replace('-', ' ').title())}</a>"
        for anchor, _, _ in figures
    )
    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Comparative Matrix Atlas</title>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;background:#f8fafc;color:#1f2937;}"
        "h1{margin:0 0 8px;font-size:28px;} .sub{max-width:1040px;line-height:1.45;color:#475467;margin:0 0 14px;}"
        "nav{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 22px;} nav a{background:#fff;border:1px solid #cbd5e1;border-radius:6px;padding:7px 9px;text-decoration:none;color:#1d4ed8;font-size:13px;}"
        "section{background:#fff;border:1px solid #d9e2ec;border-radius:8px;margin:18px 0;padding:14px;}"
        ".note{color:#344054;line-height:1.45;margin:0 0 8px;}.lead{font-size:15px;line-height:1.5;color:#344054;max-width:1120px;}"
        ".results{border-color:#9fb3c8;background:#f7fbff;}.result-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px;margin:12px 0;}"
        ".result-grid article{background:#fff;border:1px solid #d9e2ec;border-radius:8px;padding:12px;} h2{margin:0 0 8px;} h3{margin:8px 0;font-size:16px;}"
        "table{border-collapse:collapse;width:100%;font-size:12px;background:#fff;margin:8px 0 14px;} th,td{border:1px solid #d9e2ec;padding:6px;text-align:left;vertical-align:top;} th{background:#e5e9f0;} ul{padding-left:20px;line-height:1.5;}"
        "</style></head><body>"
        "<h1>Comparative Matrix Atlas</h1>"
        "<p class='sub'>A matrix-first comparison of all studied parliaments across target attention, entity grammar, policy measures, migrant cohorts, policy agency, narrative framing, argument schemes, migration direction, sentiment, and concreteness. Row-normalized matrices show within-parliament emphasis; the similarity matrix uses all encoded levels together.</p>"
        f"<nav>{nav}</nav>"
        f"{first_results_html}"
        f"{''.join(parts)}"
        "</body></html>"
    )
    output_path.write_text(document, encoding="utf-8")


def build_atlas(project_root: Path, output_dir: Path) -> dict[str, Path]:
    df = read_mentions(project_root)
    rows = ordered_sources(df)
    feature_matrix = comparative_feature_matrix(df, rows)
    z_features = zscore_columns(feature_matrix)
    similarity = country_similarity(feature_matrix)

    target_shares = target_share_matrix(df, rows=rows, top_targets=TOP_TARGETS)
    target_counts = target_count_matrix(df, rows=rows, top_targets=TOP_TARGETS)
    targets = target_shares.columns.tolist()
    target_sentiment, target_sentiment_counts = target_metric_matrix(
        df,
        "sentence_sentiment_value",
        rows=rows,
        targets=targets,
        min_cell_count=MIN_TARGET_CELL_COUNT,
    )
    target_concreteness, target_concreteness_counts = target_metric_matrix(
        df,
        "concreteness_score",
        rows=rows,
        targets=targets,
        min_cell_count=MIN_TARGET_CELL_COUNT,
    )
    first_results_html, first_results_md, first_result_tables = build_first_results(
        df,
        similarity,
        z_features,
        target_shares,
        target_counts,
        target_sentiment,
        target_sentiment_counts,
        target_concreteness,
        target_concreteness_counts,
    )

    figures: list[tuple[str, str, go.Figure]] = [
        (
            "country-similarity",
            "Country-to-country cosine similarity computed from the full comparative feature matrix: profile metrics, category shares, and top-target shares. This is the closest thing to a single all-level comparison.",
            heatmap(similarity, title="Country similarity across all analytical levels", colorbar="Similarity", colorscale="RdBu", zmin=-1, zmax=1, height=760),
        ),
        (
            "integrated-feature-matrix",
            "Column-wise z-scores show where each parliament is over- or under-represented on each encoded feature. Values are clipped at +/-2.5 so the pattern remains readable.",
            heatmap(z_features, title="Integrated feature matrix: relative emphasis by parliament", colorbar="z-score", colorscale="RdBu", zmin=-2.5, zmax=2.5, height=980),
        ),
        (
            "target-attention",
            "Rows sum to each parliament's total migration entity mentions, so cells compare target emphasis rather than raw corpus size.",
            heatmap(target_shares, title="Target attention matrix: within-parliament share", colorbar="Share", colorscale="Blues", height=820, percentage=True),
        ),
        (
            "target-sentiment",
            f"Mean sentence sentiment for source-target cells with at least {MIN_TARGET_CELL_COUNT} mentions. Empty cells are hidden because the evidence is too sparse.",
            heatmap(target_sentiment, title="Target sentiment matrix", colorbar="Mean sentiment", colorscale="RdBu", height=820, extra_counts=target_sentiment_counts),
        ),
        (
            "target-concreteness",
            f"Mean concreteness for source-target cells with at least {MIN_TARGET_CELL_COUNT} mentions. Empty cells are hidden because the evidence is too sparse.",
            heatmap(target_concreteness, title="Target concreteness matrix", colorbar="Mean concreteness", colorscale="Viridis", height=820, extra_counts=target_concreteness_counts),
        ),
    ]

    for column, label in ANALYTICAL_LEVELS:
        if column not in df.columns:
            continue
        matrix = share_matrix(df, column, rows=rows)
        figures.append(
            (
                label.lower().replace(" ", "-"),
                f"Within-parliament distribution over {label.lower()}. These are row-normalized shares, not raw counts.",
                heatmap(matrix, title=f"{label} matrix: within-parliament share", colorbar="Share", colorscale="Blues", height=720, percentage=True),
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "atlas": output_dir / "comparative_matrix_atlas.html",
        "feature_matrix": output_dir / "comparative_feature_matrix.csv",
        "feature_matrix_zscore": output_dir / "comparative_feature_matrix_zscore.csv",
        "country_similarity": output_dir / "country_similarity_all_levels.csv",
        "target_attention_share": output_dir / "target_attention_share_matrix.csv",
        "target_attention_count": output_dir / "target_attention_count_matrix.csv",
        "target_sentiment": output_dir / "target_sentiment_matrix.csv",
        "target_concreteness": output_dir / "target_concreteness_matrix.csv",
        "first_results_markdown": output_dir / "comparative_matrix_first_results.md",
    }
    write_html(figures, paths["atlas"], first_results_html=first_results_html)
    feature_matrix.to_csv(paths["feature_matrix"])
    z_features.to_csv(paths["feature_matrix_zscore"])
    similarity.to_csv(paths["country_similarity"])
    target_shares.to_csv(paths["target_attention_share"])
    target_counts.to_csv(paths["target_attention_count"])
    target_sentiment.to_csv(paths["target_sentiment"])
    target_concreteness.to_csv(paths["target_concreteness"])
    paths["first_results_markdown"].write_text(first_results_md, encoding="utf-8")
    for name, table in first_result_tables.items():
        table_path = output_dir / f"first_results_{name}.csv"
        table.to_csv(table_path, index=False)
        paths[f"first_results_{name}"] = table_path
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", help="Project root. Defaults to auto-detection.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to comparison interactive figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve() if args.project_root else find_project_root()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(project_root)
    paths = build_atlas(project_root, output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
