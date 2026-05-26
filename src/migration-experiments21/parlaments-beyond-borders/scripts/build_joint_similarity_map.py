"""Build a joint source-target correspondence-analysis similarity map.

The default input is the combined dyadic table produced by the all-country
pipeline:

    data/processed/ALL_AVAILABLE_COUNTRIES_dyadic_data_model/
        resolved_migration_country_mentions.parquet

Outputs are written to:

    data/processed/ALL_AVAILABLE_COUNTRIES_comparisons/
        figures_cross_country/interactive/

The map is a correspondence-analysis biplot. Source parliaments and target
countries that point in a similar direction are associated through
disproportionate attention, not literal Euclidean closeness.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go


SOURCE_CANDIDATES = [
    "source_country",
    "source_parliament",
    "parliament",
    "country",
    "source",
]
TARGET_CANDIDATES = [
    "target_country_name",
    "target_country",
    "target_entity",
    "entity_content",
    "entity_text",
    "entity",
]
TARGET_ISO3_CANDIDATES = [
    "target_country_iso3",
    "target_iso3",
    "target_country_code",
]
CATEGORY_CANDIDATES = [
    "entity_scope",
    "entity_category",
    "target_category",
]
YEAR_CANDIDATES = [
    "source_year",
    "year",
]
SENTIMENT_CANDIDATES = [
    "sentence_sentiment_value",
    "sentiment",
]

DEFAULT_TOP_TARGETS = 35
DEFAULT_MIN_TARGET_MENTIONS = 25
DEFAULT_MIN_SOURCE_MENTIONS = 25


@dataclass(frozen=True)
class CorrespondenceResult:
    row_coords: pd.DataFrame
    col_coords: pd.DataFrame
    inertia: np.ndarray
    explained_pct: np.ndarray
    total_count: float


def find_project_root(start: Path | None = None) -> Path:
    """Find the parlaments-beyond-borders root without doing work at import."""
    if start is None:
        start = Path.cwd()
    start = start.resolve()
    candidates = [start, *start.parents]
    script_parent = Path(__file__).resolve().parents[1]
    candidates.extend([script_parent, *script_parent.parents])

    for path in candidates:
        if (path / "data" / "processed").exists() and (path / "scripts").exists():
            return path
    raise FileNotFoundError(
        "Could not find project root. Run this from parlaments-beyond-borders "
        "or pass --project-root explicitly."
    )


def default_input_path(project_root: Path) -> Path:
    return (
        project_root
        / "data"
        / "processed"
        / "ALL_AVAILABLE_COUNTRIES_dyadic_data_model"
        / "resolved_migration_country_mentions.parquet"
    )


def default_output_dir(project_root: Path) -> Path:
    return (
        project_root
        / "data"
        / "processed"
        / "ALL_AVAILABLE_COUNTRIES_comparisons"
        / "figures_cross_country"
        / "interactive"
    )


def first_existing_column(columns: Iterable[str], candidates: list[str], role: str) -> str:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    raise ValueError(
        f"Could not find a {role} column. Tried {candidates}. "
        f"Available columns: {sorted(column_set)}"
    )


def optional_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input extension {path.suffix!r}; use parquet or csv.")


def build_contingency(
    df: pd.DataFrame,
    *,
    source_col: str,
    target_col: str,
    target_iso3_col: str | None = None,
    category_col: str | None = None,
    allowed_categories: set[str] | None = None,
    top_targets: int = DEFAULT_TOP_TARGETS,
    min_target_mentions: int = DEFAULT_MIN_TARGET_MENTIONS,
    min_source_mentions: int = DEFAULT_MIN_SOURCE_MENTIONS,
    keep_self_targets: bool = False,
) -> pd.DataFrame:
    data = df.copy()
    data = data[data[source_col].notna() & data[target_col].notna()]
    data[source_col] = data[source_col].astype(str).str.strip()
    data[target_col] = data[target_col].astype(str).str.strip()
    data = data[(data[source_col] != "") & (data[target_col] != "")]

    if category_col and allowed_categories:
        data = data[data[category_col].isin(allowed_categories)]

    if not keep_self_targets and target_iso3_col:
        data = data[data[target_iso3_col].isna() | (data[source_col] != data[target_iso3_col])]

    source_counts = data[source_col].value_counts()
    keep_sources = source_counts[source_counts >= min_source_mentions].index
    data = data[data[source_col].isin(keep_sources)]

    target_counts = data[target_col].value_counts()
    target_counts = target_counts[target_counts >= min_target_mentions]
    keep_targets = target_counts.head(top_targets).index
    data = data[data[target_col].isin(keep_targets)]

    table = pd.crosstab(data[source_col], data[target_col])
    table = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0]

    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError(
            "Need at least a 2x2 non-empty source-target table after filtering; "
            f"got shape {table.shape}. Lower --min-target-mentions or --min-source-mentions."
        )
    return table


def correspondence_analysis(table: pd.DataFrame, dims: int = 2) -> CorrespondenceResult:
    counts = table.to_numpy(dtype=float)
    total = counts.sum()
    if total <= 0:
        raise ValueError("Correspondence analysis requires a positive total count.")

    p = counts / total
    row_masses = p.sum(axis=1)
    col_masses = p.sum(axis=0)

    expected = np.outer(row_masses, col_masses)
    standardized = (p - expected) / np.sqrt(expected)
    u, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)

    inertia = singular_values**2
    explained_pct = inertia / inertia.sum() * 100 if inertia.sum() else np.zeros_like(inertia)
    keep_dims = min(dims, len(singular_values))

    row_principal = (u[:, :keep_dims] * singular_values[:keep_dims]) / np.sqrt(row_masses[:, None])
    col_principal = (vt[:keep_dims, :].T * singular_values[:keep_dims]) / np.sqrt(col_masses[:, None])

    row_coords = pd.DataFrame(row_principal, index=table.index)
    col_coords = pd.DataFrame(col_principal, index=table.columns)
    for coord_df in (row_coords, col_coords):
        coord_df.columns = [f"dim_{i + 1}" for i in range(keep_dims)]
        for i in range(keep_dims, dims):
            coord_df[f"dim_{i + 1}"] = 0.0

    row_coords["mass"] = row_masses
    row_coords["mentions"] = counts.sum(axis=1)
    col_coords["mass"] = col_masses
    col_coords["mentions"] = counts.sum(axis=0)

    return CorrespondenceResult(
        row_coords=row_coords,
        col_coords=col_coords,
        inertia=inertia,
        explained_pct=explained_pct,
        total_count=total,
    )


def standardized_residuals(table: pd.DataFrame) -> pd.DataFrame:
    counts = table.to_numpy(dtype=float)
    total = counts.sum()
    expected = np.outer(counts.sum(axis=1), counts.sum(axis=0)) / total
    residuals = (counts - expected) / np.sqrt(expected)
    return pd.DataFrame(residuals, index=table.index, columns=table.columns)


def top_positive_associations(residuals: pd.DataFrame, n: int = 3) -> dict[str, str]:
    labels: dict[str, str] = {}
    for source, row in residuals.iterrows():
        strongest = row.sort_values(ascending=False).head(n)
        labels[source] = "<br>".join(f"{target}: {value:.1f}" for target, value in strongest.items())
    return labels


def combined_coordinates(
    table: pd.DataFrame,
    result: CorrespondenceResult,
    residuals: pd.DataFrame,
) -> pd.DataFrame:
    source_assoc = top_positive_associations(residuals)
    target_assoc = top_positive_associations(residuals.T)

    rows = result.row_coords.reset_index()
    rows = rows.rename(columns={rows.columns[0]: "label"})
    rows["type"] = "source parliament"
    rows["top_positive_residuals"] = rows["label"].map(source_assoc)

    cols = result.col_coords.reset_index()
    cols = cols.rename(columns={cols.columns[0]: "label"})
    cols["type"] = "target country/entity"
    cols["top_positive_residuals"] = cols["label"].map(target_assoc)

    coords = pd.concat([rows, cols], ignore_index=True)
    coords["mentions"] = coords["mentions"].astype(int)
    return coords[
        [
            "type",
            "label",
            "dim_1",
            "dim_2",
            "mass",
            "mentions",
            "top_positive_residuals",
        ]
    ]


def make_figure(coords: pd.DataFrame, result: CorrespondenceResult, table: pd.DataFrame) -> go.Figure:
    source = coords[coords["type"] == "source parliament"]
    target = coords[coords["type"] == "target country/entity"]

    max_source = max(float(source["mentions"].max()), 1.0)
    max_target = max(float(target["mentions"].max()), 1.0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=source["dim_1"],
            y=source["dim_2"],
            mode="markers+text",
            name="Source parliament",
            text=source["label"],
            textposition="top center",
            marker={
                "size": 14 + 32 * np.sqrt(source["mentions"] / max_source),
                "color": "#2563eb",
                "line": {"color": "white", "width": 1},
                "opacity": 0.9,
            },
            customdata=np.stack(
                [
                    source["mentions"],
                    source["mass"],
                    source["top_positive_residuals"].fillna(""),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Type: source parliament<br>"
                "Mentions in table: %{customdata[0]:,}<br>"
                "Mass: %{customdata[1]:.3f}<br>"
                "Strongest positive residuals:<br>%{customdata[2]}"
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=target["dim_1"],
            y=target["dim_2"],
            mode="markers+text",
            name="Target country/entity",
            text=target["label"],
            textposition="bottom center",
            marker={
                "size": 10 + 24 * np.sqrt(target["mentions"] / max_target),
                "color": "#f97316",
                "symbol": "diamond",
                "line": {"color": "white", "width": 1},
                "opacity": 0.85,
            },
            customdata=np.stack(
                [
                    target["mentions"],
                    target["mass"],
                    target["top_positive_residuals"].fillna(""),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Type: target country/entity<br>"
                "Mentions in table: %{customdata[0]:,}<br>"
                "Mass: %{customdata[1]:.3f}<br>"
                "Strongest positive residuals:<br>%{customdata[2]}"
                "<extra></extra>"
            ),
        )
    )

    x_title = f"CA dimension 1 ({result.explained_pct[0]:.1f}% inertia)"
    y_title = f"CA dimension 2 ({result.explained_pct[1]:.1f}% inertia)"
    fig.update_layout(
        title={
            "text": (
                "Joint Similarity Map: Source Parliaments and Migration Target Countries"
                f"<br><sup>{int(result.total_count):,} mentions, "
                f"{table.shape[0]} sources x {table.shape[1]} targets. "
                "Shared direction indicates disproportionate attention.</sup>"
            ),
            "x": 0.02,
            "xanchor": "left",
        },
        template="plotly_white",
        width=1200,
        height=850,
        legend={"orientation": "h", "x": 0.02, "y": 1.06},
        margin={"l": 70, "r": 30, "t": 120, "b": 70},
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#94a3b8")
    fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="#94a3b8")
    fig.update_xaxes(zeroline=False)
    fig.update_yaxes(zeroline=False, scaleanchor="x", scaleratio=1)
    return fig


def write_outputs(
    table: pd.DataFrame,
    result: CorrespondenceResult,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    residuals = standardized_residuals(table)
    coords = combined_coordinates(table, result, residuals)
    fig = make_figure(coords, result, table)

    html_path = output_dir / "joint_similarity_map.html"
    coords_path = output_dir / "joint_similarity_map_coords.csv"
    table_path = output_dir / "joint_similarity_map_contingency.csv"

    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    coords.to_csv(coords_path, index=False)
    table.to_csv(table_path)
    return html_path, coords_path, table_path


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    project_root = Path(args.project_root).resolve() if args.project_root else find_project_root()
    input_path = Path(args.input).resolve() if args.input else default_input_path(project_root)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(project_root)

    df = read_table(input_path)
    source_col = args.source_col or first_existing_column(df.columns, SOURCE_CANDIDATES, "source")
    target_col = args.target_col or first_existing_column(df.columns, TARGET_CANDIDATES, "target")
    target_iso3_col = args.target_iso3_col or optional_column(df.columns, TARGET_ISO3_CANDIDATES)
    category_col = args.category_col or optional_column(df.columns, CATEGORY_CANDIDATES)

    allowed_categories = set(args.allowed_categories.split(",")) if args.allowed_categories else None
    table = build_contingency(
        df,
        source_col=source_col,
        target_col=target_col,
        target_iso3_col=target_iso3_col,
        category_col=category_col,
        allowed_categories=allowed_categories,
        top_targets=args.top_targets,
        min_target_mentions=args.min_target_mentions,
        min_source_mentions=args.min_source_mentions,
        keep_self_targets=args.keep_self_targets,
    )
    result = correspondence_analysis(table)
    paths = write_outputs(table, result, output_dir)

    print(f"Input: {input_path}")
    print(f"Source column: {source_col}")
    print(f"Target column: {target_col}")
    if target_iso3_col:
        print(f"Self-target filter column: {target_iso3_col}")
    print(f"Contingency table: {table.shape[0]} sources x {table.shape[1]} targets")
    print(f"Total mentions in CA table: {int(result.total_count):,}")
    print(
        "Explained inertia: "
        f"dim1={result.explained_pct[0]:.1f}%, dim2={result.explained_pct[1]:.1f}%"
    )
    print(f"Wrote HTML: {paths[0]}")
    print(f"Wrote coordinates: {paths[1]}")
    print(f"Wrote contingency table: {paths[2]}")
    return paths


def self_test() -> None:
    """Check CA math against Greenacre's smoking/staff example."""
    table = pd.DataFrame(
        [
            [4, 2, 3, 2],
            [4, 3, 7, 4],
            [25, 10, 12, 4],
            [18, 24, 33, 13],
            [10, 6, 7, 2],
        ],
        index=["SM", "JM", "SE", "JE", "SC"],
        columns=["none", "light", "medium", "heavy"],
    )
    result = correspondence_analysis(table)
    if not np.isclose(result.explained_pct[0], 87.76, atol=0.15):
        raise AssertionError(f"Unexpected dim1 inertia: {result.explained_pct[0]:.3f}")
    if not np.isclose(result.explained_pct[1], 11.76, atol=0.15):
        raise AssertionError(f"Unexpected dim2 inertia: {result.explained_pct[1]:.3f}")
    print("Self-test passed: Greenacre smoking/staff CA inertia reproduced.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", help="Project root. Defaults to auto-detection.")
    parser.add_argument("--input", help="Input parquet/csv. Defaults to resolved dyadic parquet.")
    parser.add_argument("--output-dir", help="Output directory for HTML and CSV files.")
    parser.add_argument("--source-col", help="Override detected source column.")
    parser.add_argument("--target-col", help="Override detected target column.")
    parser.add_argument("--target-iso3-col", help="Override detected target ISO3 column.")
    parser.add_argument("--category-col", help="Override detected category column.")
    parser.add_argument(
        "--allowed-categories",
        help="Comma-separated category whitelist, for example country,european_union.",
    )
    parser.add_argument("--top-targets", type=int, default=DEFAULT_TOP_TARGETS)
    parser.add_argument("--min-target-mentions", type=int, default=DEFAULT_MIN_TARGET_MENTIONS)
    parser.add_argument("--min-source-mentions", type=int, default=DEFAULT_MIN_SOURCE_MENTIONS)
    parser.add_argument("--keep-self-targets", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="Run CA math self-test and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run(args)


if __name__ == "__main__":
    main()
