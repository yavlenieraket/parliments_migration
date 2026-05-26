"""Generate visualization bundles for every studied country.

This script is the code-side counterpart to the notebooks. It ensures that all
studied countries have the same per-country visualization bundle and then builds
the combined dyadic visualizations across all countries.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import plotly.express as px
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_extended_analysis as runner  # noqa: E402
from src import data_model, visualize  # noqa: E402
from src.config import PROCESSED_DIR  # noqa: E402


STUDIED_COUNTRIES = runner.DEFAULT_COUNTRIES
COMBINED_PREFIX = "ALL_AVAILABLE_COUNTRIES"

alt.data_transformers.disable_max_rows()

VISUAL_EXPLANATIONS = {
    "concreteness_density_by_weog": {
        "title": "Concreteness Density By Region Group",
        "question": "Are migration references grounded in concrete wording, and does this differ by broad region group?",
        "read": "Look for whether the distribution is shifted toward higher values. A right-shift means the debate uses more named places, dates, institutions, numbers, or event-like language.",
    },
    "concreteness_by_year_region": {
        "title": "Concreteness By Year And Region",
        "question": "Does concreteness change over time for European, non-European, EU, or special territory references?",
        "read": "Compare year-to-year movement. Spikes often mean parliamentarians were reacting to visible events rather than abstract policy categories.",
    },
    "country_concreteness_year_lines": {
        "title": "Country-Level Concreteness Over Time",
        "question": "Which mentioned countries become more or less concrete in parliamentary talk?",
        "read": "Each line is a mentioned country/entity. Higher points mean that references to that country are more fact-anchored in that year.",
    },
    "concreteness_feature_pattern_heatmap": {
        "title": "Repeated Agency And Narrative Patterns",
        "question": "Which combinations of policy agency and narrative frame repeat, and how concrete are they?",
        "read": "Cells show repeated discourse patterns. Use this to identify where the same frame is attached again and again to the same kind of agency claim.",
    },
    "concreteness_quote_panels": {
        "title": "Quote Panels: What People Actually Say",
        "question": "What concrete and abstract language sits behind the scores?",
        "read": "Use this as an audit view. It gives excerpts rather than only aggregates, so you can check whether a score or classification makes interpretive sense.",
    },
    "diffusion_top_targets": {
        "title": "Top Policy Diffusion Targets",
        "question": "Which countries/entities are invoked most often as policy references?",
        "read": "Longer bars indicate more repeated use as comparison cases, models, warnings, or institutional references.",
    },
    "cohort_policy_heatmap": {
        "title": "Migrant Cohorts By Policy Measures",
        "question": "Which migrant groups are connected to which policy instruments?",
        "read": "Dense cells show where a parliament repeatedly links a cohort, such as refugees or students, with a policy field, such as asylum procedure or integration.",
    },
    "country_concreteness_bubble": {
        "title": "Country Concreteness Bubble",
        "question": "Which countries are mentioned often and concretely?",
        "read": "Large bubbles are frequent targets. Higher positions mean more concrete language. Color separates regional or analytical groups.",
    },
    "country_year_concreteness_heatmap": {
        "title": "Country-Year Concreteness Heatmap",
        "question": "Which country-year cells are unusually concrete?",
        "read": "Hotter cells identify country references tied to visible events in specific years.",
    },
    "country_cohort_heatmap": {
        "title": "Country By Migrant Cohort",
        "question": "Which migrant cohorts are associated with which mentioned countries?",
        "read": "Cells show repeated association between a target country and a migrant category.",
    },
    "country_policy_heatmap": {
        "title": "Country By Policy Measure",
        "question": "Which policy tools are associated with which mentioned countries?",
        "read": "Use this to see whether a country is discussed through asylum, border control, integration, returns, visas, or other policy fields.",
    },
    "policy_agency_network": {
        "title": "Policy Agency Network",
        "question": "Who is treated as a model, pressure target, partner, competitor, or neutral reference?",
        "read": "Nodes are source and target countries/entities. Edges summarize directional policy agency. Thick links mean repeated references.",
    },
    "policy_agency_country_heatmap": {
        "title": "Policy Agency By Target Country",
        "question": "How is each mentioned country positioned in policy reasoning?",
        "read": "Read across a country row to see whether it is mainly a model, intervention target, cooperation partner, competitor, or neutral report case.",
    },
    "policy_hubs_pagerank": {
        "title": "Policy Hubs PageRank",
        "question": "Which countries become hubs in the directed policy-reference network?",
        "read": "Higher PageRank means the country is structurally central, not just frequently mentioned once.",
    },
    "narrative_ternary": {
        "title": "Narrative Ternary",
        "question": "Is migration imagined through solidarity, risk, benefit, or administrative language?",
        "read": "Points closer to one corner indicate stronger association with that narrative pole.",
    },
    "narrative_mirror_bars": {
        "title": "Narrative Mirror Bars",
        "question": "How much of the discourse is solidarity, anti-solidarity, benefit, or administrative framing?",
        "read": "Compare positive/supportive language against threat/control/legal-administrative framing.",
    },
    "evidence_visibility_map": {
        "title": "Evidence Visibility Map",
        "question": "Which places become visible as concrete migration evidence?",
        "read": "Darker or larger map marks indicate countries/entities that parliamentarians cite with concrete details.",
    },
    "fact_density_timeline": {
        "title": "Fact Density Timeline",
        "question": "When do speeches become especially concrete?",
        "read": "Points higher on the chart are more concrete. Hover to inspect named entities and event-like anchors.",
    },
    "direction_agenda_split_bars": {
        "title": "Internal vs External Migration Agenda",
        "question": "Does the parliament discuss migration into itself or migration elsewhere?",
        "read": "The split shows inbound/internal, outbound, external/transnational, and ambiguous migration-direction talk.",
    },
    "bilateral_concreteness_matrix": {
        "title": "Combined Bilateral Concreteness Matrix",
        "question": "Across all studied parliaments, which source-target country pairs are frequent and concrete?",
        "read": "Rows are speaking parliaments, columns are mentioned countries. Color is mean concreteness; dot size is mention volume.",
    },
    "attention_asymmetry_bars": {
        "title": "Attention Asymmetry",
        "question": "Which country pairs are unequal in attention?",
        "read": "Positive and negative bars show whether A talks about B more than B talks about A. Larger absolute values mean stronger asymmetry.",
    },
    "shock_moria_fire_delta_heatmap": {
        "title": "Moria Fire Shock Window",
        "question": "Which source-target arcs changed around the Moria camp fire?",
        "read": "Cells show post-event minus pre-event mentions in a 90-day window.",
    },
    "shock_belarus_crisis_delta_heatmap": {
        "title": "Belarus Border Crisis Shock Window",
        "question": "Which parliamentary attention arcs switched on around the Belarus border crisis?",
        "read": "Positive cells indicate more target-country attention after the event than before.",
    },
    "shock_ukraine_war_delta_heatmap": {
        "title": "Ukraine War Shock Window",
        "question": "How did country references change around Russia's full-scale invasion of Ukraine?",
        "read": "Large positive cells identify parliaments and targets whose attention increased after February 24, 2022.",
    },
    "comparison_summary_table": {
        "title": "Cross-Country Summary Table",
        "question": "What is the one-row profile of each parliament?",
        "read": "Use this first. It lists volume, speakers, most mentioned country/entity, mean concreteness, sentiment, and dominant discourse labels.",
    },
    "comparison_mentions_by_year_source": {
        "title": "Mentions By Year And Parliament",
        "question": "When does each parliament talk more intensely about migration-related foreign countries?",
        "read": "Compare peaks across lines. Peaks often indicate shock-event years or parliamentary agenda shifts.",
    },
    "comparison_total_mentions_by_source": {
        "title": "Total Mentions By Parliament",
        "question": "Which parliament contributes the largest volume of migration country references?",
        "read": "This is a volume baseline. Use it before comparing percentages so large and small corpora are not confused.",
    },
    "comparison_top_targets_heatmap": {
        "title": "Top Targets Compared Across Parliaments",
        "question": "Do parliaments look at the same countries or different ones?",
        "read": "Rows are speaking parliaments and columns are mentioned countries/entities. Dark cells reveal shared or country-specific attention targets.",
    },
    "comparison_mean_concreteness_by_source": {
        "title": "Mean Concreteness By Parliament",
        "question": "Which parliament speaks in more grounded, fact-like language?",
        "read": "Higher bars mean more concrete mention windows on average. Interpret with the quote/context views, not alone.",
    },
    "comparison_concreteness_target_heatmap": {
        "title": "Concreteness Of Shared Targets",
        "question": "When different parliaments mention the same target, do they speak about it equally concretely?",
        "read": "Use this to find the same target being treated as a concrete event in one parliament but abstractly in another.",
    },
    "comparison_agency_composition": {
        "title": "Policy Agency Composition",
        "question": "Which parliament uses others as models, pressure targets, partners, competitors, or neutral examples?",
        "read": "This normalizes within each parliament, so it compares style of referencing rather than raw volume.",
    },
    "comparison_narrative_polarity": {
        "title": "Narrative Polarity Composition",
        "question": "How do parliaments differ in solidarity, risk, benefit, and administrative imagination?",
        "read": "Compare the shares within each bar to see discourse flavor across countries.",
    },
    "comparison_direction_agenda": {
        "title": "Direction Agenda Composition",
        "question": "Who focuses on inbound domestic migration and who discusses migration between third countries?",
        "read": "Internal/inbound shares indicate domestic agenda focus; external/transnational shares indicate observation of wider migration processes.",
    },
    "comparison_migrant_cohorts": {
        "title": "Migrant Cohort Composition",
        "question": "Which migrant groups dominate each parliament's country references?",
        "read": "Compare shares of refugees, asylum seekers, students, economic migrants, high-skilled workers, and general migration.",
    },
    "comparison_policy_measure_composition": {
        "title": "Policy Measure Composition",
        "question": "Which policy instruments dominate each parliament's migration country talk?",
        "read": "This normalized view shows whether debate is more about borders, asylum procedure, integration, returns, visas, security, or law.",
    },
    "comparison_policy_measure_heatmap": {
        "title": "Policy Measure Heatmap",
        "question": "Where are policy-measure differences concentrated?",
        "read": "Dark cells show a policy field taking a large share of a parliament's migration country references.",
    },
    "comparison_sentiment_by_source": {
        "title": "Mean Sentiment By Parliament",
        "question": "How does the sentence-level emotional tone differ across parliaments?",
        "read": "Use as a broad signal only. Sentiment is averaged over mention windows and should be interpreted with context examples.",
    },
    "comparison_high_concreteness_fact_counts": {
        "title": "High-Concreteness Fact Counts",
        "question": "Which parliament gives the most event-like, named, concrete migration evidence?",
        "read": "Higher bars mean more high-concreteness snippets that can be inspected as real-event evidence.",
    },
    "interactive_mentions_timeline": {
        "title": "Interactive Mentions Timeline",
        "question": "How does migration-country attention move over time across all parliaments?",
        "read": "Use legend clicks to isolate countries, hover for yearly counts, and zoom into specific periods.",
    },
    "interactive_target_heatmap": {
        "title": "Interactive Target Heatmap",
        "question": "Which source parliaments share target countries, and which targets are country-specific?",
        "read": "Hover cells for exact counts. Dark cells identify highly visible targets in each parliament.",
    },
    "interactive_scope_composition": {
        "title": "Interactive Entity-Scope Composition",
        "question": "Does a parliament speak through countries, the EU, broad regions/routes, or territories?",
        "read": "This normalized chart compares target-entity grammar across countries.",
    },
    "interactive_concreteness_sentiment_scatter": {
        "title": "Interactive Concreteness And Sentiment Scatter",
        "question": "Which parliaments combine concrete language with more positive or negative sentence tone?",
        "read": "Each point is a parliament. Size is volume; x is concreteness; y is mean sentiment.",
    },
    "interactive_asymmetry_scatter": {
        "title": "Interactive Attention Asymmetry Scatter",
        "question": "Which country pairs are most unequal in reciprocal attention?",
        "read": "Far-right/far-left points are strongest asymmetries. Hover to compare A->B and B->A counts.",
    },
    "joint_similarity_map": {
        "title": "Joint Similarity Map",
        "question": "Which parliaments share a discursive geometry of migration, and which target countries cluster with them?",
        "read": "Source and target bubbles pointing in the same direction indicate disproportionate attention. Use the coordinate CSV for residual checks before making strong claims.",
    },
    "comparative_matrix_atlas": {
        "title": "Comparative Matrix Atlas",
        "question": "How do all studied parliaments compare across every encoded analytical level at once?",
        "read": "Start with the country-similarity matrix, then inspect the integrated feature matrix and row-normalized layer matrices for targets, policy, cohorts, agency, narratives, argument schemes, direction, sentiment, and concreteness.",
    },
}


def available_years_by_country(countries: list[str] = STUDIED_COUNTRIES) -> dict[str, list[int]]:
    """Return available fact years for each studied country."""
    return {
        country: [runner.year_from_fact_path(path) for path in runner.available_fact_files(country)]
        for country in countries
    }


def ensure_country_visualizations(
    country_years: dict[str, list[int]],
    rerun_existing: bool = False,
) -> dict[str, Path]:
    """Ensure every country has the same extended visualization outputs."""
    output_dirs: dict[str, Path] = {}
    for country, years in country_years.items():
        prefix = runner.country_prefix(country, years)
        country_dir = PROCESSED_DIR / prefix
        mention_path = country_dir / f"{prefix}_migration_mentions_extended.parquet"
        figures_dir = country_dir / "figures_interactive_advanced"
        if rerun_existing or not mention_path.exists() or not figures_dir.exists():
            runner.save_country_analysis(country, years)
        output_dirs[country] = country_dir
    return output_dirs


def generate_combined_dyadic_visualizations(
    country_years: dict[str, list[int]],
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Build combined resolver, bilateral, asymmetry, and shock visualizations."""
    output_dir = output_dir or (PROCESSED_DIR / f"{COMBINED_PREFIX}_dyadic_data_model")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_facts = pl.concat(
        [
            runner.load_country_facts(country, years)
            for country, years in country_years.items()
        ],
        how="vertical",
    )
    resolver_path = data_model.save_country_resolver()
    resolver = data_model.scan_country_resolver(resolver_path)
    facts_mig = data_model.migration_country_mentions(
        all_facts,
        country_resolver=resolver,
        min_hits=2,
        min_confidence=0.85,
    )
    mentions = data_model.mentions_with_concreteness(facts_mig, all_facts)

    mentions_path = output_dir / "resolved_migration_country_mentions.parquet"
    mentions.sink_parquet(mentions_path)
    resolved_mentions = pl.scan_parquet(mentions_path)

    bilateral = data_model.bilateral_matrix(resolved_mentions)
    bilateral_concrete = data_model.bilateral_concreteness(resolved_mentions)
    asymmetry = data_model.compute_asymmetry(bilateral, min_total_traffic=20)

    bilateral_path = output_dir / "bilateral_matrix.csv"
    bilateral_concrete_path = output_dir / "bilateral_concreteness.csv"
    asymmetry_path = output_dir / "asymmetry_table.csv"

    bilateral.write_csv(bilateral_path)
    bilateral_concrete.write_csv(bilateral_concrete_path)
    asymmetry.write_csv(asymmetry_path)

    shock_tables = {}
    shock_paths = {}
    for shock_name in ["moria_fire", "belarus_crisis", "ukraine_war"]:
        shock = data_model.shock_window_matrix(
            resolved_mentions,
            data_model.SHOCKS[shock_name],
            window_days=90,
        )
        shock_path = output_dir / f"shock_{shock_name}_window.csv"
        shock.write_csv(shock_path)
        shock_tables[shock_name] = shock
        shock_paths[f"shock_{shock_name}"] = shock_path

    figure_paths = visualize.save_data_model_figures(
        bilateral_concrete,
        asymmetry,
        output_dir,
        shock_tables=shock_tables,
    )
    return {
        "country_resolver": resolver_path,
        "resolved_mentions": mentions_path,
        "bilateral_matrix": bilateral_path,
        "bilateral_concreteness": bilateral_concrete_path,
        "asymmetry_table": asymmetry_path,
        **shock_paths,
        **figure_paths,
    }


def _save_chart(chart: alt.Chart, output_path: Path) -> Path:
    """Save a comparison chart as HTML, Vega-Lite JSON, and PNG when possible."""
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
    except Exception:
        return html_path


def _save_plotly(fig, output_path: Path) -> Path:
    """Save a Plotly figure as a self-contained interactive HTML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    return output_path


def _read_country_mentions(country: str, years: list[int]) -> pl.DataFrame:
    """Read one country's processed mention table and add a stable source column."""
    prefix = runner.country_prefix(country, years)
    path = PROCESSED_DIR / prefix / f"{prefix}_migration_mentions_extended.parquet"
    return (
        pl.read_parquet(path)
        .with_columns(pl.lit(country).alias("source_country"))
    )


def combined_mentions(country_years: dict[str, list[int]]) -> pl.DataFrame:
    """Return one combined processed mention table for all studied countries."""
    return pl.concat(
        [_read_country_mentions(country, years) for country, years in country_years.items()],
        how="diagonal",
    )


def _top_targets(df: pl.DataFrame, n: int = 25) -> list[str]:
    """Return the most mentioned target entities across all source countries."""
    return (
        df
        .group_by("entity_content")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(n)
        .get_column("entity_content")
        .to_list()
    )


def _stacked_share_chart(
    df: pl.DataFrame,
    category_col: str,
    output_path: Path,
    title: str,
    category_title: str,
    top_n: int | None = None,
) -> Path:
    """Save source-country normalized stacked bars for one categorical feature."""
    work = df.drop_nulls(category_col)
    if top_n:
        top_categories = (
            work
            .group_by(category_col)
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
            .head(top_n)
            .get_column(category_col)
            .to_list()
        )
        work = work.with_columns(
            pl.when(pl.col(category_col).is_in(top_categories))
            .then(pl.col(category_col))
            .otherwise(pl.lit("other"))
            .alias(category_col)
        )
    plot_df = (
        work
        .group_by(["source_country", category_col])
        .agg(pl.len().alias("n_mentions"))
        .with_columns(
            (pl.col("n_mentions") / pl.col("n_mentions").sum().over("source_country")).alias("share")
        )
        .to_pandas()
    )
    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("source_country:N", title="Speaking parliament"),
            y=alt.Y("n_mentions:Q", title="Share of mentions", stack="normalize"),
            color=alt.Color(f"{category_col}:N", title=category_title),
            tooltip=[
                alt.Tooltip("source_country:N", title="Source"),
                alt.Tooltip(f"{category_col}:N", title=category_title),
                alt.Tooltip("n_mentions:Q", title="Mentions"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        )
        .interactive()
        .properties(title=title, width=760, height=360)
    )
    return _save_chart(chart, output_path)


def save_comparison_summary_table(df: pl.DataFrame, output_path: Path) -> Path:
    """Save a compact comparative interpretation table for all countries."""
    summary = (
        df
        .group_by("source_country")
        .agg([
            pl.len().alias("n_mentions"),
            pl.col("speech_id").n_unique().alias("n_speeches"),
            pl.col("speaker_id").n_unique().alias("n_speakers"),
            pl.col("entity_content").mode().first().alias("most_mentioned_entity"),
            pl.col("concreteness_score").mean().round(3).alias("mean_concreteness"),
            pl.col("sentence_sentiment_value").mean().round(3).alias("mean_sentiment"),
            pl.col("policy_agency_type").mode().first().alias("dominant_agency"),
            pl.col("narrative_polarity").mode().first().alias("dominant_narrative_polarity"),
            pl.col("migration_direction").mode().first().alias("dominant_direction"),
            pl.col("migrant_cohort").mode().first().alias("dominant_cohort"),
            pl.col("policy_measure").mode().first().alias("dominant_policy_measure"),
        ])
        .sort("source_country")
        .to_dicts()
    )
    rows = []
    for row in summary:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['source_country']))}</td>"
            f"<td>{int(row['n_mentions']):,}</td>"
            f"<td>{int(row['n_speeches']):,}</td>"
            f"<td>{int(row['n_speakers']):,}</td>"
            f"<td>{html.escape(str(row['most_mentioned_entity']))}</td>"
            f"<td>{float(row['mean_concreteness']):.3f}</td>"
            f"<td>{float(row['mean_sentiment']):.3f}</td>"
            f"<td>{html.escape(str(row['dominant_agency']))}</td>"
            f"<td>{html.escape(str(row['dominant_narrative_polarity']))}</td>"
            f"<td>{html.escape(str(row['dominant_direction']))}</td>"
            f"<td>{html.escape(str(row['dominant_cohort']))}</td>"
            f"<td>{html.escape(str(row['dominant_policy_measure']))}</td>"
            "</tr>"
        )
    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Cross-Country Comparison Summary</title>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:28px;background:#f8fafc;color:#1f2937;}"
        "h1{font-size:24px;margin:0 0 6px;} p{color:#52606d;margin:0 0 16px;}"
        "table{border-collapse:collapse;width:100%;background:#fff;font-size:13px;}"
        "th,td{border:1px solid #d9e2ec;padding:8px;text-align:left;vertical-align:top;} th{background:#e5e9f0;position:sticky;top:0;}"
        "</style></head><body>"
        "<h1>Cross-Country Comparison Summary</h1>"
        "<p>One-row summary per speaking parliament. Use it to orient the detailed comparison charts.</p>"
        "<table><thead><tr>"
        "<th>Source</th><th>Mentions</th><th>Speeches</th><th>Speakers</th><th>Most mentioned</th>"
        "<th>Mean concreteness</th><th>Mean sentiment</th><th>Dominant agency</th><th>Dominant narrative</th>"
        "<th>Dominant direction</th><th>Dominant cohort</th><th>Dominant policy</th>"
        "</tr></thead><tbody>"
        f"{''.join(rows)}"
        "</tbody></table></body></html>"
    )
    output_path.write_text(document, encoding="utf-8")
    return output_path


def save_interactive_comparison_figures(
    df: pl.DataFrame,
    figures_dir: Path,
    top_targets: list[str],
    asymmetry_path: Path | None = None,
) -> dict[str, Path]:
    """Save richer Plotly figures for exploration in the HTML report."""
    outputs: dict[str, Path] = {}
    interactive_dir = figures_dir / "interactive"
    interactive_dir.mkdir(parents=True, exist_ok=True)

    year_df = (
        df
        .group_by(["source_country", "source_year"])
        .agg(pl.len().alias("n_mentions"))
        .sort(["source_country", "source_year"])
        .to_pandas()
    )
    fig = px.line(
        year_df,
        x="source_year",
        y="n_mentions",
        color="source_country",
        markers=True,
        title="Interactive mentions timeline by parliament",
        labels={"source_year": "Year", "n_mentions": "Migration country/entity mentions", "source_country": "Parliament"},
    )
    fig.update_layout(hovermode="x unified")
    outputs["interactive_mentions_timeline"] = _save_plotly(
        fig,
        interactive_dir / "interactive_mentions_timeline.html",
    )

    target_df = (
        df
        .filter(pl.col("entity_content").is_in(top_targets))
        .group_by(["source_country", "entity_content"])
        .agg(pl.len().alias("n_mentions"))
        .to_pandas()
    )
    target_pivot = (
        target_df
        .pivot(index="source_country", columns="entity_content", values="n_mentions")
        .fillna(0)
        .reindex(columns=top_targets, fill_value=0)
    )
    fig = px.imshow(
        target_pivot,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Interactive top-target heatmap",
        labels={"x": "Mentioned target", "y": "Speaking parliament", "color": "Mentions"},
    )
    fig.update_traces(hovertemplate="Source=%{y}<br>Target=%{x}<br>Mentions=%{z}<extra></extra>")
    outputs["interactive_target_heatmap"] = _save_plotly(
        fig,
        interactive_dir / "interactive_target_heatmap.html",
    )

    scope_df = (
        df
        .group_by(["source_country", "entity_scope"])
        .agg(pl.len().alias("n_mentions"))
        .with_columns((pl.col("n_mentions") / pl.col("n_mentions").sum().over("source_country")).alias("share"))
        .to_pandas()
    )
    fig = px.bar(
        scope_df,
        x="source_country",
        y="share",
        color="entity_scope",
        title="Interactive entity-scope composition",
        labels={"source_country": "Parliament", "share": "Within-country share", "entity_scope": "Entity scope"},
        hover_data={"n_mentions": ":,", "share": ":.1%"},
    )
    fig.update_layout(yaxis_tickformat=".0%")
    outputs["interactive_scope_composition"] = _save_plotly(
        fig,
        interactive_dir / "interactive_scope_composition.html",
    )

    scatter_df = (
        df
        .group_by("source_country")
        .agg([
            pl.len().alias("n_mentions"),
            pl.col("speech_id").n_unique().alias("n_speeches"),
            pl.col("concreteness_score").mean().alias("mean_concreteness"),
            pl.col("sentence_sentiment_value").mean().alias("mean_sentiment"),
            pl.col("entity_content").mode().first().alias("top_entity"),
            pl.col("policy_measure").mode().first().alias("dominant_policy"),
        ])
        .to_pandas()
    )
    fig = px.scatter(
        scatter_df,
        x="mean_concreteness",
        y="mean_sentiment",
        size="n_mentions",
        color="source_country",
        hover_name="source_country",
        hover_data={
            "n_mentions": ":,",
            "n_speeches": ":,",
            "top_entity": True,
            "dominant_policy": True,
            "mean_concreteness": ":.3f",
            "mean_sentiment": ":.3f",
        },
        title="Interactive concreteness vs sentiment by parliament",
        labels={"mean_concreteness": "Mean concreteness", "mean_sentiment": "Mean sentence sentiment"},
    )
    outputs["interactive_concreteness_sentiment_scatter"] = _save_plotly(
        fig,
        interactive_dir / "interactive_concreteness_sentiment_scatter.html",
    )

    if asymmetry_path and asymmetry_path.exists():
        asym = pl.read_csv(asymmetry_path)
        if not asym.is_empty():
            asym_df = (
                asym
                .with_columns([
                    pl.concat_str([pl.col("source_country"), pl.lit(" -> "), pl.col("target_country")]).alias("pair"),
                    pl.col("attention_log_ratio").abs().alias("abs_attention_log_ratio"),
                ])
                .sort("abs_attention_log_ratio", descending=True)
                .head(250)
                .to_pandas()
            )
            fig = px.scatter(
                asym_df,
                x="attention_log_ratio",
                y="total_traffic",
                size="total_traffic",
                color="source_country",
                hover_name="pair",
                hover_data={
                    "target_country": True,
                    "n_mentions": ":,",
                    "n_mentions_reverse": ":,",
                    "attention_share_AtoB": ":.1%",
                    "sentiment_gap": ":.3f",
                    "total_traffic": ":,",
                },
                title="Interactive reciprocal attention asymmetry",
                labels={"attention_log_ratio": "Attention log-ratio", "total_traffic": "Pair traffic"},
            )
            fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#667085")
            outputs["interactive_asymmetry_scatter"] = _save_plotly(
                fig,
                interactive_dir / "interactive_asymmetry_scatter.html",
            )
    return outputs


def save_cross_country_comparisons(
    country_years: dict[str, list[int]],
    country_dirs: dict[str, Path],
    asymmetry_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Generate direct comparison visualizations across all studied countries."""
    output_dir = output_dir or (PROCESSED_DIR / f"{COMBINED_PREFIX}_comparisons")
    figures_dir = output_dir / "figures_cross_country"
    figures_dir.mkdir(parents=True, exist_ok=True)
    df = combined_mentions(country_years)
    top_targets = _top_targets(df, n=25)

    outputs: dict[str, Path] = {}
    outputs["comparison_summary_table"] = save_comparison_summary_table(
        df,
        output_dir / "comparison_summary_table.html",
    )
    outputs.update(save_interactive_comparison_figures(
        df,
        figures_dir,
        top_targets,
        asymmetry_path=asymmetry_path,
    ))

    year_df = (
        df
        .group_by(["source_country", "source_year"])
        .agg(pl.len().alias("n_mentions"))
        .sort(["source_country", "source_year"])
        .to_pandas()
    )
    outputs["comparison_mentions_by_year_source"] = _save_chart(
        alt.Chart(year_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("source_year:O", title="Year"),
            y=alt.Y("n_mentions:Q", title="Migration country mentions"),
            color=alt.Color("source_country:N", title="Speaking parliament"),
            tooltip=["source_country:N", "source_year:O", "n_mentions:Q"],
        )
        .interactive()
        .properties(title="Migration country mentions by year and parliament", width=820, height=360),
        figures_dir / "comparison_mentions_by_year_source.png",
    )

    total_df = (
        df
        .group_by("source_country")
        .agg([
            pl.len().alias("n_mentions"),
            pl.col("speech_id").n_unique().alias("n_speeches"),
            pl.col("speaker_id").n_unique().alias("n_speakers"),
        ])
        .sort("n_mentions", descending=True)
        .to_pandas()
    )
    outputs["comparison_total_mentions_by_source"] = _save_chart(
        alt.Chart(total_df)
        .mark_bar()
        .encode(
            x=alt.X("n_mentions:Q", title="Migration country mentions"),
            y=alt.Y("source_country:N", title="Speaking parliament", sort="-x"),
            color=alt.Color("source_country:N", legend=None),
            tooltip=["source_country:N", "n_mentions:Q", "n_speeches:Q", "n_speakers:Q"],
        )
        .interactive()
        .properties(title="Total migration country mentions by parliament", width=720, height=280),
        figures_dir / "comparison_total_mentions_by_source.png",
    )

    target_df = (
        df
        .filter(pl.col("entity_content").is_in(top_targets))
        .group_by(["source_country", "entity_content"])
        .agg(pl.len().alias("n_mentions"))
        .to_pandas()
    )
    outputs["comparison_top_targets_heatmap"] = _save_chart(
        alt.Chart(target_df)
        .mark_rect()
        .encode(
            x=alt.X("entity_content:N", title="Mentioned country/entity", sort=top_targets),
            y=alt.Y("source_country:N", title="Speaking parliament"),
            color=alt.Color("n_mentions:Q", title="Mentions", scale=alt.Scale(scheme="blues")),
            tooltip=["source_country:N", "entity_content:N", "n_mentions:Q"],
        )
        .interactive()
        .properties(title="Top mentioned countries/entities compared across parliaments", width=900, height=320),
        figures_dir / "comparison_top_targets_heatmap.png",
    )

    source_concrete_df = (
        df
        .drop_nulls("concreteness_score")
        .group_by("source_country")
        .agg([
            pl.col("concreteness_score").mean().alias("mean_concreteness"),
            pl.col("concreteness_score").median().alias("median_concreteness"),
            pl.len().alias("n_mentions"),
        ])
        .sort("mean_concreteness", descending=True)
        .to_pandas()
    )
    outputs["comparison_mean_concreteness_by_source"] = _save_chart(
        alt.Chart(source_concrete_df)
        .mark_bar()
        .encode(
            x=alt.X("mean_concreteness:Q", title="Mean concreteness"),
            y=alt.Y("source_country:N", title="Speaking parliament", sort="-x"),
            color=alt.Color("source_country:N", legend=None),
            tooltip=["source_country:N", "n_mentions:Q", "mean_concreteness:Q", "median_concreteness:Q"],
        )
        .interactive()
        .properties(title="Mean concreteness by speaking parliament", width=720, height=280),
        figures_dir / "comparison_mean_concreteness_by_source.png",
    )

    target_concrete_df = (
        df
        .filter(pl.col("entity_content").is_in(top_targets))
        .drop_nulls("concreteness_score")
        .group_by(["source_country", "entity_content"])
        .agg([
            pl.col("concreteness_score").mean().round(3).alias("mean_concreteness"),
            pl.len().alias("n_mentions"),
        ])
        .filter(pl.col("n_mentions") >= 5)
        .to_pandas()
    )
    outputs["comparison_concreteness_target_heatmap"] = _save_chart(
        alt.Chart(target_concrete_df)
        .mark_rect()
        .encode(
            x=alt.X("entity_content:N", title="Mentioned country/entity", sort=top_targets),
            y=alt.Y("source_country:N", title="Speaking parliament"),
            color=alt.Color("mean_concreteness:Q", title="Mean concreteness", scale=alt.Scale(scheme="viridis")),
            tooltip=["source_country:N", "entity_content:N", "n_mentions:Q", "mean_concreteness:Q"],
        )
        .interactive()
        .properties(title="Concreteness of shared target countries/entities", width=900, height=320),
        figures_dir / "comparison_concreteness_target_heatmap.png",
    )

    outputs["comparison_agency_composition"] = _stacked_share_chart(
        df,
        "policy_agency_type",
        figures_dir / "comparison_agency_composition.png",
        "Policy agency composition by parliament",
        "Policy agency",
    )
    outputs["comparison_narrative_polarity"] = _stacked_share_chart(
        df,
        "narrative_polarity",
        figures_dir / "comparison_narrative_polarity.png",
        "Narrative polarity composition by parliament",
        "Narrative polarity",
    )
    outputs["comparison_direction_agenda"] = _stacked_share_chart(
        df,
        "migration_direction",
        figures_dir / "comparison_direction_agenda.png",
        "Internal vs external migration direction by parliament",
        "Direction",
    )
    outputs["comparison_migrant_cohorts"] = _stacked_share_chart(
        df,
        "migrant_cohort",
        figures_dir / "comparison_migrant_cohorts.png",
        "Migrant cohort composition by parliament",
        "Migrant cohort",
    )
    outputs["comparison_policy_measure_composition"] = _stacked_share_chart(
        df,
        "policy_measure",
        figures_dir / "comparison_policy_measure_composition.png",
        "Policy measure composition by parliament",
        "Policy measure",
        top_n=10,
    )

    policy_heat_df = (
        df
        .group_by(["source_country", "policy_measure"])
        .agg(pl.len().alias("n_mentions"))
        .with_columns((pl.col("n_mentions") / pl.col("n_mentions").sum().over("source_country")).alias("share"))
        .to_pandas()
    )
    outputs["comparison_policy_measure_heatmap"] = _save_chart(
        alt.Chart(policy_heat_df)
        .mark_rect()
        .encode(
            x=alt.X("policy_measure:N", title="Policy measure"),
            y=alt.Y("source_country:N", title="Speaking parliament"),
            color=alt.Color("share:Q", title="Within-country share", scale=alt.Scale(scheme="greens")),
            tooltip=[
                alt.Tooltip("source_country:N", title="Source"),
                alt.Tooltip("policy_measure:N", title="Policy measure"),
                alt.Tooltip("n_mentions:Q", title="Mentions"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        )
        .interactive()
        .properties(title="Policy measure intensity by parliament", width=860, height=320),
        figures_dir / "comparison_policy_measure_heatmap.png",
    )

    sentiment_df = (
        df
        .group_by("source_country")
        .agg([
            pl.col("sentence_sentiment_value").mean().alias("mean_sentiment"),
            pl.col("sentence_sentiment_value").median().alias("median_sentiment"),
            pl.len().alias("n_mentions"),
        ])
        .sort("mean_sentiment", descending=True)
        .to_pandas()
    )
    outputs["comparison_sentiment_by_source"] = _save_chart(
        alt.Chart(sentiment_df)
        .mark_bar()
        .encode(
            x=alt.X("mean_sentiment:Q", title="Mean sentence sentiment"),
            y=alt.Y("source_country:N", title="Speaking parliament", sort="-x"),
            color=alt.Color("source_country:N", legend=None),
            tooltip=["source_country:N", "n_mentions:Q", "mean_sentiment:Q", "median_sentiment:Q"],
        )
        .interactive()
        .properties(title="Mean sentence sentiment by parliament", width=720, height=280),
        figures_dir / "comparison_sentiment_by_source.png",
    )

    fact_rows = []
    for country, years in country_years.items():
        prefix = runner.country_prefix(country, years)
        path = country_dirs[country] / f"{prefix}_high_concreteness_events.csv"
        if path.exists():
            facts = pl.read_csv(path)
            fact_rows.append({
                "source_country": country,
                "high_concreteness_events": facts.height,
                "unique_entities": facts.get_column("entity_content").n_unique() if "entity_content" in facts.columns else 0,
            })
    fact_df = pd.DataFrame(fact_rows)
    outputs["comparison_high_concreteness_fact_counts"] = _save_chart(
        alt.Chart(fact_df)
        .mark_bar()
        .encode(
            x=alt.X("high_concreteness_events:Q", title="High-concreteness event snippets"),
            y=alt.Y("source_country:N", title="Speaking parliament", sort="-x"),
            color=alt.Color("source_country:N", legend=None),
            tooltip=["source_country:N", "high_concreteness_events:Q", "unique_entities:Q"],
        )
        .interactive()
        .properties(title="High-concreteness event evidence by parliament", width=720, height=280),
        figures_dir / "comparison_high_concreteness_fact_counts.png",
    )

    return outputs


def build_findings_html(country_years: dict[str, list[int]], country_dirs: dict[str, Path]) -> str:
    """Build a detailed computed report for the top of the visualization index."""
    df = combined_mentions(country_years)
    summary = (
        df
        .group_by("source_country")
        .agg([
            pl.len().alias("n_mentions"),
            pl.col("concreteness_score").mean().round(3).alias("mean_concreteness"),
            pl.col("sentence_sentiment_value").mean().round(3).alias("mean_sentiment"),
            pl.col("entity_content").mode().first().alias("top_entity"),
            pl.col("policy_agency_type").mode().first().alias("top_agency"),
            pl.col("narrative_polarity").mode().first().alias("top_narrative"),
            pl.col("migration_direction").mode().first().alias("top_direction"),
        ])
        .sort("n_mentions", descending=True)
    )
    rows = summary.to_dicts()
    volume_top = rows[0]
    concrete_top = summary.sort("mean_concreteness", descending=True).row(0, named=True)
    sentiment_top = summary.sort("mean_sentiment", descending=True).row(0, named=True)
    total_mentions = df.height

    top_entities = (
        df
        .group_by(["entity_content", "entity_scope", "target_iso3"])
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(12)
        .to_dicts()
    )
    top_entity_items = "".join(
        f"<li>{html.escape(str(row['entity_content']))} "
        f"<span class='muted'>({html.escape(str(row['entity_scope']))}, {int(row['n']):,})</span></li>"
        for row in top_entities
    )

    scope_leaders = {}
    scope_summary = (
        df
        .group_by(["source_country", "entity_scope"])
        .agg(pl.len().alias("n"))
        .with_columns((pl.col("n") / pl.col("n").sum().over("source_country")).round(3).alias("share"))
    )
    for scope in ["country", "european_union", "analytical_region", "territory_region"]:
        sub = scope_summary.filter(pl.col("entity_scope") == scope).sort("share", descending=True)
        if not sub.is_empty():
            scope_leaders[scope] = sub.row(0, named=True)

    trend = (
        df
        .group_by(["source_country", "source_year"])
        .agg(pl.len().alias("n_mentions"))
        .sort(["source_country", "source_year"])
    )
    trend_rows = []
    for country in trend.get_column("source_country").unique().sort().to_list():
        sub = trend.filter(pl.col("source_country") == country).sort("source_year")
        if sub.height < 2:
            continue
        first = sub.row(0, named=True)
        last = sub.row(sub.height - 1, named=True)
        peak = sub.sort("n_mentions", descending=True).row(0, named=True)
        trend_rows.append({
            "source_country": country,
            "first_year": first["source_year"],
            "first_mentions": first["n_mentions"],
            "last_year": last["source_year"],
            "last_mentions": last["n_mentions"],
            "delta": last["n_mentions"] - first["n_mentions"],
            "peak_year": peak["source_year"],
            "peak_mentions": peak["n_mentions"],
        })
    trend_top = sorted(trend_rows, key=lambda row: row["delta"], reverse=True)[:6]
    trend_down = sorted(trend_rows, key=lambda row: row["delta"])[:6]
    trend_up_items = "".join(
        f"<li>{html.escape(row['source_country'])}: {int(row['first_mentions']):,} in {row['first_year']} "
        f"to {int(row['last_mentions']):,} in {row['last_year']} "
        f"<span class='muted'>(delta {int(row['delta']):+,}, peak {row['peak_year']}: {int(row['peak_mentions']):,})</span></li>"
        for row in trend_top
    )
    trend_down_items = "".join(
        f"<li>{html.escape(row['source_country'])}: {int(row['first_mentions']):,} in {row['first_year']} "
        f"to {int(row['last_mentions']):,} in {row['last_year']} "
        f"<span class='muted'>(delta {int(row['delta']):+,}, peak {row['peak_year']}: {int(row['peak_mentions']):,})</span></li>"
        for row in trend_down
    )

    fact_counts = []
    for country, years in country_years.items():
        prefix = runner.country_prefix(country, years)
        path = country_dirs[country] / f"{prefix}_high_concreteness_events.csv"
        if path.exists():
            facts = pl.read_csv(path)
            top_fact = (
                facts
                .group_by("entity_content")
                .agg(pl.len().alias("n"))
                .sort("n", descending=True)
                .row(0, named=True)["entity_content"]
                if facts.height else "n/a"
            )
            fact_counts.append({"country": country, "n": facts.height, "top_fact": top_fact})
    fact_counts = sorted(fact_counts, key=lambda row: row["n"], reverse=True)
    fact_top = fact_counts[0] if fact_counts else {"country": "n/a", "n": 0}
    fact_items = "".join(
        f"<li>{html.escape(str(row['country']))}: {int(row['n']):,} snippets "
        f"<span class='muted'>(top: {html.escape(str(row['top_fact']))})</span></li>"
        for row in fact_counts[:8]
    )

    def share_items(column: str, value: str, label: str) -> str:
        share_df = (
            df
            .group_by(["source_country", column])
            .agg(pl.len().alias("n"))
            .with_columns((pl.col("n") / pl.col("n").sum().over("source_country")).round(3).alias("share"))
            .filter(pl.col(column) == value)
            .sort("share", descending=True)
            .head(8)
            .to_dicts()
        )
        return (
            f"<h4>{html.escape(label)}</h4><ul>"
            + "".join(
                f"<li>{html.escape(str(row['source_country']))}: {float(row['share']):.1%} "
                f"<span class='muted'>({int(row['n']):,} mentions)</span></li>"
                for row in share_df
            )
            + "</ul>"
        )

    asym_path = PROCESSED_DIR / f"{COMBINED_PREFIX}_dyadic_data_model" / "asymmetry_table.csv"
    asym_html = "<p class='muted'>Asymmetry table not available yet.</p>"
    if asym_path.exists():
        asym = pl.read_csv(asym_path)
        if not asym.is_empty():
            strongest = (
                asym
                .with_columns(pl.col("attention_log_ratio").abs().alias("abs_attention_log_ratio"))
                .sort("abs_attention_log_ratio", descending=True)
                .head(14)
                .to_dicts()
            )
            asym_html = (
                "<table class='findings-table'><thead><tr>"
                "<th>Direction</th><th>A->B</th><th>B->A</th><th>Log-ratio</th><th>Total traffic</th><th>Sentiment gap</th>"
                "</tr></thead><tbody>"
                + "".join(
                    "<tr>"
                    f"<td>{html.escape(str(row['source_country']))} -> {html.escape(str(row['target_country']))}</td>"
                    f"<td>{int(row['n_mentions']):,}</td>"
                    f"<td>{int(row['n_mentions_reverse']):,}</td>"
                    f"<td>{float(row['attention_log_ratio']):.3f}</td>"
                    f"<td>{int(row['total_traffic']):,}</td>"
                    f"<td>{float(row['sentiment_gap']):.3f}</td>"
                    "</tr>"
                    for row in strongest
                )
                + "</tbody></table>"
            )

    target_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['source_country']))}</td>"
        f"<td>{int(row['n_mentions']):,}</td>"
        f"<td>{html.escape(str(row['top_entity']))}</td>"
        f"<td>{float(row['mean_concreteness']):.3f}</td>"
        f"<td>{html.escape(str(row['top_agency']))}</td>"
        f"<td>{html.escape(str(row['top_narrative']))}</td>"
        f"<td>{html.escape(str(row['top_direction']))}</td>"
        "</tr>"
        for row in rows
    )
    scope_lines = "".join(
        f"<li><strong>{html.escape(scope)} leader:</strong> {html.escape(str(row['source_country']))} "
        f"at {float(row['share']):.1%} <span class='muted'>({int(row['n']):,} mentions)</span></li>"
        for scope, row in scope_leaders.items()
    )
    return (
        "<section class='findings'>"
        "<h2>Analytical Report: Asymmetries, Trends, Special Cases, and LLM Leads</h2>"
        "<p class='section-note'>This report is computed from the current cleaned target-entity tables. It should guide interpretation of the visualizations and identify where close-reading or LLM-assisted analysis is most valuable.</p>"
        "<ul class='findings-list'>"
        f"<li><strong>Corpus scale:</strong> {int(total_mentions):,} cleaned migration country/EU/region mentions across {len(country_years)} parliaments.</li>"
        f"<li><strong>Largest volume:</strong> {html.escape(str(volume_top['source_country']))} has {int(volume_top['n_mentions']):,} migration country/entity mentions in the processed set.</li>"
        f"<li><strong>Highest mean concreteness:</strong> {html.escape(str(concrete_top['source_country']))} has the highest average mention-window concreteness ({float(concrete_top['mean_concreteness']):.3f}).</li>"
        f"<li><strong>Highest mean sentence sentiment:</strong> {html.escape(str(sentiment_top['source_country']))} has the highest mean sentiment value ({float(sentiment_top['mean_sentiment']):.3f}).</li>"
        f"<li><strong>Most high-concreteness event snippets:</strong> {html.escape(str(fact_top['country']))} has {int(fact_top['n']):,} extracted high-concreteness event rows.</li>"
        f"{scope_lines}"
        "<li><strong>Repeated pattern to inspect:</strong> neutral/administrative and external/transnational labels dominate many aggregate views, so the more revealing differences are often in target-specific heatmaps, policy-measure composition, and concrete fact tables.</li>"
        "</ul>"
        "<div class='report-grid'>"
        "<article><h3>Most Visible Targets Overall</h3><ul>" + top_entity_items + "</ul></article>"
        "<article><h3>Strongest Upward Trends</h3><ul>" + trend_up_items + "</ul></article>"
        "<article><h3>Strongest Downward Trends</h3><ul>" + trend_down_items + "</ul></article>"
        "<article><h3>Most Concrete Fact Evidence</h3><ul>" + fact_items + "</ul></article>"
        "</div>"
        "<h3>Strongest Reciprocal Attention Asymmetries</h3>"
        "<p class='section-note'>Positive log-ratio means the listed direction dominates. Negative values in the source table mean the reverse direction dominates; this table sorts by absolute asymmetry.</p>"
        f"{asym_html}"
        "<h3>Comparative Discourse Tendencies</h3>"
        "<div class='report-grid'>"
        "<article>" + share_items("migrant_cohort", "refugees", "Highest refugee-share parliaments") + "</article>"
        "<article>" + share_items("policy_measure", "border_control", "Highest border-control share") + "</article>"
        "<article>" + share_items("policy_measure", "national_security", "Highest national-security share") + "</article>"
        "<article>" + share_items("narrative_polarity", "negative_risk", "Highest negative-risk narrative share") + "</article>"
        "<article>" + share_items("migration_direction", "inbound_internal", "Highest inbound/internal direction share") + "</article>"
        "<article>" + share_items("policy_agency_type", "coercion_intervention_to", "Highest coercion/intervention agency share") + "</article>"
        "</div>"
        "<h3>What Is Especially Promising To Study With LLMs</h3>"
        "<ol class='findings-list'>"
        "<li><strong>Why are some dyads asymmetric?</strong> For the top asymmetric pairs, prompt an LLM to classify whether the asymmetry is driven by border proximity, crisis visibility, EU accession/governance, war, diaspora, or historical memory.</li>"
        "<li><strong>Event-level visibility:</strong> For high-concreteness snippets, ask an LLM to extract event names, responsible actors, routes, affected cohorts, and whether the event is used as evidence for restriction, solidarity, burden-sharing, or institutional reform.</li>"
        "<li><strong>Model vs pressure references:</strong> For country pairs with high policy agency, classify whether another country is treated as a model to emulate, a warning case, a cooperation partner, or an object of pressure.</li>"
        "<li><strong>Same target, different imagination:</strong> Compare how several parliaments talk about the same target such as Turkey, Ukraine, Syria, Libya, the EU, or the Mediterranean; ask the LLM to summarize differences in risk, solidarity, administration, and policy remedy.</li>"
        "<li><strong>Domestic vs external agenda:</strong> For parliaments with high inbound/internal shares, ask whether migration is framed as capacity pressure, legal obligation, security issue, demographic need, or humanitarian responsibility.</li>"
        "</ol>"
        "<table class='findings-table'><thead><tr>"
        "<th>Source</th><th>Mentions</th><th>Most mentioned</th><th>Mean concreteness</th><th>Dominant agency</th><th>Dominant narrative</th><th>Dominant direction</th>"
        "</tr></thead><tbody>"
        f"{target_rows}"
        "</tbody></table>"
        "</section>"
    )


def _explanation_for_path(path: Path) -> dict[str, str]:
    """Return human-readable explanation metadata for a figure path."""
    stem = path.stem
    explanation = VISUAL_EXPLANATIONS.get(stem)
    if explanation:
        return explanation
    base = stem.replace("_", " ").title()
    return {
        "title": base,
        "question": "This figure is part of the migration discourse analysis bundle.",
        "read": "Open the linked file and use hover/click interactions where available. Compare it with the country profile and context tables before drawing a substantive conclusion.",
    }


def _relative_display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _figure_cards(section_dir: Path, root: Path, link_root: Path) -> list[str]:
    """Return HTML cards for all HTML/PNG figures under a directory."""
    cards = []
    for path in sorted(section_dir.rglob("*.html")) + sorted(section_dir.rglob("*.png")):
        label = _relative_display_path(path, root)
        href = _relative_display_path(path, link_root)
        explanation = _explanation_for_path(path)
        cards.append(
            "<article class='figure-card'>"
            f"<h3><a href='{html.escape(href)}'>{html.escape(explanation['title'])}</a></h3>"
            f"<p class='path'>{html.escape(label)}</p>"
            f"<p><strong>What it asks:</strong> {html.escape(explanation['question'])}</p>"
            f"<p><strong>How to read it:</strong> {html.escape(explanation['read'])}</p>"
            "</article>"
        )
    return cards


def save_visualization_index(
    country_dirs: dict[str, Path],
    combined_dir: Path,
    comparison_dir: Path | None = None,
    findings_html: str = "",
    output_path: Path | None = None,
) -> Path:
    """Save a browser index of all generated visualizations."""
    output_path = output_path or (PROCESSED_DIR / "all_studied_countries_visualization_index.html")
    link_root = output_path.parent
    blocks = []
    for country, country_dir in country_dirs.items():
        cards = []
        for folder in ["figures_altair_extended", "figures_interactive_advanced"]:
            cards.extend(_figure_cards(country_dir / folder, country_dir, link_root))
        blocks.append(
            "<section>"
            f"<h2>{html.escape(country)}</h2>"
            f"<p class='section-path'>{html.escape(_relative_display_path(country_dir, link_root))}</p>"
            f"<div class='card-grid'>{''.join(cards) if cards else '<p>No figures found</p>'}</div>"
            "</section>"
        )

    if comparison_dir is not None:
        comparison_cards = _figure_cards(comparison_dir, comparison_dir.parent, link_root)
        comparison_table = comparison_dir.parent / "comparison_summary_table.html"
        if comparison_table.exists():
            comparison_cards.insert(0, _figure_cards(comparison_dir.parent, comparison_dir.parent, link_root)[0])
        blocks.insert(
            0,
            "<section>"
            "<h2>Cross-country comparisons</h2>"
            f"<p class='section-path'>{html.escape(_relative_display_path(comparison_dir.parent, link_root))}</p>"
            "<p class='section-note'>These figures compare the five studied parliaments directly. They are the main place to look for differences and repeated patterns across countries.</p>"
            f"<div class='card-grid'>{''.join(comparison_cards) if comparison_cards else '<p>No figures found</p>'}</div>"
            "</section>",
        )

    combined_cards = _figure_cards(combined_dir / "figures_data_model", combined_dir, link_root)
    blocks.append(
        "<section>"
        "<h2>Combined dyadic model</h2>"
        f"<p class='section-path'>{html.escape(_relative_display_path(combined_dir, link_root))}</p>"
        "<p class='section-note'>These visualizations use the resolved source-country to target-country model. They answer who looks at whom, how concretely, and how attention changes around shocks.</p>"
        f"<div class='card-grid'>{''.join(combined_cards) if combined_cards else '<p>No figures found</p>'}</div>"
        "</section>"
    )

    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>All Studied Countries: Migration Visualizations</title>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:28px;background:#f8fafc;color:#1f2937;}"
        "h1{font-size:28px;margin:0 0 6px;} .sub{color:#52606d;margin:0 0 20px;max-width:980px;line-height:1.45;}"
        "section{background:#fff;border:1px solid #d9e2ec;border-radius:8px;padding:16px 18px;margin:16px 0;}"
        "h2{font-size:20px;margin:0 0 4px;} .section-path,.path{color:#697586;font-size:12px;margin:0 0 10px;}"
        ".section-note{color:#364152;margin:0 0 14px;line-height:1.45;} .card-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:12px;}"
        ".figure-card{border:1px solid #d9e2ec;border-radius:8px;padding:12px;background:#fbfdff;}"
        ".figure-card h3{font-size:15px;margin:0 0 5px;} .figure-card p{font-size:13px;line-height:1.42;margin:7px 0;color:#344054;}"
        ".findings{border-color:#b6c7d6;background:#f7fbff;} .findings-list{margin:0 0 14px 20px;line-height:1.55;}"
        ".report-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;margin:12px 0;} .report-grid article{background:#fff;border:1px solid #d9e2ec;border-radius:8px;padding:12px;} .report-grid h3,.report-grid h4{margin:0 0 8px;font-size:15px;} .report-grid ul{margin:0 0 0 18px;line-height:1.5;} .muted{color:#667085;font-size:12px;}"
        ".findings-table{border-collapse:collapse;width:100%;background:#fff;font-size:13px;} .findings-table th,.findings-table td{border:1px solid #d9e2ec;padding:7px;text-align:left;} .findings-table th{background:#e5e9f0;}"
        "a{color:#1d4ed8;text-decoration:none;} a:hover{text-decoration:underline;}"
        "</style></head><body>"
        "<h1>All Studied Countries: Migration Visualizations</h1>"
        "<p class='sub'>This index explains every generated visualization for all available studied parliaments. Start with the cross-country comparisons, then open country-specific charts for detail, and finally inspect the dyadic source-target model for asymmetry and shock windows.</p>"
        f"{findings_html}"
        f"{''.join(blocks)}"
        "</body></html>"
    )
    output_path.write_text(document, encoding="utf-8")
    return output_path


def main() -> None:
    country_years = available_years_by_country()
    country_dirs = ensure_country_visualizations(country_years, rerun_existing=False)
    combined_dir = PROCESSED_DIR / f"{COMBINED_PREFIX}_dyadic_data_model"
    outputs = generate_combined_dyadic_visualizations(country_years, combined_dir)
    comparison_outputs = save_cross_country_comparisons(
        country_years,
        country_dirs,
        asymmetry_path=outputs.get("asymmetry_table"),
    )
    comparison_dir = PROCESSED_DIR / f"{COMBINED_PREFIX}_comparisons" / "figures_cross_country"
    findings_html = build_findings_html(country_years, country_dirs)
    index_path = save_visualization_index(
        country_dirs,
        combined_dir,
        comparison_dir=comparison_dir,
        findings_html=findings_html,
    )

    print(f"visualization_index: {index_path}")
    for country, path in country_dirs.items():
        print(f"{country}: {path}")
    for name, path in comparison_outputs.items():
        print(f"{name}: {path}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
