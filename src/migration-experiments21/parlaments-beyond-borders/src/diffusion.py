"""Migrant cohort classification and policy-diffusion network construction."""

from __future__ import annotations

from pathlib import Path

import polars as pl


# Explanation: These transparent keyword sets are a local substitute for zero-shot
# classification. They can be replaced later with DeBERTa/SetFit outputs while
# keeping the same output columns.
COHORT_MARKERS = {
    "refugees": {
        "refugee", "refugees", "humanitarian protection", "subsidiary protection",
        "displaced", "war refugees",
    },
    "asylum_seekers": {
        "asylum seeker", "asylum seekers", "asylum-seeker", "asylum-seekers",
        "application for asylum", "asylum application", "request for asylum",
    },
    "students": {
        "student", "students", "university", "universities", "study visa",
        "foreign student", "foreign students",
    },
    "economic_migrants": {
        "economic migrant", "economic migrants", "labour migrant", "labor migrant",
        "workers", "seasonal workers", "work permit", "employment",
    },
    "high_skilled_workers": {
        "high-skilled", "high skilled", "talent passport", "researcher",
        "researchers", "engineer", "engineers", "qualified worker",
        "qualified workers",
    },
}

POLICY_MEASURE_MARKERS = {
    "international_law": {
        "international law", "geneva convention", "convention", "treaty",
        "european convention", "human rights", "dublin regulation",
    },
    "national_security": {
        "security", "terrorism", "terrorist", "radicalisation", "radicalization",
        "public order", "threat", "criminal", "crime",
    },
    "border_control": {
        "border", "borders", "frontier", "coast guard", "control", "controls",
        "crossing", "entry", "schengen",
    },
    "allocation_of_resources": {
        "resources", "housing", "shelter", "places", "budget", "funding",
        "cost", "costs", "reception capacity",
    },
    "integration": {
        "integration", "language", "school", "education", "training",
        "employment", "housing", "citizenship",
    },
    "returns_deportation": {
        "return", "returns", "deportation", "deportations", "expulsion",
        "expulsions", "removal", "readmission",
    },
    "asylum_procedure": {
        "asylum procedure", "procedure", "application", "applications",
        "appeal", "processing", "office for refugees",
    },
    "visas_mobility": {
        "visa", "visas", "residence permit", "permit", "mobility",
        "regularization", "regularisation",
    },
}


def _contains_any(text: str | None, markers: set[str]) -> bool:
    """Return True when any marker appears in text."""
    # Explanation: Markers are phrases, so simple lowercase substring matching is transparent.
    if not isinstance(text, str):
        return False
    lower = text.lower()
    return any(marker in lower for marker in markers)


def _first_marker(text: str | None, markers: set[str]) -> str:
    """Return the first matched marker in a deterministic order."""
    # Explanation: Marker evidence lets colleagues inspect why a label was assigned.
    if not isinstance(text, str):
        return ""
    lower = text.lower()
    for marker in sorted(markers, key=len, reverse=True):
        if marker in lower:
            return marker
    return ""


def classify_migrant_cohort(text: str | None) -> str:
    """Classify the migrant cohort discussed in a context window."""
    # Explanation: First matching cohort wins; order follows the research categories.
    for cohort, markers in COHORT_MARKERS.items():
        if _contains_any(text, markers):
            return cohort
    return "general_migration"


def matched_migrant_cohort_marker(text: str | None) -> str:
    """Return the keyword/phrase that triggered the migrant cohort label."""
    for markers in COHORT_MARKERS.values():
        marker = _first_marker(text, markers)
        if marker:
            return marker
    return ""


def classify_policy_measure(text: str | None) -> str:
    """Classify the policy measure/frame discussed in a context window."""
    # Explanation: This approximates a CAP/narrative-frame taxonomy with auditable rules.
    for measure, markers in POLICY_MEASURE_MARKERS.items():
        if _contains_any(text, markers):
            return measure
    return "general_policy"


def matched_policy_measure_marker(text: str | None) -> str:
    """Return the keyword/phrase that triggered the policy measure label."""
    for markers in POLICY_MEASURE_MARKERS.values():
        marker = _first_marker(text, markers)
        if marker:
            return marker
    return ""


def add_diffusion_classifications(df: pl.DataFrame) -> pl.DataFrame:
    """Add migrant cohort and policy measure labels to mentions."""
    # Explanation: These columns become edge attributes in the diffusion network.
    return df.with_columns([
        pl.col("context_window")
        .map_elements(classify_migrant_cohort, return_dtype=pl.Utf8)
        .alias("migrant_cohort"),
        pl.col("context_window")
        .map_elements(matched_migrant_cohort_marker, return_dtype=pl.Utf8)
        .alias("migrant_cohort_marker"),
        pl.col("context_window")
        .map_elements(classify_policy_measure, return_dtype=pl.Utf8)
        .alias("policy_measure"),
        pl.col("context_window")
        .map_elements(matched_policy_measure_marker, return_dtype=pl.Utf8)
        .alias("policy_measure_marker"),
        pl.lit("keyword_rules").alias("cohort_policy_method"),
    ])


def build_diffusion_edges(df: pl.DataFrame, source_country: str = "FRA") -> pl.DataFrame:
    """Build weighted source-country -> mentioned-country diffusion edges."""
    # Explanation: The speaker/source is France in this pilot; the target is the
    # mentioned country/entity. We keep EU and overseas labels explicit.
    return (
        df
        .with_columns([
            pl.lit(source_country).alias("source_country"),
            pl.col("entity_content").alias("target_entity"),
        ])
        .group_by([
            "source_country",
            "target_entity",
            "target_iso3",
            "geo_class",
            "region_group",
            "weog_group",
            "source_year",
            "migrant_cohort",
            "policy_measure",
            "ref_type",
        ])
        .agg([
            pl.len().alias("weight"),
            pl.col("sentence_id").n_unique().alias("n_sentences"),
            pl.col("migrant_cohort_marker").drop_nulls().unique().str.join(", ").alias("cohort_markers"),
            pl.col("policy_measure_marker").drop_nulls().unique().str.join(", ").alias("policy_markers"),
        ])
        .sort(["weight", "target_entity"], descending=[True, False])
    )


def build_target_summary(edges: pl.DataFrame) -> pl.DataFrame:
    """Aggregate diffusion edges to one row per mentioned target."""
    # Explanation: This summary is easier to read than the full cohort/measure edge table.
    return (
        edges
        .group_by(["target_entity", "target_iso3", "geo_class", "region_group", "weog_group"])
        .agg([
            pl.col("weight").sum().alias("total_mentions"),
            pl.col("migrant_cohort").mode().first().alias("top_cohort"),
            pl.col("policy_measure").mode().first().alias("top_policy_measure"),
            pl.col("ref_type").mode().first().alias("top_ref_type"),
        ])
        .sort("total_mentions", descending=True)
    )


def to_networkx_multidigraph(edges: pl.DataFrame):
    """Convert the edge table to a NetworkX MultiDiGraph."""
    # Explanation: MultiDiGraph preserves separate cohort/measure edges between
    # the same source and target.
    import networkx as nx

    graph = nx.MultiDiGraph()
    for row in edges.to_dicts():
        source = row["source_country"]
        target = row["target_entity"]
        graph.add_node(source, node_type="source_country")
        graph.add_node(
            target,
            node_type=row.get("geo_class") or "target",
            iso3=row.get("target_iso3") or "",
            region_group=row.get("region_group") or "",
            weog_group=row.get("weog_group") or "",
        )
        graph.add_edge(
            source,
            target,
            weight=int(row["weight"]),
            source_year=int(row["source_year"]),
            migrant_cohort=row["migrant_cohort"],
            policy_measure=row["policy_measure"],
            ref_type=row["ref_type"],
        )
    return graph


def save_diffusion_outputs(
    edges: pl.DataFrame,
    processed_dir: Path,
    prefix: str = "FRA_2017_2022",
) -> dict[str, Path]:
    """Save diffusion edge CSV, target summary CSV, and GraphML network."""
    # Explanation: CSVs are for audit; GraphML can be opened in Gephi/Cytoscape.
    processed_dir.mkdir(parents=True, exist_ok=True)
    edge_path = processed_dir / f"{prefix}_diffusion_edges.csv"
    summary_path = processed_dir / f"{prefix}_diffusion_target_summary.csv"
    graphml_path = processed_dir / f"{prefix}_diffusion_network.graphml"

    edges.write_csv(edge_path)
    build_target_summary(edges).write_csv(summary_path)

    graph = to_networkx_multidigraph(edges)
    import networkx as nx

    nx.write_graphml(graph, graphml_path)
    return {
        "edges_csv": edge_path,
        "target_summary_csv": summary_path,
        "graphml": graphml_path,
    }
