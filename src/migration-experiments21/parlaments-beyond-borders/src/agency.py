"""Policy diffusion agency mechanisms for country mentions."""

from __future__ import annotations

from pathlib import Path

import polars as pl


# Explanation: These categories operationalize whether another country is treated
# as a model, target of pressure, competitor, partner, or neutral event location.
AGENCY_MARKERS = {
    "learning_emulation_from": {
        "model", "example", "learn from", "inspired by", "as in", "like in",
        "follow", "adopt", "emulate", "copy", "best practice", "successful",
        "what works", "experience of", "system in",
    },
    "coercion_intervention_to": {
        "force", "pressure", "sanction", "comply", "must accept", "must take",
        "oblige", "impose", "intervention", "intervene", "require", "demand",
        "send back", "return to", "readmission",
    },
    "competition": {
        "attractive", "compete", "competition", "more attractive", "less attractive",
        "brain drain", "talent", "rank", "compared with", "relative to",
        "better than", "worse than",
    },
    "exchange_cooperation": {
        "cooperate", "cooperation", "agreement", "treaty", "joint", "partnership",
        "shared", "together", "bilateral", "multilateral", "exchange",
        "coordination", "solidarity mechanism", "european mechanism",
    },
}

AGENCY_ORDER = [
    "learning_emulation_from",
    "coercion_intervention_to",
    "competition",
    "exchange_cooperation",
    "neutral_reporting",
]


def _first_marker(text: str | None, markers: set[str]) -> str:
    """Return first matched marker, preferring longer phrases."""
    # Explanation: Longer first avoids matching "model" before "successful model" if added.
    if not isinstance(text, str):
        return ""
    lower = text.lower()
    for marker in sorted(markers, key=len, reverse=True):
        if marker in lower:
            return marker
    return ""


def classify_policy_agency(text: str | None) -> str:
    """Classify policy diffusion agency type for one country mention."""
    # Explanation: Rules are ordered from strongest agency claims to neutral reporting.
    for agency_type in AGENCY_ORDER[:-1]:
        if _first_marker(text, AGENCY_MARKERS[agency_type]):
            return agency_type
    return "neutral_reporting"


def matched_policy_agency_marker(text: str | None) -> str:
    """Return the marker that triggered the policy agency label."""
    for agency_type in AGENCY_ORDER[:-1]:
        marker = _first_marker(text, AGENCY_MARKERS[agency_type])
        if marker:
            return marker
    return ""


def agency_prompt(text: str | None, entity: str | None) -> str:
    """Return an LLM-ready prompt for replacing the rule label later."""
    # Explanation: This lets Qwen/Llama classify the same schema without changing downstream code.
    excerpt = "" if text is None else " ".join(text.replace("||", " ").split())
    return (
        "Classify the policy diffusion agency of this country/entity mention. "
        "Labels: learning_emulation_from, coercion_intervention_to, competition, "
        "exchange_cooperation, neutral_reporting. "
        f"Mentioned entity: {entity}. Context: {excerpt}"
    )


def add_policy_agency(df: pl.DataFrame) -> pl.DataFrame:
    """Add policy diffusion agency labels and evidence markers."""
    # Explanation: policy_agency_type is the current rule-based label; prompt is for LLM upgrades.
    return df.with_columns([
        pl.col("context_window")
        .map_elements(classify_policy_agency, return_dtype=pl.Utf8)
        .alias("policy_agency_type"),
        pl.col("context_window")
        .map_elements(matched_policy_agency_marker, return_dtype=pl.Utf8)
        .alias("policy_agency_marker"),
        pl.struct(["context_window", "entity_content"])
        .map_elements(
            lambda row: agency_prompt(row["context_window"], row["entity_content"]),
            return_dtype=pl.Utf8,
        )
        .alias("policy_agency_llm_prompt"),
        pl.lit("keyword_rules_llm_ready").alias("policy_agency_method"),
    ])


def build_agency_edges(df: pl.DataFrame, source_country: str = "FRA") -> pl.DataFrame:
    """Build directed dyadic network edges with agency and cohort attributes."""
    # Explanation: This is the deeper policy-agency version of the diffusion edge table.
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
            "policy_agency_type",
            "migrant_cohort",
            "policy_measure",
            "ref_type",
        ])
        .agg([
            pl.len().alias("weight"),
            pl.col("sentence_id").n_unique().alias("n_sentences"),
            pl.col("policy_agency_marker").drop_nulls().unique().str.join(", ").alias("agency_markers"),
            pl.col("migrant_cohort_marker").drop_nulls().unique().str.join(", ").alias("cohort_markers"),
            pl.col("policy_measure_marker").drop_nulls().unique().str.join(", ").alias("policy_markers"),
        ])
        .sort(["weight", "target_entity"], descending=[True, False])
    )


def to_networkx_agency_multidigraph(edges: pl.DataFrame):
    """Convert agency edges to a NetworkX MultiDiGraph."""
    # Explanation: Separate agency/cohort edges preserve multiple mechanisms per target.
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
            policy_agency_type=row["policy_agency_type"],
            migrant_cohort=row["migrant_cohort"],
            policy_measure=row["policy_measure"],
            ref_type=row["ref_type"],
        )
    return graph


def save_agency_outputs(
    edges: pl.DataFrame,
    processed_dir: Path,
    prefix: str = "FRA_2017_2022",
) -> dict[str, Path]:
    """Save policy agency edge CSV and GraphML network."""
    # Explanation: This network distinguishes FROM/TO/exchange/competition mechanisms.
    processed_dir.mkdir(parents=True, exist_ok=True)
    edge_path = processed_dir / f"{prefix}_policy_agency_edges.csv"
    graphml_path = processed_dir / f"{prefix}_policy_agency_network.graphml"
    edges.write_csv(edge_path)

    import networkx as nx

    nx.write_graphml(to_networkx_agency_multidigraph(edges), graphml_path)
    return {
        "policy_agency_edges_csv": edge_path,
        "policy_agency_graphml": graphml_path,
    }
