"""Regenerate the requested advanced migration-discourse visualizations.

This script uses the processed France 2017-2022 tables produced by the extended
analysis notebook and writes the policy-agency, narrative, evidence-map, and
fact-timeline figures into data/processed/figures_interactive_advanced/.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import quote

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import agency, events as event_tools, geo, load, visualize  # noqa: E402
from src.config import EU_ENTITIES, EUROPEAN_COUNTRIES, PROCESSED_DIR  # noqa: E402


PREFIX = "FRA_2017_2022"

EUROPE_PLACE_GAZETTEER = [
    {"place_name": "Calais", "aliases": ["Calais", "Calaisis"], "latitude": 50.9513, "longitude": 1.8587, "country": "France", "place_kind": "city/border area"},
    {"place_name": "Grande-Synthe", "aliases": ["Grande-Synthe", "Grande Synthe"], "latitude": 51.0156, "longitude": 2.3021, "country": "France", "place_kind": "camp/city"},
    {"place_name": "Dunkirk", "aliases": ["Dunkirk", "Dunkerque"], "latitude": 51.0344, "longitude": 2.3768, "country": "France", "place_kind": "port city"},
    {"place_name": "Sangatte", "aliases": ["Sangatte"], "latitude": 50.9458, "longitude": 1.7536, "country": "France", "place_kind": "border area"},
    {"place_name": "Ouistreham", "aliases": ["Ouistreham"], "latitude": 49.2776, "longitude": -0.2597, "country": "France", "place_kind": "port city"},
    {"place_name": "Paris", "aliases": ["Paris", "La Chapelle", "Porte de la Chapelle", "Aubervilliers"], "latitude": 48.8566, "longitude": 2.3522, "country": "France", "place_kind": "city"},
    {"place_name": "Menton", "aliases": ["Menton"], "latitude": 43.7745, "longitude": 7.4975, "country": "France", "place_kind": "border city"},
    {"place_name": "Briancon", "aliases": ["Briançon", "Briancon"], "latitude": 44.8994, "longitude": 6.6432, "country": "France", "place_kind": "alpine border city"},
    {"place_name": "Col de l'Echelle", "aliases": ["Col de l'Échelle", "Col de l'Echelle"], "latitude": 45.0325, "longitude": 6.6814, "country": "France", "place_kind": "mountain pass"},
    {"place_name": "Roya Valley", "aliases": ["Roya Valley", "Roya"], "latitude": 43.9387, "longitude": 7.5144, "country": "France", "place_kind": "border valley"},
    {"place_name": "Ventimiglia", "aliases": ["Ventimiglia", "Vintimille"], "latitude": 43.7912, "longitude": 7.6076, "country": "Italy", "place_kind": "border city"},
    {"place_name": "Lampedusa", "aliases": ["Lampedusa"], "latitude": 35.5010, "longitude": 12.6091, "country": "Italy", "place_kind": "island"},
    {"place_name": "Sicily", "aliases": ["Sicily", "Sicile"], "latitude": 37.5999, "longitude": 14.0154, "country": "Italy", "place_kind": "region/island"},
    {"place_name": "Rome", "aliases": ["Rome"], "latitude": 41.9028, "longitude": 12.4964, "country": "Italy", "place_kind": "city"},
    {"place_name": "Samos", "aliases": ["Samos", "Samos Island"], "latitude": 37.7547, "longitude": 26.9778, "country": "Greece", "place_kind": "island"},
    {"place_name": "Lesbos", "aliases": ["Lesbos", "Lesvos"], "latitude": 39.1641, "longitude": 26.3722, "country": "Greece", "place_kind": "island"},
    {"place_name": "Moria", "aliases": ["Moria", "Moria camp"], "latitude": 39.1322, "longitude": 26.5162, "country": "Greece", "place_kind": "camp"},
    {"place_name": "Crete", "aliases": ["Crete", "Crète"], "latitude": 35.2401, "longitude": 24.8093, "country": "Greece", "place_kind": "island"},
    {"place_name": "Aegean Sea", "aliases": ["Aegean Sea", "Aegean"], "latitude": 38.5, "longitude": 25.0, "country": "Greece/Turkey", "place_kind": "sea route"},
    {"place_name": "Evros", "aliases": ["Evros"], "latitude": 41.2444, "longitude": 26.1359, "country": "Greece", "place_kind": "border region"},
    {"place_name": "Idomeni", "aliases": ["Idomeni"], "latitude": 41.1231, "longitude": 22.5181, "country": "Greece", "place_kind": "border village"},
    {"place_name": "English Channel", "aliases": ["English Channel", "the Channel", "Channel"], "latitude": 50.35, "longitude": 0.0, "country": "France/United Kingdom", "place_kind": "sea route"},
    {"place_name": "Dover", "aliases": ["Dover"], "latitude": 51.1279, "longitude": 1.3134, "country": "United Kingdom", "place_kind": "port city"},
    {"place_name": "Kent", "aliases": ["Kent"], "latitude": 51.2787, "longitude": 0.5217, "country": "United Kingdom", "place_kind": "region"},
    {"place_name": "London", "aliases": ["London"], "latitude": 51.5072, "longitude": -0.1276, "country": "United Kingdom", "place_kind": "city"},
    {"place_name": "Sandhurst", "aliases": ["Sandhurst"], "latitude": 51.3469, "longitude": -0.8038, "country": "United Kingdom", "place_kind": "agreement place"},
    {"place_name": "Ceuta", "aliases": ["Ceuta"], "latitude": 35.8894, "longitude": -5.3213, "country": "Spain", "place_kind": "border enclave"},
    {"place_name": "Melilla", "aliases": ["Melilla"], "latitude": 35.2923, "longitude": -2.9381, "country": "Spain", "place_kind": "border enclave"},
    {"place_name": "Canary Islands", "aliases": ["Canary Islands", "Canaries"], "latitude": 28.2916, "longitude": -16.6291, "country": "Spain", "place_kind": "islands"},
    {"place_name": "Belarus-Poland border", "aliases": ["Belarus-Poland border", "Polish-Belarusian border", "border with Belarus"], "latitude": 52.1, "longitude": 23.7, "country": "Poland/Belarus", "place_kind": "border"},
    {"place_name": "Brest", "aliases": ["Brest"], "latitude": 52.0976, "longitude": 23.7341, "country": "Belarus", "place_kind": "border city"},
    {"place_name": "Mediterranean Sea", "aliases": ["Mediterranean Sea", "Mediterranean", "Central Mediterranean"], "latitude": 36.0, "longitude": 15.0, "country": "Mediterranean", "place_kind": "sea route"},
    {"place_name": "Balkan route", "aliases": ["Balkan route", "Balkans", "Balkan"], "latitude": 44.0, "longitude": 20.5, "country": "Balkans", "place_kind": "migration route"},
    {"place_name": "Dublin", "aliases": ["Dublin"], "latitude": 53.3498, "longitude": -6.2603, "country": "Ireland", "place_kind": "city/policy reference"},
    {"place_name": "Brussels", "aliases": ["Brussels", "Bruxelles"], "latitude": 50.8503, "longitude": 4.3517, "country": "Belgium", "place_kind": "EU city"},
    {"place_name": "Berlin", "aliases": ["Berlin"], "latitude": 52.52, "longitude": 13.405, "country": "Germany", "place_kind": "city"},
    {"place_name": "Munich", "aliases": ["Munich", "München"], "latitude": 48.1351, "longitude": 11.5820, "country": "Germany", "place_kind": "city"},
    {"place_name": "Vienna", "aliases": ["Vienna", "Vienne"], "latitude": 48.2082, "longitude": 16.3738, "country": "Austria", "place_kind": "city"},
    {"place_name": "Vilnius", "aliases": ["Vilnius"], "latitude": 54.6872, "longitude": 25.2797, "country": "Lithuania", "place_kind": "city"},
    {"place_name": "Kyiv", "aliases": ["Kyiv", "Kiev"], "latitude": 50.4501, "longitude": 30.5234, "country": "Ukraine", "place_kind": "city"},
]


def _read_required_parquet(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing processed parquet: {path}")
    return pl.read_parquet(path)


def _read_required_csv(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing processed CSV: {path}")
    return pl.read_csv(path)


def _normalized_speaker_id_expr(column: str) -> pl.Expr:
    return pl.col(column).str.replace_all(r"[^A-Za-z0-9]", "")


def enrich_with_party_metadata(df: pl.DataFrame) -> pl.DataFrame:
    """Attach party group and CHES orientation from ParlaMint affiliation tables."""
    affiliations = (
        load.load_affiliations()
        .filter((pl.col("country") == "FRA") & (pl.col("role") == "member"))
        .with_columns([
            _normalized_speaker_id_expr("speaker_id").alias("speaker_id_norm"),
            pl.col("start_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("affiliation_start_date"),
            pl.col("end_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("affiliation_end_date"),
        ])
    )
    organizations = (
        load.load_orgs()
        .filter((pl.col("country") == "FRA") & (pl.col("org_role") == "parliamentaryGroup"))
        .select([
            "org_id",
            pl.col("org_name").alias("party_group"),
            "party_ches_id",
            "party_left_right_orientation",
        ])
    )
    party_affiliations = affiliations.join(organizations, on="org_id", how="inner")

    enriched = (
        df
        .with_row_index("_row_id")
        .with_columns(_normalized_speaker_id_expr("speaker_id").alias("speaker_id_norm"))
        .join(party_affiliations, on="speaker_id_norm", how="left", suffix="_affiliation")
        .with_columns(
            (
                (pl.col("affiliation_start_date").is_null() | (pl.col("session_date") >= pl.col("affiliation_start_date")))
                & (pl.col("affiliation_end_date").is_null() | (pl.col("session_date") <= pl.col("affiliation_end_date")))
                & pl.col("party_group").is_not_null()
            ).alias("valid_party_affiliation")
        )
        .sort(["_row_id", "valid_party_affiliation", "affiliation_start_date"], descending=[False, True, True])
        .unique(subset=["_row_id"], keep="first", maintain_order=True)
        .with_columns([
            pl.when(pl.col("valid_party_affiliation")).then(pl.col("party_group")).otherwise(None).alias("party_group"),
            pl.when(pl.col("valid_party_affiliation")).then(pl.col("party_ches_id")).otherwise(None).alias("party_ches_id"),
            pl.when(pl.col("valid_party_affiliation")).then(pl.col("party_left_right_orientation")).otherwise(None).alias("party_left_right_orientation"),
            pl.col("speaker_ana").alias("speaker_role"),
        ])
        .drop([
            "_row_id",
            "speaker_id_norm",
            "country_affiliation",
            "org_id",
            "role",
            "start_date",
            "end_date",
            "affiliation_start_date",
            "affiliation_end_date",
            "valid_party_affiliation",
        ], strict=False)
    )
    return enriched


def add_fact_concreteness_score(events: pl.DataFrame) -> pl.DataFrame:
    """Give named-entity anchored snippets the maximum fact score for strict views."""
    has_fact_anchor = (
        pl.col("proper_noun_anchors").fill_null("").str.len_chars() > 0
    ) | (
        pl.col("countries_detected_in_context").fill_null("").str.len_chars() > 0
    ) | (
        pl.col("country_iso3_detected_in_context").fill_null("").str.len_chars() > 0
    ) | (
        pl.col("session_date").is_not_null()
    )
    return events.with_columns([
        pl.col("concreteness_score").alias("original_concreteness_score"),
        pl.when(has_fact_anchor)
        .then(pl.lit(5.0))
        .otherwise(pl.col("concreteness_score"))
        .alias("fact_concreteness_score"),
        pl.when(pl.col("proper_noun_anchors").fill_null("").str.len_chars() > 0)
        .then(pl.col("proper_noun_anchors"))
        .otherwise(pl.col("entity_content"))
        .alias("event_label"),
    ])


def europe_iso3_codes() -> list[str]:
    """Return ISO3 codes for configured European countries."""
    return sorted({
        iso3
        for country in EUROPEAN_COUNTRIES
        for iso3 in [geo.iso3_for_entity(country)]
        if iso3
    })


def concrete_event_mentions_table(events: pl.DataFrame) -> pl.DataFrame:
    """Return a readable table of all concrete event mentions."""
    return (
        events
        .sort(["session_date", "entity_content"])
        .with_columns(
            pl.col("context_window")
            .map_elements(lambda text: " ".join(str(text).replace("||", " ").split())[:900], return_dtype=pl.Utf8)
            .alias("context_excerpt")
        )
        .select([
            "source_year",
            "session_date",
            "entity_content",
            "target_iso3",
            "event_label",
            "proper_noun_anchors",
            "countries_detected_in_context",
            "country_iso3_detected_in_context",
            "original_concreteness_score",
            "fact_concreteness_score",
            "narrative_frame",
            "policy_agency_type",
            "migrant_cohort",
            "policy_measure",
            "migration_direction",
            "context_excerpt",
            "sentence_id",
        ])
    )


def europe_concrete_events(events: pl.DataFrame) -> pl.DataFrame:
    """Filter concrete event mentions to European targets plus the EU."""
    europe_iso3 = europe_iso3_codes()
    return events.filter(
        pl.col("target_iso3").is_in(europe_iso3)
        | pl.col("entity_content").is_in(list(EU_ENTITIES))
        | (pl.col("weog_group") == "european_union")
    )


def europe_conversation_map_summary(events: pl.DataFrame) -> pl.DataFrame:
    """Aggregate Europe concrete events for map hover text."""
    return (
        events
        .drop_nulls("target_iso3")
        .filter(pl.col("target_iso3").is_in(europe_iso3_codes()))
        .with_columns(
            pl.col("context_window")
            .map_elements(lambda text: " ".join(str(text).replace("||", " ").split())[:520], return_dtype=pl.Utf8)
            .alias("context_excerpt")
        )
        .group_by(["entity_content", "target_iso3"])
        .agg([
            pl.len().alias("concrete_event_mentions"),
            pl.col("fact_concreteness_score").mean().round(3).alias("mean_fact_concreteness"),
            pl.col("narrative_frame").mode().first().alias("dominant_frame"),
            pl.col("policy_agency_type").mode().first().alias("dominant_agency"),
            pl.col("event_label").drop_nulls().unique().str.join(", ").alias("top_event_labels"),
            pl.col("context_excerpt").first().alias("sample_excerpt"),
        ])
        .sort("concrete_event_mentions", descending=True)
    )


def europe_conversation_year_summary(events: pl.DataFrame) -> pl.DataFrame:
    """Aggregate European concrete events by country and year for animated maps."""
    return (
        events
        .drop_nulls("target_iso3")
        .filter(pl.col("target_iso3").is_in(europe_iso3_codes()))
        .group_by(["source_year", "entity_content", "target_iso3"])
        .agg([
            pl.len().alias("concrete_event_mentions"),
            pl.col("narrative_frame").mode().first().alias("dominant_frame"),
            pl.col("policy_agency_type").mode().first().alias("dominant_agency"),
            pl.col("event_label").drop_nulls().first().alias("sample_event_label"),
        ])
        .sort(["source_year", "concrete_event_mentions"], descending=[False, True])
    )


def _place_pattern(alias: str) -> re.Pattern:
    return re.compile(rf"(?<![A-Za-z]){re.escape(alias)}(?![A-Za-z])", flags=re.IGNORECASE)


def _matched_places(text: str) -> list[dict[str, object]]:
    """Match concrete places from the local Europe gazetteer in descending alias length."""
    matches = []
    seen = set()
    candidates = []
    for place in EUROPE_PLACE_GAZETTEER:
        for alias in place["aliases"]:
            candidates.append((alias, place))
    for alias, place in sorted(candidates, key=lambda item: len(item[0]), reverse=True):
        if place["place_name"] in seen:
            continue
        if _place_pattern(alias).search(text):
            seen.add(place["place_name"])
            matches.append(place)
    return matches


def geolocated_concrete_event_points(events: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return one point per matched concrete city/region/border mention plus unmatched events."""
    points = []
    matched_event_ids = set()
    working = (
        events
        .with_row_index("_event_id")
        .with_columns(
            pl.col("context_window")
            .map_elements(lambda text: " ".join(str(text).replace("||", " ").split()), return_dtype=pl.Utf8)
            .alias("full_context")
        )
    )
    for row in working.to_dicts():
        text = " ".join(str(row.get(field) or "") for field in [
            "entity_content",
            "event_label",
            "proper_noun_anchors",
            "countries_detected_in_context",
            "full_context",
        ])
        matches = _matched_places(text)
        if not matches:
            continue
        matched_event_ids.add(row["_event_id"])
        for place in matches:
            points.append({
                "event_id": row["_event_id"],
                "source_year": row.get("source_year"),
                "session_date": row.get("session_date"),
                "entity_content": row.get("entity_content"),
                "target_iso3": row.get("target_iso3"),
                "place_name": place["place_name"],
                "place_country": place["country"],
                "place_kind": place["place_kind"],
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "event_label": row.get("event_label"),
                "proper_noun_anchors": row.get("proper_noun_anchors"),
                "countries_detected_in_context": row.get("countries_detected_in_context"),
                "fact_concreteness_score": row.get("fact_concreteness_score"),
                "point_weight": 1,
                "narrative_frame": row.get("narrative_frame"),
                "policy_agency_type": row.get("policy_agency_type"),
                "migrant_cohort": row.get("migrant_cohort"),
                "policy_measure": row.get("policy_measure"),
                "migration_direction": row.get("migration_direction"),
                "context_window": row.get("full_context"),
                "sentence_id": row.get("sentence_id"),
            })
    point_df = pl.DataFrame(points) if points else pl.DataFrame()
    unmatched = (
        working
        .filter(~pl.col("_event_id").is_in(list(matched_event_ids)))
        .drop("_event_id")
    )
    return point_df, unmatched


def save_concrete_conversations_dashboard(output_paths: dict[str, Path], output_path: Path) -> Path:
    """Save a small HTML index for the concrete-conversations visual layer."""
    ordered = [
        ("Europe map", "europe_concrete_conversation_map"),
        ("Europe animated map by year", "europe_concrete_map_by_year"),
        ("Europe timeline", "europe_concrete_event_timeline"),
        ("Europe country-year heatmap", "europe_country_year_event_heatmap"),
        ("Europe country-frame heatmap", "europe_country_frame_heatmap"),
        ("Europe Sankey", "europe_concrete_sankey"),
        ("Europe treemap", "europe_concrete_treemap"),
        ("Geolocated point map", "europe_geolocated_concrete_event_point_map"),
        ("Geolocated place timeline", "europe_geolocated_concrete_event_timeline"),
        ("Interactive explorer", "concrete_conversations_explorer"),
        ("Europe event table", "europe_concrete_event_mentions_table"),
        ("All concrete event table", "all_concrete_event_mentions_table"),
    ]
    cards = []
    for label, key in ordered:
        path = output_paths.get(key)
        if not path:
            continue
        rel = quote(path.name)
        cards.append(
            "<article>"
            f"<h2>{label}</h2>"
            f"<p><a href='{rel}' target='_blank'>Open {path.name}</a></p>"
            f"<iframe src='{rel}' loading='lazy'></iframe>"
            "</article>"
        )
    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Concrete Conversations Dashboard</title>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;background:#f8f9fa;color:#1f2933;}"
        "h1{font-size:26px;margin:0 0 8px 0}.sub{color:#52606d;margin-bottom:18px;}"
        "article{background:#fff;border:1px solid #d9e2ec;border-radius:8px;margin:0 0 18px 0;padding:12px;}"
        "h2{font-size:17px;margin:0 0 6px 0}p{margin:0 0 8px 0}a{color:#1d4ed8}"
        "iframe{width:100%;height:620px;border:1px solid #d9e2ec;border-radius:6px;background:#fff;}"
        "</style></head><body>"
        "<h1>Concrete Conversations Dashboard</h1>"
        "<p class='sub'>Europe-focused maps, timelines, schemes, tables, and event-context visualizations.</p>"
        f"{''.join(cards)}"
        "</body></html>"
    )
    output_path.write_text(document, encoding="utf-8")
    return output_path


def direction_agenda_summary(df: pl.DataFrame, group_col: str) -> pl.DataFrame:
    """Summarize inbound/external agenda shares by country/entity or party."""
    return (
        df
        .drop_nulls(group_col)
        .with_columns(
            pl.when(pl.col("migration_direction") == "inbound_internal")
            .then(pl.lit("internal_inbound"))
            .when(pl.col("migration_direction") == "outbound_from_domestic")
            .then(pl.lit("domestic_outbound"))
            .when(pl.col("migration_direction") == "external_transnational")
            .then(pl.lit("external_transnational"))
            .otherwise(pl.lit("ambiguous"))
            .alias("agenda_direction")
        )
        .group_by([group_col, "agenda_direction"])
        .agg(pl.len().alias("n_mentions"))
        .with_columns(
            (pl.col("n_mentions") / pl.col("n_mentions").sum().over(group_col)).alias("share")
        )
        .sort([group_col, "n_mentions"], descending=[False, True])
    )


def country_year_concreteness_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Summarize concreteness by country/entity and year."""
    return (
        df
        .drop_nulls("concreteness_score")
        .group_by(["entity_content", "source_year", "target_iso3", "weog_group", "region_group"])
        .agg([
            pl.len().alias("n_mentions"),
            pl.col("concreteness_score").mean().round(3).alias("mean_concreteness"),
            pl.col("concreteness_score").median().round(3).alias("median_concreteness"),
            pl.col("narrative_frame").mode().first().alias("dominant_frame"),
            pl.col("policy_agency_type").mode().first().alias("dominant_agency"),
            pl.col("migration_direction").mode().first().alias("dominant_direction"),
        ])
        .sort(["source_year", "n_mentions"], descending=[False, True])
    )


def concreteness_feature_patterns(df: pl.DataFrame) -> pl.DataFrame:
    """Summarize repeated feature patterns and their concreteness."""
    return (
        df
        .drop_nulls("concreteness_score")
        .group_by(["policy_agency_type", "narrative_frame", "migration_direction"])
        .agg([
            pl.len().alias("n_mentions"),
            pl.col("concreteness_score").mean().round(3).alias("mean_concreteness"),
            pl.col("entity_content").mode().first().alias("typical_entity"),
            pl.col("migrant_cohort").mode().first().alias("dominant_cohort"),
            pl.col("policy_measure").mode().first().alias("dominant_policy_measure"),
        ])
        .sort(["n_mentions", "mean_concreteness"], descending=[True, True])
    )


def concreteness_quote_examples(df: pl.DataFrame, top_n: int = 12, examples_per_country: int = 2) -> pl.DataFrame:
    """Return concrete and abstract example excerpts by country/entity."""
    entities = (
        df
        .group_by("entity_content")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(top_n)
        .get_column("entity_content")
        .to_list()
    )
    working = (
        df
        .filter(pl.col("entity_content").is_in(entities))
        .with_columns(
            pl.col("context_window")
            .map_elements(lambda text: " ".join(str(text).replace("||", " ").split())[:520], return_dtype=pl.Utf8)
            .alias("context_excerpt")
        )
    )
    selected = []
    for entity in entities:
        sub = working.filter(pl.col("entity_content") == entity)
        if sub.is_empty():
            continue
        selected.append(
            sub
            .sort(["concreteness_score", "source_year"], descending=[True, False])
            .head(examples_per_country)
            .with_columns(pl.lit("most_concrete").alias("example_type"))
        )
        selected.append(
            sub
            .sort(["concreteness_score", "source_year"], descending=[False, False])
            .head(examples_per_country)
            .with_columns(pl.lit("most_abstract").alias("example_type"))
        )
    if not selected:
        return pl.DataFrame()
    return (
        pl.concat(selected, how="vertical")
        .select([
            "entity_content",
            "example_type",
            "source_year",
            "session_date",
            "concreteness_score",
            "narrative_frame",
            "policy_agency_type",
            "migration_direction",
            "context_excerpt",
            "sentence_id",
        ])
        .sort(["entity_content", "example_type", "concreteness_score"], descending=[False, False, True])
    )


def main() -> None:
    df = enrich_with_party_metadata(
        _read_required_parquet(PROCESSED_DIR / f"{PREFIX}_migration_mentions_extended.parquet")
    )
    diffusion_edges = _read_required_csv(PROCESSED_DIR / f"{PREFIX}_diffusion_edges.csv")
    agency_edges = _read_required_csv(PROCESSED_DIR / f"{PREFIX}_policy_agency_edges.csv")
    events = _read_required_csv(PROCESSED_DIR / f"{PREFIX}_high_concreteness_events.csv")
    visible_summary = _read_required_csv(PROCESSED_DIR / f"{PREFIX}_visible_country_summary.csv")

    extended_output_paths = visualize.save_extended_figures(
        df=df,
        edges=diffusion_edges,
        processed_dir=PROCESSED_DIR,
    )

    country_year_path = PROCESSED_DIR / f"{PREFIX}_country_year_concreteness_summary.csv"
    feature_patterns_path = PROCESSED_DIR / f"{PREFIX}_concreteness_feature_patterns.csv"
    quote_examples_path = PROCESSED_DIR / f"{PREFIX}_concreteness_quote_examples.csv"
    country_year_concreteness_summary(df).write_csv(country_year_path)
    concreteness_feature_patterns(df).write_csv(feature_patterns_path)
    concreteness_quote_examples(df).write_csv(quote_examples_path)

    hubs = agency.policy_hub_pagerank(agency_edges)
    hubs_path = PROCESSED_DIR / f"{PREFIX}_policy_hubs_pagerank.csv"
    hubs.write_csv(hubs_path)

    output_paths = visualize.save_advanced_figures(
        df=df,
        agency_edges=agency_edges,
        events=events,
        visible_summary=visible_summary,
        processed_dir=PROCESSED_DIR,
    )
    figures_dir = PROCESSED_DIR / "figures_interactive_advanced"
    output_paths["policy_hubs_pagerank"] = visualize.plot_policy_hubs_pagerank(
        hubs,
        figures_dir / "policy_hubs_pagerank.png",
    )
    output_paths["narrative_ternary_by_party"] = visualize.plot_narrative_ternary_by_group(
        df,
        figures_dir / "narrative_ternary_by_party.html",
        group_col="party_group",
        title="Narrative language flavor by party group",
    )
    output_paths["narrative_mirror_by_party"] = visualize.plot_narrative_mirror_bars_by_group(
        df,
        figures_dir / "narrative_mirror_by_party.png",
        group_col="party_group",
        title="Narrative Mirror: imagined risks versus benefits by party group",
    )
    output_paths["narrative_mirror_by_speaker_role"] = visualize.plot_narrative_mirror_bars_by_group(
        df,
        figures_dir / "narrative_mirror_by_speaker_role.png",
        group_col="speaker_role",
        title="Narrative Mirror by speaker role",
    )
    direction_entity_path = PROCESSED_DIR / f"{PREFIX}_direction_agenda_by_entity.csv"
    direction_party_path = PROCESSED_DIR / f"{PREFIX}_direction_agenda_by_party.csv"
    direction_agenda_summary(df, "entity_content").write_csv(direction_entity_path)
    direction_agenda_summary(df, "party_group").write_csv(direction_party_path)
    output_paths["direction_agenda_by_party"] = visualize.plot_direction_agenda_split_bars(
        df,
        figures_dir / "direction_agenda_by_party.png",
        group_col="party_group",
    )

    strict_events = (
        add_fact_concreteness_score(events)
        .filter(pl.col("fact_concreteness_score") > 4.0)
        .with_columns(pl.col("fact_concreteness_score").alias("concreteness_score"))
    )
    all_concrete_mentions = concrete_event_mentions_table(strict_events)
    europe_events = europe_concrete_events(strict_events)
    europe_mentions = concrete_event_mentions_table(europe_events)
    europe_summary = europe_conversation_map_summary(europe_events)
    europe_year_summary = europe_conversation_year_summary(europe_events)
    europe_points, europe_unmatched = geolocated_concrete_event_points(europe_events)

    strict_events_path = PROCESSED_DIR / f"{PREFIX}_high_concreteness_events_gt4.csv"
    strict_summary_path = PROCESSED_DIR / f"{PREFIX}_visible_country_summary_gt4.csv"
    all_concrete_mentions_path = PROCESSED_DIR / f"{PREFIX}_all_concrete_event_mentions.csv"
    europe_mentions_path = PROCESSED_DIR / f"{PREFIX}_europe_concrete_event_mentions.csv"
    europe_summary_path = PROCESSED_DIR / f"{PREFIX}_europe_concrete_conversation_summary.csv"
    europe_year_summary_path = PROCESSED_DIR / f"{PREFIX}_europe_concrete_conversation_year_summary.csv"
    europe_points_path = PROCESSED_DIR / f"{PREFIX}_europe_geolocated_concrete_event_points.csv"
    europe_unmatched_path = PROCESSED_DIR / f"{PREFIX}_europe_concrete_event_mentions_unmatched_places.csv"
    strict_events.write_csv(strict_events_path)
    strict_visible_summary = event_tools.visible_country_summary(strict_events)
    strict_visible_summary.write_csv(strict_summary_path)
    all_concrete_mentions.write_csv(all_concrete_mentions_path)
    europe_mentions.write_csv(europe_mentions_path)
    europe_summary.write_csv(europe_summary_path)
    europe_year_summary.write_csv(europe_year_summary_path)
    europe_points.write_csv(europe_points_path)
    europe_unmatched.write_csv(europe_unmatched_path)

    output_paths["evidence_visibility_map_gt4"] = visualize.plot_evidence_visibility_map(
        strict_visible_summary,
        figures_dir / "evidence_visibility_map_gt4.html",
    )
    output_paths["fact_density_timeline_gt4"] = visualize.plot_fact_density_timeline(
        strict_events,
        figures_dir / "fact_density_timeline_gt4.html",
    )
    output_paths["europe_concrete_conversation_map"] = visualize.plot_europe_concrete_conversation_map(
        europe_summary,
        figures_dir / "europe_concrete_conversation_map.html",
    )
    output_paths["europe_concrete_map_by_year"] = visualize.plot_europe_concrete_map_by_year(
        europe_year_summary,
        figures_dir / "europe_concrete_map_by_year.html",
    )
    output_paths["europe_concrete_event_timeline"] = visualize.plot_europe_concrete_event_timeline(
        europe_events,
        figures_dir / "europe_concrete_event_timeline.html",
    )
    output_paths["europe_country_year_event_heatmap"] = visualize.plot_europe_country_year_event_heatmap(
        europe_events,
        figures_dir / "europe_country_year_event_heatmap.png",
    )
    output_paths["europe_country_frame_heatmap"] = visualize.plot_europe_country_frame_heatmap(
        europe_events,
        figures_dir / "europe_country_frame_heatmap.png",
    )
    output_paths["europe_concrete_sankey"] = visualize.plot_europe_concrete_sankey(
        europe_events,
        figures_dir / "europe_concrete_sankey.html",
    )
    output_paths["europe_concrete_treemap"] = visualize.plot_europe_concrete_treemap(
        europe_events,
        figures_dir / "europe_concrete_treemap.html",
    )
    output_paths["europe_geolocated_concrete_event_point_map"] = visualize.plot_geolocated_concrete_event_map(
        europe_points,
        figures_dir / "europe_geolocated_concrete_event_point_map.html",
        title="Europe: concrete migration mentions geolocated to cities, regions, borders, and routes",
    )
    output_paths["europe_geolocated_concrete_event_timeline"] = visualize.plot_geolocated_event_timeline(
        europe_points,
        figures_dir / "europe_geolocated_concrete_event_timeline.html",
    )
    output_paths["concrete_conversations_explorer"] = visualize.save_concrete_conversations_explorer(
        europe_points,
        figures_dir / "concrete_conversations_explorer.html",
        title="Concrete Migration Conversations Explorer",
    )
    output_paths["all_concrete_event_mentions_table"] = visualize.save_concrete_event_mentions_table(
        strict_events,
        figures_dir / "all_concrete_event_mentions_table.html",
        title="All concrete migration event mentions",
    )
    output_paths["europe_concrete_event_mentions_table"] = visualize.save_concrete_event_mentions_table(
        europe_events,
        figures_dir / "europe_concrete_event_mentions_table.html",
        title="Europe concrete migration event mentions",
    )
    output_paths["concrete_conversations_dashboard"] = save_concrete_conversations_dashboard(
        output_paths,
        figures_dir / "concrete_conversations_dashboard.html",
    )

    print(f"policy_hubs_pagerank_csv: {hubs_path}")
    print(f"high_concreteness_events_gt4_csv: {strict_events_path}")
    print(f"visible_country_summary_gt4_csv: {strict_summary_path}")
    print(f"all_concrete_event_mentions_csv: {all_concrete_mentions_path}")
    print(f"europe_concrete_event_mentions_csv: {europe_mentions_path}")
    print(f"europe_concrete_conversation_summary_csv: {europe_summary_path}")
    print(f"europe_concrete_conversation_year_summary_csv: {europe_year_summary_path}")
    print(f"europe_geolocated_concrete_event_points_csv: {europe_points_path}")
    print(f"europe_concrete_event_mentions_unmatched_places_csv: {europe_unmatched_path}")
    print(f"direction_agenda_by_entity_csv: {direction_entity_path}")
    print(f"direction_agenda_by_party_csv: {direction_party_path}")
    print(f"country_year_concreteness_summary_csv: {country_year_path}")
    print(f"concreteness_feature_patterns_csv: {feature_patterns_path}")
    print(f"concreteness_quote_examples_csv: {quote_examples_path}")
    for name, path in extended_output_paths.items():
        print(f"{name}: {path}")
    for name, path in output_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
