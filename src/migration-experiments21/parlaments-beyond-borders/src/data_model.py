"""Lazy data model for cross-country migration reference analysis.

The facts table is mention-level data: one named entity mention in one
sentence. This module converts that raw entity text into a stable dyadic model:

    source parliament country -> mentioned target country

Everything here is written against Polars LazyFrame where possible so filters,
projections, and joins can be pushed down into parquet scans.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl

from src.config import DATA_ROOT, PROCESSED_DIR
from src.filters import COUNTRY_ALIASES
from src.geo import COUNTRY_ISO3


RESOLVER_PATH = DATA_ROOT / "country_resolver.parquet"

MIGRATION_TERMS = [
    "migrat",
    "migrant",
    "immigrat",
    "emigrat",
    "refugee",
    "asylum",
    "border",
    "deportat",
    "integration",
    "naturali",
    "citizenship",
    "diaspora",
    "smuggl",
    "traffick",
    "frontex",
    "schengen",
]

SHOCKS = {
    "mediterranean_2015": date(2015, 4, 19),
    "cologne_2015": date(2015, 12, 31),
    "brexit_vote": date(2016, 6, 23),
    "moria_fire": date(2020, 9, 9),
    "afghan_withdrawal": date(2021, 8, 15),
    "belarus_crisis": date(2021, 8, 1),
    "ukraine_war": date(2022, 2, 24),
}

# Explanation: These are not a replacement for a full Wikidata gazetteer. They are
# the high-value aliases needed for this corpus and can be extended safely because
# they are resolved through normalized exact joins, not substring matching.
COUNTRY_NAME_ALIASES = {
    "Federal Republic of Germany": "Germany",
    "German government": "Germany",
    "German Bundestag": "Germany",
    "Bundestag": "Germany",
    "Berlin": "Germany",
    "Deutsch": "Germany",
    "Deutschland": "Germany",
    "Allemagne": "Germany",
    "UK": "United Kingdom",
    "Britain": "United Kingdom",
    "Great Britain": "United Kingdom",
    "British government": "United Kingdom",
    "British Parliament": "United Kingdom",
    "Westminster": "United Kingdom",
    "London": "United Kingdom",
    "England": "United Kingdom",
    "Scotland": "United Kingdom",
    "Wales": "United Kingdom",
    "Northern Ireland": "United Kingdom",
    "United States of America": "United States",
    "USA": "United States",
    "U.S.": "United States",
    "U.S.A.": "United States",
    "US": "United States",
    "America": "United States",
    "Washington": "United States",
    "White House": "United States",
    "Italia": "Italy",
    "Italian government": "Italy",
    "Rome": "Italy",
    "Lampedusa": "Italy",
    "Sicily": "Italy",
    "Trieste": "Italy",
    "España": "Spain",
    "Espagne": "Spain",
    "Madrid": "Spain",
    "Ceuta": "Spain",
    "Melilla": "Spain",
    "Grèce": "Greece",
    "Hellenic Republic": "Greece",
    "Greek government": "Greece",
    "Athens": "Greece",
    "Lesbos": "Greece",
    "Lesvos": "Greece",
    "Moria": "Greece",
    "Samos": "Greece",
    "Turkiye": "Turkey",
    "Türkiye": "Turkey",
    "Turquie": "Turkey",
    "Turkish government": "Turkey",
    "Ankara": "Turkey",
    "Istanbul": "Turkey",
    "Belgique": "Belgium",
    "Brussels": "Belgium",
    "Pays-Bas": "Netherlands",
    "Holland": "Netherlands",
    "Dutch government": "Netherlands",
    "The Hague": "Netherlands",
    "Pologne": "Poland",
    "Warsaw": "Poland",
    "Hongrie": "Hungary",
    "Budapest": "Hungary",
    "Orban": "Hungary",
    "Orbán": "Hungary",
    "Minsk": "Belarus",
    "Kyiv": "Ukraine",
    "Kiev": "Ukraine",
    "Maroc": "Morocco",
    "Rabat": "Morocco",
    "Algérie": "Algeria",
    "Algiers": "Algeria",
    "Tunisie": "Tunisia",
    "Tunis": "Tunisia",
    "Libye": "Libya",
    "Tripoli": "Libya",
    "Syrie": "Syria",
    "Damascus": "Syria",
    "Soudan": "Sudan",
    "Khartoum": "Sudan",
    "Côte d'Ivoire": "Cote d'Ivoire",
    "Ivory Coast": "Cote d'Ivoire",
}

RESOLVER_SOURCE_BY_ALIAS = {
    "Bundestag": "institution",
    "German Bundestag": "institution",
    "White House": "institution",
    "British Parliament": "institution",
    "Westminster": "institution",
    "Berlin": "city",
    "London": "city",
    "Rome": "city",
    "Lampedusa": "city",
    "Trieste": "city",
    "Madrid": "city",
    "Ceuta": "city",
    "Melilla": "city",
    "Athens": "city",
    "Lesbos": "island",
    "Lesvos": "island",
    "Moria": "camp_or_place",
    "Samos": "island",
    "Ankara": "city",
    "Istanbul": "city",
    "Brussels": "city",
    "The Hague": "city",
    "Warsaw": "city",
    "Budapest": "city",
    "Minsk": "city",
    "Kyiv": "city",
    "Kiev": "city",
    "Rabat": "city",
    "Algiers": "city",
    "Tunis": "city",
    "Tripoli": "city",
    "Damascus": "city",
    "Khartoum": "city",
}


def normalize_expr(col: str) -> pl.Expr:
    """Normalize entity text for exact resolver joins."""
    return (
        pl.col(col)
        .str.to_lowercase()
        .str.strip_chars()
        .str.replace_all(r"[^\w\s]", "")
        .str.replace_all(r"\s+", " ")
    )


def normalize_text(value: str) -> str:
    """Python-side equivalent of normalize_expr for resolver construction."""
    return " ".join(
        "".join(char.lower() if (char.isalnum() or char.isspace() or char == "_") else " " for char in value)
        .split()
    )


def build_country_resolver_table() -> pl.DataFrame:
    """Build a mention-text to ISO3 resolver table.

    The table intentionally includes LOC, ORG, and MISC aliases but not person
    aliases. Mentions such as "Merkel" can be analyzed later as persona mentions,
    but they should not silently become country mentions by default.
    """
    rows: list[dict[str, object]] = []

    def add(alias: str, canonical: str, confidence: float, source: str) -> None:
        iso3 = COUNTRY_ISO3.get(canonical)
        if not iso3:
            return
        rows.append({
            "entity_text": alias,
            "entity_text_normalized": normalize_text(alias),
            "target_country_name": canonical,
            "target_country_iso3": iso3,
            "resolver_confidence": confidence,
            "resolver_source": source,
        })

    for country_name in COUNTRY_ISO3:
        add(country_name, country_name, 1.0, "country_name")

    for alias, canonical in COUNTRY_ALIASES.items():
        add(alias, canonical, 0.96, "existing_alias")

    for alias, canonical in COUNTRY_NAME_ALIASES.items():
        add(
            alias,
            canonical,
            0.9 if RESOLVER_SOURCE_BY_ALIAS.get(alias) in {"city", "island", "camp_or_place"} else 0.94,
            RESOLVER_SOURCE_BY_ALIAS.get(alias, "manual_alias"),
        )

    return (
        pl.DataFrame(rows)
        .unique(subset=["entity_text_normalized", "target_country_iso3"], keep="first")
        .sort(["target_country_iso3", "entity_text_normalized"])
    )


def save_country_resolver(path: Path = RESOLVER_PATH) -> Path:
    """Write the resolver parquet used by the lazy dyadic model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    build_country_resolver_table().write_parquet(path)
    return path


def scan_country_resolver(path: Path = RESOLVER_PATH) -> pl.LazyFrame:
    """Scan the resolver parquet, creating it first when needed."""
    if not path.exists():
        save_country_resolver(path)
    return pl.scan_parquet(path)


def resolve_country_mentions(
    facts: pl.LazyFrame,
    country_resolver: pl.LazyFrame | None = None,
    min_confidence: float = 0.85,
) -> pl.LazyFrame:
    """Resolve raw entity mentions to target-country ISO3 codes."""
    resolver = country_resolver if country_resolver is not None else scan_country_resolver()
    return (
        facts
        .filter(pl.col("entity_category").is_in(["LOC", "ORG", "MISC"]))
        .with_columns(normalize_expr("entity_content").alias("entity_text_normalized"))
        .join(resolver, on="entity_text_normalized", how="inner")
        .filter(pl.col("resolver_confidence") >= min_confidence)
        .filter(pl.col("target_country_iso3") != pl.col("country"))
        .rename({"country": "source_country"})
    )


def migration_keyword_pattern(terms: list[str] | None = None) -> str:
    """Return a regex for human-migration vocabulary."""
    terms = sorted(terms or MIGRATION_TERMS, key=len, reverse=True)
    return "(" + "|".join(terms) + ")"


def migration_speech_ids(
    facts: pl.LazyFrame,
    min_hits: int = 2,
    terms: list[str] | None = None,
    topic_code: str = "immig",
    include_topic: bool = True,
) -> pl.LazyFrame:
    """Return speech IDs that are genuinely migration-related."""
    text = (
        pl.concat_str([
            pl.col("sentence_content_previous").fill_null(""),
            pl.lit(" "),
            pl.col("sentence_content_current").fill_null(""),
            pl.lit(" "),
            pl.col("sentence_content_next").fill_null(""),
        ])
        .str.to_lowercase()
    )
    keyword_hit = text.str.contains(migration_keyword_pattern(terms))
    predicate = keyword_hit | (pl.col("debate_topic") == topic_code) if include_topic else keyword_hit
    return (
        facts
        .filter(predicate)
        .group_by("speech_id")
        .agg(pl.len().alias("n_migration_hits"))
        .filter(pl.col("n_migration_hits") >= min_hits)
        .select("speech_id")
    )


def migration_country_mentions(
    facts: pl.LazyFrame,
    country_resolver: pl.LazyFrame | None = None,
    min_hits: int = 2,
    min_confidence: float = 0.85,
) -> pl.LazyFrame:
    """Resolve countries and retain only migration-related speeches."""
    facts_geo = resolve_country_mentions(facts, country_resolver, min_confidence)
    speech_ids = migration_speech_ids(facts, min_hits=min_hits)
    return facts_geo.join(speech_ids, on="speech_id", how="inner")


def bilateral_matrix(facts_mig: pl.LazyFrame, year_filter: int | None = None) -> pl.DataFrame:
    """Aggregate source-country -> target-country mentions."""
    df = facts_mig
    if year_filter is not None:
        df = df.with_columns(pl.col("session_date").dt.year().alias("year"))
        df = df.filter(pl.col("year") == year_filter)
    return (
        df
        .group_by(["source_country", "target_country_iso3"])
        .agg([
            pl.len().alias("n_mentions"),
            pl.col("speech_id").n_unique().alias("n_speeches"),
            pl.col("speaker_id").n_unique().alias("n_speakers"),
            pl.col("sentence_sentiment_value").mean().alias("mean_sentiment"),
            pl.col("sentence_sentiment_value").std().alias("sd_sentiment"),
            pl.col("target_country_name").mode().first().alias("target_country_name"),
        ])
        .rename({"target_country_iso3": "target_country"})
        .sort(["source_country", "n_mentions"], descending=[False, True])
        .collect()
    )


def compute_concreteness_per_mention(facts_mig: pl.LazyFrame) -> pl.LazyFrame:
    """Add grounded-item counts in the three-sentence window around a mention."""
    return (
        facts_mig
        .with_columns(
            pl.concat_str([
                pl.col("sentence_content_previous").fill_null(""),
                pl.lit(" "),
                pl.col("sentence_content_current").fill_null(""),
                pl.lit(" "),
                pl.col("sentence_content_next").fill_null(""),
            ]).alias("window_text")
        )
        .with_columns([
            pl.col("window_text").str.count_matches(r"\b\d+([.,]\d+)?\b").alias("n_numbers"),
            pl.col("window_text").str.count_matches(r"\b(19|20)\d{2}\b").alias("n_years"),
            pl.col("window_text").str.split(" ").list.len().alias("window_tokens"),
        ])
    )


def grounded_items_by_sentence(facts: pl.LazyFrame) -> pl.LazyFrame:
    """Count entity evidence available in each sentence."""
    return (
        facts
        .group_by("sentence_id")
        .agg([
            (pl.col("entity_category") == "PER").sum().alias("n_per"),
            (pl.col("entity_category") == "LOC").sum().alias("n_loc"),
            (pl.col("entity_category") == "ORG").sum().alias("n_org"),
        ])
    )


def mentions_with_concreteness(facts_mig: pl.LazyFrame, facts: pl.LazyFrame) -> pl.LazyFrame:
    """Attach mention-level concreteness score to migration country mentions."""
    return (
        compute_concreteness_per_mention(facts_mig)
        .join(grounded_items_by_sentence(facts), on="sentence_id", how="left")
        .with_columns([
            pl.col("n_per").fill_null(0),
            pl.col("n_loc").fill_null(0),
            pl.col("n_org").fill_null(0),
        ])
        .with_columns(
            (
                pl.col("n_per")
                + pl.col("n_loc")
                + pl.col("n_org")
                + pl.col("n_numbers")
                + pl.col("n_years")
            ).alias("n_grounded_items")
        )
        .with_columns(
            (1000.0 * pl.col("n_grounded_items") / pl.max_horizontal(pl.col("window_tokens"), pl.lit(1)))
            .alias("concreteness_score")
        )
    )


def bilateral_concreteness(mentions: pl.LazyFrame) -> pl.DataFrame:
    """Aggregate concreteness to source-country -> target-country cells."""
    return (
        mentions
        .group_by(["source_country", "target_country_iso3"])
        .agg([
            pl.col("concreteness_score").mean().alias("mean_concreteness"),
            pl.col("concreteness_score").median().alias("median_concreteness"),
            pl.col("concreteness_score").std().alias("sd_concreteness"),
            pl.len().alias("n_mentions"),
            pl.col("speech_id").n_unique().alias("n_speeches"),
            pl.col("target_country_name").mode().first().alias("target_country_name"),
        ])
        .rename({"target_country_iso3": "target_country"})
        .sort(["source_country", "n_mentions"], descending=[False, True])
        .collect()
    )


def compute_asymmetry(matrix: pl.DataFrame, min_total_traffic: int = 50) -> pl.DataFrame:
    """Compare A->B with B->A for every reciprocal country pair."""
    reverse = matrix.rename({
        "source_country": "target_country",
        "target_country": "source_country",
        "n_mentions": "n_mentions_reverse",
        "n_speeches": "n_speeches_reverse",
        "mean_sentiment": "mean_sentiment_reverse",
    })
    keep_cols = [
        "source_country",
        "target_country",
        "n_mentions_reverse",
        "n_speeches_reverse",
        "mean_sentiment_reverse",
    ]
    return (
        matrix
        .join(reverse.select(keep_cols), on=["source_country", "target_country"], how="inner")
        .filter(pl.col("source_country") < pl.col("target_country"))
        .with_columns([
            (pl.col("n_mentions") / (pl.col("n_mentions") + pl.col("n_mentions_reverse")))
            .alias("attention_share_AtoB"),
            ((pl.col("n_mentions") + 1).log() - (pl.col("n_mentions_reverse") + 1).log())
            .alias("attention_log_ratio"),
            (pl.col("mean_sentiment") - pl.col("mean_sentiment_reverse")).alias("sentiment_gap"),
            (pl.col("n_mentions") + pl.col("n_mentions_reverse")).alias("total_traffic"),
        ])
        .filter(pl.col("total_traffic") >= min_total_traffic)
        .sort("attention_log_ratio", descending=True)
    )


def shock_window_matrix(
    facts_mig: pl.LazyFrame,
    shock_date: date,
    window_days: int = 90,
) -> pl.DataFrame:
    """Compare dyadic attention before and after a shock event."""
    pre_start = shock_date - timedelta(days=window_days)
    post_end = shock_date + timedelta(days=window_days)
    wide = (
        facts_mig
        .filter((pl.col("session_date") >= pre_start) & (pl.col("session_date") <= post_end))
        .with_columns(
            pl.when(pl.col("session_date") < shock_date)
            .then(pl.lit("pre"))
            .otherwise(pl.lit("post"))
            .alias("period")
        )
        .group_by(["source_country", "target_country_iso3", "period"])
        .agg([
            pl.len().alias("n_mentions"),
            pl.col("sentence_sentiment_value").mean().alias("mean_sentiment"),
            pl.col("target_country_name").mode().first().alias("target_country_name"),
        ])
        .collect()
        .pivot(
            values=["n_mentions", "mean_sentiment"],
            index=["source_country", "target_country_iso3", "target_country_name"],
            on="period",
        )
    )
    missing_defaults = {
        "n_mentions_pre": 0,
        "n_mentions_post": 0,
        "mean_sentiment_pre": None,
        "mean_sentiment_post": None,
    }
    for col_name, default_value in missing_defaults.items():
        if col_name not in wide.columns:
            wide = wide.with_columns(pl.lit(default_value).alias(col_name))
    return (
        wide
        .with_columns([
            pl.col("n_mentions_pre").fill_null(0),
            pl.col("n_mentions_post").fill_null(0),
        ])
        .with_columns([
            (pl.col("n_mentions_post") - pl.col("n_mentions_pre")).alias("delta_mentions"),
            (pl.col("mean_sentiment_post") - pl.col("mean_sentiment_pre")).alias("delta_sentiment"),
        ])
        .sort("delta_mentions", descending=True)
    )


def save_data_model_outputs(
    facts: pl.LazyFrame,
    output_dir: Path = PROCESSED_DIR / "dyadic_data_model",
    min_hits: int = 2,
) -> dict[str, Path]:
    """Materialize core dyadic-model tables for inspection and visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    resolver_path = save_country_resolver()
    resolver = scan_country_resolver(resolver_path)
    facts_mig = migration_country_mentions(facts, resolver, min_hits=min_hits)
    mentions = mentions_with_concreteness(facts_mig, facts)

    mentions_path = output_dir / "resolved_migration_country_mentions.parquet"
    matrix_path = output_dir / "bilateral_matrix.csv"
    concreteness_path = output_dir / "bilateral_concreteness.csv"
    asymmetry_path = output_dir / "asymmetry_table.csv"

    mentions.sink_parquet(mentions_path)
    matrix = bilateral_matrix(pl.scan_parquet(mentions_path))
    concrete = bilateral_concreteness(pl.scan_parquet(mentions_path))
    asymmetry = compute_asymmetry(matrix)

    matrix.write_csv(matrix_path)
    concrete.write_csv(concreteness_path)
    asymmetry.write_csv(asymmetry_path)
    return {
        "country_resolver": resolver_path,
        "resolved_mentions": mentions_path,
        "bilateral_matrix": matrix_path,
        "bilateral_concreteness": concreteness_path,
        "asymmetry_table": asymmetry_path,
    }
