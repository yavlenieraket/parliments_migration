"""Filtering functions for migration discourse and foreign country mentions."""

import re

import polars as pl

from src.config import (
    EU_ENTITIES,
    EUROPEAN_COUNTRIES,
    EXCLUDE_FROM_FOREIGN,
    FRENCH_OVERSEAS,
    MIGRATION_KEYWORDS,
    MIGRATION_TOPIC,
)


COUNTRY_ALIASES = {
    # Explanation: These variants appear because the corpus is translated to English
    # but some original French/German/Italian surface forms still survive in NER.
    "Allemagne": "Germany",
    "Deutschland": "Germany",
    "Federal Republic of Germany": "Germany",
    "German": "Germany",
    "Italie": "Italy",
    "Italia": "Italy",
    "Italian": "Italy",
    "Espagne": "Spain",
    "España": "Spain",
    "Spanish": "Spain",
    "Royaume-Uni": "United Kingdom",
    "Grande-Bretagne": "United Kingdom",
    "UK": "United Kingdom",
    "Britain": "United Kingdom",
    "British": "United Kingdom",
    "Great Britain": "United Kingdom",
    "England": "United Kingdom",
    "Hongrie": "Hungary",
    "Hungarian": "Hungary",
    "Pologne": "Poland",
    "Belgique": "Belgium",
    "Pays-Bas": "Netherlands",
    "Holland": "Netherlands",
    "Grèce": "Greece",
    "Turquie": "Turkey",
    "Maroc": "Morocco",
    "Algérie": "Algeria",
    "Tunisie": "Tunisia",
    "Libye": "Libya",
    "Syrie": "Syria",
    "Soudan": "Sudan",
    "Sénégal": "Senegal",
    "Mali": "Mali",
    "États-Unis": "United States",
    "USA": "United States",
    "U.S.": "United States",
    "U.S.A.": "United States",
    "America": "United States",
    "Guyana": "French Guiana",
    "Mahorais": "Mayotte",
    "Anjouan": "Comoros",
    "EU": "European Union",
    "US": "United States",
    "Republic of Armenia": "Armenia",
    "Côte d'Ivoire": "Cote d'Ivoire",
    "Ivory Coast": "Cote d'Ivoire",
    "Futuna": "Wallis and Futuna",
    "Mamoudzou": "Mayotte",
}

EU_ENTITY_ALIASES = {"European Union", "EU"}


def _migration_keyword_pattern() -> str:
    """Return a regex matching migration keywords as whole words/phrases."""
    # Explanation: Longer phrases are sorted first so "asylum seeker" is checked
    # before the shorter "asylum" token.
    terms = sorted(MIGRATION_KEYWORDS, key=len, reverse=True)
    escaped = [re.escape(term.lower()) for term in terms]
    return r"\b(" + "|".join(escaped) + r")\b"


def filter_migration(
    lf: pl.LazyFrame,
    use_topic: bool = True,
    use_keywords: bool = False,
) -> pl.LazyFrame:
    """Keep rows tagged as immigration and/or containing migration keywords."""
    # Explanation: CAP topic is the strongest signal; keywords catch migration rows
    # where the topic label is missing, noisy, or broader than the sentence.
    predicates = []
    if use_topic:
        predicates.append(pl.col("debate_topic") == MIGRATION_TOPIC)
    if use_keywords:
        text = pl.concat_str([
            pl.col("sentence_content_previous").fill_null(""),
            pl.lit(" "),
            pl.col("sentence_content_current").fill_null(""),
            pl.lit(" "),
            pl.col("sentence_content_next").fill_null(""),
        ]).str.to_lowercase()
        predicates.append(text.str.contains(_migration_keyword_pattern()))
    if not predicates:
        return lf

    predicate = predicates[0]
    for extra_predicate in predicates[1:]:
        predicate = predicate | extra_predicate
    return lf.filter(predicate)


def filter_country_mentions(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Keep LOC entities plus explicit European Union organization mentions."""
    # Explanation: Countries are usually LOC, but the European Union is encoded as ORG.
    return lf.filter(
        (pl.col("entity_category") == "LOC")
        | (
            (pl.col("entity_category") == "ORG")
            & pl.col("entity_content").is_in(EU_ENTITY_ALIASES)
        )
    )


def normalize_country_names(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Merge known country-name variants into a canonical English form."""
    # Explanation: Start from the original entity and overwrite only known aliases.
    expr = pl.col("entity_content")
    for variant, canonical in COUNTRY_ALIASES.items():
        expr = (
            pl.when(pl.col("entity_content") == variant)
            .then(pl.lit(canonical))
            .otherwise(expr)
        )
    return lf.with_columns(expr.alias("entity_content"))


def filter_foreign(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Exclude France itself, French cities/regions, and non-country geo terms.

    Overseas territories are kept but flagged separately (see add_geo_class).
    """
    # Explanation: Remove France, French places, and non-country geographic regions.
    return lf.filter(~pl.col("entity_content").is_in(EXCLUDE_FROM_FOREIGN))


def add_geo_class(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add a column distinguishing foreign countries from French overseas territories."""
    # Explanation: EU and overseas territories are not foreign countries, but we keep them visible.
    return lf.with_columns(
        pl.when(pl.col("entity_content").is_in(EU_ENTITIES))
        .then(pl.lit("european_union"))
        .when(pl.col("entity_content").is_in(FRENCH_OVERSEAS))
        .then(pl.lit("french_overseas"))
        .otherwise(pl.lit("foreign"))
        .alias("geo_class")
    )


def add_region_group(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add Europe / non-Europe / EU / French overseas analytical grouping."""
    # Explanation: This lets us compare European country mentions with non-European ones
    # while keeping the European Union and French overseas territories separate.
    return lf.with_columns(
        pl.when(pl.col("entity_content").is_in(EU_ENTITIES))
        .then(pl.lit("european_union"))
        .when(pl.col("entity_content").is_in(FRENCH_OVERSEAS))
        .then(pl.lit("french_overseas"))
        .when(pl.col("entity_content").is_in(EUROPEAN_COUNTRIES))
        .then(pl.lit("european_country"))
        .otherwise(pl.lit("non_european_country"))
        .alias("region_group")
    )


def build_context_window(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Concatenate previous/current/next sentences into a single context field.

    The separator '||' makes it easy to split back later if needed.
    """
    # Explanation: Context is previous + current + next sentence for manual validation.
    return lf.with_columns(
        pl.concat_str([
            pl.col("sentence_content_previous").fill_null(""),
            pl.lit(" || "),
            pl.col("sentence_content_current"),
            pl.lit(" || "),
            pl.col("sentence_content_next").fill_null(""),
        ]).alias("context_window")
    )


def build_migration_mentions(
    lf: pl.LazyFrame,
    use_topic: bool = True,
    use_keywords: bool = False,
) -> pl.DataFrame:
    """Full pipeline: facts -> migration debates -> LOC entities ->
    foreign countries (with overseas flag) -> with context window.

    Returns an eagerly-collected DataFrame.
    """
    # Explanation: The order matters: normalize names before excluding domestic terms.
    return (
        lf
        .pipe(filter_migration, use_topic=use_topic, use_keywords=use_keywords)
        .pipe(filter_country_mentions)
        .pipe(normalize_country_names)
        .pipe(filter_foreign)
        .pipe(add_geo_class)
        .pipe(add_region_group)
        .pipe(build_context_window)
        .with_columns(
            # Explanation: Keep source_year in every output, even if the input was
            # loaded from a single file without the multi-year loader.
            pl.col("session_date").dt.year().alias("source_year")
        )
        .select([
            "source_year",
            "sentence_id",
            "speech_id",
            "session_date",
            "speaker_id",
            "speaker_ana",
            "entity_content",
            "geo_class",
            "region_group",
            "context_window",
            "sentence_content_current",
            "sentence_sentiment_value",
            "sentence_sentiment_ana",
            "debate_topic",
            "country",
        ])
        .collect()
    )
