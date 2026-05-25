"""Run the extended migration-reference analysis for one or more countries.

This is the script equivalent of the France 2017-2022 extended notebook. It
discovers available Table1_Fact parquet files for each requested country and
saves country-specific processed tables, networks, event tables, and figures.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import agency, concreteness, diffusion, direction, events, evidence, filters, framing, geo, load, typology, visualize  # noqa: E402
from src.config import DATA_ROOT, PROCESSED_DIR  # noqa: E402


DEFAULT_COUNTRIES = [
    "AUT", "BEL", "BGR", "BIH", "CZE", "ESP", "EST", "FIN", "FRA",
    "GBR", "GRC", "HRV", "HUN", "ISL", "ITA", "LVA", "NLD", "NOR",
    "POL", "PRT", "SRB", "SVN", "SWE", "TUR", "UKR",
]

DOMESTIC_ENTITY_ALIASES = {
    "FRA": {"France", "French Republic", "French", "Republic of France"},
    "GRC": {"Greece", "Greek Republic", "Hellenic Republic", "Greek"},
    "TUR": {"Turkey", "Türkiye", "Turkiye", "Turkish Republic", "Turkish"},
    "ITA": {"Italy", "Italian Republic", "Italian"},
    "GBR": {"United Kingdom", "UK", "Great Britain", "Britain", "British", "England", "Scotland", "Wales", "Northern Ireland"},
}


def available_fact_files(country: str) -> list[Path]:
    """Return available facts parquet files for a country."""
    fact_dir = DATA_ROOT / "Table1_Fact" / country
    files = sorted(fact_dir.glob(f"{country}_*_facts.parquet"))
    if not files:
        raise FileNotFoundError(f"No facts parquet files found in {fact_dir}")
    return files


def available_countries() -> list[str]:
    """Return country folders that contain at least one facts parquet file."""
    fact_root = DATA_ROOT / "Table1_Fact"
    return sorted(
        path.name
        for path in fact_root.iterdir()
        if path.is_dir() and list(path.glob(f"{path.name}_*_facts.parquet"))
    )


def year_from_fact_path(path: Path) -> int:
    match = re.search(r"_(\d{4})_facts\.parquet$", path.name)
    if not match:
        raise ValueError(f"Cannot parse year from {path.name}")
    return int(match.group(1))


def load_country_facts(country: str, years: list[int] | None = None) -> pl.LazyFrame:
    """Lazy-load all available facts files for a country, optionally filtered by year."""
    files = available_fact_files(country)
    if years:
        wanted = set(years)
        files = [path for path in files if year_from_fact_path(path) in wanted]
    if not files:
        raise FileNotFoundError(f"No matching facts parquet files for {country} and years={years}")
    frames = [
        pl.scan_parquet(path).with_columns(pl.lit(year_from_fact_path(path)).alias("source_year"))
        for path in files
    ]
    return pl.concat(frames, how="vertical")


def country_prefix(country: str, years: list[int]) -> str:
    return f"{country}_{min(years)}_{max(years)}"


def remove_domestic_mentions(df: pl.DataFrame, country: str) -> pl.DataFrame:
    """Remove source-country self-mentions after normalization."""
    domestic = DOMESTIC_ENTITY_ALIASES.get(country, set())
    if not domestic:
        return df
    return df.filter(~pl.col("entity_content").is_in(sorted(domestic)))


def build_annotated_mentions(country: str, years: list[int]) -> pl.DataFrame:
    """Build the fully annotated mention table for one source country."""
    lf = load_country_facts(country, years=years)
    mentions = filters.build_migration_mentions(
        lf,
        use_topic=True,
        use_keywords=True,
    )
    mentions = remove_domestic_mentions(mentions, country)
    annotated = (
        mentions
        .pipe(typology.apply_typology)
        .pipe(geo.add_country_metadata)
        .filter((pl.col("target_iso3").is_null()) | (pl.col("target_iso3") != pl.col("country")))
        .pipe(concreteness.add_concreteness_scores)
        .pipe(diffusion.add_diffusion_classifications)
        .pipe(agency.add_policy_agency)
        .pipe(framing.add_narrative_framing)
        .pipe(direction.add_directional_schema)
    )
    return annotated


def save_country_analysis(country: str, years: list[int]) -> dict[str, Path]:
    """Run and save all extended analysis outputs for one country."""
    prefix = country_prefix(country, years)
    country_processed_dir = PROCESSED_DIR / prefix
    country_processed_dir.mkdir(parents=True, exist_ok=True)

    annotated = build_annotated_mentions(country, years)
    mention_path = country_processed_dir / f"{prefix}_migration_mentions_extended.parquet"
    annotated.write_parquet(mention_path)

    diffusion_edges = diffusion.build_diffusion_edges(annotated, source_country=country)
    agency_edges = agency.build_agency_edges(annotated, source_country=country)

    output_paths: dict[str, Path] = {"mentions_parquet": mention_path}
    output_paths.update(diffusion.save_diffusion_outputs(diffusion_edges, country_processed_dir, prefix=prefix))
    output_paths.update(agency.save_agency_outputs(agency_edges, country_processed_dir, prefix=prefix))
    output_paths.update(events.save_event_outputs(annotated, country_processed_dir, prefix=prefix))
    output_paths.update(evidence.save_evidence_outputs(annotated, diffusion_edges, country_processed_dir, prefix=prefix))
    output_paths.update(visualize.save_extended_figures(annotated, diffusion_edges, country_processed_dir))

    high_events = events.build_high_concreteness_events(annotated)
    visible_summary = events.visible_country_summary(high_events)
    output_paths.update(visualize.save_advanced_figures(
        df=annotated,
        agency_edges=agency_edges,
        events=high_events,
        visible_summary=visible_summary,
        processed_dir=country_processed_dir,
    ))

    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extended migration analysis for country folders.")
    parser.add_argument(
        "--countries",
        nargs="+",
        default=DEFAULT_COUNTRIES,
        help="Country codes to process. Defaults to all available country folders with facts files.",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=None,
        help="Optional years to process. Defaults to all available years per country.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    countries = args.countries or available_countries()
    for country in countries:
        files = available_fact_files(country)
        years = args.years or [year_from_fact_path(path) for path in files]
        print(f"\n=== {country} {min(years)}-{max(years)} ===")
        outputs = save_country_analysis(country, years)
        for name, path in outputs.items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
