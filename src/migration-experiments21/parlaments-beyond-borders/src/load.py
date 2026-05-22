"""Loading functions for ParlaMint parquet files."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.config import (
    DATA_ROOT,
    MASTER_AFFILIATIONS,
    MASTER_ORGS,
    MASTER_PEOPLE,
    SOURCE_COUNTRY,
    SOURCE_YEAR,
    SOURCE_YEARS,
)


def facts_file_for(country: str = SOURCE_COUNTRY, year: int = SOURCE_YEAR) -> Path:
    """Return the parquet path for one country-year facts file."""
    # Explanation: Keeping path construction here lets notebooks switch years cleanly.
    return DATA_ROOT / "Table1_Fact" / country / f"{country}_{year}_facts.parquet"


def facts_files_for(
    country: str = SOURCE_COUNTRY,
    years: list[int] | None = None,
) -> list[Path]:
    """Return existing facts paths for a country over multiple years."""
    # Explanation: The default is the full 2017-2022 France window.
    years = years or SOURCE_YEARS
    files = [facts_file_for(country, year) for year in years]
    missing = [path for path in files if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing facts parquet files:\n{missing_text}")
    return files


def load_facts_lazy(year: int | None = None) -> pl.LazyFrame:
    """Lazy-load the facts table for one configured country/year.

    Returns a LazyFrame - no data is read until you call .collect().
    """
    facts_file = facts_file_for(SOURCE_COUNTRY, year or SOURCE_YEAR)
    if not facts_file.exists():
        raise FileNotFoundError(
            f"Facts file not found at {facts_file}. "
            f"Check that you placed the year-specific parquet in "
            f"data/parlamint_extracted/Table1_Fact/FRA/"
        )
    return pl.scan_parquet(facts_file).with_columns(
        # Explanation: Make single-year and multi-year outputs share the same column.
        pl.lit(year or SOURCE_YEAR).alias("source_year")
    )


def load_facts_multi_year_lazy(
    years: list[int] | None = None,
    country: str = SOURCE_COUNTRY,
) -> pl.LazyFrame:
    """Lazy-load all configured country-year facts files as one LazyFrame."""
    # Explanation: Each scanned file gets a source_year before lazy concatenation.
    years = years or SOURCE_YEARS
    missing = [
        facts_file_for(country, year)
        for year in years
        if not facts_file_for(country, year).exists()
    ]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing facts parquet files:\n{missing_text}")
    frames = [
        pl.scan_parquet(facts_file_for(country, year)).with_columns(
            pl.lit(year).alias("source_year")
        )
        for year in years
    ]
    return pl.concat(frames, how="vertical")


def load_people() -> pl.DataFrame:
    """Load the master speaker demographics table."""
    return pl.read_parquet(MASTER_PEOPLE)


def load_orgs() -> pl.DataFrame:
    """Load the master organizations table (parties, parliaments)."""
    # Explanation: Some exports use Master_Organizations.parquet instead of Master_Orgs.parquet.
    if MASTER_ORGS.exists():
        return pl.read_parquet(MASTER_ORGS)
    fallback = MASTER_ORGS.parent / "Master_Organizations.parquet"
    return pl.read_parquet(fallback)


def load_affiliations() -> pl.DataFrame:
    """Load the speaker-to-organization junction table."""
    return pl.read_parquet(MASTER_AFFILIATIONS)


def inspect_schema(lf: pl.LazyFrame) -> None:
    """Print schema and row count without loading the whole table into memory."""
    print("=== Schema ===")
    for name, dtype in lf.collect_schema().items():
        print(f"  {name}: {dtype}")
    n_rows = lf.select(pl.len()).collect().item()
    print(f"\nTotal rows: {n_rows:,}")
