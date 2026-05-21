"""Loading functions for ParlaMint parquet files."""

import polars as pl

from src.config import FACTS_FILE, MASTER_AFFILIATIONS, MASTER_ORGS, MASTER_PEOPLE


def load_facts_lazy() -> pl.LazyFrame:
    """Lazy-load the facts table for the configured country/year.

    Returns a LazyFrame - no data is read until you call .collect().
    """
    if not FACTS_FILE.exists():
        raise FileNotFoundError(
            f"Facts file not found at {FACTS_FILE}. "
            f"Check that you placed FRA_2018_facts.parquet in "
            f"data/parlamint_extracted/Table1_Fact/FRA/"
        )
    return pl.scan_parquet(FACTS_FILE)


def load_people() -> pl.DataFrame:
    """Load the master speaker demographics table."""
    return pl.read_parquet(MASTER_PEOPLE)


def load_orgs() -> pl.DataFrame:
    """Load the master organizations table (parties, parliaments)."""
    return pl.read_parquet(MASTER_ORGS)


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
