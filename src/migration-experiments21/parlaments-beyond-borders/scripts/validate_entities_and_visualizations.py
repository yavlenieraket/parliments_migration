"""Validate strict entity scope and generated visualization files."""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import generate_all_country_visualizations as allviz  # noqa: E402
import run_extended_analysis as runner  # noqa: E402
from src import geo  # noqa: E402
from src.config import PROCESSED_DIR  # noqa: E402


BAD_ENTITY_EXAMPLES = {
    "Athens", "Chios", "Hatay", "Windrush", "Brexit", "East", "South",
    "North", "West", "Lampedusa", "Berlin", "London", "Rome", "Dublin",
    "Brussels", "Coquelles", "Castle", "Gorizia", "Veneto", "Mont Blanc",
}


def country_outputs() -> list[tuple[str, str, Path]]:
    years_by_country = allviz.available_years_by_country()
    outputs = []
    for country, years in years_by_country.items():
        prefix = runner.country_prefix(country, years)
        outputs.append((country, prefix, PROCESSED_DIR / prefix))
    return outputs


def validate_entities() -> list[str]:
    """Return validation errors for target entity cleanliness."""
    errors: list[str] = []
    for country, prefix, country_dir in country_outputs():
        path = country_dir / f"{prefix}_migration_mentions_extended.parquet"
        if not path.exists():
            errors.append(f"{country}: missing processed mentions table {path}")
            continue
        df = pl.read_parquet(path)
        if "entity_scope" not in df.columns:
            errors.append(f"{country}: missing entity_scope column")
            continue
        invalid = df.filter(pl.col("entity_scope") == "invalid")
        if invalid.height:
            sample = invalid.get_column("entity_content").unique().head(20).to_list()
            errors.append(f"{country}: invalid entity_scope rows={invalid.height}, sample={sample}")
        bad_found = sorted(set(df.get_column("entity_content").unique().to_list()) & BAD_ENTITY_EXAMPLES)
        if bad_found:
            errors.append(f"{country}: disallowed city/event/vague entities leaked: {bad_found}")
        allowed_scopes = {"country", "european_union", "territory_region", "analytical_region"}
        scopes = set(df.get_column("entity_scope").unique().to_list())
        extra_scopes = sorted(scopes - allowed_scopes)
        if extra_scopes:
            errors.append(f"{country}: unexpected entity scopes: {extra_scopes}")
        country_rows = df.filter(pl.col("entity_scope") == "country")
        missing_iso3 = country_rows.filter(pl.col("target_iso3").is_null())
        if missing_iso3.height:
            sample = missing_iso3.get_column("entity_content").unique().head(20).to_list()
            errors.append(f"{country}: country-scope rows missing ISO3, sample={sample}")
        recomputed_invalid = (
            df
            .with_columns(
                pl.col("entity_content")
                .map_elements(geo.entity_scope_for_entity, return_dtype=pl.Utf8)
                .alias("_recomputed_scope")
            )
            .filter(pl.col("_recomputed_scope") == "invalid")
        )
        if recomputed_invalid.height:
            sample = recomputed_invalid.get_column("entity_content").unique().head(20).to_list()
            errors.append(f"{country}: recomputed invalid entities, sample={sample}")
    return errors


def validate_visualizations() -> list[str]:
    """Return validation errors for generated chart files."""
    errors: list[str] = []
    index = PROCESSED_DIR / "all_studied_countries_visualization_index.html"
    if not index.exists():
        errors.append(f"missing visualization index: {index}")
    else:
        text = index.read_text(encoding="utf-8")
        for required in [
            "Analytical Report: Asymmetries, Trends, Special Cases, and LLM Leads",
            "What it asks:",
            "How to read it:",
            "Cross-country comparisons",
            "Combined dyadic model",
        ]:
            if required not in text:
                errors.append(f"visualization index missing explanatory text: {required}")

    for country, prefix, country_dir in country_outputs():
        html_files = sorted(country_dir.glob("figures_*/*.html"))
        png_files = sorted(country_dir.glob("figures_*/*.png"))
        vl_files = sorted(country_dir.glob("figures_*/*.vl.json"))
        if len(html_files) < 10:
            errors.append(f"{country}: expected at least 10 HTML figure files, found {len(html_files)}")
        if len(png_files) < 8:
            errors.append(f"{country}: expected at least 8 PNG figure files, found {len(png_files)}")
        if len(vl_files) < 8:
            errors.append(f"{country}: expected at least 8 Vega-Lite specs, found {len(vl_files)}")
        for path in html_files + png_files + vl_files:
            if path.stat().st_size < 500:
                errors.append(f"{country}: suspiciously small visualization file: {path}")

    comparison_dir = PROCESSED_DIR / f"{allviz.COMBINED_PREFIX}_comparisons" / "figures_cross_country"
    comparison_png = sorted(comparison_dir.rglob("*.png"))
    comparison_html = sorted(comparison_dir.rglob("*.html"))
    comparison_specs = sorted(comparison_dir.rglob("*.vl.json"))
    if len(comparison_png) < 13 or len(comparison_html) < 18 or len(comparison_specs) < 13:
        errors.append(
            "cross-country comparison bundle incomplete: "
            f"png={len(comparison_png)}, html={len(comparison_html)}, specs={len(comparison_specs)}"
        )
    for path in comparison_png + comparison_html + comparison_specs:
        if path.stat().st_size < 500:
            errors.append(f"suspiciously small comparison visualization file: {path}")

    dyadic_dir = PROCESSED_DIR / f"{allviz.COMBINED_PREFIX}_dyadic_data_model" / "figures_data_model"
    dyadic_png = sorted(dyadic_dir.glob("*.png"))
    dyadic_html = sorted(dyadic_dir.glob("*.html"))
    if len(dyadic_png) < 5 or len(dyadic_html) < 5:
        errors.append(f"dyadic visualization bundle incomplete: png={len(dyadic_png)}, html={len(dyadic_html)}")
    for path in dyadic_png + dyadic_html:
        if path.stat().st_size < 500:
            errors.append(f"suspiciously small dyadic visualization file: {path}")
    return errors


def main() -> None:
    errors = validate_entities() + validate_visualizations()
    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("VALIDATION PASSED")
    for country, prefix, country_dir in country_outputs():
        df = pl.read_parquet(country_dir / f"{prefix}_migration_mentions_extended.parquet")
        scopes = df.group_by("entity_scope").agg(pl.len().alias("n")).sort("n", descending=True)
        print(f"{country} {prefix}: {df.height:,} mentions")
        print(scopes)


if __name__ == "__main__":
    main()
