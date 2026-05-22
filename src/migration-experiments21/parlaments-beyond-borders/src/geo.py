"""Country metadata helpers for mentioned entities."""

from __future__ import annotations

import polars as pl

from src.config import EU_ENTITIES, FRENCH_OVERSEAS


# Explanation: Exact country-name mapping is safer here than running NER again,
# because the parquet already contains extracted entities in entity_content.
COUNTRY_ISO3 = {
    "Afghanistan": "AFG",
    "Albania": "ALB",
    "Algeria": "DZA",
    "Andorra": "AND",
    "Angola": "AGO",
    "Argentina": "ARG",
    "Armenia": "ARM",
    "Australia": "AUS",
    "Austria": "AUT",
    "Bangladesh": "BGD",
    "Belarus": "BLR",
    "Belgium": "BEL",
    "Brazil": "BRA",
    "Bulgaria": "BGR",
    "Burkina Faso": "BFA",
    "Burundi": "BDI",
    "Cameroon": "CMR",
    "Canada": "CAN",
    "Central African Republic": "CAF",
    "Chad": "TCD",
    "China": "CHN",
    "Comoros": "COM",
    "Cote d'Ivoire": "CIV",
    "Croatia": "HRV",
    "Cyprus": "CYP",
    "Czech Republic": "CZE",
    "Czechia": "CZE",
    "Denmark": "DNK",
    "Djibouti": "DJI",
    "Egypt": "EGY",
    "Eritrea": "ERI",
    "Ethiopia": "ETH",
    "Finland": "FIN",
    "Georgia": "GEO",
    "Germany": "DEU",
    "Greece": "GRC",
    "Guinea": "GIN",
    "Haiti": "HTI",
    "Hungary": "HUN",
    "Iceland": "ISL",
    "India": "IND",
    "Iran": "IRN",
    "Iraq": "IRQ",
    "Ireland": "IRL",
    "Israel": "ISR",
    "Italy": "ITA",
    "Japan": "JPN",
    "Jordan": "JOR",
    "Kosovo": "XKX",
    "Latvia": "LVA",
    "Lebanon": "LBN",
    "Libya": "LBY",
    "Lithuania": "LTU",
    "Luxembourg": "LUX",
    "Madagascar": "MDG",
    "Mali": "MLI",
    "Malta": "MLT",
    "Mexico": "MEX",
    "Moldova": "MDA",
    "Monaco": "MCO",
    "Morocco": "MAR",
    "Netherlands": "NLD",
    "New Zealand": "NZL",
    "Niger": "NER",
    "Nigeria": "NGA",
    "Norway": "NOR",
    "Pakistan": "PAK",
    "Poland": "POL",
    "Portugal": "PRT",
    "Qatar": "QAT",
    "Romania": "ROU",
    "Russia": "RUS",
    "Rwanda": "RWA",
    "Saudi Arabia": "SAU",
    "Senegal": "SEN",
    "Serbia": "SRB",
    "Somalia": "SOM",
    "South Africa": "ZAF",
    "South Sudan": "SSD",
    "Spain": "ESP",
    "Sudan": "SDN",
    "Suriname": "SUR",
    "Sweden": "SWE",
    "Switzerland": "CHE",
    "Syria": "SYR",
    "Tanzania": "TZA",
    "Tunisia": "TUN",
    "Turkey": "TUR",
    "Ukraine": "UKR",
    "United Kingdom": "GBR",
    "United States": "USA",
    "Yemen": "YEM",
}

# Explanation: UN WEOG-style grouping for the hypothesis comparison. France itself
# is excluded earlier, but the broader WEOG comparison still matters for mentions.
WEOG_COUNTRIES = {
    "Andorra", "Australia", "Austria", "Belgium", "Canada", "Denmark",
    "Finland", "Germany", "Greece", "Iceland", "Ireland", "Israel",
    "Italy", "Liechtenstein", "Luxembourg", "Malta", "Monaco",
    "Netherlands", "New Zealand", "Norway", "Portugal", "San Marino",
    "Spain", "Sweden", "Switzerland", "Turkey", "United Kingdom",
    "United States",
}


def iso3_for_entity(entity: str | None) -> str | None:
    """Return ISO3 country code for a canonical entity name when available."""
    # Explanation: EU and French overseas territories are not assigned foreign-state ISO3.
    if entity is None or entity in EU_ENTITIES or entity in FRENCH_OVERSEAS:
        return None
    return COUNTRY_ISO3.get(entity)


def weog_group_for_entity(entity: str | None) -> str:
    """Return WEOG / non-WEOG / special group for a mentioned entity."""
    # Explanation: Special analytical groups are kept separate from country regions.
    if entity in EU_ENTITIES:
        return "european_union"
    if entity in FRENCH_OVERSEAS:
        return "french_overseas"
    if entity in WEOG_COUNTRIES:
        return "weog"
    if entity in COUNTRY_ISO3:
        return "non_weog"
    return "unknown"


def add_country_metadata(df: pl.DataFrame) -> pl.DataFrame:
    """Add ISO3 and WEOG/non-WEOG comparison labels to a mentions dataframe."""
    # Explanation: These fields support regional concreteness and network analysis.
    return df.with_columns([
        pl.col("entity_content")
        .map_elements(iso3_for_entity, return_dtype=pl.Utf8)
        .alias("target_iso3"),
        pl.col("entity_content")
        .map_elements(weog_group_for_entity, return_dtype=pl.Utf8)
        .alias("weog_group"),
    ])
