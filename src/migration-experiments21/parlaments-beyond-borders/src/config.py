"""Project-wide paths and constants."""

from pathlib import Path


# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "parlamint_extracted"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# === Source files ===
SOURCE_COUNTRY = "FRA"
SOURCE_YEAR = 2018

FACTS_FILE = DATA_ROOT / "Table1_Fact" / SOURCE_COUNTRY / f"{SOURCE_COUNTRY}_{SOURCE_YEAR}_facts.parquet"
MASTER_PEOPLE = DATA_ROOT / "Table2_People" / "Master_People.parquet"
MASTER_ORGS = DATA_ROOT / "Table3_Orgs" / "Master_Orgs.parquet"
MASTER_AFFILIATIONS = DATA_ROOT / "Table4_Affiliations" / "Master_Affiliations.parquet"

# === Filtering constants ===
MIGRATION_TOPIC = "immigration"

# Entities to exclude when searching for "foreign" countries in French debates.
# These are France itself, French cities/regions, and non-country geographic terms.
# This list is meant to be iteratively refined as you inspect results.
FRANCE_SELF = {
    "France", "République française", "Hexagone", "métropole",
    "français", "française", "Français", "Française",
}

FRENCH_CITIES_REGIONS = {
    "Paris", "Marseille", "Lyon", "Toulouse", "Nice", "Nantes",
    "Strasbourg", "Bordeaux", "Calais", "Briançon", "Menton",
    "Île-de-France", "Bretagne", "Normandie", "Provence", "Occitanie",
    "Alsace", "Lorraine", "Aquitaine", "Auvergne",
}

# Geographic terms that are not countries. Discuss with the team how to handle
# these - some (Maghreb, Sahel) carry meaningful collective references.
GEO_NON_COUNTRY = {
    "Europe", "Méditerranée", "Atlantique", "Sahel",
    "Maghreb", "Balkans", "Moyen-Orient", "Afrique", "Asie",
    "Union européenne", "UE",
}

# French overseas territories - politically part of France but often discussed
# as separate actors in migration debates. Keep them in a separate category
# rather than mixing with "foreign" or "domestic".
FRENCH_OVERSEAS = {
    "Mayotte", "Guyane", "Réunion", "Martinique", "Guadeloupe",
    "Nouvelle-Calédonie", "Polynésie",
}

EXCLUDE_FROM_FOREIGN = FRANCE_SELF | FRENCH_CITIES_REGIONS | GEO_NON_COUNTRY
