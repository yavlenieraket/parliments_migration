"""Project-wide paths and constants."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "parlamint_extracted"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_COUNTRY = "FRA"
SOURCE_YEAR = 2018

FACTS_FILE = DATA_ROOT / "Table1_Fact" / SOURCE_COUNTRY / f"{SOURCE_COUNTRY}_{SOURCE_YEAR}_facts.parquet"
MASTER_PEOPLE = DATA_ROOT / "Table2_People" / "Master_People.parquet"
MASTER_ORGS = DATA_ROOT / "Table3_Orgs" / "Master_Orgs.parquet"
MASTER_AFFILIATIONS = DATA_ROOT / "Table4_Affiliations" / "Master_Affiliations.parquet"

# The actual ParlaMint-en.ana CAP topic code in the parquet is "immig".
MIGRATION_TOPIC = "immig"

# === Exclusion lists in English (ParlaMint-en.ana) ===

# France itself - keep capitalization variants and adjective forms
# that NER might tag as LOC.
FRANCE_SELF = {
    "France", "French Republic", "Republic of France",
    "French", "the French",
    "our country", "our nation", "the nation", "the Republic",
    "Republic", "La France", "Franco", "French State", "State",
    "the Hexagon", "Hexagon",
}

# French cities and regions that NER will frequently catch.
FRENCH_CITIES_REGIONS = {
    "Paris", "Marseille", "Lyon", "Toulouse", "Nice", "Nantes",
    "Strasbourg", "Bordeaux", "Lille", "Rennes", "Montpellier",
    "Calais", "Briançon", "Menton", "Le Havre", "Grenoble",
    "Île-de-France", "Ile-de-France", "Brittany", "Normandy",
    "Provence", "Occitanie", "Alsace", "Lorraine", "Aquitaine",
    "Auvergne", "Burgundy", "Corsica", "Picardy",
    "Alpes", "Pas", "Matignon", "Sangatte", "Calaisis",
    "Ouistreham", "Maritimes", "Seine", "Haute", "Savoie",
    "Hauts", "Grande", "Marseilles", "Synthe", "Metz", "Roissy",
}

# Geographic terms that are not countries.
# Maghreb, Sahel, Balkans may be analytically meaningful collective references,
# but they are not single countries.
GEO_NON_COUNTRY = {
    "Europe", "Mediterranean", "Atlantic", "Sahel",
    "Maghreb", "Balkans", "Middle East", "Africa", "Asia",
    "the Union",
    "Member States", "States",
    "Schengen Area", "Schengen",
    "the West", "Western Europe", "Eastern Europe",
    "North Africa", "Sub-Saharan Africa", "Central Europe",
    "sub-Saharan Africa", "the Mediterranean", "Mediterranean Sea",
    "the Channel", "Channel", "North Sea", "South",
    "Dublin", "Brussels", "Sandhurst", "Aquarius", "Travellers",
}

# French overseas territories - politically French but discussed as
# separate actors in migration debates. Keep as a distinct category.
FRENCH_OVERSEAS = {
    "Mayotte", "French Guiana", "Guyane",
    "Guyana",
    "Reunion", "Réunion", "La Réunion",
    "Martinique", "Guadeloupe",
    "New Caledonia", "Nouvelle-Calédonie",
    "French Polynesia", "Polynesia",
}

EU_ENTITIES = {
    "European Union",
    "EU",
}

EUROPEAN_COUNTRIES = {
    "Albania", "Andorra", "Austria", "Belarus", "Belgium",
    "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus",
    "Czech Republic", "Czechia", "Denmark", "Estonia", "Finland",
    "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy",
    "Kosovo", "Latvia", "Liechtenstein", "Lithuania", "Luxembourg",
    "Malta", "Moldova", "Monaco", "Montenegro", "Netherlands",
    "North Macedonia", "Norway", "Poland", "Portugal", "Romania",
    "Russia", "San Marino", "Serbia", "Slovakia", "Slovenia",
    "Spain", "Sweden", "Switzerland", "Ukraine", "United Kingdom",
    "Vatican City",
}

EXCLUDE_FROM_FOREIGN = FRANCE_SELF | FRENCH_CITIES_REGIONS | GEO_NON_COUNTRY
