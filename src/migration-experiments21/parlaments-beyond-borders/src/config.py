"""Project-wide paths and constants."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "parlamint_extracted"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_COUNTRY = "FRA"
SOURCE_YEAR = 2018
SOURCE_YEARS = list(range(2017, 2023))

FACTS_FILE = DATA_ROOT / "Table1_Fact" / SOURCE_COUNTRY / f"{SOURCE_COUNTRY}_{SOURCE_YEAR}_facts.parquet"
MASTER_PEOPLE = DATA_ROOT / "Table2_People" / "Master_People.parquet"
MASTER_ORGS = DATA_ROOT / "Table3_Orgs" / "Master_Orgs.parquet"
MASTER_AFFILIATIONS = DATA_ROOT / "Table4_Affiliations" / "Master_Affiliations.parquet"

# The actual ParlaMint-en.ana CAP topic code in the parquet is "immig".
MIGRATION_TOPIC = "immig"

# Explanation: Keyword fallback for rows that mention migration vocabulary even when
# the debate_topic tag is missing or too broad. The terms are English because this
# project uses ParlaMint-en.ana.
MIGRATION_KEYWORDS = {
    "refugee", "refugees",
    "migrant", "migrants",
    "immigrant", "immigrants",
    "asylum", "asylum seeker", "asylum seekers",
    "migration", "immigration",
}

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
    "Aubervilliers", "La Chapelle", "Bouches", "Moselle", "Nord",
    "Hérault", "Isère", "Pyrénées", "Béziers", "Sarthe", "Dieppe",
    "Roubaix", "Col de l'Échelle", "Essonne", "Mesnil", "Conflans",
    "Ain", "Loire", "Yvelines", "Calvados", "Bobigny", "Garonne",
    "Croisilles", "La Celle", "Perthus", "Palaiseau", "Amiens",
    "Cherbourg", "Chapelle", "Meurthe", "Puy", "Denain", "Franche",
    "Annemasse", "Sète", "Nanterre", "Orléans", "Dunkirk", "Marne",
    "Rhône", "Val", "Hautes",
}

# Analytical region/route entities that are allowed as target entities even
# though they are not countries. These are deliberately broad regions, migration
# routes, or territorial cases. Cities, events, people, and vague directions are
# excluded from target-entity analysis and can only appear inside context/fact
# snippets.
ANALYTICAL_REGIONS = {
    "Europe", "European States", "Western Europe", "Eastern Europe",
    "Central Europe", "Africa", "African States", "North Africa",
    "Sub-Saharan Africa", "West Africa", "Saharan Africa", "Asia",
    "Asia Minor", "Middle East", "Near East", "Levant", "Maghreb",
    "Sahel", "Balkans", "Balkan route", "Mediterranean",
    "Mediterranean Sea", "Central Mediterranean", "Atlantic",
    "North Sea", "Black Sea", "Aegean Sea", "English Channel",
    "Channel", "Schengen Area", "Schengen", "Caribbean",
    "West Indies", "Indian Ocean", "Pacific", "Sahara",
    "Darfur", "Chechnya", "Nagorny Karabakh", "Kashmir",
    "Overseas Territories",
}

# Geographic terms and NER artifacts that are not valid target entities.
GEO_NON_COUNTRY = {
    "the Union",
    "Member States", "States", "Member State",
    "the West", "the Mediterranean", "the Channel",
    "Balkan",
    "South", "North", "East", "West",
    "Dublin", "Brussels", "Sandhurst", "Aquarius", "Travellers",
    "Sorbonne", "The Republic", "Roya Valley", "Idlib", "No Border",
    "Fifth Republic", "Médecins", "Chapel", "Chapel Gate",
    "La République", "Islamic State", "Territory", "Saharan",
    "Place de la République", "Ile", "Samos", "Lampedusa",
    "Daesh", "Tindouf", "-Saharan Africa", "New York",
    "London", "Berlin", "Abidjan", "Kabul", "Marrakech", "Agadez",
    "Niamey", "Ankara", "Dakar", "Saint", "Sainte", "Terre",
    "Côte",
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
    "Wallis and Futuna",
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
