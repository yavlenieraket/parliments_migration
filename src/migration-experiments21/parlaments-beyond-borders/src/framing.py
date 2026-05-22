"""Narrative frames, argumentative schemes, and conceptual definitions."""

from __future__ import annotations

import re

import polars as pl


# Explanation: A 61-frame taxonomy for migration imagination. Each frame has
# a compact marker set for a transparent first-pass classifier.
NARRATIVE_FRAMES = {
    "economic_contribution": {"contribution", "contribute", "growth", "tax", "labour", "labor", "workforce"},
    "economic_burden": {"burden", "cost", "costs", "expense", "welfare", "benefits"},
    "labour_shortage": {"shortage", "recruit", "seasonal worker", "vacancies"},
    "skills_talent": {"talent", "high-skilled", "qualified", "researcher", "engineer"},
    "student_mobility": {"student", "university", "study visa", "campus"},
    "entrepreneurship": {"entrepreneur", "startup", "business", "investment"},
    "demographic_need": {"demographic", "ageing", "aging", "birth rate", "population decline"},
    "humanitarian_obligation": {"humanitarian", "obligation", "duty", "protection", "save lives"},
    "solidarity": {"solidarity", "share responsibility", "relocation", "burden sharing"},
    "human_rights": {"human rights", "dignity", "fundamental rights", "rights"},
    "child_protection": {"children", "minor", "minors", "unaccompanied"},
    "family_reunification": {"family reunification", "family", "families"},
    "gender_vulnerability": {"women", "girls", "violence against women", "pregnant"},
    "victimhood": {"victim", "victims", "trafficked", "persecuted"},
    "cultural_threat": {"culture", "values", "way of life", "identity", "civilization"},
    "religious_threat": {"islam", "religion", "radical", "secularism"},
    "national_identity": {"national identity", "nation", "republic", "citizenship"},
    "security_threat": {"security", "threat", "danger", "risk"},
    "terrorism": {"terrorism", "terrorist", "radicalisation", "radicalization"},
    "crime": {"crime", "criminal", "delinquency", "violence"},
    "border_crisis": {"border", "frontier", "crossing", "coast guard"},
    "sovereignty": {"sovereignty", "control our borders", "national law"},
    "rule_of_law": {"rule of law", "law", "legal", "illegal"},
    "administrative_efficiency": {"procedure", "processing", "deadline", "administrative"},
    "bureaucratic_delay": {"delay", "backlog", "waiting", "slow"},
    "asylum_right": {"right of asylum", "asylum", "refugee status"},
    "return_deportation": {"return", "deportation", "expulsion", "removal"},
    "detention_control": {"detention", "retention", "custody"},
    "regularization": {"regularization", "regularisation", "amnesty"},
    "integration_success": {"successful integration", "integrated", "language course"},
    "integration_failure": {"failed integration", "communitarianism", "ghetto"},
    "education": {"school", "education", "training"},
    "housing_pressure": {"housing", "shelter", "accommodation"},
    "healthcare": {"health", "hospital", "medical"},
    "local_capacity": {"municipality", "local", "places", "capacity"},
    "european_solidarity": {"european solidarity", "european mechanism", "relocation"},
    "eu_failure": {"europe has failed", "failure of europe", "dublin failure"},
    "schengen_dublin": {"schengen", "dublin regulation", "dublin"},
    "international_law": {"international law", "geneva convention", "convention"},
    "bilateral_cooperation": {"bilateral", "agreement with", "cooperation with"},
    "development_aid": {"development aid", "aid", "development policy"},
    "root_causes": {"root causes", "causes of migration", "poverty", "climate"},
    "war_conflict": {"war", "conflict", "civil war", "bombing"},
    "climate_migration": {"climate", "drought", "famine", "environmental"},
    "smuggling_trafficking": {"smuggler", "smugglers", "trafficking", "passeurs"},
    "sea_rescue": {"sea", "rescue", "shipwreck", "drowning", "mediterranean"},
    "camp_conditions": {"camp", "camps", "moria", "calais jungle"},
    "colonial_history": {"colonial", "colonization", "postcolonial"},
    "moral_panic": {"invasion", "flood", "wave", "submersion"},
    "technical_management": {"management", "system", "database", "fingerprint"},
    "data_statistics": {"number", "statistics", "figures", "rate"},
    "legal_category": {"definition", "category", "status"},
    "policy_model": {"model", "example", "best practice"},
    "policy_failure": {"failure", "ineffective", "does not work"},
    "competition_attractiveness": {"attractive", "competition", "talent"},
    "reciprocity_exchange": {"exchange", "reciprocal", "partnership"},
    "financial_instrument": {"fund", "funding", "budget", "money"},
    "public_opinion": {"public opinion", "citizens think", "fear of"},
    "electoral_politics": {"election", "vote", "party", "populism"},
    "media_visibility": {"media", "news", "images"},
    "neutral_reporting": set(),
}

FRAME_POLARITY = {
    "economic_contribution": "positive_benefit",
    "skills_talent": "positive_benefit",
    "student_mobility": "positive_benefit",
    "entrepreneurship": "positive_benefit",
    "humanitarian_obligation": "positive_sympathy",
    "solidarity": "positive_sympathy",
    "human_rights": "positive_sympathy",
    "child_protection": "positive_sympathy",
    "family_reunification": "positive_sympathy",
    "victimhood": "positive_sympathy",
    "economic_burden": "negative_risk",
    "cultural_threat": "negative_risk",
    "religious_threat": "negative_risk",
    "security_threat": "negative_risk",
    "terrorism": "negative_risk",
    "crime": "negative_risk",
    "border_crisis": "negative_risk",
    "moral_panic": "negative_risk",
}

DEFINITION_RE = re.compile(
    r"\b(migration|immigration|asylum|a refugee|refugees|a migrant|migrants|an immigrant|immigrants)\s+"
    r"(is|are|means|mean|refers to|consists of|can be defined as)\b",
    flags=re.IGNORECASE,
)

CONSEQUENCE_MARKERS = {"therefore", "as a result", "will lead to", "risk", "consequence", "because of this"}
PRACTICAL_REASONING_MARKERS = {"in order to", "so that", "we must", "we should", "the aim is", "to achieve"}


def _first_marker(text: str | None, markers: set[str]) -> str:
    if not isinstance(text, str):
        return ""
    lower = text.lower()
    for marker in sorted(markers, key=len, reverse=True):
        if marker and marker in lower:
            return marker
    return ""


def classify_narrative_frame(text: str | None) -> str:
    """Return the first matching narrative frame."""
    # Explanation: The first marker match is an auditable zero-shot substitute.
    for frame, markers in NARRATIVE_FRAMES.items():
        if _first_marker(text, markers):
            return frame
    return "neutral_reporting"


def matched_narrative_frame_marker(text: str | None) -> str:
    """Return marker evidence for the narrative frame."""
    for markers in NARRATIVE_FRAMES.values():
        marker = _first_marker(text, markers)
        if marker:
            return marker
    return ""


def narrative_polarity(frame: str | None) -> str:
    """Return ternary narrative polarity for plotting."""
    return FRAME_POLARITY.get(frame or "", "neutral_administrative")


def classify_argument_scheme(text: str | None) -> str:
    """Classify argument scheme as consequences, practical reasoning, definition, or other."""
    # Explanation: These are surface-pattern labels for locating examples quickly.
    if not isinstance(text, str):
        return "other"
    if DEFINITION_RE.search(text):
        return "conceptual_definition"
    if _first_marker(text, CONSEQUENCE_MARKERS):
        return "argument_from_consequences"
    if _first_marker(text, PRACTICAL_REASONING_MARKERS):
        return "practical_reasoning"
    return "other"


def extract_conceptual_definition(text: str | None) -> str:
    """Extract a compact conceptual definition candidate."""
    # Explanation: Keep this simple and auditable; LLM extraction can refine it later.
    if not isinstance(text, str):
        return ""
    match = DEFINITION_RE.search(text)
    if not match:
        return ""
    start = max(0, match.start() - 80)
    end = min(len(text), match.end() + 220)
    return " ".join(text[start:end].replace("||", " ").split())


def add_narrative_framing(df: pl.DataFrame) -> pl.DataFrame:
    """Add narrative frame, polarity, argument scheme, and definition columns."""
    # Explanation: These labels describe how migration is imagined around country references.
    return df.with_columns([
        pl.col("context_window")
        .map_elements(classify_narrative_frame, return_dtype=pl.Utf8)
        .alias("narrative_frame"),
        pl.col("context_window")
        .map_elements(matched_narrative_frame_marker, return_dtype=pl.Utf8)
        .alias("narrative_frame_marker"),
        pl.col("context_window")
        .map_elements(classify_argument_scheme, return_dtype=pl.Utf8)
        .alias("argument_scheme"),
        pl.col("context_window")
        .map_elements(extract_conceptual_definition, return_dtype=pl.Utf8)
        .alias("conceptual_definition"),
        pl.lit("keyword_rules_llm_ready").alias("narrative_method"),
    ]).with_columns(
        pl.col("narrative_frame")
        .map_elements(narrative_polarity, return_dtype=pl.Utf8)
        .alias("narrative_polarity")
    )
