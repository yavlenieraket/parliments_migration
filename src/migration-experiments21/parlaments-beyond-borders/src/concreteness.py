"""Concreteness / abstractness scoring for migration discourse contexts.

The preferred production method is a Brysbaert-style concreteness lexicon
with 1-5 word scores. This module also includes a transparent fallback
heuristic so the notebook runs without downloading external resources.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import polars as pl


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")

# Explanation: Short function words do not carry concreteness evidence.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "for", "from", "had", "has", "have", "he", "her", "his",
    "i", "in", "is", "it", "its", "may", "not", "of", "on", "or",
    "our", "she", "that", "the", "their", "them", "there", "these",
    "they", "this", "those", "to", "was", "we", "were", "which",
    "who", "will", "with", "you", "your",
}

# Explanation: These concrete markers reflect physical places, bodies, objects,
# documents, and institutions that make a migration reference more specific.
CONCRETE_MARKERS = {
    "airport", "boat", "boats", "border", "borders", "camp", "camps",
    "card", "centre", "center", "child", "children", "city", "coast",
    "detention", "document", "documents", "family", "families", "fingerprint",
    "hospital", "house", "housing", "island", "judge", "office", "passport",
    "plane", "police", "port", "prefecture", "road", "school", "ship",
    "shelter", "street", "territory", "train", "visa", "visas",
}

# Explanation: These abstract markers reflect values, principles, generalized
# claims, and speculation rather than concrete places or actors.
ABSTRACT_MARKERS = {
    "approach", "burden", "citizenship", "civilization", "cohesion",
    "concept", "debate", "dignity", "duty", "identity", "integration",
    "justice", "model", "morality", "nation", "obligation", "principle",
    "problem", "responsibility", "security", "solidarity", "sovereignty",
    "strategy", "system", "value", "values",
}


def tokenize(text: str | None) -> list[str]:
    """Tokenize English text into lowercase lexical tokens."""
    # Explanation: A small regex tokenizer is enough for reproducible pilot scoring.
    if not isinstance(text, str):
        return []
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if token.lower() not in STOPWORDS
    ]


def load_concreteness_lexicon(path: Path) -> dict[str, float]:
    """Load a Brysbaert-style concreteness lexicon from CSV/TSV."""
    # Explanation: Accept common column names used in concreteness norm files.
    if not path.exists():
        raise FileNotFoundError(path)

    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        lexicon: dict[str, float] = {}
        for row in reader:
            word = (
                row.get("Word")
                or row.get("word")
                or row.get("lemma")
                or row.get("Lemma")
            )
            score = (
                row.get("Conc.M")
                or row.get("concreteness")
                or row.get("score")
                or row.get("Score")
            )
            if not word or not score:
                continue
            try:
                lexicon[word.lower()] = float(score)
            except ValueError:
                continue
    return lexicon


def _entity_tokens(entities: list[str] | None) -> set[str]:
    """Return tokens belonging to named entities."""
    # Explanation: Named entities are treated as maximally concrete by design.
    if not entities:
        return set()
    tokens: set[str] = set()
    for entity in entities:
        tokens.update(tokenize(entity))
    return tokens


def _fallback_token_score(token: str) -> float:
    """Return transparent heuristic concreteness when no lexicon is available."""
    # Explanation: The fallback keeps the workflow runnable and auditable, but it
    # should be replaced with Brysbaert norms before making strong claims.
    if token in CONCRETE_MARKERS:
        return 4.5
    if token in ABSTRACT_MARKERS:
        return 2.0
    if token.endswith(("tion", "ity", "ism", "ness", "ment")):
        return 2.6
    return 3.2


def sentence_concreteness_score(
    text: str | None,
    entities: list[str] | None = None,
    lexicon: dict[str, float] | None = None,
) -> float | None:
    """Calculate the mean concreteness score for one context window."""
    # Explanation: Score range follows Brysbaert norms: 1 abstract, 5 concrete.
    tokens = tokenize(text)
    if not tokens:
        return None

    entity_tokens = _entity_tokens(entities)
    scores: list[float] = []
    for token in tokens:
        if token in entity_tokens:
            scores.append(5.0)
        elif lexicon and token in lexicon:
            scores.append(lexicon[token])
        else:
            scores.append(_fallback_token_score(token))
    return round(sum(scores) / len(scores), 3)


def concreteness_diagnostics(
    text: str | None,
    entities: list[str] | None = None,
) -> dict[str, object]:
    """Return transparent evidence used by the fallback concreteness scorer."""
    # Explanation: Diagnostics make the score auditable in context tables.
    tokens = tokenize(text)
    entity_tokens = _entity_tokens(entities)
    concrete_hits = sorted({token for token in tokens if token in CONCRETE_MARKERS})
    abstract_hits = sorted({token for token in tokens if token in ABSTRACT_MARKERS})
    entity_hits = sorted({token for token in tokens if token in entity_tokens})
    return {
        "token_count": len(tokens),
        "concrete_marker_hits": ", ".join(concrete_hits),
        "abstract_marker_hits": ", ".join(abstract_hits),
        "entity_token_hits": ", ".join(entity_hits),
        "concrete_marker_count": len(concrete_hits),
        "abstract_marker_count": len(abstract_hits),
    }


def concreteness_band(score: float | None) -> str:
    """Convert a numeric score into an interpretable pilot-relative band."""
    # Explanation: With the fallback heuristic, most scores cluster near 3.2.
    # These bands are therefore relative diagnostics, not universal thresholds.
    if score is None:
        return "unknown"
    if score < 3.2:
        return "abstract_leaning"
    if score >= 3.3:
        return "concrete_leaning"
    return "mixed"


def concreteness_method(lexicon: dict[str, float] | None = None) -> str:
    """Return the method label stored in the output table."""
    # Explanation: This makes it explicit whether results use lexicon scores or fallback.
    return "brysbaert_lexicon" if lexicon else "transparent_fallback_heuristic"


def add_concreteness_scores(
    df: pl.DataFrame,
    lexicon: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Add concreteness score and method columns to each mention."""
    # Explanation: We score the context window and give the mentioned entity max
    # concreteness because named places/institutions make the sentence specific.
    method = concreteness_method(lexicon)
    scored = df.with_columns([
        pl.struct(["context_window", "entity_content"])
        .map_elements(
            lambda row: sentence_concreteness_score(
                row["context_window"],
                entities=[row["entity_content"]],
                lexicon=lexicon,
            ),
            return_dtype=pl.Float64,
        )
        .alias("concreteness_score"),
        pl.lit(method).alias("concreteness_method"),
    ])
    return scored.with_columns([
        pl.col("concreteness_score")
        .map_elements(concreteness_band, return_dtype=pl.Utf8)
        .alias("concreteness_band"),
        pl.struct(["context_window", "entity_content"])
        .map_elements(
            lambda row: concreteness_diagnostics(
                row["context_window"],
                entities=[row["entity_content"]],
            )["token_count"],
            return_dtype=pl.Int64,
        )
        .alias("concreteness_token_count"),
        pl.struct(["context_window", "entity_content"])
        .map_elements(
            lambda row: concreteness_diagnostics(
                row["context_window"],
                entities=[row["entity_content"]],
            )["concrete_marker_hits"],
            return_dtype=pl.Utf8,
        )
        .alias("concrete_marker_hits"),
        pl.struct(["context_window", "entity_content"])
        .map_elements(
            lambda row: concreteness_diagnostics(
                row["context_window"],
                entities=[row["entity_content"]],
            )["abstract_marker_hits"],
            return_dtype=pl.Utf8,
        )
        .alias("abstract_marker_hits"),
    ])


def regional_concreteness_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Summarize concreteness by WEOG/non-WEOG analytical group."""
    # Explanation: This is the direct table for testing the regional hypothesis.
    return (
        df
        .drop_nulls("concreteness_score")
        .group_by("weog_group")
        .agg([
            pl.len().alias("n_mentions"),
            pl.col("concreteness_score").mean().round(3).alias("mean_concreteness"),
            pl.col("concreteness_score").median().round(3).alias("median_concreteness"),
            pl.col("concreteness_score").std().round(3).alias("sd_concreteness"),
        ])
        .sort("mean_concreteness")
    )
