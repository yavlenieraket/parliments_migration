"""
llm_insights.py -- LLM-based interpretive layer over the existing migration
mentions pipeline. Reads your per-country processed parquets, sends targeted
prompts to a Llama model served by vLLM, writes structured outputs that drop
into the existing all_studied_countries_visualization_index.html as new
sections.

This is NOT a per-mention classifier. It runs ~600 LLM calls total, taking
patterns the rule-based pipeline already surfaced (asymmetric dyads, high-
concreteness events, top policy-agency edges, similar policy profiles) and
asks the LLM to interpret WHY they look that way. Output is structured-JSON
+ short narrative summaries, not a label per row.

Tasks:
    1 asymmetry          -- WHY are top-N asymmetric dyads asymmetric?
    2 events             -- structured extraction from high-concreteness events
    3 model_pressure     -- model / warning / cooperation / pressure positioning
    4 cross_target       -- how do N parliaments imagine the SAME target?
    5 direction_framing  -- domestic vs external framing per parliament
    6 policy_convergence -- pairs of parliaments with SIMILAR policy profiles:
                            what kind of relationship (alliance, rivalry,
                            ideological friction, silent convergence)?

Usage from project root, after vLLM server is up:
    # All tasks, defaults sensible
    python scripts/llm_insights.py --task all

    # Just one task
    python scripts/llm_insights.py --task policy_convergence --top-n 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from openai import APIError, AsyncOpenAI, RateLimitError
from tenacity import (
    retry, retry_if_exception_type,
    stop_after_attempt, wait_exponential,
)

log = logging.getLogger("llm_insights")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def detect_root(start: Path) -> Path:
    for p in [start.parent, *start.parents]:
        if (p / "data" / "processed").is_dir():
            return p
    raise SystemExit("Could not find data/processed/ above this script.")

ROOT = detect_root(Path(__file__).resolve())
PROCESSED = ROOT / "data" / "processed"
OUT_DIR   = PROCESSED / "ALL_AVAILABLE_COUNTRIES_comparisons" / "llm_insights"


# ---------------------------------------------------------------------------
# Generic LLM call -- guided JSON, retries, runs against your vLLM server
# ---------------------------------------------------------------------------
@retry(stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=2, min=2, max=30),
       retry=retry_if_exception_type((RateLimitError, APIError, asyncio.TimeoutError)),
       reraise=True)
async def call(client: AsyncOpenAI, model: str, system: str,
               user: str, schema: dict | None = None,
               max_tokens: int = 500) -> dict | str:
    extra = {"guided_json": schema} if schema else {}
    resp = await client.chat.completions.create(
        model=model, max_tokens=max_tokens, temperature=0.2, top_p=0.9,
        seed=42,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        extra_body=extra,
    )
    text = resp.choices[0].message.content or ""
    if schema:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"_parse_error": True, "_raw": text[:500]}
    return text


# ---------------------------------------------------------------------------
# Data loading -- defensive, handles your per-country file layout
# ---------------------------------------------------------------------------
def discover_country_dirs() -> list[Path]:
    """Per-country folders look like FRA_2017_2022/, GRC_2015_2022/ etc."""
    out = []
    for p in sorted(PROCESSED.iterdir()):
        if p.is_dir() and "_" in p.name and not p.name.startswith("ALL_"):
            iso = p.name.split("_")[0]
            if len(iso) == 3 and iso.isupper():
                out.append(p)
    return out


def load_all_mentions() -> pl.DataFrame:
    """Concatenate every country's annotated mentions table."""
    frames = []
    for cd in discover_country_dirs():
        candidates = list(cd.glob("*_migration_mentions_extended.parquet"))
        if not candidates:
            log.warning(f"no mentions parquet in {cd.name}, skipping")
            continue
        df = pl.read_parquet(candidates[0])
        if "country" not in df.columns:
            df = df.with_columns(pl.lit(cd.name.split("_")[0]).alias("country"))
        frames.append(df)
    if not frames:
        raise SystemExit("No per-country mentions parquet files found.")
    return pl.concat(frames, how="diagonal_relaxed")


def sample_snippets(df: pl.DataFrame, n: int = 5) -> list[str]:
    """Pull up to n short context windows for prompting."""
    if df.is_empty():
        return []
    sub = df.sample(n=min(n, df.height), seed=42)
    out = []
    for row in sub.iter_rows(named=True):
        prev = (row.get("sentence_content_previous") or "")[:300]
        curr = (row.get("sentence_content_current")  or "")[:300]
        nxt  = (row.get("sentence_content_next")     or "")[:300]
        out.append(f"[{row.get('source_year','?')}] ...{prev} **{curr}** {nxt}...")
    return out


# ---------------------------------------------------------------------------
# TASK 1 -- Why are top-N dyads asymmetric?
# ---------------------------------------------------------------------------
ASYM_SCHEMA = {
    "type": "object",
    "properties": {
        "primary_driver": {"type": "string",
            "enum": ["border_proximity", "crisis_visibility",
                     "EU_accession_governance", "war_or_conflict",
                     "diaspora_or_historical_ties", "historical_memory",
                     "economic_dependency", "other"]},
        "secondary_driver": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string", "maxLength": 600},
        "evidence_spans": {"type": "array", "items": {"type": "string"},
                           "maxItems": 3},
    },
    "required": ["primary_driver", "confidence", "rationale"],
}

ASYM_SYSTEM = """You are a political-discourse analyst. Given an ASYMMETRIC pair of parliaments — one talks about the other much more than vice versa — diagnose the most likely driver of that asymmetry.

Choose ONE primary driver from this closed list:
- border_proximity (the two countries share or are very close to a border)
- crisis_visibility (one country is a high-salience migration crisis site)
- EU_accession_governance (the relationship is shaped by EU membership/accession asymmetry)
- war_or_conflict (recent or ongoing war in/around one of the countries)
- diaspora_or_historical_ties (post-colonial or large diaspora flows)
- historical_memory (long-standing political/cultural reference)
- economic_dependency (labour migration tied to economic asymmetry)
- other (use sparingly; explain in rationale)

Rationale must be ≤80 words and grounded in the evidence snippets provided.
Output ONLY a JSON object matching the schema."""


async def task_asymmetry(client: AsyncOpenAI, model: str,
                         all_mentions: pl.DataFrame, top_n: int) -> None:
    log.info("[task 1] building dyad asymmetry table...")
    df = all_mentions.filter(
        pl.col("target_iso3").is_not_null() &
        (pl.col("entity_scope") == "country")
    )
    counts = (df.group_by(["country", "target_iso3"])
                .agg(pl.len().alias("n"))
                .filter(pl.col("n") >= 10))

    rev = counts.rename({"country": "target_iso3",
                         "target_iso3": "country",
                         "n": "n_reverse"})
    pairs = (counts.join(rev, on=["country", "target_iso3"], how="left")
                   .fill_null(0)
                   .with_columns(
                       (pl.col("n") / (pl.col("n_reverse") + 1)).alias("ratio"),
                       (pl.col("n") + pl.col("n_reverse")).alias("total"),
                   )
                   .filter((pl.col("ratio") >= 3) & (pl.col("total") >= 30))
                   .sort("ratio", descending=True)
                   .head(top_n))
    log.info(f"  selected {pairs.height} asymmetric pairs (ratio>=3, total>=30)")

    results: list[dict[str, Any]] = []
    for row in pairs.iter_rows(named=True):
        src, tgt = row["country"], row["target_iso3"]
        sub_fwd = df.filter((pl.col("country") == src) &
                            (pl.col("target_iso3") == tgt))
        sub_rev = df.filter((pl.col("country") == tgt) &
                            (pl.col("target_iso3") == src))
        user = (
            f"ASYMMETRIC PAIR: {src} talks about {tgt} {row['n']} times; "
            f"{tgt} talks about {src} only {row['n_reverse']} times "
            f"(ratio {row['ratio']:.1f}).\n\n"
            f"Sample mentions {src} -> {tgt}:\n"
            + "\n".join(f"  - {s}" for s in sample_snippets(sub_fwd, n=4))
            + (f"\n\nSample mentions {tgt} -> {src}:\n"
               + "\n".join(f"  - {s}" for s in sample_snippets(sub_rev, n=2))
               if sub_rev.height else "\n\n(No reverse mentions to sample.)")
        )
        out = await call(client, model, ASYM_SYSTEM, user, ASYM_SCHEMA)
        results.append({
            "source": src, "target": tgt,
            "n_forward": row["n"], "n_reverse": row["n_reverse"],
            "ratio": float(row["ratio"]), **(out if isinstance(out, dict) else {}),
        })
        log.info(f"  {src}->{tgt}: {out.get('primary_driver','?')}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(results).write_csv(OUT_DIR / "task1_asymmetry_drivers.csv")
    log.info(f"  wrote {OUT_DIR/'task1_asymmetry_drivers.csv'}")


# ---------------------------------------------------------------------------
# TASK 2 -- structured event extraction
# ---------------------------------------------------------------------------
EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "event_name":      {"type": "string", "maxLength": 150},
        "primary_actors":  {"type": "array", "items": {"type": "string"},
                            "maxItems": 5},
        "location_or_route": {"type": "string", "maxLength": 200},
        "affected_cohort": {"type": "string",
            "enum": ["refugees", "asylum_seekers", "students",
                     "economic_migrants", "high_skilled_workers",
                     "general_migration", "unspecified"]},
        "evidentiary_use": {"type": "string",
            "enum": ["restriction", "solidarity", "burden_sharing",
                     "institutional_reform", "neutral_reporting", "other"]},
        "confidence":      {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["event_name", "evidentiary_use", "confidence"],
}

EVENT_SYSTEM = """You extract structured event metadata from a single parliamentary snippet about migration.

event_name: a short noun phrase naming the event, NOT a description ("2015 Aylan Kurdi photo", "Belarus border push 2021", "Lampedusa boat sinking October 2013"). If the snippet only generically alludes to "the crisis" or "the recent events" without naming, use "unspecified".

evidentiary_use: what argumentative work does this event do IN THIS SNIPPET? Is it cited to justify restriction, express solidarity, demand burden-sharing, push institutional reform, or just neutrally reported?

Be conservative. If a field cannot be determined from the snippet, return "unspecified". Output ONLY a JSON object."""


async def task_events(client: AsyncOpenAI, model: str,
                      all_mentions: pl.DataFrame, top_n: int) -> None:
    log.info("[task 2] extracting high-concreteness events...")
    df = all_mentions.filter(
        pl.col("concreteness_score").is_not_null() &
        (pl.col("concreteness_score") >= 3.5)
    )
    parts = []
    per_country = max(1, top_n // max(1, df["country"].n_unique()))
    for c, sub in df.group_by("country"):
        parts.append(sub.sample(n=min(per_country, sub.height), seed=42))
    selection = pl.concat(parts).head(top_n)
    log.info(f"  selected {selection.height} high-concreteness snippets")

    results: list[dict[str, Any]] = []
    for row in selection.iter_rows(named=True):
        prev = (row.get("sentence_content_previous") or "")[:300]
        curr = (row.get("sentence_content_current")  or "")[:400]
        nxt  = (row.get("sentence_content_next")     or "")[:300]
        user = (
            f"Speaker parliament: {row['country']} (year {row.get('source_year','?')})\n"
            f"Mentioned entity: {row.get('entity_content','?')}\n\n"
            f"Snippet:\n{prev}\n**{curr}**\n{nxt}\n"
        )
        out = await call(client, model, EVENT_SYSTEM, user, EVENT_SCHEMA)
        results.append({
            "country": row["country"],
            "year": row.get("source_year"),
            "entity_content": row.get("entity_content"),
            "concreteness_score": row.get("concreteness_score"),
            **(out if isinstance(out, dict) else {}),
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(results).write_csv(OUT_DIR / "task2_extracted_events.csv")
    log.info(f"  wrote {OUT_DIR/'task2_extracted_events.csv'}")


# ---------------------------------------------------------------------------
# TASK 3 -- model / warning / cooperation / pressure
# ---------------------------------------------------------------------------
AGENCY_SCHEMA = {
    "type": "object",
    "properties": {
        "positioning": {"type": "string",
            "enum": ["model_to_emulate", "warning_case",
                     "cooperation_partner", "object_of_pressure",
                     "neutral_reference", "mixed"]},
        "confidence":  {"type": "number", "minimum": 0, "maximum": 1},
        "rationale":   {"type": "string", "maxLength": 400},
        "evidence_span": {"type": "string", "maxLength": 200},
    },
    "required": ["positioning", "confidence", "rationale"],
}

AGENCY_SYSTEM = """For the relationship between a speaker parliament and a mentioned country, decide how the mentioned country is positioned in the speaker's discourse:

- model_to_emulate: held up positively, "we should do what they do"
- warning_case: held up negatively, "we should NOT become like them"
- cooperation_partner: framed as a bilateral counterpart for joint action
- object_of_pressure: framed as a target of coercion, sanction, or demand
- neutral_reference: mentioned without normative or relational stance
- mixed: snippets show MORE THAN ONE of the above

This refines a rule-based pre-classification — feel free to disagree with it if the snippets warrant. Output ONLY a JSON object."""


async def task_model_pressure(client: AsyncOpenAI, model: str,
                              all_mentions: pl.DataFrame, top_n: int) -> None:
    log.info("[task 3] re-classifying top policy-agency pairs...")
    df = all_mentions.filter(
        (pl.col("entity_scope") == "country") &
        pl.col("target_iso3").is_not_null() &
        pl.col("policy_agency_type").is_not_null() &
        (pl.col("policy_agency_type") != "neutral_reporting")
    )
    pairs = (df.group_by(["country", "target_iso3"])
               .agg(pl.len().alias("n"),
                    pl.col("policy_agency_type").mode().first().alias("rule_label"))
               .filter(pl.col("n") >= 15)
               .sort("n", descending=True)
               .head(top_n))
    log.info(f"  selected {pairs.height} pairs to re-classify")

    results: list[dict[str, Any]] = []
    for row in pairs.iter_rows(named=True):
        src, tgt = row["country"], row["target_iso3"]
        sub = df.filter((pl.col("country") == src) &
                        (pl.col("target_iso3") == tgt))
        user = (
            f"Speaker parliament: {src}\n"
            f"Mentioned country: {tgt}\n"
            f"Rule-based pre-classification: {row['rule_label']}\n"
            f"Number of mentions: {row['n']}\n\n"
            f"Sample mentions:\n"
            + "\n".join(f"  - {s}" for s in sample_snippets(sub, n=5))
        )
        out = await call(client, model, AGENCY_SYSTEM, user, AGENCY_SCHEMA)
        results.append({
            "source": src, "target": tgt, "n": row["n"],
            "rule_based_label": row["rule_label"],
            **(out if isinstance(out, dict) else {}),
        })
        log.info(f"  {src}->{tgt}: rule={row['rule_label']} llm={out.get('positioning','?')}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(results).write_csv(OUT_DIR / "task3_model_pressure.csv")
    log.info(f"  wrote {OUT_DIR/'task3_model_pressure.csv'}")


# ---------------------------------------------------------------------------
# TASK 4 -- same target, different imagination across parliaments
# ---------------------------------------------------------------------------
KEY_TARGETS = ["TUR", "UKR", "SYR", "LBY", "MAR", "AFG", "BLR"]
CROSSTGT_SYSTEM = """You compare how different parliaments imagine the SAME target country in migration debates.

Given snippets from ONE parliament about ONE target, summarize that parliament's framing in ≤80 words, focusing on the BALANCE between:
- risk framing (threat, security, burden)
- solidarity framing (humanitarian, obligation, fellow-feeling)
- administrative framing (procedure, agreement, capacity)
- policy-remedy framing (what should be DONE, restriction vs cooperation)

Be specific to the snippets. Do not generalize. End with a one-sentence dominant-frame summary."""


async def task_cross_target(client: AsyncOpenAI, model: str,
                            all_mentions: pl.DataFrame) -> None:
    log.info("[task 4] cross-parliament framing of key targets...")
    df = all_mentions.filter(pl.col("target_iso3").is_in(KEY_TARGETS))
    results: list[dict[str, Any]] = []
    for target in KEY_TARGETS:
        sub = df.filter(pl.col("target_iso3") == target)
        if sub.is_empty():
            continue
        for src, src_sub in sub.group_by("country"):
            src = src[0] if isinstance(src, tuple) else src
            if src_sub.height < 5:
                continue
            user = (
                f"Speaker parliament: {src}\n"
                f"Target country: {target}\n"
                f"Total mentions: {src_sub.height}\n\n"
                f"Snippets:\n"
                + "\n".join(f"  - {s}" for s in sample_snippets(src_sub, n=6))
            )
            summary = await call(client, model, CROSSTGT_SYSTEM, user,
                                 schema=None, max_tokens=250)
            results.append({"target": target, "source": src,
                            "n_mentions": src_sub.height,
                            "framing_summary": summary})
            log.info(f"  {target} as seen by {src}: done")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(results).write_csv(OUT_DIR / "task4_cross_target_framings.csv")
    md = ["# Cross-parliament framings of key migration targets\n"]
    for tgt in KEY_TARGETS:
        rows = [r for r in results if r["target"] == tgt]
        if not rows:
            continue
        md.append(f"\n## {tgt}\n")
        for r in sorted(rows, key=lambda x: -x["n_mentions"]):
            md.append(f"### {r['source']} ({r['n_mentions']} mentions)\n")
            md.append(r["framing_summary"] + "\n")
    (OUT_DIR / "task4_cross_target_framings.md").write_text("\n".join(md))
    log.info(f"  wrote {OUT_DIR/'task4_cross_target_framings.md'}")


# ---------------------------------------------------------------------------
# TASK 5 -- domestic vs external agenda framing
# ---------------------------------------------------------------------------
DIRFRAME_SCHEMA = {
    "type": "object",
    "properties": {
        "dominant_framing": {"type": "string",
            "enum": ["capacity_pressure", "legal_obligation",
                     "security_issue", "demographic_need",
                     "humanitarian_responsibility", "mixed"]},
        "secondary_framing": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string", "maxLength": 500},
    },
    "required": ["dominant_framing", "confidence", "rationale"],
}

DIRFRAME_SYSTEM = """A parliament's migration discourse has a high share of inbound/internal mentions (migration INTO the speaker's polity). Classify how that domestic agenda is dominantly framed.

Choose ONE of:
- capacity_pressure (overload, numbers, system strain)
- legal_obligation (rights, treaties, asylum law)
- security_issue (border, crime, threat)
- demographic_need (labour shortages, ageing, replacement)
- humanitarian_responsibility (duty to protect, moral obligation)
- mixed (snippets clearly span 2+ with no dominance)

Output ONLY a JSON object."""


async def task_direction_framing(client: AsyncOpenAI, model: str,
                                 all_mentions: pl.DataFrame) -> None:
    log.info("[task 5] framing of inbound/internal mentions per parliament...")
    df = all_mentions.filter(
        pl.col("migration_direction").is_in(["inbound_internal",
                                             "outbound_from_domestic"])
    )
    results: list[dict[str, Any]] = []
    for src, sub in df.group_by("country"):
        src = src[0] if isinstance(src, tuple) else src
        if sub.height < 20:
            continue
        user = (
            f"Parliament: {src}\n"
            f"Total inbound/internal mentions: {sub.height}\n\n"
            f"Sample snippets:\n"
            + "\n".join(f"  - {s}" for s in sample_snippets(sub, n=8))
        )
        out = await call(client, model, DIRFRAME_SYSTEM, user, DIRFRAME_SCHEMA)
        results.append({"parliament": src, "n_mentions": sub.height,
                        **(out if isinstance(out, dict) else {})})
        log.info(f"  {src}: {out.get('dominant_framing','?')}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(results).write_csv(OUT_DIR / "task5_direction_framing.csv")
    log.info(f"  wrote {OUT_DIR/'task5_direction_framing.csv'}")


# ---------------------------------------------------------------------------
# TASK 6 -- POLICY CONVERGENCE & TENSION  (NEW)
#
# For pairs of parliaments with SIMILAR policy profiles (similar share of
# (cohort, policy_measure) cells), characterize the RELATIONSHIP:
#  - silent_convergence    : similar policies, low cross-mention
#  - mutual_alignment      : explicit recognition both ways
#  - unilateral_modeling   : one models on the other, asymmetric
#  - competitive_rivalry   : same migrants, treated as competitors
#  - ideological_friction  : same policies, framed as ideological enemies
#  - mutual_complaint      : both blame the same third party
#  - administrative_coord  : Frontex/Dublin-style operational coordination
#  - diffusion_chain       : visible policy transfer in snippets
#  - no_significant_interaction
# ---------------------------------------------------------------------------
CONVERGENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "relation_type": {"type": "string", "enum": [
            "mutual_acknowledged_alignment",
            "unilateral_modeling",
            "silent_convergence",
            "competitive_rivalry",
            "ideological_friction_despite_similarity",
            "diffusion_chain",
            "mutual_complaint",
            "administrative_coordination",
            "no_significant_interaction",
        ]},
        "shared_target_cohort": {"type": "string", "maxLength": 100},
        "tension_present": {"type": "boolean"},
        "tension_nature":  {"type": "string", "maxLength": 300},
        "rationale":       {"type": "string", "maxLength": 600},
        "evidence_spans":  {"type": "array", "items": {"type": "string"},
                            "maxItems": 3},
        "confidence":      {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["relation_type", "tension_present", "rationale", "confidence"],
}

CONVERGENCE_SYSTEM = """Two parliaments have CONVERGED on similar migration-policy profiles (similar focus on the same migrant cohorts × policy measures). You are given:
- their cosine similarity over (cohort × policy_measure) shares,
- the top shared focus areas,
- mention counts in each direction (A->B and B->A),
- sample cross-mentions where available.

Diagnose their RELATIONSHIP TYPE -- what KIND of convergence is this?

- mutual_acknowledged_alignment: both explicitly recognize each other as allies/models
- unilateral_modeling: A treats B as a model, B does not reciprocate
- silent_convergence: similar policies but minimal cross-mention; convergence is likely externally driven (EU directive, shared crisis, parallel domestic pressures)
- competitive_rivalry: similar focus produces competition -- typically for the SAME migrants (students, high-skilled workers) or for the SAME resources (EU funding, burden allocation)
- ideological_friction_despite_similarity: similar policies but each frames the other as an ideological adversary (very common when both are restrictive but blame each other for "the crisis")
- diffusion_chain: visible policy transfer -- one parliament references adopting a mechanism FROM the other
- mutual_complaint: both complain about the SAME third party (the EU, a transit country, "Brussels") in similar terms
- administrative_coordination: operational cooperation (Frontex, Dublin, joint patrols) framed in technocratic/procedural terms
- no_significant_interaction: too few cross-mentions to characterize meaningfully

Also flag tension_present = true if there is observable friction, competition, blame, or contradiction between them DESPITE the policy similarity. Describe tension_nature briefly.

Be especially attentive to these diagnostic patterns:
- SAME cohort + SAME policy + asymmetric mentions  -> often modeling or rivalry
- SAME cohort + SAME policy + mutual criticism     -> often ideological friction
- SAME cohort + SAME policy + silence              -> external pressure
- SAME cohort + SAME policy + technical coop terms -> administrative coordination

Ground rationale in the snippets. Output ONLY a JSON object."""


def _policy_profiles(all_mentions: pl.DataFrame,
                     min_mentions: int = 100
                     ) -> tuple[list[str], np.ndarray, list[str]]:
    """Return (parliaments, L2-normalized profile matrix, cell names)."""
    df = all_mentions.filter(
        pl.col("cohort").is_not_null() &
        pl.col("policy_measure").is_not_null() &
        (pl.col("cohort") != "general_migration") &
        (pl.col("policy_measure") != "general_policy")
    ).with_columns(
        pl.concat_str([pl.col("cohort"), pl.lit("|"),
                       pl.col("policy_measure")]).alias("cell")
    )
    # Drop parliaments with too few mentions for stable profiles
    pop = (df.group_by("country").agg(pl.len().alias("n"))
              .filter(pl.col("n") >= min_mentions))
    keep = pop["country"].to_list()
    if len(keep) < 2:
        raise SystemExit(
            f"Need at least 2 parliaments with >={min_mentions} cohort+policy "
            f"mentions; found {len(keep)}. Lower --top-n won't help -- "
            "check your cohort/policy_measure coverage upstream."
        )
    df = df.filter(pl.col("country").is_in(keep))

    counts = (df.group_by(["country", "cell"]).agg(pl.len().alias("n"))
                .pivot(values="n", index="country", on="cell")
                .fill_null(0))
    parliaments = counts["country"].to_list()
    cells = [c for c in counts.columns if c != "country"]
    M = counts.select(cells).to_numpy().astype(float)

    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return parliaments, M / norms, cells


def _select_convergence_pairs(parliaments: list[str], M: np.ndarray,
                              cells: list[str],
                              cross_counts: dict[tuple[str, str], int],
                              top_quiet: int, top_loud: int
                              ) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Two-stratum sampling: (1) top similar overall, (2) top similar + loud."""
    n = len(parliaments)
    sim = M @ M.T  # cosine since rows are L2-normalized

    pairs: list[dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            ab = cross_counts.get((parliaments[i], parliaments[j]), 0)
            ba = cross_counts.get((parliaments[j], parliaments[i]), 0)
            pairs.append({
                "i": i, "j": j,
                "a": parliaments[i], "b": parliaments[j],
                "similarity": float(sim[i, j]),
                "n_ab": ab, "n_ba": ba, "cross_total": ab + ba,
            })

    by_sim  = sorted(pairs, key=lambda p: -p["similarity"])[:top_quiet]
    loud    = [p for p in pairs
               if p["similarity"] >= 0.5 and p["cross_total"] >= 30]
    by_loud = sorted(loud, key=lambda p: -p["cross_total"])[:top_loud]

    seen, merged = set(), []
    for p in by_sim + by_loud:
        key = (p["a"], p["b"])
        if key not in seen:
            seen.add(key)
            merged.append(p)
    return merged, sim


async def task_policy_convergence(client: AsyncOpenAI, model: str,
                                  all_mentions: pl.DataFrame,
                                  top_n: int) -> None:
    log.info("[task 6] computing policy-profile similarities...")
    parliaments, M, cells = _policy_profiles(all_mentions)
    log.info(f"  {len(parliaments)} parliaments x {len(cells)} "
             f"(cohort,policy) cells")

    cross_counts: dict[tuple[str, str], int] = {}
    rolled = (all_mentions
              .filter((pl.col("entity_scope") == "country") &
                      pl.col("target_iso3").is_not_null())
              .group_by(["country", "target_iso3"])
              .agg(pl.len().alias("n")))
    for row in rolled.iter_rows(named=True):
        cross_counts[(row["country"], row["target_iso3"])] = row["n"]

    pairs, _ = _select_convergence_pairs(
        parliaments, M, cells, cross_counts,
        top_quiet=top_n, top_loud=max(5, top_n // 3),
    )
    log.info(f"  selected {len(pairs)} pairs (quiet stratum + loud stratum)")

    results: list[dict[str, Any]] = []
    for p in pairs:
        a, b = p["a"], p["b"]
        i, j = p["i"], p["j"]
        # Top 3 shared focus cells: product of normalized shares
        shared_scores = sorted(
            zip(cells, (M[i] * M[j]).tolist()),
            key=lambda x: -x[1],
        )[:3]
        shared_str = "; ".join(f"{c.replace('|', ' / ')}"
                               for c, _ in shared_scores if _ > 0)

        ab = all_mentions.filter((pl.col("country") == a) &
                                 (pl.col("target_iso3") == b))
        ba = all_mentions.filter((pl.col("country") == b) &
                                 (pl.col("target_iso3") == a))

        user = (
            f"PARLIAMENTS WITH CONVERGENT POLICY PROFILES\n"
            f"  A = {a}\n"
            f"  B = {b}\n"
            f"  Cosine similarity of (cohort x policy_measure) profiles: "
            f"{p['similarity']:.3f}\n"
            f"  Top shared focus: {shared_str or '(none above zero)'}\n"
            f"  Cross-mentions: {a} -> {b}: {p['n_ab']}   |   "
            f"{b} -> {a}: {p['n_ba']}\n\n"
        )
        if ab.height >= 3:
            user += (f"Sample mentions {a} -> {b}:\n" +
                     "\n".join(f"  - {s}" for s in sample_snippets(ab, n=4))
                     + "\n\n")
        else:
            user += f"({a} barely mentions {b}: {ab.height} mentions.)\n\n"
        if ba.height >= 3:
            user += (f"Sample mentions {b} -> {a}:\n" +
                     "\n".join(f"  - {s}" for s in sample_snippets(ba, n=4)))
        else:
            user += f"({b} barely mentions {a}: {ba.height} mentions.)"

        out = await call(client, model, CONVERGENCE_SYSTEM, user,
                         CONVERGENCE_SCHEMA, max_tokens=700)
        results.append({
            "parliament_a": a, "parliament_b": b,
            "policy_similarity": round(p["similarity"], 4),
            "shared_focus": shared_str,
            "n_a_to_b": p["n_ab"], "n_b_to_a": p["n_ba"],
            **(out if isinstance(out, dict) else {}),
        })
        log.info(f"  {a}~{b} sim={p['similarity']:.2f}: "
                 f"{out.get('relation_type','?')}  "
                 f"tension={out.get('tension_present','?')}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(results).write_csv(OUT_DIR / "task6_policy_convergence.csv")

    # Also write a short markdown summary by relation_type for quick reading
    md = ["# Policy convergence and tension between parliaments\n",
          "Pairs grouped by LLM-diagnosed relationship type. "
          "`tension=True` flags pairs where similar policies coexist with "
          "observable friction, rivalry, or mutual blame.\n"]
    by_type: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_type.setdefault(r.get("relation_type", "unknown"), []).append(r)
    for rt in sorted(by_type):
        md.append(f"\n## {rt} ({len(by_type[rt])} pairs)\n")
        for r in sorted(by_type[rt],
                        key=lambda x: -x.get("policy_similarity", 0)):
            tflag = " ⚠ tension" if r.get("tension_present") else ""
            md.append(f"- **{r['parliament_a']} ~ {r['parliament_b']}** "
                      f"(sim {r['policy_similarity']:.2f},"
                      f" A→B {r['n_a_to_b']}, B→A {r['n_b_to_a']}){tflag}")
            md.append(f"  - shared focus: {r['shared_focus']}")
            md.append(f"  - {r.get('rationale','')}")
    (OUT_DIR / "task6_policy_convergence.md").write_text("\n".join(md))
    log.info(f"  wrote {OUT_DIR/'task6_policy_convergence.csv'}")
    log.info(f"  wrote {OUT_DIR/'task6_policy_convergence.md'}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
async def run(args: argparse.Namespace) -> None:
    log.info(f"loading all per-country mentions from {PROCESSED}...")
    all_mentions = load_all_mentions()
    log.info(f"  loaded {all_mentions.height:,} mentions across "
             f"{all_mentions['country'].n_unique()} parliaments")

    client = AsyncOpenAI(base_url=args.server, api_key="EMPTY", timeout=120.0)
    tasks = args.task.split(",") if args.task != "all" else \
        ["asymmetry", "events", "model_pressure", "cross_target",
         "direction_framing", "policy_convergence"]

    for t in tasks:
        try:
            if t == "asymmetry":
                await task_asymmetry(client, args.model, all_mentions, args.top_n)
            elif t == "events":
                await task_events(client, args.model, all_mentions, args.top_n * 4)
            elif t == "model_pressure":
                await task_model_pressure(client, args.model, all_mentions, args.top_n * 2)
            elif t == "cross_target":
                await task_cross_target(client, args.model, all_mentions)
            elif t == "direction_framing":
                await task_direction_framing(client, args.model, all_mentions)
            elif t == "policy_convergence":
                await task_policy_convergence(client, args.model, all_mentions, args.top_n)
            else:
                log.warning(f"unknown task: {t}")
        except Exception as e:
            log.error(f"task {t} failed: {e!r}")
            continue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",   default="all",
        help="all | asymmetry | events | model_pressure | cross_target | "
             "direction_framing | policy_convergence | comma-separated")
    ap.add_argument("--server", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model",  default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--top-n",  type=int, default=50)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
