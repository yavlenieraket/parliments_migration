"""


Tasks:
    1 asymmetry        -- WHY are top-N asymmetric dyads asymmetric?
    2 events           -- structured extraction from high-concreteness events
    3 model_pressure   -- model / warning / cooperation / pressure positioning
    4 cross_target     -- how do N parliaments imagine the SAME target?
    5 direction_framing-- domestic vs external framing per parliament
    6 policy_convergence
    7 argumentative_schemes
    8 parliament_profiles
    9 yearly_attention
    10 target_salience
    postprocess        -- repair known split-column output bugs

Usage from project root, after vLLM server is up:
    # All tasks, saved to a NEW folder by default:
    # data/processed/ALL_AVAILABLE_COUNTRIES_comparisons/llm_insights_phase1_2026_05_26
    python scripts/llm_prompts_phase1.py --task all

    # Just one task
    python scripts/llm_prompts_phase1.py --task argumentative_schemes

    # Override the output folder name if needed
    python scripts/llm_prompts_phase1.py --task all --run-name llm_insights_phase1_rerun
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl
from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError
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
DEFAULT_RUN_NAME = "llm_insights_phase1_2026_05_26"
OUT_DIR = (
    PROCESSED
    / "ALL_AVAILABLE_COUNTRIES_comparisons"
    / DEFAULT_RUN_NAME
)


# ---------------------------------------------------------------------------
# Generic LLM call -- guided JSON, retries, runs against your vLLM server
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> dict[str, Any] | None:
    """Parse raw, fenced, or lightly wrapped model JSON."""
    cleaned = text.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalise_structured(obj: dict[str, Any], schema: dict) -> dict[str, Any]:
    """Repair predictable local model slips while preserving the raw schema."""
    props = schema.get("properties", {})
    aliases = {
        "dominant_framing": ["dominant_frame"],
        "relation_type": ["relationship_type"],
        "positioning": ["position", "country_position", "model_relation"],
    }
    fixed = dict(obj)
    for canonical, alternates in aliases.items():
        if canonical not in fixed or fixed.get(canonical) in (None, ""):
            for alt in alternates:
                if fixed.get(alt) not in (None, ""):
                    fixed[canonical] = fixed[alt]
                    break

    if "confidence" in props and isinstance(fixed.get("confidence"), str):
        conf = fixed["confidence"].strip().lower()
        fixed["confidence"] = {
            "very high": 0.95,
            "high": 0.85,
            "medium": 0.6,
            "moderate": 0.6,
            "low": 0.35,
            "very low": 0.15,
        }.get(conf, fixed["confidence"])

    if fixed.get("evidentiary_use") == "neutral report":
        fixed["evidentiary_use"] = "neutral_reporting"

    enum_maps = {
        "positioning": {
            "learning_emulation_from": "model_to_emulate",
            "exchange_cooperation": "cooperation_partner",
            "coercion_intervention_to": "object_of_pressure",
        },
        "evidentiary_use": {
            "justify restriction": "restriction",
            "justify_restriction": "restriction",
            "express solidarity": "solidarity",
            "demand burden-sharing": "burden_sharing",
            "demand_burden_sharing": "burden_sharing",
            "push institutional reform": "institutional_reform",
            "push_institutional_reform": "institutional_reform",
            "neutral report": "neutral_reporting",
            "unspecified": "other",
        },
        "scheme": {
            "no_clear_scheme": "none_or_unclear",
            "argument_from_definition": "argument_from_sign",
        },
    }
    for key, mapping in enum_maps.items():
        if isinstance(fixed.get(key), str):
            fixed[key] = mapping.get(fixed[key], fixed[key])

    # Keep output tables stable: include every schema property, with nulls for
    # fields the local model omitted.
    return {key: fixed.get(key) for key in props}


def _csv_safe_value(value: Any) -> Any:
    """CSV writer helper: stringify nested JSON values deterministically."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_records_csv(records: list[dict[str, Any]], path: Path) -> None:
    """Write heterogeneous LLM records without Polars nested-column failures."""
    safe = [
        {key: _csv_safe_value(value) for key, value in row.items()}
        for row in records
    ]
    pl.DataFrame(safe).write_csv(path)


def _schema_instruction(schema: dict) -> str:
    props = schema.get("properties", {})
    required = schema.get("required", [])
    lines = [
        "CRITICAL JSON FORMAT RULES:",
        "Return one raw JSON object only. Do not use markdown fences.",
        "Include these keys exactly: " + ", ".join(props.keys()) + ".",
        "Required keys that must be non-null: " + ", ".join(required) + ".",
        "For confidence, use a number from 0 to 1, never words like high/medium/low.",
        "If the evidence is weak, still include all keys; use a low numeric confidence and explain uncertainty in rationale.",
    ]
    for key, spec in props.items():
        if "enum" in spec:
            lines.append(f"{key} allowed values: " + ", ".join(spec["enum"]) + ".")
    return "\n".join(lines)


@retry(stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=2, min=2, max=30),
       retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError, asyncio.TimeoutError)),
       reraise=True)
async def call(client: AsyncOpenAI, model: str, system: str,
               user: str, schema: dict | None = None,
               max_tokens: int = 500) -> dict | str:
    extra = {"guided_json": schema} if schema else {}
    system_text = system
    if schema:
        system_text = system + "\n\n" + _schema_instruction(schema)
    request: dict[str, Any] = dict(
        model=model, max_tokens=max_tokens, temperature=0.1, top_p=0.9,
        seed=42,
        messages=[{"role": "system", "content": system_text},
                  {"role": "user", "content": user}],
    )
    if schema:
        request["response_format"] = {"type": "json_object"}
        request["extra_body"] = extra
    resp = await client.chat.completions.create(**request)
    text = resp.choices[0].message.content or ""
    if schema:
        parsed = _extract_json(text)
        if parsed is None:
            return {"_parse_error": True, "_raw": text[:1000]}
        return _normalise_structured(parsed, schema)
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
        # Ensure 'country' (source) column exists
        if "country" not in df.columns:
            df = df.with_columns(pl.lit(cd.name.split("_")[0]).alias("country"))
        frames.append(df)
    if not frames:
        raise SystemExit("No per-country mentions parquet files found.")
    return pl.concat(frames, how="diagonal_relaxed")


def load_csv_per_country(filename_suffix: str) -> pl.DataFrame:
    """Concatenate per-country CSVs like *_policy_agency_edges.csv."""
    frames = []
    for cd in discover_country_dirs():
        path = next(cd.glob(f"*{filename_suffix}"), None)
        if path and path.exists():
            df = pl.read_csv(path)
            if "source_country" not in df.columns:
                df = df.with_columns(pl.lit(cd.name.split("_")[0])
                                       .alias("source_country"))
            frames.append(df)
    if not frames:
        raise SystemExit(f"No *{filename_suffix} files found.")
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


def top_distribution(df: pl.DataFrame, col: str, n: int = 5) -> list[dict[str, Any]]:
    """Return top categories as count/share dictionaries for prompt evidence."""
    if col not in df.columns or df.is_empty():
        return []
    total = df.height
    out = []
    vc = (df.filter(pl.col(col).is_not_null())
            .group_by(col)
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(n))
    for row in vc.iter_rows(named=True):
        value = row.get(col)
        if value in (None, ""):
            continue
        count = int(row["count"])
        out.append({
            "value": value,
            "count": count,
            "share": round(count / max(total, 1), 4),
        })
    return out


def safe_mean(df: pl.DataFrame, col: str) -> float | None:
    if col not in df.columns or df.is_empty():
        return None
    value = df.select(pl.col(col).cast(pl.Float64, strict=False).mean()).item()
    return None if value is None else round(float(value), 4)


def safe_unique(df: pl.DataFrame, col: str) -> int:
    if col not in df.columns or df.is_empty():
        return 0
    return int(df.select(pl.col(col).n_unique()).item())


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

ASYM_SYSTEM = """You are a political-discourse analyst. Given an ASYMMETRIC pair of parliaments — one talks about the other at a higher NORMALIZED RATE than vice versa — diagnose the most likely driver of that asymmetry.

Important scope condition: both countries in the pair have parliamentary corpus data in this project. Do NOT treat one-sided mentions of countries without parliamentary data (for example Syria, Afghanistan, Libya, Morocco, or Belarus if no source parliament is available) as reciprocal asymmetry. Those cases are target salience, not parliament-to-parliament asymmetry.

Important normalization condition: use the normalized attention ratio as the main evidence, not raw mention counts. Normalization means: mentions of the target divided by the source parliament's analyzed token volume when token counts are available; otherwise divide by the source parliament's total country-mention volume. Raw counts are context only, because parliaments have different corpus sizes.

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
    # Build parliament -> parliament mention counts. This task is only valid
    # where BOTH sides have source-parliament data in the corpus. External
    # targets such as Syria are important, but they are target salience rather
    # than reciprocal parliament-to-parliament asymmetry.
    parliament_isos = set(all_mentions["country"].unique().to_list())
    all_country_mentions = all_mentions.filter(
        pl.col("target_iso3").is_not_null() &
        (pl.col("entity_scope") == "country") &
        pl.col("country").is_in(sorted(parliament_isos))
    )
    source_mention_totals = (
        all_country_mentions
        .group_by("country")
        .agg(pl.len().alias("source_total_country_mentions"))
    )
    target_mention_totals = source_mention_totals.rename({
        "country": "target_iso3",
        "source_total_country_mentions": "target_total_country_mentions",
    })
    if "concreteness_token_count" in all_country_mentions.columns:
        denominator_type = "context_tokens"
        source_denominators = (
            all_country_mentions
            .with_columns(
                pl.col("concreteness_token_count")
                .fill_null(0)
                .cast(pl.Float64, strict=False)
                .alias("_normalization_units")
            )
            .group_by("country")
            .agg(pl.sum("_normalization_units").alias("source_normalization_units"))
        )
    else:
        denominator_type = "country_mentions"
        source_denominators = source_mention_totals.rename({
            "source_total_country_mentions": "source_normalization_units"
        })
    target_denominators = source_denominators.rename({
        "country": "target_iso3",
        "source_normalization_units": "target_normalization_units",
    })
    df = all_country_mentions.filter(
        pl.col("target_iso3").is_in(sorted(parliament_isos)) &
        (pl.col("country") != pl.col("target_iso3"))
    )
    log.info(
        f"  limiting asymmetry to {len(parliament_isos)} source parliaments; "
        f"{df.height} parliament-to-parliament mentions remain; "
        f"normalizing by {denominator_type}"
    )
    counts = (
        df.group_by(["country", "target_iso3"])
        .agg(pl.len().alias("n"))
        .join(source_mention_totals, on="country", how="left")
        .join(target_mention_totals, on="target_iso3", how="left")
        .join(source_denominators, on="country", how="left")
        .join(target_denominators, on="target_iso3", how="left")
        .with_columns([
            pl.max_horizontal(pl.col("source_normalization_units"), pl.lit(1.0))
            .alias("source_normalization_units"),
            pl.max_horizontal(pl.col("target_normalization_units"), pl.lit(1.0))
            .alias("target_normalization_units"),
        ])
        .with_columns([
            (pl.col("n") / pl.col("source_normalization_units")).alias("source_rate"),
            (pl.col("n") * 1000 / pl.col("source_normalization_units"))
            .alias("source_rate_per_1000_normalization_units"),
            (pl.col("n") * 1000 / pl.col("source_total_country_mentions"))
            .alias("source_rate_per_1000_country_mentions"),
        ])
    )

    # Self-join for reverse direction
    rev = counts.select([
        pl.col("country").alias("target_iso3"),
        pl.col("target_iso3").alias("country"),
        pl.col("n").alias("n_reverse"),
        pl.col("source_rate").alias("target_rate"),
        pl.col("source_rate_per_1000_normalization_units")
        .alias("target_rate_per_1000_normalization_units"),
        pl.col("source_rate_per_1000_country_mentions")
        .alias("target_rate_per_1000_country_mentions"),
    ])
    pairs = (
        counts
        .filter(pl.col("n") >= 10)
        .join(rev, on=["country", "target_iso3"], how="left")
        .with_columns([
            pl.col("n_reverse").fill_null(0),
            pl.col("target_rate").fill_null(0),
            pl.col("target_rate_per_1000_normalization_units").fill_null(0),
            pl.col("target_rate_per_1000_country_mentions").fill_null(0),
        ])
        .with_columns([
            (pl.col("n") + pl.col("n_reverse")).alias("total"),
            (pl.col("n") / (pl.col("n_reverse") + 1)).alias("raw_ratio"),
            ((pl.col("source_rate") + 0.5 / pl.col("source_normalization_units")) /
             ((pl.col("n_reverse") + 0.5) / pl.col("target_normalization_units")))
            .alias("normalized_ratio"),
            ((pl.col("source_rate") + 0.5 / pl.col("source_normalization_units")).log() -
             (((pl.col("n_reverse") + 0.5) / pl.col("target_normalization_units")).log()))
            .alias("normalized_log_ratio"),
        ])
        .filter((pl.col("normalized_ratio") >= 3) & (pl.col("total") >= 30))
        .sort("normalized_ratio", descending=True)
        .head(top_n)
    )
    log.info(
        f"  selected {pairs.height} normalized asymmetric pairs "
        "(normalized_ratio>=3, total>=30)"
    )

    results: list[dict[str, Any]] = []
    for row in pairs.iter_rows(named=True):
        src, tgt = row["country"], row["target_iso3"]
        sub_fwd = df.filter((pl.col("country") == src) &
                            (pl.col("target_iso3") == tgt))
        sub_rev = df.filter((pl.col("country") == tgt) &
                            (pl.col("target_iso3") == src))
        user = (
            f"ASYMMETRIC PAIR (NORMALIZED): {src} talks about {tgt} "
            f"{row['n']} times out of {row['source_normalization_units']:.0f} "
            f"{denominator_type} "
            f"({row['source_rate_per_1000_normalization_units']:.2f} per 1,000). "
            f"{tgt} talks about {src} {row['n_reverse']} times out of "
            f"{row['target_normalization_units']:.0f} {denominator_type} "
            f"({row['target_rate_per_1000_normalization_units']:.2f} per 1,000). "
            f"Normalized ratio: {row['normalized_ratio']:.1f}. "
            f"Raw count ratio: {row['raw_ratio']:.1f}.\n\n"
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
            "normalization_denominator_type": denominator_type,
            "source_normalization_units": float(row["source_normalization_units"]),
            "target_normalization_units": float(row["target_normalization_units"]),
            "source_total_country_mentions": row["source_total_country_mentions"],
            "target_total_country_mentions": row["target_total_country_mentions"],
            "source_rate_per_1000_normalization_units": float(row["source_rate_per_1000_normalization_units"]),
            "target_rate_per_1000_normalization_units": float(row["target_rate_per_1000_normalization_units"]),
            "source_rate_per_1000_country_mentions": float(row["source_rate_per_1000_country_mentions"]),
            "target_rate_per_1000_country_mentions": float(row["target_rate_per_1000_country_mentions"]),
            "normalized_ratio": float(row["normalized_ratio"]),
            "normalized_log_ratio": float(row["normalized_log_ratio"]),
            "raw_ratio": float(row["raw_ratio"]),
            # Keep ratio for backwards compatibility with old dashboards, but
            # from this version onward it is the normalized attention ratio.
            "ratio": float(row["normalized_ratio"]),
            **(out if isinstance(out, dict) else {}),
        })
        log.info(f"  {src}->{tgt}: {out.get('primary_driver','?')}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_records_csv(results, OUT_DIR / "task1_asymmetry_drivers.csv")
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
    # Exclude abstract aggregate entities that produced 162/200 'unspecified'
    # in the previous run. These are organizations or regions, not events.
    EXCLUDE_ENTITIES = {
        "European Union", "Europe", "Schengen", "Schengen Area", "EU",
        "Council of Europe", "United Nations", "UN", "NATO",
        "European Commission", "European Parliament", "European Council",
    }
    df = all_mentions.filter(
        pl.col("concreteness_score").is_not_null() &
        (pl.col("concreteness_score") >= 3.5) &
        (~pl.col("entity_content").is_in(list(EXCLUDE_ENTITIES))) &
        # Prefer entities that look like specific places (have ISO3 mapping)
        pl.col("target_iso3").is_not_null()
    )
    log.info(f"  after filtering abstract entities: {df.height} candidates")
    # Stratify by source country so we don't get only the loudest parliament
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
    write_records_csv(results, OUT_DIR / "task2_extracted_events.csv")
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
    write_records_csv(results, OUT_DIR / "task3_model_pressure.csv")
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
    write_records_csv(results, OUT_DIR / "task4_cross_target_framings.csv")
    # Also write a readable per-target markdown for quick reading
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
    "additionalProperties": False,
}

DIRFRAME_SYSTEM = """A parliament's migration discourse has a high share of inbound/internal mentions (migration INTO the speaker's polity). Classify how that domestic agenda is dominantly framed.

Choose ONE of:
- capacity_pressure (overload, numbers, system strain)
- legal_obligation (rights, treaties, asylum law)
- security_issue (border, crime, threat)
- demographic_need (labour shortages, ageing, replacement)
- humanitarian_responsibility (duty to protect, moral obligation)
- mixed (snippets clearly span 2+ with no dominance)

The JSON output MUST use the field name "dominant_framing" (with the -ing ending), NOT "dominant_frame". Output ONLY a JSON object with exactly these keys: dominant_framing, secondary_framing, confidence, rationale."""


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
    write_records_csv(results, OUT_DIR / "task5_direction_framing.csv")
    log.info(f"  wrote {OUT_DIR/'task5_direction_framing.csv'}")


# ---------------------------------------------------------------------------
# TASK 6 -- POLICY CONVERGENCE AND TENSION
# Pairs of parliaments with similar policy profiles -- characterize the
# relationship (alliance, rivalry, friction, silent convergence, etc.)
# Schema-name bug from the v2 run is fixed here: additionalProperties=false
# plus explicit field-name instructions in the prompt.
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
    "additionalProperties": False,
}

CONVERGENCE_SYSTEM = """Two parliaments have CONVERGED on similar migration-policy profiles. You are given:
- their cosine similarity over (cohort x policy_measure) shares,
- the top shared focus areas,
- mention counts in each direction (A->B and B->A),
- sample cross-mentions where available.

CRITICAL: The relation_type label and the tension_nature description MUST be consistent. If the tension_nature describes mutual complaining, the relation_type must be "mutual_complaint" -- do not label it "competitive_rivalry" or "unilateral_modeling" if the prose says otherwise. Each category is exclusive:

- mutual_acknowledged_alignment: both EXPLICITLY recognize each other as allies/models (in the snippets)
- unilateral_modeling: A treats B as a model AND B does not reciprocate at all
- silent_convergence: similar policies BUT minimal cross-mention -- convergence likely externally driven
- competitive_rivalry: they compete for the SAME migrants (students, high-skilled) or the SAME resources (EU funding) -- NOT just having opinions about each other
- ideological_friction_despite_similarity: similar policies, but each frames the other as an ideological adversary
- diffusion_chain: visible policy transfer -- one explicitly references adopting a mechanism FROM the other
- mutual_complaint: both complain (about each other, or about the same third party in similar terms)
- administrative_coordination: operational cooperation (Frontex, Dublin, joint patrols) framed in technocratic terms
- no_significant_interaction: too few cross-mentions to characterize

Output JSON MUST use field name "relation_type" (NOT "relationship_type"). Output ONLY a JSON object with the keys defined."""


def _policy_profiles(all_mentions: pl.DataFrame,
                     min_mentions: int = 100):
    """Return (parliaments, L2-normalized profile matrix, cell names)."""
    df = all_mentions.filter(
        pl.col("migrant_cohort").is_not_null() &
        pl.col("policy_measure").is_not_null() &
        (pl.col("migrant_cohort") != "general_migration") &
        (pl.col("policy_measure") != "general_policy")
    ).with_columns(
        pl.concat_str([pl.col("migrant_cohort"), pl.lit("|"),
                       pl.col("policy_measure")]).alias("cell")
    )
    pop = (df.group_by("country").agg(pl.len().alias("n"))
              .filter(pl.col("n") >= min_mentions))
    keep = pop["country"].to_list()
    if len(keep) < 2:
        raise SystemExit(
            f"Need at least 2 parliaments with >={min_mentions} mentions; "
            f"found {len(keep)}."
        )
    df = df.filter(pl.col("country").is_in(keep))
    counts = (df.group_by(["country", "cell"]).agg(pl.len().alias("n"))
                .pivot(values="n", index="country", on="cell")
                .fill_null(0))
    parliaments = counts["country"].to_list()
    cells = [c for c in counts.columns if c != "country"]
    import numpy as np
    M = counts.select(cells).to_numpy().astype(float)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return parliaments, M / norms, cells


def _select_convergence_pairs(parliaments, M, cells, cross_counts,
                              top_quiet, top_loud):
    """Two-stratum sampling: top similar + top similar+loud."""
    import numpy as np
    n = len(parliaments)
    sim = M @ M.T
    pairs = []
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
    return merged


async def task_policy_convergence(client: AsyncOpenAI, model: str,
                                  all_mentions: pl.DataFrame,
                                  top_n: int) -> None:
    import numpy as np
    log.info("[task 6] computing policy-profile similarities...")
    parliaments, M, cells = _policy_profiles(all_mentions)
    log.info(f"  {len(parliaments)} parliaments x {len(cells)} cells")

    cross_counts = {}
    rolled = (all_mentions
              .filter((pl.col("entity_scope") == "country") &
                      pl.col("target_iso3").is_not_null())
              .group_by(["country", "target_iso3"])
              .agg(pl.len().alias("n")))
    for row in rolled.iter_rows(named=True):
        cross_counts[(row["country"], row["target_iso3"])] = row["n"]

    pairs = _select_convergence_pairs(
        parliaments, M, cells, cross_counts,
        top_quiet=top_n, top_loud=max(5, top_n // 3),
    )
    log.info(f"  selected {len(pairs)} pairs")

    results = []
    for p in pairs:
        a, b, i, j = p["a"], p["b"], p["i"], p["j"]
        shared_scores = sorted(zip(cells, (M[i] * M[j]).tolist()),
                               key=lambda x: -x[1])[:3]
        shared_str = "; ".join(f"{c.replace('|', ' / ')}"
                               for c, _ in shared_scores if _ > 0)
        ab = all_mentions.filter((pl.col("country") == a) &
                                 (pl.col("target_iso3") == b))
        ba = all_mentions.filter((pl.col("country") == b) &
                                 (pl.col("target_iso3") == a))
        user = (
            f"PARLIAMENTS WITH CONVERGENT POLICY PROFILES\n"
            f"  A = {a}\n  B = {b}\n"
            f"  Cosine similarity: {p['similarity']:.3f}\n"
            f"  Top shared focus: {shared_str or '(none)'}\n"
            f"  Cross-mentions: {a}->{b}: {p['n_ab']}, "
            f"{b}->{a}: {p['n_ba']}\n\n"
        )
        if ab.height >= 3:
            user += (f"Sample {a}->{b}:\n" +
                     "\n".join(f"  - {s}" for s in sample_snippets(ab, n=4))
                     + "\n\n")
        else:
            user += f"({a} barely mentions {b}: {ab.height} mentions.)\n\n"
        if ba.height >= 3:
            user += (f"Sample {b}->{a}:\n" +
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
                 f"{out.get('relation_type','?')}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_records_csv(results, OUT_DIR / "task6_policy_convergence.csv")
    log.info(f"  wrote {OUT_DIR/'task6_policy_convergence.csv'}")


# ---------------------------------------------------------------------------
# TASK 7 -- ARGUMENTATIVE SCHEMES (Macagno's analytical layer)
#
# Your role-frames pilot already labels each mention with a relational role
# (TARGET_EXPORT, SOURCE_IMPORT, ADVERSARY_BLAME, etc.). That tells you WHAT
# RELATION the speaker assigns the target country -- but not WHAT REASONING
# MOVE they're making. Hungary's ADVERSARY_BLAME-via-cause-to-effect and
# Sweden's ADVERSARY_BLAME-via-authority-appeal are different rhetorical
# instruments that lead to very different downstream politics.
#
# This task runs over the same snippets as the role-frames pilot (snippets
# where a country is mentioned in a migration context) and classifies the
# ARGUMENTATIVE SCHEME the speaker is deploying. Cross-tabulating role x
# scheme is the analytical engine that turns descriptive work into a paper.
#
# Schemes are taken from Walton/Macagno's argumentation taxonomy, restricted
# to the six most common in political discourse.
# ---------------------------------------------------------------------------
SCHEME_SCHEMA = {
    "type": "object",
    "properties": {
        "scheme": {"type": "string", "enum": [
            "argument_from_authority",
            "argument_from_analogy",
            "argument_from_cause_to_effect",
            "argument_from_sign",
            "argument_ad_hominem",
            "argument_from_practical_reasoning",
            "no_clear_scheme",
        ]},
        "scheme_warrant": {"type": "string", "maxLength": 200},
        "evidence_span": {"type": "string", "maxLength": 200},
        "confidence":    {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["scheme", "confidence"],
    "additionalProperties": False,
}

SCHEME_SYSTEM = """You classify the ARGUMENTATIVE SCHEME a parliamentary speaker is deploying in a snippet about migration.

Argumentative schemes (Walton/Macagno taxonomy):

- argument_from_authority: speaker invokes an expert, institution, treaty, court ruling, or recognized norm to support a claim. Markers: "according to the UN", "the Court ruled", "international law says".

- argument_from_analogy: speaker compares the current case to a parallel case (another country, another era, another policy domain) to draw a conclusion. Markers: "as in Denmark", "like in 2015", "the same as".

- argument_from_cause_to_effect: speaker asserts that one fact will cause / has caused another. Markers: "this will lead to", "the result is", "because of X, Y happens".

- argument_from_sign: speaker treats a single observation as symptom of a broader condition (one rising number = whole system in crisis). Markers: "this is a sign of", "this shows that", "this proves".

- argument_ad_hominem: speaker attacks the source of an opposing position rather than the position itself. Targets the opponent's motives, character, identity, or alleged bad faith. Markers: "those who", "what they really want", attacks on competence or honesty.

- argument_from_practical_reasoning: speaker reasons from goal to means -- "we want X, so we must do Y". Markers: "in order to", "if we want", "to achieve".

- no_clear_scheme: the snippet is descriptive, narrative, or procedural without an inferential argument.

CLASSIFICATION RULES:
1. Classify on REASONING STRUCTURE, not on lexical matching. The markers are hints only.
2. If multiple schemes apply, pick the PRIMARY one driving the speaker's conclusion.
3. The scheme_warrant is the implicit premise the argument depends on (e.g. "the Court is a legitimate source" for authority arguments).
4. The evidence_span must be a substring of the CURRENT sentence (≤25 words).
5. If genuinely uncertain between two schemes, pick the one with the strongest direct evidence in the snippet and set confidence ≤ 0.6.

Output ONLY a JSON object with exactly: scheme, scheme_warrant, evidence_span, confidence."""


async def task_argumentative_schemes(client: AsyncOpenAI, model: str,
                                     all_mentions: pl.DataFrame,
                                     top_n: int) -> None:
    """Classify argumentative scheme on a stratified sample of mentions.

    Stratification: by source country AND by the existing rule-based
    policy_agency_type. This ensures coverage across both the speakers
    and the relational stances, so the cross-tab will have enough cells
    populated for chi-square testing.
    """
    log.info("[task 7] classifying argumentative schemes...")

    # We sample from rows that have substantive snippets and a target country.
    df = all_mentions.filter(
        pl.col("target_iso3").is_not_null() &
        (pl.col("entity_scope") == "country") &
        pl.col("sentence_content_current").is_not_null() &
        (pl.col("sentence_content_current").str.len_chars() > 40)
    )

    # Stratified sample: per (country, policy_agency_type) cell
    parts = []
    per_cell = max(2, top_n // 50)  # roughly: per_cell items per cell
    if "policy_agency_type" in df.columns:
        for keys, sub in df.group_by(["country", "policy_agency_type"]):
            if sub.height == 0:
                continue
            parts.append(sub.sample(n=min(per_cell, sub.height), seed=42))
    else:
        for c, sub in df.group_by("country"):
            parts.append(sub.sample(n=min(per_cell * 5, sub.height), seed=42))

    selection = pl.concat(parts).head(top_n)
    log.info(f"  selected {selection.height} snippets across "
             f"{selection['country'].n_unique()} parliaments")

    results = []
    for row in selection.iter_rows(named=True):
        prev = (row.get("sentence_content_previous") or "")[:300]
        curr = (row.get("sentence_content_current")  or "")[:500]
        nxt  = (row.get("sentence_content_next")     or "")[:300]
        user = (
            f"Speaker parliament: {row['country']} "
            f"(year {row.get('source_year','?')})\n"
            f"Mentioned country: {row.get('entity_content','?')} "
            f"({row.get('target_iso3','?')})\n"
            f"Existing role label (if any): "
            f"{row.get('policy_agency_type','unknown')}\n\n"
            f"Snippet:\n{prev}\n**{curr}**\n{nxt}"
        )
        out = await call(client, model, SCHEME_SYSTEM, user, SCHEME_SCHEMA,
                         max_tokens=300)
        # Snapshot the original metadata so we can cross-tab later
        results.append({
            "country":             row["country"],
            "year":                row.get("source_year"),
            "target_iso3":         row.get("target_iso3"),
            "entity_content":      row.get("entity_content"),
            "policy_agency_type":  row.get("policy_agency_type"),
            "migrant_cohort":      row.get("migrant_cohort"),
            "policy_measure":      row.get("policy_measure"),
            "sentiment_polarity":  row.get("sentiment_polarity"),
            **(out if isinstance(out, dict) else {}),
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_results = [
        {key: _csv_safe_value(value) for key, value in row.items()}
        for row in results
    ]
    out_df = pl.DataFrame(safe_results)
    out_df.write_csv(OUT_DIR / "task7_argumentative_schemes.csv")
    log.info(f"  wrote {OUT_DIR/'task7_argumentative_schemes.csv'}")

    # The cross-tab is THE FINDING. Compute and save it.
    if "scheme" in out_df.columns and "policy_agency_type" in out_df.columns:
        ct = (out_df.filter(pl.col("scheme").is_not_null() &
                            pl.col("policy_agency_type").is_not_null())
                    .group_by(["policy_agency_type", "scheme"])
                    .agg(pl.len().alias("n"))
                    .pivot(values="n", index="policy_agency_type",
                           on="scheme")
                    .fill_null(0))
        ct.write_csv(OUT_DIR / "task7_role_x_scheme_crosstab.csv")
        log.info(f"  wrote {OUT_DIR/'task7_role_x_scheme_crosstab.csv'}")

        # Also: country x scheme (which parliaments favor which schemes?)
        ct2 = (out_df.filter(pl.col("scheme").is_not_null())
                     .group_by(["country", "scheme"])
                     .agg(pl.len().alias("n"))
                     .pivot(values="n", index="country", on="scheme")
                     .fill_null(0))
        ct2.write_csv(OUT_DIR / "task7_country_x_scheme_crosstab.csv")
        log.info(f"  wrote {OUT_DIR/'task7_country_x_scheme_crosstab.csv'}")


# ---------------------------------------------------------------------------
# TASK 8 -- Parliament-level profiles answering the dashboard questions
# ---------------------------------------------------------------------------
PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "one_row_profile": {"type": "string", "maxLength": 700},
        "referencing_style": {"type": "string", "maxLength": 500},
        "target_concreteness_profile": {"type": "string", "maxLength": 500},
        "migration_direction_profile": {"type": "string", "maxLength": 500},
        "event_evidence_profile": {"type": "string", "maxLength": 500},
        "cohort_profile": {"type": "string", "maxLength": 500},
        "temporal_profile": {"type": "string", "maxLength": 500},
        "policy_instrument_profile": {"type": "string", "maxLength": 500},
        "narrative_sentiment_profile": {"type": "string", "maxLength": 500},
        "prompt_selection_relevance": {"type": "string", "maxLength": 500},
        "dashboard_takeaway": {"type": "string", "maxLength": 300},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "one_row_profile", "referencing_style",
        "target_concreteness_profile", "migration_direction_profile",
        "event_evidence_profile", "cohort_profile", "temporal_profile",
        "policy_instrument_profile", "narrative_sentiment_profile",
        "prompt_selection_relevance", "dashboard_takeaway", "confidence",
    ],
    "additionalProperties": False,
}

PROFILE_SYSTEM = """You write parliament-level research profiles for a dashboard and methods section.

You receive deterministic Python evidence for one parliament's migration-relevant country/entity mention windows. Answer the research questions directly and cautiously.

Important scope rules:
- The source universe is all processed country corpora, but this profile is based on migration-relevant mention windows selected by the taxonomy/filtering pipeline.
- Where ParlaMint taxonomy is available, IMIG is part of that upstream selection.
- Do not invent causal explanations. Treat yearly peaks as agenda-shift or shock-event candidates unless evidence explicitly supports more.
- Distinguish raw volume from normalized within-parliament shares.
- Mention if a pattern is descriptive and should be interpreted with context snippets.

Output ONLY a JSON object matching the schema."""


def profile_metrics_for_country(df: pl.DataFrame, country: str) -> dict[str, Any]:
    sub = df.filter(pl.col("country") == country)
    if sub.is_empty():
        return {}
    high_concrete = sub
    if "concreteness_score" in sub.columns:
        high_concrete = sub.filter(
            pl.col("concreteness_score").cast(pl.Float64, strict=False) >= 0.65
        )
    return {
        "parliament": country,
        "n_migration_country_mentions": sub.height,
        "n_speeches": safe_unique(sub, "speech_id"),
        "n_speakers": safe_unique(sub, "speaker_id"),
        "n_targets": safe_unique(sub, "target_iso3"),
        "year_range": {
            "min": sub.select(pl.col("source_year").min()).item()
                   if "source_year" in sub.columns else None,
            "max": sub.select(pl.col("source_year").max()).item()
                   if "source_year" in sub.columns else None,
        },
        "mean_concreteness": safe_mean(sub, "concreteness_score"),
        "mean_sentiment": safe_mean(sub, "sentence_sentiment_value"),
        "top_targets": top_distribution(sub, "target_iso3", 8),
        "top_entities": top_distribution(sub, "entity_content", 6),
        "policy_agency_distribution": top_distribution(sub, "policy_agency_type", 6),
        "migration_direction_distribution": top_distribution(sub, "migration_direction", 5),
        "migrant_cohort_distribution": top_distribution(sub, "migrant_cohort", 6),
        "policy_measure_distribution": top_distribution(sub, "policy_measure", 8),
        "narrative_polarity_distribution": top_distribution(sub, "narrative_polarity", 5),
        "narrative_frame_distribution": top_distribution(sub, "narrative_frame", 8),
        "sentiment_polarity_distribution": top_distribution(sub, "sentiment_polarity", 4),
        "concreteness_band_distribution": top_distribution(sub, "concreteness_band", 4),
        "yearly_peak_distribution": top_distribution(sub, "source_year", 5),
        "high_concreteness_mentions": high_concrete.height,
        "high_concreteness_examples": sample_snippets(high_concrete, n=3),
    }


async def task_parliament_profiles(client: AsyncOpenAI, model: str,
                                   all_mentions: pl.DataFrame) -> None:
    log.info("[task 8] building parliament-level question profiles...")
    countries = sorted(c for c in all_mentions["country"].unique().to_list() if c)
    questions = [
        "What is the one-row profile of this parliament?",
        "Which countries/entities dominate its attention?",
        "Does it use others as models, pressure targets, partners, competitors, or neutral examples?",
        "Does it discuss targets concretely or abstractly?",
        "Does it focus on inbound domestic migration or migration between third countries?",
        "Does it give event-like, named, concrete migration evidence?",
        "Which migrant groups dominate its country references?",
        "When does it talk more intensely about migration-related foreign countries?",
        "Which policy instruments dominate its migration country talk?",
        "How does it differ in solidarity, risk, benefit, and administrative imagination?",
        "How does its sentence-level emotional tone look, cautiously interpreted?",
        "Why should this parliament be selected for later LLM close reading?",
    ]
    results: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []

    for country in countries:
        metrics = profile_metrics_for_country(all_mentions, country)
        if not metrics:
            continue
        metrics_rows.append({
            "parliament": country,
            "n_migration_country_mentions": metrics["n_migration_country_mentions"],
            "n_speeches": metrics["n_speeches"],
            "n_speakers": metrics["n_speakers"],
            "n_targets": metrics["n_targets"],
            "mean_concreteness": metrics["mean_concreteness"],
            "mean_sentiment": metrics["mean_sentiment"],
            "high_concreteness_mentions": metrics["high_concreteness_mentions"],
            "top_targets": json.dumps(metrics["top_targets"], ensure_ascii=False),
            "policy_agency_distribution": json.dumps(metrics["policy_agency_distribution"], ensure_ascii=False),
            "migration_direction_distribution": json.dumps(metrics["migration_direction_distribution"], ensure_ascii=False),
            "migrant_cohort_distribution": json.dumps(metrics["migrant_cohort_distribution"], ensure_ascii=False),
            "policy_measure_distribution": json.dumps(metrics["policy_measure_distribution"], ensure_ascii=False),
            "narrative_frame_distribution": json.dumps(metrics["narrative_frame_distribution"], ensure_ascii=False),
            "yearly_peak_distribution": json.dumps(metrics["yearly_peak_distribution"], ensure_ascii=False),
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(metrics_rows).write_csv(OUT_DIR / "task8_parliament_profile_metrics.csv")
    log.info(f"  wrote deterministic metrics to {OUT_DIR/'task8_parliament_profile_metrics.csv'}")

    for country in countries:
        metrics = profile_metrics_for_country(all_mentions, country)
        if not metrics:
            continue
        user = (
            f"Parliament: {country}\n\n"
            "Research questions this profile must answer:\n"
            + "\n".join(f"- {q}" for q in questions)
            + "\n\nDeterministic Python evidence for this parliament:\n"
            + json.dumps(metrics, ensure_ascii=False, indent=2)
        )
        try:
            out = await call(client, model, PROFILE_SYSTEM, user,
                             PROFILE_SCHEMA, max_tokens=1000)
        except APIConnectionError:
            log.error(
                "Cannot connect to the vLLM server at the configured --server. "
                "Run this task inside the SLURM script that starts vLLM, or start "
                "vLLM before running the Python client."
            )
            raise
        results.append({
            "parliament": country,
            "n_migration_country_mentions": metrics["n_migration_country_mentions"],
            "n_speeches": metrics["n_speeches"],
            "n_speakers": metrics["n_speakers"],
            "n_targets": metrics["n_targets"],
            "mean_concreteness": metrics["mean_concreteness"],
            "mean_sentiment": metrics["mean_sentiment"],
            **(out if isinstance(out, dict) else {}),
        })
        log.info(f"  profiled {country}")

    write_records_csv(results, OUT_DIR / "task8_parliament_question_profiles.csv")
    log.info(f"  wrote {OUT_DIR/'task8_parliament_question_profiles.csv'}")


# ---------------------------------------------------------------------------
# TASK 9/10 -- deterministic year-by-year attention and external target salience
# ---------------------------------------------------------------------------
def task_yearly_attention(all_mentions: pl.DataFrame) -> None:
    """Write year-by-year country attention tables, including Ukraine."""
    log.info("[task 9] computing year-by-year target attention...")
    parliament_isos = set(all_mentions["country"].unique().to_list())
    df = all_mentions.filter(
        pl.col("target_iso3").is_not_null() &
        (pl.col("entity_scope") == "country") &
        pl.col("source_year").is_not_null()
    ).with_columns([
        pl.col("target_iso3").is_in(sorted(parliament_isos)).alias("target_has_project_parliament"),
        (pl.col("country") == pl.col("target_iso3")).alias("self_reference"),
    ]).with_columns(
        pl.when(pl.col("self_reference")).then(pl.lit("self_reference"))
          .when(pl.col("target_has_project_parliament")).then(pl.lit("reciprocal_asymmetry_candidate"))
          .otherwise(pl.lit("external_target_salience"))
          .alias("analysis_scope")
    )
    yearly = (
        df.group_by(["country", "target_iso3", "source_year", "analysis_scope"])
          .agg([
              pl.len().alias("n_mentions"),
              pl.col("speech_id").n_unique().alias("n_speeches"),
              pl.col("speaker_id").n_unique().alias("n_speakers"),
              pl.col("concreteness_score").cast(pl.Float64, strict=False).mean()
                .alias("mean_concreteness"),
              pl.col("sentence_sentiment_value").cast(pl.Float64, strict=False).mean()
                .alias("mean_sentiment"),
          ])
          .sort(["target_iso3", "country", "source_year"])
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    yearly.write_csv(OUT_DIR / "task9_yearly_target_attention.csv")

    ukraine = yearly.filter(pl.col("target_iso3") == "UKR")
    ukraine.write_csv(OUT_DIR / "task9_ukraine_yearly_attention.csv")
    if not ukraine.is_empty():
        total = (
            ukraine.group_by("source_year")
                   .agg([
                       pl.sum("n_mentions").alias("total_mentions"),
                       pl.col("country").n_unique().alias("n_source_parliaments"),
                   ])
                   .sort("source_year")
        )
        total.write_csv(OUT_DIR / "task9_ukraine_total_by_year.csv")
    log.info(f"  wrote {OUT_DIR/'task9_yearly_target_attention.csv'}")
    log.info(f"  wrote {OUT_DIR/'task9_ukraine_yearly_attention.csv'}")


def task_target_salience(all_mentions: pl.DataFrame) -> None:
    """Separate external target salience from reciprocal asymmetry candidates."""
    log.info("[task 10] separating external target salience from reciprocal asymmetry...")
    parliament_isos = set(all_mentions["country"].unique().to_list())
    df = all_mentions.filter(
        pl.col("target_iso3").is_not_null() &
        (pl.col("entity_scope") == "country") &
        (pl.col("country") != pl.col("target_iso3"))
    ).with_columns(
        pl.col("target_iso3").is_in(sorted(parliament_isos)).alias("target_has_project_parliament")
    )
    scoped = (
        df.with_columns(
            pl.when(pl.col("target_has_project_parliament"))
              .then(pl.lit("reciprocal_asymmetry_candidate"))
              .otherwise(pl.lit("external_target_salience"))
              .alias("analysis_scope")
        )
        .group_by(["country", "target_iso3", "analysis_scope"])
        .agg([
            pl.len().alias("n_mentions"),
            pl.col("speech_id").n_unique().alias("n_speeches"),
            pl.col("speaker_id").n_unique().alias("n_speakers"),
            pl.col("source_year").min().alias("first_year"),
            pl.col("source_year").max().alias("last_year"),
            pl.col("concreteness_score").cast(pl.Float64, strict=False).mean()
              .alias("mean_concreteness"),
            pl.col("sentence_sentiment_value").cast(pl.Float64, strict=False).mean()
              .alias("mean_sentiment"),
        ])
        .sort("n_mentions", descending=True)
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scoped.write_csv(OUT_DIR / "task10_target_scope_salience.csv")
    scoped.filter(pl.col("analysis_scope") == "external_target_salience").write_csv(
        OUT_DIR / "task10_external_target_salience.csv"
    )
    scoped.filter(pl.col("analysis_scope") == "reciprocal_asymmetry_candidate").write_csv(
        OUT_DIR / "task10_reciprocal_asymmetry_candidates.csv"
    )
    log.info(f"  wrote {OUT_DIR/'task10_target_scope_salience.csv'}")


# ---------------------------------------------------------------------------
# POSTPROCESS -- coalesce dual-column bug remnants, compute key cross-tabs.
# Run with: --task postprocess
# Reads outputs already on disk, normalizes them, writes clean versions.
# ---------------------------------------------------------------------------
def postprocess_run() -> None:
    """Repair known schema-name splits from earlier runs."""
    log.info("[postprocess] cleaning up known output bugs...")
    out_dir = OUT_DIR

    # Task 5: coalesce dominant_framing / dominant_frame
    p5 = out_dir / "task5_direction_framing.csv"
    if p5.exists():
        df = pl.read_csv(p5)
        cols = df.columns
        if "dominant_framing" in cols and "dominant_frame" in cols:
            df = df.with_columns(
                pl.coalesce(["dominant_framing", "dominant_frame"])
                  .alias("dominant_framing_clean")
            ).drop("dominant_framing", "dominant_frame").rename(
                {"dominant_framing_clean": "dominant_framing"}
            )
            df.write_csv(p5)
            log.info(f"  coalesced dual columns in {p5.name}")

    # Task 6: coalesce relation_type / relationship_type
    p6 = out_dir / "task6_policy_convergence.csv"
    if p6.exists():
        df = pl.read_csv(p6)
        cols = df.columns
        if "relation_type" in cols and "relationship_type" in cols:
            df = df.with_columns(
                pl.coalesce(["relation_type", "relationship_type"])
                  .alias("relation_type_clean")
            ).drop("relation_type", "relationship_type").rename(
                {"relation_type_clean": "relation_type"}
            )
            df.write_csv(p6)
            log.info(f"  coalesced dual columns in {p6.name}")

    log.info("[postprocess] done.")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
async def run(args: argparse.Namespace) -> None:
    global OUT_DIR
    if args.out_dir:
        OUT_DIR = Path(args.out_dir).expanduser().resolve()
    elif args.run_name:
        OUT_DIR = (
            PROCESSED
            / "ALL_AVAILABLE_COUNTRIES_comparisons"
            / args.run_name
        )
    log.info(f"outputs will be written to {OUT_DIR}")

    if args.task == "postprocess":
        postprocess_run()
        return

    log.info(f"loading all per-country mentions from {PROCESSED}...")
    all_mentions = load_all_mentions()
    log.info(f"  loaded {all_mentions.height:,} mentions across "
             f"{all_mentions['country'].n_unique()} parliaments")

    client = AsyncOpenAI(base_url=args.server, api_key="EMPTY", timeout=120.0)
    if args.task == "all":
        tasks = ["target_salience", "yearly_attention",
                 "asymmetry", "events", "model_pressure", "cross_target",
                 "direction_framing", "policy_convergence",
                 "argumentative_schemes", "parliament_profiles"]
    else:
        tasks = args.task.split(",")

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
            elif t == "argumentative_schemes":
                # 300 snippets is a good first pass: ~12 per parliament,
                # enough to populate a 6 x 6 cross-tab per country
                await task_argumentative_schemes(
                    client, args.model, all_mentions, 300)
            elif t == "parliament_profiles":
                await task_parliament_profiles(client, args.model, all_mentions)
            elif t == "yearly_attention":
                task_yearly_attention(all_mentions)
            elif t == "target_salience":
                task_target_salience(all_mentions)
            else:
                log.warning(f"unknown task: {t}")
        except Exception as e:
            log.error(f"task {t} failed: {e!r}")
            continue

    # Always coalesce dual columns at the end of a run
    postprocess_run()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all",
        help="all | asymmetry | events | model_pressure | cross_target | "
             "direction_framing | policy_convergence | argumentative_schemes | "
             "parliament_profiles | yearly_attention | target_salience | "
             "postprocess | comma-separated")
    ap.add_argument("--server", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model",  default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--top-n",  type=int, default=50)
    ap.add_argument("--run-name", default=DEFAULT_RUN_NAME,
        help="Output subfolder under data/processed/ALL_AVAILABLE_COUNTRIES_comparisons/. "
             "Use a new name to avoid overwriting previous results.")
    ap.add_argument("--out-dir", default=None,
        help="Absolute or relative output directory. Overrides --run-name.")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
