# LLM Prompt Phase 1 Rerun

This folder keeps the new prompt run separate from earlier outputs.

## Files

- `llm_prompts_phase1.py` -- copied from `llm_prompts_up.py`, with a new default output folder.

## Default output folder

The script writes to:

```text
data/processed/ALL_AVAILABLE_COUNTRIES_comparisons/llm_insights_phase1_2026_05_26
```

This avoids overwriting the older `llm_insights` folder.

## Run on LUMI

Copy `llm_prompts_phase1.py` into your project `scripts/` folder, then run from the project root after vLLM is up:

```bash
python scripts/llm_prompts_phase1.py --task postprocess
python scripts/llm_prompts_phase1.py --task argumentative_schemes
python scripts/llm_prompts_phase1.py --task events,direction_framing,policy_convergence
```

Or run everything:

```bash
python scripts/llm_prompts_phase1.py --task all
```

To create another separate run folder:

```bash
python scripts/llm_prompts_phase1.py --task all --run-name llm_insights_phase1_rerun_2
```
