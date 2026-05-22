# Methodology Notes: Cross-Country Migration References

This project studies migration discourse through a cross-country reference lens:
when France talks about migration, which other countries or entities are invoked,
and what kind of evidentiary work do those references do?

## Better Research Questions

### Study 1: Concrete vs Abstract Country References

**Question.** When France mentions other countries/entities in migration debates,
are those references grounded in concrete events, places, institutions, and policy
instruments, or are they framed through abstract/speculative language?

**Hypothesis.** Mentions of non-WEOG countries may be more abstract or speculative
than mentions of WEOG countries, because distant places can be used as generalized
symbols of crisis, risk, or pressure rather than as concrete policy examples.

**Operationalization.**

- Filter migration discourse with the CAP immigration topic and migration keywords.
- Keep country/entity mentions from the ParlaMint NER layer.
- Normalize country names and classify targets as WEOG, non-WEOG, European Union,
  French overseas territory, or unknown.
- Score the previous/current/next sentence context window for concreteness.
- Export context examples for the most abstract and most concrete examples by country.

**Current implementation.**

The code supports a Brysbaert-style concreteness lexicon. If no lexicon is present,
it uses a transparent fallback heuristic:

- named entities receive high concreteness because they refer to specific places
  or institutions;
- physical/institutional terms such as `border`, `camp`, `passport`, `visa`,
  `police`, `school`, and `territory` increase concreteness;
- abstract terms such as `solidarity`, `principle`, `sovereignty`, `identity`,
  `responsibility`, and `system` reduce concreteness.

The fallback bands are pilot-relative:

- `abstract_leaning`: score below 3.2
- `mixed`: score from 3.2 to below 3.3
- `concrete_leaning`: score 3.3 or higher

This is not yet a final psycholinguistic score. It is a transparent way to locate
examples for manual validation during the hackathon.

## Study 2: Institutional Cohorts and Policy Diffusion

**Question.** When France divides migrants into institutional cohorts, which
countries/entities are invoked as models, warnings, comparison cases, or policy
references?

**Hypothesis.** Different migrant cohorts should activate different cross-country
reference patterns. For example, asylum seekers may appear with legal/procedural
references, students with mobility/economic contribution, and refugees with
humanitarian or international-law references.

**Operationalization.**

- Classify each context window into migrant cohorts:
  `refugees`, `asylum_seekers`, `students`, `economic_migrants`,
  `high_skilled_workers`, or `general_migration`.
- Classify each context into policy measures:
  `international_law`, `national_security`, `border_control`,
  `allocation_of_resources`, `integration`, `returns_deportation`,
  `asylum_procedure`, `visas_mobility`, or `general_policy`.
- Build a directed network:
  source = `FRA`; target = mentioned country/entity.
- Store edge attributes: year, migrant cohort, policy measure, reference type,
  and edge weight.

**Current implementation.**

The cohort and policy labels are keyword-rule labels, with matched marker columns
stored in the output. This makes the classifications auditable. A later version
can replace these rules with zero-shot classifiers while preserving the same
output schema.

## What To Read First

Open these outputs first:

- `data/processed/FRA_2017_2022_result_notes.md`
- `data/processed/FRA_2017_2022_country_mention_profile.csv`
- `data/processed/FRA_2017_2022_country_context_examples.csv`
- `data/processed/FRA_2017_2022_cohort_policy_context_examples.csv`
- `data/processed/figures_altair_extended/country_concreteness_bubble.html`
- `data/processed/figures_altair_extended/country_year_concreteness_heatmap.html`
- `data/processed/figures_altair_extended/country_cohort_heatmap.html`
- `data/processed/figures_altair_extended/country_policy_heatmap.html`

## Literature Anchors

- Construal Level Theory: psychological distance is associated with more abstract
  mental representation.
- Brysbaert concreteness norms: concreteness can be operationalized as a
  word-level 1-5 psycholinguistic rating.
- Linguistic Category Model: language can be ordered from concrete action
  description toward more abstract traits and states.
- Migration discourse work on metaphor/framing supports the idea that migration
  can be discussed indirectly through crisis, flow, pressure, burden, security,
  or humanitarian frames.
