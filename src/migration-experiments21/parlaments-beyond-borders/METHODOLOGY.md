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

## Study 3: Policy Agency in Cross-Country References

**Question.** When France mentions another country/entity in migration debates,
is that target treated as a model to learn from, a target for pressure, a partner
for cooperation, a competitor, or simply a location where events are reported?

**Why this matters.** The project is not only counting country names. It asks
what work those country names do inside parliamentary argument. The same target
can appear as a model in one speech, a problem in another, and a cooperation
partner in a third.

**Operationalization.**

- Add `policy_agency_type` to each mention:
  `learning_emulation_from`, `coercion_intervention_to`, `competition`,
  `exchange_cooperation`, or `neutral_reporting`.
- Store `policy_agency_marker` so each classification can be audited.
- Store `policy_agency_llm_prompt`, which can be sent to Qwen, Llama, or another
  zero-shot model later.
- Build a directed dyadic edge table:
  source = `FRA`; target = mentioned country/entity; edge attributes include
  agency type, migrant cohort, policy measure, reference type, and year.

**Current implementation.**

The current version is keyword-rule plus LLM-ready prompts. No external LLM is
called in the notebook, so the output is reproducible offline. This is deliberate
for the hackathon: it gives us a stable baseline and a direct path to replacing
the rules with a stronger classifier.

## Study 4: Narrative Framing and Migration Imagination

**Question.** How is migration imagined when other countries are mentioned: as
risk, obligation, benefit, administrative problem, legal category, crisis, or
policy model?

**Operationalization.**

- Apply a 61-frame narrative taxonomy to each context window.
- Collapse frames into a ternary plotting schema:
  `positive_sympathy`, `positive_benefit`, `negative_risk`,
  `neutral_administrative`.
- Detect argumentative schemes:
  `argument_from_consequences`, `practical_reasoning`,
  `conceptual_definition`, or `other`.
- Extract short definition candidates when migration/refugee terms are followed
  by copular definition patterns such as "migration is..." or "refugees are...".

**Current implementation.**

The taxonomy is implemented as transparent phrase markers in `src/framing.py`.
This is not yet a final narrative classifier. It is a precise audit layer that
lets the team inspect which frames drive the result and which contexts should be
manually validated.

## Study 5: High-Concreteness Evidence and Event Visibility

**Question.** Which other countries become visible as concrete evidence in
French migration debate, rather than just abstract symbols?

**Operationalization.**

- Keep contexts with `concreteness_score >= 3.3`, the current pilot threshold for
  `concrete_leaning`.
- Extract dates, target entities, ISO3 codes, country mentions inside the context,
  and proper-noun anchors.
- Save a high-concreteness event table and a country visibility summary.
- Visualize this as:
  evidence visibility map;
  fact-density timeline;
  country x agency heatmap.

**Current implementation.**

Country extraction uses `country-named-entity-recognition`, which avoids common
partial-match errors such as confusing `Niger` and `Nigeria`. Proper-noun anchors
are lightweight event labels for tooltips and manual reading; they should be
treated as evidence pointers, not final event extraction.

## Study 6: Internal vs External Migration Processes

**Question.** Is France discussing migration into France, migration out of
France, or migration between other countries?

**Operationalization.**

- Add `flow_source_candidate` and `flow_destination_candidate` from surface
  source/destination patterns.
- Add `migration_direction`:
  `inbound_internal`, `outbound_from_domestic`, `external_transnational`, or
  `ambiguous`.

**Current implementation.**

This is a surface semantic-role-labeling substitute. It is good enough to split
obvious internal and external cases for exploratory analysis, but it should be
validated against hand-coded examples before being used as a final claim.

## What To Read First

Open these outputs first:

- `data/processed/FRA_2017_2022_result_notes.md`
- `data/processed/FRA_2017_2022_country_mention_profile.csv`
- `data/processed/FRA_2017_2022_country_year_concreteness_summary.csv`
- `data/processed/FRA_2017_2022_concreteness_feature_patterns.csv`
- `data/processed/FRA_2017_2022_concreteness_quote_examples.csv`
- `data/processed/FRA_2017_2022_country_context_examples.csv`
- `data/processed/FRA_2017_2022_cohort_policy_context_examples.csv`
- `data/processed/FRA_2017_2022_policy_agency_edges.csv`
- `data/processed/FRA_2017_2022_policy_hubs_pagerank.csv`
- `data/processed/FRA_2017_2022_high_concreteness_events.csv`
- `data/processed/FRA_2017_2022_high_concreteness_events_gt4.csv`
- `data/processed/FRA_2017_2022_all_concrete_event_mentions.csv`
- `data/processed/FRA_2017_2022_europe_concrete_event_mentions.csv`
- `data/processed/FRA_2017_2022_europe_concrete_conversation_summary.csv`
- `data/processed/FRA_2017_2022_europe_concrete_conversation_year_summary.csv`
- `data/processed/FRA_2017_2022_europe_geolocated_concrete_event_points.csv`
- `data/processed/FRA_2017_2022_europe_concrete_event_mentions_unmatched_places.csv`
- `data/processed/FRA_2017_2022_visible_country_summary.csv`
- `data/processed/FRA_2017_2022_visible_country_summary_gt4.csv`
- `data/processed/FRA_2017_2022_direction_agenda_by_entity.csv`
- `data/processed/FRA_2017_2022_direction_agenda_by_party.csv`
- `data/processed/figures_interactive_advanced/policy_agency_network.html`
- `data/processed/figures_interactive_advanced/policy_agency_country_heatmap.html`
- `data/processed/figures_interactive_advanced/policy_hubs_pagerank.html`
- `data/processed/figures_interactive_advanced/narrative_ternary.html`
- `data/processed/figures_interactive_advanced/narrative_ternary_by_party.html`
- `data/processed/figures_interactive_advanced/narrative_mirror_bars.html`
- `data/processed/figures_interactive_advanced/narrative_mirror_by_party.html`
- `data/processed/figures_interactive_advanced/narrative_mirror_by_speaker_role.html`
- `data/processed/figures_interactive_advanced/evidence_visibility_map.html`
- `data/processed/figures_interactive_advanced/evidence_visibility_map_gt4.html`
- `data/processed/figures_interactive_advanced/fact_density_timeline.html`
- `data/processed/figures_interactive_advanced/fact_density_timeline_gt4.html`
- `data/processed/figures_interactive_advanced/all_concrete_event_mentions_table.html`
- `data/processed/figures_interactive_advanced/europe_concrete_event_mentions_table.html`
- `data/processed/figures_interactive_advanced/concrete_conversations_dashboard.html`
- `data/processed/figures_interactive_advanced/europe_concrete_conversation_map.html`
- `data/processed/figures_interactive_advanced/europe_concrete_map_by_year.html`
- `data/processed/figures_interactive_advanced/europe_concrete_event_timeline.html`
- `data/processed/figures_interactive_advanced/europe_country_year_event_heatmap.html`
- `data/processed/figures_interactive_advanced/europe_country_frame_heatmap.html`
- `data/processed/figures_interactive_advanced/europe_concrete_sankey.html`
- `data/processed/figures_interactive_advanced/europe_concrete_treemap.html`
- `data/processed/figures_interactive_advanced/europe_geolocated_concrete_event_point_map.html`
- `data/processed/figures_interactive_advanced/europe_geolocated_concrete_event_timeline.html`
- `data/processed/figures_interactive_advanced/concrete_conversations_explorer.html`
- `data/processed/figures_interactive_advanced/direction_agenda_split_bars.html`
- `data/processed/figures_interactive_advanced/direction_agenda_by_party.html`
- `data/processed/figures_altair_extended/country_concreteness_bubble.html`
- `data/processed/figures_altair_extended/country_concreteness_year_lines.html`
- `data/processed/figures_altair_extended/concreteness_feature_pattern_heatmap.html`
- `data/processed/figures_altair_extended/concreteness_quote_panels.html`
- `data/processed/figures_altair_extended/country_year_concreteness_heatmap.html`
- `data/processed/figures_altair_extended/country_cohort_heatmap.html`
- `data/processed/figures_altair_extended/country_policy_heatmap.html`

## Current Run Snapshot

After executing `notebooks/02_fra_2017_2022_extended.ipynb`, the current local
run retained 4,256 migration country/entity mentions. The largest policy-agency
category is `neutral_reporting`, followed by `learning_emulation_from`,
`coercion_intervention_to`, and `exchange_cooperation`. The directional split is
dominated by `external_transnational`, which is exactly why the project needs
cross-country reference analysis rather than only domestic migration analysis.

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
