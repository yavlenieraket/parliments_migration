# Parliaments Beyond Borders - Migration Discourse Pilot

DHH26 hackathon project analyzing how national parliaments reference
foreign countries in migration debates. The pilot uses the translated
`ParlaMint-en.ana` corpus, so entity names and typology markers are English.

## Scope

The first pilot notebook covers France 2018. The extended notebook now covers
all available France files from 2017 through 2022:

- `FRA_2017_facts.parquet`
- `FRA_2018_facts.parquet`
- `FRA_2019_facts.parquet`
- `FRA_2020_facts.parquet`
- `FRA_2021_facts.parquet`
- `FRA_2022_facts.parquet`

## Typology

Each foreign-country mention is classified along two axes:

- **Reference type**: policy / situation / mixed / neutral
- **Sentiment**: ParlaMint's full 6-level categorical sentiment scale

Sentiment is based directly on ParlaMint's categorical
`sentence_sentiment_ana` labels, preserving intensity:

- `senti:negneg` - strongly negative
- `senti:mixneg` - mixed leaning negative
- `senti:neuneg` - neutral with slight negative tilt
- `senti:neupos` - neutral with slight positive tilt
- `senti:mixpos` - mixed leaning positive
- `senti:pospos` - strongly positive

The numeric `sentence_sentiment_value` is kept in the output for inspection,
but it is not centered at zero in this file, so it should not be interpreted
with a simple negative/positive zero threshold.

French overseas territories such as Mayotte and French Guiana are kept in the
analysis because they are important in migration debates, but they are marked
as `geo_class = french_overseas`. They are not foreign states.

The European Union is kept as `region_group = european_union`. It is captured
from explicit `ORG` mentions such as `European Union` and `EU`, because the EU
is not tagged as a country/location in the source data.

## Data

ParlaMint extracted parquet files. Not committed to the repo.
Place files at:

- `data/parlamint_extracted/Table1_Fact/FRA/FRA_2018_facts.parquet`
- `data/parlamint_extracted/Table1_Fact/GRC/GRC_2015_facts.parquet`
- `data/parlamint_extracted/Table1_Fact/TUR/TUR_2015_facts.parquet`
- `data/parlamint_extracted/Table1_Fact/ITA/ITA_2013_facts.parquet`
- `data/parlamint_extracted/Table1_Fact/GBR/GBR_2015_facts.parquet`
- `data/parlamint_extracted/Table2_People/Master_People.parquet`
- `data/parlamint_extracted/Table3_Orgs/Master_Orgs.parquet`
- `data/parlamint_extracted/Table4_Affiliations/Master_Affiliations.parquet`

## Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_fra_2018_pilot.ipynb
```

For the extended 2017-2022 analysis, run:

```bash
jupyter notebook notebooks/02_fra_2017_2022_extended.ipynb
```

For the same extended notebook workflow across all available studied country
folders, run:

```bash
jupyter notebook notebooks/03_grc_tur_ita_gbr_extended.ipynb
```

To apply the same extended pipeline to every studied country folder, run:

```bash
python scripts/run_extended_analysis.py
```

This writes each country to its own processed directory, for example
`data/processed/AUT_2015_2022/`, `data/processed/FRA_2017_2022/`,
`data/processed/NLD_2015_2022/`, and `data/processed/UKR_2015_2022/`.

To regenerate every per-country visualization bundle, the combined dyadic
visualizations, and a single browser index of all figure files, run:

```bash
python scripts/generate_all_country_visualizations.py
```

This also creates direct cross-country comparison charts under
`data/processed/ALL_AVAILABLE_COUNTRIES_comparisons/`, including comparison views
for yearly volume, total mentions, shared target countries, mean concreteness,
target-level concreteness, policy agency, narrative polarity, migration
direction, migrant cohorts, policy measures, sentiment, and high-concreteness
fact counts. The HTML index explains what each illustration asks and how to
read it. The index begins with an analytical report on asymmetries, trends,
special cases, and LLM-ready research leads. It also includes richer Plotly
interactive comparison views for timelines, target heatmaps, entity-scope
composition, concreteness/sentiment scatterplots, and reciprocal attention
asymmetry.

To verify that target entities are clean and every visualization bundle exists,
run:

```bash
python scripts/validate_entities_and_visualizations.py
```

The validation checks that the target entity layer contains only countries, the
European Union, approved analytical regions/routes, and approved territory
regions. Cities, events, people, institutions, and vague directional words are
not allowed as target entities; they can appear only inside context snippets or
fact-map place labels.

## Extended 2017-2022 Methods

For the conceptual framing, research questions, and validation logic, see
`METHODOLOGY.md`.

The extended notebook uses all available France files and adds several analysis
layers: concreteness, migrant cohorts, policy measures, policy agency,
narrative framing, high-concreteness event extraction, and internal/external
migration direction.

### Country Reference Base

Every row is a migration-related mention of another country, the EU, or a
French overseas territory. French overseas territories are retained because
they are substantively important, but they are labelled separately from foreign
states with `geo_class = french_overseas`.

### Concreteness / Abstractness

The notebook scores each migration context window on a 1-5 concreteness scale.
Named entities are treated as maximally concrete because they point to specific
places or institutions. The code can use a Brysbaert-style concreteness lexicon
if supplied; otherwise it uses a transparent fallback heuristic and records this
in `concreteness_method`.

The regional comparison uses `weog_group`:

- `weog`
- `non_weog`
- `european_union`
- `french_overseas`
- `unknown`

The WEOG/non-WEOG density chart excludes `unknown`, `european_union`, and
`french_overseas` so the regional hypothesis is tested only on country mentions.

### Cohorts / Policy Diffusion

The notebook classifies each context into migrant cohorts and policy measures
with transparent keyword rules:

- migrant cohorts: refugees, asylum seekers, students, economic migrants,
  high-skilled workers, general migration
- policy measures: international law, national security, border control,
  allocation of resources, integration, returns/deportation, asylum procedure,
  visas/mobility, general policy

It then builds a weighted directed network:

- source node: `FRA`
- target node: mentioned country/entity
- edge weight: number of mentions
- edge attributes: year, migrant cohort, policy measure, reference type

### Policy Agency Mechanisms

`src/agency.py` classifies what another country/entity is doing in the argument:

- `learning_emulation_from` - another country is used as a model or lesson
- `coercion_intervention_to` - pressure, obligation, return, sanction, or rule
  is directed toward another country
- `competition` - France is compared against other countries for attractiveness
  or strategic position
- `exchange_cooperation` - treaties, partnerships, coordination, or joint action
- `neutral_reporting` - descriptive reporting without clear policy agency

The current labels are transparent keyword-rule labels with marker evidence.
Each row also stores an LLM-ready prompt in `policy_agency_llm_prompt`, so the
same schema can later be rerun with Qwen, Llama, or another zero-shot model
without changing the downstream tables.

### Narrative Framing and Argument Schemes

`src/framing.py` adds a 61-frame narrative taxonomy for how migration is
imagined: humanitarian obligation, security threat, economic contribution,
economic burden, legal category, border crisis, policy model, sea rescue,
camp conditions, and others.

It also adds:

- ternary narrative polarity: `positive_sympathy`, `positive_benefit`,
  `negative_risk`, `neutral_administrative`
- argumentative scheme: `argument_from_consequences`, `practical_reasoning`,
  `conceptual_definition`, or `other`
- extracted definition snippets when migration/refugee terms are defined with
  patterns such as "migration is..." or "refugees are..."

### High-Concreteness Events

`src/events.py` extracts concrete evidence snippets from high-concreteness
contexts. It records the date, mentioned country/entity, ISO3 target where
available, proper-noun anchors, detected countries in the context, and the
surrounding context window. This powers the evidence visibility map and the
fact-density timeline.

### Internal vs External Processes

`src/direction.py` adds a surface semantic-role-style direction label:

- `inbound_internal` - migration is directed toward France
- `outbound_from_domestic` - migration is from France outward
- `external_transnational` - migration is between third-party places
- `ambiguous` - no reliable direction signal

The current method uses auditable surface rules. A later version can replace
this with full SRL while preserving the same output columns.

## Outputs

The notebook saves the annotated mention table here:

- `data/processed/FRA_2018_migration_mentions.parquet`

It also saves audit CSVs here:

- `data/processed/entity_distribution_min26.csv` - displayed distribution of every entity mentioned more than 25 times
- `data/processed/entity_distribution_all_for_audit.csv` - full audit distribution of every retained mentioned entity

Vega-Altair visualizations are saved here:

- `data/processed/figures_altair/*.html` - interactive browser versions
- `data/processed/figures_altair/*.png` - static image exports for slides
- `data/processed/figures_altair/*.vl.json` - Vega-Lite chart specifications

The main saved charts are:

- `entity_distribution_top10` - 10 most mentioned countries/cases
- `entity_distribution_min26` - every entity mentioned more than 25 times
- `reference_type_sentiment_heatmap` - headline reference type x 6-level sentiment matrix
- `country_sentiment_mentions_top10` - top 10 entities split by 6-level sentiment
- `entity_sentiment_heatmap_min26` - 6-level sentiment heatmap for every entity with more than 25 mentions
- `entity_distribution_heatmap_min26` - mention-volume heatmap for every entity with more than 25 mentions
- `country_reference_type_mentions_top10` - top 10 entities split by policy / situation / mixed / neutral reference type
- `policy_vs_situation_sentiment_top10` - top 10 comparison of policy references versus international situation/context references
- `policy_situation_sentiment_heatmap` - 6-level sentiment heatmap for policy vs situation references
- `region_group_distribution` - mentions split into European countries, non-European countries/cases, EU, and French overseas territories
- `region_group_sentiment` - 6-level sentiment split by the same Europe / non-Europe / EU grouping
- `country_reference_heatmap_top10` - top 10 heatmap of reference-type intensity by mentioned entity

The extended notebook saves:

- `data/processed/FRA_2017_2022_migration_mentions_extended.parquet`
- `data/processed/FRA_2017_2022_result_notes.md`
- `data/processed/FRA_2017_2022_country_mention_profile.csv`
- `data/processed/FRA_2017_2022_country_year_profile.csv`
- `data/processed/FRA_2017_2022_country_year_concreteness_summary.csv`
- `data/processed/FRA_2017_2022_concreteness_feature_patterns.csv`
- `data/processed/FRA_2017_2022_concreteness_quote_examples.csv`
- `data/processed/FRA_2017_2022_country_context_examples.csv`
- `data/processed/FRA_2017_2022_cohort_policy_context_examples.csv`
- `data/processed/FRA_2017_2022_diffusion_edges.csv`
- `data/processed/FRA_2017_2022_diffusion_target_summary.csv`
- `data/processed/FRA_2017_2022_diffusion_network.graphml`
- `data/processed/FRA_2017_2022_policy_agency_edges.csv`
- `data/processed/FRA_2017_2022_policy_agency_network.graphml`
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
- `data/processed/figures_altair_extended/concreteness_density_by_weog.*`
- `data/processed/figures_altair_extended/concreteness_by_year_region.*`
- `data/processed/figures_altair_extended/country_concreteness_year_lines.*`
- `data/processed/figures_altair_extended/concreteness_feature_pattern_heatmap.*`
- `data/processed/figures_altair_extended/concreteness_quote_panels.html`
- `data/processed/figures_altair_extended/diffusion_top_targets.*`
- `data/processed/figures_altair_extended/cohort_policy_heatmap.*`
- `data/processed/figures_altair_extended/country_concreteness_bubble.*`
- `data/processed/figures_altair_extended/country_year_concreteness_heatmap.*`
- `data/processed/figures_altair_extended/country_cohort_heatmap.*`
- `data/processed/figures_altair_extended/country_policy_heatmap.*`

Advanced interactive figures are saved here:

- `data/processed/figures_interactive_advanced/policy_agency_network.html`
- `data/processed/figures_interactive_advanced/policy_agency_country_heatmap.html`
- `data/processed/figures_interactive_advanced/policy_hubs_pagerank.html`
- `data/processed/figures_interactive_advanced/policy_hubs_pagerank.png`
- `data/processed/figures_interactive_advanced/narrative_ternary.html`
- `data/processed/figures_interactive_advanced/narrative_ternary_by_party.html`
- `data/processed/figures_interactive_advanced/narrative_mirror_bars.html`
- `data/processed/figures_interactive_advanced/narrative_mirror_bars.png`
- `data/processed/figures_interactive_advanced/narrative_mirror_by_party.html`
- `data/processed/figures_interactive_advanced/narrative_mirror_by_party.png`
- `data/processed/figures_interactive_advanced/narrative_mirror_by_speaker_role.html`
- `data/processed/figures_interactive_advanced/narrative_mirror_by_speaker_role.png`
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
- `data/processed/figures_interactive_advanced/europe_country_year_event_heatmap.png`
- `data/processed/figures_interactive_advanced/europe_country_frame_heatmap.html`
- `data/processed/figures_interactive_advanced/europe_country_frame_heatmap.png`
- `data/processed/figures_interactive_advanced/europe_concrete_sankey.html`
- `data/processed/figures_interactive_advanced/europe_concrete_treemap.html`
- `data/processed/figures_interactive_advanced/europe_geolocated_concrete_event_point_map.html`
- `data/processed/figures_interactive_advanced/europe_geolocated_concrete_event_timeline.html`
- `data/processed/figures_interactive_advanced/concrete_conversations_explorer.html`
- `data/processed/figures_interactive_advanced/direction_agenda_split_bars.html`
- `data/processed/figures_interactive_advanced/direction_agenda_split_bars.png`
- `data/processed/figures_interactive_advanced/direction_agenda_by_party.html`
- `data/processed/figures_interactive_advanced/direction_agenda_by_party.png`

The `.html` files are the primary interactive outputs. Static PNG export is
attempted where the chart type supports it; complex layered/network charts may
save only HTML and Vega-Lite JSON.

To regenerate the requested advanced visualization bundle from the processed
2017-2022 tables, run:

```bash
python scripts/generate_requested_visualizations.py
```

For non-France country folders, use `scripts/run_extended_analysis.py`; it
generates the same diffusion, agency, concreteness, high-concreteness event,
and interactive visualization outputs inside each country-specific processed
directory.

The multi-country notebook also includes the stricter dyadic data model in
`src/data_model.py`: raw `entity_content` is resolved to
`target_country_iso3`, migration filtering is applied at `speech_id` level,
and then source-target bilateral matrices, concreteness matrices, asymmetry
tables, and shock-window tables are generated. The current combined outputs for
all available studied countries are saved under:

- `data/processed/all_studied_countries_visualization_index.html`
- `data/processed/ALL_AVAILABLE_COUNTRIES_dyadic_data_model/resolved_migration_country_mentions.parquet`
- `data/processed/ALL_AVAILABLE_COUNTRIES_dyadic_data_model/bilateral_matrix.csv`
- `data/processed/ALL_AVAILABLE_COUNTRIES_dyadic_data_model/bilateral_concreteness.csv`
- `data/processed/ALL_AVAILABLE_COUNTRIES_dyadic_data_model/asymmetry_table.csv`
- `data/processed/ALL_AVAILABLE_COUNTRIES_dyadic_data_model/shock_moria_fire_window.csv`
- `data/processed/ALL_AVAILABLE_COUNTRIES_dyadic_data_model/shock_belarus_crisis_window.csv`
- `data/processed/ALL_AVAILABLE_COUNTRIES_dyadic_data_model/shock_ukraine_war_window.csv`
- `data/processed/ALL_AVAILABLE_COUNTRIES_dyadic_data_model/figures_data_model/*.html`
- `data/processed/ALL_AVAILABLE_COUNTRIES_dyadic_data_model/figures_data_model/*.png`
