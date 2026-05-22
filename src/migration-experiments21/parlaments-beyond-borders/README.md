# Parliaments Beyond Borders - Migration Discourse Pilot

DHH26 hackathon project analyzing how national parliaments reference
foreign countries in migration debates. The pilot uses the translated
`ParlaMint-en.ana` corpus, so entity names and typology markers are English.

## Pilot scope

France, year 2018 - chosen because the Collomb Law on immigration
was debated and adopted that year, producing a dense and
internationally-comparative migration discussion.

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

## Extended 2017-2022 Methods

For the conceptual framing, research questions, and validation logic, see
`METHODOLOGY.md`.

The extended notebook uses all available France files:

- `FRA_2017_facts.parquet`
- `FRA_2018_facts.parquet`
- `FRA_2019_facts.parquet`
- `FRA_2020_facts.parquet`
- `FRA_2021_facts.parquet`
- `FRA_2022_facts.parquet`

It adds two analysis layers.

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
- `data/processed/FRA_2017_2022_country_context_examples.csv`
- `data/processed/FRA_2017_2022_cohort_policy_context_examples.csv`
- `data/processed/FRA_2017_2022_diffusion_edges.csv`
- `data/processed/FRA_2017_2022_diffusion_target_summary.csv`
- `data/processed/FRA_2017_2022_diffusion_network.graphml`
- `data/processed/figures_altair_extended/concreteness_density_by_weog.*`
- `data/processed/figures_altair_extended/concreteness_by_year_region.*`
- `data/processed/figures_altair_extended/diffusion_top_targets.*`
- `data/processed/figures_altair_extended/cohort_policy_heatmap.*`
- `data/processed/figures_altair_extended/country_concreteness_bubble.*`
- `data/processed/figures_altair_extended/country_year_concreteness_heatmap.*`
- `data/processed/figures_altair_extended/country_cohort_heatmap.*`
- `data/processed/figures_altair_extended/country_policy_heatmap.*`
