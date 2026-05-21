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
- **Sentiment**: positive / negative / neutral

Sentiment buckets are based on ParlaMint's categorical
`sentence_sentiment_ana` labels:

- **negative**: `senti:negneg`, `senti:mixneg`
- **neutral**: `senti:neuneg`, `senti:neupos`
- **positive**: `senti:mixpos`, `senti:pospos`

The numeric `sentence_sentiment_value` is kept in the output for inspection,
but it is not centered at zero in this file, so it should not be interpreted
with a simple negative/positive zero threshold.

French overseas territories such as Mayotte and French Guiana are kept in the
analysis because they are important in migration debates, but they are marked
as `geo_class = french_overseas`. They are not foreign states.

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

## Outputs

The notebook saves the annotated mention table here:

- `data/processed/FRA_2018_migration_mentions.parquet`

It also saves four visualization files here:

- `data/processed/entity_distribution_all.csv` - complete distribution of every retained mentioned entity
- `data/processed/figures/entity_distribution_top40.png` - readable distribution chart for the 40 most mentioned entities
- `data/processed/figures/entity_distribution_all.png` - full all-entities distribution chart
- `data/processed/figures/country_sentiment_mentions.png` - top mentioned entities colored by positive / negative / neutral sentiment
- `data/processed/figures/country_reference_type_mentions.png` - top mentioned entities split by policy / situation / mixed / neutral reference type
- `data/processed/figures/policy_vs_situation_sentiment.png` - sentiment-colored comparison of policy references versus international situation/context references
- `data/processed/figures/country_reference_heatmap.png` - heatmap of reference-type intensity by mentioned entity
