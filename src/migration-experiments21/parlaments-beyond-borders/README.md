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
