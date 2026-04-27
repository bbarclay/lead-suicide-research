# Data Sources

Every dataset used in this repository is publicly available. This document lists each source, its origin, the license or terms of use, and how to re-download it. Most large raw files are not committed to git; the main exceptions are a small number of public supplement files that are directly consumed by the analysis scripts.

## Core outcome data

### CDC WONDER — male suicide mortality
- **File:** `cdc_wonder_male_suicide_ALL_counties_2018_2022.tsv`
- **Source:** [CDC WONDER Underlying Cause of Death](https://wonder.cdc.gov/ucd-icd10-expanded.html)
- **Query:** ICD-10 X60–X84 (intentional self-harm), male sex, ages 18+, 2018–2022, all counties
- **License:** U.S. government public domain
- **Notes:** Counties with suppressed counts (< 10 deaths over 5 years) are excluded from most analyses per CDC suppression rules

### VA Office of Mental Health and Suicide Prevention
- **Files:** `VA_National_Veteran_Suicide_Rates_2001-2023.csv`, `VA_State_Veteran_Suicide_Rates_2023.csv`, `VA_Veteran_Suicides_by_State_2001-2023_FULL.csv`, `VA_suicides_by_age_state.csv`, `VA_suicides_by_method_state.csv`
- **Source:** [VA Annual National Veteran Suicide Prevention Report 2025](https://www.mentalhealth.va.gov/suicide_prevention/data.asp)
- **License:** U.S. government public domain

## Primary exposure data

### USGS — historical mining sites
- **File:** `usgs_mining_sites_by_county.csv`
- **Source:** [USGS Mineral Resources Data System (MRDS)](https://mrdata.usgs.gov/mrds/)
- **License:** U.S. government public domain
- **Aggregation:** Counts of all recorded mining sites and the subset with lead, zinc, or copper as a listed commodity, aggregated by county FIPS.

### USGS — eagle bone lead (Slabe et al. 2022)
- **Files:** `Pb_Eagle_Femur.csv`, `Pb_Eagle_Blood.csv`, `Pb_Eagle_Liver.csv`
- **Source:** [USGS ScienceBase 10.5066/P9BXIY3B](https://www.sciencebase.gov/catalog/item/61cdf823d34ed79293fc871f)
- **Citation:** Slabe VA et al. 2022. *Science* 375:779–782. [doi:10.1126/science.abj3068](https://doi.org/10.1126/science.abj3068)
- **License:** U.S. government public domain

### USGS — soil lead
- **Files:** `usgs_soil_lead_by_state.csv`, `usgs_soil_lead_by_county_sample.csv`, `usgs_soil_lead_concentrations.csv` (large, gitignored), `usgs_soil_lead_by_county.csv` (generated intermediate; not committed)
- **Source:** [USGS National Geochemical Database — soils](https://mrdata.usgs.gov/soilgeochemistry/)
- **License:** U.S. government public domain
- **Notes:** `usgs_soil_lead_by_county.csv` is a locally generated county aggregation from the raw point file when geocoding / spatial joins are available. `build_county_lead_master.py` will skip that merge if the generated file is absent.

### EPA / state surveillance — predicted blood lead, observed blood lead, LCR, Superfund, TRI
- **Files:**
  - `epa_predicted_county_blood_lead.csv` — county-level summary of the national Random Forest childhood lead-risk model
  - `epa_observed_bll_MI_07_11_county.csv`, `epa_observed_bll_MI_14_16_county.csv`, `epa_observed_bll_OH_07_11_county.csv`, `epa_observed_bll_OH_14_16_county.csv` — county aggregates derived from tract-level observed EBLL tables
  - `Supplement B/RandomForest v1 model predictions.csv`, `Supplement B/RandomForest v2 model predictions.csv` — tract-level national model outputs from the EPA supplement release
  - `Supplement B/Supplement_B_ MI_07_11.csv`, `Supplement B/Supplement_B_MI_14_16.csv`, `Supplement B/Supplement_B_OH_07_11.csv`, `Supplement B/Supplement_B_OH_14_16.csv` — tract-level observed EBLL tables used to derive the county aggregates above
  - `epa_lcr_samples_county.csv` (large, gitignored) — Lead and Copper Rule sampling
  - `epa_superfund_by_county.csv` — Superfund sites per county
  - `tri_lead_by_state_2022.csv` — Aggregated from EPA Toxics Release Inventory 2022
  - `tri_lead_facility_2022.csv` — Facility-level TRI lead releases (2022)
- **Sources:**
  - [Data.gov / EPA ORD: A U.S. Lead Exposure Hotspots Analysis](https://catalog.data.gov/dataset/a-u-s-lead-exposure-hotspots-analysis) — public release for the Zartarian et al. 2024 supplement files and data dictionaries
  - [EPA Lead mapping overview](https://www.epa.gov/lead/mapping)
  - [EPA Toxics Release Inventory](https://www.epa.gov/toxics-release-inventory-tri-program) (pull script: `pull_tri_lead.py`)
  - [EPA SDWIS / LCR](https://www.epa.gov/enviro/sdwis-search)
  - [EPA Superfund NPL](https://www.epa.gov/superfund)
- **Citation:** Zartarian Morrison VG et al. 2024. *Environmental Science & Technology* 58(7):3311-3321. [doi:10.1021/acs.est.3c07881](https://doi.org/10.1021/acs.est.3c07881)
- **License:** EPA ScienceHub public-access license/disclaimer for the supplement release; EPA-produced datasets are public domain unless otherwise specified in upstream metadata
- **Notes:** `fetch_county_lead_extra.py` aggregates the tract-level Supplement B observed files into the committed county summaries.

### IHME — Global Burden of Disease (cross-country)
- **Files:** `ihme_gbd_lead_vs_suicide_by_country_2019.csv`, `ihme_gbd_lead_vs_suicide_panel.csv`, `ihme_gbd_lead_exposure_death_rate_by_country.csv`, `ihme_gbd_suicide_death_rate_by_country.csv`
- **Source:** [IHME GBD 2019 results](https://vizhub.healthdata.org/gbd-results/)
- **License:** IHME Free-of-Charge Non-commercial User Agreement (viz & analysis OK; republication of raw rows requires IHME notification)

### NHANES — individual-level blood lead & mental health
- **Files:** `nhanes_data/*.XPT` (gitignored — re-download via `nhanes_lead_suicide.py`)
- **Source:** [NHANES 2007–2018 public-use data](https://wwwn.cdc.gov/nchs/nhanes/)
- **Cycles used:** E (2007–08), F (2009–10), G (2011–12), H (2013–14), I (2015–16), J (2017–18)
- **Modules:** DEMO (demographics), PBCD/PbCd (blood lead), DPQ (PHQ-9 depression, including item 9 on suicidal ideation)
- **License:** U.S. government public domain

## Covariate data

### Census American Community Survey (ACS) — 5-year, 2018–2022
- **Files:** `real_census_acs_data.csv`, `census_demographics_for_lead.csv`, `census_race_ethnicity_by_county.csv`, `census_median_age_by_county.csv`, `census_living_alone_by_county.csv`, `census_pre1978_housing_by_county.csv`, `census_pre1980_housing_by_county.csv`, `census_housing_age_detail_county.csv`
- **Source:** [Census ACS 5-year API](https://www.census.gov/data/developers/data-sets/acs-5year.html)
- **License:** U.S. government public domain

### USDA — rural-urban continuum codes
- **File:** `rucc_2023.csv`
- **Source:** [USDA ERS Rural-Urban Continuum Codes 2023](https://www.ers.usda.gov/data-products/rural-urban-continuum-codes/)
- **License:** U.S. government public domain

### VA VetPop2023 — veteran population by state, age, sex
- **Files:** `VetPop2023_State_AgeSex.xlsx`, `VetPop2023_State_Population.csv`, `veteran_employment_by_county.csv`, `veteran_period_of_service_by_county.csv`
- **Source:** [VA National Center for Veterans Analysis and Statistics](https://www.va.gov/vetdata/Veteran_Population.asp)
- **License:** U.S. government public domain

### CDC PLACES (local-area health data)
- **File:** `cdc_places_county_2025.csv` (large, gitignored)
- **Source:** [CDC PLACES 2024 release](https://www.cdc.gov/places/)
- **License:** U.S. government public domain

### County Health Rankings
- **File:** `real_county_health_rankings_2024.csv`
- **Source:** [County Health Rankings & Roadmaps](https://www.countyhealthrankings.org/)
- **License:** CC BY-NC-SA 4.0 (noncommercial research and reporting OK with attribution)

### ATF firearm dealer counts
- **File:** `firearm_dealers_by_county.csv`
- **Source:** [ATF Federal Firearms Licenses (FFL)](https://www.atf.gov/firearms/listing-federal-firearms-licensees)
- **License:** U.S. government public domain

## Aggregated master dataset

- **File:** `real_county_dataset.csv` (3,255 counties, 122 variables)
- **Build:** `build_county_lead_master.py` merges the above sources by county FIPS. Re-running `build_county_lead_master.py` with the committed inputs plus any optional raw source files in place reproduces the master dataset; if `usgs_soil_lead_by_county.csv` is absent, the county soil-lead merge is skipped rather than treated as a hard failure.
- **Structure:** one row per county, one column per covariate or outcome.

## Files excluded from git (re-download instructions in code)

See `.gitignore`. Large raw files (>5 MB each) are downloaded lazily by their respective `pull_*.py` or `fetch_*.py` scripts.

## Attribution and reuse

- Original analytical code, manuscript text, and original figures in this repository are released under the MIT License (see `LICENSE`).
- U.S. federal datasets here are generally public domain or public-access government works, but you should still follow the source-specific notices linked above.
- IHME and County Health Rankings require attribution under their noncommercial terms; commercial reuse requires separate agreements.
- The EPA ScienceHub supplement files in `Supplement B/` retain their upstream license/disclaimer and are not relicensed under MIT.
- See `THIRD_PARTY_DATA_NOTICE.md` before redistributing bundled data files.
