# Lead Exposure and Male Suicide in U.S. Counties and Veterans

[![GitHub](https://img.shields.io/badge/GitHub-bbarclay%2Flead--suicide--research-blue?logo=github)](https://github.com/bbarclay/lead-suicide-research)
[![Website](https://img.shields.io/badge/Website-GitHub%20Pages-orange?logo=githubpages)](https://bbarclay.github.io/lead-suicide-research)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Reproducible data and analysis code for two manuscripts testing whether environmental and occupational lead exposure is associated with male suicide, with a particular focus on the widening veteran suicide gap.

**Status:** hypothesis-generating, ecological preprints in preparation. Individual-level validation (bone-lead measurements in veteran suicide decedents) has not yet been performed; see [Limitations](#what-this-work-does-not-prove) below.

**License:** original code, manuscript text, and original figures are released under MIT (see `LICENSE`); bundled third-party data retain their upstream terms and are documented in `DATA_SOURCES.md` and `THIRD_PARTY_DATA_NOTICE.md`.

## What's in this repository

| Path | What it is |
|---|---|
| `paper2_lead_suicide.tex` / `.pdf` | **Paper 2 — ecological study of 2,683 U.S. counties.** Compares alcohol access (null) and environmental lead exposure (significant) as predictors of male suicide, with NHANES, IHME, and military convergent evidence. |
| `paper3_veteran_lead_suicide.tex` / `.pdf` | **Paper 3 — veteran-focused.** Tests the state- and county-level mining × veteran interaction, validates against VA state-level suicide data, and introduces wildlife bone lead (eagle femur Pb, Slabe 2022) as a bioindicator. |
| `references_paper2.bib` / `references_paper3.bib` | BibTeX bibliographies for professional LaTeX compilation. |
| `figure_eagle_vet_suicide.pdf` | Publication-grade figure: eagle femur lead vs. 2023 VA veteran suicide rate by state. |
| Root-level `*.py` scripts | One script per analysis layer (see [Reproducing the results](#reproducing-the-results)). |
| `DATA_SOURCES.md` | Provenance for every dataset, with direct download URLs and license terms. |
| `THIRD_PARTY_DATA_NOTICE.md` | Licensing clarification for bundled third-party data. |
| `docs/` | GitHub Pages source for the public project website. |
| `CITATION.cff` | Academic citation metadata. |
| `CONTRIBUTING.md` | Guidelines for contributing corrections, improvements, or collaborations. |
| `CHANGELOG.md` | Version history and future work roadmap. |

## Headline findings

1. **Alcohol access is null.** Twelve independent tests of county-level alcohol outlet density, bar density, liquor-store density, excessive drinking prevalence, and dry-county status returned null or paradoxical results (ΔR² = 0.004).
2. **Historical lead/zinc/copper mining predicts county male suicide** with a monotonic dose-response gradient (β = 3.53, p < 0.001; state fixed-effects β = 1.22, p < 0.001; 1,000-iteration bootstrap β = 3.41 [2.86, 3.98]).
3. **The mining association is specific to suicide.** It does not predict drug overdose, homicide, motor-vehicle death, or general mental distress — a double dissociation that narrows the space of possible confounders.
4. **Veteran concentration amplifies the effect.** Mining × veteran interaction is significant (β = 1.20, p = 0.005 with state FE). In high-veteran counties the mining effect is β = 2.54 (p < 0.001); in low-veteran counties it is null.
5. **Eagle bone lead (wildlife bioindicator, Slabe et al. 2022 *Science*) reproduces the signal** at the state level: % of eagles with chronic lead poisoning → 2023 VA veteran suicide rate, r = +0.52, p = 0.003 (N = 31 states). Eagle Pb predicts suicide but not overdose, motor-vehicle death, or homicide — the same specificity pattern seen in the county analysis, now with non-human biology.
6. **The mining effect is 3.6× stronger in states where wildlife bone lead is independently high** (β = 1.30 vs. β = 0.36 in low-eagle-Pb states). This is a dose-response at the level of the exposure proxy itself.
7. **Within-country evidence** across 33 high-income countries (1990–2019): country-fixed-effects β = 0.245 (p < 10⁻¹⁰); Δlead-mortality → Δsuicide r = +0.37; country-level trend slopes r = +0.50 (p = 0.003).
8. **The veteran/non-veteran suicide gap widened from 1.06 (2001) to 1.33 (2023)** despite sustained VA mental-health spending. The widening is most severe in high-mining states (Montana, Utah, Arizona, Missouri, Idaho) and in age cohorts whose childhood coincided with peak leaded-gasoline exposure (55–74).

## What this work does not prove

This is an ecological study. Every finding above is community-level or wildlife-level, not individual-level.

- Specifically, we do **not** have bone-lead measurements in individual veteran suicide decedents versus matched controls. That study is the decisive test and has not been conducted. We hope it will be, and we have structured these papers to motivate it.
- The NHANES individual-level analysis of blood lead and suicidal ideation (N = 24,050 adults) is significant in the full sample (aOR = 1.21, p < 0.001) but **null in the male-only subsample** (p = 0.77), likely reflecting limited statistical power (~450 male SI events). We report this honestly rather than presenting only the favorable pooled result.
- In state-level models with demographic controls (rurality, poverty, race, veteran percentage, median age), the eagle-lead coefficient attenuates and its unique contribution cannot be cleanly separated from the rural/veteran/extractive state profile.
- The hypothesized biological mechanism (lead → prefrontal/serotonergic damage → impulsive self-harm; lead → endocrine disruption → ED → relationship collapse → suicide) is drawn from published studies in other populations. We do not validate it directly in U.S. veterans dying by suicide.

Taken together, the evidence **justifies the decisive individual-level test**; it does not substitute for it.

## Reproducing the results

```bash
# From the repository root:
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Scripts that pull from the U.S. Census API need a free key
# (register at https://api.census.gov/data/key_signup.html):
cp .env.example .env
# then edit .env and set CENSUS_API_KEY=your_key
export $(cat .env | xargs)   # or use direnv / dotenv

# Raw source data: see DATA_SOURCES.md for download instructions.
# Most CSVs committed to the repo are <5 MB; large raw files (NHANES .XPT,
# EPA LCR, CDC PLACES) must be downloaded separately before running the
# pull/fetch/build scripts.

# Rebuild the master county-level dataset (needs the committed inputs plus any
# optional raw downloads described in DATA_SOURCES.md)
python build_county_lead_master.py

# Replicate the state-level eagle × veteran suicide analysis
python eagle_lead_suicide_analysis.py          # main merge + correlations
python eagle_plus_soil_lead_analysis.py        # horse race vs soil & mining
python eagle_specificity_robustness.py         # specificity + demographic controls
python eagle_county_moderation.py              # county-level stratified by state eagle Pb
python eagle_figure.py                         # generates figure_eagle_vet_suicide.pdf

# IHME within-country evidence
python ihme_within_country_analysis.py
python ihme_developed_countries.py             # high-income country subset

# EPA TRI industrial lead emissions
python pull_tri_lead.py                        # downloads from EPA Envirofacts

# NHANES individual-level blood lead vs PHQ-9 suicidal ideation
python nhanes_lead_suicide.py                  # requires nhanes_data/*.XPT

# Compile the manuscripts (LaTeX)
pdflatex paper2_lead_suicide.tex && pdflatex paper2_lead_suicide.tex
pdflatex paper3_veteran_lead_suicide.tex && pdflatex paper3_veteran_lead_suicide.tex
```

`build_county_lead_master.py` will also merge the optional generated intermediate `usgs_soil_lead_by_county.csv` if you have produced it locally from the raw USGS point file; otherwise it logs that county soil-lead geocoding is still pending and continues without that merge.

## GitHub Pages

A scoped project website now lives in `docs/`, with deployment automation in `.github/workflows/pages.yml`.

- Push this repository to GitHub.
- In the repository's Pages settings, set the source to **GitHub Actions**.
- Pushes to `main` will then publish the site automatically.

## Citation

If you use code or data from this repository, please cite the software archive and the relevant manuscripts. See `CITATION.cff`.

## Questions, corrections, collaboration

Issues and pull requests welcome. Email `barclaybrandon@hotmail.com` for collaboration inquiries. Internal planning materials are intentionally excluded from the public repository.

The decisive individual-level test requires NCHS Research Data Center access to NHANES III linked mortality files and/or bone-lead XRF on stored veteran biospecimens. If you are an academic investigator with access to either and are interested in collaborating, please get in touch.
