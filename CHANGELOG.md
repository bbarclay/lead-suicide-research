# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2026-04-27

### Added
- Initial public release with two complete manuscripts
  - Paper 2: County-level ecological study comparing alcohol access (null) and lead exposure (significant) as predictors of male suicide across 2,683 U.S. counties
  - Paper 3: Veteran-focused analysis testing mining × veteran interaction, eagle bone lead bioindicator validation
- Complete analysis codebase with 20+ Python scripts
- Master county-level dataset (3,255 counties, 122 variables)
- GitHub Pages site with professional HTML/CSS design
- Comprehensive documentation:
  - README.md with reproduction instructions
  - DATA_SOURCES.md with full provenance for every dataset
  - THIRD_PARTY_DATA_NOTICE.md clarifying licensing boundaries
  - CITATION.cff for academic citation
- BibTeX bibliographies for both papers (references_paper2.bib, references_paper3.bib)
- GitHub Actions workflow for automated Pages deployment
- MIT License for original code, text, and figures

### Data Sources Integrated
- CDC WONDER male suicide mortality (2018-2022)
- USGS Mineral Resources Data System (historical mining sites)
- USGS eagle bone lead data (Slabe et al. 2022 Science release)
- VA Annual Suicide Prevention Report 2025 (state-level veteran rates 2001-2023)
- EPA Toxics Release Inventory (industrial lead emissions)
- County Health Rankings 2024
- Census ACS 2022 (demographics, housing age, veteran population)
- IHME Global Burden of Disease 2019 (33-country panel)
- NHANES 2007-2018 (individual-level blood lead and PHQ-9)

### Key Findings Documented
- Alcohol access null across 12 independent tests (ΔR² = 0.004)
- Historical mining predicts male suicide with dose-response gradient (β = 3.53, p < 0.001)
- Mining × veteran interaction significant (β = 1.20, p = 0.005 with state FE)
- Eagle bone lead predicts veteran suicide at state level (r = +0.52, p = 0.003)
- Veteran/non-veteran suicide gap widened from 1.06 (2001) to 1.33 (2023)

### Limitations Explicitly Stated
- Ecological design (no individual-level causation proven)
- No bone-lead measurements in veteran suicide decedents (the decisive test)
- NHANES male-only subsample null for suicidal ideation (p = 0.77, power limitation)
- County-level divorce mediation not robust to controls

## Future Work

### Planned
- Individual-level validation study (bone-lead XRF in veteran suicide decedents vs controls)
- Difference-in-differences analysis of Superfund remediation dates
- Additional international replications

### Under Consideration
- Preprint submission to arXiv/medRxiv
- Journal submission (Environmental Health Perspectives, Military Medicine, American Journal of Public Health)
- Zenodo DOI archive

