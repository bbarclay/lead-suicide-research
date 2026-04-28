# Contributing to Lead Suicide Research

Thank you for your interest in improving this research repository. This document provides guidelines for contributions.

## Ways to Contribute

### Reporting Issues
- **Data corrections:** If you find errors in the county-level data or source file mappings, open an issue with the specific FIPS code and correction
- **Methodological concerns:** For statistical or analytical questions, open an issue referencing the specific script and line number
- **Documentation improvements:** Typos, unclear instructions, or missing context in README/DATA_SOURCES.md

### Proposing Changes

#### Code contributions
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/description`
3. Make your changes with clear commit messages
4. Ensure scripts remain fresh-clone runnable (relative paths only)
5. Open a pull request describing the change and its rationale

#### Data additions
If you have additional county-level datasets that could strengthen the analysis:
- Provide the data source and license terms
- Include a data dictionary
- Add a loading/merge script following the pattern in `build_county_lead_master.py`
- Document provenance in DATA_SOURCES.md

### Research Collaboration

For substantive research collaboration inquiries (access to restricted datasets, bone-lead validation studies, grant partnerships), email `barclaybrandon@hotmail.com` directly rather than using GitHub issues.

## Code Standards

- **Python:** Follow PEP 8 style (the existing code uses minimal, readable patterns)
- **Documentation:** Each analysis script should have a docstring describing inputs, outputs, and key findings
- **Reproducibility:** No absolute paths, no hardcoded API keys, no manual steps that can't be scripted
- **Git:** Keep commits atomic (one logical change per commit)

## Scope Boundaries

This repository intentionally excludes:
- Individual-level health records (use NCHS Research Data Center for NHANES III mortality files)
- Proprietary datasets (only public-domain or properly licensed data)
- Internal planning documents (grant timelines, outreach strategy)

## Review Process

Pull requests will be reviewed for:
1. Correctness (does the code do what it claims?)
2. Reproducibility (can it run on a fresh clone?)
3. Documentation (is the change explained?)
4. License compatibility (are data sources properly attributed?)

## Code of Conduct

- Be respectful and constructive
- Cite your sources
- Acknowledge limitations explicitly
- Prioritize scientific accuracy over narrative convenience

## Priority Areas for Contribution

The following would be especially valuable:
- **Superfund remediation dates:** County-level dates of EPA remediation actions for difference-in-differences analysis
- **MOS-level exposure data:** Military Occupational Specialty-specific firing-range time estimates
- **State-level longitudinal panels:** Extended state-level datasets for 1990-2023
- **Replication in other countries:** County- or municipality-level data from other high-income countries
- **Individual-level validation:** If you have access to bone-lead XRF data with mortality follow-up

## Questions?

Open a discussion (if enabled) or email `barclaybrandon@hotmail.com`.
