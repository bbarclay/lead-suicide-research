#!/usr/bin/env python3
"""
Build the FINAL comprehensive county-level lead exposure master dataset.
Merges all successfully downloaded data sources.
"""

import pandas as pd
import os

OUT = "/Users/bobbarclay/Documents/soldiers"

def safe_str_fips(series, width=5):
    """Safely convert FIPS to zero-padded string."""
    return series.astype(str).str.strip().str.zfill(width)

print("=" * 70)
print("BUILDING FINAL COUNTY-LEVEL LEAD EXPOSURE MASTER DATASET")
print("=" * 70)

# ==================================================================
# 1. BASE: Census housing age data (3,222 counties)
# ==================================================================
housing = pd.read_csv(f"{OUT}/census_housing_age_detail_county.csv")
housing["county_fips"] = safe_str_fips(housing["county_fips"])
print(f"\n1. Base: Census housing age - {len(housing)} counties")

# ==================================================================
# 2. EPA Predicted Blood Lead Levels (RF model, 3,140 counties)
# ==================================================================
epa_file = f"{OUT}/epa_predicted_county_blood_lead.csv"
if os.path.exists(epa_file):
    epa = pd.read_csv(epa_file)
    epa["county_fips"] = safe_str_fips(epa["county_fips"])
    # Rename for clarity
    epa = epa.rename(columns={
        "mean_predicted_pct_elevated_bll": "epa_rf_predicted_pct_elevated_bll",
        "median_predicted_pct_elevated_bll": "epa_rf_median_pct_elevated_bll",
        "max_predicted_pct_elevated_bll": "epa_rf_max_pct_elevated_bll",
        "mean_predicted_pct_elevated_bll_v2": "epa_rf_v2_predicted_pct_elevated_bll",
        "n_tracts": "epa_n_census_tracts",
    })
    keep_cols = ["county_fips", "epa_n_census_tracts",
                 "epa_rf_predicted_pct_elevated_bll", "epa_rf_median_pct_elevated_bll",
                 "epa_rf_max_pct_elevated_bll", "epa_rf_v2_predicted_pct_elevated_bll",
                 "mean_pct_pre1940_housing", "mean_pct_pre1950_housing", "mean_pct_black"]
    keep_cols = [c for c in keep_cols if c in epa.columns]
    housing = housing.merge(epa[keep_cols], on="county_fips", how="left")
    n = housing["epa_rf_predicted_pct_elevated_bll"].notna().sum()
    print(f"2. EPA RF predicted BLL: matched {n} counties")

# ==================================================================
# 3. Census demographics
# ==================================================================
demo_file = f"{OUT}/census_demographics_for_lead.csv"
if os.path.exists(demo_file):
    demo = pd.read_csv(demo_file)
    demo["county_fips"] = safe_str_fips(demo["county_fips"])
    demo_cols = ["county_fips", "population", "median_income", "median_age",
                 "median_home_value", "pct_poverty", "pct_black", "pct_hispanic", "pct_renter"]
    demo_cols = [c for c in demo_cols if c in demo.columns]
    housing = housing.merge(demo[demo_cols], on="county_fips", how="left")
    print(f"3. Census demographics: merged")

# ==================================================================
# 4. Census children counts
# ==================================================================
child_file = f"{OUT}/census_children_under6_by_county.csv"
if os.path.exists(child_file):
    child = pd.read_csv(child_file)
    child["county_fips"] = safe_str_fips(child["county_fips"])
    child_keep = ["county_fips", "children_under_5", "children_under_10", "pct_children_under5"]
    child_keep = [c for c in child_keep if c in child.columns]
    housing = housing.merge(child[child_keep], on="county_fips", how="left")
    print(f"4. Census children: merged")

# ==================================================================
# 5. Michigan observed BLL (county level)
# ==================================================================
for suffix, desc in [("MI_07_11", "Michigan 2007-2011"), ("MI_14_16", "Michigan 2014-2016"),
                     ("OH_07_11", "Ohio 2007-2011"), ("OH_14_16", "Ohio 2014-2016")]:
    fpath = f"{OUT}/epa_observed_bll_{suffix}_county.csv"
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        df["county_fips"] = safe_str_fips(df["county_fips"])
        df = df.rename(columns={
            "n_tracts": f"n_tracts_{suffix}",
            "mean_observed_pct_elevated": f"observed_bll_{suffix}",
            "median_observed_pct_elevated": f"observed_bll_median_{suffix}",
            "max_observed_pct_elevated": f"observed_bll_max_{suffix}",
        })
        keep = ["county_fips"] + [c for c in df.columns if suffix in c]
        housing = housing.merge(df[keep], on="county_fips", how="left")
        n = housing[f"observed_bll_{suffix}"].notna().sum()
        print(f"5. {desc} observed BLL: {n} counties")

# ==================================================================
# 6. NY county-year blood lead (get latest year)
# ==================================================================
ny_file = f"{OUT}/ny_county_blood_lead_by_year.csv"
if os.path.exists(ny_file):
    ny = pd.read_csv(ny_file)
    ny["county_fips"] = safe_str_fips(ny["county_fips"])
    # Use most recent year
    ny_latest = ny[ny["Year"] == ny["Year"].max()].copy()
    ny_latest = ny_latest.rename(columns={
        "Tests": "ny_tests",
        "Total Elevated Blood Levels": "ny_total_elevated",
        "pct_elevated_5plus": "ny_pct_bll_ge5",
        "pct_elevated_10plus": "ny_pct_bll_ge10",
    })
    keep = ["county_fips", "ny_tests", "ny_total_elevated", "ny_pct_bll_ge5", "ny_pct_bll_ge10"]
    keep = [c for c in keep if c in ny_latest.columns]
    housing = housing.merge(ny_latest[keep], on="county_fips", how="left")
    n = housing["ny_pct_bll_ge5"].notna().sum()
    print(f"6. NY blood lead ({ny['Year'].max()}): {n} counties")

# ==================================================================
# 7. USGS soil lead (if geocoded)
# ==================================================================
usgs_file = f"{OUT}/usgs_soil_lead_by_county.csv"
if os.path.exists(usgs_file):
    usgs = pd.read_csv(usgs_file)
    usgs["county_fips"] = safe_str_fips(usgs["county_fips"])
    usgs = usgs.rename(columns={
        "mean_soil_lead_ppm": "usgs_soil_lead_ppm",
        "median_soil_lead_ppm": "usgs_soil_lead_median_ppm",
        "total_soil_samples": "usgs_n_soil_samples",
    })
    keep = ["county_fips", "usgs_soil_lead_ppm", "usgs_soil_lead_median_ppm", "usgs_n_soil_samples"]
    keep = [c for c in keep if c in usgs.columns]
    housing = housing.merge(usgs[keep], on="county_fips", how="left")
    n = housing["usgs_soil_lead_ppm"].notna().sum()
    print(f"7. USGS soil lead: {n} counties")
else:
    print(f"7. USGS soil lead: geocoding still in progress (run usgs_soil_lead_by_county.csv)")

# ==================================================================
# 8. Suicide rates and other outcomes from existing data
# ==================================================================
real_file = f"{OUT}/real_county_dataset.csv"
if os.path.exists(real_file):
    real = pd.read_csv(real_file, low_memory=False)
    if "FIPS" in real.columns:
        real["county_fips"] = safe_str_fips(real["FIPS"])
        # Key outcome and control variables
        outcome_cols = [c for c in real.columns if any(kw in c.lower()
                        for kw in ["suicide", "gun", "firearm", "death", "overdose",
                                   "veteran", "rural", "mining", "elevation",
                                   "premature", "injury", "motor"])]
        keep = ["county_fips"] + [c for c in outcome_cols if c not in housing.columns]
        if len(keep) > 1:
            housing = housing.merge(real[keep], on="county_fips", how="left")
            print(f"8. Suicide/outcome data: added {len(keep)-1} variables")

# ==================================================================
# 9. EPA Superfund sites by county
# ==================================================================
sf_file = f"{OUT}/epa_superfund_by_county.csv"
if os.path.exists(sf_file):
    sf = pd.read_csv(sf_file)
    fips_col = [c for c in sf.columns if "fips" in c.lower()]
    if fips_col:
        sf["county_fips"] = safe_str_fips(sf[fips_col[0]])
        sf_cols = [c for c in sf.columns if c not in housing.columns and "superfund" in c.lower()]
        if sf_cols:
            sf_keep = ["county_fips"] + sf_cols
            housing = housing.merge(sf[sf_keep], on="county_fips", how="left")
            print(f"9. EPA Superfund: added {len(sf_cols)} variables")

# ==================================================================
# SAVE MASTER
# ==================================================================
master_file = f"{OUT}/county_lead_exposure_master.csv"
housing.to_csv(master_file, index=False)

print(f"\n{'='*70}")
print(f"MASTER DATASET SAVED: {master_file}")
print(f"Shape: {housing.shape[0]} rows x {housing.shape[1]} columns")
print(f"{'='*70}")

# Print all columns with stats
print(f"\nALL COLUMNS ({housing.shape[1]}):")
print(f"{'#':>3}  {'Column':<55} {'N':>6}  {'Mean':>10}  {'Std':>10}")
print("-" * 90)
for i, c in enumerate(housing.columns):
    vals = pd.to_numeric(housing[c], errors="coerce")
    n = vals.notna().sum()
    if n > 10:
        print(f"{i+1:3d}  {c:<55} {n:6d}  {vals.mean():10.4f}  {vals.std():10.4f}")
    else:
        print(f"{i+1:3d}  {c:<55} {'text':>6}")

# ==================================================================
# KEY CORRELATIONS
# ==================================================================
print(f"\n{'='*70}")
print("KEY CORRELATIONS: LEAD EXPOSURE vs SUICIDE")
print(f"{'='*70}")

lead_vars = [c for c in housing.columns if any(kw in c.lower()
             for kw in ["predicted_pct", "pct_pre19", "bll", "soil_lead"])]
suicide_vars = [c for c in housing.columns if "suicide_rate" in c.lower() and "crude" not in c.lower()]

if suicide_vars:
    for sv in suicide_vars[:2]:
        print(f"\n  Correlations with {sv} (n={housing[sv].notna().sum()}):")
        for lv in lead_vars:
            n_both = housing[[lv, sv]].dropna().shape[0]
            if n_both > 50:
                corr = housing[lv].corr(housing[sv])
                print(f"    {lv:<55} r={corr:+.4f}  (n={n_both})")

# ==================================================================
# DATA SOURCE INVENTORY
# ==================================================================
print(f"\n{'='*70}")
print("COMPLETE DATA SOURCE INVENTORY")
print(f"{'='*70}")

inventory = {
    "county_lead_exposure_master.csv": {
        "desc": "MASTER merged dataset - all indicators",
        "geo": "County FIPS (5-digit)",
        "coverage": f"{housing.shape[0]} counties",
        "key_vars": "Predicted BLL, housing age, demographics, soil lead, suicide rates",
    },
    "epa_predicted_county_blood_lead.csv": {
        "desc": "EPA Random Forest model - predicted % elevated BLL by county",
        "geo": "County FIPS (aggregated from census tract)",
        "coverage": "3,140 counties nationwide",
        "key_vars": "epa_rf_predicted_pct_elevated_bll (model v1 & v2)",
    },
    "census_housing_age_detail_county.csv": {
        "desc": "Census ACS housing year built - lead paint proxy",
        "geo": "County FIPS",
        "coverage": "3,222 counties",
        "key_vars": "pct_pre1940, pct_pre1950, pct_pre1960, pct_pre1978",
    },
    "ny_county_blood_lead_by_year.csv": {
        "desc": "NY State OBSERVED blood lead by county and year",
        "geo": "County FIPS (NY only)",
        "coverage": "57 NY counties, 2000-2021",
        "key_vars": "pct_elevated_5plus, pct_elevated_10plus",
    },
    "epa_observed_bll_MI_07_11_county.csv": {
        "desc": "Michigan OBSERVED % elevated BLL (2007-2011)",
        "geo": "County FIPS",
        "coverage": "71 MI counties",
        "key_vars": "mean_observed_pct_elevated",
    },
    "epa_observed_bll_OH_07_11_county.csv": {
        "desc": "Ohio OBSERVED % elevated BLL (2007-2011)",
        "geo": "County FIPS",
        "coverage": "88 OH counties",
        "key_vars": "mean_observed_pct_elevated",
    },
    "usgs_soil_lead_concentrations.csv": {
        "desc": "USGS soil lead (Pb) concentrations - point data",
        "geo": "Lat/Lon (93,640 samples)",
        "coverage": "Continental US",
        "key_vars": "pb_value (ppm)",
    },
    "usgs_soil_lead_by_state.csv": {
        "desc": "USGS soil lead aggregated by state",
        "geo": "State",
        "coverage": "53 states/territories",
        "key_vars": "mean_soil_lead_ppm, median_soil_lead_ppm",
    },
    "michigan_BLL_under6_county_2016.csv": {
        "desc": "Michigan county BLL, children under 6 (2016)",
        "geo": "County name",
        "coverage": "85 MI counties",
        "key_vars": "all_greater5_total, all_greater5_percent",
    },
    "utah_county_blood_lead.csv": {
        "desc": "Utah county blood lead levels (2000-2009)",
        "geo": "County name",
        "coverage": "29 UT counties",
        "key_vars": "children_elevated",
    },
    "socrata_data_Children_Under_6_yrs_with_Elevated_.csv": {
        "desc": "NYC elevated blood lead by neighborhood",
        "geo": "Borough/neighborhood",
        "coverage": "NYC sub-borough areas",
        "key_vars": "BLL >=5, >=10, >=15 mcg/dL counts and rates",
    },
    "Supplement B/RandomForest v1 model predictions.csv": {
        "desc": "EPA national census tract predicted BLL (73,031 tracts)",
        "geo": "Census tract GEOID",
        "coverage": "73,031 census tracts nationwide",
        "key_vars": "RF.OH0711_var5_pred (predicted % elevated BLL)",
    },
}

for fname, info in inventory.items():
    fpath = f"{OUT}/{fname}"
    exists = os.path.exists(fpath)
    size = f"{os.path.getsize(fpath)/1024:.0f} KB" if exists else "missing"
    status = "OK" if exists else "MISSING"
    print(f"\n  [{status}] {fname} ({size})")
    print(f"    {info['desc']}")
    print(f"    Geo: {info['geo']} | Coverage: {info['coverage']}")
    print(f"    Key vars: {info['key_vars']}")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
