#!/usr/bin/env python3
"""
Additional county-level lead data sources:
1. Download Vox lead risk data (Census tract level)
2. Process the GDB from EPA hotspots
3. Try more CDC data endpoints
4. Try Kaggle CDC blood lead surveillance
5. Compile everything into a county-level master
"""

import requests
import pandas as pd
import io
import os
import json

OUT = "/Users/bobbarclay/Documents/soldiers"
import os as _os
CENSUS_KEY = _os.environ.get("CENSUS_API_KEY")
if not CENSUS_KEY:
    raise RuntimeError("Set CENSUS_API_KEY in your environment (see .env.example). Get a key free from https://api.census.gov/data/key_signup.html")

# ============================================================================
# 1. Download Vox lead exposure risk source data (Census ACS housing age)
# ============================================================================
print("\n" + "="*70)
print("1. Vox Lead Exposure Risk - Census Housing Data")
print("="*70)

# Vox uses Census B25034 (housing age) and S1701 (poverty) data
# We can get the same data directly from Census at county level

# Get B25034 at a finer level - all the age bands
year = 2022
variables = [
    "B25034_001E",  # Total
    "B25034_002E",  # Built 2020 or later
    "B25034_003E",  # Built 2010 to 2019
    "B25034_004E",  # Built 2000 to 2009
    "B25034_005E",  # Built 1990 to 1999
    "B25034_006E",  # Built 1980 to 1989
    "B25034_007E",  # Built 1970 to 1979
    "B25034_008E",  # Built 1960 to 1969
    "B25034_009E",  # Built 1950 to 1959
    "B25034_010E",  # Built 1940 to 1949
    "B25034_011E",  # Built 1939 or earlier
    "NAME"
]

url = (f"https://api.census.gov/data/{year}/acs/acs5?"
       f"get={','.join(variables)}&for=county:*&key={CENSUS_KEY}")

try:
    r = requests.get(url, timeout=30)
    if r.status_code == 200:
        data = r.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        for col in variables[:-1]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df["county_fips"] = df["state"] + df["county"]
        total = df["B25034_001E"]

        # Key lead-relevant metrics
        df["pct_pre1940"] = (df["B25034_011E"] / total * 100).round(2)
        df["pct_pre1950"] = ((df["B25034_010E"] + df["B25034_011E"]) / total * 100).round(2)
        df["pct_pre1960"] = ((df["B25034_009E"] + df["B25034_010E"] + df["B25034_011E"]) / total * 100).round(2)
        df["pct_pre1970"] = ((df["B25034_008E"] + df["B25034_009E"] + df["B25034_010E"] + df["B25034_011E"]) / total * 100).round(2)
        df["pct_pre1978"] = ((df["B25034_007E"] + df["B25034_008E"] + df["B25034_009E"] + df["B25034_010E"] + df["B25034_011E"]) / total * 100).round(2)
        # Lead paint was banned in 1978, so pre-1978 is the key threshold

        out = df[["county_fips", "NAME", "B25034_001E",
                  "pct_pre1940", "pct_pre1950", "pct_pre1960", "pct_pre1970", "pct_pre1978"]].copy()
        out.columns = ["county_fips", "name", "total_housing",
                       "pct_pre1940", "pct_pre1950", "pct_pre1960", "pct_pre1970", "pct_pre1978"]

        out.to_csv(f"{OUT}/census_housing_age_detail_county.csv", index=False)
        print(f"  Saved {len(out)} counties")
        print(f"  Mean pct pre-1978: {out['pct_pre1978'].mean():.1f}%")
        print(f"  Mean pct pre-1950: {out['pct_pre1950'].mean():.1f}%")
        print(f"  Mean pct pre-1940: {out['pct_pre1940'].mean():.1f}%")
except Exception as e:
    print(f"  Error: {e}")

# ============================================================================
# 2. Download data.cdc.gov Socrata datasets about blood lead
# ============================================================================
print("\n" + "="*70)
print("2. Searching Socrata for blood lead datasets across ALL domains")
print("="*70)

# Search across all Socrata domains for blood lead county data
search_urls = [
    "https://api.us.socrata.com/api/catalog/v1?q=childhood+blood+lead+county&limit=20",
    "https://api.us.socrata.com/api/catalog/v1?q=blood+lead+level+county+FIPS&limit=20",
    "https://api.us.socrata.com/api/catalog/v1?q=elevated+blood+lead+county&limit=20",
]

seen_ids = set()
for surl in search_urls:
    try:
        r = requests.get(surl, timeout=15)
        if r.status_code == 200:
            results = r.json().get("results", [])
            print(f"\n  Query: {surl.split('q=')[1].split('&')[0]}")
            print(f"  Found: {len(results)} datasets")

            for res in results:
                resource = res.get("resource", {})
                nid = resource.get("id", "")
                if nid in seen_ids:
                    continue
                seen_ids.add(nid)

                name = resource.get("name", "")
                domain = res.get("metadata", {}).get("domain", "")
                desc = resource.get("description", "")[:80]
                print(f"\n  [{nid}] {name}")
                print(f"    Domain: {domain}")
                print(f"    Desc: {desc}")

                # Try to download if it looks like county-level lead data
                if any(kw in (name + desc).lower() for kw in ['county', 'lead', 'blood lead']):
                    try:
                        csv_url = f"https://{domain}/api/views/{nid}/rows.csv?accessType=DOWNLOAD"
                        print(f"    Trying download: {csv_url}")
                        r2 = requests.get(csv_url, timeout=45)
                        if r2.status_code == 200 and len(r2.content) > 500:
                            df = pd.read_csv(io.StringIO(r2.text), nrows=5)
                            print(f"    Cols: {list(df.columns)[:8]}")
                            county_found = any('county' in c.lower() or 'fips' in c.lower() for c in df.columns)
                            lead_found = any('lead' in c.lower() or 'bll' in c.lower() for c in df.columns)
                            print(f"    County col: {county_found}, Lead col: {lead_found}")

                            if county_found or lead_found:
                                # Download full
                                df_full = pd.read_csv(io.StringIO(r2.text), low_memory=False)
                                safe = name[:35].replace(" ", "_").replace("/", "_").replace(":", "")
                                fname = f"socrata_{domain.split('.')[0]}_{safe}.csv"
                                df_full.to_csv(f"{OUT}/{fname}", index=False)
                                print(f"    SAVED: {fname} ({len(df_full)} rows)")
                    except Exception as e:
                        print(f"    Download error: {e}")
    except Exception as e:
        print(f"  Search error: {e}")


# ============================================================================
# 3. Try the CDC archived state surveillance data pages
# ============================================================================
print("\n" + "="*70)
print("3. CDC Archived State Blood Lead Surveillance (county-level data)")
print("="*70)

# The old CDC lead data pages have state-by-state Excel files
# Try archived URLs
import re

states_abbr = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas",
    "CA": "california", "CO": "colorado", "CT": "connecticut",
    "FL": "florida", "GA": "georgia", "ID": "idaho", "IL": "illinois",
    "IN": "indiana", "IA": "iowa", "KS": "kansas", "KY": "kentucky",
    "LA": "louisiana", "ME": "maine", "MD": "maryland", "MA": "massachusetts",
    "MI": "michigan", "MN": "minnesota", "MS": "mississippi", "MO": "missouri",
    "MT": "montana", "NE": "nebraska", "NV": "nevada", "NH": "new-hampshire",
    "NJ": "new-jersey", "NM": "new-mexico", "NY": "new-york",
    "NC": "north-carolina", "ND": "north-dakota", "OH": "ohio",
    "OK": "oklahoma", "OR": "oregon", "PA": "pennsylvania",
    "RI": "rhode-island", "SC": "south-carolina", "SD": "south-dakota",
    "TN": "tennessee", "TX": "texas", "UT": "utah", "VT": "vermont",
    "VA": "virginia", "WA": "washington", "WV": "west-virginia",
    "WI": "wisconsin", "WY": "wyoming"
}

# Try direct CDC data URLs
test_patterns = [
    "https://www.cdc.gov/lead-prevention/media/pdfs/data/{abbr}data.xlsx",
    "https://www.cdc.gov/lead-prevention/media/data-downloads/{abbr}data.xlsx",
    "https://www.cdc.gov/nceh/lead/data/tables/{abbr}.xlsx",
]

found_pattern = None
for pattern in test_patterns:
    for abbr in ["mo", "ny", "pa", "oh"]:
        url = pattern.format(abbr=abbr)
        try:
            r = requests.head(url, timeout=8, allow_redirects=True)
            if r.status_code == 200:
                print(f"  Found: {url} (size: {r.headers.get('content-length', '?')})")
                found_pattern = pattern
                break
        except:
            pass
    if found_pattern:
        break

if not found_pattern:
    print("  Direct file patterns not found; CDC uses dynamic loading")
    print("  Will need to use CDC Data Explorer or contact leadsurv@cdc.gov")


# ============================================================================
# 4. Kaggle CDC Blood Lead Surveillance (state-level, but useful reference)
# ============================================================================
print("\n" + "="*70)
print("4. Checking Kaggle CDC Blood Lead Surveillance")
print("="*70)

# The Kaggle dataset is state-level, but let's check if there are alternatives
kaggle_url = "https://www.kaggle.com/api/v1/datasets/download/cdc/childhood-blood-lead-surveillance"
print("  Kaggle requires authentication; skipping automated download")
print("  Dataset is state-level only (1997-2015)")


# ============================================================================
# 5. Process existing data files for county-level info
# ============================================================================
print("\n" + "="*70)
print("5. Checking ALL existing data files for county-level lead info")
print("="*70)

import glob

all_csvs = sorted(glob.glob(f"{OUT}/*.csv"))
county_data_files = []

for fpath in all_csvs:
    try:
        df = pd.read_csv(fpath, nrows=5, low_memory=False)
        cols = list(df.columns)
        has_county = any('county' in c.lower() or 'fips' in c.lower() for c in cols)
        has_lead = any('lead' in c.lower() or 'bll' in c.lower() or 'pb' in c.lower()
                      or 'elevated' in c.lower() for c in cols)

        if has_county and has_lead:
            fname = os.path.basename(fpath)
            county_data_files.append(fname)
            full_df = pd.read_csv(fpath, low_memory=False)
            county_col = [c for c in full_df.columns if 'county' in c.lower() or 'fips' in c.lower()][0]
            n = full_df[county_col].nunique()
            print(f"  COUNTY+LEAD: {fname} ({len(full_df)} rows, {n} unique counties/FIPS)")
    except:
        pass

print(f"\n  Found {len(county_data_files)} files with both county and lead data")


# ============================================================================
# 6. EPA EJSCREEN Lead Paint EJ Index (used in hotspot paper)
# ============================================================================
print("\n" + "="*70)
print("6. EPA EJSCREEN Lead Paint Data")
print("="*70)

# EJSCREEN data by census tract
# Try the EPA EJSCREEN data download
ejscreen_url = "https://gaftp.epa.gov/EJSCREEN/2023/EJSCREEN_2023_BG_with_AS_CNMI_GU_VI.csv.zip"
print(f"  EJSCREEN dataset is very large (~1GB); skipping full download")
print(f"  URL: {ejscreen_url}")
print(f"  The EJSCREEN lead paint indicator (DSLPM) is included in the EPA hotspots RF model")
print(f"  Our epa_predicted_county_blood_lead.csv already captures this signal")


# ============================================================================
# 7. Comprehensive county-level master merge
# ============================================================================
print("\n" + "="*70)
print("7. BUILDING COMPREHENSIVE COUNTY MASTER DATASET")
print("="*70)

# Start with housing age data
housing = pd.read_csv(f"{OUT}/census_housing_age_detail_county.csv")
print(f"  Base: {len(housing)} counties from housing age data")

# EPA predicted BLL
epa_file = f"{OUT}/epa_predicted_county_blood_lead.csv"
if os.path.exists(epa_file):
    epa = pd.read_csv(epa_file)
    housing = housing.merge(epa, on="county_fips", how="left")
    print(f"  + EPA predicted BLL: matched {housing['mean_predicted_pct_elevated_bll'].notna().sum()} counties")

# Census demographics
demo_file = f"{OUT}/census_demographics_for_lead.csv"
if os.path.exists(demo_file):
    demo = pd.read_csv(demo_file)
    housing = housing.merge(demo, on="county_fips", how="left", suffixes=("", "_demo"))
    print(f"  + Demographics: {len(housing)} rows")

# Census children
child_file = f"{OUT}/census_children_under6_by_county.csv"
if os.path.exists(child_file):
    child = pd.read_csv(child_file)
    child_cols = [c for c in child.columns if c not in housing.columns or c == "county_fips"]
    housing = housing.merge(child[child_cols], on="county_fips", how="left")
    print(f"  + Children under 5: added")

# Census pre-1980 housing (from main script)
pre1980_file = f"{OUT}/census_pre1980_housing_lead_proxy.csv"
if os.path.exists(pre1980_file):
    pre1980 = pd.read_csv(pre1980_file)
    new_cols = [c for c in pre1980.columns if c not in housing.columns or c == "county_fips"]
    if len(new_cols) > 1:
        housing = housing.merge(pre1980[new_cols], on="county_fips", how="left")
        print(f"  + Pre-1980 housing proxy: added {len(new_cols)-1} cols")

# USGS soil lead
usgs_file = f"{OUT}/usgs_soil_lead_concentrations.csv"
if os.path.exists(usgs_file):
    usgs = pd.read_csv(usgs_file, low_memory=False)
    # Check for lat/lon columns to aggregate to county
    lat_cols = [c for c in usgs.columns if 'lat' in c.lower()]
    lon_cols = [c for c in usgs.columns if 'lon' in c.lower() or 'long' in c.lower()]
    print(f"  USGS: {len(usgs)} samples, lat cols: {lat_cols}, lon cols: {lon_cols}")

# Michigan observed data
for suffix in ['MI_07_11', 'MI_14_16', 'OH_07_11', 'OH_14_16']:
    mi_file = f"{OUT}/epa_observed_bll_{suffix}_county.csv"
    if os.path.exists(mi_file):
        mi = pd.read_csv(mi_file)
        mi_rename = {c: f"{c}_{suffix}" for c in mi.columns if c != "county_fips"}
        mi = mi.rename(columns=mi_rename)
        housing = housing.merge(mi, on="county_fips", how="left")
        matched = housing[f"mean_observed_pct_elevated_{suffix}"].notna().sum()
        print(f"  + Observed BLL {suffix}: {matched} counties matched")

# NY aggregated data
ny_file = f"{OUT}/ny_county_blood_lead_aggregated.csv"
if os.path.exists(ny_file):
    ny = pd.read_csv(ny_file)
    # Get the most recent year's data
    if 'year' in ny.columns:
        ny_latest = ny[ny['year'] == ny['year'].max()].copy()
        ny_latest = ny_latest.rename(columns={
            'pct_elevated': 'ny_pct_elevated_bll',
            'tests': 'ny_tests',
            'total_eblls': 'ny_total_elevated'
        })
        # Need to match NY FIPS to county_fips (NY FIPS is just the county part)
        ny_latest['county_fips'] = '36' + ny_latest['fips'].astype(str).str.zfill(3)
        merge_cols = ['county_fips', 'ny_pct_elevated_bll', 'ny_tests', 'ny_total_elevated']
        merge_cols = [c for c in merge_cols if c in ny_latest.columns]
        if merge_cols:
            housing = housing.merge(ny_latest[merge_cols], on="county_fips", how="left")
            matched = housing['ny_pct_elevated_bll'].notna().sum()
            print(f"  + NY blood lead: {matched} counties matched")

# Existing county dataset
real_file = f"{OUT}/real_county_dataset.csv"
if os.path.exists(real_file):
    real = pd.read_csv(real_file, low_memory=False)
    # Try to find county FIPS
    fips_cols = [c for c in real.columns if 'fips' in c.lower()]
    suicide_cols = [c for c in real.columns if 'suicide' in c.lower() or 'death' in c.lower()]
    print(f"  real_county_dataset.csv: {len(real)} rows, FIPS cols: {fips_cols}, suicide cols: {suicide_cols}")

    if fips_cols:
        fips_col = fips_cols[0]
        real['county_fips'] = real[fips_col].astype(str).str.zfill(5)
        housing['county_fips'] = housing['county_fips'].astype(str).str.zfill(5)
        # Get suicide and other key cols that aren't already in housing
        useful_cols = ['county_fips'] + [c for c in real.columns
                       if c not in housing.columns and c != fips_col
                       and any(kw in c.lower() for kw in ['suicide', 'gun', 'firearm', 'death',
                                                           'veteran', 'rural', 'mining', 'elevation'])]
        if len(useful_cols) > 1:
            housing = housing.merge(real[useful_cols], on="county_fips", how="left")
            print(f"  + Real county data: added {len(useful_cols)-1} columns")

# Save final master
fname = f"{OUT}/county_lead_exposure_master.csv"
housing.to_csv(fname, index=False)
print(f"\n  MASTER DATASET SAVED: {fname}")
print(f"  Shape: {housing.shape}")
print(f"  Columns:")
for i, c in enumerate(housing.columns):
    print(f"    {i+1}. {c}")

# Quick summary stats on key lead variables
print("\n  KEY LEAD VARIABLE SUMMARY:")
lead_vars = [c for c in housing.columns if any(kw in c.lower()
             for kw in ['lead', 'bll', 'elevated', 'pre19', 'pb'])]
for lv in lead_vars:
    try:
        vals = pd.to_numeric(housing[lv], errors='coerce')
        if vals.notna().sum() > 0:
            print(f"    {lv}: n={vals.notna().sum()}, mean={vals.mean():.4f}, "
                  f"median={vals.median():.4f}, std={vals.std():.4f}")
    except:
        pass


# ============================================================================
# 8. Summary report
# ============================================================================
print("\n" + "="*70)
print("FINAL COUNTY-LEVEL LEAD DATA INVENTORY")
print("="*70)

inventory = [
    ("epa_predicted_county_blood_lead.csv",
     "EPA Random Forest predicted % children with elevated BLL",
     "County FIPS", "3,140 counties", "Model based on 2007-2016 data",
     "mean_predicted_pct_elevated_bll (RF v1), mean_predicted_pct_elevated_bll_v2 (RF v2)"),

    ("epa_observed_bll_MI_07_11_county.csv",
     "Michigan OBSERVED % children with elevated BLL 2007-2011",
     "County FIPS", "71 MI counties", "2007-2011",
     "mean_observed_pct_elevated"),

    ("epa_observed_bll_MI_14_16_county.csv",
     "Michigan OBSERVED % children with elevated BLL 2014-2016",
     "County FIPS", "71 MI counties", "2014-2016",
     "mean_observed_pct_elevated"),

    ("epa_observed_bll_OH_07_11_county.csv",
     "Ohio OBSERVED % children with elevated BLL 2007-2011",
     "County FIPS", "88 OH counties", "2007-2011",
     "mean_observed_pct_elevated"),

    ("epa_observed_bll_OH_14_16_county.csv",
     "Ohio OBSERVED % children with elevated BLL 2014-2016",
     "County FIPS", "88 OH counties", "2014-2016",
     "mean_observed_pct_elevated"),

    ("census_housing_age_detail_county.csv",
     "Census housing age (lead paint proxy) - detailed bands",
     "County FIPS", "3,221 counties", "ACS 2022 5yr",
     "pct_pre1940, pct_pre1950, pct_pre1960, pct_pre1978"),

    ("ny_county_blood_lead_aggregated.csv",
     "New York State county-level blood lead (aggregated from ZIP)",
     "County name + FIPS", "~57 NY counties", "2000-2020",
     "pct_elevated (% children with elevated BLL)"),

    ("michigan_BLL_under6_county_2016.csv",
     "Michigan county blood lead levels, children under 6",
     "County name", "85 MI counties", "2016",
     "all_greater5_total, all_greater5_percent"),

    ("michigan_BLL_under6_county_2015.csv",
     "Michigan county blood lead levels, children under 6",
     "County name", "85 MI counties", "2015",
     "all_greater5_total, all_greater5_percent"),

    ("michigan_BLL_under6_county_2014.csv",
     "Michigan county blood lead levels, children under 6",
     "County name", "85 MI counties", "2014",
     "total_n, all_greater5_total, all_greater5_percent"),

    ("usgs_soil_lead_concentrations.csv",
     "USGS soil lead concentrations (point data)",
     "Lat/Lon (needs geocoding to county)", "~14,000 sample points", "2007-2013",
     "Soil Pb concentration (ppm)"),

    ("county_lead_exposure_master.csv",
     "MERGED MASTER: All county-level lead indicators combined",
     "County FIPS", "3,221 counties", "Various",
     "Multiple lead exposure indicators"),
]

for fname, desc, geo, coverage, period, key_var in inventory:
    fpath = f"{OUT}/{fname}"
    exists = "EXISTS" if os.path.exists(fpath) else "MISSING"
    size = f"{os.path.getsize(fpath)/1024:.0f} KB" if os.path.exists(fpath) else "-"
    print(f"\n  {exists} | {fname}")
    print(f"    {desc}")
    print(f"    Geo: {geo} | Coverage: {coverage} | Period: {period}")
    print(f"    Key var: {key_var}")
    print(f"    Size: {size}")

print("\n\nDONE!")
