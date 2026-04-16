#!/usr/bin/env python3
"""
Process and aggregate environmental data already downloaded,
and fetch remaining data (Census, CDC).
"""

import requests
import pandas as pd
import io
import time
import os
import json

OUTPUT_DIR = "/Users/bobbarclay/Documents/soldiers"
import os as _os
CENSUS_API_KEY = _os.environ.get("CENSUS_API_KEY")
if not CENSUS_API_KEY:
    raise RuntimeError("Set CENSUS_API_KEY in your environment (see .env.example). Get a key free from https://api.census.gov/data/key_signup.html")

STATE_FIPS = {
    '01':'AL','02':'AK','04':'AZ','05':'AR','06':'CA','08':'CO','09':'CT',
    '10':'DE','11':'DC','12':'FL','13':'GA','15':'HI','16':'ID','17':'IL',
    '18':'IN','19':'IA','20':'KS','21':'KY','22':'LA','23':'ME','24':'MD',
    '25':'MA','26':'MI','27':'MN','28':'MS','29':'MO','30':'MT','31':'NE',
    '32':'NV','33':'NH','34':'NJ','35':'NM','36':'NY','37':'NC','38':'ND',
    '39':'OH','40':'OK','41':'OR','42':'PA','44':'RI','45':'SC','46':'SD',
    '47':'TN','48':'TX','49':'UT','50':'VT','51':'VA','53':'WA','54':'WV',
    '55':'WI','56':'WY','60':'AS','66':'GU','69':'MP','72':'PR','78':'VI'
}

STATE_ABBREV_TO_FIPS = {v: k for k, v in STATE_FIPS.items()}


# ============================================================
# 1. Process EPA Superfund raw data -> county aggregation
# ============================================================
def process_superfund():
    print("\n" + "="*70)
    print("1. AGGREGATING EPA SUPERFUND BY COUNTY")
    print("="*70)

    raw_path = os.path.join(OUTPUT_DIR, "epa_superfund_raw.csv")
    if not os.path.exists(raw_path):
        print("  No raw Superfund data found. Skipping.")
        return False

    df = pd.read_csv(raw_path, low_memory=False)
    print(f"  Loaded {len(df)} SEMS records")

    # Use state_code and county_name columns
    # state_code is 2-letter abbreviation (KS, MT, etc.)
    # county_name is the county name
    # pgm_sys_id is the unique site identifier

    # Drop rows with missing state or county
    df_valid = df.dropna(subset=['state_code']).copy()
    df_valid['county_name'] = df_valid['county_name'].fillna('UNKNOWN')
    df_valid['std_county_fips'] = df_valid['std_county_fips'].astype(str).str.strip()

    print(f"  Records with state: {len(df_valid)}")

    # Count unique sites per state/county
    county_agg = df_valid.groupby(['state_code', 'county_name']).agg(
        superfund_site_count=('pgm_sys_id', 'nunique'),
        std_county_fips=('std_county_fips', 'first'),
    ).reset_index()

    # Remove "NOT DEFINED" and "UNKNOWN" counties
    county_agg = county_agg[~county_agg['county_name'].isin(['NOT DEFINED', 'UNKNOWN', ''])]

    out_path = os.path.join(OUTPUT_DIR, "epa_superfund_by_county.csv")
    county_agg.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")
    print(f"  Counties with Superfund sites: {len(county_agg)}")
    print(f"\n  Top 20 counties:")
    print(county_agg.nlargest(20, 'superfund_site_count').to_string(index=False))
    print(f"\n  Sites per state (top 10):")
    state_totals = county_agg.groupby('state_code')['superfund_site_count'].sum().sort_values(ascending=False)
    print(state_totals.head(10).to_string())
    return True


# ============================================================
# 2. Process EPA Lead Violations + LCR data
# ============================================================
def process_lead_violations():
    print("\n" + "="*70)
    print("2. AGGREGATING EPA LEAD VIOLATIONS BY STATE/COUNTY")
    print("="*70)

    viol_path = os.path.join(OUTPUT_DIR, "epa_lead_violations_raw.csv")
    lcr_path = os.path.join(OUTPUT_DIR, "epa_lcr_lead_samples_raw.csv")

    results = []

    # Process violations (contaminant 2950 = lead)
    if os.path.exists(viol_path):
        df_viol = pd.read_csv(viol_path, low_memory=False)
        print(f"  Loaded {len(df_viol)} lead violation records")

        # Map primacy_agency_code (state FIPS as integer) to state abbreviation
        df_viol['state_fips'] = df_viol['primacy_agency_code'].astype(str).str.zfill(2)
        df_viol['state_abbrev'] = df_viol['state_fips'].map(STATE_FIPS)

        # Deduplicate - same violation can appear multiple times for different facilities
        df_viol_dedup = df_viol.drop_duplicates(subset=['pwsid', 'violation_id'])
        print(f"  Unique violations: {len(df_viol_dedup)}")

        # Extract county from PWSID if possible (first 2 chars = state FIPS)
        # PWSID format varies; use state_fips from primacy_agency_code
        state_viol = df_viol_dedup.groupby(['state_abbrev', 'state_fips']).agg(
            lead_violation_count=('violation_id', 'nunique'),
            systems_with_violations=('pwsid', 'nunique'),
            total_pop_affected=('population_served_count', 'sum'),
            health_based_violations=('is_health_based_ind', lambda x: (x == 'Y').sum()),
        ).reset_index()

        results.append(('violations', state_viol))
        print(f"\n  State-level violation summary:")
        print(state_viol.sort_values('lead_violation_count', ascending=False).head(15).to_string(index=False))

    # Process LCR samples (PB90 = 90th percentile lead)
    if os.path.exists(lcr_path):
        df_lcr = pd.read_csv(lcr_path, low_memory=False)
        print(f"\n  Loaded {len(df_lcr)} LCR lead sample records")

        df_lcr['state_fips'] = df_lcr['primacy_agency_code'].astype(str).str.zfill(2)
        df_lcr['state_abbrev'] = df_lcr['state_fips'].map(STATE_FIPS)

        # Deduplicate
        df_lcr_dedup = df_lcr.drop_duplicates(subset=['pwsid', 'sample_id', 'sar_id'])
        print(f"  Unique LCR samples: {len(df_lcr_dedup)}")

        # Action level for lead is 0.015 mg/L (15 ppb)
        ACTION_LEVEL = 0.015
        df_lcr_dedup['exceeds_action_level'] = df_lcr_dedup['sample_measure'] > ACTION_LEVEL

        state_lcr = df_lcr_dedup.groupby(['state_abbrev', 'state_fips']).agg(
            lcr_sample_count=('sar_id', 'count'),
            systems_sampled=('pwsid', 'nunique'),
            mean_lead_level_mg_L=('sample_measure', 'mean'),
            median_lead_level_mg_L=('sample_measure', 'median'),
            max_lead_level_mg_L=('sample_measure', 'max'),
            pct_exceeding_action_level=('exceeds_action_level', 'mean'),
        ).reset_index()
        state_lcr['pct_exceeding_action_level'] = (state_lcr['pct_exceeding_action_level'] * 100).round(2)
        state_lcr['mean_lead_level_mg_L'] = state_lcr['mean_lead_level_mg_L'].round(6)
        state_lcr['median_lead_level_mg_L'] = state_lcr['median_lead_level_mg_L'].round(6)

        results.append(('lcr', state_lcr))
        print(f"\n  State-level LCR summary (highest mean lead levels):")
        print(state_lcr.sort_values('mean_lead_level_mg_L', ascending=False).head(15).to_string(index=False))

    # Merge and save
    if results:
        if len(results) == 2:
            merged = results[0][1].merge(results[1][1], on=['state_abbrev', 'state_fips'], how='outer')
        else:
            merged = results[0][1]

        out_path = os.path.join(OUTPUT_DIR, "epa_lead_water_violations.csv")
        merged.to_csv(out_path, index=False)
        print(f"\n  Saved combined lead data to {out_path}")
        print(f"  States in dataset: {len(merged)}")
        return True

    return False


# ============================================================
# 3. CDC Blood Lead Surveillance
# ============================================================
def pull_cdc_blood_lead():
    print("\n" + "="*70)
    print("3. CDC BLOOD LEAD SURVEILLANCE")
    print("="*70)

    # The CDC EPHT DataExplorer is down. Try the API directly with known measure IDs.
    base_url = "https://ephtracking.cdc.gov/apigateway/api/v1"

    # Try getting content areas to confirm API is working
    print("  Checking CDC EPHT API status...")
    resp = requests.get(f"{base_url}/contentareas/json", timeout=15,
                       headers={'User-Agent': 'Mozilla/5.0 (research)'})
    if resp.status_code != 200:
        print(f"  API returned {resp.status_code}. May be down.")
    else:
        print("  API responding.")

    # Try known measure IDs for childhood lead
    # From CDC documentation, common lead measures:
    # Attempt various IDs in the range used for lead content area
    success = False

    # Use a focused set of likely measure IDs
    for mid in range(556, 570):
        for geo_type, geo_label in [(1, "state"), (2, "county")]:
            url = f"{base_url}/getCoreHolder/{mid}/1/{geo_type}/0/0/json"
            try:
                resp = requests.get(url, timeout=15,
                                   headers={'User-Agent': 'Mozilla/5.0 (research)'})
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list) and len(data) > 5:
                        df = pd.json_normalize(data)
                        print(f"  Measure {mid} at {geo_label} level: {len(df)} records!")
                        print(f"  Columns: {list(df.columns)[:10]}")

                        out_file = f"cdc_blood_lead_{geo_label}_m{mid}.csv"
                        out_path = os.path.join(OUTPUT_DIR, out_file)
                        df.to_csv(out_path, index=False)
                        print(f"  Saved to {out_path}")

                        # Also save as main file
                        main_path = os.path.join(OUTPUT_DIR, "cdc_blood_lead_surveillance.csv")
                        df.to_csv(main_path, index=False)
                        print(df.head(5).to_string())
                        success = True

                        if geo_type == 2:  # County level is best
                            return True
            except Exception as e:
                pass
            time.sleep(0.5)

        if success:
            break

    if not success:
        # Try the Connecticut state-level data from Socrata as a fallback
        print("\n  CDC EPHT API did not return lead data.")
        print("  Trying state-specific datasets from data.cdc.gov...")

        # Connecticut childhood lead dataset
        url = "https://data.cdc.gov/resource/m5fk-6xgv.csv?$limit=50000"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.text) > 200 and '"error"' not in resp.text[:100]:
                df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
                if len(df) > 5:
                    print(f"  CT Lead Poisoning data: {len(df)} rows")
                    print(f"  Columns: {list(df.columns)}")
                    out_path = os.path.join(OUTPUT_DIR, "cdc_blood_lead_surveillance.csv")
                    df.to_csv(out_path, index=False)
                    print(f"  Saved to {out_path} (Connecticut only)")
                    print(df.head(3).to_string())
                    success = True
        except:
            pass

    if not success:
        print("\n  COULD NOT RETRIEVE NATIONAL BLOOD LEAD DATA.")
        print("  The CDC Environmental Public Health Tracking Network")
        print("  DataExplorer is currently down for maintenance.")
        print("\n  MANUAL STEPS WHEN AVAILABLE:")
        print("  1. Go to https://ephtracking.cdc.gov/DataExplorer/")
        print("  2. Content area: 'Childhood Lead Poisoning'")
        print("  3. Measure: '% children with elevated BLL (>=5 ug/dL)'")
        print("  4. Geography: County level, all states")
        print("  5. Time period: most recent available")
        print("  6. Download CSV")
        print("\n  Alternative sources:")
        print("  - https://www.cdc.gov/lead-prevention/data-research/index.html")
        print("  - Individual state health department websites")

        # Create a placeholder with metadata
        placeholder = pd.DataFrame({
            'source': ['CDC EPHT Tracking Network'],
            'status': ['DataExplorer down for maintenance as of 2026-04-06'],
            'url': ['https://ephtracking.cdc.gov/DataExplorer/'],
            'content_area': ['Childhood Lead Poisoning (ID: 6)'],
            'note': ['Re-run when available or download manually']
        })
        out_path = os.path.join(OUTPUT_DIR, "cdc_blood_lead_surveillance.csv")
        placeholder.to_csv(out_path, index=False)
        print(f"  Created placeholder: {out_path}")

    return success


# ============================================================
# 4. Census Pre-1978 Housing
# ============================================================
def pull_census_housing():
    print("\n" + "="*70)
    print("4. CENSUS PRE-1978 HOUSING (ACS B25034)")
    print("="*70)

    # 2022 ACS 5-year B25034 (Year Structure Built):
    # _001E = Total
    # _007E = Built 1970 to 1979
    # _008E = Built 1960 to 1969
    # _009E = Built 1950 to 1959
    # _010E = Built 1940 to 1949
    # _011E = Built 1939 or earlier

    variables = "NAME,B25034_001E,B25034_007E,B25034_008E,B25034_009E,B25034_010E,B25034_011E"

    for year in [2022, 2021, 2020]:
        url = f"https://api.census.gov/data/{year}/acs/acs5"
        params = {
            'get': variables,
            'for': 'county:*',
            'key': CENSUS_API_KEY
        }

        print(f"  Trying Census ACS {year} 5-year...")
        try:
            resp = requests.get(url, params=params, timeout=120,
                              headers={'User-Agent': 'Mozilla/5.0'})
        except Exception as e:
            print(f"  Request failed: {e}")
            continue

        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
            continue

        try:
            data = resp.json()
        except:
            print(f"  Could not parse JSON: {resp.text[:200]}")
            continue

        if not isinstance(data, list) or len(data) < 2:
            print(f"  Unexpected format: {str(data)[:200]}")
            continue

        headers = data[0]
        rows = data[1:]
        df = pd.DataFrame(rows, columns=headers)
        print(f"  Got {len(df)} counties from {year} ACS")

        # Convert numeric
        num_cols = [c for c in df.columns if c.startswith('B25034')]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Pre-1978 proxy = 1970-1979 + 1960-1969 + 1950-1959 + 1940-1949 + pre-1939
        df['pre_1980_units'] = (df['B25034_007E'].fillna(0) + df['B25034_008E'].fillna(0) +
                                df['B25034_009E'].fillna(0) + df['B25034_010E'].fillna(0) +
                                df['B25034_011E'].fillna(0))
        df['total_units'] = df['B25034_001E']
        df['pct_pre1978_housing'] = (df['pre_1980_units'] / df['total_units'] * 100).round(2)

        df['county_fips'] = df['state'] + df['county']
        df['state_name'] = df['NAME'].apply(lambda x: x.split(', ')[-1] if ', ' in str(x) else '')
        df['county_name'] = df['NAME'].apply(lambda x: x.split(', ')[0] if ', ' in str(x) else x)

        output_df = df[['county_fips', 'state', 'county', 'state_name', 'county_name',
                        'total_units', 'pre_1980_units', 'pct_pre1978_housing',
                        'B25034_007E', 'B25034_008E', 'B25034_009E',
                        'B25034_010E', 'B25034_011E']].copy()
        output_df.columns = ['county_fips', 'state_fips', 'county_code', 'state_name',
                            'county_name', 'total_housing_units', 'pre_1980_units',
                            'pct_pre1978_housing', 'built_1970_1979', 'built_1960_1969',
                            'built_1950_1959', 'built_1940_1949', 'built_pre_1939']

        out_path = os.path.join(OUTPUT_DIR, "census_pre1978_housing_by_county.csv")
        output_df.to_csv(out_path, index=False)
        print(f"  Saved to {out_path}")
        print(f"  ACS year: {year}")
        print(f"  Counties: {len(output_df)}")
        valid = output_df['pct_pre1978_housing'].dropna()
        print(f"  Mean pct pre-1978 housing: {valid.mean():.1f}%")
        print(f"  Median: {valid.median():.1f}%")
        print(f"  Max: {valid.max():.1f}%")
        print(f"\n  Top 15 counties by pre-1978 housing:")
        top = output_df.nlargest(15, 'pct_pre1978_housing')
        print(top[['county_fips','county_name','state_name','pct_pre1978_housing',
                    'total_housing_units']].to_string(index=False))
        return True

    print("  FAILED to get Census data.")
    return False


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("="*70)
    print("ENVIRONMENTAL DATA PROCESSING & FETCHING")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    results = {}

    for name, func in [
        ('superfund', process_superfund),
        ('lead_violations', process_lead_violations),
        ('cdc_blood_lead', pull_cdc_blood_lead),
        ('census_housing', pull_census_housing),
    ]:
        try:
            results[name] = func()
        except Exception as e:
            print(f"\n  FATAL ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # USGS mining was already done successfully
    usgs_path = os.path.join(OUTPUT_DIR, "usgs_mining_sites_by_county.csv")
    if os.path.exists(usgs_path) and os.path.getsize(usgs_path) > 1000:
        results['usgs_mining'] = True
        df_usgs = pd.read_csv(usgs_path)
        print(f"\n  USGS Mining: Already complete ({len(df_usgs)} counties)")
    else:
        results['usgs_mining'] = False

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    for source, status in results.items():
        symbol = "OK" if status else "NEEDS MANUAL"
        print(f"  [{symbol}] {source}")

    print(f"\nOutput files:")
    keywords = ['superfund', 'lead', 'blood', 'housing', 'mining', 'epa', 'cdc', 'census', 'usgs', 'lcr']
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.csv') and any(kw in f.lower() for kw in keywords):
            fpath = os.path.join(OUTPUT_DIR, f)
            size = os.path.getsize(fpath)
            try:
                row_count = sum(1 for _ in open(fpath)) - 1
                print(f"  {f}: {size/1024:.1f} KB ({row_count:,} rows)")
            except:
                print(f"  {f}: {size/1024:.1f} KB")

    print(f"\nDone: {time.strftime('%Y-%m-%d %H:%M:%S')}")
