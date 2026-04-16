#!/usr/bin/env python3
"""
Pull county-level lead contamination and environmental data from federal sources.
Sources:
  1. EPA Superfund Sites (via FRS_PROGRAM_FACILITY/SEMS)
  2. EPA Lead and Copper Rule violations (SDWIS VIOLATION + LCR_SAMPLE_RESULT)
  3. CDC Blood Lead Surveillance (EPHT Tracking)
  4. Census Pre-1978 Housing (ACS B25034)
  5. USGS Mineral Resources Data System (MRDS)
"""

import requests
import pandas as pd
import io
import time
import os
import json
import sys
import zipfile

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

def safe_request(url, params=None, timeout=120, stream=False, retries=3):
    """Make a request with retries and error handling."""
    headers = {'User-Agent': 'Mozilla/5.0 (research data pull)'}
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers,
                              timeout=timeout, stream=stream)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
    return None


def fetch_epa_paginated(base_url, batch_size=10000, max_rows=300000, pause=1.5):
    """Fetch paginated data from EPA Envirofacts API."""
    all_data = []
    start_row = 0
    while start_row < max_rows:
        end_row = start_row + batch_size - 1
        url = f"{base_url}/ROWS/{start_row}:{end_row}/CSV"
        print(f"  Fetching rows {start_row}-{end_row}...")
        resp = safe_request(url, timeout=180)
        if resp is None or len(resp.text.strip()) < 50:
            print(f"  No more data at row {start_row}.")
            break
        try:
            chunk = pd.read_csv(io.StringIO(resp.text), low_memory=False)
            if len(chunk) == 0:
                break
            all_data.append(chunk)
            print(f"    Got {len(chunk)} rows")
            start_row += batch_size
            if len(chunk) < batch_size:
                break
            time.sleep(pause)
        except Exception as e:
            print(f"  Error parsing CSV: {e}")
            print(f"  Response preview: {resp.text[:300]}")
            break
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None


# ============================================================
# 1. EPA SUPERFUND SITES (SEMS via FRS_PROGRAM_FACILITY)
# ============================================================
def pull_epa_superfund():
    print("\n" + "="*70)
    print("1. EPA SUPERFUND SITES (FRS_PROGRAM_FACILITY/SEMS)")
    print("="*70)

    # The FRS_PROGRAM_FACILITY table has SEMS (Superfund) sites when filtered
    # by PGM_SYS_ACRNM = SEMS. ~16,000 records.
    base_url = "https://data.epa.gov/efservice/FRS_PROGRAM_FACILITY/PGM_SYS_ACRNM/SEMS"
    df = fetch_epa_paginated(base_url, batch_size=10000, max_rows=50000)

    if df is not None and len(df) > 0:
        print(f"\n  Total SEMS/Superfund records: {len(df)}")
        print(f"  Columns: {list(df.columns)}")

        # Save raw
        raw_path = os.path.join(OUTPUT_DIR, "epa_superfund_raw.csv")
        df.to_csv(raw_path, index=False)
        print(f"  Saved raw data to {raw_path}")

        # Key columns: state_code (or state_name), county_name, std_county_fips,
        # primary_name, pgm_sys_id
        # Identify columns (case-insensitive)
        col_map = {c.lower(): c for c in df.columns}

        state_col = col_map.get('state_name') or col_map.get('state_code')
        county_col = col_map.get('county_name') or col_map.get('std_county_name')
        fips_col = col_map.get('std_county_fips') or col_map.get('fips_code')
        site_col = col_map.get('primary_name') or col_map.get('pgm_sys_id')
        state_code_col = col_map.get('state_code')

        print(f"  Using: state={state_col}, county={county_col}, fips={fips_col}, site={site_col}")

        if county_col and state_col:
            # Remove rows with missing county
            df_valid = df.dropna(subset=[county_col])

            # Count unique sites per county (use pgm_sys_id as site identifier)
            id_col = col_map.get('pgm_sys_id', site_col)
            county_agg = df_valid.groupby([state_col, county_col]).agg(
                superfund_site_count=(id_col, 'nunique') if id_col else (county_col, 'count'),
            ).reset_index()

            # Add FIPS if available
            if fips_col:
                fips_map = df_valid.groupby([state_col, county_col])[fips_col].first().reset_index()
                county_agg = county_agg.merge(fips_map, on=[state_col, county_col], how='left')

            # Add state code
            if state_code_col and state_code_col != state_col:
                sc_map = df_valid.groupby([state_col])[state_code_col].first().reset_index()
                county_agg = county_agg.merge(sc_map, on=[state_col], how='left')

            out_path = os.path.join(OUTPUT_DIR, "epa_superfund_by_county.csv")
            county_agg.to_csv(out_path, index=False)
            print(f"\n  Saved county-level Superfund data: {out_path}")
            print(f"  Counties with Superfund sites: {len(county_agg)}")
            print(f"\n  Top 15 counties by Superfund site count:")
            print(county_agg.nlargest(15, 'superfund_site_count').to_string(index=False))
            return True
        else:
            out_path = os.path.join(OUTPUT_DIR, "epa_superfund_by_county.csv")
            df.to_csv(out_path, index=False)
            print(f"  Could not identify county columns. Saved full dataset to {out_path}")
            return True
    else:
        print("  WARNING: Could not retrieve SEMS data from Envirofacts.")
        print("  MANUAL FALLBACK:")
        print("    - https://www.epa.gov/superfund/search-superfund-sites-where-you-live")
        print("    - https://cumulis.epa.gov/supercpad/cursites/srchsites.cfm")
        out_path = os.path.join(OUTPUT_DIR, "epa_superfund_by_county.csv")
        pd.DataFrame(columns=['state_name','county_name','superfund_site_count']).to_csv(out_path, index=False)
        return False


# ============================================================
# 2. EPA LEAD AND COPPER RULE VIOLATIONS (SDWIS)
# ============================================================
def pull_epa_lead_violations():
    print("\n" + "="*70)
    print("2. EPA LEAD AND COPPER RULE VIOLATIONS (SDWIS)")
    print("="*70)

    # Two data sources:
    # A) VIOLATION table with contaminant_code=2950 (Lead) - ~114K records
    # B) LCR_SAMPLE_RESULT with contaminant_code=PB90 - ~262K records (90th percentile lead samples)
    # We'll pull both.

    # --- Part A: Lead violations (contaminant 2950) ---
    print("\n  [A] Pulling VIOLATION records for lead (contaminant_code=2950)...")
    base_url = "https://data.epa.gov/efservice/VIOLATION/CONTAMINANT_CODE/2950"
    df_viol = fetch_epa_paginated(base_url, batch_size=10000, max_rows=200000)

    if df_viol is not None and len(df_viol) > 0:
        print(f"  Total lead violation records: {len(df_viol)}")
        print(f"  Columns: {list(df_viol.columns)}")

        # Save raw violations
        raw_path = os.path.join(OUTPUT_DIR, "epa_lead_violations_raw.csv")
        df_viol.to_csv(raw_path, index=False)
        print(f"  Saved raw violations to {raw_path}")
    else:
        print("  No violation records retrieved for contaminant 2950.")
        df_viol = pd.DataFrame()

    # --- Part B: LCR sample results (PB90) ---
    print("\n  [B] Pulling LCR_SAMPLE_RESULT records for PB90...")
    base_url_lcr = "https://data.epa.gov/efservice/LCR_SAMPLE_RESULT/CONTAMINANT_CODE/PB90"
    df_lcr = fetch_epa_paginated(base_url_lcr, batch_size=10000, max_rows=300000)

    if df_lcr is not None and len(df_lcr) > 0:
        print(f"  Total LCR PB90 sample records: {len(df_lcr)}")
        print(f"  Columns: {list(df_lcr.columns)}")

        lcr_path = os.path.join(OUTPUT_DIR, "epa_lcr_lead_samples_raw.csv")
        df_lcr.to_csv(lcr_path, index=False)
        print(f"  Saved LCR samples to {lcr_path}")
    else:
        print("  No LCR sample records retrieved.")
        df_lcr = pd.DataFrame()

    # --- Now we need to join with WATER_SYSTEM to get county/state ---
    # Pull WATER_SYSTEM for geography mapping
    print("\n  [C] Pulling WATER_SYSTEM table for geographic mapping...")
    # This is a large table. We'll get the columns we need.
    # Pull key fields: pwsid, pws_name, state_code, county info
    # Unfortunately WATER_SYSTEM doesn't have county directly.
    # But GEOGRAPHIC_AREA does.
    print("    Trying GEOGRAPHIC_AREA table...")
    geo_url = "https://data.epa.gov/efservice/GEOGRAPHIC_AREA"
    df_geo = fetch_epa_paginated(geo_url, batch_size=10000, max_rows=500000, pause=1.0)

    if df_geo is not None and len(df_geo) > 0:
        print(f"  GEOGRAPHIC_AREA records: {len(df_geo)}")
        print(f"  Columns: {list(df_geo.columns)}")
    else:
        print("  GEOGRAPHIC_AREA not available. Using state from PWSID prefix instead.")
        df_geo = None

    # Combine and save
    # For violations: aggregate by state (from primacy_agency_code / epa_region)
    if len(df_viol) > 0:
        # primacy_agency_code = state FIPS code
        col_map_v = {c.lower(): c for c in df_viol.columns}
        state_col = col_map_v.get('primacy_agency_code')

        if state_col:
            df_viol['state_abbrev'] = df_viol[state_col].astype(str).str.zfill(2).map(STATE_FIPS)

            # Aggregate by state
            state_agg = df_viol.groupby(['state_abbrev', state_col]).agg(
                lead_violation_count=('violation_id', 'nunique') if 'violation_id' in df_viol.columns else (state_col, 'count'),
                systems_with_violations=('pwsid', 'nunique') if 'pwsid' in df_viol.columns else (state_col, 'count'),
            ).reset_index()
            state_agg.columns = ['state_abbrev', 'state_fips', 'lead_violation_count', 'systems_with_violations']

            # If we have geo data, try to get county
            if df_geo is not None:
                geo_cols = {c.lower(): c for c in df_geo.columns}
                if 'pwsid' in geo_cols and 'county_served' in geo_cols:
                    # Create PWSID -> county mapping
                    pws_county = df_geo[[geo_cols['pwsid'], geo_cols.get('county_served', '')]].drop_duplicates()
                    # Join with violations
                    df_viol_geo = df_viol.merge(pws_county, left_on='pwsid',
                                                right_on=geo_cols['pwsid'], how='left')
                    county_col_name = geo_cols.get('county_served', 'county')
                    county_agg = df_viol_geo.groupby(['state_abbrev', county_col_name]).agg(
                        lead_violation_count=('violation_id', 'nunique') if 'violation_id' in df_viol_geo.columns else (county_col_name, 'count'),
                        systems_with_violations=('pwsid', 'nunique') if 'pwsid' in df_viol_geo.columns else (county_col_name, 'count'),
                    ).reset_index()

                    out_path = os.path.join(OUTPUT_DIR, "epa_lead_water_violations.csv")
                    county_agg.to_csv(out_path, index=False)
                    print(f"\n  Saved county-level lead violations to {out_path}")
                    print(f"  Total counties with violations: {len(county_agg)}")
                    print(county_agg.nlargest(15, 'lead_violation_count').to_string(index=False))
                    return True

            # Fall back to state-level
            out_path = os.path.join(OUTPUT_DIR, "epa_lead_water_violations.csv")
            state_agg.to_csv(out_path, index=False)
            print(f"\n  Saved state-level lead violations to {out_path}")
            print(f"  (County not available in SDWIS without GEOGRAPHIC_AREA join)")
            print(state_agg.to_string(index=False))
            return True

    # If violations empty but LCR has data
    if len(df_lcr) > 0:
        col_map_lcr = {c.lower(): c for c in df_lcr.columns}
        state_col = col_map_lcr.get('primacy_agency_code')
        if state_col:
            df_lcr['state_abbrev'] = df_lcr[state_col].astype(str).str.zfill(2).map(STATE_FIPS)
            state_agg = df_lcr.groupby(['state_abbrev']).agg(
                lcr_sample_count=(state_col, 'count'),
                systems_sampled=('pwsid', 'nunique') if 'pwsid' in df_lcr.columns else (state_col, 'count'),
                mean_lead_level=('sample_measure', 'mean') if 'sample_measure' in df_lcr.columns else (state_col, 'count'),
            ).reset_index()

            out_path = os.path.join(OUTPUT_DIR, "epa_lead_water_violations.csv")
            state_agg.to_csv(out_path, index=False)
            print(f"\n  Saved LCR lead sample data by state to {out_path}")
            return True

    print("  FAILED to get any lead violation data.")
    out_path = os.path.join(OUTPUT_DIR, "epa_lead_water_violations.csv")
    pd.DataFrame(columns=['state','county','lead_violation_count']).to_csv(out_path, index=False)
    return False


# ============================================================
# 3. CDC BLOOD LEAD SURVEILLANCE
# ============================================================
def pull_cdc_blood_lead():
    print("\n" + "="*70)
    print("3. CDC BLOOD LEAD SURVEILLANCE")
    print("="*70)

    # The CDC Environmental Public Health Tracking Network API
    # Content area 6 = Childhood Lead Poisoning
    # The DataExplorer is currently down for maintenance.
    # We'll try the API with known measure IDs.

    base_url = "https://ephtracking.cdc.gov/apigateway/api/v1"

    # Known measure IDs for childhood lead (from prior API documentation):
    # 556 = % children with BLL >= 5 ug/dL
    # 557 = % children with BLL >= 10 ug/dL
    # 396 = Number of children tested
    # Try getting data with getCoreHolder

    success = False

    # Try various known measure IDs for lead
    measure_ids = [556, 557, 558, 559, 396, 397, 398, 399, 400, 401, 402, 403,
                   780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790]

    for mid in measure_ids:
        # Try county level (geographicTypeId=2) and state level (geographicTypeId=1)
        for geo_type, geo_label in [(2, "county"), (1, "state")]:
            # Format: getCoreHolder/{measureId}/{stratificationLevelId}/{geographicTypeId}/{temporalTypeId}/{isSmoothed}/json
            url = f"{base_url}/getCoreHolder/{mid}/1/{geo_type}/0/0/json"
            resp = safe_request(url, timeout=30, retries=1)
            if resp:
                try:
                    data = resp.json()
                    if isinstance(data, list) and len(data) > 5:
                        df = pd.json_normalize(data)
                        print(f"  Measure {mid} at {geo_label} level: {len(df)} records!")
                        print(f"  Columns: {list(df.columns)[:12]}")

                        out_path = os.path.join(OUTPUT_DIR, f"cdc_blood_lead_{geo_label}_m{mid}.csv")
                        df.to_csv(out_path, index=False)
                        print(f"  Saved to {out_path}")
                        print(df.head(5).to_string())
                        success = True

                        if geo_type == 2:  # county level found, great!
                            # Also save as the main file
                            main_path = os.path.join(OUTPUT_DIR, "cdc_blood_lead_surveillance.csv")
                            df.to_csv(main_path, index=False)
                            return True
                except (json.JSONDecodeError, ValueError):
                    pass

            time.sleep(0.3)

        if success:
            break

    # If API didn't work, try state-level data from CDC Socrata
    if not success:
        print("\n  CDC EPHT API did not return lead data (may be in maintenance).")
        print("  Trying CDC Socrata and other sources...")

        # Try various known datasets
        datasets = [
            ("https://data.cdc.gov/resource/m5fk-6xgv.csv", "CT Lead Poisoning", 50000),
            ("https://data.cdc.gov/resource/d54z-enu8.csv", "NY Blood Lead by Zip", 50000),
            ("https://data.cdc.gov/resource/tnry-kwh5.csv", "NYC BLL", 50000),
        ]

        for url, name, limit in datasets:
            print(f"  Trying {name}: {url}")
            resp = safe_request(f"{url}?$limit={limit}", timeout=60, retries=1)
            if resp and len(resp.text.strip()) > 100 and '"error"' not in resp.text[:50]:
                try:
                    df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
                    if len(df) > 5:
                        print(f"    Got {len(df)} rows!")
                        print(f"    Columns: {list(df.columns)}")
                        out_path = os.path.join(OUTPUT_DIR, "cdc_blood_lead_surveillance.csv")
                        df.to_csv(out_path, index=False)
                        print(f"    Saved to {out_path}")
                        print(df.head(5).to_string())
                        success = True
                        break
                except Exception as e:
                    print(f"    Parse error: {e}")

    if not success:
        print("\n  MANUAL STEPS NEEDED for CDC Blood Lead Data:")
        print("    The CDC Environmental Public Health Tracking Network DataExplorer")
        print("    is currently down for maintenance.")
        print("    When available:")
        print("    1. Go to https://ephtracking.cdc.gov/DataExplorer/")
        print("    2. Content area: 'Childhood Lead Poisoning'")
        print("    3. Measure: '% children tested with elevated BLL >= 5 ug/dL'")
        print("    4. Geography: County, all states")
        print("    5. Download CSV")
        print("    Alternative: https://www.cdc.gov/lead-prevention/data-research/index.html")

        out_path = os.path.join(OUTPUT_DIR, "cdc_blood_lead_surveillance.csv")
        pd.DataFrame(columns=['state','county','fips','year','pct_elevated_bll',
                              'children_tested','source_note']).to_csv(out_path, index=False)

    return success


# ============================================================
# 4. CENSUS PRE-1978 HOUSING (ACS B25034)
# ============================================================
def pull_census_housing():
    print("\n" + "="*70)
    print("4. CENSUS PRE-1978 HOUSING (ACS B25034)")
    print("="*70)

    # 2022 ACS 5-year B25034 variables:
    # _001E = Total
    # _002E = Built 2020 or later
    # _003E = Built 2010 to 2019
    # _004E = Built 2000 to 2009
    # _005E = Built 1990 to 1999
    # _006E = Built 1980 to 1989
    # _007E = Built 1970 to 1979  <-- partially pre-1978
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

        print(f"  Trying Census ACS {year} 5-year estimates...")
        resp = safe_request(url, params=params, timeout=120)

        if resp is None:
            print(f"  Failed for {year}")
            continue

        # Check for error response
        try:
            text = resp.text.strip()
            if text.startswith('{') and 'error' in text.lower():
                print(f"  API error for {year}: {text[:200]}")
                continue
            data = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"  JSON parse error for {year}: {e}")
            print(f"  Response preview: {resp.text[:300]}")
            continue

        if not isinstance(data, list) or len(data) < 2:
            print(f"  Unexpected response format for {year}")
            continue

        headers = data[0]
        rows = data[1:]

        df = pd.DataFrame(rows, columns=headers)
        print(f"  Got {len(df)} counties from {year} ACS")

        # Convert numeric columns
        num_cols = [c for c in df.columns if c.startswith('B25034')]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Pre-1978 = 1970-1979 (proxy, includes some post-1978) + 1960s + 1950s + 1940s + pre-1939
        df['pre_1980_units'] = (df['B25034_007E'] + df['B25034_008E'] +
                                df['B25034_009E'] + df['B25034_010E'] +
                                df['B25034_011E'])
        df['total_units'] = df['B25034_001E']
        df['pct_pre1978_housing'] = (df['pre_1980_units'] / df['total_units'] * 100).round(2)

        # Create FIPS
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
        print(f"  Year: {year} ACS 5-year")
        print(f"\n  Summary statistics:")
        print(f"    Counties: {len(output_df)}")
        print(f"    Mean pct pre-1978 housing: {output_df['pct_pre1978_housing'].mean():.1f}%")
        print(f"    Median: {output_df['pct_pre1978_housing'].median():.1f}%")
        print(f"    Max: {output_df['pct_pre1978_housing'].max():.1f}%")
        print(f"\n  Top 15 counties by pre-1978 housing:")
        top = output_df.nlargest(15, 'pct_pre1978_housing')
        print(top[['county_fips','county_name','state_name','pct_pre1978_housing',
                    'total_housing_units']].to_string(index=False))
        return True

    print("  FAILED to pull Census housing data.")
    return False


# ============================================================
# 5. USGS MINERAL RESOURCES DATA SYSTEM (MRDS)
# ============================================================
def pull_usgs_mining():
    print("\n" + "="*70)
    print("5. USGS MINERAL RESOURCES DATA SYSTEM (MRDS)")
    print("="*70)

    print("  Downloading MRDS CSV from USGS...")
    url = "https://mrdata.usgs.gov/mrds/mrds-csv.zip"
    resp = safe_request(url, timeout=300, stream=True)

    if resp is None:
        print("  Download failed.")
        out_path = os.path.join(OUTPUT_DIR, "usgs_mining_sites_by_county.csv")
        pd.DataFrame(columns=['state','county','mining_site_count','lead_zinc_copper_sites']).to_csv(out_path, index=False)
        return False

    zip_path = os.path.join(OUTPUT_DIR, "mrds_temp.zip")
    try:
        with open(zip_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  Downloaded {os.path.getsize(zip_path)/1024/1024:.1f} MB")

        with zipfile.ZipFile(zip_path, 'r') as z:
            csv_name = [f for f in z.namelist() if f.endswith('.csv')][0]
            with z.open(csv_name) as csvfile:
                df = pd.read_csv(csvfile, low_memory=False, encoding='latin-1')

        os.remove(zip_path)
        print(f"  Total MRDS records: {len(df)}")
        print(f"  Columns: {list(df.columns)}")

        # Filter to US
        if 'country' in df.columns:
            us_mask = df['country'].astype(str).str.upper().str.contains('UNITED STATES|^US$|^USA$', na=False)
            df_us = df[us_mask].copy()
            print(f"  US sites: {len(df_us)} of {len(df)}")
        else:
            df_us = df.copy()

        # Aggregate by state + county
        if 'state' in df_us.columns and 'county' in df_us.columns:
            county_agg = df_us.groupby(['state', 'county']).agg(
                mining_site_count=('county', 'count')
            ).reset_index()

            # Lead/zinc/copper mining sites
            if 'commod1' in df_us.columns:
                lead_keywords = ['LEAD', 'PB', 'ZINC', 'ZN', 'COPPER', 'CU']
                lead_mask = df_us['commod1'].astype(str).str.upper().str.contains(
                    '|'.join(lead_keywords), na=False)
                lead_mining = df_us[lead_mask].groupby(['state', 'county']).size().reset_index(
                    name='lead_zinc_copper_sites')
                county_agg = county_agg.merge(lead_mining, on=['state', 'county'], how='left')
                county_agg['lead_zinc_copper_sites'] = county_agg['lead_zinc_copper_sites'].fillna(0).astype(int)

            out_path = os.path.join(OUTPUT_DIR, "usgs_mining_sites_by_county.csv")
            county_agg.to_csv(out_path, index=False)
            print(f"  Saved to {out_path}")
            print(f"  Counties with mining sites: {len(county_agg)}")
            print(f"\n  Top 20 counties by mining sites:")
            print(county_agg.nlargest(20, 'mining_site_count').to_string(index=False))
            return True

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(zip_path):
            os.remove(zip_path)

    return False


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("="*70)
    print("ENVIRONMENTAL DATA PULL - COUNTY-LEVEL LEAD CONTAMINATION")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    results = {}

    for name, func in [
        ('superfund', pull_epa_superfund),
        ('lead_violations', pull_epa_lead_violations),
        ('cdc_blood_lead', pull_cdc_blood_lead),
        ('census_housing', pull_census_housing),
        ('usgs_mining', pull_usgs_mining),
    ]:
        try:
            results[name] = func()
        except Exception as e:
            print(f"\n  FATAL ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for source, status in results.items():
        symbol = "OK" if status else "FAILED/PLACEHOLDER"
        print(f"  [{symbol}] {source}")

    print(f"\nOutput files in {OUTPUT_DIR}:")
    keywords = ['superfund', 'lead', 'blood', 'housing', 'mining', 'epa', 'cdc', 'census', 'usgs', 'lcr']
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.csv') and any(kw in f.lower() for kw in keywords):
            fpath = os.path.join(OUTPUT_DIR, f)
            size = os.path.getsize(fpath)
            # Count rows
            try:
                row_count = sum(1 for _ in open(fpath)) - 1
                print(f"  {f}: {size/1024:.1f} KB ({row_count:,} rows)")
            except:
                print(f"  {f}: {size/1024:.1f} KB")

    print(f"\nCompleted: {time.strftime('%Y-%m-%d %H:%M:%S')}")
