#!/usr/bin/env python3
"""
Pull EPA Toxics Release Inventory (TRI) data for LEAD releases,
aggregate to county level.

Source: EPA TRI Basic Data Files
https://www.epa.gov/toxics-release-inventory-tri-program/tri-basic-data-files-calendar-years-1987-present
"""

import os
import io
import sys
import zipfile
import pandas as pd
import requests
import time
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent
OUTFILE = os.path.join(OUTDIR, "epa_tri_lead_by_state_year.csv")

# TRI data years for the panel (2001-2022)
YEARS = list(range(2001, 2023))


# FIPS code lookup for state+county
# We'll load the census gazetteer file if available, otherwise build from data
GAZ_FILE = os.path.join(OUTDIR, "2020_Gaz_counties_national.txt")


def download_tri_basic(year: int) -> pd.DataFrame | None:
    """Download TRI basic data file for a given year."""
    url = f"https://data.epa.gov/efservice/downloads/tri/mv_tri_basic_download/{year}_U.S./csv"
    print(f"  Trying: {url}")
    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get('Content-Type', '')
        print(f"  Status: {resp.status_code}, Content-Type: {content_type}, Size: {len(resp.content):,} bytes")

        # The response might be a zip file or raw CSV
        raw = resp.content
        if raw[:2] == b'PK':  # ZIP file magic bytes
            print("  -> Detected ZIP file")
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                names = zf.namelist()
                print(f"  -> Files in zip: {names}")
                # Read the first CSV-like file
                for name in names:
                    if name.lower().endswith('.csv') or name.lower().endswith('.txt'):
                        with zf.open(name) as f:
                            df = pd.read_csv(f, dtype=str, low_memory=False, encoding='latin-1')
                        print(f"  -> Read {len(df)} rows from {name}")
                        return df
                # If no csv/txt, try first file
                with zf.open(names[0]) as f:
                    df = pd.read_csv(f, dtype=str, low_memory=False, encoding='latin-1')
                print(f"  -> Read {len(df)} rows from {names[0]}")
                return df
        else:
            # Try as raw CSV
            df = pd.read_csv(io.BytesIO(raw), dtype=str, low_memory=False, encoding='latin-1')
            print(f"  -> Read {len(df)} rows as CSV")
            return df
    except Exception as e:
        print(f"  ERROR downloading {year}: {e}")
        return None


def try_alternative_download(year: int) -> pd.DataFrame | None:
    """Try alternative TRI download URLs."""
    # Alternative URL patterns
    alt_urls = [
        f"https://data.epa.gov/efservice/downloads/tri/mv_tri_basic_download/{year}_US/csv",
        f"https://www3.epa.gov/tri/current/US_{year}_v19/US_1a_{year}.csv",
        f"https://www3.epa.gov/tri/current/US_{year}/US_1a_{year}.csv",
    ]
    for url in alt_urls:
        print(f"  Alt trying: {url}")
        try:
            resp = requests.get(url, timeout=120, stream=True)
            if resp.status_code == 200:
                raw = resp.content
                print(f"  Status: {resp.status_code}, Size: {len(raw):,} bytes")
                if raw[:2] == b'PK':
                    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                        names = zf.namelist()
                        print(f"  -> ZIP files: {names}")
                        for name in names:
                            if name.lower().endswith('.csv') or name.lower().endswith('.txt'):
                                with zf.open(name) as f:
                                    df = pd.read_csv(f, dtype=str, low_memory=False, encoding='latin-1')
                                print(f"  -> Read {len(df)} rows")
                                return df
                else:
                    df = pd.read_csv(io.BytesIO(raw), dtype=str, low_memory=False, encoding='latin-1')
                    print(f"  -> Read {len(df)} rows")
                    return df
            else:
                print(f"  Status: {resp.status_code}")
        except Exception as e:
            print(f"  Error: {e}")
    return None


def try_envirofacts_api() -> pd.DataFrame | None:
    """Try the EPA Envirofacts REST API for TRI data."""
    print("\n--- Trying Envirofacts REST API ---")

    # Try various table/endpoint combos
    base = "https://data.epa.gov/efservice"
    endpoints = [
        "/TRI_FACILITY/ROWS/0:5/JSON",
        "/MV_TRI_BASIC_DOWNLOAD/ROWS/0:5/JSON",
        "/V_TRI_FORM_R/CHEMICAL_NAME/LEAD/ROWS/0:5/JSON",
    ]
    for ep in endpoints:
        url = base + ep
        print(f"  Trying: {url}")
        try:
            resp = requests.get(url, timeout=30)
            print(f"  Status: {resp.status_code}, Content-Type: {resp.headers.get('Content-Type','')}")
            if resp.status_code == 200:
                print(f"  First 500 chars: {resp.text[:500]}")
        except Exception as e:
            print(f"  Error: {e}")

    # Full pull if any endpoint works - try paginated approach
    # TRI_RELEASE or similar
    print("\n  Trying paginated pull of TRI release data for Lead...")
    all_rows = []

    table_patterns = [
        "TRI_RELEASE_QTY/CHEMICAL_NAME/LEAD",
        "TRI_RELEASE_QTY/CHEMICAL_NAME/LEAD COMPOUNDS",
    ]

    for pattern in table_patterns:
        url = f"{base}/{pattern}/ROWS/0:100/JSON"
        print(f"  Trying: {url}")
        try:
            resp = requests.get(url, timeout=30)
            print(f"  Status: {resp.status_code}")
            if resp.status_code == 200 and resp.text.strip().startswith('['):
                data = resp.json()
                if data:
                    print(f"  Got {len(data)} rows! Keys: {list(data[0].keys())}")
                    all_rows.extend(data)
        except Exception as e:
            print(f"  Error: {e}")

    if all_rows:
        return pd.DataFrame(all_rows)
    return None


def try_tri_explorer_download() -> pd.DataFrame | None:
    """
    Try downloading from TRI Explorer which can export CSV.
    This uses a form POST to get data.
    """
    print("\n--- Trying TRI Explorer ---")
    url = "https://enviro.epa.gov/triexplorer/release_chem"

    # TRI Explorer form params for lead releases, all states, all counties
    params = {
        'detail': '1',
        'sort': 'CHEMICAL',
        'chemical': '007439921',  # Lead CAS number
        'year': '2022',
        'tab_rpt': '1',
        'fld': 'RESSION',
        'fld': 'LNAMEUNIT',
        'fld': 'E1',  # fugitive air
        'fld': 'E2',  # stack air
        'fld': 'E3',  # water
        'fld': 'E5',  # underground injection
        'fld': 'E51',
        'fld': 'E52',
        'fld': 'E53',
        'fld': 'E54',
        'fld': 'TSFDSP',  # total on-site disposal/release
        'triession': 'TRIALL',
        'tristate': 'All+states',
        'tricounty': '',
        'trizip': '',
        'triPOW': '',
        'triEPA': '',
        'triSIC': '',
        'triNAICS': '',
        'triPartner': '',
        'triFederalFacility': '',
        'output_type': 'CSV',
    }
    try:
        resp = requests.post(url, data=params, timeout=60)
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"  Size: {len(resp.content):,} bytes")
            print(f"  First 500 chars: {resp.text[:500]}")
            if 'html' not in resp.text[:200].lower():
                df = pd.read_csv(io.StringIO(resp.text), dtype=str)
                print(f"  -> Got {len(df)} rows")
                return df
    except Exception as e:
        print(f"  Error: {e}")
    return None


def load_fips_lookup() -> pd.DataFrame:
    """Load county FIPS codes from census gazetteer file."""
    if os.path.exists(GAZ_FILE):
        print(f"\nLoading FIPS from {GAZ_FILE}")
        gaz = pd.read_csv(GAZ_FILE, sep='\t', dtype=str)
        # Clean column names
        gaz.columns = [c.strip() for c in gaz.columns]
        print(f"  Columns: {list(gaz.columns)}")
        print(f"  Rows: {len(gaz)}")
        return gaz
    return pd.DataFrame()


def normalize_county(name: str) -> str:
    """Normalize county name for matching."""
    if pd.isna(name):
        return ''
    name = str(name).upper().strip()
    # Remove common suffixes for matching
    for suffix in [' COUNTY', ' PARISH', ' BOROUGH', ' CENSUS AREA',
                   ' MUNICIPALITY', ' CITY AND BOROUGH', ' CITY']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name.strip()


def main():
    print("=" * 70)
    print("EPA TRI Lead Release Data - County Aggregation")
    print("=" * 70)

    all_dfs = []

    # ---- Approach 1: TRI Basic Data File downloads ----
    print("\n--- Approach 1: TRI Basic Data Files ---")
    for year in YEARS:
        print(f"\nDownloading {year}...")
        df = download_tri_basic(year)
        if df is None:
            df = try_alternative_download(year)
        if df is not None:
            df['_YEAR'] = str(year)
            all_dfs.append(df)
            print(f"  SUCCESS for {year}: {len(df)} rows, {len(df.columns)} columns")
            print(f"  Columns (first 20): {list(df.columns[:20])}")
        else:
            print(f"  FAILED for {year}")
        time.sleep(1)  # Be polite

    if not all_dfs:
        # ---- Approach 2: Envirofacts API ----
        print("\n--- Approach 2: Envirofacts API ---")
        df = try_envirofacts_api()
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        # ---- Approach 3: TRI Explorer ----
        df = try_tri_explorer_download()
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("\n*** ALL download approaches failed. ***")
        print("Generating synthetic county-level TRI data from known EPA summaries...")
        generate_fallback_data()
        return

    # Combine all years
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n--- Combined data: {len(combined)} rows, {len(combined.columns)} columns ---")
    print(f"Columns: {list(combined.columns)}")

    # Identify relevant columns - TRI basic files have standardized names
    # Look for key columns
    col_map = identify_columns(combined)
    print(f"\nColumn mapping: {col_map}")

    if not col_map:
        print("ERROR: Could not identify required columns")
        print("Sample of first 5 rows:")
        print(combined.head())
        return

    # Filter for Lead and Lead compounds
    chem_col = col_map.get('chemical')
    if chem_col:
        lead_mask = combined[chem_col].str.upper().str.contains('LEAD', na=False)
        lead_df = combined[lead_mask].copy()
        print(f"\nFiltered to Lead/Lead compounds: {len(lead_df)} rows")
        if len(lead_df) > 0:
            print(f"  Chemicals found: {lead_df[chem_col].unique()}")
    else:
        lead_df = combined.copy()
        print("WARNING: No chemical column found, using all data")

    # Process and aggregate
    process_and_aggregate(lead_df, col_map)


def identify_columns(df: pd.DataFrame) -> dict:
    """Identify the relevant columns in the TRI basic data file."""
    cols = [c.upper().strip() for c in df.columns]
    orig_cols = list(df.columns)

    mapping = {}

    # Chemical name
    for i, c in enumerate(cols):
        if 'CHEMICAL' in c and 'NAME' in c:
            mapping['chemical'] = orig_cols[i]
            break
        elif c == 'CHEMICAL':
            mapping['chemical'] = orig_cols[i]
            break
    if 'chemical' not in mapping:
        for i, c in enumerate(cols):
            if 'CHEM' in c:
                mapping['chemical'] = orig_cols[i]
                break

    # State
    for i, c in enumerate(cols):
        if 'ST' == c or c == 'STATE' or ('FACILITY' in c and 'STATE' in c) or c == 'ST_ABBR':
            mapping['state'] = orig_cols[i]
            break
    if 'state' not in mapping:
        for i, c in enumerate(cols):
            if 'STATE' in c and 'COUNTY' not in c:
                mapping['state'] = orig_cols[i]
                break

    # County
    for i, c in enumerate(cols):
        if 'COUNTY' in c and 'FIPS' not in c:
            mapping['county'] = orig_cols[i]
            break

    # County FIPS
    for i, c in enumerate(cols):
        if 'COUNTY' in c and 'FIPS' in c:
            mapping['county_fips'] = orig_cols[i]
            break

    # State FIPS
    for i, c in enumerate(cols):
        if 'STATE' in c and 'FIPS' in c:
            mapping['state_fips'] = orig_cols[i]
            break

    # FIPS (combined)
    for i, c in enumerate(cols):
        if c == 'FIPS' or c == 'COUNTY_FIPS_CODE':
            mapping['fips'] = orig_cols[i]
            break

    # Total on-site release amounts
    # TRI basic files typically have columns like:
    # 5.1 - FUGITIVE AIR, 5.2 - STACK AIR, 5.3 - WATER,
    # 5.4 - UNDERGROUND, 5.5.1-5.5.4 - LAND
    # ON-SITE RELEASE TOTAL
    for i, c in enumerate(cols):
        if 'ON-SITE' in c and 'TOTAL' in c and 'RELEASE' in c:
            mapping['onsite_total'] = orig_cols[i]
            break
        elif 'ONSITE' in c and 'TOTAL' in c:
            mapping['onsite_total'] = orig_cols[i]
            break
        elif 'ON_SITE' in c and 'TOTAL' in c:
            mapping['onsite_total'] = orig_cols[i]
            break

    if 'onsite_total' not in mapping:
        # Look for total release
        for i, c in enumerate(cols):
            if 'TOTAL' in c and 'RELEASE' in c:
                mapping['onsite_total'] = orig_cols[i]
                break

    if 'onsite_total' not in mapping:
        # Look for individual release columns and sum them
        release_cols = []
        for i, c in enumerate(cols):
            if any(x in c for x in ['FUGITIVE', 'STACK', '5.1', '5.2', '5.3', '5.4', '5.5']):
                release_cols.append(orig_cols[i])
        if release_cols:
            mapping['release_components'] = release_cols

    # Year
    for i, c in enumerate(cols):
        if c == 'YEAR' or c == 'REPORTING_YEAR' or ('REPORT' in c and 'YEAR' in c):
            mapping['year'] = orig_cols[i]
            break

    # Facility name (for counting)
    for i, c in enumerate(cols):
        if 'FACILITY' in c and 'NAME' in c:
            mapping['facility'] = orig_cols[i]
            break

    # TRI Facility ID
    for i, c in enumerate(cols):
        if 'TRIFID' in c or ('TRI' in c and 'ID' in c) or c == 'TRIFD':
            mapping['tri_id'] = orig_cols[i]
            break
    if 'tri_id' not in mapping:
        for i, c in enumerate(cols):
            if 'FACILITY' in c and 'ID' in c:
                mapping['tri_id'] = orig_cols[i]
                break

    return mapping


def process_and_aggregate(df: pd.DataFrame, col_map: dict):
    """Process lead data and aggregate to county level."""

    # Get release amounts
    if 'onsite_total' in col_map:
        release_col = col_map['onsite_total']
        df['release_lbs'] = pd.to_numeric(df[release_col], errors='coerce').fillna(0)
    elif 'release_components' in col_map:
        df['release_lbs'] = 0
        for rc in col_map['release_components']:
            df['release_lbs'] += pd.to_numeric(df[rc], errors='coerce').fillna(0)
    else:
        print("ERROR: No release amount columns found")
        return

    # Get year
    if 'year' in col_map:
        df['year'] = df[col_map['year']].astype(str).str.strip()
    elif '_YEAR' in df.columns:
        df['year'] = df['_YEAR']
    else:
        df['year'] = 'unknown'

    # Get state and county
    state_col = col_map.get('state', None)
    county_col = col_map.get('county', None)

    if state_col:
        df['state'] = df[state_col].astype(str).str.strip()
    if county_col:
        df['county'] = df[county_col].astype(str).str.strip()

    # Build FIPS
    if 'fips' in col_map:
        df['FIPS'] = df[col_map['fips']].astype(str).str.strip().str.zfill(5)
    elif 'state_fips' in col_map and 'county_fips' in col_map:
        df['FIPS'] = (
            df[col_map['state_fips']].astype(str).str.strip().str.zfill(2) +
            df[col_map['county_fips']].astype(str).str.strip().str.zfill(3)
        )
    else:
        # We'll need to build FIPS from state+county using gazetteer
        df['FIPS'] = ''
        gaz = load_fips_lookup()
        if len(gaz) > 0:
            # Build lookup from gazetteer
            fips_col = [c for c in gaz.columns if 'GEOID' in c.upper() or 'FIPS' in c.upper()]
            name_col = [c for c in gaz.columns if 'NAME' in c.upper()]
            state_gaz_col = [c for c in gaz.columns if 'USPS' in c.upper() or 'STATE' in c.upper()]

            if fips_col and name_col and state_gaz_col:
                lookup = {}
                for _, row in gaz.iterrows():
                    key = (str(row[state_gaz_col[0]]).strip().upper(),
                           normalize_county(str(row[name_col[0]])))
                    lookup[key] = str(row[fips_col[0]]).strip().zfill(5)

                for idx in df.index:
                    st = str(df.at[idx, 'state']).strip().upper() if state_col else ''
                    ct = normalize_county(str(df.at[idx, 'county'])) if county_col else ''
                    key = (st, ct)
                    if key in lookup:
                        df.at[idx, 'FIPS'] = lookup[key]

    # Facility ID for counting unique facilities
    fac_col = col_map.get('tri_id', col_map.get('facility', None))
    if fac_col:
        df['fac_id'] = df[fac_col].astype(str)
    else:
        df['fac_id'] = df.index.astype(str)

    print(f"\nData summary before aggregation:")
    print(f"  Rows: {len(df)}")
    print(f"  Years: {sorted(df['year'].unique())}")
    print(f"  Total release (lbs): {df['release_lbs'].sum():,.0f}")
    if state_col:
        print(f"  States: {df['state'].nunique()}")

    # Aggregate by state and year
    group_cols = ['state', 'year']
    
    # Ensure all group columns exist
    for gc in group_cols:
        if gc not in df.columns:
            df[gc] = 'unknown'

    agg = df.groupby(group_cols).agg(
        total_lead_released_lbs=('release_lbs', 'sum'),
        n_facilities=('fac_id', 'nunique')
    ).reset_index()

    # Sort
    agg = agg.sort_values(['state', 'year']).reset_index(drop=True)

    # Reorder columns
    final_cols = ['state', 'year', 'total_lead_released_lbs', 'n_facilities']
    agg = agg[final_cols]

    print(f"\n--- Aggregated state-level panel data ---")
    print(f"  Rows: {len(agg)}")
    print(f"  States: {agg['state'].nunique()}")
    print(f"  Total release across all years: {agg['total_lead_released_lbs'].sum():,.0f} lbs")
    print(f"\nTop 10 states by total lead released (all years):")
    top10 = agg.groupby('state')['total_lead_released_lbs'].sum().sort_values(ascending=False).head(10)
    print(top10)


    # Save
    agg.to_csv(OUTFILE, index=False)
    print(f"\nSaved {len(agg)} rows to {OUTFILE}")


def generate_fallback_data():
    """
    If all download methods fail, try one more creative approach:
    Use the Envirofacts ECHO/TRI search with specific params.
    """
    print("\nAttempting final fallback: ECHO facility search for lead...")

    # ECHO API for facilities with TRI lead releases
    # https://echo.epa.gov/tools/data-downloads
    url = "https://echodata.epa.gov/echo/tri_download.download?p_chemical=007439921&p_view=TRFA&output=CSV"

    print(f"  Trying ECHO: {url}")
    try:
        resp = requests.get(url, timeout=120)
        print(f"  Status: {resp.status_code}, Size: {len(resp.content):,}")
        if resp.status_code == 200 and len(resp.content) > 100:
            df = pd.read_csv(io.BytesIO(resp.content), dtype=str, low_memory=False, encoding='latin-1')
            print(f"  Got {len(df)} rows")
            print(f"  Columns: {list(df.columns)}")
            return
    except Exception as e:
        print(f"  Error: {e}")

    print("\n*** All approaches exhausted. Please download TRI data manually from:")
    print("  https://www.epa.gov/toxics-release-inventory-tri-program/tri-basic-data-files-calendar-years-1987-present")
    print("  or https://enviro.epa.gov/triexplorer/tri_release.chemical")


if __name__ == '__main__':
    main()
