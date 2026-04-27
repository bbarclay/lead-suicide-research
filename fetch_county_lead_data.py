#!/usr/bin/env python3
"""
Fetch COUNTY-LEVEL lead exposure data from every available source.
Saves each dataset with descriptive filenames.
"""

import requests
import pandas as pd
import io
import json
import os
import time
import zipfile
import tempfile
from pathlib import Path

OUT = Path(__file__).resolve().parent
import os as _os
CENSUS_KEY = _os.environ.get("CENSUS_API_KEY")
if not CENSUS_KEY:
    raise RuntimeError("Set CENSUS_API_KEY in your environment (see .env.example). Get a key free from https://api.census.gov/data/key_signup.html")

# Track what we get
results_log = []

def log(source, status, filename=None, geo_level=None, time_period=None, key_var=None, n_rows=None, n_counties=None):
    entry = {
        "source": source,
        "status": status,
        "filename": filename,
        "geo_level": geo_level,
        "time_period": time_period,
        "key_variable": key_var,
        "n_rows": n_rows,
        "n_counties": n_counties
    }
    results_log.append(entry)
    print(f"\n{'='*70}")
    print(f"SOURCE: {source}")
    print(f"STATUS: {status}")
    if filename: print(f"FILE: {filename}")
    if geo_level: print(f"GEO LEVEL: {geo_level}")
    if time_period: print(f"TIME PERIOD: {time_period}")
    if key_var: print(f"KEY VARIABLE: {key_var}")
    if n_rows: print(f"ROWS: {n_rows}")
    if n_counties: print(f"COUNTIES: {n_counties}")
    print(f"{'='*70}")


# ============================================================================
# 1. CDC EPHT API - County-level childhood blood lead data
# ============================================================================
def fetch_cdc_epht():
    """Try multiple CDC EPHT API endpoint formats for county-level blood lead data."""
    print("\n\n" + "#"*70)
    print("# 1. CDC Environmental Public Health Tracking Network API")
    print("#"*70)

    # Measure IDs: 1156 = % children BLL >= 5, 1534 = % children BLL >= 3.5, 1155 = % tested
    # Geographic Type 2 = County

    base = "https://ephtracking.cdc.gov/apigateway/api/v1"

    # Try different endpoint patterns
    endpoints_to_try = [
        # Pattern 1: getCoreHolder/{measureId}/{geographicTypeId}/{geographicItemsId}/{temporalTypeId}/{temporalItemsId}/{isSmoothed}
        f"{base}/getCoreHolder/1156/2/ALL/ALL/ALL/0",
        f"{base}/getCoreHolder/1534/2/ALL/ALL/ALL/0",
        f"{base}/getCoreHolder/1155/2/ALL/ALL/ALL/0",
        # Pattern 2: getData with query params
        f"{base}/getData?measureId=1156&geographicTypeId=2&isSmoothed=0",
        f"{base}/getData?measureId=1534&geographicTypeId=2&isSmoothed=0",
        # Pattern 3: with temporal
        f"{base}/getCoreHolder/1156/2/ALL/1/ALL/0",
        # Pattern 4: geographic items as state IDs
        f"{base}/getCoreHolder/1156/2/0/1/0/0",
    ]

    for endpoint in endpoints_to_try:
        try:
            print(f"\n  Trying: {endpoint[:100]}...")
            r = requests.get(endpoint, timeout=30)
            data = r.json()

            # Check if it's an error
            if isinstance(data, dict) and data.get("code") in [400, 401, 404]:
                print(f"  Error {data.get('code')}: {data.get('message', 'unknown')}")
                continue

            # If it's a list with data
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                print(f"  SUCCESS! Got {len(df)} records")
                print(f"  Columns: {list(df.columns)}")
                print(df.head())

                fname = "cdc_epht_county_blood_lead.csv"
                df.to_csv(OUT / fname, index=False)
                log("CDC EPHT API", "SUCCESS", fname, "County", "varies",
                    "% children with elevated BLL", len(df))
                return df

        except Exception as e:
            print(f"  Exception: {e}")
            continue

    # Also try listing geographic items and temporal items
    try:
        print("\n  Listing available geographic items for measure 1156...")
        r = requests.get(f"{base}/geographicItems/77/1156/2", timeout=15)
        if r.status_code == 200:
            items = r.json()
            if isinstance(items, list):
                print(f"  Found {len(items)} geographic items")
                if len(items) > 0:
                    print(f"  First few: {items[:5]}")

                    # Try to get temporal items
                    r2 = requests.get(f"{base}/temporalItems/77/1156/2/0", timeout=15)
                    if r2.status_code == 200:
                        temps = r2.json()
                        print(f"  Found {len(temps)} temporal items: {temps[:5]}")

                        # Now try to get actual data with specific items
                        geo_ids = ",".join([str(item.get("id", item.get("itemId", ""))) for item in items[:50]])
                        temp_ids = ",".join([str(item.get("id", item.get("parentTemporal", {}).get("id", ""))) for item in temps[:5]])

                        url = f"{base}/getCoreHolder/1156/2/{geo_ids}/1/{temp_ids}/0"
                        r3 = requests.get(url, timeout=30)
                        data = r3.json()
                        if isinstance(data, list) and len(data) > 0:
                            df = pd.DataFrame(data)
                            print(f"  SUCCESS with specific IDs! Got {len(df)} records")
                            fname = "cdc_epht_county_blood_lead.csv"
                            df.to_csv(OUT / fname, index=False)
                            log("CDC EPHT API", "SUCCESS", fname, "County", "varies",
                                "% children with elevated BLL", len(df))
                            return df
    except Exception as e:
        print(f"  Exception during geographic/temporal exploration: {e}")

    log("CDC EPHT API", "PARTIAL - Data Explorer down for maintenance, API requires token",
        None, "County (available when API accessible)", "2005-present",
        "% children tested with BLL >= 3.5 or 5 mcg/dL")
    return None


# ============================================================================
# 2. New York State - County & ZIP level blood lead (Socrata Open Data)
# ============================================================================
def fetch_ny_blood_lead():
    """Download NY State county-level blood lead data from health.data.ny.gov."""
    print("\n\n" + "#"*70)
    print("# 2. New York State Blood Lead Data (County & ZIP level)")
    print("#"*70)

    datasets = {
        # County-level: BLL >= 10 mcg/dL
        "ny_county_bll10": {
            "url": "https://health.data.ny.gov/api/views/iebf-7vjk/rows.csv?accessType=DOWNLOAD",
            "desc": "County-level BLL >= 10 mcg/dL"
        },
        # ZIP-level: elevated incidence
        "ny_zip_elevated": {
            "url": "https://health.data.ny.gov/api/views/d54z-enu8/rows.csv?accessType=DOWNLOAD",
            "desc": "ZIP-level elevated blood lead"
        }
    }

    for name, info in datasets.items():
        try:
            print(f"\n  Fetching {info['desc']}...")
            r = requests.get(info["url"], timeout=60)
            if r.status_code == 200 and len(r.content) > 100:
                df = pd.read_csv(io.StringIO(r.text))
                print(f"  Got {len(df)} rows, columns: {list(df.columns)}")
                print(df.head(3))

                fname = f"cdc_blood_lead_{name}.csv"
                df.to_csv(OUT / fname, index=False)

                # Aggregate to county if ZIP-level
                geo = "ZIP code" if "zip" in name else "County"
                n_counties = df["County"].nunique() if "County" in df.columns else None
                years = f"{df['Year'].min()}-{df['Year'].max()}" if "Year" in df.columns else "unknown"

                log(f"NY State ({info['desc']})", "SUCCESS", fname, geo, years,
                    "Blood lead tests, % elevated", len(df), n_counties)
            else:
                print(f"  Failed: status {r.status_code}")
                log(f"NY State ({info['desc']})", f"FAILED: HTTP {r.status_code}")
        except Exception as e:
            print(f"  Error: {e}")
            log(f"NY State ({info['desc']})", f"ERROR: {e}")


# ============================================================================
# 3. Michigan - County-level blood lead (MiTracking)
# ============================================================================
def fetch_michigan_lead():
    """Download Michigan county-level blood lead data."""
    print("\n\n" + "#"*70)
    print("# 3. Michigan Blood Lead Data")
    print("#"*70)

    # GitHub repo with Michigan BLL data
    urls = [
        "https://raw.githubusercontent.com/AmeliaMN/BLL/main/data/mi_bll_county.csv",
        "https://raw.githubusercontent.com/AmeliaMN/BLL/master/data/mi_bll_county.csv",
        "https://raw.githubusercontent.com/AmeliaMN/BLL/main/mi_bll_county.csv",
        "https://raw.githubusercontent.com/AmeliaMN/BLL/master/mi_bll_county.csv",
    ]

    for url in urls:
        try:
            print(f"  Trying: {url}")
            r = requests.get(url, timeout=15)
            if r.status_code == 200 and len(r.content) > 100 and not r.text.startswith("404"):
                df = pd.read_csv(io.StringIO(r.text))
                print(f"  Got {len(df)} rows, columns: {list(df.columns)}")
                fname = "michigan_county_blood_lead.csv"
                df.to_csv(OUT / fname, index=False)
                log("Michigan (GitHub BLL repo)", "SUCCESS", fname, "County",
                    "varies", "Blood lead levels", len(df),
                    df.iloc[:,0].nunique() if len(df.columns) > 0 else None)
                return
        except Exception as e:
            print(f"  Error: {e}")

    # Try the GitHub API to find the correct file structure
    try:
        print("  Checking repo structure...")
        r = requests.get("https://api.github.com/repos/AmeliaMN/BLL/contents/", timeout=15)
        if r.status_code == 200:
            contents = r.json()
            print(f"  Repo contents: {[c['name'] for c in contents]}")
            # Look for data files
            for item in contents:
                if item["type"] == "dir":
                    r2 = requests.get(item["url"], timeout=15)
                    if r2.status_code == 200:
                        subcontents = r2.json()
                        csv_files = [f for f in subcontents if f["name"].endswith(".csv")]
                        for cf in csv_files:
                            print(f"  Found CSV: {cf['name']} -> {cf.get('download_url','')}")
                            try:
                                r3 = requests.get(cf["download_url"], timeout=30)
                                if r3.status_code == 200:
                                    df = pd.read_csv(io.StringIO(r3.text))
                                    print(f"    Got {len(df)} rows, cols: {list(df.columns)[:5]}")
                                    fname = f"michigan_{cf['name']}"
                                    df.to_csv(OUT / fname, index=False)
                                    log(f"Michigan ({cf['name']})", "SUCCESS", fname,
                                        "County/ZIP", "varies", "Blood lead levels", len(df))
                            except Exception as e:
                                print(f"    Error: {e}")
    except Exception as e:
        print(f"  Repo exploration error: {e}")

    log("Michigan BLL Data", "PARTIAL - repo structure explored")


# ============================================================================
# 4. CDC State Surveillance Data - Try to access county-level Excel files
# ============================================================================
def fetch_cdc_state_surveillance():
    """Try to download CDC state surveillance Excel files with county data."""
    print("\n\n" + "#"*70)
    print("# 4. CDC State Blood Lead Surveillance (county-level in state files)")
    print("#"*70)

    # The CDC hosts Excel files for each state. Try common URL patterns.
    base_urls = [
        "https://www.cdc.gov/lead-prevention/media/data-downloads/state-data/",
        "https://www.cdc.gov/nceh/lead/data/tables/",
        "https://www.cdc.gov/lead-prevention/media/pdfs/data/",
    ]

    states = ["alabama", "alaska", "arizona", "arkansas", "california", "colorado",
              "connecticut", "delaware", "florida", "georgia", "idaho", "illinois",
              "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine",
              "maryland", "massachusetts", "michigan", "minnesota", "mississippi",
              "missouri", "montana", "nebraska", "nevada", "new-hampshire",
              "new-jersey", "new-mexico", "new-york", "north-carolina", "north-dakota",
              "ohio", "oklahoma", "oregon", "pennsylvania", "rhode-island",
              "south-carolina", "south-dakota", "tennessee", "texas", "utah",
              "vermont", "virginia", "washington", "west-virginia", "wisconsin", "wyoming"]

    # Try to find the actual download pattern
    found_any = False
    for base in base_urls:
        for ext in [".xlsx", ".xls", ".csv"]:
            test_state = "missouri"
            url = f"{base}{test_state}{ext}"
            try:
                r = requests.head(url, timeout=10, allow_redirects=True)
                if r.status_code == 200:
                    print(f"  Found pattern: {url}")
                    found_any = True
                    break
            except:
                pass
        if found_any:
            break

    if not found_any:
        # Try archived CDC patterns
        archive_urls = [
            "https://www.cdc.gov/nceh/lead/data/state/modata.htm",
            "https://www.cdc.gov/nceh/lead/data/tables/missouri.htm",
        ]
        for url in archive_urls:
            try:
                r = requests.get(url, timeout=15, allow_redirects=True)
                if r.status_code == 200:
                    print(f"  Found archived page: {url} (length: {len(r.text)})")
                    # Parse for download links
                    import re
                    links = re.findall(r'href="([^"]*\.(?:xlsx|xls|csv))"', r.text, re.IGNORECASE)
                    if links:
                        print(f"  Download links found: {links}")
                    else:
                        print(f"  No direct download links found in archived page")
            except Exception as e:
                print(f"  Error: {e}")

    log("CDC State Surveillance Files",
        "EXPLORED - CDC dynamically loads data; direct file downloads require manual access",
        None, "County (within state files)", "2012-2022",
        "# children tested, # with BLL >= 3.5, 5, 10 mcg/dL by county")


# ============================================================================
# 5. USGS National Geochemical Survey - Soil Lead
# ============================================================================
def fetch_usgs_soil_lead():
    """Download USGS National Geochemical Survey soil data (includes lead)."""
    print("\n\n" + "#"*70)
    print("# 5. USGS National Geochemical Survey - Soil Lead Concentrations")
    print("#"*70)

    # The full dataset is at ngdbsoil-csv.zip (54MB)
    url = "https://mrdata.usgs.gov/ngdb/soil/ngdbsoil-csv.zip"

    try:
        print(f"  Downloading USGS soil geochemistry data ({url})...")
        print(f"  (This is ~54MB, may take a minute)")
        r = requests.get(url, timeout=120, stream=True)

        if r.status_code == 200:
            # Save to temp and extract
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
                total = 0
                for chunk in r.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                    total += len(chunk)
                tmp_path = tmp.name

            print(f"  Downloaded {total/1e6:.1f} MB")

            # Extract and look for lead data
            with zipfile.ZipFile(tmp_path, 'r') as zf:
                names = zf.namelist()
                print(f"  ZIP contains: {names[:10]}...")

                # Find the main data file
                csv_files = [n for n in names if n.endswith('.csv')]
                print(f"  CSV files: {csv_files[:5]}")

                for csv_name in csv_files:
                    # Read a small sample first
                    with zf.open(csv_name) as f:
                        # Read first few lines to check columns
                        sample = f.read(10000).decode('utf-8', errors='replace')
                        cols = sample.split('\n')[0] if sample else ""

                        if 'Pb' in cols or 'pb' in cols or 'lead' in cols.lower() or 'LEAD' in cols:
                            print(f"  Found lead data in {csv_name}")
                            print(f"  Columns: {cols[:200]}")

                        # Check for lat/lon or FIPS
                        if any(x in cols.lower() for x in ['lat', 'lon', 'fips', 'county']):
                            print(f"  Has geographic coords in {csv_name}")

                # Read the full main file
                main_csv = csv_files[0] if csv_files else None
                if main_csv:
                    print(f"\n  Reading {main_csv}...")
                    with zf.open(main_csv) as f:
                        df = pd.read_csv(f, low_memory=False)
                    print(f"  Shape: {df.shape}")
                    print(f"  Columns: {list(df.columns)[:20]}")

                    # Find lead-related columns
                    lead_cols = [c for c in df.columns if 'pb' in c.lower() or 'lead' in c.lower()]
                    geo_cols = [c for c in df.columns if any(x in c.lower() for x in ['lat', 'lon', 'fips', 'county', 'state'])]
                    print(f"  Lead columns: {lead_cols}")
                    print(f"  Geographic columns: {geo_cols}")

                    if lead_cols:
                        # Extract just the lead and geographic data
                        keep_cols = geo_cols + lead_cols + [c for c in df.columns if c in ['SiteId', 'SampleId', 'Depth']]
                        keep_cols = [c for c in keep_cols if c in df.columns]
                        df_lead = df[keep_cols].copy()

                        fname = "usgs_soil_lead_concentrations.csv"
                        df_lead.to_csv(OUT / fname, index=False)

                        # Try to get basic stats
                        for lc in lead_cols[:3]:
                            print(f"\n  {lc} stats:")
                            try:
                                vals = pd.to_numeric(df[lc], errors='coerce')
                                print(f"    Mean: {vals.mean():.2f}, Median: {vals.median():.2f}")
                                print(f"    Min: {vals.min():.2f}, Max: {vals.max():.2f}")
                                print(f"    Non-null: {vals.notna().sum()}")
                            except:
                                pass

                        log("USGS Soil Geochemistry", "SUCCESS", fname,
                            "Point data (lat/lon, can aggregate to county)",
                            "2007-2013", "Soil lead (Pb) concentration in ppm/mg/kg",
                            len(df_lead))

                        # Clean up
                        os.unlink(tmp_path)
                        return df_lead

            os.unlink(tmp_path)
        else:
            print(f"  HTTP {r.status_code}")
    except Exception as e:
        print(f"  Error: {e}")

    log("USGS Soil Geochemistry", "ATTEMPTED - see error above")
    return None


# ============================================================================
# 6. EPA ECHO - Lead violations by facility (can aggregate to county)
# ============================================================================
def fetch_epa_echo():
    """Download EPA ECHO lead violation data aggregated to county."""
    print("\n\n" + "#"*70)
    print("# 6. EPA ECHO - Lead Violations by County")
    print("#"*70)

    # EPA ECHO SDWA API - Safe Drinking Water Act violations for lead
    # Get facilities with lead violations
    url = ("https://data.epa.gov/efservice/VIOLATION/CONTAMINANT_CODE/1030/"
           "CSV/rows/0:50000")

    try:
        print(f"  Fetching EPA SDWA lead violations...")
        r = requests.get(url, timeout=60)
        if r.status_code == 200 and len(r.content) > 100:
            df = pd.read_csv(io.StringIO(r.text))
            print(f"  Got {len(df)} rows")
            print(f"  Columns: {list(df.columns)[:10]}")

            fname = "epa_sdwa_lead_violations_raw.csv"
            df.to_csv(OUT / fname, index=False)
            log("EPA ECHO SDWA", "SUCCESS", fname, "Facility (can aggregate to county)",
                "varies", "Lead water violations", len(df))
            return df
        else:
            print(f"  HTTP {r.status_code}")
    except Exception as e:
        print(f"  Error from EPA Envirofacts: {e}")

    # Try ECHO REST API
    try:
        print(f"\n  Trying ECHO REST API for lead violations...")
        url2 = ("https://echodata.epa.gov/echo/sdw_rest_services.get_download?"
                "p_contaminant_code=1030&output=CSV")
        r = requests.get(url2, timeout=60)
        if r.status_code == 200 and len(r.content) > 200:
            df = pd.read_csv(io.StringIO(r.text))
            print(f"  Got {len(df)} rows")
            fname = "epa_echo_lead_violations.csv"
            df.to_csv(OUT / fname, index=False)
            log("EPA ECHO REST", "SUCCESS", fname, "Facility with county",
                "varies", "SDWA lead violations", len(df))
            return df
    except Exception as e:
        print(f"  Error: {e}")

    log("EPA ECHO", "ATTEMPTED - may need direct ECHO download")
    return None


# ============================================================================
# 7. Census ACS - Pre-1980 housing (lead paint proxy) by county
# ============================================================================
def fetch_census_housing():
    """Download Census ACS pre-1980 housing data as lead paint exposure proxy."""
    print("\n\n" + "#"*70)
    print("# 7. Census ACS - Pre-1980 Housing (Lead Paint Proxy)")
    print("#"*70)

    # ACS 5-year estimates: Year structure built
    # B25034: Year Structure Built
    # B25034_001E = Total, B25034_009E = 1970-1979, B25034_010E = 1960-1969,
    # B25034_011E = 1940-1959, B25034_012E = Before 1939 (actually now _011 in recent)

    year = 2022  # Most recent ACS

    variables = [
        "B25034_001E",  # Total housing units
        "B25034_008E",  # Built 1970 to 1979
        "B25034_009E",  # Built 1960 to 1969
        "B25034_010E",  # Built 1940 to 1959
        "B25034_011E",  # Built 1939 or earlier
        "NAME"
    ]

    url = (f"https://api.census.gov/data/{year}/acs/acs5?"
           f"get={','.join(variables)}&for=county:*&key={CENSUS_KEY}")

    try:
        print(f"  Fetching ACS {year} housing age data...")
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            data = r.json()
            df = pd.DataFrame(data[1:], columns=data[0])

            # Convert to numeric
            for col in variables[:-1]:  # skip NAME
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Calculate % pre-1980 housing
            df["total_housing"] = df["B25034_001E"]
            df["pre_1980_units"] = (df["B25034_008E"].fillna(0) +
                                     df["B25034_009E"].fillna(0) +
                                     df["B25034_010E"].fillna(0) +
                                     df["B25034_011E"].fillna(0))
            df["pct_pre1980_housing"] = (df["pre_1980_units"] / df["total_housing"] * 100).round(2)
            df["county_fips"] = df["state"] + df["county"]

            # Also get pre-1950 (higher lead risk)
            df["pre_1960_units"] = df["B25034_010E"].fillna(0) + df["B25034_011E"].fillna(0)
            df["pct_pre1960_housing"] = (df["pre_1960_units"] / df["total_housing"] * 100).round(2)
            df["pre_1940_units"] = df["B25034_011E"].fillna(0)
            df["pct_pre1940_housing"] = (df["pre_1940_units"] / df["total_housing"] * 100).round(2)

            keep = ["county_fips", "NAME", "state", "county", "total_housing",
                    "pre_1980_units", "pct_pre1980_housing",
                    "pre_1960_units", "pct_pre1960_housing",
                    "pre_1940_units", "pct_pre1940_housing"]
            df_out = df[keep]

            fname = "census_pre1980_housing_lead_proxy.csv"
            df_out.to_csv(OUT / fname, index=False)

            print(f"  Got {len(df_out)} counties")
            print(f"  Mean % pre-1980: {df_out['pct_pre1980_housing'].mean():.1f}%")
            print(f"  Mean % pre-1940: {df_out['pct_pre1940_housing'].mean():.1f}%")
            print(df_out.head())

            log("Census ACS Pre-1980 Housing", "SUCCESS", fname,
                "County (FIPS)", f"ACS {year} 5-year",
                "% pre-1980, pre-1960, pre-1940 housing (lead paint proxy)",
                len(df_out), len(df_out))
            return df_out
    except Exception as e:
        print(f"  Error: {e}")

    log("Census ACS Housing", "FAILED")
    return None


# ============================================================================
# 8. Census ACS - Poverty, demographics (confounders) by county
# ============================================================================
def fetch_census_demographics():
    """Download county-level demographics for lead exposure analysis."""
    print("\n\n" + "#"*70)
    print("# 8. Census ACS Demographics (confounders)")
    print("#"*70)

    year = 2022
    variables = [
        "B01003_001E",  # Total population
        "B19013_001E",  # Median household income
        "B17001_001E",  # Population for poverty status
        "B17001_002E",  # Below poverty level
        "B02001_003E",  # Black alone
        "B03003_003E",  # Hispanic
        "B01002_001E",  # Median age
        "B25003_001E",  # Total occupied housing
        "B25003_002E",  # Owner occupied
        "B25003_003E",  # Renter occupied
        "B25077_001E",  # Median home value
        "NAME"
    ]

    url = (f"https://api.census.gov/data/{year}/acs/acs5?"
           f"get={','.join(variables)}&for=county:*&key={CENSUS_KEY}")

    try:
        print(f"  Fetching ACS {year} demographics...")
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            data = r.json()
            df = pd.DataFrame(data[1:], columns=data[0])

            for col in variables[:-1]:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df["county_fips"] = df["state"] + df["county"]
            df["pct_poverty"] = (df["B17001_002E"] / df["B17001_001E"] * 100).round(2)
            df["pct_black"] = (df["B02001_003E"] / df["B01003_001E"] * 100).round(2)
            df["pct_hispanic"] = (df["B03003_003E"] / df["B01003_001E"] * 100).round(2)
            df["pct_renter"] = (df["B25003_003E"] / df["B25003_001E"] * 100).round(2)

            df_out = df[["county_fips", "NAME", "B01003_001E", "B19013_001E",
                         "B01002_001E", "B25077_001E",
                         "pct_poverty", "pct_black", "pct_hispanic", "pct_renter"]].copy()
            df_out.columns = ["county_fips", "name", "population", "median_income",
                              "median_age", "median_home_value",
                              "pct_poverty", "pct_black", "pct_hispanic", "pct_renter"]

            fname = "census_demographics_for_lead.csv"
            df_out.to_csv(OUT / fname, index=False)
            print(f"  Got {len(df_out)} counties")

            log("Census ACS Demographics", "SUCCESS", fname,
                "County (FIPS)", f"ACS {year} 5-year",
                "Income, poverty, race, housing tenure", len(df_out), len(df_out))
            return df_out
    except Exception as e:
        print(f"  Error: {e}")

    log("Census ACS Demographics", "FAILED")
    return None


# ============================================================================
# 9. EPA Superfund Sites - Lead contamination by county
# ============================================================================
def fetch_epa_superfund():
    """Download EPA Superfund sites with lead contamination."""
    print("\n\n" + "#"*70)
    print("# 9. EPA Superfund Sites with Lead Contamination")
    print("#"*70)

    # Try Envirofacts
    url = ("https://data.epa.gov/efservice/SEMS_ACTIVE_SITES/"
           "CONTAMINANT_NAME/LEAD/CSV/rows/0:10000")

    try:
        print(f"  Fetching EPA Superfund lead sites...")
        r = requests.get(url, timeout=60)
        if r.status_code == 200 and len(r.content) > 100:
            df = pd.read_csv(io.StringIO(r.text))
            print(f"  Got {len(df)} rows")
            print(f"  Columns: {list(df.columns)[:10]}")

            fname = "epa_superfund_lead_sites.csv"
            df.to_csv(OUT / fname, index=False)

            # Count by county if available
            county_col = [c for c in df.columns if 'county' in c.lower()]
            n_counties = df[county_col[0]].nunique() if county_col else None

            log("EPA Superfund Lead Sites", "SUCCESS", fname,
                "Site-level (with county)", "varies",
                "Superfund sites with lead contamination", len(df), n_counties)
            return df
    except Exception as e:
        print(f"  Error: {e}")

    log("EPA Superfund Lead Sites", "ATTEMPTED")
    return None


# ============================================================================
# 10. Wisconsin - County blood lead data
# ============================================================================
def fetch_wisconsin_lead():
    """Download Wisconsin county-level blood lead data."""
    print("\n\n" + "#"*70)
    print("# 10. Wisconsin Blood Lead Data")
    print("#"*70)

    # Wisconsin DHS publishes county-level data
    urls = [
        "https://www.dhs.wisconsin.gov/lead/data-tables.htm",
    ]

    try:
        # Try the open data portal
        print("  Checking Wisconsin DHS lead data...")
        r = requests.get(urls[0], timeout=15, allow_redirects=True)
        if r.status_code == 200:
            import re
            # Look for CSV/Excel download links
            links = re.findall(r'href="([^"]*(?:\.csv|\.xlsx|\.xls)[^"]*)"', r.text, re.IGNORECASE)
            links += re.findall(r'href="([^"]*lead[^"]*(?:download|data)[^"]*)"', r.text, re.IGNORECASE)
            if links:
                print(f"  Found data links: {links[:5]}")
                for link in links[:3]:
                    if not link.startswith("http"):
                        link = "https://www.dhs.wisconsin.gov" + link
                    try:
                        r2 = requests.get(link, timeout=30)
                        if r2.status_code == 200 and len(r2.content) > 200:
                            if link.endswith('.csv'):
                                df = pd.read_csv(io.StringIO(r2.text))
                            else:
                                df = pd.read_excel(io.BytesIO(r2.content))
                            print(f"  Got: {len(df)} rows from {link}")
                            fname = "wisconsin_county_blood_lead.csv"
                            df.to_csv(OUT / fname, index=False)
                            log("Wisconsin DHS", "SUCCESS", fname, "County",
                                "varies", "Blood lead levels", len(df))
                            return
                    except Exception as e:
                        print(f"  Error with {link}: {e}")
            else:
                print("  No direct download links found on page")
    except Exception as e:
        print(f"  Error: {e}")

    log("Wisconsin Lead Data", "EXPLORED - data available via DHS website")


# ============================================================================
# 11. Connecticut - County/town level lead data (data.gov)
# ============================================================================
def fetch_connecticut_lead():
    """Download Connecticut lead data from data.gov/Socrata."""
    print("\n\n" + "#"*70)
    print("# 11. Connecticut Lead Data")
    print("#"*70)

    # CT has town-level data on data.ct.gov
    urls = [
        "https://data.ct.gov/api/views/mhb5-bcrh/rows.csv?accessType=DOWNLOAD",
        "https://data.ct.gov/api/views/gctx-z3uh/rows.csv?accessType=DOWNLOAD",
    ]

    for url in urls:
        try:
            print(f"  Trying: {url}")
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 200:
                df = pd.read_csv(io.StringIO(r.text))
                print(f"  Got {len(df)} rows, columns: {list(df.columns)[:8]}")

                # Check if lead-related
                lead_cols = [c for c in df.columns if 'lead' in c.lower() or 'bll' in c.lower()]
                if lead_cols:
                    fname = "connecticut_town_blood_lead.csv"
                    df.to_csv(OUT / fname, index=False)
                    log("Connecticut Lead Data", "SUCCESS", fname,
                        "Town", "varies", str(lead_cols[:3]), len(df))
                    return
        except Exception as e:
            print(f"  Error: {e}")

    log("Connecticut Lead Data", "EXPLORED")


# ============================================================================
# 12. data.cdc.gov Socrata API - Search for lead datasets
# ============================================================================
def fetch_cdc_socrata():
    """Search data.cdc.gov for lead-related datasets."""
    print("\n\n" + "#"*70)
    print("# 12. data.cdc.gov Socrata - Lead Datasets")
    print("#"*70)

    # Search the Socrata discovery API
    url = "https://api.us.socrata.com/api/catalog/v1?q=blood+lead+county&domains=data.cdc.gov&limit=20"

    try:
        print(f"  Searching data.cdc.gov for lead datasets...")
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            print(f"  Found {len(results)} datasets")

            for i, result in enumerate(results):
                res = result.get("resource", {})
                name = res.get("name", "unknown")
                desc = res.get("description", "")[:100]
                link = result.get("link", "")
                nid = res.get("id", "")
                print(f"\n  [{i+1}] {name}")
                print(f"      ID: {nid}")
                print(f"      Desc: {desc}")

                # Try to download promising datasets
                if any(kw in name.lower() + desc.lower() for kw in ['county', 'lead', 'blood']):
                    try:
                        csv_url = f"https://data.cdc.gov/api/views/{nid}/rows.csv?accessType=DOWNLOAD"
                        print(f"      Downloading from {csv_url}...")
                        r2 = requests.get(csv_url, timeout=60)
                        if r2.status_code == 200 and len(r2.content) > 200:
                            df = pd.read_csv(io.StringIO(r2.text))
                            print(f"      Got {len(df)} rows, cols: {list(df.columns)[:6]}")

                            # Check for county-level data
                            county_cols = [c for c in df.columns if 'county' in c.lower() or 'fips' in c.lower()]
                            lead_cols = [c for c in df.columns if 'lead' in c.lower() or 'bll' in c.lower()]

                            if county_cols or lead_cols:
                                safe_name = name[:40].replace(" ", "_").replace("/", "_")
                                fname = f"cdc_socrata_{safe_name}.csv"
                                df.to_csv(OUT / fname, index=False)
                                log(f"data.cdc.gov: {name[:50]}", "SUCCESS", fname,
                                    "County" if county_cols else "varies",
                                    "varies", str(lead_cols[:3]) if lead_cols else name,
                                    len(df))
                    except Exception as e:
                        print(f"      Download error: {e}")
    except Exception as e:
        print(f"  Error: {e}")

    # Also search for PLACES county data
    try:
        print(f"\n  Also downloading PLACES county data...")
        # PLACES 2024 county data
        places_url = "https://data.cdc.gov/api/views/swc5-untb/rows.csv?accessType=DOWNLOAD"
        print(f"  Downloading PLACES county dataset (may be large)...")
        r = requests.get(places_url, timeout=120)
        if r.status_code == 200 and len(r.content) > 1000:
            df = pd.read_csv(io.StringIO(r.text), low_memory=False)
            print(f"  PLACES: {len(df)} rows, {len(df.columns)} columns")
            print(f"  Columns: {list(df.columns)[:10]}")

            # Check for lead-related measures
            if 'MeasureId' in df.columns:
                measures = df['MeasureId'].unique()
                print(f"  Measures: {measures[:20]}")
            if 'Measure' in df.columns:
                measures = df['Measure'].unique()
                lead_m = [m for m in measures if 'lead' in str(m).lower()]
                print(f"  Lead-related measures: {lead_m}")

            # Save a sample or the lead-relevant subset
            fname = "cdc_places_county_2025.csv"
            df.to_csv(OUT / fname, index=False)
            log("CDC PLACES County Data", "SUCCESS", fname,
                "County", "2025 release", "Health measures (BRFSS-based)", len(df))
    except Exception as e:
        print(f"  PLACES error: {e}")


# ============================================================================
# 13. EPA Lead and Copper Rule (LCR) water sampling data
# ============================================================================
def fetch_epa_lcr():
    """Download EPA Lead and Copper Rule water data."""
    print("\n\n" + "#"*70)
    print("# 13. EPA Lead and Copper Rule Water Sampling")
    print("#"*70)

    # EPA SDWIS data
    url = "https://data.epa.gov/efservice/LCR_SAMPLE_RESULT/CSV/rows/0:50000"

    try:
        print(f"  Fetching EPA LCR sample results...")
        r = requests.get(url, timeout=90)
        if r.status_code == 200 and len(r.content) > 200:
            df = pd.read_csv(io.StringIO(r.text))
            print(f"  Got {len(df)} rows")
            print(f"  Columns: {list(df.columns)[:10]}")

            fname = "epa_lcr_samples_county.csv"
            df.to_csv(OUT / fname, index=False)

            log("EPA LCR Water Sampling", "SUCCESS", fname,
                "Water system (can aggregate to county)", "varies",
                "Lead in drinking water (mg/L)", len(df))
            return df
    except Exception as e:
        print(f"  Error: {e}")

    log("EPA LCR Water Sampling", "ATTEMPTED")
    return None


# ============================================================================
# 14. Census - Children under 6 by county (denominator for lead rates)
# ============================================================================
def fetch_census_children():
    """Download count of children under 6 by county (for lead testing rates)."""
    print("\n\n" + "#"*70)
    print("# 14. Census - Children Under 6 by County")
    print("#"*70)

    year = 2022
    # B09001: Population Under 18, B01001_003E-007E: Male under 5, etc.
    variables = [
        "B09001_001E",  # Population under 18
        "B01001_003E",  # Male under 5
        "B01001_027E",  # Female under 5
        "B01001_004E",  # Male 5-9
        "B01001_028E",  # Female 5-9
        "B01003_001E",  # Total population
        "NAME"
    ]

    url = (f"https://api.census.gov/data/{year}/acs/acs5?"
           f"get={','.join(variables)}&for=county:*&key={CENSUS_KEY}")

    try:
        print(f"  Fetching ACS {year} children population data...")
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            data = r.json()
            df = pd.DataFrame(data[1:], columns=data[0])
            for col in variables[:-1]:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df["county_fips"] = df["state"] + df["county"]
            df["children_under_5"] = df["B01001_003E"] + df["B01001_027E"]
            df["children_under_10"] = df["children_under_5"] + df["B01001_004E"] + df["B01001_028E"]
            df["total_pop"] = df["B01003_001E"]
            df["pct_children_under5"] = (df["children_under_5"] / df["total_pop"] * 100).round(2)

            df_out = df[["county_fips", "NAME", "total_pop", "children_under_5",
                         "children_under_10", "pct_children_under5"]]

            fname = "census_children_under6_by_county.csv"
            df_out.to_csv(OUT / fname, index=False)
            print(f"  Got {len(df_out)} counties")

            log("Census Children Under 5", "SUCCESS", fname,
                "County (FIPS)", f"ACS {year}",
                "Children under 5 count", len(df_out), len(df_out))
            return df_out
    except Exception as e:
        print(f"  Error: {e}")

    log("Census Children", "FAILED")
    return None


# ============================================================================
# 15. EPA TRI - Toxic Release Inventory lead emissions by county
# ============================================================================
def fetch_epa_tri():
    """Download EPA TRI lead air emissions data."""
    print("\n\n" + "#"*70)
    print("# 15. EPA Toxics Release Inventory - Lead Emissions")
    print("#"*70)

    # TRI Explorer data
    url = ("https://data.epa.gov/efservice/TRI_FACILITY/"
           "TRI_CHEM_ID/LEAD/CSV/rows/0:50000")

    try:
        print(f"  Fetching EPA TRI lead facility data...")
        r = requests.get(url, timeout=60)
        if r.status_code == 200 and len(r.content) > 100:
            df = pd.read_csv(io.StringIO(r.text))
            print(f"  Got {len(df)} rows")
            print(f"  Columns: {list(df.columns)[:10]}")

            fname = "epa_tri_lead_facilities.csv"
            df.to_csv(OUT / fname, index=False)

            county_col = [c for c in df.columns if 'county' in c.lower()]
            n_counties = df[county_col[0]].nunique() if county_col else None

            log("EPA TRI Lead Emissions", "SUCCESS", fname,
                "Facility (with county)", "varies",
                "Lead releases to air/water/land", len(df), n_counties)
            return df
    except Exception as e:
        print(f"  Error: {e}")

    # Alternative: try TRI BASIC download
    try:
        url2 = ("https://data.epa.gov/efservice/MV_TRI_BASIC_DOWNLOAD/"
                "TRI_CHEM_ID/00743921/CSV/rows/0:50000")  # Lead CAS# 7439-92-1
        print(f"  Trying TRI basic download (CAS 7439-92-1)...")
        r = requests.get(url2, timeout=60)
        if r.status_code == 200 and len(r.content) > 100:
            df = pd.read_csv(io.StringIO(r.text))
            print(f"  Got {len(df)} rows")
            fname = "epa_tri_lead_releases.csv"
            df.to_csv(OUT / fname, index=False)
            log("EPA TRI Lead Releases", "SUCCESS", fname,
                "Facility (with county/FIPS)", "varies",
                "Lead releases (lbs)", len(df))
    except Exception as e:
        print(f"  Error: {e}")

    log("EPA TRI Lead", "ATTEMPTED")


# ============================================================================
# 16. Aggregate NY ZIP data to county level
# ============================================================================
def aggregate_ny_to_county():
    """Aggregate the NY ZIP-level data we already have to county level."""
    print("\n\n" + "#"*70)
    print("# 16. Aggregate Existing NY Data to County Level")
    print("#"*70)

    # Check what we already have
    ny_file = OUT / "cdc_blood_lead_ny_county.csv"
    if ny_file.exists():
        try:
            df = pd.read_csv(ny_file)
            print(f"  Existing NY data: {len(df)} rows")
            print(f"  Columns: {list(df.columns)}")

            # Aggregate by county and year
            if 'county' in df.columns and 'year' in df.columns:
                agg = df.groupby(['county', 'fips', 'year']).agg({
                    'tests': 'sum',
                    'total_eblls': 'sum',
                }).reset_index()
                agg['pct_elevated'] = (agg['total_eblls'] / agg['tests'] * 100).round(2)
                agg = agg[agg['tests'] > 0]

                fname = "ny_county_blood_lead_aggregated.csv"
                agg.to_csv(OUT / fname, index=False)
                print(f"  Aggregated to {len(agg)} county-year observations")
                print(f"  Counties: {agg['county'].nunique()}")
                print(f"  Years: {sorted(agg['year'].unique())}")
                print(agg.head(10))

                log("NY County Aggregated", "SUCCESS", fname,
                    "County", f"{agg['year'].min()}-{agg['year'].max()}",
                    "% children with elevated BLL", len(agg), agg['county'].nunique())
        except Exception as e:
            print(f"  Error: {e}")


# ============================================================================
# 17. California CDPH lead data
# ============================================================================
def fetch_california_lead():
    """Try to download California county-level blood lead data."""
    print("\n\n" + "#"*70)
    print("# 17. California County Blood Lead Data")
    print("#"*70)

    # CA CDPH publishes data through their open data portal
    urls = [
        "https://data.ca.gov/api/3/action/package_search?q=blood+lead+county",
        "https://data.chhs.ca.gov/api/3/action/package_search?q=blood+lead",
    ]

    for url in urls:
        try:
            print(f"  Searching: {url}")
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                results = data.get("result", {}).get("results", [])
                print(f"  Found {len(results)} datasets")
                for res in results:
                    name = res.get("title", "unknown")
                    print(f"  - {name}")
                    for resource in res.get("resources", []):
                        if resource.get("format", "").upper() == "CSV":
                            dl_url = resource.get("url", "")
                            print(f"    CSV: {dl_url}")
                            try:
                                r2 = requests.get(dl_url, timeout=30)
                                if r2.status_code == 200:
                                    df = pd.read_csv(io.StringIO(r2.text))
                                    print(f"    Got {len(df)} rows, cols: {list(df.columns)[:6]}")
                                    safe = name[:30].replace(" ", "_")
                                    fname = f"california_{safe}.csv"
                                    df.to_csv(OUT / fname, index=False)
                                    log(f"California: {name[:40]}", "SUCCESS", fname,
                                        "varies", "varies", name, len(df))
                            except Exception as e:
                                print(f"    Error: {e}")
        except Exception as e:
            print(f"  Error: {e}")

    log("California Lead Data", "SEARCHED")


# ============================================================================
# 18. HUD Lead-Based Paint data
# ============================================================================
def fetch_hud_lead_paint():
    """Try to download HUD lead-based paint hazard data."""
    print("\n\n" + "#"*70)
    print("# 18. HUD Lead-Based Paint Data")
    print("#"*70)

    # HUD has lead hazard data; try their data portal
    urls = [
        "https://hudgis-hud.opendata.arcgis.com/api/v3/datasets?q=lead",
        "https://services.arcgis.com/VTyQ9soqVukalItT/arcgis/rest/services/Lead_Based_Paint/FeatureServer/0/query?where=1=1&outFields=*&f=json&resultRecordCount=100",
    ]

    for url in urls:
        try:
            print(f"  Trying: {url[:80]}")
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                if "data" in data:
                    print(f"  Found {len(data['data'])} datasets")
                    for item in data.get("data", [])[:5]:
                        attrs = item.get("attributes", {})
                        print(f"  - {attrs.get('name', 'unknown')}")
                elif "features" in data:
                    features = data["features"]
                    print(f"  Got {len(features)} features")
                    if features:
                        print(f"  Fields: {list(features[0].get('attributes', {}).keys())[:8]}")
        except Exception as e:
            print(f"  Error: {e}")

    log("HUD Lead Paint", "EXPLORED")


# ============================================================================
# 19. CDC WONDER - Environmental health data
# ============================================================================
def fetch_cdc_wonder():
    """Check CDC WONDER for county-level environmental health data."""
    print("\n\n" + "#"*70)
    print("# 19. CDC WONDER / Environmental Health")
    print("#"*70)

    # WONDER API for county-level cause of death data (to get suicide rates for matching)
    # This is complementary - not lead data but the outcome variable
    print("  CDC WONDER requires agreement to terms; skipping automated download.")
    print("  Suicide rates by county already obtained through other means.")
    log("CDC WONDER", "SKIPPED - requires terms agreement")


# ============================================================================
# 20. Compile and merge all county-level data
# ============================================================================
def compile_master_dataset():
    """Merge all county-level datasets into a master file."""
    print("\n\n" + "#"*70)
    print("# 20. COMPILING MASTER COUNTY-LEVEL LEAD DATASET")
    print("#"*70)

    master = None

    # Start with housing data (most complete county coverage)
    housing_file = OUT / "census_pre1980_housing_lead_proxy.csv"
    if housing_file.exists():
        master = pd.read_csv(housing_file)
        print(f"  Base: housing data with {len(master)} counties")

    # Merge demographics
    demo_file = OUT / "census_demographics_for_lead.csv"
    if demo_file.exists():
        demo = pd.read_csv(demo_file)
        if master is not None:
            master = master.merge(demo, on="county_fips", how="left", suffixes=("", "_demo"))
            print(f"  + demographics: {len(master)} rows")

    # Merge children counts
    child_file = OUT / "census_children_under6_by_county.csv"
    if child_file.exists():
        child = pd.read_csv(child_file)
        if master is not None:
            master = master.merge(child[["county_fips", "children_under_5", "children_under_10"]],
                                  on="county_fips", how="left")
            print(f"  + children: {len(master)} rows")

    # Merge existing county data we already have
    existing_files = {
        "real_county_dataset.csv": "county_fips",
        "epa_superfund_by_county.csv": "county_fips",
    }

    for fname, key in existing_files.items():
        fpath = OUT / fname
        if fpath.exists():
            try:
                df = pd.read_csv(fpath)
                if key in df.columns and master is not None:
                    # Only merge columns we don't already have
                    new_cols = [c for c in df.columns if c not in master.columns or c == key]
                    if len(new_cols) > 1:
                        master = master.merge(df[new_cols], on=key, how="left")
                        print(f"  + {fname}: {len(new_cols)-1} new columns")
            except Exception as e:
                print(f"  Error with {fname}: {e}")

    if master is not None:
        fname = "county_lead_exposure_master.csv"
        master.to_csv(OUT / fname, index=False)
        print(f"\n  MASTER DATASET: {master.shape[0]} rows x {master.shape[1]} columns")
        print(f"  Columns: {list(master.columns)}")
        log("MASTER COMPILATION", "SUCCESS", fname,
            "County (FIPS)", "Various", "Multiple lead exposure indicators",
            len(master), len(master))
        return master

    log("MASTER COMPILATION", "NO DATA TO COMPILE")
    return None


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("COUNTY-LEVEL LEAD EXPOSURE DATA COLLECTION")
    print("="*70)
    print(f"Output directory: {OUT}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Run all data collection functions
    fetch_cdc_epht()
    fetch_ny_blood_lead()
    fetch_michigan_lead()
    fetch_cdc_state_surveillance()
    fetch_usgs_soil_lead()
    fetch_epa_echo()
    fetch_census_housing()
    fetch_census_demographics()
    fetch_epa_superfund()
    fetch_wisconsin_lead()
    fetch_connecticut_lead()
    fetch_cdc_socrata()
    fetch_epa_lcr()
    fetch_census_children()
    fetch_epa_tri()
    aggregate_ny_to_county()
    fetch_california_lead()
    fetch_hud_lead_paint()
    fetch_cdc_wonder()

    # Compile master dataset
    compile_master_dataset()

    # Print summary
    print("\n\n" + "="*70)
    print("FINAL SUMMARY OF ALL DATA SOURCES")
    print("="*70)

    for entry in results_log:
        print(f"\n  {entry['source']}")
        print(f"    Status: {entry['status']}")
        if entry.get('filename'):
            fpath = OUT / entry['filename']
            size = fpath.stat().st_size / 1024 if fpath.exists() else 0
            print(f"    File: {entry['filename']} ({size:.0f} KB)")
        if entry.get('geo_level'):
            print(f"    Geo: {entry['geo_level']}")
        if entry.get('key_variable'):
            print(f"    Key var: {entry['key_variable']}")
        if entry.get('n_counties'):
            print(f"    Counties: {entry['n_counties']}")

    # Save log
    log_df = pd.DataFrame(results_log)
    log_df.to_csv(OUT / "county_lead_data_sources_log.csv", index=False)
    print(f"\n\nLog saved to: {OUT / 'county_lead_data_sources_log.csv'}")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
