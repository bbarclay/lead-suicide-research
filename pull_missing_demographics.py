#!/usr/bin/env python3
"""
Pull Census ACS 2022 5-Year data for all US counties:
  1. Median age (B01002)
  2. Race/ethnicity (B03002)
  3. Household structure / living alone (B11001)

Saves each to a CSV in the working directory.
"""

import csv
import io
import json
import os
import sys
import urllib.request
import urllib.error

import os as _os
API_KEY = _os.environ.get("CENSUS_API_KEY")
if not API_KEY:
    raise RuntimeError("Set CENSUS_API_KEY in your environment (see .env.example). Get a key free from https://api.census.gov/data/key_signup.html")
BASE_URL = "https://api.census.gov/data/2022/acs/acs5"
OUT_DIR = "/Users/bobbarclay/Documents/soldiers"


def fetch_census(variables, description=""):
    """Fetch variables for all counties from the Census ACS API."""
    var_str = ",".join(variables)
    url = (
        f"{BASE_URL}?get=NAME,{var_str}"
        f"&for=county:*&in=state:*"
        f"&key={API_KEY}"
    )
    print(f"Fetching {description}...")
    print(f"  URL: {url[:120]}...")
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        print(f"  Received {len(data) - 1} county rows (plus header)")
        return data
    except urllib.error.HTTPError as e:
        print(f"  HTTP Error {e.code}: {e.reason}")
        body = e.read().decode("utf-8", errors="replace")
        print(f"  Response: {body[:500]}")
        sys.exit(1)
    except Exception as e:
        print(f"  Error: {e}")
        sys.exit(1)


def make_fips(state, county):
    """Create 5-digit FIPS code from state and county codes."""
    return state.zfill(2) + county.zfill(3)


def safe_float(val):
    """Convert to float, returning None for missing/null values."""
    if val is None or val == "" or val == "null":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def pct(numerator, denominator):
    """Compute percentage, returning None if denominator is zero or missing."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return round((numerator / denominator) * 100, 2)


# ─────────────────────────────────────────────
# 1. MEDIAN AGE (B01002)
# ─────────────────────────────────────────────
def pull_median_age():
    variables = ["B01002_001E"]
    raw = fetch_census(variables, "Median Age (B01002)")
    header = raw[0]

    rows = []
    for row in raw[1:]:
        name = row[header.index("NAME")]
        state = row[header.index("state")]
        county = row[header.index("county")]
        median_age = safe_float(row[header.index("B01002_001E")])
        fips = make_fips(state, county)
        rows.append({
            "FIPS": fips,
            "county_name": name,
            "median_age": median_age,
        })

    out_path = os.path.join(OUT_DIR, "census_median_age_by_county.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["FIPS", "county_name", "median_age"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved {len(rows)} rows to {out_path}")
    return rows


# ─────────────────────────────────────────────
# 2. RACE / ETHNICITY (B03002)
# ─────────────────────────────────────────────
def pull_race_ethnicity():
    variables = [
        "B03002_001E",  # total
        "B03002_003E",  # White alone, not Hispanic
        "B03002_004E",  # Black alone, not Hispanic
        "B03002_005E",  # American Indian/Alaska Native alone, not Hispanic
        "B03002_012E",  # Hispanic/Latino
    ]
    raw = fetch_census(variables, "Race/Ethnicity (B03002)")
    header = raw[0]

    rows = []
    for row in raw[1:]:
        name = row[header.index("NAME")]
        state = row[header.index("state")]
        county = row[header.index("county")]
        fips = make_fips(state, county)

        total = safe_float(row[header.index("B03002_001E")])
        white_nh = safe_float(row[header.index("B03002_003E")])
        black = safe_float(row[header.index("B03002_004E")])
        native = safe_float(row[header.index("B03002_005E")])
        hispanic = safe_float(row[header.index("B03002_012E")])

        rows.append({
            "FIPS": fips,
            "county_name": name,
            "total_population": total,
            "white_nh": white_nh,
            "black_nh": black,
            "native_american_nh": native,
            "hispanic_latino": hispanic,
            "pct_white_nh": pct(white_nh, total),
            "pct_black": pct(black, total),
            "pct_native_american": pct(native, total),
            "pct_hispanic": pct(hispanic, total),
        })

    out_path = os.path.join(OUT_DIR, "census_race_ethnicity_by_county.csv")
    fieldnames = [
        "FIPS", "county_name", "total_population",
        "white_nh", "black_nh", "native_american_nh", "hispanic_latino",
        "pct_white_nh", "pct_black", "pct_native_american", "pct_hispanic",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved {len(rows)} rows to {out_path}")
    return rows


# ─────────────────────────────────────────────
# 3. LIVING ALONE (B11001)
# ─────────────────────────────────────────────
def pull_living_alone():
    variables = [
        "B11001_001E",  # total households
        "B11001_008E",  # nonfamily householder living alone
    ]
    raw = fetch_census(variables, "Living Alone (B11001)")
    header = raw[0]

    rows = []
    for row in raw[1:]:
        name = row[header.index("NAME")]
        state = row[header.index("state")]
        county = row[header.index("county")]
        fips = make_fips(state, county)

        total_hh = safe_float(row[header.index("B11001_001E")])
        living_alone = safe_float(row[header.index("B11001_008E")])

        rows.append({
            "FIPS": fips,
            "county_name": name,
            "total_households": total_hh,
            "living_alone": living_alone,
            "pct_living_alone": pct(living_alone, total_hh),
        })

    out_path = os.path.join(OUT_DIR, "census_living_alone_by_county.csv")
    fieldnames = [
        "FIPS", "county_name", "total_households",
        "living_alone", "pct_living_alone",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved {len(rows)} rows to {out_path}")
    return rows


# ─────────────────────────────────────────────
# SUMMARY STATISTICS
# ─────────────────────────────────────────────
def summarize(label, values):
    """Print summary stats for a list of numeric values."""
    clean = [v for v in values if v is not None]
    if not clean:
        print(f"  {label}: no valid data")
        return
    n = len(clean)
    mean_val = sum(clean) / n
    sorted_v = sorted(clean)
    median_val = sorted_v[n // 2] if n % 2 == 1 else (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2
    print(f"  {label}:")
    print(f"    N = {n}, Min = {sorted_v[0]:.2f}, Max = {sorted_v[-1]:.2f}")
    print(f"    Mean = {mean_val:.2f}, Median = {median_val:.2f}")


def main():
    print("=" * 60)
    print("Census ACS 2022 5-Year Data Pull for All US Counties")
    print("=" * 60)
    print()

    age_rows = pull_median_age()
    print()
    race_rows = pull_race_ethnicity()
    print()
    alone_rows = pull_living_alone()

    print()
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    summarize("Median Age", [r["median_age"] for r in age_rows])
    print()
    summarize("Pct White (non-Hispanic)", [r["pct_white_nh"] for r in race_rows])
    summarize("Pct Black (non-Hispanic)", [r["pct_black"] for r in race_rows])
    summarize("Pct Native American (non-Hispanic)", [r["pct_native_american"] for r in race_rows])
    summarize("Pct Hispanic/Latino", [r["pct_hispanic"] for r in race_rows])
    print()
    summarize("Pct Living Alone", [r["pct_living_alone"] for r in alone_rows])

    print()
    print("=" * 60)
    print("ALL DONE. Files saved:")
    print(f"  1. {os.path.join(OUT_DIR, 'census_median_age_by_county.csv')}")
    print(f"  2. {os.path.join(OUT_DIR, 'census_race_ethnicity_by_county.csv')}")
    print(f"  3. {os.path.join(OUT_DIR, 'census_living_alone_by_county.csv')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
