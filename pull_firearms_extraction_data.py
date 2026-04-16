#!/usr/bin/env python3
"""
Pull firearm dealer counts and extraction employment change data.

1. Firearm dealers by county (Census CBP NAICS 45111 as proxy for gun stores)
   - Computes dealers_per_10k using ACS population data
2. Extraction employment change 2012 vs 2021 (NAICS 21)

Data sources:
  - Census County Business Patterns API (2021 and 2012)
  - Existing real_census_acs_data.csv for population denominators
"""

import csv
import json
import os
import sys
import time
import urllib.request
import urllib.error

BASE_DIR = "/Users/bobbarclay/Documents/soldiers"
import os as _os
API_KEY = _os.environ.get("CENSUS_API_KEY")
if not API_KEY:
    raise RuntimeError("Set CENSUS_API_KEY in your environment (see .env.example). Get a key free from https://api.census.gov/data/key_signup.html")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_json(url, label="data", retries=3):
    """Fetch JSON from a URL with retries."""
    for attempt in range(retries):
        try:
            print(f"  Fetching {label} (attempt {attempt+1})...")
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Mozilla/5.0 (research)")
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
                print(f"  -> Got {len(data)} rows (including header)")
                return data
        except urllib.error.HTTPError as e:
            print(f"  HTTP error {e.code}: {e.reason}")
            if attempt < retries - 1:
                time.sleep(3)
            else:
                raise
        except Exception as e:
            print(f"  Error: {e}")
            if attempt < retries - 1:
                time.sleep(3)
            else:
                raise


def load_population():
    """Load county population from existing ACS data."""
    pop = {}
    path = os.path.join(BASE_DIR, "real_census_acs_data.csv")
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fips = row["FIPS"]
            try:
                pop[fips] = float(row["total_population"])
            except (ValueError, KeyError):
                pass
    print(f"Loaded population for {len(pop)} counties")
    return pop


def fips_from_parts(state, county):
    """Build 5-digit FIPS from state and county codes."""
    return str(state).zfill(2) + str(county).zfill(3)


# ---------------------------------------------------------------------------
# 1. Firearm dealers by county (sporting goods stores as proxy)
# ---------------------------------------------------------------------------

def pull_firearm_dealers():
    """Pull sporting goods store data as a proxy for firearm dealers."""
    print("\n" + "=" * 70)
    print("PULLING FIREARM DEALER PROXY DATA (NAICS 45111 - Sporting Goods Stores)")
    print("=" * 70)

    pop = load_population()

    # Try multiple NAICS codes for gun-related retail
    # 45111 = Sporting Goods Stores (includes gun shops)
    # 451110 = Sporting Goods Stores (6-digit, same category)
    # We'll also try 423910 = Sporting & Recreational Goods Wholesalers

    results = {}

    # --- Try NAICS 45111 (Sporting Goods Stores) for 2021 ---
    url_45111 = (
        f"https://api.census.gov/data/2021/cbp?"
        f"get=NAICS2017,ESTAB,EMP&for=county:*&NAICS2017=45111"
        f"&key={API_KEY}"
    )
    data_45111 = None
    try:
        data_45111 = fetch_json(url_45111, "NAICS 45111 (Sporting Goods Stores, 2021)")
    except Exception as e:
        print(f"  Failed to fetch NAICS 45111: {e}")

    # --- Try 451110 (more specific 6-digit) ---
    url_451110 = (
        f"https://api.census.gov/data/2021/cbp?"
        f"get=NAICS2017,ESTAB,EMP&for=county:*&NAICS2017=451110"
        f"&key={API_KEY}"
    )
    data_451110 = None
    try:
        data_451110 = fetch_json(url_451110, "NAICS 451110 (Sporting Goods Stores 6-digit, 2021)")
    except Exception as e:
        print(f"  Failed to fetch NAICS 451110: {e}")

    # --- Try 423910 (Sporting Goods Wholesalers) ---
    url_423910 = (
        f"https://api.census.gov/data/2021/cbp?"
        f"get=NAICS2017,ESTAB,EMP&for=county:*&NAICS2017=423910"
        f"&key={API_KEY}"
    )
    data_423910 = None
    try:
        data_423910 = fetch_json(url_423910, "NAICS 423910 (Sporting Goods Wholesalers, 2021)")
    except Exception as e:
        print(f"  Failed to fetch NAICS 423910: {e}")

    # Process results - prefer 451110, then 45111
    primary_data = data_451110 or data_45111
    primary_label = "451110" if data_451110 else "45111"

    if primary_data:
        header = primary_data[0]
        naics_idx = header.index("NAICS2017")
        estab_idx = header.index("ESTAB")
        emp_idx = header.index("EMP")
        state_idx = header.index("state")
        county_idx = header.index("county")

        for row in primary_data[1:]:
            fips = fips_from_parts(row[state_idx], row[county_idx])
            try:
                estab = int(row[estab_idx])
                emp = int(row[emp_idx]) if row[emp_idx] else 0
            except (ValueError, TypeError):
                estab = 0
                emp = 0
            results[fips] = {
                "naics": row[naics_idx],
                "sporting_goods_estab": estab,
                "sporting_goods_emp": emp,
                "wholesale_estab": 0,
                "wholesale_emp": 0,
            }
        print(f"  Primary data ({primary_label}): {len(results)} counties")

    # Add wholesale data if available
    if data_423910:
        header = data_423910[0]
        estab_idx = header.index("ESTAB")
        emp_idx = header.index("EMP")
        state_idx = header.index("state")
        county_idx = header.index("county")

        for row in data_423910[1:]:
            fips = fips_from_parts(row[state_idx], row[county_idx])
            try:
                estab = int(row[estab_idx])
                emp = int(row[emp_idx]) if row[emp_idx] else 0
            except (ValueError, TypeError):
                estab = 0
                emp = 0
            if fips in results:
                results[fips]["wholesale_estab"] = estab
                results[fips]["wholesale_emp"] = emp
            else:
                results[fips] = {
                    "naics": "423910",
                    "sporting_goods_estab": 0,
                    "sporting_goods_emp": 0,
                    "wholesale_estab": estab,
                    "wholesale_emp": emp,
                }

    # Compute per capita rates and write CSV
    outpath = os.path.join(BASE_DIR, "firearm_dealers_by_county.csv")
    fieldnames = [
        "FIPS", "state_fips", "county_fips",
        "sporting_goods_estab", "sporting_goods_emp",
        "wholesale_estab", "wholesale_emp",
        "total_gun_related_estab", "total_gun_related_emp",
        "population", "dealers_per_10k",
        "naics_source"
    ]

    rows_written = 0
    with_pop = 0
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for fips in sorted(results.keys()):
            r = results[fips]
            state_fips = fips[:2]
            county_fips = fips[2:]

            total_estab = r["sporting_goods_estab"] + r["wholesale_estab"]
            total_emp = r["sporting_goods_emp"] + r["wholesale_emp"]
            population = pop.get(fips, None)

            dealers_per_10k = ""
            if population and population > 0:
                dealers_per_10k = round(total_estab / (population / 10000), 4)
                with_pop += 1

            writer.writerow({
                "FIPS": fips,
                "state_fips": state_fips,
                "county_fips": county_fips,
                "sporting_goods_estab": r["sporting_goods_estab"],
                "sporting_goods_emp": r["sporting_goods_emp"],
                "wholesale_estab": r["wholesale_estab"],
                "wholesale_emp": r["wholesale_emp"],
                "total_gun_related_estab": total_estab,
                "total_gun_related_emp": total_emp,
                "population": int(population) if population else "",
                "dealers_per_10k": dealers_per_10k,
                "naics_source": primary_label,
            })
            rows_written += 1

    print(f"\nWrote {rows_written} counties to {outpath}")
    print(f"  {with_pop} counties matched with population data")

    # Print some summary stats
    per_10k_vals = []
    for fips, r in results.items():
        total_estab = r["sporting_goods_estab"] + r["wholesale_estab"]
        population = pop.get(fips, None)
        if population and population > 0:
            per_10k_vals.append(total_estab / (population / 10000))
    if per_10k_vals:
        per_10k_vals.sort()
        print(f"\n  dealers_per_10k summary:")
        print(f"    Min:    {per_10k_vals[0]:.4f}")
        print(f"    Median: {per_10k_vals[len(per_10k_vals)//2]:.4f}")
        print(f"    Mean:   {sum(per_10k_vals)/len(per_10k_vals):.4f}")
        print(f"    Max:    {per_10k_vals[-1]:.4f}")
        print(f"    N:      {len(per_10k_vals)}")

    return outpath


# ---------------------------------------------------------------------------
# 2. Extraction employment change 2012 vs 2021
# ---------------------------------------------------------------------------

def pull_extraction_change():
    """Pull NAICS 21 (Mining/Extraction) for 2012 and 2021, compute change."""
    print("\n" + "=" * 70)
    print("PULLING EXTRACTION EMPLOYMENT CHANGE (NAICS 21, 2012 vs 2021)")
    print("=" * 70)

    pop = load_population()

    # --- 2021 data ---
    url_2021 = (
        f"https://api.census.gov/data/2021/cbp?"
        f"get=NAICS2017,EMP,ESTAB&for=county:*&NAICS2017=21"
        f"&key={API_KEY}"
    )
    data_2021 = fetch_json(url_2021, "NAICS 21 (Mining, 2021)")

    # --- 2012 data ---
    url_2012 = (
        f"https://api.census.gov/data/2012/cbp?"
        f"get=NAICS2012,EMP,ESTAB&for=county:*&NAICS2012=21"
        f"&key={API_KEY}"
    )
    data_2012 = fetch_json(url_2012, "NAICS 21 (Mining, 2012)")

    # Parse 2021
    records_2021 = {}
    if data_2021:
        header = data_2021[0]
        emp_idx = header.index("EMP")
        estab_idx = header.index("ESTAB")
        state_idx = header.index("state")
        county_idx = header.index("county")

        for row in data_2021[1:]:
            fips = fips_from_parts(row[state_idx], row[county_idx])
            try:
                emp = int(row[emp_idx]) if row[emp_idx] else 0
                estab = int(row[estab_idx]) if row[estab_idx] else 0
            except (ValueError, TypeError):
                emp, estab = 0, 0
            records_2021[fips] = {"emp_2021": emp, "estab_2021": estab}
        print(f"  2021 data: {len(records_2021)} counties")

    # Parse 2012
    records_2012 = {}
    if data_2012:
        header = data_2012[0]
        emp_idx = header.index("EMP")
        estab_idx = header.index("ESTAB")
        state_idx = header.index("state")
        county_idx = header.index("county")

        for row in data_2012[1:]:
            fips = fips_from_parts(row[state_idx], row[county_idx])
            try:
                emp = int(row[emp_idx]) if row[emp_idx] else 0
                estab = int(row[estab_idx]) if row[estab_idx] else 0
            except (ValueError, TypeError):
                emp, estab = 0, 0
            records_2012[fips] = {"emp_2012": emp, "estab_2012": estab}
        print(f"  2012 data: {len(records_2012)} counties")

    # Merge
    all_fips = sorted(set(records_2021.keys()) | set(records_2012.keys()))
    print(f"  Total unique counties: {len(all_fips)}")

    outpath = os.path.join(BASE_DIR, "extraction_employment_change_2012_2021.csv")
    fieldnames = [
        "FIPS", "state_fips", "county_fips",
        "emp_2012", "estab_2012",
        "emp_2021", "estab_2021",
        "extraction_emp_change", "extraction_pct_change",
        "estab_change", "estab_pct_change",
        "population",
        "extraction_emp_per_10k_2021",
    ]

    rows_written = 0
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for fips in all_fips:
            state_fips = fips[:2]
            county_fips = fips[2:]

            r12 = records_2012.get(fips, {"emp_2012": 0, "estab_2012": 0})
            r21 = records_2021.get(fips, {"emp_2021": 0, "estab_2021": 0})

            emp_2012 = r12["emp_2012"]
            emp_2021 = r21["emp_2021"]
            estab_2012 = r12["estab_2012"]
            estab_2021 = r21["estab_2021"]

            emp_change = emp_2021 - emp_2012
            estab_change = estab_2021 - estab_2012

            emp_pct = ""
            if emp_2012 > 0:
                emp_pct = round((emp_change / emp_2012) * 100, 2)

            estab_pct = ""
            if estab_2012 > 0:
                estab_pct = round((estab_change / estab_2012) * 100, 2)

            population = pop.get(fips, None)
            emp_per_10k = ""
            if population and population > 0:
                emp_per_10k = round(emp_2021 / (population / 10000), 4)

            writer.writerow({
                "FIPS": fips,
                "state_fips": state_fips,
                "county_fips": county_fips,
                "emp_2012": emp_2012,
                "estab_2012": estab_2012,
                "emp_2021": emp_2021,
                "estab_2021": estab_2021,
                "extraction_emp_change": emp_change,
                "extraction_pct_change": emp_pct,
                "estab_change": estab_change,
                "estab_pct_change": estab_pct,
                "population": int(population) if population else "",
                "extraction_emp_per_10k_2021": emp_per_10k,
            })
            rows_written += 1

    print(f"\nWrote {rows_written} counties to {outpath}")

    # Summary stats
    changes = [records_2021.get(fips, {"emp_2021": 0})["emp_2021"] -
               records_2012.get(fips, {"emp_2012": 0})["emp_2012"]
               for fips in all_fips]
    pct_changes = []
    for fips in all_fips:
        e12 = records_2012.get(fips, {"emp_2012": 0})["emp_2012"]
        e21 = records_2021.get(fips, {"emp_2021": 0})["emp_2021"]
        if e12 > 0:
            pct_changes.append(((e21 - e12) / e12) * 100)

    print(f"\n  Employment change summary:")
    print(f"    Counties with 2012 data: {len(records_2012)}")
    print(f"    Counties with 2021 data: {len(records_2021)}")
    if changes:
        print(f"    Mean emp change:     {sum(changes)/len(changes):.1f}")
        print(f"    Total emp change:    {sum(changes)}")
    if pct_changes:
        pct_changes.sort()
        print(f"    Median pct change:   {pct_changes[len(pct_changes)//2]:.1f}%")
        print(f"    Mean pct change:     {sum(pct_changes)/len(pct_changes):.1f}%")

    # How many counties lost extraction jobs?
    lost = sum(1 for c in changes if c < 0)
    gained = sum(1 for c in changes if c > 0)
    same = sum(1 for c in changes if c == 0)
    print(f"    Counties that lost jobs:   {lost}")
    print(f"    Counties that gained jobs: {gained}")
    print(f"    Counties unchanged:        {same}")

    return outpath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("PULLING FIREARM DEALER & EXTRACTION EMPLOYMENT DATA")
    print("=" * 70)

    try:
        path1 = pull_firearm_dealers()
    except Exception as e:
        print(f"\nERROR pulling firearm dealer data: {e}")
        import traceback
        traceback.print_exc()
        path1 = None

    try:
        path2 = pull_extraction_change()
    except Exception as e:
        print(f"\nERROR pulling extraction data: {e}")
        import traceback
        traceback.print_exc()
        path2 = None

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    if path1:
        print(f"  1. {path1}")
    if path2:
        print(f"  2. {path2}")
    print()
