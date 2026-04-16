"""
Pull veteran detail data from Census ACS 5-Year 2022 API.

Tables:
  B21002 - Veterans by Period of Service (by county)
  B21005 - Veteran Status by Age by Employment Status (by county)

Output:
  veteran_period_of_service_by_county.csv
  veteran_employment_by_county.csv
"""

import urllib.request
import json
import csv
import sys
import time

import os as _os
API_KEY = _os.environ.get("CENSUS_API_KEY")
if not API_KEY:
    raise RuntimeError("Set CENSUS_API_KEY in your environment (see .env.example). Get a key free from https://api.census.gov/data/key_signup.html")
BASE_URL = "https://api.census.gov/data/2022/acs/acs5"


def fetch_census(variables, geo="county:*", in_geo="", retries=3):
    """Fetch data from Census API with retries."""
    var_str = ",".join(variables)
    url = f"{BASE_URL}?get=NAME,{var_str}&for={geo}"
    if in_geo:
        url += f"&in={in_geo}"
    url += f"&key={API_KEY}"

    for attempt in range(retries):
        try:
            print(f"  Requesting: {url[:120]}...")
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            print(f"  Got {len(data)-1} rows")
            return data
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(5)
            else:
                raise


def pull_b21002():
    """
    B21002: Period of Military Service for Civilian Veterans 18+

    Key variables (estimates only, suffix E):
      B21002_001E  Total civilian veterans 18+
      B21002_002E  Served September 2001 or later (Gulf War II / post-9/11)
      B21002_003E  Served September 2001 or later AND August 1990 to August 2001 (both Gulf War periods)
      B21002_004E  Served September 2001 or later AND other period
      B21002_005E  Gulf War (August 1990 to August 2001), no earlier period
      B21002_006E  Gulf War AND Vietnam era
      B21002_007E  Gulf War AND other period
      B21002_008E  Vietnam era, no other period
      B21002_009E  Vietnam era AND Korean War
      B21002_010E  Vietnam era AND other period
      B21002_011E  Korean War, no other period
      B21002_012E  World War II, no other period
      B21002_013E  Other period only
    """
    print("\n=== Pulling Table B21002: Veterans by Period of Service ===")

    variables = [f"B21002_{i:03d}E" for i in range(1, 14)]
    data = fetch_census(variables)

    header = data[0]
    rows = data[1:]

    # Build output rows
    output = []
    for row in rows:
        # Map columns by header
        d = dict(zip(header, row))

        state_fips = d.get("state", "")
        county_fips = d.get("county", "")
        fips = state_fips + county_fips
        county_name = d.get("NAME", "")

        def safe_int(key):
            val = d.get(key, None)
            if val is None or val == "":
                return 0
            try:
                return int(val)
            except (ValueError, TypeError):
                return 0

        total = safe_int("B21002_001E")

        # Gulf War II (post-9/11) = served Sep 2001 or later (all sub-categories)
        # B21002_002E = Sep 2001+ only
        # B21002_003E = Sep 2001+ AND Aug 1990 - Aug 2001
        # B21002_004E = Sep 2001+ AND other period
        gw2_only = safe_int("B21002_002E")
        gw2_and_gw1 = safe_int("B21002_003E")
        gw2_and_other = safe_int("B21002_004E")
        gulf_war_2 = gw2_only + gw2_and_gw1 + gw2_and_other

        # Gulf War I (Aug 1990 - Aug 2001, no Sep 2001+ service)
        # B21002_005E = GW1 no earlier
        # B21002_006E = GW1 AND Vietnam
        # B21002_007E = GW1 AND other
        gw1_only = safe_int("B21002_005E")
        gw1_and_vietnam = safe_int("B21002_006E")
        gw1_and_other = safe_int("B21002_007E")
        gulf_war_1 = gw1_only + gw1_and_vietnam + gw1_and_other

        # Vietnam era (no GW service)
        # B21002_008E = Vietnam only
        # B21002_009E = Vietnam AND Korean War
        # B21002_010E = Vietnam AND other
        vietnam_only = safe_int("B21002_008E")
        vietnam_and_korea = safe_int("B21002_009E")
        vietnam_and_other = safe_int("B21002_010E")
        vietnam = vietnam_only + vietnam_and_korea + vietnam_and_other

        # Korean War (no Vietnam or GW)
        korean = safe_int("B21002_011E")

        # WWII
        wwii = safe_int("B21002_012E")

        # Other period only
        other = safe_int("B21002_013E")

        # Percent post-9/11
        pct_post911 = (gulf_war_2 / total * 100) if total > 0 else 0.0

        output.append({
            "FIPS": fips,
            "county_name": county_name,
            "total_veterans": total,
            "gulf_war_2_vets": gulf_war_2,
            "gulf_war_1_vets": gulf_war_1,
            "vietnam_vets": vietnam,
            "korean_war_vets": korean,
            "wwii_vets": wwii,
            "other_period_vets": other,
            "pct_post911_vets": round(pct_post911, 2),
        })

    # Save CSV
    outpath = "/Users/bobbarclay/Documents/soldiers/veteran_period_of_service_by_county.csv"
    fieldnames = [
        "FIPS", "county_name", "total_veterans",
        "gulf_war_2_vets", "gulf_war_1_vets", "vietnam_vets",
        "korean_war_vets", "wwii_vets", "other_period_vets",
        "pct_post911_vets",
    ]
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output)

    print(f"\n  Saved {len(output)} counties to {outpath}")

    # Summary statistics
    totals = [r["total_veterans"] for r in output if r["total_veterans"] > 0]
    gw2s = [r["gulf_war_2_vets"] for r in output]
    pcts = [r["pct_post911_vets"] for r in output if r["total_veterans"] > 0]

    print("\n  --- B21002 Summary Statistics ---")
    print(f"  Counties with data: {len(totals)}")
    print(f"  Total veterans across all counties: {sum(totals):,}")
    print(f"  Mean veterans per county: {sum(totals)/len(totals):,.0f}")
    print(f"  Median veterans per county: {sorted(totals)[len(totals)//2]:,}")
    print(f"  Total post-9/11 (Gulf War II) veterans: {sum(gw2s):,}")
    print(f"  Mean pct_post911_vets: {sum(pcts)/len(pcts):.2f}%")
    print(f"  Min pct_post911_vets: {min(pcts):.2f}%")
    print(f"  Max pct_post911_vets: {max(pcts):.2f}%")

    # Top 10 counties by total veterans
    top10 = sorted(output, key=lambda x: x["total_veterans"], reverse=True)[:10]
    print("\n  Top 10 counties by total veterans:")
    for r in top10:
        print(f"    {r['county_name']:40s}  {r['total_veterans']:>8,}  ({r['pct_post911_vets']:.1f}% post-9/11)")

    return output


def pull_b21005():
    """
    B21005: Age by Veteran Status by Employment Status for Civilian Pop 18-64

    Correct variable mapping (from Census API metadata):
      _001E  Total 18 to 64
      _002E  18 to 34 total
      _003E    18-34 Veteran
      _004E      In labor force
      _005E        Employed
      _006E        Unemployed
      _007E      Not in labor force
      _008E    18-34 Nonveteran
      ...
      _013E  35 to 54 total
      _014E    35-54 Veteran
      _015E      In labor force
      _016E        Employed
      _017E        Unemployed
      _018E      Not in labor force
      _019E    35-54 Nonveteran
      ...
      _024E  55 to 64 total
      _025E    55-64 Veteran
      _026E      In labor force
      _027E        Employed
      _028E        Unemployed
      _029E      Not in labor force
      _030E    55-64 Nonveteran
    """
    print("\n=== Pulling Table B21005: Veteran Employment by Age ===")

    variables = [
        "B21005_001E",  # Total pop 18-64
        "B21005_003E",  # Veterans 18-34
        "B21005_005E",  # Employed veterans 18-34
        "B21005_006E",  # Unemployed veterans 18-34
        "B21005_007E",  # Not in labor force veterans 18-34
        "B21005_014E",  # Veterans 35-54
        "B21005_016E",  # Employed veterans 35-54
        "B21005_017E",  # Unemployed veterans 35-54
        "B21005_018E",  # Not in labor force veterans 35-54
        "B21005_025E",  # Veterans 55-64
        "B21005_027E",  # Employed veterans 55-64
        "B21005_028E",  # Unemployed veterans 55-64
        "B21005_029E",  # Not in labor force veterans 55-64
    ]

    data = fetch_census(variables)

    header = data[0]
    rows = data[1:]

    output = []
    for row in rows:
        d = dict(zip(header, row))

        state_fips = d.get("state", "")
        county_fips = d.get("county", "")
        fips = state_fips + county_fips
        county_name = d.get("NAME", "")

        def safe_int(key):
            val = d.get(key, None)
            if val is None or val == "":
                return 0
            try:
                return int(val)
            except (ValueError, TypeError):
                return 0

        total_pop_18_64 = safe_int("B21005_001E")

        vets_18_34 = safe_int("B21005_003E")
        employed_vets_18_34 = safe_int("B21005_005E")
        unemployed_vets_18_34 = safe_int("B21005_006E")

        vets_35_54 = safe_int("B21005_014E")
        employed_vets_35_54 = safe_int("B21005_016E")
        unemployed_vets_35_54 = safe_int("B21005_017E")

        vets_55_64 = safe_int("B21005_025E")
        employed_vets_55_64 = safe_int("B21005_027E")
        unemployed_vets_55_64 = safe_int("B21005_028E")

        total_vets_working_age = vets_18_34 + vets_35_54 + vets_55_64
        total_employed_vets = employed_vets_18_34 + employed_vets_35_54 + employed_vets_55_64
        total_unemployed_vets = unemployed_vets_18_34 + unemployed_vets_35_54 + unemployed_vets_55_64

        vet_employment_rate = (total_employed_vets / total_vets_working_age * 100) if total_vets_working_age > 0 else 0.0
        vet_unemployment_rate = (total_unemployed_vets / (total_employed_vets + total_unemployed_vets) * 100) if (total_employed_vets + total_unemployed_vets) > 0 else 0.0

        output.append({
            "FIPS": fips,
            "county_name": county_name,
            "total_pop_18_64": total_pop_18_64,
            "total_vets_working_age": total_vets_working_age,
            "vets_18_34": vets_18_34,
            "employed_vets_18_34": employed_vets_18_34,
            "unemployed_vets_18_34": unemployed_vets_18_34,
            "vets_35_54": vets_35_54,
            "employed_vets_35_54": employed_vets_35_54,
            "unemployed_vets_35_54": unemployed_vets_35_54,
            "vets_55_64": vets_55_64,
            "employed_vets_55_64": employed_vets_55_64,
            "unemployed_vets_55_64": unemployed_vets_55_64,
            "total_employed_vets": total_employed_vets,
            "total_unemployed_vets": total_unemployed_vets,
            "vet_employment_rate": round(vet_employment_rate, 2),
            "vet_unemployment_rate": round(vet_unemployment_rate, 2),
        })

    # Save CSV
    outpath = "/Users/bobbarclay/Documents/soldiers/veteran_employment_by_county.csv"
    fieldnames = list(output[0].keys())
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output)

    print(f"\n  Saved {len(output)} counties to {outpath}")

    # Summary stats
    vets_wa = [r["total_vets_working_age"] for r in output if r["total_vets_working_age"] > 0]
    emp_rates = [r["vet_employment_rate"] for r in output if r["total_vets_working_age"] > 0]
    unemp_rates = [r["vet_unemployment_rate"] for r in output if r["total_vets_working_age"] > 0]

    print("\n  --- B21005 Summary Statistics ---")
    print(f"  Counties with data: {len(vets_wa)}")
    print(f"  Total working-age veterans (18-64): {sum(vets_wa):,}")
    total_emp = sum(r["total_employed_vets"] for r in output)
    total_unemp = sum(r["total_unemployed_vets"] for r in output)
    print(f"  Total employed veterans: {total_emp:,}")
    print(f"  Total unemployed veterans: {total_unemp:,}")
    print(f"  Mean vet employment rate: {sum(emp_rates)/len(emp_rates):.2f}%")
    print(f"  Mean vet unemployment rate: {sum(unemp_rates)/len(unemp_rates):.2f}%")
    print(f"  Min vet employment rate: {min(emp_rates):.2f}%")
    print(f"  Max vet employment rate: {max(emp_rates):.2f}%")

    # Top 10 counties by working-age veterans
    top10 = sorted(output, key=lambda x: x["total_vets_working_age"], reverse=True)[:10]
    print("\n  Top 10 counties by working-age veteran population:")
    for r in top10:
        print(f"    {r['county_name']:40s}  {r['total_vets_working_age']:>8,} vets  "
              f"(emp: {r['vet_employment_rate']:.1f}%, unemp: {r['vet_unemployment_rate']:.1f}%)")

    return output


if __name__ == "__main__":
    print("=" * 70)
    print("Census ACS 2022 5-Year - Veteran Detail Data Pull")
    print("=" * 70)

    # Pull B21002 - Period of Service
    try:
        b21002_data = pull_b21002()
    except Exception as e:
        print(f"\nERROR pulling B21002: {e}")
        b21002_data = None

    # Pull B21005 - Employment by Age
    try:
        b21005_data = pull_b21005()
    except Exception as e:
        print(f"\nERROR pulling B21005: {e}")
        b21005_data = None

    print("\n" + "=" * 70)
    print("DONE")
    if b21002_data:
        print(f"  B21002 (Period of Service): {len(b21002_data)} counties")
    if b21005_data:
        print(f"  B21005 (Employment):        {len(b21005_data)} counties")
    print("=" * 70)
