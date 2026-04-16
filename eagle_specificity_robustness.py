"""
Specificity and robustness for the eagle-Pb → veteran-suicide association.

Tests:
  1. Does eagle bone Pb predict suicide but NOT non-impulsivity outcomes?
     (negative controls: motor vehicle deaths, cancer deaths)
  2. Does eagle Pb survive demographic controls at state level?
     (rurality %, poverty %, % white, veteran %)
  3. Age-stratified VA data: is the eagle-Pb association strongest
     in the generation with peak childhood leaded-gasoline exposure?
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

# --- Eagle bone Pb by state ---
femur = pd.read_csv("Pb_Eagle_Femur.csv")
femur.columns = [c.strip() for c in femur.columns]
femur["State"] = femur["State"].str.strip()
femur["Pb"] = femur["DW Lead (µg/g)"].astype(float)
femur["chronic"] = (femur["Pb"] >= 10).astype(int)
eagle = femur.groupby("State").agg(
    n_femur=("Pb", "count"),
    mean_femur_Pb=("Pb", "mean"),
    pct_chronic=("chronic", "mean"),
).reset_index()

# --- State VA veteran suicide 2023 ---
va = pd.read_csv("VA_State_Veteran_Suicide_Rates_2023.csv").rename(columns={
    "Veteran_Suicide_Rate_per_100000": "vet_suicide_rate_2023",
})[["State", "vet_suicide_rate_2023"]]

# --- State demographics from the county master dataset ---
county = pd.read_csv("real_county_dataset.csv", low_memory=False)
scol = next((c for c in ["state", "State", "state_name"] if c in county.columns), None)

def grab(name_candidates):
    for c in county.columns:
        lc = c.lower()
        if any(k in lc for k in name_candidates):
            return c
    return None

cand = {
    "male_suicide_rate":  ["male_suicide_rate", "m_suicide_rate"],
    "overdose":           ["drug_overdose", "overdose_rate"],
    "veteran_pct":        ["veteran_percent", "pct_veteran", "percent_veteran", "veteran_pct"],
    "rural_pct":          ["pct_rural", "percent_rural", "rural_percent"],
    "poverty":            ["poverty_rate", "pct_poverty", "percent_poverty"],
    "white_pct":          ["percent_white", "pct_white", "white_percent", "nh_white"],
    "median_age":         ["median_age"],
    "mv_deaths":          ["motor_vehicle", "mv_death", "traffic"],
    "firearm_rate":       ["firearm_rate", "firearm_death", "gun_death"],
    "homicide":           ["homicide"],
}
mapping = {}
for std, cands in cand.items():
    col = None
    for c in county.columns:
        lc = c.lower()
        for k in cands:
            if k in lc:
                col = c; break
        if col: break
    mapping[std] = col

print("Column matches found in real_county_dataset.csv:")
for k, v in mapping.items():
    print(f"  {k:20s} -> {v}")

# Build state aggregates
state_dem = county.groupby(scol).agg({v: "mean" for v in mapping.values() if v}).reset_index()
state_dem = state_dem.rename(columns={scol: "State"})
# Rename mapped columns to standard names
rev = {v: k for k, v in mapping.items() if v}
state_dem = state_dem.rename(columns=rev)

# --- Merge ---
panel = eagle.merge(va, on="State", how="left").merge(state_dem, on="State", how="left")
for c in panel.columns:
    if c != "State":
        panel[c] = pd.to_numeric(panel[c], errors="coerce")

# --- 1. Specificity: eagle Pb vs outcomes ---
print("\n" + "=" * 70)
print("SPECIFICITY: eagle bone Pb correlated with what?")
print("=" * 70)
outcomes = ["vet_suicide_rate_2023", "male_suicide_rate", "overdose",
            "mv_deaths", "firearm_rate", "homicide"]
for y in outcomes:
    if y not in panel.columns: continue
    d = panel[["mean_femur_Pb", y]].dropna()
    if len(d) < 6: continue
    r, p = stats.pearsonr(d["mean_femur_Pb"], d[y])
    rs, ps = stats.spearmanr(d["mean_femur_Pb"], d[y])
    print(f"  mean eagle Pb -> {y:25s}  n={len(d):2d}  r={r:+.3f} (p={p:.3f})  ρ={rs:+.3f} (p={ps:.3f})")

print("\npct_chronic (% eagles w/ Pb >= 10 µg/g DW) vs outcomes:")
for y in outcomes:
    if y not in panel.columns: continue
    d = panel[["pct_chronic", y]].dropna()
    if len(d) < 6: continue
    r, p = stats.pearsonr(d["pct_chronic"], d[y])
    rs, ps = stats.spearmanr(d["pct_chronic"], d[y])
    print(f"  pct_chronic -> {y:25s}  n={len(d):2d}  r={r:+.3f} (p={p:.3f})  ρ={rs:+.3f} (p={ps:.3f})")

# --- 2. Robustness: does eagle Pb survive demographic controls? ---
print("\n" + "=" * 70)
print("ROBUSTNESS: eagle Pb -> vet suicide, with demographic controls")
print("=" * 70)
for dv in ["vet_suicide_rate_2023", "male_suicide_rate"]:
    ctrls = [c for c in ["rural_pct", "poverty", "white_pct", "veteran_pct", "median_age"]
             if c in panel.columns]
    sub = panel[["n_femur", "mean_femur_Pb", dv] + ctrls].dropna()
    if len(sub) < 10:
        print(f"\n{dv}: n={len(sub)}, skipping multivariate")
        continue
    X = sm.add_constant(sub[["mean_femur_Pb"] + ctrls])
    y = sub[dv]
    w = np.sqrt(sub["n_femur"])
    fit = sm.WLS(y, X, weights=w).fit(cov_type="HC3")
    print(f"\nDV = {dv}   N = {len(sub)}   R² = {fit.rsquared:.3f}")
    print(fit.summary2().tables[1].round(3).to_string())

# --- 3. Age-stratified VA suicide ---
print("\n" + "=" * 70)
print("AGE STRATIFICATION: VA veteran suicide by age x eagle Pb")
print("(expectation: strongest in 55-74 — peak leaded-gasoline childhood cohort)")
print("=" * 70)

try:
    age = pd.read_csv("VA_suicides_by_age_state.csv")
    print(f"Columns: {list(age.columns)[:12]}")
    print(f"First rows:\n{age.head(3).to_string()}")
    # Attempt an auto-detect: find 'state', 'age group', 'rate'
    cols_lower = {c: c.lower() for c in age.columns}
    state_c = next((c for c, lc in cols_lower.items() if "state" in lc), None)
    agegrp_c = next((c for c, lc in cols_lower.items() if ("age" in lc and ("group" in lc or "range" in lc or "_" in lc))), None)
    rate_c = next((c for c, lc in cols_lower.items() if "rate" in lc), None)
    if state_c and agegrp_c and rate_c:
        print(f"\nMatching columns: state={state_c}  age={agegrp_c}  rate={rate_c}")
        age[rate_c] = pd.to_numeric(age[rate_c], errors="coerce")
        age_recent = age[age["Year"].astype(str).str.strip().isin(["2021", "2022", "2023"])].copy()
        if age_recent.empty:
            age_recent = age
        for age_val, grp in age_recent.groupby(agegrp_c):
            state_rate = grp.groupby(state_c)[rate_c].mean().reset_index()
            state_rate.columns = ["State", "rate"]
            d = eagle.merge(state_rate, on="State", how="inner")
            d = d[d["n_femur"] >= 5].dropna(subset=["mean_femur_Pb", "rate"])
            if len(d) < 8: continue
            r, p = stats.pearsonr(d["mean_femur_Pb"], d["rate"])
            print(f"  Age {str(age_val):12s} n={len(d):2d}  r={r:+.3f}  p={p:.3f}")
except Exception as e:
    print(f"(could not run age stratification: {e})")

panel.to_csv("eagle_specificity_panel.csv", index=False)
print("\nSaved: eagle_specificity_panel.csv")
