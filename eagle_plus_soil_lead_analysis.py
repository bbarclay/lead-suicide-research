"""
Stack multiple convergent lead-exposure measures at state level:
  - eagle femur Pb  (Slabe 2022, USGS — ammunition/environmental)
  - USGS soil Pb     (geogenic + anthropogenic baseline)
  - mining sites     (historical exposure proxy)
against veteran and male suicide rates.

Question: among these, which layer(s) independently predict suicide?
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

# --- Eagle
femur = pd.read_csv("Pb_Eagle_Femur.csv")
femur.columns = [c.strip() for c in femur.columns]
femur["State"] = femur["State"].str.strip()
femur["Pb"] = femur["DW Lead (µg/g)"].astype(float)
femur["chronic"] = (femur["Pb"] >= 10).astype(int)
femur_state = femur.groupby("State").agg(
    n_femur=("Pb", "count"),
    mean_femur_Pb=("Pb", "mean"),
    median_femur_Pb=("Pb", "median"),
    pct_chronic=("chronic", "mean"),
).reset_index()

# --- Soil Pb (state abbreviation -> full state name)
abbrev = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}
soil = pd.read_csv("usgs_soil_lead_by_state.csv")
soil["State"] = soil["state"].map(abbrev)
soil = soil.dropna(subset=["State"])[["State", "mean_soil_lead_ppm", "median_soil_lead_ppm",
                                      "p90_soil_lead_ppm", "n_samples"]]

# --- Mining
mining = pd.read_csv("usgs_mining_sites_by_county.csv")
mining_state = mining.groupby("state").agg(
    total_mining_sites=("mining_site_count", "sum"),
    total_lead_zinc_copper_sites=("lead_zinc_copper_sites", "sum"),
    n_counties=("county", "count"),
).reset_index().rename(columns={"state": "State"})
mining_state["log_leadzinc_sites"] = np.log1p(mining_state["total_lead_zinc_copper_sites"])

# --- Outcomes
va = pd.read_csv("VA_State_Veteran_Suicide_Rates_2023.csv").rename(columns={
    "Veteran_Suicide_Rate_per_100000": "vet_suicide_rate_2023",
})[["State", "vet_suicide_rate_2023"]]

# Male suicide rate from county data
county = pd.read_csv("real_county_dataset.csv", low_memory=False)
state_col = next((c for c in ["state", "State", "state_name"] if c in county.columns), None)
rate_col = None
for c in county.columns:
    lc = c.lower()
    if "suicide" in lc and "rate" in lc and ("male" in lc or lc.startswith("m_")):
        rate_col = c; break
if rate_col is None:
    for c in county.columns:
        if "suicide" in c.lower() and "rate" in c.lower():
            rate_col = c; break
male_state = county.groupby(state_col)[rate_col].mean().reset_index().rename(
    columns={state_col: "State", rate_col: "male_suicide_rate"}
)

# --- Merge all
panel = femur_state.merge(soil, on="State", how="left") \
                   .merge(mining_state, on="State", how="left") \
                   .merge(va, on="State", how="left") \
                   .merge(male_state, on="State", how="left")

# Force numeric
for c in panel.columns:
    if c != "State":
        panel[c] = pd.to_numeric(panel[c], errors="coerce")

# Log-transform skewed soil measures
panel["log_soil_Pb"] = np.log1p(panel["mean_soil_lead_ppm"])
panel["log_soil_p90"] = np.log1p(panel["p90_soil_lead_ppm"])

print("=" * 70)
print("PANEL SUMMARY (n_femur >= 1, all available states)")
print("=" * 70)
_cov_cols = ["mean_femur_Pb", "mean_soil_lead_ppm", "log_leadzinc_sites", "vet_suicide_rate_2023"]
print(f"States with all four layers (eagle+soil+mining+vet): {panel.dropna(subset=_cov_cols).shape[0]}")

# --- Bivariate screen ---
print("\nBivariate correlations with MALE suicide rate")
for x in ["mean_femur_Pb", "pct_chronic", "log_soil_Pb", "log_soil_p90", "log_leadzinc_sites"]:
    d = panel[[x, "male_suicide_rate"]].dropna()
    if len(d) < 6: continue
    r, p = stats.pearsonr(d[x], d["male_suicide_rate"])
    rs, ps = stats.spearmanr(d[x], d["male_suicide_rate"])
    print(f"  {x:22s} n={len(d):2d}  r={r:+.3f} (p={p:.3f})  ρ={rs:+.3f} (p={ps:.3f})")

print("\nBivariate correlations with VET suicide rate (2023)")
for x in ["mean_femur_Pb", "pct_chronic", "log_soil_Pb", "log_soil_p90", "log_leadzinc_sites"]:
    d = panel[[x, "vet_suicide_rate_2023"]].dropna()
    if len(d) < 6: continue
    r, p = stats.pearsonr(d[x], d["vet_suicide_rate_2023"])
    rs, ps = stats.spearmanr(d[x], d["vet_suicide_rate_2023"])
    print(f"  {x:22s} n={len(d):2d}  r={r:+.3f} (p={p:.3f})  ρ={rs:+.3f} (p={ps:.3f})")

# --- Multivariate horse race: four Pb proxies simultaneously ---
print("\n" + "=" * 70)
print("HORSE RACE (WLS, weights = sqrt(n_femur))")
print("All four lead proxies entered simultaneously")
print("=" * 70)

for dv in ["vet_suicide_rate_2023", "male_suicide_rate"]:
    predictors = ["mean_femur_Pb", "log_soil_Pb", "log_leadzinc_sites"]
    sub = panel[["n_femur", dv] + predictors].dropna()
    if len(sub) < 8:
        print(f"\n{dv}: n={len(sub)} — skipping")
        continue
    X = sm.add_constant(sub[predictors])
    w = np.sqrt(sub["n_femur"])
    fit = sm.WLS(sub[dv], X, weights=w).fit(cov_type="HC3")
    print(f"\nDV = {dv}   N = {len(sub)}   R² = {fit.rsquared:.3f}")
    print(fit.summary2().tables[1].round(3).to_string())

# --- Compare: state-level vs county-level mining coefficient ---
print("\n" + "=" * 70)
print("STATE-LEVEL vs COUNTY-LEVEL: which scale does mining work at?")
print("=" * 70)
# County-level from the master dataset, if available
cty = county.copy()
cty["log_leadzinc"] = np.log1p(cty["lead_zinc_copper_sites"]) if "lead_zinc_copper_sites" in cty.columns else np.nan
if rate_col and "lead_zinc_copper_sites" in cty.columns:
    sub = cty[[rate_col, "log_leadzinc"]].dropna()
    if len(sub) > 100:
        r, p = stats.pearsonr(sub["log_leadzinc"], sub[rate_col])
        print(f"County-level: log(leadzinc sites) → male suicide  n={len(sub)}  r={r:+.3f} p={p:.3f}")

panel.to_csv("eagle_soil_mining_suicide_state_panel.csv", index=False)
print("\nSaved: eagle_soil_mining_suicide_state_panel.csv")
