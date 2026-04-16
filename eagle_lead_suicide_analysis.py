"""
State-level merge: eagle bone lead (Slabe et al. 2022, USGS) vs.
VA veteran suicide rate, mining site density, and male suicide rate.

Purpose: add a wildlife bioindicator layer of convergent evidence to Paper 2/3.
Eagle bone Pb is independent of human behavioral/cultural confounds and reflects
decades of integrated environmental lead exposure. If it tracks the same geography
as veteran suicide, that's biological validation of the mining-density proxy.
"""

import numpy as np
import pandas as pd
from scipy import stats

# ---------- 1. Eagle bone lead by state ----------
femur = pd.read_csv("Pb_Eagle_Femur.csv")
femur.columns = [c.strip() for c in femur.columns]
femur["State"] = femur["State"].str.strip()
femur["Pb_ug_g_DW"] = femur["DW Lead (µg/g)"].astype(float)
# Chronic poisoning threshold used in the paper: >= 10 µg/g DW
femur["chronic"] = (femur["Pb_ug_g_DW"] >= 10).astype(int)

# Blood (ante-mortem) complement
blood = pd.read_csv("Pb_Eagle_Blood.csv")
blood.columns = [c.strip() for c in blood.columns]
blood["State"] = blood["State"].str.strip()
blood["Pb_ug_dL"] = blood["WW Lead (µg/dL)"].astype(float)

# State aggregates
femur_state = (
    femur.groupby("State")
    .agg(
        n_femur=("Pb_ug_g_DW", "count"),
        mean_femur_Pb=("Pb_ug_g_DW", "mean"),
        median_femur_Pb=("Pb_ug_g_DW", "median"),
        pct_chronic=("chronic", "mean"),
    )
    .reset_index()
)

blood_state = (
    blood.groupby("State")
    .agg(
        n_blood=("Pb_ug_dL", "count"),
        mean_blood_Pb=("Pb_ug_dL", "mean"),
        median_blood_Pb=("Pb_ug_dL", "median"),
    )
    .reset_index()
)

eagle_state = femur_state.merge(blood_state, on="State", how="outer")

# ---------- 2. VA veteran suicide rate ----------
va = pd.read_csv("VA_State_Veteran_Suicide_Rates_2023.csv")
va = va.rename(columns={
    "State": "State",
    "Veteran_Suicide_Rate_per_100000": "vet_suicide_rate_2023",
    "Veteran_Suicide_Deaths_2023": "vet_suicide_deaths_2023",
})

# ---------- 3. Mining density by state ----------
mining = pd.read_csv("usgs_mining_sites_by_county.csv")
mining_state = (
    mining.groupby("state")
    .agg(
        total_mining_sites=("mining_site_count", "sum"),
        total_lead_zinc_copper_sites=("lead_zinc_copper_sites", "sum"),
        n_counties=("county", "count"),
    )
    .reset_index()
    .rename(columns={"state": "State"})
)
mining_state["mining_sites_per_county"] = (
    mining_state["total_mining_sites"] / mining_state["n_counties"]
)
mining_state["log_total_mining_sites"] = np.log1p(mining_state["total_mining_sites"])
mining_state["log_leadzinc_sites"] = np.log1p(mining_state["total_lead_zinc_copper_sites"])

# ---------- 4. Male suicide rate by state (from real_county_dataset if available) ----------
county = pd.read_csv("real_county_dataset.csv", low_memory=False)
# Find a male suicide rate column and a state column
state_col = None
for c in ["state", "State", "STATE", "state_name", "State_name"]:
    if c in county.columns:
        state_col = c
        break

rate_col = None
for c in county.columns:
    lc = c.lower()
    if ("male" in lc and "suicide" in lc and "rate" in lc) or lc in (
        "male_suicide_rate",
        "m_suicide_rate",
    ):
        rate_col = c
        break
if rate_col is None:
    # fall back to any column that looks like a suicide rate
    for c in county.columns:
        if "suicide" in c.lower() and "rate" in c.lower():
            rate_col = c
            break

if state_col and rate_col:
    male_state = (
        county.groupby(state_col)[rate_col]
        .mean()
        .reset_index()
        .rename(columns={state_col: "State", rate_col: "male_suicide_rate"})
    )
else:
    male_state = pd.DataFrame(columns=["State", "male_suicide_rate"])

# ---------- 5. Build the final panel ----------
panel = (
    eagle_state.merge(va, on="State", how="left")
    .merge(mining_state, on="State", how="left")
    .merge(male_state, on="State", how="left")
)

# Force numeric on everything we'll regress
for c in ["vet_suicide_rate_2023", "vet_suicide_deaths_2023",
          "mean_femur_Pb", "median_femur_Pb", "pct_chronic",
          "mean_blood_Pb", "median_blood_Pb",
          "log_leadzinc_sites", "log_total_mining_sites",
          "total_lead_zinc_copper_sites", "total_mining_sites",
          "male_suicide_rate", "n_femur"]:
    if c in panel.columns:
        panel[c] = pd.to_numeric(panel[c], errors="coerce")

# Only keep states with adequate eagle sample (n >= 5 for reliable state mean)
panel_reliable = panel[panel["n_femur"] >= 5].copy()

# Also build a full panel using ALL states with any eagle data (weighted by n)
panel_all = panel[panel["n_femur"] >= 1].copy()

print("=" * 70)
print("STATE-LEVEL PANEL: EAGLE BONE Pb x VETERAN SUICIDE x MINING")
print("=" * 70)
print(f"\nStates with eagle femur data: {len(panel)}")
print(f"States with n_femur >= 5 (reliable): {len(panel_reliable)}")

cols_show = [
    "State", "n_femur", "mean_femur_Pb", "median_femur_Pb", "pct_chronic",
    "vet_suicide_rate_2023", "total_lead_zinc_copper_sites", "male_suicide_rate",
]
cols_show = [c for c in cols_show if c in panel_reliable.columns]
print("\nReliable states (n_femur >= 5):")
print(panel_reliable[cols_show].sort_values("mean_femur_Pb", ascending=False).to_string(index=False))

# ---------- 6. Bivariate correlations ----------
print("\n" + "=" * 70)
print("BIVARIATE CORRELATIONS (n_femur >= 5)")
print("=" * 70)

def corr_report(x, y, label, df):
    d = df[[x, y]].dropna()
    if len(d) < 4:
        print(f"{label}: insufficient data (n={len(d)})")
        return
    r_p, p_p = stats.pearsonr(d[x], d[y])
    r_s, p_s = stats.spearmanr(d[x], d[y])
    print(
        f"{label:55s}  n={len(d):2d}  "
        f"Pearson r={r_p:+.3f} (p={p_p:.3f})  "
        f"Spearman ρ={r_s:+.3f} (p={p_s:.3f})"
    )

for y in ["vet_suicide_rate_2023", "male_suicide_rate"]:
    for x in ["mean_femur_Pb", "median_femur_Pb", "pct_chronic",
              "mean_blood_Pb", "log_leadzinc_sites", "log_total_mining_sites"]:
        if x in panel_reliable.columns and y in panel_reliable.columns:
            corr_report(x, y, f"{x} -> {y}", panel_reliable)
    print()

# Internal validity: does mining density predict eagle bone lead?
print("INTERNAL VALIDITY: does mining site density predict eagle bone lead?")
corr_report("log_leadzinc_sites", "mean_femur_Pb",
            "log(Pb/Zn/Cu sites) -> mean eagle femur Pb", panel_reliable)
corr_report("log_total_mining_sites", "mean_femur_Pb",
            "log(total mining sites) -> mean eagle femur Pb", panel_reliable)

# ---------- 7. Partial / multivariate: does eagle Pb add beyond mining? ----------
print("\n" + "=" * 70)
print("MULTIVARIATE: does eagle bone Pb predict vet suicide after controlling for mining?")
print("=" * 70)

try:
    import statsmodels.api as sm

    for dv in ["vet_suicide_rate_2023", "male_suicide_rate"]:
        sub = panel_reliable[["mean_femur_Pb", "log_leadzinc_sites", dv]].dropna()
        if len(sub) < 6:
            print(f"\n{dv}: insufficient data (n={len(sub)})")
            continue
        X = sm.add_constant(sub[["mean_femur_Pb", "log_leadzinc_sites"]])
        y = sub[dv]
        fit = sm.OLS(y, X).fit(cov_type="HC3")
        print(f"\nDV = {dv}   N = {len(sub)}   R² = {fit.rsquared:.3f}")
        print(fit.summary2().tables[1].round(3).to_string())
except Exception as e:
    print(f"(statsmodels unavailable: {e})")

# ---------- 7b. Weighted analysis using ALL 38 states ----------
print("\n" + "=" * 70)
print("WEIGHTED (WLS) USING ALL 38 STATES — weights = sqrt(n_femur)")
print("=" * 70)
try:
    import statsmodels.api as sm
    for dv in ["vet_suicide_rate_2023", "male_suicide_rate"]:
        sub = panel_all[["mean_femur_Pb", "log_leadzinc_sites", "n_femur", dv]].dropna()
        if len(sub) < 6:
            print(f"\n{dv}: insufficient data (n={len(sub)})")
            continue
        X = sm.add_constant(sub[["mean_femur_Pb", "log_leadzinc_sites"]])
        y = sub[dv]
        w = np.sqrt(sub["n_femur"])
        fit = sm.WLS(y, X, weights=w).fit(cov_type="HC3")
        print(f"\nDV = {dv}   N = {len(sub)}   R² = {fit.rsquared:.3f}")
        print(fit.summary2().tables[1].round(3).to_string())
except Exception as e:
    print(f"(statsmodels unavailable: {e})")

# ---------- 7c. Simple rank test: do top-Pb states have higher suicide? ----------
print("\n" + "=" * 70)
print("RANK COMPARISON: top-quartile eagle Pb states vs bottom-quartile")
print("=" * 70)
pr = panel_reliable.dropna(subset=["mean_femur_Pb", "vet_suicide_rate_2023"]).copy()
pr["Pb_q"] = pd.qcut(pr["mean_femur_Pb"], 2, labels=["low", "high"])
for dv in ["vet_suicide_rate_2023", "male_suicide_rate"]:
    if dv not in pr.columns: continue
    g = pr.dropna(subset=[dv]).groupby("Pb_q", observed=True)[dv].agg(["mean", "median", "count"])
    print(f"\n{dv}:")
    print(g.round(2).to_string())
    # Mann-Whitney
    lo = pr[pr["Pb_q"] == "low"][dv].dropna()
    hi = pr[pr["Pb_q"] == "high"][dv].dropna()
    if len(lo) >= 3 and len(hi) >= 3:
        u, p = stats.mannwhitneyu(hi, lo, alternative="greater")
        print(f"  Mann-Whitney (high > low) p = {p:.3f}")

# ---------- 8. Save output ----------
panel.to_csv("eagle_lead_suicide_state_panel.csv", index=False)
print("\n\nSaved: eagle_lead_suicide_state_panel.csv")
