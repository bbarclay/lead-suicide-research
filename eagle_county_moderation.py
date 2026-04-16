"""
Does the county-level mining × veteran interaction predict suicide
MORE in states with high eagle bone Pb (high bioavailable lead) than in
low-eagle-Pb states?

If yes, that's a dose-response at the level of the exposure proxy itself:
the hypothesized mechanism (mining × veteran → suicide) is stronger where
we have independent biological evidence that environmental lead is elevated.
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

# --- County data ---
county = pd.read_csv("real_county_dataset.csv", low_memory=False)
print(f"County rows: {len(county)}")

# --- Eagle state Pb ---
femur = pd.read_csv("Pb_Eagle_Femur.csv")
femur.columns = [c.strip() for c in femur.columns]
femur["State"] = femur["State"].str.strip()
femur["Pb"] = femur["DW Lead (µg/g)"].astype(float)
femur["chronic"] = (femur["Pb"] >= 10).astype(int)
state_eagle = femur.groupby("State").agg(
    n_femur=("Pb", "count"),
    eagle_mean_Pb=("Pb", "mean"),
    eagle_pct_chronic=("chronic", "mean"),
).reset_index()

# Merge eagle data to counties by state
merged = county.merge(state_eagle, left_on="state_name", right_on="State", how="left")
print(f"Counties with eagle state data: {merged['eagle_mean_Pb'].notna().sum()}")

# --- Columns ---
DV = "male_suicide_rate_cdc"
mining_col = "lead_zinc_copper" if "lead_zinc_copper" in merged.columns else "mining_sites"
merged["log_mining"] = np.log1p(merged[mining_col])
merged["log_mining_centered"] = merged["log_mining"] - merged["log_mining"].mean()
merged["vet_centered"] = merged["veteran_pct"] - merged["veteran_pct"].mean()
merged["mining_x_vet"] = merged["log_mining_centered"] * merged["vet_centered"]

# --- Baseline model: does the county mining × vet interaction hold? ---
predictors = ["log_mining", "veteran_pct", "mining_x_vet",
              "chr_pct_rural", "poverty_rate", "pct_white_nh", "median_age"]
sub = merged[[DV] + predictors].dropna()
X = sm.add_constant(sub[predictors])
fit = sm.OLS(sub[DV], X).fit(cov_type="HC3")
print(f"\nBaseline: county mining × vet on male suicide  (N = {len(sub)})")
row = fit.summary2().tables[1].round(3)
print(row.loc[[r for r in row.index if r in ["const"] + predictors]].to_string())
print(f"R² = {fit.rsquared:.3f}")

# --- Split by eagle Pb level ---
print("\n" + "=" * 70)
print("STRATIFIED: does mining×vet interaction differ by state-level eagle Pb?")
print("=" * 70)

sub2 = merged.dropna(subset=predictors + [DV, "eagle_mean_Pb"]).copy()
sub2 = sub2[sub2["n_femur"] >= 5]
if len(sub2) > 100:
    med = sub2["eagle_mean_Pb"].median()
    sub2["eagle_high"] = (sub2["eagle_mean_Pb"] >= med).astype(int)
    for level in [0, 1]:
        s = sub2[sub2["eagle_high"] == level]
        X = sm.add_constant(s[predictors])
        fit = sm.OLS(s[DV], X).fit(cov_type="HC3")
        lbl = "HIGH eagle Pb states" if level == 1 else "LOW eagle Pb states"
        print(f"\n{lbl}  N = {len(s)}  median state Pb threshold = {med:.1f} µg/g")
        row = fit.summary2().tables[1].round(3)
        print(row.loc[["log_mining", "veteran_pct", "mining_x_vet"]].to_string())
        print(f"  R² = {fit.rsquared:.3f}")

# --- Three-way interaction: mining × vet × eagle_Pb ---
print("\n" + "=" * 70)
print("THREE-WAY INTERACTION: mining × veteran × state eagle Pb")
print("=" * 70)
s = sub2.copy()
s["eagle_centered"] = s["eagle_mean_Pb"] - s["eagle_mean_Pb"].mean()
s["mining_x_vet_x_eagle"] = s["log_mining_centered"] * s["vet_centered"] * s["eagle_centered"]
preds3 = ["log_mining", "veteran_pct", "eagle_mean_Pb",
          "mining_x_vet", "mining_x_vet_x_eagle",
          "chr_pct_rural", "poverty_rate", "pct_white_nh", "median_age"]
X = sm.add_constant(s[preds3])
fit = sm.OLS(s[DV], X).fit(cov_type="HC3")
row = fit.summary2().tables[1].round(4)
print(f"N = {len(s)}  R² = {fit.rsquared:.3f}")
show = ["const", "log_mining", "veteran_pct", "eagle_mean_Pb",
        "mining_x_vet", "mining_x_vet_x_eagle"]
print(row.loc[[r for r in show if r in row.index]].to_string())
