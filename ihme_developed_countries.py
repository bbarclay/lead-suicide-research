"""Verify the '21 developed countries β=0.15' IHME claim in Paper 2."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

panel = pd.read_csv("ihme_gbd_lead_vs_suicide_panel.csv")
panel = panel.dropna(subset=["lead_death_rate", "suicide_rate"])

# OECD / high-income country list (approx.)
high_income = [
    "United States", "Canada", "United Kingdom", "Germany", "France", "Italy",
    "Spain", "Netherlands", "Belgium", "Switzerland", "Austria", "Sweden",
    "Norway", "Denmark", "Finland", "Ireland", "Portugal", "Japan",
    "Australia", "New Zealand", "South Korea", "Republic of Korea",
    "Israel", "Greece", "Luxembourg", "Iceland", "Czech Republic", "Czechia",
    "Slovakia", "Slovenia", "Estonia", "Latvia", "Lithuania", "Poland",
    "Hungary",
]

sub = panel[panel["country"].isin(high_income)].copy()
print(f"Countries in high-income subset: {sub['country'].nunique()}")
print(f"Names: {sorted(sub['country'].unique().tolist())}")
print(f"Country-year rows: {len(sub)}")

# Country + year fixed effects
X = pd.get_dummies(sub["country"], prefix="c", drop_first=True)
X["lead_death_rate"] = sub["lead_death_rate"].values
X["year"] = sub["year"].values
X = sm.add_constant(X).astype(float)
y = sub["suicide_rate"].astype(float)
fit = sm.OLS(y, X).fit(cov_type="HC3")
row = fit.summary2().tables[1].round(4)
keep_rows = [r for r in row.index if r in ("const", "lead_death_rate", "year")]
print(f"\nOLS with country FE + year (high-income, N={len(sub)}):")
print(row.loc[keep_rows].to_string())
print(f"R² = {fit.rsquared:.3f}")

# First-difference within country
sub = sub.sort_values(["country", "year"])
sub["d_lead"] = sub.groupby("country")["lead_death_rate"].diff()
sub["d_suicide"] = sub.groupby("country")["suicide_rate"].diff()
diff_df = sub.dropna(subset=["d_lead", "d_suicide"])
r, p = stats.pearsonr(diff_df["d_lead"], diff_df["d_suicide"])
print(f"\nWithin high-income country Δlead → Δsuicide:")
print(f"  N = {len(diff_df)} country-year pairs")
print(f"  Pearson r = {r:+.3f}  p = {p:.2e}")

# Country trend slopes
slopes = []
for country, grp in sub.groupby("country"):
    if len(grp) < 15: continue
    ls = np.polyfit(grp["year"], grp["lead_death_rate"], 1)[0]
    ss = np.polyfit(grp["year"], grp["suicide_rate"], 1)[0]
    slopes.append({"country": country, "lead_slope": ls, "suicide_slope": ss})
slopes = pd.DataFrame(slopes)
r, p = stats.pearsonr(slopes["lead_slope"], slopes["suicide_slope"])
print(f"\nCountry-level trend-slope correlation in {len(slopes)} high-income countries:")
print(f"  r(Δlead, Δsuicide) = {r:+.3f}  p = {p:.2e}")
print(slopes.sort_values("lead_slope").to_string(index=False))
