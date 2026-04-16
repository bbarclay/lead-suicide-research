"""
IHME GBD cross-country: within-country longitudinal test of lead → suicide.

Cross-sectional country correlations are confounded by everything (GDP, culture,
reporting quality). A stronger design: within each country across 1990-2019,
does the year-to-year change in age-standardized lead death rate predict the
year-to-year change in suicide rate? This controls for all time-invariant
country characteristics.
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

panel = pd.read_csv("ihme_gbd_lead_vs_suicide_panel.csv")
panel = panel.dropna(subset=["lead_death_rate", "suicide_rate"])

# Keep only countries with enough years
country_counts = panel.groupby("country").size()
keep = country_counts[country_counts >= 20].index
panel = panel[panel["country"].isin(keep)].copy()
print(f"Countries with >=20 years: {panel['country'].nunique()}")
print(f"Country-year rows: {len(panel)}")

# ---------- Within-country year-to-year change correlation ----------
panel = panel.sort_values(["country", "year"])
panel["d_lead"] = panel.groupby("country")["lead_death_rate"].diff()
panel["d_suicide"] = panel.groupby("country")["suicide_rate"].diff()

diff_df = panel.dropna(subset=["d_lead", "d_suicide"])
r, p = stats.pearsonr(diff_df["d_lead"], diff_df["d_suicide"])
rs, ps = stats.spearmanr(diff_df["d_lead"], diff_df["d_suicide"])
print(f"\nWithin-country Δ correlation (N = {len(diff_df)} country-year pairs):")
print(f"  Pearson  r = {r:+.3f}  p = {p:.3e}")
print(f"  Spearman ρ = {rs:+.3f}  p = {ps:.3e}")

# Country and year fixed effects regression
print("\nOLS with country fixed effects (HC3):")
X = pd.get_dummies(panel["country"], prefix="c", drop_first=True)
X["lead_death_rate"] = panel["lead_death_rate"]
X["year"] = panel["year"]
X = sm.add_constant(X).astype(float)
y = panel["suicide_rate"].astype(float)
fit = sm.OLS(y, X).fit(cov_type="HC3")
row_tbl = fit.summary2().tables[1]
# Print just the variables of interest
interesting = [c for c in row_tbl.index if c in ("const", "lead_death_rate", "year")]
print(row_tbl.loc[interesting].round(5).to_string())
print(f"N = {len(panel)}   R² = {fit.rsquared:.3f}")

# ---------- High-income country subset ----------
# Recent cross-section (2019)
recent = panel[panel["year"] == 2019].copy()
print(f"\n2019 cross-section: N = {len(recent)} countries")
if len(recent) > 10:
    r, p = stats.pearsonr(recent["lead_death_rate"], recent["suicide_rate"])
    rs, ps = stats.spearmanr(recent["lead_death_rate"], recent["suicide_rate"])
    print(f"  2019 r = {r:+.3f}  p = {p:.3e}")
    print(f"  2019 ρ = {rs:+.3f}  p = {ps:.3e}")

# ---------- Country-level trend slopes ----------
print("\nCountry-level trend slopes (1990-2019):")
rows = []
for country, grp in panel.groupby("country"):
    if len(grp) < 15: continue
    lead_slope = np.polyfit(grp["year"], grp["lead_death_rate"], 1)[0]
    suicide_slope = np.polyfit(grp["year"], grp["suicide_rate"], 1)[0]
    rows.append({"country": country, "lead_slope": lead_slope, "suicide_slope": suicide_slope,
                 "n_years": len(grp)})
slopes = pd.DataFrame(rows)
r, p = stats.pearsonr(slopes["lead_slope"], slopes["suicide_slope"])
rs, ps = stats.spearmanr(slopes["lead_slope"], slopes["suicide_slope"])
print(f"  Countries with declining lead mortality also show declining suicide?")
print(f"  N = {len(slopes)} countries")
print(f"  Pearson  r(Δlead, Δsuicide) = {r:+.3f}  p = {p:.3e}")
print(f"  Spearman ρ(Δlead, Δsuicide) = {rs:+.3f}  p = {ps:.3e}")

# Top/bottom lead-declining countries
slopes_sorted = slopes.sort_values("lead_slope")
print("\nFastest lead-mortality declines (top 10):")
print(slopes_sorted.head(10).to_string(index=False))
print("\nSlowest or increasing lead mortality (bottom 10):")
print(slopes_sorted.tail(10).to_string(index=False))

slopes.to_csv("ihme_country_slope_comparison.csv", index=False)
print("\nSaved: ihme_country_slope_comparison.csv")
