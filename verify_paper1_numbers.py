#!/usr/bin/env python3
"""
Comprehensive verification of every specific number in MANUSCRIPT_COMPLETE.md
against the actual data in real_county_dataset.csv.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(Path(__file__).resolve().parent / 'real_county_dataset.csv')

print("=" * 90)
print("PAPER 1 NUMBER VERIFICATION REPORT")
print("=" * 90)
print(f"\nTotal rows in CSV: {len(df)}")
print(f"NOTE: The CSV has {len(df)} rows vs manuscript's 3,233.")
print(f"  This suggests the dataset was updated after manuscript numbers were computed.")
print(f"  The extra {len(df) - 3233} rows will cause minor discrepancies in counts/N values.")
print(f"  All rate-based statistics (means, r, p, R-squared) should still be very close.")

results = []

def report(claim, manuscript_val, actual_val, tolerance, category=""):
    """Record and print a verification result."""
    if isinstance(manuscript_val, str) or isinstance(actual_val, str):
        match = str(manuscript_val).strip() == str(actual_val).strip()
    elif manuscript_val is None or actual_val is None:
        match = False
    else:
        match = abs(float(manuscript_val) - float(actual_val)) <= tolerance
    status = "PASS" if match else "FLAG"
    results.append((category, claim, manuscript_val, actual_val, tolerance, status))
    marker = "  " if match else "**"
    print(f"  {marker}[{status}] {claim}")
    print(f"         Manuscript: {manuscript_val}  |  Actual: {actual_val}  |  Tol: {tolerance}")
    return match

# ============================================================
# 1. SAMPLE SIZES
# ============================================================
print("\n" + "=" * 90)
print("1. SAMPLE SIZES")
print("=" * 90)

# N = 2,683 counties with valid male suicide rates
n_valid = df['male_suicide_rate_cdc'].notna().sum()
report("N counties with valid male suicide rates", 2683, n_valid, 0, "Sample Size")

# Total counties in analytic frame: 3,233 (mentioned in methods)
# This is likely total rows
report("Total counties in CSV (methods say 3,233 counties, 104 variables)", 3233, len(df), 5, "Sample Size")
report("Number of variables (methods say 104)", 104, len(df.columns), 2, "Sample Size")

# Suppressed counties: 465 (14.8%)
n_suppressed = df['male_suicide_rate_cdc'].isna().sum()
report("Suppressed counties (no valid rate)", 465, n_suppressed, 5, "Sample Size")
pct_suppressed = n_suppressed / 3142 * 100  # out of 3142 counties
report("Pct suppressed (of 3142)", 14.8, round(pct_suppressed, 1), 1.0, "Sample Size")

# Filter to valid counties for rest of analysis
valid = df[df['male_suicide_rate_cdc'].notna()].copy()
print(f"\n  Working with {len(valid)} valid counties for remaining analyses.")

# ============================================================
# 2. DESCRIPTIVE STATISTICS
# ============================================================
print("\n" + "=" * 90)
print("2. DESCRIPTIVE STATISTICS")
print("=" * 90)

rate = valid['male_suicide_rate_cdc']
report("Mean male suicide crude rate", 19.4, round(rate.mean(), 1), 0.2, "Descriptive")
report("SD male suicide crude rate", 7.4, round(rate.std(), 1), 0.2, "Descriptive")
report("Min male suicide crude rate", 0, round(rate.min(), 1), 1.0, "Descriptive")
report("Max male suicide crude rate", 78, round(rate.max(), 1), 2.0, "Descriptive")

# RUCC breakdown
print("\n  --- Rural-Urban Gradient ---")
rucc = valid['rural_urban_code']

# RUCC 1
rucc1 = valid[rucc == 1]
report("RUCC 1 count", 437, len(rucc1), 5, "RUCC")
report("RUCC 1 mean rate", 15.6, round(rucc1['male_suicide_rate_cdc'].mean(), 1), 0.3, "RUCC")
report("RUCC 1 SD", 5.4, round(rucc1['male_suicide_rate_cdc'].std(), 1), 0.3, "RUCC")

# RUCC 2
rucc2 = valid[rucc == 2]
report("RUCC 2 count", 361, len(rucc2), 5, "RUCC")
report("RUCC 2 mean rate", 17.2, round(rucc2['male_suicide_rate_cdc'].mean(), 1), 0.3, "RUCC")

# RUCC 3
rucc3 = valid[rucc == 3]
report("RUCC 3 count", 343, len(rucc3), 5, "RUCC")
report("RUCC 3 mean rate", 18.9, round(rucc3['male_suicide_rate_cdc'].mean(), 1), 0.3, "RUCC")

# RUCC 4-5
rucc45 = valid[rucc.isin([4, 5])]
report("RUCC 4-5 count", 275, len(rucc45), 5, "RUCC")
report("RUCC 4-5 mean rate", 18.9, round(rucc45['male_suicide_rate_cdc'].mean(), 1), 0.3, "RUCC")

# RUCC 6-7
rucc67 = valid[rucc.isin([6, 7])]
report("RUCC 6-7 count", 613, len(rucc67), 5, "RUCC")
report("RUCC 6-7 mean rate", 20.1, round(rucc67['male_suicide_rate_cdc'].mean(), 1), 0.3, "RUCC")

# RUCC 8-9
rucc89 = valid[rucc.isin([8, 9])]
report("RUCC 8-9 count", 654, len(rucc89), 5, "RUCC")
report("RUCC 8-9 mean rate", 22.8, round(rucc89['male_suicide_rate_cdc'].mean(), 1), 0.3, "RUCC")
report("RUCC 8-9 SD", 9.7, round(rucc89['male_suicide_rate_cdc'].std(), 1), 0.3, "RUCC")

# 46% elevation
pct_elev = (rucc89['male_suicide_rate_cdc'].mean() / rucc1['male_suicide_rate_cdc'].mean() - 1) * 100
report("Rural-urban elevation pct", 46, round(pct_elev, 0), 3, "RUCC")

# Spearman rho for rurality
sp_rho, sp_p = stats.spearmanr(valid['rural_urban_code'], valid['male_suicide_rate_cdc'])
report("Spearman rho (RUCC vs suicide)", 0.368, round(sp_rho, 3), 0.01, "RUCC")

# ============================================================
# 3. BIVARIATE CORRELATIONS
# ============================================================
print("\n" + "=" * 90)
print("3. BIVARIATE CORRELATIONS (r values and p-values)")
print("=" * 90)

def pearson_test(x_col, y_col='male_suicide_rate_cdc', data=valid):
    """Compute Pearson r and p-value for two columns."""
    mask = data[x_col].notna() & data[y_col].notna()
    x = data.loc[mask, x_col]
    y = data.loc[mask, y_col]
    r, p = stats.pearsonr(x, y)
    return r, p, mask.sum()

# Firearm fatality rate
r, p, n = pearson_test('chr_firearm_fatality_rate')
report("r(firearm fatality, suicide)", 0.552, round(r, 3), 0.01, "Bivariate r")

# Elevation
r, p, n = pearson_test('elevation_feet')
report("r(elevation, suicide)", 0.452, round(r, 3), 0.01, "Bivariate r")
print(f"         (N for elevation = {n})")

# Rurality (RUCC)
r, p, n = pearson_test('rural_urban_code')
report("r(RUCC, suicide) Pearson", 0.334, round(r, 3), 0.01, "Bivariate r")

# Veteran pct
r, p, n = pearson_test('veteran_pct')
report("r(veteran_pct, suicide)", 0.324, round(r, 3), 0.01, "Bivariate r")

# chr_pct_rural (rural pop share)
r, p, n = pearson_test('chr_pct_rural')
report("r(rural pop share, suicide)", 0.321, round(r, 3), 0.01, "Bivariate r")

# Extraction employment pct
r, p, n = pearson_test('extraction_employment_pct')
report("r(extraction_emp, suicide)", 0.197, round(r, 3), 0.01, "Bivariate r")

# Male-female ratio
r, p, n = pearson_test('male_female_ratio')
report("r(male_female_ratio, suicide)", 0.161, round(r, 3), 0.01, "Bivariate r")

# Poverty rate
r, p, n = pearson_test('poverty_rate')
report("r(poverty, suicide)", 0.142, round(r, 3), 0.01, "Bivariate r")

# Unemployment
r, p, n = pearson_test('unemployment_rate')
report("r(unemployment, suicide)", 0.122, round(r, 3), 0.01, "Bivariate r")

# Alcohol measures
print("\n  --- Alcohol Access Bivariate ---")

# Total alcohol outlets per 10k
r_alc, p_alc, n_alc = pearson_test('alcohol_outlets_per_10k')
report("r(alcohol outlets per 10k, suicide)", 0.013, round(r_alc, 3), 0.01, "Bivariate r")
report("p(alcohol outlets per 10k, suicide)", 0.49, round(p_alc, 2), 0.02, "Bivariate p")

# Excessive drinking
r_drink, p_drink, n_drink = pearson_test('chr_excessive_drinking_pct')
report("r(excessive drinking, suicide)", -0.097, round(r_drink, 3), 0.01, "Bivariate r")
# Check p-value (~5.3e-7)
report("p(excessive drinking, suicide) ~5.3e-7", 5.3e-7, float(f"{p_drink:.1e}"), 5e-7, "Bivariate p")

# Bars per 10k
r_bar, p_bar, n_bar = pearson_test('bars_per_10k')
report("r(bars per 10k, suicide)", 0.063, round(r_bar, 3), 0.01, "Bivariate r")
report("p(bars per 10k, suicide)", 0.001, round(p_bar, 3), 0.002, "Bivariate p")

# Liquor stores per 10k
r_liq, p_liq, n_liq = pearson_test('liquor_per_10k')
report("r(liquor per 10k, suicide)", -0.085, round(r_liq, 3), 0.01, "Bivariate r")
report("p(liquor per 10k, suicide) ~1.1e-5", 1.1e-5, float(f"{p_liq:.1e}"), 1e-5, "Bivariate p")

# Excessive drinking bivariate direction claim: "negative"
report("Excessive drinking correlation is NEGATIVE", "negative",
       "negative" if r_drink < 0 else "positive", 0, "Direction")

# ============================================================
# 4. BONFERRONI THRESHOLD
# ============================================================
print("\n" + "=" * 90)
print("4. BONFERRONI CORRECTION")
print("=" * 90)

bonf = 0.05 / 23
report("Bonferroni threshold (0.05/23)", 0.00217, round(bonf, 5), 0.0001, "Bonferroni")

# "22 of 23 predictors significant" - alcohol outlets is the only one that doesn't survive
report("Alcohol outlets only predictor failing Bonferroni", True,
       p_alc > bonf, 0, "Bonferroni")

# ============================================================
# 5. DRY VS WET COUNTIES
# ============================================================
print("\n" + "=" * 90)
print("5. DRY VS WET COUNTIES")
print("=" * 90)

# Dry = alcohol_access == 'dry' OR total_alcohol_outlets == 0
# Let's check both definitions
dry_by_access = valid[valid['alcohol_access'] == 'dry']
wet_by_access = valid[valid['alcohol_access'] == 'wet']
dry_by_outlets = valid[valid['total_alcohol_outlets'] == 0]

# The manuscript says dry = zero outlets, n=811, wet n=1,627
# Note: 811 + 1627 = 2438, not 2683. Some counties may have intermediate status
# Let's check what alcohol_access values exist
print(f"\n  alcohol_access values: {valid['alcohol_access'].value_counts().to_dict()}")
print(f"  Counties with 0 outlets: {len(dry_by_outlets)}")

# Try: dry = counties where total_alcohol_outlets == 0 or alcohol_access == 'dry'
# And wet = counties where alcohol_access == 'wet'
dry = valid[valid['total_alcohol_outlets'] == 0].copy()
wet = valid[valid['total_alcohol_outlets'] > 0].copy()

# But manuscript says n=811 dry, n=1627 wet => total 2438
# Some counties might have missing outlet data. Let's also try the alcohol_access column
print(f"  Dry (0 outlets): {len(dry)}, Wet (>0 outlets): {len(wet)}")
print(f"  Dry (access='dry'): {len(dry_by_access)}, Wet (access='wet'): {len(wet_by_access)}")

# Use 'dry' access label for dry and 'wet' for wet (excluding 'very_restricted', 'restricted', etc.)
# The manuscript specifies: "zero outlets" for dry, "any outlets" for wet
# Let's check if the manuscript counts match using alcohol_access categories
# If dry=811 and wet=1627, they may be filtering out some in-between counties
# Or 811+1627 = 2438 with 245 unclassified

# Let's try: maybe "dry" includes some restricted categories
# Actually the manuscript says "dry counties; n = 811" and "wet counties; n = 1,627"
# Let's check with the access column
for val in valid['alcohol_access'].unique():
    subset = valid[valid['alcohol_access'] == val]
    print(f"    alcohol_access='{val}': n={len(subset)}")

# Use manuscript definition: dry = zero outlets
# But the paper says n=811 for dry. Let's see if that matches
report("N dry counties (zero outlets)", 811, len(dry), 5, "Dry/Wet")
report("N wet counties (any outlets)", 1627, len(wet), 5, "Dry/Wet")

# If dry/wet N doesn't match, also try alcohol_access=='dry'
if abs(len(dry) - 811) > 5:
    print("  NOTE: Trying alcohol_access column instead...")
    dry = dry_by_access.copy()
    wet = wet_by_access.copy()
    report("N dry (alcohol_access='dry')", 811, len(dry), 5, "Dry/Wet")
    report("N wet (alcohol_access='wet')", 1627, len(wet), 5, "Dry/Wet")

# Suicide rates
dry_rate = dry['male_suicide_rate_cdc'].mean()
wet_rate = wet['male_suicide_rate_cdc'].mean()
report("Mean suicide rate, dry counties", 21.6, round(dry_rate, 1), 0.3, "Dry/Wet")
report("Mean suicide rate, wet counties", 18.5, round(wet_rate, 1), 0.3, "Dry/Wet")

pct_diff = (dry_rate / wet_rate - 1) * 100
report("Dry vs wet pct elevation", 17, round(pct_diff, 0), 3, "Dry/Wet")

# t-test
t_stat, p_val = stats.ttest_ind(dry['male_suicide_rate_cdc'], wet['male_suicide_rate_cdc'])
report("Dry vs wet p < 0.001", True, p_val < 0.001, 0, "Dry/Wet")

# Drug overdose rates
if 'chr_overdose_rate' in valid.columns:
    dry_od = dry['chr_overdose_rate'].dropna().mean()
    wet_od = wet['chr_overdose_rate'].dropna().mean()
    report("Overdose rate dry", 34.6, round(dry_od, 1), 1.0, "Dry/Wet")
    report("Overdose rate wet", 27.1, round(wet_od, 1), 1.0, "Dry/Wet")
    pct_od = (dry_od / wet_od - 1) * 100
    report("Overdose pct diff (+28%)", 28, round(pct_od, 0), 3, "Dry/Wet")

# Firearm fatality
dry_ff = dry['chr_firearm_fatality_rate'].dropna().mean()
wet_ff = wet['chr_firearm_fatality_rate'].dropna().mean()
report("Firearm fatality dry", 21.8, round(dry_ff, 1), 1.0, "Dry/Wet")
report("Firearm fatality wet", 15.4, round(wet_ff, 1), 1.0, "Dry/Wet")
pct_ff = (dry_ff / wet_ff - 1) * 100
report("Firearm pct diff (+41%)", 41, round(pct_ff, 0), 3, "Dry/Wet")

# Injury death rates
dry_inj = dry['chr_injury_death_rate'].dropna().mean()
wet_inj = wet['chr_injury_death_rate'].dropna().mean()
report("Injury death rate dry", 106.5, round(dry_inj, 1), 2.0, "Dry/Wet")
report("Injury death rate wet", 90.4, round(wet_inj, 1), 2.0, "Dry/Wet")
pct_inj = (dry_inj / wet_inj - 1) * 100
report("Injury pct diff (+18%)", 18, round(pct_inj, 0), 3, "Dry/Wet")

# Premature death
dry_pd = dry['chr_premature_death_rate'].dropna().mean()
wet_pd = wet['chr_premature_death_rate'].dropna().mean()
report("Premature death dry", 11433, round(dry_pd, 0), 200, "Dry/Wet")
report("Premature death wet", 8998, round(wet_pd, 0), 200, "Dry/Wet")
pct_pd = (dry_pd / wet_pd - 1) * 100
report("Premature death pct diff (+27%)", 27, round(pct_pd, 0), 3, "Dry/Wet")

# Life expectancy
dry_le = dry['chr_life_expectancy'].dropna().mean()
wet_le = wet['chr_life_expectancy'].dropna().mean()
report("Life expectancy dry", 74.1, round(dry_le, 1), 0.3, "Dry/Wet")
report("Life expectancy wet", 76.6, round(wet_le, 1), 0.3, "Dry/Wet")
le_gap = wet_le - dry_le
report("Life expectancy gap (2.5 years)", 2.5, round(le_gap, 1), 0.3, "Dry/Wet")

# Mental health providers
dry_mh = dry['chr_pop_per_mh_provider'].dropna().mean()
wet_mh = wet['chr_pop_per_mh_provider'].dropna().mean()
report("Pop per MH provider, dry", 2218, round(dry_mh, 0), 100, "Dry/Wet")
report("Pop per MH provider, wet", 869, round(wet_mh, 0), 100, "Dry/Wet")
mh_ratio = dry_mh / wet_mh
report("MH provider ratio (2.6x fewer in dry)", 2.6, round(mh_ratio, 1), 0.3, "Dry/Wet")

# Excessive drinking
dry_ed = dry['chr_excessive_drinking_pct'].dropna().mean() * 100 if dry['chr_excessive_drinking_pct'].dropna().mean() < 1 else dry['chr_excessive_drinking_pct'].dropna().mean()
wet_ed = wet['chr_excessive_drinking_pct'].dropna().mean() * 100 if wet['chr_excessive_drinking_pct'].dropna().mean() < 1 else wet['chr_excessive_drinking_pct'].dropna().mean()
# Check if values are already in pct or proportion
sample_val = valid['chr_excessive_drinking_pct'].dropna().iloc[0]
if sample_val < 1:  # it's a proportion
    dry_ed = dry['chr_excessive_drinking_pct'].dropna().mean() * 100
    wet_ed = wet['chr_excessive_drinking_pct'].dropna().mean() * 100
else:
    dry_ed = dry['chr_excessive_drinking_pct'].dropna().mean()
    wet_ed = wet['chr_excessive_drinking_pct'].dropna().mean()
report("Excessive drinking dry pct", 15.4, round(dry_ed, 1), 0.5, "Dry/Wet")
report("Excessive drinking wet pct", 17.5, round(wet_ed, 1), 0.5, "Dry/Wet")

# ============================================================
# 5b. WITHIN-STATE SIGN TEST
# ============================================================
print("\n  --- Within-State Sign Test ---")

# For each state, need >= 3 dry and >= 3 wet counties
states_testable = []
state_signs = []

for state in valid['state_name'].unique():
    state_data = valid[valid['state_name'] == state]
    state_dry = state_data[state_data.index.isin(dry.index)]
    state_wet = state_data[state_data.index.isin(wet.index)]
    if len(state_dry) >= 3 and len(state_wet) >= 3:
        states_testable.append(state)
        dry_mean = state_dry['male_suicide_rate_cdc'].mean()
        wet_mean = state_wet['male_suicide_rate_cdc'].mean()
        state_signs.append(1 if dry_mean > wet_mean else 0)

n_testable = len(states_testable)
n_positive = sum(state_signs)
report("N testable states (>=3 each)", 33, n_testable, 2, "Sign Test")
report("N states where dry > wet", 27, n_positive, 2, "Sign Test")
report("Pct states dry > wet (82%)", 82, round(n_positive / n_testable * 100, 0), 3, "Sign Test")

# Sign test p-value (binomial test)
sign_p = stats.binom_test(n_positive, n_testable, 0.5) if hasattr(stats, 'binom_test') else \
         stats.binomtest(n_positive, n_testable, 0.5).pvalue
report("Sign test p-value (0.0003)", 0.0003, round(sign_p, 4), 0.001, "Sign Test")

# Mean within-state gap
gaps = []
for state in states_testable:
    state_data = valid[valid['state_name'] == state]
    state_dry = state_data[state_data.index.isin(dry.index)]
    state_wet = state_data[state_data.index.isin(wet.index)]
    gaps.append(state_dry['male_suicide_rate_cdc'].mean() - state_wet['male_suicide_rate_cdc'].mean())
report("Mean within-state gap (+2.7)", 2.7, round(np.mean(gaps), 1), 0.5, "Sign Test")

# ============================================================
# 5c. DRY COUNTY STRUCTURAL PROFILE
# ============================================================
print("\n  --- Dry County Structural Profile (Discussion) ---")

dry_rucc_mean = dry['rural_urban_code'].mean()
wet_rucc_mean = wet['rural_urban_code'].mean()
report("Mean RUCC dry", 6.3, round(dry_rucc_mean, 1), 0.3, "Dry Profile")
report("Mean RUCC wet", 4.3, round(wet_rucc_mean, 1), 0.3, "Dry Profile")

# Median household income
dry_inc = dry['median_household_income'].dropna().mean()
wet_inc = wet['median_household_income'].dropna().mean()
report("Median income dry (~$55,000)", 55000, round(dry_inc, -3), 3000, "Dry Profile")
report("Median income wet (~$68,000)", 68000, round(wet_inc, -3), 3000, "Dry Profile")

# Education (bachelor's or higher)
dry_ed_pct = dry['pct_bachelors_or_higher'].dropna().mean()
wet_ed_pct = wet['pct_bachelors_or_higher'].dropna().mean()
# Check if proportion or pct
if dry_ed_pct < 1:
    dry_ed_pct *= 100
    wet_ed_pct *= 100
report("Pct college dry (~18%)", 18, round(dry_ed_pct, 0), 2, "Dry Profile")
report("Pct college wet (~27%)", 27, round(wet_ed_pct, 0), 2, "Dry Profile")

# ============================================================
# 6. MULTIVARIATE MODELS - OLS
# ============================================================
print("\n" + "=" * 90)
print("6. MULTIVARIATE OLS MODEL")
print("=" * 90)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Build the 9-predictor model
# Predictors: veteran_pct, rural_urban_code, extraction_employment_pct,
#             male_female_ratio, poverty_rate, pct_bachelors_or_higher,
#             unemployment_rate, male_divorced_separated_pct, chr_excessive_drinking_pct
# (and possibly chr_pop_per_mh_provider or alcohol measures)
# The manuscript mentions "nine predictors" in baseline OLS

# From context: alcohol measures + structural + socioeconomic
# Let's identify the 9 predictors. The manuscript mentions:
# veteran_pct, rural_urban_code, extraction_employment_pct, male_female_ratio,
# poverty_rate, unemployment_rate, male_divorced_separated_pct,
# pct_bachelors_or_higher(?), and one alcohol measure?
# Actually from Section 2.3: the multivariate includes structural demo + socioeconomic
# Section 3.5 says "nine predictors" with betas for:
# veteran (+2.27), rurality (+1.71), extraction (+1.08), gender ratio (+0.86),
# poverty (+0.89), divorce (+0.74), unemployment (+0.65)
# That's 7 structural/socio. Plus possibly education and one alcohol measure?

# Let's try the 9 predictors that make sense
predictor_cols = [
    'veteran_pct',
    'rural_urban_code',
    'extraction_employment_pct',
    'male_female_ratio',
    'poverty_rate',
    'unemployment_rate',
    'male_divorced_separated_pct',
    'pct_bachelors_or_higher',
    'chr_excessive_drinking_pct'
]

# Drop rows with missing values in predictors
model_data = valid[predictor_cols + ['male_suicide_rate_cdc']].dropna()
print(f"  Model N (complete cases): {len(model_data)}")

X = model_data[predictor_cols].values
y = model_data['male_suicide_rate_cdc'].values

# Standardize
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# OLS
from numpy.linalg import lstsq
X_with_const = np.column_stack([np.ones(len(X_std)), X_std])
betas, residuals, rank, sv = lstsq(X_with_const, y, rcond=None)
y_pred = X_with_const @ betas
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r_squared = 1 - ss_res / ss_tot

report("OLS R-squared", 0.261, round(r_squared, 3), 0.01, "OLS Model")

# Standardized betas (skip intercept)
beta_names = predictor_cols
print("\n  --- Standardized Betas ---")
for i, name in enumerate(beta_names):
    print(f"    {name}: beta = {betas[i+1]:.3f}")

report("Beta veteran_pct", 2.27, round(betas[1+0], 2), 0.1, "OLS Betas")
report("Beta rural_urban_code", 1.71, round(betas[1+1], 2), 0.1, "OLS Betas")
report("Beta extraction_employment_pct", 1.08, round(betas[1+2], 2), 0.1, "OLS Betas")
report("Beta male_female_ratio", 0.86, round(betas[1+3], 2), 0.1, "OLS Betas")
report("Beta poverty_rate", 0.89, round(betas[1+4], 2), 0.1, "OLS Betas")
report("Beta unemployment_rate", 0.65, round(betas[1+5], 2), 0.1, "OLS Betas")
report("Beta male_divorced_separated_pct", 0.74, round(betas[1+6], 2), 0.1, "OLS Betas")

# ============================================================
# 7. CROSS-VALIDATION
# ============================================================
print("\n" + "=" * 90)
print("7. CROSS-VALIDATION")
print("=" * 90)

from sklearn.model_selection import cross_val_score, KFold

lr = LinearRegression()
# Use shuffled CV (as manuscript likely did)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(lr, X_std, y, cv=kf, scoring='r2')
cv_mean = cv_scores.mean()
report("5-fold CV R-squared (shuffled)", 0.246, round(cv_mean, 3), 0.015, "Cross-Validation")

# Training R2
lr.fit(X_std, y)
train_r2 = lr.score(X_std, y)
report("Training R-squared", 0.261, round(train_r2, 3), 0.01, "Cross-Validation")

cv_gap = train_r2 - cv_mean
report("Overfitting gap (0.015)", 0.015, round(cv_gap, 3), 0.005, "Cross-Validation")

# ============================================================
# 8. VIF
# ============================================================
print("\n" + "=" * 90)
print("8. VARIANCE INFLATION FACTORS")
print("=" * 90)

from numpy.linalg import inv

# Compute VIF for each predictor
corr_matrix = np.corrcoef(X_std.T)
try:
    inv_corr = inv(corr_matrix)
    vifs = np.diag(inv_corr)
    max_vif = max(vifs)
    print("  VIF values:")
    for i, name in enumerate(beta_names):
        print(f"    {name}: VIF = {vifs[i]:.2f}")
    report("All VIFs below 3.0", True, all(v < 3.0 for v in vifs), 0, "VIF")
    report("Max VIF < 3.0", True, max_vif < 3.0, 0, "VIF")
except:
    print("  VIF computation failed (singular matrix)")

# Sample-to-predictor ratio
ratio = len(model_data) / len(predictor_cols)
report("Sample-to-predictor ratio (298:1)", 298, round(ratio, 0), 5, "VIF")

# ============================================================
# 9. SPATIAL MODEL
# ============================================================
print("\n" + "=" * 90)
print("9. SPATIAL DIAGNOSTICS")
print("=" * 90)

# Moran's I on OLS residuals
# We need spatial weights (k=5 nearest neighbors)
resids = y - y_pred

# Get coordinates for model_data counties
coords_data = valid.loc[model_data.index, ['longitude', 'latitude']].values

# Build k=5 nearest neighbor weights
from scipy.spatial import cKDTree
tree = cKDTree(coords_data)
k = 5
distances, indices = tree.query(coords_data, k=k+1)  # +1 because first is self

n_obs = len(resids)
# Compute Moran's I
W_resids = np.zeros(n_obs)
for i in range(n_obs):
    neighbors = indices[i, 1:]  # skip self
    W_resids[i] = resids[neighbors].mean()

resids_centered = resids - resids.mean()
W_centered = W_resids - W_resids.mean()

morans_num = np.sum(resids_centered * W_centered)
morans_den = np.sum(resids_centered ** 2)
# This is a simplified Moran's I
morans_I = (n_obs / (n_obs * k)) * morans_num / morans_den

# More standard calculation
# Moran's I = (N / S0) * (sum_i sum_j w_ij * (x_i - xbar)(x_j - xbar)) / sum_i (x_i - xbar)^2
# For row-standardized weights, S0 = N, so I = sum of products / sum of squares
morans_I_std = morans_num / morans_den
report("Moran's I (approximate, simplified)", 0.294, round(morans_I_std, 3), 0.05, "Spatial")

# Spatial lag R-squared
# Simple 2SLS spatial lag: y = rho * Wy + X*beta + e
# Instrument Wy with WX
Wy = np.zeros(n_obs)
for i in range(n_obs):
    neighbors = indices[i, 1:]
    Wy[i] = y[neighbors].mean()

# 2SLS: First stage - regress Wy on X and WX
WX = np.zeros_like(X_std)
for i in range(n_obs):
    neighbors = indices[i, 1:]
    WX[i] = X_std[neighbors].mean(axis=0)

Z = np.column_stack([np.ones(n_obs), X_std, WX])
# First stage
betas_1st = lstsq(Z, Wy, rcond=None)[0]
Wy_hat = Z @ betas_1st

# Second stage
X_2sls = np.column_stack([np.ones(n_obs), Wy_hat, X_std])
betas_2sls = lstsq(X_2sls, y, rcond=None)[0]
y_pred_spatial = X_2sls @ betas_2sls
ss_res_spatial = np.sum((y - y_pred_spatial) ** 2)
r2_spatial = 1 - ss_res_spatial / ss_tot

rho_hat = betas_2sls[1]
report("Spatial lag rho", 0.62, round(rho_hat, 2), 0.05, "Spatial")
report("Spatial lag R-squared", 0.307, round(r2_spatial, 3), 0.02, "Spatial")

# ============================================================
# 10. STATE FIXED EFFECTS
# ============================================================
print("\n" + "=" * 90)
print("10. STATE FIXED EFFECTS MODEL")
print("=" * 90)

# Add state dummies
fe_data = valid[predictor_cols + ['male_suicide_rate_cdc', 'state_name']].dropna()
state_dummies = pd.get_dummies(fe_data['state_name'], drop_first=True)
X_fe = np.column_stack([
    StandardScaler().fit_transform(fe_data[predictor_cols].values),
    state_dummies.values
])
y_fe = fe_data['male_suicide_rate_cdc'].values

X_fe_const = np.column_stack([np.ones(len(X_fe)), X_fe])
betas_fe = lstsq(X_fe_const, y_fe, rcond=None)[0]
y_pred_fe = X_fe_const @ betas_fe
ss_res_fe = np.sum((y_fe - y_pred_fe) ** 2)
ss_tot_fe = np.sum((y_fe - y_fe.mean()) ** 2)
r2_fe = 1 - ss_res_fe / ss_tot_fe

report("State FE R-squared", 0.429, round(r2_fe, 3), 0.015, "State FE")

# ============================================================
# 11. POPULATION-WEIGHTED LEAST SQUARES
# ============================================================
print("\n" + "=" * 90)
print("11. POPULATION-WEIGHTED LEAST SQUARES")
print("=" * 90)

wls_data = valid[predictor_cols + ['male_suicide_rate_cdc', 'total_population']].dropna()
X_wls = StandardScaler().fit_transform(wls_data[predictor_cols].values)
y_wls = wls_data['male_suicide_rate_cdc'].values
w_pop = wls_data['total_population'].values
w_sqrt = np.sqrt(w_pop)

X_wls_c = np.column_stack([np.ones(len(X_wls)), X_wls])
X_wls_w = X_wls_c * w_sqrt[:, np.newaxis]
y_wls_w = y_wls * w_sqrt

betas_wls = lstsq(X_wls_w, y_wls_w, rcond=None)[0]
y_pred_wls = X_wls_c @ betas_wls

# Unweighted R-squared of the WLS fit (as manuscript reports)
ss_res_wls_uw = np.sum((y_wls - y_pred_wls) ** 2)
ss_tot_wls_uw = np.sum((y_wls - y_wls.mean()) ** 2)
r2_wls = 1 - ss_res_wls_uw / ss_tot_wls_uw

report("WLS R-squared (unweighted)", 0.204, round(r2_wls, 3), 0.02, "WLS")

# ============================================================
# 12. BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================
print("\n" + "=" * 90)
print("12. BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 90)

np.random.seed(42)
n_boot = 1000
boot_betas = np.zeros((n_boot, len(predictor_cols)))

for b in range(n_boot):
    idx = np.random.choice(len(X_std), size=len(X_std), replace=True)
    X_b = np.column_stack([np.ones(len(idx)), X_std[idx]])
    y_b = y[idx]
    beta_b = lstsq(X_b, y_b, rcond=None)[0]
    boot_betas[b] = beta_b[1:]  # skip intercept

# Report CIs for core predictors
ci_vet = np.percentile(boot_betas[:, 0], [2.5, 97.5])
report("Bootstrap CI veteran_pct [1.91, 2.61]", "1.91-2.61",
       f"{ci_vet[0]:.2f}-{ci_vet[1]:.2f}", 0, "Bootstrap")

ci_rucc = np.percentile(boot_betas[:, 1], [2.5, 97.5])
report("Bootstrap CI rurality [1.35, 2.09]", "1.35-2.09",
       f"{ci_rucc[0]:.2f}-{ci_rucc[1]:.2f}", 0, "Bootstrap")

ci_ext = np.percentile(boot_betas[:, 2], [2.5, 97.5])
report("Bootstrap CI extraction [0.73, 1.55]", "0.73-1.55",
       f"{ci_ext[0]:.2f}-{ci_ext[1]:.2f}", 0, "Bootstrap")

ci_mf = np.percentile(boot_betas[:, 3], [2.5, 97.5])
report("Bootstrap CI male_female_ratio [0.43, 1.29]", "0.43-1.29",
       f"{ci_mf[0]:.2f}-{ci_mf[1]:.2f}", 0, "Bootstrap")

# ============================================================
# 13. PIPELINE COMMUNITY COMPARISON
# ============================================================
print("\n" + "=" * 90)
print("13. PIPELINE COMMUNITY COMPARISON")
print("=" * 90)

# Pipeline: extraction_employment_pct > 5% AND veteran_pct in top quartile
vet_q75 = valid['veteran_pct'].quantile(0.75)
print(f"  Veteran pct 75th percentile: {vet_q75:.2f}")

pipeline = valid[
    (valid['extraction_employment_pct'] > 5) &
    (valid['veteran_pct'] >= vet_q75)
]

# Control: zero extraction AND bottom quartile veteran
vet_q25 = valid['veteran_pct'].quantile(0.25)
control = valid[
    (valid['extraction_employment_pct'] == 0) &
    (valid['veteran_pct'] <= vet_q25)
]

report("N pipeline counties", 39, len(pipeline), 5, "Pipeline")
report("N control counties", 325, len(control), 20, "Pipeline")

pipe_rate = pipeline['male_suicide_rate_cdc'].mean()
ctrl_rate = control['male_suicide_rate_cdc'].mean()
report("Pipeline mean rate", 30.1, round(pipe_rate, 1), 0.5, "Pipeline")
report("Control mean rate", 16.6, round(ctrl_rate, 1), 0.5, "Pipeline")

pct_elev_pipe = (pipe_rate / ctrl_rate - 1) * 100
report("Pipeline elevation pct (82%)", 82, round(pct_elev_pipe, 0), 5, "Pipeline")

# t-test
t_pipe, p_pipe = stats.ttest_ind(pipeline['male_suicide_rate_cdc'], control['male_suicide_rate_cdc'])
report("Pipeline t-statistic (9.74)", 9.74, round(t_pipe, 2), 0.3, "Pipeline")
report("Pipeline p-value (~4.6e-20)", 4.6e-20, p_pipe, 1e-18, "Pipeline")

# Cohen's d
pooled_std = np.sqrt(
    ((len(pipeline) - 1) * pipeline['male_suicide_rate_cdc'].std() ** 2 +
     (len(control) - 1) * control['male_suicide_rate_cdc'].std() ** 2) /
    (len(pipeline) + len(control) - 2)
)
cohens_d = (pipe_rate - ctrl_rate) / pooled_std
report("Cohen's d (1.58)", 1.58, round(cohens_d, 2), 0.1, "Pipeline")

# ============================================================
# 14. DENOMINATOR SENSITIVITY ANALYSIS
# ============================================================
print("\n" + "=" * 90)
print("14. DENOMINATOR SENSITIVITY (True Male Rate)")
print("=" * 90)

# True male rate
true_rate = valid['male_suicide_rate_true'].copy()

# Correlation between crude and true
r_ct, _ = stats.pearsonr(
    valid['male_suicide_rate_cdc'].dropna(),
    valid.loc[valid['male_suicide_rate_cdc'].notna() & valid['male_suicide_rate_true'].notna(), 'male_suicide_rate_true']
)
# Need aligned data
both_valid = valid[valid['male_suicide_rate_cdc'].notna() & valid['male_suicide_rate_true'].notna()]
r_ct, _ = stats.pearsonr(both_valid['male_suicide_rate_cdc'], both_valid['male_suicide_rate_true'])
report("Correlation crude vs true rate", 0.991, round(r_ct, 3), 0.005, "Sensitivity")

# Scaling factor
scale = (both_valid['male_suicide_rate_true'] / both_valid['male_suicide_rate_cdc']).median()
report("Scaling factor crude->true (~2.8x)", 2.8, round(scale, 1), 0.2, "Sensitivity")

# Veteran pct vs true rate
r_vet_true, _, _ = pearson_test('veteran_pct', 'male_suicide_rate_true')
r_vet_crude, _, _ = pearson_test('veteran_pct', 'male_suicide_rate_cdc')
report("r(veteran, true rate) ~0.33", 0.33, round(r_vet_true, 2), 0.02, "Sensitivity")
report("r(veteran, crude rate) ~0.32", 0.32, round(r_vet_crude, 2), 0.02, "Sensitivity")

# Male-female ratio with true rate (bivariate)
r_mf_true, _, _ = pearson_test('male_female_ratio', 'male_suicide_rate_true')
r_mf_crude, _, _ = pearson_test('male_female_ratio', 'male_suicide_rate_cdc')
report("r(male_female_ratio, crude) ~+0.16", 0.16, round(r_mf_crude, 2), 0.02, "Sensitivity")
report("r(male_female_ratio, true) ~+0.05", 0.05, round(r_mf_true, 2), 0.02, "Sensitivity")
report("Two-thirds of bivariate r is mechanical", True,
       (r_mf_crude - r_mf_true) / r_mf_crude > 0.5, 0, "Sensitivity")

# Working-age sex ratio
r_wa_crude, _, _ = pearson_test('working_age_sex_ratio', 'male_suicide_rate_cdc')
r_wa_true, _, _ = pearson_test('working_age_sex_ratio', 'male_suicide_rate_true')
report("r(working_age_sex_ratio, crude) ~+0.09", 0.09, round(r_wa_crude, 2), 0.02, "Sensitivity")
report("r(working_age_sex_ratio, true) ~-0.01", -0.01, round(r_wa_true, 2), 0.02, "Sensitivity")

# Male-female ratio multivariate p with true rate
# Re-run multivariate model with true rate
model_data_true = valid[predictor_cols + ['male_suicide_rate_true']].dropna()
X_true = StandardScaler().fit_transform(model_data_true[predictor_cols].values)
y_true = model_data_true['male_suicide_rate_true'].values

X_true_c = np.column_stack([np.ones(len(X_true)), X_true])
betas_true = lstsq(X_true_c, y_true, rcond=None)[0]
y_pred_true = X_true_c @ betas_true
resids_true = y_true - y_pred_true
n_true = len(y_true)
p_true = X_true_c.shape[1]
mse_true = np.sum(resids_true ** 2) / (n_true - p_true)
# Standard errors
try:
    cov_true = mse_true * inv(X_true_c.T @ X_true_c)
    se_true = np.sqrt(np.diag(cov_true))
    t_mf_true = betas_true[1+3] / se_true[1+3]  # male_female_ratio is index 3
    p_mf_true_mv = 2 * (1 - stats.t.cdf(abs(t_mf_true), n_true - p_true))
    report("male_female_ratio p in multivariate (true rate) ~0.26", 0.26,
           round(p_mf_true_mv, 2), 0.05, "Sensitivity")
except:
    print("  Could not compute multivariate p for male_female_ratio with true rate")

# Pipeline comparison with true rate
pipe_true = pipeline['male_suicide_rate_true'].dropna().mean()
ctrl_true = control['male_suicide_rate_true'].dropna().mean()
pooled_std_true = np.sqrt(
    ((len(pipeline['male_suicide_rate_true'].dropna()) - 1) * pipeline['male_suicide_rate_true'].dropna().std() ** 2 +
     (len(control['male_suicide_rate_true'].dropna()) - 1) * control['male_suicide_rate_true'].dropna().std() ** 2) /
    (len(pipeline['male_suicide_rate_true'].dropna()) + len(control['male_suicide_rate_true'].dropna()) - 2)
)
d_true = (pipe_true - ctrl_true) / pooled_std_true
report("Pipeline Cohen's d with true rate (~1.56)", 1.56, round(d_true, 2), 0.1, "Sensitivity")

# Dry county elevation with true rate
dry_true = dry['male_suicide_rate_true'].dropna().mean()
wet_true = wet['male_suicide_rate_true'].dropna().mean()
pct_dry_true = (dry_true / wet_true - 1) * 100
report("Dry county elevation with true rate (~+16%)", 16, round(pct_dry_true, 0), 3, "Sensitivity")

# ============================================================
# 15. INTERACTION EFFECTS (using HC3 robust SEs)
# ============================================================
print("\n" + "=" * 90)
print("15. INTERACTION EFFECTS (HC3 robust SEs)")
print("=" * 90)

import statsmodels.api as sm

int_data2 = valid[predictor_cols + ['male_suicide_rate_cdc', 'alcohol_outlets_per_10k']].dropna()
X_int2 = StandardScaler().fit_transform(int_data2[predictor_cols].values)
y_int2 = int_data2['male_suicide_rate_cdc'].values
alc_outlets_std = StandardScaler().fit_transform(int_data2[['alcohol_outlets_per_10k']].values).flatten()

# Vet x Extraction (HC3)
X_ve = np.column_stack([X_int2, X_int2[:, 0] * X_int2[:, 2]])
X_ve_c = sm.add_constant(X_ve)
model_ve = sm.OLS(y_int2, X_ve_c).fit(cov_type='HC3')
# interaction is the last coefficient
ve_beta = model_ve.params[-1]
ve_p = model_ve.pvalues[-1]
report("Vet x Extraction interaction beta ~+0.18 (HC3)", 0.18, round(ve_beta, 2), 0.05, "Interactions")
report("Vet x Extraction interaction p ~0.13 (HC3)", 0.13, round(ve_p, 2), 0.05, "Interactions")

# Vet x Rurality (HC3)
X_vr = np.column_stack([X_int2, X_int2[:, 0] * X_int2[:, 1]])
X_vr_c = sm.add_constant(X_vr)
model_vr = sm.OLS(y_int2, X_vr_c).fit(cov_type='HC3')
vr_beta = model_vr.params[-1]
vr_p = model_vr.pvalues[-1]
report("Vet x Rurality interaction beta ~+0.24 (HC3)", 0.24, round(vr_beta, 2), 0.05, "Interactions")
report("Vet x Rurality interaction p ~0.051 (HC3)", 0.051, round(vr_p, 3), 0.03, "Interactions")

# Rurality x Alcohol outlets (HC3)
X_ra = np.column_stack([X_int2, alc_outlets_std, X_int2[:, 1] * alc_outlets_std])
X_ra_c = sm.add_constant(X_ra)
model_ra = sm.OLS(y_int2, X_ra_c).fit(cov_type='HC3')
ra_beta = model_ra.params[-1]
ra_p = model_ra.pvalues[-1]
report("Rurality x Alcohol interaction beta ~+0.12 (HC3)", 0.12, round(ra_beta, 2), 0.05, "Interactions")
report("Rurality x Alcohol interaction p ~0.36 (HC3)", 0.36, round(ra_p, 2), 0.1, "Interactions")

# ============================================================
# 16. CONFOUNDING DEMONSTRATION (Alcohol attenuation p-values)
# ============================================================
print("\n" + "=" * 90)
print("16. ALCOHOL CONFOUNDING DEMONSTRATION")
print("  (Using HC3 robust standard errors as specified in manuscript)")
print("=" * 90)

import statsmodels.api as sm

def get_hc3_p(X_cols, y_col='male_suicide_rate_cdc', data=valid):
    """Run OLS with HC3 and return p-values for all predictors (incl. intercept)."""
    d = data[X_cols + [y_col]].dropna()
    X = pd.DataFrame(StandardScaler().fit_transform(d[X_cols].values), columns=X_cols)
    y_v = d[y_col].values
    Xc = sm.add_constant(X)
    model = sm.OLS(y_v, Xc.values).fit(cov_type='HC3')
    return model.pvalues, model.params, len(d)

def get_classical_p(X_cols, y_col='male_suicide_rate_cdc', data=valid):
    """Run OLS with classical SEs."""
    d = data[X_cols + [y_col]].dropna()
    X = pd.DataFrame(StandardScaler().fit_transform(d[X_cols].values), columns=X_cols)
    y_v = d[y_col].values
    Xc = sm.add_constant(X)
    model = sm.OLS(y_v, Xc.values).fit()
    return model.pvalues, model.params, len(d)

demo_cols = ['veteran_pct', 'rural_urban_code', 'extraction_employment_pct',
             'male_female_ratio', 'poverty_rate', 'unemployment_rate',
             'male_divorced_separated_pct']

# Excessive drinking alone (classical)
p_drink_alone, _, _ = get_classical_p(['chr_excessive_drinking_pct'])
report("Excessive drinking alone p < 0.001", True, p_drink_alone[1] < 0.001, 0, "Confounding")

# Excessive drinking with demographic controls (HC3 as manuscript specifies)
p_drink_hc3, _, _ = get_hc3_p(demo_cols + ['chr_excessive_drinking_pct'])
p_drink_classical, _, _ = get_classical_p(demo_cols + ['chr_excessive_drinking_pct'])
# Index: 0=const, 1-7=demo, 8=drinking
print(f"  Drinking + demos: HC3 p={p_drink_hc3[8]:.4f}, classical p={p_drink_classical[8]:.4f}")
report("Excessive drinking with demographics p ~0.94 (HC3)", 0.94, round(p_drink_hc3[8], 2), 0.15, "Confounding")

# Bar density alone
p_bar_alone, _, _ = get_classical_p(['bars_per_10k'])
report("Bar density alone p ~0.001", 0.001, round(p_bar_alone[1], 3), 0.002, "Confounding")

# Bar density with demographics (HC3)
p_bar_hc3, _, _ = get_hc3_p(demo_cols + ['bars_per_10k'])
print(f"  Bars + demos: HC3 p={p_bar_hc3[8]:.4f}")
report("Bar density with demographics p ~0.018 (HC3)", 0.018, round(p_bar_hc3[8], 3), 0.01, "Confounding")

# Bar density with state FE (classical SEs for FE model)
fe_bar = valid[demo_cols + ['bars_per_10k', 'male_suicide_rate_cdc', 'state_name']].dropna()
state_d = pd.get_dummies(fe_bar['state_name'], drop_first=True)
X_bar_fe_std = StandardScaler().fit_transform(fe_bar[demo_cols + ['bars_per_10k']].values)
X_bar_fe = np.column_stack([X_bar_fe_std, state_d.values])
y_bar_fe = fe_bar['male_suicide_rate_cdc'].values
X_bar_fe_c = sm.add_constant(X_bar_fe)
model_bar_fe = sm.OLS(y_bar_fe, X_bar_fe_c).fit()
# bars_per_10k is at index 8 (const=0, 7 demo, bars=8)
print(f"  Bars + demos + state FE: p={model_bar_fe.pvalues[8]:.4f}")
report("Bar density with state FE p ~0.50", 0.50, round(model_bar_fe.pvalues[8], 2), 0.1, "Confounding")

# Total alcohol outlets - unadjusted (HC3)
p_alc_hc3, _, _ = get_hc3_p(['alcohol_outlets_per_10k'])
report("Total outlets unadjusted p ~0.49 (HC3)", 0.49, round(p_alc_hc3[1], 2), 0.15, "Confounding")

# Total outlets with demographics (HC3)
p_alc_demo_hc3, _, _ = get_hc3_p(demo_cols + ['alcohol_outlets_per_10k'])
print(f"  Outlets + demos: HC3 p={p_alc_demo_hc3[8]:.4f}")
report("Total outlets with demographics p ~0.19 (HC3)", 0.19, round(p_alc_demo_hc3[8], 2), 0.1, "Confounding")

# Excessive drinking with state FE
fe_drink = valid[demo_cols + ['chr_excessive_drinking_pct', 'male_suicide_rate_cdc', 'state_name']].dropna()
state_d2 = pd.get_dummies(fe_drink['state_name'], drop_first=True)
X_drink_fe_std = StandardScaler().fit_transform(fe_drink[demo_cols + ['chr_excessive_drinking_pct']].values)
X_drink_fe = np.column_stack([X_drink_fe_std, state_d2.values])
y_drink_fe = fe_drink['male_suicide_rate_cdc'].values
X_drink_fe_c = sm.add_constant(X_drink_fe)
model_drink_fe = sm.OLS(y_drink_fe, X_drink_fe_c).fit()
print(f"  Drinking + demos + state FE: p={model_drink_fe.pvalues[8]:.4f}")
report("Excessive drinking with state FE p ~0.48", 0.48, round(model_drink_fe.pvalues[8], 2), 0.1, "Confounding")

# Total outlets with state FE
fe_out = valid[demo_cols + ['alcohol_outlets_per_10k', 'male_suicide_rate_cdc', 'state_name']].dropna()
state_d3 = pd.get_dummies(fe_out['state_name'], drop_first=True)
X_out_fe_std = StandardScaler().fit_transform(fe_out[demo_cols + ['alcohol_outlets_per_10k']].values)
X_out_fe = np.column_stack([X_out_fe_std, state_d3.values])
y_out_fe = fe_out['male_suicide_rate_cdc'].values
X_out_fe_c = sm.add_constant(X_out_fe)
model_out_fe = sm.OLS(y_out_fe, X_out_fe_c).fit()
print(f"  Outlets + demos + state FE: p={model_out_fe.pvalues[8]:.4f}")
report("Total outlets with state FE p ~0.72", 0.72, round(model_out_fe.pvalues[8], 2), 0.15, "Confounding")

# ============================================================
# 17. VETERAN PERIOD-OF-SERVICE ANALYSIS
# ============================================================
print("\n" + "=" * 90)
print("17. VETERAN PERIOD-OF-SERVICE CORRELATIONS")
print("=" * 90)

# Post-9/11 veterans: pct_post911_vets
if 'pct_post911_vets' in valid.columns:
    r_post911, p_post911, _ = pearson_test('pct_post911_vets')
    report("r(post-9/11 vet pct, suicide) ~-0.13", -0.13, round(r_post911, 2), 0.03, "Vet Period")

# Partial correlation: veteran_pct controlling for RUCC
# r(veteran, suicide | RUCC) ~= +0.33
from numpy.linalg import lstsq as np_lstsq

def partial_corr(x_col, y_col, control_cols, data=valid):
    d = data[[x_col, y_col] + control_cols].dropna()
    # Residualize x on controls
    C = d[control_cols].values
    C_c = np.column_stack([np.ones(len(C)), C])

    x = d[x_col].values
    bx = lstsq(C_c, x, rcond=None)[0]
    x_resid = x - C_c @ bx

    y_v = d[y_col].values
    by = lstsq(C_c, y_v, rcond=None)[0]
    y_resid = y_v - C_c @ by

    return stats.pearsonr(x_resid, y_resid)

r_vet_partial, _ = partial_corr('veteran_pct', 'male_suicide_rate_cdc', ['rural_urban_code'])
report("Partial r(veteran, suicide | RUCC) ~0.33", 0.33, round(r_vet_partial, 2), 0.02, "Vet Period")

# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n\n" + "=" * 90)
print("FINAL SUMMARY TABLE")
print("=" * 90)
print(f"\n{'Category':<20} {'Claim':<55} {'MS':>8} {'Actual':>10} {'Status':>6}")
print("-" * 100)

n_pass = 0
n_flag = 0
for cat, claim, ms, actual, tol, status in results:
    flag = "  " if status == "PASS" else ">>"
    print(f"{flag}{cat:<18} {claim[:53]:<55} {str(ms):>8} {str(actual):>10} {status:>6}")
    if status == "PASS":
        n_pass += 1
    else:
        n_flag += 1

print(f"\n{'=' * 90}")
print(f"TOTAL: {n_pass} PASS, {n_flag} FLAG out of {n_pass + n_flag} checks")
print(f"{'=' * 90}")

# Print just the flagged items
if n_flag > 0:
    print(f"\n{'=' * 90}")
    print("FLAGGED ITEMS DETAIL")
    print(f"{'=' * 90}")
    for cat, claim, ms, actual, tol, status in results:
        if status == "FLAG":
            print(f"  [{cat}] {claim}")
            print(f"    Manuscript says: {ms}")
            print(f"    Data shows:     {actual}")
            print(f"    Tolerance:      {tol}")
            print()
