#!/usr/bin/env python3
"""
THE REAL TEST: County-level suicide rates + county-level demographics + elevation
===================================================================================
ALL REAL DATA. This is the analysis that matters.
"""
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import pearsonr, ttest_ind
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
OUTPUT_DIR = Path(__file__).resolve().parent

# Load master
df = pd.read_csv(OUTPUT_DIR / 'real_county_dataset.csv', dtype={'FIPS': str, 'state_fips': str})
df['FIPS'] = df['FIPS'].str.zfill(5)

# Use county-level suicide rate from CHR
df['suicide_rate'] = df['chr_suicide_rate']

# Bars per 10K
df['bars_per_10k'] = (df['bar_establishments'] / (df['total_population'] / 10000)).replace([np.inf, -np.inf], np.nan)

# Analysis subset: counties with county-level suicide data
adf = df.dropna(subset=['suicide_rate']).copy()
print(f"Analysis dataset: {len(adf)} counties with REAL county-level suicide rates")
print(f"Suicide rate: mean={adf['suicide_rate'].mean():.1f}, sd={adf['suicide_rate'].std():.1f}, "
      f"range={adf['suicide_rate'].min():.1f}-{adf['suicide_rate'].max():.1f}")

# ============================================================
# 1. BIVARIATE CORRELATIONS
# ============================================================
print("\n" + "=" * 70)
print("BIVARIATE CORRELATIONS WITH COUNTY SUICIDE RATE (n=%d)" % len(adf))
print("=" * 70)

predictors = [
    ('elevation_feet', 'Elevation (feet)'),
    ('male_female_ratio', 'Male:Female Ratio'),
    ('working_age_sex_ratio', 'Working-Age Sex Ratio (25-64)'),
    ('veteran_pct', 'Veteran %'),
    ('rural_urban_code', 'Rural-Urban Code (RUCC)'),
    ('extraction_employment_pct', 'Extraction Employment %'),
    ('male_divorced_separated_pct', 'Male Divorced/Separated %'),
    ('pop_density_sqmi', 'Population Density'),
    ('bars_per_10k', 'Bars per 10K Pop'),
    ('unemployment_rate', 'Unemployment Rate'),
    ('poverty_rate', 'Poverty Rate'),
    ('median_household_income', 'Median Household Income'),
    ('pct_bachelors_or_higher', '% Bachelors Degree+'),
    ('pct_no_internet', '% No Internet Access'),
    ('housing_vacancy_rate', 'Housing Vacancy Rate'),
    ('homeownership_rate', 'Homeownership Rate'),
    ('pct_moved_diff_state', '% Moved from Different State'),
    ('alcohol_strictness', 'Alcohol Strictness (state)'),
    ('chr_excessive_drinking_pct', 'Excessive Drinking %'),
    ('chr_firearm_fatality_rate', 'Firearm Fatality Rate'),
    ('chr_overdose_rate', 'Drug Overdose Rate'),
    ('chr_pop_per_mh_provider', 'Pop per MH Provider'),
    ('chr_freq_mental_distress', 'Frequent Mental Distress'),
    ('chr_social_associations', 'Social Associations'),
    ('chr_income_inequality', 'Income Inequality'),
    ('chr_pct_rural', '% Rural (CHR)'),
    ('chr_smoking_pct', 'Adult Smoking %'),
    ('chr_physical_inactivity', 'Physical Inactivity %'),
    ('chr_food_insecurity', 'Food Insecurity %'),
]

print(f"\n{'Variable':<40} {'r':>8} {'p':>12} {'n':>6} {'Sig':>5}")
print("-" * 75)

corr_results = []
for var, label in predictors:
    if var in adf.columns:
        valid = adf[[var, 'suicide_rate']].dropna()
        if len(valid) > 30:
            r, p = pearsonr(valid[var], valid['suicide_rate'])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            print(f"{label:<40} {r:>+8.4f} {p:>12.6f} {len(valid):>6} {sig:>5}")
            corr_results.append({'variable': var, 'label': label, 'r': r, 'p': p, 'n': len(valid), 'sig': sig})

# ============================================================
# 2. ALTITUDE HIERARCHICAL REGRESSION (THE BIG TEST)
# ============================================================
print("\n" + "=" * 70)
print("ALTITUDE HIERARCHICAL REGRESSION — REAL COUNTY SUICIDE DATA")
print("=" * 70)

elev_df = adf.dropna(subset=['elevation_feet']).copy()
print(f"\nCounties with elevation + suicide: n = {len(elev_df)}")

models = [
    ('Model 1: Altitude only', ['elevation_feet']),
    ('Model 2: + Veteran %', ['elevation_feet', 'veteran_pct']),
    ('Model 3: + Rurality', ['elevation_feet', 'veteran_pct', 'rural_urban_code']),
    ('Model 4: + Gender ratio', ['elevation_feet', 'veteran_pct', 'rural_urban_code', 'male_female_ratio']),
    ('Model 5: + Extraction', ['elevation_feet', 'veteran_pct', 'rural_urban_code', 'male_female_ratio', 'extraction_employment_pct']),
    ('Model 6: + Poverty + Education', ['elevation_feet', 'veteran_pct', 'rural_urban_code', 'male_female_ratio',
                                          'extraction_employment_pct', 'poverty_rate', 'pct_bachelors_or_higher']),
    ('Model 7: Full', ['elevation_feet', 'veteran_pct', 'rural_urban_code', 'male_female_ratio',
                        'extraction_employment_pct', 'poverty_rate', 'pct_bachelors_or_higher',
                        'male_divorced_separated_pct', 'unemployment_rate', 'pct_no_internet']),
]

alt_trajectory = []
for name, preds in models:
    cols = ['suicide_rate'] + preds
    mdf = elev_df[cols].dropna()
    if len(mdf) < 50:
        continue
    
    y = mdf['suicide_rate']
    X = mdf[preds]
    scaler = StandardScaler()
    X_std = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    X_std = sm.add_constant(X_std)
    
    m = sm.OLS(y, X_std).fit()
    
    ab = m.params['elevation_feet']
    ap = m.pvalues['elevation_feet']
    asig = '***' if ap < 0.001 else '**' if ap < 0.01 else '*' if ap < 0.05 else 'n.s.'
    
    print(f"\n{name} (n={len(mdf)}, R²={m.rsquared:.4f})")
    for pred in preds:
        b = m.params[pred]
        p = m.pvalues[pred]
        s = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        tag = " ← ALTITUDE" if pred == 'elevation_feet' else ""
        print(f"  {pred:<35} β={b:>+8.4f}  p={p:.6f} {s}{tag}")
    
    alt_trajectory.append({'model': name, 'n': len(mdf), 'alt_beta': ab, 'alt_p': ap,
                           'alt_sig': asig, 'r_squared': m.rsquared})

print("\n" + "=" * 70)
print("ALTITUDE β TRAJECTORY:")
print("=" * 70)
for a in alt_trajectory:
    print(f"  {a['model']:<45} β={a['alt_beta']:>+8.4f}  p={a['alt_p']:.6f} {a['alt_sig']}  R²={a['r_squared']:.4f}")

if alt_trajectory:
    initial = alt_trajectory[0]['alt_beta']
    final = alt_trajectory[-1]['alt_beta']
    pct = ((final - initial) / abs(initial)) * 100
    print(f"\n  β change: {initial:+.4f} → {final:+.4f} ({pct:+.1f}%)")
    if alt_trajectory[-1]['alt_p'] > 0.05:
        print("  ★★★ ALTITUDE LOSES SIGNIFICANCE WITH REAL COUNTY DATA ★★★")
    elif alt_trajectory[-1]['alt_p'] > 0.01:
        print("  ★ Altitude weakened but still marginally significant")
    else:
        print("  Altitude remains significant, but β reduced substantially")

# ============================================================
# 3. FULL MODEL WITHOUT ALTITUDE
# ============================================================
print("\n" + "=" * 70)
print("FULL MODEL WITHOUT ALTITUDE (all counties)")
print("=" * 70)

full_preds = ['veteran_pct', 'rural_urban_code', 'male_female_ratio', 'extraction_employment_pct',
              'poverty_rate', 'pct_bachelors_or_higher', 'male_divorced_separated_pct',
              'unemployment_rate', 'pct_no_internet']
cols = ['suicide_rate'] + full_preds
mdf = adf[cols].dropna()

y = mdf['suicide_rate']
X = mdf[full_preds]
scaler = StandardScaler()
X_std = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
X_std = sm.add_constant(X_std)
m = sm.OLS(y, X_std).fit()

print(f"n = {len(mdf)}, R² = {m.rsquared:.4f}, Adj R² = {m.rsquared_adj:.4f}")
print(f"\n{'Predictor':<35} {'β (std)':>10} {'p':>12} {'Sig':>5}")
print("-" * 65)
for pred in full_preds:
    b = m.params[pred]
    p = m.pvalues[pred]
    s = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
    print(f"{pred:<35} {b:>+10.4f} {p:>12.6f} {s}")

# ============================================================
# 4. PIPELINE ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("PIPELINE ANALYSIS — REAL COUNTY SUICIDE RATES")
print("=" * 70)

pipeline = adf[(adf['extraction_employment_pct'] > 5) & (adf['veteran_pct'] > adf['veteran_pct'].quantile(0.75))]
control = adf[(adf['extraction_employment_pct'] == 0) & (adf['veteran_pct'] < adf['veteran_pct'].quantile(0.25))]

print(f"Pipeline: n={len(pipeline)}, Control: n={len(control)}")
if len(pipeline) >= 5 and len(control) >= 5:
    for var, label in [('suicide_rate', 'County Suicide Rate'), ('male_female_ratio', 'M:F Ratio'),
                        ('veteran_pct', 'Veteran %'), ('chr_pop_per_mh_provider', 'Pop per MH Provider')]:
        if var in pipeline.columns:
            pm = pipeline[var].mean()
            cm = control[var].mean()
            print(f"  {label:<30} Pipeline: {pm:>8.2f}  Control: {cm:>8.2f}  Diff: {pm-cm:>+8.2f}")
    
    t, p = ttest_ind(pipeline['suicide_rate'].dropna(), control['suicide_rate'].dropna())
    d = ((pipeline['suicide_rate'].mean() - control['suicide_rate'].mean()) /
         np.sqrt((pipeline['suicide_rate'].std()**2 + control['suicide_rate'].std()**2) / 2))
    pct = ((pipeline['suicide_rate'].mean() - control['suicide_rate'].mean()) / control['suicide_rate'].mean()) * 100
    print(f"\n  t={t:.3f}, p={p:.6f}, Cohen's d={d:.3f}, difference: {pct:+.1f}%")

# ============================================================
# 5. ALCOHOL / BAR TEST
# ============================================================
print("\n" + "=" * 70)
print("BAR DENSITY + ALCOHOL POLICY — REAL COUNTY DATA")
print("=" * 70)

bar_df = adf.dropna(subset=['bars_per_10k']).copy()
y = bar_df['suicide_rate']

# Model 1: bars only
X1 = sm.add_constant(bar_df[['bars_per_10k']])
m1 = sm.OLS(y, X1).fit()
print(f"Bars only: β={m1.params['bars_per_10k']:.4f}, p={m1.pvalues['bars_per_10k']:.6f}, R²={m1.rsquared:.4f}")

# Model 2: bars + full controls
ctrl = ['bars_per_10k', 'veteran_pct', 'rural_urban_code', 'male_female_ratio', 'extraction_employment_pct']
m2df = bar_df[['suicide_rate'] + ctrl].dropna()
X2 = sm.add_constant(m2df[ctrl])
m2 = sm.OLS(m2df['suicide_rate'], X2).fit()
print(f"\nBars + controls (n={len(m2df)}):")
for p in ctrl:
    sig = '***' if m2.pvalues[p] < 0.001 else '**' if m2.pvalues[p] < 0.01 else '*' if m2.pvalues[p] < 0.05 else 'n.s.'
    print(f"  {p:<35} β={m2.params[p]:>+8.4f}  p={m2.pvalues[p]:.6f} {sig}")

# Save correlations
pd.DataFrame(corr_results).to_csv(OUTPUT_DIR / 'REAL_county_correlations.csv', index=False)
print("\n→ Saved REAL_county_correlations.csv")
print("\n" + "=" * 70)
print("ALL RESULTS FROM REAL COUNTY-LEVEL DATA — NO SIMULATIONS")
print("=" * 70)
