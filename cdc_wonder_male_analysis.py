#!/usr/bin/env python3
"""
FINAL ANALYSIS: CDC WONDER County-Level MALE Suicide Rates
===========================================================
Parses real CDC WONDER data (ICD X60-X84, males, 2018-2022),
merges with master county dataset, and runs ALL key analyses
with male-specific county-level suicide rates.

Uses only: pandas, numpy, scipy (no statsmodels/sklearn needed).
"""
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu
import re

warnings.filterwarnings('ignore')
OUTPUT_DIR = Path(__file__).resolve().parent


def ols_regression(y, X_df):
    """Manual OLS with standardized coefficients, p-values, R²."""
    from scipy.stats import t as t_dist

    # Standardize predictors
    X_std = (X_df - X_df.mean()) / X_df.std()
    X = np.column_stack([np.ones(len(X_std)), X_std.values])
    y_arr = y.values

    # OLS: β = (X'X)^-1 X'y
    try:
        beta = np.linalg.lstsq(X, y_arr, rcond=None)[0]
    except:
        return None

    y_hat = X @ beta
    residuals = y_arr - y_hat
    n, k = X.shape

    if n <= k:
        return None

    # R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_arr - y_arr.mean())**2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - k) if n > k else r_sq

    # Standard errors
    mse = ss_res / (n - k)
    try:
        var_beta = mse * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(var_beta))
    except:
        return None

    # t-values and p-values
    t_vals = beta / se
    p_vals = 2 * t_dist.sf(np.abs(t_vals), df=n - k)

    # Package results
    names = ['const'] + list(X_df.columns)
    results = {}
    for i, name in enumerate(names):
        results[name] = {'beta': beta[i], 'se': se[i], 't': t_vals[i], 'p': p_vals[i]}

    return {'params': results, 'r_squared': r_sq, 'adj_r_squared': adj_r_sq, 'n': n, 'k': k}


def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'n.s.'


# ============================================================
# STEP 1: PARSE CDC WONDER TSV
# ============================================================
print("=" * 80)
print("STEP 1: PARSING CDC WONDER COUNTY-LEVEL MALE SUICIDE DATA")
print("=" * 80)

# Try full dataset first, fall back to partial
full_path = OUTPUT_DIR / 'cdc_wonder_male_suicide_ALL_counties_2018_2022.tsv'
partial_path = OUTPUT_DIR / 'cdc_wonder_male_suicide_county_2018_2022.tsv'
if full_path.exists():
    tsv = pd.read_csv(full_path, sep='\t', dtype={'FIPS': str})
    print(f"Using FULL dataset: {len(tsv)} rows")
    # Already clean format: FIPS, County, Deaths, Population, Crude_Rate
    tsv['FIPS'] = tsv['FIPS'].astype(str).str.zfill(5)
    # Extract FIPS from County column if FIPS column is empty/missing
    mask = tsv['FIPS'].isna() | (tsv['FIPS'] == '00nan') | (tsv['FIPS'].str.len() < 5)
    if mask.any():
        extracted = tsv.loc[mask, 'County'].str.extract(r'\((\d{5})\)')
        tsv.loc[mask, 'FIPS'] = extracted[0]
    tsv['male_suicide_rate_cdc'] = pd.to_numeric(tsv['Crude_Rate'], errors='coerce')
    tsv['male_suicide_deaths'] = pd.to_numeric(tsv['Deaths'].astype(str).str.replace(',', ''), errors='coerce')
    tsv['male_suicide_pop'] = pd.to_numeric(tsv['Population'].astype(str).str.replace(',', ''), errors='coerce')
else:
    tsv = pd.read_csv(partial_path, sep='\t')
    print(f"Using partial dataset: {len(tsv)} rows")
    county_col = tsv.columns[0]
    rate_col = tsv.columns[3]
    tsv['FIPS'] = tsv[county_col].str.extract(r'\((\d{5})\)')
    def parse_rate(val):
        if pd.isna(val) or 'Suppressed' in str(val) or 'Unreliable' in str(val):
            return np.nan
        match = re.match(r'([\d.]+)', str(val))
        return float(match.group(1)) if match else np.nan
    tsv['male_suicide_rate_cdc'] = tsv[rate_col].apply(parse_rate)
    tsv['male_suicide_deaths'] = pd.to_numeric(tsv['Deaths'], errors='coerce')
    tsv['male_suicide_pop'] = pd.to_numeric(tsv['Population'], errors='coerce')

# Remove total row
tsv = tsv[~tsv['County'].astype(str).str.contains('Total', na=False)]
cdc = tsv[['FIPS', 'male_suicide_rate_cdc', 'male_suicide_deaths', 'male_suicide_pop']].dropna(subset=['FIPS'])
valid_rates = cdc['male_suicide_rate_cdc'].dropna()
print(f"\nParsed: {len(cdc)} counties total")
print(f"  With valid rate: {len(valid_rates)}")
print(f"  Suppressed: {cdc['male_suicide_rate_cdc'].isna().sum()}")
print(f"  Rate range: {valid_rates.min():.1f} - {valid_rates.max():.1f}")
print(f"  Rate mean: {valid_rates.mean():.1f}, median: {valid_rates.median():.1f}")

# ============================================================
# STEP 2: MERGE WITH MASTER DATASET
# ============================================================
print("\n" + "=" * 80)
print("STEP 2: MERGING WITH MASTER COUNTY DATASET")
print("=" * 80)

master = pd.read_csv(OUTPUT_DIR / 'real_county_dataset.csv', dtype={'FIPS': str, 'state_fips': str})
master['FIPS'] = master['FIPS'].str.zfill(5)
cdc['FIPS'] = cdc['FIPS'].str.zfill(5)

for col in ['male_suicide_rate_cdc', 'male_suicide_deaths', 'male_suicide_pop']:
    if col in master.columns:
        master = master.drop(columns=[col])

master = master.merge(cdc[['FIPS', 'male_suicide_rate_cdc', 'male_suicide_deaths', 'male_suicide_pop']],
                       on='FIPS', how='left')

matched = master['male_suicide_rate_cdc'].notna().sum()
print(f"Master dataset: {len(master)} counties")
print(f"Matched CDC WONDER male suicide rate: {matched} counties")
print(f"Coverage: {matched/len(master)*100:.1f}%")

master.to_csv(OUTPUT_DIR / 'real_county_dataset.csv', index=False)
print("Saved updated master dataset.")

# ============================================================
# ANALYSIS SUBSET
# ============================================================
adf = master.dropna(subset=['male_suicide_rate_cdc']).copy()
print(f"\nAnalysis dataset: {len(adf)} counties with REAL county-level male suicide rates")
print(f"  Mean: {adf['male_suicide_rate_cdc'].mean():.1f}/100K, SD: {adf['male_suicide_rate_cdc'].std():.1f}")
print(f"  Range: {adf['male_suicide_rate_cdc'].min():.1f} - {adf['male_suicide_rate_cdc'].max():.1f}")

# ============================================================
# STEP 3: BIVARIATE CORRELATIONS
# ============================================================
print("\n" + "=" * 80)
print(f"BIVARIATE CORRELATIONS WITH CDC WONDER MALE SUICIDE RATE (n={len(adf)})")
print("=" * 80)

predictors = [
    ('elevation_feet', 'Elevation (feet)'),
    ('male_female_ratio', 'Male:Female Ratio'),
    ('working_age_sex_ratio', 'Working-Age Sex Ratio'),
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
    ('alcohol_outlets_per_10k', 'All Alcohol Outlets per 10K'),
    ('liquor_per_10k', 'Liquor Stores per 10K'),
]

print(f"\n{'Variable':<40} {'r':>8} {'p':>12} {'n':>6} {'Sig':>5}")
print("-" * 75)

corr_results = []
for var, label in predictors:
    if var in adf.columns:
        valid = adf[[var, 'male_suicide_rate_cdc']].dropna()
        if len(valid) > 30:
            r, p = pearsonr(valid[var], valid['male_suicide_rate_cdc'])
            s = sig_stars(p)
            print(f"{label:<40} {r:>+8.4f} {p:>12.6f} {len(valid):>6} {s:>5}")
            corr_results.append({'variable': var, 'label': label, 'r': r, 'p': p, 'n': len(valid), 'sig': s})

pd.DataFrame(corr_results).to_csv(OUTPUT_DIR / 'REAL_male_suicide_correlations.csv', index=False)

# ============================================================
# STEP 4: ALTITUDE HIERARCHICAL REGRESSION
# ============================================================
print("\n" + "=" * 80)
print("ALTITUDE HIERARCHICAL REGRESSION — REAL COUNTY MALE SUICIDE RATES")
print("THIS IS THE KEY TEST")
print("=" * 80)

elev_df = adf.dropna(subset=['elevation_feet']).copy()
print(f"\nCounties with elevation + male suicide rate: n = {len(elev_df)}")

alt_trajectory = []
if len(elev_df) >= 50:
    model_specs = [
        ('Model 1: Altitude only', ['elevation_feet']),
        ('Model 2: + Veteran %', ['elevation_feet', 'veteran_pct']),
        ('Model 3: + Rurality', ['elevation_feet', 'veteran_pct', 'rural_urban_code']),
        ('Model 4: + Gender ratio', ['elevation_feet', 'veteran_pct', 'rural_urban_code', 'male_female_ratio']),
        ('Model 5: + Extraction', ['elevation_feet', 'veteran_pct', 'rural_urban_code', 'male_female_ratio', 'extraction_employment_pct']),
        ('Model 6: + Poverty + Education', ['elevation_feet', 'veteran_pct', 'rural_urban_code', 'male_female_ratio',
                                              'extraction_employment_pct', 'poverty_rate', 'pct_bachelors_or_higher']),
        ('Model 7: + Divorce + Unemployment', ['elevation_feet', 'veteran_pct', 'rural_urban_code', 'male_female_ratio',
                                                 'extraction_employment_pct', 'poverty_rate', 'pct_bachelors_or_higher',
                                                 'male_divorced_separated_pct', 'unemployment_rate']),
        ('Model 8: Full', ['elevation_feet', 'veteran_pct', 'rural_urban_code', 'male_female_ratio',
                            'extraction_employment_pct', 'poverty_rate', 'pct_bachelors_or_higher',
                            'male_divorced_separated_pct', 'unemployment_rate', 'pct_no_internet']),
    ]

    for name, preds in model_specs:
        actual_preds = [p for p in preds if p in elev_df.columns]
        cols = ['male_suicide_rate_cdc'] + actual_preds
        mdf = elev_df[cols].dropna()
        if len(mdf) < 50:
            continue

        result = ols_regression(mdf['male_suicide_rate_cdc'], mdf[actual_preds])
        if result is None:
            continue

        ab = result['params']['elevation_feet']['beta']
        ap = result['params']['elevation_feet']['p']

        print(f"\n{name} (n={result['n']}, R²={result['r_squared']:.4f}, Adj R²={result['adj_r_squared']:.4f})")
        for pred in actual_preds:
            b = result['params'][pred]['beta']
            p = result['params'][pred]['p']
            tag = " ← ALTITUDE" if pred == 'elevation_feet' else ""
            print(f"  {pred:<35} β={b:>+8.4f}  p={p:.6f} {sig_stars(p)}{tag}")

        alt_trajectory.append({'model': name, 'n': result['n'], 'alt_beta': ab, 'alt_p': ap,
                               'alt_sig': sig_stars(ap), 'r_squared': result['r_squared'],
                               'adj_r_squared': result['adj_r_squared']})

    print("\n" + "-" * 80)
    print("ALTITUDE β TRAJECTORY:")
    print("-" * 80)
    for a in alt_trajectory:
        print(f"  {a['model']:<50} β={a['alt_beta']:>+8.4f}  p={a['alt_p']:.6f} {a['alt_sig']}  R²={a['r_squared']:.4f}")

    if alt_trajectory:
        initial = alt_trajectory[0]['alt_beta']
        final = alt_trajectory[-1]['alt_beta']
        pct = ((final - initial) / abs(initial)) * 100
        print(f"\n  β change: {initial:+.4f} → {final:+.4f} ({pct:+.1f}%)")
        if alt_trajectory[-1]['alt_p'] > 0.05:
            print("  ★★★ ALTITUDE LOSES SIGNIFICANCE WITH REAL COUNTY MALE SUICIDE DATA ★★★")
        elif alt_trajectory[-1]['alt_p'] > 0.01:
            print("  ★★ Altitude weakened to marginal significance")
        else:
            print(f"  Altitude remains significant at p = {alt_trajectory[-1]['alt_p']:.6f}")

    pd.DataFrame(alt_trajectory).to_csv(OUTPUT_DIR / 'REAL_male_altitude_trajectory.csv', index=False)
else:
    print(f"  Only {len(elev_df)} counties — checking if that's enough...")
    if len(elev_df) >= 20:
        # Still run basic altitude correlation
        valid = elev_df[['elevation_feet', 'male_suicide_rate_cdc']].dropna()
        r, p = pearsonr(valid['elevation_feet'], valid['male_suicide_rate_cdc'])
        print(f"  Altitude vs male suicide: r = {r:+.4f}, p = {p:.6f} (n={len(valid)})")

# ============================================================
# STEP 5: FULL MODEL WITHOUT ALTITUDE
# ============================================================
print("\n" + "=" * 80)
print("FULL MODEL WITHOUT ALTITUDE — ALL COUNTIES WITH MALE SUICIDE RATE")
print("=" * 80)

full_preds = ['veteran_pct', 'rural_urban_code', 'male_female_ratio', 'extraction_employment_pct',
              'poverty_rate', 'pct_bachelors_or_higher', 'male_divorced_separated_pct',
              'unemployment_rate', 'pct_no_internet']
cols = ['male_suicide_rate_cdc'] + full_preds
mdf = adf[cols].dropna()

if len(mdf) >= 50:
    result = ols_regression(mdf['male_suicide_rate_cdc'], mdf[full_preds])
    if result:
        print(f"n = {result['n']}, R² = {result['r_squared']:.4f}, Adj R² = {result['adj_r_squared']:.4f}")
        print(f"\n{'Predictor':<35} {'β (std)':>10} {'p':>12} {'Sig':>5}")
        print("-" * 65)
        for pred in full_preds:
            b = result['params'][pred]['beta']
            p = result['params'][pred]['p']
            print(f"{pred:<35} {b:>+10.4f} {p:>12.6f} {sig_stars(p)}")
else:
    print(f"  Only {len(mdf)} counties with all variables — running reduced model")
    reduced = ['veteran_pct', 'rural_urban_code', 'male_female_ratio', 'poverty_rate']
    cols2 = ['male_suicide_rate_cdc'] + reduced
    mdf2 = adf[cols2].dropna()
    if len(mdf2) >= 20:
        result = ols_regression(mdf2['male_suicide_rate_cdc'], mdf2[reduced])
        if result:
            print(f"  Reduced model (n={result['n']}, R²={result['r_squared']:.4f}):")
            for pred in reduced:
                b = result['params'][pred]['beta']
                p = result['params'][pred]['p']
                print(f"    {pred:<35} β={b:>+8.4f}  p={p:.6f} {sig_stars(p)}")

# ============================================================
# STEP 6: PIPELINE ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("PIPELINE ANALYSIS — REAL COUNTY MALE SUICIDE RATES")
print("=" * 80)

pipeline = adf[(adf['extraction_employment_pct'] > 5) & (adf['veteran_pct'] > adf['veteran_pct'].quantile(0.75))]
control = adf[(adf['extraction_employment_pct'] == 0) & (adf['veteran_pct'] < adf['veteran_pct'].quantile(0.25))]

print(f"Pipeline: n={len(pipeline)}, Control: n={len(control)}")

if len(pipeline) >= 3 and len(control) >= 3:
    for var, label in [('male_suicide_rate_cdc', 'Male Suicide Rate (CDC)'),
                        ('chr_suicide_rate', 'Total Suicide Rate (CHR)'),
                        ('male_female_ratio', 'M:F Ratio'),
                        ('veteran_pct', 'Veteran %'),
                        ('chr_overdose_rate', 'Overdose Rate'),
                        ('chr_pop_per_mh_provider', 'Pop per MH Provider')]:
        if var in pipeline.columns:
            pv = pipeline[var].dropna()
            cv = control[var].dropna()
            if len(pv) >= 2 and len(cv) >= 2:
                print(f"  {label:<30} Pipeline: {pv.mean():>8.1f}  Control: {cv.mean():>8.1f}  Diff: {pv.mean()-cv.mean():>+8.1f}")

    p_sr = pipeline['male_suicide_rate_cdc'].dropna()
    c_sr = control['male_suicide_rate_cdc'].dropna()
    if len(p_sr) >= 2 and len(c_sr) >= 2:
        t, p = ttest_ind(p_sr, c_sr)
        pooled_sd = np.sqrt((p_sr.std()**2 + c_sr.std()**2) / 2)
        d = (p_sr.mean() - c_sr.mean()) / pooled_sd if pooled_sd > 0 else 0
        pct = ((p_sr.mean() - c_sr.mean()) / c_sr.mean()) * 100
        print(f"\n  t={t:.3f}, p={p:.6f}, Cohen's d={d:.3f}, difference: {pct:+.1f}%")
        if p < 0.001:
            print("  ★★★ PIPELINE EFFECT CONFIRMED WITH REAL MALE SUICIDE RATES ★★★")
        elif p < 0.05:
            print("  ★ Pipeline effect significant at p < 0.05")
else:
    print("  Not enough counties in pipeline/control for this subset")
    # Try relaxed criteria
    pipeline2 = adf[(adf['extraction_employment_pct'] > 2) & (adf['veteran_pct'] > adf['veteran_pct'].median())]
    control2 = adf[(adf['extraction_employment_pct'] == 0) & (adf['veteran_pct'] < adf['veteran_pct'].median())]
    print(f"  Relaxed criteria — Pipeline: n={len(pipeline2)}, Control: n={len(control2)}")
    if len(pipeline2) >= 3 and len(control2) >= 3:
        p_sr = pipeline2['male_suicide_rate_cdc'].dropna()
        c_sr = control2['male_suicide_rate_cdc'].dropna()
        if len(p_sr) >= 2 and len(c_sr) >= 2:
            t, p = ttest_ind(p_sr, c_sr)
            pct = ((p_sr.mean() - c_sr.mean()) / c_sr.mean()) * 100
            print(f"  Pipeline: {p_sr.mean():.1f}/100K vs Control: {c_sr.mean():.1f}/100K ({pct:+.1f}%)")
            print(f"  t={t:.3f}, p={p:.6f}")

# ============================================================
# STEP 7: BAR DENSITY / ALCOHOL CONFOUNDING
# ============================================================
print("\n" + "=" * 80)
print("BAR DENSITY + ALCOHOL — CONFOUNDING TEST WITH MALE SUICIDE")
print("=" * 80)

alcohol_vars = [
    ('bars_per_10k', 'Bars per 10K'),
    ('alcohol_outlets_per_10k', 'All Outlets per 10K'),
    ('liquor_per_10k', 'Liquor Stores per 10K'),
    ('chr_excessive_drinking_pct', 'Excessive Drinking %'),
]

controls = ['veteran_pct', 'rural_urban_code', 'male_female_ratio', 'extraction_employment_pct', 'poverty_rate']

for avar, alabel in alcohol_vars:
    if avar not in adf.columns:
        continue

    # Model 1: alcohol var only
    m1_df = adf[['male_suicide_rate_cdc', avar]].dropna()
    if len(m1_df) < 30:
        continue

    r1 = ols_regression(m1_df['male_suicide_rate_cdc'], m1_df[[avar]])
    if r1 is None:
        continue

    # Model 2: + controls
    m2_cols = [c for c in ['male_suicide_rate_cdc', avar] + controls if c in adf.columns]
    m2_df = adf[m2_cols].dropna()
    m2_preds = [c for c in [avar] + controls if c in m2_df.columns]
    r2 = ols_regression(m2_df['male_suicide_rate_cdc'], m2_df[m2_preds])
    if r2 is None:
        continue

    p1 = r1['params'][avar]['p']
    p2 = r2['params'][avar]['p']
    b1 = r1['params'][avar]['beta']
    b2 = r2['params'][avar]['beta']

    print(f"\n  {alabel}:")
    print(f"    Alone:     β={b1:+.4f}, p={p1:.6f} {sig_stars(p1)}, R²={r1['r_squared']:.4f}")
    print(f"    +Controls: β={b2:+.4f}, p={p2:.6f} {sig_stars(p2)}, R²={r2['r_squared']:.4f}")

    if p1 < 0.05 and p2 > 0.05:
        print(f"    → LOSES SIGNIFICANCE with demographic controls ✓")
    elif p2 > 0.05:
        print(f"    → Not significant even alone")
    elif b2 < 0:
        print(f"    → Stays significant but NEGATIVE direction (more access = LESS suicide)")
    else:
        print(f"    → Remains significant with controls")

# ============================================================
# STEP 8: DRY vs WET — MALE SUICIDE
# ============================================================
print("\n" + "=" * 80)
print("DRY vs WET COUNTY COMPARISON — MALE SUICIDE RATES")
print("=" * 80)

if 'alcohol_access' in adf.columns:
    groups = {}
    for access in ['dry', 'very_restricted', 'wet']:
        g = adf[adf['alcohol_access'] == access]
        rate = g['male_suicide_rate_cdc'].dropna()
        groups[access] = g
        print(f"  {access}: n={len(rate)}, mean={rate.mean():.1f}/100K" if len(rate) > 0 else f"  {access}: n=0")

    outcomes = [
        ('male_suicide_rate_cdc', 'Male Suicide Rate (CDC)'),
        ('chr_suicide_rate', 'Total Suicide Rate (CHR)'),
        ('chr_overdose_rate', 'Drug Overdose Rate'),
        ('chr_firearm_fatality_rate', 'Firearm Fatality Rate'),
        ('chr_injury_death_rate', 'Injury Death Rate'),
        ('chr_excessive_drinking_pct', 'Excessive Drinking %'),
        ('chr_premature_death_rate', 'Premature Death Rate'),
        ('chr_life_expectancy', 'Life Expectancy'),
        ('veteran_pct', 'Veteran %'),
        ('chr_pop_per_mh_provider', 'Pop per MH Provider'),
    ]

    print(f"\n  {'Outcome':<40} {'Dry':>10} {'Restricted':>10} {'Wet':>10} {'p(D vs W)':>10} {'Δ%':>8}")
    print(f"  {'-'*82}")

    dry_wet_results = []
    for var, label in outcomes:
        if var not in adf.columns:
            continue
        dry_vals = groups.get('dry', pd.DataFrame())[var].dropna() if 'dry' in groups else pd.Series(dtype=float)
        wet_vals = groups.get('wet', pd.DataFrame())[var].dropna() if 'wet' in groups else pd.Series(dtype=float)
        restr_vals = groups.get('very_restricted', pd.DataFrame())[var].dropna() if 'very_restricted' in groups else pd.Series(dtype=float)

        if len(dry_vals) >= 3 and len(wet_vals) >= 3:
            dm, rm, wm = dry_vals.mean(), restr_vals.mean() if len(restr_vals) > 0 else np.nan, wet_vals.mean()
            try:
                _, p = mannwhitneyu(dry_vals, wet_vals, alternative='two-sided')
            except:
                p = np.nan
            pct_diff = ((dm - wm) / wm) * 100 if wm != 0 else 0
            print(f"  {label:<40} {dm:>10.1f} {rm:>10.1f} {wm:>10.1f} {p:>10.4f} {pct_diff:>+7.0f}%")
            dry_wet_results.append({'variable': var, 'label': label, 'dry_mean': dm, 'restricted_mean': rm,
                                     'wet_mean': wm, 'pct_diff': pct_diff, 'p_value': p})

    pd.DataFrame(dry_wet_results).to_csv(OUTPUT_DIR / 'REAL_male_dry_wet_comparison.csv', index=False)

# ============================================================
# STEP 9: TOP/BOTTOM COUNTIES
# ============================================================
print("\n" + "=" * 80)
print("TOP 20 HIGHEST MALE SUICIDE RATE COUNTIES (CDC WONDER)")
print("=" * 80)

top = adf.nlargest(20, 'male_suicide_rate_cdc')
print(f"\n  {'County':<45} {'Rate':>8} {'Vet%':>6} {'M:F':>6} {'RUCC':>5} {'Access':>12}")
print(f"  {'-'*85}")
for _, r in top.iterrows():
    access = r.get('alcohol_access', 'N/A')
    print(f"  {str(r['NAME'])[:44]:<45} {r['male_suicide_rate_cdc']:>8.1f} {r['veteran_pct']:>6.1f} "
          f"{r['male_female_ratio']:>6.3f} {r['rural_urban_code']:>5.0f} {access:>12}")

print("\n\nBOTTOM 20 LOWEST MALE SUICIDE RATE COUNTIES")
print("-" * 80)
bottom = adf.nsmallest(20, 'male_suicide_rate_cdc')
for _, r in bottom.iterrows():
    access = r.get('alcohol_access', 'N/A')
    print(f"  {str(r['NAME'])[:44]:<45} {r['male_suicide_rate_cdc']:>8.1f} {r['veteran_pct']:>6.1f} "
          f"{r['male_female_ratio']:>6.3f} {r['rural_urban_code']:>5.0f} {access:>12}")

# ============================================================
# STEP 10: MALE vs TOTAL COMPARISON
# ============================================================
print("\n" + "=" * 80)
print("COMPARISON: CDC WONDER MALE RATE vs CHR TOTAL RATE")
print("=" * 80)

both = adf.dropna(subset=['male_suicide_rate_cdc', 'chr_suicide_rate'])
if len(both) > 10:
    r, p = pearsonr(both['male_suicide_rate_cdc'], both['chr_suicide_rate'])
    ratio = both['male_suicide_rate_cdc'].mean() / both['chr_suicide_rate'].mean()
    print(f"  Counties with both: {len(both)}")
    print(f"  Correlation: r = {r:.4f}, p = {p:.6f}")
    print(f"  Male mean: {both['male_suicide_rate_cdc'].mean():.1f} vs Total mean: {both['chr_suicide_rate'].mean():.1f}")
    print(f"  Male/Total ratio: {ratio:.2f}x (males ~{ratio:.1f}x the total rate)")

# ============================================================
# STEP 11: RURALITY GRADIENT
# ============================================================
print("\n" + "=" * 80)
print("RURALITY GRADIENT — MALE SUICIDE RATES")
print("=" * 80)

rucc_labels = {
    1: 'Metro ≥1M', 2: 'Metro 250K-1M', 3: 'Metro <250K',
    4: 'Nonmetro ≥20K adj', 5: 'Nonmetro ≥20K nonadj',
    6: 'Nonmetro 2.5-20K adj', 7: 'Nonmetro 2.5-20K nonadj',
    8: 'Rural adj', 9: 'Rural nonadj'
}

print(f"\n  {'RUCC':<5} {'Description':<30} {'N':>5} {'Mean Rate':>10} {'SD':>8}")
print(f"  {'-'*62}")
for code in sorted(adf['rural_urban_code'].dropna().unique()):
    subset = adf[adf['rural_urban_code'] == code]['male_suicide_rate_cdc'].dropna()
    if len(subset) > 0:
        label = rucc_labels.get(int(code), f'RUCC {int(code)}')
        print(f"  {int(code):<5} {label:<30} {len(subset):>5} {subset.mean():>10.1f} {subset.std():>8.1f}")

valid = adf[['rural_urban_code', 'male_suicide_rate_cdc']].dropna()
if len(valid) > 10:
    rho, p = spearmanr(valid['rural_urban_code'], valid['male_suicide_rate_cdc'])
    print(f"\n  Spearman ρ (rurality vs male suicide): {rho:+.4f}, p = {p:.6f}")

# ============================================================
# STEP 12: STATE-LEVEL AGGREGATION FROM COUNTY DATA
# ============================================================
print("\n" + "=" * 80)
print("STATE-LEVEL: WEIGHTED MALE SUICIDE RATES FROM COUNTY DATA")
print("=" * 80)

state_agg = adf.groupby('state_name').agg(
    male_suicide_weighted=('male_suicide_rate_cdc', lambda x: np.average(x, weights=adf.loc[x.index, 'male_suicide_pop'].fillna(1))),
    male_suicide_mean=('male_suicide_rate_cdc', 'mean'),
    n_counties=('male_suicide_rate_cdc', 'count'),
    veteran_pct_mean=('veteran_pct', 'mean'),
    extraction_mean=('extraction_employment_pct', 'mean'),
    mf_ratio_mean=('male_female_ratio', 'mean'),
).sort_values('male_suicide_weighted', ascending=False)

print(f"\n  {'State':<25} {'Weighted Rate':>14} {'N Counties':>11} {'Vet%':>6} {'Extr%':>7}")
print(f"  {'-'*66}")
for state, row in state_agg.head(15).iterrows():
    print(f"  {state:<25} {row['male_suicide_weighted']:>14.1f} {int(row['n_counties']):>11} "
          f"{row['veteran_pct_mean']:>6.1f} {row['extraction_mean']:>7.1f}")

print(f"\n  ... (bottom 5)")
for state, row in state_agg.tail(5).iterrows():
    print(f"  {state:<25} {row['male_suicide_weighted']:>14.1f} {int(row['n_counties']):>11} "
          f"{row['veteran_pct_mean']:>6.1f} {row['extraction_mean']:>7.1f}")

# State-level correlations with weighted male suicide
if len(state_agg) > 10:
    print(f"\n  State-level correlations (n={len(state_agg)} states with data):")
    for var, label in [('veteran_pct_mean', 'Veteran %'), ('extraction_mean', 'Extraction %'),
                        ('mf_ratio_mean', 'M:F Ratio')]:
        valid = state_agg[[var, 'male_suicide_weighted']].dropna()
        if len(valid) > 5:
            r, p = pearsonr(valid[var], valid['male_suicide_weighted'])
            print(f"    {label:<25} r = {r:+.4f}, p = {p:.6f} {sig_stars(p)}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("=" * 80)
print("FINAL SUMMARY — CDC WONDER COUNTY-LEVEL MALE SUICIDE ANALYSIS")
print("=" * 80)
print("=" * 80)

print(f"""
  DATA: {len(adf)} counties with CDC WONDER male suicide rates (2018-2022)
  SOURCE: ICD-10 codes X60-X84, males only, county-level
  MASTER: {len(master)} counties × {len(master.columns)} variables

  TOP CORRELATES OF MALE SUICIDE (by |r|):""")

if corr_results:
    sorted_corrs = sorted(corr_results, key=lambda x: abs(x['r']), reverse=True)
    for c in sorted_corrs[:12]:
        print(f"    {c['label']:<40} r = {c['r']:+.4f}  {c['sig']}  (n={c['n']})")

print(f"""
  ALL REAL DATA. NO SIMULATIONS. NO SYNTHETIC DATA.
  Every number from CDC WONDER, Census ACS, Census CBP, or County Health Rankings.
""")
print("=" * 80)
