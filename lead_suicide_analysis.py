#!/usr/bin/env python3
"""
Lead Mining and Suicide: Extended Statistical Analysis
======================================================
Tests the lead-suicide connection across multiple analytical frameworks:
dose-response, state fixed effects, interactions, specificity, negative controls,
spatial analysis, bootstrapping, and attributable fraction estimation.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

OUT_DIR = '/Users/bobbarclay/Documents/soldiers/'

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 80)
print("LEAD MINING AND SUICIDE: COMPREHENSIVE STATISTICAL ANALYSIS")
print("=" * 80)

df = pd.read_csv(f'{OUT_DIR}real_county_dataset.csv')
print(f"\nDataset: {df.shape[0]} counties, {df.shape[1]} columns")
print(f"Counties with male suicide rate: {df['male_suicide_rate_true'].notna().sum()}")
print(f"Counties with lead/zinc/copper sites > 0: {(df['lead_zinc_copper'] > 0).sum()}")
print(f"Mean lead/zinc/copper sites (all): {df['lead_zinc_copper'].mean():.2f}")
print(f"Mean lead/zinc/copper sites (where > 0): {df.loc[df['lead_zinc_copper'] > 0, 'lead_zinc_copper'].mean():.2f}")

# Working subset: counties with suicide data
w = df.dropna(subset=['male_suicide_rate_true']).copy()
print(f"\nWorking sample: {len(w)} counties with suicide data")


# ============================================================
# 1. DOSE-RESPONSE TEST
# ============================================================
print("\n" + "=" * 80)
print("1. DOSE-RESPONSE TEST: Suicide Rate by Lead Mining Quintile")
print("=" * 80)

# Because lead_zinc_copper is very skewed (many zeros), create categories instead
# of strict quintiles. We'll use: 0 sites, 1-2, 3-10, 11-50, 51+
bins = [-1, 0, 2, 10, 50, 9999]
labels = ['0 sites', '1-2 sites', '3-10 sites', '11-50 sites', '51+ sites']
w['lead_category'] = pd.cut(w['lead_zinc_copper'], bins=bins, labels=labels)

dose_response = w.groupby('lead_category', observed=True).agg(
    n_counties=('male_suicide_rate_true', 'count'),
    mean_suicide_rate=('male_suicide_rate_true', 'mean'),
    median_suicide_rate=('male_suicide_rate_true', 'median'),
    std_suicide_rate=('male_suicide_rate_true', 'std'),
    mean_lead_sites=('lead_zinc_copper', 'mean')
).round(2)

# Add SE and 95% CI
dose_response['se'] = (dose_response['std_suicide_rate'] / np.sqrt(dose_response['n_counties'])).round(3)
dose_response['ci_lower'] = (dose_response['mean_suicide_rate'] - 1.96 * dose_response['se']).round(2)
dose_response['ci_upper'] = (dose_response['mean_suicide_rate'] + 1.96 * dose_response['se']).round(2)

print("\n" + dose_response.to_string())

# Also try proper quintiles on log(1 + lead_zinc_copper)
w['log_lead'] = np.log1p(w['lead_zinc_copper'])
# Since ~72% of counties have 0 lead sites, qcut can't make 5 equal bins.
# Use the dose categories already created for the trend test.
# Additionally, among counties WITH lead sites, split into terciles.
lead_only = w[w['lead_zinc_copper'] > 0].copy()
lead_only['lead_tercile'] = pd.qcut(lead_only['lead_zinc_copper'], q=3, labels=['Low lead (T1)', 'Mid lead (T2)', 'High lead (T3)'])
tercile_table = lead_only.groupby('lead_tercile', observed=True).agg(
    n_counties=('male_suicide_rate_true', 'count'),
    mean_lead_sites=('lead_zinc_copper', 'mean'),
    mean_suicide_rate=('male_suicide_rate_true', 'mean'),
    median_suicide_rate=('male_suicide_rate_true', 'median'),
).round(2)
print(f"\nAmong counties WITH lead/zinc/copper sites (n={len(lead_only)}), terciles:")
print(tercile_table.to_string())
print(f"  (For reference, counties with 0 sites: mean suicide = {w.loc[w['lead_zinc_copper']==0, 'male_suicide_rate_true'].mean():.2f})")

# Test for linear trend using Spearman on the dose categories
w['category_rank'] = w['lead_category'].cat.codes
rho, p_trend = stats.spearmanr(w['category_rank'], w['male_suicide_rate_true'])
print(f"\nSpearman trend test: rho = {rho:.4f}, p = {p_trend:.2e}")

# ANOVA across categories
groups = [g['male_suicide_rate_true'].values for _, g in w.groupby('lead_category', observed=True)]
f_stat, p_anova = stats.f_oneway(*groups)
print(f"One-way ANOVA across dose categories: F = {f_stat:.2f}, p = {p_anova:.2e}")

# Save
dose_response.to_csv(f'{OUT_DIR}lead_dose_response.csv')
print("\n[Saved: lead_dose_response.csv]")


# ============================================================
# 2. STATE FIXED EFFECTS
# ============================================================
print("\n" + "=" * 80)
print("2. STATE FIXED EFFECTS: Does Lead Mining Predict Suicide WITHIN States?")
print("=" * 80)

w2 = w.dropna(subset=['state_name', 'poverty_rate', 'unemployment_rate', 'median_age',
                        'veteran_pct', 'pct_pre1980_housing', 'rural_urban_code']).copy()

# Model without state FE
model_no_fe = smf.ols(
    'male_suicide_rate_true ~ lead_zinc_copper + poverty_rate + unemployment_rate + '
    'median_age + veteran_pct + pct_pre1980_housing + rural_urban_code',
    data=w2
).fit(cov_type='HC1')

# Model with state FE
model_fe = smf.ols(
    'male_suicide_rate_true ~ lead_zinc_copper + poverty_rate + unemployment_rate + '
    'median_age + veteran_pct + pct_pre1980_housing + rural_urban_code + C(state_name)',
    data=w2
).fit(cov_type='HC1')

print("\nWithout State Fixed Effects:")
print(f"  Lead coefficient: {model_no_fe.params['lead_zinc_copper']:.4f}")
print(f"  Std error:        {model_no_fe.bse['lead_zinc_copper']:.4f}")
print(f"  t-statistic:      {model_no_fe.tvalues['lead_zinc_copper']:.3f}")
print(f"  p-value:          {model_no_fe.pvalues['lead_zinc_copper']:.4e}")
print(f"  R-squared:        {model_no_fe.rsquared:.4f}")
print(f"  N:                {int(model_no_fe.nobs)}")

print("\nWith State Fixed Effects (50 state dummies):")
print(f"  Lead coefficient: {model_fe.params['lead_zinc_copper']:.4f}")
print(f"  Std error:        {model_fe.bse['lead_zinc_copper']:.4f}")
print(f"  t-statistic:      {model_fe.tvalues['lead_zinc_copper']:.3f}")
print(f"  p-value:          {model_fe.pvalues['lead_zinc_copper']:.4e}")
print(f"  R-squared:        {model_fe.rsquared:.4f}")
print(f"  N:                {int(model_fe.nobs)}")

pct_change = ((model_fe.params['lead_zinc_copper'] - model_no_fe.params['lead_zinc_copper']) /
              abs(model_no_fe.params['lead_zinc_copper'])) * 100
print(f"\n  Coefficient change with state FE: {pct_change:+.1f}%")
print(f"  --> {'Lead effect SURVIVES' if model_fe.pvalues['lead_zinc_copper'] < 0.05 else 'Lead effect DOES NOT survive'} state fixed effects (p={model_fe.pvalues['lead_zinc_copper']:.4e})")

# Log-transformed lead for robustness
model_fe_log = smf.ols(
    'male_suicide_rate_true ~ log_lead + poverty_rate + unemployment_rate + '
    'median_age + veteran_pct + pct_pre1980_housing + rural_urban_code + C(state_name)',
    data=w2
).fit(cov_type='HC1')

print(f"\n  Log(1+lead) with state FE: coef = {model_fe_log.params['log_lead']:.4f}, "
      f"p = {model_fe_log.pvalues['log_lead']:.4e}")

# Save state FE results
fe_results = pd.DataFrame({
    'Model': ['No state FE', 'State FE', 'State FE (log lead)'],
    'Coefficient': [model_no_fe.params['lead_zinc_copper'], model_fe.params['lead_zinc_copper'],
                    model_fe_log.params['log_lead']],
    'Std_Error': [model_no_fe.bse['lead_zinc_copper'], model_fe.bse['lead_zinc_copper'],
                  model_fe_log.bse['log_lead']],
    'T_stat': [model_no_fe.tvalues['lead_zinc_copper'], model_fe.tvalues['lead_zinc_copper'],
               model_fe_log.tvalues['log_lead']],
    'P_value': [model_no_fe.pvalues['lead_zinc_copper'], model_fe.pvalues['lead_zinc_copper'],
                model_fe_log.pvalues['log_lead']],
    'R_squared': [model_no_fe.rsquared, model_fe.rsquared, model_fe_log.rsquared],
    'N': [int(model_no_fe.nobs), int(model_fe.nobs), int(model_fe_log.nobs)]
})
fe_results.to_csv(f'{OUT_DIR}lead_state_fixed_effects.csv', index=False)
print("[Saved: lead_state_fixed_effects.csv]")


# ============================================================
# 3. LEAD MINING x VETERAN INTERACTION
# ============================================================
print("\n" + "=" * 80)
print("3. LEAD MINING x VETERAN INTERACTION")
print("=" * 80)

w2['high_veteran'] = (w2['veteran_pct'] > w2['veteran_pct'].median()).astype(int)
w2['lead_x_veteran'] = w2['lead_zinc_copper'] * w2['veteran_pct']
w2['lead_x_high_vet'] = w2['lead_zinc_copper'] * w2['high_veteran']

# Continuous interaction
model_vet_int = smf.ols(
    'male_suicide_rate_true ~ lead_zinc_copper * veteran_pct + poverty_rate + '
    'unemployment_rate + median_age + pct_pre1980_housing + rural_urban_code + C(state_name)',
    data=w2
).fit(cov_type='HC1')

print("\nContinuous interaction (lead_zinc_copper * veteran_pct):")
for var in ['lead_zinc_copper', 'veteran_pct', 'lead_zinc_copper:veteran_pct']:
    print(f"  {var:40s} coef={model_vet_int.params[var]:8.4f}  p={model_vet_int.pvalues[var]:.4e}")

# Binary interaction
model_vet_bin = smf.ols(
    'male_suicide_rate_true ~ lead_zinc_copper * high_veteran + poverty_rate + '
    'unemployment_rate + median_age + pct_pre1980_housing + rural_urban_code + C(state_name)',
    data=w2
).fit(cov_type='HC1')

print("\nBinary interaction (lead * above-median veteran %):")
for var in ['lead_zinc_copper', 'high_veteran', 'lead_zinc_copper:high_veteran']:
    print(f"  {var:40s} coef={model_vet_bin.params[var]:8.4f}  p={model_vet_bin.pvalues[var]:.4e}")

# Show marginal effects: suicide rate in 4 groups
print("\nMean suicide rate by lead presence x veteran status:")
w2['has_lead'] = (w2['lead_zinc_copper'] > 0).astype(int)
interaction_table = w2.groupby(['has_lead', 'high_veteran']).agg(
    n=('male_suicide_rate_true', 'count'),
    mean_suicide=('male_suicide_rate_true', 'mean'),
    se=('male_suicide_rate_true', lambda x: x.std() / np.sqrt(len(x)))
).round(2)
print(interaction_table.to_string())

interaction_table.to_csv(f'{OUT_DIR}lead_veteran_interaction.csv')
print("\n[Saved: lead_veteran_interaction.csv]")


# ============================================================
# 4. LEAD MINING x NATIVE AMERICAN INTERACTION
# ============================================================
print("\n" + "=" * 80)
print("4. LEAD MINING x NATIVE AMERICAN INTERACTION")
print("=" * 80)

w2['high_native'] = (w2['pct_native_american'] > w2['pct_native_american'].median()).astype(int)
w2['lead_x_native'] = w2['lead_zinc_copper'] * w2['pct_native_american']

# Continuous interaction
model_native_int = smf.ols(
    'male_suicide_rate_true ~ lead_zinc_copper * pct_native_american + poverty_rate + '
    'unemployment_rate + median_age + veteran_pct + pct_pre1980_housing + rural_urban_code + C(state_name)',
    data=w2
).fit(cov_type='HC1')

print("\nContinuous interaction (lead_zinc_copper * pct_native_american):")
for var in ['lead_zinc_copper', 'pct_native_american', 'lead_zinc_copper:pct_native_american']:
    print(f"  {var:45s} coef={model_native_int.params[var]:8.4f}  p={model_native_int.pvalues[var]:.4e}")

# Binary interaction
model_native_bin = smf.ols(
    'male_suicide_rate_true ~ lead_zinc_copper * high_native + poverty_rate + '
    'unemployment_rate + median_age + veteran_pct + pct_pre1980_housing + rural_urban_code + C(state_name)',
    data=w2
).fit(cov_type='HC1')

print("\nBinary interaction (lead * above-median Native American %):")
for var in ['lead_zinc_copper', 'high_native', 'lead_zinc_copper:high_native']:
    print(f"  {var:45s} coef={model_native_bin.params[var]:8.4f}  p={model_native_bin.pvalues[var]:.4e}")

# 4-group table
print("\nMean suicide rate by lead presence x Native American concentration:")
na_table = w2.groupby(['has_lead', 'high_native']).agg(
    n=('male_suicide_rate_true', 'count'),
    mean_suicide=('male_suicide_rate_true', 'mean'),
    mean_pct_native=('pct_native_american', 'mean')
).round(2)
print(na_table.to_string())

na_table.to_csv(f'{OUT_DIR}lead_native_american_interaction.csv')
print("\n[Saved: lead_native_american_interaction.csv]")


# ============================================================
# 5. SPECIFICITY TEST: Multiple Outcomes
# ============================================================
print("\n" + "=" * 80)
print("5. SPECIFICITY TEST: Does Lead Mining Predict Multiple Health Outcomes?")
print("=" * 80)

outcomes = {
    'male_suicide_rate_true': 'Male Suicide Rate',
    'chr_firearm_fatality_rate': 'Firearm Fatality Rate',
    'chr_overdose_rate': 'Drug Overdose Rate',
    'chr_premature_death_rate': 'Premature Death Rate (YPLL)',
    'chr_injury_death_rate': 'Injury Death Rate',
    'chr_homicide_rate': 'Homicide Rate'
}

specificity_results = []
for outcome_var, label in outcomes.items():
    tmp = w2.dropna(subset=[outcome_var])
    if len(tmp) < 100:
        continue
    try:
        m = smf.ols(
            f'{outcome_var} ~ lead_zinc_copper + poverty_rate + unemployment_rate + '
            f'median_age + veteran_pct + pct_pre1980_housing + rural_urban_code + C(state_name)',
            data=tmp
        ).fit(cov_type='HC1')
        specificity_results.append({
            'Outcome': label,
            'Coefficient': round(m.params['lead_zinc_copper'], 5),
            'Std_Error': round(m.bse['lead_zinc_copper'], 5),
            'T_stat': round(m.tvalues['lead_zinc_copper'], 3),
            'P_value': m.pvalues['lead_zinc_copper'],
            'N': int(m.nobs),
            'Significant': 'Yes' if m.pvalues['lead_zinc_copper'] < 0.05 else 'No'
        })
    except Exception as e:
        print(f"  Could not fit model for {label}: {e}")

spec_df = pd.DataFrame(specificity_results)
# Multiple testing correction
if len(spec_df) > 0:
    _, pvals_corrected, _, _ = multipletests(spec_df['P_value'], method='fdr_bh')
    spec_df['P_FDR'] = pvals_corrected
    spec_df['FDR_Significant'] = ['Yes' if p < 0.05 else 'No' for p in pvals_corrected]

print("\nLead mining coefficient across health outcomes (with state FE + controls):\n")
print(spec_df.to_string(index=False, float_format='%.5f'))

spec_df.to_csv(f'{OUT_DIR}lead_specificity_test.csv', index=False)
print("\n[Saved: lead_specificity_test.csv]")


# ============================================================
# 6. NEGATIVE CONTROL TEST
# ============================================================
print("\n" + "=" * 80)
print("6. NEGATIVE CONTROL: Does Lead Mining Predict What It Shouldn't?")
print("=" * 80)

neg_controls = {
    'homeownership_rate': 'Homeownership Rate',
    'pct_bachelors_or_higher': 'Bachelor\'s Degree %',
    'pct_hs_or_higher': 'High School Diploma %',
    'chr_social_associations': 'Social Associations Rate',
    'pct_broadband': 'Broadband Internet %',
}

neg_results = []
for var, label in neg_controls.items():
    tmp = w2.dropna(subset=[var])
    if len(tmp) < 100:
        continue
    try:
        m = smf.ols(
            f'{var} ~ lead_zinc_copper + poverty_rate + unemployment_rate + '
            f'median_age + rural_urban_code + C(state_name)',
            data=tmp
        ).fit(cov_type='HC1')
        neg_results.append({
            'Outcome': label,
            'Coefficient': round(m.params['lead_zinc_copper'], 5),
            'Std_Error': round(m.bse['lead_zinc_copper'], 5),
            'T_stat': round(m.tvalues['lead_zinc_copper'], 3),
            'P_value': m.pvalues['lead_zinc_copper'],
            'N': int(m.nobs),
            'Significant': 'Yes' if m.pvalues['lead_zinc_copper'] < 0.05 else 'No'
        })
    except Exception as e:
        print(f"  Could not fit model for {label}: {e}")

neg_df = pd.DataFrame(neg_results)
print("\nLead mining coefficient on negative control outcomes (should be null):\n")
print(neg_df.to_string(index=False, float_format='%.5f'))

# Compare: how many health outcomes significant vs how many negative controls significant
n_health_sig = spec_df['Significant'].eq('Yes').sum() if len(spec_df) > 0 else 0
n_neg_sig = neg_df['Significant'].eq('Yes').sum() if len(neg_df) > 0 else 0
print(f"\nSpecificity summary:")
print(f"  Health outcomes significantly predicted by lead: {n_health_sig}/{len(spec_df)}")
print(f"  Negative controls significantly predicted by lead: {n_neg_sig}/{len(neg_df)}")
if n_health_sig > n_neg_sig:
    print("  --> Lead mining shows SPECIFICITY for health outcomes over social/economic controls")

neg_df.to_csv(f'{OUT_DIR}lead_negative_controls.csv', index=False)
print("\n[Saved: lead_negative_controls.csv]")


# ============================================================
# 7. SPATIAL/GEOGRAPHIC CHECK
# ============================================================
print("\n" + "=" * 80)
print("7. SPATIAL CHECK: Geographic Clustering of Lead Mining")
print("=" * 80)

# Top states by lead mining sites
state_lead = w.groupby('state_name').agg(
    n_counties=('lead_zinc_copper', 'count'),
    counties_with_lead=('lead_zinc_copper', lambda x: (x > 0).sum()),
    total_lead_sites=('lead_zinc_copper', 'sum'),
    mean_lead_sites=('lead_zinc_copper', 'mean'),
    mean_suicide=('male_suicide_rate_true', 'mean')
).sort_values('total_lead_sites', ascending=False).round(2)

print("\nTop 15 states by total lead/zinc/copper mining sites:")
print(state_lead.head(15).to_string())

# Is the association driven by a few states?
# Re-run the model excluding top lead states one at a time
print("\n\nLeave-one-state-out sensitivity (top 10 lead states):")
top_lead_states = state_lead.head(10).index.tolist()
base_coef = model_fe.params['lead_zinc_copper']
base_p = model_fe.pvalues['lead_zinc_copper']
print(f"  {'State excluded':25s} {'Coef':>10s} {'P-value':>12s} {'Change':>10s}")
print(f"  {'(none - full model)':25s} {base_coef:10.4f} {base_p:12.4e} {'---':>10s}")

loo_results = []
for state in top_lead_states:
    tmp = w2[w2['state_name'] != state]
    try:
        m = smf.ols(
            'male_suicide_rate_true ~ lead_zinc_copper + poverty_rate + unemployment_rate + '
            'median_age + veteran_pct + pct_pre1980_housing + rural_urban_code + C(state_name)',
            data=tmp
        ).fit(cov_type='HC1')
        coef = m.params['lead_zinc_copper']
        pval = m.pvalues['lead_zinc_copper']
        change_pct = ((coef - base_coef) / abs(base_coef)) * 100
        print(f"  {state:25s} {coef:10.4f} {pval:12.4e} {change_pct:+9.1f}%")
        loo_results.append({
            'State_Excluded': state,
            'Coefficient': round(coef, 5),
            'P_value': pval,
            'Pct_Change': round(change_pct, 1)
        })
    except:
        print(f"  {state:25s} -- model failed --")

loo_df = pd.DataFrame(loo_results)
loo_df.to_csv(f'{OUT_DIR}lead_leave_one_state_out.csv', index=False)
print("\n[Saved: lead_leave_one_state_out.csv]")

# Regional analysis
region_map = {
    'Connecticut': 'Northeast', 'Maine': 'Northeast', 'Massachusetts': 'Northeast',
    'New Hampshire': 'Northeast', 'Rhode Island': 'Northeast', 'Vermont': 'Northeast',
    'New Jersey': 'Northeast', 'New York': 'Northeast', 'Pennsylvania': 'Northeast',
    'Illinois': 'Midwest', 'Indiana': 'Midwest', 'Michigan': 'Midwest', 'Ohio': 'Midwest',
    'Wisconsin': 'Midwest', 'Iowa': 'Midwest', 'Kansas': 'Midwest', 'Minnesota': 'Midwest',
    'Missouri': 'Midwest', 'Nebraska': 'Midwest', 'North Dakota': 'Midwest', 'South Dakota': 'Midwest',
    'Delaware': 'South', 'Florida': 'South', 'Georgia': 'South', 'Maryland': 'South',
    'North Carolina': 'South', 'South Carolina': 'South', 'Virginia': 'South',
    'District of Columbia': 'South', 'West Virginia': 'South', 'Alabama': 'South',
    'Kentucky': 'South', 'Mississippi': 'South', 'Tennessee': 'South', 'Arkansas': 'South',
    'Louisiana': 'South', 'Oklahoma': 'South', 'Texas': 'South',
    'Arizona': 'West', 'Colorado': 'West', 'Idaho': 'West', 'Montana': 'West',
    'Nevada': 'West', 'New Mexico': 'West', 'Utah': 'West', 'Wyoming': 'West',
    'Alaska': 'West', 'California': 'West', 'Hawaii': 'West', 'Oregon': 'West',
    'Washington': 'West'
}

w2['region'] = w2['state_name'].map(region_map)
print("\nLead mining coefficient by Census region:")
for region in ['Northeast', 'Midwest', 'South', 'West']:
    tmp = w2[w2['region'] == region]
    if len(tmp) < 50:
        continue
    try:
        m = smf.ols(
            'male_suicide_rate_true ~ lead_zinc_copper + poverty_rate + unemployment_rate + '
            'median_age + veteran_pct + pct_pre1980_housing + rural_urban_code + C(state_name)',
            data=tmp
        ).fit(cov_type='HC1')
        print(f"  {region:12s}: coef = {m.params['lead_zinc_copper']:8.4f}, "
              f"p = {m.pvalues['lead_zinc_copper']:.4e}, N = {int(m.nobs)}")
    except:
        print(f"  {region:12s}: model failed")


# ============================================================
# 8. BOOTSTRAP THE KEY FINDING
# ============================================================
print("\n" + "=" * 80)
print("8. BOOTSTRAP: 1000 Resamples of the Lead Mining Coefficient")
print("=" * 80)

np.random.seed(42)
n_boot = 1000
boot_coefs = []
boot_data = w2.dropna(subset=['male_suicide_rate_true', 'lead_zinc_copper', 'poverty_rate',
                               'unemployment_rate', 'median_age', 'veteran_pct',
                               'pct_pre1980_housing', 'rural_urban_code', 'state_name']).copy()

print(f"Bootstrap sample size: {len(boot_data)} counties")
print("Running 1000 bootstrap iterations...")

for i in range(n_boot):
    sample = boot_data.sample(n=len(boot_data), replace=True)
    try:
        m = smf.ols(
            'male_suicide_rate_true ~ lead_zinc_copper + poverty_rate + unemployment_rate + '
            'median_age + veteran_pct + pct_pre1980_housing + rural_urban_code + C(state_name)',
            data=sample
        ).fit()
        boot_coefs.append(m.params['lead_zinc_copper'])
    except:
        pass

boot_coefs = np.array(boot_coefs)
print(f"\nBootstrap results ({len(boot_coefs)} successful iterations):")
print(f"  Mean coefficient:     {boot_coefs.mean():.5f}")
print(f"  Median coefficient:   {np.median(boot_coefs):.5f}")
print(f"  Std deviation:        {boot_coefs.std():.5f}")
print(f"  2.5th percentile:     {np.percentile(boot_coefs, 2.5):.5f}")
print(f"  97.5th percentile:    {np.percentile(boot_coefs, 97.5):.5f}")
print(f"  Bootstrap 95% CI:     [{np.percentile(boot_coefs, 2.5):.5f}, {np.percentile(boot_coefs, 97.5):.5f}]")
print(f"  % positive:           {(boot_coefs > 0).mean() * 100:.1f}%")
print(f"  OLS point estimate:   {model_fe.params['lead_zinc_copper']:.5f}")

# Bias-corrected percentile
ols_coef = model_fe.params['lead_zinc_copper']
z0 = stats.norm.ppf((boot_coefs < ols_coef).mean())
alpha_lower = stats.norm.cdf(2 * z0 + stats.norm.ppf(0.025))
alpha_upper = stats.norm.cdf(2 * z0 + stats.norm.ppf(0.975))
bc_lower = np.percentile(boot_coefs, alpha_lower * 100)
bc_upper = np.percentile(boot_coefs, alpha_upper * 100)
print(f"  Bias-corrected 95% CI: [{bc_lower:.5f}, {bc_upper:.5f}]")

# Save bootstrap distribution
boot_df = pd.DataFrame({'bootstrap_coefficient': boot_coefs})
boot_df.to_csv(f'{OUT_DIR}lead_bootstrap_coefficients.csv', index=False)
print("\n[Saved: lead_bootstrap_coefficients.csv]")


# ============================================================
# 9. ATTRIBUTABLE FRACTION
# ============================================================
print("\n" + "=" * 80)
print("9. ATTRIBUTABLE FRACTION: Excess Deaths from Lead Mining Exposure")
print("=" * 80)

# Calculate the difference between lead-exposed and non-exposed counties
exposed = w[w['lead_zinc_copper'] > 0]['male_suicide_rate_true']
unexposed = w[w['lead_zinc_copper'] == 0]['male_suicide_rate_true']

mean_exposed = exposed.mean()
mean_unexposed = unexposed.mean()
rate_ratio = mean_exposed / mean_unexposed
excess_pct = (mean_exposed - mean_unexposed) / mean_unexposed * 100

print(f"\nRaw comparison:")
print(f"  Counties with lead mining (n={len(exposed)}):")
print(f"    Mean male suicide rate: {mean_exposed:.2f} per 100k")
print(f"  Counties without lead mining (n={len(unexposed)}):")
print(f"    Mean male suicide rate: {mean_unexposed:.2f} per 100k")
print(f"  Rate ratio: {rate_ratio:.3f}")
print(f"  Excess rate: {excess_pct:+.1f}%")

# Adjusted attributable fraction (using regression coefficient)
# Mean lead sites among exposed counties
mean_lead_exposed = w.loc[w['lead_zinc_copper'] > 0, 'lead_zinc_copper'].mean()
adjusted_coef = model_fe.params['lead_zinc_copper']
adjusted_excess = adjusted_coef * mean_lead_exposed
adjusted_pct = adjusted_excess / mean_unexposed * 100

print(f"\nAdjusted estimate (using state FE model):")
print(f"  Adjusted coefficient: {adjusted_coef:.5f} per lead site")
print(f"  Mean lead sites among exposed counties: {mean_lead_exposed:.1f}")
print(f"  Adjusted excess rate: {adjusted_excess:.2f} per 100k")
print(f"  Adjusted excess percentage: {adjusted_pct:+.1f}%")

# National excess deaths calculation
# US male population ~ 165 million; male suicide rate ~ 23 per 100k
# Proportion of US males in lead mining counties
total_male_pop_lead = df.loc[df['lead_zinc_copper'] > 0, 'male_population'].sum()
total_male_pop_all = df['male_population'].sum()
pct_in_lead_counties = total_male_pop_lead / total_male_pop_all * 100

print(f"\nNational impact estimation:")
print(f"  Male population in lead mining counties: {total_male_pop_lead:,.0f}")
print(f"  Total male population in dataset: {total_male_pop_all:,.0f}")
print(f"  % of males in lead mining counties: {pct_in_lead_counties:.1f}%")

# Using the raw rate difference
rate_diff_raw = mean_exposed - mean_unexposed  # per 100k
excess_deaths_raw = (rate_diff_raw / 100000) * total_male_pop_lead
print(f"\n  Raw excess rate difference: {rate_diff_raw:.2f} per 100k males")
print(f"  Estimated raw excess male suicides in lead counties: {excess_deaths_raw:.0f}")

# Using the adjusted coefficient
excess_deaths_adj = (adjusted_excess / 100000) * total_male_pop_lead
print(f"  Adjusted excess rate (from regression): {adjusted_excess:.2f} per 100k males")
print(f"  Estimated adjusted excess male suicides in lead counties: {excess_deaths_adj:.0f}")

# Population Attributable Fraction (PAF)
# PAF = Pe * (RR - 1) / [Pe * (RR - 1) + 1]
Pe = len(exposed) / (len(exposed) + len(unexposed))  # proportion exposed
PAF = Pe * (rate_ratio - 1) / (Pe * (rate_ratio - 1) + 1)
print(f"\n  Population Attributable Fraction (PAF):")
print(f"    Proportion of counties exposed: {Pe:.3f}")
print(f"    Rate ratio (unadjusted): {rate_ratio:.3f}")
print(f"    PAF = {PAF:.4f} ({PAF*100:.2f}%)")

# Using US total male suicides (~38,000 in recent years)
us_male_suicides = 38000  # approximate
attributable_deaths = PAF * us_male_suicides
print(f"    Estimated US male suicides: ~{us_male_suicides:,}")
print(f"    Attributable to lead mining exposure: ~{attributable_deaths:.0f} deaths")

# Save attributable fraction results
af_results = pd.DataFrame({
    'Metric': [
        'Counties with lead mining', 'Counties without lead mining',
        'Mean suicide rate (exposed)', 'Mean suicide rate (unexposed)',
        'Rate ratio (unadjusted)', 'Excess percentage (unadjusted)',
        'Adjusted coefficient (per lead site)', 'Mean lead sites (exposed)',
        'Adjusted excess rate', 'Adjusted excess percentage',
        'Male population in lead counties', 'Raw excess deaths',
        'Adjusted excess deaths', 'Population Attributable Fraction',
        'Estimated attributable deaths (US)'
    ],
    'Value': [
        len(exposed), len(unexposed),
        round(mean_exposed, 2), round(mean_unexposed, 2),
        round(rate_ratio, 3), round(excess_pct, 1),
        round(adjusted_coef, 5), round(mean_lead_exposed, 1),
        round(adjusted_excess, 2), round(adjusted_pct, 1),
        total_male_pop_lead, round(excess_deaths_raw, 0),
        round(excess_deaths_adj, 0), round(PAF, 4),
        round(attributable_deaths, 0)
    ]
})
af_results.to_csv(f'{OUT_DIR}lead_attributable_fraction.csv', index=False)
print("\n[Saved: lead_attributable_fraction.csv]")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY OF KEY FINDINGS")
print("=" * 80)

print(f"""
1. DOSE-RESPONSE: {'Clear gradient' if rho > 0.05 and p_trend < 0.05 else 'Weak/no gradient'} observed.
   Spearman rho = {rho:.4f}, p = {p_trend:.2e}
   Suicide rate in highest vs lowest category: {dose_response['mean_suicide_rate'].iloc[-1]:.1f} vs {dose_response['mean_suicide_rate'].iloc[0]:.1f}

2. STATE FIXED EFFECTS: Lead coefficient {'survives' if model_fe.pvalues['lead_zinc_copper'] < 0.05 else 'does not survive'} within-state analysis.
   Coefficient: {model_fe.params['lead_zinc_copper']:.4f} (p = {model_fe.pvalues['lead_zinc_copper']:.4e})

3. VETERAN INTERACTION: {'Significant' if model_vet_int.pvalues.get('lead_zinc_copper:veteran_pct', 1) < 0.05 else 'Not significant'} synergy between lead mining and veteran concentration.
   Interaction p = {model_vet_int.pvalues.get('lead_zinc_copper:veteran_pct', float('nan')):.4e}

4. NATIVE AMERICAN INTERACTION: {'Significant' if model_native_int.pvalues.get('lead_zinc_copper:pct_native_american', 1) < 0.05 else 'Not significant'} differential effect.
   Interaction p = {model_native_int.pvalues.get('lead_zinc_copper:pct_native_american', float('nan')):.4e}

5. SPECIFICITY: Lead mining predicts {n_health_sig}/{len(spec_df)} health outcomes tested.

6. NEGATIVE CONTROLS: Lead mining predicts {n_neg_sig}/{len(neg_df)} negative control outcomes.
   {'GOOD: Specificity confirmed' if n_neg_sig < n_health_sig else 'CONCERN: Lead predicts non-health outcomes too'}

7. SPATIAL STABILITY: Coefficient {'remains significant' if all(r['P_value'] < 0.05 for r in loo_results) else 'varies in significance'} after removing individual top-lead states.
   Range of coefficients: [{min(r['Coefficient'] for r in loo_results):.5f}, {max(r['Coefficient'] for r in loo_results):.5f}]

8. BOOTSTRAP: {(boot_coefs > 0).mean()*100:.1f}% of resamples show positive coefficient.
   95% CI: [{np.percentile(boot_coefs, 2.5):.5f}, {np.percentile(boot_coefs, 97.5):.5f}]

9. ATTRIBUTABLE FRACTION: PAF = {PAF*100:.2f}%, representing ~{attributable_deaths:.0f} excess male
   suicides nationally attributable to lead mining county residence.
""")

print("=" * 80)
print("CSV outputs saved:")
print("  lead_dose_response.csv")
print("  lead_state_fixed_effects.csv")
print("  lead_veteran_interaction.csv")
print("  lead_native_american_interaction.csv")
print("  lead_specificity_test.csv")
print("  lead_negative_controls.csv")
print("  lead_leave_one_state_out.csv")
print("  lead_bootstrap_coefficients.csv")
print("  lead_attributable_fraction.csv")
print("=" * 80)
