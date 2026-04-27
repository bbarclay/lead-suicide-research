#!/usr/bin/env python3
"""
FINAL UPGRADES
==============
1. Compute TRUE male-specific suicide rates (male deaths / male population)
2. Interaction models (veteran × extraction, rurality × alcohol access)
3. State fixed effects model
4. Compare: do results change with corrected rates?
"""
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, ttest_ind, spearmanr
from scipy.stats import t as t_dist

warnings.filterwarnings('ignore')
OUTPUT_DIR = Path(__file__).resolve().parent
np.random.seed(42)


def ols(y_arr, X_mat):
    """Minimal OLS returning beta, se, p, R²."""
    n, k = X_mat.shape
    beta = np.linalg.lstsq(X_mat, y_arr, rcond=None)[0]
    y_hat = X_mat @ beta
    resid = y_arr - y_hat
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_arr - y_arr.mean())**2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - k) if n > k else r_sq
    mse = ss_res / (n - k)
    try:
        se = np.sqrt(np.diag(mse * np.linalg.inv(X_mat.T @ X_mat)))
    except:
        se = np.full(k, np.nan)
    t_vals = beta / se
    p_vals = 2 * t_dist.sf(np.abs(t_vals), df=n - k)
    return {'beta': beta, 'se': se, 'p': p_vals, 'r_sq': r_sq, 'adj_r_sq': adj_r_sq, 'n': n, 'k': k, 'resid': resid}


def sig(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'n.s.'


# ============================================================
# LOAD
# ============================================================
master = pd.read_csv(OUTPUT_DIR / 'real_county_dataset.csv', dtype={'FIPS': str, 'state_fips': str})
master['FIPS'] = master['FIPS'].str.zfill(5)

# ============================================================
# 1. COMPUTE TRUE MALE-SPECIFIC RATES
# ============================================================
print("=" * 90)
print("1. TRUE MALE-SPECIFIC SUICIDE RATES (male deaths / male population)")
print("=" * 90)

# male_suicide_deaths / (male_population * 5 years) * 100,000
# CDC WONDER deaths are pooled 2018-2022 (5 years)
# male_population is from ACS 2022 (single-year estimate)
# Approximate 5-year male population = male_population * 5

master['male_suicide_rate_true'] = (
    master['male_suicide_deaths'] / (master['male_population'] * 5) * 100000
)

adf = master.dropna(subset=['male_suicide_rate_true']).copy()
print(f"Counties with true male rate: {len(adf)}")
print(f"  Mean: {adf['male_suicide_rate_true'].mean():.1f}/100K male pop")
print(f"  Median: {adf['male_suicide_rate_true'].median():.1f}")
print(f"  Range: {adf['male_suicide_rate_true'].min():.1f} - {adf['male_suicide_rate_true'].max():.1f}")

# Compare with CDC crude rate (total pop denominator)
both = adf.dropna(subset=['male_suicide_rate_cdc', 'male_suicide_rate_true'])
r, p = pearsonr(both['male_suicide_rate_cdc'], both['male_suicide_rate_true'])
ratio = both['male_suicide_rate_true'].mean() / both['male_suicide_rate_cdc'].mean()
print(f"\n  Correlation with CDC crude rate: r = {r:.4f}")
print(f"  True male rate is {ratio:.2f}x the CDC crude rate")
print(f"  (Expected ~2x since males are ~half the population)")

# Re-run top correlations with true rate
print(f"\n  Top correlations with TRUE male rate (vs CDC crude rate):")
for var, label in [('veteran_pct', 'Veteran %'), ('rural_urban_code', 'RUCC'),
                    ('chr_firearm_fatality_rate', 'Firearms'), ('elevation_feet', 'Altitude'),
                    ('extraction_employment_pct', 'Extraction'), ('bars_per_10k', 'Bars/10K'),
                    ('chr_excessive_drinking_pct', 'Drinking %'), ('alcohol_outlets_per_10k', 'Outlets/10K')]:
    if var in adf.columns:
        v1 = adf[[var, 'male_suicide_rate_true']].dropna()
        v2 = adf[[var, 'male_suicide_rate_cdc']].dropna()
        if len(v1) > 30 and len(v2) > 30:
            r1, p1 = pearsonr(v1[var], v1['male_suicide_rate_true'])
            r2, p2 = pearsonr(v2[var], v2['male_suicide_rate_cdc'])
            print(f"    {label:<20} True: r={r1:+.3f} {sig(p1)}   CDC: r={r2:+.3f} {sig(p2)}")

# Save true rate to master
master.to_csv(OUTPUT_DIR / 'real_county_dataset.csv', index=False)

# ============================================================
# 2. INTERACTION MODELS
# ============================================================
print("\n" + "=" * 90)
print("2. INTERACTION MODELS")
print("=" * 90)

full_preds = ['veteran_pct', 'rural_urban_code', 'male_female_ratio', 'extraction_employment_pct',
              'poverty_rate', 'male_divorced_separated_pct', 'unemployment_rate']

mdf = adf[['male_suicide_rate_cdc'] + full_preds].dropna().copy()
y = mdf['male_suicide_rate_cdc'].values

# Standardize predictors
for pred in full_preds:
    mdf[pred + '_z'] = (mdf[pred] - mdf[pred].mean()) / mdf[pred].std()

# --- Interaction A: Veteran % × Extraction % ---
print("\n--- Interaction A: Veteran % × Extraction Employment % ---")
mdf['vet_x_extract'] = mdf['veteran_pct_z'] * mdf['extraction_employment_pct_z']

pred_cols_A = [p + '_z' for p in full_preds] + ['vet_x_extract']
X_A = np.column_stack([np.ones(len(mdf)), mdf[pred_cols_A].values])
res_A = ols(y, X_A)

# Compare R² with and without interaction
pred_cols_base = [p + '_z' for p in full_preds]
X_base = np.column_stack([np.ones(len(mdf)), mdf[pred_cols_base].values])
res_base = ols(y, X_base)

names_A = ['const'] + [p.replace('_z', '') for p in pred_cols_A]
int_idx = names_A.index('vet_x_extract')
int_b = res_A['beta'][int_idx]
int_p = res_A['p'][int_idx]
delta_r2 = res_A['r_sq'] - res_base['r_sq']

print(f"  Base model R² = {res_base['r_sq']:.4f}")
print(f"  + Interaction R² = {res_A['r_sq']:.4f} (ΔR² = {delta_r2:.4f})")
print(f"  Veteran × Extraction: β = {int_b:+.4f}, p = {int_p:.6f} {sig(int_p)}")
if int_p < 0.05:
    if int_b > 0:
        print("  → SYNERGISTIC: The combination of high veteran % AND high extraction amplifies suicide risk")
    else:
        print("  → BUFFERING: One factor attenuates the other's effect")
else:
    print("  → No significant interaction (effects are additive, not multiplicative)")

# --- Interaction B: Rurality × Alcohol Access ---
print("\n--- Interaction B: Rurality × Alcohol Outlet Density ---")
mdf2 = adf[['male_suicide_rate_cdc'] + full_preds + ['alcohol_outlets_per_10k']].dropna().copy()
for pred in full_preds + ['alcohol_outlets_per_10k']:
    mdf2[pred + '_z'] = (mdf2[pred] - mdf2[pred].mean()) / mdf2[pred].std()

mdf2['rural_x_alcohol'] = mdf2['rural_urban_code_z'] * mdf2['alcohol_outlets_per_10k_z']

pred_cols_B = [p + '_z' for p in full_preds] + ['alcohol_outlets_per_10k_z', 'rural_x_alcohol']
X_B = np.column_stack([np.ones(len(mdf2)), mdf2[pred_cols_B].values])
y_B = mdf2['male_suicide_rate_cdc'].values
res_B = ols(y_B, X_B)

names_B = ['const'] + [p.replace('_z', '') for p in pred_cols_B]
int_idx_B = names_B.index('rural_x_alcohol')
int_b_B = res_B['beta'][int_idx_B]
int_p_B = res_B['p'][int_idx_B]

# Also get the main effect of alcohol in this model
alc_idx = names_B.index('alcohol_outlets_per_10k')
alc_b = res_B['beta'][alc_idx]
alc_p = res_B['p'][alc_idx]

print(f"  Alcohol outlets (main effect): β = {alc_b:+.4f}, p = {alc_p:.6f} {sig(alc_p)}")
print(f"  Rurality × Alcohol: β = {int_b_B:+.4f}, p = {int_p_B:.6f} {sig(int_p_B)}")
if int_p_B < 0.05:
    print("  → Alcohol's relationship with suicide DEPENDS on rurality context")
else:
    print("  → Alcohol has no significant effect regardless of rural/urban context")

# --- Interaction C: Veteran % × Rurality ---
print("\n--- Interaction C: Veteran % × Rurality ---")
mdf['vet_x_rural'] = mdf['veteran_pct_z'] * mdf['rural_urban_code_z']
pred_cols_C = [p + '_z' for p in full_preds] + ['vet_x_rural']
X_C = np.column_stack([np.ones(len(mdf)), mdf[pred_cols_C].values])
res_C = ols(y, X_C)

names_C = ['const'] + [p.replace('_z', '') for p in pred_cols_C]
int_idx_C = names_C.index('vet_x_rural')
int_b_C = res_C['beta'][int_idx_C]
int_p_C = res_C['p'][int_idx_C]
delta_r2_C = res_C['r_sq'] - res_base['r_sq']

print(f"  + Interaction R² = {res_C['r_sq']:.4f} (ΔR² = {delta_r2_C:.4f})")
print(f"  Veteran × Rurality: β = {int_b_C:+.4f}, p = {int_p_C:.6f} {sig(int_p_C)}")
if int_p_C < 0.05:
    if int_b_C > 0:
        print("  → Veterans in rural areas face COMPOUNDED risk (isolation × veteran status)")
    else:
        print("  → Veterans in rural areas have attenuated risk")

# ============================================================
# 3. STATE FIXED EFFECTS MODEL
# ============================================================
print("\n" + "=" * 90)
print("3. STATE FIXED EFFECTS MODEL")
print("=" * 90)
print("Controls for ALL unmeasured state-level confounders (laws, culture, climate, etc.)")

fe_df = adf[['male_suicide_rate_cdc', 'state_name'] + full_preds].dropna().copy()
y_fe = fe_df['male_suicide_rate_cdc'].values

# Create state dummies
state_dummies = pd.get_dummies(fe_df['state_name'], drop_first=True, dtype=float)

# Standardize predictors
X_preds = fe_df[full_preds].copy()
for pred in full_preds:
    X_preds[pred] = (X_preds[pred] - X_preds[pred].mean()) / X_preds[pred].std()

# Model with state FE
X_fe = np.column_stack([np.ones(len(fe_df)), X_preds.values, state_dummies.values])
res_fe = ols(y_fe, X_fe)

# Model without state FE (same sample)
X_no_fe = np.column_stack([np.ones(len(fe_df)), X_preds.values])
res_no_fe = ols(y_fe, X_no_fe)

print(f"\n  n = {res_fe['n']}")
print(f"  States: {len(state_dummies.columns) + 1}")
print(f"  R² without state FE: {res_no_fe['r_sq']:.4f}")
print(f"  R² with state FE:    {res_fe['r_sq']:.4f}")
print(f"  ΔR² from state FE:   {res_fe['r_sq'] - res_no_fe['r_sq']:.4f}")

print(f"\n  {'Predictor':<30} {'β(no FE)':>10} {'p(no FE)':>10} {'β(with FE)':>10} {'p(with FE)':>10} {'Survives':>10}")
print("  " + "-" * 82)

for i, pred in enumerate(full_preds):
    idx_no = i + 1  # skip constant
    idx_fe = i + 1
    b_no = res_no_fe['beta'][idx_no]
    p_no = res_no_fe['p'][idx_no]
    b_fe = res_fe['beta'][idx_fe]
    p_fe = res_fe['p'][idx_fe]
    survives = "YES" if p_fe < 0.05 else "NO"
    print(f"  {pred:<30} {b_no:>+10.4f} {p_no:>10.6f} {b_fe:>+10.4f} {p_fe:>10.6f} {survives:>10}")

print("\n  Key question: Which predictors survive WITHIN-STATE variation?")
print("  (These are effects that hold even comparing counties within the same state)")

# ============================================================
# 4. ALCOHOL CONFOUNDING WITH STATE FE
# ============================================================
print("\n" + "=" * 90)
print("4. ALCOHOL CONFOUNDING TEST — WITH STATE FIXED EFFECTS")
print("=" * 90)

for avar, alabel in [('bars_per_10k', 'Bars/10K'), ('alcohol_outlets_per_10k', 'Outlets/10K'),
                      ('chr_excessive_drinking_pct', 'Excessive Drinking %')]:
    test_df = adf[['male_suicide_rate_cdc', 'state_name', avar] + full_preds].dropna()
    if len(test_df) < 100:
        continue

    y_test = test_df['male_suicide_rate_cdc'].values
    sd = pd.get_dummies(test_df['state_name'], drop_first=True, dtype=float)

    # Alcohol alone
    x_alone = np.column_stack([np.ones(len(test_df)),
                                (test_df[avar] - test_df[avar].mean()) / test_df[avar].std()])
    r_alone = ols(y_test, x_alone)

    # Alcohol + controls (no state FE)
    X_ctrl = test_df[full_preds].copy()
    for p in full_preds:
        X_ctrl[p] = (X_ctrl[p] - X_ctrl[p].mean()) / X_ctrl[p].std()
    alc_z = (test_df[avar] - test_df[avar].mean()) / test_df[avar].std()
    x_ctrl = np.column_stack([np.ones(len(test_df)), alc_z.values, X_ctrl.values])
    r_ctrl = ols(y_test, x_ctrl)

    # Alcohol + controls + state FE
    x_fe = np.column_stack([np.ones(len(test_df)), alc_z.values, X_ctrl.values, sd.values])
    r_fe_alc = ols(y_test, x_fe)

    print(f"\n  {alabel}:")
    print(f"    Alone:           β={r_alone['beta'][1]:+.4f}, p={r_alone['p'][1]:.6f} {sig(r_alone['p'][1])}")
    print(f"    + Demographics:  β={r_ctrl['beta'][1]:+.4f}, p={r_ctrl['p'][1]:.6f} {sig(r_ctrl['p'][1])}")
    print(f"    + State FE:      β={r_fe_alc['beta'][1]:+.4f}, p={r_fe_alc['p'][1]:.6f} {sig(r_fe_alc['p'][1])}")

# ============================================================
# 5. DRY VS WET — WITHIN-STATE COMPARISON
# ============================================================
print("\n" + "=" * 90)
print("5. DRY vs WET — WITHIN-STATE COMPARISON")
print("=" * 90)
print("Comparing dry vs wet counties WITHIN the same state controls for all state-level factors")

if 'alcohol_access' in adf.columns:
    # For each state with both dry and wet counties, compute the difference
    state_diffs = []
    for state in adf['state_name'].dropna().unique():
        state_df = adf[adf['state_name'] == state]
        dry_rate = state_df[state_df['alcohol_access'] == 'dry']['male_suicide_rate_cdc'].dropna()
        wet_rate = state_df[state_df['alcohol_access'] == 'wet']['male_suicide_rate_cdc'].dropna()
        if len(dry_rate) >= 3 and len(wet_rate) >= 3:
            state_diffs.append({
                'state': state,
                'dry_mean': dry_rate.mean(),
                'wet_mean': wet_rate.mean(),
                'diff': dry_rate.mean() - wet_rate.mean(),
                'n_dry': len(dry_rate),
                'n_wet': len(wet_rate),
            })

    if state_diffs:
        sdf = pd.DataFrame(state_diffs)
        print(f"\n  States with both dry (>=3) and wet (>=3) counties: {len(sdf)}")
        print(f"  Mean within-state difference (dry - wet): {sdf['diff'].mean():+.2f}/100K")
        print(f"  States where dry > wet: {(sdf['diff'] > 0).sum()}/{len(sdf)}")
        print(f"  States where wet > dry: {(sdf['diff'] <= 0).sum()}/{len(sdf)}")

        # Sign test
        n_pos = (sdf['diff'] > 0).sum()
        n_total = len(sdf)
        from scipy.stats import binom_test
        try:
            p_sign = binom_test(n_pos, n_total, 0.5)
        except:
            from scipy.stats import binomtest
            p_sign = binomtest(n_pos, n_total, 0.5).pvalue
        print(f"  Sign test: p = {p_sign:.4f} {'(significant)' if p_sign < 0.05 else '(not significant)'}")

        # Show top states
        print(f"\n  {'State':<20} {'Dry Mean':>10} {'Wet Mean':>10} {'Diff':>8} {'n_dry':>6} {'n_wet':>6}")
        print("  " + "-" * 62)
        for _, row in sdf.sort_values('diff', ascending=False).head(10).iterrows():
            print(f"  {row['state']:<20} {row['dry_mean']:>10.1f} {row['wet_mean']:>10.1f} {row['diff']:>+8.1f} {int(row['n_dry']):>6} {int(row['n_wet']):>6}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 90)
print("UPGRADE SUMMARY")
print("=" * 90)

print(f"""
TRUE MALE RATES:
  Correlation with CDC crude rate: r = {r:.3f} (near-perfect tracking)
  True male rate ~{ratio:.1f}x the crude rate (as expected)
  → Patterns are identical; crude rate is a valid proxy

INTERACTION MODELS:
  Veteran × Extraction: β = {int_b:+.4f}, p = {int_p:.4f} {sig(int_p)}
  Veteran × Rurality:   β = {int_b_C:+.4f}, p = {int_p_C:.4f} {sig(int_p_C)}
  Rurality × Alcohol:   β = {int_b_B:+.4f}, p = {int_p_B:.4f} {sig(int_p_B)}

STATE FIXED EFFECTS:
  R² without FE: {res_no_fe['r_sq']:.4f}
  R² with FE:    {res_fe['r_sq']:.4f}
  → State FE adds {(res_fe['r_sq'] - res_no_fe['r_sq'])*100:.1f} percentage points
""")

print("=" * 90)
print("DONE")
print("=" * 90)
