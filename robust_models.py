#!/usr/bin/env python3
"""
ROBUSTNESS CHECKS — Alternative Models
========================================
1. HC3 Robust Standard Errors (heteroscedasticity-consistent)
2. Moran's I test for spatial autocorrelation
3. Spatial Lag Model (via 2SLS with spatially-lagged DV)
4. Weighted Least Squares (population-weighted)
5. Bootstrap Confidence Intervals (1000 resamples)
6. Median Regression (robust to outliers)
7. Compare all models side-by-side

All implemented with numpy/scipy only (no statsmodels/pysal).
"""
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, t as t_dist, chi2, norm
from scipy.spatial import KDTree

warnings.filterwarnings('ignore')
OUTPUT_DIR = Path('/Users/bobbarclay/Documents/soldiers')
np.random.seed(42)


def ols_full(y, X_df, weights=None):
    """Full OLS with HC3 robust SEs, hat matrix, leverage."""
    X_std = (X_df - X_df.mean()) / X_df.std()
    X = np.column_stack([np.ones(len(X_std)), X_std.values])
    y_arr = y.values.astype(float)
    n, k = X.shape

    if weights is not None:
        W = np.diag(np.sqrt(weights))
        Xw = W @ X
        yw = W @ y_arr
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    else:
        beta = np.linalg.lstsq(X, y_arr, rcond=None)[0]

    y_hat = X @ beta
    resid = y_arr - y_hat
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_arr - y_arr.mean())**2)
    r_sq = 1 - ss_res / ss_tot
    adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - k)
    mse = ss_res / (n - k)

    XtX_inv = np.linalg.inv(X.T @ X)

    # Classical SEs
    se_classical = np.sqrt(np.diag(mse * XtX_inv))

    # Hat matrix diagonal (leverage)
    H = X @ XtX_inv @ X.T
    h = np.diag(H)

    # HC3 robust SEs: Var(beta) = (X'X)^-1 X' diag(e_i^2/(1-h_ii)^2) X (X'X)^-1
    e_hc3 = resid**2 / (1 - h)**2
    meat = X.T @ np.diag(e_hc3) @ X
    V_hc3 = XtX_inv @ meat @ XtX_inv
    se_hc3 = np.sqrt(np.diag(V_hc3))

    names = ['const'] + list(X_df.columns)
    params = {}
    for i, name in enumerate(names):
        t_classical = beta[i] / se_classical[i]
        p_classical = 2 * t_dist.sf(abs(t_classical), df=n - k)
        t_robust = beta[i] / se_hc3[i]
        p_robust = 2 * t_dist.sf(abs(t_robust), df=n - k)
        params[name] = {
            'beta': beta[i],
            'se_classical': se_classical[i], 't_classical': t_classical, 'p_classical': p_classical,
            'se_hc3': se_hc3[i], 't_hc3': t_robust, 'p_hc3': p_robust,
        }

    f_stat = ((ss_tot - ss_res) / (k - 1)) / mse
    return {'params': params, 'r_squared': r_sq, 'adj_r_squared': adj_r_sq,
            'n': n, 'k': k, 'residuals': resid, 'y_hat': y_hat, 'f_stat': f_stat,
            'beta': beta, 'X': X, 'y': y_arr, 'XtX_inv': XtX_inv, 'h': h}


def sig(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'n.s.'


# ============================================================
# LOAD DATA
# ============================================================
print("=" * 90)
print("ROBUSTNESS CHECKS — ALTERNATIVE MODELS")
print("=" * 90)

master = pd.read_csv(OUTPUT_DIR / 'real_county_dataset.csv', dtype={'FIPS': str, 'state_fips': str})
master['FIPS'] = master['FIPS'].str.zfill(5)
adf = master.dropna(subset=['male_suicide_rate_cdc']).copy()

full_preds = ['veteran_pct', 'rural_urban_code', 'male_female_ratio', 'extraction_employment_pct',
              'poverty_rate', 'pct_bachelors_or_higher', 'male_divorced_separated_pct',
              'unemployment_rate', 'pct_no_internet']

mdf = adf[['male_suicide_rate_cdc'] + full_preds + ['latitude', 'longitude', 'total_population']].dropna()
print(f"Analysis dataset: {len(mdf)} counties with all variables + coordinates")

y = mdf['male_suicide_rate_cdc']
X_df = mdf[full_preds]

# ============================================================
# MODEL 1: OLS WITH CLASSICAL vs HC3 ROBUST SEs
# ============================================================
print("\n" + "=" * 90)
print("MODEL 1: OLS — CLASSICAL vs HC3 ROBUST STANDARD ERRORS")
print("=" * 90)

result = ols_full(y, X_df)

print(f"\nn = {result['n']}, R² = {result['r_squared']:.4f}, Adj R² = {result['adj_r_squared']:.4f}")
print(f"\n{'Predictor':<30} {'β':>8} {'SE(OLS)':>8} {'p(OLS)':>10} {'SE(HC3)':>8} {'p(HC3)':>10} {'Change':>8}")
print("-" * 85)

for pred in full_preds:
    p = result['params'][pred]
    se_change = ((p['se_hc3'] - p['se_classical']) / p['se_classical']) * 100
    print(f"{pred:<30} {p['beta']:>+8.3f} {p['se_classical']:>8.3f} {p['p_classical']:>10.6f} "
          f"{p['se_hc3']:>8.3f} {p['p_hc3']:>10.6f} {se_change:>+7.1f}%")

print("\nKey question: Do any predictors LOSE significance with HC3?")
for pred in full_preds:
    p = result['params'][pred]
    if p['p_classical'] < 0.05 and p['p_hc3'] >= 0.05:
        print(f"  YES: {pred} loses significance (p goes from {p['p_classical']:.4f} to {p['p_hc3']:.4f})")
    elif p['p_classical'] < 0.001 and p['p_hc3'] >= 0.001:
        print(f"  WEAKENED: {pred} (p goes from {p['p_classical']:.6f} to {p['p_hc3']:.6f})")

lost = sum(1 for pred in full_preds if result['params'][pred]['p_classical'] < 0.05 and result['params'][pred]['p_hc3'] >= 0.05)
if lost == 0:
    print("  NO — All significant predictors remain significant with HC3 robust SEs")

# ============================================================
# MODEL 2: MORAN'S I — SPATIAL AUTOCORRELATION TEST
# ============================================================
print("\n" + "=" * 90)
print("MODEL 2: MORAN'S I — SPATIAL AUTOCORRELATION TEST")
print("=" * 90)

# Build k-nearest-neighbors spatial weights (k=5)
coords = mdf[['latitude', 'longitude']].values
tree = KDTree(coords)
k_neighbors = 5
distances, indices = tree.query(coords, k=k_neighbors + 1)  # +1 because includes self

# Build row-standardized weights matrix (sparse-ish via neighbors)
n_sp = len(mdf)
resid = result['residuals']
resid_mean = resid.mean()
resid_dev = resid - resid_mean

# Moran's I = (n/S0) * (sum_i sum_j w_ij * (x_i - xbar)(x_j - xbar)) / sum_i (x_i - xbar)^2
numerator = 0.0
S0 = 0.0
for i in range(n_sp):
    neighbors = indices[i, 1:]  # skip self
    w = 1.0 / k_neighbors  # row-standardized
    for j in neighbors:
        numerator += w * resid_dev[i] * resid_dev[j]
        S0 += w

denominator = np.sum(resid_dev**2)
morans_I = (n_sp / S0) * (numerator / denominator)

# Expected value and variance under randomization
E_I = -1.0 / (n_sp - 1)
# Approximate variance (normality assumption)
S1 = 0
S2 = 0
# For row-standardized knn, simplified variance:
var_I = (n_sp**2 * (n_sp - 1) * S0**(-2) *
         (n_sp * np.sum(resid_dev**4) / (np.sum(resid_dev**2)**2) - 1) /
         ((n_sp - 1) * (n_sp - 2) * (n_sp - 3)))

# Simpler z-score approximation
z_moran = (morans_I - E_I) / np.sqrt(var_I) if var_I > 0 else 0
p_moran = 2 * (1 - norm.cdf(abs(z_moran)))

print(f"\nSpatial weights: {k_neighbors}-nearest neighbors, row-standardized")
print(f"Moran's I = {morans_I:.4f}")
print(f"Expected I = {E_I:.4f}")
print(f"z = {z_moran:.2f}, p = {p_moran:.6f}")

if p_moran < 0.001:
    print("\n*** SIGNIFICANT SPATIAL AUTOCORRELATION DETECTED ***")
    print("Residuals are spatially clustered — standard errors are underestimated.")
    print("Spatial regression model needed (see Model 3 below).")
elif p_moran < 0.05:
    print("\n* Modest spatial autocorrelation detected.")
else:
    print("\nNo significant spatial autocorrelation.")

# ============================================================
# MODEL 3: SPATIAL LAG MODEL (via manual 2SLS)
# ============================================================
print("\n" + "=" * 90)
print("MODEL 3: SPATIAL LAG MODEL")
print("=" * 90)
print("y = ρWy + Xβ + ε  (estimated via spatial two-stage least squares)")

# Create spatially-lagged dependent variable (Wy)
Wy = np.zeros(n_sp)
for i in range(n_sp):
    neighbors = indices[i, 1:]
    Wy[i] = np.mean(y.values[neighbors])

# 2SLS: Instrument Wy with WX (spatially lagged predictors)
X_std = (X_df - X_df.mean()) / X_df.std()
WX = np.zeros_like(X_std.values)
for i in range(n_sp):
    neighbors = indices[i, 1:]
    WX[i] = np.mean(X_std.values[neighbors], axis=0)

# Stage 1: Regress Wy on X, WX
Z1 = np.column_stack([np.ones(n_sp), X_std.values, WX])
Wy_hat = Z1 @ np.linalg.lstsq(Z1, Wy, rcond=None)[0]

# Stage 2: Regress y on Wy_hat, X
X_s2 = np.column_stack([np.ones(n_sp), Wy_hat, X_std.values])
y_arr = y.values.astype(float)
beta_s2 = np.linalg.lstsq(X_s2, y_arr, rcond=None)[0]

y_hat_s2 = X_s2 @ beta_s2
resid_s2 = y_arr - y_hat_s2
ss_res_s2 = np.sum(resid_s2**2)
ss_tot_s2 = np.sum((y_arr - y_arr.mean())**2)
r_sq_s2 = 1 - ss_res_s2 / ss_tot_s2
n_s2, k_s2 = X_s2.shape
adj_r_sq_s2 = 1 - (1 - r_sq_s2) * (n_s2 - 1) / (n_s2 - k_s2)

mse_s2 = ss_res_s2 / (n_s2 - k_s2)
try:
    se_s2 = np.sqrt(np.diag(mse_s2 * np.linalg.inv(X_s2.T @ X_s2)))
except:
    se_s2 = np.full(k_s2, np.nan)

rho = beta_s2[1]
rho_se = se_s2[1]
rho_t = rho / rho_se if rho_se > 0 else 0
rho_p = 2 * t_dist.sf(abs(rho_t), df=n_s2 - k_s2)

print(f"\nSpatial autoregressive coefficient (ρ): {rho:.4f}")
print(f"  SE = {rho_se:.4f}, t = {rho_t:.2f}, p = {rho_p:.6f} {sig(rho_p)}")
print(f"\nR² = {r_sq_s2:.4f} (vs OLS R² = {result['r_squared']:.4f})")
print(f"Adj R² = {adj_r_sq_s2:.4f}")

print(f"\n{'Predictor':<30} {'β(Spatial)':>10} {'p(Spatial)':>12} {'β(OLS)':>10} {'p(OLS)':>10}")
print("-" * 75)

s2_names = ['const', 'Wy (spatial lag)'] + list(X_df.columns)
for i, name in enumerate(s2_names):
    b = beta_s2[i]
    s = se_s2[i]
    t_val = b / s if s > 0 else 0
    p_val = 2 * t_dist.sf(abs(t_val), df=n_s2 - k_s2)

    if name in result['params']:
        ols_b = result['params'][name]['beta']
        ols_p = result['params'][name]['p_classical']
        print(f"{name:<30} {b:>+10.4f} {p_val:>12.6f} {sig(p_val):>4} {ols_b:>+10.4f} {ols_p:>10.6f}")
    elif name == 'Wy (spatial lag)':
        print(f"{name:<30} {b:>+10.4f} {p_val:>12.6f} {sig(p_val):>4} {'---':>10} {'---':>10}")
    else:
        print(f"{name:<30} {b:>+10.4f} {p_val:>12.6f} {sig(p_val):>4}")

# Check if key predictors change
print("\nDo key predictors survive spatial correction?")
for i, pred in enumerate(full_preds):
    idx = i + 2  # offset for const and Wy
    b = beta_s2[idx]
    s = se_s2[idx]
    t_val = b / s if s > 0 else 0
    p_val = 2 * t_dist.sf(abs(t_val), df=n_s2 - k_s2)
    ols_p = result['params'][pred]['p_classical']
    status = "HOLDS" if p_val < 0.05 else "LOST"
    if ols_p < 0.05:
        print(f"  {pred:<30} OLS p={ols_p:.6f} → Spatial p={p_val:.6f}  [{status}]")

# ============================================================
# MODEL 4: WEIGHTED LEAST SQUARES (population-weighted)
# ============================================================
print("\n" + "=" * 90)
print("MODEL 4: WEIGHTED LEAST SQUARES (population-weighted)")
print("=" * 90)

pop = mdf['total_population'].values.astype(float)
pop_weights = pop / pop.mean()  # normalize

result_wls = ols_full(y, X_df, weights=pop_weights)

print(f"\nn = {result_wls['n']}, R² = {result_wls['r_squared']:.4f}")
print(f"\n{'Predictor':<30} {'β(WLS)':>10} {'p(WLS)':>10} {'β(OLS)':>10} {'p(OLS)':>10}")
print("-" * 65)
for pred in full_preds:
    pw = result_wls['params'][pred]
    po = result['params'][pred]
    print(f"{pred:<30} {pw['beta']:>+10.4f} {pw['p_classical']:>10.6f} {po['beta']:>+10.4f} {po['p_classical']:>10.6f}")

# ============================================================
# MODEL 5: BOOTSTRAP CONFIDENCE INTERVALS (1000 resamples)
# ============================================================
print("\n" + "=" * 90)
print("MODEL 5: BOOTSTRAP CONFIDENCE INTERVALS (1000 resamples)")
print("=" * 90)

n_boot = 1000
boot_betas = np.zeros((n_boot, len(full_preds)))
X_std_full = (X_df - X_df.mean()) / X_df.std()
X_mat = np.column_stack([np.ones(len(X_std_full)), X_std_full.values])
y_arr = y.values.astype(float)

for b in range(n_boot):
    idx = np.random.choice(len(y_arr), len(y_arr), replace=True)
    X_b = X_mat[idx]
    y_b = y_arr[idx]
    try:
        beta_b = np.linalg.lstsq(X_b, y_b, rcond=None)[0]
        boot_betas[b] = beta_b[1:]  # skip intercept
    except:
        boot_betas[b] = np.nan

print(f"\n{'Predictor':<30} {'β':>8} {'Boot 2.5%':>10} {'Boot 97.5%':>10} {'Boot SE':>8} {'OLS SE':>8} {'Significant':>12}")
print("-" * 90)
for i, pred in enumerate(full_preds):
    b_vals = boot_betas[:, i]
    b_vals = b_vals[~np.isnan(b_vals)]
    lo = np.percentile(b_vals, 2.5)
    hi = np.percentile(b_vals, 97.5)
    b_se = np.std(b_vals)
    b_mean = result['params'][pred]['beta']
    ols_se = result['params'][pred]['se_classical']
    is_sig = "YES" if (lo > 0 or hi < 0) else "NO (crosses 0)"
    print(f"{pred:<30} {b_mean:>+8.3f} {lo:>+10.3f} {hi:>+10.3f} {b_se:>8.3f} {ols_se:>8.3f} {is_sig:>12}")

# ============================================================
# MODEL 6: MEDIAN REGRESSION (Least Absolute Deviations)
# ============================================================
print("\n" + "=" * 90)
print("MODEL 6: MEDIAN REGRESSION (Iteratively Reweighted Least Squares)")
print("=" * 90)

# LAD via IRLS
X_lad = np.column_stack([np.ones(len(X_std_full)), X_std_full.values])
y_lad = y.values.astype(float)
beta_lad = np.linalg.lstsq(X_lad, y_lad, rcond=None)[0]  # OLS start

for iteration in range(50):
    resid_lad = y_lad - X_lad @ beta_lad
    weights_lad = 1.0 / (np.abs(resid_lad) + 1e-6)
    W_lad = np.diag(weights_lad)
    try:
        beta_new = np.linalg.inv(X_lad.T @ W_lad @ X_lad) @ X_lad.T @ W_lad @ y_lad
    except:
        break
    if np.max(np.abs(beta_new - beta_lad)) < 1e-6:
        beta_lad = beta_new
        break
    beta_lad = beta_new

y_hat_lad = X_lad @ beta_lad
resid_final = y_lad - y_hat_lad
mae = np.mean(np.abs(resid_final))

print(f"\nMAE = {mae:.2f} (vs OLS RMSE = {np.sqrt(np.mean(result['residuals']**2)):.2f})")
print(f"\n{'Predictor':<30} {'β(Median)':>10} {'β(OLS)':>10} {'Agree':>8}")
print("-" * 60)
lad_names = ['const'] + list(X_df.columns)
for i, name in enumerate(lad_names):
    if name == 'const':
        continue
    b_lad = beta_lad[i]
    b_ols = result['params'][name]['beta']
    agree = "YES" if (b_lad * b_ols > 0) else "NO"
    print(f"{name:<30} {b_lad:>+10.4f} {b_ols:>+10.4f} {agree:>8}")

# ============================================================
# SUMMARY COMPARISON TABLE
# ============================================================
print("\n" + "=" * 90)
print("SUMMARY: ALL MODELS COMPARED — DO KEY FINDINGS HOLD?")
print("=" * 90)

print(f"\n{'Predictor':<25} {'OLS':>8} {'HC3':>8} {'Spatial':>8} {'WLS':>8} {'Boot CI':>10} {'Median':>8} {'ROBUST?':>8}")
print("-" * 90)

for i, pred in enumerate(full_preds):
    ols_sig = sig(result['params'][pred]['p_classical'])
    hc3_sig = sig(result['params'][pred]['p_hc3'])

    # Spatial
    idx_s = i + 2
    b_s = beta_s2[idx_s]
    s_s = se_s2[idx_s]
    t_s = b_s / s_s if s_s > 0 else 0
    p_s = 2 * t_dist.sf(abs(t_s), df=n_s2 - k_s2)
    spatial_sig = sig(p_s)

    wls_sig = sig(result_wls['params'][pred]['p_classical'])

    # Bootstrap
    b_vals = boot_betas[:, i]
    b_vals = b_vals[~np.isnan(b_vals)]
    lo = np.percentile(b_vals, 2.5)
    hi = np.percentile(b_vals, 97.5)
    boot_sig = "YES" if (lo > 0 or hi < 0) else "no"

    # Median
    b_med = beta_lad[i + 1]
    b_ols_val = result['params'][pred]['beta']
    med_agree = "+" if b_med * b_ols_val > 0 else "-"

    # Robust across all?
    all_sig = all([
        result['params'][pred]['p_classical'] < 0.05,
        result['params'][pred]['p_hc3'] < 0.05,
        p_s < 0.05,
        result_wls['params'][pred]['p_classical'] < 0.05,
        (lo > 0 or hi < 0),
    ])
    robust = "ROBUST" if all_sig else "MIXED"

    print(f"{pred:<25} {ols_sig:>8} {hc3_sig:>8} {spatial_sig:>8} {wls_sig:>8} {boot_sig:>10} {med_agree:>8} {robust:>8}")

print(f"""
Legend: *** p<.001, ** p<.01, * p<.05, n.s. not significant
Boot CI: YES = 95% CI excludes 0
Median: + = same direction as OLS, - = opposite
ROBUST = significant in ALL models
""")

# Final R² comparison
print(f"R² Comparison:")
print(f"  OLS:         {result['r_squared']:.4f}")
print(f"  Spatial Lag: {r_sq_s2:.4f}")
print(f"  WLS:         {result_wls['r_squared']:.4f}")

# Moran's I on spatial model residuals
resid_s2_dev = resid_s2 - resid_s2.mean()
num_s2 = 0.0
for i in range(n_sp):
    neighbors = indices[i, 1:]
    for j in neighbors:
        num_s2 += (1.0/k_neighbors) * resid_s2_dev[i] * resid_s2_dev[j]
den_s2 = np.sum(resid_s2_dev**2)
morans_I_s2 = (n_sp / S0) * (num_s2 / den_s2)
print(f"\nMoran's I (OLS residuals):     {morans_I:.4f} (p = {p_moran:.6f})")
print(f"Moran's I (Spatial residuals): {morans_I_s2:.4f}")
if abs(morans_I_s2) < abs(morans_I):
    print(f"  Spatial model REDUCED autocorrelation by {(1 - abs(morans_I_s2)/abs(morans_I))*100:.0f}%")

print("\n" + "=" * 90)
print("ROBUSTNESS VERIFICATION COMPLETE")
print("=" * 90)
