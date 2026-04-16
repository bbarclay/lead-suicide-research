#!/usr/bin/env python3
"""
International Lead Exposure Burden vs. Suicide Rates Analysis
=============================================================
Data source: IHME Global Burden of Disease Study 2023 via Our World in Data API

Indicators:
- Lead exposure death rate: OWID indicator 1173823
  "Age-standardized Deaths from all causes attributed to lead exposure per 100,000 people"
- Suicide death rate: OWID indicator 1165226
  "Age-standardized deaths from self-harm per 100,000 people"

Both are age-standardized rates per 100,000 people.
"""

import json
import os
import urllib.request
import csv
import io
import sys

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUTPUT_DIR = "/Users/bobbarclay/Documents/soldiers"

# ──────────────────────────────────────────────────────────────
# 1. Download data from OWID API
# ──────────────────────────────────────────────────────────────

def fetch_owid_indicator(indicator_id, label):
    """Download indicator data + metadata from the OWID API."""
    base = "https://api.ourworldindata.org/v1/indicators"

    print(f"Downloading {label} (indicator {indicator_id})...")

    # Data
    req = urllib.request.Request(
        f"{base}/{indicator_id}.data.json",
        headers={'User-Agent': 'Mozilla/5.0 (research)'}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())

    # Metadata (for entity names)
    req = urllib.request.Request(
        f"{base}/{indicator_id}.metadata.json",
        headers={'User-Agent': 'Mozilla/5.0 (research)'}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        meta = json.loads(resp.read().decode())

    # Build entity lookup
    entities = {e['id']: e for e in meta['dimensions']['entities']['values']}

    # Build DataFrame
    rows = []
    for val, year, ent_id in zip(data['values'], data['years'], data['entities']):
        ent = entities.get(ent_id, {})
        rows.append({
            'entity_id': ent_id,
            'country': ent.get('name', f'Unknown_{ent_id}'),
            'code': ent.get('code', ''),
            'year': year,
            'value': val
        })

    df = pd.DataFrame(rows)
    print(f"  -> {len(df)} observations, {df['country'].nunique()} entities, "
          f"years {df['year'].min()}-{df['year'].max()}")
    return df, meta


# Fetch both datasets
lead_df, lead_meta = fetch_owid_indicator(1173823, "Lead exposure death rate")
suicide_df, suicide_meta = fetch_owid_indicator(1165226, "Suicide death rate")

# ──────────────────────────────────────────────────────────────
# 2. Save raw data as CSV
# ──────────────────────────────────────────────────────────────

lead_csv = os.path.join(OUTPUT_DIR, "ihme_gbd_lead_exposure_death_rate_by_country.csv")
suicide_csv = os.path.join(OUTPUT_DIR, "ihme_gbd_suicide_death_rate_by_country.csv")

lead_df.to_csv(lead_csv, index=False)
suicide_df.to_csv(suicide_csv, index=False)
print(f"\nSaved: {lead_csv}")
print(f"Saved: {suicide_csv}")

# ──────────────────────────────────────────────────────────────
# 3. Merge datasets — use 2019 as primary year (pre-COVID,
#    latest year with both indicators commonly available)
# ──────────────────────────────────────────────────────────────

# Filter to countries only (have ISO 3-letter codes, exclude regions)
# Also exclude aggregate entities
EXCLUDE_ENTITIES = {
    'World', 'Africa', 'Asia', 'Europe', 'North America', 'South America',
    'Oceania', 'European Union', 'High-income', 'Low-income',
    'Lower-middle-income', 'Upper-middle-income', 'High SDI', 'High-middle SDI',
    'Middle SDI', 'Low-middle SDI', 'Low SDI', 'World Bank Upper Middle Income',
    'World Bank Lower Middle Income', 'World Bank Low Income',
    'World Bank High Income', 'African Union', 'Commonwealth',
    'Central Asia', 'Central Europe', 'Central Latin America',
    'Central sub-Saharan Africa', 'East Asia', 'Eastern Europe',
    'Eastern sub-Saharan Africa', 'High-income Asia Pacific',
    'High-income North America', 'Latin America and Caribbean',
    'Middle East and North Africa', 'North Africa and Middle East',
    'South Asia', 'Southeast Asia', 'Southern Latin America',
    'Southern sub-Saharan Africa', 'Sub-Saharan Africa',
    'Tropical Latin America', 'Western Europe',
    'Western sub-Saharan Africa', 'Andean Latin America',
    'Australasia', 'Caribbean', 'Central Europe, Eastern Europe, and Central Asia',
    'G20', 'Latin America & Caribbean - World Bank',
    'Middle East & North Africa - World Bank',
    'OECD Countries', 'South Asia - World Bank',
    'Sub-Saharan Africa - World Bank', 'East Asia & Pacific - World Bank',
    'Europe & Central Asia - World Bank', 'North America (WB)',
}

def filter_countries(df):
    """Keep only rows with valid ISO codes and that aren't aggregate regions."""
    mask = (
        df['code'].notna() &
        (df['code'].str.len() == 3) &
        (~df['country'].isin(EXCLUDE_ENTITIES))
    )
    return df[mask].copy()

lead_countries = filter_countries(lead_df)
suicide_countries = filter_countries(suicide_df)

print(f"\nAfter filtering to countries:")
print(f"  Lead data: {lead_countries['country'].nunique()} countries")
print(f"  Suicide data: {suicide_countries['country'].nunique()} countries")

# ──────────────────────────────────────────────────────────────
# 4. Merge on multiple years and analyze
# ──────────────────────────────────────────────────────────────

# Find common years
lead_years = set(lead_countries['year'].unique())
suicide_years = set(suicide_countries['year'].unique())
common_years = sorted(lead_years & suicide_years)
print(f"\nCommon years: {common_years[0]}-{common_years[-1]} ({len(common_years)} years)")

# Use the most recent year available in both + also 2019 for robustness
analysis_years = [2019]
if common_years[-1] != 2019:
    analysis_years.append(common_years[-1])

results = {}

for year in analysis_years:
    print(f"\n{'='*70}")
    print(f"ANALYSIS FOR YEAR {year}")
    print(f"{'='*70}")

    lead_yr = lead_countries[lead_countries['year'] == year][['code', 'country', 'value']].copy()
    lead_yr.rename(columns={'value': 'lead_death_rate'}, inplace=True)

    suicide_yr = suicide_countries[suicide_countries['year'] == year][['code', 'country', 'value']].copy()
    suicide_yr.rename(columns={'value': 'suicide_rate'}, inplace=True)

    merged = pd.merge(lead_yr, suicide_yr, on=['code', 'country'], how='inner')
    merged = merged.dropna(subset=['lead_death_rate', 'suicide_rate'])

    print(f"Countries with both measures: {len(merged)}")
    print(f"Lead death rate: mean={merged['lead_death_rate'].mean():.2f}, "
          f"median={merged['lead_death_rate'].median():.2f}, "
          f"range=[{merged['lead_death_rate'].min():.2f}, {merged['lead_death_rate'].max():.2f}]")
    print(f"Suicide rate:    mean={merged['suicide_rate'].mean():.2f}, "
          f"median={merged['suicide_rate'].median():.2f}, "
          f"range=[{merged['suicide_rate'].min():.2f}, {merged['suicide_rate'].max():.2f}]")

    # ── Pearson correlation ──
    r_pearson, p_pearson = stats.pearsonr(merged['lead_death_rate'], merged['suicide_rate'])
    print(f"\nPearson r  = {r_pearson:.4f}  (p = {p_pearson:.4e})")

    # ── Spearman rank correlation (robust to outliers) ──
    r_spearman, p_spearman = stats.spearmanr(merged['lead_death_rate'], merged['suicide_rate'])
    print(f"Spearman rho = {r_spearman:.4f}  (p = {p_spearman:.4e})")

    # ── Kendall tau ──
    tau, p_tau = stats.kendalltau(merged['lead_death_rate'], merged['suicide_rate'])
    print(f"Kendall tau  = {tau:.4f}  (p = {p_tau:.4e})")

    # ── Linear regression ──
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        merged['lead_death_rate'], merged['suicide_rate']
    )
    print(f"\nLinear regression: suicide = {slope:.3f} * lead + {intercept:.3f}")
    print(f"  R-squared = {r_value**2:.4f}, p = {p_value:.4e}, SE = {std_err:.4f}")

    # ── Top and bottom countries ──
    print(f"\nTop 10 countries by LEAD exposure death rate ({year}):")
    top_lead = merged.nlargest(10, 'lead_death_rate')
    for _, row in top_lead.iterrows():
        print(f"  {row['country']:30s}  lead={row['lead_death_rate']:6.2f}  suicide={row['suicide_rate']:6.2f}")

    print(f"\nTop 10 countries by SUICIDE rate ({year}):")
    top_suicide = merged.nlargest(10, 'suicide_rate')
    for _, row in top_suicide.iterrows():
        print(f"  {row['country']:30s}  lead={row['lead_death_rate']:6.2f}  suicide={row['suicide_rate']:6.2f}")

    # ── Quartile analysis ──
    merged['lead_quartile'] = pd.qcut(merged['lead_death_rate'], 4, labels=['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)'])
    quartile_means = merged.groupby('lead_quartile', observed=True)['suicide_rate'].agg(['mean', 'median', 'count'])
    print(f"\nSuicide rate by lead exposure quartile ({year}):")
    print(quartile_means.to_string())

    # ANOVA across quartiles
    groups = [g['suicide_rate'].values for _, g in merged.groupby('lead_quartile', observed=True)]
    f_stat, p_anova = stats.f_oneway(*groups)
    print(f"  ANOVA F={f_stat:.3f}, p={p_anova:.4e}")

    results[year] = {
        'merged': merged,
        'r_pearson': r_pearson,
        'p_pearson': p_pearson,
        'r_spearman': r_spearman,
        'p_spearman': p_spearman,
        'tau': tau,
        'p_tau': p_tau,
        'slope': slope,
        'intercept': intercept,
        'r_sq': r_value**2,
        'n': len(merged)
    }

# ──────────────────────────────────────────────────────────────
# 5. Multi-year panel analysis
# ──────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("PANEL ANALYSIS: All overlapping years (1990-2023)")
print(f"{'='*70}")

lead_panel = lead_countries[['code', 'country', 'year', 'value']].rename(columns={'value': 'lead_death_rate'})
suicide_panel = suicide_countries[['code', 'country', 'year', 'value']].rename(columns={'value': 'suicide_rate'})

panel = pd.merge(lead_panel, suicide_panel, on=['code', 'country', 'year'], how='inner')
panel = panel.dropna(subset=['lead_death_rate', 'suicide_rate'])

print(f"Panel observations: {len(panel)}")
print(f"Countries: {panel['country'].nunique()}, Years: {panel['year'].nunique()}")

# Overall panel correlation
r_panel, p_panel = stats.pearsonr(panel['lead_death_rate'], panel['suicide_rate'])
rho_panel, p_rho_panel = stats.spearmanr(panel['lead_death_rate'], panel['suicide_rate'])
print(f"Panel Pearson r = {r_panel:.4f} (p = {p_panel:.4e})")
print(f"Panel Spearman rho = {rho_panel:.4f} (p = {p_rho_panel:.4e})")

# Within-country correlations (country fixed effects logic)
print(f"\nWithin-country correlations (countries with 10+ years of data):")
country_corrs = []
for country, grp in panel.groupby('country'):
    if len(grp) >= 10:
        r, p = stats.pearsonr(grp['lead_death_rate'], grp['suicide_rate'])
        country_corrs.append({'country': country, 'r': r, 'p': p, 'n': len(grp)})

cc_df = pd.DataFrame(country_corrs)
print(f"  Countries analyzed: {len(cc_df)}")
print(f"  Mean within-country r: {cc_df['r'].mean():.4f}")
print(f"  Median within-country r: {cc_df['r'].median():.4f}")
print(f"  Countries with positive r: {(cc_df['r'] > 0).sum()} ({(cc_df['r'] > 0).mean()*100:.1f}%)")
print(f"  Countries with significant positive r (p<0.05): {((cc_df['r'] > 0) & (cc_df['p'] < 0.05)).sum()}")
print(f"  Countries with significant negative r (p<0.05): {((cc_df['r'] < 0) & (cc_df['p'] < 0.05)).sum()}")

# ── Country-level temporal trends ──
# For each country, compute the change in lead death rate and suicide rate
# over the full period, then correlate the changes
print(f"\nChange analysis: Do countries where lead exposure fell more also see bigger suicide changes?")
changes = []
for country, grp in panel.groupby('country'):
    grp_sorted = grp.sort_values('year')
    if len(grp_sorted) >= 10:
        first5 = grp_sorted.head(5)
        last5 = grp_sorted.tail(5)
        lead_change = last5['lead_death_rate'].mean() - first5['lead_death_rate'].mean()
        suicide_change = last5['suicide_rate'].mean() - first5['suicide_rate'].mean()
        changes.append({
            'country': country,
            'lead_change': lead_change,
            'suicide_change': suicide_change
        })

changes_df = pd.DataFrame(changes)
r_change, p_change = stats.pearsonr(changes_df['lead_change'], changes_df['suicide_change'])
rho_change, p_rho_change = stats.spearmanr(changes_df['lead_change'], changes_df['suicide_change'])
print(f"  Countries: {len(changes_df)}")
print(f"  Pearson r (change in lead vs change in suicide): {r_change:.4f} (p={p_change:.4e})")
print(f"  Spearman rho: {rho_change:.4f} (p={p_rho_change:.4e})")


# ──────────────────────────────────────────────────────────────
# 6. Save merged data
# ──────────────────────────────────────────────────────────────

# Save the 2019 cross-sectional dataset
merged_csv = os.path.join(OUTPUT_DIR, "ihme_gbd_lead_vs_suicide_by_country_2019.csv")
results[2019]['merged'].to_csv(merged_csv, index=False)
print(f"\nSaved merged 2019 data: {merged_csv}")

# Save full panel
panel_csv = os.path.join(OUTPUT_DIR, "ihme_gbd_lead_vs_suicide_panel.csv")
panel.to_csv(panel_csv, index=False)
print(f"Saved full panel: {panel_csv}")

# ──────────────────────────────────────────────────────────────
# 7. Visualization
# ──────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("International Lead Exposure Burden vs. Suicide Rates\n"
             "(IHME Global Burden of Disease Study 2023, via OWID API)",
             fontsize=14, fontweight='bold', y=0.98)

# ── Plot 1: Scatter for 2019 ──
ax = axes[0, 0]
m2019 = results[2019]['merged']
ax.scatter(m2019['lead_death_rate'], m2019['suicide_rate'],
           alpha=0.5, edgecolors='navy', facecolors='steelblue', s=40)

# Regression line
x_range = np.linspace(m2019['lead_death_rate'].min(), m2019['lead_death_rate'].max(), 100)
ax.plot(x_range, results[2019]['slope'] * x_range + results[2019]['intercept'],
        'r-', linewidth=2, label=f"r={results[2019]['r_pearson']:.3f}, p={results[2019]['p_pearson']:.2e}")

# Label notable countries
for _, row in m2019.nlargest(5, 'lead_death_rate').iterrows():
    ax.annotate(row['country'], (row['lead_death_rate'], row['suicide_rate']),
                fontsize=7, ha='left', va='bottom', alpha=0.7)
for _, row in m2019.nlargest(5, 'suicide_rate').iterrows():
    if row['country'] not in m2019.nlargest(5, 'lead_death_rate')['country'].values:
        ax.annotate(row['country'], (row['lead_death_rate'], row['suicide_rate']),
                    fontsize=7, ha='left', va='bottom', alpha=0.7)

ax.set_xlabel("Lead Exposure Death Rate (per 100K, age-standardized)")
ax.set_ylabel("Suicide Rate (per 100K, age-standardized)")
ax.set_title(f"Cross-Sectional: 2019 (n={results[2019]['n']} countries)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Plot 2: Quartile box plot ──
ax = axes[0, 1]
m2019_q = results[2019]['merged'].copy()
quartile_order = ['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)']
box_data = [m2019_q[m2019_q['lead_quartile'] == q]['suicide_rate'].values for q in quartile_order]
bp = ax.boxplot(box_data, labels=quartile_order, patch_artist=True,
                boxprops=dict(facecolor='lightsteelblue', edgecolor='navy'),
                medianprops=dict(color='red', linewidth=2))
colors = ['#d4e6f1', '#85c1e9', '#3498db', '#1a5276']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_xlabel("Lead Exposure Death Rate Quartile")
ax.set_ylabel("Suicide Rate (per 100K)")
ax.set_title("Suicide Rate by Lead Burden Quartile (2019)")
ax.grid(True, alpha=0.3, axis='y')

# ── Plot 3: Time trends for selected countries ──
ax = axes[1, 0]
# Pick countries spanning the lead exposure range
highlight_countries = ['United States', 'Russia', 'India', 'China', 'Japan',
                       'Germany', 'Brazil', 'Nigeria', 'South Korea', 'Mexico']
highlight_countries = [c for c in highlight_countries if c in panel['country'].unique()]

for country in highlight_countries:
    grp = panel[panel['country'] == country].sort_values('year')
    ax.plot(grp['lead_death_rate'], grp['suicide_rate'], '-o', markersize=2,
            alpha=0.7, label=country)
    # Mark start and end
    if len(grp) > 0:
        ax.annotate(country, (grp.iloc[-1]['lead_death_rate'], grp.iloc[-1]['suicide_rate']),
                    fontsize=6, alpha=0.7)

ax.set_xlabel("Lead Exposure Death Rate (per 100K)")
ax.set_ylabel("Suicide Rate (per 100K)")
ax.set_title("Temporal Trajectories (1990-2023)")
ax.legend(fontsize=6, ncol=2, loc='upper right')
ax.grid(True, alpha=0.3)

# ── Plot 4: Changes correlation ──
ax = axes[1, 1]
ax.scatter(changes_df['lead_change'], changes_df['suicide_change'],
           alpha=0.4, edgecolors='darkgreen', facecolors='mediumseagreen', s=30)
slope_c, intercept_c, _, _, _ = stats.linregress(changes_df['lead_change'], changes_df['suicide_change'])
x_range = np.linspace(changes_df['lead_change'].min(), changes_df['lead_change'].max(), 100)
ax.plot(x_range, slope_c * x_range + intercept_c, 'r-', linewidth=2,
        label=f"r={r_change:.3f}, p={p_change:.2e}")
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel("Change in Lead Death Rate (last 5yr avg - first 5yr avg)")
ax.set_ylabel("Change in Suicide Rate")
ax.set_title(f"Changes Over Time: Lead vs Suicide (n={len(changes_df)})")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig_path = os.path.join(OUTPUT_DIR, "international_lead_vs_suicide.png")
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
print(f"\nSaved figure: {fig_path}")
plt.close()


# ──────────────────────────────────────────────────────────────
# 8. Additional: Log-transformed analysis
# ──────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("LOG-TRANSFORMED ANALYSIS (2019)")
print(f"{'='*70}")

m = results[2019]['merged'].copy()
m = m[(m['lead_death_rate'] > 0) & (m['suicide_rate'] > 0)]
m['log_lead'] = np.log10(m['lead_death_rate'])
m['log_suicide'] = np.log10(m['suicide_rate'])

r_log, p_log = stats.pearsonr(m['log_lead'], m['log_suicide'])
rho_log, p_rho_log = stats.spearmanr(m['log_lead'], m['log_suicide'])
print(f"Log-log Pearson r = {r_log:.4f} (p = {p_log:.4e})")
print(f"Log-log Spearman rho = {rho_log:.4f} (p = {p_rho_log:.4e})")


# ──────────────────────────────────────────────────────────────
# 9. Regional analysis
# ──────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("REGIONAL ANALYSIS (2019)")
print(f"{'='*70}")

# Simple continent assignment by code prefix
# Use a proper mapping from the lead_df which includes regional entities
# Instead, let's use a rough WHO region mapping

# We'll use the data to define regions by checking which OWID regional entities
# each country falls into. For simplicity, use a manual mapping.

# Broad region mapping based on UN geoscheme
region_map = {
    'AFG': 'South Asia', 'ALB': 'Europe', 'DZA': 'Africa', 'AND': 'Europe',
    'AGO': 'Africa', 'ATG': 'Americas', 'ARG': 'Americas', 'ARM': 'Europe',
    'AUS': 'Oceania', 'AUT': 'Europe', 'AZE': 'Europe', 'BHS': 'Americas',
    'BHR': 'Middle East', 'BGD': 'South Asia', 'BRB': 'Americas', 'BLR': 'Europe',
    'BEL': 'Europe', 'BLZ': 'Americas', 'BEN': 'Africa', 'BTN': 'South Asia',
    'BOL': 'Americas', 'BIH': 'Europe', 'BWA': 'Africa', 'BRA': 'Americas',
    'BRN': 'East Asia', 'BGR': 'Europe', 'BFA': 'Africa', 'BDI': 'Africa',
    'KHM': 'East Asia', 'CMR': 'Africa', 'CAN': 'Americas', 'CPV': 'Africa',
    'CAF': 'Africa', 'TCD': 'Africa', 'CHL': 'Americas', 'CHN': 'East Asia',
    'COL': 'Americas', 'COM': 'Africa', 'COG': 'Africa', 'COD': 'Africa',
    'CRI': 'Americas', 'CIV': 'Africa', 'HRV': 'Europe', 'CUB': 'Americas',
    'CYP': 'Europe', 'CZE': 'Europe', 'DNK': 'Europe', 'DJI': 'Africa',
    'DMA': 'Americas', 'DOM': 'Americas', 'ECU': 'Americas', 'EGY': 'Middle East',
    'SLV': 'Americas', 'GNQ': 'Africa', 'ERI': 'Africa', 'EST': 'Europe',
    'SWZ': 'Africa', 'ETH': 'Africa', 'FJI': 'Oceania', 'FIN': 'Europe',
    'FRA': 'Europe', 'GAB': 'Africa', 'GMB': 'Africa', 'GEO': 'Europe',
    'DEU': 'Europe', 'GHA': 'Africa', 'GRC': 'Europe', 'GRD': 'Americas',
    'GTM': 'Americas', 'GIN': 'Africa', 'GNB': 'Africa', 'GUY': 'Americas',
    'HTI': 'Americas', 'HND': 'Americas', 'HUN': 'Europe', 'ISL': 'Europe',
    'IND': 'South Asia', 'IDN': 'East Asia', 'IRN': 'Middle East', 'IRQ': 'Middle East',
    'IRL': 'Europe', 'ISR': 'Middle East', 'ITA': 'Europe', 'JAM': 'Americas',
    'JPN': 'East Asia', 'JOR': 'Middle East', 'KAZ': 'Central Asia',
    'KEN': 'Africa', 'KIR': 'Oceania', 'PRK': 'East Asia', 'KOR': 'East Asia',
    'KWT': 'Middle East', 'KGZ': 'Central Asia', 'LAO': 'East Asia',
    'LVA': 'Europe', 'LBN': 'Middle East', 'LSO': 'Africa', 'LBR': 'Africa',
    'LBY': 'Africa', 'LTU': 'Europe', 'LUX': 'Europe', 'MDG': 'Africa',
    'MWI': 'Africa', 'MYS': 'East Asia', 'MDV': 'South Asia', 'MLI': 'Africa',
    'MLT': 'Europe', 'MHL': 'Oceania', 'MRT': 'Africa', 'MUS': 'Africa',
    'MEX': 'Americas', 'FSM': 'Oceania', 'MDA': 'Europe', 'MCO': 'Europe',
    'MNG': 'East Asia', 'MNE': 'Europe', 'MAR': 'Africa', 'MOZ': 'Africa',
    'MMR': 'East Asia', 'NAM': 'Africa', 'NRU': 'Oceania', 'NPL': 'South Asia',
    'NLD': 'Europe', 'NZL': 'Oceania', 'NIC': 'Americas', 'NER': 'Africa',
    'NGA': 'Africa', 'MKD': 'Europe', 'NOR': 'Europe', 'OMN': 'Middle East',
    'PAK': 'South Asia', 'PLW': 'Oceania', 'PAN': 'Americas', 'PNG': 'Oceania',
    'PRY': 'Americas', 'PER': 'Americas', 'PHL': 'East Asia', 'POL': 'Europe',
    'PRT': 'Europe', 'QAT': 'Middle East', 'ROU': 'Europe', 'RUS': 'Europe',
    'RWA': 'Africa', 'KNA': 'Americas', 'LCA': 'Americas', 'VCT': 'Americas',
    'WSM': 'Oceania', 'SMR': 'Europe', 'STP': 'Africa', 'SAU': 'Middle East',
    'SEN': 'Africa', 'SRB': 'Europe', 'SYC': 'Africa', 'SLE': 'Africa',
    'SGP': 'East Asia', 'SVK': 'Europe', 'SVN': 'Europe', 'SLB': 'Oceania',
    'SOM': 'Africa', 'ZAF': 'Africa', 'SSD': 'Africa', 'ESP': 'Europe',
    'LKA': 'South Asia', 'SDN': 'Africa', 'SUR': 'Americas', 'SWE': 'Europe',
    'CHE': 'Europe', 'SYR': 'Middle East', 'TWN': 'East Asia', 'TJK': 'Central Asia',
    'TZA': 'Africa', 'THA': 'East Asia', 'TLS': 'East Asia', 'TGO': 'Africa',
    'TON': 'Oceania', 'TTO': 'Americas', 'TUN': 'Africa', 'TUR': 'Europe',
    'TKM': 'Central Asia', 'TUV': 'Oceania', 'UGA': 'Africa', 'UKR': 'Europe',
    'ARE': 'Middle East', 'GBR': 'Europe', 'USA': 'Americas', 'URY': 'Americas',
    'UZB': 'Central Asia', 'VUT': 'Oceania', 'VEN': 'Americas', 'VNM': 'East Asia',
    'YEM': 'Middle East', 'ZMB': 'Africa', 'ZWE': 'Africa', 'PSE': 'Middle East',
}

m2019r = results[2019]['merged'].copy()
m2019r['region'] = m2019r['code'].map(region_map)

print("\nSuicide rate and lead death rate by region (2019):")
region_stats = m2019r.groupby('region').agg(
    n=('country', 'count'),
    lead_mean=('lead_death_rate', 'mean'),
    lead_median=('lead_death_rate', 'median'),
    suicide_mean=('suicide_rate', 'mean'),
    suicide_median=('suicide_rate', 'median')
).sort_values('lead_mean', ascending=False)
print(region_stats.to_string())

# Within-region correlations
print("\nWithin-region correlations (2019):")
for region, grp in m2019r.groupby('region'):
    if len(grp) >= 5:
        r, p = stats.pearsonr(grp['lead_death_rate'], grp['suicide_rate'])
        rho, p_rho = stats.spearmanr(grp['lead_death_rate'], grp['suicide_rate'])
        print(f"  {region:20s}  n={len(grp):3d}  Pearson r={r:+.3f} (p={p:.3f})  "
              f"Spearman rho={rho:+.3f} (p={p_rho:.3f})")


# ──────────────────────────────────────────────────────────────
# 10. Summary
# ──────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"""
DATA:
- Lead exposure death rate: Age-standardized deaths from all causes
  attributed to lead exposure per 100,000 people (IHME GBD 2023)
- Suicide rate: Age-standardized deaths from self-harm per 100,000
  people (IHME GBD 2023)
- Both downloaded via Our World in Data API from IHME GBD results

CROSS-SECTIONAL (2019):
- {results[2019]['n']} countries with both measures
- Pearson r = {results[2019]['r_pearson']:.4f} (p = {results[2019]['p_pearson']:.4e})
- Spearman rho = {results[2019]['r_spearman']:.4f} (p = {results[2019]['p_spearman']:.4e})
- R-squared = {results[2019]['r_sq']:.4f}
- Slope: {results[2019]['slope']:.3f} additional suicides per 100K for each
  additional lead-attributable death per 100K

PANEL (1990-2023):
- Pearson r = {r_panel:.4f} (p = {p_panel:.4e})
- Spearman rho = {rho_panel:.4f} (p = {p_rho_panel:.4e})

WITHIN-COUNTRY (temporal):
- Mean within-country r = {cc_df['r'].mean():.4f}
- {(cc_df['r'] > 0).sum()}/{len(cc_df)} countries show positive association over time

CHANGES OVER TIME:
- Correlation between lead change and suicide change: r = {r_change:.4f} (p = {p_change:.4e})

FILES SAVED:
- {lead_csv}
- {suicide_csv}
- {merged_csv}
- {panel_csv}
- {fig_path}
""")
