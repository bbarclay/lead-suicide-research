"""
NHANES Analysis: Blood Lead Levels and Suicidal Ideation (PHQ-9 Item 9)
First analysis to test this in a nationally representative US sample.
"""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')
import os

# Load and combine all cycles
all_data = []
cycles = [
    ("2007", "DPQ_E.XPT", "PbCd_E.XPT", "DEMO_E.XPT"),
    ("2009", "DPQ_F.XPT", "PBCD_F.XPT", "DEMO_F.XPT"),
    ("2011", "DPQ_G.XPT", "PBCD_G.XPT", "DEMO_G.XPT"),
    ("2013", "DPQ_H.XPT", "PBCD_H.XPT", "DEMO_H.XPT"),
    ("2015", "DPQ_I.XPT", "PBCD_I.XPT", "DEMO_I.XPT"),
    ("2017", "DPQ_J.XPT", "PBCD_J.XPT", "DEMO_J.XPT"),
]

for year, dpq_file, lead_file, demo_file in cycles:
    try:
        dpq = pd.read_sas(f"nhanes_data/{year}_{dpq_file}", format='xport')
        lead = pd.read_sas(f"nhanes_data/{year}_{lead_file}", format='xport')
        demo = pd.read_sas(f"nhanes_data/{year}_{demo_file}", format='xport')

        # Merge on SEQN
        merged = dpq.merge(lead, on='SEQN', how='inner').merge(demo, on='SEQN', how='inner')
        merged['cycle'] = year
        all_data.append(merged)
        print(f"  {year}: DPQ={len(dpq)}, Lead={len(lead)}, Demo={len(demo)}, Merged={len(merged)}")
    except Exception as e:
        print(f"  {year}: ERROR - {e}")

df = pd.concat(all_data, ignore_index=True)
print(f"\nTotal merged records: {len(df)}")

# Key variables:
# DPQ090 = PHQ-9 Item 9: "Thoughts that you would be better off dead, or of hurting yourself"
#   0 = Not at all, 1 = Several days, 2 = More than half the days, 3 = Nearly every day
# LBXBPB = Blood lead (ug/dL)
# RIAGENDR = Gender (1=Male, 2=Female)
# RIDAGEYR = Age in years
# RIDRETH1 = Race/ethnicity
# INDFMPIR = Poverty-income ratio
# DMDEDUC2 = Education (adults 20+)

# Standardize blood lead variable name
for col in ['LBXBPB', 'LBDBPBSI']:
    if col in df.columns:
        print(f"Found lead variable: {col}")

# Check what columns we have
dpq_cols = [c for c in df.columns if c.startswith('DPQ')]
print(f"DPQ columns: {dpq_cols}")
lead_cols = [c for c in df.columns if 'PB' in c.upper() or 'LEAD' in c.upper() or 'LBX' in c.upper()]
print(f"Lead columns: {lead_cols[:10]}")

# PHQ-9 Item 9 (suicidal ideation)
if 'DPQ090' in df.columns:
    print(f"\nDPQ090 (suicidal ideation) distribution:")
    print(df['DPQ090'].value_counts().sort_index())

# Blood lead
if 'LBXBPB' in df.columns:
    bl = df['LBXBPB']
    print(f"\nBlood lead (LBXBPB): N={bl.notna().sum()}, mean={bl.mean():.2f}, median={bl.median():.2f}")

# ============================================================
# ANALYSIS: Blood Lead → Suicidal Ideation
# ============================================================
print("\n" + "="*70)
print("NHANES ANALYSIS: Blood Lead → Suicidal Ideation (PHQ-9 Item 9)")
print("="*70)

# Create binary suicidal ideation variable (any positive response)
df['suicidal_ideation'] = (df['DPQ090'] >= 1).astype(float)
df['suicidal_ideation'] = df['suicidal_ideation'].where(df['DPQ090'].notna())

# Males only (matching our paper's focus on male suicide)
df['male'] = (df['RIAGENDR'] == 1).astype(int)

# Analysis sample: adults 20+ with both blood lead and PHQ-9
mask = (df['LBXBPB'].notna() &
        df['suicidal_ideation'].notna() &
        df['RIDAGEYR'].notna() &
        (df['RIDAGEYR'] >= 20))

sample = df[mask].copy()
print(f"\nAnalysis sample (adults 20+): N = {len(sample)}")
print(f"Males: {(sample['male']==1).sum()}, Females: {(sample['male']==0).sum()}")
print(f"Suicidal ideation prevalence: {sample['suicidal_ideation'].mean()*100:.1f}%")
print(f"Mean blood lead: {sample['LBXBPB'].mean():.2f} ug/dL")

# --- ALL ADULTS ---
print("\n--- ALL ADULTS ---")
# Bivariate
si_yes = sample[sample['suicidal_ideation']==1]['LBXBPB']
si_no = sample[sample['suicidal_ideation']==0]['LBXBPB']
t, p = stats.ttest_ind(si_yes, si_no)
print(f"Mean BLL: SI+ = {si_yes.mean():.3f}, SI- = {si_no.mean():.3f}")
print(f"T-test: t = {t:.3f}, p = {p:.6f}")

# Logistic regression: unadjusted
sample['log_lead'] = np.log(sample['LBXBPB'] + 0.01)
m1 = smf.logit('suicidal_ideation ~ log_lead', data=sample).fit(disp=0)
print(f"\nUnadjusted logistic: log(BLL) OR = {np.exp(m1.params['log_lead']):.3f}, "
      f"p = {m1.pvalues['log_lead']:.6f}")

# Adjusted for age, sex, race, poverty, education
sample['age_c'] = sample['RIDAGEYR'] - sample['RIDAGEYR'].mean()
m2 = smf.logit('suicidal_ideation ~ log_lead + age_c + male + C(RIDRETH1) + INDFMPIR',
                data=sample.dropna(subset=['INDFMPIR', 'RIDRETH1'])).fit(disp=0)
print(f"Adjusted logistic: log(BLL) OR = {np.exp(m2.params['log_lead']):.3f}, "
      f"p = {m2.pvalues['log_lead']:.6f}")

# --- MALES ONLY ---
print("\n--- MALES ONLY ---")
males = sample[sample['male']==1].copy()
print(f"N males: {len(males)}")
print(f"SI prevalence (males): {males['suicidal_ideation'].mean()*100:.1f}%")

si_yes_m = males[males['suicidal_ideation']==1]['LBXBPB']
si_no_m = males[males['suicidal_ideation']==0]['LBXBPB']
t, p = stats.ttest_ind(si_yes_m, si_no_m)
print(f"Mean BLL: SI+ = {si_yes_m.mean():.3f}, SI- = {si_no_m.mean():.3f}")
print(f"T-test: t = {t:.3f}, p = {p:.6f}")

m3 = smf.logit('suicidal_ideation ~ log_lead', data=males).fit(disp=0)
print(f"\nUnadjusted logistic (males): log(BLL) OR = {np.exp(m3.params['log_lead']):.3f}, "
      f"p = {m3.pvalues['log_lead']:.6f}")

m4 = smf.logit('suicidal_ideation ~ log_lead + age_c + C(RIDRETH1) + INDFMPIR',
                data=males.dropna(subset=['INDFMPIR', 'RIDRETH1'])).fit(disp=0)
print(f"Adjusted logistic (males): log(BLL) OR = {np.exp(m4.params['log_lead']):.3f}, "
      f"p = {m4.pvalues['log_lead']:.6f}")

# --- QUARTILE ANALYSIS ---
print("\n--- BLOOD LEAD QUARTILE ANALYSIS (all adults) ---")
sample['bll_quartile'] = pd.qcut(sample['LBXBPB'], 4, labels=['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)'])
qtab = sample.groupby('bll_quartile').agg(
    n=('suicidal_ideation', 'count'),
    si_pct=('suicidal_ideation', lambda x: x.mean()*100),
    mean_bll=('LBXBPB', 'mean')
).round(2)
print(qtab)

# Test for trend
quartile_si = sample.groupby('bll_quartile')['suicidal_ideation'].mean()
trend = stats.spearmanr(range(4), quartile_si.values)
print(f"Trend test: rho = {trend.correlation:.3f}, p = {trend.pvalue:.4f}")

# --- MALES QUARTILE ---
print("\n--- BLOOD LEAD QUARTILE ANALYSIS (males only) ---")
males['bll_quartile'] = pd.qcut(males['LBXBPB'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
qtab_m = males.groupby('bll_quartile').agg(
    n=('suicidal_ideation', 'count'),
    si_pct=('suicidal_ideation', lambda x: x.mean()*100),
    mean_bll=('LBXBPB', 'mean')
).round(2)
print(qtab_m)

trend_m = stats.spearmanr(range(4), males.groupby('bll_quartile')['suicidal_ideation'].mean().values)
print(f"Males trend test: rho = {trend_m.correlation:.3f}, p = {trend_m.pvalue:.4f}")

# --- Q4 vs Q1 odds ratio ---
print("\n--- Q4 vs Q1 COMPARISON ---")
q1 = sample[sample['bll_quartile'] == 'Q1 (lowest)']['suicidal_ideation']
q4 = sample[sample['bll_quartile'] == 'Q4 (highest)']['suicidal_ideation']
# Odds ratio
or_q4q1 = (q4.mean()/(1-q4.mean())) / (q1.mean()/(1-q1.mean()))
print(f"Q4 SI prevalence: {q4.mean()*100:.2f}%")
print(f"Q1 SI prevalence: {q1.mean()*100:.2f}%")
print(f"Crude OR (Q4 vs Q1): {or_q4q1:.3f}")

print("\n" + "="*70)
print("DONE. If blood lead predicts suicidal ideation, this is the first")
print("US individual-level finding linking lead to suicide risk.")
print("="*70)
