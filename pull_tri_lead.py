"""
Pull EPA TRI lead-compound facility-level release data for 2022,
aggregate to state/county, and correlate with state suicide + eagle Pb.

Strategy:
  1. Fetch form-level lead reports (chemical name contains LEAD) with facility info.
  2. Fetch release quantities joined by doc_ctrl_num.
  3. Aggregate to county kg/yr total + air + on-site vs off-site.
  4. Merge with state-level vet suicide + eagle Pb.
"""
import io
import urllib.request
import pandas as pd
import numpy as np

# ---- Chunked fetcher (Envirofacts caps at ~10k rows per query) ----
def fetch_csv(url):
    req = urllib.request.Request(url, headers={"Accept": "text/csv"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return pd.read_csv(io.BytesIO(r.read()), low_memory=False)

def fetch_all(base, page=9000):
    frames = []
    start = 0
    while True:
        url = f"{base}/ROWS/{start}:{start+page-1}/CSV"
        try:
            df = fetch_csv(url)
        except Exception as e:
            print(f"  fetch failed at {start}: {e}")
            break
        if len(df) == 0:
            break
        frames.append(df)
        print(f"  fetched {start}-{start+len(df)-1} ({len(df)} rows)")
        if len(df) < page:
            break
        start += page
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

print("STEP 1. Pull form-level lead reports (2022) with facility location")
form_url = ("https://data.epa.gov/efservice/tri_chem_info/chem_name/CONTAINING/"
            "LEAD/tri_reporting_form/reporting_year/2022/tri_facility")
forms = fetch_all(form_url)
print(f"Total form rows: {len(forms)}")

# Keep just what we need
wanted = [c for c in ["doc_ctrl_num", "reporting_year", "tri_chem_id",
                      "chem_name", "state_abbr", "state_county_fips_code",
                      "county_name", "facility_name", "fac_latitude", "fac_longitude"]
          if c in forms.columns]
forms = forms[wanted].drop_duplicates(subset="doc_ctrl_num")
print(f"Unique forms after dedup: {len(forms)}")
print("Chem names (top 20):")
print(forms["chem_name"].value_counts().head(20))

# ---- Pull release quantities for these forms ----
print("\nSTEP 2. Pull release quantities for these forms")
# Pull all release rows for 2022, then filter to doc_ctrl_nums we have
rel_url = ("https://data.epa.gov/efservice/tri_reporting_form/reporting_year/"
           "2022/tri_release_qty")
rel = fetch_all(rel_url)
print(f"Total release rows fetched: {len(rel)}")

if len(rel) > 0:
    rel = rel[rel["doc_ctrl_num"].isin(forms["doc_ctrl_num"])]
    print(f"Lead-relevant release rows: {len(rel)}")

# Aggregate releases per form (sum across environmental media)
rel["total_release"] = pd.to_numeric(rel["total_release"], errors="coerce").fillna(0)
rel_by_form = rel.groupby("doc_ctrl_num").agg(
    total_release_lb=("total_release", "sum"),
    air_fug=("total_release", lambda x: rel.loc[x.index][rel.loc[x.index]["environmental_medium"]=="AIR FUG"]["total_release"].sum()),
).reset_index()

# Simpler: pivot by medium
wide = rel.pivot_table(index="doc_ctrl_num",
                       columns="environmental_medium",
                       values="total_release",
                       aggfunc="sum",
                       fill_value=0).reset_index()
wide.columns = ["doc_ctrl_num"] + [f"rel_{c}" for c in wide.columns[1:]]
wide["total_release_lb"] = wide.filter(like="rel_").sum(axis=1)
print(f"Wide release table: {wide.shape}")

# ---- Merge with facility info ----
merged = forms.merge(wide, on="doc_ctrl_num", how="left")
merged["total_release_lb"] = merged["total_release_lb"].fillna(0)
merged["air_release_lb"] = (
    merged.get("rel_AIR FUG", 0).fillna(0) + merged.get("rel_AIR STACK", 0).fillna(0)
)
print(f"\nMerged facility-form-release: {len(merged)} rows, "
      f"{merged['state_abbr'].nunique()} states")

# ---- Aggregate to state level ----
state = merged.groupby("state_abbr").agg(
    n_facilities=("doc_ctrl_num", "nunique"),
    total_pb_lb=("total_release_lb", "sum"),
    air_pb_lb=("air_release_lb", "sum"),
).reset_index()
state["log_total_pb"] = np.log1p(state["total_pb_lb"])
state["log_air_pb"] = np.log1p(state["air_pb_lb"])
print("\nTop 10 states by TRI lead releases (2022):")
print(state.sort_values("total_pb_lb", ascending=False).head(10).to_string(index=False))

# Save
merged.to_csv("tri_lead_facility_2022.csv", index=False)
state.to_csv("tri_lead_by_state_2022.csv", index=False)
print("\nSaved: tri_lead_facility_2022.csv, tri_lead_by_state_2022.csv")

# ---- Quick state-level correlation with VA vet suicide ----
abbrev = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}
state["State"] = state["state_abbr"].map(abbrev)

va = pd.read_csv("VA_State_Veteran_Suicide_Rates_2023.csv").rename(columns={
    "Veteran_Suicide_Rate_per_100000": "vet_suicide_rate_2023",
})[["State", "vet_suicide_rate_2023"]]

panel = state.merge(va, on="State", how="inner")
for c in panel.columns:
    if c not in ("state_abbr", "State"):
        panel[c] = pd.to_numeric(panel[c], errors="coerce")

# Normalize by state population (use total_vets_working_age from county data as pop proxy)
try:
    county = pd.read_csv("real_county_dataset.csv", low_memory=False)
    pop = county.groupby("state_name")["male_population"].sum().reset_index()
    pop.columns = ["State", "male_pop"]
    panel = panel.merge(pop, on="State", how="left")
    panel["pb_per_cap"] = panel["total_pb_lb"] / panel["male_pop"]
    panel["log_pb_per_cap"] = np.log1p(panel["pb_per_cap"] * 1e6)
except Exception as e:
    print(f"Couldn't merge pop: {e}")

print(f"\nTRI lead x VA veteran suicide correlation (N={len(panel)} states):")
from scipy import stats
for x in ["total_pb_lb", "log_total_pb", "air_pb_lb", "log_air_pb", "log_pb_per_cap"]:
    if x not in panel.columns: continue
    d = panel[[x, "vet_suicide_rate_2023"]].dropna()
    r, p = stats.pearsonr(d[x], d["vet_suicide_rate_2023"])
    rs, ps = stats.spearmanr(d[x], d["vet_suicide_rate_2023"])
    print(f"  {x:20s}  n={len(d):2d}  r={r:+.3f} (p={p:.3f})  ρ={rs:+.3f} (p={ps:.3f})")

# Merge with eagle Pb for horse race
try:
    femur = pd.read_csv("Pb_Eagle_Femur.csv")
    femur.columns = [c.strip() for c in femur.columns]
    femur["State"] = femur["State"].str.strip()
    femur["Pb"] = femur["DW Lead (µg/g)"].astype(float)
    femur["chronic"] = (femur["Pb"] >= 10).astype(int)
    eagle = femur.groupby("State").agg(
        n_femur=("Pb","count"),
        mean_femur_Pb=("Pb","mean"),
        pct_chronic=("chronic","mean"),
    ).reset_index()
    full = panel.merge(eagle, on="State", how="inner")
    print(f"\nHorse race: TRI lead + eagle Pb (N={len(full)})")
    import statsmodels.api as sm
    for dv in ["vet_suicide_rate_2023"]:
        sub = full[["log_pb_per_cap", "mean_femur_Pb", dv, "n_femur"]].dropna()
        X = sm.add_constant(sub[["log_pb_per_cap", "mean_femur_Pb"]])
        w = np.sqrt(sub["n_femur"])
        fit = sm.WLS(sub[dv], X, weights=w).fit(cov_type="HC3")
        print(f"\nDV={dv}  N={len(sub)}  R²={fit.rsquared:.3f}")
        print(fit.summary2().tables[1].round(3).to_string())
except Exception as e:
    print(f"Horse race merge failed: {e}")

# Per-capita TRI (need state pop)
# Approximate with VA vet pop or skip for now
print("\nTop 10 states by TRI lead per facility (rough concentration proxy):")
state["pb_per_facility"] = state["total_pb_lb"] / state["n_facilities"].replace(0, np.nan)
print(state.sort_values("pb_per_facility", ascending=False).head(10).to_string(index=False))
