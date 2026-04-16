"""
Comprehensive analysis of ALL identified lead mine/smelter Superfund counties.
"""
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('real_county_dataset.csv')
df['FIPS_str'] = df['FIPS'].astype(str).str.zfill(5)

# COMPREHENSIVE LIST - all identified lead mine/smelter counties
sites = {
    # RURAL MINING/SMELTER COMMUNITIES
    'Deer Lodge, MT (Anaconda smelter)': '30023',
    'Shoshone, ID (Bunker Hill)': '16079',
    'Lemhi, ID (Blackbird Mine)': '16059',
    'Silver Bow, MT (Butte/Berkeley Pit)': '30093',
    'Lake, CO (Leadville)': '08065',
    'Cherokee, KS (Galena)': '20021',
    'Ottawa, OK (Picher/Tar Creek)': '40115',
    'Iron, MO (Lead Belt)': '29093',
    'Washington, MO (Lead Belt)': '29221',
    'Jefferson, MO (Herculaneum)': '29099',
    'Lewis & Clark, MT (East Helena)': '30049',
    'St. Francois, MO (Lead Belt)': '29187',
    'Madison, MO (Lead Belt)': '29123',
    'Dent, MO (Lead Belt)': '29065',
    'Cascade, MT (Great Falls smelter)': '30013',
    'Kay, OK (Blackwell Zinc)': '40071',
    'Washington, OK (Bartlesville Zinc)': '40147',
    'Jasper, MO (Joplin/Tri-State)': '29097',
    'Newton, MO (Tri-State)': '29145',
    'Crawford, KS (Tri-State ext)': '20037',
    'Tooele, UT (Intl Smelter)': '49045',
    'San Juan, CO (Silverton)': '08111',
    'Mineral, CO (Creede)': '08079',
    'Pitkin, CO (Aspen/Smuggler)': '08097',
    'Pueblo, CO (Colorado Smelter)': '08101',
    'Kootenai, ID (CDA Basin)': '16055',
    'Benewah, ID (CDA Basin downstream)': '16009',
    'Gila, AZ (Hayden Smelter)': '04007',
    'Wythe, VA (Austinville Mine)': '51197',
    'Polk, TN (Copper Basin)': '47139',
    'Unicoi, TN (Bumpass Cove)': '47171',
    'Jo Daviess, IL (Galena lead)': '17085',
    'Dubuque, IA (Upper MS Valley)': '19061',
    'Grant, WI (WI Lead Region)': '55043',
    'Lafayette, WI (WI Lead Region)': '55065',
    'Iowa, WI (WI Lead Region)': '55049',
    'Ste. Genevieve, MO (SE MO Lead)': '29186',
    'Carbon, PA (Palmerton Zinc)': '42025',
    'Lyon, NV (Carson River)': '32019',
    'Storey, NV (Virginia City)': '32029',
    'Dona Ana, NM (Stephenson Mine)': '35013',
    # URBAN SMELTER SITES (for comparison)
    'Douglas, NE (Omaha ASARCO)': '31055',
    'El Paso, TX (ASARCO smelter)': '48141',
    'Dallas, TX (RSR Corp)': '48113',
    'Pierce, WA (Tacoma ASARCO)': '53053',
    'Lake, IN (East Chicago USS)': '18089',
    'Marion, IN (Indianapolis)': '18097',
    'Madison, IL (Granite City)': '17119',
    'Los Angeles, CA (Exide Vernon)': '06037',
    'Harris, TX (Houston Lead)': '48201',
    'Salt Lake, UT (Murray Smelter)': '49035',
    'Northampton, PA (Bethlehem)': '42095',
    'Denver, CO (ASARCO Globe)': '08031',
    'Berks, PA (Hamburg Battery)': '42011',
}

nat_avg = df['male_suicide_rate_true'].mean()
valid = df['male_suicide_rate_true'].dropna()
total = len(valid)

# Classify as rural vs urban based on rural_urban_code
rural_rates = []
urban_rates = []
all_rates = []

print(f"{'County':<45} {'Rate':>6} {'%ile':>6} {'xNatl':>6} {'RU':>4} {'Mining':>7}")
print("="*78)

for name, fips in sites.items():
    row = df[df['FIPS_str'] == fips]
    if len(row) > 0:
        r = row.iloc[0]
        sr = r.get('male_suicide_rate_true', None)
        ru = r.get('rural_urban_code', None)
        ms = r.get('mining_sites', None)
        if pd.notna(sr):
            pctile = (valid < sr).sum() / total * 100
            ratio = sr / nat_avg
            ru_str = f'{int(ru)}' if pd.notna(ru) else '?'
            ms_str = f'{int(ms)}' if pd.notna(ms) else '?'
            print(f"{name:<45} {sr:>5.1f} {pctile:>5.0f}% {ratio:>5.2f}x {ru_str:>4} {ms_str:>7}")
            all_rates.append(sr)
            if pd.notna(ru) and ru >= 4:  # Rural
                rural_rates.append(sr)
            else:
                urban_rates.append(sr)

print()
print("="*78)
print(f"ALL lead mine/smelter counties (N={len(all_rates)}):")
print(f"  Mean: {np.mean(all_rates):.1f}/100K, Median: {np.median(all_rates):.1f}")
print(f"  National avg: {nat_avg:.1f}/100K")
print(f"  Ratio: {np.mean(all_rates)/nat_avg:.2f}x")
above = sum(1 for r in all_rates if r > nat_avg)
print(f"  Above national avg: {above}/{len(all_rates)} ({above/len(all_rates)*100:.0f}%)")
p90 = valid.quantile(0.90)
top10 = sum(1 for r in all_rates if r > p90)
print(f"  In top 10%: {top10}/{len(all_rates)} ({top10/len(all_rates)*100:.0f}%)")

t, p = stats.ttest_ind(all_rates, valid.values)
print(f"  T-test vs all counties: t={t:.2f}, p={p:.6f}")

print()
print(f"RURAL mine/smelter counties (RUCC>=4, N={len(rural_rates)}):")
print(f"  Mean: {np.mean(rural_rates):.1f}/100K")
print(f"  Ratio: {np.mean(rural_rates)/nat_avg:.2f}x")
above_r = sum(1 for r in rural_rates if r > nat_avg)
print(f"  Above national avg: {above_r}/{len(rural_rates)} ({above_r/len(rural_rates)*100:.0f}%)")
top10_r = sum(1 for r in rural_rates if r > p90)
print(f"  In top 10%: {top10_r}/{len(rural_rates)} ({top10_r/len(rural_rates)*100:.0f}%)")
t2, p2 = stats.ttest_ind(rural_rates, valid.values)
print(f"  T-test vs all counties: t={t2:.2f}, p={p2:.6f}")

print()
print(f"URBAN smelter counties (RUCC<4, N={len(urban_rates)}):")
print(f"  Mean: {np.mean(urban_rates):.1f}/100K")
print(f"  Ratio: {np.mean(urban_rates)/nat_avg:.2f}x")
above_u = sum(1 for r in urban_rates if r > nat_avg)
print(f"  Above national avg: {above_u}/{len(urban_rates)} ({above_u/len(urban_rates)*100:.0f}%)")
t3, p3 = stats.ttest_ind(urban_rates, valid.values)
print(f"  T-test vs all counties: t={t3:.2f}, p={p3:.6f}")

print()
print("RURAL vs URBAN mine/smelter comparison:")
t4, p4 = stats.ttest_ind(rural_rates, urban_rates)
print(f"  Rural mean: {np.mean(rural_rates):.1f} vs Urban mean: {np.mean(urban_rates):.1f}")
print(f"  Difference: {np.mean(rural_rates) - np.mean(urban_rates):.1f}/100K")
print(f"  T-test: t={t4:.2f}, p={p4:.6f}")
