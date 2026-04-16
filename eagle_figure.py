"""Publication-quality figure: eagle bone Pb vs. veteran suicide, by state."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

panel = pd.read_csv("eagle_soil_mining_suicide_state_panel.csv")
panel = panel.dropna(subset=["mean_femur_Pb", "vet_suicide_rate_2023", "n_femur"])

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

# Panel A: scatter — mean eagle bone Pb vs vet suicide
ax = axes[0]
sizes = 18 * np.sqrt(panel["n_femur"])
sc = ax.scatter(panel["mean_femur_Pb"], panel["vet_suicide_rate_2023"],
                s=sizes, alpha=0.55, edgecolor="k", linewidth=0.4, color="#c0392b")
# Label notable states
for st in ["Utah", "Nevada", "Montana", "Wyoming", "Idaho", "South Dakota",
           "West Virginia", "Florida", "Kentucky", "Washington", "California", "Virginia"]:
    r = panel[panel["State"] == st]
    if not r.empty:
        ax.annotate(st, (r["mean_femur_Pb"].iloc[0], r["vet_suicide_rate_2023"].iloc[0]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
# Weighted regression line
w = np.sqrt(panel["n_femur"])
coef = np.polyfit(panel["mean_femur_Pb"], panel["vet_suicide_rate_2023"], 1, w=w)
xline = np.linspace(panel["mean_femur_Pb"].min(), panel["mean_femur_Pb"].max(), 100)
ax.plot(xline, np.polyval(coef, xline), "k--", lw=1, alpha=0.7)

r, p = stats.pearsonr(panel["mean_femur_Pb"], panel["vet_suicide_rate_2023"])
ax.text(0.03, 0.94, f"Pearson r = {r:+.2f}\np = {p:.3f}\nN = {len(panel)} states",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.85))
ax.set_xlabel("Mean eagle femur lead (µg/g DW)", fontsize=10)
ax.set_ylabel("Veteran male suicide rate, 2023 (per 100K)", fontsize=10)
ax.set_title("A. Eagle bone lead predicts state-level veteran suicide",
             fontsize=10, loc="left")
ax.grid(True, alpha=0.3, linestyle="--")

# Panel B: pct_chronic (% eagles with chronic Pb poisoning) vs vet suicide
ax = axes[1]
d = panel.dropna(subset=["pct_chronic", "vet_suicide_rate_2023"])
ax.scatter(d["pct_chronic"] * 100, d["vet_suicide_rate_2023"],
           s=18 * np.sqrt(d["n_femur"]), alpha=0.55, edgecolor="k",
           linewidth=0.4, color="#2c3e50")
for st in ["Utah", "Nevada", "Montana", "Wyoming", "Idaho", "South Dakota",
           "West Virginia", "Florida", "Kentucky", "Washington", "California", "Virginia"]:
    r_ = d[d["State"] == st]
    if not r_.empty:
        ax.annotate(st, (r_["pct_chronic"].iloc[0] * 100, r_["vet_suicide_rate_2023"].iloc[0]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
coef = np.polyfit(d["pct_chronic"] * 100, d["vet_suicide_rate_2023"], 1,
                  w=np.sqrt(d["n_femur"]))
xline = np.linspace((d["pct_chronic"]*100).min(), (d["pct_chronic"]*100).max(), 100)
ax.plot(xline, np.polyval(coef, xline), "k--", lw=1, alpha=0.7)

r, p = stats.pearsonr(d["pct_chronic"], d["vet_suicide_rate_2023"])
ax.text(0.03, 0.94, f"Pearson r = {r:+.2f}\np = {p:.3f}\nN = {len(d)} states",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.85))
ax.set_xlabel("% eagles with chronic Pb poisoning (≥10 µg/g DW)", fontsize=10)
ax.set_ylabel("Veteran male suicide rate, 2023 (per 100K)", fontsize=10)
ax.set_title("B. % of eagles chronically poisoned tracks veteran suicide",
             fontsize=10, loc="left")
ax.grid(True, alpha=0.3, linestyle="--")

# Size-legend note
fig.text(0.5, -0.01, "Marker area ∝ √(eagle femur samples per state) (Slabe et al. 2022, USGS)",
         ha="center", fontsize=8, style="italic")
plt.tight_layout()
plt.savefig("figure_eagle_vet_suicide.png", dpi=200, bbox_inches="tight")
plt.savefig("figure_eagle_vet_suicide.pdf", bbox_inches="tight")
print("Saved: figure_eagle_vet_suicide.png / .pdf")
