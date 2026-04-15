import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# Output directory
# ============================================================
outdir = "cut_decision_plots"
os.makedirs(outdir, exist_ok=True)

# ============================================================
# Read input data
# ============================================================
df = pd.read_parquet("ttbar_SemiLeptonic_bdt_score_pdgid_weights.parquet")

is_signal = (df["PDG_ID"] == 0)
is_background = (df["PDG_ID"] != 0)

sig_scores = df.loc[is_signal, "BDT_score"].values
bkg_scores = df.loc[is_background, "BDT_score"].values

sig_weights = df.loc[is_signal, "weights"].values
bkg_weights = df.loc[is_background, "weights"].values

# ============================================================
# BDT thresholds
# ============================================================
bdt_cuts = np.linspace(0.0, 1.0, 1001)

# ============================================================
# Unweighted efficiencies
# ============================================================
N_sig = len(sig_scores)
N_bkg = len(bkg_scores)


sig_eff = np.array([
    np.sum(sig_scores >= cut) / N_sig
    for cut in bdt_cuts
])

bkg_rej = np.array([
    1.0 - np.sum(bkg_scores >= cut) / N_bkg
    for cut in bdt_cuts
])

# ============================================================
# Weighted S, B, significance
# ============================================================

S = np.array([
    np.sum(sig_weights[sig_scores >= cut])
    for cut in bdt_cuts
])

B = np.array([
    np.sum(bkg_weights[bkg_scores >= cut])
    for cut in bdt_cuts
])

with np.errstate(divide="ignore", invalid="ignore"):
    Z = np.where(
        (S + B) > 0,
        S/ np.sqrt(S + B),
        0.0
    )


# for the background significance 
B0 = np.sum(bkg_weights)
with np.errstate(divide="ignore", invalid="ignore"):
    Z_bkg = np.divide(
            (B0-B),
            np.sqrt(B0),
            out=np.zeros_like(B),
            where= B0 > 0
    )

table_Z = pd.DataFrame({
    "BDT_cut":bdt_cuts,
    "Signal_eff": sig_eff,
    "bkg_rej":bkg_rej,
    "bkg_sig":Z_bkg
    })

#pd.set_option("display.max_rows", None)
print("\nBDT scan: yields and background rejection")
print("=" * 90)
print(table_Z.round(4))
print("=" * 90)
# ============================================================
# Significance uncertainty (Poisson propagation)
# ============================================================

with np.errstate(divide="ignore", invalid="ignore"):
    Z_err = np.where(
        (S > 0) & (S + B > 0),
        np.sqrt(S * (S + 2 * B)**2 + S**2 * B) / (2 * (S + B)**(3/2)),
        0.0
    )

# ============================================================
# Balanced working point
# ============================================================
idx_balanced = np.argmin(np.abs(sig_eff - bkg_rej))

balanced_cut = bdt_cuts[idx_balanced]
balanced_eff = sig_eff[idx_balanced]
balanced_rej = bkg_rej[idx_balanced]
balanced_Z = Z[idx_balanced]
balanced_Zerr = Z_err[idx_balanced]

table_df_0 = pd.DataFrame(
    {
        "BDT_cut": [balanced_cut],
        "Signal_eff": [balanced_eff],
        "Background_rej": [balanced_rej],
        "Significance": [balanced_Z],
        #"Significance_err": [balanced_Zerr],
    }
)
print(table_df_0.to_string(index=False))

# ============================================================
# All tighter working points
# ============================================================
tighter_indices = np.where(bkg_rej > balanced_rej)[0]

# ============================================================
# MANUAL selection of tighter WP
# ============================================================
MANUAL_TIGHTER_INDEX = 35  # <-- CHANGE THIS

idx_tighter = tighter_indices[MANUAL_TIGHTER_INDEX]

tighter_cut = bdt_cuts[idx_tighter]
tighter_eff = sig_eff[idx_tighter]
tighter_rej = bkg_rej[idx_tighter]
tighter_Z = Z[idx_tighter]
tighter_Zerr = Z_err[idx_tighter]

# ============================================================
# Print table for ALL tighter points
# ============================================================
print("\nTIGHTER WORKING POINTS (Bkg rejection > balanced WP)")
print("=" * 95)
print(f"{'Idx':<6} {'BDT cut':<10} {'Sig eff':<10} {'Bkg rej':<10} {'Significance':<14} {'Error':<10}")
print("-" * 95)

for i, idx in enumerate(tighter_indices):
    print(
        f"{i:<6} "
        f"{bdt_cuts[idx]:<10.3f} "
        f"{sig_eff[idx]:<10.6f} "
        f"{bkg_rej[idx]:<10.6f} "
        f"{Z[idx]:<14.2f} "
        #f"{Z_err[idx]:<10.2f}"
    )

print("=" * 95)

# ============================================================
# Print selected tighter WP
# ============================================================
print("\nSELECTED TIGHTER WORKING POINT (MANUAL)")
print("=" * 60)
print(f"Table index       : {MANUAL_TIGHTER_INDEX}")
print(f"BDT cut           : {tighter_cut:.3f}")
print(f"Signal efficiency : {tighter_eff:.6f}")
print(f"Background rej.   : {tighter_rej:.6f}")
print(f"Significance      : {tighter_Z:.2f} ± {tighter_Zerr:.2f}")
print("=" * 60)

# ============================================================
# Save table to CSV
# ============================================================
table_df = pd.DataFrame({
    "BDT_cut": bdt_cuts[tighter_indices],
    "Signal_eff": sig_eff[tighter_indices],
    "Background_rej": bkg_rej[tighter_indices],
    "Significance": Z[tighter_indices],
    #"Significance_err": Z_err[tighter_indices],
})

table_df.to_csv(f"{outdir}/tighter_working_points.csv", index=False)

# ============================================================
# Combined plot: top (eff/rej), bottom (significance)
# ============================================================
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1,
    figsize=(8, 9),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1]}
)

# ---------- Top panel ----------
ax_top.plot(bdt_cuts, sig_eff, linewidth=2, label="Signal efficiency")
ax_top.plot(bdt_cuts, bkg_rej, linewidth=2, label="Background rejection")

ax_top.axvline(
    balanced_cut,
    linestyle="--",
    linewidth=1.5,
    label=f"Balanced WP (BDT = {balanced_cut:.3f})"
)

'''
ax_top.axvline(
    tighter_cut,
    linestyle=":",
    linewidth=1.5,
    label=f"Tighter WP (BDT = {tighter_cut:.3f})"
)
'''

ax_top.set_ylabel("Efficiency and Rejection")
ax_top.set_ylim(-0.05, 1.05)
ax_top.grid(True, linestyle="--", alpha=0.5)
ax_top.legend()

# ---------- Bottom panel ----------
ax_bot.plot(
    bdt_cuts,
    Z,
    linewidth=2,
    label=r"$S/\sqrt{S+B}$"
)

ax_bot.axvline(
    balanced_cut,
    linestyle="--",
    linewidth=1.5,
    label=f"Balanced WP (BDT = {balanced_cut:.3f}, Z = {balanced_Z:.3f})"
)

'''
ax_bot.axvline(
    tighter_cut,
    linestyle=":",
    linewidth=1.5,
    label=f"Tighter WP (BDT = {tighter_cut:.3f}, Z = {tighter_Z:.3f})"
)
'''
ax_bot.set_xlabel("BDT cut")
ax_bot.set_ylabel(r"$S/\sqrt{S+B}$")
ax_bot.grid(True, linestyle="--", alpha=0.5)
ax_bot.legend()

plt.tight_layout()
plt.savefig(f"{outdir}/bdt_performance_combined.png", dpi=300)
plt.savefig(f"{outdir}/bdt_performance_combined.pdf")
plt.show()
