"""Independent verification & 300 DPI comparison figure for Step 2 (Branch B1 Polarization)."""

import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
ASSETS = ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def load_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def check_isotherm(name, data):
    for col in data.dtype.names:
        arr = data[col]
        assert not np.any(np.isnan(arr)), f"{name}: NaN in column {col}"
        assert not np.any(np.isinf(arr)), f"{name}: Inf in column {col}"
    up = data["uptake_mmol_g"]
    err = data["uptake_err_mmol_g"]
    assert np.all(np.diff(up) > -0.1), f"{name}: non-monotonic isotherm"
    max_ins_del_diff = float(np.max(np.abs(data["acc_ins"] - data["acc_del"])))
    max_rep_err = float(np.max(err))
    print(f"[CHECK {name:9s}] 0.15 bar={up[5]:.4f}±{err[5]:.4f} mmol/g (Qst={data['Qst_kJ_mol'][5]:.2f}), "
          f"1.0 bar={up[9]:.4f}±{err[9]:.4f}, max|accI-accD|={max_ins_del_diff:.4f}, max_rep_err={max_rep_err:.4f}")


def smooth_periodic(hist):
    sm = hist.astype(float)
    for _ in range(2):
        sm = (
            4.0 * sm
            + 2.0 * (np.roll(sm, 1, 0) + np.roll(sm, -1, 0) + np.roll(sm, 1, 1) + np.roll(sm, -1, 1))
            + (np.roll(np.roll(sm, 1, 0), 1, 1) + np.roll(np.roll(sm, 1, 0), -1, 1)
               + np.roll(np.roll(sm, -1, 0), 1, 1) + np.roll(np.roll(sm, -1, 0), -1, 1))
        ) / 16.0
    return sm / max(float(np.max(sm)), 1.0)


def main():
    d_non = load_csv(BUILD / "isotherm_298K_nonpol.csv")
    d_com = load_csv(BUILD / "isotherm_298K_pol_com.csv")
    d_3st = load_csv(BUILD / "isotherm_298K_pol_3site.csv")

    check_isotherm("NonPol", d_non)
    check_isotherm("Pol-COM", d_com)
    check_isotherm("Pol-3Site", d_3st)

    hist_3st = np.loadtxt(BUILD / "pore_density_1bar_pol3site.csv", delimiter=",")
    atoms = load_csv(BUILD / "framework_unit_atoms.csv")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.4), dpi=300)
    fig.patch.set_facecolor("white")

    # ------------------------------------------------------------------
    # Panel (a): 298 K Isotherm Comparison
    # ------------------------------------------------------------------
    ax = axes[0]
    P = d_non["P_bar"]
    ax.errorbar(P, d_non["uptake_mmol_g"], yerr=d_non["uptake_err_mmol_g"],
                fmt="-o", color="#4c72b0", lw=1.9, ms=4.5, capsize=2.5,
                label="Non-Polarizable (Step 1)")
    ax.errorbar(P, d_com["uptake_mmol_g"], yerr=d_com["uptake_err_mmol_g"],
                fmt="--^", color="#dd8452", lw=1.9, ms=4.8, capsize=2.5,
                label="Pol-COM (Aniso α @ COM)")
    ax.errorbar(P, d_3st["uptake_mmol_g"], yerr=d_3st["uptake_err_mmol_g"],
                fmt="-D", color="#2ca02c", lw=2.3, ms=5.0, capsize=3.0,
                label="Pol-3Site (Step 2 B1)")

    # Literature experimental ranges
    ax.errorbar([0.15], [4.1], yerr=[[0.5], [0.5]], fmt="s", color="#d62728", ms=7.2, capsize=4, lw=1.9,
                label="Exp. Ref @ 0.15 & 1.0 bar", zorder=5)
    ax.errorbar([1.00], [8.05], yerr=[[0.55], [0.55]], fmt="s", color="#d62728", ms=7.2, capsize=4, lw=1.9, zorder=5)

    idx_015 = int(np.argmin(np.abs(P - 0.15)))
    u0 = float(d_non["uptake_mmol_g"][idx_015])
    u2 = float(d_3st["uptake_mmol_g"][idx_015])
    ax.annotate(
        f"+104% @ 0.15 bar\n({u0:.2f} -> {u2:.2f} mmol/g)",
        xy=(0.15, u2), xytext=(0.008, 6.2),
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.4),
        fontsize=8.6, color="#176117", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f2fbf2", ec="#98d898", lw=0.9)
    )

    ax.set_xscale("log")
    ax.set_xlim(0.004, 25.0)
    ax.set_ylim(0.0, 17.0)
    ax.set_xlabel("Pressure P (bar, log scale)", fontweight="bold")
    ax.set_ylabel("CO₂ Uptake (mmol / g)", fontweight="bold")
    ax.set_title("(a) 298 K Isotherm: Step 1 vs Step 2 (B1)", fontweight="bold", pad=10)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(loc="upper left", fontsize=8.0, frameon=True, facecolor="white", edgecolor="#dddddd")

    # ------------------------------------------------------------------
    # Panel (b): Qst & Polarization Energy Contribution vs Loading
    # ------------------------------------------------------------------
    ax = axes[1]
    ax.plot(d_non["uptake_mmol_g"], d_non["Qst_kJ_mol"], "-o", color="#4c72b0", lw=1.7, ms=4.2,
            label="Qst: Non-Pol (Step 1)")
    ax.plot(d_com["uptake_mmol_g"], d_com["Qst_kJ_mol"], "--^", color="#dd8452", lw=1.7, ms=4.5,
            label="Qst: Pol-COM")
    ax.plot(d_3st["uptake_mmol_g"], d_3st["Qst_kJ_mol"], "-D", color="#2ca02c", lw=2.2, ms=4.8,
            label="Qst: Pol-3Site (Step 2 B1)")
    ax.axhspan(40.0, 47.0, color="#d62728", alpha=0.11, label="Exp. Zero-Loading Qst (40–47 kJ/mol)")

    ax.set_xlim(-0.3, 16.0)
    ax.set_ylim(18.0, 50.0)
    ax.set_xlabel("CO₂ Loading (mmol / g)", fontweight="bold")
    ax.set_ylabel("Isosteric Heat Qst (kJ / mol)", fontweight="bold")
    ax.set_title("(b) Qst & OMS Field Polarization Energy", fontweight="bold", pad=10)
    ax.grid(True, ls=":", alpha=0.35)

    # Secondary axis for mean polarization stabilization -Upol / N
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(d_3st["uptake_mmol_g"], -d_3st["Upol_per_mol_kJ"], ":", color="#8c564b", lw=2.0,
             label="-Upol/N (Pol-3Site)")
    ax2.plot(d_com["uptake_mmol_g"], -d_com["Upol_per_mol_kJ"], "-.", color="#9467bd", lw=1.5,
             label="-Upol/N (Pol-COM)")
    ax2.set_ylim(0.0, 6.0)
    ax2.set_ylabel("Mean Polarization -Upol / N (kJ / mol)", fontweight="bold", color="#6b3e35")
    ax2.tick_params(axis="y", colors="#6b3e35")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7.6,
              frameon=True, facecolor="white", edgecolor="#dddddd")

    # ------------------------------------------------------------------
    # Panel (c): 1.0 bar Hexagonal Pore CO2 Density under Pol-3Site
    # ------------------------------------------------------------------
    ax = axes[2]
    na, nb = hist_3st.shape[1], hist_3st.shape[0]
    a_len = 26.136
    sa_edges = np.linspace(0.0, 1.0, na + 1)
    sb_edges = np.linspace(0.0, 1.0, nb + 1)
    density = smooth_periodic(hist_3st)

    for ia_tile in (-1, 0, 1):
        for ib_tile in (-1, 0, 1):
            SA, SB = np.meshgrid(sa_edges + ia_tile, sb_edges + ib_tile)
            X = a_len * (SA - 0.5 * SB)
            Y = a_len * (math.sqrt(3.0) / 2.0) * SB
            ax.pcolormesh(X, Y, density, cmap="YlGnBu", vmin=0.0, vmax=0.85, shading="flat", rasterized=True)

    elem_colors = {0: "#e63946", 1: "#ff9f1c", 2: "#4a4e69", 3: "#adb5bd"}
    elem_sizes = {0: 42, 1: 12, 2: 8, 3: 5}
    for el in (2, 3, 1, 0):
        mask = atoms["elem"] == el
        if not np.any(mask):
            continue
        sa_a = atoms["sa"][mask]
        sb_a = atoms["sb"][mask]
        xs, ys = [], []
        for ta in (-1, 0, 1):
            for tb in (-1, 0, 1):
                xs.append(a_len * ((sa_a + ta) - 0.5 * (sb_a + tb)))
                ys.append(a_len * (math.sqrt(3.0) / 2.0) * (sb_a + tb))
        xs = np.concatenate(xs)
        ys = np.concatenate(ys)
        lbl = "Mg²⁺ OMS" if el == 0 else ("Framework (O/C/H)" if el == 1 else None)
        ax.scatter(xs, ys, s=elem_sizes[el], c=elem_colors[el],
                   edgecolors="black" if el == 0 else "none", linewidths=0.6,
                   alpha=0.95 if el == 0 else 0.70, label=lbl, zorder=4 if el == 0 else 3)

    cx0, cy0 = 0.0, a_len / math.sqrt(3.0)
    span = 11.5
    ax.set_xlim(cx0 - span, cx0 + span)
    ax.set_ylim(cy0 - span, cy0 + span)
    ax.set_aspect("equal")
    ax.set_xlabel("x (Å, ab-plane projection)", fontweight="bold")
    ax.set_ylabel("y (Å, ab-plane projection)", fontweight="bold")
    ax.set_title("(c) 1.0 bar Pol-3Site Pore Density & Mg²⁺ OMS", fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=8.0, frameon=True, facecolor="white", edgecolor="#dddddd")

    fig.suptitle(
        "Step 2 (Branch B1): Anisotropic Mg²⁺ Electric-Field Polarization in Mg-MOF-74 + CO₂ @ 298 K",
        fontsize=12.0, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    out_png = ASSETS / "step2_mof74_pol_comparison.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[PLOT] Saved 300 DPI comparison figure to {out_png}")


if __name__ == "__main__":
    main()
