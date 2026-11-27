"""Figures for the Auto-MLIP / GPU-CBMC Nested MC study (legacy 4-figure & JCTC 5-figure modes).

All plotted numbers are read strictly from summary JSONs (`phase2_phase3_summary.json` or
`jctc_summary.json`), `arch_inputs_mgmof74_r2scan.npz`, and `data/exp_isotherms/*.csv`.
No simulation results are hardcoded.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "docs" / "results" / "phase2_phase3_summary.json"
JCTC_SUMMARY = ROOT / "docs" / "results" / "jctc_summary.json"
PRESUB_SUMMARY = ROOT / "build" / "jctc_presubmission" / "matched_budget_summary.json"
PRESUB_SUMMARY_PUBLISHED = ROOT / "docs" / "results" / "presubmission" / "matched_budget_summary.json"
PRESUB_TRACES = ROOT / "build" / "jctc_presubmission" / "matched_budget_traces.npz"
ARCH_NPZ = ROOT / "docs" / "results" / "arch_inputs_mgmof74_r2scan.npz"
EXP = ROOT / "data" / "exp_isotherms"
OUT = ROOT / "assets"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "axes.titleweight": "normal",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 8,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "mathtext.fontset": "dejavusans",
})

C = {
    "exp1": "#4B5563", "exp2": "#9CA3AF",
    "auto": "#1F4E9C", "hl": "#2A9D8F", "remc": "#0F766E", "grid": "#7FB3D5",
    "oneb": "#6366F1", "macedac": "#C2410C", "h2o": "#0284C7",
    "mp0d3": "#B83227", "mp0": "#7D8B3A",
    "direct": "#222222", "uff": "#8E6CB0", "mercado": "#D98E04",
    "pass": "#2E7D32", "fail": "#E08E0B", "ink": "#1F2937", "muted": "#6B7280",
}
MG_SUPER_MASS = 4368.744  # g/mol, Mg-MOF-74 1x1x2 supercell (from run JSON fw_mass_gmol)


def panel_label(ax, s, x=-0.13, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")


def load_exp(fn):
    xs, ys = [], []
    with (EXP / fn).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            xs.append(float(row["P_bar"]))
            ys.append(float(row["uptake_mmol_g"]))
    return np.array(xs), np.array(ys)


def pts(s, key):
    return sorted(s["runs"][key]["points"], key=lambda p: p["P_bar"])


def series(s, key):
    p = pts(s, key)
    return (np.array([x["P_bar"] for x in p]), np.array([x["uptake_mmol_g"] for x in p]),
            np.array([x["uptake_sem_mmol_g"] for x in p]))


def mean_trace_mmol(pt, mass=MG_SUPER_MASS):
    tr = np.asarray(pt["box_N_trace"], dtype=float)
    return tr.mean(axis=1) * 1000.0 / mass


# --------------------------------------------------------------------------------------------
# Figure 1: method overview
# --------------------------------------------------------------------------------------------

def _box(ax, x, y, w, h, title, body, fc, ec, title_size=9.5, body_size=8.3, body_lines=1.45):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1.0))
    ax.add_patch(Rectangle((x, y + h - 4.2), w, 4.2, fc=ec, ec=ec, lw=0))
    ax.text(x + 1.2, y + h - 2.1, title, color="white", fontsize=title_size, fontweight="bold", va="center")
    ax.text(x + 1.2, y + h - 5.6, body, color=C["ink"], fontsize=body_size, va="top", linespacing=body_lines)


def _arrow(ax, p0, p1, color="#374151", rad=0.0, lw=1.2, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>,head_length=5,head_width=3", lw=lw, color=color, ls=ls,
                                 shrinkA=0, shrinkB=0))


def fig_architecture(s):
    arch = np.load(ARCH_NPZ)
    fig = plt.figure(figsize=(13.6, 8.4))

    # ---------------- (a) surrogate construction ----------------
    axa = fig.add_axes([0.015, 0.515, 0.635, 0.445])
    axa.set_xlim(0, 100); axa.set_ylim(0, 44); axa.axis("off")
    axa.text(0, 43.2, "a", fontsize=13, fontweight="bold", va="top")
    axa.text(3.2, 43.0, "One-time surrogate construction  (per framework, guest and target MLIP)",
             fontsize=10.5, va="top", color=C["ink"])

    blue_fc, blue_ec = "#EEF3FA", "#2F5D8A"
    _box(axa, 0.5, 12, 22.5, 23, "Inputs",
         "Rigid framework (CIF,\n  any space group)\nRigid guest (\u2264 12 sites)\nTarget MLIP:\n  MACE, with or without D3\nT, list of pressures", blue_fc, blue_ec)
    _box(axa, 25.8, 12, 22.5, 23, "Typing",
         "Framework atoms grouped by\n  element + neighbour\n  signature (6\u20138 classes)\nGuest atoms grouped into\n  symmetry channels\nNo hand-assigned\n  force-field types", blue_fc, blue_ec)
    _box(axa, 51.1, 12, 22.5, 23, "Target sampling",
         "1,200 single-guest poses,\n  stratified by distance\n  to the framework\n192 in-pore dimers\nBatched MACE (\u00b1 D3)\n  energies", blue_fc, blue_ec)
    _box(axa, 76.4, 12, 23.1, 23, "Constrained fit",
         "1B: radial basis per\n  (channel, class) + $r^{-12}$ core\n  + neutral Ewald modes\n2B: scaled LJ/Coulomb\n  + COM residual table\n85/15 hold-out check", blue_fc, blue_ec)
    for x0 in (23.0, 48.3, 73.6):
        _arrow(axa, (x0 + 0.2, 23.5), (x0 + 2.6, 23.5))
    d0 = s["distillation"]["5e543d0520048f84"]
    m2_d0 = d0.get("mae_2b_deployed_holdout_kJ_mol")
    if m2_d0 is not None:
        m2_d0_txt = f"2B deployed hold-out MAE = {m2_d0:.2f} kJ/mol"
    else:
        m2_d0_txt = f"2B in-sample fit MAE = {d0['mae_2b_kJ_mol']:.2f} kJ/mol"
    axa.text(50, 7.6, "Cached by SHA-256 key; 56\u2013210 s per (framework, guest, MLIP) on one RTX 3090",
             ha="center", fontsize=8.3, color=C["muted"], style="italic")
    axa.text(50, 3.6, f"Example on the right (Mg-MOF-74, CO$_2$, r2SCAN): 15 % hold-out 1B R$^2$ = {d0['r2_1b']:.3f}, "
                      f"1B MAE = {d0['mae_1b_kJ_mol']:.2f} kJ/mol, {m2_d0_txt}",
             ha="center", fontsize=8.0, color=C["muted"], style="italic")

    # real outputs: 1B grid slice and 2B residual table
    axg = fig.add_axes([0.665, 0.545, 0.17, 0.37])
    Z = np.clip(arch["o_min_kJ"], -45.0, 5.0)
    n = Z.shape[0]
    a = float(arch["a"]); g = np.deg2rad(float(arch["gamma_deg"]))
    A = np.array([a, 0.0]); B = np.array([a * np.cos(g), a * np.sin(g)])
    fa = (np.arange(n + 1) / n)
    FA, FB = np.meshgrid(fa, fa)
    for i in (0, 1):
        for j in (0, 1):
            X = (FA + i) * A[0] + (FB + j) * B[0]
            Y = (FA + i) * A[1] + (FB + j) * B[1]
            im = axg.pcolormesh(X, Y, Z, cmap="YlGnBu_r", vmin=-45, vmax=5, shading="flat", rasterized=True)
    cx, cy = 0.5 * (A + B) + 0.5 * (A + B)
    half = 15.0
    axg.set_xlim(cx - half, cx + half); axg.set_ylim(cy - half, cy + half)
    axg.set_aspect("equal"); axg.set_xticks([]); axg.set_yticks([])
    for sp in axg.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.6)
    axg.plot([cx - half + 2, cx - half + 12], [cy - half + 2] * 2, color="#111827", lw=2.2)
    axg.text(cx - half + 7, cy - half + 3.2, "10 \u00c5", color="#111827", ha="center", fontsize=8, fontweight="bold")
    axg.set_title("1B grid, Mg-MOF-74 (r2SCAN)\nO(CO$_2$) channel, minimum along c", fontsize=8.3, color=C["ink"])
    cax = fig.add_axes([0.665, 0.522, 0.17, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("U$_{1B}$ (kJ mol$^{-1}$)", fontsize=7.8, labelpad=1)
    cb.ax.tick_params(labelsize=7)

    axl = fig.add_axes([0.875, 0.61, 0.11, 0.27])
    lut = arch["lut_K"] * 8.31446261815324e-3
    r = np.linspace(float(arch["lut_rmin"]), float(arch["lut_rmax"]), len(lut))
    axl.axhline(0, color="#9CA3AF", lw=0.6)
    axl.plot(r, lut, color=blue_ec, lw=1.4)
    axl.set_xlabel("r$_{COM}$ (\u00c5)", fontsize=8)
    axl.set_ylabel("\u0394U$_{2B}$ (kJ mol$^{-1}$)", fontsize=8)
    axl.tick_params(labelsize=7)
    axl.set_title("2B residual table", fontsize=8.3, color=C["ink"])

    # ---------------- (b) nested MC cycle ----------------
    axb = fig.add_axes([0.015, 0.03, 0.635, 0.45])
    axb.set_xlim(0, 100); axb.set_ylim(0, 46); axb.axis("off")
    axb.text(0, 45.4, "b", fontsize=13, fontweight="bold", va="top")
    axb.text(3.2, 45.2, "One outer cycle of the nested Monte Carlo chain  (repeated for every box)",
             fontsize=10.5, va="top", color=C["ink"])

    _box(axb, 0.5, 14.5, 19.5, 17, "Accepted state",
         "$x=(N,\\{\\mathbf{r}_i,\\mathbf{q}_i\\})$\n4 independent boxes\nper pressure", "#F3F4F6", "#4B5563")
    _box(axb, 27.5, 24.5, 39.5, 17.5, "GPU inner burst  (one CUDA block per box)",
         "k = 16 moves on the shifted surrogate $U_{\\rm ref}(x)+\\Delta\\varepsilon\\,N$\n"
         "translate, rotate, insert, delete (25 % each)\n"
         "insert/delete: K = 128 Rosenbluth trials, one per thread\n"
         "1B tricubic grids + 2B table in device memory", "#FDF5EA", "#B5651D")
    _box(axb, 74.5, 14.5, 25.0, 17, "Target energy",
         "$U_{\\rm MLIP}(x')$ for all boxes\nin one batched call\n(MACE, optionally + D3)", "#EEF3FA", "#2F5D8A")
    _box(axb, 27.5, 1.0, 39.5, 17.0, "Outer Metropolis test", "", "#F1EEF8", "#5B4B8A")
    axb.text(47.25, 9.4,
             r"$\alpha=\min\left\{1,\ \exp\left(-\beta\,[\,\Delta U_{\rm MLIP}-\Delta U_{\rm ref}-\Delta\varepsilon\,\Delta N\,]\right)\right\}$",
             ha="center", va="center", fontsize=10.5, color=C["ink"])
    axb.text(47.25, 4.0, "accept: commit x\u2032      reject: restore x on the device",
             ha="center", va="center", fontsize=8.3, color=C["ink"])

    _arrow(axb, (11.0, 31.6), (27.3, 35.5), rad=-0.25)
    axb.text(13.5, 37.0, "$x$", fontsize=9.5)
    _arrow(axb, (67.2, 33.0), (80.0, 31.7), rad=-0.25)
    axb.text(70.5, 36.0, "candidate $x'$", fontsize=8.5)
    _arrow(axb, (86.5, 14.3), (67.2, 9.5), rad=-0.25)
    axb.text(77.5, 7.2, "$U_{\\rm MLIP}(x')$", fontsize=8.5)
    _arrow(axb, (27.3, 9.5), (10.5, 14.3), rad=-0.25)

    axb.text(0.5, 8.4, "$\\Delta\\varepsilon$: per-molecule MLIP\u2013surrogate offset,\nset once per pressure (median over\n"
                       "boxes, then a 40-cycle pilot). The chain\nsamples the MLIP ensemble for any $\\Delta\\varepsilon$;\n"
                       "it only changes efficiency.",
             fontsize=7.2, color=C["muted"], va="top", linespacing=1.3)

    # ---------------- (c) multi-GPU layouts ----------------
    axc = fig.add_axes([0.665, 0.03, 0.325, 0.45])
    axc.set_xlim(0, 100); axc.set_ylim(0, 46); axc.axis("off")
    axc.text(0, 45.4, "c", fontsize=13, fontweight="bold", va="top")
    axc.text(5.5, 45.2, "Multi-GPU execution", fontsize=10.5, va="top", color=C["ink"])
    shades = ["#C7D7EC", "#93B4DA", "#5E8FC6", "#2F5D8A"]

    def gpu_row(y0, label, fill_fn, note):
        axc.text(0, y0 + 13.6, label, fontsize=8.8, fontweight="bold", color=C["ink"])
        for gi in range(4):
            gx = 2 + gi * 24.5
            axc.add_patch(FancyBboxPatch((gx, y0), 22.0, 11.0, boxstyle="round,pad=0,rounding_size=0.8",
                                         fc="white", ec="#6B7280", lw=0.8))
            axc.text(gx + 11.0, y0 + 8.9, f"GPU {gi}", ha="center", fontsize=7.6, color=C["muted"])
            fill_fn(gi, gx, y0)
        axc.text(0, y0 - 2.6, note, fontsize=7.8, color=C["muted"], va="top")

    def fill_pressure(gi, gx, y0):
        for bi in range(4):
            axc.add_patch(Rectangle((gx + 2.0 + bi * 4.6, y0 + 1.8), 3.8, 4.4, fc=shades[gi], ec="none"))
        axc.text(gx + 11.0, y0 + 6.6, f"$P_{gi + 1}$", ha="center", fontsize=8, color=C["ink"])

    def fill_replica(gi, gx, y0):
        for bi in range(2):
            axc.add_patch(Rectangle((gx + 6.4 + bi * 4.8, y0 + 1.8), 3.8, 4.4, fc=shades[2], ec="none"))
        axc.text(gx + 11.0, y0 + 6.6, "$P$", ha="center", fontsize=8, color=C["ink"])

    p1 = s["runs"]["phase3_scale_1gpu_pressure"]["total_wall_s"]; p4 = s["runs"]["phase3_scale_4gpu_pressure"]["total_wall_s"]
    r1 = s["runs"]["phase3_scale_1gpu_replica"]["points"][0]["elapsed_s"]; r4 = s["runs"]["phase3_scale_4gpu_replica"]["points"][0]["elapsed_s"]
    gpu_row(26.0, "Pressure queue: one pressure per GPU (4 boxes each)", fill_pressure,
            f"4 pressures, end-to-end: {p1:.0f} s \u2192 {p4:.0f} s on 4\u00d7RTX 3090 ({p1 / p4:.2f}\u00d7)")
    gpu_row(5.0, "Replica sharding: one pressure, 8 boxes split 2 per GPU", fill_replica,
            f"MC time: {r1:.0f} s \u2192 {r4:.0f} s ({r1 / r4:.2f}\u00d7); chains merged for statistics")

    fig.savefig(OUT / "fig1_architecture.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved fig1_architecture.png")


# --------------------------------------------------------------------------------------------
# Figure 2: Experiment 1 (Mg-MOF-74 + CO2)
# --------------------------------------------------------------------------------------------

def fig_exp1(s):
    fig, axs = plt.subplots(2, 2, figsize=(11.0, 8.6))
    (ax_a, ax_b), (ax_c, ax_d) = axs
    fig.subplots_adjust(hspace=0.62, wspace=0.30)

    # (a) isotherms, log-log
    for fn, col, mk, lab in [("mgmof74_10.1021acs.jpcc.7b08239.Isotherm1.csv", C["exp1"], "o", "Exp. (10.1021/acs.jpcc.7b08239)"),
                             ("mgmof74_10.1039c5dt03137k.Isotherm1.csv", C["exp2"], "s", "Exp. (10.1039/c5dt03137k)")]:
        x, y = load_exp(fn)
        m = (x > 0.004) & (x < 1.3)
        ax_a.plot(x[m], y[m], mk, ms=4.2, mfc="none", mec=col, mew=1.0, ls="none", label=lab)
    for key, col, mk, lab, ls in [("exp1_auto_mlip_r2scan_nod3", C["auto"], "D", "Nested MC, r2SCAN (no D3)", "-"),
                                  ("exp1_grid_only_r2scan", C["grid"], "^", "Surrogate only (1B+2B), r2SCAN", "--"),
                                  ("exp1_auto_mlip_mp0_d3", C["mp0d3"], "v", "Nested MC, MP0 + D3", "-")]:
        p, q, e = series(s, key)
        ax_a.errorbar(p, q, yerr=e, fmt=mk + ls, color=col, ms=5.5, lw=1.4, capsize=2.5, label=lab)
    pm = pts(s, "exp1_mp0_nod3_0p28")[0]
    ax_a.errorbar([pm["P_bar"]], [pm["uptake_mmol_g"]], yerr=[pm["uptake_sem_mmol_g"]], fmt="P", color=C["mp0"],
                  ms=7, label="Nested MC, MP0 (no D3)")
    ax_a.set_xscale("log"); ax_a.set_yscale("log")
    ax_a.set_xlim(0.008, 1.4); ax_a.set_ylim(0.1, 30)
    ax_a.set_xlabel("Pressure (bar)"); ax_a.set_ylabel("CO$_2$ uptake (mmol g$^{-1}$)")
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7.2, handlelength=1.8, columnspacing=1.0)
    ax_a.set_title("Isotherms at 298 K", loc="left")
    panel_label(ax_a, "a")

    # (b) relaxation vs wall time at 0.28 bar
    d_wall = s["runs"]["exp1_auto_mlip_r2scan_nod3"]["distillation"]["distill_wall_s"]
    runs_b = [("exp1_direct_r2scan_0p28", C["direct"], "Direct MLIP-GCMC", 0.0, "-"),
              ("exp1_uff_r2scan_0p28", C["uff"], "UFF proposer", 0.0, "-"),
              ("exp1_mercado_r2scan_0p28", C["mercado"], "Mercado-2016 proposer", 0.0, "-"),
              ("exp1_auto_mlip_r2scan_nod3", C["auto"], "Auto-MLIP proposer, empty start", d_wall, "-"),
              ("exp1_auto_mlip_r2scan_nod3_highload_0p28", C["hl"], "Auto-MLIP proposer, loaded start", d_wall, "-")]
    for key, col, lab, off, ls in runs_b:
        pt = [p for p in s["runs"][key]["points"] if abs(p["P_bar"] - 0.28) < 1e-6][0]
        y = mean_trace_mmol(pt)
        t = off + pt["elapsed_s"] * (np.arange(1, len(y) + 1) / len(y))
        ax_b.plot(t, y, color=col, lw=1.3, ls=ls, label=lab)
        ax_b.plot(t[-1], y[-1], "o", color=col, ms=3.5)
    ax_b.axvspan(0, d_wall, color="#E5E7EB", alpha=0.7, lw=0)
    ax_b.text(d_wall / 2, 7.65, "distillation", ha="center", fontsize=7.2, color=C["muted"], rotation=90, va="top")
    ax_b.axhspan(4.745, 6.937, color="#D1D5DB", alpha=0.35, lw=0)
    ax_b.text(780, 5.05, "experimental range\nat 0.28 bar", fontsize=7.2, color=C["muted"], ha="right")
    ax_b.set_xlim(0, 800); ax_b.set_ylim(0, 8)
    ax_b.set_xlabel("Wall-clock time on one RTX 3090 (s)")
    ax_b.set_ylabel("Box-averaged uptake (mmol g$^{-1}$)")
    ax_b.set_title("Relaxation at 0.28 bar, same target (r2SCAN, no D3)", loc="left")
    ax_b.legend(loc="lower right", fontsize=7.2, bbox_to_anchor=(1.0, 0.04))
    panel_label(ax_b, "b")

    # (c) nested MC minus surrogate-only
    P = [0.012, 0.28, 1.0]
    width = 0.36
    for k, (auto_key, grid_key, col, lab) in enumerate([("exp1_auto_mlip_r2scan_nod3", "exp1_grid_only_r2scan", C["auto"], "r2SCAN (no D3)"),
                                                        ("exp1_auto_mlip_mp0_d3", "exp1_grid_only_mp0_d3", C["mp0d3"], "MP0 + D3")]):
        a = {round(p["P_bar"], 3): p for p in pts(s, auto_key)}
        g = {round(p["P_bar"], 3): p for p in pts(s, grid_key)}
        d = [a[p]["uptake_mmol_g"] - g[p]["uptake_mmol_g"] for p in P]
        e = [np.hypot(a[p]["uptake_sem_mmol_g"], g[p]["uptake_sem_mmol_g"]) for p in P]
        xs = np.arange(3) + (k - 0.5) * width
        ax_c.bar(xs, d, width=width * 0.92, yerr=e, color=col, alpha=0.85, capsize=2.5, label=lab, ecolor="#374151")
        for xi, di, ei in zip(xs, d, e):
            ax_c.text(xi, di + (ei + 0.12 if di >= 0 else -ei - 0.12), f"{di:+.2f}", ha="center",
                      va="bottom" if di >= 0 else "top", fontsize=7.6)
    ax_c.axhline(0, color="#374151", lw=0.8)
    ax_c.set_xticks(np.arange(3)); ax_c.set_xticklabels(["0.012 bar", "0.28 bar", "1.0 bar"])
    ax_c.set_ylabel("q$_{nested}$ \u2212 q$_{surrogate}$ (mmol g$^{-1}$)")
    ax_c.set_ylim(-1.2, 3.2)
    ax_c.set_title("What the outer MLIP correction changes", loc="left")
    ax_c.legend(loc="upper left", fontsize=7.6)
    panel_label(ax_c, "c")

    # (d) target functional at 0.28 bar
    items = [("MP0\n(no D3)", pts(s, "exp1_mp0_nod3_0p28")[0], C["mp0"]),
             ("r2SCAN\n(no D3)", [p for p in pts(s, "exp1_auto_mlip_r2scan_nod3") if abs(p["P_bar"] - 0.28) < 1e-6][0], C["auto"]),
             ("MP0\n+ D3", [p for p in pts(s, "exp1_auto_mlip_mp0_d3") if abs(p["P_bar"] - 0.28) < 1e-6][0], C["mp0d3"])]
    xs = np.arange(len(items))
    ax_d.axhspan(4.745, 6.937, color="#D1D5DB", alpha=0.45, lw=0, label="experiment (two datasets)")
    for xi, (lab, p, col) in zip(xs, items):
        ax_d.bar(xi, p["uptake_mmol_g"], width=0.55, color=col, alpha=0.88, yerr=p["uptake_sem_mmol_g"], capsize=2.5, ecolor="#374151")
        ax_d.text(xi, p["uptake_mmol_g"] + 0.6, f"{p['uptake_mmol_g']:.2f} mmol/g\nQ$_{{st}}$ = {p['Qst_fluc_kJ_mol']:.1f} kJ/mol",
                  ha="center", va="bottom", fontsize=7.8)
    ax_d.set_xticks(xs); ax_d.set_xticklabels([i[0] for i in items])
    ax_d.set_ylabel("CO$_2$ uptake at 0.28 bar (mmol g$^{-1}$)")
    ax_d.set_ylim(0, 23.5)
    ax_d.set_title("Effect of the target MLIP (same sampler)", loc="left")
    ax_d.legend(loc="upper left", fontsize=7.6)
    panel_label(ax_d, "d")

    fig.savefig(OUT / "fig2_exp1_mgmof74.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved fig2_exp1_mgmof74.png")


# --------------------------------------------------------------------------------------------
# Figure 3: Experiment 2 (transferability)
# --------------------------------------------------------------------------------------------

def _iso_panel(ax, exp_specs, sim_specs, xlim, ylim, title, xlog=False):
    for fn, col, mk, lab in exp_specs:
        x, y = load_exp(fn)
        m = (x <= xlim[1] * 1.02) & (x > 0)
        ax.plot(x[m], y[m], mk, ms=3.8, mfc="none", mec=col, mew=0.9, ls="none", label=lab)
    for p, q, e, col, mk, lab in sim_specs:
        ax.errorbar(p, q, yerr=e, fmt=mk + "-", color=col, ms=5.5, lw=1.4, capsize=2.5, label=lab)
    if xlog:
        ax.set_xscale("log")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("Pressure (bar)"); ax.set_ylabel("Uptake (mmol g$^{-1}$)")
    ax.set_title(title, loc="left")


def fig_exp2(s):
    fig, axs = plt.subplots(2, 3, figsize=(14.2, 8.6))
    fig.subplots_adjust(hspace=0.45, wspace=0.62)
    (a, b, c), (d, e, f) = axs
    lab_sim = "Nested MC, r2SCAN (no D3)"

    p, q, er = series(s, "exp2_nimof74_co2")
    _iso_panel(a, [("nimof74_10.1021acs.jpcc.5b03700.Isotherm47.csv", C["exp1"], "o", "Exp. (10.1021/acs.jpcc.5b03700)"),
                   ("nimof74_10.108015422119.2014.884507.Isotherm5.csv", C["exp2"], "s", "Exp. (10.1080/15422119.2014.884507)")],
               [(p, q, er, "#0F766E", "D", lab_sim)], (0, 1.05), (0, 8.3), "Ni-MOF-74 + CO$_2$, 298 K")
    a.legend(loc="lower right", bbox_to_anchor=(1.0, 0.12), fontsize=7.0)
    panel_label(a, "a")

    p, q, er = series(s, "exp2_cubtc_co2")
    _iso_panel(b, [("hkust1_10.1002aic.15837.Isotherm7.csv", C["exp1"], "o", "Exp. (10.1002/aic.15837)"),
                   ("hkust1_10.1007s1045001395272.Isotherm50.csv", C["exp2"], "s", "Exp. (10.1007/s10450-013-9527-2)")],
               [(p, q, er, C["mercado"], "D", lab_sim)], (0, 1.08), (0, 5.0), "Cu-BTC (HKUST-1) + CO$_2$, 298 K")
    b.legend(loc="upper left", fontsize=7.0)
    panel_label(b, "b")

    p1 = pts(s, "exp2_irmof1_co2")[0]
    p5 = pts(s, "exp2_irmof1_co2_5bar")[0]
    _iso_panel(c, [("irmof1_10.1021acs.jctc.5b00093.Isotherm1.csv", C["exp1"], "o", "Exp. (10.1021/acs.jctc.5b00093)"),
                   ("irmof1_10.1038srep12045.Isotherm2.csv", C["exp2"], "s", "Exp. (10.1038/srep12045)")],
               [(np.array([1.0, 5.0]), np.array([p1["uptake_mmol_g"], p5["uptake_mmol_g"]]),
                 np.array([p1["uptake_sem_mmol_g"], p5["uptake_sem_mmol_g"]]), "#6D28D9", "D", lab_sim)],
               (0, 6.6), (0, 7.0), "IRMOF-1 + CO$_2$, 298 K")
    c.legend(loc="upper left", fontsize=7.0)
    panel_label(c, "c")

    # (d) IRMOF-1 + CH4 with MP0+D3, inset single-molecule energies
    p, q, er = series(s, "exp2_irmof1_ch4_mp0d3")
    d.errorbar(p, q, yerr=er, fmt="o-", color=C["mp0d3"], ms=5.5, lw=1.4, capsize=2.5, label="Nested MC, MP0 + D3")
    for pi, qi in zip(p, q):
        d.text(pi, qi + 0.45, f"{qi:.2f}", ha="center", fontsize=7.6)
    d.set_xlim(0, 5.6); d.set_ylim(0, 9.5)
    d.set_xlabel("Pressure (bar)"); d.set_ylabel("CH$_4$ uptake (mmol g$^{-1}$)")
    d.set_title("IRMOF-1 + CH$_4$, 298 K", loc="left")
    d.legend(loc="upper left", fontsize=7.4)
    d.text(0.02, 0.80, "no CH$_4$ reference isotherm\nin data/exp_isotherms", transform=d.transAxes, fontsize=7.0, color=C["muted"], va="top")
    diag = s["irmof1_ch4_mlip_diagnostic"]
    ins = d.inset_axes([0.50, 0.20, 0.47, 0.36])
    xx = np.arange(4); w = 0.26
    ins.bar(xx - w, diag["r2scan_nod3_ase_kJ_mol"], w, color=C["auto"], label="r2SCAN")
    ins.bar(xx, diag["mp0_d3_fast_kJ_mol"], w, color=C["mp0d3"], label="MP0+D3")
    ins.bar(xx + w, diag["mp0_nod3_fast_kJ_mol"], w, color=C["mp0"], label="MP0")
    ins.axhline(0, color="#374151", lw=0.6)
    ins.set_xticks(xx); ins.set_xticklabels([f"{v:.2f}" for v in diag["poses_dmin_A"]], fontsize=6.4)
    ins.set_xlabel("pose, distance to framework (\u00c5)", fontsize=6.4, labelpad=1.5)
    ins.set_ylabel("U$_{int}$ (kJ/mol)", fontsize=6.6, labelpad=1)
    ins.tick_params(labelsize=6.2)
    ins.legend(fontsize=5.8, loc="upper center", ncol=3, handlelength=0.8, columnspacing=0.6)
    ins.set_ylim(-42, 16)
    ins.set_title("single CH$_4$, direct MACE", fontsize=6.8)
    panel_label(d, "d")

    # (e) parity against experiment
    ex = s["experimental_interpolated_mmol_g"]
    rows = []
    for mof, key, P_list, col, mk in [("Mg-MOF-74", "exp1_auto_mlip_r2scan_nod3", [0.012, 0.28, 1.0], C["auto"], "D"),
                                       ("Ni-MOF-74", "exp2_nimof74_co2", [0.012, 0.28, 1.0], "#0F766E", "s"),
                                       ("Cu-BTC", "exp2_cubtc_co2", [0.15, 1.0], C["mercado"], "^"),
                                       ("IRMOF-1", "exp2_irmof1_co2_5bar", [5.0], "#6D28D9", "o")]:
        sim = {round(p["P_bar"], 3): p for p in pts(s, key)}
        for P in P_list:
            vals = [v for v in ex[mof][str(P)].values() if v is not None]
            sp = sim[round(P, 3)]
            rows.append((mof, P, np.mean(vals), min(vals), max(vals), sp["uptake_mmol_g"], sp["uptake_sem_mmol_g"], col, mk, sp["diag_status"]))
    lims = (0.005, 30)
    xx = np.logspace(np.log10(lims[0]), np.log10(lims[1]), 50)
    e.plot(xx, xx, color="#374151", lw=0.8)
    e.plot(xx, 2 * xx, color="#9CA3AF", lw=0.7, ls="--"); e.plot(xx, 0.5 * xx, color="#9CA3AF", lw=0.7, ls="--")
    e.plot(xx, 10 * xx, color="#D1D5DB", lw=0.7, ls=":"); e.plot(xx, 0.1 * xx, color="#D1D5DB", lw=0.7, ls=":")
    seen = set()
    for mof, P, xm, xlo, xhi, ys, ye, col, mk, st in rows:
        e.errorbar(xm, ys, xerr=[[xm - xlo], [xhi - xm]], yerr=ye, fmt=mk, color=col, ms=5.5,
                   mfc=col if st == "pass" else "white", mew=1.1, capsize=1.8, lw=0.9,
                   label=mof if mof not in seen else None)
        seen.add(mof)
        off = {("Mg-MOF-74", 1.0): (6, 2), ("Mg-MOF-74", 0.28): (-34, 4), ("Mg-MOF-74", 0.012): (5, -11),
               ("Ni-MOF-74", 1.0): (6, -4), ("Ni-MOF-74", 0.28): (6, -9), ("Ni-MOF-74", 0.012): (-38, -3),
               ("Cu-BTC", 1.0): (-30, -6), ("Cu-BTC", 0.15): (5, -11), ("IRMOF-1", 5.0): (-30, 3)}.get((mof, P), (4, -9))
        e.annotate(f"{P:g} bar", (xm, ys), xytext=off, textcoords="offset points", fontsize=6.3, color=col)
    e.set_xscale("log"); e.set_yscale("log"); e.set_xlim(*lims); e.set_ylim(*lims)
    e.set_xlabel("Experiment (mmol g$^{-1}$, interpolated)"); e.set_ylabel("Nested MC, r2SCAN (mmol g$^{-1}$)")
    e.set_title("Parity with experiment, CO$_2$", loc="left")
    e.text(0.6, 1.5, "\u00d72", fontsize=7, color="#6B7280"); e.text(0.9, 0.3, "\u00d70.5", fontsize=7, color="#6B7280")
    e.text(0.25, 4.5, "\u00d710", fontsize=7, color="#9CA3AF"); e.text(0.22, 0.031, "\u00d70.1", fontsize=7, color="#9CA3AF")
    leg1 = e.legend(loc="upper left", fontsize=7.0)
    e.add_artist(leg1)
    e.legend(handles=[Line2D([], [], marker="o", color="#374151", ls="none", mfc="#374151", label="all gates passed"),
                      Line2D([], [], marker="o", color="#374151", ls="none", mfc="white", label="not all gates passed")],
             loc="lower right", fontsize=6.8)
    panel_label(e, "e")

    # (f) surrogate fidelity for the caches used in production
    caches = [("5e543d0520048f84", "Mg-MOF-74, CO$_2$\nr2SCAN"), ("e523efe1b8f2e137", "Mg-MOF-74, CO$_2$\nMP0+D3"),
              ("269c6d37da0d1de5", "Mg-MOF-74, CO$_2$\nMP0"), ("c829b9bb903aaded", "Ni-MOF-74, CO$_2$\nr2SCAN"),
              ("9be1cad17f79fedb", "Cu-BTC, CO$_2$\nr2SCAN"), ("0ca4d69d7bafef2b", "IRMOF-1, CO$_2$\nr2SCAN"),
              ("5134add5a31071bc", "IRMOF-1, CH$_4$\nMP0+D3")]
    y = np.arange(len(caches))[::-1]
    m1 = [s["distillation"][c]["mae_1b_kJ_mol"] for c, _ in caches]
    has_deployed_2b = all(s["distillation"][c].get("mae_2b_deployed_holdout_kJ_mol") is not None for c, _ in caches)
    if has_deployed_2b:
        m2 = [s["distillation"][c]["mae_2b_deployed_holdout_kJ_mol"] for c, _ in caches]
        lab2 = "2B deployed hold-out MAE"
        xlab_f = "Error on 15 % hold-out (kJ mol$^{-1}$)"
    else:
        m2 = [
            s["distillation"][c].get("mae_2b_deployed_holdout_kJ_mol", s["distillation"][c]["mae_2b_kJ_mol"])
            for c, _ in caches
        ]
        lab2 = "2B in-sample fit MAE (legacy)"
        xlab_f = "MAE (1B: 15 % hold-out; 2B: legacy in-sample, kJ mol$^{-1}$)"
    r2 = [s["distillation"][c]["r2_1b"] for c, _ in caches]
    tw = [s["distillation"][c]["distill_wall_s"] for c, _ in caches]
    f.barh(y + 0.18, m1, height=0.34, color="#2F5D8A", label="1B hold-out MAE")
    f.barh(y - 0.18, m2, height=0.34, color="#D98E04", label=lab2)
    for yi, a1, rr, tt in zip(y, m1, r2, tw):
        f.text(a1 + 0.06, yi + 0.18, f"R\u00b2 = {rr:.3f}   ({tt:.0f} s)", va="center", fontsize=6.9)
    f.set_yticks(y); f.set_yticklabels([c[1] for c in caches], fontsize=7.0)
    f.set_xlim(0, 4.2)
    f.set_xlabel(xlab_f)
    f.set_title("Surrogate fidelity (caches used in production)", loc="left")
    f.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=6.8)
    panel_label(f, "f", x=-0.42)

    fig.savefig(OUT / "fig3_exp2_transferability.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved fig3_exp2_transferability.png")


# --------------------------------------------------------------------------------------------
# Figure 4: Experiment 3 (physics checks, efficiency, convergence)
# --------------------------------------------------------------------------------------------

def fig_exp3(s):
    fig, axs = plt.subplots(2, 3, figsize=(14.2, 8.4))
    fig.subplots_adjust(hspace=0.48, wspace=0.38)
    (a, b, c), (d, e, f) = axs
    wr = s["widom"]["phase3_widom_r2scan"]; wm = s["widom"]["phase3_widom_mp0d3"]

    # (a) Widom
    labs = ["r2SCAN\n(no D3)", "MP0\n+ D3"]
    q0 = [wr["widom"]["Qst_0_kJ_mol"], wm["widom"]["Qst_0_kJ_mol"]]
    q0e = [wr["widom"]["Qst_0_sem_kJ_mol"], wm["widom"]["Qst_0_sem_kJ_mol"]]
    kh = [wr["widom"]["K_H_mmol_g_bar"], wm["widom"]["K_H_mmol_g_bar"]]
    khe = [wr["widom"]["K_H_sem_mmol_g_bar"], wm["widom"]["K_H_sem_mmol_g_bar"]]
    a.bar([0, 1], q0, yerr=q0e, width=0.5, color=[C["auto"], C["mp0d3"]], alpha=0.88, capsize=2.5, ecolor="#374151")
    for i in range(2):
        kh_txt = f"{kh[i]:.0f} \u00b1 {khe[i]:.0f}" if kh[i] > 100 else f"{kh[i]:.2f} \u00b1 {khe[i]:.2f}"
        a.text(i, q0[i] + q0e[i] + 1.2, f"Q$_{{st}}$(0) = {q0[i]:.1f} \u00b1 {q0e[i]:.1f}\nK$_H$ = {kh_txt}",
               ha="center", fontsize=7.6)
    a.set_xticks([0, 1]); a.set_xticklabels(labs)
    a.set_ylim(0, 66); a.set_xlim(-0.6, 1.6)
    a.set_ylabel("Q$_{st}$(0) (kJ mol$^{-1}$)")
    a.set_title("Widom insertion, Mg-MOF-74 + CO$_2$", loc="left")
    a.text(0.02, 0.97, "6,000 insertions; K$_H$ in mmol g$^{-1}$ bar$^{-1}$", transform=a.transAxes, fontsize=7.0,
           color=C["muted"], va="top")
    panel_label(a, "a")

    # (b) energy decomposition from equilibrium configurations
    Pl = ["0.012", "0.28", "1.0"]
    dr = wr["mace_d3_decomposition"]; dm = wm["mace_d3_decomposition"]
    x = np.arange(3); w = 0.36
    r_mace = [-z["mean_u_mace_per_mol_kJ_mol"] for z in dr]
    m_mace = [-z["mean_u_mace_per_mol_kJ_mol"] for z in dm]
    m_d3 = [-z["mean_u_d3_per_mol_kJ_mol"] for z in dm]
    b.bar(x - w / 2, r_mace, w, color=C["auto"], alpha=0.88, label="r2SCAN: MACE")
    b.bar(x + w / 2, m_mace, w, color="#E8A39B", label="MP0+D3: MACE part")
    b.bar(x + w / 2, m_d3, w, bottom=m_mace, color=C["mp0d3"], label="MP0+D3: D3 part")
    for i in range(3):
        tot = m_mace[i] + m_d3[i]
        b.text(x[i] + w / 2, tot + 0.8, f"{tot:.1f}\n(D3 {100 * m_d3[i] / tot:.0f}%)", ha="center", fontsize=7.0)
        nb = dr[i]["per_box"]
        lab = f"{r_mace[i]:.1f}" + ("\n(N=1)" if sum(bb["N"] for bb in nb) <= 1 else "")
        b.text(x[i] - w / 2, r_mace[i] + 0.8, lab, ha="center", fontsize=7.0)
    b.set_xticks(x); b.set_xticklabels([p + " bar" for p in Pl])
    b.set_ylim(0, 64)
    b.set_ylabel(r"$-\langle U\rangle/N$ in equilibrium (kJ mol$^{-1}$)")
    b.set_title("Where the binding energy comes from", loc="left")
    b.legend(loc="upper center", fontsize=6.6, ncol=3, handlelength=1.0, columnspacing=0.8)
    panel_label(b, "b")

    # (c) k_steps
    ks = s["kstep_sweep_mgmof74_0p28"]
    kk = np.array(ks["k_steps"]); acc = np.array(ks["outer_acc_pct"]); esjd = np.array(ks["esjd_N"])
    c.plot(kk, acc, "o-", color=C["auto"], lw=1.4, ms=5, label="outer acceptance")
    c.set_xlabel("Inner moves per outer MLIP call, k"); c.set_ylabel("Outer acceptance (%)", color=C["auto"])
    c.tick_params(axis="y", colors=C["auto"])
    c.set_ylim(0, 80)
    c2 = c.twinx(); c2.spines["right"].set_visible(True); c2.spines["top"].set_visible(False)
    c2.plot(kk, esjd, "s--", color=C["mercado"], lw=1.4, ms=5, label="ESJD$_N$")
    c2.set_ylabel("ESJD$_N$ (molecules$^2$ per cycle)", color=C["mercado"]); c2.tick_params(axis="y", colors=C["mercado"])
    c2.set_ylim(0.10, 0.30)
    c.axvline(ks["selected"], color="#6B7280", lw=0.9, ls=":")
    c.text(ks["selected"] + 0.4, 74, f"k = {ks['selected']} used", fontsize=7.4, color=C["muted"])
    h1, l1 = c.get_legend_handles_labels(); h2, l2 = c2.get_legend_handles_labels()
    c.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=7.2)
    c.set_title("Calibrating the inner burst length", loc="left")
    panel_label(c, "c")

    # (d) multi-GPU
    p1 = s["runs"]["phase3_scale_1gpu_pressure"]["total_wall_s"]; p4 = s["runs"]["phase3_scale_4gpu_pressure"]["total_wall_s"]
    r1w = s["runs"]["phase3_scale_1gpu_replica"]["total_wall_s"]; r4w = s["runs"]["phase3_scale_4gpu_replica"]["total_wall_s"]
    r1 = s["runs"]["phase3_scale_1gpu_replica"]["points"][0]["elapsed_s"]; r4 = s["runs"]["phase3_scale_4gpu_replica"]["points"][0]["elapsed_s"]
    cats = ["pressure queue\nend-to-end", "replica sharding\nend-to-end", "replica sharding\nMC only"]
    one = [p1, r1w, r1]; four = [p4, r4w, r4]
    x = np.arange(3); w = 0.36
    d.bar(x - w / 2, one, w, color="#9CA3AF", label="1 \u00d7 RTX 3090")
    d.bar(x + w / 2, four, w, color="#0F766E", label="4 \u00d7 RTX 3090")
    for i in range(3):
        d.text(x[i] - w / 2, one[i] + 2, f"{one[i]:.0f}", ha="center", fontsize=7.4)
        d.text(x[i] + w / 2, four[i] + 2, f"{four[i]:.0f}\n{one[i] / four[i]:.2f}\u00d7", ha="center", fontsize=7.4, color="#0F766E")
    d.set_xticks(x); d.set_xticklabels(cats, fontsize=8)
    d.set_ylabel("Time (s)"); d.set_ylim(0, 150)
    d.set_title("Multi-GPU throughput (60 + 200 cycles)", loc="left")
    d.legend(loc="upper right", fontsize=7.4)
    panel_label(d, "d")

    # (e) convergence audit
    prod = [("exp1_auto_mlip_r2scan_nod3", "Mg-MOF-74 r2SCAN", "D", C["auto"]),
            ("exp1_auto_mlip_r2scan_nod3_highload_0p28", "Mg-MOF-74 r2SCAN", "D", C["auto"]),
            ("exp1_mp0_nod3_0p28", "Mg-MOF-74 MP0", "P", C["mp0"]),
            ("exp1_auto_mlip_mp0_d3", "Mg-MOF-74 MP0+D3", "v", C["mp0d3"]),
            ("exp2_nimof74_co2", "Ni-MOF-74", "s", "#0F766E"),
            ("exp2_cubtc_co2", "Cu-BTC", "^", C["mercado"]),
            ("exp2_irmof1_co2", "IRMOF-1 CO$_2$", "o", "#6D28D9"),
            ("exp2_irmof1_co2_5bar", "IRMOF-1 CO$_2$", "o", "#6D28D9"),
            ("exp2_irmof1_ch4_mp0d3", "IRMOF-1 CH$_4$", "h", "#9D174D")]
    seen = set()
    n_pass = n_tot = 0
    for key, lab, mk, col in prod:
        for p in s["runs"][key]["points"]:
            if key == "exp2_irmof1_co2" and abs(p["P_bar"] - 5.0) < 1e-6:
                continue  # superseded by exp2_irmof1_co2_5bar
            if p["diag_status"] == "insufficient":
                continue
            ess = max(p["N_ess_bulk"], 1.0); rh = max(p["N_rhat_max"] - 1.0, 1e-4)
            ok = p["diag_status"] == "pass"
            n_tot += 1; n_pass += int(ok)
            e.plot(ess, rh, mk, color=col, ms=6.2, mfc=col if ok else "white", mew=1.2, ls="none",
                   label=lab if lab not in seen else None)
            seen.add(lab)
    e.axvline(400, color="#B91C1C", lw=0.8, ls="--"); e.axhline(0.01, color="#B91C1C", lw=0.8, ls="--")
    e.fill_between([400, 5000], 1e-4, 0.01, color="#DCFCE7", alpha=0.6, lw=0)
    e.text(4500, 1.4e-4, "gate region", fontsize=7.0, color=C["pass"], ha="right")
    e.set_xscale("log"); e.set_yscale("log")
    e.set_xlim(3, 5000); e.set_ylim(1e-4, 1.5)
    e.set_xlabel("Bulk ESS of N (4 chains)"); e.set_ylabel("$\\hat{R}$ \u2212 1 for N")
    e.set_title(f"Convergence of production points ({n_pass}/{n_tot} pass all gates)", loc="left")
    e.legend(loc="upper right", fontsize=6.6, ncol=1)
    e.text(0.02, 0.03, "filled: all gates passed\n(N, U, OMS, MCSE, drift)", transform=e.transAxes, fontsize=6.6, color=C["muted"])
    panel_label(e, "e")

    # (f) consistency checks (strictly data-driven from summary JSON)
    def pt_at_opt(key, P):
        run = s.get("runs", {}).get(key)
        if not run:
            return None
        for p in run.get("points", []):
            if abs(float(p["P_bar"]) - float(P)) < 1e-6:
                return p
        return None

    def _fmt_shift(p):
        de = p.get("delta_eps_kJ_mol")
        return f"\u0394\u03b5 = {float(de):.2f}" if de is not None else ""

    rowsf = []
    p_mg_e = pt_at_opt("exp1_auto_mlip_r2scan_nod3", 0.28)
    p_mg_hl = pt_at_opt("exp1_auto_mlip_r2scan_nod3_highload_0p28", 0.28)
    if p_mg_e is not None:
        rowsf.append((f"Mg-MOF-74 0.28 bar, empty start ({_fmt_shift(p_mg_e)})", p_mg_e, C["auto"]))
    if p_mg_hl is not None:
        rowsf.append((f"Mg-MOF-74 0.28 bar, loaded start ({_fmt_shift(p_mg_hl)})", p_mg_hl, C["hl"]))

    irmof_entries = []
    p_ir5 = pt_at_opt("exp2_irmof1_co2_5bar", 5.0)
    if p_ir5 is not None:
        irmof_entries.append((f"IRMOF-1 5 bar, {_fmt_shift(p_ir5)} kJ/mol", p_ir5, "#6D28D9"))
    for zkey in ("exp2_irmof1_co2_5bar_shift0", "exp2_irmof1_co2_5bar_zero_shift", "jctc_remc_irmof1_co2_r2scan"):
        p_ir0 = pt_at_opt(zkey, 5.0)
        if p_ir0 is not None:
            irmof_entries.append((f"IRMOF-1 5 bar, {_fmt_shift(p_ir0)} kJ/mol", p_ir0, "#A78BFA"))
            break
    p_irst = pt_at_opt("exp2_irmof1_co2", 5.0)
    if p_irst is not None:
        irmof_entries.append((f"IRMOF-1 5 bar, {_fmt_shift(p_irst)} kJ/mol (stalled)", p_irst, "#D1D5DB"))
    if irmof_entries:
        if rowsf:
            rowsf.append(None)
        rowsf.extend(irmof_entries)

    scale_entries = []
    p_s1 = pt_at_opt("phase3_scale_1gpu_replica", 0.28)
    p_s4 = pt_at_opt("phase3_scale_4gpu_replica", 0.28)
    if p_s1 is not None:
        scale_entries.append(("Mg-MOF-74 0.28 bar short run, 1 GPU (8 boxes)", p_s1, "#6B7280"))
    if p_s4 is not None:
        scale_entries.append(("Mg-MOF-74 0.28 bar short run, 4 GPUs (2 boxes each)", p_s4, "#0F766E"))
    if scale_entries:
        if rowsf:
            rowsf.append(None)
        rowsf.extend(scale_entries)

    yy = 0.0
    max_x = 7.6
    for r in rowsf:
        if r is None:
            yy -= 0.5
            continue
        lab, p, col = r
        up_v = float(p["uptake_mmol_g"])
        se_v = float(p.get("uptake_sem_mmol_g", 0.0) or 0.0)
        max_x = max(max_x, up_v + 2 * se_v + 1.1)
        f.errorbar(up_v, yy, xerr=2 * se_v, fmt="o", color=col, ms=5.2, capsize=2.5, lw=1.2)
        f.text(0.25, yy + 0.30, lab, fontsize=6.9, color=C["ink"], va="bottom")
        f.text(up_v + 2 * se_v + 0.12, yy, f"{up_v:.2f}", fontsize=6.6,
               va="center", color=col if col != "#D1D5DB" else C["muted"])
        yy -= 1.15
    f.set_yticks([]); f.spines["left"].set_visible(False)
    f.set_ylim(yy + 0.6, 0.9)
    f.set_xlabel("Uptake (mmol g$^{-1}$), bars: \u00b12 s.e.")
    f.set_xlim(0.2, max_x)
    f.set_title("Internal consistency checks", loc="left")
    panel_label(f, "f", x=-0.06)

    fig.savefig(OUT / "fig4_exp3_physics_scaling.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved fig4_exp3_physics_scaling.png")


# ============================================================================================
# JCTC v2 5-Figure Publication Suite (--mode jctc)
# ============================================================================================

def _jctc_find_run(s, preferred_keys, **criteria):
    """Find (run_key, run_dict) in `s['runs']` by exact preferred keys or metadata criteria."""
    runs = s.get("runs", {})
    for k in preferred_keys:
        if k in runs:
            return k, runs[k]
    if not criteria:
        return None, None
    for k, r in runs.items():
        ok = True
        for ck, cv in criteria.items():
            if ck == "model_contains":
                if str(cv).lower() not in str(r.get("model", "")).lower():
                    ok = False
                    break
            elif ck == "model_not_contains":
                if str(cv).lower() in str(r.get("model", "")).lower():
                    ok = False
                    break
            elif ck == "supercell":
                if list(r.get("supercell") or []) != list(cv):
                    ok = False
                    break
            else:
                if r.get(ck) != cv:
                    ok = False
                    break
        if ok:
            return k, r
    return None, None


def _jctc_pt_at(run, P_bar, tol=1e-4):
    if not run:
        return None
    for p in run.get("points", []):
        if abs(float(p["P_bar"]) - float(P_bar)) <= tol:
            return p
    return None


def _jctc_series(run):
    ps = sorted(run.get("points", []), key=lambda x: float(x["P_bar"]))
    return (
        np.array([float(x["P_bar"]) for x in ps]),
        np.array([float(x["uptake_mmol_g"]) for x in ps]),
        np.array([float(x.get("uptake_sem_mmol_g", 0.0) or 0.0) for x in ps]),
    )


def _jctc_trace_mmol(pt, default_mass=MG_SUPER_MASS):
    tr = np.asarray(pt.get("box_N_trace") or pt.get("box_N_history") or [], dtype=float)
    if tr.ndim != 2 or tr.shape[0] == 0:
        return np.array([], dtype=float)
    mass = float(pt.get("fw_mass_gmol") or default_mass)
    return tr.mean(axis=1) * 1000.0 / mass


def _short_model_label(model_str: str) -> str:
    s = str(model_str or "").strip()
    low = s.lower()
    stem_low = Path(s).stem.lower()
    if "mace-dac-1" in stem_low or "mace-dac-1" in low:
        return "MACE-DAC-1"
    if low == "medium":
        return "MACE-MP-0"
    if low == "mace-matpes-r2scan-0":
        return "r2SCAN"
    if s.endswith(".model"):
        return Path(s).stem
    return s


def _draw_unavailable(ax, title, msg):
    ax.set_title(title, loc="left")
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
            fontsize=8.5, color=C["muted"], style="italic")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def fig_jctc1_architecture(s, arch_npz_path: Path, out_dir: Path, presub: dict = None):
    """JCTC Fig. 1: Three-step workflow (1. Auto grid cache; 2. Hand-written CUDA CBMC burst; 3. Batched MLIP outer check)."""
    fig = plt.figure(figsize=(14.0, 8.8))

    # (a) Step 1: Automated reference grid construction & caching
    axa = fig.add_axes([0.015, 0.52, 0.635, 0.44])
    axa.set_xlim(0, 100); axa.set_ylim(0, 44); axa.axis("off")
    axa.text(0, 43.4, "a", fontsize=13, fontweight="bold", va="top")
    axa.text(3.0, 43.2, "Step 1 — Automated on-demand reference-potential distillation & caching (once per MOF, guest, MLIP)",
             fontsize=10.2, va="top", color=C["ink"], fontweight="bold")

    blue_fc, blue_ec = "#EEF3FA", "#2F5D8A"
    _box(axa, 0.5, 12.2, 22.5, 23, "1. Inputs",
         "Rigid MOF (CIF,\n  any space group)\nRigid guest (\u2264 12 sites)\nTarget MLIP + SHA-256\nT, pressure ladder", blue_fc, blue_ec)
    _box(axa, 25.8, 12.2, 22.5, 23, "2. Auto typing",
         "Framework grouped by\n  element + coordination\n  graph (6\u20138 classes)\nGuest symmetry channels\nZero manual FF typing", blue_fc, blue_ec)
    _box(axa, 51.1, 12.2, 22.5, 23, "3. Target poses",
         "1,200 1B single poses +\n  192 in-pore 2B dimers\nUnified continuous MIC\n  hard-core (1.20/1.48/\n  0.50 \u00c5), $N_{\\max}=128$", blue_fc, blue_ec)
    _box(axa, 76.4, 12.2, 23.1, 23, "4. Grid & LUT cache",
         "1B: radial + neutral Ewald\n  \u2192 FP32 tricubic grids\n2B: scaled LJ/DSF + 128-\n  bin COM residual LUT\n85/15 & 80/20 hold-out", blue_fc, blue_ec)
    for x0 in (23.0, 48.3, 73.6):
        _arrow(axa, (x0 + 0.2, 23.7), (x0 + 2.6, 23.7))

    dists = list(s.get("distillation", {}).values())
    walls = [float(d["distill_wall_s"]) for d in dists if d.get("distill_wall_s")]
    if presub is not None:
        s12 = presub["surrogates"]["1B_2B"]["summary"]
        s1 = presub["surrogates"]["1B_only"]["summary"]
        walls.extend([float(s12["distill_wall_s"]), float(s1["distill_wall_s"])])
        ex_txt = (
            f"Mg-MOF-74 + CO$_2$ (r2SCAN, v1.4): 1B hold-out R$^2$ = {float(s12['r2_1b']):.3f} "
            f"(MAE {float(s12['mae_1b_kJ_mol']):.2f} kJ/mol); 2B deployed hold-out MAE "
            f"{float(s12['mae_2b_classical_holdout_kJ_mol']):.2f} \u2192 {float(s12['mae_2b_deployed_holdout_kJ_mol']):.2f} kJ/mol"
        )
    else:
        ex_txt = "Cached Surrogate: 1B tricubic grids + 2B residual LUT uploaded once to GPU memory"
    wall_txt = f"{min(walls):.0f}\u2013{max(walls):.0f} s" if len(walls) >= 2 else "~40\u2013213 s"
    axa.text(50, 7.7, f"Cached on disk by SHA-256 key ({wall_txt} on 1\u00d7RTX 3090); uploaded once (H\u2192D) to GPU global memory",
             ha="center", fontsize=8.1, color=C["muted"], style="italic")
    axa.text(50, 3.6, ex_txt, ha="center", fontsize=7.8, color=C["muted"], style="italic")

    # Real 1B grid + 2B LUT panels
    axg = fig.add_axes([0.665, 0.55, 0.17, 0.365])
    axl = fig.add_axes([0.875, 0.615, 0.11, 0.265])
    if arch_npz_path.is_file():
        arch = np.load(arch_npz_path)
        Z = np.clip(arch["o_min_kJ"], -45.0, 5.0)
        n = Z.shape[0]
        a = float(arch["a"]); g = np.deg2rad(float(arch["gamma_deg"]))
        A = np.array([a, 0.0]); B = np.array([a * np.cos(g), a * np.sin(g)])
        fa = (np.arange(n + 1) / n)
        FA, FB = np.meshgrid(fa, fa)
        for i in (0, 1):
            for j in (0, 1):
                X = (FA + i) * A[0] + (FB + j) * B[0]
                Y = (FA + i) * A[1] + (FB + j) * B[1]
                im = axg.pcolormesh(X, Y, Z, cmap="YlGnBu_r", vmin=-45, vmax=5, shading="flat", rasterized=True)
        cx, cy = (A + B)
        half = 15.0
        axg.set_xlim(cx - half, cx + half); axg.set_ylim(cy - half, cy + half)
        axg.set_aspect("equal"); axg.set_xticks([]); axg.set_yticks([])
        for sp in axg.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.6)
        axg.plot([cx - half + 2, cx - half + 12], [cy - half + 2] * 2, color="#111827", lw=2.2)
        axg.text(cx - half + 7, cy - half + 3.2, "10 \u00c5", color="#111827", ha="center", fontsize=8, fontweight="bold")
        axg.set_title("1B grid, Mg-MOF-74 (r2SCAN)\nO(CO$_2$) channel, min along c", fontsize=8.2, color=C["ink"])
        cax = fig.add_axes([0.665, 0.526, 0.17, 0.012])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label("U$_{1B}$ (kJ mol$^{-1}$)", fontsize=7.8, labelpad=1)
        cb.ax.tick_params(labelsize=7)

        lut = arch["lut_K"] * 8.31446261815324e-3
        r = np.linspace(float(arch["lut_rmin"]), float(arch["lut_rmax"]), len(lut))
        axl.axhline(0, color="#9CA3AF", lw=0.6)
        axl.plot(r, lut, color=blue_ec, lw=1.4)
        axl.set_xlabel("r$_{COM}$ (\u00c5)", fontsize=8)
        axl.set_ylabel("\u0394U$_{2B}$ (kJ mol$^{-1}$)", fontsize=8)
        axl.tick_params(labelsize=7)
        axl.set_title("2B residual LUT", fontsize=8.2, color=C["ink"])
    else:
        _draw_unavailable(axg, "1B grid", "arch_inputs NPZ\nnot provided")
        _draw_unavailable(axl, "2B LUT", "arch_inputs NPZ\nnot provided")

    # (b) Step 2 (CUDA inner burst) & Step 3 (Batched MLIP outer check + accept/rollback)
    axb = fig.add_axes([0.015, 0.025, 0.635, 0.465])
    axb.set_xlim(0, 100); axb.set_ylim(0, 47); axb.axis("off")
    axb.text(0, 46.4, "b", fontsize=13, fontweight="bold", va="top")
    axb.text(3.0, 46.2, "Steps 2 & 3 — Hand-written CUDA multi-step GCMC inner burst + batched full-system MLIP outer check",
             fontsize=10.2, va="top", color=C["ink"], fontweight="bold")

    _box(axb, 0.5, 16.0, 19.5, 17.5, "GPU accepted state",
         "$x_b=(N_b,\\{\\mathbf{r}_i,\\mathbf{q}_i\\})$\n$B$ independent boxes\nin device memory\n($N_b \\leq N_{\\max}=128$)", "#F3F4F6", "#4B5563", body_size=8.0)
    _box(axb, 25.5, 25.2, 43.0, 17.8, "Step 2: Hand-written CUDA burst (1 block = 128 threads / box)",
         "\u2022 $k$ moves on device ($\\tilde{U}_{\\rm ref}+\\Delta\\varepsilon N$, zero host I/O inside burst)\n"
         "\u2022 ins/del: 128 threads evaluate $K=128$ CBMC trial poses in parallel\n"
         "\u2022 Unified continuous periodic MIC hard-core + FP64 shared memory\n"
         "\u2022 Exact full FP64 $\\tilde{U}_{\\rm ref}(x'_b)$ recompute at end of $k$-step burst", "#FDF5EA", "#B5651D", title_size=8.9, body_size=7.8, body_lines=1.34)
    _box(axb, 74.0, 16.0, 25.5, 17.5, "Step 3a: Batched MLIP",
         "1 batched MACE(\u00b1D3) call\nacross all $B$ end states\n(static fw graph on GPU;\nFP32 msg + FP64 sum)", "#EEF3FA", "#2F5D8A", body_size=7.9)
    _box(axb, 25.5, 4.4, 43.0, 16.5, "Step 3b: Outer Metropolis & GPU commit / rollback", "", "#F1EEF8", "#5B4B8A", title_size=8.9)
    axb.text(47.0, 12.8,
             r"$\alpha_b=\min\left\{1,\ \exp\left(-\beta\,[\,\Delta\tilde{U}_{\rm MLIP}-\Delta\tilde{U}_{\rm ref}-\Delta\varepsilon\,\Delta N\,]\right)\right\}$",
             ha="center", va="center", fontsize=9.6, color=C["ink"])
    axb.text(47.0, 7.4,
             "accept: commit $x'_b$ on GPU   |   reject: roll back to $x_b$ on GPU\n"
             "optional $\\mu$-REMC: swap $P_m\\leftrightarrow P_{m+1}$ every $n_{\\rm swap}$ cycles (0 MLIP calls)",
             ha="center", va="center", fontsize=7.4, color=C["ink"], linespacing=1.26)

    _arrow(axb, (10.5, 33.7), (25.3, 36.4), rad=-0.22)
    axb.text(11.5, 38.6, "in-device\nload $x_b$", fontsize=7.6, color=C["ink"])
    _arrow(axb, (68.7, 33.8), (82.0, 33.7), rad=-0.22)
    axb.text(70.0, 38.0, "D\u2192H C ABI (~29 KB)\n+ H\u2192D guest edges", fontsize=7.3, color="#B5651D")
    _arrow(axb, (86.5, 15.8), (68.7, 11.8), rad=-0.22)
    axb.text(75.2, 9.0, "$\\tilde{U}_{\\rm MLIP}(x'_b)$\n(1 batch / $k$ steps)", fontsize=7.5, color="#2F5D8A")
    _arrow(axb, (25.3, 11.8), (10.5, 15.8), rad=-0.22)
    axb.text(8.8, 9.6, "H\u2192D mask ($B$ ints)\ncommit / rollback", fontsize=7.3, color="#5B4B8A")

    axb.text(50.0, 0.3,
             "Amortization: 1 batched MLIP call per $k=24$ GPU moves (~1,536 CBMC trials/box); "
             "frozen $(\\Delta\\varepsilon, k)$ + common MIC hard-core $\\Rightarrow$ exact $\\tilde{\\pi}_{\\rm MLIP}$.",
             ha="center", va="bottom", fontsize=7.1, color=C["muted"])

    # (c) Multi-GPU & REMC parallelization
    axc = fig.add_axes([0.665, 0.025, 0.325, 0.465])
    axc.set_xlim(0, 100); axc.set_ylim(0, 47); axc.axis("off")
    axc.text(0, 46.4, "c", fontsize=13, fontweight="bold", va="top")
    axc.text(5.5, 46.2, "Multi-GPU isolated ladders & sharding", fontsize=10.2, va="top", color=C["ink"], fontweight="bold")
    shades = ["#C7D7EC", "#93B4DA", "#5E8FC6", "#2F5D8A"]

    def _draw_row(y0, label, mode_tag, note):
        axc.text(0, y0 + 13.0, label, fontsize=8.4, fontweight="bold", color=C["ink"])
        for gi in range(4):
            gx = 2 + gi * 24.5
            axc.add_patch(FancyBboxPatch((gx, y0), 22.0, 10.5, boxstyle="round,pad=0,rounding_size=0.8",
                                         fc="white", ec="#6B7280", lw=0.8))
            axc.text(gx + 11.0, y0 + 8.5, f"GPU {gi}", ha="center", fontsize=7.4, color=C["muted"])
            if mode_tag == "remc":
                for bi in range(4):
                    axc.add_patch(Rectangle((gx + 2.0 + bi * 4.6, y0 + 1.5), 3.8, 4.0, fc=shades[bi], ec="none"))
                axc.text(gx + 11.0, y0 + 6.2, f"$P_{gi + 1}$", ha="center", fontsize=7.8, color=C["ink"])
                if gi < 3:
                    _arrow(axc, (gx + 22.2, y0 + 5.2), (gx + 24.3, y0 + 5.2), color=C["remc"], lw=1.1)
                    _arrow(axc, (gx + 24.3, y0 + 3.4), (gx + 22.2, y0 + 3.4), color=C["remc"], lw=1.1)
            else:
                for bi in range(2):
                    axc.add_patch(Rectangle((gx + 6.4 + bi * 4.8, y0 + 1.5), 3.8, 4.0, fc=shades[2], ec="none"))
                axc.text(gx + 11.0, y0 + 6.2, "$P$", ha="center", fontsize=7.8, color=C["ink"])
        axc.text(0, y0 - 2.3, note, fontsize=7.4, color=C["muted"], va="top", linespacing=1.25)

    remc_note = "1 pressure/GPU, 4 isolated ladders (colors: never cross-mix);\nzero-MLIP fugacity swaps boost whole-ladder ESS rate 2.26\u20132.29\u00d7"
    p1_r = s.get("runs", {}).get("phase3_scale_1gpu_replica")
    p4_r = s.get("runs", {}).get("phase3_scale_4gpu_replica")
    if p1_r and p4_r and p1_r.get("points") and p4_r.get("points"):
        r1 = float(p1_r["points"][0]["elapsed_s"]); r4 = float(p4_r["points"][0]["elapsed_s"])
        rep_note = f"Single-$P$ sharding: {r1:.0f} s \u2192 {r4:.0f} s ({r1 / max(r4, 1e-6):.2f}\u00d7 pure MC on 4\u00d7RTX 3090)"
    else:
        rep_note = "Independent chains sharded across GPUs and merged for Vehtari stats"

    _draw_row(26.5, "Isolated-ladder \u03bc-REMC across GPUs (8 GPUs in formal 2\u00d72)", "remc", remc_note)
    _draw_row(5.5, "Replica sharding for single-pressure scaling", "replica", rep_note)

    out_p = out_dir / "fig1_architecture.png"
    fig.savefig(out_p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_p.name}")


def fig_jctc2_parity_efficiency(s, exp_dir: Path, out_dir: Path):
    """JCTC Fig. 2: Exact parity (Direct vs Nested), relaxation traces, efficiency, & Mg-MOF-74 ablation."""
    fig, axs = plt.subplots(2, 2, figsize=(11.8, 8.8))
    (ax_a, ax_b), (ax_c, ax_d) = axs
    fig.subplots_adjust(hspace=0.42, wspace=0.32)

    # (a) Exact Parity Test: Direct MLIP-GCMC vs Auto-Distilled Nested MC on IRMOF-1 + CO2 @ 1.0 bar
    _, r_dir_par = _jctc_find_run(
        s, ["jctc_parity_direct_irmof1_co2_1bar"],
        mof="IRMOF-1", guest="co2", proposer="direct",
    )
    _, r_nes_par = _jctc_find_run(
        s, ["jctc_parity_nested_irmof1_co2_1bar", "jctc_remc_irmof1_co2_r2scan"],
        mof="IRMOF-1", guest="co2", proposer="auto_mlip", grid_only=False,
    )
    p_dir = _jctc_pt_at(r_dir_par, 1.0)
    p_nes = _jctc_pt_at(r_nes_par, 1.0)
    if p_dir is not None and p_nes is not None:
        xs = np.array([0, 1], dtype=float)
        ups = [float(p_dir["uptake_mmol_g"]), float(p_nes["uptake_mmol_g"])]
        sems = [float(p_dir.get("uptake_sem_mmol_g", 0.0) or 0.0), float(p_nes.get("uptake_sem_mmol_g", 0.0) or 0.0)]
        cols = [C["direct"], C["auto"]]
        ax_a.bar(xs, ups, yerr=[2 * e for e in sems], width=0.45, color=cols, alpha=0.88, capsize=4, ecolor="#111827")
        for xi, u, se in zip(xs, ups, sems):
            ax_a.text(xi, u + 2 * se + max(0.03, 0.05 * max(ups)), f"{u:.3f} \u00b1 {se:.3f}", ha="center", fontsize=8.0)
        comb_se = math.hypot(sems[0], sems[1])
        z_val = abs(ups[1] - ups[0]) / comb_se if comb_se > 1e-12 else 0.0
        ed = float(p_dir.get("ess_bulk_N_per_gpu_s") or 0.0)
        en = float(p_nes.get("ess_bulk_N_per_gpu_s") or 0.0)
        sp_txt = f"ESS/GPU-s speedup: {en / ed:.1f}\u00d7" if ed > 1e-9 else ""
        q_d = p_dir.get("Qst_fluc_kJ_mol"); q_n = p_nes.get("Qst_fluc_kJ_mol")
        q_txt = f"\nQ$_{{st}}$: {q_d:.1f} vs {q_n:.1f} kJ/mol" if (q_d is not None and q_n is not None) else ""
        ax_a.text(
            0.03, 0.96,
            f"Unified hard-core $\\tilde{{U}}_{{\\rm MLIP}}$\n$z$-score = {z_val:.2f} (unbiased)\n{sp_txt}{q_txt}".strip(),
            transform=ax_a.transAxes, va="top", fontsize=7.6, color=C["ink"],
            bbox=dict(boxstyle="round,pad=0.35", fc="#F9FAFB", ec="#D1D5DB", lw=0.7),
        )
        ax_a.set_xticks(xs); ax_a.set_xticklabels(["Direct MLIP-GCMC\n(unified hard-core)", "Auto-Distilled\nNested MC"])
        ax_a.set_ylabel("CO$_2$ uptake (mmol g$^{-1}$), bars: \u00b12 s.e.")
        ax_a.set_ylim(0, max(ups) * 1.48 + 0.15)
        ax_a.set_title("Exact parity: IRMOF-1 + CO$_2$, 1.0 bar (r2SCAN)", loc="left")
    else:
        _draw_unavailable(ax_a, "Exact parity: IRMOF-1 + CO$_2$, 1.0 bar", "Parity runs not present in archive")
    panel_label(ax_a, "a")

    # (b) Relaxation vs wall-clock time at Mg-MOF-74 0.28 bar (r2SCAN)
    ablation_specs = [
        (["jctc_exp1_direct_r2scan_0p28", "exp1_direct_r2scan_0p28"], C["direct"], "Direct MLIP-GCMC", False, "-"),
        (["jctc_exp1_uff_r2scan", "exp1_uff_r2scan_0p28"], C["uff"], "UFF proposer", False, "-"),
        (["jctc_exp1_mercado_r2scan", "exp1_mercado_r2scan_0p28"], C["mercado"], "Mercado-2016 proposer", False, "-"),
        (["jctc_exp1_auto_mlip_1b_only_r2scan"], C["oneb"], "Auto-MLIP (1B-only)", True, "--"),
        (["jctc_exp1_auto_mlip_r2scan_no_remc", "exp1_auto_mlip_r2scan_nod3"], C["auto"], "Auto-MLIP 1B+2B (no REMC)", True, "-"),
        (["jctc_exp1_auto_mlip_r2scan_remc"], C["remc"], "Auto-MLIP 1B+2B (REMC, empty)", True, "-"),
        (["jctc_exp1_auto_mlip_r2scan_highload_0p28", "exp1_auto_mlip_r2scan_nod3_highload_0p28"], C["hl"], "Auto-MLIP 1B+2B (loaded start)", True, ":"),
    ]
    d_wall_max = 0.0
    t_max_b = 100.0
    plotted_b = 0
    for keys, col, lab, add_dist, ls in ablation_specs:
        _, run = _jctc_find_run(s, keys)
        pt = _jctc_pt_at(run, 0.28)
        if pt is None:
            continue
        y = _jctc_trace_mmol(pt)
        if len(y) == 0:
            continue
        off = float((run.get("distillation") or {}).get("distill_wall_s", 0.0) or 0.0) if add_dist else 0.0
        d_wall_max = max(d_wall_max, off)
        el = float(pt.get("elapsed_s", 0.0) or 0.0)
        t = off + el * (np.arange(1, len(y) + 1) / float(len(y)))
        ax_b.plot(t, y, color=col, lw=1.3, ls=ls, label=lab)
        ax_b.plot(t[-1], y[-1], "o", color=col, ms=3.2)
        t_max_b = max(t_max_b, float(t[-1]))
        plotted_b += 1

    if plotted_b > 0:
        if d_wall_max > 0:
            ax_b.axvspan(0, d_wall_max, color="#E5E7EB", alpha=0.65, lw=0)
            ax_b.text(d_wall_max / 2.0, 8.2, "distillation", ha="center", va="top", fontsize=7.0, color=C["muted"], rotation=90)
        ax_b.axhspan(4.745, 6.937, color="#D1D5DB", alpha=0.35, lw=0)
        ax_b.set_xlim(0, min(max(t_max_b * 1.04, 200.0), 1500.0))
        ax_b.set_ylim(0, 8.8)
        ax_b.set_xlabel("Wall-clock time on one RTX 3090 (s)")
        ax_b.set_ylabel("Box-averaged CO$_2$ uptake (mmol g$^{-1}$)")
        ax_b.set_title("Relaxation at 0.28 bar, Mg-MOF-74 (r2SCAN)", loc="left")
        ax_b.legend(loc="lower right", fontsize=6.8)
    else:
        _draw_unavailable(ax_b, "Relaxation at 0.28 bar, Mg-MOF-74", "No Mg-MOF-74 0.28 bar traces found")
    panel_label(ax_b, "b")

    # (c) Sampling efficiency: ESS_bulk(N)/GPU-s and time-to-MCSE=0.10 mmol/g at Mg-MOF-74 0.28 bar
    eff_rows = []
    for keys, col, lab, _, _ in ablation_specs[:6]:
        _, run = _jctc_find_run(s, keys)
        pt = _jctc_pt_at(run, 0.28)
        if pt is None:
            continue
        ess_rate = float(pt.get("ess_bulk_N_per_gpu_s") or 0.0)
        t_samp = pt.get("time_to_mcse_0p1_s")
        t_cold = pt.get("time_to_mcse_0p1_coldstart_s")
        eff_rows.append((lab, col, ess_rate, t_samp, t_cold))

    if eff_rows:
        yy = np.arange(len(eff_rows))[::-1]
        rates = [max(r[2], 1e-4) for r in eff_rows]
        cols_c = [r[1] for r in eff_rows]
        ax_c.barh(yy, rates, height=0.56, color=cols_c, alpha=0.88)
        ax_c.set_xscale("log")
        max_r = max(rates)
        ax_c.set_xlim(1e-4, max(10.0, max_r * 18.0))
        for yi, (_, _, rate, ts, tc) in zip(yy, eff_rows):
            if rate <= 1e-4:
                txt = "stalled / unestimable"
            else:
                t_txt = f" | t(0.1): {ts:.0f}s (cold {tc:.0f}s)" if (ts is not None and tc is not None) else ""
                txt = f"{rate:.3f} ESS/GPU-s{t_txt}"
            ax_c.text(max(rate, 1e-4) * 1.18, yi, txt, va="center", fontsize=6.9, color=C["ink"])
        ax_c.set_yticks(yy); ax_c.set_yticklabels([r[0] for r in eff_rows], fontsize=7.6)
        ax_c.set_xlabel("Bulk ESS of $N$ per GPU-second (log scale)")
        ax_c.set_title("Sampler efficiency & time to MCSE = 0.10 mmol/g", loc="left")
    else:
        _draw_unavailable(ax_c, "Sampler efficiency (0.28 bar)", "No ablation runs found")
    panel_label(ax_c, "c")

    # (d) Mg-MOF-74 isotherms & ablation (REMC, 1B-only, grid_only, MP0+D3 vs Exp)
    for fn, col, mk, lab in [
        ("mgmof74_10.1021acs.jpcc.7b08239.Isotherm1.csv", C["exp1"], "o", "Exp. (10.1021/acs.jpcc.7b08239)"),
        ("mgmof74_10.1039c5dt03137k.Isotherm1.csv", C["exp2"], "s", "Exp. (10.1039/c5dt03137k)"),
    ]:
        if (exp_dir / fn).is_file():
            x, y = load_exp(fn)
            m = (x > 0.004) & (x < 1.3)
            ax_d.plot(x[m], y[m], mk, ms=4.0, mfc="none", mec=col, mew=0.95, ls="none", label=lab)
    iso_specs = [
        (["jctc_exp1_auto_mlip_r2scan_remc", "jctc_exp1_auto_mlip_r2scan_no_remc", "exp1_auto_mlip_r2scan_nod3"], C["auto"], "D", "-", "Nested MC (1B+2B, r2SCAN)"),
        (["jctc_exp1_auto_mlip_1b_only_r2scan"], C["oneb"], "s", ":", "Nested MC (1B-only, r2SCAN)"),
        (["jctc_exp1_grid_only_r2scan", "exp1_grid_only_r2scan"], C["grid"], "^", "--", "Surrogate only (grid-only)"),
        (["jctc_remc_mgmof74_co2_mp0_d3", "exp1_auto_mlip_mp0_d3"], C["mp0d3"], "v", "-", "Nested MC (REMC, MP0+D3)"),
        (["jctc_macedac_mgmof74_co2"], C["macedac"], "*", "-", "Nested MC (REMC, MACE-DAC-1)"),
    ]
    for keys, col, mk, ls, lab in iso_specs:
        _, run = _jctc_find_run(s, keys)
        if run and run.get("points"):
            p, q, e = _jctc_series(run)
            ax_d.errorbar(p, q, yerr=e, fmt=mk + ls, color=col, ms=5.5, lw=1.35, capsize=2.5, label=lab)
    ax_d.set_xscale("log"); ax_d.set_yscale("log")
    ax_d.set_xlim(0.008, 1.4); ax_d.set_ylim(0.1, 35)
    ax_d.set_xlabel("Pressure (bar)"); ax_d.set_ylabel("CO$_2$ uptake (mmol g$^{-1}$)")
    ax_d.set_title("Mg-MOF-74 + CO$_2$ isotherm & ablations (298 K)", loc="left")
    ax_d.legend(loc="lower right", fontsize=6.8)
    panel_label(ax_d, "d")

    out_p = out_dir / "fig2_parity_efficiency.png"
    fig.savefig(out_p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_p.name}")


_PRESUB_GROUPS = [
    ("A", "A: 1B-only* + REMC OFF", "A\n1B-only*\nREMC OFF", "#64748B", "--", "o"),
    ("B", "B: 1B+2B + REMC OFF", "B\n1B+2B\nREMC OFF", "#D97706", "-.", "s"),
    ("C", "C: 1B-only* + REMC ON", "C\n1B-only*\nREMC ON", "#2563EB", "-", "^"),
    ("D", "D: 1B+2B + REMC ON", "D\n1B+2B\nREMC ON", "#0F766E", "-", "D"),
]


def fig_jctc2_presubmission_ablation(ps: dict, out_dir: Path):
    """JCTC Fig. 2 (pre-submission 2x2 matched-budget ablation, Mg-MOF-74 / CO2, 298 K, 0.28 bar).

    All numbers are read directly from build/jctc_presubmission/matched_budget_summary.json.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12.6, 9.2))
    (ax_a, ax_b), (ax_c, ax_d) = axs
    fig.subplots_adjust(hspace=0.40, wspace=0.28)

    runs = ps["runs"]
    gs_map = ps["group_summaries"]
    contrasts = ps["factorial_contrasts"]
    budget_wall_s = 7200.0
    budget_gpu_s = 57600.0

    # ---- (a) ESS_bulk(N) at 0.28 bar vs cumulative whole-ladder GPU-s ----
    ax_a.axhline(400.0, color="#DC2626", lw=1.0, ls=":", zorder=1)
    ax_a.text(1.5, 412.0, "ESS gate = 400", fontsize=7.6, color="#DC2626", va="bottom", ha="left")
    ax_a.axvline(budget_gpu_s / 1000.0, color="#9CA3AF", lw=0.9, ls="--", zorder=1)

    for grp, leg_lab, _, col, ls, mk in _PRESUB_GROUPS:
        g_runs = [r for r in runs if r["group"] == grp]
        grid_t = np.linspace(2.4, 57.6, 24)
        interp_y = []
        for r in g_runs:
            ckpts = r["empirical_convergence"]["checkpoints"]
            xs = np.array([c["total_ladder_gpu_s"] / 1000.0 for c in ckpts], dtype=float)
            ys = np.array([float(c.get("N_ess_bulk") or 0.0) for c in ckpts], dtype=float)
            ax_a.plot(xs, ys, color=col, lw=1.05, ls=ls, alpha=0.42, zorder=2)
            passed_fin = bool(r["primary_028bar"]["converged"])
            ax_a.scatter(
                [xs[-1]], [ys[-1]],
                s=34, marker=mk,
                facecolor=col if passed_fin else "white",
                edgecolor=col, linewidths=1.3, zorder=5,
            )
            if len(xs) >= 2:
                interp_y.append(np.interp(grid_t, xs, ys))
        if interp_y:
            med_y = np.median(np.vstack(interp_y), axis=0)
            ax_a.plot(grid_t, med_y, color=col, lw=2.2, ls=ls, label=leg_lab, zorder=4)

    ax_a.set_xlim(0, 61.5)
    ax_a.set_ylim(0, 780)
    ax_a.set_xlabel("Cumulative whole-ladder compute (10$^3$ GPU-s; 8$\\times$RTX 3090)")
    ax_a.set_ylabel("Bulk ESS of loading $N$ at 0.28 bar")
    ax_a.set_title("Mixing growth vs. whole-ladder budget (3 seeds/group)", loc="left")
    handles_a, labels_a = ax_a.get_legend_handles_labels()
    handles_a.extend([
        Line2D([0], [0], marker="o", color="none", mfc="#374151", mec="#374151", ms=5.5, label="Filled: endpoint 11/11 PASS"),
        Line2D([0], [0], marker="o", color="none", mfc="white", mec="#374151", ms=5.5, mew=1.2, label="Open: not converged in budget"),
    ])
    ax_a.legend(handles=handles_a, loc="upper left", fontsize=7.0, handlelength=2.0)
    panel_label(ax_a, "a")

    # ---- (b) Fixed-budget efficiency ESS_bulk(N) / total_ladder_GPU_s ----
    x_pos = np.arange(4, dtype=float)
    med_effs = []
    for idx, (grp, _, _, col, _, mk) in enumerate(_PRESUB_GROUPS):
        g_runs = [r for r in runs if r["group"] == grp]
        vals_k = [float(r["primary_028bar"]["ess_bulk_N_per_total_gpu_s"]) * 1000.0 for r in g_runs]
        med_v = float(gs_map[grp]["ess_bulk_N_per_total_gpu_s"]["median"]) * 1000.0
        med_effs.append(med_v)
        ax_b.bar(idx, med_v, width=0.52, color=col, alpha=0.28, edgecolor=col, lw=1.3, zorder=2)
        offsets = np.linspace(-0.14, 0.14, len(g_runs))
        for s_i, (dx, r, vk) in enumerate(zip(offsets, g_runs, vals_k), start=1):
            passed_fin = bool(r["primary_028bar"]["converged"])
            ax_b.scatter(
                [idx + dx], [vk],
                s=46, marker=mk,
                facecolor=col if passed_fin else "white",
                edgecolor=col, linewidths=1.4, zorder=5,
            )
        ax_b.text(
            idx, max(vals_k) + 0.55,
            f"med {med_v:.2f}\n[{min(vals_k):.2f}, {max(vals_k):.2f}]",
            ha="center", va="bottom", fontsize=7.3, color=C["ink"],
        )

    c_over_a = float(contrasts["C_over_A_REMC_effect_1B_only"]["eff_ratio"])
    d_over_b = float(contrasts["D_over_B_REMC_effect_1B_2B"]["eff_ratio"])
    b_over_a = float(contrasts["B_over_A_2B_effect_REMC_OFF"]["eff_ratio"])
    d_over_c = float(contrasts["D_over_C_2B_effect_REMC_ON"]["eff_ratio"])
    s12_sum = ps["surrogates"]["1B_2B"]["summary"]
    mae_cls = float(s12_sum["mae_2b_classical_holdout_kJ_mol"])
    mae_dep = float(s12_sum["mae_2b_deployed_holdout_kJ_mol"])
    ratio_txt = (
        f"Whole-ladder $\\mu$-REMC ESS$_N$/GPU-s gain:\n"
        f"  {c_over_a:.2f}$\\times$ (1B: C/A), {d_over_b:.2f}$\\times$ (1B+2B: D/B)\n"
        f"2B hold-out MAE: {mae_cls:.2f} $\\rightarrow$ {mae_dep:.2f} kJ mol$^{{-1}}$,\n"
        f"  matched-budget ESS$_N$/GPU-s: {b_over_a:.2f}$\\times$ (OFF), {d_over_c:.2f}$\\times$ (ON)\n"
        f"*1B-only retains unscaled classical guest–guest terms"
    )
    ax_b.text(
        0.025, 0.97, ratio_txt, transform=ax_b.transAxes,
        va="top", ha="left", fontsize=6.85, color=C["ink"], linespacing=1.22,
        bbox=dict(boxstyle="round,pad=0.32", fc="#F9FAFB", ec="#D1D5DB", lw=0.7),
    )
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels([g[2] for g in _PRESUB_GROUPS], fontsize=8.0)
    ax_b.set_ylim(0, 18.6)
    ax_b.set_ylabel("Whole-ladder ESS$_{\\rm bulk}(N)$ per 10$^3$ GPU-s")
    ax_b.set_title("0.28 bar efficiency (8$\\times$RTX 3090, 7,200 s = 57.6k GPU-s/run)", loc="left")
    panel_label(ax_b, "b")

    # ---- (c) 0.28 bar per-seed point estimates +/- MCSE_eff & endpoint vs 2-ckpt pass counts ----
    all_q = [float(r["primary_028bar"]["uptake_mmol_g"]) for r in runs]
    grand_med = float(np.median(all_q))
    ax_c.axhline(grand_med, color="#9CA3AF", lw=0.9, ls="--", zorder=1)

    for g_idx, (grp, _, _, col, _, mk) in enumerate(_PRESUB_GROUPS):
        g_runs = [r for r in runs if r["group"] == grp]
        offsets = np.linspace(-0.21, 0.21, len(g_runs))
        for s_i, (dx, r) in enumerate(zip(offsets, g_runs), start=1):
            p0 = r["primary_028bar"]
            ec = r["empirical_convergence"]
            q_val = float(p0["uptake_mmol_g"])
            mcse = float(p0["mcse_eff_mmol_g"])
            fin_pass = bool(p0["converged"])
            conf_pass = not bool(ec["right_censored"])
            ax_c.errorbar(
                g_idx + dx, q_val, yerr=mcse,
                fmt="none", ecolor=col, elinewidth=1.2, capsize=3.0, zorder=3,
            )
            ax_c.scatter(
                [g_idx + dx], [q_val],
                s=54 if conf_pass else 44,
                marker=mk,
                facecolor=col if fin_pass else "white",
                edgecolor="#111827" if conf_pass else col,
                linewidths=1.8 if conf_pass else 1.3,
                zorder=5,
            )
            if conf_pass:
                ax_c.scatter([g_idx + dx], [q_val + mcse + 0.021], s=48, marker="*", color=col, edgecolor="#111827", linewidths=0.4, zorder=6)
            ax_c.text(g_idx + dx, q_val - mcse - 0.012, f"S{s_i}", ha="center", va="top", fontsize=6.1, color=C["muted"])
        gs = gs_map[grp]
        n_fin = int(gs["n_passed_final"])
        n_conf = int(gs["n_passed_confirmed_empirical"])
        rh_med = float(gs["rhat_N"]["median"])
        ess_n_med = float(gs["ess_bulk_N"]["median"])
        ess_u_med = float(gs["ess_bulk_U"]["median"])
        ess_o_med = float(gs["ess_bulk_OMS"]["median"])
        ax_c.text(
            g_idx, 5.916,
            f"End {n_fin}/3 | 2-ckpt {n_conf}/3\n"
            f"$\\hat{{R}}_N$={rh_med:.3f}\n"
            f"ESS {ess_n_med:.0f}/{ess_u_med:.0f}/{ess_o_med:.0f}",
            ha="center", va="bottom", fontsize=5.85, color=C["ink"], linespacing=1.18,
            bbox=dict(boxstyle="round,pad=0.16", fc="#F9FAFB", ec="#E5E7EB", lw=0.6),
        )

    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels([g[2] for g in _PRESUB_GROUPS], fontsize=8.0)
    ax_c.set_xlim(-0.55, 3.55)
    ax_c.set_ylim(5.90, 6.49)
    ax_c.set_ylabel("CO$_2$ uptake at 0.28 bar (mmol g$^{-1}$, $\\pm$MCSE$_{\\rm eff}$)")
    ax_c.set_title("Per-seed fixed-budget point estimates & pass counts", loc="left")
    leg_c = [
        Line2D([0], [0], marker="*", color="none", mfc="#0F766E", mec="#111827", mew=0.6, ms=8.0, label="Star + black ring: 2-ckpt confirmed PASS (0/3, 0/3, 1/3, 1/3)"),
        Line2D([0], [0], marker="D", color="none", mfc="#0F766E", mec="#0F766E", mew=1.2, ms=5.5, label="Filled: endpoint 11/11 PASS (0/3, 0/3, 1/3, 2/3)"),
        Line2D([0], [0], marker="o", color="none", mfc="white", mec="#64748B", mew=1.2, ms=5.5, label="Open: not converged within 7,200 s budget"),
        Line2D([0], [0], color="#9CA3AF", lw=1.0, ls="--", label=f"12-run median point estimate: {grand_med:.2f} mmol g$^{{-1}}$"),
    ]
    ax_c.legend(handles=leg_c, loc="upper left", fontsize=6.45, ncol=1)
    panel_label(ax_c, "c")

    # ---- (d) Empirical time-to-convergence (confirmed 2 consecutive 5-min ckpts) + budget cutoff ----
    y_rows = []
    for grp, _, _, col, _, mk in _PRESUB_GROUPS:
        g_runs = [r for r in runs if r["group"] == grp]
        for r_i, r in enumerate(g_runs, start=1):
            y_rows.append((f"{grp}-S{r_i}", col, mk, r))

    yy = np.arange(len(y_rows))[::-1]
    ax_d.axvline(budget_wall_s, color="#DC2626", lw=1.2, ls="--", zorder=1)
    ax_d.text(budget_wall_s - 150, len(y_rows) - 0.15, "Budget cutoff:\n7,200 s (57.6k GPU-s)",
              ha="right", va="top", fontsize=6.8, color="#DC2626",
              bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.88))

    for yi, (lbl, col, mk, r) in zip(yy, y_rows):
        ec = r["empirical_convergence"]
        p0 = r["primary_028bar"]
        if not ec["right_censored"] and ec["time_to_convergence_s"] is not None:
            t_c = float(ec["time_to_convergence_s"])
            t_cf = float(ec["confirmation_time_s"])
            ax_d.hlines(yi, 0, t_c, color=col, lw=2.6, alpha=0.85, zorder=3)
            ax_d.hlines(yi, t_c, t_cf, color=col, lw=1.2, ls=":", zorder=3)
            ax_d.scatter([t_c], [yi], s=52, marker=mk, facecolor=col, edgecolor="#111827", linewidths=1.3, zorder=5)
            ax_d.scatter([t_cf], [yi], s=26, marker="|", color="#111827", linewidths=1.5, zorder=5)
            ax_d.text(
                t_cf + 150, yi, f"{t_c:,.0f} s (conf. {t_cf:,.0f} s)",
                va="center", ha="left", fontsize=6.6, color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.92), zorder=6,
            )
        elif ec.get("unconfirmed_terminal_pass"):
            ax_d.hlines(yi, 0, budget_wall_s, color=col, lw=2.0, ls="-", alpha=0.55, zorder=2)
            ax_d.scatter([budget_wall_s], [yi], s=48, marker=mk, facecolor=col, edgecolor=col, linewidths=1.2, zorder=5)
            ax_d.annotate("", xy=(budget_wall_s + 480, yi), xytext=(budget_wall_s + 40, yi),
                          arrowprops=dict(arrowstyle="->", color=col, lw=1.1))
            ax_d.text(budget_wall_s + 530, yi, "Endpoint pass @ 7,200 s (unconf.)", va="center", ha="left", fontsize=6.35, color=col)
        else:
            ax_d.hlines(yi, 0, budget_wall_s, color=col, lw=1.5, ls="--", alpha=0.40, zorder=2)
            ax_d.scatter([budget_wall_s], [yi], s=42, marker=mk, facecolor="white", edgecolor=col, linewidths=1.2, zorder=5)
            ax_d.annotate("", xy=(budget_wall_s + 480, yi), xytext=(budget_wall_s + 40, yi),
                          arrowprops=dict(arrowstyle="->", color=col, lw=1.0, alpha=0.75))
            n_g = sum(1 for v in p0["gates"].values() if v)
            ax_d.text(budget_wall_s + 530, yi, f"Not converged in budget ({n_g}/11)", va="center", ha="left", fontsize=6.35, color=C["muted"])

    # Horizontal group separators
    for sep in (2.5, 5.5, 8.5):
        ax_d.axhline(sep, color="#E5E7EB", lw=0.7, zorder=0)

    ax_d.set_yticks(yy)
    ax_d.set_yticklabels([row[0] for row in y_rows], fontsize=7.5)
    ax_d.set_xlim(0, 11650)
    ax_d.set_ylim(-0.7, len(y_rows) + 0.15)
    ax_d.set_xticks([0, 1800, 3600, 5400, 7200])
    ax_d.set_xlabel("Empirical time to 2-ckpt convergence, wall-clock (s; 8 GPUs)")
    ax_d.set_title("Per-seed convergence time & budget cutoff (no extrapolation)", loc="left")

    ax_d_top = ax_d.twiny()
    ax_d_top.set_xlim(ax_d.get_xlim()[0] * 8.0 / 1000.0, ax_d.get_xlim()[1] * 8.0 / 1000.0)
    ax_d_top.set_xticks([0, 14.4, 28.8, 43.2, 57.6])
    ax_d_top.set_xlabel("Whole-ladder compute (10$^3$ GPU-s)", fontsize=8.2, labelpad=4)
    ax_d_top.tick_params(labelsize=7.6)
    ax_d_top.spines["right"].set_visible(False)
    ax_d_top.spines["top"].set_visible(True)
    ax_d_top.spines["top"].set_linewidth(0.7)
    panel_label(ax_d, "d", y=1.12)

    out_p = out_dir / "fig2_parity_efficiency.png"
    fig.savefig(out_p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_p.name}")


def fig_si_presubmission_diagnostics(ps: dict, traces_npz_path: Path, out_dir: Path):
    """SI Figure: Empty vs high-loading branch convergence, REMC swap rates/RTs, and P(N)/P(OMS) distributions."""
    npz = np.load(traces_npz_path)
    runs = ps["runs"]
    conv_mmol_g = 1000.0 / MG_SUPER_MASS

    fig, axs = plt.subplots(2, 2, figsize=(12.4, 8.8))
    (ax_a, ax_b), (ax_c, ax_d) = axs
    fig.subplots_adjust(hspace=0.38, wspace=0.28)

    # ---- (a) Empty-start (boxes 0,1) vs high-loading-start (boxes 2,3) equilibration & cumulative convergence ----
    # Plot initial 3,500 total cycles (including 1,500 eq cycles) averaged over all 3 replicates per group
    n_show = 3500
    win = 40
    kernel = np.ones(win, dtype=float) / float(win)
    ax_a.axvline(1500, color="#DC2626", lw=0.95, ls=":", zorder=1)
    ax_a.text(1540, 8.85, "eq_cycles = 1,500\n(production starts)", fontsize=7.2, color="#DC2626", va="top", ha="left")
    ax_a.axhline(6.27, color="#9CA3AF", lw=0.85, ls="--", zorder=1)

    for grp, _, _, col, _, _ in _PRESUB_GROUPS:
        g_runs = [r for r in runs if r["group"] == grp]
        e_list, h_list = [], []
        for r in g_runs:
            arr_all = np.asarray(npz[f"{r['run_id']}_box_N_all_028"], dtype=float)
            k_len = min(n_show, arr_all.shape[0])
            init_n = r["primary_028bar"].get("init_N_per_box") or [0, 0, 36, 36]
            e0 = np.array([float(np.mean(init_n[:2])) * conv_mmol_g])
            h0 = np.array([float(np.mean(init_n[2:])) * conv_mmol_g])
            e_list.append(np.concatenate([e0, arr_all[:k_len, :2].mean(axis=1) * conv_mmol_g]))
            h_list.append(np.concatenate([h0, arr_all[:k_len, 2:].mean(axis=1) * conv_mmol_g]))
        e_mean = np.mean(np.vstack(e_list), axis=0)
        h_mean = np.mean(np.vstack(h_list), axis=0)
        e_sm = np.convolve(e_mean, kernel, mode="valid")
        h_sm = np.convolve(h_mean, kernel, mode="valid")
        xs_c = np.concatenate([[0.0], np.arange(len(e_sm), dtype=float) + win / 2.0])
        lw_g = 1.75 if grp in ("C", "D") else 1.2
        ax_a.plot(xs_c, np.concatenate([[e_mean[0]], e_sm]), color=col, lw=lw_g, ls="-", alpha=0.92)
        ax_a.plot(xs_c, np.concatenate([[h_mean[0]], h_sm]), color=col, lw=lw_g, ls="--", alpha=0.88)

    ax_a.set_xlim(0, n_show)
    ax_a.set_ylim(0, 9.2)
    ax_a.set_xlabel("Outer cycle index (3-replicate mean, 40-cycle smoothing)")
    ax_a.set_ylabel("Branch-averaged CO$_2$ uptake (mmol g$^{-1}$)")
    ax_a.set_title("Empty vs. high-loading branch convergence at 0.28 bar", loc="left")
    leg_a = [
        Line2D([0], [0], color=col, lw=1.7, label=f"Group {grp}") for grp, _, _, col, _, _ in _PRESUB_GROUPS
    ] + [
        Line2D([0], [0], color="#374151", lw=1.4, ls="-", label="Solid: empty start ($N_0=0$)"),
        Line2D([0], [0], color="#374151", lw=1.4, ls="--", label="Dashed: loaded start ($N_0\\approx 36$)"),
    ]
    ax_a.legend(handles=leg_a, loc="lower right", ncol=2, fontsize=6.9, frameon=True, facecolor="white", edgecolor="#E5E7EB")
    panel_label(ax_a, "a")

    # ---- (b) Adjacent pressure-pair swap acceptance rates (C vs D; A/B = 0%) + round-trip counts ----
    pair_labels = [p["pair"] for p in runs[0]["ladder_swaps"]["pairs"]]
    xp = np.arange(len(pair_labels), dtype=float)
    w = 0.36
    for idx, (grp, col, lab) in enumerate([("C", "#2563EB", "C: 1B-only + REMC ON"), ("D", "#0F766E", "D: 1B+2B + REMC ON")]):
        g_runs = [r for r in runs if r["group"] == grp]
        mat = np.array([[p["prod_rate"] * 100.0 for p in r["ladder_swaps"]["pairs"]] for r in g_runs], dtype=float)
        med = np.median(mat, axis=0)
        lo = med - np.min(mat, axis=0)
        hi = np.max(mat, axis=0) - med
        rts = [int(r["ladder_swaps"]["sum_round_trips_prod"]) for r in g_runs]
        ax_b.bar(
            xp + (idx - 0.5) * w, med, width=w * 0.92,
            yerr=[lo, hi], capsize=2.5, color=col, alpha=0.85,
            label=f"{lab} (RT: {min(rts)}–{max(rts)})",
        )
    ax_b.set_xticks(xp)
    ax_b.set_xticklabels(pair_labels, rotation=28, ha="right", fontsize=7.4)
    ax_b.set_ylim(0, 62)
    ax_b.set_ylabel("Production swap acceptance (%)")
    ax_b.set_title("Adjacent-rung $\\mu$-REMC swaps (A/B strictly 0 attempts)", loc="left")
    ax_b.legend(loc="upper left", fontsize=7.3)
    ax_b.text(
        0.98, 0.95,
        "A & B (REMC OFF):\n0 swap attempts,\n0 round trips\n(verified on all 6 runs)",
        transform=ax_b.transAxes, ha="right", va="top", fontsize=7.0, color=C["muted"],
        bbox=dict(boxstyle="round,pad=0.28", fc="#F9FAFB", ec="#D1D5DB", lw=0.7),
    )
    panel_label(ax_b, "b")

    # ---- (c) Pooled P(N) distribution at 0.28 bar ----
    for grp, leg_lab, _, col, ls, _ in _PRESUB_GROUPS:
        g_runs = [r for r in runs if r["group"] == grp]
        pooled = np.concatenate([np.asarray(npz[f"{r['run_id']}_N_028"]).ravel() for r in g_runs])
        vals, cnts = np.unique(pooled, return_counts=True)
        probs = cnts.astype(float) / float(pooled.size)
        ax_c.plot(vals, probs, color=col, lw=1.8 if grp in ("C", "D") else 1.3, ls=ls, label=f"{leg_lab} (max N={int(vals.max())})")
    ax_c.set_xlim(10, 42)
    ax_c.set_xlabel("Adsorbed CO$_2$ molecules $N$ in 1$\\times$1$\\times$2 supercell ($N_{\\max}=128$)")
    ax_c.set_ylabel("Probability $P(N)$")
    ax_c.set_title("Pooled production loading distribution $P(N)$ at 0.28 bar", loc="left")
    ax_c.legend(loc="upper left", fontsize=7.1)
    panel_label(ax_c, "c")

    # ---- (d) Pooled P(N_OMS) distribution at 0.28 bar ----
    for grp, leg_lab, _, col, ls, _ in _PRESUB_GROUPS:
        g_runs = [r for r in runs if r["group"] == grp]
        pooled = np.concatenate([np.asarray(npz[f"{r['run_id']}_oms_occ_028"]).ravel() for r in g_runs])
        vals, cnts = np.unique(pooled, return_counts=True)
        probs = cnts.astype(float) / float(pooled.size)
        ax_d.plot(vals, probs, color=col, lw=1.8 if grp in ("C", "D") else 1.3, ls=ls, label=f"{grp}: $\\langle N_{{\\rm OMS}}\\rangle={pooled.mean():.2f}$")
    ax_d.set_xlim(8, 36)
    ax_d.set_xlabel("Occupied Mg$^{2+}$ open metal sites $N_{\\rm OMS}$ (36 sites total)")
    ax_d.set_ylabel("Probability $P(N_{\\rm OMS})$")
    ax_d.set_title("Pooled OMS occupancy distribution at 0.28 bar", loc="left")
    ax_d.legend(loc="upper left", fontsize=7.4)
    panel_label(ax_d, "d")

    out_p = out_dir / "si_fig_presubmission_diagnostics.png"
    fig.savefig(out_p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_p.name}")


def fig_jctc3_transferability_finetuned(s, exp_dir: Path, out_dir: Path):
    """JCTC Fig. 3: Cross-MOF transferability, MACE-DAC-1 comparison, parity vs exp, & honest hold-out fidelity."""
    fig, axs = plt.subplots(2, 3, figsize=(14.4, 8.8))
    fig.subplots_adjust(hspace=0.46, wspace=0.58)
    (a, b, c), (d, e, f) = axs

    def _sim_pair(r2_keys, dac_keys, col_r2):
        specs = []
        _, r_r2 = _jctc_find_run(s, r2_keys)
        if r_r2 and r_r2.get("points"):
            p, q, er = _jctc_series(r_r2)
            specs.append((p, q, er, col_r2, "D", "Nested MC, r2SCAN (REMC)"))
        _, r_dac = _jctc_find_run(s, dac_keys)
        if r_dac and r_dac.get("points"):
            p, q, er = _jctc_series(r_dac)
            specs.append((p, q, er, C["macedac"], "*", "Nested MC, MACE-DAC-1"))
        return specs

    # (a) Ni-MOF-74 + CO2
    _iso_panel(
        a,
        [("nimof74_10.1021acs.jpcc.5b03700.Isotherm47.csv", C["exp1"], "o", "Exp. (10.1021/acs.jpcc.5b03700)"),
         ("nimof74_10.108015422119.2014.884507.Isotherm5.csv", C["exp2"], "s", "Exp. (10.1080/15422119.2014.884507)")],
        _sim_pair(["jctc_remc_nimof74_co2_r2scan", "exp2_nimof74_co2"], ["jctc_macedac_nimof74_co2"], "#0F766E"),
        (0, 1.06), (0, 8.5), "Ni-MOF-74 + CO$_2$, 298 K",
    )
    a.legend(loc="lower right", fontsize=6.8)
    panel_label(a, "a")

    # (b) Cu-BTC + CO2
    _iso_panel(
        b,
        [("hkust1_10.1002aic.15837.Isotherm7.csv", C["exp1"], "o", "Exp. (10.1002/aic.15837)"),
         ("hkust1_10.1007s1045001395272.Isotherm50.csv", C["exp2"], "s", "Exp. (10.1007/s10450-013-9527-2)")],
        _sim_pair(["jctc_remc_cubtc_co2_r2scan", "exp2_cubtc_co2"], ["jctc_macedac_cubtc_co2"], C["mercado"]),
        (0, 1.08), (0, 5.5), "Cu-BTC (HKUST-1) + CO$_2$, 298 K",
    )
    b.legend(loc="upper left", fontsize=6.8)
    panel_label(b, "b")

    # (c) IRMOF-1 + CO2
    _iso_panel(
        c,
        [("irmof1_10.1021acs.jctc.5b00093.Isotherm1.csv", C["exp1"], "o", "Exp. (10.1021/acs.jctc.5b00093)"),
         ("irmof1_10.1038srep12045.Isotherm2.csv", C["exp2"], "s", "Exp. (10.1038/srep12045)")],
        _sim_pair(["jctc_remc_irmof1_co2_r2scan", "exp2_irmof1_co2"], ["jctc_macedac_irmof1_co2"], "#6D28D9"),
        (0, 6.2), (0, 7.0), "IRMOF-1 + CO$_2$, 298 K (auto shift)",
    )
    c.legend(loc="upper left", fontsize=6.8)
    panel_label(c, "c")

    # (d) IRMOF-1 + CH4 (MP0+D3 REMC) + optional H2O inset if present
    _, r_ch4 = _jctc_find_run(s, ["jctc_remc_irmof1_ch4_mp0_d3", "exp2_irmof1_ch4_mp0d3"], mof="IRMOF-1", guest="ch4")
    if r_ch4 and r_ch4.get("points"):
        p, q, er = _jctc_series(r_ch4)
        d.errorbar(p, q, yerr=er, fmt="o-", color=C["mp0d3"], ms=5.5, lw=1.4, capsize=2.5, label="IRMOF-1 + CH$_4$ (MP0+D3, REMC)")
        for pi, qi in zip(p, q):
            d.text(pi, qi + 0.35, f"{qi:.2f}", ha="center", fontsize=7.4)
        d.set_xlim(0, max(5.6, float(np.max(p)) * 1.15))
        d.set_ylim(0, max(9.5, float(np.max(q)) * 1.35))
        d.set_xlabel("Pressure (bar)"); d.set_ylabel("CH$_4$ uptake (mmol g$^{-1}$)")
        d.set_title("Cross-guest: IRMOF-1 + CH$_4$, 298 K", loc="left")
        d.legend(loc="upper left", fontsize=7.0)
        # Optional H2O inset if MACE-DAC-1 H2O runs exist
        h2o_runs = []
        for hkey, hcol, hlab in [
            ("jctc_macedac_cubtc_h2o", C["h2o"], "Cu-BTC H$_2$O"),
            ("jctc_macedac_mgmof74_h2o", C["macedac"], "Mg-MOF-74 H$_2$O"),
        ]:
            _, hr = _jctc_find_run(s, [hkey])
            if hr and hr.get("points"):
                h2o_runs.append((hr, hcol, hlab))
        if h2o_runs:
            ins = d.inset_axes([0.48, 0.18, 0.49, 0.40])
            for hr, hcol, hlab in h2o_runs:
                hp, hq, he = _jctc_series(hr)
                ins.errorbar(hp * 1000.0, hq, yerr=he, fmt="o-", color=hcol, ms=3.5, lw=1.1, capsize=1.5, label=hlab)
            ins.set_xlabel("P(H$_2$O) (mbar)", fontsize=6.2)
            ins.set_ylabel("mmol g$^{-1}$", fontsize=6.2)
            ins.tick_params(labelsize=6.0)
            ins.set_title("MACE-DAC-1 H$_2$O", fontsize=6.6)
            ins.legend(fontsize=5.6, loc="upper left")
    else:
        _draw_unavailable(d, "Cross-guest: IRMOF-1 + CH$_4$", "CH$_4$ run not present in archive")
    panel_label(d, "d")

    # (e) Parity vs Experiment (CO2, r2SCAN vs MACE-DAC-1)
    ex = s.get("experimental_interpolated_mmol_g") or {}
    parity_rows = []
    parity_specs = [
        ("Mg-MOF-74", ["jctc_exp1_auto_mlip_r2scan_remc", "jctc_exp1_auto_mlip_r2scan_no_remc", "exp1_auto_mlip_r2scan_nod3"], C["auto"], "D", "r2SCAN"),
        ("Ni-MOF-74", ["jctc_remc_nimof74_co2_r2scan", "exp2_nimof74_co2"], "#0F766E", "s", "r2SCAN"),
        ("Cu-BTC", ["jctc_remc_cubtc_co2_r2scan", "exp2_cubtc_co2"], C["mercado"], "^", "r2SCAN"),
        ("IRMOF-1", ["jctc_remc_irmof1_co2_r2scan", "exp2_irmof1_co2_5bar"], "#6D28D9", "o", "r2SCAN"),
        ("Mg-MOF-74", ["jctc_macedac_mgmof74_co2"], C["macedac"], "*", "MACE-DAC-1"),
        ("Ni-MOF-74", ["jctc_macedac_nimof74_co2"], C["macedac"], "P", "MACE-DAC-1"),
        ("Cu-BTC", ["jctc_macedac_cubtc_co2"], C["macedac"], "X", "MACE-DAC-1"),
        ("IRMOF-1", ["jctc_macedac_irmof1_co2"], C["macedac"], "h", "MACE-DAC-1"),
    ]
    for mof, keys, col, mk, mlip_tag in parity_specs:
        _, run = _jctc_find_run(s, keys)
        if not run or mof not in ex:
            continue
        for pt in run.get("points", []):
            pkey = f"{float(pt['P_bar']):g}"
            exp_map = ex[mof].get(pkey) or ex[mof].get(str(pt["P_bar"]))
            if not exp_map:
                continue
            vals = [float(v) for v in exp_map.values() if v is not None]
            if not vals:
                continue
            parity_rows.append((
                f"{mof} ({mlip_tag})", float(pt["P_bar"]), float(np.mean(vals)), min(vals), max(vals),
                float(pt["uptake_mmol_g"]), float(pt.get("uptake_sem_mmol_g", 0.0) or 0.0),
                col, mk, pt.get("diag_status", "unknown"),
            ))
    if parity_rows:
        lims = (0.005, 30.0)
        xx = np.logspace(np.log10(lims[0]), np.log10(lims[1]), 50)
        e.plot(xx, xx, color="#374151", lw=0.8)
        e.plot(xx, 2 * xx, color="#9CA3AF", lw=0.7, ls="--"); e.plot(xx, 0.5 * xx, color="#9CA3AF", lw=0.7, ls="--")
        e.plot(xx, 10 * xx, color="#D1D5DB", lw=0.7, ls=":"); e.plot(xx, 0.1 * xx, color="#D1D5DB", lw=0.7, ls=":")
        seen = set()
        for lab, P, xm, xlo, xhi, ys, ye, col, mk, st in parity_rows:
            e.errorbar(
                xm, max(ys, lims[0] * 1.1), xerr=[[xm - xlo], [xhi - xm]], yerr=ye,
                fmt=mk, color=col, ms=5.8, mfc=col if st == "pass" else "white", mew=1.1,
                capsize=1.8, lw=0.9, label=lab if lab not in seen else None,
            )
            seen.add(lab)
        e.set_xscale("log"); e.set_yscale("log"); e.set_xlim(*lims); e.set_ylim(*lims)
        e.set_xlabel("Experiment (mmol g$^{-1}$, interpolated)"); e.set_ylabel("Nested MC (mmol g$^{-1}$)")
        e.set_title("Parity with experiment (CO$_2$, 298 K)", loc="left")
        e.legend(loc="upper left", fontsize=6.2, ncol=1)
    else:
        _draw_unavailable(e, "Parity with experiment (CO$_2$)", "No matched experimental points")
    panel_label(e, "e")

    # (f) Honest Hold-Out Surrogate Fidelity (1B hold-out vs 2B deployed hold-out vs 2B classical hold-out)
    dist_items = [
        (cid, d) for cid, d in sorted(s.get("distillation", {}).items())
        if not d.get("distill_1b_only")
    ]
    if dist_items:
        y = np.arange(len(dist_items))[::-1]
        m1 = [float(d.get("mae_1b_kJ_mol", 0.0) or 0.0) for _, d in dist_items]
        has_dep = all(d.get("mae_2b_deployed_holdout_kJ_mol") is not None for _, d in dist_items)
        has_cls = any(d.get("mae_2b_classical_holdout_kJ_mol") is not None for _, d in dist_items)
        m2_dep = [
            float(d["mae_2b_deployed_holdout_kJ_mol"]) if d.get("mae_2b_deployed_holdout_kJ_mol") is not None
            else float(d.get("mae_2b_kJ_mol", 0.0) or 0.0)
            for _, d in dist_items
        ]
        m2_cls = [
            float(d["mae_2b_classical_holdout_kJ_mol"]) if d.get("mae_2b_classical_holdout_kJ_mol") is not None else 0.0
            for _, d in dist_items
        ]
        r2 = [float(d.get("r2_1b", 0.0) or 0.0) for _, d in dist_items]
        tw = [float(d.get("distill_wall_s", 0.0) or 0.0) for _, d in dist_items]

        if has_cls:
            f.barh(y + 0.25, m1, height=0.23, color="#2F5D8A", label="1B hold-out MAE")
            f.barh(y, m2_dep, height=0.23, color="#D98E04",
                   label="2B deployed hold-out MAE" if has_dep else "2B fit MAE (mixed/legacy)")
            f.barh(y - 0.25, m2_cls, height=0.23, color="#9CA3AF", label="2B classical unscaled hold-out")
        else:
            f.barh(y + 0.18, m1, height=0.34, color="#2F5D8A", label="1B hold-out MAE")
            f.barh(y - 0.18, m2_dep, height=0.34, color="#D98E04",
                   label="2B deployed hold-out MAE" if has_dep else "2B in-sample fit MAE (legacy)")

        max_bar = max(max(m1, default=1.0), max(m2_dep, default=1.0), max(m2_cls, default=0.0))
        for yi, a1, rr, tt in zip(y, m1, r2, tw):
            f.text(a1 + 0.06, yi + (0.25 if has_cls else 0.18), f"R\u00b2={rr:.3f} ({tt:.0f}s)", va="center", fontsize=6.6)
        ylabs = [
            f"{Path(str(d.get('cif_path', ''))).stem}, {str(d.get('guest_name', '')).upper()}\n"
            f"{_short_model_label(str(d.get('model', '')))}" + ("+D3" if d.get("dispersion") else "")
            for _, d in dist_items
        ]
        f.set_yticks(y); f.set_yticklabels(ylabs, fontsize=6.6)
        f.set_xlim(0, max(4.2, max_bar * 1.45))
        f.set_xlabel("Hold-out MAE (kJ mol$^{-1}$)" if has_dep else "MAE (1B hold-out; 2B legacy, kJ mol$^{-1}$)")
        f.set_title("Surrogate hold-out fidelity", loc="left")
        f.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1 if has_cls else 2, fontsize=6.6)
    else:
        _draw_unavailable(f, "Surrogate hold-out fidelity", "No distillation caches in archive")
    panel_label(f, "f", x=-0.42)

    out_p = out_dir / "fig3_transferability_finetuned.png"
    fig.savefig(out_p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_p.name}")


def fig_jctc4_physics_diagnostics(s, exp_dir: Path, out_dir: Path):
    """JCTC Fig. 4: Functional sensitivity, Widom zero-loading, MACE/D3 decomposition, & 1x1x4 finite-size."""
    fig, axs = plt.subplots(2, 2, figsize=(11.8, 8.4))
    (ax_a, ax_b), (ax_c, ax_d) = axs
    fig.subplots_adjust(hspace=0.42, wspace=0.32)

    # (a) Target MLIP functional sensitivity at Mg-MOF-74 0.28 bar
    func_cands = [
        ("MP0\n(no D3)", ["exp1_mp0_nod3_0p28"], C["mp0"]),
        ("r2SCAN\n(no D3)", ["jctc_exp1_auto_mlip_r2scan_remc", "jctc_exp1_auto_mlip_r2scan_no_remc", "exp1_auto_mlip_r2scan_nod3"], C["auto"]),
        ("MACE-DAC-1", ["jctc_macedac_mgmof74_co2"], C["macedac"]),
        ("MP0\n+ D3", ["jctc_remc_mgmof74_co2_mp0_d3", "exp1_auto_mlip_mp0_d3"], C["mp0d3"]),
    ]
    func_items = []
    for lab, keys, col in func_cands:
        _, run = _jctc_find_run(s, keys)
        pt = _jctc_pt_at(run, 0.28)
        if pt is not None:
            func_items.append((lab, pt, col))

    if func_items:
        xs = np.arange(len(func_items))
        ax_a.axhspan(4.745, 6.937, color="#D1D5DB", alpha=0.45, lw=0, label="experiment (two datasets)")
        max_u = 8.0
        for xi, (lab, p, col) in zip(xs, func_items):
            up = float(p["uptake_mmol_g"]); se = float(p.get("uptake_sem_mmol_g", 0.0) or 0.0)
            max_u = max(max_u, up + se)
            ax_a.bar(xi, up, width=0.52, color=col, alpha=0.88, yerr=se, capsize=2.8, ecolor="#374151")
            qf = p.get("Qst_fluc_kJ_mol")
            qtxt = f"\nQ$_{{st}}$={qf:.1f} kJ/mol" if qf is not None else ""
            ax_a.text(xi, up + se + 0.45, f"{up:.2f} mmol/g{qtxt}", ha="center", va="bottom", fontsize=7.4)
        ax_a.set_xticks(xs); ax_a.set_xticklabels([x[0] for x in func_items])
        ax_a.set_ylabel("CO$_2$ uptake at 0.28 bar (mmol g$^{-1}$)")
        ax_a.set_ylim(0, max_u * 1.28 + 1.5)
        ax_a.set_title("Target MLIP functional sensitivity (Mg-MOF-74)", loc="left")
        ax_a.legend(loc="upper left", fontsize=7.4)
    else:
        _draw_unavailable(ax_a, "Target MLIP functional sensitivity", "No Mg-MOF-74 0.28 bar runs found")
    panel_label(ax_a, "a")

    # (b) Widom zero-loading Qst(0) & K_H
    wid_dict = s.get("widom") or {}
    wid_items = []
    for wk, wv in sorted(wid_dict.items()):
        if isinstance(wv, dict) and isinstance(wv.get("widom"), dict):
            m = _short_model_label(str(wv.get("model", wk)))
            lab = f"{m}\n" + ("+ D3" if wv.get("dispersion") else "(no D3)")
            col = C["macedac"] if "dac" in m.lower() else (C["mp0d3"] if wv.get("dispersion") else C["auto"])
            wid_items.append((lab, wv["widom"], col))
    if wid_items:
        xs = np.arange(len(wid_items))
        q0 = [float(w["Qst_0_kJ_mol"]) for _, w, _ in wid_items]
        q0e = [float(w.get("Qst_0_sem_kJ_mol", 0.0) or 0.0) for _, w, _ in wid_items]
        ax_b.bar(xs, q0, yerr=q0e, width=0.5, color=[x[2] for x in wid_items], alpha=0.88, capsize=2.8, ecolor="#374151")
        for i, (_, w, _) in enumerate(wid_items):
            kh = float(w["K_H_mmol_g_bar"]); khe = float(w.get("K_H_sem_mmol_g_bar", 0.0) or 0.0)
            kh_txt = f"{kh:.0f} \u00b1 {khe:.0f}" if kh > 100 else f"{kh:.2f} \u00b1 {khe:.2f}"
            ax_b.text(i, q0[i] + q0e[i] + 1.2, f"Q$_{{st}}$(0)={q0[i]:.1f}\u00b1{q0e[i]:.1f}\nK$_H$={kh_txt}", ha="center", fontsize=7.4)
        ax_b.set_xticks(xs); ax_b.set_xticklabels([x[0] for x in wid_items], fontsize=8.0)
        ax_b.set_ylim(0, max(q0) * 1.32 + 5.0)
        ax_b.set_ylabel("Q$_{st}$(0) (kJ mol$^{-1}$)")
        ax_b.set_title("Widom zero-loading insertion, Mg-MOF-74 + CO$_2$", loc="left")
    else:
        _draw_unavailable(ax_b, "Widom zero-loading insertion", "No Widom JSONs in archive")
    panel_label(ax_b, "b")

    # (c) MACE vs D3 decomposition across pressures
    decomp_reports = [
        (wk, wv) for wk, wv in sorted(wid_dict.items())
        if isinstance(wv, dict) and wv.get("mace_d3_decomposition")
    ]
    if decomp_reports:
        press_set = sorted({float(z["P_bar"]) for _, wv in decomp_reports for z in wv["mace_d3_decomposition"]})
        x = np.arange(len(press_set))
        n_rep = len(decomp_reports)
        w = 0.72 / max(1, n_rep)
        max_stack = 45.0
        for ri, (wk, wv) in enumerate(decomp_reports):
            dmap = {round(float(z["P_bar"]), 4): z for z in wv["mace_d3_decomposition"]}
            m_vals = [-float(dmap[round(p, 4)]["mean_u_mace_per_mol_kJ_mol"]) if round(p, 4) in dmap else 0.0 for p in press_set]
            d_vals = [-float(dmap[round(p, 4)]["mean_u_d3_per_mol_kJ_mol"]) if round(p, 4) in dmap else 0.0 for p in press_set]
            for mv, dv in zip(m_vals, d_vals):
                max_stack = max(max_stack, mv + dv)
            xpos = x + (ri - 0.5 * (n_rep - 1)) * w
            disp_on = bool(wv.get("dispersion"))
            m_lab = _short_model_label(str(wv.get("model", wk)))
            if disp_on:
                if "dac" in m_lab.lower():
                    ax_c.bar(xpos, m_vals, w * 0.92, color="#FDBA74", label=f"{m_lab}+D3: MACE")
                    ax_c.bar(xpos, d_vals, w * 0.92, bottom=m_vals, color=C["macedac"], label=f"{m_lab}+D3: D3")
                else:
                    ax_c.bar(xpos, m_vals, w * 0.92, color="#E8A39B", label=f"{m_lab}+D3: MACE")
                    ax_c.bar(xpos, d_vals, w * 0.92, bottom=m_vals, color=C["mp0d3"], label=f"{m_lab}+D3: D3")
            else:
                ax_c.bar(xpos, m_vals, w * 0.92, color=C["auto"], alpha=0.88, label=f"{m_lab}: MACE")
        ax_c.set_xticks(x)
        ax_c.set_xticklabels([f"{p:g}" for p in press_set], rotation=35, ha="right", fontsize=7.2)
        ax_c.set_xlabel("Pressure (bar)")
        ax_c.set_ylim(0, max_stack * 1.32)
        ax_c.set_ylabel(r"$-\langle U\rangle/N$ in equilibrium (kJ mol$^{-1}$)")
        ax_c.set_title("Intrinsic MACE vs D3 dispersion decomposition", loc="left")
        ax_c.legend(loc="upper center", ncol=2, fontsize=6.2)
    else:
        _draw_unavailable(ax_c, "MACE vs D3 decomposition", "No energy decomposition data in archive")
    panel_label(ax_c, "c")

    # (d) Independent supercell finite-size verification (1x1x2 vs 1x1x4 Mg-MOF-74 @ 0.28 bar)
    _, r_112 = _jctc_find_run(
        s, ["jctc_exp1_auto_mlip_r2scan_remc", "jctc_exp1_auto_mlip_r2scan_no_remc", "exp1_auto_mlip_r2scan_nod3"],
        mof="Mg-MOF-74", guest="co2", supercell=[1, 1, 2], proposer="auto_mlip", grid_only=False,
    )
    _, r_114 = _jctc_find_run(
        s, ["jctc_finitesize_mgmof74_1x1x4_co2_r2scan_0p28"],
        mof="Mg-MOF-74", guest="co2", supercell=[1, 1, 4],
    )
    p112 = _jctc_pt_at(r_112, 0.28)
    p114 = _jctc_pt_at(r_114, 0.28)
    if p112 is not None and p114 is not None:
        xs = np.array([0, 1], dtype=float)
        ups = [float(p112["uptake_mmol_g"]), float(p114["uptake_mmol_g"])]
        sems = [float(p112.get("uptake_sem_mmol_g", 0.0) or 0.0), float(p114.get("uptake_sem_mmol_g", 0.0) or 0.0)]
        ax_d.bar(xs, ups, yerr=[2 * e for e in sems], width=0.45, color=[C["auto"], C["remc"]], alpha=0.88, capsize=4, ecolor="#111827")
        for xi, u, se, pt in zip(xs, ups, sems, [p112, p114]):
            qf = pt.get("Qst_fluc_kJ_mol")
            qtxt = f"\nQ$_{{st}}$={qf:.1f} kJ/mol" if qf is not None else ""
            ax_d.text(xi, u + 2 * se + 0.2, f"{u:.3f} \u00b1 {se:.3f}{qtxt}", ha="center", fontsize=7.6)
        comb = math.hypot(sems[0], sems[1])
        z_fs = abs(ups[1] - ups[0]) / comb if comb > 1e-12 else 0.0
        ax_d.text(
            0.03, 0.95, f"Independent GCMC check\n$z$-score = {z_fs:.2f}",
            transform=ax_d.transAxes, va="top", fontsize=7.6,
            bbox=dict(boxstyle="round,pad=0.3", fc="#F9FAFB", ec="#D1D5DB", lw=0.7),
        )
        ax_d.set_xticks(xs); ax_d.set_xticklabels(["1\u00d71\u00d72 supercell\n(c = 13.7 \u00c5)", "1\u00d71\u00d74 supercell\n(c = 27.4 \u00c5)"])
        ax_d.set_ylabel("CO$_2$ uptake at 0.28 bar (mmol g$^{-1}$)")
        ax_d.set_ylim(0, max(ups) * 1.45 + 0.5)
        ax_d.set_title("Supercell finite-size verification (Mg-MOF-74)", loc="left")
    else:
        _draw_unavailable(ax_d, "Supercell finite-size verification", "1\u00d71\u00d74 finite-size run not present in archive")
    panel_label(ax_d, "d")

    out_p = out_dir / "fig4_physics_diagnostics.png"
    fig.savefig(out_p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_p.name}")


def fig_jctc5_convergence_scaling(s, out_dir: Path):
    """JCTC Fig. 5: Vehtari convergence audit, REMC swap rates, data-driven consistency, & GPU scaling/cost."""
    fig, axs = plt.subplots(2, 2, figsize=(13.6, 8.8))
    (ax_a, ax_b), (ax_c, ax_d) = axs
    fig.subplots_adjust(hspace=0.40, wspace=0.56)

    # (a) Vehtari convergence audit across ALL non-grid-only production runs
    n_pass = n_tot = 0
    color_cycle = [C["auto"], C["remc"], C["macedac"], C["mp0d3"], C["mercado"], "#6D28D9", C["uff"], C["direct"]]
    markers = ["D", "s", "^", "o", "v", "P", "X", "h"]
    mof_styles = {}
    for rkey, run in sorted(s.get("runs", {}).items()):
        if run.get("grid_only"):
            continue
        mof_tag = f"{run.get('mof', rkey)} ({run.get('guest', '').upper()})"
        if mof_tag not in mof_styles:
            idx = len(mof_styles)
            mof_styles[mof_tag] = (color_cycle[idx % len(color_cycle)], markers[idx % len(markers)])
        col, mk = mof_styles[mof_tag]
        for pt in run.get("points", []):
            st = pt.get("diag_status", "unknown")
            if st in ("insufficient", "unknown"):
                continue
            rhat = pt.get("N_rhat_max")
            ess_b = pt.get("N_ess_bulk")
            if rhat is None or ess_b is None:
                continue
            ok = st == "pass"
            n_tot += 1; n_pass += int(ok)
            ax_a.plot(
                max(float(ess_b), 1.0), max(float(rhat) - 1.0, 1e-4), mk,
                color=col, ms=6.0, mfc=col if ok else "white", mew=1.15, ls="none",
                label=mof_tag,
            )
    if n_tot > 0:
        # Deduplicate legend handles
        handles, labels = ax_a.get_legend_handles_labels()
        by_lab = dict(zip(labels, handles))
        ax_a.axvline(400, color="#B91C1C", lw=0.8, ls="--")
        ax_a.axhline(0.01, color="#B91C1C", lw=0.8, ls="--")
        ax_a.fill_between([400, 10000], 1e-4, 0.01, color="#DCFCE7", alpha=0.6, lw=0)
        ax_a.text(8500, 1.5e-4, "gate region", fontsize=7.0, color=C["pass"], ha="right")
        ax_a.set_xscale("log"); ax_a.set_yscale("log")
        ax_a.set_xlim(3, 10000); ax_a.set_ylim(1e-4, 1.5)
        ax_a.set_xlabel("Vehtari Bulk ESS of $N$"); ax_a.set_ylabel("$\\hat{R}_N - 1$")
        ax_a.set_title(f"Vehtari convergence audit ({n_pass}/{n_tot} pass all gates)", loc="left")
        ax_a.legend(by_lab.values(), by_lab.keys(), loc="upper right", fontsize=6.4)
        ax_a.text(0.02, 0.03, "filled: all gates pass\n(N, U, OMS, MCSE, drift)", transform=ax_a.transAxes, fontsize=6.6, color=C["muted"])
    else:
        _draw_unavailable(ax_a, "Vehtari convergence audit", "No estimable production points found")
    panel_label(ax_a, "a")

    # (b) REMC swap acceptance rates: flagship pair-by-pair + per-run summaries
    remc_runs = (s.get("audit") or {}).get("remc_runs") or {}
    remc_rows = []
    flagship_key = "jctc_exp1_auto_mlip_r2scan_remc"
    if flagship_key in remc_runs:
        for pr in remc_runs[flagship_key].get("pairs", []):
            if pr.get("acc_rate") is not None:
                plab = str(pr["pair_label"]).replace(" <-> ", "\u2194").replace(" bar", "")
                remc_rows.append((f"Mg-MOF-74 r2SCAN ({plab} bar)", 100.0 * float(pr["acc_rate"]), None, None, C["remc"]))
    for rkey, rinfo in sorted(remc_runs.items()):
        if rkey == flagship_key and len(remc_runs) > 1:
            continue
        rates = [100.0 * float(pr["acc_rate"]) for pr in rinfo.get("pairs", []) if pr.get("acc_rate") is not None]
        if not rates:
            continue
        run = s.get("runs", {}).get(rkey, {})
        mlip_short = _short_model_label(str(run.get("model", "")))
        if run.get("dispersion") and "dac" not in mlip_short.lower():
            mlip_short += "+D3"
        lab = f"{run.get('mof', rkey)} {str(run.get('guest', '')).upper()} ({mlip_short}, {len(rates)}p)"
        mean_v, lo_v, hi_v = float(np.mean(rates)), float(np.min(rates)), float(np.max(rates))
        col_b = C["macedac"] if "dac" in mlip_short.lower() else C["auto"]
        if len(remc_runs) == 1:
            for pr in rinfo.get("pairs", []):
                if pr.get("acc_rate") is not None:
                    remc_rows.append((f"{run.get('mof', rkey)} ({pr['pair_label']})", 100.0 * float(pr["acc_rate"]), None, None, C["remc"]))
        else:
            remc_rows.append((lab, mean_v, lo_v, hi_v, col_b))
    if remc_rows:
        yy = np.arange(len(remc_rows))[::-1]
        ax_b.axvspan(20.0, 50.0, color="#DCFCE7", alpha=0.55, lw=0, label="target 20\u201350% band")
        for yi, (lab, v, lo, hi, col_b) in zip(yy, remc_rows):
            ax_b.barh(yi, v, height=0.58, color=col_b, alpha=0.86)
            if lo is not None and hi is not None and (hi - lo) > 0.2:
                ax_b.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt="none", ecolor="#374151", elinewidth=0.9, capsize=2.2)
                txt = f"{v:.1f}% ({lo:.0f}\u2013{hi:.0f}%)"
                tx = hi + 1.5
            else:
                txt = f"{v:.1f}%"
                tx = v + 1.5
            ax_b.text(tx, yi, txt, va="center", fontsize=6.2, color=C["ink"])
        ax_b.set_yticks(yy); ax_b.set_yticklabels([r[0] for r in remc_rows], fontsize=6.4)
        max_x_b = max([r[3] if r[3] is not None else r[1] for r in remc_rows] + [60.0])
        ax_b.set_xlim(0, max_x_b + 24.0)
        ax_b.set_xlabel("REMC swap acceptance rate (%)")
        ax_b.set_title("Pressure-REMC swap acceptance (flagship + per-run mean)", loc="left")
        ax_b.legend(loc="upper right", fontsize=6.8)
    else:
        # Fallback: plot proposal symmetry ratio prod_ins_att / prod_del_att across runs
        sym_rows = []
        for rkey, run in sorted(s.get("runs", {}).items()):
            for pt in run.get("points", []):
                sr = pt.get("proposal_symmetry_ratio")
                if sr is not None and (int(pt.get("prod_ins_att", 0)) + int(pt.get("prod_del_att", 0))) >= 20:
                    sym_rows.append((f"{rkey} ({pt['P_bar']:g}b)", float(sr)))
        if sym_rows:
            sym_rows = sym_rows[:10]
            yy = np.arange(len(sym_rows))[::-1]
            ax_b.axvline(1.0, color="#374151", lw=0.9, ls="--")
            ax_b.barh(yy, [v for _, v in sym_rows], height=0.55, color=C["auto"], alpha=0.82)
            ax_b.set_yticks(yy); ax_b.set_yticklabels([x[0] for x in sym_rows], fontsize=6.5)
            ax_b.set_xlabel("Proposal symmetry ratio (ins_att / del_att)")
            ax_b.set_title("Insertion / deletion proposal symmetry audit", loc="left")
        else:
            _draw_unavailable(ax_b, "REMC swap & symmetry audit", "No REMC or exchange statistics available")
    panel_label(ax_b, "b")

    # (c) Data-driven internal consistency forest plot (ZERO hardcoded values)
    cons_groups = []
    # Group 1: Mg-MOF-74 0.28 bar empty vs loaded vs 1x1x4
    g1 = []
    for lab, keys, col in [
        ("Mg-MOF-74 0.28 bar, REMC empty", ["jctc_exp1_auto_mlip_r2scan_remc"], C["remc"]),
        ("Mg-MOF-74 0.28 bar, no-REMC empty", ["jctc_exp1_auto_mlip_r2scan_no_remc", "exp1_auto_mlip_r2scan_nod3"], C["auto"]),
        ("Mg-MOF-74 0.28 bar, loaded start", ["jctc_exp1_auto_mlip_r2scan_highload_0p28", "exp1_auto_mlip_r2scan_nod3_highload_0p28"], C["hl"]),
        ("Mg-MOF-74 0.28 bar, 1\u00d71\u00d74 supercell", ["jctc_finitesize_mgmof74_1x1x4_co2_r2scan_0p28"], "#0369A1"),
    ]:
        _, r = _jctc_find_run(s, keys)
        pt = _jctc_pt_at(r, 0.28)
        if pt is not None:
            de = pt.get("delta_eps_kJ_mol")
            sfx = f" (\u0394\u03b5={de:.2f})" if de is not None else ""
            g1.append((lab + sfx, pt, col))
    if g1:
        cons_groups.append(g1)

    # Group 2: IRMOF-1 1.0 bar Direct vs Nested
    g2 = []
    for lab, keys, col in [
        ("IRMOF-1 1.0 bar, Direct (unified HC)", ["jctc_parity_direct_irmof1_co2_1bar"], C["direct"]),
        ("IRMOF-1 1.0 bar, Nested MC", ["jctc_parity_nested_irmof1_co2_1bar", "jctc_remc_irmof1_co2_r2scan"], "#6D28D9"),
    ]:
        _, r = _jctc_find_run(s, keys)
        pt = _jctc_pt_at(r, 1.0)
        if pt is not None:
            g2.append((lab, pt, col))
    if g2:
        cons_groups.append(g2)

    if cons_groups:
        yy = 0.0
        max_x = 1.0
        for gi, grp in enumerate(cons_groups):
            if gi > 0:
                yy -= 0.55
            for lab, pt, col in grp:
                u = float(pt["uptake_mmol_g"]); se = float(pt.get("uptake_sem_mmol_g", 0.0) or 0.0)
                max_x = max(max_x, u + 2 * se)
                ax_c.errorbar(u, yy, xerr=2 * se, fmt="o", color=col, ms=5.2, capsize=2.5, lw=1.2)
                ax_c.text(0.05, yy + 0.28, lab, fontsize=6.8, color=C["ink"], va="bottom")
                ax_c.text(u + 2 * se + max(0.08, 0.02 * max_x), yy, f"{u:.2f}", fontsize=6.6, va="center", color=col)
                yy -= 1.15
        ax_c.set_yticks([]); ax_c.spines["left"].set_visible(False)
        ax_c.set_ylim(yy + 0.6, 0.9)
        ax_c.set_xlim(0.0, max_x * 1.25 + 0.5)
        ax_c.set_xlabel("Uptake (mmol g$^{-1}$), bars: \u00b12 s.e.")
        ax_c.set_title("Internal consistency checks (data-driven)", loc="left")
    else:
        _draw_unavailable(ax_c, "Internal consistency checks", "No consistency pairs present")
    panel_label(ax_c, "c", x=-0.06)

    # (d) Multi-GPU scaling or distillation-vs-sampling GPU cost breakdown
    p1 = s.get("runs", {}).get("phase3_scale_1gpu_pressure")
    p4 = s.get("runs", {}).get("phase3_scale_4gpu_pressure")
    r1 = s.get("runs", {}).get("phase3_scale_1gpu_replica")
    r4 = s.get("runs", {}).get("phase3_scale_4gpu_replica")
    if p1 and p4 and r1 and r4:
        cats = ["pressure queue\nend-to-end", "replica sharding\nend-to-end", "replica sharding\nMC only"]
        one = [float(p1["total_wall_s"]), float(r1["total_wall_s"]), float(r1["points"][0]["elapsed_s"])]
        four = [float(p4["total_wall_s"]), float(r4["total_wall_s"]), float(r4["points"][0]["elapsed_s"])]
        x = np.arange(3); w = 0.36
        ax_d.bar(x - w / 2, one, w, color="#9CA3AF", label="1 \u00d7 RTX 3090")
        ax_d.bar(x + w / 2, four, w, color="#0F766E", label="4 \u00d7 RTX 3090")
        for i in range(3):
            ax_d.text(x[i] - w / 2, one[i] + 2, f"{one[i]:.0f}", ha="center", fontsize=7.4)
            ax_d.text(x[i] + w / 2, four[i] + 2, f"{four[i]:.0f}\n{one[i] / max(four[i], 1e-6):.2f}\u00d7", ha="center", fontsize=7.4, color="#0F766E")
        ax_d.set_xticks(x); ax_d.set_xticklabels(cats, fontsize=8.0)
        ax_d.set_ylabel("Wall-clock time (s)"); ax_d.set_ylim(0, max(one) * 1.25)
        ax_d.set_title("Multi-GPU scaling throughput", loc="left")
        ax_d.legend(loc="upper right", fontsize=7.4)
    else:
        # Plot per-run GPU-time breakdown (distillation vs sampling) on log scale, sorted by total cost
        cost_rows = []
        for rkey, run in sorted(s.get("runs", {}).items()):
            if run.get("grid_only"):
                continue
            d_s = float((run.get("distillation") or {}).get("distill_wall_s", 0.0) or 0.0)
            samp_s = sum(float(p.get("gpu_s_sampling", p.get("elapsed_s", 0.0)) or 0.0) for p in run.get("points", []))
            if samp_s > 0.0:
                mlip_s = _short_model_label(str(run.get("model", "")))
                if run.get("dispersion") and "dac" not in mlip_s.lower():
                    mlip_s += "+D3"
                n_pts_r = len(run.get("points", []))
                if "1b_only" in rkey:
                    mode_s = "1B-only"
                elif "uff" in rkey:
                    mode_s = "UFF"
                elif "mercado" in rkey:
                    mode_s = "Mercado"
                elif "highload" in rkey:
                    mode_s = "loaded"
                elif "no_remc" in rkey:
                    mode_s = "no-REMC"
                elif "finitesize" in rkey:
                    mode_s = "1x1x4"
                elif "direct" in rkey:
                    mode_s = "Direct"
                elif run.get("parallel_mode") == "remc":
                    mode_s = "REMC"
                else:
                    mode_s = str(run.get("proposer", ""))
                disp_lab = f"{run.get('mof', rkey)} {str(run.get('guest', '')).upper()} ({mlip_s}, {mode_s}, {n_pts_r}P)"
                cost_rows.append((disp_lab, max(d_s, 0.0), samp_s))
        if cost_rows:
            cost_rows = sorted(cost_rows, key=lambda x: x[1] + x[2], reverse=True)[:10]
            yy = np.arange(len(cost_rows))[::-1]
            d_arr = [c[1] for c in cost_rows]
            s_arr = [c[2] for c in cost_rows]
            ax_d.barh(yy - 0.18, s_arr, height=0.34, color=C["auto"], alpha=0.88, label="Nested / Direct MC sampling")
            d_plot = [v if v > 0 else float("nan") for v in d_arr]
            ax_d.barh(yy + 0.18, d_plot, height=0.34, color=C["mercado"], label="One-time distillation")
            ax_d.set_xscale("log")
            max_tot = max(c[1] + c[2] for c in cost_rows)
            ax_d.set_xlim(10.0, max_tot * 18.0)
            for yi, (_, d_v, s_v) in zip(yy, cost_rows):
                pct = 100.0 * d_v / max(d_v + s_v, 1e-9)
                txt = f"{s_v:.0f}s MC" + (f" ({d_v:.0f}s dist, {pct:.1f}%)" if d_v > 0 else " (direct)")
                ax_d.text(max(s_v, d_v, 15.0) * 1.15, yi, txt, va="center", fontsize=6.2, color=C["ink"])
            ax_d.set_yticks(yy); ax_d.set_yticklabels([c[0] for c in cost_rows], fontsize=6.4)
            ax_d.set_xlabel("GPU-seconds (RTX 3090, log scale)")
            ax_d.set_title("Distillation vs sampling GPU-time budget (top 10 runs)", loc="left")
            ax_d.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=6.8)
        else:
            _draw_unavailable(ax_d, "Multi-GPU scaling & cost", "No timing breakdown data available")
    panel_label(ax_d, "d")

    out_p = out_dir / "fig5_convergence_scaling.png"
    fig.savefig(out_p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_p.name}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate publication figures (legacy, jctc, or presubmission mode)")
    p.add_argument("--mode", choices=["legacy", "jctc", "presubmission"], default="legacy", help="Figure generation mode")
    p.add_argument("--summary", default="", help="Path to summary JSON (defaults per mode)")
    p.add_argument("--presubmission-summary", default=str(PRESUB_SUMMARY), help="Path to matched_budget_summary.json")
    p.add_argument("--presubmission-traces", default=str(PRESUB_TRACES), help="Path to matched_budget_traces.npz")
    p.add_argument("--arch-npz", default=str(ARCH_NPZ), help="Path to arch_inputs_mgmof74_r2scan.npz")
    p.add_argument("--exp-dir", default=str(EXP), help="Path to experimental isotherms directory")
    p.add_argument("--out-dir", default=str(OUT), help="Output directory for PNG figures")
    return p.parse_args()


def main():
    global EXP, OUT, ARCH_NPZ
    args = parse_args()
    EXP = Path(args.exp_dir)
    OUT = Path(args.out_dir)
    ARCH_NPZ = Path(args.arch_npz)
    OUT.mkdir(parents=True, exist_ok=True)

    def _resolve_presub_summary(arg_path: str) -> Path:
        p = Path(arg_path)
        if p.is_file():
            return p
        if PRESUB_SUMMARY_PUBLISHED.is_file():
            return PRESUB_SUMMARY_PUBLISHED
        return p

    if args.mode == "presubmission":
        ps_path = _resolve_presub_summary(args.presubmission_summary)
        pt_path = Path(args.presubmission_traces)
        ps = json.loads(ps_path.read_text(encoding="utf-8"))
        if ARCH_NPZ.is_file():
            s_opt = json.loads(JCTC_SUMMARY.read_text(encoding="utf-8")) if JCTC_SUMMARY.is_file() else {}
            fig_jctc1_architecture(s_opt, ARCH_NPZ, OUT, presub=ps)
        fig_jctc2_presubmission_ablation(ps, OUT)
        if pt_path.is_file():
            fig_si_presubmission_diagnostics(ps, pt_path, OUT)
    elif args.mode == "jctc":
        summary_path = Path(args.summary) if args.summary else JCTC_SUMMARY
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        ps_path = _resolve_presub_summary(args.presubmission_summary)
        pt_path = Path(args.presubmission_traces)
        ps = json.loads(ps_path.read_text(encoding="utf-8")) if ps_path.is_file() else None
        fig_jctc1_architecture(s, ARCH_NPZ, OUT, presub=ps)
        if ps is not None:
            fig_jctc2_presubmission_ablation(ps, OUT)
            if pt_path.is_file():
                fig_si_presubmission_diagnostics(ps, pt_path, OUT)
        else:
            fig_jctc2_parity_efficiency(s, EXP, OUT)
        fig_jctc3_transferability_finetuned(s, EXP, OUT)
        fig_jctc4_physics_diagnostics(s, EXP, OUT)
        fig_jctc5_convergence_scaling(s, OUT)
    else:
        summary_path = Path(args.summary) if args.summary else SUMMARY
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        fig_architecture(s)
        fig_exp1(s)
        fig_exp2(s)
        fig_exp3(s)


if __name__ == "__main__":
    main()
