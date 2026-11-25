"""Figures for the Auto-MLIP / GPU-CBMC Nested MC preprint.

All numbers are read from docs/results/phase2_phase3_summary.json, docs/results/arch_inputs_mgmof74_r2scan.npz
and data/exp_isotherms/*.csv. Nothing is typed in by hand except values that were only ever recorded in logs
(they are marked LOG below).
"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "docs" / "results" / "phase2_phase3_summary.json"
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
    "auto": "#1F4E9C", "hl": "#2A9D8F", "grid": "#7FB3D5",
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
    axa.text(50, 7.6, "Cached by SHA-256 key; 56\u2013210 s per (framework, guest, MLIP) on one RTX 3090",
             ha="center", fontsize=8.3, color=C["muted"], style="italic")
    axa.text(50, 3.6, f"Example on the right (Mg-MOF-74, CO$_2$, r2SCAN): 15 % hold-out 1B R$^2$ = {d0['r2_1b']:.3f}, "
                      f"1B MAE = {d0['mae_1b_kJ_mol']:.2f} kJ/mol, 2B MAE = {d0['mae_2b_kJ_mol']:.2f} kJ/mol",
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
    m2 = [s["distillation"][c]["mae_2b_kJ_mol"] for c, _ in caches]
    r2 = [s["distillation"][c]["r2_1b"] for c, _ in caches]
    tw = [s["distillation"][c]["distill_wall_s"] for c, _ in caches]
    f.barh(y + 0.18, m1, height=0.34, color="#2F5D8A", label="1B hold-out MAE")
    f.barh(y - 0.18, m2, height=0.34, color="#D98E04", label="2B hold-out MAE")
    for yi, a1, rr, tt in zip(y, m1, r2, tw):
        f.text(a1 + 0.06, yi + 0.18, f"R\u00b2 = {rr:.3f}   ({tt:.0f} s)", va="center", fontsize=6.9)
    f.set_yticks(y); f.set_yticklabels([c[1] for c in caches], fontsize=7.0)
    f.set_xlim(0, 4.2)
    f.set_xlabel("Error on 15 % hold-out (kJ mol$^{-1}$)")
    f.set_title("Surrogate fidelity (caches used in production)", loc="left")
    f.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7.0)
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

    # (f) consistency checks
    def pt_at(key, P):
        return [p for p in s["runs"][key]["points"] if abs(p["P_bar"] - P) < 1e-6][0]
    rowsf = [
        ("Mg-MOF-74 0.28 bar, empty start (\u0394\u03b5 = 0.58)", pt_at("exp1_auto_mlip_r2scan_nod3", 0.28), C["auto"]),
        ("Mg-MOF-74 0.28 bar, loaded start (\u0394\u03b5 = 0.39)", pt_at("exp1_auto_mlip_r2scan_nod3_highload_0p28", 0.28), C["hl"]),
        None,
        ("IRMOF-1 5 bar, \u0394\u03b5 = 2.50 kJ/mol", pt_at("exp2_irmof1_co2_5bar", 5.0), "#6D28D9"),
        ("IRMOF-1 5 bar, \u0394\u03b5 = 0.00 kJ/mol (LOG)", {"uptake_mmol_g": 3.8506, "uptake_sem_mmol_g": 0.2123}, "#A78BFA"),
        ("IRMOF-1 5 bar, \u0394\u03b5 = 10.53 kJ/mol (stalled)", pt_at("exp2_irmof1_co2", 5.0), "#D1D5DB"),
        None,
        ("Mg-MOF-74 0.28 bar short run, 1 GPU (8 boxes)", pt_at("phase3_scale_1gpu_replica", 0.28), "#6B7280"),
        ("Mg-MOF-74 0.28 bar short run, 4 GPUs (2 boxes each)", pt_at("phase3_scale_4gpu_replica", 0.28), "#0F766E"),
    ]
    yy = 0
    for r in rowsf:
        if r is None:
            yy -= 0.5
            continue
        lab, p, col = r
        f.errorbar(p["uptake_mmol_g"], yy, xerr=2 * p["uptake_sem_mmol_g"], fmt="o", color=col, ms=5.2, capsize=2.5, lw=1.2)
        f.text(0.25, yy + 0.30, lab, fontsize=6.9, color=C["ink"], va="bottom")
        f.text(p["uptake_mmol_g"] + 2 * p["uptake_sem_mmol_g"] + 0.12, yy, f"{p['uptake_mmol_g']:.2f}", fontsize=6.6,
               va="center", color=col if col != "#D1D5DB" else C["muted"])
        yy -= 1.15
    f.set_yticks([]); f.spines["left"].set_visible(False)
    f.set_ylim(yy + 0.6, 0.9)
    f.set_xlabel("Uptake (mmol g$^{-1}$), bars: \u00b12 s.e.")
    f.set_xlim(0.2, 7.6)
    f.set_title("Internal consistency checks", loc="left")
    panel_label(f, "f", x=-0.06)

    fig.savefig(OUT / "fig4_exp3_physics_scaling.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved fig4_exp3_physics_scaling.png")


def main():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    fig_architecture(s)
    fig_exp1(s)
    fig_exp2(s)
    fig_exp3(s)


if __name__ == "__main__":
    main()
