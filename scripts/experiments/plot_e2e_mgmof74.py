"""Generate publication-grade figures for Mg-MOF-74 + CO2 E2E (OMS physics & 4-GPU scaling)."""

import csv
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent.parent.parent
E2E_DIR = ROOT / "build" / "e2e_mgmof74_co2"
ASSETS_DIR = ROOT / "assets"
EXP_DIR = ROOT / "data" / "exp_isotherms"


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.labelweight": "medium",
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.9,
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "text.color": "#0F172A",
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "grid.color": "#E2E8F0",
        "grid.linewidth": 0.7,
        "legend.frameon": True,
        "legend.facecolor": "#FFFFFF",
        "legend.edgecolor": "#E2E8F0",
        "legend.framealpha": 0.94,
    })


def load_exp_csv(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = float(r["P_bar"])
            u = float(r["uptake_mmol_g"])
            if 0.001 <= p <= 0.35:
                rows.append((p, u))
    rows.sort()
    return np.array([x[0] for x in rows]), np.array([x[1] for x in rows])


def load_cli_csv(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((
                float(r["P_bar"]),
                float(r["uptake_mmol_g"]),
                float(r["uptake_sem_mmol_g"]),
                float(r["Qst_kJ_mol"]),
            ))
    rows.sort()
    return rows


def plot_oms_physics():
    p_jpcc, u_jpcc = load_exp_csv(EXP_DIR / "mgmof74_10.1021acs.jpcc.7b08239.Isotherm1.csv")
    p_dt, u_dt = load_exp_csv(EXP_DIR / "mgmof74_10.1039c5dt03137k.Isotherm1.csv")
    cli_uff = load_cli_csv(E2E_DIR / "cfg_uff.csv")
    cli_merc = load_cli_csv(E2E_DIR / "cfg_mercado.csv")

    mlip_merc4 = json.loads((E2E_DIR / "mlip_r2scan_4gpu.json").read_text(encoding="utf-8"))
    mlip_uff4 = json.loads((E2E_DIR / "mlip_r2scan_uffprop_4gpu.json").read_text(encoding="utf-8"))
    oms_geom = json.loads((E2E_DIR / "oms_geometry.json").read_text(encoding="utf-8"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.6), gridspec_kw={"wspace": 0.25})
    fig.suptitle(
        "Mg-MOF-74 + CO₂ (298 K): Open-Metal-Site (OMS) Adsorption Physics & Microscopic Coordination",
        fontsize=13, fontweight="bold", y=1.01, color="#0F172A",
    )

    # --- Panel (a): Adsorption Isotherm & Qst ---
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(True, which="major", alpha=0.8)

    p_band = np.geomspace(0.0025, 0.20, 120)
    u_hi = np.interp(np.log10(p_band), np.log10(p_jpcc), u_jpcc)
    u_lo = np.interp(np.log10(p_band), np.log10(p_dt), u_dt)
    ax1.fill_between(p_band, u_lo, u_hi, color="#94A3B8", alpha=0.22, label="NIST ISODB Exp. band (298 K)")
    ax1.plot(p_jpcc, u_jpcc, "s--", color="#334155", lw=1.4, ms=4.5, label="Exp. JPCC 2017")
    ax1.plot(p_dt, u_dt, "^:", color="#64748B", lw=1.4, ms=4.5, label="Exp. Dalton 2015")

    ax1.plot(
        [r[0] for r in cli_uff], [r[1] for r in cli_uff],
        "o--", color="#64748B", lw=1.7, ms=5.0, markerfacecolor="#FFFFFF",
        label="Classical UFF+REPEAT (Qst 26.2–27.0 kJ/mol)",
    )
    ax1.plot(
        [r[0] for r in cli_merc], [r[1] for r in cli_merc],
        "v-.", color="#0F766E", lw=1.8, ms=5.5,
        label="Classical Mercado 2016 (Qst 34.8–36.6 kJ/mol)",
    )

    pts_m = sorted(mlip_merc4["points"], key=lambda x: x["P_bar"])
    ax1.errorbar(
        [pt["P_bar"] for pt in pts_m],
        [pt["uptake_mmol_g"] for pt in pts_m],
        yerr=[pt["uptake_sem_mmol_g"] for pt in pts_m],
        fmt="o-", color="#1D4ED8", lw=2.2, ms=6.5, capsize=3.5, elinewidth=1.3,
        label="MLIP r2SCAN+D3 (Mercado prop., MolQst 42.9–45.9)",
    )

    pts_u = sorted(mlip_uff4["points"], key=lambda x: x["P_bar"])
    ax1.errorbar(
        [pt["P_bar"] for pt in pts_u],
        [pt["uptake_mmol_g"] for pt in pts_u],
        yerr=[pt["uptake_sem_mmol_g"] for pt in pts_u],
        fmt="D-", color="#B45309", lw=2.0, ms=6.0, capsize=3.5, elinewidth=1.3,
        label="MLIP r2SCAN+D3 (Generic UFF+QEq prop., MolQst 37.0–43.5)",
    )

    ax1.axhline(8.24, color="#475569", ls=":", lw=1.2)
    ax1.text(0.00215, 8.45, "1:1 Mg²⁺ OMS saturation (~8.24 mmol/g)", fontsize=8.4, color="#334155")

    ax1.annotate(
        "MLIP+D3 captures OMS affinity,\nyet foundation model overbinds Exp. ~2–4×",
        xy=(0.05, 10.14), xytext=(0.0023, 9.75),
        fontsize=8.2, color="#1E3A8A",
        bbox=dict(boxstyle="round,pad=0.30", fc="#EFF6FF", ec="#BFDBFE", lw=0.8),
        arrowprops=dict(arrowstyle="->", color="#1D4ED8", lw=1.0, connectionstyle="arc3,rad=-0.10"),
    )

    ax1.set_xscale("log")
    ax1.set_xlim(0.0020, 0.23)
    ax1.set_ylim(-0.4, 17.6)
    ax1.set_xlabel("CO₂ Pressure (bar, log scale)")
    ax1.set_ylabel("CO₂ Uptake (mmol / g)")
    ax1.set_title("(a) 298 K Adsorption Isotherm vs. NIST ISODB & Classical FF", loc="left", pad=10)
    ax1.legend(loc="upper left", ncol=1, fontsize=7.6, handlelength=2.0)

    # --- Panel (b): Microscopic Mg2+---O(CO2) Coordination Density (0.05 bar) ---
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, which="major", alpha=0.8)

    d_grid = np.linspace(1.95, 4.45, 400)
    ax2.axvspan(1.95, 2.60, color="#DBEAFE", alpha=0.38, zorder=0)
    ax2.axvline(2.60, color="#475569", ls="--", lw=1.2, zorder=1)
    ax2.text(2.275, 1.73, "OMS End-On Zone\n(d < 2.60 Å)", ha="center", va="top",
             fontsize=8.5, color="#1E3A8A", fontweight="bold")

    series = [
        ("Classical UFF+REPEAT (0% < 2.60 Å)", np.array(oms_geom["classical_0.05bar"]["uff"]["d_mg_o"]), "#64748B", "--", 1.8),
        ("MLIP 4-GPU (UFF+QEq prop., 29.4% < 2.60 Å)", np.array(oms_geom["mlip_4gpu_uff"]["0.05"]["d_mg_o"]), "#B45309", "-", 2.1),
        ("Classical Mercado 2016 (49.1% < 2.60 Å)", np.array(oms_geom["classical_0.05bar"]["mercado"]["d_mg_o"]), "#0F766E", "-.", 1.7),
        ("MLIP 4-GPU (Mercado prop., 50.9% < 2.60 Å)", np.array(oms_geom["mlip_4gpu_mercado"]["0.05"]["d_mg_o"]), "#1D4ED8", "-", 2.4),
    ]

    for lbl, arr, col, ls, lw in series:
        bw = 0.28 if len(arr) < 15 else "scott"
        kde = gaussian_kde(arr, bw_method=bw)
        y = kde(d_grid)
        ax2.plot(d_grid, y, linestyle=ls, color=col, lw=lw, label=lbl)
        if "MLIP 4-GPU" in lbl:
            ax2.fill_between(d_grid, 0, y, color=col, alpha=0.10)

    # Callout on bent coordination angle pointing cleanly to the MLIP Mercado peak
    ax2.annotate(
        "OMS peak ~2.25–2.50 Å\n∠Mg–O–C med: 135.6° (Mercado)\n                   134.0° (UFF+QEq)\nMg_occ: 61.8% vs 22.2%",
        xy=(2.46, 0.92), xytext=(2.82, 0.92),
        fontsize=8.1, color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.32", fc="#F8FAFC", ec="#CBD5E1", lw=0.8),
        arrowprops=dict(arrowstyle="->", color="#1D4ED8", lw=1.0, connectionstyle="arc3,rad=0.12"),
    )

    ax2.set_xlim(1.95, 4.45)
    ax2.set_ylim(0.0, 1.82)
    ax2.set_xlabel("Nearest Periodic Mg²⁺···O(CO₂) Distance (Å)")
    ax2.set_ylabel("Normalized Probability Density (Å⁻¹)")
    ax2.set_title("(b) Microscopic Mg²⁺···O(CO₂) Coordination at 0.05 bar", loc="left", pad=10)
    ax2.legend(loc="upper right", fontsize=7.8)

    out1 = ASSETS_DIR / "e2e_mgmof74_oms_physics.png"
    fig.savefig(out1, dpi=300, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"Saved {out1}")


def plot_multigpu_scaling():
    m1 = json.loads((E2E_DIR / "mlip_r2scan_1gpu.json").read_text(encoding="utf-8"))
    m4 = json.loads((E2E_DIR / "mlip_r2scan_4gpu.json").read_text(encoding="utf-8"))
    u4 = json.loads((E2E_DIR / "mlip_r2scan_uffprop_4gpu.json").read_text(encoding="utf-8"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2), gridspec_kw={"wspace": 0.27})
    fig.suptitle(
        "Mg-MOF-74 + CO₂ E2E Performance: 1× vs 4× RTX 3090 (GPUs 0–3) & Statistical Parity",
        fontsize=13, fontweight="bold", y=1.01, color="#0F172A",
    )

    # --- Panel (a): Stacked Wall Time Breakdown ---
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(True, axis="y", alpha=0.8)

    runs = [
        ("1 GPU (GPU 0)\nMercado prop.", m1),
        ("4 GPUs (0–3)\nMercado prop.", m4),
        ("4 GPUs (0–3)\nUFF+QEq prop.", u4),
    ]
    labels = [r[0] for r in runs]
    x = np.arange(len(runs))
    width = 0.48

    mc_times = []
    other_times = []
    tot_walls = []
    ess_rates = []
    for _, d in runs:
        tot_w = float(d["total_wall_s"])
        pts = d["points"]
        tot_ess = sum(float(p["ess_N"]) for p in pts)
        if len(d.get("gpus", [])) > 1:
            mc_w = max(float(p["elapsed_s"]) for p in pts)
        else:
            mc_w = sum(float(p["elapsed_s"]) for p in pts)
        mc_times.append(mc_w)
        other_times.append(max(0.0, tot_w - mc_w))
        tot_walls.append(tot_w)
        ess_rates.append(tot_ess / tot_w)

    ax1.bar(x, mc_times, width, color="#1D4ED8", label="Nested MC Production + Eq Loop (bottleneck)")
    ax1.bar(x, other_times, width, bottom=mc_times, color="#93C5FD", label="3D Grid Build + MACE Init + 3-Stage Shift Pilot")

    for i in range(len(runs)):
        speed_txt = ""
        if i == 1:
            sp = tot_walls[0] / tot_walls[1]
            sp_mc = mc_times[0] / mc_times[1]
            speed_txt = f"\n({sp:.2f}× total | {sp_mc:.2f}× MC)"
        elif i == 2:
            sp = tot_walls[0] / tot_walls[2]
            speed_txt = f"\n({sp:.2f}× vs 1-GPU)"
        ax1.text(
            x[i], tot_walls[i] + 8,
            f"{tot_walls[i]:.1f} s{speed_txt}\nESS/s = {ess_rates[i]:.2f}",
            ha="center", va="bottom", fontsize=9.0, fontweight="bold" if i > 0 else "normal",
            color="#0F172A",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 415)
    ax1.set_ylabel("End-to-End Wall Time (s) for 4-Point Isotherm")
    ax1.set_title("(a) 1× vs 4× RTX 3090 Wall-Time Breakdown", loc="left", pad=10)
    ax1.legend(loc="upper right", fontsize=8.3)

    # --- Panel (b): Statistical Parity (1-GPU vs 4-GPU Mercado Proposer) ---
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, alpha=0.8)

    p1_map = {round(float(p["P_bar"]), 6): p for p in m1["points"]}
    p4_map = {round(float(p["P_bar"]), 6): p for p in m4["points"]}
    pressures = sorted(p1_map.keys())

    u1 = [p1_map[p]["uptake_mmol_g"] for p in pressures]
    s1 = [p1_map[p]["uptake_sem_mmol_g"] for p in pressures]
    u4_v = [p4_map[p]["uptake_mmol_g"] for p in pressures]
    s4 = [p4_map[p]["uptake_sem_mmol_g"] for p in pressures]

    lim = (1.0, 12.6)
    ax2.plot(lim, lim, "--", color="#64748B", lw=1.3, label="Exact parity (y = x)")
    ax2.errorbar(
        u1, u4_v, xerr=s1, yerr=s4,
        fmt="o", color="#1D4ED8", ecolor="#60A5FA", ms=7.0, capsize=3.5, elinewidth=1.4,
        label="Mg-MOF-74 + CO₂ (40 pilot + 80 eq + 200 prod)",
    )

    label_pos = {
        0.003: (1.55, 4.35),
        0.01: (6.35, 3.90),
        0.05: (9.30, 7.50),
        0.15: (7.95, 11.20),
    }
    for p in pressures:
        pt1, pt4 = p1_map[p], p4_map[p]
        z = abs(pt1["uptake_mmol_g"] - pt4["uptake_mmol_g"]) / math.sqrt(
            pt1["uptake_sem_mmol_g"] ** 2 + pt4["uptake_sem_mmol_g"] ** 2 + 1e-12
        )
        tx, ty = label_pos[p]
        ax2.annotate(
            f"{p:g} bar\n(Δ = {z:.2f}σ)",
            xy=(pt1["uptake_mmol_g"], pt4["uptake_mmol_g"]),
            xytext=(tx, ty),
            fontsize=8.5, color="#1E3A8A",
            arrowprops=dict(arrowstyle="->", color="#64748B", lw=0.9),
        )

    ax2.text(
        0.04, 0.95,
        "4-GPU pressure pool reproduces\n1-GPU isotherm within ≤ 2.50σ\n(0.39σ, 2.50σ, 0.81σ, 0.00σ)",
        transform=ax2.transAxes, va="top", ha="left", fontsize=8.4, color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.35", fc="#F8FAFC", ec="#CBD5E1", lw=0.8),
    )

    ax2.set_xlim(*lim)
    ax2.set_ylim(*lim)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("1-GPU (GPU 0) MLIP Uptake (mmol / g)")
    ax2.set_ylabel("4-GPU (GPUs 0–3) MLIP Uptake (mmol / g)")
    ax2.set_title("(b) 1-GPU vs 4-GPU Reweighting Parity", loc="left", pad=10)
    ax2.legend(loc="lower right", fontsize=8.3)

    out2 = ASSETS_DIR / "e2e_mgmof74_multigpu_scaling.png"
    fig.savefig(out2, dpi=300, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"Saved {out2}")


if __name__ == "__main__":
    setup_style()
    plot_oms_physics()
    plot_multigpu_scaling()
