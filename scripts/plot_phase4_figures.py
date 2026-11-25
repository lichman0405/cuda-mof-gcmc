"""Generate publication-grade Phase 4 figures from verified Phase 2-3 JSON outputs."""

import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = ROOT / "docs" / "results" / "phase2_phase3_summary.json"
EXP_DIR = ROOT / "data" / "exp_isotherms"
ASSETS_DIR = ROOT / "assets"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})


def load_csv_isotherm(filename: str):
    xs, ys = [], []
    with (EXP_DIR / filename).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["P_bar"]))
            ys.append(float(row["uptake_mmol_g"]))
    return np.array(xs), np.array(ys)


def extract_series(summary: dict, run_key: str):
    pts = sorted(summary["runs"][run_key]["points"], key=lambda x: x["P_bar"])
    p = np.array([pt["P_bar"] for pt in pts], dtype=float)
    q = np.array([pt["uptake_mmol_g"] for pt in pts], dtype=float)
    e = np.array([pt["uptake_sem_mmol_g"] for pt in pts], dtype=float)
    return p, q, e, pts


def plot_architecture(summary: dict):
    fig, ax = plt.subplots(figsize=(14.5, 6.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")
    ax.grid(False)

    ax.text(
        50, 48.8,
        "Auto-MLIP + 128-Thread GPU Rosenbluth CBMC Nested Monte Carlo Architecture",
        ha="center", va="center", fontsize=14.5, fontweight="bold", color="#0F172A"
    )
    ax.text(
        50, 45.6,
        "Zero Manual Force Fields | Environment-Aware 1B+2B Self-Distillation -> Exact Many-Body MLIP Reweighting -> Multi-GPU Scaling",
        ha="center", va="center", fontsize=10.2, color="#475569"
    )

    boxes = [
        (
            2.5, 11, 20.5, 30, "#EFF6FF", "#1D4ED8",
            "1. Universal Input",
            "Any Rigid MOF CIF\n(230 Space Groups)\n+ Any Rigid Guest\n(<=12 sites, <=6 chans)\n\nASE Symmetry Expand\nScreened Ewald QEq\nSymmetric Channel Map",
        ),
        (
            27.0, 11, 21.5, 30, "#ECFDF5", "#047857",
            "2. Online Self-Distill",
            "Zero-Manual-Typing 1B+2B\n(mlip_distill.py)\n\n1B: 1200 MLIP Poses\n  Non-uniform OMS RBFs\n  + Ewald Null-Space q\n  R2 = 0.985-0.991\n2B: 192 Dimers -> Pore LUT\n53-210 s (Disk-Cached)",
        ),
        (
            52.5, 11, 21.5, 30, "#FFFBEB", "#B45309",
            "3. CUDA GPU-CBMC",
            "Zero-I/O Proposer ABI\n(src/proposer_dll.cu)\n\n128-Thread Block-Parallel\n  Rosenbluth Trials (K=128)\n  O(h4) Tricubic Grids\n  + Scaled 2B Pore LUT\nReversible Inner Burst\n(k = 16 steps / outer eval)",
        ),
        (
            78.0, 11, 19.5, 30, "#F5F3FF", "#6D28D9",
            "4. Exact Nested MC",
            "Exact Target MLIP\n(MACE + GPU DFT-D3)\n\nMedian Delta-Eps Shift\nUnbiased Metropolis\nR-hat / ESS / MCSE Gates\nBidirectional Check\nMulti-GPU: Pressure &\n  Replica (3.25x MC)",
        ),
    ]

    for x, y, w, h, fc, ec, title, body in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.6,rounding_size=1.8",
            facecolor=fc, edgecolor=ec, linewidth=2.0
        )
        ax.add_patch(patch)
        hdr = FancyBboxPatch(
            (x + 0.3, y + h - 5.2), w - 0.6, 4.9,
            boxstyle="round,pad=0.2,rounding_size=1.2",
            facecolor=ec, edgecolor=ec, linewidth=0.5
        )
        ax.add_patch(hdr)
        ax.text(x + w / 2, y + h - 2.75, title, ha="center", va="center",
                fontsize=10.8, fontweight="bold", color="white")
        ax.text(x + 1.0, y + h - 7.0, body, ha="left", va="top",
                fontsize=9.1, color="#1E293B", linespacing=1.42)

    for x_start, x_end in [(23.7, 26.4), (49.2, 51.9), (74.7, 77.4)]:
        arr = FancyArrowPatch(
            (x_start, 26), (x_end, 26),
            arrowstyle="-|>,head_length=7,head_width=4.5",
            linewidth=2.4, color="#334155"
        )
        ax.add_patch(arr)

    # Return arrow from Stage 4 back to Stage 3 for outer Nested MC reweighting
    fb = FancyArrowPatch(
        (87.8, 10.2), (63.2, 10.2),
        connectionstyle="arc3,rad=0.14",
        arrowstyle="-|>,head_length=6,head_width=4",
        linewidth=1.8, linestyle="--", color="#6D28D9"
    )
    ax.add_patch(fb)
    ax.text(75.5, 4.4, "Exact MH Reweighting on Zero-I/O Device Pointers (Roll Back on Reject)",
            ha="center", va="center", fontsize=9.2, fontweight="bold", color="#6D28D9")

    # Bottom verified throughput note
    note = (
        "Verified Mg-MOF-74 CO2 (298 K, 0.28 bar):  57.5% insertion acc vs 3.65% Direct MLIP (15.8x)  |  "
        "ESJD_N 0.131 vs 0.016 (8.0x)  |  1.0 bar r2SCAN No-D3: 8.148 vs 8.170 mmol/g Exp (0.26% dev)"
    )
    ax.text(50, 1.4, note, ha="center", va="center", fontsize=9.2,
            color="#0F172A", bbox=dict(boxstyle="round,pad=0.4", facecolor="#F1F5F9", edgecolor="#CBD5E1"))

    fig.tight_layout()
    out = ASSETS_DIR / "architecture_auto_mlip_cbmc.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


def plot_fig1(summary: dict):
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.2))
    axA, axB, axC, axD = axes.ravel()

    # Panel A: Mg-MOF-74 Isotherms & Method Comparison
    p_e1, q_e1 = load_csv_isotherm("mgmof74_10.1021acs.jpcc.7b08239.Isotherm1.csv")
    m1 = (p_e1 >= 0.003) & (p_e1 <= 1.2)
    p_e2, q_e2 = load_csv_isotherm("mgmof74_10.1039c5dt03137k.Isotherm1.csv")
    m2 = (p_e2 >= 0.003) & (p_e2 <= 1.2)

    axA.semilogx(p_e1[m1], q_e1[m1], "o--", color="#475569", ms=5, lw=1.6, label="Exp (jpcc.7b08239)")
    axA.semilogx(p_e2[m2], q_e2[m2], "s:", color="#94A3B8", ms=4.5, lw=1.4, label="Exp (c5dt03137k)")

    p_r2, q_r2, e_r2, _ = extract_series(summary, "exp1_auto_mlip_r2scan_nod3")
    axA.errorbar(p_r2, q_r2, yerr=e_r2, fmt="-D", color="#1D4ED8", lw=2.3, ms=7, capsize=3.5,
                 label="Auto-MLIP + CBMC (r2SCAN No-D3)", zorder=5)

    p_go, q_go, e_go, _ = extract_series(summary, "exp1_grid_only_r2scan")
    axA.errorbar(p_go, q_go, yerr=e_go, fmt="--^", color="#0EA5E9", lw=1.7, ms=6.5, capsize=3,
                 label="Grid-Only 1B+2B (r2SCAN, no MB)")

    p_mp3, q_mp3, e_mp3, _ = extract_series(summary, "exp1_auto_mlip_mp0_d3")
    axA.errorbar(p_mp3, q_mp3, yerr=e_mp3, fmt="-v", color="#DC2626", lw=1.8, ms=6.5, capsize=3,
                 label="Auto-MLIP (MP0+D3 PBE, overbinding)")

    # Single-point baselines at 0.28 bar
    single_pts = [
        ("exp1_mercado_r2scan_0p28", "P", "#D97706", "Mercado2016 + CBMC"),
        ("exp1_uff_r2scan_0p28", "X", "#9333EA", "UFF + CBMC"),
        ("exp1_direct_r2scan_0p28", "o", "#0F172A", "Direct MLIP (300s cap)"),
        ("exp1_mp0_nod3_0p28", "*", "#10B981", "Auto-MLIP (MP0 No-D3)"),
    ]
    for key, marker, col, lbl in single_pts:
        pt = summary["runs"][key]["points"][0]
        axA.errorbar([pt["P_bar"]], [pt["uptake_mmol_g"]], yerr=[pt["uptake_sem_mmol_g"]],
                     fmt=marker, color=col, ms=8.5, capsize=3, label=lbl, zorder=6)

    axA.set_xlabel("Pressure (bar, log scale)")
    axA.set_ylabel("CO2 Uptake (mmol/g)")
    axA.set_title("A. Mg-MOF-74 CO2 @ 298 K: Methods vs. Experiment")
    axA.set_xlim(0.008, 1.35)
    axA.set_ylim(-0.3, 21.0)
    axA.legend(loc="upper left", fontsize=7.6, frameon=True, ncol=1)

    # Panel B: 0.28 bar Method Comparison Bar Chart (r2SCAN target)
    labels_b = [
        "Direct MLIP\n(300s)",
        "UFF+\nCBMC",
        "Grid-Only\n(1B+2B)",
        "Mercado+\nCBMC",
        "Auto-MLIP\n(Empty)",
        "Auto-MLIP\n(High-Load)",
    ]
    keys_b = [
        ("exp1_direct_r2scan_0p28", 0),
        ("exp1_uff_r2scan_0p28", 0),
        ("exp1_grid_only_r2scan", 1),
        ("exp1_mercado_r2scan_0p28", 0),
        ("exp1_auto_mlip_r2scan_nod3", 1),
        ("exp1_auto_mlip_r2scan_nod3_highload_0p28", 0),
    ]
    vals_b = [summary["runs"][k]["points"][idx]["uptake_mmol_g"] for k, idx in keys_b]
    errs_b = [summary["runs"][k]["points"][idx]["uptake_sem_mmol_g"] for k, idx in keys_b]
    accs_b = [
        3.65,  # Direct ins acc (prod_ins_acc/att = 158/4330)
        summary["runs"]["exp1_uff_r2scan_0p28"]["points"][0]["prod_outer_acc_pct"],
        100.0,
        summary["runs"]["exp1_mercado_r2scan_0p28"]["points"][0]["prod_outer_acc_pct"],
        summary["runs"]["exp1_auto_mlip_r2scan_nod3"]["points"][1]["prod_outer_acc_pct"],
        summary["runs"]["exp1_auto_mlip_r2scan_nod3_highload_0p28"]["points"][0]["prod_outer_acc_pct"],
    ]
    cols_b = ["#64748B", "#9333EA", "#0EA5E9", "#D97706", "#1D4ED8", "#047857"]
    x_b = np.arange(len(labels_b))
    bars = axB.bar(x_b, vals_b, yerr=errs_b, color=cols_b, width=0.62, capsize=4, edgecolor="#1E293B", linewidth=0.8)
    axB.axhspan(4.74, 6.80, color="#94A3B8", alpha=0.18, label="Exp @ ~0.28 bar (4.74-6.80)")
    for rect, v, acc, kpair in zip(bars, vals_b, accs_b, keys_b):
        tag = f"{v:.2f}\n(Ins 3.6%)" if kpair[0] == "exp1_direct_r2scan_0p28" else (f"{v:.2f}\n(No MB)" if "grid_only" in kpair[0] else f"{v:.2f}\n({acc:.1f}% acc)")
        axB.text(rect.get_x() + rect.get_width() / 2, v + 0.35, tag, ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    axB.set_xticks(x_b)
    axB.set_xticklabels(labels_b, fontsize=8.4)
    axB.set_ylabel("CO2 Uptake @ 0.28 bar (mmol/g)")
    axB.set_title("B. 0.28 bar Target Parity & Acceptance (r2SCAN No-D3)")
    axB.set_ylim(0, 8.2)
    axB.legend(loc="upper left", frameon=True)

    # Panel C: Bidirectional Trace (Empty vs High-Loading vs Direct MLIP)
    pt_emp = summary["runs"]["exp1_auto_mlip_r2scan_nod3"]["points"][1]
    pt_hl = summary["runs"]["exp1_auto_mlip_r2scan_nod3_highload_0p28"]["points"][0]
    pt_dir = summary["runs"]["exp1_direct_r2scan_0p28"]["points"][0]
    conv_mg = 1000.0 / pt_emp["fw_mass_gmol"]

    tr_emp = np.mean(np.array(pt_emp["box_N_trace"]), axis=1) * conv_mg
    tr_hl = np.mean(np.array(pt_hl["box_N_trace"]), axis=1) * conv_mg
    tr_dir = np.mean(np.array(pt_dir["box_N_trace"]), axis=1) * conv_mg

    c_emp = np.linspace(0, pt_emp["eq_cycles"] + pt_emp["prod_cycles"], len(tr_emp))
    c_hl = np.linspace(0, pt_hl["eq_cycles"] + pt_hl["prod_cycles"], len(tr_hl))
    c_dir = np.linspace(0, pt_dir["elapsed_s"], len(tr_dir))

    axC.plot(c_emp, tr_emp, color="#1D4ED8", lw=1.8, alpha=0.9, label=f"Empty Init -> {pt_emp['uptake_mmol_g']:.3f}+-{pt_emp['uptake_sem_mmol_g']:.3f}")
    axC.plot(c_hl, tr_hl, color="#047857", lw=1.8, alpha=0.88, label=f"High-Load Init -> {pt_hl['uptake_mmol_g']:.3f}+-{pt_hl['uptake_sem_mmol_g']:.3f}")
    axC.axhline(6.318, color="#0F172A", ls=":", lw=1.3, label="Consensus Plateau (~6.32 mmol/g, diff 0.12%)")
    axC.plot(c_dir * 4.0, tr_dir, color="#DC2626", lw=1.8, ls="--", label=f"Direct MLIP (300s trace scaled x4) -> {pt_dir['uptake_mmol_g']:.2f}")
    axC.axvline(1200, color="#64748B", ls="--", lw=1.1, label="Burn-in Cutoff (1200 cyc)")
    axC.set_xlabel("Nested MC Cycle (Direct trace shown at 4x s-to-cycle)")
    axC.set_ylabel("Box-Mean CO2 Uptake (mmol/g)")
    axC.set_title("C. Bidirectional Convergence Trace @ 0.28 bar")
    axC.set_ylim(0, 8.6)
    axC.legend(loc="lower right", fontsize=8.0, frameon=True)

    # Panel D: MLIP Functional Physics Audit @ 0.28 bar (MP0 No-D3 vs r2SCAN No-D3 vs MP0+D3)
    labels_d = ["MP0 No-D3\n(PBE)", "Grid-Only\n(r2SCAN 1B+2B)", "r2SCAN No-D3\n(Auto-MLIP)", "MP0+D3\n(PBE+D3)"]
    vals_d = [
        summary["runs"]["exp1_mp0_nod3_0p28"]["points"][0]["uptake_mmol_g"],
        summary["runs"]["exp1_grid_only_r2scan"]["points"][1]["uptake_mmol_g"],
        summary["runs"]["exp1_auto_mlip_r2scan_nod3"]["points"][1]["uptake_mmol_g"],
        summary["runs"]["exp1_auto_mlip_mp0_d3"]["points"][1]["uptake_mmol_g"],
    ]
    qsts_d = [
        summary["runs"]["exp1_mp0_nod3_0p28"]["points"][0]["Qst_fluc_kJ_mol"],
        summary["runs"]["exp1_grid_only_r2scan"]["points"][1]["Qst_fluc_kJ_mol"],
        summary["runs"]["exp1_auto_mlip_r2scan_nod3"]["points"][1]["Qst_fluc_kJ_mol"],
        summary["runs"]["exp1_auto_mlip_mp0_d3"]["points"][1]["Qst_molar_kJ_mol"],
    ]
    cols_d = ["#10B981", "#0EA5E9", "#1D4ED8", "#DC2626"]
    x_d = np.arange(len(labels_d))
    bars_d = axD.bar(x_d, vals_d, color=cols_d, width=0.55, edgecolor="#1E293B", linewidth=0.8)
    axD.axhspan(4.74, 7.04, color="#94A3B8", alpha=0.22, label="Exp @ 0.28-0.31 bar")
    for rect, v, qv in zip(bars_d, vals_d, qsts_d):
        axD.text(rect.get_x() + rect.get_width() / 2, v + 0.45,
                 f"{v:.2f} mmol/g\n(Qst={qv:.1f} kJ/mol)", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    axD.annotate(
        "+2.33 mmol/g\nmany-body OMS\ncooperativity",
        xy=(1.0, 4.1), xytext=(0.4, 12.2),
        arrowprops=dict(arrowstyle="->", color="#0284C7", lw=1.5),
        fontsize=8.3, fontweight="bold", color="#0369A1",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E0F2FE", edgecolor="#7DD3FC")
    )
    axD.set_xticks(x_d)
    axD.set_xticklabels(labels_d, fontsize=8.8)
    axD.set_ylabel("CO2 Uptake @ 0.28 bar (mmol/g)")
    axD.set_title("D. MLIP Functional Sensitivity & Many-Body Gap")
    axD.set_ylim(0, 22.8)
    axD.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    out = ASSETS_DIR / "fig1_mgmof74_method_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


def plot_fig2(summary: dict):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.4))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

    # Panel A: Mg-MOF-74 + CO2
    p_e, q_e = load_csv_isotherm("mgmof74_10.1021acs.jpcc.7b08239.Isotherm1.csv")
    m = (p_e >= 0.005) & (p_e <= 1.2)
    ax1.plot(p_e[m], q_e[m], "o--", color="#475569", ms=5, lw=1.5, label="Exp (jpcc.7b08239)")
    p, q, e, _ = extract_series(summary, "exp1_auto_mlip_r2scan_nod3")
    ax1.errorbar(p, q, yerr=e, fmt="-D", color="#1D4ED8", lw=2.0, ms=6.5, capsize=3.5, label="r2SCAN No-D3")
    ax1.set_title("A. Mg-MOF-74 + CO2 @ 298 K")
    ax1.set_xlabel("Pressure (bar)")
    ax1.set_ylabel("Uptake (mmol/g)")
    ax1.annotate("1.0 bar: 8.15 vs 8.17\n(0.26% dev)", xy=(1.0, 8.15), xytext=(0.38, 5.2),
                 arrowprops=dict(arrowstyle="->", lw=1.2, color="#1D4ED8"), fontsize=8.3, color="#1E3A8A")
    ax1.legend(loc="upper left", frameon=True)

    # Panel B: Ni-MOF-74 + CO2
    p_e, q_e = load_csv_isotherm("nimof74_10.1021acs.jpcc.5b03700.Isotherm47.csv")
    ax2.plot(p_e, q_e, "o--", color="#475569", ms=4.5, lw=1.4, label="Exp (jpcc.5b03700)")
    p, q, e, _ = extract_series(summary, "exp2_nimof74_co2")
    ax2.errorbar(p, q, yerr=e, fmt="-D", color="#0F766E", lw=2.0, ms=6.5, capsize=3.5, label="r2SCAN No-D3")
    ax2.set_title("B. Ni-MOF-74 + CO2 @ 298 K")
    ax2.set_xlabel("Pressure (bar)")
    ax2.set_ylabel("Uptake (mmol/g)")
    ax2.text(0.05, 3.0, "Qst = 25.2-25.5 kJ/mol\n(Exp ~38-41 kJ/mol;\nr2SCAN Ni d-band sensitivity)",
             fontsize=8.0, color="#0F172A", bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0FDFA", edgecolor="#99F6E4"))
    ax2.legend(loc="upper left", frameon=True)

    # Panel C: Cu-BTC + CO2
    p_e, q_e = load_csv_isotherm("hkust1_10.1002aic.15837.Isotherm7.csv")
    m = p_e <= 1.08
    ax3.plot(p_e[m], q_e[m], "o--", color="#475569", ms=4, lw=1.4, label="Exp (aic.15837)")
    p, q, e, _ = extract_series(summary, "exp2_cubtc_co2")
    ax3.errorbar(p, q, yerr=e, fmt="-D", color="#D97706", lw=2.0, ms=6.5, capsize=3.5, label="r2SCAN No-D3 (2/2 PASS)")
    ax3.set_title("C. Cu-BTC (HKUST-1) + CO2 @ 298 K")
    ax3.set_xlabel("Pressure (bar)")
    ax3.set_ylabel("Uptake (mmol/g)")
    ax3.text(0.06, 2.2, "0.15 bar: Rhat=1.0021\n1.0 bar: Rhat=1.0069\nQst=17.6-18.1 kJ/mol",
             fontsize=8.0, bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFBEB", edgecolor="#FDE68A"))
    ax3.legend(loc="upper left", frameon=True)

    # Panel D: IRMOF-1 + CO2
    p_e1, q_e1 = load_csv_isotherm("irmof1_10.1021acs.jctc.5b00093.Isotherm1.csv")
    p_e2, q_e2 = load_csv_isotherm("irmof1_10.1038srep12045.Isotherm2.csv")
    m1 = p_e1 <= 6.5
    m2 = p_e2 <= 6.5
    ax4.plot(p_e1[m1], q_e1[m1], "o--", color="#475569", ms=5, lw=1.4, label="Exp (jctc.5b00093)")
    ax4.plot(p_e2[m2], q_e2[m2], "s:", color="#94A3B8", ms=5, lw=1.4, label="Exp (srep12045)")
    pt1 = summary["runs"]["exp2_irmof1_co2"]["points"][0]
    pt5 = summary["runs"]["exp2_irmof1_co2_5bar"]["points"][0]
    ax4.errorbar([pt1["P_bar"], pt5["P_bar"]], [pt1["uptake_mmol_g"], pt5["uptake_mmol_g"]],
                 yerr=[pt1["uptake_sem_mmol_g"], pt5["uptake_sem_mmol_g"]],
                 fmt="-D", color="#7C3AED", lw=2.0, ms=6.5, capsize=3.5, label="r2SCAN No-D3")
    ax4.set_title("D. IRMOF-1 + CO2 @ 298 K")
    ax4.set_xlabel("Pressure (bar)")
    ax4.set_ylabel("Uptake (mmol/g)")
    ax4.set_xlim(-0.2, 6.6)
    ax4.set_ylim(-0.2, 6.8)
    ax4.legend(loc="upper left", frameon=True)

    # Panel E: IRMOF-1 + CH4 (MP0+D3) + r2SCAN OOD Diagnostic Inset
    p_ch4, q_ch4, e_ch4, _ = extract_series(summary, "exp2_irmof1_ch4_mp0d3")
    ax5.errorbar(p_ch4, q_ch4, yerr=e_ch4, fmt="-o", color="#DC2626", lw=2.1, ms=7, capsize=3.5,
                 label="Auto-MLIP MP0+D3 (1B R2=0.987)")
    for pv, qv, ev in zip(p_ch4, q_ch4, e_ch4):
        ax5.text(pv, qv + 0.35, f"{qv:.2f}+-{ev:.2f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#991B1B")
    ax5.set_title("E. IRMOF-1 + CH4 @ 298 K (MP0+D3 & OOD Audit)")
    ax5.set_xlabel("Pressure (bar)")
    ax5.set_ylabel("CH4 Uptake (mmol/g)")
    ax5.set_xlim(-0.2, 5.6)
    ax5.set_ylim(0.0, 9.0)
    diag = summary["irmof1_ch4_mlip_diagnostic"]
    ood_txt = (
        "Direct ASE MACE Cross-Check:\n"
        f"r2SCAN No-D3: {np.mean(diag['r2scan_nod3_ase_kJ_mol']):.1f} kJ/mol (OOD)\n"
        f"MP0+D3: {np.mean(diag['mp0_d3_fast_kJ_mol']):.1f} kJ/mol (Physical)\n"
        f"MP0 No-D3: {np.mean(diag['mp0_nod3_fast_kJ_mol']):+.1f} kJ/mol"
    )
    ax5.text(0.25, 5.2, ood_txt, fontsize=7.9,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#FEF2F2", edgecolor="#FECACA"))
    ax5.legend(loc="upper left", frameon=True, fontsize=8.0)

    # Panel F: Distillation R2 & MAE across all MOFs / Guests / Models
    dist_items = [
        ("Mg74\nr2SCAN", "MgMOF-74__co2__mace-matpes-r2scan-0__disp_False"),
        ("Mg74\nMP0+D3", "MgMOF-74__co2__medium__disp_True"),
        ("Mg74\nMP0", "MgMOF-74__co2__medium__disp_False"),
        ("Ni74\nr2SCAN", "NiMOF-74__co2__mace-matpes-r2scan-0__disp_False"),
        ("CuBTC\nr2SCAN", "Cu-BTC__co2__mace-matpes-r2scan-0__disp_False"),
        ("IRMOF1\nCO2", "IRMOF-1__co2__mace-matpes-r2scan-0__disp_False"),
        ("IRMOF1\nCH4", "IRMOF-1__ch4__medium__disp_True"),
    ]
    lbls_f = [x[0] for x in dist_items]
    r2s_f = [summary["distillation"][x[1]]["r2_1b"] for x in dist_items]
    mae1_f = [summary["distillation"][x[1]]["mae_1b_kJ_mol"] for x in dist_items]
    mae2_f = [summary["distillation"][x[1]]["mae_2b_kJ_mol"] for x in dist_items]
    x_f = np.arange(len(lbls_f))
    w = 0.36
    b1 = ax6.bar(x_f - w/2, mae1_f, width=w, color="#2563EB", label="1B MAE (kJ/mol)")
    b2 = ax6.bar(x_f + w/2, mae2_f, width=w, color="#F59E0B", label="2B Pore LUT MAE (kJ/mol)")
    for i, r2v in enumerate(r2s_f):
        ax6.text(x_f[i], max(mae1_f[i], mae2_f[i]) + 0.14, f"R2={r2v:.3f}", ha="center", va="bottom", fontsize=7.3, fontweight="bold", color="#1E3A8A")
    ax6.set_xticks(x_f)
    ax6.set_xticklabels(lbls_f, fontsize=7.8, rotation=18, ha="center")
    ax6.set_ylabel("Held-Out Distillation MAE (kJ/mol)")
    ax6.set_title("F. Online 1B+2B Self-Distillation Accuracy (15% Holdout)")
    ax6.set_ylim(0, 3.2)
    ax6.legend(loc="upper right", frameon=True, fontsize=8.0)

    fig.tight_layout()
    out = ASSETS_DIR / "fig2_cross_mof_generalization.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


def plot_fig3(summary: dict):
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.0))
    axA, axB, axC, axD = axes.ravel()

    # Panel A: Widom & MACE vs D3 Energy Decomposition
    w_r2 = summary["widom"]["phase3_widom_r2scan"]
    w_mp = summary["widom"]["phase3_widom_mp0d3"]
    dec_r2 = w_r2["mace_d3_decomposition"]
    dec_mp = w_mp["mace_d3_decomposition"]
    press_labels = ["0 bar\n(Widom Qst0)", "0.012 bar\n(U/N)", "0.28 bar\n(U/N)", "1.0 bar\n(U/N)"]
    x_a = np.arange(len(press_labels))
    r2_mace = [w_r2["widom"]["Qst_0_kJ_mol"]] + [-d["mean_u_mace_per_mol_kJ_mol"] for d in dec_r2]
    mp_mace = [w_mp["widom"]["Qst_0_kJ_mol"]] + [-d["mean_u_mace_per_mol_kJ_mol"] for d in dec_mp]
    mp_d3 = [0.0] + [-d["mean_u_d3_per_mol_kJ_mol"] for d in dec_mp]

    w = 0.34
    axA.bar(x_a - w/2, r2_mace, width=w, color="#1D4ED8", label="r2SCAN No-D3 (MACE only)")
    axA.bar(x_a + w/2, mp_mace, width=w, color="#F97316", label="MP0+D3: MACE part")
    axA.bar(x_a + w/2, mp_d3, width=w, bottom=mp_mace, color="#DC2626", alpha=0.85, label="MP0+D3: D3(BJ) part")
    axA.text(x_a[0] - w/2, r2_mace[0] + 0.8, f"{r2_mace[0]:.1f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#1E3A8A")
    axA.text(x_a[0] + w/2, mp_mace[0] + 0.8, f"{mp_mace[0]:.1f}\n(K_H=8450\nvs 7.31)", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color="#991B1B")
    for i in range(1, 4):
        tot = mp_mace[i] + mp_d3[i]
        axA.text(x_a[i] + w/2, tot + 0.8, f"{tot:.1f}\n({100*mp_d3[i]/tot:.0f}% D3)", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color="#991B1B")
        axA.text(x_a[i] - w/2, r2_mace[i] + 0.8, f"{r2_mace[i]:.1f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#1E3A8A")
    axA.set_xticks(x_a)
    axA.set_xticklabels(press_labels, fontsize=8.5)
    axA.set_ylabel("Binding Magnitude -<U>/N or Qst(0) (kJ/mol)")
    axA.set_title("A. Widom Henry Regime & MACE / D3 Decomposition (Mg-MOF-74)")
    axA.set_ylim(0, 61)
    axA.legend(loc="upper right", frameon=True, fontsize=8.0)

    # Panel B: k_steps Calibration Sweep + Finite Size Inset
    ks = np.array([5, 10, 15, 25])
    accs = np.array([68.2, 50.1, 39.9, 25.1])
    esjd = np.array([0.1595, 0.2374, 0.2688, 0.2575])
    ess_s = np.array([0.673, 0.607, 0.733, 1.029])

    l1 = axB.plot(ks, accs, "-o", color="#1D4ED8", lw=2.1, ms=7, label="Outer Acceptance (%)")
    axB.set_xlabel("Inner GPU-CBMC Steps per Outer MLIP Call (k_steps)")
    axB.set_ylabel("Outer Acceptance (%)", color="#1D4ED8")
    axB.tick_params(axis="y", labelcolor="#1D4ED8")
    axB.set_ylim(0, 82)
    axB.axvline(16, color="#047857", ls="--", lw=1.4)
    axB.text(16.4, 72, "Selected k=16\n(~32-40% acc)", fontsize=8.2, fontweight="bold", color="#047857")

    axB2 = axB.twinx()
    axB2.spines["top"].set_visible(False)
    axB2.grid(False)
    l2 = axB2.plot(ks, esjd, "--s", color="#D97706", lw=1.9, ms=6.5, label="ESJD_N")
    l3 = axB2.plot(ks, ess_s * 0.25, ":^", color="#DC2626", lw=1.9, ms=6.5, label="0.25 * ESS_N/s")
    axB2.set_ylabel("ESJD_N / Scaled ESS_N per s", color="#B45309")
    axB2.tick_params(axis="y", labelcolor="#B45309")
    axB2.set_ylim(0.10, 0.34)
    lines = l1 + l2 + l3
    axB.legend(lines, [l.get_label() for l in lines], loc="lower left", frameon=True, fontsize=7.8)
    axB.set_title("B. Inner-Burst Calibration (k_steps) & 1x1x2 vs 1x1x4 Finite-Size")

    # Finite-size note box inside B
    fs = w_r2["finite_size_check"]
    fs_txt = "1x1x2 vs 1x1x4 Max |dU/N|:\n" + "\n".join(
        [f"  {item['P_bar']} bar: {item['max_abs_diff_kJ_mol']:.2e} kJ/mol" for item in fs]
    )
    axB.text(13.8, 12.0, fs_txt, fontsize=7.6, bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8FAFC", edgecolor="#CBD5E1"))

    # Panel C: Multi-GPU Scaling (1x vs 4x RTX 3090)
    s1p = summary["runs"]["phase3_scale_1gpu_pressure"]["total_wall_s"]
    s4p = summary["runs"]["phase3_scale_4gpu_pressure"]["total_wall_s"]
    s1r_wall = summary["runs"]["phase3_scale_1gpu_replica"]["total_wall_s"]
    s4r_wall = summary["runs"]["phase3_scale_4gpu_replica"]["total_wall_s"]
    s1r_mc = summary["runs"]["phase3_scale_1gpu_replica"]["points"][0]["elapsed_s"]
    s4r_mc = summary["runs"]["phase3_scale_4gpu_replica"]["points"][0]["elapsed_s"]

    cats = ["Pressure Queue\n(4P End-to-End)", "Replica Parallel\n(8 Box End-to-End)", "Replica Parallel\n(Pure MC Kernel)"]
    t_1gpu = [s1p, s1r_wall, s1r_mc]
    t_4gpu = [s4p, s4r_wall, s4r_mc]
    x_c = np.arange(len(cats))
    w_c = 0.34
    b_1 = axC.bar(x_c - w_c/2, t_1gpu, width=w_c, color="#64748B", label="1x RTX 3090")
    b_4 = axC.bar(x_c + w_c/2, t_4gpu, width=w_c, color="#0F766E", label="4x RTX 3090")
    for i, (t1, t4) in enumerate(zip(t_1gpu, t_4gpu)):
        sp = t1 / t4
        axC.text(x_c[i] - w_c/2, t1 + 2.5, f"{t1:.1f}s", ha="center", va="bottom", fontsize=8.3, fontweight="bold")
        axC.text(x_c[i] + w_c/2, t4 + 2.5, f"{t4:.1f}s\n({sp:.2f}x)", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color="#0F766E")
    axC.set_xticks(x_c)
    axC.set_xticklabels(cats, fontsize=8.7)
    axC.set_ylabel("Wall Time (s)")
    axC.set_title("C. Multi-GPU Scaling Across 4x RTX 3090 (Mg-MOF-74 r2SCAN)")
    axC.set_ylim(0, 155)
    axC.legend(loc="upper right", frameon=True)

    # Panel D: Honest Convergence Audit Across Production Runs
    audit_rows = [
        ("Mg-MOF-74 (r2SCAN 0.012)", summary["runs"]["exp1_auto_mlip_r2scan_nod3"]["points"][0]),
        ("Mg-MOF-74 (r2SCAN 0.28)", summary["runs"]["exp1_auto_mlip_r2scan_nod3"]["points"][1]),
        ("Mg-MOF-74 (r2SCAN 1.0)", summary["runs"]["exp1_auto_mlip_r2scan_nod3"]["points"][2]),
        ("Mg-MOF-74 (MP0 NoD3 0.28)", summary["runs"]["exp1_mp0_nod3_0p28"]["points"][0]),
        ("Ni-MOF-74 (r2SCAN 0.28)", summary["runs"]["exp2_nimof74_co2"]["points"][1]),
        ("Ni-MOF-74 (r2SCAN 1.0)", summary["runs"]["exp2_nimof74_co2"]["points"][2]),
        ("Cu-BTC (r2SCAN 0.15)", summary["runs"]["exp2_cubtc_co2"]["points"][0]),
        ("Cu-BTC (r2SCAN 1.0)", summary["runs"]["exp2_cubtc_co2"]["points"][1]),
        ("IRMOF-1 CO2 (r2SCAN 1.0)", summary["runs"]["exp2_irmof1_co2"]["points"][0]),
        ("IRMOF-1 CH4 (MP0+D3 1.0)", summary["runs"]["exp2_irmof1_ch4_mp0d3"]["points"][0]),
    ]
    y_d = np.arange(len(audit_rows))
    rhats = [pt["N_rhat_max"] for _, pt in audit_rows]
    esss = [pt["N_ess_bulk"] for _, pt in audit_rows]
    passes = [pt["diag_status"] == "pass" for _, pt in audit_rows]
    bar_cols = ["#16A34A" if p else "#F59E0B" for p in passes]

    axD.barh(y_d, esss, color=bar_cols, edgecolor="#1E293B", linewidth=0.7, height=0.66)
    axD.axvline(400, color="#DC2626", ls="--", lw=1.6, label="Gate: ESS_bulk >= 400 (100/chain)")
    for i, (rh, es, ok) in enumerate(zip(rhats, esss, passes)):
        tag = f"ESS={es:.0f}, Rhat={rh:.4f} [{'PASS' if ok else 'PARTIAL'}]"
        axD.text(es + 38, i, tag, va="center", ha="left", fontsize=7.5, fontweight="bold",
                 color="#166534" if ok else "#92400E")
    axD.set_yticks(y_d)
    axD.set_yticklabels([lbl for lbl, _ in audit_rows], fontsize=8.0)
    axD.invert_yaxis()
    axD.set_xlabel("Rank-Normalized Bulk ESS (N)")
    axD.set_title("D. Honest Convergence Audit (Split-Rhat & Bulk ESS)")
    axD.set_xlim(0, 3550)
    axD.legend(loc="lower right", frameon=True, fontsize=8.0)

    fig.tight_layout()
    out = ASSETS_DIR / "fig3_physics_widom_scaling.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


def main():
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    plot_architecture(summary)
    plot_fig1(summary)
    plot_fig2(summary)
    plot_fig3(summary)


if __name__ == "__main__":
    main()
