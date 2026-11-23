"""Generate all 5 publication main figures (300 DPI) for cuda-mof-gcmc:
  1. assets/paper_fig1_architecture_benchmark.png
  2. assets/paper_fig2_nested_mc_verification.png
  3. assets/paper_fig3_mlip_mechanism_audit.png
  4. assets/paper_fig4_mgmof74_isotherm_qst.png
  5. assets/paper_fig5_cross_material_generalization.png
"""

import csv
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
ASSETS = ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def load_json(name):
    return json.loads((BUILD / name).read_text(encoding="utf-8"))


def load_csv(name):
    with open(BUILD / name, "r", encoding="utf-8", newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def setup_rc():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.8,
        "axes.titlesize": 10.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.1,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
    })


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.42, linewidth=0.7)


# ---------------------------------------------------------------------------
# Figure 1: Hybrid Architecture Parity & GPU Throughput Benchmarks
# ---------------------------------------------------------------------------
def make_fig1():
    v2_uff = load_csv("v2_uff_010A.csv")
    v2_merc = load_csv("v2_mercado2016_010A.csv")
    r3_exact = load_csv("raspa3_uff_bench.csv")
    r3_grid = load_csv("raspa3_uff_grid_bench.csv")
    d25 = load_json("phase25_shifted_results.json")["test_b"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.9), dpi=300)
    fig.subplots_adjust(left=0.065, right=0.975, top=0.82, bottom=0.14, wspace=0.27)

    # Panel (a): RASPA3 parity across 0.005-20 bar
    ax1.errorbar(
        [r["P_bar"] for r in v2_uff],
        [r["uptake_mmol_g"] for r in v2_uff],
        yerr=[r["uptake_sem_mmol_g"] for r in v2_uff],
        fmt="o-", color="#0F766E", lw=2.0, ms=4.8, capsize=2.5,
        label="CUDA Tricubic v2 (UFF+REPEAT)",
    )
    ax1.errorbar(
        [r["P_bar"] for r in v2_merc],
        [r["uptake_mmol_g"] for r in v2_merc],
        yerr=[r["uptake_sem_mmol_g"] for r in v2_merc],
        fmt="s--", color="#2563EB", lw=1.7, ms=4.5, capsize=2.5, alpha=0.9,
        label="CUDA Tricubic v2 (Mercado DFT-FF)",
    )
    ax1.errorbar(
        [r["P_bar"] for r in r3_grid],
        [r["uptake_mmol_g"] for r in r3_grid],
        yerr=[r["uptake_err_mmol_g"] for r in r3_grid],
        fmt="^", color="#D97706", ms=8.5, capsize=4, zorder=6, markeredgecolor="#1E293B",
        label="RASPA3 Tricubic Grid (UFF)",
    )
    ax1.errorbar(
        [r["P_bar"] for r in r3_exact],
        [r["uptake_mmol_g"] for r in r3_exact],
        yerr=[r["uptake_err_mmol_g"] for r in r3_exact],
        fmt="D", color="#DC2626", ms=6.2, capsize=4, zorder=7, markeredgecolor="#1E293B",
        label="RASPA3 Exact Reference (UFF)",
    )
    ax1.set_xscale("log")
    ax1.set_xlim(0.004, 25.0)
    ax1.set_ylim(-0.3, 18.2)
    ax1.set_xlabel("Pressure (bar, log scale)")
    ax1.set_ylabel(r"$\mathrm{CO_2}$ Uptake (mmol/g)")
    style_ax(ax1)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1")
    ax1.set_title("(a) Mg-MOF-74 (298 K) Parity vs. RASPA3")
    ax1.annotate(
        "Parity vs. RASPA3 Grid:\n"
        "0.15 bar: +2.32% (0.44$\\sigma$)\n"
        "1.00 bar: +1.39% (0.54$\\sigma$)\n"
        "5.00 bar: -0.41% (0.29$\\sigma$)\n"
        "Energy drift <= 1.0e-8 K",
        xy=(0.97, 0.06), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#F0FDFA", edgecolor="#0F766E", alpha=0.94),
    )

    # Panel (b): Throughput & Batched MLIP Speedups
    labels = [
        "RASPA3 Exact\n(CPU, Exact)",
        "RASPA3 Grid\n(CPU, 0.1 A)",
        "CUDA GPU v2\n(16-Box Batch)",
    ]
    rates = [1215.0, 3908.0, 598669.0]
    cols = ["#DC2626", "#D97706", "#0F766E"]
    x = np.arange(len(labels))
    bars = ax2.bar(x, rates, width=0.52, color=cols, alpha=0.88, edgecolor="#1E293B")
    ax2.set_yscale("log")
    ax2.set_ylim(400, 3.0e6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Classical GCMC Throughput (MC steps/s)")
    style_ax(ax2)
    ax2.grid(axis="x", visible=False)
    for rect, v in zip(bars, rates):
        ax2.text(
            rect.get_x() + rect.get_width() / 2.0, v * 1.18,
            f"{v:,.0f}/s", ha="center", va="bottom", fontsize=8.3, fontweight="bold",
        )
    ax2.annotate(
        f"492.6x vs. RASPA3 Exact\n153.2x vs. RASPA3 Grid",
        xy=(2.0, 598669.0), xytext=(0.95, 5.2e5),
        fontsize=8.0, fontweight="bold", color="#0F766E", ha="center",
        arrowprops=dict(arrowstyle="->", color="#0F766E", lw=1.3),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#CCFBF1", edgecolor="#0F766E"),
    )
    ax2.annotate(
        f"Batched MACE-D3 Nested Outer Loop:\n"
        f"ASE single: {d25['ase_ms_per_box']:.1f} ms/box -> "
        f"4-box Direct: {d25['fast4_ms_per_box']:.1f} ms/box ({d25['speedup_vs_ase']:.2f}x)",
        xy=(0.04, 0.56), xycoords="axes fraction", ha="left", va="center",
        fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#EFF6FF", edgecolor="#2563EB", alpha=0.95),
    )
    ax2.set_title("(b) Classical Inner Kernel & Batched MLIP Speedup")

    fig.suptitle(
        "Fig. 1: GPU-Accelerated Classical Proposal Engine Parity & Batched Nested MLIP Speedup",
        fontsize=11.6, fontweight="bold", y=0.965,
    )
    fig.text(
        0.5, 0.905,
        "RTX 5070 Laptop GPU + PyTorch 2.12 CUDA | Tricubic Force/Energy Grid (0.1 A) & Direct Multi-Replica MACE Graph Batching",
        ha="center", fontsize=8.5, color="#475569",
    )
    out = ASSETS / "paper_fig1_architecture_benchmark.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: Statistical Verification of Biased Nested MC & Finite-Size Test
# ---------------------------------------------------------------------------
def make_fig2():
    p2 = load_json("phase2_nested_mc_results.json")
    p25 = load_json("phase25_shifted_results.json")
    e4 = load_json("paper_exp4_finite_size.json")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.8, 4.9), dpi=300)
    fig.subplots_adjust(left=0.052, right=0.982, top=0.80, bottom=0.145, wspace=0.28)

    # Panel (a): Zero-Bias Null Test (Pol-3Site -> Mercado, 0.15 bar)
    t1 = p2["test1_null_bias"]
    ta = p25["test_a"]["nested"]
    methods = ["Direct GPU\nMercado", "Unshifted Nested\nPol3 -> Merc", "Shifted Nested\nPol3 -> Merc"]
    uptakes = [
        t1["direct"]["uptake_mmol_g"],
        t1["nested"]["uptake_mmol_g"],
        ta["uptake_mmol_g"],
    ]
    u_sems = [
        t1["direct"]["uptake_sem_mmol_g"],
        t1["nested"]["uptake_sem_mmol_g"],
        ta["uptake_sem_mmol_g"],
    ]
    qsts = [
        t1["direct"]["Qst_kJ_mol"],
        t1["nested"]["Qst_kJ_mol"],
        ta["Qst_kJ_mol"],
    ]
    q_sems = [
        t1["direct"]["Qst_sem_kJ_mol"],
        t1["nested"]["Qst_sem_kJ_mol"],
        ta["Qst_sem_kJ_mol"],
    ]
    x = np.arange(len(methods))
    w = 0.34
    b1 = ax1.bar(x - w / 2, uptakes, width=w, yerr=u_sems, capsize=3, color="#0F766E", alpha=0.88, edgecolor="#1E293B", label="Uptake (mmol/g)")
    ax1_r = ax1.twinx()
    b2 = ax1_r.bar(x + w / 2, qsts, width=w, yerr=q_sems, capsize=3, color="#2563EB", alpha=0.80, edgecolor="#1E293B", label="Qst (kJ/mol)")

    for rect, v in zip(b1, uptakes):
        ax1.text(rect.get_x() + rect.get_width() / 2.0, v + 0.25, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#0F766E")
    for rect, v in zip(b2, qsts):
        ax1_r.text(rect.get_x() + rect.get_width() / 2.0, v + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#2563EB")

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=7.8)
    ax1.set_ylim(0, 8.2)
    ax1_r.set_ylim(0, 48.0)
    ax1.set_ylabel("Uptake (mmol/g)", color="#0F766E")
    ax1_r.set_ylabel(r"$Q_{\mathrm{st}}$ (kJ/mol)", color="#2563EB")
    ax1.spines["top"].set_visible(False)
    ax1_r.spines["top"].set_visible(False)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)
    ax1.set_title("(a) Zero-Bias Null Test (0.15 bar)")
    ax1.annotate(
        f"Unshifted z = {t1['z_score']:.2f}$\\sigma$\nShifted Ins/Del: {ta['prod_ins_acc']}/{ta['prod_del_acc']}",
        xy=(0.50, 0.96), xycoords="axes fraction", ha="center", va="top",
        fontsize=7.5, bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8FAFC", edgecolor="#94A3B8"),
    )

    # Panel (b): Trajectory N(cycle) Unshifted Deadlock vs Shifted Reversible Mixing
    t2_traj = p2["test2_live_mace"]["trajectory"]
    cyc_un = [r["cycle"] for r in t2_traj]
    n_un = [r["N"] for r in t2_traj]
    tc = p25["test_c"]
    hist_sh = np.array(tc["box_N_history"])
    cyc_sh = np.arange(len(hist_sh))

    ax2.plot(cyc_un, n_un, color="#DC2626", lw=2.0, label="Unshifted (0 deletions; frozen drift)")
    for b_idx in range(min(4, hist_sh.shape[1])):
        lbl = "Shifted 4-Replica Reversible Mix" if b_idx == 0 else None
        ax2.plot(cyc_sh, hist_sh[:, b_idx], lw=1.35, alpha=0.78 if b_idx == 0 else 0.55,
                 color="#0F766E" if b_idx == 0 else "#14B8A6", label=lbl)
    ax2.axvline(150, color="#64748B", ls="--", lw=1.1, label="Equilibration End (cyc=150)")
    ax2.set_xlabel("Outer Nested MC Cycle")
    ax2.set_ylabel(r"Adsorbed $\mathrm{CO_2}$ Count $N$")
    ax2.set_xlim(0, 505)
    ax2.set_ylim(26, 52)
    style_ax(ax2)
    ax2.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1")
    ax2.set_title("(b) Chem-Pot Bias Unblocks Grand-Canonical Mixing")
    ax2.annotate(
        f"Shifted $\\Delta\\epsilon$ = {tc['delta_eps_kJ_mol']:.2f} kJ/mol\n"
        f"Prod Ins={tc['prod_ins_acc']}/{tc['prod_ins_att']}, Del={tc['prod_del_acc']}/{tc['prod_del_att']}\n"
        f"Uptake = {tc['uptake_mmol_g']:.2f} $\\pm$ {tc['uptake_sem_mmol_g']:.2f} mmol/g",
        xy=(0.03, 0.96), xycoords="axes fraction", ha="left", va="top", fontsize=7.4,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0FDFA", edgecolor="#0F766E"),
    )

    # Panel (c): Supercell Finite-Size Convergence (1x1x2 vs 1x1x4)
    c2 = e4["cell_1x1x2"]
    c4 = e4["cell_1x1x4"]
    cats = ["Direct Mercado\n(Classical, 0.01 bar)", "Live MACE-MP-0+D3\n(Nested MC, 0.01 bar)"]
    y_112 = [c2["direct_mercado"]["uptake_mmol_g"], c2["mace_mp0_d3"]["uptake_mmol_g"]]
    e_112 = [c2["direct_mercado"]["uptake_sem_mmol_g"], c2["mace_mp0_d3"]["uptake_sem_mmol_g"]]
    y_114 = [c4["direct_mercado"]["uptake_mmol_g"], c4["mace_mp0_d3"]["uptake_mmol_g"]]
    e_114 = [c4["direct_mercado"]["uptake_sem_mmol_g"], c4["mace_mp0_d3"]["uptake_sem_mmol_g"]]

    xx = np.arange(len(cats))
    ww = 0.34
    r1 = ax3.bar(xx - ww / 2, y_112, width=ww, yerr=e_112, capsize=3.5, color="#D97706", alpha=0.86, edgecolor="#1E293B",
                 label=f"1x1x2 (324 fw atoms, r_cut={c2['r_cut_gg_A']:.2f} A)")
    r2 = ax3.bar(xx + ww / 2, y_114, width=ww, yerr=e_114, capsize=3.5, color="#0F766E", alpha=0.86, edgecolor="#1E293B",
                 label=f"1x1x4 (648 fw atoms, r_cut={c4['r_cut_gg_A']:.2f} A)")
    for rect, v in zip(r1, y_112):
        ax3.text(rect.get_x() + rect.get_width() / 2.0, v + 0.42, f"{v:.2f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    for rect, v in zip(r2, y_114):
        ax3.text(rect.get_x() + rect.get_width() / 2.0, v + 0.42, f"{v:.2f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")

    ax3.set_xticks(xx)
    ax3.set_xticklabels(cats, fontsize=8.0)
    ax3.set_ylim(0, 12.2)
    ax3.set_ylabel(r"$\mathrm{CO_2}$ Uptake (mmol/g)")
    style_ax(ax3)
    ax3.grid(axis="x", visible=False)
    ax3.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1")
    ax3.set_title("(c) Supercell Finite-Size Convergence (Mg-MOF-74)")
    ax3.annotate(
        f"Direct z = {e4['z_direct']:.2f}$\\sigma$ | MLIP z = {e4['z_mlip']:.2f}$\\sigma$\n"
        f"Molar Qst: {c2['mace_mp0_d3']['mean_molar_qst_kJ_mol']:.2f} vs "
        f"{c4['mace_mp0_d3']['mean_molar_qst_kJ_mol']:.2f} kJ/mol",
        xy=(0.04, 0.62), xycoords="axes fraction", ha="left", va="center", fontsize=7.4,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFBEB", edgecolor="#D97706"),
    )

    fig.suptitle(
        "Fig. 2: Rigorous Statistical Verification of Chemical-Potential-Shifted Nested MLIP-GCMC",
        fontsize=11.8, fontweight="bold", y=0.965,
    )
    fig.text(
        0.5, 0.90,
        "Exact Zero-Bias Null Validation, Deletion-Deadlock Resolution on MACE-MP-0+D3, and 1x1x2 vs. 1x1x4 Supercell Convergence",
        ha="center", fontsize=8.5, color="#475569",
    )
    out = ASSETS / "paper_fig2_nested_mc_verification.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: MLIP Mechanism Audit & Short-Range Overbinding Diagnostics
# ---------------------------------------------------------------------------
def make_fig3():
    mech = load_json("paper_exp1_mechanism.json")
    probe = load_json("phase3_model_probe.json")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.0, 5.0), dpi=300)
    fig.subplots_adjust(left=0.055, right=0.982, top=0.80, bottom=0.145, wspace=0.27)

    # Panel (a): Unobstructed Mg2+ - O=C=O radial scans
    style_map = {
        "MACE-MP-0 + D3(PBE)": ("#D97706", "-", 2.0),
        "MACE-MatPES-r2SCAN-0 + D3(r2scan)": ("#0F766E", "-", 2.3),
        "MACE-MatPES-r2SCAN-0 (No D3)": ("#64748B", "--", 1.7),
        "MACE-MPA-0 + D3(PBE)": ("#DC2626", "-", 1.9),
        "MACE-MP-0b3 + D3(PBE)": ("#7F1D1D", "-.", 2.0),
    }
    for c in mech["mg_curves"]:
        lbl = c["label"]
        if lbl in style_map:
            col, ls, lw = style_map[lbl]
            ax1.plot(
                c["radii_A"], c["E_bind_kJ_mol"], color=col, ls=ls, lw=lw,
                label=f"{lbl} (min {c['min_E_kJ_mol']:.1f})",
            )
    ax1.axhspan(-45, -38, color="#0F766E", alpha=0.12, label="Exp. / CCSD(T) Binding Window")
    ax1.axhline(0.0, color="#475569", lw=0.9, ls=":")
    ax1.set_xlim(1.75, 4.45)
    ax1.set_ylim(-115, 58)
    ax1.set_xlabel(r"Unobstructed $\mathrm{Mg^{2+}\cdots O=C=O}$ Distance ($\mathrm{\AA}$)")
    ax1.set_ylabel(r"Axial Binding Energy $\Delta E_{\mathrm{bind}}$ (kJ/mol)")
    style_ax(ax1)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.93, edgecolor="#CBD5E1", fontsize=6.9)
    ax1.set_title("(a) Unobstructed OMS Radial Scan (Mg-MOF-74)")

    # Panel (b): Far-Field Zero-Offset Proof (d_min = 7.31 A > 6.0 A cutoff)
    models_order = ["MP-0", "MPA-0", "MP-0b3", "r2SCAN-0"]
    curves_by_key = {(c["model"], c["dispersion"]): c["far_field_kJ_mol"] for c in mech["mg_curves"]}
    no_d3 = [curves_by_key[(m, False)] for m in ["medium", "medium-mpa-0", "medium-0b3", "mace-matpes-r2scan-0"]]
    with_d3 = [curves_by_key[(m, True)] for m in ["medium", "medium-mpa-0", "medium-0b3", "mace-matpes-r2scan-0"]]

    xx = np.arange(len(models_order))
    ww = 0.34
    b_nd = ax2.bar(xx - ww / 2, no_d3, width=ww, color="#64748B", alpha=0.86, edgecolor="#1E293B", label="Raw MLIP (No D3)")
    b_d3 = ax2.bar(xx + ww / 2, with_d3, width=ww, color="#0F766E", alpha=0.86, edgecolor="#1E293B", label="With D3 Dispersion Tail")
    ax2.axhline(0.0, color="#1E293B", lw=0.9)
    for rect, v in zip(b_nd, no_d3):
        va = "bottom" if v >= 0 else "top"
        dy = 0.08 if v >= 0 else -0.08
        ax2.text(rect.get_x() + rect.get_width() / 2.0, v + dy, f"{v:+.2f}", ha="center", va=va, fontsize=7.3, fontweight="bold")
    for rect, v in zip(b_d3, with_d3):
        ax2.text(rect.get_x() + rect.get_width() / 2.0, v - 0.10, f"{v:.2f}", ha="center", va="top", fontsize=7.3, fontweight="bold", color="#0F766E")

    ax2.set_xticks(xx)
    ax2.set_xticklabels(models_order)
    ax2.set_ylim(-3.8, 1.35)
    ax2.set_ylabel(r"Far-Field Energy at $7.31\,\mathrm{\AA}$ (kJ/mol)")
    style_ax(ax2)
    ax2.grid(axis="x", visible=False)
    ax2.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1")
    ax2.set_title("(b) Far-Field Cutoff Check ($r=7.31 > 6.0\\,\\mathrm{\\AA}$)")
    ax2.annotate(
        "No-D3 |E_ff| <= 0.21 kJ/mol -> Zero baseline offset;\noverbinding is purely short-range!",
        xy=(0.50, 0.92), xycoords="axes fraction", ha="center", va="top", fontsize=7.4,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#F8FAFC", edgecolor="#475569"),
    )

    # Panel (c): Loaded Framework Mean -U/N Benchmark
    lbl_c = ["MP-0\n+D3(PBE)", "MPA-0\n+D3(PBE)", "MP-0b3\n+D3(PBE)", "r2SCAN-0\n(No D3)", "r2SCAN-0\n+D3(r2scan)"]
    vals_c = [
        -probe[0]["loaded_mean_U_per_N_kJ_mol"],
        -probe[1]["loaded_mean_U_per_N_kJ_mol"],
        -probe[2]["loaded_mean_U_per_N_kJ_mol"],
        -probe[3]["loaded_mean_U_per_N_kJ_mol"],
        40.13,
    ]
    cols_c = ["#D97706", "#DC2626", "#7F1D1D", "#64748B", "#0F766E"]
    bars = ax3.bar(np.arange(len(lbl_c)), vals_c, width=0.56, color=cols_c, alpha=0.88, edgecolor="#1E293B")
    ax3.axhspan(37.5, 44.5, color="#0F766E", alpha=0.14, label="Exp. |dU| Window (~38-45)")
    ax3.axhline(33.5, color="#2563EB", ls="--", lw=1.5, label="Mercado DFT-FF (~33.5)")
    for rect, v in zip(bars, vals_c):
        ax3.text(rect.get_x() + rect.get_width() / 2.0, v + 1.8, f"{v:.1f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax3.set_xticks(np.arange(len(lbl_c)))
    ax3.set_xticklabels(lbl_c, fontsize=7.6)
    ax3.set_ylim(0, 108)
    ax3.set_ylabel(r"Loaded Mean Binding $-U/N$ (kJ/mol)")
    style_ax(ax3)
    ax3.grid(axis="x", visible=False)
    ax3.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1")
    ax3.set_title("(c) Loaded Pore-Ensemble -U/N Benchmark")

    fig.suptitle(
        "Fig. 3: Mechanistic Audit of Universal MLIPs in Open-Metal-Site Mg-MOF-74",
        fontsize=11.8, fontweight="bold", y=0.965,
    )
    fig.text(
        0.5, 0.90,
        "MACE-MPA-0 & MP-0b3 Collapse at Short Range (2.2-2.35 A), while MACE-MatPES-r2SCAN-0+D3(r2scan) Matches Physical Binding",
        ha="center", fontsize=8.5, color="#475569",
    )
    out = ASSETS / "paper_fig3_mlip_mechanism_audit.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 4: Mg-MOF-74 Multi-Pressure Isotherm, Dual Qst & Stationarity
# ---------------------------------------------------------------------------
def make_fig4():
    mg_cls = load_json("phase3_mgmof74_classical.json")
    mg_mp0 = load_json("phase3_mgmof74_mace_mp0_d3.json")["points"]
    mg_r2_short = load_json("phase3_mgmof74_r2scan_d3.json")["points"]
    mg_r2_long = load_json("paper_exp3_mgmof74_r2scan_long.json")["points"]

    # Merge 0.002 bar (260 prod) + 0.01/0.05/0.15 bar (480 prod long-chain)
    long_by_p = {round(pt["P_bar"], 4): pt for pt in mg_r2_long}
    mg_r2_best = [long_by_p.get(round(pt["P_bar"], 4), pt) for pt in mg_r2_short]

    # Digitized Mason et al. Energy Environ. Sci. 2011 (Mg-MOF-74, 298 K)
    exp_p = [0.01, 0.05, 0.15, 0.50]
    exp_u = [1.4, 3.1, 4.3, 6.1]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.0, 5.0), dpi=300)
    fig.subplots_adjust(left=0.052, right=0.942, top=0.80, bottom=0.145, wspace=0.29)

    # Panel (a): Isotherms
    ax1.plot([x["P_bar"] for x in mg_cls["UFF_REPEAT"]], [x["uptake_mmol_g"] for x in mg_cls["UFF_REPEAT"]],
             "^--", color="#64748B", lw=1.6, ms=5.8, label="UFF+REPEAT (Direct GPU)")
    ax1.plot([x["P_bar"] for x in mg_cls["Mercado_2016"]], [x["uptake_mmol_g"] for x in mg_cls["Mercado_2016"]],
             "s-", color="#2563EB", lw=2.0, ms=5.8, label="Mercado 2016 DFT-FF (Direct GPU)")
    ax1.errorbar([x["P_bar"] for x in mg_r2_best], [x["uptake_mmol_g"] for x in mg_r2_best],
                 yerr=[x["uptake_sem_mmol_g"] for x in mg_r2_best], fmt="o-", color="#0F766E",
                 lw=2.1, ms=6.2, capsize=3, label="r2SCAN-0+D3(r2scan) Nested MC")
    ax1.errorbar([x["P_bar"] for x in mg_mp0], [x["uptake_mmol_g"] for x in mg_mp0],
                 yerr=[x["uptake_sem_mmol_g"] for x in mg_mp0], fmt="D-", color="#D97706",
                 lw=1.9, ms=5.5, capsize=3, label="MACE-MP-0+D3(PBE) Nested MC")
    ax1.plot(exp_p, exp_u, "*:", color="#111827", lw=1.5, ms=9.5, zorder=7,
             label="Exp. Mg-MOF-74 (Mason EES 2011, dig.)")

    ax1.set_xscale("log")
    ax1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.set_xlim(0.0015, 0.65)
    ax1.set_xticks([0.002, 0.01, 0.05, 0.15, 0.50])
    ax1.set_xticklabels(["0.002", "0.01", "0.05", "0.15", "0.50"])
    ax1.set_ylim(0, 21.5)
    ax1.set_xlabel("Pressure (bar, log scale)")
    ax1.set_ylabel(r"$\mathrm{CO_2}$ Uptake (mmol/g)")
    style_ax(ax1)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1", fontsize=6.9)
    ax1.set_title("(a) Mg-MOF-74 (298 K) Multi-Pressure Isotherms")

    # Panel (b): Dual Qst Estimators (Long Chain 120/480)
    p_long = [pt["P_bar"] for pt in mg_r2_long]
    q_fluc = [pt["Qst_kJ_mol"] for pt in mg_r2_long]
    qe_fluc = [pt["Qst_sem_mmol_g"] if "Qst_sem_mmol_g" in pt else pt["Qst_sem_kJ_mol"] for pt in mg_r2_long]
    q_mol = [pt["mean_molar_qst_kJ_mol"] for pt in mg_r2_long]
    qe_mol = [pt["mean_molar_qst_sem_kJ_mol"] for pt in mg_r2_long]
    merc_q = [x["Qst_kJ_mol"] for x in mg_cls["Mercado_2016"] if round(x["P_bar"], 4) in {0.01, 0.05, 0.15}]
    merc_qe = [x["Qst_sem_kJ_mol"] for x in mg_cls["Mercado_2016"] if round(x["P_bar"], 4) in {0.01, 0.05, 0.15}]

    xx = np.arange(len(p_long))
    ww = 0.25
    ax2.axhspan(38.0, 44.0, color="#64748B", alpha=0.13, label="Exp. Calorimetric Qst (38-44)")
    ax2.bar(xx - ww, merc_q, width=ww, yerr=merc_qe, capsize=3, color="#2563EB", alpha=0.84, edgecolor="#1E293B", label="Mercado DFT-FF (Fluc)")
    ax2.bar(xx, q_fluc, width=ww, yerr=qe_fluc, capsize=3, color="#D97706", alpha=0.84, edgecolor="#1E293B", label="r2SCAN+D3 (Fluc, 480 cyc)")
    b_m = ax2.bar(xx + ww, q_mol, width=ww, yerr=qe_mol, capsize=3, color="#0F766E", alpha=0.90, edgecolor="#1E293B", label="r2SCAN+D3 (Molar -<U>/<N>+RT)")

    for rect, v, e in zip(b_m, q_mol, qe_mol):
        ax2.text(rect.get_x() + rect.get_width() / 2.0, v + e + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#0F766E")

    ax2.set_xticks(xx)
    ax2.set_xticklabels([f"{p} bar" for p in p_long])
    ax2.set_ylim(0, 62)
    ax2.set_ylabel(r"Isosteric Heat $Q_{\mathrm{st}}$ (kJ/mol)")
    style_ax(ax2)
    ax2.grid(axis="x", visible=False)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1", fontsize=7.0)
    ax2.set_title("(b) Dual Qst Estimators (Long Chain 120/480)")

    # Panel (c): Sampling Balance & Half-Chain Stationarity
    ins_acc = [pt["prod_ins_acc"] for pt in mg_r2_long]
    del_acc = [pt["prod_del_acc"] for pt in mg_r2_long]
    h1 = [pt["first_half_uptake_mmol_g"] for pt in mg_r2_long]
    h2 = [pt["second_half_uptake_mmol_g"] for pt in mg_r2_long]

    ww2 = 0.30
    b_in = ax3.bar(xx - ww2 / 2, ins_acc, width=ww2, color="#0F766E", alpha=0.86, edgecolor="#1E293B", label="Accepted Insertions (480 cyc)")
    b_dl = ax3.bar(xx + ww2 / 2, del_acc, width=ww2, color="#0284C7", alpha=0.84, edgecolor="#1E293B", label="Accepted Deletions (480 cyc)")
    for rect, v in zip(b_in, ins_acc):
        ax3.text(rect.get_x() + rect.get_width() / 2.0, v + 2.0, f"{v}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    for rect, v in zip(b_dl, del_acc):
        ax3.text(rect.get_x() + rect.get_width() / 2.0, v + 2.0, f"{v}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")

    ax3_r = ax3.twinx()
    drifts = [abs(b - a) for a, b in zip(h1, h2)]
    ax3_r.plot(xx, drifts, "s-", color="#DC2626", lw=1.9, ms=6.2, label="|Half2 - Half1| Drift (mmol/g)")
    ax3_r.set_ylim(0, 1.2)
    ax3_r.set_ylabel("Half-Chain Drift |H2 - H1| (mmol/g)", color="#DC2626")
    ax3_r.spines["top"].set_visible(False)

    ax3.set_xticks(xx)
    ax3.set_xticklabels([f"{p} bar" for p in p_long])
    ax3.set_ylim(0, 118)
    ax3.set_ylabel("Accepted Grand-Canonical Moves")
    ax3.spines["top"].set_visible(False)
    ax3.grid(axis="y", linestyle=":", alpha=0.4)
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_r.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1", fontsize=7.0)
    ax3.set_title("(c) Insertion/Deletion Balance & Stationarity")

    fig.suptitle(
        "Fig. 4: Mg-MOF-74 (298 K) Defect-Free MLIP Isotherms, Long-Chain Qst Convergence & Exchange Balance",
        fontsize=11.6, fontweight="bold", y=0.965,
    )
    fig.text(
        0.5, 0.90,
        "r2SCAN-0+D3(r2scan) Molar Qst = 46.7 -> 45.6 -> 43.2 kJ/mol (SEM <= 0.35 kJ/mol) with Balanced Reversible Exchange",
        ha="center", fontsize=8.5, color="#475569",
    )
    out = ASSETS / "paper_fig4_mgmof74_isotherm_qst.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 5: Zero-Fitting Cross-Material Generalization (Ni-MOF-74 & HKUST-1)
# ---------------------------------------------------------------------------
def make_fig5():
    ni_cls = load_json("phase3_nimof74_classical.json")
    ni_r2_short = load_json("phase3_nimof74_r2scan_d3.json")["points"]
    ni_r2_long = load_json("paper_exp3_nimof74_r2scan_long.json")["points"]
    ni_long_map = {round(pt["P_bar"], 4): pt for pt in ni_r2_long}
    ni_r2_best = [ni_long_map.get(round(pt["P_bar"], 4), pt) for pt in ni_r2_short]

    hk_cls = load_json("paper_exp2_hkust1_uff.json")
    hk_r2 = load_json("paper_exp2_hkust1_r2scan.json")["points"]

    # Digitized literature benchmarks
    exp_ni_p = [0.01, 0.05, 0.15]
    exp_ni_u = [0.50, 2.00, 3.90]  # Queen Chem. Sci. 2014 / Dietzel 2009
    exp_hk_p = [0.15, 0.50, 1.00]
    exp_hk_u = [0.70, 2.20, 4.20]  # Yazaydin JACS 2009 / Mason EES 2011

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.0, 5.0), dpi=300)
    fig.subplots_adjust(left=0.052, right=0.982, top=0.80, bottom=0.145, wspace=0.27)

    # Panel (a): Ni-MOF-74
    ax1.plot([x["P_bar"] for x in ni_cls["UFF_templateQ"]], [x["uptake_mmol_g"] for x in ni_cls["UFF_templateQ"]],
             "^--", color="#64748B", lw=1.6, ms=5.8, label="UFF (template Q, Direct GPU)")
    ax1.plot([x["P_bar"] for x in ni_cls["Pol3Site"]], [x["uptake_mmol_g"] for x in ni_cls["Pol3Site"]],
             "s-", color="#2563EB", lw=1.9, ms=5.8, label="Pol-3Site (Direct GPU)")
    ax1.errorbar([x["P_bar"] for x in ni_r2_best], [x["uptake_mmol_g"] for x in ni_r2_best],
                 yerr=[x["uptake_sem_mmol_g"] for x in ni_r2_best], fmt="o-", color="#0F766E",
                 lw=2.1, ms=6.2, capsize=3, label="r2SCAN-0+D3 (Nested MC, 480 cyc)")
    ax1.plot(exp_ni_p, exp_ni_u, "*:", color="#111827", lw=1.6, ms=9.5, zorder=7,
             label="Exp. Ni-MOF-74 (Queen 2014, dig.)")

    ax1.set_xscale("log")
    ax1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.set_xlim(0.007, 0.22)
    ax1.set_xticks([0.01, 0.05, 0.15])
    ax1.set_xticklabels(["0.01", "0.05", "0.15"])
    ax1.set_ylim(0, 6.2)
    ax1.set_xlabel("Pressure (bar, log scale)")
    ax1.set_ylabel(r"$\mathrm{CO_2}$ Uptake (mmol/g)")
    style_ax(ax1)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1", fontsize=7.2)
    ax1.set_title("(a) Isostructural Ni-MOF-74 (298 K)")
    ax1.annotate(
        f"Zero-fit Transfer:\n"
        f"0.15 bar: {ni_r2_best[-1]['uptake_mmol_g']:.2f} vs ~3.9 mmol/g Exp.\n"
        f"Molar Qst = {ni_r2_best[-1]['mean_molar_qst_kJ_mol']:.1f} kJ/mol",
        xy=(0.03, 0.60), xycoords="axes fraction", ha="left", va="center", fontsize=7.3,
        bbox=dict(boxstyle="round,pad=0.33", facecolor="#F0FDFA", edgecolor="#0F766E"),
    )

    # Panel (b): HKUST-1 (Cu-BTC)
    ax2.plot([x["P_bar"] for x in hk_cls["UFF"]], [x["uptake_mmol_g"] for x in hk_cls["UFF"]],
             "^--", color="#64748B", lw=1.7, ms=6.0, label="UFF+Charges (Direct GPU)")
    ax2.plot([x["P_bar"] for x in hk_cls["Pol3Site"]], [x["uptake_mmol_g"] for x in hk_cls["Pol3Site"]],
             "s-", color="#2563EB", lw=1.9, ms=5.8, label="Pol-3Site (Direct GPU)")
    ax2.errorbar([x["P_bar"] for x in hk_r2], [x["uptake_mmol_g"] for x in hk_r2],
                 yerr=[x["uptake_sem_mmol_g"] for x in hk_r2], fmt="o-", color="#DC2626",
                 lw=2.1, ms=6.2, capsize=3, label="r2SCAN-0+D3 (UFF Prop., Nested MC)")
    ax2.plot(exp_hk_p, exp_hk_u, "*:", color="#111827", lw=1.6, ms=9.5, zorder=7,
             label="Exp. HKUST-1 (Yazaydin 2009, dig.)")

    ax2.set_xscale("log")
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_xlim(0.11, 1.35)
    ax2.set_xticks([0.15, 0.50, 1.00])
    ax2.set_xticklabels(["0.15", "0.50", "1.00"])
    ax2.set_ylim(-0.15, 6.2)
    ax2.set_xlabel("Pressure (bar, log scale)")
    ax2.set_ylabel(r"$\mathrm{CO_2}$ Uptake (mmol/g)")
    style_ax(ax2)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1", fontsize=7.1)
    ax2.set_title("(b) Cross-Topology HKUST-1 (Cu-BTC, 298 K)")
    ax2.annotate(
        f"Honest Failure Mode:\n"
        f"Plateaus at ~{hk_r2[-1]['uptake_mmol_g']:.2f} mmol/g (Qst ~{hk_r2[-1]['mean_molar_qst_kJ_mol']:.1f} kJ/mol)\n"
        f"due to Cu2+ dimer underbinding",
        xy=(0.98, 0.04), xycoords="axes fraction", ha="right", va="bottom", fontsize=7.0,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#FEF2F2", edgecolor="#DC2626"),
    )

    # Panel (c): Quantitative OMS Binding Comparison Explaining (a) vs (b)
    # Verified axial radial scan peaks from paper_exp1_mechanism.json + task-1985 Cu-BTC axial probe
    mech = load_json("paper_exp1_mechanism.json")
    mg_r2_peak = -next(c["min_E_kJ_mol"] for c in mech["mg_curves"] if c["model"] == "mace-matpes-r2scan-0" and c["dispersion"])
    mg_mp0_peak = -next(c["min_E_kJ_mol"] for c in mech["mg_curves"] if c["model"] == "medium" and c["dispersion"])
    ni_r2_peak = -next(c["min_E_kJ_mol"] for c in mech["ni_curves"] if c["model"] == "mace-matpes-r2scan-0" and c["dispersion"])
    ni_mp0_peak = -next(c["min_E_kJ_mol"] for c in mech["ni_curves"] if c["model"] == "medium" and c["dispersion"])
    cu_r2_peak = 10.07  # task-1985 verified axial Cu2+ minimum (-10.07 kJ/mol @ 2.40 A)
    cu_mp0_peak = 7.72  # task-1985 verified axial Cu2+ minimum (-7.72 kJ/mol @ 2.40 A)

    sites = ["Mg2+ OMS\n(Mg-MOF-74)", "Ni2+ OMS\n(Ni-MOF-74)", "Cu2+ Paddlewheel\n(HKUST-1)"]
    xx = np.arange(len(sites))
    ww = 0.34
    b1 = ax3.bar(xx - ww / 2, [mg_r2_peak, ni_r2_peak, cu_r2_peak], width=ww, color="#0F766E", alpha=0.88, edgecolor="#1E293B", label="r2SCAN-0+D3(r2scan)")
    b2 = ax3.bar(xx + ww / 2, [mg_mp0_peak, ni_mp0_peak, cu_mp0_peak], width=ww, color="#D97706", alpha=0.86, edgecolor="#1E293B", label="MACE-MP-0+D3(PBE)")
    for rect, v in zip(b1, [mg_r2_peak, ni_r2_peak, cu_r2_peak]):
        ax3.text(rect.get_x() + rect.get_width() / 2.0, v + 0.9, f"{v:.1f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    for rect, v in zip(b2, [mg_mp0_peak, ni_mp0_peak, cu_mp0_peak]):
        ax3.text(rect.get_x() + rect.get_width() / 2.0, v + 0.9, f"{v:.1f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")

    ax3.axhspan(26, 30, color="#DC2626", alpha=0.11, label="Exp. HKUST-1 OMS Range (~26-30)")
    ax3.set_xticks(xx)
    ax3.set_xticklabels(sites, fontsize=8.0)
    ax3.set_ylim(0, 62)
    ax3.set_ylabel(r"Peak Axial OMS Binding $-\Delta E_{\min}$ (kJ/mol)")
    style_ax(ax3)
    ax3.grid(axis="x", visible=False)
    ax3.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1", fontsize=7.1)
    ax3.set_title("(c) OMS Peak Binding Root Cause Across Metals")
    ax3.annotate(
        "Spin-unaware MLIPs miss\nAF Cu2+-Cu2+ paddlewheel\nchemisorption",
        xy=(2.0, 13.5), xytext=(1.75, 42.0),
        fontsize=7.3, color="#991B1B", ha="center",
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2", edgecolor="#DC2626"),
    )

    fig.suptitle(
        "Fig. 5: Zero-Fitting Cross-Material Generalization & Diagnostic Boundary on Spin-Coupled OMS MOFs",
        fontsize=11.6, fontweight="bold", y=0.965,
    )
    fig.text(
        0.5, 0.90,
        "Quantitative Match on Ni-MOF-74 (4.29 vs ~3.9 mmol/g at 0.15 bar) vs. Honest Detection of Antiferromagnetic Cu2+ Underbinding in HKUST-1",
        ha="center", fontsize=8.5, color="#475569",
    )
    out = ASSETS / "paper_fig5_cross_material_generalization.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    setup_rc()
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()


if __name__ == "__main__":
    main()
