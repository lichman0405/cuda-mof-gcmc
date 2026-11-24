"""Generate two 300 DPI Stage-1 validation figures directly from verified
build/*.json and curated data/exp_isotherms/*.csv datasets:
  1. assets/stage1_four_mof_nist_isodb.png
  2. assets/stage1_qst_fp32_diagnostics.png
"""

import csv
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
EXP = ROOT / "data" / "exp_isotherms"
ASSETS = ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def load_json(name: str):
    return json.loads((BUILD / name).read_text(encoding="utf-8"))


def load_exp_csv(name: str, p_min: float, p_max: float):
    with open(EXP / name, "r", encoding="utf-8", newline="") as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    return [r for r in rows if p_min <= r["P_bar"] <= p_max]


def setup_rc():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.8,
        "axes.titlesize": 10.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.0,
        "legend.fontsize": 7.3,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
    })


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.42, linewidth=0.7)


def format_log_x(ax, ticks):
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])


def make_fig_four_mof():
    # Load simulation data
    mg_cls = load_json("phase3_mgmof74_classical.json")
    mg_r2_short = load_json("phase3_mgmof74_r2scan_d3.json")["points"]
    mg_r2_long = load_json("paper_exp3_mgmof74_r2scan_long.json")["points"]
    mg_r2_deep = load_json("stage1_mgmof74_r2scan_deep.json")["points"]

    mg_map = {round(pt["P_bar"], 4): pt for pt in mg_r2_short}
    for pt in mg_r2_long:
        mg_map[round(pt["P_bar"], 4)] = pt
    for pt in mg_r2_deep:
        mg_map[round(pt["P_bar"], 4)] = pt
    mg_r2 = [mg_map[k] for k in sorted(mg_map.keys())]

    ni_cls = load_json("phase3_nimof74_classical.json")
    ni_r2_short = load_json("phase3_nimof74_r2scan_d3.json")["points"]
    ni_r2_long = load_json("paper_exp3_nimof74_r2scan_long.json")["points"]
    ni_map = {round(pt["P_bar"], 4): pt for pt in ni_r2_short}
    for pt in ni_r2_long:
        ni_map[round(pt["P_bar"], 4)] = pt
    ni_r2 = [ni_map[k] for k in sorted(ni_map.keys())]

    hk_cls = load_json("paper_exp2_hkust1_uff.json")
    hk_r2 = load_json("stage1_hkust1_r2scan_eq.json")["points"]

    ir_uff = load_json("stage1_irmof1_uff.json")["UFF"]
    ir_r2 = load_json("stage1_irmof1_r2scan.json")["points"]

    # Load curated NIST ISODB experimental tables
    exp_mg_1 = load_exp_csv("mgmof74_10.1021acs.jpcc.7b08239.Isotherm1.csv", 0.001, 0.40)
    exp_mg_2 = load_exp_csv("mgmof74_10.1039c5dt03137k.Isotherm1.csv", 0.001, 0.25)
    exp_ni = load_exp_csv("nimof74_10.1021acs.jpcc.5b03700.Isotherm47.csv", 0.005, 0.25)
    exp_hk = load_exp_csv("hkust1_10.1002aic.15837.Isotherm7.csv", 0.10, 1.15)
    exp_ir_1 = load_exp_csv("irmof1_10.1021acs.jctc.5b00093.Isotherm1.csv", 0.3, 6.5)
    exp_ir_2 = load_exp_csv("irmof1_10.1038srep12045.Isotherm2.csv", 0.05, 6.5)

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.6), dpi=300)
    fig.subplots_adjust(left=0.065, right=0.975, top=0.895, bottom=0.07, hspace=0.30, wspace=0.22)

    # Panel (a): Mg-MOF-74
    ax = axes[0, 0]
    p_cls = [x for x in mg_cls["UFF_REPEAT"] if x["P_bar"] <= 0.20]
    m_cls = [x for x in mg_cls["Mercado_2016"] if x["P_bar"] <= 0.20]
    ax.plot([x["P_bar"] for x in p_cls], [x["uptake_mmol_g"] for x in p_cls],
            "^--", color="#64748B", lw=1.6, ms=5.5, label="UFF+REPEAT (Direct GPU)")
    ax.plot([x["P_bar"] for x in m_cls], [x["uptake_mmol_g"] for x in m_cls],
            "s-", color="#2563EB", lw=1.8, ms=5.5, label="Mercado 2016 DFT-FF (Direct GPU)")
    ax.errorbar([x["P_bar"] for x in mg_r2], [x["uptake_mmol_g"] for x in mg_r2],
                yerr=[x["uptake_sem_mmol_g"] for x in mg_r2], fmt="o-", color="#0F766E",
                lw=2.0, ms=6.0, capsize=3, label="r2SCAN-0+D3 (1200 cyc @ 0.01/0.05)")
    ax.plot([r["P_bar"] for r in exp_mg_1], [r["uptake_mmol_g"] for r in exp_mg_1],
            "*:", color="#111827", lw=1.5, ms=8.5, zorder=7, label="ISODB 10.1021/acs.jpcc.7b08239")
    ax.plot([r["P_bar"] for r in exp_mg_2], [r["uptake_mmol_g"] for r in exp_mg_2],
            "x:", color="#475569", lw=1.2, ms=5.5, zorder=6, label="ISODB 10.1039/c5dt03137k")
    format_log_x(ax, [0.002, 0.01, 0.05, 0.15])
    ax.set_xlim(0.0015, 0.25)
    ax.set_ylim(0, 16.8)
    ax.set_xlabel("Pressure (bar, log scale)")
    ax.set_ylabel(r"$\mathrm{CO_2}$ Uptake (mmol/g)")
    style_ax(ax)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1")
    ax.set_title("(a) Mg-MOF-74 (298 K) — Closed-Shell OMS")
    ax.annotate(
        "Systematic ~1.8-3.0x overprediction\nvs. NIST ISODB (DFT ground truth needed)",
        xy=(0.97, 0.96), xycoords="axes fraction", ha="right", va="top", fontsize=7.3,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#FFFBEB", edgecolor="#D97706"),
    )

    # Panel (b): Ni-MOF-74
    ax = axes[0, 1]
    ax.plot([x["P_bar"] for x in ni_cls["UFF_templateQ"]], [x["uptake_mmol_g"] for x in ni_cls["UFF_templateQ"]],
            "^--", color="#64748B", lw=1.6, ms=5.5, label="UFF (template Q, Direct GPU)")
    ax.plot([x["P_bar"] for x in ni_cls["Pol3Site"]], [x["uptake_mmol_g"] for x in ni_cls["Pol3Site"]],
            "s-", color="#2563EB", lw=1.8, ms=5.5, label="Pol-3Site (Direct GPU)")
    ax.errorbar([x["P_bar"] for x in ni_r2], [x["uptake_mmol_g"] for x in ni_r2],
                yerr=[x["uptake_sem_mmol_g"] for x in ni_r2], fmt="o-", color="#0F766E",
                lw=2.0, ms=6.0, capsize=3, label="r2SCAN-0+D3 (480 cyc @ 0.05/0.15)")
    ax.plot([r["P_bar"] for r in exp_ni], [r["uptake_mmol_g"] for r in exp_ni],
            "*:", color="#111827", lw=1.5, ms=8.5, zorder=7, label="ISODB 10.1021/acs.jpcc.5b03700")
    format_log_x(ax, [0.01, 0.05, 0.15])
    ax.set_xlim(0.007, 0.22)
    ax.set_ylim(0, 6.2)
    ax.set_xlabel("Pressure (bar, log scale)")
    ax.set_ylabel(r"$\mathrm{CO_2}$ Uptake (mmol/g)")
    style_ax(ax)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1")
    ax.set_title("(b) Ni-MOF-74 (298 K) — Open-Shell Isostructural OMS")
    ax.annotate(
        f"0.15 bar: {ni_r2[-1]['uptake_mmol_g']:.2f} vs 4.46 mmol/g (ISODB)",
        xy=(0.03, 0.56), xycoords="axes fraction", ha="left", va="center", fontsize=7.4,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#F0FDFA", edgecolor="#0F766E"),
    )

    # Panel (c): HKUST-1
    ax = axes[1, 0]
    ax.plot([x["P_bar"] for x in hk_cls["UFF"]], [x["uptake_mmol_g"] for x in hk_cls["UFF"]],
            "^--", color="#64748B", lw=1.6, ms=5.5, label="UFF+Charges (Direct GPU)")
    ax.plot([x["P_bar"] for x in hk_cls["Pol3Site"]], [x["uptake_mmol_g"] for x in hk_cls["Pol3Site"]],
            "s-", color="#2563EB", lw=1.8, ms=5.5, label="Pol-3Site (Direct GPU)")
    ax.errorbar([x["P_bar"] for x in hk_r2], [x["uptake_mmol_g"] for x in hk_r2],
                yerr=[x["uptake_sem_mmol_g"] for x in hk_r2], fmt="o-", color="#0F766E",
                lw=2.0, ms=6.0, capsize=3, label="r2SCAN-0+D3 (PilotStage3, 480 cyc)")
    ax.plot([r["P_bar"] for r in exp_hk], [r["uptake_mmol_g"] for r in exp_hk],
            "-", color="#111827", lw=1.6, alpha=0.82, zorder=6, label="ISODB 10.1002/aic.15837 (n=79 pts)")
    format_log_x(ax, [0.15, 0.50, 1.00])
    ax.set_xlim(0.10, 1.30)
    ax.set_ylim(0, 6.2)
    ax.set_xlabel("Pressure (bar, log scale)")
    ax.set_ylabel(r"$\mathrm{CO_2}$ Uptake (mmol/g)")
    style_ax(ax)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1")
    ax.set_title("(c) HKUST-1 (298 K) — Equilibrated Paddlewheel OMS")
    ax.annotate(
        f"Monotonic equilibration (Ins/Del=76/52):\n1.0 bar = {hk_r2[-1]['uptake_mmol_g']:.2f} vs 4.53 mmol/g (ISODB)",
        xy=(0.97, 0.05), xycoords="axes fraction", ha="right", va="bottom", fontsize=7.4,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#F0FDFA", edgecolor="#0F766E"),
    )

    # Panel (d): IRMOF-1 (Non-OMS Control)
    ax = axes[1, 1]
    ax.errorbar([x["P_bar"] for x in ir_uff], [x["uptake_mmol_g"] for x in ir_uff],
                yerr=[x["uptake_sem_mmol_g"] for x in ir_uff], fmt="^--", color="#64748B",
                lw=1.7, ms=5.8, capsize=3, label="UFF 0-charge (Direct GPU)")
    ax.errorbar([x["P_bar"] for x in ir_r2], [x["uptake_mmol_g"] for x in ir_r2],
                yerr=[x["uptake_sem_mmol_g"] for x in ir_r2], fmt="o-", color="#DC2626",
                lw=2.0, ms=6.0, capsize=3, label="r2SCAN-0+D3 (PilotStage3, 360 cyc)")
    ax.plot([r["P_bar"] for r in exp_ir_1], [r["uptake_mmol_g"] for r in exp_ir_1],
            "*:", color="#111827", lw=1.6, ms=8.5, zorder=7, label="ISODB 10.1021/acs.jctc.5b00093")
    ax.plot([r["P_bar"] for r in exp_ir_2], [r["uptake_mmol_g"] for r in exp_ir_2],
            "d:", color="#475569", lw=1.3, ms=5.0, zorder=6, label="ISODB 10.1038/srep12045")
    format_log_x(ax, [0.5, 1.0, 2.0, 5.0])
    ax.set_xlim(0.38, 6.8)
    ax.set_ylim(0, 15.8)
    ax.set_xlabel("Pressure (bar, log scale)")
    ax.set_ylabel(r"$\mathrm{CO_2}$ Uptake (mmol/g)")
    style_ax(ax)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1")
    ax.set_title("(d) IRMOF-1 / MOF-5 (298 K) — Non-OMS Positive Control")
    ax.annotate(
        "Even with ZERO open metal sites,\nr2SCAN-0+D3 overshoots ISODB by 2.5-2.9x\n(+4-5 kJ/mol dispersion/model bias)",
        xy=(0.04, 0.56), xycoords="axes fraction", ha="left", va="center", fontsize=7.4,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#FEF2F2", edgecolor="#DC2626"),
    )

    fig.suptitle(
        "Stage-1 Four-MOF Benchmark vs. Curated NIST ISODB Experimental Tables (298 K)",
        fontsize=12.0, fontweight="bold", y=0.965,
    )
    fig.text(
        0.5, 0.93,
        "Non-OMS IRMOF-1 isolates a systematic ~4-5 kJ/mol background overbinding in MACE-MatPES-r2SCAN-0+D3(r2scan), motivating Stage-2 DFT single-point checks",
        ha="center", fontsize=8.6, color="#475569",
    )
    out = ASSETS / "stage1_four_mof_nist_isodb.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def make_fig_qst_fp32():
    fp32 = load_json("stage1_fp32_parity.json")
    mg_480 = load_json("paper_exp3_mgmof74_r2scan_long.json")["points"]
    mg_1200 = load_json("stage1_mgmof74_r2scan_deep.json")["points"]
    mg_313 = load_json("stage1_mgmof74_313K.json")["points"][0]
    ni_298 = [pt for pt in load_json("paper_exp3_nimof74_r2scan_long.json")["points"] if abs(pt["P_bar"] - 0.05) < 1e-6][0]
    ni_313 = load_json("stage1_nimof74_313K.json")["points"][0]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.8, 4.9), dpi=300)
    fig.subplots_adjust(left=0.052, right=0.980, top=0.80, bottom=0.145, wspace=0.27)

    # Panel (a): FP32 vs FP64 Parity
    metrics = ["Max |dU_tot|", "Max |dU/N|", "Max Pair |ddU|"]
    r2_vals = [
        fp32[0]["max_abs_err_total_kJ_mol"],
        fp32[0]["max_abs_err_per_mol_kJ_mol"],
        fp32[0]["max_pair_dU_err_kJ_mol"],
    ]
    mp_vals = [
        fp32[1]["max_abs_err_total_kJ_mol"],
        fp32[1]["max_abs_err_per_mol_kJ_mol"],
        fp32[1]["max_pair_dU_err_kJ_mol"],
    ]
    xx = np.arange(len(metrics))
    ww = 0.34
    b1 = ax1.bar(xx - ww / 2, r2_vals, width=ww, color="#0F766E", alpha=0.88, edgecolor="#1E293B", label="r2SCAN-0+D3 (<= 0.141 kB*T)")
    b2 = ax1.bar(xx + ww / 2, mp_vals, width=ww, color="#D97706", alpha=0.86, edgecolor="#1E293B", label="MACE-MP-0+D3 (<= 0.190 kB*T)")
    for rect, v in zip(b1, r2_vals):
        ax1.text(rect.get_x() + rect.get_width() / 2.0, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    for rect, v in zip(b2, mp_vals):
        ax1.text(rect.get_x() + rect.get_width() / 2.0, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    ax1.set_xticks(xx)
    ax1.set_xticklabels(metrics)
    ax1.set_ylim(0, 0.62)
    ax1.set_ylabel("FP32 vs. FP64 Max Error (kJ/mol)")
    style_ax(ax1)
    ax1.grid(axis="x", visible=False)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1")
    ax1.set_title("(a) Live FP32 vs. FP64 Parity (15 Snapshots)")

    # Panel (b): Differential Qst Convergence (480 vs 1200) vs Integral Enthalpy
    p480 = {round(pt["P_bar"], 4): pt for pt in mg_480}
    pts_480 = [p480[0.01], p480[0.05]]
    pts_1200 = mg_1200  # 0.01, 0.05 bar

    x2 = np.arange(2)
    w2 = 0.25
    q480 = [pt["Qst_kJ_mol"] for pt in pts_480]
    e480 = [pt["Qst_sem_kJ_mol"] for pt in pts_480]
    q1200 = [pt["Qst_kJ_mol"] for pt in pts_1200]
    e1200 = [pt["Qst_sem_kJ_mol"] for pt in pts_1200]
    qint = [pt["mean_molar_qst_kJ_mol"] for pt in pts_1200]
    eint = [pt["mean_molar_qst_sem_kJ_mol"] for pt in pts_1200]

    ax2.bar(x2 - w2, q480, width=w2, yerr=e480, capsize=3, color="#D97706", alpha=0.85, edgecolor="#1E293B", label="Diff Qst (Fluc, 480 cyc)")
    b_d = ax2.bar(x2, q1200, width=w2, yerr=e1200, capsize=3, color="#0F766E", alpha=0.90, edgecolor="#1E293B", label="Diff Qst (Fluc, 1200 cyc)")
    b_i = ax2.bar(x2 + w2, qint, width=w2, yerr=eint, capsize=3, color="#2563EB", alpha=0.84, edgecolor="#1E293B", label="Integral -<U/N>+RT (1200 cyc)")

    for rect, v, e in zip(b_d, q1200, e1200):
        ax2.text(rect.get_x() + rect.get_width() / 2.0, v + e + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#0F766E")
    for rect, v, e in zip(b_i, qint, eint):
        ax2.text(rect.get_x() + rect.get_width() / 2.0, v + e + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#2563EB")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(["0.01 bar\n(6.34 mmol/g)", "0.05 bar\n(9.47 > 8.24 OMS)"])
    ax2.set_ylim(0, 66)
    ax2.set_ylabel("Heat / Enthalpy (kJ/mol)")
    style_ax(ax2)
    ax2.grid(axis="x", visible=False)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1", fontsize=6.9)
    ax2.set_title("(b) Differential Qst Convergence vs. Integral")
    ax2.annotate(
        "0.05 bar Diff SEM:\n+-10.56 -> +-1.97 kJ/mol (5.4x)",
        xy=(1.0, 38.5), xytext=(0.48, 12.0), ha="center", fontsize=7.3,
        arrowprops=dict(arrowstyle="->", color="#0F766E", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0FDFA", edgecolor="#0F766E"),
    )

    # Panel (c): 298 K vs 313 K Thermal Desorption
    cats = ["Mg-MOF-74\n(0.01 bar)", "Ni-MOF-74\n(0.05 bar)"]
    u298 = [mg_1200[0]["uptake_mmol_g"], ni_298["uptake_mmol_g"]]
    eu298 = [mg_1200[0]["uptake_sem_mmol_g"], ni_298["uptake_sem_mmol_g"]]
    u313 = [mg_313["uptake_mmol_g"], ni_313["uptake_mmol_g"]]
    eu313 = [mg_313["uptake_sem_mmol_g"], ni_313["uptake_sem_mmol_g"]]

    x3 = np.arange(len(cats))
    w3 = 0.34
    r1 = ax3.bar(x3 - w3 / 2, u298, width=w3, yerr=eu298, capsize=3.5, color="#0F766E", alpha=0.88, edgecolor="#1E293B", label="298 K (r2SCAN-0+D3)")
    r2 = ax3.bar(x3 + w3 / 2, u313, width=w3, yerr=eu313, capsize=3.5, color="#DC2626", alpha=0.84, edgecolor="#1E293B", label="313 K (r2SCAN-0+D3)")
    for rect, v, e in zip(r1, u298, eu298):
        ax3.text(rect.get_x() + rect.get_width() / 2.0, v + e + 0.16, f"{v:.2f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    for rect, v, e in zip(r2, u313, eu313):
        ax3.text(rect.get_x() + rect.get_width() / 2.0, v + e + 0.16, f"{v:.2f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(cats)
    ax3.set_ylim(0, 8.2)
    ax3.set_ylabel(r"$\mathrm{CO_2}$ Uptake (mmol/g)")
    style_ax(ax3)
    ax3.grid(axis="x", visible=False)
    ax3.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1")
    ax3.set_title("(c) Thermal Desorption Check (298 vs. 313 K)")

    fig.suptitle(
        "Stage-1 Numerical & Thermodynamic Diagnostics: FP32 Parity, 1200-Cycle Diff Qst & Thermal Swing",
        fontsize=11.6, fontweight="bold", y=0.965,
    )
    fig.text(
        0.5, 0.90,
        "Verified FP32 noise <= 0.19 kB*T, 5.4x fluctuation Qst SEM reduction separating differential vs. integral heat, and consistent 313 K desorption",
        ha="center", fontsize=8.5, color="#475569",
    )
    out = ASSETS / "stage1_qst_fp32_diagnostics.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    setup_rc()
    make_fig_four_mof()
    make_fig_qst_fp32()


if __name__ == "__main__":
    main()
