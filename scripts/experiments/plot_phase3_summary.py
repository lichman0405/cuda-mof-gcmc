"""Generate Phase 3 300 DPI 3-panel publication summary figure:
  (a) Multi-Foundation-Model Loaded Binding Benchmark on Mg-MOF-74
  (b) Mg-MOF-74 (298 K) Multi-Pressure CO2 Isotherms (Classical vs MLIPs)
  (c) Zero-Fitting Cross-Material Generalization on Ni-MOF-74 (298 K)
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
ASSETS = ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def load_json(name):
    return json.loads((BUILD / name).read_text(encoding="utf-8"))


def main():
    probe = load_json("phase3_model_probe.json")
    mg_cls = load_json("phase3_mgmof74_classical.json")
    mg_mp0 = load_json("phase3_mgmof74_mace_mp0_d3.json")["points"]
    mg_r2 = load_json("phase3_mgmof74_r2scan_d3.json")["points"]
    ni_uff0 = load_json("phase3_nimof74_uff.json")
    ni_cls = load_json("phase3_nimof74_classical.json")
    ni_r2 = load_json("phase3_nimof74_r2scan_d3.json")["points"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.9,
        "axes.titlesize": 10.1,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.2,
        "legend.fontsize": 7.5,
    })
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.1), dpi=300)
    fig.subplots_adjust(left=0.055, right=0.98, top=0.81, bottom=0.14, wspace=0.28)

    # Panel (a): Multi-Foundation-Model Loaded -U/N Diagnostic on Mg-MOF-74
    ax1 = axes[0]
    # Add exact measured values from probe + r2SCAN No-D3 & r2SCAN+D3(r2scan) (-27.45 and -40.13 kJ/mol from task-1829 / task-1836)
    labels = [
        "MP-0\n+D3(PBE)",
        "MPA-0\n+D3(PBE)",
        "MP-0b3\n+D3(PBE)",
        "r2SCAN-0\n(No D3)",
        "r2SCAN-0\n+D3(r2scan)",
    ]
    # Exact values: probe[0]=-41.63, probe[1]=-82.84, probe[2]=-92.15, probe[3](NoD3)=-25.47, task-1836 mean=40.13
    vals = [
        -probe[0]["loaded_mean_U_per_N_kJ_mol"],
        -probe[1]["loaded_mean_U_per_N_kJ_mol"],
        -probe[2]["loaded_mean_U_per_N_kJ_mol"],
        -probe[3]["loaded_mean_U_per_N_kJ_mol"],
        40.13,
    ]
    cols = ["#D97706", "#DC2626", "#991B1B", "#64748B", "#0F766E"]
    bars = ax1.bar(np.arange(len(labels)), vals, width=0.56, color=cols, alpha=0.86, edgecolor="#1E293B")
    ax1.axhspan(37.5, 44.5, color="#0F766E", alpha=0.14, label="Exp. |ΔU| Range (~38–45 kJ/mol)")
    ax1.axhline(33.5, color="#2563EB", linestyle="--", linewidth=1.6, label="Mercado 2016 DFT-FF (~33.5 kJ/mol)")
    for rect, v in zip(bars, vals):
        ax1.text(rect.get_x() + rect.get_width() / 2.0, v + 1.8, f"{v:.1f}", ha="center", va="bottom", fontsize=8.1, fontweight="bold")
    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_xticklabels(labels, fontsize=7.9)
    ax1.set_ylim(0, 112)
    ax1.set_ylabel(r"Loaded Mean Binding $-U/N$ ($\mathrm{kJ/mol\,CO_2}$)")
    ax1.grid(axis="y", linestyle=":", alpha=0.45)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1")
    ax1.set_title("(a) Mg-MOF-74 Foundation Model Overbinding Audit")

    # Panel (b): Mg-MOF-74 (298 K) Multi-Pressure Isotherms
    ax2 = axes[1]
    p_cls = [x["P_bar"] for x in mg_cls["UFF_REPEAT"] if x["P_bar"] <= 0.16]
    u_uff = [x["uptake_mmol_g"] for x in mg_cls["UFF_REPEAT"] if x["P_bar"] <= 0.16]
    u_merc = [x["uptake_mmol_g"] for x in mg_cls["Mercado_2016"] if x["P_bar"] <= 0.16]
    p_mp0 = [x["P_bar"] for x in mg_mp0]
    u_mp0 = [x["uptake_mmol_g"] for x in mg_mp0]
    e_mp0 = [x["uptake_sem_mmol_g"] for x in mg_mp0]
    p_r2 = [x["P_bar"] for x in mg_r2]
    u_r2 = [x["uptake_mmol_g"] for x in mg_r2]
    e_r2 = [x["uptake_sem_mmol_g"] for x in mg_r2]

    ax2.plot(p_cls, u_uff, "^--", color="#64748B", linewidth=1.6, markersize=6, label="UFF+REPEAT (Direct GPU)")
    ax2.plot(p_cls, u_merc, "s-", color="#2563EB", linewidth=2.0, markersize=6, label="Mercado 2016 (Direct GPU)")
    ax2.errorbar(p_r2, u_r2, yerr=e_r2, fmt="o-", color="#0F766E", linewidth=2.1, markersize=6.5, capsize=3, label="r2SCAN-0+D3(r2scan) Nested MC")
    ax2.errorbar(p_mp0, u_mp0, yerr=e_mp0, fmt="D-", color="#D97706", linewidth=1.9, markersize=5.8, capsize=3, label="MACE-MP-0+D3(PBE) Nested MC")
    ax2.errorbar([0.15], [4.1], yerr=[[0.5], [0.5]], fmt="*", color="#1E293B", markersize=11, capsize=4, zorder=6, label="Exp. Defect-Limited Range (0.15 bar)")
    ax2.set_xscale("log")
    ax2.set_xlim(0.0015, 0.22)
    ax2.set_xticks([0.002, 0.01, 0.05, 0.15])
    ax2.set_xticklabels(["0.002", "0.01", "0.05", "0.15"])
    ax2.set_ylim(0, 17.5)
    ax2.set_xlabel("Pressure (bar, log scale)")
    ax2.set_ylabel(r"$\mathrm{CO_2}$ Uptake at 298 K (mmol/g)")
    ax2.grid(True, which="both", linestyle=":", alpha=0.45)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1")
    ax2.set_title("(b) Mg-MOF-74 (298 K): Classical vs MLIP Isotherms")

    # Panel (c): Cross-Material Generalization on Ni-MOF-74 (298 K)
    ax3 = axes[2]
    p_ni = [x["P_bar"] for x in ni_r2]
    u_ni_uff0 = [x["uptake_mmol_g"] for x in ni_uff0]
    u_ni_uffq = [x["uptake_mmol_g"] for x in ni_cls["UFF_templateQ"]]
    u_ni_pol3 = [x["uptake_mmol_g"] for x in ni_cls["Pol3Site"]]
    u_ni_mlip = [x["uptake_mmol_g"] for x in ni_r2]
    e_ni_mlip = [x["uptake_sem_mmol_g"] for x in ni_r2]

    ax3.plot(p_ni, u_ni_uff0, "v:", color="#94A3B8", linewidth=1.6, markersize=5.8, label="Zero-Q Generic UFF (Qst=17.4 kJ/mol)")
    ax3.plot(p_ni, u_ni_uffq, "^--", color="#64748B", linewidth=1.6, markersize=5.8, label="UFF+Template-Q (Qst=26.9 kJ/mol)")
    ax3.plot(p_ni, u_ni_pol3, "s--", color="#2563EB", linewidth=1.8, markersize=5.8, label="Pol-3Site Proposer (Qst=34.5 kJ/mol)")
    ax3.errorbar(p_ni, u_ni_mlip, yerr=e_ni_mlip, fmt="o-", color="#0F766E", linewidth=2.2, markersize=6.8, capsize=3, label="r2SCAN-0+D3(r2scan) Nested MC (Zero-Fit)")
    ax3.errorbar([0.15], [3.9], yerr=[[0.4], [0.5]], fmt="*", color="#1E293B", markersize=11, capsize=4, zorder=6, label="Exp. Ni-MOF-74 Range (0.15 bar)")
    ax3.set_xscale("log")
    ax3.set_xlim(0.008, 0.22)
    ax3.set_xticks([0.01, 0.05, 0.15])
    ax3.set_xticklabels(["0.01", "0.05", "0.15"])
    ax3.set_ylim(0, 6.4)
    ax3.set_xlabel("Pressure (bar, log scale)")
    ax3.set_ylabel(r"$\mathrm{CO_2}$ Uptake at 298 K (mmol/g)")
    ax3.grid(True, which="both", linestyle=":", alpha=0.45)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1")
    ax3.set_title("(c) Ni-MOF-74 (298 K): Zero-Fit Cross-Material Test")

    fig.suptitle(
        "Phase 3 AI4Science Benchmark: Universal MLIP Foundation-Model Audit, Mg-MOF-74 Isotherms & Ni-MOF-74 Generalization",
        fontsize=11.4, fontweight="bold", y=0.95
    )
    out_png = ASSETS / "phase3_mlip_isotherm_and_generalization.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
