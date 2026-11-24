"""Head-to-head ESS / GPU-s & SJD_N / GPU-s speedup benchmark:
Compares GPU-proposed Nested MLIP-GCMC against:
  1) Serial ASE Direct MLIP-GCMC (literature MLIP-MC baseline + hard-sphere screen), and
  2) Strong Batched-GPU Hard-Sphere-Screened Direct MLIP-GCMC (exact same FastMaceD3Target).
"""

import argparse
import csv
import gc
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from direct_mlip_gcmc import SerialAseMaceD3Target, run_direct_gcmc
from guests import load_guest
from nested_mc_engine import CudaGcmcProposer, FastMaceD3Target, K_TO_KJMOL, run_nested_mc
from qeq_charges import extract_framework_for_proposer
from run_mlip_gcmc import compute_grid_dims

ROOT = Path(__file__).resolve().parent.parent

SYSTEM_DEFAULTS = {
    "mgmof74_co2": {
        "cif": ROOT / "data" / "MgMOF-74.cif",
        "supercell": (1, 1, 2),
        "guest": "co2",
        "T_K": 298.0,
        "P_bar": 0.05,
        "model": "mace-matpes-r2scan-0",
        "dispersion": True,
        "xc": "r2scan",
        "mace_chunk": None,
        "warm_proposer": "mercado",
        "warm_charge": "auto",
        "warm_pilot": 40,
        "warm_eq": 60,
        "max_mol": 128,
    },
    "irmof1_ch4": {
        "cif": ROOT / "data" / "IRMOF-1.cif",
        "supercell": (1, 1, 1),
        "guest": "ch4",
        "T_K": 298.0,
        "P_bar": 5.0,
        "model": "medium",
        "dispersion": True,
        "xc": "pbe",
        "mace_chunk": 1,
        "warm_proposer": "uff",
        "warm_charge": "auto",
        "warm_pilot": 30,
        "warm_eq": 40,
        "max_mol": 256,
    },
}


def _free_gpu_cache():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _calibrate_shift_only(prop: CudaGcmcProposer, target: FastMaceD3Target, k_steps: int, pilot_cycles: int, seed: int):
    n_b = prop.n_boxes
    prop.equilibrate(20000)
    cal1_st = prop.get_accepted_quat()
    if np.mean([s["N"] for s in cal1_st]) < 4.0:
        for boost in (-4.0, -8.0, -12.0, -16.0):
            prop.set_chem_pot_shift(boost)
            prop.equilibrate(12000)
            cal1_st = prop.get_accepted_quat()
            if np.mean([s["N"] for s in cal1_st]) >= 4.0:
                break
    cal1_u = target.eval_batch(cal1_st)
    d1 = [float(((cal1_u[b][0] - cal1_st[b]["U_ref_K"]) * K_TO_KJMOL) / cal1_st[b]["N"])
          for b in range(n_b) if cal1_st[b]["N"] > 0]
    de1 = float(max(-24.0, min(2.0, np.mean(d1)))) if d1 else 0.0
    prop.set_chem_pot_shift(de1)
    prop.equilibrate(25000)

    cal2_st = prop.get_accepted_quat()
    cal2_u = target.eval_batch(cal2_st)
    d2 = [float(((cal2_u[b][0] - cal2_st[b]["U_ref_K"]) * K_TO_KJMOL) / cal2_st[b]["N"])
          for b in range(n_b) if cal2_st[b]["N"] > 0]
    de2 = float(max(-24.0, min(2.0, np.mean(d2)))) if d2 else de1
    prop.set_chem_pot_shift(de2)
    prop.equilibrate(15000)

    if pilot_cycles >= 16:
        p1 = pilot_cycles // 2
        p2 = pilot_cycles - p1
        _ = run_nested_mc(prop, None, k_steps=k_steps, eq_cycles=p1, prod_cycles=0,
                          seed=seed + 5, batch_target_fn=target.eval_batch)
        s3a = prop.get_accepted_quat()
        u3a = target.eval_batch(s3a)
        d3a = [float(((u3a[b][0] - s3a[b]["U_ref_K"]) * K_TO_KJMOL) / s3a[b]["N"])
               for b in range(n_b) if s3a[b]["N"] > 0]
        de3a = float(max(-24.0, min(4.0, np.mean(d3a)))) if d3a else de2
        prop.set_chem_pot_shift(de3a)
        _ = run_nested_mc(prop, None, k_steps=k_steps, eq_cycles=p2, prod_cycles=0,
                          seed=seed + 9, batch_target_fn=target.eval_batch)
        s3b = prop.get_accepted_quat()
        u3b = target.eval_batch(s3b)
        d3b = [float(((u3b[b][0] - s3b[b]["U_ref_K"]) * K_TO_KJMOL) / s3b[b]["N"])
               for b in range(n_b) if s3b[b]["N"] > 0]
        de_final = float(max(-24.0, min(4.0, np.mean(d3b)))) if d3b else de3a
        prop.set_chem_pot_shift(de_final)
        prop.equilibrate(10000)
    else:
        de_final = de2
    return de_final


def _format_row(system_name: str, method_name: str, kind: str, res: dict, k_steps: int,
                n_boxes: int, T_K: float, P_bar: float, mlip_evals: int,
                model_name: str, xc_name: str, hs_rejected: int = 0) -> dict:
    return {
        "system": system_name,
        "method": method_name,
        "kind": kind,
        "model": model_name,
        "xc": xc_name,
        "T_K": float(T_K),
        "P_bar": float(P_bar),
        "n_boxes": int(n_boxes),
        "k_steps": int(k_steps),
        "elapsed_s": float(res["elapsed_s"]),
        "mlip_evals": int(mlip_evals),
        "hs_rejected": int(hs_rejected),
        "uptake_mmol_g": float(res["uptake_mmol_g"]),
        "uptake_sem_mmol_g": float(res["uptake_sem_mmol_g"]),
        "Qst_kJ_mol": float(res["Qst_kJ_mol"]),
        "Qst_sem_kJ_mol": float(res["Qst_sem_kJ_mol"]),
        "acc_rate": float(res.get("prod_outer_acc_rate", res.get("prod_move_acc_rate", 0.0))),
        "prod_ins_acc": int(res["prod_ins_acc"]),
        "prod_ins_att": int(res["prod_ins_att"]),
        "prod_del_acc": int(res["prod_del_acc"]),
        "prod_del_att": int(res["prod_del_att"]),
        "tau_int_N": float(res["tau_int_N"]),
        "ess_N": float(res["ess_N"]),
        "ess_N_per_s": float(res["ess_N_per_s"]),
        "tau_int_U": float(res["tau_int_U"]),
        "ess_U": float(res["ess_U"]),
        "ess_U_per_s": float(res["ess_U_per_s"]),
        "esjd_N": float(res["esjd_N"]),
        "sjd_N_per_s": float(res["sjd_N_per_s"]),
        "dN_moves_per_s": float(res["dN_moves_per_s"]),
    }


def _compute_speedups(rows: list) -> list:
    by_sys = {}
    for r in rows:
        by_sys.setdefault(r["system"], {})[r["method"]] = r
    out = []
    for r in rows:
        row = dict(r)
        m_map = by_sys[r["system"]]
        ser = m_map.get("direct_serial_ase")
        bat = m_map.get("direct_batched_gpu")
        for tag, base in (("vs_serial", ser), ("vs_batched", bat)):
            if base is None:
                row[f"speedup_essN_{tag}"] = None
                row[f"speedup_sjdN_{tag}"] = None
                row[f"speedup_dN_{tag}"] = None
            else:
                row[f"speedup_essN_{tag}"] = float(r["ess_N_per_s"] / max(1e-9, base["ess_N_per_s"]))
                row[f"speedup_sjdN_{tag}"] = float(r["sjd_N_per_s"] / max(1e-9, base["sjd_N_per_s"]))
                row[f"speedup_dN_{tag}"] = float(r["dN_moves_per_s"] / max(1e-9, base["dN_moves_per_s"]))
        out.append(row)
    return out


def benchmark_system(
    system_name: str,
    cif_path: str,
    supercell: tuple,
    guest_name: str,
    T_K: float,
    P_bar: float,
    model: str,
    dispersion: bool,
    dispersion_xc: str,
    mace_chunk: int = None,
    warm_proposer: str = "mercado",
    warm_charge: str = "auto",
    warm_pilot: int = 40,
    warm_eq: int = 60,
    max_mol: int = 128,
):
    print(f"\n================ BENCHMARK: {system_name} ({cif_path}, {guest_name}, T={T_K}K, P={P_bar}bar, {model}+D3({dispersion_xc})) ================")
    ff_warm = 1 if warm_proposer == "mercado" else 0
    guest_warm = load_guest(guest_name, proposer=warm_proposer)
    guest_uff = load_guest(guest_name, proposer="uff")
    fw_warm = extract_framework_for_proposer(cif_path, charge_mode=warm_charge)
    grid_dims = compute_grid_dims(fw_warm["uc"], 0.11)

    target = FastMaceD3Target(
        cif_path=cif_path, supercell=supercell, model=model, device="cuda",
        dispersion=dispersion, dispersion_xc=dispersion_xc, guest=guest_warm,
        mace_chunk=mace_chunk, uc_atoms=fw_warm["uc"],
    )

    # 1. Shared MLIP equilibrium warm-start
    print(f"  [Warm-Start] Running {warm_proposer} pilot={warm_pilot} + eq={warm_eq} to get shared 4 equilibrated states...")
    p_warm = CudaGcmcProposer.from_atoms(
        fw_data=fw_warm, guest=guest_warm, supercell=supercell,
        T_K=T_K, pressure_bar=P_bar, ff_mode=ff_warm, pol_mode=0,
        grid_dims=grid_dims, n_boxes=4, max_mol=max_mol, seed=20261700,
    )
    de_warm = _calibrate_shift_only(p_warm, target, k_steps=25, pilot_cycles=warm_pilot, seed=20261701)
    res_warm = run_nested_mc(
        p_warm, None, k_steps=25, eq_cycles=warm_eq, prod_cycles=0,
        seed=20261702, batch_target_fn=target.eval_batch,
    )
    init_states = res_warm["final_states"]
    conv = 1000.0 / p_warm.fw_mass_gmol
    fw_mass_gmol = p_warm.fw_mass_gmol
    print(f"  [Warm-Start Done] shift={de_warm:.2f} kJ/mol, N={[s['N'] for s in init_states]} ({[round(s['N']*conv, 2) for s in init_states]} mmol/g)")
    p_warm.close()

    rows = []

    # 2. Direct Batched-GPU with Hard-Sphere Screen (reuses FastMaceD3Target)
    is_mg = (system_name == "mgmof74_co2")
    print("  [1/N] Running direct_batched_gpu (n_boxes=4, r_hs=1.20 A)...")
    target.n_evals = 0
    res_bat = run_direct_gcmc(
        uc_atoms=fw_warm["uc"], guest=guest_uff, supercell=supercell,
        T_K=T_K, pressure_bar=P_bar, batch_target_fn=target.eval_batch,
        n_boxes=4, eq_steps=0, prod_steps=280 if is_mg else 240,
        initial_states=init_states, r_hs_A=1.20, max_mol=max_mol,
        seed=20261710, max_wall_s=75.0, log_every=80, fw_mass_gmol=fw_mass_gmol,
    )
    rows.append(_format_row(system_name, "direct_batched_gpu", "direct_batched", res_bat, 1, 4, T_K, P_bar, res_bat["mlip_evals"], model, dispersion_xc, res_bat["hs_rejected"]))
    print(f"    -> elapsed={res_bat['elapsed_s']:.1f}s steps={res_bat['prod_steps_done']} evals={res_bat['mlip_evals']} HSrej={res_bat['hs_rejected']} essN/s={res_bat['ess_N_per_s']:.3f} sjdN/s={res_bat['sjd_N_per_s']:.3f} dN/s={res_bat['dN_moves_per_s']:.3f}")

    # 3. Nested-GPU runs reusing FastMaceD3Target
    if is_mg:
        for k_val, p_cyc, s_off in ((10, 180, 20), (25, 240, 30), (50, 180, 40)):
            m_name = f"nested_mercado_k{k_val}"
            print(f"  [Nested] Running {m_name} (n_boxes=4, prod={p_cyc} cyc)...")
            prop = CudaGcmcProposer.from_atoms(
                fw_data=fw_warm, guest=guest_warm, supercell=supercell,
                T_K=T_K, pressure_bar=P_bar, ff_mode=1, pol_mode=0,
                grid_dims=grid_dims, n_boxes=4, max_mol=max_mol, seed=20261700 + s_off,
            )
            prop.set_chem_pot_shift(de_warm)
            prop.set_accepted_quat(init_states)
            target.n_evals = 0
            r_k = run_nested_mc(
                prop, None, k_steps=k_val, eq_cycles=0, prod_cycles=p_cyc,
                seed=20261705 + s_off, batch_target_fn=target.eval_batch,
            )
            rows.append(_format_row(system_name, m_name, "nested_gpu", r_k, k_val, 4, T_K, P_bar, target.n_evals, model, dispersion_xc))
            print(f"    -> elapsed={r_k['elapsed_s']:.1f}s uptake={r_k['uptake_mmol_g']:.3f}+-{r_k['uptake_sem_mmol_g']:.3f} Acc={r_k['prod_outer_acc_rate']*100:.1f}% essN/s={r_k['ess_N_per_s']:.3f} sjdN/s={r_k['sjd_N_per_s']:.3f} dN/s={r_k['dN_moves_per_s']:.3f}")
            prop.close()
            _free_gpu_cache()
    else:
        for k_val, p_cyc, s_off in ((25, 200, 70), (50, 200, 80)):
            m_name = f"nested_uff_qeq_k{k_val}"
            print(f"  [Nested] Running {m_name} (n_boxes=4, prod={p_cyc} cyc)...")
            prop_u = CudaGcmcProposer.from_atoms(
                fw_data=fw_warm, guest=guest_uff, supercell=supercell,
                T_K=T_K, pressure_bar=P_bar, ff_mode=0, pol_mode=0,
                grid_dims=grid_dims, n_boxes=4, max_mol=max_mol, seed=20261700 + s_off,
            )
            prop_u.set_chem_pot_shift(de_warm)
            prop_u.set_accepted_quat(init_states)
            target.n_evals = 0
            r_u = run_nested_mc(
                prop_u, None, k_steps=k_val, eq_cycles=0, prod_cycles=p_cyc,
                seed=20261705 + s_off, batch_target_fn=target.eval_batch,
            )
            rows.append(_format_row(system_name, m_name, "nested_gpu", r_u, k_val, 4, T_K, P_bar, target.n_evals, model, dispersion_xc))
            print(f"    -> elapsed={r_u['elapsed_s']:.1f}s uptake={r_u['uptake_mmol_g']:.3f}+-{r_u['uptake_sem_mmol_g']:.3f} Acc={r_u['prod_outer_acc_rate']*100:.1f}% essN/s={r_u['ess_N_per_s']:.3f} sjdN/s={r_u['sjd_N_per_s']:.3f} dN/s={r_u['dN_moves_per_s']:.3f}")
            prop_u.close()
            _free_gpu_cache()

    # Release FastMaceD3Target before loading SerialAseMaceD3Target so 12 GB VRAM stays clean
    del target
    _free_gpu_cache()

    # 4. Serial ASE Direct GCMC baseline (MLIP-MC style)
    print("  [Baseline] Running direct_serial_ase (n_boxes=1, r_hs=1.20 A)...")
    serial_target = SerialAseMaceD3Target(
        uc_atoms=fw_warm["uc"], supercell=supercell, model=model, device="cuda",
        dispersion=dispersion, dispersion_xc=dispersion_xc, guest=guest_uff,
    )
    res_ser = run_direct_gcmc(
        uc_atoms=fw_warm["uc"], guest=guest_uff, supercell=supercell,
        T_K=T_K, pressure_bar=P_bar, target_fn=serial_target,
        n_boxes=1, eq_steps=0, prod_steps=220 if is_mg else 200,
        initial_states=init_states[:1], r_hs_A=1.20, max_mol=max_mol,
        seed=20261790, max_wall_s=75.0, log_every=50, fw_mass_gmol=fw_mass_gmol,
    )
    rows.append(_format_row(system_name, "direct_serial_ase", "direct_serial", res_ser, 1, 1, T_K, P_bar, res_ser["mlip_evals"], model, dispersion_xc, res_ser["hs_rejected"]))
    print(f"    -> elapsed={res_ser['elapsed_s']:.1f}s steps={res_ser['prod_steps_done']} evals={res_ser['mlip_evals']} HSrej={res_ser['hs_rejected']} essN/s={res_ser['ess_N_per_s']:.3f} sjdN/s={res_ser['sjd_N_per_s']:.3f} dN/s={res_ser['dN_moves_per_s']:.3f}")
    del serial_target
    _free_gpu_cache()

    return rows


def plot_benchmark(rows: list, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.2,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.0,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.1), dpi=300)
    fig.subplots_adjust(left=0.07, right=0.95, top=0.83, bottom=0.16, wspace=0.30)

    mg_rows = {r["method"]: r for r in rows if r["system"] == "mgmof74_co2"}
    ir_rows = {r["method"]: r for r in rows if r["system"] == "irmof1_ch4"}

    groups = []
    if mg_rows:
        n_key = "nested_mercado_k50" if "nested_mercado_k50" in mg_rows else "nested_mercado_k25"
        groups.append(("Mg-MOF-74 + CO2\n(0.05 bar, r2SCAN+D3)", mg_rows, n_key))
    if ir_rows:
        n_key = "nested_uff_qeq_k50" if ("nested_uff_qeq_k50" in ir_rows and ir_rows["nested_uff_qeq_k50"]["sjd_N_per_s"] > ir_rows.get("nested_uff_qeq_k25", {}).get("sjd_N_per_s", 0.0)) else "nested_uff_qeq_k25"
        groups.append(("IRMOF-1 + CH4\n(5.0 bar, MACE-MP0+D3)", ir_rows, n_key))

    x = np.arange(len(groups), dtype=np.float64)
    w = 0.24
    s_vals = [g[1]["direct_serial_ase"]["sjd_N_per_s"] for g in groups]
    b_vals = [g[1]["direct_batched_gpu"]["sjd_N_per_s"] for g in groups]
    n_vals = [g[1][g[2]]["sjd_N_per_s"] for g in groups]

    ax1.bar(x - w, s_vals, width=w, color="#94A3B8", edgecolor="#475569", label="Direct Serial ASE (HS-screened)")
    ax1.bar(x,     b_vals, width=w, color="#3B82F6", edgecolor="#1D4ED8", label="Direct Batched GPU (HS-screened)")
    ax1.bar(x + w, n_vals, width=w, color="#059669", edgecolor="#065F46", label="GPU Nested MC (Best K)")

    ax1.set_yscale("log")
    ymax = max(max(n_vals), max(b_vals), max(s_vals)) * 5.0
    ymin = max(1e-3, min(min(s_vals), min(b_vals), min(n_vals)) * 0.45)
    ax1.set_ylim(ymin, ymax)
    ax1.set_xticks(x)
    ax1.set_xticklabels([g[0] for g in groups])
    ax1.set_ylabel(r"Particle-Number Jump Rate $\mathrm{SJD}_N / \mathrm{GPU\text{-}s}$ ($\mathrm{s}^{-1}$)")
    ax1.set_title("(a) Sampling Speedup on Same RTX 5070 (Log Scale)")
    ax1.grid(axis="y", which="both", linestyle=":", alpha=0.45)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1")

    for i, (_lbl, mp, n_key) in enumerate(groups):
        sp_ser = mp[n_key]["speedup_sjdN_vs_serial"]
        sp_bat = mp[n_key]["speedup_sjdN_vs_batched"]
        k_tag = f"K={mp[n_key]['k_steps']}"
        ax1.text(
            x[i] + w, n_vals[i] * 1.18,
            f"{sp_ser:.1f}x vs Serial\n{sp_bat:.1f}x vs Batched\n({k_tag})",
            ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#065F46",
        )
        for bx, val in ((x[i] - w, s_vals[i]), (x[i], b_vals[i])):
            ax1.text(bx, val * 1.12, f"{val:.3f}", ha="center", va="bottom", fontsize=7.4, color="#334155")
        ax1.text(x[i] + w, n_vals[i] * 0.58, f"{n_vals[i]:.2f}", ha="center", va="center", fontsize=7.6, color="white", fontweight="bold")

    # Panel (b): K-step sweep and equilibrium uptake parity on Mg-MOF-74 + CO2
    if mg_rows:
        order = ["direct_batched_gpu", "nested_mercado_k10", "nested_mercado_k25", "nested_mercado_k50"]
        labels = ["Direct\nBatched", "Nested\nK=10", "Nested\nK=25", "Nested\nK=50"]
        sel = [mg_rows[k] for k in order if k in mg_rows]
        x2 = np.arange(len(sel), dtype=np.float64)
        sjd_list = [r["sjd_N_per_s"] for r in sel]
        ess_list = [r["ess_N_per_s"] for r in sel]
        ax2.plot(x2, sjd_list, "o-", color="#059669", linewidth=2.0, markersize=6, label=r"$\mathrm{SJD}_N / \mathrm{GPU\text{-}s}$")
        ax2.plot(x2, ess_list, "s--", color="#2563EB", linewidth=1.8, markersize=5.5, label=r"$\mathrm{ESS}_N / \mathrm{GPU\text{-}s}$")
        ax2.set_xticks(x2)
        ax2.set_xticklabels(labels[:len(sel)], fontsize=8.4)
        ax2.set_ylabel("Throughput per GPU-second (1/s)")
        ax2.set_title("(b) Mg-MOF-74 + CO2: K-Scaling & Unbiased Uptake")
        ax2.grid(True, linestyle=":", alpha=0.45)
        ax2.spines["top"].set_visible(False)
        ax2.set_ylim(0, max(sjd_list) * 1.28)
        ax2_r = ax2.twinx()
        ups = [r["uptake_mmol_g"] for r in sel]
        errs = [r["uptake_sem_mmol_g"] for r in sel]
        ax2_r.errorbar(x2 + 0.06, ups, yerr=errs, fmt="^:", color="#DC2626", capsize=3, markersize=5.5, label="Uptake (mmol/g)")
        ax2_r.set_ylabel("Equilibrium Uptake (mmol/g)", color="#DC2626")
        ax2_r.tick_params(axis="y", colors="#DC2626")
        ax2_r.set_ylim(0, max(ups) * 1.45)
        ax2_r.spines["top"].set_visible(False)
        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = ax2_r.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.6, frameon=True, facecolor="white", edgecolor="#CBD5E1")

    fig.suptitle("Head-to-Head GPU Sampling Speedup: CUDA Nested MC vs Direct MLIP-GCMC", fontsize=11.8, fontweight="bold", y=0.95)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Benchmark ESS / GPU-s and SJD_N / GPU-s speedup vs Direct MLIP-GCMC")
    p.add_argument("--systems", nargs="+", default=["mgmof74_co2", "irmof1_ch4"], choices=["mgmof74_co2", "irmof1_ch4"])
    p.add_argument("--model", default="", help="Optional override model for all selected systems")
    p.add_argument("--xc", default="", help="Optional override D3 xc for all selected systems")
    p.add_argument("--dispersion", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--reuse-existing", action="store_true", help="Keep rows for unselected systems from existing --out-json")
    p.add_argument("--out-json", default=str(ROOT / "build" / "bench_speedup_ess.json"))
    p.add_argument("--out-csv", default=str(ROOT / "build" / "bench_speedup_ess.csv"))
    p.add_argument("--out-png", default=str(ROOT / "assets" / "benchmark_ess_speedup.png"))
    args = p.parse_args()

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)

    all_rows = []
    if args.reuse_existing and out_json.exists():
        prev = json.loads(out_json.read_text(encoding="utf-8"))
        prev_model = prev.get("model", "mace-matpes-r2scan-0")
        prev_xc = prev.get("xc", "r2scan")
        for r in prev.get("rows", []):
            if r["system"] not in args.systems:
                row = dict(r)
                row.setdefault("model", prev_model)
                row.setdefault("xc", prev_xc)
                all_rows.append(row)

    for sys_name in args.systems:
        cfg = SYSTEM_DEFAULTS[sys_name]
        model = args.model if args.model else cfg["model"]
        xc = args.xc if args.xc else cfg["xc"]
        disp = cfg["dispersion"] if args.dispersion == "auto" else (args.dispersion == "on")
        all_rows.extend(
            benchmark_system(
                system_name=sys_name,
                cif_path=str(cfg["cif"]),
                supercell=cfg["supercell"],
                guest_name=cfg["guest"],
                T_K=cfg["T_K"],
                P_bar=cfg["P_bar"],
                model=model,
                dispersion=disp,
                dispersion_xc=xc,
                mace_chunk=cfg["mace_chunk"],
                warm_proposer=cfg["warm_proposer"],
                warm_charge=cfg["warm_charge"],
                warm_pilot=cfg["warm_pilot"],
                warm_eq=cfg["warm_eq"],
                max_mol=cfg["max_mol"],
            )
        )

    all_rows = _compute_speedups(all_rows)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "systems": list({r["system"] for r in all_rows}),
        "rows": all_rows,
    }, indent=2), encoding="utf-8")

    cols = list(all_rows[0].keys()) if all_rows else []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)

    plot_benchmark(all_rows, out_png)

    print("\n================ SPEEDUP SUMMARY ================")
    for r in all_rows:
        sp_s = f"{r['speedup_sjdN_vs_serial']:.1f}x" if r["speedup_sjdN_vs_serial"] is not None else "-"
        sp_b = f"{r['speedup_sjdN_vs_batched']:.1f}x" if r["speedup_sjdN_vs_batched"] is not None else "-"
        es_s = f"{r['speedup_essN_vs_serial']:.1f}x" if r["speedup_essN_vs_serial"] is not None else "-"
        es_b = f"{r['speedup_essN_vs_batched']:.1f}x" if r["speedup_essN_vs_batched"] is not None else "-"
        print(
            f"{r['system']:12s} | {r['method']:20s} | {r['elapsed_s']:5.1f}s | "
            f"Uptake={r['uptake_mmol_g']:.3f}+-{r['uptake_sem_mmol_g']:.3f} | "
            f"ESS_N/s={r['ess_N_per_s']:6.3f} ({es_s} ser, {es_b} bat) | "
            f"SJD_N/s={r['sjd_N_per_s']:6.3f} ({sp_s} ser, {sp_b} bat) | dN/s={r['dN_moves_per_s']:6.3f}"
        )
    print(f"Saved: {out_json}, {out_csv}, {out_png}")


if __name__ == "__main__":
    main()
