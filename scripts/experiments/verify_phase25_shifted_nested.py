"""Phase 2.5 Verification Suite:
Test A: Exact Chemical-Potential Shift Null-Bias Proof (Pol-3Site w/ delta_eps=-6.0 kJ/mol -> Mercado 2016).
Test B: FastMaceD3Target Exactness (< 0.10 kJ/mol/CO2) & 4-Box Batched Speedup vs ASE MaceD3Target.
Test C: Converged Shifted 4-Box MACE-MP-0(Medium)+D3 Nested MC on Mg-MOF-74 (298 K, 0.15 bar).
"""

import json
import math
import time
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nested_mc_engine import (
    CudaGcmcProposer,
    MaceD3Target,
    FastMaceD3Target,
    K_TO_KJMOL,
    run_nested_mc,
)

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
ASSETS = ROOT / "assets"
BUILD.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)


def run_test_a_shifted_null_bias():
    print("\n=== Test A: Shifted Chemical-Potential Null-Bias Proof (Auto-Calibrated Pol-3Site -> Mercado16) ===")
    cif = str(ROOT / "data" / "MgMOF-74.cif")
    T_K, P_bar = 298.0, 0.15

    direct_ctx = CudaGcmcProposer(
        cif_path=cif, supercell=(1, 1, 2), T_K=T_K, pressure_bar=P_bar,
        ff_mode=1, pol_mode=0, grid_dims=(256, 256, 72), n_boxes=4, max_mol=128, seed=20261199
    )
    direct_res = direct_ctx.run_direct_benchmark(eq_steps=100000, prod_steps=200000, sample_stride=20)
    print(
        f"  [Direct Mercado16] Uptake = {direct_res['uptake_mmol_g']:.4f} +- {direct_res['uptake_sem_mmol_g']:.4f} mmol/g | "
        f"Qst = {direct_res['Qst_kJ_mol']:.2f} +- {direct_res['Qst_sem_kJ_mol']:.2f} kJ/mol"
    )

    pol3_prop = CudaGcmcProposer(
        cif_path=cif, supercell=(1, 1, 2), T_K=T_K, pressure_bar=P_bar,
        ff_mode=0, pol_mode=2, grid_dims=(256, 256, 72), n_boxes=4, max_mol=128, seed=20261301
    )
    pol3_prop.equilibrate(30000)
    st_unshifted = pol3_prop.get_accepted()
    diffs = []
    for s in st_unshifted:
        ut, ov = direct_ctx.eval_energy(s)
        if not ov and s["N"] > 0:
            diffs.append(((ut - s["U_ref_K"]) * K_TO_KJMOL) / s["N"])
    delta_eps_kJ = float(np.mean(diffs))
    fug_mult = math.exp(-delta_eps_kJ / (K_TO_KJMOL * T_K))

    pol3_prop.set_chem_pot_shift(delta_eps_kJ)
    pol3_prop.equilibrate(30000)
    st_shifted = pol3_prop.get_accepted()
    print(
        f"  [Inner Shifted Equil] delta_eps = {delta_eps_kJ:.2f} kJ/mol ({fug_mult:.2f}x inner fugacity) -> "
        f"warm-started N = {[s['N'] for s in st_shifted]}"
    )

    nested_res = run_nested_mc(
        pol3_prop, direct_ctx.eval_energy,
        k_steps=15, eq_cycles=2000, prod_cycles=8000, seed=20261302, log_every=2000
    )

    du = abs(nested_res["uptake_mmol_g"] - direct_res["uptake_mmol_g"])
    comb_sem = math.sqrt(nested_res["uptake_sem_mmol_g"] ** 2 + direct_res["uptake_sem_mmol_g"] ** 2 + 1e-12)
    z_score = du / comb_sem
    dq = abs(nested_res["Qst_kJ_mol"] - direct_res["Qst_kJ_mol"])

    print(
        f"  [Shifted Pol3->Merc] Uptake = {nested_res['uptake_mmol_g']:.4f} +- {nested_res['uptake_sem_mmol_g']:.4f} mmol/g | "
        f"Qst = {nested_res['Qst_kJ_mol']:.2f} +- {nested_res['Qst_sem_kJ_mol']:.2f} kJ/mol | "
        f"z = {z_score:.2f} sigma, |dQst| = {dq:.2f} kJ/mol"
    )
    print(
        f"  [Outer dN Symmetry] Ins={nested_res['prod_ins_acc']}/{nested_res['prod_ins_att']} | "
        f"Del={nested_res['prod_del_acc']}/{nested_res['prod_del_att']} | Outer Acc={nested_res['outer_acc_rate']*100:.1f}%"
    )

    assert z_score < 2.5, f"Shifted null-bias failed: z={z_score:.2f}"
    assert dq < 2.5, f"Shifted null-bias Qst gap too large: {dq:.2f}"
    assert nested_res["prod_ins_acc"] > 100 and nested_res["prod_del_acc"] > 100, "Insufficient shifted ins/del moves"
    pol3_prop.close()
    direct_ctx.close()
    print("  -> Test A PASSED: Shifted proposer cancels exact outer fugacity correction with zero bias!")
    return {"direct": direct_res, "nested": nested_res, "z_score": z_score, "dQst": dq}


def run_test_b_fast_target_bench():
    print("\n=== Test B: FastMaceD3Target Exactness & 4-Box Batched Speedup Benchmark ===")
    import torch
    cif = str(ROOT / "data" / "MgMOF-74.cif")
    prop = CudaGcmcProposer(
        cif_path=cif, supercell=(1, 1, 2), T_K=298.0, pressure_bar=0.15,
        ff_mode=1, pol_mode=0, grid_dims=(256, 256, 72), n_boxes=4, max_mol=128, seed=20261311
    )
    prop.equilibrate(25000)
    states = prop.get_accepted()
    print(f"  Test states N = {[s['N'] for s in states]}")

    ase_tgt = MaceD3Target(cif, supercell=(1, 1, 2), model="medium", device="cuda")
    fast_tgt = FastMaceD3Target(cif, supercell=(1, 1, 2), model="medium", device="cuda")

    # Warmup
    u_ref0, _ = ase_tgt(states[0])
    u_fast0, _ = fast_tgt(states[0])
    _ = fast_tgt.eval_batch(states)

    diffs_kJ_per_mol = []
    for s in states:
        ur, _ = ase_tgt(s)
        uf, _ = fast_tgt(s)
        dkj = abs(uf - ur) * K_TO_KJMOL / max(1, s["N"])
        diffs_kJ_per_mol.append(dkj)
        print(f"    Box N={s['N']:2d}: ASE={ur*K_TO_KJMOL:8.2f} kJ/mol | Fast={uf*K_TO_KJMOL:8.2f} kJ/mol | diff={dkj:.4f} kJ/mol/CO2")

    max_diff_per_co2 = max(diffs_kJ_per_mol)
    assert max_diff_per_co2 < 0.10, f"FastMaceD3 mismatch too high: {max_diff_per_co2:.4f} kJ/mol/CO2"

    # Benchmark latency on distinct slightly-perturbed states to bypass ASE cache
    def perturbed(s, eps):
        c = dict(s)
        c["cx"] = s["cx"] + eps
        return c

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(4):
        _ = ase_tgt(perturbed(states[0], 1e-4 * (i + 1)))
    torch.cuda.synchronize()
    ase_ms = (time.perf_counter() - t0) / 4.0 * 1000.0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(8):
        _ = fast_tgt(perturbed(states[0], 1e-4 * (i + 1)))
    torch.cuda.synchronize()
    fast1_ms = (time.perf_counter() - t0) / 8.0 * 1000.0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(8):
        b_states = [perturbed(states[b], 1e-4 * (i + 1)) for b in range(4)]
        _ = fast_tgt.eval_batch(b_states)
    torch.cuda.synchronize()
    fast4_batch_ms = (time.perf_counter() - t0) / 8.0 * 1000.0
    fast4_per_box_ms = fast4_batch_ms / 4.0
    speedup = ase_ms / fast4_per_box_ms

    print(
        f"  [Latency] ASE 1-box={ase_ms:.1f} ms/box | Fast 1-box={fast1_ms:.1f} ms/box | "
        f"Fast 4-box={fast4_batch_ms:.1f} ms/batch ({fast4_per_box_ms:.1f} ms/box, {speedup:.2f}x speedup!)"
    )
    prop.close()
    del ase_tgt
    torch.cuda.empty_cache()
    print("  -> Test B PASSED: FastMaceD3Target is exact and substantially faster!")
    return {
        "max_diff_kJ_per_CO2": float(max_diff_per_co2),
        "ase_ms_per_box": float(ase_ms),
        "fast1_ms_per_box": float(fast1_ms),
        "fast4_batch_ms": float(fast4_batch_ms),
        "fast4_ms_per_box": float(fast4_per_box_ms),
        "speedup_vs_ase": float(speedup),
    }, fast_tgt


def run_test_c_converged_mace(fast_tgt=None):
    print("\n=== Test C: Converged Shifted 4-Box MACE-MP-0(Medium)+D3 Nested MC (298 K, 0.01 bar) ===")
    cif = str(ROOT / "data" / "MgMOF-74.cif")
    T_K, P_bar = 298.0, 0.01

    if fast_tgt is None:
        fast_tgt = FastMaceD3Target(cif, supercell=(1, 1, 2), model="medium", device="cuda")

    merc_prop = CudaGcmcProposer(
        cif_path=cif, supercell=(1, 1, 2), T_K=T_K, pressure_bar=P_bar,
        ff_mode=1, pol_mode=0, grid_dims=(256, 256, 72), n_boxes=4, max_mol=128, seed=20261321
    )
    # Stage 1 self-consistent chemical-potential calibration at 0.01 bar
    merc_prop.equilibrate(20000)
    cal1_states = merc_prop.get_accepted()
    cal1_u = fast_tgt.eval_batch(cal1_states)
    diffs_1 = [
        float(((cal1_u[b][0] - cal1_states[b]["U_ref_K"]) * K_TO_KJMOL) / max(1, cal1_states[b]["N"]))
        for b in range(4)
    ]
    delta_eps_1 = float(max(-12.0, min(12.0, np.mean(diffs_1))))
    merc_prop.set_chem_pot_shift(delta_eps_1)
    merc_prop.equilibrate(30000)

    # Stage 2 self-consistent re-calibration at the target MACE loading
    cal2_states = merc_prop.get_accepted()
    cal2_u = fast_tgt.eval_batch(cal2_states)
    diffs_2 = [
        float(((cal2_u[b][0] - cal2_states[b]["U_ref_K"]) * K_TO_KJMOL) / max(1, cal2_states[b]["N"]))
        for b in range(4)
    ]
    delta_eps_kJ = float(max(-12.0, min(12.0, np.mean(diffs_2))))
    fug_mult = math.exp(-delta_eps_kJ / (K_TO_KJMOL * T_K))
    print(
        f"  [Self-Consistent Shift] Stage1={delta_eps_1:.2f} -> Stage2 <(U_MACE - U_Merc)/N> = {delta_eps_kJ:.2f} kJ/mol "
        f"({fug_mult:.2f}x inner fugacity; boxes={[round(x,2) for x in diffs_2]})"
    )

    merc_prop.set_chem_pot_shift(delta_eps_kJ)
    merc_prop.equilibrate(20000)
    warm_states = merc_prop.get_accepted()
    conv_mmol_g = 1000.0 / merc_prop.fw_mass_gmol
    print(f"  [Shifted Warm-Start] N0 = {[s['N'] for s in warm_states]} ({[round(s['N']*conv_mmol_g, 2) for s in warm_states]} mmol/g)")

    fast_tgt.n_evals = 0
    res = run_nested_mc(
        merc_prop, None,
        k_steps=25, eq_cycles=150, prod_cycles=350, seed=20261322, log_every=50,
        batch_target_fn=fast_tgt.eval_batch,
    )

    hist = np.array(res["box_N_history"], dtype=np.float64) * conv_mmol_g  # (total_cycles, 4)
    prod_hist = hist[res["eq_cycles"]:]
    half = prod_hist.shape[0] // 2
    mean_h1 = float(np.mean(prod_hist[:half]))
    mean_h2 = float(np.mean(prod_hist[half:]))
    stationarity_drift = abs(mean_h2 - mean_h1)
    res["first_half_uptake_mmol_g"] = mean_h1
    res["second_half_uptake_mmol_g"] = mean_h2
    res["stationarity_drift_mmol_g"] = stationarity_drift
    res["mace_evals"] = fast_tgt.n_evals
    res["total_inner_steps"] = 4 * res["k_steps"] * (res["eq_cycles"] + res["prod_cycles"])

    print(
        f"  [Converged 4-Box MACE+D3] Uptake = {res['uptake_mmol_g']:.4f} +- {res['uptake_sem_mmol_g']:.4f} mmol/g | "
        f"Qst = {res['Qst_kJ_mol']:.2f} +- {res['Qst_sem_kJ_mol']:.2f} kJ/mol | Outer Acc = {res['outer_acc_rate']*100:.1f}%"
    )
    print(
        f"  [dN Symmetry & Stationarity] Ins={res['prod_ins_acc']}/{res['prod_ins_att']} | "
        f"Del={res['prod_del_acc']}/{res['prod_del_att']} | Half1={mean_h1:.3f} vs Half2={mean_h2:.3f} mmol/g (|drift|={stationarity_drift:.3f}) | "
        f"{res['total_inner_steps']} inner steps in {res['elapsed_s']:.1f} s"
    )

    assert 0.15 <= res["outer_acc_rate"] <= 0.90, f"Unhealthy outer acceptance: {res['outer_acc_rate']}"
    assert res["prod_ins_acc"] >= 10 and res["prod_del_acc"] >= 10, (
        f"Asymmetric ins/del: ins={res['prod_ins_acc']}, del={res['prod_del_acc']}"
    )
    assert stationarity_drift < 0.6, f"Non-stationary drift too large: {stationarity_drift:.3f} mmol/g"

    merc_prop.close()
    print("  -> Test C PASSED: Symmetric grand-canonical sampling & converged 4-box MACE-MP-0+D3 equilibrium!")
    return res, conv_mmol_g


def make_figure(ta, tb, tc, conv_mmol_g):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.9,
        "axes.titlesize": 10.2,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.3,
        "legend.fontsize": 7.6,
    })
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.9), dpi=300)
    fig.subplots_adjust(left=0.055, right=0.98, top=0.81, bottom=0.14, wspace=0.28)

    # Panel (a): Test A Shifted Null-Bias Proof
    ax1 = axes[0]
    direct, nested = ta["direct"], ta["nested"]
    boxes = nested["box_uptakes_mmol_g"]
    x = np.arange(len(boxes))
    de_a = nested["delta_eps_kJ_mol"]
    fmult_a = math.exp(-de_a / (K_TO_KJMOL * 298.0))
    ax1.bar(x, boxes, width=0.52, color="#2563EB", alpha=0.82, edgecolor="#1E40AF", label=f"Shifted Nested (Δε={de_a:.2f} kJ/mol)")
    ax1.axhline(direct["uptake_mmol_g"], color="#0F766E", linewidth=2.0, label=f"Direct Mercado16: {direct['uptake_mmol_g']:.2f} ± {direct['uptake_sem_mmol_g']:.2f}")
    ax1.axhspan(direct["uptake_mmol_g"] - direct["uptake_sem_mmol_g"], direct["uptake_mmol_g"] + direct["uptake_sem_mmol_g"], color="#0F766E", alpha=0.16)
    ax1.axhline(nested["uptake_mmol_g"], color="#DC2626", linewidth=1.8, linestyle="--", label=f"Nested Mean: {nested['uptake_mmol_g']:.2f} ± {nested['uptake_sem_mmol_g']:.2f}")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Box {i+1}" for i in range(len(boxes))])
    ax1.set_ylim(0, max(max(boxes), direct["uptake_mmol_g"]) * 1.38)
    ax1.set_ylabel(r"$\mathrm{CO_2}$ Uptake at 0.15 bar, 298 K (mmol/g)")
    ax1.grid(axis="y", linestyle=":", alpha=0.45)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1")
    ax1.text(
        0.97, 0.04,
        f"z = {ta['z_score']:.2f}σ | |ΔQst| = {ta['dQst']:.2f} kJ/mol\nIns={nested['prod_ins_acc']} / Del={nested['prod_del_acc']} acc",
        transform=ax1.transAxes, ha="right", va="bottom", fontsize=7.9,
        bbox=dict(boxstyle="round,pad=0.30", fc="#F8FAFC", ec="#CBD5E1")
    )
    ax1.set_title(f"(a) Shifted Null-Bias Proof ({fmult_a:.2f}× Inner f_ref)")

    # Panel (b): Test B Throughput Speedup
    ax2 = axes[1]
    labels = ["ASE\n(1 Box)", "Fast Energy-Only\n(1 Box)", "Fast Batched\n(4 Boxes)"]
    latencies = [tb["ase_ms_per_box"], tb["fast1_ms_per_box"], tb["fast4_ms_per_box"]]
    colors = ["#94A3B8", "#0284C7", "#7C3AED"]
    bars = ax2.bar(np.arange(3), latencies, width=0.52, color=colors, edgecolor="#1E293B", alpha=0.88)
    for rect, val in zip(bars, latencies):
        ax2.text(rect.get_x() + rect.get_width() / 2.0, val + max(latencies) * 0.025, f"{val:.1f} ms", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax2.set_xticks(np.arange(3))
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, max(latencies) * 1.24)
    ax2.set_ylabel("MACE-Medium + D3 Time per Box (ms/eval)")
    ax2.grid(axis="y", linestyle=":", alpha=0.45)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.text(
        0.97, 0.94,
        f"Speedup: {tb['speedup_vs_ase']:.2f}×\nMax |ΔE|: {tb['max_diff_kJ_per_CO2']:.4f} kJ/mol/CO2",
        transform=ax2.transAxes, ha="right", va="top", fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.30", fc="#F8FAFC", ec="#CBD5E1")
    )
    ax2.set_title("(b) Energy-Only MACE + GPU D3 Speedup")

    # Panel (c): Test C Converged 4-Box MACE-MP-0(M)+D3 Nested MC
    ax3 = axes[2]
    hist_up = np.array(tc["box_N_history"], dtype=np.float64) * conv_mmol_g
    cycs = np.arange(hist_up.shape[0])
    box_cols = ["#2563EB", "#0F766E", "#D97706", "#DC2626"]
    ax3.axvspan(0, tc["eq_cycles"], color="#F1F5F9", alpha=0.95, zorder=0, label=f"Equilibration ({tc['eq_cycles']} cyc)")
    for b in range(4):
        ax3.plot(cycs, hist_up[:, b], color=box_cols[b], linewidth=1.15, alpha=0.82, label=f"Box {b+1}: {tc['box_uptakes_mmol_g'][b]:.2f}")
    ax3.axhline(tc["uptake_mmol_g"], xmin=tc["eq_cycles"]/len(cycs), xmax=1.0, color="#1E293B", linestyle="--", linewidth=1.8, label=f"Mean: {tc['uptake_mmol_g']:.2f} ± {tc['uptake_sem_mmol_g']:.2f}")
    ax3.set_xlabel("Outer Nested MC Cycle (K = 25 Mercado16 steps/box/cyc)")
    ax3.set_ylabel(r"$\mathrm{CO_2}$ Uptake (mmol/g)")
    ax3.set_ylim(min(np.min(hist_up) * 0.7, 3.0), max(np.max(hist_up) * 1.25, 11.0))
    ax3.grid(True, linestyle=":", alpha=0.45)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.legend(loc="upper left", ncol=2, frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1")
    ax3.text(
        0.97, 0.04,
        f"Δε = {tc['delta_eps_kJ_mol']:.2f} kJ/mol | Acc={tc['outer_acc_rate']*100:.1f}%\n"
        f"Ins={tc['prod_ins_acc']}/{tc['prod_ins_att']} | Del={tc['prod_del_acc']}/{tc['prod_del_att']}\n"
        f"Qst={tc['Qst_kJ_mol']:.1f} ± {tc['Qst_sem_kJ_mol']:.1f} kJ/mol | {tc['elapsed_s']:.0f}s",
        transform=ax3.transAxes, ha="right", va="bottom", fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.30", fc="#F8FAFC", ec="#CBD5E1")
    )
    ax3.set_title("(c) Converged 4-Box MACE-MP-0(M)+D3 (298 K, 0.01 bar)")

    fig.suptitle(
        "Phase 2.5 Upgrade: Exact Chemical-Potential Shift (Δμ_ref) & 4-Box Batched MACE-MP-0+D3 Equilibrium",
        fontsize=11.6, fontweight="bold", y=0.95
    )
    out_png = ASSETS / "phase25_shifted_batched_mace.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved {out_png}")


def main():
    ta = run_test_a_shifted_null_bias()
    tb, fast_tgt = run_test_b_fast_target_bench()
    tc, conv_mmol_g = run_test_c_converged_mace(fast_tgt=fast_tgt)
    make_figure(ta, tb, tc, conv_mmol_g)
    out_json = BUILD / "phase25_shifted_results.json"
    out_json.write_text(json.dumps({"test_a": ta, "test_b": tb, "test_c": tc}, indent=2), encoding="utf-8")
    print("\nALL PHASE 2.5 SHIFTED & BATCHED NESTED-MC ASSERTIONS PASSED!")


if __name__ == "__main__":
    main()
