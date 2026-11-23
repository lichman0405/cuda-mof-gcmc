"""Paper Experiment 4 (Detail 4): Supercell Finite-Size Check (1x1x2 vs 1x1x4) on Mg-MOF-74 (298 K, 0.01 bar).
Compares Direct Mercado 2016 GPU GCMC and Shifted 4-Box MACE-MP-0(Medium)+D3(PBE) Nested MC.
"""

import json
import math
from pathlib import Path
import numpy as np

from nested_mc_engine import CudaGcmcProposer, FastMaceD3Target, K_TO_KJMOL, run_nested_mc

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
BUILD.mkdir(parents=True, exist_ok=True)


def run_cell_check(sc, eq_cyc=100, prod_cyc=240):
    cif = str(ROOT / "data" / "MgMOF-74.cif")
    T_K, P_bar = 298.0, 0.01
    label = f"{sc[0]}x{sc[1]}x{sc[2]}"

    # Direct Mercado 2016
    direct_ctx = CudaGcmcProposer(
        cif_path=cif, supercell=sc, T_K=T_K, pressure_bar=P_bar,
        ff_mode=1, pol_mode=0, grid_dims=(256, 256, 72), n_boxes=4, max_mol=180, seed=20262400 + sc[2]
    )
    direct_res = direct_ctx.run_direct_benchmark(eq_steps=100000, prod_steps=200000, sample_stride=20)
    print(f"  [{label} Direct Mercado16] uptake={direct_res['uptake_mmol_g']:.4f} +- {direct_res['uptake_sem_mmol_g']:.4f} mmol/g | Qst={direct_res['Qst_kJ_mol']:.2f} kJ/mol | r_cut_gg={direct_ctx.r_cut_gg:.3f} A")

    # MACE-MP-0(Medium)+D3(PBE) Nested MC
    tgt = FastMaceD3Target(cif, supercell=sc, model="medium", device="cuda", dispersion=True, dispersion_xc="pbe")
    direct_ctx.equilibrate(20000)
    st1 = direct_ctx.get_accepted()
    if np.mean([s["N"] for s in st1]) < 4.0:
        direct_ctx.set_chem_pot_shift(-8.0)
        direct_ctx.equilibrate(15000)
        st1 = direct_ctx.get_accepted()
    ev1 = tgt.eval_batch(st1)
    de1 = float(max(-12.0, min(12.0, np.mean([((ev1[b][0] - st1[b]["U_ref_K"]) * K_TO_KJMOL) / max(1, st1[b]["N"]) for b in range(4)]))))
    direct_ctx.set_chem_pot_shift(de1)
    direct_ctx.equilibrate(25000)
    st2 = direct_ctx.get_accepted()
    ev2 = tgt.eval_batch(st2)
    de2 = float(max(-12.0, min(12.0, np.mean([((ev2[b][0] - st2[b]["U_ref_K"]) * K_TO_KJMOL) / max(1, st2[b]["N"]) for b in range(4)]))))
    direct_ctx.set_chem_pot_shift(de2)
    direct_ctx.equilibrate(20000)
    print(f"  [{label} MACE+D3 Shifted] de2={de2:.2f} kJ/mol, warm N={[s['N'] for s in direct_ctx.get_accepted()]}")

    res = run_nested_mc(
        direct_ctx, None, k_steps=25, eq_cycles=eq_cyc, prod_cycles=prod_cyc,
        seed=20262450 + sc[2], log_every=85, batch_target_fn=tgt.eval_batch
    )
    print(
        f"  [{label} MACE-MP-0+D3] uptake={res['uptake_mmol_g']:.4f} +- {res['uptake_sem_mmol_g']:.4f} mmol/g | "
        f"Fluc Qst={res['Qst_kJ_mol']:.2f} | Molar Qst={res['mean_molar_qst_kJ_mol']:.2f} kJ/mol | Acc={res['outer_acc_rate']*100:.1f}%"
    )
    rc_gg = direct_ctx.r_cut_gg
    direct_ctx.close()
    del tgt
    return {
        "supercell": label,
        "r_cut_gg_A": rc_gg,
        "direct_mercado": direct_res,
        "mace_mp0_d3": {k: v for k, v in res.items() if k != "box_N_history" and k != "trajectory"},
    }


def main():
    print("\n=== Experiment 4: Finite-Size Check (1x1x2 vs 1x1x4) on Mg-MOF-74 (298 K, 0.01 bar) ===")
    r12 = run_cell_check((1, 1, 2), eq_cyc=100, prod_cyc=240)
    r14 = run_cell_check((1, 1, 4), eq_cyc=100, prod_cyc=240)

    dz_dir = abs(r12["direct_mercado"]["uptake_mmol_g"] - r14["direct_mercado"]["uptake_mmol_g"]) / math.sqrt(
        r12["direct_mercado"]["uptake_sem_mmol_g"] ** 2 + r14["direct_mercado"]["uptake_sem_mmol_g"] ** 2 + 1e-12
    )
    dz_mlip = abs(r12["mace_mp0_d3"]["uptake_mmol_g"] - r14["mace_mp0_d3"]["uptake_mmol_g"]) / math.sqrt(
        r12["mace_mp0_d3"]["uptake_sem_mmol_g"] ** 2 + r14["mace_mp0_d3"]["uptake_sem_mmol_g"] ** 2 + 1e-12
    )
    print(f"\n[Finite-Size Consistency] Direct Mercado z={dz_dir:.2f} sigma | MACE-MP-0+D3 z={dz_mlip:.2f} sigma")
    assert dz_mlip < 2.5, f"Finite-size discrepancy too large: z={dz_mlip:.2f}"

    out = {"cell_1x1x2": r12, "cell_1x1x4": r14, "z_direct": dz_dir, "z_mlip": dz_mlip}
    out_path = BUILD / "paper_exp4_finite_size.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
