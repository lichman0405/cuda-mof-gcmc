"""Step 3.2 Multi-Foundation-Model Probe on Mg-MOF-74 (1x1x2):
Compares MACE-MP-0(medium)+D3, MACE-MPA-0+D3, MACE-MP-0b3+D3, and MACE-MatPES-r2SCAN-0 (no D3)
on 1-CO2 OMS radial/pore binding, loaded-state energy vs Mercado 2016, and 4-box batched latency.
"""

import json
import math
import time
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nested_mc_engine import CudaGcmcProposer, FastMaceD3Target, K_TO_KJMOL

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
BUILD.mkdir(parents=True, exist_ok=True)


def build_diverse_1mol_states(prop: CudaGcmcProposer, n_poses: int = 30):
    """Harvest diverse valid 1-CO2 microstates inside pores using classical GCMC."""
    rng = np.random.default_rng(20261501)
    # Use a 1-box proposer or sample states from low pressure Monte Carlo
    H = prop.H_cols
    states = []
    # Also run a low-pressure equilibrator or generate non-overlapping 1-CO2 poses evaluated via eval_energy
    while len(states) < n_poses:
        f = rng.random(3)
        c = H @ f
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        st = {
            "N": 1,
            "cx": np.array([c[0]], dtype=np.float64),
            "cy": np.array([c[1]], dtype=np.float64),
            "cz": np.array([c[2]], dtype=np.float64),
            "ux": np.array([v[0]], dtype=np.float64),
            "uy": np.array([v[1]], dtype=np.float64),
            "uz": np.array([v[2]], dtype=np.float64),
        }
        u_ref, ov = prop.eval_energy(st)
        if not ov and u_ref * K_TO_KJMOL < -5.0:
            st["U_ref_K"] = u_ref
            states.append(st)
    # Sort from strongest to weakest in Mercado 2016 to guarantee deep OMS poses too
    states.sort(key=lambda s: s["U_ref_K"])
    return states


def main():
    cif = str(ROOT / "data" / "MgMOF-74.cif")
    # 1. Harvest loaded N~27 states + 30 diverse 1-CO2 poses
    merc_prop = CudaGcmcProposer(
        cif_path=cif, supercell=(1, 1, 2), T_K=298.0, pressure_bar=0.15,
        ff_mode=1, pol_mode=0, grid_dims=(256, 256, 72), n_boxes=4, max_mol=128, seed=20261511
    )
    merc_prop.equilibrate(30000)
    loaded_states = merc_prop.get_accepted()
    print(f"Loaded Mercado16 test states N = {[s['N'] for s in loaded_states]}")

    # Generate 24 random low-energy 1-CO2 poses + run short low-P MC for 6 deep OMS poses
    low_p = CudaGcmcProposer(
        cif_path=cif, supercell=(1, 1, 2), T_K=298.0, pressure_bar=0.002,
        ff_mode=1, pol_mode=0, grid_dims=(256, 256, 72), n_boxes=4, max_mol=128, seed=20261512
    )
    oms_poses = []
    for step_idx in range(12):
        low_p.equilibrate(500)
        for s in low_p.get_accepted():
            if s["N"] >= 1 and len(oms_poses) < 15:
                # extract the strongest-bound single molecule? extract molecule 0 for clean 1-CO2 probe
                one = {
                    "N": 1,
                    "cx": s["cx"][:1].copy(), "cy": s["cy"][:1].copy(), "cz": s["cz"][:1].copy(),
                    "ux": s["ux"][:1].copy(), "uy": s["uy"][:1].copy(), "uz": s["uz"][:1].copy(),
                }
                u_ref, ov = merc_prop.eval_energy(one)
                if not ov:
                    one["U_ref_K"] = u_ref
                    oms_poses.append(one)
    rand_poses = build_diverse_1mol_states(merc_prop, n_poses=15)
    poses_1mol = oms_poses[:15] + rand_poses[:15]
    u_merc_1mol_kJ = np.array([p["U_ref_K"] * K_TO_KJMOL for p in poses_1mol])
    print(f"30 1-CO2 poses Mercado range: [{u_merc_1mol_kJ.min():.2f}, {u_merc_1mol_kJ.max():.2f}] kJ/mol")

    candidates = [
        ("medium", True, "MACE-MP-0 (Medium) + D3"),
        ("medium-mpa-0", True, "MACE-MPA-0 (Medium) + D3"),
        ("medium-0b3", True, "MACE-MP-0b3 (Medium) + D3"),
        ("mace-matpes-r2scan-0", False, "MACE-MatPES-r2SCAN-0 (No D3)"),
    ]

    results = []
    for model_name, disp, label in candidates:
        print(f"\n--- Probing {label} (model={model_name}, disp={disp}) ---")
        try:
            tgt = FastMaceD3Target(cif, supercell=(1, 1, 2), model=model_name, device="cuda", dispersion=disp)
        except Exception as exc:
            print(f"  SKIP {label}: failed to load ({exc})")
            continue

        # 30 single-molecule poses in chunks of 4
        u_mlip_1mol_kJ = []
        for i0 in range(0, len(poses_1mol), 4):
            res_b = tgt.eval_batch(poses_1mol[i0:i0 + 4])
            u_mlip_1mol_kJ.extend([r[0] * K_TO_KJMOL for r in res_b])
        u_mlip_1mol_kJ = np.array(u_mlip_1mol_kJ)

        r_corr = float(np.corrcoef(u_merc_1mol_kJ, u_mlip_1mol_kJ)[0, 1])
        r2 = r_corr * r_corr
        dd_1mol = u_mlip_1mol_kJ - u_merc_1mol_kJ
        sigma_dd = float(np.std(dd_1mol, ddof=1))

        # 4 loaded states + latency benchmark
        _ = tgt.eval_batch(loaded_states)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            loaded_evals = tgt.eval_batch(loaded_states)
        torch.cuda.synchronize()
        batch_ms = (time.perf_counter() - t0) / 5.0 * 1000.0
        ms_per_box = batch_ms / 4.0

        u_mace_loaded_per_mol = [
            float(loaded_evals[b][0] * K_TO_KJMOL / loaded_states[b]["N"])
            for b in range(4)
        ]
        delta_eps_loaded = [
            float((loaded_evals[b][0] - loaded_states[b]["U_ref_K"]) * K_TO_KJMOL / loaded_states[b]["N"])
            for b in range(4)
        ]

        entry = {
            "model": model_name,
            "dispersion": disp,
            "label": label,
            "min_1mol_kJ_mol": float(u_mlip_1mol_kJ.min()),
            "mean_1mol_kJ_mol": float(u_mlip_1mol_kJ.mean()),
            "r2_vs_mercado_1mol": r2,
            "sigma_ddE_1mol_kJ_mol": sigma_dd,
            "loaded_mean_U_per_N_kJ_mol": float(np.mean(u_mace_loaded_per_mol)),
            "loaded_delta_eps_vs_mercado_kJ_mol": float(np.mean(delta_eps_loaded)),
            "batch4_ms": float(batch_ms),
            "ms_per_box": float(ms_per_box),
        }
        results.append(entry)
        print(
            f"  1-mol min={entry['min_1mol_kJ_mol']:.2f} kJ/mol, R^2={r2:.3f}, sigma(ddE)={sigma_dd:.2f} kJ/mol | "
            f"Loaded <U/N>={entry['loaded_mean_U_per_N_kJ_mol']:.2f} kJ/mol, "
            f"delta_eps={entry['loaded_delta_eps_vs_mercado_kJ_mol']:+.2f} kJ/mol | "
            f"Speed={ms_per_box:.1f} ms/box ({batch_ms:.1f} ms/batch)"
        )
        del tgt
        torch.cuda.empty_cache()

    merc_prop.close()
    low_p.close()

    out_path = BUILD / "phase3_model_probe.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
