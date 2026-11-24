"""Step 2 (Item 7): Verify FP32 FastMaceD3Target vs FP64 MACE+D3 energy parity.

Generates 15 non-overlapping Mg-MOF-74 (1x1x2) pore snapshots across 3 loadings
(N=1, N=16, N=40 CO2) via CudaGcmcProposer and compares binding energies and
pairwise transition differences between the live FP32 fast engine and full FP64
MACE + TorchDFTD3 calculations.
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
from ase import Atoms
from ase.io import read as ase_read
from mace.calculators import mace_mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nested_mc_engine import CudaGcmcProposer, FastMaceD3Target, K_TO_KJMOL

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
BUILD.mkdir(parents=True, exist_ok=True)
KB_T_KJMOL = 298.0 * K_TO_KJMOL  # ~2.4777 kJ/mol at 298 K
EV_TO_KJMOL = 96.48533212


def generate_snapshots(cif_path: Path, eq_steps_per_snap=1000):
    target_pressures = [(1, 0.002), (16, 0.05), (40, 0.30)]
    snapshots = []
    for n_target, p_bar in target_pressures:
        prop = CudaGcmcProposer(
            cif_path=str(cif_path),
            supercell=(1, 1, 2),
            T_K=298.0,
            pressure_bar=p_bar,
            ff_mode=1,
            pol_mode=0,
            grid_dims=(256, 256, 72),
            n_boxes=4,
            max_mol=128,
            seed=20263200 + n_target,
        )
        prop.equilibrate(25000)
        collected = 0
        tries = 0
        while collected < 5 and tries < 20:
            tries += 1
            prop.equilibrate(eq_steps_per_snap)
            boxes = prop.get_accepted()
            # Sort by closeness to desired loading N
            boxes.sort(key=lambda s: abs(int(s["N"]) - n_target))
            st = boxes[0]
            if int(st["N"]) > 0:
                snapshots.append({
                    "N": int(st["N"]),
                    "cx": st["cx"].copy(),
                    "cy": st["cy"].copy(),
                    "cz": st["cz"].copy(),
                    "ux": st["ux"].copy(),
                    "uy": st["uy"].copy(),
                    "uz": st["uz"].copy(),
                    "target_bin": n_target,
                })
                collected += 1
        prop.close()
    return snapshots


def build_ase_atoms(fw_base, st):
    n = st["N"]
    if n == 0:
        return fw_base.copy()
    c_xyz = np.column_stack([st["cx"][:n], st["cy"][:n], st["cz"][:n]])
    u_xyz = np.column_stack([st["ux"][:n], st["uy"][:n], st["uz"][:n]])
    o1 = c_xyz + 1.16 * u_xyz
    o2 = c_xyz - 1.16 * u_xyz
    pos = np.empty((3 * n, 3), dtype=np.float64)
    pos[0::3] = c_xyz
    pos[1::3] = o1
    pos[2::3] = o2
    guests = Atoms("CO2" * n, positions=pos)
    full = fw_base.copy() + guests
    full.set_cell(fw_base.get_cell())
    full.set_pbc(True)
    return full


def eval_model_pair(model_name: str, xc: str, snapshots, fw_base):
    print(f"\n=== Checking FP32 vs FP64 Parity: {model_name} (xc={xc}) ===")
    # 1. FP32 fast engine (exact live GCMC target)
    fast32 = FastMaceD3Target(
        cif_path="data/MgMOF-74.cif",
        supercell=(1, 1, 2),
        model=model_name,
        device="cuda",
        dispersion=True,
        dispersion_xc=xc,
    )
    e32_K = [fast32.eval_batch([st])[0][0] for st in snapshots]
    e32_kJ = np.array([e * K_TO_KJMOL for e in e32_K], dtype=np.float64)
    del fast32
    torch.cuda.empty_cache()

    # 2. Full FP64 ASE reference
    torch.set_default_dtype(torch.float64)
    calc64 = mace_mp(
        model=model_name,
        dispersion=True,
        dispersion_xc=xc,
        default_dtype="float64",
        device="cuda",
    )
    fw64 = fw_base.copy()
    fw64.calc = calc64
    e_fw64 = float(fw64.get_potential_energy()) * EV_TO_KJMOL

    co2_64 = Atoms("CO2", positions=[[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]])
    co2_64.set_cell([30.0, 30.0, 30.0])
    co2_64.set_pbc(True)
    co2_64.calc = calc64
    e_co2_64 = float(co2_64.get_potential_energy()) * EV_TO_KJMOL

    e64_kJ = []
    for idx, st in enumerate(snapshots):
        at = build_ase_atoms(fw_base, st)
        at.calc = calc64
        e_tot = float(at.get_potential_energy()) * EV_TO_KJMOL
        e_bind = e_tot - e_fw64 - st["N"] * e_co2_64
        e64_kJ.append(e_bind)
    e64_kJ = np.array(e64_kJ, dtype=np.float64)
    del calc64
    torch.set_default_dtype(torch.float32)
    torch.cuda.empty_cache()

    Ns = np.array([st["N"] for st in snapshots], dtype=np.float64)
    abs_err_tot = np.abs(e32_kJ - e64_kJ)
    abs_err_per_mol = abs_err_tot / Ns
    # Pairwise transition error |dU_32 - dU_64| within same loading bin
    pair_errs = []
    for i in range(len(snapshots)):
        for j in range(i + 1, len(snapshots)):
            d32 = e32_kJ[j] - e32_kJ[i]
            d64 = e64_kJ[j] - e64_kJ[i]
            pair_errs.append(abs(d32 - d64))
    pair_errs = np.array(pair_errs, dtype=np.float64)

    summary = {
        "model": model_name,
        "xc": xc,
        "n_snapshots": len(snapshots),
        "N_values": [int(n) for n in Ns],
        "e_bind_fp32_kJ_mol": [float(x) for x in e32_kJ],
        "e_bind_fp64_kJ_mol": [float(x) for x in e64_kJ],
        "max_abs_err_total_kJ_mol": float(np.max(abs_err_tot)),
        "rms_err_total_kJ_mol": float(np.sqrt(np.mean(abs_err_tot ** 2))),
        "max_abs_err_per_mol_kJ_mol": float(np.max(abs_err_per_mol)),
        "rms_err_per_mol_kJ_mol": float(np.sqrt(np.mean(abs_err_per_mol ** 2))),
        "max_pair_dU_err_kJ_mol": float(np.max(pair_errs)),
        "max_pair_dU_err_kBT_298K": float(np.max(pair_errs) / KB_T_KJMOL),
        "rms_pair_dU_err_kJ_mol": float(np.sqrt(np.mean(pair_errs ** 2))),
    }
    print(
        f"  Max |dU_tot| = {summary['max_abs_err_total_kJ_mol']:.5f} kJ/mol | "
        f"Max |dU/N| = {summary['max_abs_err_per_mol_kJ_mol']:.5f} kJ/mol/CO2\n"
        f"  Max pairwise |ddU| = {summary['max_pair_dU_err_kJ_mol']:.5f} kJ/mol "
        f"({summary['max_pair_dU_err_kBT_298K']:.5f} kB*T @ 298 K)"
    )
    return summary


def main():
    cif_path = ROOT / "data" / "MgMOF-74.cif"
    uc = ase_read(str(cif_path))
    fw_base = uc.repeat((1, 1, 2))
    snapshots = generate_snapshots(cif_path)
    print(f"Generated {len(snapshots)} snapshots with N={[s['N'] for s in snapshots]}")

    results = [
        eval_model_pair("mace-matpes-r2scan-0", "r2scan", snapshots, fw_base),
        eval_model_pair("medium", "pbe", snapshots, fw_base),
    ]
    out_path = BUILD / "stage1_fp32_parity.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
