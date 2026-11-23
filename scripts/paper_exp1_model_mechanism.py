"""Paper Experiment 1 (Detail 1): Mechanism Audit Across 4 MACE Foundation Models.
Verifies:
  1. Exact far-field 7.31 A pore-center zero-interaction energy (No-D3 & +D3) & isolated CO2 consistency.
  2. Unobstructed Mg2+-O(CO2) radial binding curves across MP-0, MPA-0, MP-0b3, MatPES-r2SCAN-0 (No-D3 & +D3).
  3. Unobstructed Ni2+-O(CO2) radial binding curves for MatPES-r2SCAN-0+D3(r2scan) & MP-0+D3(pbe).
"""

import json
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from mace.calculators import mace_mp

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
BUILD.mkdir(parents=True, exist_ok=True)


def find_pore_center_and_open_ray(cif_path: str, metal_sym: str):
    fw = ase_read(cif_path).repeat((1, 1, 2))
    cell = np.array(fw.get_cell(), dtype=np.float64)
    inv_cell = np.linalg.inv(cell)
    pos = fw.get_positions()
    syms = np.array(fw.get_chemical_symbols())

    rng = np.random.default_rng(42)
    best_d, best_c = -1.0, None
    for _ in range(3000):
        c = rng.random(3) @ cell
        f = (pos - c) @ inv_cell
        f -= np.round(f)
        d = float(np.linalg.norm(f @ cell, axis=1).min())
        if d > best_d:
            best_d, best_c = d, c

    m_idx = int(np.where(syms == metal_sym)[0][0])
    m_pos = pos[m_idx]
    best_md, best_u = -1.0, None
    for _ in range(1500):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        probe = m_pos + 2.4 * u
        f = (pos - probe) @ inv_cell
        f -= np.round(f)
        dists = np.linalg.norm(f @ cell, axis=1)
        dists[m_idx] = 99.0
        md = float(dists.min())
        if md > best_md:
            best_md, best_u = md, u
    return fw, best_c, best_d, m_pos, best_u, best_md


def eval_curve(fw, pore_c, metal_pos, ray_u, radii, model_name: str, disp: bool, xc: str):
    calc = mace_mp(
        model=model_name, dispersion=disp, dispersion_xc=xc,
        default_dtype="float32", device="cuda"
    )
    fwc = fw.copy()
    fwc.calc = calc
    e_fw = float(fwc.get_potential_energy())

    co2_pbc = Atoms("CO2", positions=[[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]], cell=[30, 30, 30], pbc=True)
    co2_pbc.calc = calc
    e_co2_pbc = float(co2_pbc.get_potential_energy())

    co2_nopbc = Atoms("CO2", positions=[[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]], pbc=False)
    co2_nopbc.calc = calc
    e_co2_nopbc = float(co2_nopbc.get_potential_energy())

    sp = fw.copy() + Atoms("CO2", positions=[pore_c, pore_c + [0, 0, 1.16], pore_c - [0, 0, 1.16]])
    sp.calc = calc
    e_far_kJ = (float(sp.get_potential_energy()) - e_fw - e_co2_pbc) * 96.485332

    energies_kJ = []
    for r in radii:
        o1 = metal_pos + r * ray_u
        c = o1 + 1.16 * ray_u
        o2 = c + 1.16 * ray_u
        at = fw.copy() + Atoms("CO2", positions=[c, o1, o2])
        at.calc = calc
        eb = (float(at.get_potential_energy()) - e_fw - e_co2_pbc) * 96.485332
        energies_kJ.append(float(eb))

    return {
        "model": model_name,
        "dispersion": disp,
        "xc": xc,
        "e_co2_pbc_eV": e_co2_pbc,
        "e_co2_nopbc_eV": e_co2_nopbc,
        "far_field_kJ_mol": float(e_far_kJ),
        "radii_A": [float(r) for r in radii],
        "E_bind_kJ_mol": energies_kJ,
        "min_E_kJ_mol": float(min(energies_kJ)),
        "r_min_A": float(radii[int(np.argmin(energies_kJ))]),
    }


def main():
    mg_fw, mg_pore_c, mg_pore_d, mg_pos, mg_u, mg_open_d = find_pore_center_and_open_ray(str(ROOT / "data" / "MgMOF-74.cif"), "Mg")
    ni_fw, ni_pore_c, ni_pore_d, ni_pos, ni_u, ni_open_d = find_pore_center_and_open_ray(str(ROOT / "data" / "NiMOF-74.cif"), "Ni")
    print(f"Mg-MOF-74: pore_center min_d={mg_pore_d:.3f} A, open ray min_other={mg_open_d:.3f} A")
    print(f"Ni-MOF-74: pore_center min_d={ni_pore_d:.3f} A, open ray min_other={ni_open_d:.3f} A")

    radii_mg = [2.00, 2.20, 2.35, 2.50, 2.65, 2.85, 3.10, 3.50, 4.20, 5.20]
    radii_ni = [1.95, 2.15, 2.30, 2.45, 2.65, 2.90, 3.30, 4.00, 5.00]

    mg_specs = [
        ("medium", False, "pbe", "MACE-MP-0 (No D3)"),
        ("medium", True, "pbe", "MACE-MP-0 + D3(PBE)"),
        ("medium-mpa-0", False, "pbe", "MACE-MPA-0 (No D3)"),
        ("medium-mpa-0", True, "pbe", "MACE-MPA-0 + D3(PBE)"),
        ("medium-0b3", False, "pbe", "MACE-MP-0b3 (No D3)"),
        ("medium-0b3", True, "pbe", "MACE-MP-0b3 + D3(PBE)"),
        ("mace-matpes-r2scan-0", False, "r2scan", "MACE-MatPES-r2SCAN-0 (No D3)"),
        ("mace-matpes-r2scan-0", True, "r2scan", "MACE-MatPES-r2SCAN-0 + D3(r2scan)"),
    ]

    mg_curves = []
    for m, disp, xc, label in mg_specs:
        res = eval_curve(mg_fw, mg_pore_c, mg_pos, mg_u, radii_mg, m, disp, xc)
        res["label"] = label
        mg_curves.append(res)
        print(f"  [Mg] {label:36s}: far={res['far_field_kJ_mol']:+.2f} | min={res['min_E_kJ_mol']:+.2f} kJ/mol @ {res['r_min_A']:.2f} A")

    ni_specs = [
        ("mace-matpes-r2scan-0", True, "r2scan", "MACE-MatPES-r2SCAN-0 + D3(r2scan)"),
        ("medium", True, "pbe", "MACE-MP-0 + D3(PBE)"),
    ]
    ni_curves = []
    for m, disp, xc, label in ni_specs:
        res = eval_curve(ni_fw, ni_pore_c, ni_pos, ni_u, radii_ni, m, disp, xc)
        res["label"] = label
        ni_curves.append(res)
        print(f"  [Ni] {label:36s}: far={res['far_field_kJ_mol']:+.2f} | min={res['min_E_kJ_mol']:+.2f} kJ/mol @ {res['r_min_A']:.2f} A")

    out = {
        "mg_pore_center_dist_A": mg_pore_d,
        "mg_open_ray_other_dist_A": mg_open_d,
        "ni_pore_center_dist_A": ni_pore_d,
        "ni_open_ray_other_dist_A": ni_open_d,
        "mg_curves": mg_curves,
        "ni_curves": ni_curves,
    }
    out_path = BUILD / "paper_exp1_mechanism.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
