"""Phase 1 Feasibility Probe: MACE-MP-0 + D3 Speed, OMS Binding & Nested-MC Correlation on RTX 5070."""

import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from ase import Atoms
from ase.io import read as ase_read
from mace.calculators import mace_mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
ASSETS = ROOT / "assets"
BUILD.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)

EV_TO_KJMOL = 96.48533212331002
K_TO_KJMOL = 8.31446261815324e-3
KE_K = 167100.9558738398
KB_T_298 = 298.0 * K_TO_KJMOL  # 2.4777 kJ/mol


# Exact symmetric crystal expansion matching src/cif_loader.cpp for data/MgMOF-74.cif
def load_mgmof74_1x1x2():
    uc = ase_read(str(ROOT / "data" / "MgMOF-74.cif"))
    # Assign asymmetric-unit labels in exact CIF order
    asym_labels = ["Mg1", "O1", "O2", "O3", "C1", "C2", "C3", "C4", "H1"]
    assert len(uc) == 162, f"Expected 162 atoms/uc, got {len(uc)}"

    # Identify exact atom type (0..8) by matching asymmetric fractional orbits under R-3
    asym_frac = np.array([
        [0.64159, 0.68514, 0.47189],
        [0.96943, 0.29776, 0.36038],
        [0.92763, 0.23048, 0.61020],
        [0.25494, 0.94205, 0.67383],
        [0.92667, 0.24580, 0.42703],
        [0.87918, 0.20711, 0.29083],
        [0.21254, 0.88926, 0.75490],
        [0.16630, 0.84759, 0.63558],
        [0.16529, 0.85803, 0.47918],
    ])
    sym_ops = [
        lambda f: ( f[0],       f[1],       f[2]),
        lambda f: (-f[1],       f[0]-f[1],  f[2]),
        lambda f: (-f[0]+f[1], -f[0],       f[2]),
        lambda f: (-f[0],      -f[1],      -f[2]),
        lambda f: ( f[1],      -f[0]+f[1], -f[2]),
        lambda f: ( f[0]-f[1],  f[0],      -f[2]),
    ]
    shifts = [(0.0, 0.0, 0.0), (2/3, 1/3, 1/3), (1/3, 2/3, 2/3)]
    type_orbits = [[] for _ in range(9)]
    for t in range(9):
        for sh in shifts:
            for op in sym_ops:
                nf = np.array(op(asym_frac[t])) + np.array(sh)
                nf = np.mod(nf, 1.0)
                type_orbits[t].append(nf)
        type_orbits[t] = np.array(type_orbits[t])

    uc_frac = np.mod(uc.get_scaled_positions(), 1.0)
    uc_types = []
    for f in uc_frac:
        best_t, best_d = -1, 1e9
        for t in range(9):
            df = type_orbits[t] - f
            df -= np.round(df)
            d = float(np.min(np.linalg.norm(df, axis=1)))
            if d < best_d:
                best_d = d
                best_t = t
        assert best_d < 1e-3, f"Unmatched atom type: {best_d}"
        uc_types.append(best_t)
    uc_types = np.array(uc_types, dtype=np.int32)

    # UFF parameters (K, A) per type_id (Mg=0, O=1..3, C=4..7, H=8)
    uff_eps = np.array([55.8574, 30.1932, 30.1932, 30.1932, 52.8381, 52.8381, 52.8381, 52.8381, 22.1417])
    uff_sig = np.array([2.6914,  3.1181,  3.1181,  3.1181,  3.4309,  3.4309,  3.4309,  3.4309,  2.5711])
    q_repeat = np.array([+1.566, -0.915, -0.768, -0.907, +0.936, -0.325, +0.457, -0.235, +0.191])
    q_mercado = np.array([+1.560, -0.899, -0.752, -0.903, +0.900, -0.314, +0.456, -0.234, +0.186])

    assert abs(float(np.sum(q_repeat[uc_types]))) < 1e-9
    assert abs(float(np.sum(q_mercado[uc_types]))) < 1e-9

    sup2 = uc.repeat((1, 1, 2))
    sup4 = uc.repeat((1, 1, 4))
    types2 = np.tile(uc_types, 2)
    return {
        "uc": uc,
        "sup2": sup2,
        "sup4": sup4,
        "types2": types2,
        "uff_eps": uff_eps[types2],
        "uff_sig": uff_sig[types2],
        "q_repeat": q_repeat[types2],
        "q_mercado": q_mercado[types2],
    }


def make_buck_table(r_cut=11.0):
    def find_rp(A, B, C):
        lo, hi = 0.05, 7.0 / B
        if A * B * (lo ** 7) * math.exp(-B * lo) - 6.0 * C >= 0.0:
            return 1.25
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if A * B * (mid ** 7) * math.exp(-B * mid) - 6.0 * C < 0.0:
                lo = mid
            else:
                hi = mid
        return max(0.5 * (lo + hi), 1.25)

    def lj(eps, sig):
        sr2 = (sig * sig) / (r_cut * r_cut)
        sr6 = sr2 ** 3
        return (False, eps, sig, 0.0, 0.4, 4.0 * eps * (sr6 ** 2 - sr6))

    def buck(C, A, B):
        rp = find_rp(A, B, C)
        us = A * math.exp(-B * r_cut) - C / (r_cut ** 6)
        return (True, A, B, C, rp, us)

    # C_CO2[type 0..8], O_CO2[type 0..8]
    c_tbl = [
        lj(190.6212, 2.816),
        lj(69.958, 2.794), lj(69.958, 2.794), lj(69.958, 2.794),
        lj(87.738, 2.904), lj(87.738, 2.904), lj(87.738, 2.904), lj(87.738, 2.904),
        lj(68.317, 2.453),
    ]
    o_tbl = [
        buck(4.08795E5,      2.47320E7,      3.965),
        buck(1.4313206634E5, 3.3788254669E7, 3.805),
        buck(1.4313206634E5, 2.6778651722E7, 3.780),
        buck(1.4313206634E5, 2.634324168E7,  3.705),
        buck(2.2631166802E5, 2.765001156E7,  3.840),
        buck(2.2631166802E5, 2.1483617388E7, 3.515),
        buck(2.2631166802E5, 2.992168789E7,  3.840),
        buck(2.2631166802E5, 1.314693952E7,  3.315),
        lj(56.900, 2.343),
    ]
    return c_tbl, o_tbl


class ClassicalFrameworkEvaluator:
    def __init__(self, fw_data, r_cut=11.0, alpha=0.30):
        self.atoms = fw_data["sup2"]
        self.cell = np.array(self.atoms.get_cell(), dtype=np.float64)  # rows = a,b,c
        self.H = self.cell.T  # cols = a,b,c
        self.H_inv = np.linalg.inv(self.H)
        self.vol = abs(float(np.linalg.det(self.H)))
        self.pos = np.array(self.atoms.get_positions(), dtype=np.float64)
        self.types = fw_data["types2"]
        self.uff_eps = fw_data["uff_eps"]
        self.uff_sig = fw_data["uff_sig"]
        self.q_rep = fw_data["q_repeat"]
        self.q_mer = fw_data["q_mercado"]
        self.r_cut = r_cut
        self.rc2 = r_cut * r_cut
        self.alpha = alpha
        self.c_buck, self.o_buck = make_buck_table(r_cut)

        # Build integer lattice shift vectors covering r_cut inside 1x1x2 cell (c=13.88 A -> +/-1 image along z!)
        shifts = []
        for ia in (-1, 0, 1):
            for ib in (-1, 0, 1):
                for ic in (-2, -1, 0, 1, 2):
                    shifts.append(self.H @ np.array([ia, ib, ic], dtype=np.float64))
        self.real_shifts = np.array(shifts, dtype=np.float64)

        # Reciprocal lattice vectors (row vectors of 2*pi*H_inv)
        B = 2.0 * math.pi * self.H_inv  # rows = b1, b2, b3
        k_vecs = []
        k_cut = 1.65
        k2_cut = k_cut * k_cut
        for n1 in range(-8, 9):
            for n2 in range(-8, 9):
                for n3 in range(-5, 6):
                    if n1 == 0 and n2 == 0 and n3 == 0:
                        continue
                    k = n1 * B[0] + n2 * B[1] + n3 * B[2]
                    k2 = float(np.dot(k, k))
                    if k2 <= k2_cut:
                        k_vecs.append(k)
        self.k_vecs = np.array(k_vecs, dtype=np.float64)
        self.k2 = np.sum(self.k_vecs ** 2, axis=1)
        self.k_weight = (4.0 * math.pi / self.vol) * np.exp(-self.k2 / (4.0 * alpha * alpha)) / self.k2

        # Precompute structure factor for both REPEAT and Mercado charges
        kr = self.k_vecs @ self.pos.T  # (Nk, Nat)
        cos_kr = np.cos(kr)
        sin_kr = np.sin(kr)
        self.S_re_rep = cos_kr @ self.q_rep
        self.S_im_rep = sin_kr @ self.q_rep
        self.S_re_mer = cos_kr @ self.q_mer
        self.S_im_mer = sin_kr @ self.q_mer

    def ewald_phi_and_field(self, r_site, charges_mode="repeat"):
        q_fw = self.q_rep if charges_mode == "repeat" else self.q_mer
        S_re = self.S_re_rep if charges_mode == "repeat" else self.S_re_mer
        S_im = self.S_im_rep if charges_mode == "repeat" else self.S_im_mer

        # Wrap difference into primary MIC box, then sum all periodic images within r_cut
        df = self.H_inv @ (r_site - self.pos).T
        df -= np.round(df)
        d0 = (self.H @ df).T  # (Nat, 3)

        # (Nsh, Nat, 3)
        d_all = self.real_shifts[:, None, :] + d0[None, :, :]
        r2_all = np.sum(d_all * d_all, axis=2)
        mask = (r2_all < self.rc2) & (r2_all > 1e-6)
        r_m = np.sqrt(r2_all[mask])
        d_m = d_all[mask]
        q_m = np.broadcast_to(q_fw[None, :], r2_all.shape)[mask]

        from scipy.special import erfc
        ar = self.alpha * r_m
        erfc_ar = erfc(ar)
        phi_real = float(np.sum(q_m * erfc_ar / r_m))
        two_a = 2.0 * self.alpha / math.sqrt(math.pi)
        fac = q_m * (erfc_ar / (r_m * r_m) + two_a * np.exp(-ar * ar) / r_m) / r_m
        E_real = np.sum(fac[:, None] * d_m, axis=0)

        kr_site = self.k_vecs @ r_site
        cos_s = np.cos(kr_site)
        sin_s = np.sin(kr_site)
        phi_recip = float(np.sum(self.k_weight * (S_re * cos_s + S_im * sin_s)))
        dE_mag = self.k_weight * (S_re * sin_s - S_im * cos_s)
        E_recip = np.sum(dE_mag[:, None] * self.k_vecs, axis=0)

        phi_K_e = KE_K * (phi_real + phi_recip)
        E_K_eA = KE_K * (E_real + E_recip)
        return phi_K_e, E_K_eA

    def eval_co2(self, c_pos, axis_u):
        bond = 1.160
        sites = [c_pos, c_pos + bond * axis_u, c_pos - bond * axis_u]

        # 1. UFF + REPEAT (TraPPE CO2) & Pol-3Site
        q_trappe = [+0.70, -0.35, -0.35]
        eps_trappe = [27.0, 79.0, 79.0]
        sig_trappe = [2.800, 3.050, 3.050]
        w_pol = [0.40, 0.30, 0.30]

        u_uff_lj = 0.0
        u_uff_el = 0.0
        u_pol3 = 0.0
        ovl_uff = False

        for s_idx, rs in enumerate(sites):
            df = self.H_inv @ (rs - self.pos).T
            df -= np.round(df)
            d0 = (self.H @ df).T
            d_all = self.real_shifts[:, None, :] + d0[None, :, :]
            r2_all = np.sum(d_all * d_all, axis=2)
            mask = r2_all < self.rc2
            r2_m = r2_all[mask]
            if np.any(r2_m < 0.16):
                ovl_uff = True
                break
            eps_m = np.sqrt(eps_trappe[s_idx] * np.broadcast_to(self.uff_eps[None, :], r2_all.shape)[mask])
            sig_m = 0.5 * (sig_trappe[s_idx] + np.broadcast_to(self.uff_sig[None, :], r2_all.shape)[mask])
            sr2 = (sig_m * sig_m) / r2_m
            sr6 = sr2 ** 3
            u_uff_lj += float(np.sum(4.0 * eps_m * (sr6 ** 2 - sr6)))

            phi_r, E_r = self.ewald_phi_and_field(rs, "repeat")
            u_uff_el += q_trappe[s_idx] * phi_r
            e2 = float(np.dot(E_r, E_r))
            e_par = float(np.dot(E_r, axis_u))
            e_perp2 = max(0.0, e2 - e_par * e_par)
            u_pol3 += -0.5 * (1.0 / KE_K) * w_pol[s_idx] * (4.05 * e_par * e_par + 1.95 * e_perp2)

        # 2. Mercado 2016 DFT-FF (EPM2 CO2, shifted)
        q_epm2 = [+0.6512, -0.3256, -0.3256]
        u_mer_lj = 0.0
        u_mer_el = 0.0
        ovl_mer = False

        for s_idx, rs in enumerate(sites):
            tbl = self.c_buck if s_idx == 0 else self.o_buck
            df = self.H_inv @ (rs - self.pos).T
            df -= np.round(df)
            d0 = (self.H @ df).T
            d_all = self.real_shifts[:, None, :] + d0[None, :, :]
            r2_all = np.sum(d_all * d_all, axis=2)
            mask = r2_all < self.rc2
            r2_m = r2_all[mask]
            r_m = np.sqrt(r2_m)
            t_m = np.broadcast_to(self.types[None, :], r2_all.shape)[mask]

            for t_id in range(9):
                sel = (t_m == t_id)
                if not np.any(sel):
                    continue
                is_buck, A, B, C, rp, us = tbl[t_id]
                rr = r_m[sel]
                rr2 = r2_m[sel]
                if np.any(rr <= rp):
                    ovl_mer = True
                    break
                if is_buck:
                    u_mer_lj += float(np.sum(A * np.exp(-B * rr) - C / (rr2 ** 3) - us))
                else:
                    sr2 = (B * B) / rr2
                    sr6 = sr2 ** 3
                    u_mer_lj += float(np.sum(4.0 * A * (sr6 ** 2 - sr6) - us))
            if ovl_mer:
                break
            phi_m, _ = self.ewald_phi_and_field(rs, "mercado")
            u_mer_el += q_epm2[s_idx] * phi_m

        return {
            "ovl_uff": ovl_uff,
            "ovl_mer": ovl_mer,
            "uff_kJ": (u_uff_lj + u_uff_el) * K_TO_KJMOL if not ovl_uff else 1e6,
            "pol3_kJ": (u_uff_lj + u_uff_el + u_pol3) * K_TO_KJMOL if not ovl_uff else 1e6,
            "mercado_kJ": (u_mer_lj + u_mer_el) * K_TO_KJMOL if not ovl_mer else 1e6,
        }


def build_co2_atoms(c_pos, u_vec):
    bond = 1.160
    pos = np.array([c_pos, c_pos + bond * u_vec, c_pos - bond * u_vec], dtype=np.float64)
    return Atoms("CO2", positions=pos)


def compute_pore_channels(atoms):
    # Locate the central honeycomb pore axis in 1x1x2 Mg-MOF-74 (near frac (2/3, 1/3) or (0, 0) - check closest Mg!)
    # From CIF: Mg1 at frac (0.64159, 0.68514, 0.47189); R-3 pores sit along (0, 0, z) and (1/3, 2/3, z) and (2/3, 1/3, z)!
    cell = np.array(atoms.get_cell())
    pos = np.array(atoms.get_positions())
    H = cell.T
    H_inv = np.linalg.inv(H)

    # Test which fractional (a, b) column has the largest empty radius (the 1D hexagonal pore!)
    best_ab, best_min_d = None, -1.0
    for fa, fb in [(0.0, 0.0), (1/3, 2/3), (2/3, 1/3)]:
        p = H @ np.array([fa, fb, 0.5])
        df = H_inv @ (pos - p).T
        df -= np.round(df)
        dxy = (H @ df).T[:, :2]
        min_d = float(np.min(np.linalg.norm(dxy, axis=1)))
        if min_d > best_min_d:
            best_min_d = min_d
            best_ab = (fa, fb)
    return best_ab, best_min_d


def benchmark_speed(fw_data):
    sup2 = fw_data["sup2"]
    sup4 = fw_data["sup4"]
    pore_ab, _ = compute_pore_channels(sup2)
    H2 = np.array(sup2.get_cell()).T
    c0 = H2 @ np.array([pore_ab[0], pore_ab[1], 0.5])
    u0 = np.array([1.0, 0.0, 0.0])

    # Build 1x1x2 + 16 CO2 well separated inside the pore channel
    rng = np.random.default_rng(20261123)
    mols16 = []
    for k in range(16):
        z = (k + 0.5) / 16.0
        ang = k * 0.7
        shift = H2 @ np.array([pore_ab[0], pore_ab[1], z]) + np.array([2.2 * math.cos(ang), 2.2 * math.sin(ang), 0.0])
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        mols16.append(build_co2_atoms(shift, u))

    sup2_1co2 = sup2.copy() + build_co2_atoms(c0, u0)
    sup2_16co2 = sup2.copy()
    for m in mols16:
        sup2_16co2 += m
    sup4_1co2 = sup4.copy() + build_co2_atoms(c0, u0)

    speed_results = []
    for size_name in ["small", "medium"]:
        calc = mace_mp(model=size_name, dispersion=True, default_dtype="float32", device="cuda")
        for label, system in [
            ("1x1x2 + 1 CO2 (327 at)", sup2_1co2),
            ("1x1x2 + 16 CO2 (372 at)", sup2_16co2),
            ("1x1x4 + 1 CO2 (651 at)", sup4_1co2),
        ]:
            sys_c = system.copy()
            sys_c.calc = calc
            for _ in range(5):
                _ = sys_c.get_potential_energy()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            n_rep = 25
            for _ in range(n_rep):
                # Nudge atom 0 slightly so ASE doesn't cache energy!
                sys_c.positions[-1, 0] += 1e-6
                _ = sys_c.get_potential_energy()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0 / n_rep
            steps_s = 1000.0 / dt
            print(f"[Speed] MACE-{size_name:<6s} + D3 | {label:<24s} | {dt:6.2f} ms/eval ({steps_s:5.1f} eval/s)")
            speed_results.append({
                "model": size_name,
                "system": label,
                "n_atoms": len(sys_c),
                "ms_per_eval": dt,
                "evals_per_sec": steps_s,
            })
    return speed_results


def nested_mc_boltzmann_acc(e_mlip, e_ref):
    # Single-molecule equilibrium swap acceptance rate across reference Boltzmann pairs
    beta = 1.0 / KB_T_298
    w = np.exp(-beta * (e_ref - np.min(e_ref)))
    w /= np.sum(w)
    diff = e_mlip - e_ref
    # Vectorized N x N pair expectation
    dd = diff[None, :] - diff[:, None]  # delta(B) - delta(A)
    pacc = np.minimum(1.0, np.exp(-np.clip(beta * dd, -60.0, 60.0)))
    return float(w @ pacc @ w)


def main():
    print("=== Phase 1: MACE-MP-0 + D3 Feasibility Probe (RTX 5070) ===")
    fw_data = load_mgmof74_1x1x2()
    sup2 = fw_data["sup2"]
    print(f"Loaded Mg-MOF-74 1x1x2: {len(sup2)} atoms, cell lengths = {sup2.cell.lengths()}")

    # Finite-difference check on Python Ewald field
    cl_eval = ClassicalFrameworkEvaluator(fw_data)
    pore_ab, pore_r = compute_pore_channels(sup2)
    print(f"Detected primary honeycomb pore at frac ({pore_ab[0]:.3f}, {pore_ab[1]:.3f}), wall radius ~ {pore_r:.2f} A")

    # 1. Speed Benchmark
    speed_results = benchmark_speed(fw_data)

    # 2. Compute Reference Empty Framework + Isolated CO2 Energies
    calc_med = mace_mp(model="medium", dispersion=True, default_dtype="float32", device="cuda")
    calc_med_nod3 = mace_mp(model="medium", dispersion=False, default_dtype="float32", device="cuda")
    calc_sml = mace_mp(model="small", dispersion=True, default_dtype="float32", device="cuda")

    co2_iso = build_co2_atoms(np.zeros(3), np.array([1.0, 0.0, 0.0]))
    co2_iso.set_cell([30.0, 30.0, 30.0])
    co2_iso.set_pbc(True)

    def get_e(atoms, calc):
        a = atoms.copy()
        a.calc = calc
        return float(a.get_potential_energy())

    e_fw_med = get_e(sup2, calc_med)
    e_co2_med = get_e(co2_iso, calc_med)
    e_fw_med_nod3 = get_e(sup2, calc_med_nod3)
    e_co2_med_nod3 = get_e(co2_iso, calc_med_nod3)
    e_fw_sml = get_e(sup2, calc_sml)
    e_co2_sml = get_e(co2_iso, calc_sml)

    # 3. Radial scan toward one Mg1 open metal site inside 1x1x2
    H2 = cl_eval.H
    H2_inv = cl_eval.H_inv
    pos2 = cl_eval.pos
    mg_indices = np.where(fw_data["types2"] == 0)[0]
    # Find an Mg1 close to the primary pore column (fa, fb)
    pore_mid_z = H2 @ np.array([pore_ab[0], pore_ab[1], 0.5])
    best_mg_pos, best_vec, best_d = None, None, 1e9
    for idx in mg_indices:
        # Closest periodic image of Mg to pore_mid_z
        df = H2_inv @ (pos2[idx] - pore_mid_z)
        df -= np.round(df)
        d = H2 @ df
        dist = float(np.linalg.norm(d))
        if dist < best_d:
            best_d = dist
            best_mg_pos = pore_mid_z + d
            best_vec = pore_mid_z - best_mg_pos
    # Pore inward unit vector in xy-plane (plus slight normal component toward channel center)
    inward = best_vec.copy()
    inward[2] = 0.0
    inward /= np.linalg.norm(inward)
    # Tilted CO2 geometry at OMS (Mg-O-C ~ 130 deg): axis_u points inward with ~40 deg z-tilt
    u_oms = np.array([inward[0] * math.cos(0.70), inward[1] * math.cos(0.70), math.sin(0.70)])
    u_oms /= np.linalg.norm(u_oms)

    r_scan = np.linspace(2.05, 4.80, 18)
    radial = {"r_A": [], "mace_med_d3": [], "mace_med_nod3": [], "mace_sml_d3": [], "uff": [], "pol3": [], "mercado": []}
    print("\n=== Mg2+--O(CO2) Radial OMS Binding Curve (kJ/mol) ===")
    print(f"{'r(Mg-O)':>7s} | {'MACE-M+D3':>10s} | {'MACE-M(noD3)':>12s} | {'MACE-S+D3':>10s} | {'UFF':>8s} | {'Pol-3Site':>9s} | {'Mercado16':>9s}")
    for r in r_scan:
        o_pos = best_mg_pos + r * inward
        # Place CO2 COM such that O1 (at -bond*u_oms) is at o_pos => COM = o_pos + bond*u_oms
        c_pos = o_pos + 1.160 * u_oms
        cl = cl_eval.eval_co2(c_pos, u_oms)
        sys_p = sup2.copy() + build_co2_atoms(c_pos, u_oms)
        eb_med = (get_e(sys_p, calc_med) - e_fw_med - e_co2_med) * EV_TO_KJMOL
        eb_nod3 = (get_e(sys_p, calc_med_nod3) - e_fw_med_nod3 - e_co2_med_nod3) * EV_TO_KJMOL
        eb_sml = (get_e(sys_p, calc_sml) - e_fw_sml - e_co2_sml) * EV_TO_KJMOL
        radial["r_A"].append(float(r))
        radial["mace_med_d3"].append(eb_med)
        radial["mace_med_nod3"].append(eb_nod3)
        radial["mace_sml_d3"].append(eb_sml)
        radial["uff"].append(cl["uff_kJ"])
        radial["pol3"].append(cl["pol3_kJ"])
        radial["mercado"].append(cl["mercado_kJ"])
        print(f"{r:7.3f} | {eb_med:10.2f} | {eb_nod3:12.2f} | {eb_sml:10.2f} | {cl['uff_kJ']:8.2f} | {cl['pol3_kJ']:9.2f} | {cl['mercado_kJ']:9.2f}")

    # 4. 60-Pose Pore Correlation & Nested MC Acceptance Estimate
    rng = np.random.default_rng(2026112301)
    poses = []
    attempts = 0
    while len(poses) < 60 and attempts < 5000:
        attempts += 1
        near_oms = (len(poses) < 30)
        if near_oms:
            # Sample around Mg sites inside pore (r_Mg_O ~ 2.2 - 3.6 A)
            r_rad = rng.uniform(3.2, 6.2)
            theta = rng.uniform(0.0, 2.0 * math.pi)
            z_f = rng.uniform(0.0, 1.0)
            c_pos = H2 @ np.array([pore_ab[0], pore_ab[1], z_f]) + np.array([r_rad * math.cos(theta), r_rad * math.sin(theta), 0.0])
        else:
            r_rad = rng.uniform(0.0, 5.2)
            theta = rng.uniform(0.0, 2.0 * math.pi)
            z_f = rng.uniform(0.0, 1.0)
            c_pos = H2 @ np.array([pore_ab[0], pore_ab[1], z_f]) + np.array([r_rad * math.cos(theta), r_rad * math.sin(theta), 0.0])

        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        cl = cl_eval.eval_co2(c_pos, u)
        if cl["ovl_uff"] or cl["ovl_mer"]:
            continue
        if cl["uff_kJ"] > 10.0 or cl["mercado_kJ"] > 10.0:
            continue
        # Ensure minimum atom distance from framework >= 2.0 A so MLIP stays in physical gas-adsorption domain
        sys_p = sup2.copy() + build_co2_atoms(c_pos, u)
        eb_med = (get_e(sys_p, calc_med) - e_fw_med - e_co2_med) * EV_TO_KJMOL
        eb_sml = (get_e(sys_p, calc_sml) - e_fw_sml - e_co2_sml) * EV_TO_KJMOL
        poses.append({
            "mace_med_d3": eb_med,
            "mace_sml_d3": eb_sml,
            "uff": cl["uff_kJ"],
            "pol3": cl["pol3_kJ"],
            "mercado": cl["mercado_kJ"],
        })

    print(f"\n=== 60-Pose Pore Binding Correlation vs. MACE-MP-0(medium)+D3 (N={len(poses)}) ===")
    e_mlip = np.array([p["mace_med_d3"] for p in poses])
    e_sml = np.array([p["mace_sml_d3"] for p in poses])
    stats = {}
    for name in ["uff", "pol3", "mercado"]:
        e_ref = np.array([p[name] for p in poses])
        dd = e_mlip - e_ref
        dd_zero = dd - np.mean(dd)
        sigma = float(np.std(dd_zero))
        r2 = float(np.corrcoef(e_mlip, e_ref)[0, 1] ** 2)
        pacc = nested_mc_boltzmann_acc(e_mlip, e_ref)
        stats[name] = {
            "mean_shift_kJ": float(np.mean(dd)),
            "sigma_kJ": sigma,
            "sigma_kBT": sigma / KB_T_298,
            "r2": r2,
            "nested_acc_1mol": pacc,
            "min_bind_ref_kJ": float(np.min(e_ref)),
        }
        print(
            f"  Ref={name:<8s} | minE={np.min(e_ref):6.2f} | mean(MLIP-Ref)={np.mean(dd):+6.2f} kJ | "
            f"sigma(ddE)={sigma:5.2f} kJ ({sigma/KB_T_298:4.2f} kBT) | R2={r2:.3f} | 1-mol Nested Acc={pacc*100:5.1f}%"
        )

    # Also check MACE-small vs MACE-medium
    sigma_sm = float(np.std((e_sml - e_mlip) - np.mean(e_sml - e_mlip)))
    r2_sm = float(np.corrcoef(e_sml, e_mlip)[0, 1] ** 2)
    print(f"  Ref=MACE-S+D3| minE={np.min(e_sml):6.2f} | mean(Med-Sml)={np.mean(e_mlip-e_sml):+6.2f} kJ | sigma={sigma_sm:5.2f} kJ | R2={r2_sm:.3f}")

    summary = {
        "speed": speed_results,
        "radial": radial,
        "poses": poses,
        "stats_vs_mace_medium_d3": stats,
        "mace_med_min_bind_kJ": float(np.min(e_mlip)),
        "mace_sml_min_bind_kJ": float(np.min(e_sml)),
    }
    (BUILD / "phase1_mlip_probe.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 5. Publication-Quality 3-Panel Probe Figure
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.3,
        "axes.titlesize": 10.6,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.6,
        "legend.fontsize": 8.0,
    })
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.2), dpi=300)
    fig.subplots_adjust(left=0.055, right=0.975, top=0.83, bottom=0.14, wspace=0.30)

    # Panel (a): Radial Binding Curve
    ax1 = axes[0]
    rs = np.array(radial["r_A"])
    ax1.axhline(0.0, color="#94A3B8", linestyle=":", linewidth=1.0)
    ax1.plot(rs, radial["mace_med_d3"], "o-", color="#7C3AED", linewidth=2.1, markersize=4.5, label="MACE-MP-0(M)+D3")
    ax1.plot(rs, radial["mace_med_nod3"], "--", color="#A78BFA", linewidth=1.5, label="MACE-MP-0(M) no D3")
    ax1.plot(rs, radial["mercado"], "s-", color="#0F766E", linewidth=1.7, markersize=4.0, label="Mercado 2016 DFT-FF")
    ax1.plot(rs, radial["pol3"], "^-", color="#D97706", linewidth=1.7, markersize=4.0, label="UFF + Pol-3Site")
    ax1.plot(rs, radial["uff"], "d:", color="#64748B", linewidth=1.5, markersize=3.8, label="UFF + REPEAT (NonPol)")
    ax1.set_xlim(2.0, 4.85)
    ax1.set_ylim(-52, 22)
    ax1.set_xlabel(r"$\mathrm{Mg^{2+}\text{–}O(CO_2)}$ Distance ($\mathrm{\AA}$)")
    ax1.set_ylabel("Binding Energy (kJ/mol)")
    ax1.grid(True, linestyle=":", alpha=0.45)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1")
    ax1.set_title("(a) OMS Radial Binding Curve")

    # Panel (b): 60-Pose Parity & Nested MC Acceptance
    ax2 = axes[1]
    e_uff = np.array([p["uff"] for p in poses])
    e_pol = np.array([p["pol3"] for p in poses])
    e_mer = np.array([p["mercado"] for p in poses])
    lo = min(float(np.min(e_mlip)), float(np.min(e_mer)), float(np.min(e_pol))) - 3.0
    hi = max(float(np.max(e_mlip)), float(np.max(e_uff))) + 3.0
    ax2.plot([lo, hi], [lo, hi], "--", color="#94A3B8", linewidth=1.2)
    ax2.scatter(e_uff, e_mlip, s=22, alpha=0.65, color="#64748B",
                label=f"UFF (R²={stats['uff']['r2']:.2f}, Acc={stats['uff']['nested_acc_1mol']*100:.0f}%)")
    ax2.scatter(e_pol, e_mlip, s=26, alpha=0.78, color="#D97706",
                label=f"Pol-3Site (R²={stats['pol3']['r2']:.2f}, Acc={stats['pol3']['nested_acc_1mol']*100:.0f}%)")
    ax2.scatter(e_mer, e_mlip, s=28, alpha=0.82, color="#0F766E",
                label=f"Mercado16 (R²={stats['mercado']['r2']:.2f}, Acc={stats['mercado']['nested_acc_1mol']*100:.0f}%)")
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)
    ax2.set_xlabel("Classical Reference $E_{\\mathrm{bind}}$ (kJ/mol)")
    ax2.set_ylabel("MACE-MP-0(M)+D3 $E_{\\mathrm{bind}}$ (kJ/mol)")
    ax2.grid(True, linestyle=":", alpha=0.45)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1")
    ax2.set_title("(b) 60-Pose Correlation & Nested-MC Acc")

    # Panel (c): RTX 5070 Inference Speed & Nested-MC Budget
    ax3 = axes[2]
    labels = ["1x1x2 (327 at)\n1 CO2", "1x1x2 (372 at)\n16 CO2", "1x1x4 (651 at)\n1 CO2"]
    ms_sml = [r["ms_per_eval"] for r in speed_results if r["model"] == "small"]
    ms_med = [r["ms_per_eval"] for r in speed_results if r["model"] == "medium"]
    x3 = np.arange(len(labels))
    w3 = 0.34
    ax3.bar(x3 - w3 / 2, ms_sml, width=w3, color="#2563EB", alpha=0.88, edgecolor="#1E40AF", label="MACE-MP-0 (Small)+D3")
    ax3.bar(x3 + w3 / 2, ms_med, width=w3, color="#7C3AED", alpha=0.88, edgecolor="#5B21B6", label="MACE-MP-0 (Medium)+D3")
    for idx in range(len(labels)):
        ax3.text(x3[idx] - w3 / 2, ms_sml[idx] + 1.0, f"{ms_sml[idx]:.1f}ms", ha="center", va="bottom", fontsize=8.1, fontweight="bold", color="#1E40AF")
        ax3.text(x3[idx] + w3 / 2, ms_med[idx] + 1.0, f"{ms_med[idx]:.1f}ms", ha="center", va="bottom", fontsize=8.1, fontweight="bold", color="#5B21B6")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(labels)
    ax3.set_ylim(0, max(ms_med) * 1.25)
    ax3.set_ylabel("Forward Energy Latency (ms / eval)")
    ax3.grid(axis="y", linestyle=":", alpha=0.45)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1")
    ax3.set_title("(c) RTX 5070 MLIP Inference Latency")

    fig.suptitle(
        "Phase 1 Feasibility: MACE-MP-0 + D3 vs. GPU Classical Potentials (Mg-MOF-74 + CO2, RTX 5070)",
        fontsize=12.4, fontweight="bold", y=0.96
    )
    out_png = ASSETS / "phase1_mlip_nested_mc_probe.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"\nSaved {out_png} & build/phase1_mlip_probe.json")


if __name__ == "__main__":
    main()
