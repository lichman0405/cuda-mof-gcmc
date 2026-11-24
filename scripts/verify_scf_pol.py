"""Exact Applequist matrix-inversion unit test & GPU GCMC verification for Level 2 Pol-SCF."""

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "build" / ("mof_gcmc_cli.exe" if os.name == "nt" else "mof_gcmc_cli")
BUILD = ROOT / "build"

KE_K = 167100.9558738398
THOLE_A = 2.1304


def mic_super(H, H_inv, dv):
    df = H_inv @ dv
    df = df - np.round(df)
    return H @ df


def verify_exact_matrix_solve():
    proc = subprocess.run([str(EXE), "--test-scf"], cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError("--test-scf failed")
    print(proc.stdout.strip())

    data = json.loads((BUILD / "scf_test_dump.json").read_text(encoding="utf-8"))
    H = np.array(data["H_super"], dtype=np.float64)
    H_inv = np.array(data["H_super_inv"], dtype=np.float64)
    rc = float(data["r_cut_A"])
    rc2 = rc * rc
    alpha = float(data["dsf_alpha"])
    bond = float(data["bond_A"])
    q_sites = data["q_sites"]
    a_par = float(data["alpha_par"])
    a_perp = float(data["alpha_perp"])
    a_iso = float(data["alpha_iso"])
    mols = data["molecules"]
    N = len(mols)

    erfc_rc = math.erfc(alpha * rc)
    exp_rc = math.exp(-alpha * alpha * rc2)
    dsf_deriv = erfc_rc / rc2 + (2.0 * alpha / math.sqrt(math.pi)) * exp_rc / rc

    # 1. Recompute E0 independently in Python and verify against GPU E0
    e0_py = np.zeros((N, 3), dtype=np.float64)
    e0_gpu = np.array([m["e0"] for m in mols], dtype=np.float64)
    c_arr = np.array([m["c"] for m in mols], dtype=np.float64)
    u_arr = np.array([m["u"] for m in mols], dtype=np.float64)
    efw_arr = np.array([m["efw"] for m in mols], dtype=np.float64)
    mu_gpu = np.array([m["mu"] for m in mols], dtype=np.float64)

    off_s = [0.0, bond, -bond]
    for i in range(N):
        e = efw_arr[i].copy()
        for j in range(N):
            if i == j:
                continue
            dm = mic_super(H, H_inv, c_arr[i] - c_arr[j])
            if np.dot(dm, dm) > (rc + bond) ** 2:
                continue
            for b in range(3):
                qb = q_sites[b]
                if qb == 0.0:
                    continue
                d = dm - off_s[b] * u_arr[j]
                r2 = float(np.dot(d, d))
                if r2 >= rc2 or r2 < 0.25:
                    continue
                r = math.sqrt(r2)
                fac = KE_K * qb * (math.erfc(alpha * r) / r2 + (2.0 * alpha / math.sqrt(math.pi)) * math.exp(-alpha * alpha * r2) / r - dsf_deriv) / r
                e += fac * d
        e0_py[i] = e

    max_e0_err = float(np.max(np.abs(e0_py - e0_gpu)))
    assert max_e0_err < 1.0e-10, f"E0 mismatch: {max_e0_err}"

    # 2. Construct exact 3N x 3N Applequist-Thole linear system: (I - A * T) mu = A E0
    M = np.eye(3 * N, dtype=np.float64)
    rhs = np.zeros(3 * N, dtype=np.float64)
    A_blocks = []
    A_inv_blocks = []
    T_blocks = {}

    for i in range(N):
        ui = u_arr[i].reshape(3, 1)
        Ai = (1.0 / KE_K) * (a_perp * np.eye(3) + (a_par - a_perp) * (ui @ ui.T))
        Ai_inv = KE_K * ((1.0 / a_perp) * np.eye(3) + (1.0 / a_par - 1.0 / a_perp) * (ui @ ui.T))
        A_blocks.append(Ai)
        A_inv_blocks.append(Ai_inv)
        rhs[3 * i:3 * i + 3] = Ai @ e0_py[i]

    inv_s = 1.0 / ((a_iso * a_iso) ** (1.0 / 6.0))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = mic_super(H, H_inv, c_arr[i] - c_arr[j])
            r2 = float(np.dot(d, d))
            if r2 >= rc2 or r2 < 0.25:
                continue
            r = math.sqrt(r2)
            u_t = r * inv_s
            au3 = THOLE_A * (u_t ** 3)
            exp_t = math.exp(-au3) if au3 < 50.0 else 0.0
            fe = 1.0 - exp_t
            ft = 1.0 - (1.0 + au3) * exp_t
            Tij = KE_K * (3.0 * ft / (r2 * r2 * r) * np.outer(d, d) - fe / (r2 * r) * np.eye(3))
            T_blocks[(i, j)] = Tij
            M[3 * i:3 * i + 3, 3 * j:3 * j + 3] = -(A_blocks[i] @ Tij)

    mu_exact = np.linalg.solve(M, rhs).reshape(N, 3)
    u_pol_exact = -0.5 * float(np.sum(mu_exact * e0_py))

    # 3. Direct variational optimality check: U_var(mu) = 0.5 sum mu^T A^-1 mu - sum mu.E0 - 0.5 sum_{i!=j} mu_i^T T_ij mu_j
    u_var = 0.0
    for i in range(N):
        u_var += 0.5 * float(mu_exact[i].T @ A_inv_blocks[i] @ mu_exact[i]) - float(np.dot(mu_exact[i], e0_py[i]))
        for j in range(N):
            if (i, j) in T_blocks:
                u_var -= 0.5 * float(mu_exact[i].T @ T_blocks[(i, j)] @ mu_exact[j])

    max_mu_err = float(np.max(np.abs(mu_gpu - mu_exact)))
    u_gpu = float(data["u_pol_scf"])
    du_err = abs(u_gpu - u_pol_exact)
    var_err = abs(u_var - u_pol_exact)

    print(f"[Exact 36x36 Matrix Check]")
    print(f"  max|E0_GPU - E0_Py|       = {max_e0_err:.3e} K/(e*A)")
    print(f"  max|mu_GPU - mu_Inv|      = {max_mu_err:.3e} e*A")
    print(f"  |U_SCF_GPU - U_Inv|       = {du_err:.3e} K  (U_GPU = {u_gpu:.8f} K)")
    print(f"  |U_Var     - U_Inv|       = {var_err:.3e} K")

    assert max_mu_err < 1.0e-7, f"Dipole matrix mismatch: {max_mu_err}"
    assert du_err < 1.0e-6, f"SCF energy mismatch: {du_err}"
    assert var_err < 1.0e-9, f"Variational identity mismatch: {var_err}"


def run_cfg(cfg_rel):
    t0 = time.perf_counter()
    proc = subprocess.run([str(EXE), "--config", str(ROOT / cfg_rel)], cwd=str(ROOT), capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f"CLI failed for {cfg_rel}")
    print(f"\n=== {cfg_rel} ({dt:.2f} s) ===")
    for line in proc.stdout.strip().splitlines():
        print("  " + line)


def load_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def main():
    verify_exact_matrix_solve()

    if "--check-only" not in sys.argv:
        run_cfg("configs/mg_mof74_co2_298K_scf.json")
        run_cfg("configs/mg_mof74_fluegas_298K_scf.json")
        run_cfg("configs/mg_mof74_co2_298K_uff.json")

    scf_co2 = load_csv(BUILD / "cli_uff_298K_scf.csv")
    scf_flue = load_csv(BUILD / "cli_fluegas_298K_scf.csv")
    nonpol = load_csv(BUILD / "cli_uff_298K.csv")

    md_c = float(np.max(scf_co2["max_drift_K"]))
    db_c = float(np.max(np.abs(scf_co2["acc_ins"] - scf_co2["acc_del"])))
    md_f = float(np.max(scf_flue["max_drift_K"]))
    db_f0 = float(np.max(np.abs(scf_flue["acc_i0"] - scf_flue["acc_d0"])))
    db_f1 = float(np.max(np.abs(scf_flue["acc_i1"] - scf_flue["acc_d1"])))

    print(f"\n[SCF GCMC Stability]")
    print(f"  CO2 SCF  max_drift = {md_c:.3e} K, max|acc_ins - acc_del| = {db_c:.6f}")
    print(f"  Flue SCF max_drift = {md_f:.3e} K, max|acc_i - acc_d|     = {max(db_f0, db_f1):.6f}")
    assert md_c < 1.0e-6 and md_f < 1.0e-6
    assert db_c < 0.005 and db_f0 < 0.005 and db_f1 < 0.005

    print("\n[Summary: NonPol vs Pol-SCF (298 K, 0.10A Tricubic UFF)]")
    for i in range(len(nonpol)):
        p = float(nonpol["P_bar"][i])
        un = float(nonpol["uptake_mmol_g"][i])
        us = float(scf_co2["uptake_mmol_g"][i])
        upol = float(scf_co2["avg_Upol_kJ"][i])
        qst_s = float(scf_co2["Qst_kJ_mol"][i])
        print(f"  P={p:5.2f} bar | NonPol={un:7.4f} -> Pol-SCF={us:7.4f} mmol/g | Qst={qst_s:5.2f} kJ/mol | Upol={upol:6.3f} kJ/mol")
        if p <= 1.0:
            assert us > un
        else:
            assert us > 12.0

    for row in scf_flue:
        print(f"  Flue 15:85 P={float(row['P_bar']):4.2f} bar | CO2={float(row['uptake0_mmol_g']):6.4f}, N2={float(row['uptake1_mmol_g']):6.4f}, S(CO2/N2)={float(row['selectivity']):5.2f}")
        assert float(row["selectivity"]) > 35.0

    print("\nALL LEVEL-2 SCF MATRIX & GCMC ASSERTIONS PASSED!")


if __name__ == "__main__":
    main()
