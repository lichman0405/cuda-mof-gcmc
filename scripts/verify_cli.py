"""End-to-end verification of the unified JSON/CLI GPU GCMC engine (mof_gcmc_cli.exe)."""

import math
import os
import subprocess
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "build" / ("mof_gcmc_cli.exe" if os.name == "nt" else "mof_gcmc_cli")
BUILD = ROOT / "build"


def load_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def run_cfg(cfg_rel):
    t0 = time.perf_counter()
    proc = subprocess.run([str(EXE), "--config", str(ROOT / cfg_rel)], cwd=str(ROOT),
                          capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f"CLI failed for {cfg_rel} (rc={proc.returncode})")
    print(f"=== {cfg_rel} ({dt:.2f} s) ===")
    for line in proc.stdout.strip().splitlines():
        print("  " + line)
    print()


def main():
    presets = [
        "configs/mg_mof74_co2_298K_uff.json",
        "configs/mg_mof74_co2_298K_mercado.json",
        "configs/mg_mof74_fluegas_298K_pol3site.json",
        "configs/mg_mof74_co2_313K_psa.json",
    ]
    for p in presets:
        run_cfg(p)

    cli_uff = load_csv(BUILD / "cli_uff_298K.csv")
    cli_merc = load_csv(BUILD / "cli_mercado_298K.csv")
    cli_flue = load_csv(BUILD / "cli_fluegas_298K_pol3site.csv")
    cli_313 = load_csv(BUILD / "cli_mercado_313K.csv")

    r3_grid = load_csv(BUILD / "raspa3_uff_grid_bench.csv")
    v2_merc = load_csv(BUILD / "v2_mercado2016_010A.csv")

    # 1. Zero-drift & detailed balance across all 4 CLI presets
    for name, arr in [("CLI UFF 298K", cli_uff), ("CLI Mercado 298K", cli_merc), ("CLI Mercado 313K", cli_313)]:
        md = float(np.max(arr["max_drift_K"]))
        db = float(np.max(np.abs(arr["acc_ins"] - arr["acc_del"])))
        assert md < 1.0e-4, f"{name} drift too high: {md}"
        assert db < 0.005, f"{name} detailed balance gap too high: {db}"

    md_f = float(np.max(cli_flue["max_drift_K"]))
    db_f0 = float(np.max(np.abs(cli_flue["acc_i0"] - cli_flue["acc_d0"])))
    db_f1 = float(np.max(np.abs(cli_flue["acc_i1"] - cli_flue["acc_d1"])))
    assert md_f < 1.0e-4, f"Flue gas drift too high: {md_f}"
    assert db_f0 < 0.005 and db_f1 < 0.005, f"Flue gas detailed balance failed: {db_f0}, {db_f1}"

    # 2. CLI UFF 298 K vs RASPA3 Tricubic Grid
    print("[Assert 1] CLI UFF 298K vs RASPA3 Tricubic Grid:")
    for i, r3 in enumerate(r3_grid):
        u_cli = float(cli_uff["uptake_mmol_g"][i])
        s_cli = float(cli_uff["uptake_sem_mmol_g"][i])
        u_r3 = float(r3["uptake_mmol_g"])
        s_r3 = float(r3["uptake_err_mmol_g"])
        rel = (u_cli - u_r3) / u_r3
        z = abs(u_cli - u_r3) / math.sqrt(s_cli**2 + s_r3**2)
        print(f"  P={float(r3['P_bar']):5.2f} bar: CLI={u_cli:.4f}+-{s_cli:.4f} vs RASPA3-Grid={u_r3:.4f}+-{s_r3:.4f} (diff={rel*100:+.2f}%, {z:.2f} sigma)")
        assert abs(rel) <= 0.04 and z <= 2.0

    # 3. CLI Mercado 298 K vs v2_mercado2016_010A.csv
    print("[Assert 2] CLI Mercado 298K vs v2_mercado2016_010A:")
    for row in cli_merc:
        p = float(row["P_bar"])
        idx = int(np.argmin(np.abs(v2_merc["P_bar"] - p)))
        u_cli = float(row["uptake_mmol_g"])
        u_v2 = float(v2_merc["uptake_mmol_g"][idx])
        rel = (u_cli - u_v2) / u_v2
        print(f"  P={p:5.2f} bar: CLI={u_cli:.4f} vs v2={u_v2:.4f} (diff={rel*100:+.2f}%)")
        assert abs(rel) <= 0.03

    # 4. Clausius-Clapeyron thermodynamic consistency: T=313 K < T=298 K
    print("[Assert 3] Thermodynamic T-dependence (313 K < 298 K):")
    for i in range(len(cli_merc)):
        p = float(cli_merc["P_bar"][i])
        u298 = float(cli_merc["uptake_mmol_g"][i])
        u313 = float(cli_313["uptake_mmol_g"][i])
        print(f"  P={p:5.2f} bar: 298 K = {u298:.4f} -> 313 K = {u313:.4f} mmol/g (delta={u313-u298:+.4f})")
        assert u313 < u298

    # 5. Flue gas Pol-3Site 0.10A Tricubic selectivity
    idx1 = int(np.argmin(np.abs(cli_flue["P_bar"] - 1.0)))
    sel1 = float(cli_flue["selectivity"][idx1])
    print(f"[Assert 4] Flue gas (15:85 CO2/N2, 1.0 bar, 0.10A Tricubic Pol-3Site): S(CO2/N2) = {sel1:.2f}, max_drift = {md_f:.2e} K")
    assert sel1 > 45.0

    print("\nALL UNIFIED CLI ASSERTIONS PASSED!")


if __name__ == "__main__":
    main()
