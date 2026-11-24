"""Run native Windows RASPA3 Tricubic-Grid back-to-back benchmark on Mg-MOF-74 + CO2 @ 298 K."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RASPA3_ENV = ROOT / "build" / "tools" / "raspa3_env"
RASPA3_EXE = RASPA3_ENV / "bin" / "raspa3.exe"
BENCH_DIR = ROOT / "build" / "raspa3_grid_bench"
BENCH_DIR.mkdir(parents=True, exist_ok=True)


def make_ff_json():
    return {
        "MixingRule": "Lorentz-Berthelot",
        "TruncationMethod": "truncated",
        "TailCorrections": False,
        "CutOffVDW": 11.0,
        "UseInterpolationGrids": ["C_co2", "O_co2"],
        "SpacingVDWGrid": 0.15,
        "SpacingCoulombGrid": 0.15,
        "NumberOfGridTestPoints": 1000,
        "InterpolationScheme": 3,
        "PseudoAtoms": [
            {"name": "Mg1", "framework": True, "print_to_output": True, "element": "Mg", "print_as": "Mg", "mass": 24.305, "charge": 1.566},
            {"name": "O1",  "framework": True, "print_to_output": True, "element": "O",  "print_as": "O",  "mass": 15.999, "charge": -0.915},
            {"name": "O2",  "framework": True, "print_to_output": True, "element": "O",  "print_as": "O",  "mass": 15.999, "charge": -0.768},
            {"name": "O3",  "framework": True, "print_to_output": True, "element": "O",  "print_as": "O",  "mass": 15.999, "charge": -0.907},
            {"name": "C1",  "framework": True, "print_to_output": True, "element": "C",  "print_as": "C",  "mass": 12.011, "charge": 0.936},
            {"name": "C2",  "framework": True, "print_to_output": True, "element": "C",  "print_as": "C",  "mass": 12.011, "charge": -0.325},
            {"name": "C3",  "framework": True, "print_to_output": True, "element": "C",  "print_as": "C",  "mass": 12.011, "charge": 0.457},
            {"name": "C4",  "framework": True, "print_to_output": True, "element": "C",  "print_as": "C",  "mass": 12.011, "charge": -0.235},
            {"name": "H1",  "framework": True, "print_to_output": True, "element": "H",  "print_as": "H",  "mass": 1.008,  "charge": 0.191},
            {"name": "C_co2", "framework": False, "print_to_output": True, "element": "C", "print_as": "C", "mass": 12.011, "charge": 0.70},
            {"name": "O_co2", "framework": False, "print_to_output": True, "element": "O", "print_as": "O", "mass": 15.999, "charge": -0.35},
        ],
        "SelfInteractions": [
            {"name": "Mg1", "type": "lennard-jones", "parameters": [55.8574, 2.6914]},
            {"name": "O1",  "type": "lennard-jones", "parameters": [30.1932, 3.1181]},
            {"name": "O2",  "type": "lennard-jones", "parameters": [30.1932, 3.1181]},
            {"name": "O3",  "type": "lennard-jones", "parameters": [30.1932, 3.1181]},
            {"name": "C1",  "type": "lennard-jones", "parameters": [52.8381, 3.4309]},
            {"name": "C2",  "type": "lennard-jones", "parameters": [52.8381, 3.4309]},
            {"name": "C3",  "type": "lennard-jones", "parameters": [52.8381, 3.4309]},
            {"name": "C4",  "type": "lennard-jones", "parameters": [52.8381, 3.4309]},
            {"name": "H1",  "type": "lennard-jones", "parameters": [22.1417, 2.5711]},
            {"name": "C_co2", "type": "lennard-jones", "parameters": [27.0, 2.800]},
            {"name": "O_co2", "type": "lennard-jones", "parameters": [79.0, 3.050]},
        ],
    }


def make_co2_json():
    return {
        "CriticalTemperature": 304.1282,
        "CriticalPressure": 7377300.0,
        "AcentricFactor": 0.22394,
        "Type": "rigid",
        "pseudoAtoms": [
            ["O_co2", [0.0, 0.0, 1.16]],
            ["C_co2", [0.0, 0.0, 0.0]],
            ["O_co2", [0.0, 0.0, -1.16]],
        ],
    }


def make_sim_json(P_bar, eq_cycles=3000, prod_cycles=15000):
    return {
        "SimulationType": "MonteCarlo",
        "NumberOfCycles": prod_cycles,
        "NumberOfInitializationCycles": eq_cycles,
        "PrintEvery": 2000,
        "Systems": [
            {
                "Type": "Framework",
                "Name": "MgMOF-74",
                "NumberOfUnitCells": [1, 1, 4],
                "ChargeMethod": "Ewald",
                "ExternalTemperature": 298.0,
                "ExternalPressure": P_bar * 1.0e5,
            }
        ],
        "Components": [
            {
                "Name": "CO2",
                "IdealGasRosenbluthWeight": 1.0,
                "TranslationProbability": 0.5,
                "RotationProbability": 0.5,
                "ReinsertionProbability": 0.5,
                "SwapProbability": 1.0,
                "CreateNumberOfMolecules": 0,
            }
        ],
    }


def parse_raspa3_output(run_dir):
    out_files = sorted((run_dir / "output").glob("*.txt"))
    assert out_files, f"No output file in {run_dir / 'output'}"
    txt = out_files[0].read_text(encoding="utf-8", errors="replace")

    m_mmol = re.search(
        r"Abs\. loading average\s+([\d.eE+-]+)\s+\+/-\s+([\d.eE+-]+)\s+\[mol/kg-framework\]",
        txt,
    )
    m_molec = re.search(
        r"Abs\. loading average\s+([\d.eE+-]+)\s+\+/-\s+([\d.eE+-]+)\s+\[molecules/cell\]",
        txt,
    )
    m_qst = re.search(
        r"Enthalpy of adsorption:\s*[\d.eE+-]+\s+\+/-\s+[\d.eE+-]+\s+\[K\]\s*\n\s*([-+]?[\d.eE+-]+)\s+\+/-\s+([\d.eE+-]+)\s+\[kJ/mol\]",
        txt,
    )
    m_steps = re.search(r"Production count MC-steps:\s*(\d+)", txt)
    m_ptime = re.search(r"Production simulation time:\s*([\d.eE+-]+)\s*\[s\]", txt)

    uptake = float(m_mmol.group(1)) if m_mmol else float("nan")
    uptake_err = float(m_mmol.group(2)) if m_mmol else float("nan")
    molec = float(m_molec.group(1)) if m_molec else float("nan")
    qst = -float(m_qst.group(1)) if m_qst else float("nan")
    qst_err = float(m_qst.group(2)) if m_qst else float("nan")
    prod_mc_steps = int(m_steps.group(1)) if m_steps else 0
    prod_time_s = float(m_ptime.group(1)) if m_ptime else float("nan")
    return uptake, uptake_err, molec, qst, qst_err, prod_mc_steps, prod_time_s


def run_case(args):
    tag, p_bar, eq_cycles, prod_cycles, env = args
    d = BENCH_DIR / tag
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    shutil.copy2(ROOT / "data" / "MgMOF-74.cif", d / "MgMOF-74.cif")
    (d / "force_field.json").write_text(json.dumps(make_ff_json(), indent=2), encoding="utf-8")
    (d / "CO2.json").write_text(json.dumps(make_co2_json(), indent=2), encoding="utf-8")
    (d / "simulation.json").write_text(json.dumps(make_sim_json(p_bar, eq_cycles, prod_cycles), indent=2), encoding="utf-8")

    t0 = time.perf_counter()
    proc = subprocess.run([str(RASPA3_EXE)], cwd=str(d), env=env, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"[RASPA3-Grid {tag}] FAILED rc={proc.returncode}")
        print(proc.stdout[-800:])
        print(proc.stderr[-800:])
        raise RuntimeError(f"RASPA3-Grid failed for {tag} (rc={proc.returncode})")

    up, up_e, mol, qst, qst_e, prod_steps, prod_t = parse_raspa3_output(d)
    rate = (prod_steps / prod_t) if (prod_t and prod_t > 0.0) else 0.0
    print(f"[RASPA3-Grid {p_bar:4.2f} bar] uptake = {up:.4f} +/- {up_e:.4f} mmol/g (N={mol:.3f}), "
          f"Qst = {qst:.2f} +/- {qst_e:.2f} kJ/mol, prod={prod_steps} steps in {prod_t:.2f} s ({rate:.0f} steps/s), wall={dt:.2f} s")
    return (p_bar, up, up_e, mol, qst, qst_e, prod_steps, prod_t, dt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    env_root = str(RASPA3_ENV.resolve())
    env["PATH"] = f"{env_root}\\bin;{env_root}\\Library\\bin;" + env.get("PATH", "")

    if args.smoke:
        run_case(("smoke", 0.15, 100, 200, env))
        return

    cases = [
        ("p015", 0.15, 3000, 15000, env),
        ("p100", 1.00, 3000, 15000, env),
        ("p500", 5.00, 2000,  8000, env),
    ]
    with ThreadPoolExecutor(max_workers=3) as ex:
        rows = list(ex.map(run_case, cases))
    rows.sort(key=lambda r: r[0])

    csv_path = ROOT / "build" / "raspa3_uff_grid_bench.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("P_bar,uptake_mmol_g,uptake_err_mmol_g,N_mean,Qst_kJ_mol,Qst_err_kJ_mol,prod_mc_steps,prod_time_s,wall_s\n")
        for r in rows:
            f.write(f"{r[0]:.3f},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f},{r[4]:.4f},{r[5]:.4f},{r[6]},{r[7]:.3f},{r[8]:.3f}\n")
    print(f"Saved RASPA3 Tricubic-Grid results to {csv_path}")


if __name__ == "__main__":
    main()
