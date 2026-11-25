"""Phase 2 Controlled Benchmark & Cross-MOF Generalization Orchestrator.

Implements the benchmark matrix reported in docs/TECHNICAL_REPORT.md with strict convergence gates:
- Exp 1 (Mg-MOF-74 + CO2, 298 K, 0.012 / 0.28 / 1.0 bar, No-D3 & +D3):
  1. auto_mlip (1B+2B + 128-thread GPU-CBMC Nested MC), including empty & high_loading check
  2. Direct MLIP-GCMC baseline (ESS/GPU-s & MCSE wall-time comparison)
  3. UFF+QEq Nested MC and Mercado Nested MC baselines
  4. Unreweighted auto_mlip --grid-only ablation
- Exp 2 (Cross-metal / cross-topology / cross-guest zero-parameter generalization):
  NiMOF-74 + CO2, Cu-BTC + CO2, IRMOF-1 + CO2 (r2SCAN), IRMOF-1 + CH4 (MP0+D3).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from direct_mlip_gcmc import run_direct_gcmc
from guests import load_guest
from mcmc_diagnostics import evaluate_v2_convergence
from multi_gpu import parse_gpu_ids
from nested_mc_engine import FastMaceD3Target
from qeq_charges import extract_framework_for_proposer


def evaluate_phase2_hard_gate(pt: dict) -> dict:
    """Apply strict report statistical convergence thresholds to a pressure point."""
    v2 = pt.get("diagnostics") or {}
    n_chains = int(pt.get("n_boxes", 0))
    ns = v2.get("N_stats") or {}
    us = v2.get("U_stats") or {}
    omss = v2.get("OMS_stats") or {}

    rhat_n = ns.get("rhat_max")
    rhat_u = us.get("rhat_max")
    rhat_oms = omss.get("rhat_max") if omss else None
    ess_bulk_n = float(ns.get("ess_bulk", 0.0) or 0.0)
    ess_tail_n = float(ns.get("ess_tail", 0.0) or 0.0)
    uptake = float(pt.get("uptake_mmol_g", 0.0) or 0.0)
    sem = float(pt.get("uptake_sem_mmol_g", 0.0) or 0.0)

    mcse_limit = max(0.05, 0.01 * abs(uptake))
    reasons = []
    if n_chains < 4:
        reasons.append(f"n_chains={n_chains} < 4")
    if rhat_n is None or float(rhat_n) >= 1.01:
        reasons.append(f"rhat_N={rhat_n} >= 1.01")
    if rhat_u is None or float(rhat_u) >= 1.01:
        reasons.append(f"rhat_U={rhat_u} >= 1.01")
    if rhat_oms is not None and float(rhat_oms) >= 1.01:
        reasons.append(f"rhat_OMS={rhat_oms} >= 1.01")
    if ess_bulk_n < 100.0 * max(1, n_chains):
        reasons.append(f"ess_bulk_N={ess_bulk_n:.1f} < {100 * max(1, n_chains)}")
    if ess_tail_n < 50.0 * max(1, n_chains):
        reasons.append(f"ess_tail_N={ess_tail_n:.1f} < {50 * max(1, n_chains)}")
    if sem > mcse_limit:
        reasons.append(f"MCSE={sem:.4f} > limit {mcse_limit:.4f}")

    return {
        "hard_gate_pass": len(reasons) == 0,
        "reasons": reasons,
        "uptake_mmol_g": uptake,
        "uptake_sem_mmol_g": sem,
        "mcse_limit_mmol_g": mcse_limit,
        "rhat_N": rhat_n,
        "rhat_U": rhat_u,
        "rhat_OMS": rhat_oms,
        "ess_bulk_N": ess_bulk_n,
        "ess_tail_N": ess_tail_n,
        "ess_N_per_s": pt.get("ess_N_per_s"),
        "Qst_kJ_mol": pt.get("Qst_kJ_mol"),
        "Qst_sem_kJ_mol": pt.get("Qst_sem_kJ_mol"),
        "qst_status": pt.get("qst_status"),
    }


def _run_direct_worker(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    fw_data = extract_framework_for_proposer(args.cif, charge_mode="auto")
    guest = load_guest(args.guest)
    use_disp = args.dispersion == "on"
    t0 = time.perf_counter()
    target = FastMaceD3Target(
        cif_path=args.cif,
        supercell=tuple(args.supercell),
        model=args.model,
        device="cuda:0",
        dispersion=use_disp,
        dispersion_xc=args.xc,
        guest=guest,
        dtype=args.dtype,
        uc_atoms=fw_data["uc"],
    )
    model_load_s = float(time.perf_counter() - t0)
    conv_mmol_g = 1000.0 / float(np.sum(target.fw.get_masses()))
    pts = []
    for idx, p_bar in enumerate(args.pressures):
        res = run_direct_gcmc(
            uc_atoms=fw_data["uc"],
            guest=guest,
            supercell=tuple(args.supercell),
            T_K=args.temp,
            pressure_bar=float(p_bar),
            batch_target_fn=target.eval_batch,
            n_boxes=args.boxes,
            eq_steps=args.direct_eq_steps,
            prod_steps=args.direct_prod_steps,
            seed=args.seed + idx * 100,
            max_wall_s=args.direct_max_wall_s,
            log_every=max(25, (args.direct_eq_steps + args.direct_prod_steps) // 6),
            fw_mass_gmol=float(np.sum(target.fw.get_masses())),
        )
        res["P_bar"] = float(p_bar)
        res["T_K"] = float(args.temp)
        res["diagnostics"] = evaluate_v2_convergence(
            prod_N_history=res["prod_N_history"],
            prod_U_history_kJ=res["prod_U_history"],
            conv_mmol_g=conv_mmol_g,
            T_K=float(args.temp),
            prod_ins_acc=res["prod_ins_acc"],
            prod_del_acc=res["prod_del_acc"],
            prod_ins_att=res["prod_ins_att"],
            prod_del_att=res["prod_del_att"],
            prod_outer_acc_rate=res["prod_move_acc_rate"],
            uptake_sem_mmol_g=res["uptake_sem_mmol_g"],
        )
        pts.append(res)

    out_p = Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(
        json.dumps(
            {
                "method": "direct_mlip_gcmc",
                "cif": str(args.cif),
                "supercell": list(args.supercell),
                "guest": args.guest,
                "T_K": float(args.temp),
                "model": args.model,
                "dispersion": use_disp,
                "dispersion_xc": args.xc,
                "model_load_wall_s": model_load_s,
                "points": pts,
            },
            indent=2,
            default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o),
        ),
        encoding="utf-8",
    )
    print(f"[Direct worker DONE] saved {out_p}")


def _build_job_specs(args) -> List[dict]:
    mode = args.mode
    k_steps = int(args.k_steps)
    if mode == "smoke":
        boxes, pilot, eq_c, prod_c = 4, 30, 60, 120
        dir_eq, dir_pr, dir_wall = 20, 40, 60.0
    elif mode == "probe":
        boxes, pilot, eq_c, prod_c = 4, 40, 80, 200
        dir_eq, dir_pr, dir_wall = 40, 120, 180.0
    else:
        boxes, pilot, eq_c, prod_c = 4, 40, 250, 4800
        dir_eq, dir_pr, dir_wall = 60, 400, 300.0

    out_dir = Path(args.out_dir)
    ckpt_root = out_dir / "checkpoints"
    jobs: List[dict] = []

    model_cfgs = [
        ("r2scan_nod3", args.model, "off", ""),
        ("mp0_d3", "medium", "on", "pbe"),
    ]

    if args.suite in ("exp1_mgmof74", "all"):
        press_str = ["0.28"] if mode == "smoke" else ["0.012", "0.28", "1.0"]
        for m_tag, m_name, disp_flag, xc_name in model_cfgs:
            xc_args = ["--xc", xc_name] if xc_name else []
            for prop_name in ("auto_mlip", "uff", "mercado"):
                jname = f"exp1_{prop_name}_{m_tag}"
                jobs.append({
                    "name": jname,
                    "kind": "nested",
                    "out_json": str(out_dir / f"{jname}.json"),
                    "cmd": [
                        sys.executable, "scripts/run_mlip_gcmc.py",
                        "--cif", "data/MgMOF-74.cif", "--supercell", "1", "1", "2",
                        "--guest", "co2", "--temp", "298", "--pressures", *press_str,
                        "--proposer", prop_name, "--cbmc-k", "128",
                        "--model", m_name, "--dispersion", disp_flag, *xc_args,
                        "--boxes", str(boxes), "--k-steps", str(k_steps),
                        "--pilot-cycles", str(pilot), "--eq-cycles", str(eq_c), "--prod-cycles", str(prod_c),
                        "--distill-cache-dir", args.distill_cache_dir,
                        "--checkpoint-dir", str(ckpt_root / jname),
                        "--checkpoint-every", "100",
                        "--resume-mode", args.resume_mode,
                        "--out-json", str(out_dir / f"{jname}.json"),
                    ],
                })
            # Unreweighted surrogate ablation (grid_only)
            jname_go = f"exp1_grid_only_{m_tag}"
            jobs.append({
                "name": jname_go,
                "kind": "nested",
                "out_json": str(out_dir / f"{jname_go}.json"),
                "cmd": [
                    sys.executable, "scripts/run_mlip_gcmc.py",
                    "--cif", "data/MgMOF-74.cif", "--supercell", "1", "1", "2",
                    "--guest", "co2", "--temp", "298", "--pressures", *press_str,
                    "--proposer", "auto_mlip", "--grid-only", "--cbmc-k", "128",
                    "--model", m_name, "--dispersion", disp_flag, *xc_args,
                    "--boxes", str(boxes), "--k-steps", str(k_steps),
                    "--eq-cycles", str(eq_c), "--prod-cycles", str(prod_c),
                    "--distill-cache-dir", args.distill_cache_dir,
                    "--resume-mode", args.resume_mode,
                    "--out-json", str(out_dir / f"{jname_go}.json"),
                ],
            })
            if mode != "smoke":
                jname_hl = f"exp1_auto_mlip_{m_tag}_highload_0p28"
                jobs.append({
                    "name": jname_hl,
                    "kind": "nested",
                    "out_json": str(out_dir / f"{jname_hl}.json"),
                    "cmd": [
                        sys.executable, "scripts/run_mlip_gcmc.py",
                        "--cif", "data/MgMOF-74.cif", "--supercell", "1", "1", "2",
                        "--guest", "co2", "--temp", "298", "--pressures", "0.28",
                        "--proposer", "auto_mlip", "--cbmc-k", "128",
                        "--init-mode", "high_loading", "--init-loading", "8.5",
                        "--model", m_name, "--dispersion", disp_flag, *xc_args,
                        "--boxes", str(boxes), "--k-steps", str(k_steps),
                        "--pilot-cycles", str(pilot), "--eq-cycles", str(eq_c), "--prod-cycles", str(prod_c),
                        "--distill-cache-dir", args.distill_cache_dir,
                        "--checkpoint-dir", str(ckpt_root / jname_hl),
                        "--checkpoint-every", "100",
                        "--resume-mode", args.resume_mode,
                        "--out-json", str(out_dir / f"{jname_hl}.json"),
                    ],
                })
            # Direct MLIP-GCMC comparison at 0.28 bar
            jname_dir = f"exp1_direct_{m_tag}_0p28"
            jobs.append({
                "name": jname_dir,
                "kind": "direct",
                "out_json": str(out_dir / f"{jname_dir}.json"),
                "cmd": [
                    sys.executable, "scripts/run_phase2_matrix.py", "--direct-worker",
                    "--cif", "data/MgMOF-74.cif", "--supercell", "1", "1", "2",
                    "--guest", "co2", "--temp", "298", "--pressures", "0.28",
                    "--model", m_name, "--dispersion", disp_flag, *xc_args,
                    "--boxes", str(boxes), "--direct-eq-steps", str(dir_eq),
                    "--direct-prod-steps", str(dir_pr), "--direct-max-wall-s", str(dir_wall),
                    "--out-json", str(out_dir / f"{jname_dir}.json"),
                ],
            })

    if args.suite in ("exp2_generalization", "all"):
        gen_cases = [
            ("nimof74_co2", "data/NiMOF-74.cif", ["1", "1", "2"], "co2", ["0.012", "0.28", "1.0"], args.model, "off", ""),
            ("cubtc_co2", "data/Cu-BTC.cif", ["1", "1", "1"], "co2", ["0.15", "1.0"], args.model, "off", ""),
            ("irmof1_co2", "data/IRMOF-1.cif", ["1", "1", "1"], "co2", ["1.0", "5.0"], args.model, "off", ""),
            ("irmof1_ch4", "data/IRMOF-1.cif", ["1", "1", "1"], "ch4", ["1.0", "5.0"], "medium", "on", "pbe"),
        ]
        for tag, cif, sc, gname, p_list, m_name, disp_flag, xc_name in gen_cases:
            p_run = [p_list[0]] if mode == "smoke" else p_list
            xc_args = ["--xc", xc_name] if xc_name else []
            jname = f"exp2_{tag}"
            jobs.append({
                "name": jname,
                "kind": "nested",
                "out_json": str(out_dir / f"{jname}.json"),
                "cmd": [
                    sys.executable, "scripts/run_mlip_gcmc.py",
                    "--cif", cif, "--supercell", *sc,
                    "--guest", gname, "--temp", "298", "--pressures", *p_run,
                    "--proposer", "auto_mlip", "--cbmc-k", "128",
                    "--model", m_name, "--dispersion", disp_flag, *xc_args,
                    "--boxes", str(boxes), "--k-steps", str(k_steps),
                    "--pilot-cycles", str(pilot), "--eq-cycles", str(eq_c), "--prod-cycles", str(prod_c),
                    "--distill-cache-dir", args.distill_cache_dir,
                    "--checkpoint-dir", str(ckpt_root / jname),
                    "--checkpoint-every", "100",
                    "--resume-mode", args.resume_mode,
                    "--out-json", str(out_dir / f"{jname}.json"),
                ],
            })
    return jobs


def _run_job_on_gpu(job: dict, gpu_id: int) -> dict:
    cmd = list(job["cmd"])
    if job["kind"] == "direct":
        cmd.extend(["--gpu", str(gpu_id)])
    else:
        cmd.extend(["--gpus", str(gpu_id)])
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, check=False)
    wall = float(time.perf_counter() - t0)
    out_p = Path(job["out_json"])
    gates = []
    if proc.returncode == 0 and out_p.exists():
        data = json.loads(out_p.read_text(encoding="utf-8"))
        for pt in data.get("points", []):
            g = evaluate_phase2_hard_gate(pt)
            g["P_bar"] = pt.get("P_bar")
            gates.append(g)
    return {
        "name": job["name"],
        "gpu_id": int(gpu_id),
        "returncode": int(proc.returncode),
        "wall_s": wall,
        "out_json": job["out_json"],
        "gates": gates,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 Multi-GPU Benchmark & Generalization Runner")
    p.add_argument("--suite", choices=["exp1_mgmof74", "exp2_generalization", "all"], default="all")
    p.add_argument("--mode", choices=["smoke", "probe", "full"], default="probe")
    p.add_argument("--gpus", nargs="+", default=["0"], help="Confirmed idle GPU IDs (only among 0..3)")
    p.add_argument("--k-steps", type=int, default=16, help="Inner GPU steps per outer Nested MC cycle")
    p.add_argument("--model", default="mace-matpes-r2scan-0")
    p.add_argument("--xc", default="pbe")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--distill-cache-dir", default="build/distill_cache")
    p.add_argument("--out-dir", default="build/phase2_runs")
    p.add_argument("--resume-mode", choices=["none", "skip", "extend", "fresh"], default="extend")
    p.add_argument("--dry-run", action="store_true", help="Print planned jobs and exit without running")

    # Internal direct-worker mode arguments
    p.add_argument("--direct-worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--cif", default="data/MgMOF-74.cif")
    p.add_argument("--supercell", nargs=3, type=int, default=[1, 1, 2])
    p.add_argument("--guest", default="co2")
    p.add_argument("--temp", type=float, default=298.0)
    p.add_argument("--pressures", nargs="+", type=float, default=[0.28])
    p.add_argument("--dispersion", choices=["on", "off"], default="on")
    p.add_argument("--boxes", type=int, default=4)
    p.add_argument("--seed", type=int, default=20262000)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--direct-eq-steps", type=int, default=40)
    p.add_argument("--direct-prod-steps", type=int, default=120)
    p.add_argument("--direct-max-wall-s", type=float, default=180.0)
    p.add_argument("--out-json", default="build/direct_out.json")
    return p.parse_args()


def main():
    args = parse_args()
    if args.direct_worker:
        _run_direct_worker(args)
        return

    gpu_ids = parse_gpu_ids(args.gpus)
    jobs = _build_job_specs(args)
    if args.dry_run:
        print(json.dumps({"gpus": gpu_ids, "n_jobs": len(jobs), "jobs": jobs}, indent=2))
        return

    import queue
    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)

    results = []
    def _worker(gid: int):
        local_res = []
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                break
            print(f"[Phase2 GPU {gid}] Starting {job['name']} ...")
            r = _run_job_on_gpu(job, gid)
            print(f"[Phase2 GPU {gid}] Finished {job['name']} rc={r['returncode']} ({r['wall_s']:.1f}s)")
            local_res.append(r)
            q.task_done()
        return local_res

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as ex:
        futs = [ex.submit(_worker, gid) for gid in gpu_ids]
        for f in as_completed(futs):
            results.extend(f.result())

    summary_path = Path(args.out_dir) / "phase2_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print(f"[Phase2 Matrix Complete] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
