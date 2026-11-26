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
from run_metadata import sha256_file

OMS_ATOMIC_NUMBERS = {12, 13, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30}

SUITE_CHOICES = [
    "exp1_mgmof74",
    "exp2_generalization",
    "all",
    "jctc_parity",
    "jctc_exp1_ablation",
    "jctc_remc_convergence",
    "jctc_finitesize",
    "jctc_mace_dac",
    "jctc_v2",
]


def _extract_oms_metal_positions(fw_atoms) -> np.ndarray:
    z_arr = np.asarray(fw_atoms.get_atomic_numbers(), dtype=np.int32)
    pos = np.asarray(fw_atoms.get_positions(), dtype=np.float64)
    mask = np.isin(z_arr, list(OMS_ATOMIC_NUMBERS))
    return pos[mask]


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
    fw_mass_gmol = float(np.sum(target.fw.get_masses()))
    conv_mmol_g = 1000.0 / fw_mass_gmol
    oms_metal_pos = _extract_oms_metal_positions(target.fw)
    oms_cell_rows = np.asarray(target.fw.get_cell(), dtype=np.float64)
    model_file_path = getattr(target, "model_file_path", None)
    model_sha256 = sha256_file(model_file_path) if model_file_path else None
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
            max_mol=int(args.max_mol),
            seed=args.seed + idx * 100,
            max_wall_s=args.direct_max_wall_s,
            log_every=max(25, (args.direct_eq_steps + args.direct_prod_steps) // 6),
            fw_mass_gmol=fw_mass_gmol,
            match_distilled_hard_core=bool(args.match_distilled_hard_core),
            oms_metal_pos=oms_metal_pos if len(oms_metal_pos) > 0 else None,
            oms_cell_rows=oms_cell_rows,
        )
        res["P_bar"] = float(p_bar)
        res["T_K"] = float(args.temp)
        res["gpu_id"] = int(args.gpu)
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
            oms_occ_history=res.get("prod_oms_occ_history") or None,
        )
        pts.append(res)

    out_p = Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(
        json.dumps(
            {
                "method": "direct_mlip_gcmc",
                "proposer": "direct",
                "cif": str(args.cif),
                "supercell": list(args.supercell),
                "guest": args.guest,
                "T_K": float(args.temp),
                "model": args.model,
                "model_file_path": model_file_path,
                "model_sha256": model_sha256,
                "dtype": args.dtype,
                "dispersion": use_disp,
                "dispersion_xc": args.xc,
                "max_mol": int(args.max_mol),
                "match_distilled_hard_core": bool(args.match_distilled_hard_core),
                "gpus": [int(args.gpu)],
                "parallel_mode": "direct",
                "model_load_wall_s": model_load_s,
                "metadata": {
                    "cif_path": str(args.cif),
                    "cif_sha256": sha256_file(args.cif),
                    "model": args.model,
                    "model_file_path": model_file_path,
                    "model_sha256": model_sha256,
                    "max_mol": int(args.max_mol),
                    "match_distilled_hard_core": bool(args.match_distilled_hard_core),
                },
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

    if args.suite in (
        "jctc_parity",
        "jctc_exp1_ablation",
        "jctc_remc_convergence",
        "jctc_finitesize",
        "jctc_mace_dac",
        "jctc_v2",
    ):
        jobs.extend(_build_jctc_v2_jobs(args))
    return jobs


def _build_jctc_v2_jobs(args) -> List[dict]:
    """Build JCTC v2 publication benchmark sub-suites."""
    mode = args.mode
    k_steps = int(args.k_steps)
    base_max_mol = int(args.max_mol)
    remc_swap_every = int(args.remc_swap_every)
    if mode == "smoke":
        boxes, pilot, eq_c, prod_c = 4, 20, 40, 80
        dir_eq, dir_pr, dir_wall = 20, 40, 60.0
        parity_dir_eq, parity_dir_pr, parity_dir_wall = 20, 60, 120.0
    elif mode == "probe":
        boxes, pilot, eq_c, prod_c = 4, 40, 100, 300
        dir_eq, dir_pr, dir_wall = 40, 120, 180.0
        parity_dir_eq, parity_dir_pr, parity_dir_wall = 80, 300, 600.0
    else:
        boxes, pilot, eq_c, prod_c = 4, 40, 400, 4800
        dir_eq, dir_pr, dir_wall = 60, 400, 600.0
        parity_dir_eq, parity_dir_pr, parity_dir_wall = 400, 4000, 3600.0

    out_dir = Path(args.out_dir)
    ckpt_root = out_dir / "checkpoints"
    jobs: List[dict] = []

    def _add_nested(
        jname: str,
        suite_tag: str,
        cif: str,
        sc: List[str],
        guest: str,
        pressures: List[str],
        model: str,
        dispersion: str,
        xc: str = "",
        proposer: str = "auto_mlip",
        parallel_mode: Optional[str] = None,
        adapt_shift: str = "on",
        adapt_k: str = "on",
        distill_1b_only: bool = False,
        grid_only: bool = False,
        init_mode: Optional[str] = None,
        init_loading: Optional[float] = None,
        max_mol: Optional[int] = None,
    ):
        xc_args = ["--xc", xc] if xc else []
        cmd = [
            sys.executable, "scripts/run_mlip_gcmc.py",
            "--cif", cif, "--supercell", *sc,
            "--guest", guest, "--temp", "298", "--pressures", *pressures,
            "--proposer", proposer, "--cbmc-k", "128",
            "--model", model, "--dtype", args.dtype,
            "--dispersion", dispersion, *xc_args,
            "--boxes", str(boxes), "--k-steps", str(k_steps),
            "--max-mol", str(base_max_mol if max_mol is None else int(max_mol)),
            "--eq-cycles", str(eq_c), "--prod-cycles", str(prod_c),
            "--distill-cache-dir", args.distill_cache_dir,
            "--resume-mode", args.resume_mode,
            "--out-json", str(out_dir / f"{jname}.json"),
        ]
        if grid_only:
            cmd.append("--grid-only")
        else:
            cmd.extend([
                "--pilot-cycles", str(pilot),
                "--adapt-shift", adapt_shift,
                "--adapt-k", adapt_k,
                "--checkpoint-dir", str(ckpt_root / jname),
                "--checkpoint-every", "100",
            ])
        if distill_1b_only:
            cmd.append("--distill-1b-only")
        if parallel_mode is not None:
            cmd.extend(["--parallel-mode", parallel_mode])
            if parallel_mode == "remc":
                cmd.extend(["--remc-swap-every", str(remc_swap_every)])
        if init_mode is not None:
            cmd.extend(["--init-mode", init_mode])
        if init_loading is not None:
            cmd.extend(["--init-loading", str(init_loading)])
        jobs.append({
            "name": jname,
            "suite": suite_tag,
            "kind": "nested",
            "out_json": str(out_dir / f"{jname}.json"),
            "cmd": cmd,
        })

    def _add_direct(
        jname: str,
        suite_tag: str,
        cif: str,
        sc: List[str],
        guest: str,
        pressures: List[str],
        model: str,
        dispersion: str,
        xc: str,
        eq_steps: int,
        prod_steps: int,
        max_wall_s: float,
        match_distilled_hard_core: bool = True,
        max_mol: Optional[int] = None,
    ):
        xc_args = ["--xc", xc] if xc else []
        cmd = [
            sys.executable, "scripts/run_phase2_matrix.py", "--direct-worker",
            "--cif", cif, "--supercell", *sc,
            "--guest", guest, "--temp", "298", "--pressures", *pressures,
            "--model", model, "--dtype", args.dtype,
            "--dispersion", dispersion, *xc_args,
            "--boxes", str(boxes),
            "--max-mol", str(base_max_mol if max_mol is None else int(max_mol)),
            "--direct-eq-steps", str(eq_steps),
            "--direct-prod-steps", str(prod_steps),
            "--direct-max-wall-s", str(max_wall_s),
            "--out-json", str(out_dir / f"{jname}.json"),
        ]
        if match_distilled_hard_core:
            cmd.append("--match-distilled-hard-core")
        jobs.append({
            "name": jname,
            "suite": suite_tag,
            "kind": "direct",
            "out_json": str(out_dir / f"{jname}.json"),
            "cmd": cmd,
        })

    # 1. Exact Parity Test: Direct MLIP-GCMC (unified hard-core) vs Auto-Distilled Nested MC
    if args.suite in ("jctc_parity", "jctc_v2"):
        _add_direct(
            jname="jctc_parity_direct_irmof1_co2_1bar",
            suite_tag="jctc_parity",
            cif="data/IRMOF-1.cif", sc=["1", "1", "1"], guest="co2", pressures=["1.0"],
            model=args.model, dispersion="off", xc="",
            eq_steps=parity_dir_eq, prod_steps=parity_dir_pr, max_wall_s=parity_dir_wall,
            match_distilled_hard_core=True,
        )
        _add_nested(
            jname="jctc_parity_nested_irmof1_co2_1bar",
            suite_tag="jctc_parity",
            cif="data/IRMOF-1.cif", sc=["1", "1", "1"], guest="co2", pressures=["1.0"],
            model=args.model, dispersion="off", proposer="auto_mlip",
            adapt_shift="on", adapt_k="on", init_mode="empty",
        )

    # 2. Experiment 1 Ablation Matrix on Mg-MOF-74 + CO2 (r2SCAN, 298 K)
    if args.suite in ("jctc_exp1_ablation", "jctc_v2"):
        p_ablation = ["0.012", "0.28"] if mode == "smoke" else ["0.012", "0.28", "1.0"]
        # 2a. auto_mlip (1B+2B) with REMC (empty start)
        _add_nested(
            jname="jctc_exp1_auto_mlip_r2scan_remc",
            suite_tag="jctc_exp1_ablation",
            cif="data/MgMOF-74.cif", sc=["1", "1", "2"], guest="co2", pressures=p_ablation,
            model=args.model, dispersion="off", proposer="auto_mlip",
            parallel_mode="remc", adapt_shift="on", adapt_k="on", init_mode="empty",
        )
        # 2b. auto_mlip (1B+2B) high-loading start check at 0.28 bar
        _add_nested(
            jname="jctc_exp1_auto_mlip_r2scan_highload_0p28",
            suite_tag="jctc_exp1_ablation",
            cif="data/MgMOF-74.cif", sc=["1", "1", "2"], guest="co2", pressures=["0.28"],
            model=args.model, dispersion="off", proposer="auto_mlip",
            adapt_shift="on", adapt_k="on", init_mode="high_loading", init_loading=8.5,
        )
        # 2c. auto_mlip (1B+2B) without REMC (--parallel-mode pressure)
        _add_nested(
            jname="jctc_exp1_auto_mlip_r2scan_no_remc",
            suite_tag="jctc_exp1_ablation",
            cif="data/MgMOF-74.cif", sc=["1", "1", "2"], guest="co2", pressures=p_ablation,
            model=args.model, dispersion="off", proposer="auto_mlip",
            parallel_mode="pressure", adapt_shift="on", adapt_k="on", init_mode="empty",
        )
        # 2d. auto_mlip 1B-only ablation (--distill-1b-only)
        _add_nested(
            jname="jctc_exp1_auto_mlip_1b_only_r2scan",
            suite_tag="jctc_exp1_ablation",
            cif="data/MgMOF-74.cif", sc=["1", "1", "2"], guest="co2", pressures=p_ablation,
            model=args.model, dispersion="off", proposer="auto_mlip",
            parallel_mode="pressure", adapt_shift="on", adapt_k="on",
            distill_1b_only=True, init_mode="empty",
        )
        # 2e. Classical proposers: UFF and Mercado (with adaptive shift & k)
        _add_nested(
            jname="jctc_exp1_uff_r2scan",
            suite_tag="jctc_exp1_ablation",
            cif="data/MgMOF-74.cif", sc=["1", "1", "2"], guest="co2", pressures=p_ablation,
            model=args.model, dispersion="off", proposer="uff",
            parallel_mode="pressure", adapt_shift="on", adapt_k="on",
        )
        _add_nested(
            jname="jctc_exp1_mercado_r2scan",
            suite_tag="jctc_exp1_ablation",
            cif="data/MgMOF-74.cif", sc=["1", "1", "2"], guest="co2", pressures=p_ablation,
            model=args.model, dispersion="off", proposer="mercado",
            parallel_mode="pressure", adapt_shift="on", adapt_k="on",
        )
        # 2f. Unreweighted surrogate ablation (grid_only)
        _add_nested(
            jname="jctc_exp1_grid_only_r2scan",
            suite_tag="jctc_exp1_ablation",
            cif="data/MgMOF-74.cif", sc=["1", "1", "2"], guest="co2", pressures=p_ablation,
            model=args.model, dispersion="off", proposer="auto_mlip",
            grid_only=True,
        )
        # 2g. Direct MLIP-GCMC baseline at 0.28 bar (unified hard-core)
        _add_direct(
            jname="jctc_exp1_direct_r2scan_0p28",
            suite_tag="jctc_exp1_ablation",
            cif="data/MgMOF-74.cif", sc=["1", "1", "2"], guest="co2", pressures=["0.28"],
            model=args.model, dispersion="off", xc="",
            eq_steps=dir_eq, prod_steps=dir_pr, max_wall_s=dir_wall,
            match_distilled_hard_core=True,
        )

    # 3. REMC Convergence & Zero-Manual-Override Cross-MOF Suite
    if args.suite in ("jctc_remc_convergence", "jctc_v2"):
        remc_cases = [
            ("mgmof74_co2_mp0_d3", "data/MgMOF-74.cif", ["1", "1", "2"], "co2", ["0.012", "0.28", "1.0"], "medium", "on", "pbe", max(160, base_max_mol)),
            ("irmof1_co2_r2scan", "data/IRMOF-1.cif", ["1", "1", "1"], "co2", ["1.0", "5.0"], args.model, "off", "", base_max_mol),
            ("nimof74_co2_r2scan", "data/NiMOF-74.cif", ["1", "1", "2"], "co2", ["0.012", "0.28", "1.0"], args.model, "off", "", base_max_mol),
            ("cubtc_co2_r2scan", "data/Cu-BTC.cif", ["1", "1", "1"], "co2", ["0.15", "1.0"], args.model, "off", "", base_max_mol),
            ("irmof1_ch4_mp0_d3", "data/IRMOF-1.cif", ["1", "1", "1"], "ch4", ["1.0", "5.0"], "medium", "on", "pbe", base_max_mol),
        ]
        for tag, cif, sc, gname, p_list, m_name, disp_flag, xc_name, cap in remc_cases:
            p_run = p_list[:2] if mode == "smoke" else p_list
            _add_nested(
                jname=f"jctc_remc_{tag}",
                suite_tag="jctc_remc_convergence",
                cif=cif, sc=sc, guest=gname, pressures=p_run,
                model=m_name, dispersion=disp_flag, xc=xc_name,
                proposer="auto_mlip", parallel_mode="remc",
                adapt_shift="on", adapt_k="on", max_mol=cap,
            )

    # 4. Independent Supercell Finite-Size Check (1x1x4 Mg-MOF-74 + CO2 @ 0.28 bar)
    if args.suite in ("jctc_finitesize", "jctc_v2"):
        _add_nested(
            jname="jctc_finitesize_mgmof74_1x1x4_co2_r2scan_0p28",
            suite_tag="jctc_finitesize",
            cif="data/MgMOF-74.cif", sc=["1", "1", "4"], guest="co2", pressures=["0.28"],
            model=args.model, dispersion="off", proposer="auto_mlip",
            adapt_shift="on", adapt_k="on", max_mol=max(256, base_max_mol),
        )

    # 5. Fine-Tuned MACE-DAC-1 Transferability & Optional H2O Suite
    if args.suite in ("jctc_mace_dac", "jctc_v2"):
        dac_model = str(args.mace_dac_model or "").strip()
        if not dac_model:
            if args.suite == "jctc_mace_dac":
                raise ValueError("--suite jctc_mace_dac requires --mace-dac-model <path_or_name>")
        else:
            dac_disp = str(args.mace_dac_dispersion)
            dac_xc = str(args.mace_dac_xc) if dac_disp == "on" else ""
            dac_cases = [
                ("mgmof74_co2", "data/MgMOF-74.cif", ["1", "1", "2"], "co2", ["0.012", "0.28", "1.0"], base_max_mol),
                ("nimof74_co2", "data/NiMOF-74.cif", ["1", "1", "2"], "co2", ["0.012", "0.28", "1.0"], base_max_mol),
                ("cubtc_co2", "data/Cu-BTC.cif", ["1", "1", "1"], "co2", ["0.15", "1.0"], base_max_mol),
                ("irmof1_co2", "data/IRMOF-1.cif", ["1", "1", "1"], "co2", ["1.0", "5.0"], base_max_mol),
            ]
            if args.enable_h2o:
                dac_cases.extend([
                    ("cubtc_h2o", "data/Cu-BTC.cif", ["1", "1", "1"], "h2o", ["0.005", "0.015", "0.025"], max(160, base_max_mol)),
                    ("mgmof74_h2o", "data/MgMOF-74.cif", ["1", "1", "2"], "h2o", ["0.005", "0.015", "0.025"], max(160, base_max_mol)),
                ])
            for tag, cif, sc, gname, p_list, cap in dac_cases:
                p_run = p_list[:2] if mode == "smoke" else p_list
                _add_nested(
                    jname=f"jctc_macedac_{tag}",
                    suite_tag="jctc_mace_dac",
                    cif=cif, sc=sc, guest=gname, pressures=p_run,
                    model=dac_model, dispersion=dac_disp, xc=dac_xc,
                    proposer="auto_mlip", parallel_mode="remc",
                    adapt_shift="on", adapt_k="on", max_mol=cap,
                )

    return jobs


def _partition_gpu_slots(gpu_ids: List[int], gpus_per_job: int) -> List[List[int]]:
    step = max(1, int(gpus_per_job))
    if len(gpu_ids) <= step:
        return [list(gpu_ids)]
    slots = [list(gpu_ids[i : i + step]) for i in range(0, len(gpu_ids), step)]
    return [s for s in slots if s]


def _run_job_on_gpu(job: dict, gpu_slot) -> dict:
    slot_ids = [int(gpu_slot)] if isinstance(gpu_slot, (int, np.integer)) else [int(g) for g in gpu_slot]
    cmd = list(job["cmd"])
    if job["kind"] == "direct":
        cmd.extend(["--gpu", str(slot_ids[0])])
    else:
        cmd.extend(["--gpus", *[str(g) for g in slot_ids]])
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
        "suite": job.get("suite"),
        "gpu_id": int(slot_ids[0]),
        "gpu_ids": slot_ids,
        "returncode": int(proc.returncode),
        "wall_s": wall,
        "out_json": job["out_json"],
        "gates": gates,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 / JCTC v2 Multi-GPU Benchmark & Generalization Runner")
    p.add_argument("--suite", choices=SUITE_CHOICES, default="all")
    p.add_argument("--mode", choices=["smoke", "probe", "full"], default="probe")
    p.add_argument(
        "--gpus", nargs="+", default=["0"],
        help="GPU IDs (space or comma separated, e.g. 0 1 2 3 4 5 6 7 or 0,1,2,3,4,5,6,7)",
    )
    p.add_argument("--gpus-per-job", type=int, default=1, help="Number of GPUs assigned per concurrent job slot")
    p.add_argument("--k-steps", type=int, default=16, help="Inner GPU steps per outer Nested MC cycle")
    p.add_argument("--max-mol", type=int, default=128, help="Base max guest molecules per supercell")
    p.add_argument("--remc-swap-every", type=int, default=2, help="Outer cycles between REMC pressure-swap attempts")
    p.add_argument("--model", default="mace-matpes-r2scan-0")
    p.add_argument("--xc", default="pbe")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--mace-dac-model", default="", help="Optional MACE-DAC-1 fine-tuned checkpoint path or name")
    p.add_argument("--mace-dac-dispersion", choices=["off", "on"], default="off", help="D3 dispersion for MACE-DAC-1 runs")
    p.add_argument("--mace-dac-xc", default="pbe", help="D3 functional for MACE-DAC-1 when dispersion is on")
    p.add_argument("--enable-h2o", action="store_true", help="Include H2O adsorption benchmarks on Cu-BTC & Mg-MOF-74 in jctc_mace_dac")
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
    p.add_argument("--match-distilled-hard-core", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--out-json", default="build/direct_out.json")
    return p.parse_args()


def main():
    args = parse_args()
    if args.direct_worker:
        _run_direct_worker(args)
        return

    gpu_ids = parse_gpu_ids(args.gpus)
    gpu_slots = _partition_gpu_slots(gpu_ids, args.gpus_per_job)
    jobs = _build_job_specs(args)
    if args.dry_run:
        print(json.dumps({"suite": args.suite, "gpus": gpu_ids, "gpu_slots": gpu_slots, "n_jobs": len(jobs), "jobs": jobs}, indent=2))
        return

    import queue
    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)

    results = []
    def _worker(slot_ids: List[int]):
        local_res = []
        slot_label = ",".join(str(g) for g in slot_ids)
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                break
            print(f"[Matrix GPU {slot_label}] Starting {job['name']} ...")
            r = _run_job_on_gpu(job, slot_ids)
            print(f"[Matrix GPU {slot_label}] Finished {job['name']} rc={r['returncode']} ({r['wall_s']:.1f}s)")
            local_res.append(r)
            q.task_done()
        return local_res

    with ThreadPoolExecutor(max_workers=len(gpu_slots)) as ex:
        futs = [ex.submit(_worker, slot) for slot in gpu_slots]
        for f in as_completed(futs):
            results.extend(f.result())

    summary_path = Path(args.out_dir) / "phase2_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"suite": args.suite, "gpus": gpu_ids, "gpu_slots": gpu_slots, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"[Matrix Complete] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
