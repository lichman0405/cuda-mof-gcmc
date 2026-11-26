#!/usr/bin/env python3
"""Pre-submission JCTC matched-budget 2x2 ablation driver for Mg-MOF-74 / CO2 at 298 K.

Implements docs/JCTC_PRESUBMISSION_CHECKLIST.zh-CN.md §2 exactly:
- System: data/MgMOF-74.cif, 1x1x2, CO2 (rigid 3-site), 298 K, mace-matpes-r2scan-0, D3 OFF
- Surrogates (v1.4 unified hard-core, N_max=128, 240x240x64):
    1B-only (unscaled classical gg) vs 1B+2B residual (1200 1B / 192 2B samples, shared 1B grid)
- Swap factor: REMC swaps OFF (0 swap attempts) vs REMC swaps ON (every 2 outer cycles)
- 2x2 factorial groups: A (1B, OFF), B (1B+2B, OFF), C (1B, ON), D (1B+2B, ON)
- Pressures (bar): 0.012 0.04 0.08 0.16 0.22 0.28 0.55 1.0 (8 RTX 3090 GPUs, 1 pressure/GPU)
- 4 ladders/pressures: 2 empty (N=0) + 2 dispersed high-loading (~8.5 mmol/g), shared per replicate
- 3 replicates in rotated execution order, 7,200 s sampling wall clock each, 300 s checkpoints.
"""

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from guests import load_guest
from mlip_distill import DISTILL_VERSION, derive_only_1b_cache_from_full, load_distill_cache, make_distilled_guest_spec
from multi_gpu import _configure_worker_env, _init_worker_target, run_distill_on_gpu, run_pressures_parallel, run_remc_parallel
from nested_mc_engine import K_TO_KJMOL, _deserialize_single_box_state, _serialize_single_box_state, check_hard_core_state_cpp
from qeq_charges import extract_framework_for_proposer
from run_metadata import collect_run_metadata
from run_mlip_gcmc import build_legal_dispersed_high_loading_states, extract_oms_metal_positions, lookup_pressure_override

CIF_REL = "data/MgMOF-74.cif"
SUPERCELL = (1, 1, 2)
GUEST_NAME = "co2"
TEMP_K = 298.0
PRESSURES_BAR = [0.012, 0.04, 0.08, 0.16, 0.22, 0.28, 0.55, 1.0]
PRIMARY_P_BAR = 0.28
MODEL_NAME = "mace-matpes-r2scan-0"
EXPECTED_MODEL_SHA256 = "8f147ecffa1d06a696e5648b13b81abda9c96fda9d0fb3faba3cd3fb27d9bba9"
DISPERSION = False
DTYPE = "float32"
MACE_CHUNK = 2
N_BOXES = 4
MAX_MOL = 128
GRID_DIMS = (240, 240, 64)
DISTILL_1B = 1200
DISTILL_2B = 192
DISTILL_SEED = 2026112551
CBMC_K = 128
K_STEPS = 24
P_INS_DEL = 0.50
EQ_CYCLES = 1500
REMC_SWAP_EVERY = 2
GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
INIT_HIGH_LOADING_MMOL_G = 8.5

CALIB_SEED = 2026112500
CALIB_PILOT_CYCLES = 40
REPLICATE_SEEDS = [2026112601, 2026112602, 2026112603]
ROTATED_ORDERS = {
    2026112601: ["A", "B", "C", "D"],
    2026112602: ["B", "D", "A", "C"],
    2026112603: ["D", "C", "B", "A"],
}
GROUPS = {
    "A": {"only_1b": True, "remc_enable_swaps": False, "label": "1B-only (classical gg) + REMC OFF"},
    "B": {"only_1b": False, "remc_enable_swaps": False, "label": "1B+2B + REMC OFF"},
    "C": {"only_1b": True, "remc_enable_swaps": True, "label": "1B-only (classical gg) + REMC ON"},
    "D": {"only_1b": False, "remc_enable_swaps": True, "label": "1B+2B + REMC ON"},
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _check_gpus_idle(gpus: List[int], max_used_mib: int = 250, settle_wait_s: float = 15.0) -> Dict[str, Any]:
    """Query nvidia-smi compute apps and GPU memory (allowing trailing util window to settle) before starting."""
    gpu_set = set(int(g) for g in gpus)
    deadline = time.perf_counter() + max(0.0, float(settle_wait_s))
    while True:
        try:
            mem_out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                text=True,
            )
            apps_out = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"],
                text=True,
            )
        except Exception as exc:
            return {"checked": False, "error": str(exc), "idle": False, "details": []}

        details = []
        busy = []
        mem_busy = False
        for line in mem_out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            idx, util, mem = int(parts[0]), int(parts[1]), int(parts[2])
            if idx in gpu_set:
                rec = {"gpu": idx, "util_pct": util, "memory_used_mib": mem}
                details.append(rec)
                if mem > max_used_mib:
                    mem_busy = True
                    busy.append(rec)
                elif util > 10:
                    busy.append(rec)
        active_apps = [ln.strip() for ln in apps_out.strip().splitlines() if ln.strip()]
        idle = (len(busy) == 0 and len(active_apps) == 0)
        if idle or mem_busy or len(active_apps) > 0 or time.perf_counter() >= deadline:
            return {
                "checked": True,
                "idle": idle,
                "busy_gpus": busy,
                "active_compute_apps": active_apps,
                "details": details,
            }
        time.sleep(1.5)


def _git_info() -> Dict[str, Any]:
    def _run(cmd):
        try:
            return subprocess.check_output(cmd, cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(_run(["git", "status", "--porcelain"])),
    }


def _build_group_cli_command(
    group: str,
    rep_seed: int,
    shift_map_path: Path,
    init_states_path: Path,
    out_dir: Path,
    eq_cycles: int,
    prod_cycles: int,
    max_wall_s: float,
    ckpt_every_s: float,
) -> List[str]:
    g = GROUPS[group]
    cmd = [
        ".venv-mlip/bin/python",
        "scripts/run_mlip_gcmc.py",
        "--cif", CIF_REL,
        "--supercell", *[str(x) for x in SUPERCELL],
        "--guest", GUEST_NAME,
        "--temp", str(TEMP_K),
        "--pressures", *[str(p) for p in PRESSURES_BAR],
        "--proposer", "auto_mlip",
        "--model", MODEL_NAME,
        "--dispersion", "off",
        "--dtype", DTYPE,
        "--mace-chunk", str(MACE_CHUNK),
        "--boxes", str(N_BOXES),
        "--max-mol", str(MAX_MOL),
        "--grid-dims", *[str(d) for d in GRID_DIMS],
        "--distill-samples-1b", str(DISTILL_1B),
        "--distill-samples-2b", str(DISTILL_2B),
        "--cbmc-k", str(CBMC_K),
        "--k-steps", str(K_STEPS),
        "--p-ins-del", str(P_INS_DEL),
        "--eq-cycles", str(eq_cycles),
        "--prod-cycles", str(prod_cycles),
        "--adapt-k", "off",
        "--adapt-shift", "off",
        "--pilot-cycles", "0",
        "--parallel-mode", "remc",
        "--gpus", *[str(i) for i in GPUS],
        "--remc-swap-every", str(REMC_SWAP_EVERY),
        "--remc-swap", "on" if g["remc_enable_swaps"] else "off",
        "--shift-map-json", str(shift_map_path),
        "--init-states-json", str(init_states_path),
        "--max-wall-s", str(max_wall_s),
        "--diag-checkpoint-every-s", str(ckpt_every_s),
        "--seed", str(rep_seed),
        "--out-json", str(out_dir / "cli_equiv.json"),
    ]
    if g["only_1b"]:
        cmd.append("--distill-1b-only")
    return cmd


def _calibrate_surrogate(
    only_1b: bool,
    cache_npz_path: str,
    fw_data: dict,
    guest,
    cache_dir: Path,
    worker_threads: int,
) -> dict:
    """Run preset 8-GPU calibration (Stage 1/2/3 with pilot_cycles=40) on independent seed."""
    run_cfg = {
        "cif_path": str(ROOT / CIF_REL),
        "fw_data": fw_data,
        "guest": guest,
        "supercell": SUPERCELL,
        "T_K": TEMP_K,
        "ff_mode": 2,
        "pol_mode": 0,
        "grid_dims": GRID_DIMS,
        "n_boxes": N_BOXES,
        "max_mol": MAX_MOL,
        "k_steps": K_STEPS,
        "cbmc_k": CBMC_K,
        "p_ins_del": P_INS_DEL,
        "grid_only": False,
        "distill_1b_only": bool(only_1b),
        "distill_cache_path": cache_npz_path,
        "eq_cycles": 0,
        "prod_cycles": 2,
        "base_seed": CALIB_SEED,
        "shift_mode": "auto",
        "pilot_cycles": CALIB_PILOT_CYCLES,
        "shift_override": None,
        "adapt_shift_frac": 0.0,
        "adapt_k_frac": 0.0,
        "k_min": 2,
        "k_max": K_STEPS,
        "remc_swap_every": REMC_SWAP_EVERY,
        "init_mode": "auto",
        "init_loading": 8.0,
        "checkpoint_dir": "",
        "checkpoint_every": 0,
        "resume_mode": "none",
        "snapshot_stride": 0,
        "model": MODEL_NAME,
        "dtype": DTYPE,
        "dispersion": DISPERSION,
        "dispersion_xc": "pbe",
        "mace_chunk": MACE_CHUNK,
        "worker_threads": int(worker_threads),
    }
    pending = list(enumerate(PRESSURES_BAR))
    t0 = time.perf_counter()
    pts = run_pressures_parallel(pending, GPUS, run_cfg)
    wall_s = float(time.perf_counter() - t0)
    pts_sorted = sorted(pts, key=lambda x: float(x["P_bar"]))
    shifts = {f"{float(pt['P_bar']):g}": float(pt["delta_eps_kJ_mol"]) for pt in pts_sorted}
    per_p_info = {}
    for pt in pts_sorted:
        k = f"{float(pt['P_bar']):g}"
        per_p_info[k] = {
            "P_bar": float(pt["P_bar"]),
            "delta_eps_kJ_mol": float(pt["delta_eps_kJ_mol"]),
            "prop_grid_wall_s": float(pt.get("prop_grid_wall_s", 0.0)),
            "shift_calib_wall_s": float(pt.get("shift_calib_wall_s", 0.0)),
            "shift_calibration": pt.get("shift_calibration", {}),
        }
    return {
        "surrogate": "1B_only" if only_1b else "1B_2B",
        "only_1b": bool(only_1b),
        "calib_seed": CALIB_SEED,
        "pilot_cycles": CALIB_PILOT_CYCLES,
        "wall_s": wall_s,
        "distill_cache_path": cache_npz_path,
        "shifts_kJ_mol": shifts,
        "per_pressure": per_p_info,
    }


def _init_states_worker(gpu_id: int, parent_cvd: str | None, worker_threads: int, cache_2b_path: str,
                        rep_seeds: List[int], out_q):
    """Build 2 empty + 2 dispersed high-loading (~8.5 mmol/g) states per replicate/pressure on GPU."""
    prop = None
    try:
        _configure_worker_env(gpu_id, parent_cvd, worker_threads)
        fw_data = extract_framework_for_proposer(str(ROOT / CIF_REL), charge_mode="auto")
        guest = make_distilled_guest_spec(load_guest(GUEST_NAME))
        dist_2b = load_distill_cache(cache_2b_path)
        dist_1b = None  # checked cleanly via prop_2b + C++ HC + MACE target; 1B checked in main

        run_cfg_stub = {
            "cif_path": str(ROOT / CIF_REL),
            "supercell": SUPERCELL,
            "model": MODEL_NAME,
            "dtype": DTYPE,
            "dispersion": DISPERSION,
            "dispersion_xc": "pbe",
            "mace_chunk": MACE_CHUNK,
            "fw_data": fw_data,
            "guest": guest,
        }
        target = _init_worker_target(run_cfg_stub)
        model_sha = getattr(target, "model_sha256", None)
        if model_sha != EXPECTED_MODEL_SHA256:
            raise RuntimeError(f"Model SHA-256 mismatch: got {model_sha}, expected {EXPECTED_MODEL_SHA256}")

        from nested_mc_engine import CudaGcmcProposer
        prop = CudaGcmcProposer.from_distilled(
            fw_data=fw_data,
            guest=guest,
            distill_res=dist_2b,
            supercell=SUPERCELL,
            T_K=TEMP_K,
            pressure_bar=PRIMARY_P_BAR,
            n_boxes=2,
            max_mol=MAX_MOL,
            seed=rep_seeds[0],
            device_id=0,
            cbmc_k=CBMC_K,
            p_ins_del=P_INS_DEL,
        )
        conv_mmol_g = 1000.0 / prop.fw_mass_gmol
        target_N = int(round(INIT_HIGH_LOADING_MMOL_G / conv_mmol_g))
        fw_super = target.fw
        fw_pos = np.asarray(fw_super.get_positions(), dtype=np.float64)
        oms_metal_pos = extract_oms_metal_positions(fw_super)
        guest_local_pos = guest.real_local_positions()
        H_unit_cols = np.ascontiguousarray(fw_data["H_unit_cols"], dtype=np.float64)
        fw_unit_frac = np.ascontiguousarray(np.mod(np.asarray(fw_data["frac"], dtype=np.float64), 1.0))

        results_by_rep = {}
        for rep_seed in rep_seeds:
            by_p = {}
            summary_by_p = {}
            for p_idx, p_bar in enumerate(PRESSURES_BAR):
                p_key = f"{float(p_bar):g}"
                hl_seed = int(rep_seed) + 1009 * (p_idx + 1)
                hl_states, hl_sum = build_legal_dispersed_high_loading_states(
                    prop=prop,
                    target=target,
                    target_N=target_N,
                    seed=hl_seed,
                    fw_pos=fw_pos,
                    oms_metal_pos=oms_metal_pos,
                    guest_local_pos=guest_local_pos,
                )
                empty_box = {
                    "N": 0, "cx": np.zeros(0), "cy": np.zeros(0), "cz": np.zeros(0),
                    "qw": np.ones(0), "qx": np.zeros(0), "qy": np.zeros(0), "qz": np.zeros(0),
                }
                four_boxes = [dict(empty_box), dict(empty_box), hl_states[0], hl_states[1]]
                # Verify C++ hard-core + CUDA eval on all 4 boxes
                u_ref_list = []
                for b, st in enumerate(four_boxes):
                    ov_cpp = check_hard_core_state_cpp(
                        H_unit_cols, SUPERCELL, fw_unit_frac, guest, st, max_mol=MAX_MOL,
                    )
                    u_cu, ov_cu = prop.eval_energy_quat(st)
                    if ov_cpp or ov_cu or not math.isfinite(float(u_cu)):
                        raise RuntimeError(f"Generated init state rep={rep_seed} P={p_bar} box={b} failed HC/CUDA check")
                    u_ref_list.append(float(u_cu))

                by_p[p_key] = [_serialize_single_box_state(st) for st in four_boxes]
                print(
                    f"  [InitStates rep={rep_seed} P={p_key} bar] "
                    f"N={[int(st['N']) for st in four_boxes]} oms={[0, 0] + list(hl_sum['init_oms_occ_per_box'] or [0, 0])}"
                )
                summary_by_p[p_key] = {
                    "P_bar": float(p_bar),
                    "hl_seed": hl_seed,
                    "target_N_high": target_N,
                    "actual_N_per_box": [int(st["N"]) for st in four_boxes],
                    "actual_loading_mmol_g": [round(int(st["N"]) * conv_mmol_g, 4) for st in four_boxes],
                    "init_U_tgt_K": [0.0, 0.0] + list(hl_sum["init_U_tgt_K"]),
                    "init_U_ref_K": u_ref_list,
                    "init_oms_occ_per_box": [0, 0] + list(hl_sum["init_oms_occ_per_box"] or [0, 0]),
                    "init_min_gg_A": hl_sum["init_min_gg_A"],
                }
            results_by_rep[int(rep_seed)] = {
                "by_pressure": by_p,
                "summary_by_pressure": summary_by_p,
                "conv_mmol_g": float(conv_mmol_g),
                "model_sha256": model_sha,
            }
        out_q.put(("ok", results_by_rep))
    except BaseException as exc:
        import traceback
        out_q.put(("err", f"{exc}\n{traceback.format_exc()}"))
    finally:
        if prop is not None:
            try:
                prop.close()
            except Exception:
                pass


def cmd_prepare(args) -> None:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = out_root / "distill_cache"
    calib_dir = out_root / "calibration"
    init_dir = out_root / "init_states"
    cache_dir.mkdir(parents=True, exist_ok=True)
    calib_dir.mkdir(parents=True, exist_ok=True)
    init_dir.mkdir(parents=True, exist_ok=True)

    cif_path = ROOT / CIF_REL
    cif_sha = _sha256_file(cif_path)

    fw_data = extract_framework_for_proposer(str(cif_path), charge_mode="auto")
    guest = make_distilled_guest_spec(load_guest(GUEST_NAME))

    # 1. Distill 1B-only and 1B+2B in isolated GPU subprocesses with IDENTICAL distill_seed
    base_distill_cfg = {
        "cif_path": str(cif_path),
        "fw_data": fw_data,
        "guest": guest,
        "supercell": SUPERCELL,
        "grid_dims": GRID_DIMS,
        "n_1b": DISTILL_1B,
        "n_2b": DISTILL_2B,
        "distill_seed": DISTILL_SEED,
        "cache_dir": str(cache_dir),
        "model": MODEL_NAME,
        "dtype": DTYPE,
        "dispersion": DISPERSION,
        "dispersion_xc": "pbe",
        "mace_chunk": MACE_CHUNK,
        "worker_threads": int(args.worker_threads),
    }

    t_d1 = time.perf_counter()
    res_2b = run_distill_on_gpu(GPUS[0], {**base_distill_cfg, "only_1b": False})
    t_2b_wall = float(time.perf_counter() - t_d1)

    res_1b = derive_only_1b_cache_from_full(
        full_cache_npz_path=res_2b["cache_npz_path"],
        cache_dir=str(cache_dir),
        cif_path=str(cif_path),
        supercell=SUPERCELL,
        guest=guest,
        grid_dims=GRID_DIMS,
        model_name=MODEL_NAME,
        dispersion=DISPERSION,
        dispersion_xc="pbe",
        dtype=DTYPE,
        n_1b=DISTILL_1B,
        n_2b=DISTILL_2B,
        distill_seed=DISTILL_SEED,
    )
    t_1b_wall = float(res_1b["summary"]["distill_wall_s"])

    d_1b = load_distill_cache(res_1b["cache_npz_path"])
    d_2b = load_distill_cache(res_2b["cache_npz_path"])

    # Strict invariance verification between 1B-only and 1B+2B
    if d_1b["summary"].get("version") != DISTILL_VERSION or d_2b["summary"].get("version") != DISTILL_VERSION:
        raise RuntimeError("Unexpected distillation version in cache")
    for tag, d_chk in [("1B_only", d_1b), ("1B_2B", d_2b)]:
        m_sha = d_chk["summary"].get("model_sha256")
        if m_sha != EXPECTED_MODEL_SHA256:
            raise RuntimeError(f"{tag} cache model SHA-256 {m_sha} != expected {EXPECTED_MODEL_SHA256}")
    for key_1b in ("n_1b_evaluated", "n_1b_kept", "q_env_eff", "mae_1b_kJ_mol", "r2_1b"):
        if d_1b["summary"].get(key_1b) != d_2b["summary"].get(key_1b):
            raise RuntimeError(f"1B invariance mismatch on '{key_1b}': {d_1b['summary'].get(key_1b)} vs {d_2b['summary'].get(key_1b)}")
    if not np.array_equal(d_1b["grids_u_lj"], d_2b["grids_u_lj"]):
        max_diff = float(np.max(np.abs(d_1b["grids_u_lj"] - d_2b["grids_u_lj"])))
        raise RuntimeError(f"1B grids differ between 1B-only and 1B+2B caches! max|diff|={max_diff}")
    if d_1b["gg_q_scale"] != 1.0 or d_1b["gg_lj_scale"] != 1.0 or not np.all(d_1b["gg_com_lut_K"] == 0.0):
        raise RuntimeError("1B-only cache must have unit classical guest-guest scales and zero 2B LUT")
    print(
        f"[Prepare] Verified bit-identical 1B grids & coefficients "
        f"(shape={d_1b['grids_u_lj'].shape}, n_1b={d_1b['summary']['n_1b_kept']}, "
        f"MAE_1B={d_1b['summary']['mae_1b_kJ_mol']:.3f} kJ/mol, R2_1B={d_1b['summary']['r2_1b']:.5f})"
    )

    # 2. Separate independent-seed 8-GPU Δε calibration for 1B-only and 1B+2B
    cal_1b = _calibrate_surrogate(True, res_1b["cache_npz_path"], fw_data, guest, calib_dir, args.worker_threads)
    cal_1b["distill_wall_s"] = t_1b_wall
    cal_2b = _calibrate_surrogate(False, res_2b["cache_npz_path"], fw_data, guest, calib_dir, args.worker_threads)
    cal_2b["distill_wall_s"] = t_2b_wall
    cal_1b_path = calib_dir / "shift_1b_only.json"
    cal_2b_path = calib_dir / "shift_1b_2b.json"
    _atomic_write_json(cal_1b_path, cal_1b)
    _atomic_write_json(cal_2b_path, cal_2b)

    # 3. Generate shared pre-validated initial states for all 3 replicates in isolated GPU worker
    t_init0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=_init_states_worker,
        args=(GPUS[0], os.environ.get("CUDA_VISIBLE_DEVICES"), args.worker_threads,
              res_2b["cache_npz_path"], REPLICATE_SEEDS, q),
    )
    p.start()
    tag, payload_init = q.get()
    p.join(timeout=15.0)
    if tag != "ok":
        raise RuntimeError(f"Init states worker failed:\n{payload_init}")
    init_gen_wall_s = float(time.perf_counter() - t_init0)

    init_files = {}
    for rep_seed in REPLICATE_SEEDS:
        rep_path = init_dir / f"init_rep_{rep_seed}.json"
        _atomic_write_json(rep_path, payload_init[rep_seed])
        init_files[str(rep_seed)] = {
            "path": str(rep_path),
            "sha256": _sha256_file(rep_path),
            "summary_by_pressure": payload_init[rep_seed]["summary_by_pressure"],
        }

    # 4. Build exact runnable CLI commands per group & save protocol.json
    commands = {}
    print("\n=== Formal 4-Group Runnable Commands (shown for Replicate 1) ===")
    for rep_seed in REPLICATE_SEEDS:
        commands[str(rep_seed)] = {}
        for grp in ["A", "B", "C", "D"]:
            g = GROUPS[grp]
            s_path = cal_1b_path if g["only_1b"] else cal_2b_path
            i_path = Path(init_files[str(rep_seed)]["path"])
            r_dir = out_root / "runs" / f"rep_{rep_seed}_{grp}"
            cmd_list = _build_group_cli_command(
                grp, rep_seed, s_path, i_path, r_dir,
                eq_cycles=EQ_CYCLES, prod_cycles=200000, max_wall_s=7200.0, ckpt_every_s=300.0,
            )
            cmd_str = " ".join(shlex.quote(tok) for tok in cmd_list)
            commands[str(rep_seed)][grp] = cmd_str
            if rep_seed == REPLICATE_SEEDS[0]:
                print(f"  [{grp}: {g['label']}]\n    {cmd_str}")

    protocol = {
        "protocol_version": "jctc_presubmission_v1",
        "git": _git_info(),
        "system": {
            "cif": CIF_REL,
            "cif_sha256": cif_sha,
            "supercell": list(SUPERCELL),
            "guest": GUEST_NAME,
            "T_K": TEMP_K,
            "pressures_bar": PRESSURES_BAR,
            "primary_pressure_bar": PRIMARY_P_BAR,
            "model": MODEL_NAME,
            "expected_model_sha256": EXPECTED_MODEL_SHA256,
            "verified_model_sha256": payload_init[REPLICATE_SEEDS[0]]["model_sha256"],
            "dispersion": DISPERSION,
            "dtype": DTYPE,
            "mace_chunk": MACE_CHUNK,
        },
        "sampling": {
            "n_boxes": N_BOXES,
            "max_mol": MAX_MOL,
            "grid_dims": list(GRID_DIMS),
            "distill_version": DISTILL_VERSION,
            "distill_1b_samples": DISTILL_1B,
            "distill_2b_samples": DISTILL_2B,
            "distill_seed": DISTILL_SEED,
            "cbmc_k": CBMC_K,
            "k_steps": K_STEPS,
            "p_ins_del": P_INS_DEL,
            "eq_cycles": EQ_CYCLES,
            "adapt_k": "off",
            "adapt_shift": "off",
            "pilot_cycles": 0,
            "gpus": GPUS,
            "remc_swap_every": REMC_SWAP_EVERY,
            "formal_budget_wall_s": 7200.0,
            "diag_checkpoint_every_s": 300.0,
            "calib_seed": CALIB_SEED,
            "calib_pilot_cycles": CALIB_PILOT_CYCLES,
            "replicate_seeds": REPLICATE_SEEDS,
            "rotated_orders": {str(k): v for k, v in ROTATED_ORDERS.items()},
        },
        "surrogates": {
            "1B_only": {
                "cache_npz_path": res_1b["cache_npz_path"],
                "cache_npz_sha256": _sha256_file(Path(res_1b["cache_npz_path"])),
                "distill_sha256": res_1b["distill_sha256"],
                "distill_wall_s": t_1b_wall,
                "summary": res_1b["summary"],
                "calib_path": str(cal_1b_path),
                "shifts_kJ_mol": cal_1b["shifts_kJ_mol"],
                "calib_wall_s": cal_1b["wall_s"],
            },
            "1B_2B": {
                "cache_npz_path": res_2b["cache_npz_path"],
                "cache_npz_sha256": _sha256_file(Path(res_2b["cache_npz_path"])),
                "distill_sha256": res_2b["distill_sha256"],
                "distill_wall_s": t_2b_wall,
                "summary": res_2b["summary"],
                "calib_path": str(cal_2b_path),
                "shifts_kJ_mol": cal_2b["shifts_kJ_mol"],
                "calib_wall_s": cal_2b["wall_s"],
            },
            "shared_1b_grid_bit_exact": True,
            "derived_1b_from_full": True,
        },
        "init_states": {
            "target_high_loading_mmol_g": INIT_HIGH_LOADING_MMOL_G,
            "generation_wall_s": init_gen_wall_s,
            "replicates": init_files,
        },
        "groups": GROUPS,
        "commands": commands,
    }
    proto_path = out_root / "protocol.json"
    _atomic_write_json(proto_path, protocol)
    print(f"\n[Prepare] Saved protocol to {proto_path}")


def _save_run_archives(
    run_dir: Path,
    run_meta: dict,
    pts: List[dict],
) -> None:
    """Save compact run_result.json and compressed run_traces.npz preserving full raw arrays."""
    run_dir.mkdir(parents=True, exist_ok=True)
    npz_payload = {"pressures_bar": np.array([float(pt["P_bar"]) for pt in pts], dtype=np.float64)}
    slim_pts = []
    trace_keys = [
        ("N", "prod_N_history", np.int32),
        ("U_tgt_kJ", "prod_U_history", np.float64),
        ("U_ref_kJ", "prod_U_ref_history", np.float64),
        ("oms_occ", "prod_oms_occ_history", np.int32),
        ("accepted", "prod_accepted_history", np.int8),
        ("dN", "prod_dN_history", np.int32),
        ("proposed_dN", "prod_proposed_dN_history", np.int32),
    ]
    for pt in pts:
        p_tag = f"{float(pt['P_bar']):g}".replace(".", "p")
        pt_copy = dict(pt)
        pt_copy.pop("remc_run_summary", None)
        for suffix, key, dt in trace_keys:
            arr = np.asarray(pt.get(key, []), dtype=dt)
            npz_payload[f"p_{p_tag}_{suffix}"] = arr
        box_n_hist = np.asarray(pt.get("box_N_history", []), dtype=np.int32)
        npz_payload[f"p_{p_tag}_box_N_all"] = box_n_hist
        # Store slim metadata in JSON without duplicating giant raw step arrays already saved lossless in NPZ
        for large_k in [
            "prod_N_history", "prod_U_history", "prod_U_ref_history",
            "prod_oms_occ_history", "prod_accepted_history",
            "prod_dN_history", "prod_proposed_dN_history",
            "box_N_history", "trajectory", "prod_trial_marg_per_box",
        ]:
            pt_copy.pop(large_k, None)
        slim_pts.append(pt_copy)

    npz_tmp = run_dir / "run_traces.tmp.npz"
    npz_final = run_dir / "run_traces.npz"
    np.savez_compressed(npz_tmp, **npz_payload)
    os.replace(npz_tmp, npz_final)

    res_payload = {
        **run_meta,
        "traces_npz": str(npz_final),
        "traces_npz_sha256": _sha256_file(npz_final),
        "points": slim_pts,
    }
    _atomic_write_json(run_dir / "run_result.json", res_payload)


def _execute_single_ablation_run(
    protocol: dict,
    group: str,
    rep_seed: int,
    run_dir: Path,
    eq_cycles: int,
    prod_cycles: int,
    max_wall_s: float,
    ckpt_every_s: float,
    worker_threads: int,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    g = GROUPS[group]
    surr_key = "1B_only" if g["only_1b"] else "1B_2B"
    surr = protocol["surrogates"][surr_key]
    shift_map = surr["shifts_kJ_mol"]
    init_entry = protocol["init_states"]["replicates"][str(rep_seed)]
    init_data = json.loads(Path(init_entry["path"]).read_text(encoding="utf-8"))
    init_by_p = init_data["by_pressure"]

    fw_data = extract_framework_for_proposer(str(ROOT / CIF_REL), charge_mode="auto")
    guest = make_distilled_guest_spec(load_guest(GUEST_NAME))
    args_stub = argparse.Namespace(
        cif=CIF_REL,
        supercell=list(SUPERCELL),
        temp=TEMP_K,
        guest=GUEST_NAME,
        guest_json="",
        charge_mode="auto",
        proposer="auto_mlip",
        cbmc_k=CBMC_K,
        p_ins_del=P_INS_DEL,
        grid_only=False,
        distill_1b_only=bool(g["only_1b"]),
        adapt_shift="off",
        adapt_k="off",
        k_min=2,
        k_max=K_STEPS,
        remc_swap_every=REMC_SWAP_EVERY,
        shift_mode="auto",
        shift_override=None,
        init_mode="override",
        init_loading=INIT_HIGH_LOADING_MMOL_G,
        model=MODEL_NAME,
        dtype=DTYPE,
        dispersion="off",
        xc="pbe",
        mace_chunk=MACE_CHUNK,
        grid_spacing=0.11,
        k_steps=K_STEPS,
        eq_cycles=eq_cycles,
        prod_cycles=prod_cycles,
        pilot_cycles=0,
        boxes=N_BOXES,
        seed=int(rep_seed),
        gpus=[str(x) for x in GPUS],
        parallel_mode="remc",
        pressures=PRESSURES_BAR,
        max_mol=MAX_MOL,
        device="cuda:0",
    )
    run_meta_std = collect_run_metadata(
        args=args_stub,
        fw_data=fw_data,
        guest=guest,
        grid_dims=GRID_DIMS,
        prop_info={"distill_sha256": surr["distill_sha256"]},
    )

    ckpt_path = run_dir / "checkpoints.json"

    def _on_ckpt(latest_ckpt: dict, all_ckpts: List[dict]) -> None:
        _atomic_write_json(ckpt_path, {"latest": latest_ckpt, "checkpoints": all_ckpts})

    run_cfg = {
        "cif_path": str(ROOT / CIF_REL),
        "fw_data": fw_data,
        "guest": guest,
        "supercell": SUPERCELL,
        "T_K": TEMP_K,
        "ff_mode": 2,
        "pol_mode": 0,
        "grid_dims": GRID_DIMS,
        "n_boxes": N_BOXES,
        "max_mol": MAX_MOL,
        "k_steps": K_STEPS,
        "cbmc_k": CBMC_K,
        "p_ins_del": P_INS_DEL,
        "grid_only": False,
        "distill_1b_only": bool(g["only_1b"]),
        "distill_cache_path": surr["cache_npz_path"],
        "eq_cycles": int(eq_cycles),
        "prod_cycles": int(prod_cycles),
        "base_seed": int(rep_seed),
        "shift_mode": "auto",
        "pilot_cycles": 0,
        "shift_override": None,
        "shift_override_map": shift_map,
        "adapt_shift_frac": 0.0,
        "adapt_k_frac": 0.0,
        "k_min": 2,
        "k_max": K_STEPS,
        "remc_swap_every": REMC_SWAP_EVERY,
        "remc_enable_swaps": bool(g["remc_enable_swaps"]),
        "max_wall_s": float(max_wall_s),
        "diag_checkpoint_every_s": float(ckpt_every_s),
        "primary_pressure_bar": PRIMARY_P_BAR,
        "checkpoint_callback": _on_ckpt,
        "init_mode": "auto",
        "init_loading": INIT_HIGH_LOADING_MMOL_G,
        "init_states_by_pressure": init_by_p,
        "checkpoint_dir": "",
        "checkpoint_every": 0,
        "resume_mode": "none",
        "snapshot_stride": 0,
        "run_metadata": run_meta_std,
        "model": MODEL_NAME,
        "dtype": DTYPE,
        "dispersion": DISPERSION,
        "dispersion_xc": "pbe",
        "mace_chunk": MACE_CHUNK,
        "worker_threads": int(worker_threads),
    }

    gpu_health_before = _check_gpus_idle(GPUS)
    if gpu_health_before.get("checked") and not gpu_health_before.get("idle"):
        raise RuntimeError(f"GPU contention detected before run rep={rep_seed} {group}: {gpu_health_before}")

    t_total0 = time.perf_counter()
    pending = list(enumerate(PRESSURES_BAR))
    pts = run_remc_parallel(pending, GPUS, run_cfg)
    t_write0 = time.perf_counter()

    remc_summary = pts[0].get("remc_run_summary", {}) if pts else {}
    run_header = {
        "run_id": f"rep_{rep_seed}_{group}",
        "group": group,
        "group_label": g["label"],
        "only_1b": bool(g["only_1b"]),
        "remc_enable_swaps": bool(g["remc_enable_swaps"]),
        "replicate_seed": int(rep_seed),
        "git": _git_info(),
        "cli_command": protocol["commands"][str(rep_seed)][group],
        "gpu_health_before": gpu_health_before,
        "surrogate_key": surr_key,
        "cache_npz_path": surr["cache_npz_path"],
        "cache_npz_sha256": surr["cache_npz_sha256"],
        "distill_sha256": surr["distill_sha256"],
        "distill_wall_s": surr["distill_wall_s"],
        "calib_wall_s": surr["calib_wall_s"],
        "shifts_kJ_mol": shift_map,
        "init_states_path": init_entry["path"],
        "init_states_sha256": init_entry["sha256"],
        "model_sha256": protocol["system"]["verified_model_sha256"],
        "cif_sha256": protocol["system"]["cif_sha256"],
        "budget_wall_s_requested": float(max_wall_s),
        "diag_checkpoint_every_s": float(ckpt_every_s),
        "remc_run_summary": remc_summary,
        "metadata": run_meta_std,
    }
    _save_run_archives(run_dir, run_header, pts)
    write_wall_s = float(time.perf_counter() - t_write0)
    total_wall_s = float(time.perf_counter() - t_total0)
    # Persist write/total timings cleanly
    saved_res = json.loads((run_dir / "run_result.json").read_text(encoding="utf-8"))
    saved_res["write_wall_s"] = write_wall_s
    saved_res["total_wall_s"] = total_wall_s
    _atomic_write_json(run_dir / "run_result.json", saved_res)
    return saved_res


def cmd_smoke(args) -> None:
    out_root = Path(args.out_root)
    proto = json.loads((out_root / "protocol.json").read_text(encoding="utf-8"))
    rep_seed = REPLICATE_SEEDS[0]
    smoke_root = out_root / "smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)

    results = {}
    for grp in ["A", "B", "C", "D"]:
        print(f"\n========== [SMOKE] Group {grp} ({GROUPS[grp]['label']}) ==========")
        res = _execute_single_ablation_run(
            protocol=proto,
            group=grp,
            rep_seed=rep_seed,
            run_dir=smoke_root / f"smoke_{grp}",
            eq_cycles=4,
            prod_cycles=1000,
            max_wall_s=float(args.smoke_wall_s),
            ckpt_every_s=float(args.smoke_ckpt_s),
            worker_threads=args.worker_threads,
        )
        results[grp] = res

    # Strict smoke invariance assertions
    ref_init = {
        pt["P_bar"]: (pt["init_state_summary"]["init_N_per_box"], pt["init_state_summary"].get("init_U_tgt_K"))
        for pt in results["A"]["points"]
    }
    for grp, res in results.items():
        g = GROUPS[grp]
        assert len(res["points"]) == len(PRESSURES_BAR)
        for pt in res["points"]:
            n_exp, u_exp = ref_init[pt["P_bar"]]
            assert pt["init_state_summary"]["init_N_per_box"] == n_exp, f"{grp} init N mismatch"
            assert np.allclose(pt["init_state_summary"]["init_U_tgt_K"], u_exp, atol=1e-4), f"{grp} init U mismatch"
            if not g["remc_enable_swaps"]:
                assert pt["remc_total_swap_att_up"] == 0 and pt["remc_swap_att_up"] == 0, (
                    f"{grp} recorded non-zero swap attempts!"
                )
        if g["remc_enable_swaps"]:
            tot_att = sum(int(pt["remc_total_swap_att_up"]) for pt in res["points"])
            assert tot_att > 0, f"{grp} enabled swaps but recorded 0 attempts"
        if g["only_1b"]:
            assert res["distill_sha256"] == proto["surrogates"]["1B_only"]["distill_sha256"]
        else:
            assert res["distill_sha256"] == proto["surrogates"]["1B_2B"]["distill_sha256"]

    summary = {
        grp: {
            "completed_cycles": r["remc_run_summary"]["completed_cycles"],
            "sampling_wall_s": r["remc_run_summary"]["sampling_wall_s"],
            "stop_reason": r["remc_run_summary"]["stop_reason"],
            "total_swap_attempts": sum(pt["remc_total_swap_att_up"] for pt in r["points"]),
            "total_swap_accepted": sum(pt["remc_total_swap_acc_up"] for pt in r["points"]),
            "n_checkpoints": len(r["remc_run_summary"]["checkpoints"]),
        }
        for grp, r in results.items()
    }
    _atomic_write_json(smoke_root / "smoke_summary.json", summary)
    print("\n[SMOKE] ALL 4-GROUP INVARIANCE ASSERTIONS PASSED:")
    print(json.dumps(summary, indent=2))


def cmd_run_formal(args) -> None:
    out_root = Path(args.out_root)
    proto = json.loads((out_root / "protocol.json").read_text(encoding="utf-8"))

    rep_filter = set(int(x) for x in args.replicates) if args.replicates else set(REPLICATE_SEEDS)
    grp_filter = set(args.groups) if args.groups else {"A", "B", "C", "D"}

    schedule = []
    for rep_seed in REPLICATE_SEEDS:
        if rep_seed not in rep_filter:
            continue
        for grp in ROTATED_ORDERS[rep_seed]:
            if grp in grp_filter:
                schedule.append((rep_seed, grp))

    print(f"[Formal] Execution schedule ({len(schedule)} runs): {schedule}")
    for idx, (rep_seed, grp) in enumerate(schedule, 1):
        run_dir = out_root / "runs" / f"rep_{rep_seed}_{grp}"
        res_file = run_dir / "run_result.json"
        npz_file = run_dir / "run_traces.npz"
        if args.skip_done and res_file.exists() and npz_file.exists():
            try:
                old = json.loads(res_file.read_text(encoding="utf-8"))
                if (
                    len(old.get("points", [])) == len(PRESSURES_BAR)
                    and old.get("remc_run_summary", {}).get("sampling_wall_s", 0.0) >= float(args.budget_wall_s) * 0.99
                ):
                    print(f"[{idx}/{len(schedule)}] Skipping completed rep={rep_seed} group={grp}")
                    continue
            except Exception:
                pass

        print(f"\n==================== [{idx}/{len(schedule)}] Formal Run rep={rep_seed} group={grp} ({GROUPS[grp]['label']}) ====================")
        res = _execute_single_ablation_run(
            protocol=proto,
            group=grp,
            rep_seed=rep_seed,
            run_dir=run_dir,
            eq_cycles=EQ_CYCLES,
            prod_cycles=200000,
            max_wall_s=float(args.budget_wall_s),
            ckpt_every_s=float(args.ckpt_every_s),
            worker_threads=args.worker_threads,
        )
        rsum = res.get("remc_run_summary", {})
        print(
            f"[DONE {idx}/{len(schedule)}] rep={rep_seed} {grp}: cyc={rsum.get('completed_cycles')} "
            f"prod={rsum.get('prod_cycles_done')} wall={rsum.get('sampling_wall_s', 0.0):.1f}s "
            f"stop={rsum.get('stop_reason')}"
        )


def main():
    p = argparse.ArgumentParser(description="JCTC pre-submission matched-budget ablation runner")
    p.add_argument("cmd", choices=["prepare", "smoke", "run-formal"])
    p.add_argument("--out-root", default=str(ROOT / "build" / "jctc_presubmission"))
    p.add_argument("--worker-threads", type=int, default=4)
    p.add_argument("--smoke-wall-s", type=float, default=45.0)
    p.add_argument("--smoke-ckpt-s", type=float, default=15.0)
    p.add_argument("--budget-wall-s", type=float, default=7200.0)
    p.add_argument("--ckpt-every-s", type=float, default=300.0)
    p.add_argument("--replicates", nargs="*", type=int, default=None)
    p.add_argument("--groups", nargs="*", default=None)
    p.add_argument("--skip-done", action="store_true")
    args = p.parse_args()

    if args.cmd == "prepare":
        cmd_prepare(args)
    elif args.cmd == "smoke":
        cmd_smoke(args)
    elif args.cmd == "run-formal":
        cmd_run_formal(args)


if __name__ == "__main__":
    main()
