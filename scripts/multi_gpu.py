"""Multi-GPU parallel orchestration (pressure-point queue & replica box sharding) for Nested MLIP-GCMC."""

import inspect
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np

from mcmc_diagnostics import compute_independent_qst, evaluate_v2_convergence
from nested_mc_engine import (
    K_TO_KJMOL,
    attempt_remc_swaps,
    compute_ess_stats,
)


def parse_gpu_ids(gpus_arg: Optional[Sequence[str]]) -> List[int]:
    """Parse --gpus argument supporting space- or comma-separated non-negative integers."""
    if not gpus_arg:
        return []
    seen = set()
    ids: List[int] = []
    for token in gpus_arg:
        for part in str(token).split(","):
            p = part.strip()
            if not p:
                continue
            val = int(p)
            if val < 0:
                raise ValueError(f"GPU id must be >= 0, got {val}")
            if val not in seen:
                seen.add(val)
                ids.append(val)
    if not ids:
        raise ValueError("No valid GPU IDs parsed from --gpus")
    return ids


def resolve_worker_cuda_visible_devices(gpu_id: int, parent_cvd: Optional[str]) -> str:
    """Resolve child CUDA_VISIBLE_DEVICES honoring any parent CUDA_VISIBLE_DEVICES mask."""
    if not parent_cvd or not parent_cvd.strip():
        return str(int(gpu_id))
    parts = [x.strip() for x in parent_cvd.split(",") if x.strip()]
    if not parts:
        return str(int(gpu_id))
    key = str(int(gpu_id))
    if key in parts:
        return key
    if 0 <= int(gpu_id) < len(parts):
        return parts[int(gpu_id)]
    return key


def split_boxes_across_gpus(total_boxes: int, gpu_ids: Sequence[int]) -> List[Tuple[int, int]]:
    """Split `total_boxes` as evenly as possible across up to `min(len(gpu_ids), total_boxes)` GPUs."""
    total_boxes = int(total_boxes)
    if total_boxes <= 0:
        raise ValueError(f"total_boxes must be > 0, got {total_boxes}")
    if not gpu_ids:
        raise ValueError("gpu_ids must be non-empty")
    n_workers = min(len(gpu_ids), total_boxes)
    base = total_boxes // n_workers
    rem = total_boxes % n_workers
    return [(int(gpu_ids[i]), base + (1 if i < rem else 0)) for i in range(n_workers)]


def split_pressures_across_gpus(n_pressures: int, gpu_ids: Sequence[int]) -> List[Tuple[int, List[int]]]:
    """Contiguously partition `n_pressures` sorted pressure indices across `min(len(gpu_ids), n_pressures)` GPUs."""
    n_pressures = int(n_pressures)
    if n_pressures <= 0:
        raise ValueError(f"n_pressures must be > 0, got {n_pressures}")
    if not gpu_ids:
        raise ValueError("gpu_ids must be non-empty")
    n_workers = min(len(gpu_ids), n_pressures)
    base, rem = divmod(n_pressures, n_workers)
    out: List[Tuple[int, List[int]]] = []
    cur = 0
    for r in range(n_workers):
        cnt = base + (1 if r < rem else 0)
        out.append((int(gpu_ids[r]), list(range(cur, cur + cnt))))
        cur += cnt
    return out


def _configure_worker_env(gpu_id: int, parent_cvd: Optional[str], cpu_threads: int = 4) -> str:
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    visible = resolve_worker_cuda_visible_devices(gpu_id, parent_cvd)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    t_str = str(max(1, int(cpu_threads)))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, t_str)
    return visible


def _init_worker_target(run_cfg: dict, verify_first: bool = False):
    import torch
    from nested_mc_engine import FastMaceD3Target

    torch.set_num_threads(max(1, int(run_cfg.get("worker_threads", 4))))
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Worker GPU CUDA unavailable (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})"
        )
    target = FastMaceD3Target(
        cif_path=run_cfg["cif_path"],
        supercell=tuple(run_cfg["supercell"]),
        model=run_cfg["model"],
        device="cuda:0",
        dispersion=bool(run_cfg["dispersion"]),
        dispersion_xc=run_cfg["dispersion_xc"],
        guest=run_cfg["guest"],
        mace_chunk=run_cfg["mace_chunk"],
        uc_atoms=run_cfg["fw_data"]["uc"],
        dtype=run_cfg.get("dtype", "float32"),
    )
    return target


def merge_replica_shard_results(
    shards: List[dict],
    T_K: float,
    eq_cycles: int,
    wall_elapsed_s: float,
    conv_mmol_g: float,
    diagnostics_fn: Optional[Callable[[dict], dict]] = None,
) -> dict:
    """Exactly merge disjoint replica shards into a single pressure-point result dict."""
    if not shards:
        raise ValueError("shards must be non-empty")

    total_boxes = sum(int(s["n_boxes"]) for s in shards)
    prod_cycles = int(shards[0]["prod_cycles"])
    k_steps = int(shards[0]["k_steps"])
    tot_cycles = int(eq_cycles) + prod_cycles
    dt = max(float(wall_elapsed_s), 1e-6)

    box_moments = [m for s in shards for m in s["box_moments"]]
    box_uptakes = np.array([u for s in shards for u in s["box_uptakes_mmol_g"]], dtype=np.float64)
    box_qst = [None if q is None else float(q) for s in shards for q in s.get("box_qst_kJ_mol", [])]

    mean_up = float(np.mean(box_uptakes))
    sem_up = float(np.std(box_uptakes, ddof=1) / math.sqrt(total_boxes)) if total_boxes > 1 else 0.0
    valid_box_qst = [q for q in box_qst if q is not None]
    per_box_mean_qst = float(np.mean(valid_box_qst)) if valid_box_qst else None
    per_box_sem_qst = (
        float(np.std(valid_box_qst, ddof=1) / math.sqrt(len(valid_box_qst)))
        if len(valid_box_qst) > 1 else (0.0 if len(valid_box_qst) == 1 else None)
    )

    prod_denom = max(1, prod_cycles)
    box_molar_bind = []
    box_molar_qst = []
    for m in box_moments:
        mN = float(m["sum_N"]) / prod_denom if prod_cycles > 0 else 0.0
        mU = float(m["sum_U"]) / prod_denom if prod_cycles > 0 else 0.0
        box_molar_bind.append((-mU / mN) * K_TO_KJMOL if mN > 1e-12 else 0.0)
        box_molar_qst.append((float(T_K) - mU / mN) * K_TO_KJMOL if mN > 1e-12 else 0.0)
    mean_mb = float(np.mean(box_molar_bind))
    mean_mq = float(np.mean(box_molar_qst))
    sem_mq = float(np.std(box_molar_qst, ddof=1) / math.sqrt(total_boxes)) if total_boxes > 1 else 0.0

    def _concat_cyc(key: str, cast_fn):
        if not all(key in s and len(s[key]) == prod_cycles for s in shards):
            return []
        return [[cast_fn(x) for s in shards for x in s[key][cyc]] for cyc in range(prod_cycles)]

    box_N_history = [
        [int(x) for s in shards for x in s["box_N_history"][cyc]]
        for cyc in range(tot_cycles)
    ] if all("box_N_history" in s and len(s["box_N_history"]) == tot_cycles for s in shards) else []
    prod_N_history = _concat_cyc("prod_N_history", int)
    prod_U_history = _concat_cyc("prod_U_history", float)
    prod_U_ref_history = _concat_cyc("prod_U_ref_history", float)
    prod_accepted_history = _concat_cyc("prod_accepted_history", int)
    prod_dN_history = _concat_cyc("prod_dN_history", int)
    prod_proposed_dN_history = _concat_cyc("prod_proposed_dN_history", int)
    prod_oms_occ_history = _concat_cyc("prod_oms_occ_history", int)

    prod_ins_att = int(sum(int(s["prod_ins_att"]) for s in shards))
    prod_ins_acc = int(sum(int(s["prod_ins_acc"]) for s in shards))
    prod_del_att = int(sum(int(s["prod_del_att"]) for s in shards))
    prod_del_acc = int(sum(int(s["prod_del_acc"]) for s in shards))

    qst_res = compute_independent_qst(
        prod_N_history, prod_U_history, T_K=float(T_K),
        prod_ins_acc=prod_ins_acc, prod_del_acc=prod_del_acc,
    )
    ess_stats = compute_ess_stats(prod_N_history, prod_U_history, dt)

    ab_keys = (
        "total_trials", "rejected", "acc_empty_to_empty", "acc_identity_nonempty",
        "acc_same_N_changed", "acc_dN_changed", "nested_trials", "nested_accepted",
        "direct_trials", "direct_accepted",
    )
    if all(isinstance(s.get("acceptance_breakdown"), dict) for s in shards):
        ab = {k: int(sum(int(s["acceptance_breakdown"].get(k, 0)) for s in shards)) for k in ab_keys}
        ab["effective_move_acc_rate"] = float(
            (ab["acc_same_N_changed"] + ab["acc_dN_changed"]) / max(1, ab["total_trials"])
        )
    else:
        ab = {}

    hist = np.array(prod_N_history, dtype=np.float64) * float(conv_mmol_g)
    if hist.ndim == 2 and hist.shape[0] >= 2:
        half = hist.shape[0] // 2
        mean_h1 = float(np.mean(hist[:half]))
        mean_h2 = float(np.mean(hist[half:]))
        drift = abs(mean_h2 - mean_h1)
    elif hist.ndim == 2 and hist.shape[0] == 1:
        mean_h1 = float(np.mean(hist))
        mean_h2 = mean_h1
        drift = 0.0
    else:
        mean_h1 = 0.0
        mean_h2 = 0.0
        drift = 0.0

    def _box_weighted_mean(key: str) -> float:
        return float(sum(float(s[key]) * int(s["n_boxes"]) for s in shards) / total_boxes)

    merged = {
        "n_boxes": int(total_boxes),
        "k_steps": k_steps,
        "cbmc_k": int(shards[0].get("cbmc_k", 128)),
        "p_ins_del": float(shards[0].get("p_ins_del", 0.50)),
        "grid_only": bool(shards[0].get("grid_only", False)),
        "eq_cycles": int(eq_cycles),
        "prod_cycles": prod_cycles,
        "delta_eps_kJ_mol": _box_weighted_mean("delta_eps_kJ_mol"),
        "elapsed_s": dt,
        "shard_elapsed_s": [float(s["elapsed_s"]) for s in shards],
        "uptake_mmol_g": mean_up,
        "uptake_sem_mmol_g": sem_up,
        "box_uptakes_mmol_g": box_uptakes.tolist(),
        "Qst_estimator": "independent_fluctuation",
        "Qst_kJ_mol": qst_res["qst_fluctuation_kJ_mol"],
        "Qst_sem_kJ_mol": qst_res["qst_fluctuation_sem_kJ_mol"],
        "qst_status": qst_res["qst_status"],
        "qst_reasons": qst_res["qst_reasons"],
        "qst_chain_jackknife_sem_kJ_mol": qst_res.get("qst_chain_jackknife_sem_kJ_mol"),
        "qst_block_jackknife_sem_kJ_mol": qst_res.get("qst_block_jackknife_sem_kJ_mol"),
        "per_chain_qst_kJ_mol": qst_res.get("per_chain_qst_kJ_mol", []),
        "Qst_per_box_mean_kJ_mol": per_box_mean_qst,
        "Qst_per_box_sem_kJ_mol": per_box_sem_qst,
        "box_qst_kJ_mol": box_qst,
        "box_moments": box_moments,
        "mean_molar_binding_kJ_mol": mean_mb,
        "mean_molar_qst_kJ_mol": mean_mq,
        "mean_molar_qst_sem_kJ_mol": sem_mq,
        "tau_int_N": ess_stats["tau_int_N"],
        "ess_N": ess_stats["ess_N"],
        "ess_N_per_s": ess_stats["ess_N_per_s"],
        "tau_int_U": ess_stats["tau_int_U"],
        "ess_U": ess_stats["ess_U"],
        "ess_U_per_s": ess_stats["ess_U_per_s"],
        "esjd_N": ess_stats["esjd_N"],
        "sjd_N_per_s": ess_stats["sjd_N_per_s"],
        "dN_moves_per_s": ess_stats["dN_moves_per_s"],
        "outer_acc_rate": _box_weighted_mean("outer_acc_rate"),
        "prod_outer_acc_rate": _box_weighted_mean("prod_outer_acc_rate"),
        "prod_ins_att": prod_ins_att,
        "prod_ins_acc": prod_ins_acc,
        "prod_del_att": prod_del_att,
        "prod_del_acc": prod_del_acc,
        "acceptance_breakdown": ab,
        "mean_inner_acc_per_burst": _box_weighted_mean("mean_inner_acc_per_burst"),
        "effective_mc_steps_per_target_eval": _box_weighted_mean("effective_mc_steps_per_target_eval"),
        "box_N_history": box_N_history,
        "prod_N_history": prod_N_history,
        "prod_U_history": prod_U_history,
        "prod_U_ref_history": prod_U_ref_history,
        "prod_accepted_history": prod_accepted_history,
        "prod_dN_history": prod_dN_history,
        "prod_proposed_dN_history": prod_proposed_dN_history,
        "prod_oms_occ_history": prod_oms_occ_history,
        "trajectory": [],
        "final_states": [fs for s in shards for fs in s["final_states"]],
        "P_bar": float(shards[0]["P_bar"]),
        "T_K": float(T_K),
        "fugacity_Pa": float(shards[0]["fugacity_Pa"]),
        "fw_mass_gmol": float(shards[0].get("fw_mass_gmol", 1000.0 / conv_mmol_g)),
        "first_half_uptake_mmol_g": mean_h1,
        "second_half_uptake_mmol_g": mean_h2,
        "stationarity_drift_mmol_g": drift,
        "mlip_evals": int(sum(int(s["mlip_evals"]) for s in shards)),
        "total_inner_steps": int(sum(int(s["total_inner_steps"]) for s in shards)),
        "parallel_mode": "replica",
    }
    if any("max_N_observed" in s for s in shards):
        merged["max_N_observed"] = int(max(int(s.get("max_N_observed", 0)) for s in shards))
    if "max_mol" in shards[0]:
        merged["max_mol"] = int(shards[0]["max_mol"])
    if "delta_eps_initial_kJ_mol" in shards[0]:
        merged["delta_eps_initial_kJ_mol"] = _box_weighted_mean("delta_eps_initial_kJ_mol")
    if "k_steps_initial" in shards[0]:
        merged["k_steps_initial"] = int(shards[0]["k_steps_initial"])
    if "metadata" in shards[0]:
        merged["metadata"] = shards[0]["metadata"]
    if diagnostics_fn is not None:
        merged["diagnostics"] = diagnostics_fn(merged)
    else:
        merged["diagnostics"] = evaluate_v2_convergence(
            prod_N_history=prod_N_history,
            prod_U_history_kJ=prod_U_history,
            conv_mmol_g=float(conv_mmol_g),
            T_K=float(T_K),
            prod_ins_acc=prod_ins_acc,
            prod_del_acc=prod_del_acc,
            prod_ins_att=prod_ins_att,
            prod_del_att=prod_del_att,
            prod_outer_acc_rate=merged["prod_outer_acc_rate"],
            uptake_sem_mmol_g=sem_up,
            acceptance_breakdown=ab,
            oms_occ_history=prod_oms_occ_history or None,
        )
        from run_mlip_gcmc import evaluate_convergence_diagnostics
        merged["legacy_diagnostics"] = evaluate_convergence_diagnostics(merged)
    return merged


def _distill_worker_main(gpu_id: int, parent_cvd: Optional[str], req: dict, result_q):
    try:
        _configure_worker_env(gpu_id, parent_cvd, req.get("worker_threads", 4))
        import torch
        from mlip_distill import distill_mlip_surrogate
        from nested_mc_engine import FastMaceD3Target

        torch.set_num_threads(max(1, int(req.get("worker_threads", 4))))
        target = FastMaceD3Target(
            cif_path=req["cif_path"],
            supercell=tuple(req["supercell"]),
            model=req["model"],
            device="cuda:0",
            dispersion=bool(req["dispersion"]),
            dispersion_xc=req["dispersion_xc"],
            guest=req["guest"],
            mace_chunk=req.get("mace_chunk"),
            uc_atoms=req["fw_data"]["uc"],
            dtype=req.get("dtype", "float32"),
        )
        distill_kwargs = dict(
            cif_path=req["cif_path"],
            fw_data=req["fw_data"],
            guest=req["guest"],
            supercell=tuple(req["supercell"]),
            target=target,
            grid_dims=tuple(req["grid_dims"]),
            n_1b=int(req["n_1b"]),
            n_2b=int(req["n_2b"]),
            distill_seed=int(req["distill_seed"]),
            cache_dir=req["cache_dir"],
            model_name=req["model"],
            dispersion=bool(req["dispersion"]),
            dispersion_xc=req["dispersion_xc"],
            dtype=req.get("dtype", "float32"),
        )
        only_1b = bool(req.get("only_1b", False))
        if "only_1b" in inspect.signature(distill_mlip_surrogate).parameters:
            distill_kwargs["only_1b"] = only_1b
        elif only_1b:
            raise RuntimeError("distill_mlip_surrogate does not support only_1b=True in this version of mlip_distill.py")
        res = distill_mlip_surrogate(**distill_kwargs)
        result_q.put(("ok", res["cache_npz_path"]))
    except BaseException as exc:
        import traceback
        result_q.put(("err", f"[Distill GPU {gpu_id}] {exc}\n{traceback.format_exc()}"))


def run_distill_on_gpu(gpu_id: int, req: dict) -> dict:
    """Run one-time MLIP 1B+2B distillation in an isolated spawn subprocess before multi-GPU dispatch."""
    from mlip_distill import load_distill_cache

    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    p = ctx.Process(target=_distill_worker_main, args=(int(gpu_id), parent_cvd, req, result_q))
    p.start()
    msg = result_q.get()
    p.join(timeout=30.0)
    if msg[0] == "err":
        raise RuntimeError(msg[1])
    return load_distill_cache(msg[1])


def _pressure_worker_main(
    worker_rank: int,
    gpu_id: int,
    parent_cvd: Optional[str],
    run_cfg: dict,
    task_q,
    result_q,
):
    try:
        visible = _configure_worker_env(gpu_id, parent_cvd, run_cfg.get("worker_threads", 4))
        from mlip_distill import load_distill_cache
        from run_mlip_gcmc import run_pressure_point

        grid_only = bool(run_cfg.get("grid_only", False))
        target = None if grid_only else _init_worker_target(run_cfg)
        distill_res = (
            load_distill_cache(run_cfg["distill_cache_path"])
            if run_cfg.get("distill_cache_path")
            else None
        )
        consistency_checked = False
        while True:
            item = task_q.get()
            if item is None:
                break
            idx, p_bar = item
            print(f"\n[GPU {gpu_id} (visible={visible})] Pressure Point idx={idx+1}: {p_bar} bar")
            t0 = time.perf_counter()
            pt = run_pressure_point(
                fw_data=run_cfg["fw_data"],
                guest=run_cfg["guest"],
                supercell=run_cfg["supercell"],
                T_K=run_cfg["T_K"],
                P_bar=p_bar,
                ff_mode=run_cfg["ff_mode"],
                pol_mode=run_cfg["pol_mode"],
                grid_dims=run_cfg["grid_dims"],
                n_boxes=run_cfg["n_boxes"],
                max_mol=run_cfg["max_mol"],
                k_steps=run_cfg["k_steps"],
                eq_cycles=run_cfg["eq_cycles"],
                prod_cycles=run_cfg["prod_cycles"],
                seed=int(run_cfg["base_seed"]) + int(idx) * 100,
                shift_mode=run_cfg["shift_mode"],
                target=target,
                pilot_cycles=run_cfg["pilot_cycles"],
                shift_override=run_cfg["shift_override"],
                verify_consistency=(worker_rank == 0 and not consistency_checked and target is not None),
                device_id=0,
                init_mode=run_cfg.get("init_mode", "auto"),
                init_loading=float(run_cfg.get("init_loading", 8.0)),
                checkpoint_dir=run_cfg.get("checkpoint_dir", ""),
                checkpoint_every=int(run_cfg.get("checkpoint_every", 0)),
                resume_mode=run_cfg.get("resume_mode", "none"),
                snapshot_stride=int(run_cfg.get("snapshot_stride", 0)),
                run_metadata=run_cfg.get("run_metadata"),
                grid_only=grid_only,
                cbmc_k=int(run_cfg.get("cbmc_k", 128)),
                p_ins_del=float(run_cfg.get("p_ins_del", 0.50)),
                distill_res=distill_res,
                adapt_shift_frac=float(run_cfg.get("adapt_shift_frac", 0.0)),
                adapt_k_frac=float(run_cfg.get("adapt_k_frac", 0.0)),
                k_min=int(run_cfg.get("k_min", 2)),
                k_max=int(run_cfg.get("k_max", 24)),
                target_outer_acc=float(run_cfg.get("target_outer_acc", 0.35)),
                adapt_window=int(run_cfg.get("adapt_window", 20)),
            )
            consistency_checked = True
            if target is not None and getattr(target, "model_sha256", None):
                if isinstance(pt.get("metadata"), dict) and not pt["metadata"].get("model_sha256"):
                    pt["metadata"] = dict(pt["metadata"])
                    pt["metadata"]["model_sha256"] = target.model_sha256
            pt["gpu_id"] = int(gpu_id)
            pt["parallel_mode"] = "pressure"
            pt["point_wall_s"] = float(time.perf_counter() - t0)
            result_q.put(("ok", idx, pt))
    except BaseException as exc:
        import traceback
        result_q.put(("err", gpu_id, f"[GPU {gpu_id}] {exc}\n{traceback.format_exc()}"))


def _replica_worker_main(
    shard_rank: int,
    gpu_id: int,
    n_boxes_shard: int,
    parent_cvd: Optional[str],
    run_cfg: dict,
    cmd_q,
    result_q,
):
    try:
        visible = _configure_worker_env(gpu_id, parent_cvd, run_cfg.get("worker_threads", 4))
        from mlip_distill import load_distill_cache
        from run_mlip_gcmc import run_pressure_point

        grid_only = bool(run_cfg.get("grid_only", False))
        target = None if grid_only else _init_worker_target(run_cfg)
        distill_res = (
            load_distill_cache(run_cfg["distill_cache_path"])
            if run_cfg.get("distill_cache_path")
            else None
        )
        consistency_checked = False
        while True:
            item = cmd_q.get()
            if item is None:
                break
            idx, p_bar = item
            print(
                f"\n[GPU {gpu_id} (visible={visible}) shard={shard_rank} boxes={n_boxes_shard}] "
                f"Pressure Point idx={idx+1}: {p_bar} bar"
            )
            pt = run_pressure_point(
                fw_data=run_cfg["fw_data"],
                guest=run_cfg["guest"],
                supercell=run_cfg["supercell"],
                T_K=run_cfg["T_K"],
                P_bar=p_bar,
                ff_mode=run_cfg["ff_mode"],
                pol_mode=run_cfg["pol_mode"],
                grid_dims=run_cfg["grid_dims"],
                n_boxes=n_boxes_shard,
                max_mol=run_cfg["max_mol"],
                k_steps=run_cfg["k_steps"],
                eq_cycles=run_cfg["eq_cycles"],
                prod_cycles=run_cfg["prod_cycles"],
                seed=int(run_cfg["base_seed"]) + int(idx) * 100 + shard_rank * 10007,
                shift_mode=run_cfg["shift_mode"],
                target=target,
                pilot_cycles=run_cfg["pilot_cycles"],
                shift_override=run_cfg["shift_override"],
                verify_consistency=(shard_rank == 0 and not consistency_checked and target is not None),
                device_id=0,
                init_mode=run_cfg.get("init_mode", "auto"),
                init_loading=float(run_cfg.get("init_loading", 8.0)),
                checkpoint_dir=run_cfg.get("checkpoint_dir", ""),
                checkpoint_every=int(run_cfg.get("checkpoint_every", 0)),
                checkpoint_tag=f"shard{shard_rank}",
                resume_mode=run_cfg.get("resume_mode", "none"),
                snapshot_stride=int(run_cfg.get("snapshot_stride", 0)),
                run_metadata=run_cfg.get("run_metadata"),
                grid_only=grid_only,
                cbmc_k=int(run_cfg.get("cbmc_k", 128)),
                p_ins_del=float(run_cfg.get("p_ins_del", 0.50)),
                distill_res=distill_res,
                adapt_shift_frac=float(run_cfg.get("adapt_shift_frac", 0.0)),
                adapt_k_frac=float(run_cfg.get("adapt_k_frac", 0.0)),
                k_min=int(run_cfg.get("k_min", 2)),
                k_max=int(run_cfg.get("k_max", 24)),
                target_outer_acc=float(run_cfg.get("target_outer_acc", 0.35)),
                adapt_window=int(run_cfg.get("adapt_window", 20)),
            )
            consistency_checked = True
            if target is not None and getattr(target, "model_sha256", None):
                if isinstance(pt.get("metadata"), dict) and not pt["metadata"].get("model_sha256"):
                    pt["metadata"] = dict(pt["metadata"])
                    pt["metadata"]["model_sha256"] = target.model_sha256
            result_q.put(("ok", shard_rank, gpu_id, idx, pt))
    except BaseException as exc:
        import traceback
        result_q.put(("err", shard_rank, gpu_id, -1, f"[GPU {gpu_id} shard={shard_rank}] {exc}\n{traceback.format_exc()}"))


def run_pressures_parallel(
    pending: List[Tuple[int, float]],
    gpu_ids: List[int],
    run_cfg: dict,
    on_point_done: Optional[Callable[[dict], None]] = None,
) -> List[dict]:
    """Run independent pressure points across a pool of GPU worker processes."""
    if not pending:
        return []
    active_gpus = list(gpu_ids[: min(len(gpu_ids), len(pending))])
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    ctx = mp.get_context("spawn")
    task_q = ctx.Queue()
    result_q = ctx.Queue()

    for item in pending:
        task_q.put(item)
    for _ in active_gpus:
        task_q.put(None)

    procs = [
        ctx.Process(
            target=_pressure_worker_main,
            args=(rank, gid, parent_cvd, run_cfg, task_q, result_q),
        )
        for rank, gid in enumerate(active_gpus)
    ]
    for p in procs:
        p.start()

    results = []
    try:
        while len(results) < len(pending):
            try:
                msg = result_q.get(timeout=2.0)
            except queue.Empty:
                dead_bad = [p for p in procs if not p.is_alive() and p.exitcode not in (0, None)]
                if dead_bad:
                    raise RuntimeError(f"Pressure worker crashed unexpectedly with exit code {dead_bad[0].exitcode}")
                continue
            status, payload_a, payload_b = msg
            if status == "err":
                raise RuntimeError(payload_b)
            results.append(payload_b)
            if on_point_done is not None:
                on_point_done(payload_b)
    except BaseException:
        for p in procs:
            if p.is_alive():
                p.terminate()
        raise
    finally:
        for p in procs:
            p.join(timeout=10.0)

    return sorted(results, key=lambda pt: float(pt["P_bar"]))


def run_replicas_parallel(
    pending: List[Tuple[int, float]],
    gpu_ids: List[int],
    run_cfg: dict,
    on_point_done: Optional[Callable[[dict], None]] = None,
) -> List[dict]:
    """Split `n_boxes` replicas across GPUs for each pressure point and merge exact statistics."""
    if not pending:
        return []
    shard_plan = split_boxes_across_gpus(int(run_cfg["n_boxes"]), gpu_ids)
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    ctx = mp.get_context("spawn")
    cmd_qs = [ctx.Queue() for _ in shard_plan]
    result_q = ctx.Queue()

    procs = [
        ctx.Process(
            target=_replica_worker_main,
            args=(rank, gid, n_b, parent_cvd, run_cfg, cmd_qs[rank], result_q),
        )
        for rank, (gid, n_b) in enumerate(shard_plan)
    ]
    for p in procs:
        p.start()

    results = []
    try:
        for idx, p_bar in pending:
            t_point_0 = time.perf_counter()
            for q in cmd_qs:
                q.put((idx, p_bar))
            shards_by_rank = {}
            while len(shards_by_rank) < len(shard_plan):
                try:
                    msg = result_q.get(timeout=2.0)
                except queue.Empty:
                    dead_bad = [p for p in procs if not p.is_alive() and p.exitcode not in (0, None)]
                    if dead_bad:
                        raise RuntimeError(f"Replica worker crashed unexpectedly with exit code {dead_bad[0].exitcode}")
                    continue
                status, s_rank, _gid, _idx, data = msg
                if status == "err":
                    raise RuntimeError(data)
                shards_by_rank[s_rank] = data
            point_wall_s = float(time.perf_counter() - t_point_0)
            ordered_shards = [shards_by_rank[r] for r in range(len(shard_plan))]
            mc_wall_s = max(float(s["elapsed_s"]) for s in ordered_shards)
            conv_mmol_g = 1000.0 / float(ordered_shards[0]["fw_mass_gmol"])
            merged = merge_replica_shard_results(
                shards=ordered_shards,
                T_K=float(run_cfg["T_K"]),
                eq_cycles=int(run_cfg["eq_cycles"]),
                wall_elapsed_s=mc_wall_s,
                conv_mmol_g=conv_mmol_g,
            )
            merged["point_wall_s"] = point_wall_s
            merged["gpu_ids"] = [gid for gid, _ in shard_plan]
            diag = merged["diagnostics"]
            q_str = (
                f"{merged['Qst_kJ_mol']:.2f} +- {merged['Qst_sem_kJ_mol']:.2f}"
                if merged.get("Qst_kJ_mol") is not None and merged.get("Qst_sem_kJ_mol") is not None
                else f"N/A ({merged.get('qst_status', 'unknown')})"
            )
            print(
                f"\n  [4xGPU Replica Merged P={p_bar:.4g} bar] "
                f"Uptake={merged['uptake_mmol_g']:.4f} +- {merged['uptake_sem_mmol_g']:.4f} mmol/g | "
                f"Fluc Qst={q_str} | "
                f"tauN={merged['tau_int_N']:.1f} ESS/s={merged['ess_N_per_s']:.2f} | "
                f"mc_wall={mc_wall_s:.1f}s point_wall={point_wall_s:.1f}s | diag={diag.get('status', 'unknown')}"
            )
            results.append(merged)
            if on_point_done is not None:
                on_point_done(merged)

        for q in cmd_qs:
            q.put(None)
    except BaseException:
        for p in procs:
            if p.is_alive():
                p.terminate()
        raise
    finally:
        for p in procs:
            p.join(timeout=10.0)

    return sorted(results, key=lambda pt: float(pt["P_bar"]))


def _remc_worker_main(
    worker_rank: int,
    gpu_id: int,
    assigned_items: List[Tuple[int, int, float]],
    parent_cvd: Optional[str],
    run_cfg: dict,
    cmd_q,
    result_q,
):
    """Multi-GPU lockstep REMC worker owning a contiguous subset of sorted pressures."""
    proposers = {}
    try:
        visible = _configure_worker_env(gpu_id, parent_cvd, run_cfg.get("worker_threads", 4))
        from mlip_distill import load_distill_cache
        from nested_mc_engine import NestedMCChain, _serialize_single_box_state
        from run_mlip_gcmc import evaluate_convergence_diagnostics, prepare_pressure_proposer

        grid_only = bool(run_cfg.get("grid_only", False))
        target = None if grid_only else _init_worker_target(run_cfg)
        distill_res = (
            load_distill_cache(run_cfg["distill_cache_path"])
            if run_cfg.get("distill_cache_path")
            else None
        )

        chains = {}
        preps = {}
        p_by_m = {}
        ready_info = {}
        consistency_checked = False
        eff_adapt_shift = (
            0.0
            if (grid_only or run_cfg.get("shift_override") is not None)
            else float(run_cfg.get("adapt_shift_frac", 0.0))
        )
        eff_adapt_k = 0.0 if grid_only else float(run_cfg.get("adapt_k_frac", 0.0))

        for m_idx, orig_idx, p_bar in assigned_items:
            print(
                f"\n[REMC GPU {gpu_id} (visible={visible}) rank={worker_rank}] "
                f"Preparing P[{m_idx}]={p_bar} bar"
            )
            prep = prepare_pressure_proposer(
                fw_data=run_cfg["fw_data"],
                guest=run_cfg["guest"],
                supercell=run_cfg["supercell"],
                T_K=float(run_cfg["T_K"]),
                P_bar=float(p_bar),
                ff_mode=int(run_cfg["ff_mode"]),
                pol_mode=int(run_cfg["pol_mode"]),
                grid_dims=run_cfg["grid_dims"],
                n_boxes=int(run_cfg["n_boxes"]),
                max_mol=int(run_cfg["max_mol"]),
                k_steps=int(run_cfg["k_steps"]),
                seed=int(run_cfg["base_seed"]) + int(orig_idx) * 100,
                shift_mode=run_cfg["shift_mode"],
                target=target,
                pilot_cycles=int(run_cfg["pilot_cycles"]),
                shift_override=run_cfg["shift_override"],
                verify_consistency=(worker_rank == 0 and not consistency_checked and target is not None),
                device_id=0,
                init_mode=run_cfg.get("init_mode", "auto"),
                init_loading=float(run_cfg.get("init_loading", 8.0)),
                grid_only=grid_only,
                cbmc_k=int(run_cfg.get("cbmc_k", 128)),
                p_ins_del=float(run_cfg.get("p_ins_del", 0.50)),
                distill_res=distill_res,
            )
            consistency_checked = True
            prop = prep["prop"]
            proposers[m_idx] = prop
            preps[m_idx] = prep
            p_by_m[m_idx] = float(p_bar)
            chains[m_idx] = NestedMCChain(
                proposer=prop,
                target_fn=None,
                k_steps=int(run_cfg["k_steps"]),
                eq_cycles=int(run_cfg["eq_cycles"]),
                prod_cycles=int(run_cfg["prod_cycles"]),
                seed=int(run_cfg["base_seed"]) + int(orig_idx) * 100 + 17,
                log_every=0,
                batch_target_fn=None if grid_only else target.eval_batch,
                oms_metal_pos=prep["oms_metal_pos"] if len(prep["oms_metal_pos"]) > 0 else None,
                oms_cell_rows=prep["oms_cell_rows"],
                guest_local_pos=prep["guest_local_pos"],
                snapshot_stride=int(run_cfg.get("snapshot_stride", 0)),
                grid_only=grid_only,
                adapt_shift_frac=eff_adapt_shift,
                adapt_k_frac=eff_adapt_k,
                k_min=int(run_cfg.get("k_min", 2)),
                k_max=int(run_cfg.get("k_max", 24)),
                target_outer_acc=float(run_cfg.get("target_outer_acc", 0.35)),
                adapt_window=int(run_cfg.get("adapt_window", 20)),
            )
            ready_info[m_idx] = {
                "fugacity_Pa": float(prop.f_Pa),
                "N_curr": chains[m_idx].N_curr.tolist(),
                "fw_mass_gmol": float(prop.fw_mass_gmol),
                "model_sha256": getattr(target, "model_sha256", None) if target is not None else None,
            }

        result_q.put(("ready", worker_rank, ready_info))

        while True:
            cmd = cmd_q.get()
            if cmd is None or cmd[0] == "stop":
                break
            op = cmd[0]
            if op == "step":
                n_steps = int(cmd[1])
                n_map = {}
                for m_idx in chains:
                    chains[m_idx].step_cycles(n_steps)
                    n_map[m_idx] = chains[m_idx].N_curr.tolist()
                result_q.put(("stepped", worker_rank, n_map))
            elif op == "get_boxes":
                req_list = cmd[1]
                payload_boxes = {}
                for m_idx, b in req_list:
                    ch = chains[int(m_idx)]
                    bi = int(b)
                    payload_boxes[(int(m_idx), bi)] = (
                        _serialize_single_box_state(ch.curr[bi]),
                        int(ch.N_curr[bi]),
                        float(ch.U_tgt_curr[bi]),
                    )
                result_q.put(("boxes", worker_rank, payload_boxes))
            elif op == "set_boxes":
                updates_by_m = cmd[1]
                for m_idx, box_map in updates_by_m.items():
                    chains[int(m_idx)].apply_swapped_boxes(box_map)
                result_q.put(("set_ok", worker_rank, True))
            elif op == "finalize":
                swap_stats_by_m, wall_s, n_pressures_total, swap_every = cmd[1], float(cmd[2]), int(cmd[3]), int(cmd[4])
                out_by_m = {}
                eq_cycles = int(run_cfg["eq_cycles"])
                T_K = float(run_cfg["T_K"])
                for m_idx, ch in chains.items():
                    prep = preps[m_idx]
                    prop = proposers[m_idx]
                    conv_mmol_g = prep["conv_mmol_g"]
                    res = ch.finalize(elapsed_s=wall_s)
                    hist = np.asarray(res["box_N_history"], dtype=np.float64) * conv_mmol_g
                    prod_hist = hist[eq_cycles:]
                    if prod_hist.ndim == 2 and prod_hist.shape[0] >= 2:
                        half = prod_hist.shape[0] // 2
                        mean_h1 = float(np.mean(prod_hist[:half]))
                        mean_h2 = float(np.mean(prod_hist[half:]))
                        drift = abs(mean_h2 - mean_h1)
                    elif prod_hist.ndim == 2 and prod_hist.shape[0] == 1:
                        mean_h1 = float(np.mean(prod_hist))
                        mean_h2 = mean_h1
                        drift = 0.0
                    else:
                        mean_h1, mean_h2, drift = 0.0, 0.0, 0.0

                    sw = swap_stats_by_m[m_idx]
                    res["P_bar"] = float(p_by_m[m_idx])
                    res["T_K"] = T_K
                    res["fugacity_Pa"] = float(prop.f_Pa)
                    res["fw_mass_gmol"] = float(prop.fw_mass_gmol)
                    res["first_half_uptake_mmol_g"] = mean_h1
                    res["second_half_uptake_mmol_g"] = mean_h2
                    res["stationarity_drift_mmol_g"] = drift
                    res["mlip_evals"] = 0 if grid_only else int(res["target_evals"])
                    res["cbmc_k"] = int(run_cfg.get("cbmc_k", 128))
                    res["p_ins_del"] = float(run_cfg.get("p_ins_del", 0.50))
                    res["prop_grid_wall_s"] = prep["prop_grid_wall_s"]
                    res["shift_calib_wall_s"] = prep["shift_calib_wall_s"]
                    res["shift_calibration"] = prep["shift_calib"]
                    res["init_state_summary"] = prep["init_state_summary"]
                    res["gpu_id"] = int(gpu_id)
                    res["parallel_mode"] = "remc"
                    res["remc_swap_every"] = swap_every
                    res["remc_n_pressures"] = n_pressures_total
                    res["remc_isotherm_wall_s"] = wall_s
                    res["remc_swap_att_up"] = int(sw["prod_att_up"])
                    res["remc_swap_acc_up"] = int(sw["prod_acc_up"])
                    res["remc_swap_rate_up"] = sw["prod_rate_up"]
                    res["remc_total_swap_att_up"] = int(sw["tot_att_up"])
                    res["remc_total_swap_acc_up"] = int(sw["tot_acc_up"])
                    res["remc_total_swap_rate_up"] = sw["tot_rate_up"]
                    if run_cfg.get("run_metadata") is not None:
                        meta_copy = dict(run_cfg["run_metadata"])
                        if target is not None and getattr(target, "model_sha256", None) and not meta_copy.get("model_sha256"):
                            meta_copy["model_sha256"] = target.model_sha256
                        res["metadata"] = meta_copy
                    res["diagnostics"] = evaluate_v2_convergence(
                        prod_N_history=res["prod_N_history"],
                        prod_U_history_kJ=res["prod_U_history"],
                        conv_mmol_g=conv_mmol_g,
                        T_K=T_K,
                        prod_ins_acc=res["prod_ins_acc"],
                        prod_del_acc=res["prod_del_acc"],
                        prod_ins_att=res["prod_ins_att"],
                        prod_del_att=res["prod_del_att"],
                        prod_outer_acc_rate=res["prod_outer_acc_rate"],
                        uptake_sem_mmol_g=res["uptake_sem_mmol_g"],
                        acceptance_breakdown=res.get("acceptance_breakdown"),
                        oms_occ_history=res.get("prod_oms_occ_history") or None,
                    )
                    res["legacy_diagnostics"] = evaluate_convergence_diagnostics(res)
                    out_by_m[m_idx] = res
                result_q.put(("finalized", worker_rank, out_by_m))
                break
            else:
                raise RuntimeError(f"Unknown REMC worker command: {op!r}")
    except BaseException as exc:
        import traceback
        result_q.put(("err", worker_rank, f"[REMC GPU {gpu_id} rank={worker_rank}] {exc}\n{traceback.format_exc()}"))
    finally:
        for prop in proposers.values():
            try:
                prop.close()
            except Exception:
                pass


def run_remc_parallel(
    pending: List[Tuple[int, float]],
    gpu_ids: List[int],
    run_cfg: dict,
    on_point_done: Optional[Callable[[dict], None]] = None,
) -> List[dict]:
    """Run lockstep multi-GPU Grand Canonical Parallel Tempering across sorted pressures."""
    if len(pending) < 2:
        raise ValueError(f"REMC requires at least 2 pressures, got {len(pending)}")

    sorted_items = sorted(pending, key=lambda x: float(x[1]))
    M = len(sorted_items)
    plan = split_pressures_across_gpus(M, gpu_ids)
    n_workers = len(plan)
    m_to_rank = {}
    worker_assignments = []
    for rank, (gid, m_list) in enumerate(plan):
        assigned = []
        for m in m_list:
            m_to_rank[m] = rank
            orig_idx, p_bar = sorted_items[m]
            assigned.append((m, int(orig_idx), float(p_bar)))
        worker_assignments.append((rank, gid, assigned))

    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    ctx = mp.get_context("spawn")
    cmd_qs = [ctx.Queue() for _ in range(n_workers)]
    result_q = ctx.Queue()

    procs = [
        ctx.Process(
            target=_remc_worker_main,
            args=(rank, gid, assigned, parent_cvd, run_cfg, cmd_qs[rank], result_q),
        )
        for rank, gid, assigned in worker_assignments
    ]
    for p in procs:
        p.start()

    def _collect_from(expected_tag: str, expected_ranks: set) -> dict:
        got = {}
        while len(got) < len(expected_ranks):
            try:
                msg = result_q.get(timeout=2.0)
            except queue.Empty:
                dead_bad = [p for p in procs if not p.is_alive() and p.exitcode not in (0, None)]
                if dead_bad:
                    raise RuntimeError(f"REMC worker crashed unexpectedly with exit code {dead_bad[0].exitcode}")
                continue
            tag, r_rank, payload = msg
            if tag == "err":
                raise RuntimeError(payload)
            if tag != expected_tag:
                raise RuntimeError(f"REMC protocol mismatch: expected {expected_tag!r}, got {tag!r}")
            got[int(r_rank)] = payload
        return got

    all_ranks = set(range(n_workers))
    try:
        ready_by_rank = _collect_from("ready", all_ranks)
        fugacities_Pa = [0.0] * M
        for r_info in ready_by_rank.values():
            for m, info in r_info.items():
                fugacities_Pa[int(m)] = float(info["fugacity_Pa"])

        n_b = int(run_cfg["n_boxes"])
        eq_cycles = int(run_cfg["eq_cycles"])
        prod_cycles = int(run_cfg["prod_cycles"])
        total_cycles = eq_cycles + prod_cycles
        swap_stride = max(1, int(run_cfg.get("remc_swap_every", 2)))
        base_seed = int(run_cfg["base_seed"])
        ladder_rngs = [np.random.default_rng(base_seed + 7919 * (b + 1)) for b in range(n_b)]

        prod_swap_att_up = np.zeros(M, dtype=np.int64)
        prod_swap_acc_up = np.zeros(M, dtype=np.int64)
        tot_swap_att_up = np.zeros(M, dtype=np.int64)
        tot_swap_acc_up = np.zeros(M, dtype=np.int64)

        t_mc0 = time.perf_counter()
        cyc_done = 0
        swap_round = 0
        log_every = max(50, total_cycles // 5) if total_cycles > 0 else 0
        next_log = log_every

        while cyc_done < total_cycles:
            batch = min(swap_stride, total_cycles - cyc_done)
            if cyc_done < eq_cycles and cyc_done + batch > eq_cycles:
                batch = eq_cycles - cyc_done

            for q in cmd_qs:
                q.put(("step", batch))
            stepped = _collect_from("stepped", all_ranks)
            N_mat = [None] * M
            for r_map in stepped.values():
                for m, n_vec in r_map.items():
                    N_mat[int(m)] = n_vec
            cyc_done += batch

            if cyc_done < total_cycles:
                parity = swap_round % 2
                swap_round += 1
                accepted_pairs, att_up, acc_up = attempt_remc_swaps(
                    N_mat, fugacities_Pa, parity=parity, ladder_rngs=ladder_rngs
                )
                tot_swap_att_up += att_up
                tot_swap_acc_up += acc_up
                if cyc_done > eq_cycles:
                    prod_swap_att_up += att_up
                    prod_swap_acc_up += acc_up

                if accepted_pairs:
                    req_by_rank = {}
                    for m_lo, m_hi, b in accepted_pairs:
                        req_by_rank.setdefault(m_to_rank[m_lo], []).append((m_lo, b))
                        req_by_rank.setdefault(m_to_rank[m_hi], []).append((m_hi, b))
                    for r, req_list in req_by_rank.items():
                        cmd_qs[r].put(("get_boxes", req_list))
                    boxes_by_rank = _collect_from("boxes", set(req_by_rank.keys()))
                    fetched = {}
                    for b_dict in boxes_by_rank.values():
                        fetched.update(b_dict)

                    upd_by_rank = {}
                    for m_lo, m_hi, b in accepted_pairs:
                        upd_by_rank.setdefault(m_to_rank[m_lo], {}).setdefault(m_lo, {})[b] = fetched[(m_hi, b)]
                        upd_by_rank.setdefault(m_to_rank[m_hi], {}).setdefault(m_hi, {})[b] = fetched[(m_lo, b)]
                    for r, updates_by_m in upd_by_rank.items():
                        cmd_qs[r].put(("set_boxes", updates_by_m))
                    _collect_from("set_ok", set(upd_by_rank.keys()))

            if next_log > 0 and cyc_done >= next_log:
                elapsed = time.perf_counter() - t_mc0
                rates = [
                    f"{tot_swap_acc_up[m] / max(1, tot_swap_att_up[m]) * 100:.0f}%"
                    for m in range(M - 1)
                ]
                means = [f"{float(np.mean(N_mat[m])):.1f}" for m in range(M)]
                print(
                    f"  [Multi-GPU REMC cyc {cyc_done:4d}/{total_cycles}] <N>={means} | "
                    f"SwapRates={rates} ({elapsed:.1f}s)"
                )
                while next_log <= cyc_done:
                    next_log += log_every

        wall_s = float(time.perf_counter() - t_mc0)
        swap_stats_by_m = {}
        for m in range(M):
            ap, cp = int(prod_swap_att_up[m]), int(prod_swap_acc_up[m])
            at, ct = int(tot_swap_att_up[m]), int(tot_swap_acc_up[m])
            swap_stats_by_m[m] = {
                "prod_att_up": ap,
                "prod_acc_up": cp,
                "prod_rate_up": (float(cp) / float(ap)) if ap > 0 else (0.0 if m < M - 1 else None),
                "tot_att_up": at,
                "tot_acc_up": ct,
                "tot_rate_up": (float(ct) / float(at)) if at > 0 else (0.0 if m < M - 1 else None),
            }

        for q in cmd_qs:
            q.put(("finalize", swap_stats_by_m, wall_s, M, swap_stride))
        fin_by_rank = _collect_from("finalized", all_ranks)

        results = []
        for out_map in fin_by_rank.values():
            for pt in out_map.values():
                pt["point_wall_s"] = wall_s
                results.append(pt)
        results = sorted(results, key=lambda pt: float(pt["P_bar"]))
        for pt in results:
            if on_point_done is not None:
                on_point_done(pt)
        return results
    except BaseException:
        for p in procs:
            if p.is_alive():
                p.terminate()
        raise
    finally:
        for p in procs:
            p.join(timeout=10.0)
