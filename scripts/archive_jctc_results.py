"""JCTC Results Archiver, Independent Diagnostics Recomputer, and Reproducibility Auditor.

Scans a directory of raw MLIP-GCMC / Direct MLIP-GCMC / Widom JSON outputs, validates every run,
independently recomputes Vehtari R-hat, bulk/tail ESS, MCSE, half-split drift, and fluctuation Q_st
directly from full production trajectories (`prod_N_history`, `prod_U_history`, `prod_oms_occ_history`),
and audits:
  - Sampling efficiency (`ess_bulk_N_per_gpu_s`, `time_to_mcse_0p1_s` sampling / amortized / cold-start)
  - Proposal symmetry ratio (`prod_ins_att / max(1, prod_del_att)`)
  - Capacity saturation safety (`max_N_observed` vs `max_mol`)
  - Honest 2B deployed hold-out fidelity (`mae_2b_deployed_holdout_kJ_mol`, `mae_2b_classical_holdout_kJ_mol`, `n_2b_val`)
  - Provenance completeness (`model_sha256` non-null audit)
  - REMC pressure-pair swap acceptance rates

Outputs:
  1. `jctc_summary.json` (structured summary linked to full uncompressed trajectories)
  2. `jctc_traces.npz` (companion compressed NumPy archive of 100% full production traces so every
     diagnostic is independently recomputable without raw JSONs)
  3. `jctc_audit_report.md` (publication-ready Markdown audit report and tables)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mcmc_diagnostics import K_TO_KJMOL, compute_independent_qst, evaluate_v2_convergence


KNOWN_FW_MASS_GMOL = {
    ("MgMOF-74.cif", (1, 1, 2)): 4368.744,
    ("MgMOF-74.cif", (1, 1, 4)): 8737.488,
    ("NiMOF-74.cif", (1, 1, 2)): 5610.672,
    ("Cu-BTC.cif", (1, 1, 1)): 9677.824,
    ("IRMOF-1.cif", (1, 1, 1)): 6158.144,
}

MOF_NAME_MAP = {
    "mgmof-74": "Mg-MOF-74",
    "mgmof74": "Mg-MOF-74",
    "nimof-74": "Ni-MOF-74",
    "nimof74": "Ni-MOF-74",
    "cu-btc": "Cu-BTC",
    "cubtc": "Cu-BTC",
    "hkust-1": "Cu-BTC",
    "hkust1": "Cu-BTC",
    "irmof-1": "IRMOF-1",
    "irmof1": "IRMOF-1",
}

EXP_MANIFEST_KEY_MAP = {
    "Mg-MOF-74": "Mg-MOF-74",
    "Ni-MOF-74": "Ni-MOF-74",
    "Cu-BTC": "HKUST-1",
    "IRMOF-1": "IRMOF-1",
}


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sanitize_key(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(s)).strip("_")


def _canonical_mof_name(cif_path: str) -> str:
    stem = Path(str(cif_path)).stem.lower()
    if stem in MOF_NAME_MAP:
        return MOF_NAME_MAP[stem]
    cleaned = stem.replace("_", "").replace("-", "")
    for k, v in MOF_NAME_MAP.items():
        if k.replace("-", "") == cleaned:
            return v
    return Path(str(cif_path)).stem


def _downsample_trace(arr_2d: np.ndarray, max_points: int = 250) -> List[List[float]]:
    if arr_2d.ndim != 2 or arr_2d.shape[0] == 0:
        return []
    n = arr_2d.shape[0]
    if n <= max_points:
        return [[float(v) for v in row] for row in arr_2d]
    idx = np.linspace(0, n - 1, max_points, dtype=int)
    return [[float(v) for v in arr_2d[i]] for i in idx]


def _resolve_fw_mass_gmol(run: dict, pt: dict, prod_n: Optional[np.ndarray]) -> Optional[float]:
    for cand in (pt.get("fw_mass_gmol"), run.get("fw_mass_gmol")):
        if cand is not None and float(cand) > 0.0:
            return float(cand)
    if prod_n is not None and prod_n.size > 0:
        mean_n = float(np.mean(prod_n))
        stored_up = pt.get("uptake_mmol_g")
        if mean_n > 1e-9 and stored_up is not None and float(stored_up) > 1e-9:
            return float(1000.0 * mean_n / float(stored_up))
    cif = str(run.get("cif", ""))
    sc = tuple(int(x) for x in run.get("supercell", [1, 1, 1]))
    key = (Path(cif).name, sc)
    if key in KNOWN_FW_MASS_GMOL:
        return KNOWN_FW_MASS_GMOL[key]
    # Last-resort ASE lookup if CIF exists locally
    try:
        cif_p = Path(cif)
        if not cif_p.is_file():
            cif_p = ROOT_DIR / cif
        if cif_p.is_file():
            from ase.io import read as ase_read
            uc = ase_read(str(cif_p))
            return float(np.sum(uc.repeat(sc).get_masses()))
    except Exception:
        pass
    return None


def _extract_model_sha256(run: dict, pt: Optional[dict] = None) -> Optional[str]:
    candidates = []
    if pt is not None:
        candidates.append((pt.get("metadata") or {}).get("model_sha256"))
        candidates.append(pt.get("model_sha256"))
    candidates.extend([
        (run.get("metadata") or {}).get("model_sha256"),
        (run.get("distillation") or {}).get("model_sha256"),
        run.get("model_sha256"),
    ])
    if pt is None:
        for p in run.get("points") or []:
            candidates.append((p.get("metadata") or {}).get("model_sha256"))
            candidates.append(p.get("model_sha256"))
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip()
    return None


def _extract_max_mol(run: dict, pt: dict) -> int:
    for cand in (
        pt.get("max_mol"),
        (pt.get("metadata") or {}).get("max_mol"),
        run.get("max_mol"),
        (run.get("metadata") or {}).get("max_mol"),
    ):
        if cand is not None:
            try:
                val = int(cand)
                if val > 0:
                    return val
            except (TypeError, ValueError):
                pass
    return 128


def _parse_remc_swaps(run: dict) -> Optional[dict]:
    """Normalize REMC pressure-pair swap acceptance statistics from run-level or per-point fields."""
    pts = sorted(run.get("points") or [], key=lambda x: float(x.get("P_bar", 0.0)))
    # Native multi_gpu.py / nested_mc_engine.py schema: per-point remc_swap_*_up linking pts[m] <-> pts[m+1]
    if len(pts) >= 2 and any("remc_swap_att_up" in pt or "remc_swap_rate_up" in pt for pt in pts):
        pairs_norm: List[Dict[str, Any]] = []
        swap_every = None
        for m in range(len(pts) - 1):
            pt_lo, pt_hi = pts[m], pts[m + 1]
            if swap_every is None and pt_lo.get("remc_swap_every") is not None:
                swap_every = int(pt_lo["remc_swap_every"])
            att = pt_lo.get("remc_swap_att_up")
            acc = pt_lo.get("remc_swap_acc_up")
            rate = pt_lo.get("remc_swap_rate_up")
            if rate is None and att is not None and acc is not None and int(att) > 0:
                rate = float(acc) / float(int(att))
            if att is None and acc is None and rate is None:
                continue
            p_lo = float(pt_lo["P_bar"])
            p_hi = float(pt_hi["P_bar"])
            pairs_norm.append({
                "pair_label": f"{p_lo:g} <-> {p_hi:g} bar",
                "p_low": p_lo,
                "p_high": p_hi,
                "attempts": None if att is None else int(att),
                "accepted": None if acc is None else int(acc),
                "acc_rate": None if rate is None else float(rate),
            })
        if pairs_norm:
            if swap_every is None and run.get("remc_swap_every") is not None:
                swap_every = int(run["remc_swap_every"])
            valid_rates = [p["acc_rate"] for p in pairs_norm if p["acc_rate"] is not None]
            return {
                "swap_every": swap_every,
                "pairs": pairs_norm,
                "mean_swap_acc_rate": float(np.mean(valid_rates)) if valid_rates else None,
                "min_swap_acc_rate": float(np.min(valid_rates)) if valid_rates else None,
            }

    raw = (
        run.get("remc_swap_stats")
        or run.get("remc_stats")
        or run.get("remc")
        or (run.get("metadata") or {}).get("remc_swap_stats")
    )
    if not raw and pts:
        for pt in pts:
            pr = pt.get("remc_swap_stats") or pt.get("remc_stats") or pt.get("remc")
            if pr:
                raw = pr
                break
    if not raw:
        return None

    pairs_norm: List[Dict[str, Any]] = []
    swap_every = None
    if isinstance(raw, dict):
        swap_every = raw.get("swap_every", raw.get("remc_swap_every"))
        raw_pairs = (
            raw.get("pairs")
            or raw.get("pair_stats")
            or raw.get("pair_acceptance_rates")
            or raw.get("swap_acceptance_rates")
            or raw.get("pressure_pairs")
        )
        if raw_pairs is None:
            # Maybe `raw` itself is a dict mapping pair label -> rate/stats
            if all(isinstance(v, (int, float, dict)) for v in raw.values()):
                raw_pairs = raw
    elif isinstance(raw, list):
        raw_pairs = raw
    else:
        raw_pairs = None

    if isinstance(raw_pairs, list):
        for item in raw_pairs:
            if not isinstance(item, dict):
                continue
            p_lo = item.get("p_low", item.get("p_lo", item.get("p_i", item.get("P_low"))))
            p_hi = item.get("p_high", item.get("p_hi", item.get("p_j", item.get("P_high"))))
            if (p_lo is None or p_hi is None) and isinstance(item.get("pair"), (list, tuple)) and len(item["pair"]) == 2:
                p_lo, p_hi = item["pair"][0], item["pair"][1]
            attempts = item.get("attempts", item.get("swap_attempts", item.get("att")))
            accepted = item.get("accepted", item.get("swap_accepted", item.get("acc")))
            rate = item.get("acc_rate", item.get("acceptance_rate", item.get("rate")))
            if rate is None and attempts is not None and accepted is not None:
                rate = float(accepted) / float(max(1, int(attempts)))
            label = item.get("pair_label") or item.get("label")
            if not label:
                if p_lo is not None and p_hi is not None:
                    label = f"{float(p_lo):g} <-> {float(p_hi):g} bar"
                else:
                    label = f"pair_{len(pairs_norm)}"
            pairs_norm.append({
                "pair_label": str(label),
                "p_low": None if p_lo is None else float(p_lo),
                "p_high": None if p_hi is None else float(p_hi),
                "attempts": None if attempts is None else int(attempts),
                "accepted": None if accepted is None else int(accepted),
                "acc_rate": None if rate is None else float(rate),
            })
    elif isinstance(raw_pairs, dict):
        for k, v in raw_pairs.items():
            if k in ("swap_every", "remc_swap_every", "total_attempts", "total_accepted", "mean_acc_rate"):
                continue
            nums = re.findall(r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", str(k))
            p_lo = float(nums[0]) if len(nums) >= 2 else None
            p_hi = float(nums[1]) if len(nums) >= 2 else None
            if isinstance(v, dict):
                att = v.get("attempts", v.get("att"))
                acc = v.get("accepted", v.get("acc"))
                rate = v.get("acc_rate", v.get("acceptance_rate", v.get("rate")))
                if rate is None and att is not None and acc is not None:
                    rate = float(acc) / float(max(1, int(att)))
            elif isinstance(v, (int, float)):
                att, acc, rate = None, None, float(v)
            else:
                continue
            label = f"{p_lo:g} <-> {p_hi:g} bar" if (p_lo is not None and p_hi is not None) else str(k)
            pairs_norm.append({
                "pair_label": label,
                "p_low": p_lo,
                "p_high": p_hi,
                "attempts": None if att is None else int(att),
                "accepted": None if acc is None else int(acc),
                "acc_rate": None if rate is None else float(rate),
            })

    if not pairs_norm:
        return None
    valid_rates = [p["acc_rate"] for p in pairs_norm if p["acc_rate"] is not None]
    return {
        "swap_every": None if swap_every is None else int(swap_every),
        "pairs": pairs_norm,
        "mean_swap_acc_rate": float(np.mean(valid_rates)) if valid_rates else None,
        "min_swap_acc_rate": float(np.min(valid_rates)) if valid_rates else None,
    }


def _compute_molar_qst(prod_n: np.ndarray, prod_u_kJ: np.ndarray, T_K: float) -> Tuple[Optional[float], Optional[float]]:
    if prod_n.ndim != 2 or prod_u_kJ.shape != prod_n.shape or prod_n.shape[0] == 0:
        return None, None
    n_c = prod_n.shape[1]
    rt = float(T_K) * K_TO_KJMOL
    per_box = []
    for c in range(n_c):
        mn = float(np.mean(prod_n[:, c]))
        mu = float(np.mean(prod_u_kJ[:, c]))
        if mn > 1e-12:
            per_box.append(rt - mu / mn)
    if not per_box:
        return None, None
    mean_v = float(np.mean(per_box))
    sem_v = float(np.std(per_box, ddof=1) / math.sqrt(len(per_box))) if len(per_box) > 1 else 0.0
    return mean_v, sem_v


def _interpolate_experimental_isotherms(
    exp_dir: Path, mof_pressures: Dict[str, List[float]]
) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    manifest_p = exp_dir / "manifest.json"
    if not manifest_p.is_file():
        return {}
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for mof, p_list in sorted(mof_pressures.items()):
        mkey = EXP_MANIFEST_KEY_MAP.get(mof, mof)
        entries = manifest.get(mkey)
        if not entries:
            continue
        out[mof] = {}
        loaded_curves = []
        for ent in entries:
            csv_rel = ent.get("csv", "")
            csv_p = exp_dir / Path(csv_rel).name
            if not csv_p.is_file():
                continue
            px, uy = [], []
            with csv_p.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    px.append(float(row["P_bar"]))
                    uy.append(float(row["uptake_mmol_g"]))
            if px:
                order = np.argsort(px)
                loaded_curves.append((csv_p.name, np.asarray(px)[order], np.asarray(uy)[order]))
        for P in sorted(set(float(x) for x in p_list)):
            pkey = f"{P:g}"
            out[mof][pkey] = {}
            for fname, px_arr, uy_arr in loaded_curves:
                if P < float(px_arr[0]) - 1e-9 or P > float(px_arr[-1]) + 1e-9:
                    out[mof][pkey][fname] = None
                else:
                    out[mof][pkey][fname] = float(np.interp(P, px_arr, uy_arr))
    return out


def _effective_n_gpus_for_point(run: dict, pt: dict, n_points: int) -> float:
    pmode = str(pt.get("parallel_mode", run.get("parallel_mode", "single"))).lower()
    run_gpus = run.get("gpus") or [0]
    n_run_gpus = float(max(1, len(run_gpus))) if isinstance(run_gpus, list) else 1.0
    if pmode == "remc":
        n_remc_p = int(pt.get("remc_n_pressures") or max(1, n_points))
        active_gpus = min(n_run_gpus, float(max(1, n_remc_p)))
        return float(active_gpus / float(max(1, n_remc_p)))
    if pt.get("gpu_ids") and isinstance(pt["gpu_ids"], list):
        return float(max(1, len(pt["gpu_ids"])))
    if pt.get("gpu_id") is not None:
        return 1.0
    if pmode == "replica":
        return n_run_gpus
    return 1.0


def _audit_distillation(dist: dict, cache_id: str) -> dict:
    is_1b_only = bool(dist.get("only_1b", False) or dist.get("distill_1b_only", False))
    mae_2b_dep = dist.get("mae_2b_deployed_holdout_kJ_mol")
    mae_2b_cls = dist.get("mae_2b_classical_holdout_kJ_mol")
    r2_2b_dep = dist.get("r2_2b_deployed_holdout")
    n_2b_val = dist.get("n_2b_val")
    has_honest_2b = bool(mae_2b_dep is not None and n_2b_val is not None and int(n_2b_val) > 0)
    if is_1b_only:
        two_b_status = "1b_only_ablation"
    elif has_honest_2b:
        two_b_status = "honest_deployed_holdout"
    else:
        two_b_status = "missing_deployed_holdout_legacy"

    out = dict(dist)
    out["cache_id"] = cache_id
    out["distill_1b_only"] = is_1b_only
    out["r2_2b_deployed_holdout"] = None if r2_2b_dep is None else float(r2_2b_dep)
    out["mae_2b_deployed_holdout_kJ_mol"] = None if mae_2b_dep is None else float(mae_2b_dep)
    out["mae_2b_classical_holdout_kJ_mol"] = None if mae_2b_cls is None else float(mae_2b_cls)
    out["n_2b_val"] = None if n_2b_val is None else int(n_2b_val)
    out["has_honest_2b_holdout"] = has_honest_2b
    out["two_b_holdout_audit"] = two_b_status
    out["model_sha256_present"] = bool(isinstance(dist.get("model_sha256"), str) and dist["model_sha256"].strip())
    return out


def archive_results_dir(
    results_dir: Path,
    out_json: Path,
    out_traces: Path,
    out_report: Path,
    exp_dir: Optional[Path] = None,
    embed_traces: bool = False,
) -> dict:
    """Scan `results_dir`, recompute all diagnostics from full traces, write summary + NPZ + audit report."""
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    skip_names = {"phase2_summary.json", "jctc_summary.json", "manifest.json"}
    json_files = sorted(
        p for p in results_dir.rglob("*.json")
        if p.name not in skip_names
        and not p.name.startswith("distill_")
        and not p.name.startswith("ckpt_")
        and "checkpoints" not in p.parts
        and not p.name.endswith(".tmp.json")
    )

    runs_out: Dict[str, Any] = {}
    widom_out: Dict[str, Any] = {}
    distillation_map: Dict[str, Any] = {}
    extras_out: Dict[str, Any] = {}
    trace_arrays: Dict[str, np.ndarray] = {}
    mof_pressures: Dict[str, List[float]] = {}

    audit_missing_sha: List[str] = []
    audit_missing_2b_holdout: List[str] = []
    audit_saturation_fails: List[dict] = []
    audit_saturation_warns: List[dict] = []
    audit_recomp_discrepancies: List[dict] = []
    audit_asymmetric_proposals: List[dict] = []
    audit_remc_runs: Dict[str, Any] = {}

    n_pts_tot = n_pts_pass = n_pts_fail = n_pts_insuff = n_pts_unk = 0

    for jp in json_files:
        try:
            raw = json.loads(jp.read_text(encoding="utf-8"))
        except Exception as exc:
            audit_recomp_discrepancies.append({"file": str(jp), "error": f"json_parse_error: {exc}"})
            continue
        if not isinstance(raw, dict):
            continue

        stem = jp.stem
        # Preserve relative stem uniqueness if nested subdirectories exist
        rel_parts = jp.relative_to(results_dir).with_suffix("").parts
        run_key = "_".join(rel_parts) if len(rel_parts) > 1 else stem

        if "widom" in raw or "mace_d3_decomposition" in raw or "finite_size_check" in raw:
            widom_out[run_key] = raw
            continue

        if "points" not in raw or not isinstance(raw["points"], list):
            if "k_steps" in raw and "outer_acc_pct" in raw:
                extras_out[run_key] = raw
            continue

        n_pts = len(raw["points"])
        is_direct = (raw.get("method") == "direct_mlip_gcmc") or (raw.get("proposer") == "direct")
        proposer = "direct" if is_direct else str(raw.get("proposer", "unknown"))
        grid_only = bool(raw.get("grid_only", False))
        cif_str = str(raw.get("cif", ""))
        mof_name = _canonical_mof_name(cif_str)
        guest_name = str(raw.get("guest", "co2")).lower()
        model_name = str(raw.get("model", "unknown"))
        run_model_sha = _extract_model_sha256(raw)

        # Distillation audit
        dist_raw = raw.get("distillation")
        dist_audited = None
        distill_wall_s = 0.0
        if isinstance(dist_raw, dict) and dist_raw:
            sha_full = str(dist_raw.get("distill_sha256", run_key))
            cid = sha_full[:16]
            dist_audited = _audit_distillation(dist_raw, cid)
            distill_wall_s = float(dist_raw.get("distill_wall_s", 0.0) or 0.0)
            if cid not in distillation_map:
                distillation_map[cid] = dict(dist_audited)
                distillation_map[cid]["used_by_runs"] = []
            if run_key not in distillation_map[cid]["used_by_runs"]:
                distillation_map[cid]["used_by_runs"].append(run_key)
            if not dist_audited["distill_1b_only"] and not dist_audited["has_honest_2b_holdout"]:
                if cid not in audit_missing_2b_holdout:
                    audit_missing_2b_holdout.append(cid)
            if not dist_audited["model_sha256_present"]:
                tag = f"distillation:{cid}"
                if tag not in audit_missing_sha:
                    audit_missing_sha.append(tag)

        # Model SHA-256 completeness audit for any MLIP-backed run
        needs_mlip_sha = (not grid_only) or (proposer == "auto_mlip")
        if needs_mlip_sha and not run_model_sha:
            audit_missing_sha.append(f"run:{run_key}")

        remc_summary = _parse_remc_swaps(raw)
        if remc_summary is not None:
            audit_remc_runs[run_key] = remc_summary

        archived_pts: List[dict] = []
        for pt_idx, pt in enumerate(sorted(raw["points"], key=lambda x: float(x.get("P_bar", 0.0)))):
            n_pts_tot += 1
            p_bar = float(pt.get("P_bar", 0.0))
            T_K = float(pt.get("T_K", raw.get("T_K", 298.0)))
            mof_pressures.setdefault(mof_name, []).append(p_bar)

            prod_n_raw = pt.get("prod_N_history")
            prod_u_raw = pt.get("prod_U_history")
            prod_oms_raw = pt.get("prod_oms_occ_history")
            box_n_raw = pt.get("box_N_history")

            has_full_traces = (
                isinstance(prod_n_raw, list) and len(prod_n_raw) > 0
                and isinstance(prod_u_raw, list) and len(prod_u_raw) == len(prod_n_raw)
            )
            prod_n_arr = np.asarray(prod_n_raw, dtype=np.int32) if has_full_traces else None
            prod_u_arr = np.asarray(prod_u_raw, dtype=np.float64) if has_full_traces else None
            prod_oms_arr = (
                np.asarray(prod_oms_raw, dtype=np.int32)
                if isinstance(prod_oms_raw, list) and len(prod_oms_raw) > 0
                else None
            )
            box_n_arr = (
                np.asarray(box_n_raw, dtype=np.int32)
                if isinstance(box_n_raw, list) and len(box_n_raw) > 0
                else None
            )

            fw_mass_gmol = _resolve_fw_mass_gmol(raw, pt, prod_n_arr)
            conv_mmol_g = (1000.0 / fw_mass_gmol) if (fw_mass_gmol is not None and fw_mass_gmol > 0.0) else None

            prod_ins_att = int(pt.get("prod_ins_att", 0))
            prod_ins_acc = int(pt.get("prod_ins_acc", 0))
            prod_del_att = int(pt.get("prod_del_att", 0))
            prod_del_acc = int(pt.get("prod_del_acc", 0))
            prod_outer_acc = float(
                pt.get("prod_outer_acc_rate", pt.get("prod_move_acc_rate", 0.0)) or 0.0
            )
            outer_acc = float(
                pt.get("outer_acc_rate", pt.get("move_acc_rate", prod_outer_acc)) or 0.0
            )

            # Proposal symmetry ratio
            sym_ratio = float(prod_ins_att) / float(max(1, prod_del_att))
            sym_flag = bool((prod_ins_att + prod_del_att) >= 20 and (sym_ratio < 0.5 or sym_ratio > 2.0))
            if sym_flag:
                audit_asymmetric_proposals.append({
                    "run": run_key,
                    "P_bar": p_bar,
                    "prod_ins_att": prod_ins_att,
                    "prod_del_att": prod_del_att,
                    "symmetry_ratio": sym_ratio,
                })

            # max_N_observed vs max_mol saturation safety audit
            max_mol = _extract_max_mol(raw, pt)
            max_n_cands = [int(pt.get("max_N_observed", 0) or 0)]
            if prod_n_arr is not None and prod_n_arr.size > 0:
                max_n_cands.append(int(np.max(prod_n_arr)))
            if box_n_arr is not None and box_n_arr.size > 0:
                max_n_cands.append(int(np.max(box_n_arr)))
            max_n_obs = max(max_n_cands)
            max_n_frac = float(max_n_obs) / float(max(1, max_mol))
            sat_fail = bool(max_n_obs >= max_mol)
            sat_warn = bool((not sat_fail) and max_n_obs >= int(math.ceil(0.90 * max_mol)))
            sat_status = "FAIL_SATURATED" if sat_fail else ("WARN_NEAR_CAP" if sat_warn else "OK")
            if sat_fail:
                audit_saturation_fails.append({
                    "run": run_key, "P_bar": p_bar, "max_N_observed": max_n_obs, "max_mol": max_mol,
                })
            elif sat_warn:
                audit_saturation_warns.append({
                    "run": run_key, "P_bar": p_bar, "max_N_observed": max_n_obs, "max_mol": max_mol,
                })

            # Save full trajectories to companion NPZ
            tkey = f"{_sanitize_key(run_key)}__p{pt_idx}"
            if prod_n_arr is not None:
                trace_arrays[f"{tkey}__prod_N"] = prod_n_arr
            if prod_u_arr is not None:
                trace_arrays[f"{tkey}__prod_U"] = prod_u_arr
            if prod_oms_arr is not None:
                trace_arrays[f"{tkey}__prod_OMS"] = prod_oms_arr
            if box_n_arr is not None:
                trace_arrays[f"{tkey}__box_N"] = box_n_arr

            # Recompute thermodynamics & Vehtari diagnostics from full trajectories
            recompute_parity_ok = True
            recompute_notes: List[str] = []
            if has_full_traces and conv_mmol_g is not None and prod_n_arr.ndim == 2:
                n_prod_cyc, n_boxes = prod_n_arr.shape
                box_ups = np.mean(prod_n_arr.astype(np.float64), axis=0) * conv_mmol_g
                uptake_val = float(np.mean(box_ups))
                uptake_sem = float(np.std(box_ups, ddof=1) / math.sqrt(n_boxes)) if n_boxes > 1 else 0.0

                stored_up = pt.get("uptake_mmol_g")
                if stored_up is not None and abs(uptake_val - float(stored_up)) > 1e-3:
                    recompute_parity_ok = False
                    recompute_notes.append(
                        f"uptake_diff({uptake_val:.5f}_vs_stored_{float(stored_up):.5f})"
                    )

                v2 = evaluate_v2_convergence(
                    prod_N_history=prod_n_arr.tolist(),
                    prod_U_history_kJ=prod_u_arr.tolist(),
                    conv_mmol_g=conv_mmol_g,
                    T_K=T_K,
                    prod_ins_acc=prod_ins_acc,
                    prod_del_acc=prod_del_acc,
                    prod_ins_att=prod_ins_att,
                    prod_del_att=prod_del_att,
                    prod_outer_acc_rate=prod_outer_acc,
                    uptake_sem_mmol_g=uptake_sem,
                    acceptance_breakdown=pt.get("acceptance_breakdown"),
                    oms_occ_history=prod_oms_arr.tolist() if prod_oms_arr is not None else None,
                )
                qst_fluc = v2.get("qst_fluctuation_kJ_mol")
                qst_fluc_sem = v2.get("qst_fluctuation_sem_kJ_mol")
                qst_status = v2.get("qst_status", "unknown")
                qst_molar, qst_molar_sem = _compute_molar_qst(prod_n_arr.astype(np.float64), prod_u_arr, T_K)

                # Cross-check against pre-computed diagnostics if present
                stored_diag = pt.get("diagnostics") or {}
                if isinstance(stored_diag, dict) and stored_diag.get("status"):
                    if stored_diag["status"] != v2["status"]:
                        recompute_parity_ok = False
                        recompute_notes.append(
                            f"diag_status({v2['status']}_vs_stored_{stored_diag['status']})"
                        )
                    st_rhat = (stored_diag.get("N_stats") or {}).get("rhat_max")
                    rc_rhat = (v2.get("N_stats") or {}).get("rhat_max")
                    if st_rhat is not None and rc_rhat is not None and abs(float(st_rhat) - float(rc_rhat)) > 1e-3:
                        recompute_parity_ok = False
                        recompute_notes.append(f"rhat_N({rc_rhat:.4f}_vs_stored_{float(st_rhat):.4f})")
            else:
                n_boxes = int(pt.get("n_boxes", 0))
                n_prod_cyc = int(pt.get("prod_cycles", pt.get("prod_steps_done", 0)))
                uptake_val = float(pt.get("uptake_mmol_g", 0.0) or 0.0)
                uptake_sem = float(pt.get("uptake_sem_mmol_g", 0.0) or 0.0)
                v2 = pt.get("diagnostics") or {"status": "unknown", "reasons": ["missing_full_traces"]}
                qst_fluc = pt.get("Qst_kJ_mol")
                qst_fluc_sem = pt.get("Qst_sem_kJ_mol")
                qst_status = pt.get("qst_status", v2.get("qst_status", "unknown"))
                qst_molar = pt.get("mean_molar_qst_kJ_mol")
                qst_molar_sem = pt.get("mean_molar_qst_sem_kJ_mol")
                recompute_parity_ok = False
                recompute_notes.append("missing_full_production_traces")

            if not recompute_parity_ok:
                audit_recomp_discrepancies.append({
                    "run": run_key, "P_bar": p_bar, "notes": recompute_notes,
                })

            diag_status = str(v2.get("status", "unknown"))
            if diag_status == "pass":
                n_pts_pass += 1
            elif diag_status == "fail":
                n_pts_fail += 1
            elif diag_status == "insufficient":
                n_pts_insuff += 1
            else:
                n_pts_unk += 1

            n_stats = v2.get("N_stats") or {}
            u_stats = v2.get("U_stats") or {}
            oms_stats = v2.get("OMS_stats") or {}

            ess_bulk_n = float(n_stats.get("ess_bulk", 0.0) or 0.0)
            ess_tail_n = float(n_stats.get("ess_tail", 0.0) or 0.0)
            ess_mean_n = float(n_stats.get("ess_mean", 0.0) or 0.0)
            eff_mcse = v2.get("mcse_mmol_g")

            elapsed_s = float(pt.get("elapsed_s", 0.0) or 0.0)
            n_gpus_eff = _effective_n_gpus_for_point(raw, pt, n_pts)
            gpu_s_sampling = max(elapsed_s * n_gpus_eff, 1e-9)
            distill_amort_s = distill_wall_s / float(max(1, n_pts))

            ess_bulk_per_gpu_s = float(ess_bulk_n / gpu_s_sampling)
            ess_bulk_per_gpu_s_amort = float(ess_bulk_n / (gpu_s_sampling + distill_amort_s))
            ess_bulk_per_gpu_s_cold = float(ess_bulk_n / (gpu_s_sampling + distill_wall_s))

            if eff_mcse is not None and math.isfinite(float(eff_mcse)) and float(eff_mcse) > 0.0:
                t_mcse_samp = float(gpu_s_sampling * ((float(eff_mcse) / 0.10) ** 2))
                t_mcse_amort = float(t_mcse_samp + distill_amort_s)
                t_mcse_cold = float(t_mcse_samp + distill_wall_s)
            else:
                t_mcse_samp = None
                t_mcse_amort = None
                t_mcse_cold = None

            pt_trace_src = box_n_arr if box_n_arr is not None else prod_n_arr
            box_trace_compact = _downsample_trace(pt_trace_src, max_points=250) if pt_trace_src is not None else []

            pt_rec: Dict[str, Any] = {
                "P_bar": p_bar,
                "T_K": T_K,
                "n_boxes": int(n_boxes),
                "eq_cycles": int(pt.get("eq_cycles", pt.get("eq_steps", 0))),
                "prod_cycles": int(n_prod_cyc),
                "k_steps": pt.get("k_steps"),
                "fw_mass_gmol": fw_mass_gmol,
                "uptake_mmol_g": uptake_val,
                "uptake_sem_mmol_g": uptake_sem,
                "mcse_mmol_g": eff_mcse,
                "mcse_autocorr_mmol_g": v2.get("mcse_autocorr_mmol_g"),
                "Qst_fluc_kJ_mol": qst_fluc,
                "Qst_fluc_sem_kJ_mol": qst_fluc_sem,
                "qst_status": qst_status,
                "Qst_molar_kJ_mol": qst_molar,
                "Qst_molar_sem_kJ_mol": qst_molar_sem,
                "delta_eps_kJ_mol": pt.get("delta_eps_kJ_mol"),
                "outer_acc_pct": 100.0 * outer_acc,
                "prod_outer_acc_pct": 100.0 * prod_outer_acc,
                "prod_ins_acc": prod_ins_acc,
                "prod_ins_att": prod_ins_att,
                "prod_del_acc": prod_del_acc,
                "prod_del_att": prod_del_att,
                "proposal_symmetry_ratio": sym_ratio,
                "proposal_asymmetry_flag": sym_flag,
                "max_N_observed": int(max_n_obs),
                "max_mol": int(max_mol),
                "max_N_fraction_of_cap": max_n_frac,
                "max_mol_saturated": sat_fail,
                "max_mol_near_saturation": sat_warn,
                "saturation_safety_flag": sat_status,
                "match_distilled_hard_core": pt.get("match_distilled_hard_core", raw.get("match_distilled_hard_core")),
                "elapsed_s": elapsed_s,
                "point_wall_s": pt.get("point_wall_s", elapsed_s),
                "n_gpus_effective": n_gpus_eff,
                "gpu_s_sampling": gpu_s_sampling,
                "mlip_evals": int(pt.get("mlip_evals", 0) or 0),
                "ess_N": pt.get("ess_N"),
                "ess_N_per_s": pt.get("ess_N_per_s"),
                "esjd_N": pt.get("esjd_N"),
                "dN_moves_per_s": pt.get("dN_moves_per_s"),
                "N_rhat_max": n_stats.get("rhat_max"),
                "N_ess_bulk": ess_bulk_n,
                "N_ess_tail": ess_tail_n,
                "N_ess_mean": ess_mean_n,
                "U_rhat_max": u_stats.get("rhat_max"),
                "U_ess_bulk": float(u_stats.get("ess_bulk", 0.0) or 0.0),
                "U_ess_tail": float(u_stats.get("ess_tail", 0.0) or 0.0),
                "OMS_rhat_max": oms_stats.get("rhat_max") if oms_stats else None,
                "OMS_ess_bulk": float(oms_stats.get("ess_bulk", 0.0) or 0.0) if oms_stats else None,
                "max_abs_chain_drift_mmol_g": v2.get("max_abs_chain_drift_mmol_g"),
                "pooled_abs_drift_mmol_g": v2.get("pooled_abs_drift_mmol_g"),
                "ess_bulk_N_per_gpu_s": ess_bulk_per_gpu_s,
                "ess_bulk_N_per_gpu_s_amortized": ess_bulk_per_gpu_s_amort,
                "ess_bulk_N_per_gpu_s_coldstart": ess_bulk_per_gpu_s_cold,
                "time_to_mcse_0p1_s": t_mcse_samp,
                "time_to_mcse_0p1_amortized_s": t_mcse_amort,
                "time_to_mcse_0p1_coldstart_s": t_mcse_cold,
                "diag_status": diag_status,
                "diag_reasons": list(v2.get("reasons", [])),
                "recomputed_from_full_traces": bool(has_full_traces and conv_mmol_g is not None),
                "recompute_parity_ok": recompute_parity_ok,
                "recompute_notes": recompute_notes,
                "trace_key": tkey if has_full_traces else None,
                "box_N_trace": box_trace_compact,
            }
            if embed_traces and has_full_traces:
                pt_rec["prod_N_history"] = prod_n_arr.tolist()
                pt_rec["prod_U_history"] = prod_u_arr.tolist()
                if prod_oms_arr is not None:
                    pt_rec["prod_oms_occ_history"] = prod_oms_arr.tolist()
            archived_pts.append(pt_rec)

        runs_out[run_key] = {
            "source_json": str(jp.relative_to(results_dir)),
            "mof": mof_name,
            "cif": cif_str,
            "supercell": raw.get("supercell"),
            "guest": guest_name,
            "T_K": float(raw.get("T_K", 298.0)),
            "proposer": proposer,
            "model": model_name,
            "model_sha256": run_model_sha,
            "dtype": raw.get("dtype", "float32"),
            "dispersion": bool(raw.get("dispersion", False)),
            "dispersion_xc": raw.get("dispersion_xc", ""),
            "grid_only": grid_only,
            "distill_1b_only": bool(
                raw.get("distill_1b_only", False)
                or raw.get("only_1b", False)
                or (dist_audited or {}).get("distill_1b_only", False)
            ),
            "init_mode": raw.get("init_mode", "auto"),
            "gpus": raw.get("gpus", [0]),
            "parallel_mode": raw.get("parallel_mode", "direct" if is_direct else "single"),
            "total_wall_s": raw.get("total_wall_s"),
            "distillation": dist_audited,
            "remc_swap_summary": remc_summary,
            "points": archived_pts,
        }

    # Save companion compressed traces NPZ
    out_traces.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_traces, **trace_arrays)
    traces_sha = _sha256_file(out_traces)

    exp_interp = {}
    if exp_dir is not None and exp_dir.is_dir():
        exp_interp = _interpolate_experimental_isotherms(exp_dir, mof_pressures)

    summary = {
        "schema_version": "jctc_v2",
        "source_results_dir": str(results_dir),
        "traces_npz": out_traces.name,
        "traces_sha256": traces_sha,
        "audit": {
            "n_runs": len(runs_out),
            "n_widom_reports": len(widom_out),
            "n_distillation_caches": len(distillation_map),
            "n_points_total": n_pts_tot,
            "n_points_pass": n_pts_pass,
            "n_points_fail": n_pts_fail,
            "n_points_insufficient": n_pts_insuff,
            "n_points_unknown": n_pts_unk,
            "model_sha256_complete": len(audit_missing_sha) == 0,
            "model_sha256_missing": audit_missing_sha,
            "honest_2b_holdout_complete": len(audit_missing_2b_holdout) == 0,
            "honest_2b_holdout_missing": audit_missing_2b_holdout,
            "saturation_safety_pass": len(audit_saturation_fails) == 0,
            "saturation_failures": audit_saturation_fails,
            "saturation_warnings": audit_saturation_warns,
            "recompute_parity_all_ok": len(audit_recomp_discrepancies) == 0,
            "recompute_discrepancies": audit_recomp_discrepancies,
            "asymmetric_proposal_points": audit_asymmetric_proposals,
            "remc_runs": audit_remc_runs,
        },
        "runs": runs_out,
        "distillation": distillation_map,
        "widom": widom_out,
        "extras": extras_out,
        "experimental_interpolated_mmol_g": exp_interp,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = build_audit_markdown(summary)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(md, encoding="utf-8")
    return summary


def verify_archive_recomputability(summary_json: Path, traces_npz: Path) -> Dict[str, Any]:
    """Independently load `jctc_summary.json` + `jctc_traces.npz` and verify full diagnostic reproducibility."""
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    npz = np.load(traces_npz)
    checked = 0
    mismatches: List[str] = []

    for run_key, run in summary.get("runs", {}).items():
        for pt in run.get("points", []):
            tkey = pt.get("trace_key")
            if not tkey:
                continue
            n_key, u_key = f"{tkey}__prod_N", f"{tkey}__prod_U"
            if n_key not in npz or u_key not in npz:
                mismatches.append(f"{run_key} P={pt['P_bar']}: missing NPZ arrays {n_key}/{u_key}")
                continue
            prod_n = npz[n_key]
            prod_u = npz[u_key]
            oms_key = f"{tkey}__prod_OMS"
            prod_oms = npz[oms_key] if oms_key in npz else None
            fw_mass = pt.get("fw_mass_gmol")
            if not fw_mass:
                continue
            conv = 1000.0 / float(fw_mass)
            box_ups = np.mean(prod_n.astype(np.float64), axis=0) * conv
            sem = float(np.std(box_ups, ddof=1) / math.sqrt(prod_n.shape[1])) if prod_n.shape[1] > 1 else 0.0
            v2 = evaluate_v2_convergence(
                prod_N_history=prod_n.tolist(),
                prod_U_history_kJ=prod_u.tolist(),
                conv_mmol_g=conv,
                T_K=float(pt.get("T_K", 298.0)),
                prod_ins_acc=int(pt.get("prod_ins_acc", 0)),
                prod_del_acc=int(pt.get("prod_del_acc", 0)),
                prod_ins_att=int(pt.get("prod_ins_att", 0)),
                prod_del_att=int(pt.get("prod_del_att", 0)),
                prod_outer_acc_rate=float(pt.get("prod_outer_acc_pct", 0.0)) / 100.0,
                uptake_sem_mmol_g=sem,
                oms_occ_history=prod_oms.tolist() if prod_oms is not None else None,
            )
            checked += 1
            if v2["status"] != pt.get("diag_status"):
                mismatches.append(f"{run_key} P={pt['P_bar']}: status {v2['status']} != {pt.get('diag_status')}")
            rh_rc = (v2.get("N_stats") or {}).get("rhat_max")
            rh_st = pt.get("N_rhat_max")
            if rh_rc is not None and rh_st is not None and abs(float(rh_rc) - float(rh_st)) > 1e-6:
                mismatches.append(f"{run_key} P={pt['P_bar']}: rhat_N {rh_rc} != {rh_st}")
            eb_rc = float((v2.get("N_stats") or {}).get("ess_bulk", 0.0) or 0.0)
            eb_st = float(pt.get("N_ess_bulk", 0.0) or 0.0)
            if abs(eb_rc - eb_st) > 1e-4:
                mismatches.append(f"{run_key} P={pt['P_bar']}: ess_bulk {eb_rc} != {eb_st}")
            q_rc = v2.get("qst_fluctuation_kJ_mol")
            q_st = pt.get("Qst_fluc_kJ_mol")
            if q_rc is not None and q_st is not None and abs(float(q_rc) - float(q_st)) > 1e-4:
                mismatches.append(f"{run_key} P={pt['P_bar']}: Qst {q_rc} != {q_st}")

    npz.close()
    return {"checked_points": checked, "ok": len(mismatches) == 0, "mismatches": mismatches}


def _fmt(val: Optional[float], digits: int = 2, suffix: str = "") -> str:
    if val is None:
        return "N/A"
    try:
        fv = float(val)
        if not math.isfinite(fv):
            return "N/A"
        return f"{fv:.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def build_audit_markdown(summary: dict) -> str:
    """Render a clean publication-grade Markdown audit report from `jctc_summary.json`."""
    au = summary["audit"]
    lines: List[str] = [
        "# JCTC v2 Simulation & Reproducibility Audit Report",
        "",
        f"- **Source directory**: `{summary.get('source_results_dir')}`",
        f"- **Traces archive**: `{summary.get('traces_npz')}` (SHA-256: `{str(summary.get('traces_sha256') or 'N/A')[:16]}...`)",
        f"- **Runs / Points**: {au['n_runs']} runs, {au['n_points_total']} pressure points "
        f"(**PASS**: {au['n_points_pass']}, **FAIL**: {au['n_points_fail']}, "
        f"**INSUFFICIENT**: {au['n_points_insufficient']}, **UNKNOWN**: {au['n_points_unknown']})",
        f"- **`model_sha256` completeness**: {'PASS' if au['model_sha256_complete'] else 'FAIL (' + str(len(au['model_sha256_missing'])) + ' missing)'}",
        f"- **2B deployed hold-out completeness**: {'PASS' if au['honest_2b_holdout_complete'] else 'WARN (' + str(len(au['honest_2b_holdout_missing'])) + ' legacy/missing)'}",
        f"- **`max_mol` saturation safety**: {'PASS' if au['saturation_safety_pass'] else 'FAIL (' + str(len(au['saturation_failures'])) + ' saturated)'}",
        f"- **Full-trace recomputation parity**: {'PASS' if au['recompute_parity_all_ok'] else 'WARN (' + str(len(au['recompute_discrepancies'])) + ' discrepancies)'}",
        "",
        "## 1. Per-Point Convergence, Thermodynamics, Efficiency & Safety",
        "",
        "| Run | System | Proposer / Mode | Model | P (bar) | Uptake (mmol/g) | MCSE | Qst fluc (kJ/mol) | R-hat (N / U / OMS) | ESS_bulk(N) | ESS/GPU-s | t(MCSE=0.1) s (samp/amort/cold) | Ins/Del Sym | maxN/cap | Status |",
        "|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---|---:|---|---|",
    ]

    for rkey, run in sorted(summary.get("runs", {}).items()):
        sys_str = f"{run['mof']} + {run['guest'].upper()}"
        pmode = f"{run['proposer']}"
        if run.get("grid_only"):
            pmode += " (grid)"
        elif run.get("distill_1b_only"):
            pmode += " (1B)"
        elif run.get("parallel_mode") == "remc":
            pmode += " (REMC)"
        mstr = f"{run['model']}" + ("+D3" if run.get("dispersion") else "")
        for pt in run.get("points", []):
            up_str = f"{_fmt(pt['uptake_mmol_g'], 3)} ± {_fmt(pt['uptake_sem_mmol_g'], 3)}"
            mcse_str = _fmt(pt.get("mcse_mmol_g"), 4)
            q_str = (
                f"{_fmt(pt.get('Qst_fluc_kJ_mol'), 1)} ± {_fmt(pt.get('Qst_fluc_sem_kJ_mol'), 1)}"
                if pt.get("Qst_fluc_kJ_mol") is not None
                else f"N/A ({pt.get('qst_status', 'unk')})"
            )
            rh_str = f"{_fmt(pt.get('N_rhat_max'), 3)} / {_fmt(pt.get('U_rhat_max'), 3)} / {_fmt(pt.get('OMS_rhat_max'), 3)}"
            t_str = (
                f"{_fmt(pt.get('time_to_mcse_0p1_s'), 0)} / "
                f"{_fmt(pt.get('time_to_mcse_0p1_amortized_s'), 0)} / "
                f"{_fmt(pt.get('time_to_mcse_0p1_coldstart_s'), 0)}"
            )
            sym_str = f"{_fmt(pt.get('proposal_symmetry_ratio'), 2)}" + (" (!)" if pt.get("proposal_asymmetry_flag") else "")
            cap_str = f"{pt.get('max_N_observed')}/{pt.get('max_mol')} ({pt.get('saturation_safety_flag')})"
            st_str = str(pt.get("diag_status", "unknown")).upper()
            lines.append(
                f"| `{rkey}` | {sys_str} | {pmode} | `{mstr}` | {pt['P_bar']:g} | {up_str} | {mcse_str} | "
                f"{q_str} | {rh_str} | {_fmt(pt.get('N_ess_bulk'), 0)} | {_fmt(pt.get('ess_bulk_N_per_gpu_s'), 3)} | "
                f"{t_str} | {sym_str} | {cap_str} | **{st_str}** |"
            )

    lines.extend([
        "",
        "## 2. Distillation Fidelity & Provenance Audit",
        "",
        "| Cache ID | System | Model | `model_sha256` | 1B R² / MAE (kJ/mol) | 2B Deployed Hold-out MAE | 2B Classical Hold-out MAE | `n_2b_val` | 2B Audit | Wall (s) |",
        "|---|---|---|---|---|---:|---:|---:|---|---:|",
    ])
    for cid, d in sorted(summary.get("distillation", {}).items()):
        sys_str = f"{_canonical_mof_name(str(d.get('cif_path', '')))} + {str(d.get('guest_name', '')).upper()}"
        mstr = f"{d.get('model')}" + ("+D3" if d.get("dispersion") else "")
        sha = d.get("model_sha256")
        sha_str = f"`{sha[:12]}...`" if isinstance(sha, str) and sha.strip() else "**NULL**"
        b1_str = f"{_fmt(d.get('r2_1b'), 4)} / {_fmt(d.get('mae_1b_kJ_mol'), 2)}"
        m2_dep = _fmt(d.get("mae_2b_deployed_holdout_kJ_mol"), 2)
        m2_cls = _fmt(d.get("mae_2b_classical_holdout_kJ_mol"), 2)
        n2v = str(d.get("n_2b_val")) if d.get("n_2b_val") is not None else "N/A"
        lines.append(
            f"| `{cid}` | {sys_str} | `{mstr}` | {sha_str} | {b1_str} | {m2_dep} | {m2_cls} | "
            f"{n2v} | {d.get('two_b_holdout_audit')} | {_fmt(d.get('distill_wall_s'), 1)} |"
        )

    if au.get("remc_runs"):
        lines.extend([
            "",
            "## 3. REMC Replica-Exchange Swap Acceptance Rates",
            "",
            "| Run | Swap Every | Pressure Pair | Attempts | Accepted | Acceptance Rate |",
            "|---|---:|---|---:|---:|---:|",
        ])
        for rkey, rinfo in sorted(au["remc_runs"].items()):
            se = rinfo.get("swap_every")
            se_str = str(se) if se is not None else "N/A"
            for pair in rinfo.get("pairs", []):
                rate = pair.get("acc_rate")
                r_str = f"{100.0 * float(rate):.1f}%" if rate is not None else "N/A"
                att = str(pair["attempts"]) if pair.get("attempts") is not None else "N/A"
                acc = str(pair["accepted"]) if pair.get("accepted") is not None else "N/A"
                lines.append(f"| `{rkey}` | {se_str} | {pair['pair_label']} | {att} | {acc} | {r_str} |")

    flags: List[str] = []
    if au["model_sha256_missing"]:
        flags.append(f"- **Missing `model_sha256`**: {', '.join(f'`{x}`' for x in au['model_sha256_missing'])}")
    if au["honest_2b_holdout_missing"]:
        flags.append(
            f"- **Missing deployed 2B hold-out metrics (legacy cache)**: "
            f"{', '.join(f'`{x}`' for x in au['honest_2b_holdout_missing'])}"
        )
    for item in au["saturation_failures"]:
        flags.append(
            f"- **SATURATED `max_mol`**: `{item['run']}` @ {item['P_bar']} bar reached "
            f"N={item['max_N_observed']} >= cap {item['max_mol']}"
        )
    for item in au["saturation_warnings"]:
        flags.append(
            f"- **Near `max_mol` cap (>=90%)**: `{item['run']}` @ {item['P_bar']} bar reached "
            f"N={item['max_N_observed']}/{item['max_mol']}"
        )
    for item in au["asymmetric_proposal_points"]:
        flags.append(
            f"- **Proposal asymmetry**: `{item['run']}` @ {item['P_bar']} bar "
            f"(ins_att={item['prod_ins_att']}, del_att={item['prod_del_att']}, ratio={item['symmetry_ratio']:.2f})"
        )
    for item in au["recompute_discrepancies"]:
        flags.append(f"- **Recomputation discrepancy**: {item}")

    lines.extend(["", "## 4. Flagged Issues & Warnings", ""])
    if flags:
        lines.extend(flags)
    else:
        lines.append("No provenance, saturation, symmetry, or recomputation anomalies detected.")
    lines.append("")
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description="Archive & independently audit JCTC MLIP-GCMC benchmark outputs")
    p.add_argument("--results-dir", required=True, help="Directory containing raw run/widom JSON files")
    p.add_argument("--out-json", default="", help="Output path for jctc_summary.json (default: <results-dir>/jctc_summary.json)")
    p.add_argument("--out-traces", default="", help="Output path for jctc_traces.npz (default: sibling of --out-json)")
    p.add_argument("--out-report", default="", help="Output path for jctc_audit_report.md (default: sibling of --out-json)")
    p.add_argument("--exp-dir", default=str(ROOT_DIR / "data" / "exp_isotherms"), help="Experimental isotherms directory")
    p.add_argument("--embed-traces", action="store_true", help="Also embed full prod_*_history arrays inside jctc_summary.json")
    p.add_argument("--verify", action="store_true", help="Run round-trip archive recomputation verification after saving")
    return p.parse_args()


def main():
    args = parse_args()
    res_dir = Path(args.results_dir)
    out_json = Path(args.out_json) if args.out_json else (res_dir / "jctc_summary.json")
    out_traces = Path(args.out_traces) if args.out_traces else (out_json.parent / "jctc_traces.npz")
    out_report = Path(args.out_report) if args.out_report else (out_json.parent / "jctc_audit_report.md")
    exp_dir = Path(args.exp_dir) if args.exp_dir else None

    summary = archive_results_dir(
        results_dir=res_dir,
        out_json=out_json,
        out_traces=out_traces,
        out_report=out_report,
        exp_dir=exp_dir,
        embed_traces=bool(args.embed_traces),
    )
    au = summary["audit"]
    print(
        f"[Archive DONE] {au['n_runs']} runs, {au['n_points_total']} points "
        f"(pass={au['n_points_pass']}, fail={au['n_points_fail']}, insuff={au['n_points_insufficient']}) | "
        f"model_sha256_ok={au['model_sha256_complete']} | sat_ok={au['saturation_safety_pass']}"
    )
    print(f"  Summary: {out_json}")
    print(f"  Traces:  {out_traces}")
    print(f"  Report:  {out_report}")

    if args.verify:
        chk = verify_archive_recomputability(out_json, out_traces)
        print(f"  [Verify Archive] checked={chk['checked_points']} ok={chk['ok']}")
        if not chk["ok"]:
            for m in chk["mismatches"]:
                print(f"    MISMATCH: {m}")
            sys.exit(2)


if __name__ == "__main__":
    main()
