"""Run provenance metadata collection and strict checkpoint/resume compatibility verification."""

import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional


def sha256_file(path: Optional[str]) -> Optional[str]:
    """Return lowercase hex SHA-256 of a file, or None if missing."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_json_obj(obj: Any) -> str:
    """Return deterministic SHA-256 hex digest of a JSON-serializable object."""
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def get_git_info(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Capture git HEAD commit, dirty status, and short diff stat."""
    root = repo_root or Path(__file__).resolve().parent.parent
    info: Dict[str, Any] = {"commit": None, "dirty": None, "diff_stat": None}
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root), stderr=subprocess.DEVNULL, text=True
        ).strip()
        info["commit"] = head
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(root), stderr=subprocess.DEVNULL, text=True
        ).strip()
        info["dirty"] = bool(status)
        if status:
            stat = subprocess.check_output(
                ["git", "diff", "--stat"], cwd=str(root), stderr=subprocess.DEVNULL, text=True
            ).strip()
            info["diff_stat"] = stat
    except Exception:
        pass
    return info


def _guest_metadata(guest) -> Dict[str, Any]:
    sites_all = [
        {
            "symbol": s.symbol,
            "Z": int(s.Z),
            "pos": [float(v) for v in s.pos],
            "q": float(s.q),
            "eps_K": float(s.eps_K),
            "sigma_A": float(s.sigma_A),
            "lj_type": int(s.lj_type),
        }
        for s in guest.sites
    ]
    real_geom = [
        {"symbol": s.symbol, "Z": int(s.Z), "pos": [round(float(v), 8) for v in s.pos]}
        for s in guest.real_sites()
    ]
    return {
        "name": str(guest.name),
        "formula": str(guest.formula),
        "is_isotropic": bool(guest.is_isotropic),
        "Tc_K": float(guest.Tc_K),
        "Pc_bar": float(guest.Pc_bar),
        "omega": float(guest.omega),
        "sites": sites_all,
        "guest_real_geom_sha256": sha256_json_obj(real_geom),
        "guest_full_params_sha256": sha256_json_obj(sites_all),
    }


def _package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for mod_name in ("numpy", "scipy", "torch", "ase", "mace", "torch_dftd"):
        try:
            m = __import__(mod_name)
            versions[mod_name] = str(getattr(m, "__version__", "installed"))
        except Exception:
            versions[mod_name] = None
    try:
        import torch
        versions["cuda_version"] = str(torch.version.cuda)
        versions["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            versions["device_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return versions


def collect_run_metadata(
    args,
    fw_data: dict,
    guest,
    grid_dims,
    target=None,
    prop_info: Optional[dict] = None,
    cli_argv: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Collect full §4.3 metadata dict for provenance tracking and resume verification."""
    cif_path = str(args.cif)
    charges = [round(float(q), 8) for q in fw_data["charges"]]
    guest_meta = _guest_metadata(guest)

    model_file = getattr(target, "model_file_path", None) if target is not None else None
    model_sha = sha256_file(model_file) if model_file else None
    mace_r_max = getattr(target, "r_max", None) if target is not None else None
    d3_cutoff = getattr(target, "rc_d3", None) if target is not None else None
    dtype_str = str(getattr(args, "dtype", getattr(target, "dtype", "float32")))

    meta = {
        "cli_argv": list(sys.argv if cli_argv is None else cli_argv),
        "git": get_git_info(),
        "cif_path": cif_path,
        "cif_name": Path(cif_path).name,
        "cif_sha256": sha256_file(cif_path),
        "supercell": [int(x) for x in args.supercell],
        "T_K": float(args.temp),
        "charge_mode": str(args.charge_mode),
        "charge_source": str(fw_data["charge_source"]),
        "framework_charges_sha256": sha256_json_obj(charges),
        "n_unit_atoms": int(len(fw_data["Z"])),
        "guest_meta": guest_meta,
        "guest_real_geom_sha256": guest_meta["guest_real_geom_sha256"],
        "proposer": str(args.proposer),
        "model": str(args.model),
        "model_file_path": model_file,
        "model_sha256": model_sha,
        "dispersion": bool(args.dispersion == "on"),
        "dispersion_xc": str(args.xc),
        "dtype": dtype_str,
        "mace_chunk": getattr(args, "mace_chunk", None),
        "mace_r_max_A": mace_r_max,
        "d3_cutoff_A": d3_cutoff,
        "grid_dims": [int(x) for x in grid_dims],
        "grid_spacing": float(getattr(args, "grid_spacing", 0.11)),
        "r_cut_gg_A": None if prop_info is None else prop_info.get("r_cut_gg"),
        "min_perp_width_A": None if prop_info is None else prop_info.get("min_perp_width_A"),
        "k_steps": int(args.k_steps),
        "eq_cycles": int(args.eq_cycles),
        "prod_cycles": int(args.prod_cycles),
        "pilot_cycles": int(args.pilot_cycles),
        "shift_mode": str(args.shift_mode),
        "shift_override": None if args.shift_override is None else float(args.shift_override),
        "init_mode": str(getattr(args, "init_mode", "auto")),
        "init_loading": float(getattr(args, "init_loading", 8.0)),
        "boxes": int(args.boxes),
        "max_mol": int(args.max_mol),
        "seed": int(args.seed),
        "device": str(args.device),
        "versions": _package_versions(),
    }
    return meta


def validate_resume_metadata(old_meta: dict, new_meta: dict, allow_extend_prod: bool = False) -> None:
    """Reject any scientific/configuration conflict when resuming or extending checkpoints."""
    mismatches: List[str] = []

    old_cif_sha = old_meta.get("cif_sha256")
    new_cif_sha = new_meta.get("cif_sha256")
    if old_cif_sha and new_cif_sha:
        if old_cif_sha != new_cif_sha:
            mismatches.append(f"cif_sha256: {old_cif_sha[:12]} != {new_cif_sha[:12]}")
    elif old_meta.get("cif_name") != new_meta.get("cif_name"):
        mismatches.append(f"cif_name: {old_meta.get('cif_name')} != {new_meta.get('cif_name')}")

    exact_keys = [
        "framework_charges_sha256",
        "guest_real_geom_sha256",
        "supercell",
        "proposer",
        "model",
        "dispersion",
        "dispersion_xc",
        "dtype",
        "grid_dims",
        "k_steps",
        "init_mode",
        "boxes",
        "max_mol",
    ]
    for k in exact_keys:
        if k in old_meta and k in new_meta and old_meta[k] != new_meta[k]:
            mismatches.append(f"{k}: {old_meta[k]} != {new_meta[k]}")

    if old_meta.get("model_sha256") and new_meta.get("model_sha256"):
        if old_meta["model_sha256"] != new_meta["model_sha256"]:
            mismatches.append(
                f"model_sha256: {old_meta['model_sha256'][:12]} != {new_meta['model_sha256'][:12]}"
            )

    if "T_K" in old_meta and "T_K" in new_meta:
        if not math.isclose(float(old_meta["T_K"]), float(new_meta["T_K"]), rel_tol=1e-6, abs_tol=1e-6):
            mismatches.append(f"T_K: {old_meta['T_K']} != {new_meta['T_K']}")

    if old_meta.get("shift_override") != new_meta.get("shift_override"):
        mismatches.append(f"shift_override: {old_meta.get('shift_override')} != {new_meta.get('shift_override')}")

    if not allow_extend_prod:
        for k in ("eq_cycles", "prod_cycles"):
            if k in old_meta and k in new_meta and old_meta[k] != new_meta[k]:
                mismatches.append(f"{k}: {old_meta[k]} != {new_meta[k]}")
    else:
        if "eq_cycles" in old_meta and "eq_cycles" in new_meta and old_meta["eq_cycles"] != new_meta["eq_cycles"]:
            mismatches.append(f"eq_cycles: {old_meta['eq_cycles']} != {new_meta['eq_cycles']}")

    if mismatches:
        raise ValueError("Cannot resume due to configuration mismatch: " + "; ".join(mismatches))
