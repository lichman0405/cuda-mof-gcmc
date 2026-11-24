"""Unified Hybrid GPU-GCMC + MLIP Nested Monte Carlo CLI Engine.

Example:
  .venv-mlip\\Scripts\\python.exe -u scripts\\run_mlip_gcmc.py ^
      --cif data\\MgMOF-74.cif --supercell 1 1 2 --temp 298 ^
      --pressures 0.002 0.01 0.05 0.15 --proposer mercado ^
      --model mace-matpes-r2scan-0 --dispersion off ^
      --boxes 4 --k-steps 25 --eq-cycles 120 --prod-cycles 320 ^
      --out-json build\\phase3_mgmof74_r2scan.json
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time
import numpy as np

from multi_gpu import (
    parse_gpu_ids,
    resolve_worker_cuda_visible_devices,
    run_pressures_parallel,
    run_replicas_parallel,
)
from nested_mc_engine import (
    CudaGcmcProposer,
    FastMaceD3Target,
    K_TO_KJMOL,
    run_nested_mc,
    verify_proposer_target_consistency,
)
from guests import load_guest
from qeq_charges import extract_framework_for_proposer

PROPOSER_MAP = {
    "uff": (0, 0),       # Generic UFF + REPEAT (0 charge supported) + TraPPE CO2
    "pol3site": (0, 2),  # UFF + 3-site framework E-field polarization + Thole
    "mercado": (1, 0),   # Mercado 2016 DFT-FF + EPM2 CO2
}


def parse_args():
    p = argparse.ArgumentParser(description="Hybrid GPU-GCMC + MLIP Nested Monte Carlo Engine")
    p.add_argument("--cif", required=True, help="Path to MOF CIF file")
    p.add_argument("--supercell", nargs=3, type=int, default=[1, 1, 2], help="Supercell replicas na nb nc")
    p.add_argument("--temp", type=float, default=298.0, help="Temperature in Kelvin")
    p.add_argument("--pressures", nargs="+", type=float, required=True, help="Pressures in bar")
    p.add_argument("--guest", default="co2", help="Rigid guest preset (co2, ch4, n2, h2, ar, h2o, nh3)")
    p.add_argument("--guest-json", default="", help="Optional custom rigid guest JSON path")
    p.add_argument("--proposer", choices=["uff", "pol3site", "mercado"], default="mercado", help="Inner GPU classical proposer")
    p.add_argument("--charge-mode", choices=["auto", "cif", "qeq", "zero"], default="auto", help="Inner framework charge mode")
    p.add_argument("--model", default="medium", help="MACE foundation model name or checkpoint path")
    p.add_argument("--dispersion", choices=["on", "off"], default="on", help="Enable TorchDFTD3 D3(BJ)")
    p.add_argument("--xc", default="pbe", help="D3(BJ) functional parameter set (e.g. pbe, r2scan)")
    p.add_argument("--device", default="cuda", help="PyTorch MLIP device")
    p.add_argument("--mace-chunk", type=int, default=None, help="Micro-batch size for MACE forward (auto by VRAM if omitted)")
    p.add_argument("--grid-dims", nargs=3, type=int, default=None, help="3D tricubic grid dims (auto from --grid-spacing if omitted)")
    p.add_argument("--grid-spacing", type=float, default=0.11, help="Target unit-cell grid spacing (Angstrom) for auto grid dims")
    p.add_argument("--boxes", type=int, default=4, help="Parallel simulation replicas")
    p.add_argument("--max-mol", type=int, default=128, help="Max guest molecules per supercell")
    p.add_argument("--k-steps", type=int, default=25, help="Inner GPU GCMC steps per outer Nested MC cycle")
    p.add_argument("--eq-cycles", type=int, default=120, help="Outer equilibration cycles")
    p.add_argument("--prod-cycles", type=int, default=320, help="Outer production cycles")
    p.add_argument("--pilot-cycles", type=int, default=60, help="Target-ensemble pilot burn-in cycles for Stage-3 shift calibration (0 disables)")
    p.add_argument("--shift-override", type=float, default=None, help="Optional fixed chem-pot shift (kJ/mol) overriding auto calibration")
    p.add_argument("--seed", type=int, default=20261400, help="Base PRNG seed")
    p.add_argument("--shift-mode", choices=["auto", "none"], default="auto", help="Multi-stage self-consistent chem-pot shift")
    p.add_argument("--gpus", nargs="+", default=None, help="GPU IDs for multi-GPU execution (e.g. --gpus 0 1 2 3 or 0,1,2,3)")
    p.add_argument("--parallel-mode", choices=["auto", "pressure", "replica"], default="auto", help="Multi-GPU parallel strategy")
    p.add_argument("--worker-threads", type=int, default=4, help="CPU threads per multi-GPU worker process")
    p.add_argument("--resume", action="store_true", help="Resume from existing --out-json, skipping completed pressures")
    p.add_argument("--out-json", required=True, help="Output JSON path")
    p.add_argument("--out-csv", default="", help="Optional summary CSV path")
    return p.parse_args()


def compute_grid_dims(uc_atoms, grid_spacing: float = 0.11):
    lengths = np.linalg.norm(np.array(uc_atoms.get_cell(), dtype=np.float64), axis=1)
    dims = []
    for L in lengths:
        n = int(math.ceil(float(L) / float(grid_spacing)))
        n = ((n + 7) // 8) * 8
        dims.append(max(32, min(256, n)))
    return tuple(dims)


def parse_cuda_device_id(device_str: str) -> int:
    s = str(device_str).strip().lower()
    if s.startswith("cuda:"):
        return int(s.split(":", 1)[1])
    return 0


def evaluate_convergence_diagnostics(res: dict) -> dict:
    uptake = float(res["uptake_mmol_g"])
    sem = float(res["uptake_sem_mmol_g"])
    drift = float(res["stationarity_drift_mmol_g"])
    ins_acc = int(res["prod_ins_acc"])
    del_acc = int(res["prod_del_acc"])
    outer_acc = float(res["prod_outer_acc_rate"])

    ins_acc_ok = (ins_acc >= 5) if uptake >= 1e-4 else True
    del_acc_ok = (del_acc >= 5) if uptake >= 1e-4 else True
    drift_limit = max(0.25, 3.0 * sem)
    drift_ok = bool(drift <= drift_limit)
    outer_acc_ok = bool(outer_acc >= 0.05)

    warnings = []
    if not ins_acc_ok:
        warnings.append(f"Low production insertions accepted ({ins_acc} < 5 at uptake {uptake:.3f} mmol/g)")
    if not del_acc_ok:
        warnings.append(f"Low production deletions accepted ({del_acc} < 5 at uptake {uptake:.3f} mmol/g)")
    if not drift_ok:
        warnings.append(f"High half-split drift ({drift:.3f} > max(0.25, 3*SEM)={drift_limit:.3f} mmol/g)")
    if not outer_acc_ok:
        warnings.append(f"Low production outer acceptance rate ({outer_acc*100:.1f}% < 5.0%)")

    return {
        "converged": bool(ins_acc_ok and del_acc_ok and drift_ok and outer_acc_ok),
        "ins_acc_ok": ins_acc_ok,
        "del_acc_ok": del_acc_ok,
        "drift_ok": drift_ok,
        "outer_acc_ok": outer_acc_ok,
        "warnings": warnings,
    }


def save_outputs(out_json: Path, out_csv: Path | None, payload: dict):
    payload["points"] = sorted(payload["points"], key=lambda pt: float(pt["P_bar"]))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_json = out_json.with_suffix(out_json.suffix + ".tmp")
    tmp_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_json, out_json)

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        tmp_csv = out_csv.with_suffix(out_csv.suffix + ".tmp")
        with tmp_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "P_bar", "uptake_mmol_g", "uptake_sem_mmol_g", "Qst_kJ_mol", "Qst_sem_kJ_mol",
                "delta_eps_kJ_mol", "outer_acc_rate", "prod_ins_acc", "prod_ins_att",
                "prod_del_acc", "prod_del_att", "stationarity_drift_mmol_g",
                "tau_int_N", "ess_N", "ess_N_per_s", "elapsed_s", "converged"
            ])
            for pt in payload["points"]:
                conv = pt.get("diagnostics", {}).get("converged", True)
                w.writerow([
                    pt["P_bar"], pt["uptake_mmol_g"], pt["uptake_sem_mmol_g"],
                    pt["Qst_kJ_mol"], pt["Qst_sem_kJ_mol"], pt["delta_eps_kJ_mol"],
                    pt["outer_acc_rate"], pt["prod_ins_acc"], pt["prod_ins_att"],
                    pt["prod_del_acc"], pt["prod_del_att"], pt["stationarity_drift_mmol_g"],
                    pt.get("tau_int_N", 1.0), pt.get("ess_N", 0.0), pt.get("ess_N_per_s", 0.0),
                    pt["elapsed_s"], conv,
                ])
        os.replace(tmp_csv, out_csv)


def _cif_matches(old_cif: str, new_cif: str) -> bool:
    p_old, p_new = Path(old_cif), Path(new_cif)
    try:
        if p_old.resolve() == p_new.resolve():
            return True
    except OSError:
        pass
    return p_old.name == p_new.name


def run_pressure_point(
    fw_data: dict,
    guest,
    supercell,
    T_K: float,
    P_bar: float,
    ff_mode: int,
    pol_mode: int,
    grid_dims,
    n_boxes: int,
    max_mol: int,
    k_steps: int,
    eq_cycles: int,
    prod_cycles: int,
    seed: int,
    shift_mode: str,
    target: FastMaceD3Target,
    pilot_cycles: int = 60,
    shift_override=None,
    verify_consistency: bool = False,
    device_id: int = 0,
):
    prop = CudaGcmcProposer.from_atoms(
        fw_data=fw_data,
        guest=guest,
        supercell=tuple(supercell),
        T_K=T_K,
        pressure_bar=P_bar,
        ff_mode=ff_mode,
        pol_mode=pol_mode,
        grid_dims=tuple(grid_dims),
        n_boxes=n_boxes,
        max_mol=max_mol,
        seed=seed,
        device_id=device_id,
    )
    if verify_consistency:
        cons = verify_proposer_target_consistency(prop, target)
        print(
            f"  [Consistency OK] unit={cons['n_unit_atoms']} super={cons['n_super_atoms']} "
            f"max_cell_err={cons['max_cell_err_A']:.2e} A max_pos_err={cons['max_pos_err_A']:.2e} A "
            f"min_perp={cons['min_perp_width_A']:.2f} A"
        )
    conv_mmol_g = 1000.0 / prop.fw_mass_gmol

    if shift_override is not None:
        de_final = float(shift_override)
        prop.set_chem_pot_shift(de_final)
        prop.equilibrate(30000)
        fug_mult = math.exp(-de_final / (K_TO_KJMOL * T_K))
        print(f"  [P={P_bar:.4g} bar] Override Shift={de_final:.2f} kJ/mol ({fug_mult:.2f}x inner f_ref)")
    elif shift_mode == "auto":
        prop.equilibrate(20000)
        cal1_st = prop.get_accepted_quat()
        if np.mean([s["N"] for s in cal1_st]) < 4.0:
            for boost in (-4.0, -8.0, -12.0, -16.0):
                prop.set_chem_pot_shift(boost)
                prop.equilibrate(15000)
                cal1_st = prop.get_accepted_quat()
                if np.mean([s["N"] for s in cal1_st]) >= 4.0:
                    break
        cal1_u = target.eval_batch(cal1_st)
        diffs1 = [
            float(((cal1_u[b][0] - cal1_st[b]["U_ref_K"]) * K_TO_KJMOL) / cal1_st[b]["N"])
            for b in range(n_boxes) if cal1_st[b]["N"] > 0 and not cal1_u[b][1]
        ]
        # Cap initial classical positive shift to +2.0 kJ/mol to avoid starving inner proposals
        de1 = float(max(-24.0, min(2.0, np.mean(diffs1)))) if diffs1 else 0.0
        prop.set_chem_pot_shift(de1)
        prop.equilibrate(30000)

        cal2_st = prop.get_accepted_quat()
        cal2_u = target.eval_batch(cal2_st)
        diffs2 = [
            float(((cal2_u[b][0] - cal2_st[b]["U_ref_K"]) * K_TO_KJMOL) / cal2_st[b]["N"])
            for b in range(n_boxes) if cal2_st[b]["N"] > 0 and not cal2_u[b][1]
        ]
        de2 = float(max(-24.0, min(2.0, np.mean(diffs2)))) if diffs2 else de1
        prop.set_chem_pot_shift(de2)
        prop.equilibrate(20000)

        if pilot_cycles > 0:
            # Stage 3: two-step target-ensemble pilot burn-in without artificial 0.5 classical damping
            if pilot_cycles >= 20:
                p1 = pilot_cycles // 2
                p2 = pilot_cycles - p1
                _ = run_nested_mc(
                    prop, None, k_steps=k_steps, eq_cycles=p1, prod_cycles=0,
                    seed=seed + 5, log_every=p1 + 10, batch_target_fn=target.eval_batch,
                )
                cal3a_st = prop.get_accepted_quat()
                cal3a_u = target.eval_batch(cal3a_st)
                diffs3a = [
                    float(((cal3a_u[b][0] - cal3a_st[b]["U_ref_K"]) * K_TO_KJMOL) / cal3a_st[b]["N"])
                    for b in range(n_boxes) if cal3a_st[b]["N"] > 0 and not cal3a_u[b][1]
                ]
                de3a = float(max(-24.0, min(4.0, np.mean(diffs3a)))) if diffs3a else de2
                prop.set_chem_pot_shift(de3a)
                if abs(de3a - de2) > 1.0:
                    prop.equilibrate(10000)
                _ = run_nested_mc(
                    prop, None, k_steps=k_steps, eq_cycles=p2, prod_cycles=0,
                    seed=seed + 9, log_every=p2 + 10, batch_target_fn=target.eval_batch,
                )
                cal3b_st = prop.get_accepted_quat()
                cal3b_u = target.eval_batch(cal3b_st)
                diffs3b = [
                    float(((cal3b_u[b][0] - cal3b_st[b]["U_ref_K"]) * K_TO_KJMOL) / cal3b_st[b]["N"])
                    for b in range(n_boxes) if cal3b_st[b]["N"] > 0 and not cal3b_u[b][1]
                ]
                de_final = float(max(-24.0, min(4.0, np.mean(diffs3b)))) if diffs3b else de3a
                prop.set_chem_pot_shift(de_final)
                if abs(de_final - de3a) > 1.0:
                    prop.equilibrate(10000)
            else:
                _ = run_nested_mc(
                    prop, None, k_steps=k_steps, eq_cycles=pilot_cycles, prod_cycles=0,
                    seed=seed + 5, log_every=pilot_cycles + 10, batch_target_fn=target.eval_batch,
                )
                cal3_st = prop.get_accepted_quat()
                cal3_u = target.eval_batch(cal3_st)
                diffs3 = [
                    float(((cal3_u[b][0] - cal3_st[b]["U_ref_K"]) * K_TO_KJMOL) / cal3_st[b]["N"])
                    for b in range(n_boxes) if cal3_st[b]["N"] > 0 and not cal3_u[b][1]
                ]
                de_final = float(max(-24.0, min(4.0, np.mean(diffs3)))) if diffs3 else de2
                prop.set_chem_pot_shift(de_final)
                prop.equilibrate(10000)

            fug_mult = math.exp(-de_final / (K_TO_KJMOL * T_K))
            print(
                f"  [P={P_bar:.4g} bar] Shift Stage1={de1:.2f} -> Stage2={de2:.2f} -> "
                f"PilotStage3({pilot_cycles}cyc)={de_final:.2f} kJ/mol ({fug_mult:.2f}x inner f_ref)"
            )
        else:
            de_final = de2
            fug_mult = math.exp(-de_final / (K_TO_KJMOL * T_K))
            print(f"  [P={P_bar:.4g} bar] Shift Stage1={de1:.2f} -> Stage2={de_final:.2f} kJ/mol ({fug_mult:.2f}x inner f_ref)")
    else:
        de_final = 0.0
        prop.equilibrate(30000)

    warm_st = prop.get_accepted_quat()
    print(f"  [P={P_bar:.4g} bar] Warm-start N={[s['N'] for s in warm_st]} ({[round(s['N']*conv_mmol_g, 2) for s in warm_st]} mmol/g)")

    target.n_evals = 0
    res = run_nested_mc(
        prop, None,
        k_steps=k_steps, eq_cycles=eq_cycles, prod_cycles=prod_cycles,
        seed=seed + 17, log_every=max(50, (eq_cycles + prod_cycles) // 5),
        batch_target_fn=target.eval_batch,
    )

    hist = np.array(res["box_N_history"], dtype=np.float64) * conv_mmol_g
    prod_hist = hist[eq_cycles:]
    if prod_hist.shape[0] >= 2:
        half = prod_hist.shape[0] // 2
        mean_h1 = float(np.mean(prod_hist[:half]))
        mean_h2 = float(np.mean(prod_hist[half:]))
        drift = abs(mean_h2 - mean_h1)
    elif prod_hist.shape[0] == 1:
        mean_h1 = float(np.mean(prod_hist))
        mean_h2 = mean_h1
        drift = 0.0
    else:
        mean_h1 = 0.0
        mean_h2 = 0.0
        drift = 0.0

    res["P_bar"] = float(P_bar)
    res["T_K"] = float(T_K)
    res["fugacity_Pa"] = float(prop.f_Pa)
    res["fw_mass_gmol"] = float(prop.fw_mass_gmol)
    res["first_half_uptake_mmol_g"] = mean_h1
    res["second_half_uptake_mmol_g"] = mean_h2
    res["stationarity_drift_mmol_g"] = drift
    res["mlip_evals"] = int(target.n_evals)
    res["total_inner_steps"] = int(n_boxes * k_steps * (eq_cycles + prod_cycles))
    diag = evaluate_convergence_diagnostics(res)
    res["diagnostics"] = diag

    print(
        f"  [P={P_bar:.4g} bar DONE] Uptake={res['uptake_mmol_g']:.4f} +- {res['uptake_sem_mmol_g']:.4f} mmol/g | "
        f"Fluc Qst={res['Qst_kJ_mol']:.2f} +- {res['Qst_sem_kJ_mol']:.2f} | "
        f"Molar Qst={res['mean_molar_qst_kJ_mol']:.2f} +- {res['mean_molar_qst_sem_kJ_mol']:.2f} kJ/mol | "
        f"Ins={res['prod_ins_acc']}/{res['prod_ins_att']} Del={res['prod_del_acc']}/{res['prod_del_att']} | "
        f"Acc={res['outer_acc_rate']*100:.1f}% | tauN={res['tau_int_N']:.1f} ESS/s={res['ess_N_per_s']:.2f} | "
        f"|drift|={drift:.3f} | conv={diag['converged']} | {res['elapsed_s']:.1f}s"
    )
    if not diag["converged"]:
        for msg in diag["warnings"]:
            print(f"  [DIAG WARNING] P={P_bar:.4g} bar: {msg}")
    prop.close()
    if getattr(target, "device", None) is not None and target.device.type == "cuda":
        target.torch.cuda.empty_cache()
    return res


def main():
    t_start_all = time.perf_counter()
    args = parse_args()
    ff_mode, pol_mode = PROPOSER_MAP[args.proposer]
    use_disp = args.dispersion == "on"
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv) if args.out_csv else None

    gpu_ids = parse_gpu_ids(args.gpus) if args.gpus else []
    if len(gpu_ids) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = resolve_worker_cuda_visible_devices(
            gpu_ids[0], os.environ.get("CUDA_VISIBLE_DEVICES")
        )
        args.device = "cuda:0"
        device_id = 0
    else:
        device_id = parse_cuda_device_id(args.device)

    guest = load_guest(name=args.guest, guest_json=args.guest_json, proposer=args.proposer)
    guest_label = Path(args.guest_json).stem if args.guest_json else args.guest.strip().lower()
    if ff_mode == 1 and guest.formula != "CO2":
        raise ValueError(f"--proposer mercado only supports CO2; use --proposer uff for '{guest.name}'.")

    fw_data = extract_framework_for_proposer(args.cif, charge_mode=args.charge_mode)
    grid_dims = tuple(args.grid_dims) if args.grid_dims is not None else compute_grid_dims(fw_data["uc"], args.grid_spacing)

    eff_mode = "single"
    if len(gpu_ids) > 1:
        if args.parallel_mode == "auto":
            eff_mode = "pressure" if len(args.pressures) >= 2 else ("replica" if args.boxes >= 2 else "pressure")
        else:
            eff_mode = args.parallel_mode

    print(
        f"=== Hybrid GPU-GCMC + MLIP Engine ===\n"
        f"  CIF={args.cif} supercell={args.supercell} T={args.temp} K Pressures={args.pressures}\n"
        f"  Guest={guest.name} ({guest.formula}, {len(guest.sites)} sites) | Grid={grid_dims} | Charges={fw_data['charge_source']}\n"
        f"  Proposer={args.proposer} (ff={ff_mode}, pol={pol_mode}) | MLIP={args.model} (dispersion={use_disp}, xc={args.xc})\n"
        f"  GPUs={gpu_ids if gpu_ids else [device_id]} | ParallelMode={eff_mode}"
    )

    payload = {
        "cif": str(args.cif),
        "supercell": list(args.supercell),
        "T_K": float(args.temp),
        "guest": guest_label,
        "charge_mode": args.charge_mode,
        "charge_source": fw_data["charge_source"],
        "proposer": args.proposer,
        "model": args.model,
        "dispersion": use_disp,
        "dispersion_xc": args.xc,
        "gpus": gpu_ids if gpu_ids else [device_id],
        "parallel_mode": eff_mode,
        "points": [],
    }

    if args.resume and out_json.exists():
        existing = json.loads(out_json.read_text(encoding="utf-8"))
        mismatches = []
        if not _cif_matches(str(existing.get("cif", "")), str(args.cif)):
            mismatches.append(f"cif: {existing.get('cif')} != {args.cif}")
        if list(existing.get("supercell", [])) != list(args.supercell):
            mismatches.append(f"supercell: {existing.get('supercell')} != {list(args.supercell)}")
        if not math.isclose(float(existing.get("T_K", -1.0)), float(args.temp), rel_tol=1e-6, abs_tol=1e-6):
            mismatches.append(f"T_K: {existing.get('T_K')} != {args.temp}")
        if existing.get("guest", guest_label) != guest_label:
            mismatches.append(f"guest: {existing.get('guest')} != {guest_label}")
        if existing.get("proposer") != args.proposer:
            mismatches.append(f"proposer: {existing.get('proposer')} != {args.proposer}")
        if existing.get("model") != args.model:
            mismatches.append(f"model: {existing.get('model')} != {args.model}")
        if bool(existing.get("dispersion")) != bool(use_disp):
            mismatches.append(f"dispersion: {existing.get('dispersion')} != {use_disp}")
        if existing.get("dispersion_xc") != args.xc:
            mismatches.append(f"dispersion_xc: {existing.get('dispersion_xc')} != {args.xc}")
        if mismatches:
            raise ValueError(
                f"Cannot --resume '{out_json}' due to configuration mismatch: " + "; ".join(mismatches)
            )
        payload["points"] = list(existing.get("points", []))
        print(f"  [RESUME] Loaded {len(payload['points'])} existing pressure point(s) from {out_json}")

    pending = []
    for i, p_bar in enumerate(args.pressures):
        if args.resume and any(
            math.isclose(float(pt["P_bar"]), float(p_bar), rel_tol=1e-6, abs_tol=1e-9)
            for pt in payload["points"]
        ):
            print(f"  [RESUME] Skipping completed P={p_bar} bar")
            continue
        pending.append((i, p_bar))

    if len(gpu_ids) > 1 and args.parallel_mode == "auto":
        eff_mode = "pressure" if len(pending) >= 2 else ("replica" if args.boxes >= 2 else "pressure")
        payload["parallel_mode"] = eff_mode

    if pending:
        if len(gpu_ids) > 1:
            run_cfg = {
                "cif_path": str(args.cif),
                "fw_data": fw_data,
                "guest": guest,
                "supercell": tuple(args.supercell),
                "T_K": float(args.temp),
                "ff_mode": ff_mode,
                "pol_mode": pol_mode,
                "grid_dims": grid_dims,
                "n_boxes": int(args.boxes),
                "max_mol": int(args.max_mol),
                "k_steps": int(args.k_steps),
                "eq_cycles": int(args.eq_cycles),
                "prod_cycles": int(args.prod_cycles),
                "base_seed": int(args.seed),
                "shift_mode": args.shift_mode,
                "pilot_cycles": int(args.pilot_cycles),
                "shift_override": args.shift_override,
                "model": args.model,
                "dispersion": use_disp,
                "dispersion_xc": args.xc,
                "mace_chunk": args.mace_chunk,
                "worker_threads": int(args.worker_threads),
            }

            def _on_point(pt: dict):
                payload["points"].append(pt)
                payload["total_wall_s"] = float(time.perf_counter() - t_start_all)
                save_outputs(out_json, out_csv, payload)
                print(f"  [Saved checkpoint] {out_json} ({len(payload['points'])} points)")

            if eff_mode == "pressure":
                run_pressures_parallel(pending, gpu_ids, run_cfg, on_point_done=_on_point)
            else:
                run_replicas_parallel(pending, gpu_ids, run_cfg, on_point_done=_on_point)
        else:
            t_str = str(max(1, int(args.worker_threads)))
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                os.environ.setdefault(var, t_str)
            target = FastMaceD3Target(
                cif_path=args.cif,
                supercell=tuple(args.supercell),
                model=args.model,
                device=args.device,
                dispersion=use_disp,
                dispersion_xc=args.xc,
                guest=guest,
                mace_chunk=args.mace_chunk,
                uc_atoms=fw_data["uc"],
            )
            target.torch.set_num_threads(max(1, int(args.worker_threads)))

            consistency_checked = False
            for i, p_bar in pending:
                print(f"\n--- Pressure Point {i+1}/{len(args.pressures)}: {p_bar} bar ---")
                t_pt = time.perf_counter()
                pt = run_pressure_point(
                    fw_data=fw_data,
                    guest=guest,
                    supercell=args.supercell,
                    T_K=args.temp,
                    P_bar=p_bar,
                    ff_mode=ff_mode,
                    pol_mode=pol_mode,
                    grid_dims=grid_dims,
                    n_boxes=args.boxes,
                    max_mol=args.max_mol,
                    k_steps=args.k_steps,
                    eq_cycles=args.eq_cycles,
                    prod_cycles=args.prod_cycles,
                    seed=args.seed + i * 100,
                    shift_mode=args.shift_mode,
                    target=target,
                    pilot_cycles=args.pilot_cycles,
                    shift_override=args.shift_override,
                    verify_consistency=(not consistency_checked),
                    device_id=device_id,
                )
                pt["point_wall_s"] = float(time.perf_counter() - t_pt)
                consistency_checked = True
                payload["points"].append(pt)
                payload["total_wall_s"] = float(time.perf_counter() - t_start_all)
                save_outputs(out_json, out_csv, payload)
                print(f"  [Saved checkpoint] {out_json} ({len(payload['points'])} points)")
    else:
        payload["total_wall_s"] = float(time.perf_counter() - t_start_all)
        save_outputs(out_json, out_csv, payload)
        print("  [RESUME] All requested pressures already completed.")

    payload["total_wall_s"] = float(time.perf_counter() - t_start_all)
    save_outputs(out_json, out_csv, payload)
    print(f"\nTotal Wall Time: {payload['total_wall_s']:.2f} s")
    print(f"Final JSON: {out_json}")
    if out_csv is not None:
        print(f"Final CSV:  {out_csv}")


if __name__ == "__main__":
    main()
