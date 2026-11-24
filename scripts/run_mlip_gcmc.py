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
import numpy as np

from nested_mc_engine import (
    CudaGcmcProposer,
    FastMaceD3Target,
    K_TO_KJMOL,
    run_nested_mc,
    verify_proposer_target_consistency,
)

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
    p.add_argument("--proposer", choices=["uff", "pol3site", "mercado"], default="mercado", help="Inner GPU classical proposer")
    p.add_argument("--model", default="medium", help="MACE foundation model name or checkpoint path")
    p.add_argument("--dispersion", choices=["on", "off"], default="on", help="Enable TorchDFTD3 D3(BJ)")
    p.add_argument("--xc", default="pbe", help="D3(BJ) functional parameter set (e.g. pbe, r2scan)")
    p.add_argument("--device", default="cuda", help="PyTorch MLIP device")
    p.add_argument("--grid-dims", nargs=3, type=int, default=[256, 256, 72], help="3D tricubic grid dims")
    p.add_argument("--boxes", type=int, default=4, help="Parallel simulation replicas")
    p.add_argument("--max-mol", type=int, default=128, help="Max CO2 molecules per supercell")
    p.add_argument("--k-steps", type=int, default=25, help="Inner GPU GCMC steps per outer Nested MC cycle")
    p.add_argument("--eq-cycles", type=int, default=120, help="Outer equilibration cycles")
    p.add_argument("--prod-cycles", type=int, default=320, help="Outer production cycles")
    p.add_argument("--pilot-cycles", type=int, default=60, help="Target-ensemble pilot burn-in cycles for Stage-3 shift calibration (0 disables)")
    p.add_argument("--shift-override", type=float, default=None, help="Optional fixed chem-pot shift (kJ/mol) overriding auto calibration")
    p.add_argument("--seed", type=int, default=20261400, help="Base PRNG seed")
    p.add_argument("--shift-mode", choices=["auto", "none"], default="auto", help="Multi-stage self-consistent chem-pot shift")
    p.add_argument("--resume", action="store_true", help="Resume from existing --out-json, skipping completed pressures")
    p.add_argument("--out-json", required=True, help="Output JSON path")
    p.add_argument("--out-csv", default="", help="Optional summary CSV path")
    return p.parse_args()


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
                "prod_del_acc", "prod_del_att", "stationarity_drift_mmol_g", "elapsed_s", "converged"
            ])
            for pt in payload["points"]:
                conv = pt.get("diagnostics", {}).get("converged", True)
                w.writerow([
                    pt["P_bar"], pt["uptake_mmol_g"], pt["uptake_sem_mmol_g"],
                    pt["Qst_kJ_mol"], pt["Qst_sem_kJ_mol"], pt["delta_eps_kJ_mol"],
                    pt["outer_acc_rate"], pt["prod_ins_acc"], pt["prod_ins_att"],
                    pt["prod_del_acc"], pt["prod_del_att"], pt["stationarity_drift_mmol_g"],
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
    cif_path: str,
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
):
    prop = CudaGcmcProposer(
        cif_path=cif_path,
        supercell=tuple(supercell),
        T_K=T_K,
        pressure_bar=P_bar,
        ff_mode=ff_mode,
        pol_mode=pol_mode,
        grid_dims=tuple(grid_dims),
        n_boxes=n_boxes,
        max_mol=max_mol,
        seed=seed,
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
        cal1_st = prop.get_accepted()
        if np.mean([s["N"] for s in cal1_st]) < 4.0:
            for boost in (-4.0, -8.0, -12.0):
                prop.set_chem_pot_shift(boost)
                prop.equilibrate(15000)
                cal1_st = prop.get_accepted()
                if np.mean([s["N"] for s in cal1_st]) >= 4.0:
                    break
        cal1_u = target.eval_batch(cal1_st)
        diffs1 = [
            float(((cal1_u[b][0] - cal1_st[b]["U_ref_K"]) * K_TO_KJMOL) / cal1_st[b]["N"])
            for b in range(n_boxes) if cal1_st[b]["N"] > 0 and not cal1_u[b][1]
        ]
        # Cap initial classical positive shift to +2.0 kJ/mol to avoid starving inner proposals
        de1 = float(max(-12.0, min(2.0, np.mean(diffs1)))) if diffs1 else 0.0
        prop.set_chem_pot_shift(de1)
        prop.equilibrate(30000)

        cal2_st = prop.get_accepted()
        cal2_u = target.eval_batch(cal2_st)
        diffs2 = [
            float(((cal2_u[b][0] - cal2_st[b]["U_ref_K"]) * K_TO_KJMOL) / cal2_st[b]["N"])
            for b in range(n_boxes) if cal2_st[b]["N"] > 0 and not cal2_u[b][1]
        ]
        de2 = float(max(-12.0, min(2.0, np.mean(diffs2)))) if diffs2 else de1
        prop.set_chem_pot_shift(de2)
        prop.equilibrate(20000)

        if pilot_cycles > 0:
            # Stage 3: brief target-ensemble pilot burn-in to refine chem-pot shift
            _ = run_nested_mc(
                prop, None,
                k_steps=k_steps, eq_cycles=pilot_cycles, prod_cycles=0,
                seed=seed + 5, log_every=pilot_cycles + 10,
                batch_target_fn=target.eval_batch,
            )
            cal3_st = prop.get_accepted()
            cal3_u = target.eval_batch(cal3_st)
            diffs3 = [
                float(((cal3_u[b][0] - cal3_st[b]["U_ref_K"]) * K_TO_KJMOL) / cal3_st[b]["N"])
                for b in range(n_boxes) if cal3_st[b]["N"] > 0 and not cal3_u[b][1]
            ]
            if diffs3:
                de_final = float(max(-12.0, min(4.0, 0.5 * de2 + 0.5 * float(np.mean(diffs3)))))
            else:
                de_final = de2
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

    warm_st = prop.get_accepted()
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
        f"Acc={res['outer_acc_rate']*100:.1f}% | |drift|={drift:.3f} | conv={diag['converged']} | {res['elapsed_s']:.1f}s"
    )
    if not diag["converged"]:
        for msg in diag["warnings"]:
            print(f"  [DIAG WARNING] P={P_bar:.4g} bar: {msg}")
    prop.close()
    return res


def main():
    args = parse_args()
    ff_mode, pol_mode = PROPOSER_MAP[args.proposer]
    use_disp = args.dispersion == "on"
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv) if args.out_csv else None

    print(
        f"=== Hybrid GPU-GCMC + MLIP Engine ===\n"
        f"  CIF={args.cif} supercell={args.supercell} T={args.temp} K Pressures={args.pressures}\n"
        f"  Proposer={args.proposer} (ff={ff_mode}, pol={pol_mode}) | MLIP={args.model} (dispersion={use_disp}, xc={args.xc})"
    )

    payload = {
        "cif": str(args.cif),
        "supercell": list(args.supercell),
        "T_K": float(args.temp),
        "proposer": args.proposer,
        "model": args.model,
        "dispersion": use_disp,
        "dispersion_xc": args.xc,
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

    if pending:
        target = FastMaceD3Target(
            cif_path=args.cif,
            supercell=tuple(args.supercell),
            model=args.model,
            device=args.device,
            dispersion=use_disp,
            dispersion_xc=args.xc,
        )

        consistency_checked = False
        for i, p_bar in pending:
            print(f"\n--- Pressure Point {i+1}/{len(args.pressures)}: {p_bar} bar ---")
            pt = run_pressure_point(
                cif_path=args.cif,
                supercell=args.supercell,
                T_K=args.temp,
                P_bar=p_bar,
                ff_mode=ff_mode,
                pol_mode=pol_mode,
                grid_dims=args.grid_dims,
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
            )
            consistency_checked = True
            payload["points"].append(pt)
            save_outputs(out_json, out_csv, payload)
            print(f"  [Saved checkpoint] {out_json} ({len(payload['points'])} points)")
    else:
        save_outputs(out_json, out_csv, payload)
        print("  [RESUME] All requested pressures already completed.")

    print(f"\nFinal JSON: {out_json}")
    if out_csv is not None:
        print(f"Final CSV:  {out_csv}")


if __name__ == "__main__":
    main()
