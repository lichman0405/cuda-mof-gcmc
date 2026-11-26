"""Gate 1 Unified Hard-Core Verification Suite.

Verifies that Direct MLIP-GCMC (via compiled C++ host predicate) and CUDA auto_mlip
(via GPU/CPU kernels using the identical include/hard_core.hpp geometry/MIC code) agree
100% on continuous coordinates, periodic MIC, strict < squared-distance thresholds,
virtual-site exclusion (Z <= 0), N_max cap, finite bounded near-wall grid energies,
rollback/REMC swap sync, and grand-canonical P(N)/<N>/Var(N) stationary parity.
"""

import math
from pathlib import Path
import numpy as np
import pytest

from direct_mlip_gcmc import (
    DISTILLED_FW_R_HS_HEAVY_A,
    DISTILLED_FW_R_HS_H_A,
    DISTILLED_GG_R_HS_A,
    check_hard_core_state,
    run_direct_gcmc,
)
from guests import GuestSite, GuestSpec, load_guest
from mlip_distill import DISTILL_VERSION, make_distilled_guest_spec
from nested_mc_engine import (
    CudaGcmcProposer,
    K_TO_KJMOL,
    NestedMCChain,
    quat_rotate_batch,
    run_nested_mc,
)
from qeq_charges import extract_framework_for_proposer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _quat_from_axis_angle(axis, angle_rad: float):
    ax = np.asarray(axis, dtype=np.float64)
    ax = ax / np.linalg.norm(ax)
    h = 0.5 * float(angle_rad)
    s = math.sin(h)
    return (math.cos(h), float(ax[0] * s), float(ax[1] * s), float(ax[2] * s))


def _bruteforce_periodic_hard_core(
    uc_atoms,
    supercell,
    guest: GuestSpec,
    state: dict,
    max_mol: int = 128,
    r_fw_h: float = 1.20,
    r_fw_heavy: float = 1.48,
    r_gg: float = 0.50,
) -> bool:
    """Independent brute-force 27-image periodic reference checker in pure Python."""
    n = int(state["N"])
    if n < 0 or n > int(max_mol):
        return True
    if n == 0:
        return False

    H_u = np.asarray(uc_atoms.get_cell(), dtype=np.float64)  # rows A, B, C
    inv_H_u = np.linalg.inv(H_u)
    fw_u_frac = np.mod(np.asarray(uc_atoms.get_scaled_positions(), dtype=np.float64), 1.0)

    sc = tuple(int(x) for x in supercell)
    H_s = H_u * np.array(sc, dtype=np.float64)[:, None]
    inv_H_s = np.linalg.inv(H_s)

    real_sites = [s for s in guest.sites if int(s.Z) > 0]
    if not real_sites:
        return False
    local_real = np.array([s.pos for s in real_sites], dtype=np.float64)
    z_real = np.array([int(s.Z) for s in real_sites], dtype=np.int32)

    cx = np.asarray(state["cx"], dtype=np.float64)
    cy = np.asarray(state["cy"], dtype=np.float64)
    cz = np.asarray(state["cz"], dtype=np.float64)
    qw = np.asarray(state["qw"], dtype=np.float64)
    qx = np.asarray(state["qx"], dtype=np.float64)
    qy = np.asarray(state["qy"], dtype=np.float64)
    qz = np.asarray(state["qz"], dtype=np.float64)

    grid_1d = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    shifts_27 = np.stack(np.meshgrid(grid_1d, grid_1d, grid_1d, indexing="ij"), axis=-1).reshape(-1, 3)

    if guest.is_isotropic:
        offs = np.broadcast_to(local_real[None, :, :], (n, len(real_sites), 3)).copy()
    else:
        offs = quat_rotate_batch(qw, qx, qy, qz, local_real)
    coms = np.stack([cx, cy, cz], axis=1)
    site_pos = coms[:, None, :] + offs  # (N, N_real, 3)

    # 1. Framework-guest check against unit-cell periodicity
    pts_flat = site_pos.reshape(-1, 3)
    pts_uf = pts_flat @ inv_H_u
    df_fw = (pts_uf[:, None, :] - fw_u_frac[None, :, :])
    df_fw -= np.floor(df_fw + 0.5)
    df_fw_all = df_fw[:, :, None, :] + shifts_27[None, None, :, :]
    dr_fw_all = df_fw_all @ H_u
    r2_fw_min = np.min(np.sum(dr_fw_all * dr_fw_all, axis=-1), axis=(1, 2)).reshape(n, len(real_sites))

    cut2_arr = np.where(z_real == 1, r_fw_h * r_fw_h, r_fw_heavy * r_fw_heavy)
    if bool(np.any(r2_fw_min < cut2_arr[None, :])):
        return True

    # 2. Guest-guest real-atom check across distinct molecules i < j
    r_gg2 = r_gg * r_gg
    for i in range(n):
        for j in range(i + 1, n):
            diff = site_pos[j][None, :, :] - site_pos[i][:, None, :]
            df_gg = diff @ inv_H_s
            df_gg -= np.floor(df_gg + 0.5)
            df_gg_all = df_gg[:, :, None, :] + shifts_27[None, None, :, :]
            dr_gg_all = df_gg_all @ H_s
            r2_gg_min = float(np.min(np.sum(dr_gg_all * dr_gg_all, axis=-1)))
            if r2_gg_min < r_gg2:
                return True

    return False


def _make_smooth_distill_stub(guest: GuestSpec, grid_dims=(48, 48, 24), corner_spike: float = 0.0):
    """Build valid bounded 1B grid (optionally with steep near-wall corners to test non-rejection)."""
    nx, ny, nz = [int(d) for d in grid_dims]
    n_chan = max(1, max((s.lj_type for s in guest.sites if s.lj_type >= 0), default=0) + 1)
    grids = np.zeros((n_chan, nz, ny, nx), dtype=np.float32)
    sa = np.linspace(0.0, 2.0 * math.pi, nx, endpoint=False)
    sb = np.linspace(0.0, 2.0 * math.pi, ny, endpoint=False)
    sc = np.linspace(0.0, 2.0 * math.pi, nz, endpoint=False)
    gz, gy, gx = np.meshgrid(sc, sb, sa, indexing="ij")
    base = (-450.0 + 120.0 * np.cos(gx) + 80.0 * np.sin(gy) - 50.0 * np.cos(gz)).astype(np.float32)
    if corner_spike != 0.0:
        base[::2, ::2, ::2] = float(corner_spike)
    for c in range(n_chan):
        grids[c] = base
    return {
        "grids_u_lj": np.ascontiguousarray(grids, dtype=np.float32),
        "gg_com_lut_K": np.zeros(128, dtype=np.float32),
        "gg_q_scale": 1.0,
        "gg_lj_scale": 1.0,
        "gg_lut_rmin_A": 2.0,
        "gg_lut_rmax_A": 10.0,
    }


def _place_site0_near_fw_atom(
    uc_atoms,
    atom_idx: int,
    dx_A: float,
    dy_A: float,
    dz_A: float,
    cell_shift=(0, 0, 0),
    quat=(1.0, 0.0, 0.0, 0.0),
) -> dict:
    """Create a 1-molecule state whose COM is at fw_atom[atom_idx] + shift + (dx, dy, dz)."""
    H_u = np.asarray(uc_atoms.get_cell(), dtype=np.float64)
    fw_pos_u = np.asarray(uc_atoms.get_positions(), dtype=np.float64)
    sh = np.asarray(cell_shift, dtype=np.float64) @ H_u
    c = fw_pos_u[atom_idx] + sh + np.array([dx_A, dy_A, dz_A], dtype=np.float64)
    return {
        "N": 1,
        "cx": np.array([c[0]], dtype=np.float64),
        "cy": np.array([c[1]], dtype=np.float64),
        "cz": np.array([c[2]], dtype=np.float64),
        "qw": np.array([quat[0]], dtype=np.float64),
        "qx": np.array([quat[1]], dtype=np.float64),
        "qy": np.array([quat[2]], dtype=np.float64),
        "qz": np.array([quat[3]], dtype=np.float64),
    }


def test_version_and_unified_constants():
    assert DISTILL_VERSION == "v1.4"
    assert DISTILLED_FW_R_HS_H_A == 1.20
    assert DISTILLED_FW_R_HS_HEAVY_A == 1.48
    assert DISTILLED_GG_R_HS_A == 0.50


@pytest.mark.parametrize(
    "cif_name,supercell,grid_dims",
    [
        ("MgMOF-74.cif", (1, 1, 2), (48, 48, 24)),
        ("IRMOF-1.cif", (1, 1, 1), (48, 48, 48)),
    ],
)
def test_gate1_boundary_cutoffs_virtual_sites_and_nmax(cif_name, supercell, grid_dims):
    """Test exact cutoff boundaries (1.20 H, 1.48 heavy, 0.50 gg), MIC images, virtual sites, and N_max."""
    cif = str(DATA / cif_name)
    fw_data = extract_framework_for_proposer(cif, charge_mode="auto")
    uc = fw_data["uc"]
    H_u = np.asarray(uc.get_cell(), dtype=np.float64)
    sc_H = H_u * np.array(supercell, dtype=np.float64)[:, None]

    # Pick an isolated framework atom (closest neighbor > 1.55 A, wait actually in MgMOF-74 C-H bond is ~1.09 A,
    # so placing a probe along outward radial vector away from bonded neighbors tests exact atom distance cleanly)
    fw_pos_u = np.asarray(uc.get_positions(), dtype=np.float64)
    fw_frac_u = np.mod(np.asarray(uc.get_scaled_positions(), dtype=np.float64), 1.0)
    # Find atom 0's unit-cell neighbor directions and choose a direction maximizing clearance to all other atoms
    idx_ref = 0
    df = fw_frac_u - fw_frac_u[idx_ref][None, :]
    df -= np.floor(df + 0.5)
    dr = df @ H_u
    dists = np.linalg.norm(dr, axis=1)
    # Use centroid pore center to find an open direction from a pore-facing atom
    pore_center = np.mean(fw_pos_u, axis=0)
    best_atom = None
    best_dir = None
    best_clearance = -1.0
    rng_scan = np.random.default_rng(20261126)
    dirs = rng_scan.normal(size=(64, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    for ai in range(len(fw_pos_u)):
        for dvec in dirs[:16]:
            probe = fw_pos_u[ai] + 1.52 * dvec
            pf = (probe @ np.linalg.inv(H_u))
            dfp = pf[None, :] - fw_frac_u
            dfp -= np.floor(dfp + 0.5)
            d_all = np.linalg.norm(dfp @ H_u, axis=1)
            d_other = float(np.min(np.delete(d_all, ai)))
            if d_other > best_clearance:
                best_clearance = d_other
                best_atom = ai
                best_dir = dvec
    assert best_clearance > 1.85, f"Could not find isolated pore-facing atom direction in {cif_name}"

    def _assert_both_agree(guest_obj, state, expected_ovl: bool, desc: str, check_cuda_prop=None):
        py_ov = check_hard_core_state(uc, supercell, guest_obj, state, max_mol=128)
        bf_ov = _bruteforce_periodic_hard_core(uc, supercell, guest_obj, state, max_mol=128)
        assert py_ov == bf_ov == expected_ovl, f"[{cif_name} {desc}] py={py_ov} bf={bf_ov} expected={expected_ovl}"
        if check_cuda_prop is not None:
            u_cu, cu_ov = check_cuda_prop.eval_energy_quat(state)
            assert cu_ov == expected_ovl, f"[{cif_name} {desc}] CUDA eval_energy_quat ov={cu_ov} != {expected_ovl}"
            if not cu_ov:
                assert math.isfinite(float(u_cu)), f"[{cif_name} {desc}] non-finite legal U_ref={u_cu}"

    # --- A. Heavy-atom guest (Ar, Z=18, 1 site at COM) ---
    ar_guest = make_distilled_guest_spec(load_guest("ar"))
    prop_ar = CudaGcmcProposer.from_distilled(
        fw_data=fw_data, guest=ar_guest, distill_res=_make_smooth_distill_stub(ar_guest, grid_dims, corner_spike=1.5e5),
        supercell=supercell, T_K=298.0, pressure_bar=0.28, n_boxes=2, max_mol=16, seed=101,
    )
    try:
        # Empty box
        _assert_both_agree(ar_guest, {"N": 0, "cx": np.zeros(0), "cy": np.zeros(0), "cz": np.zeros(0),
                                      "qw": np.ones(0), "qx": np.zeros(0), "qy": np.zeros(0), "qz": np.zeros(0)},
                           False, "empty", prop_ar)

        # Heavy fw inside/outside across cell shift boundaries
        for eps, exp_ov, label in [
            (1.48 - 2e-4, True, "heavy_fw_inside"),
            (1.48 + 2e-4, False, "heavy_fw_outside_near_wall"),
            (1.55, False, "heavy_fw_near_wall_1.55A"),
        ]:
            for sh in [(0, 0, 0), (1, 0, 0), (0, -1, supercell[2])]:
                st = _place_site0_near_fw_atom(
                    uc, best_atom, eps * best_dir[0], eps * best_dir[1], eps * best_dir[2], cell_shift=sh
                )
                _assert_both_agree(ar_guest, st, exp_ov, f"{label}_sh{sh}", prop_ar)

        # Guest-guest inside/outside (0.50 A) including periodic supercell boundary wrap
        c0 = fw_pos_u[best_atom] + 2.80 * best_dir
        # Make sure c0 itself is well clear of fw
        c0_f = (c0 @ np.linalg.inv(H_u))
        assert float(np.min(np.linalg.norm(((c0_f[None, :] - fw_frac_u) - np.floor((c0_f[None, :] - fw_frac_u) + 0.5)) @ H_u, axis=1))) > 1.80
        for dg, exp_ov, lab in [(0.4998, True, "gg_inside"), (0.5002, False, "gg_outside")]:
            for sc_sh in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 1.0)]:
                c1 = c0 + np.array([dg, 0.0, 0.0]) + (np.asarray(sc_sh) @ sc_H)
                st_pair = {
                    "N": 2,
                    "cx": np.array([c0[0], c1[0]]), "cy": np.array([c0[1], c1[1]]), "cz": np.array([c0[2], c1[2]]),
                    "qw": np.array([1.0, 1.0]), "qx": np.array([0.0, 0.0]),
                    "qy": np.array([0.0, 0.0]), "qz": np.array([0.0, 0.0]),
                }
                _assert_both_agree(ar_guest, st_pair, exp_ov, f"{lab}_sc{sc_sh}", prop_ar)

        # N_max check: 16 legal vs 17 illegal (find open grid points > 2.0 A from fw and > 1.0 A from each other)
        open_pts = []
        pts_cand = rng_scan.random((4000, 3)) @ sc_H
        for p in pts_cand:
            st1 = {"N": 1, "cx": np.array([p[0]]), "cy": np.array([p[1]]), "cz": np.array([p[2]]),
                   "qw": np.array([1.0]), "qx": np.array([0.0]), "qy": np.array([0.0]), "qz": np.array([0.0])}
            if check_hard_core_state(uc, supercell, ar_guest, st1, max_mol=16):
                continue
            if open_pts:
                d_prev = np.min(np.linalg.norm(np.asarray(open_pts) - p[None, :], axis=1))
                if d_prev < 1.2:
                    continue
            open_pts.append(p)
            if len(open_pts) >= 17:
                break
        assert len(open_pts) == 17
        arr_16 = np.asarray(open_pts[:16])
        st_16 = {
            "N": 16,
            "cx": arr_16[:, 0], "cy": arr_16[:, 1], "cz": arr_16[:, 2],
            "qw": np.ones(16), "qx": np.zeros(16), "qy": np.zeros(16), "qz": np.zeros(16),
        }
        assert not check_hard_core_state(uc, supercell, ar_guest, st_16, max_mol=16)
        u16, ov16 = prop_ar.eval_energy_quat(st_16)
        assert not ov16 and math.isfinite(u16)
        prop_ar.set_accepted_quat([st_16, {"N": 0, "cx": np.zeros(0), "cy": np.zeros(0), "cz": np.zeros(0),
                                           "qw": np.ones(0), "qx": np.zeros(0), "qy": np.zeros(0), "qz": np.zeros(0)}])

        arr_17 = np.asarray(open_pts[:17])
        st_17 = {
            "N": 17,
            "cx": arr_17[:, 0], "cy": arr_17[:, 1], "cz": arr_17[:, 2],
            "qw": np.ones(17), "qx": np.zeros(17), "qy": np.zeros(17), "qz": np.zeros(17),
        }
        assert check_hard_core_state(uc, supercell, ar_guest, st_17, max_mol=16)
        _, ov17 = prop_ar.eval_energy_quat(st_17)
        assert ov17 is True
    finally:
        prop_ar.close()

    # --- B. Element-specific H (Z==1, 1.20 A) via CH4 oriented along best_dir ---
    ch4_guest = make_distilled_guest_spec(load_guest("ch4"))
    prop_ch4 = CudaGcmcProposer.from_distilled(
        fw_data=fw_data, guest=ch4_guest, distill_res=_make_smooth_distill_stub(ch4_guest, grid_dims, corner_spike=1.5e5),
        supercell=supercell, T_K=298.0, pressure_bar=0.28, n_boxes=1, max_mol=32, seed=102,
    )
    try:
        # Construct orientation putting H site 1 (local (d, d, d), |h|=1.09 A) pointing opposite best_dir
        h1_loc = np.array(ch4_guest.sites[1].pos, dtype=np.float64)
        h1_unit = h1_loc / np.linalg.norm(h1_loc)
        target_unit = -best_dir
        cross_v = np.cross(h1_unit, target_unit)
        dot_v = float(np.dot(h1_unit, target_unit))
        if np.linalg.norm(cross_v) < 1e-10:
            q_rot = (1.0, 0.0, 0.0, 0.0) if dot_v > 0 else (0.0, 1.0, 0.0, 0.0)
        else:
            q_rot = _quat_from_axis_angle(cross_v, math.acos(max(-1.0, min(1.0, dot_v))))
        # Place H1 at exact distance d_h along best_dir from fw atom (so C is at d_h + 1.09 >= 2.28 A!)
        for d_h, exp_ov, lab in [(1.1998, True, "ch4_H_inside"), (1.2002, False, "ch4_H_outside")]:
            com_offset = (d_h + float(np.linalg.norm(h1_loc))) * best_dir
            st_h = _place_site0_near_fw_atom(uc, best_atom, com_offset[0], com_offset[1], com_offset[2], quat=q_rot)
            _assert_both_agree(ch4_guest, st_h, exp_ov, lab, prop_ch4)
    finally:
        prop_ch4.close()

    # --- C. Virtual-site guest (Z == 0 excluded!): build rigid dumbbell with Z=0 at COM and Z=7 at +/-1.6 A ---
    virt_guest = GuestSpec(
        name="virt_probe",
        formula="X2",
        is_isotropic=False,
        Tc_K=126.0,
        Pc_bar=34.0,
        omega=0.04,
        sites=[
            GuestSite("X", 0, (0.0, 0.0, 0.0), +0.8, 0.0, 0.0, -1),
            GuestSite("N", 7, (+1.60, 0.0, 0.0), -0.4, 36.0, 3.31, 0),
            GuestSite("N", 7, (-1.60, 0.0, 0.0), -0.4, 36.0, 3.31, 0),
        ],
    )
    prop_v = CudaGcmcProposer.from_distilled(
        fw_data=fw_data, guest=virt_guest, distill_res=_make_smooth_distill_stub(virt_guest, grid_dims),
        supercell=supercell, T_K=298.0, pressure_bar=0.28, n_boxes=1, max_mol=16, seed=103,
    )
    try:
        # Place virtual COM X at 0.25 A from best_atom along an axis keeping +/-1.60 N atoms > 1.60 A away
        ortho = np.cross(best_dir, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(ortho) < 1e-6:
            ortho = np.cross(best_dir, np.array([0.0, 1.0, 0.0]))
        ortho /= np.linalg.norm(ortho)
        # Rotate molecular x-axis (1,0,0) onto ortho
        x_ax = np.array([1.0, 0.0, 0.0])
        cv = np.cross(x_ax, ortho)
        dv = float(np.dot(x_ax, ortho))
        q_o = (1.0, 0.0, 0.0, 0.0) if np.linalg.norm(cv) < 1e-10 else _quat_from_axis_angle(cv, math.acos(max(-1.0, min(1.0, dv))))
        st_virt_fw = _place_site0_near_fw_atom(uc, best_atom, 0.25 * best_dir[0], 0.25 * best_dir[1], 0.25 * best_dir[2], quat=q_o)
        # Check whether N atoms at +/-1.6 A accidentally hit another fw atom; if clear in brute-force, both MUST say False (virtual site ignored!)
        if not _bruteforce_periodic_hard_core(uc, supercell, virt_guest, st_virt_fw, max_mol=16):
            _assert_both_agree(virt_guest, st_virt_fw, False, "virtual_site_near_fw", prop_v)

        # Place two molecules so their virtual X sites are 0.14 A apart, while all real N atoms are > 0.50 A apart
        c_open = c0
        # Molecule A along (1,0,0), Molecule B rotated 90 deg around z: N sites separated by ~2.26 A, X sites ~0.14 A!
        st_virt_gg = {
            "N": 2,
            "cx": np.array([c_open[0], c_open[0] + 0.10]),
            "cy": np.array([c_open[1], c_open[1] + 0.10]),
            "cz": np.array([c_open[2], c_open[2]]),
            "qw": np.array([1.0, math.cos(math.pi / 4.0)]),
            "qx": np.array([0.0, 0.0]),
            "qy": np.array([0.0, 0.0]),
            "qz": np.array([0.0, math.sin(math.pi / 4.0)]),
        }
        if not _bruteforce_periodic_hard_core(uc, supercell, virt_guest, st_virt_gg, max_mol=16):
            _assert_both_agree(virt_guest, st_virt_gg, False, "virtual_site_gg_0.15A", prop_v)
    finally:
        prop_v.close()


@pytest.mark.parametrize(
    "cif_name,supercell,guest_name,grid_dims",
    [
        ("MgMOF-74.cif", (1, 1, 2), "co2", (48, 48, 24)),
        ("IRMOF-1.cif", (1, 1, 1), "ch4", (48, 48, 48)),
    ],
)
def test_gate1_random_batch_exact_parity_2000(cif_name, supercell, guest_name, grid_dims):
    """Fixed-seed 2,000 random single/multi-guest poses: Direct == CUDA == Brute-force (0 mismatches)."""
    cif = str(DATA / cif_name)
    fw_data = extract_framework_for_proposer(cif, charge_mode="auto")
    uc = fw_data["uc"]
    guest = make_distilled_guest_spec(load_guest(guest_name))
    H_u = np.asarray(uc.get_cell(), dtype=np.float64)
    sc_H = H_u * np.array(supercell, dtype=np.float64)[:, None]

    # Inject corner spikes > 1e5 on grid to ensure old corner rejection is dead on auto_mlip!
    distill_res = _make_smooth_distill_stub(guest, grid_dims=grid_dims, corner_spike=1.2e5)
    prop = CudaGcmcProposer.from_distilled(
        fw_data=fw_data, guest=guest, distill_res=distill_res,
        supercell=supercell, T_K=298.0, pressure_bar=0.28, n_boxes=1, max_mol=32, seed=20260001,
    )
    rng = np.random.default_rng(2026112600 if guest_name == "co2" else 2026112601)
    n_cases = 2000
    n_mismatches = 0
    n_legal = 0
    n_illegal = 0
    try:
        for k in range(n_cases):
            n = 1 if (k % 3 != 0) else int(rng.integers(2, 6))
            pos = rng.random((n, 3)) @ sc_H
            # Shift some poses across periodic cell walls
            if k % 5 == 0:
                pos += rng.integers(-1, 2, size=(n, 3)) @ sc_H
            u1, u2, u3 = rng.random(n), 2.0 * math.pi * rng.random(n), 2.0 * math.pi * rng.random(n)
            s1, s2 = np.sqrt(1.0 - u1), np.sqrt(u1)
            st = {
                "N": n,
                "cx": np.ascontiguousarray(pos[:, 0], dtype=np.float64),
                "cy": np.ascontiguousarray(pos[:, 1], dtype=np.float64),
                "cz": np.ascontiguousarray(pos[:, 2], dtype=np.float64),
                "qw": np.ascontiguousarray(s2 * np.cos(u3), dtype=np.float64),
                "qx": np.ascontiguousarray(s1 * np.sin(u2), dtype=np.float64),
                "qy": np.ascontiguousarray(s1 * np.cos(u2), dtype=np.float64),
                "qz": np.ascontiguousarray(s2 * np.sin(u3), dtype=np.float64),
            }
            ov_dir = check_hard_core_state(uc, supercell, guest, st, max_mol=32)
            u_cu, ov_cu = prop.eval_energy_quat(st)
            ov_bf = _bruteforce_periodic_hard_core(uc, supercell, guest, st, max_mol=32)
            if not (ov_dir == ov_cu == ov_bf):
                n_mismatches += 1
            if ov_cu:
                n_illegal += 1
            else:
                n_legal += 1
                assert math.isfinite(float(u_cu)), f"Legal pose {k} gave non-finite U_ref={u_cu}"
    finally:
        prop.close()

    assert n_mismatches == 0, f"Detected {n_mismatches}/{n_cases} hard-core parity mismatches on {cif_name}/{guest_name}"
    assert n_legal > 50 and n_illegal > 50, f"Poor test coverage: legal={n_legal}, illegal={n_illegal}"


def test_gate1_cbmc_burst_rollback_swap_validation():
    """Verify 1B-only and 1B+2B bursts never produce illegal states, rollback is bit-exact, and swap rejects illegal states."""
    cif = str(DATA / "MgMOF-74.cif")
    fw_data = extract_framework_for_proposer(cif, charge_mode="auto")
    uc = fw_data["uc"]
    guest = make_distilled_guest_spec(load_guest("co2"))

    for use_2b in (False, True):
        dist = _make_smooth_distill_stub(guest, (48, 48, 20))
        if not use_2b:
            dist["gg_com_lut_K"] = np.zeros(128, dtype=np.float32)
            dist["gg_q_scale"] = 1.0
            dist["gg_lj_scale"] = 1.0
        prop = CudaGcmcProposer.from_distilled(
            fw_data=fw_data, guest=guest, distill_res=dist,
            supercell=(1, 1, 2), T_K=298.0, pressure_bar=0.5,
            n_boxes=4, max_mol=32, seed=20265501 + int(use_2b), cbmc_k=128, p_ins_del=0.50,
        )
        try:
            prop.set_chem_pot_shift(-18.0)  # drive strong loading to exercise dense pairs & bursts
            for step in range(40):
                trials = prop.propose_k_quat(24)
                for b, tr in enumerate(trials):
                    assert not check_hard_core_state(uc, (1, 1, 2), guest, tr, max_mol=32)
                    u_re, ov_re = prop.eval_energy_quat(tr)
                    assert not ov_re and math.isfinite(u_re)
                    assert abs(u_re - tr["U_ref_K"]) < 1e-8
                if step % 3 == 0:
                    before = prop.get_accepted_quat()
                    prop.commit([0, 0, 0, 0])
                    after = prop.get_accepted_quat()
                    for b in range(4):
                        assert after[b]["N"] == before[b]["N"]
                        assert after[b]["U_ref_K"] == before[b]["U_ref_K"]
                        assert np.array_equal(after[b]["cx"], before[b]["cx"])
                else:
                    prop.commit([1, 0, 1, 1])

            # Reject illegal state in set_accepted_quat without corrupting GPU accepted state
            acc_before = prop.get_accepted_quat()
            fw0 = np.asarray(uc.get_positions(), dtype=np.float64)[0]
            bad_state = {
                "N": 1,
                "cx": np.array([fw0[0]]), "cy": np.array([fw0[1]]), "cz": np.array([fw0[2]]),
                "qw": np.array([1.0]), "qx": np.array([0.0]), "qy": np.array([0.0]), "qz": np.array([0.0]),
            }
            with pytest.raises(ValueError, match="illegal hard-core"):
                prop.set_accepted_quat([bad_state, acc_before[1], acc_before[2], acc_before[3]])
            acc_after = prop.get_accepted_quat()
            assert acc_after[0]["N"] == acc_before[0]["N"]

            # Test REMC NestedMCChain.apply_swapped_boxes rejects illegal state and syncs legal swaps
            chain = NestedMCChain(
                proposer=prop, target_fn=lambda s: prop.eval_energy_quat(s),
                k_steps=8, eq_cycles=4, prod_cycles=4, seed=20265510,
                batch_target_fn=lambda sts: [prop.eval_energy_quat(s) for s in sts],
            )
            acc_now = prop.get_accepted_quat()
            chain.apply_swapped_boxes({0: (acc_now[1], acc_now[1]["N"], acc_now[1]["U_ref_K"])})
            assert int(chain.N_curr[0]) == int(acc_now[1]["N"])
            assert abs(float(chain.U_ref_curr[0]) - float(acc_now[1]["U_ref_K"])) < 1e-8
            with pytest.raises((ValueError, RuntimeError)):
                chain.apply_swapped_boxes({0: (bad_state, 1, 0.0)})
        finally:
            prop.close()


@pytest.mark.slow
def test_gate1_grand_canonical_pn_mean_variance_parity():
    """Verify Direct GCMC and CUDA auto_mlip Nested MC produce indistinguishable P(N), <N>, and Var(N) under unified HC."""
    from ase import Atoms

    # Compact cubic nanopore (10x10x10 A with single Mg2+ wall atom at origin) enabling rapid full-support GCMC sampling
    uc = Atoms("Mg", positions=[[0.0, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=True)
    fw_data = {
        "uc": uc,
        "H_unit_cols": np.ascontiguousarray(np.asarray(uc.get_cell(), dtype=np.float64).T),
        "frac": np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        "Z": np.array([12], dtype=np.int32),
        "charges": np.array([0.0], dtype=np.float64),
        "type_ids": np.array([0], dtype=np.int32),
        "charge_source": "zero",
    }
    # Guest with mild steric LJ (eps=1.0 K, sig=1.2 A) so both target and surrogate have full non-zero
    # support across the entire continuous hard-core domain [0.50 A, inf), while still exercising nonzero
    # 1B + 2B residual reweighting between U_ref and U_tgt!
    guest = GuestSpec(
        name="ar_soft",
        formula="Ar",
        is_isotropic=True,
        Tc_K=150.687,
        Pc_bar=48.63,
        omega=-0.00219,
        sites=[GuestSite("Ar", 18, (0.0, 0.0, 0.0), 0.0, 1.0, 1.2, 0)],
    )

    # Exact fast target Hamiltonian evaluated identically for both Direct & Nested MC (with non-zero 1B/target discrepancy!)
    # Surrogate grid gives mild constant pore field (-350 K); target has attractive pore pocket (-750 K) + soft Ar pair!
    distill_res = _make_smooth_distill_stub(guest, grid_dims=(32, 32, 32))
    distill_res["grids_u_lj"][:] = -350.0  # pure reference offset (-350 K vs target ~ -750 K)

    def target_energy(state: dict):
        n = int(state["N"])
        if n > 16:
            return 1.0e9, True
        if n == 0:
            return 0.0, False
        if check_hard_core_state(uc, (1, 1, 1), guest, state, max_mol=16):
            return 1.0e9, True
        pos = np.stack([state["cx"], state["cy"], state["cz"]], axis=1)
        # 1-body cosine pore well
        u1 = -750.0 * n + 150.0 * np.sum(np.cos(2.0 * math.pi * pos / 10.0))
        # 2-body soft pair repulsion
        u2 = 0.0
        if n >= 2:
            df = (pos[:, None, :] - pos[None, :, :]) / 10.0
            df -= np.floor(df + 0.5)
            dr = df * 10.0
            r2 = np.sum(dr * dr, axis=-1)
            iu, ju = np.triu_indices(n, k=1)
            pair_r = np.sqrt(r2[iu, ju])
            u2 = float(np.sum(420.0 * np.exp(-pair_r / 1.4)))
        return float(u1 + u2), False

    T_K, P_bar = 298.0, 12.0
    dir_res = run_direct_gcmc(
        uc_atoms=uc, guest=guest, supercell=(1, 1, 1), T_K=T_K, pressure_bar=P_bar,
        batch_target_fn=lambda sts: [target_energy(s) for s in sts],
        n_boxes=8, eq_steps=2000, prod_steps=20000, max_mol=16,
        match_distilled_hard_core=True, seed=20269901,
    )
    assert dir_res["used_cpp_hard_core"] is True

    prop = CudaGcmcProposer.from_distilled(
        fw_data=fw_data, guest=guest, distill_res=distill_res,
        supercell=(1, 1, 1), T_K=T_K, pressure_bar=P_bar,
        n_boxes=8, max_mol=16, seed=20269902, cbmc_k=64, p_ins_del=0.60,
    )
    try:
        prop.set_chem_pot_shift(-400.0 * K_TO_KJMOL)
        nest_res = run_nested_mc(
            prop, None, k_steps=6, eq_cycles=400, prod_cycles=5000, seed=20269903,
            batch_target_fn=lambda sts: [target_energy(s) for s in sts],
        )
    finally:
        prop.close()

    dn = np.asarray(dir_res["prod_N_history"], dtype=np.float64)
    nn = np.asarray(nest_res["prod_N_history"], dtype=np.float64)
    m_dir, m_nest = float(np.mean(dn)), float(np.mean(nn))
    se_dir = float(np.std(np.mean(dn, axis=0), ddof=1) / math.sqrt(dn.shape[1]))
    se_nest = float(np.std(np.mean(nn, axis=0), ddof=1) / math.sqrt(nn.shape[1]))
    se_m = math.hypot(se_dir, se_nest)

    v_dir = float(np.var(dn, ddof=1))
    v_nest = float(np.var(nn, ddof=1))
    bm_v_dir = [float(np.var(dn[:, b], ddof=1)) for b in range(dn.shape[1])]
    bm_v_nest = [float(np.var(nn[:, b], ddof=1)) for b in range(nn.shape[1])]
    se_v = math.hypot(np.std(bm_v_dir, ddof=1) / math.sqrt(len(bm_v_dir)), np.std(bm_v_nest, ddof=1) / math.sqrt(len(bm_v_nest)))

    p_dir = np.bincount(dn.astype(int).ravel(), minlength=17) / float(dn.size)
    p_nest = np.bincount(nn.astype(int).ravel(), minlength=17) / float(nn.size)
    l1_diff = float(np.sum(np.abs(p_dir - p_nest)))

    assert 1.5 <= m_dir <= 10.5
    assert abs(m_dir - m_nest) < max(0.08, 3.0 * se_m), f"Mean N mismatch: dir={m_dir:.4f}+-{se_dir:.4f} vs nest={m_nest:.4f}+-{se_nest:.4f}"
    assert abs(v_dir - v_nest) < max(0.10, 3.5 * se_v), f"Var N mismatch: dir={v_dir:.4f} vs nest={v_nest:.4f} (se_v={se_v:.4f})"
    assert l1_diff < 0.04, f"P(N) L1 distance too large: {l1_diff:.4f}, p_dir={np.round(p_dir, 4)}, p_nest={np.round(p_nest, 4)}"
    for k in range(17):
        if p_dir[k] > 0.02 or p_nest[k] > 0.02:
            se_k = math.hypot(
                float(np.std(np.mean(dn == k, axis=0), ddof=1) / math.sqrt(dn.shape[1])),
                float(np.std(np.mean(nn == k, axis=0), ddof=1) / math.sqrt(nn.shape[1])),
            )
            assert abs(p_dir[k] - p_nest[k]) < max(0.01, 3.2 * se_k), (
                f"P(N={k}) mismatch: dir={p_dir[k]:.4f} vs nest={p_nest[k]:.4f} (se={se_k:.4f})"
            )
