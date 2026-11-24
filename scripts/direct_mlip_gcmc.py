"""Exact Grand-Canonical Direct MLIP-GCMC baselines (serial ASE & hard-sphere batched GPU)."""

import math
import time
import numpy as np

from nested_mc_engine import (
    EV_TO_K,
    K_TO_KJMOL,
    compute_ess_stats,
    compute_pooled_jackknife_qst,
    quat_rotate_batch,
)

KB_SI = 1.380649e-23
R_BAR_L = 0.08314462618


def compute_pure_fugacity_pr_py(
    T_K: float, P_bar: float, Tc_K: float, Pc_bar: float, omega: float
) -> float:
    """Exact Peng-Robinson cubic EOS pure-component fugacity in Pa (matches forcefield.hpp)."""
    if Tc_K <= 0.0 or Pc_bar <= 0.0 or P_bar <= 0.0:
        return float(P_bar * 1.0e5)

    Tr = T_K / Tc_K
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega * omega
    sqrt_alpha = 1.0 + kappa * (1.0 - math.sqrt(Tr))
    alpha = sqrt_alpha * sqrt_alpha

    a = 0.4572355289213822 * R_BAR_L * R_BAR_L * Tc_K * Tc_K / Pc_bar * alpha
    b = 0.07779607390388846 * R_BAR_L * Tc_K / Pc_bar

    A = a * P_bar / (R_BAR_L * T_K * R_BAR_L * T_K)
    B = b * P_bar / (R_BAR_L * T_K)

    c2 = -(1.0 - B)
    c1 = A - 3.0 * B * B - 2.0 * B
    c0 = -(A * B - B * B - B * B * B)

    p = c1 - c2 * c2 / 3.0
    q = 2.0 * c2 * c2 * c2 / 27.0 - c2 * c1 / 3.0 + c0
    disc = 0.25 * q * q + p * p * p / 27.0
    sqrt2 = math.sqrt(2.0)

    def _eval_phi(Z: float) -> float:
        arg1 = Z - B
        denom = Z + (1.0 - sqrt2) * B
        if denom == 0.0:
            return 1e30
        term2 = (Z + (1.0 + sqrt2) * B) / denom
        if arg1 <= 0.0 or term2 <= 0.0:
            return 1e30
        ln_phi = (Z - 1.0) - math.log(arg1) - (A / (2.0 * sqrt2 * B)) * math.log(term2)
        return math.exp(ln_phi)

    best_phi = 1.0
    if disc >= 0.0:
        sqrt_d = math.sqrt(disc)
        u = math.copysign(abs(-0.5 * q + sqrt_d) ** (1.0 / 3.0), -0.5 * q + sqrt_d)
        v = math.copysign(abs(-0.5 * q - sqrt_d) ** (1.0 / 3.0), -0.5 * q - sqrt_d)
        best_Z = u + v - c2 / 3.0
        best_phi = _eval_phi(best_Z)
    else:
        r = math.sqrt(-p * p * p / 27.0)
        theta = math.acos(max(-1.0, min(1.0, -0.5 * q / r)))
        m = 2.0 * math.sqrt(-p / 3.0)
        best_phi = 1e30
        for k in range(3):
            zk = m * math.cos((theta + 2.0 * math.pi * k) / 3.0) - c2 / 3.0
            if zk > B:
                phik = _eval_phi(zk)
                if phik < best_phi:
                    best_phi = phik

    return float(best_phi * P_bar * 1.0e5)


def rand_uniform_quat(rng: np.random.Generator):
    """Shoemake (1992) uniform unit quaternion on SO(3)."""
    u1, u2, u3 = rng.random(3)
    u2 *= 2.0 * math.pi
    u3 *= 2.0 * math.pi
    s1 = math.sqrt(1.0 - u1)
    s2 = math.sqrt(u1)
    return (
        s2 * math.cos(u3),
        s1 * math.sin(u2),
        s1 * math.cos(u2),
        s2 * math.sin(u3),
    )


def rot_quat_small(rng: np.random.Generator, max_angle: float, qw: float, qx: float, qy: float, qz: float):
    """Symmetric isotropic small-angle quaternion perturbation."""
    z = 2.0 * rng.random() - 1.0
    phi = 2.0 * math.pi * rng.random()
    r_xy = math.sqrt(max(0.0, 1.0 - z * z))
    kx = r_xy * math.cos(phi)
    ky = r_xy * math.sin(phi)
    kz = z

    half = 0.5 * (2.0 * rng.random() - 1.0) * max_angle
    sh = math.sin(half)
    ch = math.cos(half)
    dqw, dqx, dqy, dqz = ch, sh * kx, sh * ky, sh * kz

    nqw = dqw * qw - dqx * qx - dqy * qy - dqz * qz
    nqx = dqw * qx + dqx * qw + dqy * qz - dqz * qy
    nqy = dqw * qy - dqx * qz + dqy * qw + dqz * qx
    nqz = dqw * qz + dqx * qy - dqy * qx + dqz * qw
    inv = 1.0 / math.sqrt(nqw * nqw + nqx * nqx + nqy * nqy + nqz * nqz)
    return nqw * inv, nqx * inv, nqy * inv, nqz * inv


def has_hard_sphere_overlap(
    trial_guest_pos: np.ndarray,
    fw_pos: np.ndarray,
    other_guest_pos: np.ndarray,
    cell_rows: np.ndarray,
    inv_cell_rows: np.ndarray,
    r_hs_A: float = 1.20,
    shifts_27: np.ndarray = None,
) -> bool:
    """Return True if any trial real atom approaches a framework or other guest atom within r_hs_A."""
    if r_hs_A <= 0.0 or len(trial_guest_pos) == 0:
        return False
    ref = fw_pos if len(other_guest_pos) == 0 else np.concatenate([fw_pos, other_guest_pos], axis=0)
    r2_cut = r_hs_A * r_hs_A

    # dr_raw shape: (n_trial_sites, n_ref, 3)
    dr_raw = trial_guest_pos[:, None, :] - ref[None, :, :]
    df = dr_raw @ inv_cell_rows
    df -= np.round(df)
    if shifts_27 is None:
        dr = df @ cell_rows
        r2 = np.sum(dr * dr, axis=-1)
        return bool(np.any(r2 < r2_cut))

    # Full 27-image minimum image check for skewed/small cells
    df_all = df[:, :, None, :] + shifts_27[None, None, :, :]
    dr_all = df_all @ cell_rows
    r2_all = np.sum(dr_all * dr_all, axis=-1)
    return bool(np.any(r2_all < r2_cut))


class SerialAseMaceD3Target:
    """Standard ASE-based serial single-point MLIP target evaluator (analogous to ASE GCMC / MLIP-MC)."""

    def __init__(
        self,
        uc_atoms,
        supercell=(1, 1, 2),
        model="medium",
        device="cuda",
        dispersion: bool = True,
        dispersion_xc: str = "pbe",
        guest=None,
    ):
        from ase import Atoms
        from mace.calculators import mace_mp
        from guests import load_guest

        self.Atoms = Atoms
        self.uc = uc_atoms.copy()
        self.fw = self.uc.repeat(tuple(supercell))
        self.guest = load_guest("co2") if guest is None else guest
        self.guest_local_pos = self.guest.real_local_positions()
        self.guest_symbols_unit = [s.symbol for s in self.guest.real_sites()]
        self.guest_n_real = len(self.guest_symbols_unit)

        self.calc = mace_mp(
            model=model,
            dispersion=bool(dispersion),
            dispersion_xc=str(dispersion_xc),
            default_dtype="float32",
            device=device,
        )
        fw_copy = self.fw.copy()
        fw_copy.calc = self.calc
        self.e_fw_eV = float(fw_copy.get_potential_energy())

        g_iso = self.guest.to_ase_atoms(box_A=30.0)
        g_iso.calc = self.calc
        self.e_guest_eV = float(g_iso.get_potential_energy())
        self.n_evals = 0

    def __call__(self, state):
        n = int(state["N"])
        if n == 0:
            return 0.0, False
        c = np.stack([state["cx"], state["cy"], state["cz"]], axis=1)
        offsets = quat_rotate_batch(
            state["qw"], state["qx"], state["qy"], state["qz"], self.guest_local_pos
        )
        gp = (c[:, None, :] + offsets).reshape(n * self.guest_n_real, 3)
        sys_at = self.fw.copy() + self.Atoms(self.guest_symbols_unit * n, positions=gp)
        sys_at.calc = self.calc
        e_tot = float(sys_at.get_potential_energy())
        self.n_evals += 1
        u_int_K = (e_tot - self.e_fw_eV - n * self.e_guest_eV) * EV_TO_K
        return float(u_int_K), False


def _copy_box_state(st: dict) -> dict:
    n = int(st["N"])
    if n == 0:
        return {
            "N": 0,
            "cx": np.zeros((0,), dtype=np.float64),
            "cy": np.zeros((0,), dtype=np.float64),
            "cz": np.zeros((0,), dtype=np.float64),
            "qw": np.zeros((0,), dtype=np.float64),
            "qx": np.zeros((0,), dtype=np.float64),
            "qy": np.zeros((0,), dtype=np.float64),
            "qz": np.zeros((0,), dtype=np.float64),
        }
    if "qw" in st:
        qw = np.asarray(st["qw"], dtype=np.float64).copy()
        qx = np.asarray(st["qx"], dtype=np.float64).copy()
        qy = np.asarray(st["qy"], dtype=np.float64).copy()
        qz = np.asarray(st["qz"], dtype=np.float64).copy()
    else:
        qw = np.ones((n,), dtype=np.float64)
        qx = np.zeros((n,), dtype=np.float64)
        qy = np.zeros((n,), dtype=np.float64)
        qz = np.zeros((n,), dtype=np.float64)
    return {
        "N": n,
        "cx": np.asarray(st["cx"], dtype=np.float64).copy(),
        "cy": np.asarray(st["cy"], dtype=np.float64).copy(),
        "cz": np.asarray(st["cz"], dtype=np.float64).copy(),
        "qw": qw,
        "qx": qx,
        "qy": qy,
        "qz": qz,
    }


def _all_guest_positions(state: dict, local_pos: np.ndarray) -> np.ndarray:
    n = int(state["N"])
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    c = np.stack([state["cx"], state["cy"], state["cz"]], axis=1)
    offs = quat_rotate_batch(state["qw"], state["qx"], state["qy"], state["qz"], local_pos)
    return (c[:, None, :] + offs).reshape(n * local_pos.shape[0], 3)


def run_direct_gcmc(
    uc_atoms,
    guest,
    supercell=(1, 1, 2),
    T_K: float = 298.0,
    pressure_bar: float = 0.15,
    target_fn=None,
    batch_target_fn=None,
    n_boxes: int = 1,
    eq_steps: int = 200,
    prod_steps: int = 1000,
    initial_states=None,
    r_hs_A: float = 1.20,
    delta_trans: float = 1.0,
    max_angle: float = 0.55,
    max_mol: int = 128,
    seed: int = 20261500,
    max_wall_s: float = None,
    log_every: int = 0,
    fw_mass_gmol: float = None,
):
    """Exact Grand-Canonical Direct MLIP-GCMC matching C++ RigidGuestDef move probabilities and PR fugacity."""
    if target_fn is None and batch_target_fn is None:
        raise ValueError("Either target_fn or batch_target_fn must be provided")

    rng = np.random.default_rng(seed)
    fw = uc_atoms.repeat(tuple(int(x) for x in supercell))
    fw_pos = np.asarray(fw.get_positions(), dtype=np.float64)
    cell_rows = np.asarray(fw.get_cell(), dtype=np.float64)  # row vectors A, B, C
    inv_cell_rows = np.linalg.inv(cell_rows)
    vol_super = abs(float(np.linalg.det(cell_rows)))
    mass_gmol = float(fw_mass_gmol) if fw_mass_gmol is not None else float(np.sum(fw.get_masses()))
    conv_mmol_g = 1000.0 / mass_gmol

    perp_widths = 1.0 / np.linalg.norm(inv_cell_rows, axis=0)
    min_perp = float(np.min(perp_widths))
    if r_hs_A > 0.0 and min_perp < 4.0 * r_hs_A:
        grid_1d = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        shifts_27 = np.stack(np.meshgrid(grid_1d, grid_1d, grid_1d, indexing="ij"), axis=-1).reshape(-1, 3)
    else:
        shifts_27 = None

    f_Pa = compute_pure_fugacity_pr_py(
        float(T_K), float(pressure_bar), float(guest.Tc_K), float(guest.Pc_bar), float(guest.omega)
    )
    beta_f_V = (f_Pa * vol_super * 1.0e-30) / (KB_SI * float(T_K))
    local_pos = guest.real_local_positions()
    n_real = local_pos.shape[0]

    n_b = int(n_boxes)
    if initial_states is not None:
        if len(initial_states) != n_b:
            raise ValueError(f"Expected {n_b} initial_states, got {len(initial_states)}")
        boxes = [_copy_box_state(st) for st in initial_states]
    else:
        boxes = [_copy_box_state({"N": 0}) for _ in range(n_b)]

    if batch_target_fn is not None:
        init_evals = batch_target_fn(boxes)
    else:
        init_evals = [target_fn(boxes[b]) for b in range(n_b)]
    U_curr = np.zeros(n_b, dtype=np.float64)
    for b in range(n_b):
        ut, ov = init_evals[b]
        if ov:
            raise RuntimeError(f"Initial state of box {b} has target overlap")
        U_curr[b] = float(ut)

    total_steps = int(eq_steps) + int(prod_steps)
    hs_rejected = 0
    mlip_evals = 0
    acc_total = np.zeros(n_b, dtype=np.int64)
    prod_acc_total = np.zeros(n_b, dtype=np.int64)
    prod_ins_att = prod_ins_acc = 0
    prod_del_att = prod_del_acc = 0
    prod_trans_att = prod_trans_acc = 0
    prod_rot_att = prod_rot_acc = 0

    sum_N = np.zeros(n_b, dtype=np.float64)
    sum_N2 = np.zeros(n_b, dtype=np.float64)
    sum_U = np.zeros(n_b, dtype=np.float64)
    sum_UN = np.zeros(n_b, dtype=np.float64)
    box_N_history = []
    prod_N_history = []
    prod_U_history = []

    t0 = time.perf_counter()
    steps_completed = 0
    prod_steps_done = 0

    for step in range(total_steps):
        if max_wall_s is not None and step >= int(eq_steps) and prod_steps_done >= 20:
            if (time.perf_counter() - t0) >= float(max_wall_s):
                break

        in_prod = step >= int(eq_steps)
        proposals = [None] * n_b
        eval_indices = []
        eval_states = []

        for b in range(n_b):
            st = boxes[b]
            cur_N = int(st["N"])
            if guest.is_isotropic:
                r3 = int(rng.integers(0, 3))
                mv = 0 if r3 == 0 else (2 if r3 == 1 else 3)
            else:
                mv = int(rng.integers(0, 4))

            if mv in (0, 1):
                if cur_N == 0:
                    proposals[b] = ("skip", mv, -1, None)
                    continue
                idx = int(rng.integers(0, cur_N))
                tri = _copy_box_state(st)
                if mv == 0:
                    c_new = np.array(
                        [tri["cx"][idx], tri["cy"][idx], tri["cz"][idx]], dtype=np.float64
                    ) + rng.uniform(-delta_trans, delta_trans, size=3)
                    f_new = (c_new @ inv_cell_rows) % 1.0
                    c_new = f_new @ cell_rows
                    tri["cx"][idx], tri["cy"][idx], tri["cz"][idx] = c_new[0], c_new[1], c_new[2]
                else:
                    nqw, nqx, nqy, nqz = rot_quat_small(
                        rng, max_angle, tri["qw"][idx], tri["qx"][idx], tri["qy"][idx], tri["qz"][idx]
                    )
                    tri["qw"][idx], tri["qx"][idx], tri["qy"][idx], tri["qz"][idx] = nqw, nqx, nqy, nqz

                trial_offs = quat_rotate_batch(
                    np.array([tri["qw"][idx]]), np.array([tri["qx"][idx]]),
                    np.array([tri["qy"][idx]]), np.array([tri["qz"][idx]]), local_pos,
                )[0]
                trial_pos = np.array([tri["cx"][idx], tri["cy"][idx], tri["cz"][idx]])[None, :] + trial_offs
                if cur_N > 1:
                    mask_other = np.ones(cur_N, dtype=bool)
                    mask_other[idx] = False
                    other_st = {
                        "N": cur_N - 1,
                        "cx": tri["cx"][mask_other], "cy": tri["cy"][mask_other], "cz": tri["cz"][mask_other],
                        "qw": tri["qw"][mask_other], "qx": tri["qx"][mask_other],
                        "qy": tri["qy"][mask_other], "qz": tri["qz"][mask_other],
                    }
                    other_pos = _all_guest_positions(other_st, local_pos)
                else:
                    other_pos = np.zeros((0, 3), dtype=np.float64)

                if has_hard_sphere_overlap(trial_pos, fw_pos, other_pos, cell_rows, inv_cell_rows, r_hs_A, shifts_27):
                    hs_rejected += 1
                    proposals[b] = ("hs_ovl", mv, idx, None)
                    continue

                proposals[b] = ("eval", mv, idx, tri)
                eval_indices.append(b)
                eval_states.append(tri)

            elif mv == 2:
                if cur_N >= int(max_mol):
                    proposals[b] = ("skip", mv, -1, None)
                    continue
                c_new = rng.random(3) @ cell_rows
                if guest.is_isotropic:
                    nqw, nqx, nqy, nqz = 1.0, 0.0, 0.0, 0.0
                else:
                    nqw, nqx, nqy, nqz = rand_uniform_quat(rng)
                trial_offs = quat_rotate_batch(
                    np.array([nqw]), np.array([nqx]), np.array([nqy]), np.array([nqz]), local_pos
                )[0]
                trial_pos = c_new[None, :] + trial_offs
                other_pos = _all_guest_positions(st, local_pos)
                if has_hard_sphere_overlap(trial_pos, fw_pos, other_pos, cell_rows, inv_cell_rows, r_hs_A, shifts_27):
                    hs_rejected += 1
                    proposals[b] = ("hs_ovl", mv, -1, None)
                    continue

                tri = {
                    "N": cur_N + 1,
                    "cx": np.append(st["cx"], c_new[0]),
                    "cy": np.append(st["cy"], c_new[1]),
                    "cz": np.append(st["cz"], c_new[2]),
                    "qw": np.append(st["qw"], nqw),
                    "qx": np.append(st["qx"], nqx),
                    "qy": np.append(st["qy"], nqy),
                    "qz": np.append(st["qz"], nqz),
                }
                proposals[b] = ("eval", mv, -1, tri)
                eval_indices.append(b)
                eval_states.append(tri)

            else:  # mv == 3 delete
                if cur_N == 0:
                    proposals[b] = ("skip", mv, -1, None)
                    continue
                idx = int(rng.integers(0, cur_N))
                tri = _copy_box_state(st)
                last = cur_N - 1
                if idx != last:
                    for key in ("cx", "cy", "cz", "qw", "qx", "qy", "qz"):
                        tri[key][idx] = tri[key][last]
                for key in ("cx", "cy", "cz", "qw", "qx", "qy", "qz"):
                    tri[key] = tri[key][:last].copy()
                tri["N"] = last
                if last == 0:
                    proposals[b] = ("zero_del", mv, idx, (tri, 0.0, False))
                else:
                    proposals[b] = ("eval", mv, idx, tri)
                    eval_indices.append(b)
                    eval_states.append(tri)

        eval_map = {}
        if eval_states:
            if batch_target_fn is not None:
                res_list = batch_target_fn(eval_states)
            else:
                res_list = [target_fn(s) for s in eval_states]
            mlip_evals += len(eval_states)
            for b, r in zip(eval_indices, res_list):
                eval_map[b] = r

        for b in range(n_b):
            kind, mv, _idx, payload = proposals[b]
            cur_N = int(boxes[b]["N"])
            accepted = False
            u_tri = U_curr[b]

            if kind == "eval":
                tri = payload
                u_tri, ov = eval_map[b]
                if not ov:
                    dU = float(u_tri - U_curr[b])
                    arg = -dU / float(T_K)
                    if mv in (0, 1):
                        if dU <= 0.0 or (arg > -50.0 and rng.random() < math.exp(arg)):
                            accepted = True
                    elif mv == 2:
                        log_acc = math.log(max(1e-300, beta_f_V / float(cur_N + 1))) + arg
                        if log_acc >= 0.0 or (log_acc > -50.0 and rng.random() < math.exp(log_acc)):
                            accepted = True
                    else:
                        log_acc = math.log(max(1e-300, float(cur_N) / max(1e-300, beta_f_V))) + arg
                        if log_acc >= 0.0 or (log_acc > -50.0 and rng.random() < math.exp(log_acc)):
                            accepted = True
            elif kind == "zero_del":
                tri, u_tri, ov = payload
                if not ov:
                    dU = float(u_tri - U_curr[b])
                    log_acc = math.log(max(1e-300, float(cur_N) / max(1e-300, beta_f_V))) - dU / float(T_K)
                    if log_acc >= 0.0 or (log_acc > -50.0 and rng.random() < math.exp(log_acc)):
                        accepted = True

            if in_prod:
                if mv == 0 and cur_N > 0:
                    prod_trans_att += 1
                    if accepted:
                        prod_trans_acc += 1
                elif mv == 1 and cur_N > 0:
                    prod_rot_att += 1
                    if accepted:
                        prod_rot_acc += 1
                elif mv == 2:
                    prod_ins_att += 1
                    if accepted:
                        prod_ins_acc += 1
                elif mv == 3 and cur_N > 0:
                    prod_del_att += 1
                    if accepted:
                        prod_del_acc += 1

            if accepted:
                boxes[b] = tri
                U_curr[b] = float(u_tri)
                acc_total[b] += 1
                if in_prod:
                    prod_acc_total[b] += 1

        N_now = [int(boxes[b]["N"]) for b in range(n_b)]
        box_N_history.append(N_now)
        if in_prod:
            prod_steps_done += 1
            prod_N_history.append(N_now)
            prod_U_history.append([float(U_curr[b] * K_TO_KJMOL) for b in range(n_b)])
            for b in range(n_b):
                dn = float(N_now[b])
                du = float(U_curr[b])
                sum_N[b] += dn
                sum_N2[b] += dn * dn
                sum_U[b] += du
                sum_UN[b] += du * dn

        steps_completed += 1
        if log_every > 0 and ((step + 1) % log_every == 0):
            elapsed = time.perf_counter() - t0
            mean_up_now = float(np.mean(N_now) * conv_mmol_g)
            print(
                f"  [DirectGCMC step {step+1:4d}/{total_steps}] N={N_now} ({mean_up_now:.2f} mmol/g) "
                f"Acc={np.mean(acc_total)/(step+1)*100:5.1f}% HSrej={hs_rejected} ({elapsed:.1f}s)"
            )

    dt = time.perf_counter() - t0
    prod_denom = max(1, prod_steps_done)
    tot_denom = max(1, steps_completed)
    box_uptakes = (sum_N / prod_denom) * conv_mmol_g if prod_steps_done > 0 else np.zeros(n_b, dtype=np.float64)
    box_moments = [
        {
            "n_samples": int(prod_steps_done),
            "sum_N": float(sum_N[b]),
            "sum_N2": float(sum_N2[b]),
            "sum_U": float(sum_U[b]),
            "sum_UN": float(sum_UN[b]),
        }
        for b in range(n_b)
    ]
    jack = compute_pooled_jackknife_qst(box_moments, float(T_K))
    ess_stats = compute_ess_stats(prod_N_history, prod_U_history, dt)

    return {
        "n_boxes": n_b,
        "eq_steps": int(eq_steps),
        "prod_steps_requested": int(prod_steps),
        "prod_steps_done": int(prod_steps_done),
        "steps_completed": int(steps_completed),
        "elapsed_s": float(dt),
        "fugacity_Pa": float(f_Pa),
        "uptake_mmol_g": float(np.mean(box_uptakes)),
        "uptake_sem_mmol_g": float(np.std(box_uptakes, ddof=1) / math.sqrt(n_b)) if n_b > 1 else 0.0,
        "box_uptakes_mmol_g": box_uptakes.tolist(),
        "Qst_estimator": "pooled_jackknife",
        "Qst_kJ_mol": float(jack["Qst_kJ_mol"]),
        "Qst_sem_kJ_mol": float(jack["Qst_sem_kJ_mol"]),
        "box_moments": box_moments,
        "tau_int_N": ess_stats["tau_int_N"],
        "ess_N": ess_stats["ess_N"],
        "ess_N_per_s": ess_stats["ess_N_per_s"],
        "tau_int_U": ess_stats["tau_int_U"],
        "ess_U": ess_stats["ess_U"],
        "ess_U_per_s": ess_stats["ess_U_per_s"],
        "esjd_N": ess_stats["esjd_N"],
        "sjd_N_per_s": ess_stats["sjd_N_per_s"],
        "dN_moves_per_s": ess_stats["dN_moves_per_s"],
        "move_acc_rate": float(np.mean(acc_total) / tot_denom),
        "prod_move_acc_rate": float(np.mean(prod_acc_total) / prod_denom) if prod_steps_done > 0 else 0.0,
        "prod_ins_att": int(prod_ins_att),
        "prod_ins_acc": int(prod_ins_acc),
        "prod_del_att": int(prod_del_att),
        "prod_del_acc": int(prod_del_acc),
        "prod_trans_att": int(prod_trans_att),
        "prod_trans_acc": int(prod_trans_acc),
        "prod_rot_att": int(prod_rot_att),
        "prod_rot_acc": int(prod_rot_acc),
        "hs_rejected": int(hs_rejected),
        "mlip_evals": int(mlip_evals),
        "box_N_history": box_N_history,
        "prod_N_history": prod_N_history,
        "prod_U_history": prod_U_history,
        "final_states": [_copy_box_state(s) for s in boxes],
    }
