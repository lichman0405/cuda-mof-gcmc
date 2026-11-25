"""Zero-manual-typing on-the-fly 1B+2B MLIP self-distillation for CUDA Nested GCMC."""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from guests import GuestSite, GuestSpec, load_guest
from run_metadata import sha256_file, sha256_json_obj

K_TO_KJMOL = 8.31446261815324e-3
KE_K = 167100.9558738398
DISTILL_VERSION = "v1_1b2b"

# Mild generic LJ defaults (eps_K, sigma_A) used ONLY for guest-guest steric core
# when a guest site carries zero classical LJ parameters (e.g. SPC/E or TIP-style H).
DEFAULT_SITE_LJ = {
    1: (15.0, 2.20),
    6: (33.0, 3.40),
    7: (36.0, 3.26),
    8: (60.0, 3.00),
    18: (119.8, 3.405),
}

METAL_ATOMIC_NUMBERS = {
    3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 55, 56,
    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
}


def make_distilled_guest_spec(guest: GuestSpec) -> GuestSpec:
    """Assign each symmetry-distinct real atom site a unique distilled grid channel (<= 6) and steric core."""
    chan_map: Dict[Tuple[int, float], int] = {}
    new_sites: List[GuestSite] = []
    for s in guest.sites:
        if int(s.Z) <= 0:
            new_sites.append(
                GuestSite(
                    symbol=s.symbol, Z=0, pos=s.pos, q=float(s.q),
                    eps_K=0.0, sigma_A=0.0, lj_type=-1,
                )
            )
            continue
        rad = round(math.sqrt(sum(v * v for v in s.pos)), 3)
        key = (int(s.Z), rad)
        if key not in chan_map:
            if len(chan_map) >= 6:
                raise ValueError(
                    f"Guest '{guest.name}' has more than 6 distinct real-atom grid channels; cannot distill."
                )
            chan_map[key] = len(chan_map)
        eps_k = float(s.eps_K)
        sig_a = float(s.sigma_A)
        if eps_k <= 0.0 or sig_a <= 0.0:
            eps_k, sig_a = DEFAULT_SITE_LJ.get(int(s.Z), (30.0, 3.00))
        new_sites.append(
            GuestSite(
                symbol=s.symbol,
                Z=int(s.Z),
                pos=s.pos,
                q=float(s.q),
                eps_K=eps_k,
                sigma_A=sig_a,
                lj_type=chan_map[key],
            )
        )
    return GuestSpec(
        name=guest.name,
        formula=guest.formula,
        is_isotropic=guest.is_isotropic,
        Tc_K=guest.Tc_K,
        Pc_bar=guest.Pc_bar,
        omega=guest.omega,
        sites=new_sites,
    )


def assign_framework_environments(uc_atoms, charges: Optional[np.ndarray] = None) -> dict:
    """Automatically classify rigid MOF unit-cell atoms by periodic coordination signature."""
    from ase.data import covalent_radii

    Z = np.asarray(uc_atoms.get_atomic_numbers(), dtype=np.int32)
    n = len(Z)
    H = np.asarray(uc_atoms.get_cell(), dtype=np.float64)
    inv_H = np.linalg.inv(H)
    frac = np.mod(np.asarray(uc_atoms.get_scaled_positions(), dtype=np.float64), 1.0)
    pos = frac @ H

    perp_w = 1.0 / np.linalg.norm(inv_H, axis=0)
    max_rcov = float(np.max([covalent_radii[int(z)] for z in Z]))
    r_search = 2.0 * max_rcov + 0.40
    n_rep = [max(1, int(math.ceil(r_search / float(w)))) for w in perp_w]
    shifts_f = np.stack(
        np.meshgrid(
            np.arange(-n_rep[0], n_rep[0] + 1),
            np.arange(-n_rep[1], n_rep[1] + 1),
            np.arange(-n_rep[2], n_rep[2] + 1),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    shift_vecs = shifts_f @ H

    df = frac[None, :, :] - frac[:, None, :]
    df -= np.round(df)
    dr0 = df @ H

    neighbors: List[List[int]] = [[] for _ in range(n)]
    r_cov_arr = np.array([float(covalent_radii[int(z)]) for z in Z], dtype=np.float64)
    thresh2 = (r_cov_arr[:, None] + r_cov_arr[None, :] + 0.35) ** 2

    for sv in shift_vecs:
        dr = dr0 + sv.reshape(1, 1, 3)
        r2 = np.sum(dr * dr, axis=-1)
        bonded = (r2 > 0.16) & (r2 < thresh2)
        ii, jj = np.nonzero(bonded)
        for i, j in zip(ii.tolist(), jj.tolist()):
            neighbors[i].append(int(Z[j]))

    sigs = [(int(Z[i]), tuple(sorted(neighbors[i]))) for i in range(n)]
    unique_sigs = sorted(set(sigs), key=lambda s: (s[0], len(s[1]), s[1]))
    sig_to_id = {s: idx for idx, s in enumerate(unique_sigs)}
    env_ids = np.array([sig_to_id[s] for s in sigs], dtype=np.int32)
    counts = np.bincount(env_ids, minlength=len(unique_sigs)).astype(np.float64)

    q_arr = np.zeros(n, dtype=np.float64) if charges is None else np.asarray(charges, dtype=np.float64)
    is_metal = np.array([int(z) in METAL_ATOMIC_NUMBERS for z in Z], dtype=bool)
    is_polar = np.abs(q_arr) >= 0.40
    # Mark coordinatively unsaturated or polar sites as priority pocket anchors
    active_mask = is_metal | is_polar
    if not np.any(active_mask):
        active_mask = Z > 1

    return {
        "env_ids": env_ids,
        "unique_sigs": unique_sigs,
        "env_counts": counts,
        "frac": frac,
        "pos": pos,
        "H_unit": H,
        "inv_H_unit": inv_H,
        "Z": Z,
        "active_mask": active_mask,
        "is_metal_mask": is_metal,
    }


def _quat_rotate_np(q: np.ndarray, local_xyz: np.ndarray) -> np.ndarray:
    qxyz = q[:, 1:4][:, None, :]
    v = local_xyz[None, :, :]
    t = 2.0 * np.cross(qxyz, v)
    return v + q[:, 0:1][:, None] * t + np.cross(qxyz, t)


def _rand_uniform_quats(rng: np.random.Generator, n: int) -> np.ndarray:
    u1 = rng.random(n)
    u2 = (2.0 * math.pi) * rng.random(n)
    u3 = (2.0 * math.pi) * rng.random(n)
    s1 = np.sqrt(np.maximum(0.0, 1.0 - u1))
    s2 = np.sqrt(np.maximum(0.0, u1))
    return np.stack([s2 * np.cos(u3), s1 * np.sin(u2), s1 * np.cos(u2), s2 * np.sin(u3)], axis=1)


def _min_periodic_distances(pts: np.ndarray, ref_frac: np.ndarray, H: np.ndarray, inv_H: np.ndarray, device=None) -> np.ndarray:
    """Compute exact minimum periodic distance from points (M, 3) to ref_frac (R, 3)."""
    if len(pts) == 0:
        return np.zeros(0, dtype=np.float64)
    pts_frac = pts @ inv_H
    if device is not None:
        try:
            import torch
            if torch.cuda.is_available():
                H_t = torch.tensor(H, device=device, dtype=torch.float64)
                pf_t = torch.tensor(pts_frac, device=device, dtype=torch.float64)
                rf_t = torch.tensor(ref_frac, device=device, dtype=torch.float64)
                out = torch.empty(len(pts), device=device, dtype=torch.float64)
                chunk = 8192
                for s0 in range(0, len(pts), chunk):
                    s1 = min(len(pts), s0 + chunk)
                    df = pf_t[s0:s1, None, :] - rf_t[None, :, :]
                    df -= torch.round(df)
                    dr = df @ H_t
                    out[s0:s1] = torch.sqrt(torch.min(torch.sum(dr * dr, dim=-1), dim=1).values)
                return out.cpu().numpy()
        except Exception:
            pass
    out = np.empty(len(pts), dtype=np.float64)
    chunk = 4096
    for s0 in range(0, len(pts), chunk):
        s1 = min(len(pts), s0 + chunk)
        df = pts_frac[s0:s1, None, :] - ref_frac[None, :, :]
        df -= np.round(df)
        dr = df @ H
        out[s0:s1] = np.sqrt(np.min(np.sum(dr * dr, axis=-1), axis=1))
    return out


def _eval_unit_basis_torch(
    pts_cart: np.ndarray,
    site_chan_idx: Optional[np.ndarray],
    n_pts_per_mol: int,
    n_chan: int,
    guest_chan_q: np.ndarray,
    env_info: dict,
    r_cut: float = 8.0,
    ewald_alpha: float = 0.30,
    k_max: float = 2.0,
    mu_centers: np.ndarray = None,
    sigma_rbf: float = 0.45,
    device=None,
    chunk_pts: int = 8192,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate per-(channel, env) radial basis + per-env Ewald potential on GPU/CPU.

    If `site_chan_idx` is None, evaluates per-point basis summed into `(M, n_env, n_rad)` and `phi (M, n_env)`
    for all grid points `pts_cart (M, 3)` belonging to a template point; if `site_chan_idx` is provided
    for molecule atoms (`pts_cart` shape `(n_mol * n_real, 3)`), aggregates directly to `(n_mol, ...)`.
    """
    import torch

    use_dev = device if (device is not None and torch.cuda.is_available()) else torch.device("cpu")
    H_u = env_info["H_unit"]
    inv_H_u = env_info["inv_H_unit"]
    fw_frac = env_info["frac"]
    fw_pos_u = env_info["pos"]
    env_ids = env_info["env_ids"]
    n_env = len(env_info["unique_sigs"])
    vol_u = abs(float(np.linalg.det(H_u)))

    perp_w = 1.0 / np.linalg.norm(inv_H_u, axis=0)
    n_rep = [max(1, int(math.ceil(r_cut / float(w))) + 1) for w in perp_w]
    shifts_f = np.stack(
        np.meshgrid(
            np.arange(-n_rep[0], n_rep[0] + 1),
            np.arange(-n_rep[1], n_rep[1] + 1),
            np.arange(-n_rep[2], n_rep[2] + 1),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    shifts_cart = shifts_f @ H_u

    # Reciprocal Ewald vectors (half-sphere)
    b_vecs = 2.0 * math.pi * inv_H_u.T
    nb = np.linalg.norm(b_vecs, axis=1)
    m_max = [int(math.ceil(k_max / max(1e-3, float(w)))) + 1 for w in nb]
    k_vecs = []
    k_w = []
    inv_4a2 = 1.0 / (4.0 * ewald_alpha * ewald_alpha)
    pref = 2.0 * KE_K * (4.0 * math.pi / vol_u)
    kmax2 = k_max * k_max
    for mc in range(0, m_max[2] + 1):
        mb_min = 0 if mc == 0 else -m_max[1]
        for mb in range(mb_min, m_max[1] + 1):
            ma_min = 1 if (mc == 0 and mb == 0) else -m_max[0]
            for ma in range(ma_min, m_max[0] + 1):
                kv = ma * b_vecs[0] + mb * b_vecs[1] + mc * b_vecs[2]
                k2 = float(np.dot(kv, kv))
                if 1e-12 < k2 <= kmax2:
                    k_vecs.append(kv)
                    k_w.append(pref * math.exp(-k2 * inv_4a2) / k2)

    k_arr = np.asarray(k_vecs, dtype=np.float64)  # (K, 3)
    kw_arr = np.asarray(k_w, dtype=np.float64)    # (K,)
    kr_fw = fw_pos_u @ k_arr.T                     # (N_fw, K)
    cos_fw = np.cos(kr_fw)
    sin_fw = np.sin(kr_fw)
    env_onehot = np.eye(n_env, dtype=np.float64)[env_ids]  # (N_fw, n_env)
    # W_C[k, alpha] = kw[k] * sum_{j in alpha} cos(k . r_j)
    W_C = kw_arr[:, None] * (cos_fw.T @ env_onehot)  # (K, n_env)
    W_S = kw_arr[:, None] * (sin_fw.T @ env_onehot)  # (K, n_env)

    H_t = torch.tensor(H_u, device=use_dev, dtype=torch.float32)
    inv_H_t = torch.tensor(inv_H_u, device=use_dev, dtype=torch.float32)
    fw_f_t = torch.tensor(fw_frac, device=use_dev, dtype=torch.float32)
    sh_t = torch.tensor(shifts_cart, device=use_dev, dtype=torch.float32)
    env_oh_t = torch.tensor(env_onehot, device=use_dev, dtype=torch.float32)
    k_t = torch.tensor(k_arr, device=use_dev, dtype=torch.float32)
    WC_t = torch.tensor(W_C, device=use_dev, dtype=torch.float32)
    WS_t = torch.tensor(W_S, device=use_dev, dtype=torch.float32)
    mu_t = torch.tensor(mu_centers, device=use_dev, dtype=torch.float32)

    n_rad = 1 + len(mu_centers)
    M = len(pts_cart)
    out_rad = np.empty((M, n_env, n_rad), dtype=np.float32)
    out_phi = np.empty((M, n_env), dtype=np.float32)
    out_dmin = np.empty(M, dtype=np.float32)

    rc2 = float(r_cut * r_cut)
    r_on = float(r_cut - 1.5)
    inv_span = 1.0 / max(1e-6, float(r_cut - r_on))
    inv_2sig2 = 1.0 / (2.0 * float(sigma_rbf) * float(sigma_rbf))

    with torch.no_grad():
        for s0 in range(0, M, chunk_pts):
            s1 = min(M, s0 + chunk_pts)
            m_b = s1 - s0
            p_c = torch.tensor(pts_cart[s0:s1], device=use_dev, dtype=torch.float32)
            p_f = torch.remainder(p_c @ inv_H_t, 1.0)
            p_wrap = p_f @ H_t

            # Exact MIC baseline difference: (m_b, N_fw, 3)
            df0 = p_f[:, None, :] - fw_f_t[None, :, :]
            df0 = df0 - torch.round(df0)
            dr0 = df0 @ H_t

            # Reciprocal Ewald: (m_b, n_env)
            kr_p = p_wrap @ k_t.T  # (m_b, K)
            phi_acc = torch.cos(kr_p) @ WC_t + torch.sin(kr_p) @ WS_t

            rad_acc = torch.zeros((m_b, n_env, n_rad), device=use_dev, dtype=torch.float32)
            dmin_b = torch.full((m_b,), 1.0e6, device=use_dev, dtype=torch.float32)

            # Process shifts in batches to bound VRAM
            sh_chunk = 27
            for k0 in range(0, sh_t.shape[0], sh_chunk):
                k1 = min(sh_t.shape[0], k0 + sh_chunk)
                dr = dr0.unsqueeze(1) + sh_t[k0:k1].view(1, -1, 1, 3)  # (m_b, S_c, N_fw, 3)
                r2 = torch.sum(dr * dr, dim=-1)                         # (m_b, S_c, N_fw)
                r2_loc_min = torch.min(r2.reshape(m_b, -1), dim=1).values
                dmin_b = torch.minimum(dmin_b, torch.sqrt(torch.clamp(r2_loc_min, min=1.0e-8)))

                mask = r2 < rc2
                if not torch.any(mask):
                    continue
                r = torch.sqrt(torch.clamp(r2, min=1.44))  # clamp >= 1.2 A to avoid div0
                # Smooth cosine taper on [r_on, r_cut]
                t_tap = torch.clamp((r - r_on) * inv_span, 0.0, 1.0)
                f_cut = torch.where(mask, 0.5 * (1.0 + torch.cos(math.pi * t_tap)), torch.zeros_like(r))

                # Real-space Ewald erfc(alpha * r) / r
                phi_real = (KE_K * torch.erfc(ewald_alpha * r) / r) * mask.to(r.dtype)  # (m_b, S_c, N_fw)
                # Sum over S_c -> (m_b, N_fw) @ (N_fw, n_env)
                phi_acc = phi_acc + torch.sum(phi_real, dim=1) @ env_oh_t

                # Radial basis: [core12, exp_gauss_0..]
                core = torch.pow(2.20 / torch.clamp(r, min=1.25), 12.0) * f_cut
                # Sum core over S_c -> (m_b, N_fw)
                core_fw = torch.sum(core, dim=1)
                rad_acc[:, :, 0] += core_fw @ env_oh_t

                # Gaussians in chunks over mu_t
                diff_r = r.unsqueeze(-1) - mu_t.view(1, 1, 1, -1)  # (m_b, S_c, N_fw, B)
                g_val = torch.exp(-(diff_r * diff_r) * inv_2sig2) * f_cut.unsqueeze(-1)
                g_fw = torch.sum(g_val, dim=1)  # (m_b, N_fw, B)
                rad_acc[:, :, 1:] += torch.einsum("mnb,ne->meb", g_fw, env_oh_t)

            out_rad[s0:s1] = rad_acc.cpu().numpy()
            out_phi[s0:s1] = phi_acc.cpu().numpy()
            out_dmin[s0:s1] = dmin_b.cpu().numpy()

    if site_chan_idx is None:
        return out_rad, out_phi, out_dmin

    # Aggregate molecules -> (n_mol, n_chan, n_env, n_rad) and (n_mol, n_env) for electrostatics
    n_mol = M // n_pts_per_mol
    out_rad_m = np.zeros((n_mol, n_chan, n_env, n_rad), dtype=np.float64)
    out_phi_m = np.zeros((n_mol, n_env), dtype=np.float64)
    rad_r = out_rad.reshape(n_mol, n_pts_per_mol, n_env, n_rad)
    phi_r = out_phi.reshape(n_mol, n_pts_per_mol, n_env)
    dmin_r = out_dmin.reshape(n_mol, n_pts_per_mol)
    for s_idx in range(n_pts_per_mol):
        ch = int(site_chan_idx[s_idx])
        out_rad_m[:, ch, :, :] += rad_r[:, s_idx, :, :]
        out_phi_m += float(guest_chan_q[ch]) * phi_r[:, s_idx, :]
    mol_dmin = np.min(dmin_r, axis=1)
    return out_rad_m, out_phi_m, mol_dmin


def _unscaled_gg_pair_energy_K(
    st_a: dict, st_b: dict,
    cell_H: np.ndarray, cell_inv_H: np.ndarray,
    guest: GuestSpec,
    r_cut_gg: float,
    dsf_alpha: float = 0.20,
) -> Tuple[float, float, float, bool]:
    """Compute exact unscaled (u_lj_K, u_q_K, r_com_A, overlap) matching pair_rigid_guest."""
    ca = np.array([float(st_a["cx"][0]), float(st_a["cy"][0]), float(st_a["cz"][0])], dtype=np.float64)
    cb = np.array([float(st_b["cx"][0]), float(st_b["cy"][0]), float(st_b["cz"][0])], dtype=np.float64)
    df = (cb - ca) @ cell_inv_H
    df -= np.round(df)
    dm = df @ cell_H
    r2c = float(np.dot(dm, dm))
    rc_com = math.sqrt(r2c)

    max_rad = max((math.sqrt(sum(v * v for v in s.pos)) for s in guest.sites), default=0.0)
    if r2c > (r_cut_gg + 2.0 * max_rad) ** 2:
        return 0.0, 0.0, rc_com, False

    lxyz = np.array([s.pos for s in guest.sites], dtype=np.float64)
    qa = np.array([[st_a["qw"][0], st_a["qx"][0], st_a["qy"][0], st_a["qz"][0]]], dtype=np.float64)
    qb = np.array([[st_b["qw"][0], st_b["qx"][0], st_b["qy"][0], st_b["qz"][0]]], dtype=np.float64)
    oa = _quat_rotate_np(qa, lxyz)[0]
    ob = _quat_rotate_np(qb, lxyz)[0]

    rc2 = r_cut_gg * r_cut_gg
    erfc_rc = math.erfc(dsf_alpha * r_cut_gg)
    exp_rc = math.exp(-dsf_alpha * dsf_alpha * rc2)
    dsf_shift = erfc_rc / r_cut_gg
    dsf_deriv = erfc_rc / rc2 + (2.0 * dsf_alpha / math.sqrt(math.pi)) * exp_rc / r_cut_gg

    u_lj = 0.0
    u_q = 0.0
    n_s = len(guest.sites)
    for ia in range(n_s):
        sa = guest.sites[ia]
        for ib in range(n_s):
            sb = guest.sites[ib]
            has_lj = (sa.eps_K > 0.0 and sb.eps_K > 0.0 and sa.sigma_A > 0.0 and sb.sigma_A > 0.0)
            has_q = (sa.q != 0.0 and sb.q != 0.0)
            if not has_lj and not has_q:
                continue
            d = dm + ob[ib] - oa[ia]
            r2 = float(np.dot(d, d))
            if r2 >= rc2:
                continue
            if r2 < 0.25:
                return 0.0, 0.0, rc_com, True
            if has_lj:
                sig = 0.5 * (sa.sigma_A + sb.sigma_A)
                eps = math.sqrt(sa.eps_K * sb.eps_K)
                sr2 = (sig * sig) / r2
                sr6 = sr2 * sr2 * sr2
                sc2 = (sig * sig) / rc2
                sc6 = sc2 * sc2 * sc2
                u_lj += 4.0 * eps * ((sr6 * sr6 - sr6) - (sc6 * sc6 - sc6))
            if has_q:
                r = math.sqrt(r2)
                u_q += KE_K * sa.q * sb.q * (math.erfc(dsf_alpha * r) / r - dsf_shift + dsf_deriv * (r - r_cut_gg))
    return u_lj, u_q, rc_com, False


def _solve_1b_model(
    X_rad: np.ndarray,
    X_phi: np.ndarray,
    y_K: np.ndarray,
    dmin: np.ndarray,
    env_counts: np.ndarray,
    n_chan: int,
    n_env: int,
    n_rad: int,
    active_q_modes: bool,
    seed: int = 123,
) -> dict:
    """Fit constrained 1-body surrogate and report honest held-out validation metrics."""
    from scipy.linalg import null_space
    from scipy.optimize import lsq_linear

    N = len(y_K)
    # Build electroneutrality null-space basis for effective framework charges (n_env - 1 modes)
    if active_q_modes and n_env >= 2:
        ns = null_space(env_counts.reshape(1, -1))  # (n_env, n_env - 1)
        X_ew = X_phi @ ns
        n_q_modes = ns.shape[1]
    else:
        ns = np.zeros((n_env, 0), dtype=np.float64)
        X_ew = np.zeros((N, 0), dtype=np.float64)
        n_q_modes = 0

    X_sr = X_rad.reshape(N, n_chan * n_env * n_rad)
    n_sr = X_sr.shape[1]
    X_full = np.hstack([X_sr, X_ew]) if n_q_modes > 0 else X_sr
    n_params = X_full.shape[1]

    # Bounds: core exponent weights (k == 0) >= 25.0 K to guarantee short-range repulsive wall
    lb = np.full(n_params, -2.0e5, dtype=np.float64)
    ub = np.full(n_params, +2.0e5, dtype=np.float64)
    for c in range(n_chan):
        for e in range(n_env):
            col_core = (c * n_env + e) * n_rad
            lb[col_core] = 25.0
            ub[col_core] = 5.0e4
    for m in range(n_q_modes):
        lb[n_sr + m] = -5.0
        ub[n_sr + m] = +5.0

    # Boltzmann-inspired weights: emphasize binding minima and zero-crossing, prevent high wall outliers from skewing pocket
    y_kJ = y_K * K_TO_KJMOL
    w_sample = np.where(
        y_kJ <= 15.0,
        np.where(dmin <= 2.75, 1.4, 1.0),
        1.0 / np.sqrt(1.0 + np.maximum(0.0, y_kJ - 15.0) / 20.0),
    )

    # Regularization matrix (light Tikhonov on splines, moderate on Ewald modes)
    reg_diag = np.full(n_params, 1.0e-2, dtype=np.float64)
    for c in range(n_chan):
        for e in range(n_env):
            reg_diag[(c * n_env + e) * n_rad] = 1.0e-4
    if n_q_modes > 0:
        reg_diag[n_sr:] = 2.0e2

    def _fit_subset(idx_sub: np.ndarray) -> np.ndarray:
        sw = np.sqrt(w_sample[idx_sub])[:, None]
        A_w = X_full[idx_sub] * sw
        b_w = y_K[idx_sub] * sw[:, 0]
        A_aug = np.vstack([A_w, np.diag(np.sqrt(reg_diag))])
        b_aug = np.concatenate([b_w, np.zeros(n_params, dtype=np.float64)])
        res = lsq_linear(A_aug, b_aug, bounds=(lb, ub), method="trf", lsmr_tol="auto")
        return res.x

    # 85/15 split for honest held-out metrics
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_val = max(20, int(round(0.15 * N)))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    theta_cv = _fit_subset(tr_idx)
    pred_val = (X_full[val_idx] @ theta_cv) * K_TO_KJMOL
    true_val = y_kJ[val_idx]
    ss_res = float(np.sum((true_val - pred_val) ** 2))
    ss_tot = float(np.sum((true_val - np.mean(true_val)) ** 2))
    r2_val = 1.0 - (ss_res / max(1e-12, ss_tot))
    mae_val = float(np.mean(np.abs(true_val - pred_val)))
    pocket_mask = (true_val <= -15.0) | (dmin[val_idx] <= 2.75)
    mae_pocket = float(np.mean(np.abs(true_val[pocket_mask] - pred_val[pocket_mask]))) if np.any(pocket_mask) else mae_val

    # Final refit on all samples
    all_idx = np.arange(N, dtype=np.int64)
    theta_all = _fit_subset(all_idx)
    w_sr = theta_all[:n_sr].reshape(n_chan, n_env, n_rad)
    q_env = (ns @ theta_all[n_sr:]) if n_q_modes > 0 else np.zeros(n_env, dtype=np.float64)

    return {
        "w_sr": w_sr,
        "q_env": q_env,
        "r2_1b": float(r2_val),
        "mae_1b_kJ_mol": float(mae_val),
        "mae_pocket_kJ_mol": float(mae_pocket),
        "n_val": int(n_val),
        "n_train": int(len(tr_idx)),
    }


def _solve_2b_model(
    u_lj0: np.ndarray,
    u_q0: np.ndarray,
    r_com: np.ndarray,
    u2_true_K: np.ndarray,
    r_cut_gg: float,
) -> dict:
    """Fit effective pore 2-body guest-guest scaling + smoothed COM radial residual LUT."""
    from scipy.optimize import lsq_linear

    r_min_A = 2.0
    r_max_A = float(r_cut_gg)
    n_lut = 128
    lut_r = np.linspace(r_min_A, r_max_A, n_lut, dtype=np.float64)

    mu_2b = np.linspace(2.5, max(3.2, r_max_A - 0.7), 8, dtype=np.float64)
    sig_2b = max(0.45, 0.7 * float(mu_2b[1] - mu_2b[0]))
    r_on = max(r_min_A + 0.5, r_max_A - 1.5)
    inv_span = 1.0 / max(1e-6, r_max_A - r_on)

    def _rad_basis(r_arr: np.ndarray) -> np.ndarray:
        t_tap = np.clip((r_arr - r_on) * inv_span, 0.0, 1.0)
        f_cut = 0.5 * (1.0 + np.cos(math.pi * t_tap))
        f_cut = np.where(r_arr >= r_max_A, 0.0, f_cut)
        diff = r_arr[:, None] - mu_2b[None, :]
        return np.exp(-(diff * diff) / (2.0 * sig_2b * sig_2b)) * f_cut[:, None]

    if len(u2_true_K) < 16 or np.max(np.abs(u2_true_K)) < 1e-6:
        return {
            "gg_q_scale": 1.0,
            "gg_lj_scale": 1.0,
            "r_min_A": r_min_A,
            "r_max_A": r_max_A,
            "gg_com_lut_K": np.zeros(n_lut, dtype=np.float32),
            "r2_2b": 1.0,
            "mae_2b_kJ_mol": 0.0,
            "mean_offset_classical_gg_kJ_mol": 0.0,
            "mean_offset_distilled_2b_kJ_mol": 0.0,
        }

    B_2b = _rad_basis(r_com)
    X = np.column_stack([u_lj0, u_q0, B_2b])  # shape (M, 2 + 8)
    n_p = X.shape[1]

    lb = np.full(n_p, -2000.0, dtype=np.float64)
    ub = np.full(n_p, +2000.0, dtype=np.float64)
    lb[0], ub[0] = 0.15, 2.50  # gg_lj_scale
    lb[1], ub[1] = 0.00, 2.50  # gg_q_scale^2

    # Mild prior pulling c_lj, c_q2 toward 1.0 and radial corrections toward 0.0
    prior = np.zeros(n_p, dtype=np.float64)
    prior[0] = 1.0
    prior[1] = 1.0
    reg_w = np.full(n_p, 1.0e-1, dtype=np.float64)
    reg_w[0] = 5.0e1
    reg_w[1] = 5.0e1

    y_kJ = u2_true_K * K_TO_KJMOL
    sw = np.sqrt(np.where(y_kJ <= 10.0, 1.0, 1.0 / (1.0 + np.maximum(0.0, y_kJ - 10.0) / 15.0)))
    A_w = X * sw[:, None]
    b_w = u2_true_K * sw
    A_aug = np.vstack([A_w, np.diag(np.sqrt(reg_w))])
    b_aug = np.concatenate([b_w, np.sqrt(reg_w) * prior])

    res = lsq_linear(A_aug, b_aug, bounds=(lb, ub), method="trf", lsmr_tol="auto")
    theta = res.x
    gg_lj_scale = float(theta[0])
    gg_q_scale = math.sqrt(max(0.0, float(theta[1])))
    w_rad = theta[2:]

    B_lut = _rad_basis(lut_r)
    lut_K = (B_lut @ w_rad).astype(np.float64)
    lut_K[-1] = 0.0

    pred_K = X @ theta
    ss_res = float(np.sum((u2_true_K - pred_K) ** 2))
    ss_tot = float(np.sum((u2_true_K - np.mean(u2_true_K)) ** 2))
    r2_2b = 1.0 - ss_res / max(1e-12, ss_tot)
    mae_2b_kJ = float(np.mean(np.abs(u2_true_K - pred_K)) * K_TO_KJMOL)
    offset_raw_kJ = float(np.mean(u2_true_K - (u_lj0 + u_q0)) * K_TO_KJMOL)
    offset_fit_kJ = float(np.mean(u2_true_K - pred_K) * K_TO_KJMOL)

    return {
        "gg_q_scale": float(gg_q_scale),
        "gg_lj_scale": float(gg_lj_scale),
        "r_min_A": float(r_min_A),
        "r_max_A": float(r_max_A),
        "gg_com_lut_K": lut_K.astype(np.float32),
        "r2_2b": float(r2_2b),
        "mae_2b_kJ_mol": float(mae_2b_kJ),
        "mean_offset_classical_gg_kJ_mol": float(offset_raw_kJ),
        "mean_offset_distilled_2b_kJ_mol": float(offset_fit_kJ),
    }


def _render_unit_cell_grids(
    grid_dims: Tuple[int, int, int],
    env_info: dict,
    dist_guest: GuestSpec,
    fit_1b: dict,
    mu_centers: np.ndarray,
    sigma_rbf: float,
    r_cut: float,
    device=None,
) -> np.ndarray:
    """Render C-contiguous (n_chan, nz, ny, nx) float32 3D grids in Kelvin on GPU."""
    nx, ny, nz = [int(d) for d in grid_dims]
    H_u = env_info["H_unit"]
    w_sr = fit_1b["w_sr"]      # (n_chan, n_env, n_rad)
    q_env = fit_1b["q_env"]    # (n_env,)

    # Extract ordered channels and per-channel atomic numbers / charges
    chan_info: Dict[int, Tuple[int, float]] = {}
    for s in dist_guest.sites:
        if s.lj_type >= 0 and s.lj_type not in chan_info:
            chan_info[int(s.lj_type)] = (int(s.Z), float(s.q))
    n_chan = len(chan_info)
    chan_Z = [chan_info[c][0] for c in range(n_chan)]
    chan_q = np.array([chan_info[c][1] for c in range(n_chan)], dtype=np.float64)

    # Exact C-contiguous (nz, ny, nx) fractional mesh
    sa = np.arange(nx, dtype=np.float64) / float(nx)
    sb = np.arange(ny, dtype=np.float64) / float(ny)
    sc = np.arange(nz, dtype=np.float64) / float(nz)
    kz, ky, kx = np.meshgrid(sc, sb, sa, indexing="ij")
    f_pts = np.stack([kx.ravel(), ky.ravel(), kz.ravel()], axis=1)
    cart_pts = f_pts @ H_u

    rad_pts, phi_pts, dmin_pts = _eval_unit_cell_pts(
        cart_pts, env_info, mu_centers, sigma_rbf, r_cut, device=device
    )

    total = nx * ny * nz
    grids = np.empty((n_chan, nz, ny, nx), dtype=np.float32)
    phi_term = phi_pts @ q_env  # (total,)

    for c in range(n_chan):
        # V_c = sum_{e,b} w_sr[c,e,b] * rad[p,e,b] + chan_q[c] * phi_term[p]
        val = np.einsum("peb,eb->p", rad_pts, w_sr[c], optimize=True) + chan_q[c] * phi_term
        r_hard = 1.20 if chan_Z[c] == 1 else 1.48
        ovl = (dmin_pts < r_hard) | (val > 1.0e5)
        val = np.where(ovl, 1.0e6, np.clip(val, -2.0e5, 1.0e5))
        grids[c] = val.astype(np.float32).reshape(nz, ny, nx)

    return np.ascontiguousarray(grids, dtype=np.float32)


def _eval_unit_cell_pts(
    cart_pts: np.ndarray,
    env_info: dict,
    mu_centers: np.ndarray,
    sigma_rbf: float,
    r_cut: float,
    device=None,
):
    dummy_q = np.zeros(1, dtype=np.float64)
    return _eval_unit_basis_torch(
        cart_pts,
        site_chan_idx=None,
        n_pts_per_mol=1,
        n_chan=1,
        guest_chan_q=dummy_q,
        env_info=env_info,
        r_cut=r_cut,
        mu_centers=mu_centers,
        sigma_rbf=sigma_rbf,
        device=device,
        chunk_pts=16384,
    )


def compute_distill_cache_key(
    cif_path: str,
    supercell: Tuple[int, int, int],
    guest: GuestSpec,
    grid_dims: Tuple[int, int, int],
    model_name: str,
    model_file_path: Optional[str],
    dispersion: bool,
    dispersion_xc: str,
    dtype: str,
    n_1b: int,
    n_2b: int,
    distill_seed: int,
) -> str:
    real_geom = [
        {"symbol": s.symbol, "Z": int(s.Z), "pos": [round(float(v), 8) for v in s.pos]}
        for s in guest.real_sites()
    ]
    sites_all = [
        {"symbol": s.symbol, "Z": int(s.Z), "pos": [round(float(v), 8) for v in s.pos],
         "q": round(float(s.q), 8), "eps": round(float(s.eps_K), 8), "sig": round(float(s.sigma_A), 8)}
        for s in guest.sites
    ]
    key_obj = {
        "version": DISTILL_VERSION,
        "cif_sha256": sha256_file(cif_path) or Path(cif_path).name,
        "supercell": [int(x) for x in supercell],
        "guest_real_geom_sha256": sha256_json_obj(real_geom),
        "guest_sites_sha256": sha256_json_obj(sites_all),
        "grid_dims": [int(x) for x in grid_dims],
        "model": str(model_name),
        "model_sha256": sha256_file(model_file_path) if model_file_path else None,
        "dispersion": bool(dispersion),
        "dispersion_xc": str(dispersion_xc),
        "dtype": str(dtype),
        "n_1b": int(n_1b),
        "n_2b": int(n_2b),
        "distill_seed": int(distill_seed),
    }
    return sha256_json_obj(key_obj)


def load_distill_cache(cache_npz_path: str) -> dict:
    p_npz = Path(cache_npz_path)
    p_json = p_npz.with_suffix(".json")
    data = np.load(p_npz, allow_pickle=False)
    meta = json.loads(p_json.read_text(encoding="utf-8"))
    return {
        "cache_hit": True,
        "cache_npz_path": str(p_npz),
        "cache_json_path": str(p_json),
        "distill_sha256": meta["distill_sha256"],
        "grids_u_lj": np.ascontiguousarray(data["grids_u_lj"], dtype=np.float32),
        "gg_com_lut_K": np.ascontiguousarray(data["gg_com_lut_K"], dtype=np.float32),
        "gg_q_scale": float(meta["gg_q_scale"]),
        "gg_lj_scale": float(meta["gg_lj_scale"]),
        "gg_lut_rmin_A": float(meta["gg_lut_rmin_A"]),
        "gg_lut_rmax_A": float(meta["gg_lut_rmax_A"]),
        "summary": meta,
    }


def distill_mlip_surrogate(
    cif_path: str,
    fw_data: dict,
    guest: GuestSpec,
    supercell: Tuple[int, int, int],
    target,
    grid_dims: Tuple[int, int, int],
    n_1b: int = 960,
    n_2b: int = 192,
    distill_seed: int = 20260001,
    cache_dir: str = "build/distill_cache",
    model_name: str = "medium",
    dispersion: bool = True,
    dispersion_xc: str = "pbe",
    dtype: str = "float32",
) -> dict:
    """Distill target MLIP (1B unit-cell grids + 2B effective pore pair LUT) on-the-fly with disk caching."""
    import time

    t0 = time.perf_counter()
    dist_guest = make_distilled_guest_spec(guest)
    model_file = getattr(target, "model_file_path", None) if target is not None else None
    cache_key = compute_distill_cache_key(
        cif_path=cif_path,
        supercell=tuple(supercell),
        guest=dist_guest,
        grid_dims=tuple(grid_dims),
        model_name=model_name,
        model_file_path=model_file,
        dispersion=dispersion,
        dispersion_xc=dispersion_xc,
        dtype=dtype,
        n_1b=n_1b,
        n_2b=n_2b,
        distill_seed=distill_seed,
    )

    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    p_npz = cdir / f"distill_{cache_key[:16]}.npz"
    p_json = cdir / f"distill_{cache_key[:16]}.json"
    if p_npz.exists() and p_json.exists():
        meta_chk = json.loads(p_json.read_text(encoding="utf-8"))
        if meta_chk.get("distill_sha256") == cache_key and list(meta_chk.get("grid_dims", [])) == list(grid_dims):
            out = load_distill_cache(str(p_npz))
            out["distill_wall_s"] = float(time.perf_counter() - t0)
            print(f"  [Auto-MLIP Distill CACHE HIT] {p_npz} (R2_1b={meta_chk['r2_1b']:.4f}, R2_2b={meta_chk['r2_2b']:.4f})")
            return out

    if target is None:
        raise RuntimeError("Target MLIP evaluator is required on distillation cache miss.")

    import time as _t
    t_prep = _t.perf_counter()
    rng = np.random.default_rng(distill_seed)
    uc = fw_data["uc"]
    env_info = assign_framework_environments(uc, charges=fw_data.get("charges"))
    n_env = len(env_info["unique_sigs"])

    fw_super = uc.repeat(tuple(supercell))
    H_super = np.asarray(fw_super.get_cell(), dtype=np.float64)
    inv_H_super = np.linalg.inv(H_super)
    perp_super = 1.0 / np.linalg.norm(inv_H_super, axis=0)
    r_cut_gg = min(11.0, 0.48 * float(np.min(perp_super)))
    fw_super_frac = np.mod(np.asarray(fw_super.get_scaled_positions(), dtype=np.float64), 1.0)
    fw_super_pos = fw_super_frac @ H_super
    active_super = fw_super_pos[np.tile(env_info["active_mask"], int(np.prod(supercell)))]

    real_sites = [s for s in dist_guest.sites if s.Z > 0]
    n_real = len(real_sites)
    local_real = np.array([s.pos for s in real_sites], dtype=np.float64)
    site_chan_idx = np.array([s.lj_type for s in real_sites], dtype=np.int32)
    chan_q_map: Dict[int, float] = {}
    for s in real_sites:
        if s.lj_type not in chan_q_map:
            chan_q_map[s.lj_type] = float(s.q)
    n_chan = len(chan_q_map)
    guest_chan_q = np.array([chan_q_map[c] for c in range(n_chan)], dtype=np.float64)
    active_q_modes = (n_env >= 2) and (float(np.max(np.abs(guest_chan_q))) > 1e-8)

    torch_dev = getattr(target, "device", None)

    # 1. Generate stratified single-molecule candidate pool in supercell
    n_pool = max(int(n_1b) * 14, 10000)
    n_near = n_pool // 2
    anchor_idx = rng.integers(0, len(active_super), size=n_near)
    u_dir = rng.normal(size=(n_near, 3))
    u_dir /= np.maximum(1e-12, np.linalg.norm(u_dir, axis=1, keepdims=True))
    radii = rng.uniform(1.85, 4.10, size=(n_near, 1))
    c_near = ((active_super[anchor_idx] + radii * u_dir) @ inv_H_super % 1.0) @ H_super
    c_uni = rng.random((n_pool - n_near, 3)) @ H_super
    c_pool = np.vstack([c_near, c_uni])
    q_pool = np.array([[1.0, 0.0, 0.0, 0.0]] * n_pool) if dist_guest.is_isotropic else _rand_uniform_quats(rng, n_pool)

    offs_pool = _quat_rotate_np(q_pool, local_real)
    pts_pool = (c_pool[:, None, :] + offs_pool).reshape(-1, 3)
    dmin_pts = _min_periodic_distances(pts_pool, fw_super_frac, H_super, inv_H_super, device=torch_dev)
    dmin_mol = np.min(dmin_pts.reshape(n_pool, n_real), axis=1)

    has_h = any(s.Z == 1 for s in real_sites)
    d_clash = 1.65 if has_h else 1.78
    legal = np.where((dmin_mol >= d_clash) & (dmin_mol <= 12.0))[0]
    d_leg = dmin_mol[legal]

    b1 = legal[(d_leg >= d_clash) & (d_leg <= 2.75)]
    b2 = legal[(d_leg > 2.75) & (d_leg <= 3.85)]
    b3 = legal[d_leg > 3.85]
    q1 = int(round(0.45 * n_1b))
    q2 = int(round(0.35 * n_1b))
    q3 = n_1b - q1 - q2

    chosen = []
    for arr, quota in ((b1, q1), (b2, q2), (b3, q3)):
        if len(arr) > 0:
            take = min(quota, len(arr))
            chosen.append(rng.choice(arr, size=take, replace=False))
    sel_1b = np.concatenate(chosen) if chosen else np.zeros(0, dtype=np.int64)
    if len(sel_1b) < n_1b:
        rem = np.setdiff1d(legal, sel_1b, assume_unique=False)
        need = min(len(rem), n_1b - len(sel_1b))
        if need > 0:
            sel_1b = np.concatenate([sel_1b, rng.choice(rem, size=need, replace=False)])

    states_1b = [
        {
            "N": 1,
            "cx": np.array([float(c_pool[i, 0])]),
            "cy": np.array([float(c_pool[i, 1])]),
            "cz": np.array([float(c_pool[i, 2])]),
            "qw": np.array([float(q_pool[i, 0])]),
            "qx": np.array([float(q_pool[i, 1])]),
            "qy": np.array([float(q_pool[i, 2])]),
            "qz": np.array([float(q_pool[i, 3])]),
        }
        for i in sel_1b
    ]
    t_mace1_0 = _t.perf_counter()
    ev_1b = target.eval_batch(states_1b)
    t_mace1_s = float(_t.perf_counter() - t_mace1_0)

    y_1b_all = np.array([float(e[0]) for e in ev_1b], dtype=np.float64)
    keep_1b = np.isfinite(y_1b_all) & (y_1b_all <= 1.5e4)
    sel_kept = sel_1b[keep_1b]
    states_1b_kept = [st for st, ok in zip(states_1b, keep_1b) if ok]
    y_1b = y_1b_all[keep_1b]
    pts_1b = (c_pool[sel_kept][:, None, :] + offs_pool[sel_kept]).reshape(-1, 3)

    mu_centers = np.linspace(1.80, 6.40, 14, dtype=np.float64)
    sigma_rbf = 0.45
    r_cut_1b = 8.0
    n_rad = 1 + len(mu_centers)

    X_rad, X_phi, mol_dmin = _eval_unit_basis_torch(
        pts_1b,
        site_chan_idx=site_chan_idx,
        n_pts_per_mol=n_real,
        n_chan=n_chan,
        guest_chan_q=guest_chan_q,
        env_info=env_info,
        r_cut=r_cut_1b,
        mu_centers=mu_centers,
        sigma_rbf=sigma_rbf,
        device=torch_dev,
    )
    fit_1b = _solve_1b_model(
        X_rad, X_phi, y_1b, mol_dmin, env_info["env_counts"],
        n_chan=n_chan, n_env=n_env, n_rad=n_rad,
        active_q_modes=active_q_modes, seed=distill_seed + 11,
    )

    # 2. Pore-confined 2-body sampling (reusing evaluated 1-body poses + supercell lattice translations)
    low_idx = np.where(y_1b <= (15.0 / K_TO_KJMOL))[0]
    if len(low_idx) < 20:
        low_idx = np.argsort(y_1b)[:min(80, len(y_1b))]

    H_u = env_info["H_unit"]
    sc_shifts = []
    for ia in range(int(supercell[0])):
        for ib in range(int(supercell[1])):
            for ic in range(int(supercell[2])):
                sc_shifts.append(np.array([ia, ib, ic], dtype=np.float64) @ H_u)
    sc_shifts = np.asarray(sc_shifts, dtype=np.float64)

    dimer_pairs = []
    u_lj0_list, u_q0_list, rcom_list, e1_sum_list = [], [], [], []
    max_attempts = max(int(n_2b) * 40, 4000)
    for _ in range(max_attempts):
        if len(dimer_pairs) >= int(n_2b):
            break
        ia = int(rng.choice(low_idx))
        ib = int(rng.choice(low_idx))
        st_a = states_1b_kept[ia]
        st_b0 = states_1b_kept[ib]
        sh_b = sc_shifts[int(rng.integers(0, len(sc_shifts)))]
        cb_pos = np.array([st_b0["cx"][0] + sh_b[0], st_b0["cy"][0] + sh_b[1], st_b0["cz"][0] + sh_b[2]])
        cb_pos = ((cb_pos @ inv_H_super) % 1.0) @ H_super
        st_b = {
            "N": 1,
            "cx": np.array([float(cb_pos[0])]), "cy": np.array([float(cb_pos[1])]), "cz": np.array([float(cb_pos[2])]),
            "qw": st_b0["qw"], "qx": st_b0["qx"], "qy": st_b0["qy"], "qz": st_b0["qz"],
        }
        ulj0, uq0, rc_com, ov = _unscaled_gg_pair_energy_K(
            st_a, st_b, H_super, inv_H_super, dist_guest, r_cut_gg=r_cut_gg, dsf_alpha=0.20
        )
        if ov or rc_com < 2.60 or rc_com >= r_cut_gg - 0.1:
            continue
        # Ensure min intermolecular real-atom distance >= 2.30 A
        qa = np.array([[st_a["qw"][0], st_a["qx"][0], st_a["qy"][0], st_a["qz"][0]]])
        qb = np.array([[st_b["qw"][0], st_b["qx"][0], st_b["qy"][0], st_b["qz"][0]]])
        pa = np.array([st_a["cx"][0], st_a["cy"][0], st_a["cz"][0]])[None, :] + _quat_rotate_np(qa, local_real)[0]
        pb = np.array([st_b["cx"][0], st_b["cy"][0], st_b["cz"][0]])[None, :] + _quat_rotate_np(qb, local_real)[0]
        df_gg = (pa[:, None, :] - pb[None, :, :]) @ inv_H_super
        df_gg -= np.round(df_gg)
        dr_gg = df_gg @ H_super
        dmin_gg = float(math.sqrt(np.min(np.sum(dr_gg * dr_gg, axis=-1))))
        if dmin_gg < 2.30:
            continue
        st_ab = {
            "N": 2,
            "cx": np.array([st_a["cx"][0], st_b["cx"][0]], dtype=np.float64),
            "cy": np.array([st_a["cy"][0], st_b["cy"][0]], dtype=np.float64),
            "cz": np.array([st_a["cz"][0], st_b["cz"][0]], dtype=np.float64),
            "qw": np.array([st_a["qw"][0], st_b["qw"][0]], dtype=np.float64),
            "qx": np.array([st_a["qx"][0], st_b["qx"][0]], dtype=np.float64),
            "qy": np.array([st_a["qy"][0], st_b["qy"][0]], dtype=np.float64),
            "qz": np.array([st_a["qz"][0], st_b["qz"][0]], dtype=np.float64),
        }
        dimer_pairs.append(st_ab)
        u_lj0_list.append(ulj0)
        u_q0_list.append(uq0)
        rcom_list.append(rc_com)
        e1_sum_list.append(float(y_1b[ia] + y_1b[ib]))

    t_mace2_0 = _t.perf_counter()
    ev_2b = target.eval_batch(dimer_pairs) if dimer_pairs else []
    t_mace2_s = float(_t.perf_counter() - t_mace2_0)

    if dimer_pairs:
        u2_true = np.array([float(ev[0]) - e1s for ev, e1s in zip(ev_2b, e1_sum_list)], dtype=np.float64)
        u_lj0_arr = np.asarray(u_lj0_list, dtype=np.float64)
        u_q0_arr = np.asarray(u_q0_list, dtype=np.float64)
        rcom_arr = np.asarray(rcom_list, dtype=np.float64)
        valid_2b = np.isfinite(u2_true) & (u2_true < 1.5e4)
        fit_2b = _solve_2b_model(
            u_lj0_arr[valid_2b], u_q0_arr[valid_2b], rcom_arr[valid_2b], u2_true[valid_2b], r_cut_gg=r_cut_gg
        )
    else:
        fit_2b = _solve_2b_model(
            np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), r_cut_gg=r_cut_gg
        )

    # 3. Synthesize 3D unit-cell grids on GPU
    t_grid0 = _t.perf_counter()
    grids_u_lj = _render_unit_cell_grids(
        grid_dims=tuple(grid_dims),
        env_info=env_info,
        dist_guest=dist_guest,
        fit_1b=fit_1b,
        mu_centers=mu_centers,
        sigma_rbf=sigma_rbf,
        r_cut=r_cut_1b,
        device=torch_dev,
    )
    t_grid_s = float(_t.perf_counter() - t_grid0)
    total_wall_s = float(_t.perf_counter() - t0)

    meta = {
        "version": DISTILL_VERSION,
        "distill_sha256": cache_key,
        "cif_path": str(cif_path),
        "supercell": [int(x) for x in supercell],
        "guest_name": dist_guest.name,
        "model": str(model_name),
        "dispersion": bool(dispersion),
        "dispersion_xc": str(dispersion_xc),
        "dtype": str(dtype),
        "grid_dims": [int(x) for x in grid_dims],
        "n_chan": int(n_chan),
        "n_fw_envs": int(n_env),
        "fw_env_sigs": [[s[0], list(s[1])] for s in env_info["unique_sigs"]],
        "q_env_eff": [float(q) for q in fit_1b["q_env"]],
        "n_1b_evaluated": int(len(sel_1b)),
        "n_1b_kept": int(len(y_1b)),
        "n_2b_evaluated": int(len(dimer_pairs)),
        "r2_1b": fit_1b["r2_1b"],
        "mae_1b_kJ_mol": fit_1b["mae_1b_kJ_mol"],
        "mae_pocket_kJ_mol": fit_1b["mae_pocket_kJ_mol"],
        "r2_2b": fit_2b["r2_2b"],
        "mae_2b_kJ_mol": fit_2b["mae_2b_kJ_mol"],
        "mean_offset_classical_gg_kJ_mol": fit_2b["mean_offset_classical_gg_kJ_mol"],
        "mean_offset_distilled_2b_kJ_mol": fit_2b["mean_offset_distilled_2b_kJ_mol"],
        "gg_q_scale": fit_2b["gg_q_scale"],
        "gg_lj_scale": fit_2b["gg_lj_scale"],
        "gg_lut_rmin_A": fit_2b["r_min_A"],
        "gg_lut_rmax_A": fit_2b["r_max_A"],
        "t_mace1_s": t_mace1_s,
        "t_mace2_s": t_mace2_s,
        "t_grid_render_s": t_grid_s,
        "distill_wall_s": total_wall_s,
    }

    tmp_npz = p_npz.with_suffix(".tmp.npz")
    with tmp_npz.open("wb") as f:
        np.savez(f, grids_u_lj=grids_u_lj, gg_com_lut_K=fit_2b["gg_com_lut_K"])
    os.replace(tmp_npz, p_npz)

    tmp_json = p_json.with_suffix(".tmp.json")
    tmp_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(tmp_json, p_json)

    print(
        f"  [Auto-MLIP Distill DONE in {total_wall_s:.2f}s] envs={n_env}, chans={n_chan} | "
        f"1B: R2={meta['r2_1b']:.4f}, MAE={meta['mae_1b_kJ_mol']:.2f} (pocket {meta['mae_pocket_kJ_mol']:.2f}) kJ/mol | "
        f"2B: q_scale={meta['gg_q_scale']:.3f}, lj_scale={meta['gg_lj_scale']:.3f}, "
        f"MAE={meta['mae_2b_kJ_mol']:.2f} kJ/mol | saved {p_npz}"
    )

    return {
        "cache_hit": False,
        "cache_npz_path": str(p_npz),
        "cache_json_path": str(p_json),
        "distill_sha256": cache_key,
        "grids_u_lj": grids_u_lj,
        "gg_com_lut_K": fit_2b["gg_com_lut_K"],
        "gg_q_scale": fit_2b["gg_q_scale"],
        "gg_lj_scale": fit_2b["gg_lj_scale"],
        "gg_lut_rmin_A": fit_2b["r_min_A"],
        "gg_lut_rmax_A": fit_2b["r_max_A"],
        "distill_wall_s": total_wall_s,
        "summary": meta,
    }
