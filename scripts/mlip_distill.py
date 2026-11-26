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
DISTILL_VERSION = "v1.4"

UNIFIED_HARD_CORE_RULE = {
    "name": "unified_continuous_mic_v1",
    "fw_h_A": 1.20,
    "fw_heavy_A": 1.48,
    "gg_A": 0.50,
    "virtual_sites_excluded": True,
    "continuous_mic": True,
}

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
    is_metal = np.array([int(z) in METAL_ATOMIC_NUMBERS for z in Z], dtype=bool)
    both_metal = is_metal[:, None] & is_metal[None, :]

    for sv in shift_vecs:
        dr = dr0 + sv.reshape(1, 1, 3)
        r2 = np.sum(dr * dr, axis=-1)
        bonded = (r2 > 0.16) & (r2 < thresh2) & (~both_metal)
        ii, jj = np.nonzero(bonded)
        for i, j in zip(ii.tolist(), jj.tolist()):
            neighbors[i].append(int(Z[j]))

    sigs = [(int(Z[i]), tuple(sorted(neighbors[i]))) for i in range(n)]
    unique_sigs = sorted(set(sigs), key=lambda s: (s[0], len(s[1]), s[1]))
    sig_to_id = {s: idx for idx, s in enumerate(unique_sigs)}
    env_ids = np.array([sig_to_id[s] for s in sigs], dtype=np.int32)
    counts = np.bincount(env_ids, minlength=len(unique_sigs)).astype(np.float64)

    q_arr = np.zeros(n, dtype=np.float64) if charges is None else np.asarray(charges, dtype=np.float64)
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

    sh_chunk = 3
    sh_eff = min(sh_chunk, int(sh_t.shape[0]))
    # Cap GPU intermediate (eff_chunk, sh_eff, n_fw, n_rad) tensor to ~96 MB (24M floats)
    max_pts_mem = max(128, int(24_000_000 // max(1, int(fw_f_t.shape[0]) * sh_eff * n_rad)))
    eff_chunk = min(int(chunk_pts), max_pts_mem)

    with torch.no_grad():
        for s0 in range(0, M, eff_chunk):
            s1 = min(M, s0 + eff_chunk)
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
    env_sigs: Optional[list] = None,
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

    nonmetal_zs = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}
    is_metal_env = [
        (int(env_sigs[e][0]) not in nonmetal_zs) if (env_sigs is not None and e < len(env_sigs)) else True
        for e in range(n_env)
    ]

    # Bounds & gentle distance-aware Tikhonov regularization:
    # prevent multi-site collinear cancellation (e.g., tetrahedral CH4 C vs 4*H) without crushing OMS/dispersion tails
    lb = np.full(n_params, -1200.0, dtype=np.float64)
    ub = np.full(n_params, +2000.0, dtype=np.float64)
    reg_diag = np.full(n_params, 6.0, dtype=np.float64)
    for c in range(n_chan):
        for e in range(n_env):
            base = (c * n_env + e) * n_rad
            # k == 0: short-range repulsive core
            lb[base] = 25.0
            ub[base] = 5.0e4
            reg_diag[base] = 1.0e-4
            for k in range(1, n_rad):
                col = base + k
                if k <= 3:  # mu <= 2.22 A (inner wall transition)
                    lb[col] = -2500.0 if is_metal_env[e] else -1200.0
                    ub[col] = 6000.0
                    reg_diag[col] = 2.0
                elif k <= 5:  # mu = 2.46, 2.72 A (vdW / OMS pocket)
                    lb[col] = -2500.0 if is_metal_env[e] else -1200.0
                    ub[col] = 4000.0 if is_metal_env[e] else 2500.0
                    reg_diag[col] = 2.0
                else:  # mu >= 3.00 A (tail; moderate ridge prevents cross-site offset drift without crushing vdW)
                    lb[col] = -1000.0
                    ub[col] = 1500.0
                    reg_diag[col] = 6.0
    for m in range(n_q_modes):
        lb[n_sr + m] = -5.0
        ub[n_sr + m] = +5.0
        reg_diag[n_sr + m] = 2.0e2

    # Boltzmann-inspired weights: emphasize OMS pockets & binding minima, prevent high-wall outliers from skewing pocket
    y_kJ = y_K * K_TO_KJMOL
    w_sample = np.where(
        y_kJ <= -10.0,
        2.5,
        np.where(
            y_kJ <= 15.0,
            np.where(dmin <= 2.80, 1.5, 1.0),
            1.0 / (1.0 + np.maximum(0.0, y_kJ - 15.0) / 15.0),
        ),
    )

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


def _eval_deployed_2b_lut(
    r_com: np.ndarray,
    lut_K: np.ndarray,
    r_min_A: float,
    r_max_A: float,
) -> np.ndarray:
    """Evaluate deployed 2B COM residual LUT matching src/proposer_dll.cu:356-376 bit-for-bit."""
    r = np.asarray(r_com, dtype=np.float64)
    out = np.zeros_like(r, dtype=np.float64)
    lut_f32 = np.asarray(lut_K, dtype=np.float32)
    n_bins = int(lut_f32.size)
    rmin = float(r_min_A)
    rmax = float(r_max_A)
    if n_bins < 2 or rmax <= rmin:
        return out

    lut = lut_f32.astype(np.float64)
    inv_dr = float(n_bins - 1) / (rmax - rmin)

    active = r < rmax
    if not np.any(active):
        return out

    u_pos = (r[active] - rmin) * inv_dr
    vals = np.empty_like(u_pos, dtype=np.float64)

    low = u_pos <= 0.0
    if np.any(low):
        slope = lut[1] - lut[0]
        if slope > 0.0:
            slope = 0.0
        vals[low] = lut[0] + slope * u_pos[low]

    mid = ~low
    if np.any(mid):
        u_m = u_pos[mid]
        idx0 = np.floor(u_m).astype(np.int64)
        high = idx0 >= (n_bins - 1)
        vals_m = np.empty_like(u_m, dtype=np.float64)
        if np.any(high):
            vals_m[high] = lut[n_bins - 1]
        in_range = ~high
        if np.any(in_range):
            i0 = idx0[in_range]
            frac = u_m[in_range] - i0.astype(np.float64)
            v0 = lut[i0]
            v1 = lut[i0 + 1]
            vals_m[in_range] = v0 + frac * (v1 - v0)
        vals[mid] = vals_m

    out[active] = vals
    return out


def _solve_2b_model(
    u_lj0: np.ndarray,
    u_q0: np.ndarray,
    r_com: np.ndarray,
    u2_true_K: np.ndarray,
    r_cut_gg: float,
    seed: int = 20260001,
) -> dict:
    """Fit effective pore 2-body guest-guest scaling + smoothed COM radial residual LUT with honest hold-out validation."""
    from scipy.optimize import lsq_linear

    u_lj0 = np.asarray(u_lj0, dtype=np.float64)
    u_q0 = np.asarray(u_q0, dtype=np.float64)
    r_com = np.asarray(r_com, dtype=np.float64)
    u2_true_K = np.asarray(u2_true_K, dtype=np.float64)
    M = int(len(u2_true_K))

    r_min_A = 2.0
    r_max_A = float(r_cut_gg)
    n_lut = 128
    lut_r = np.linspace(r_min_A, r_max_A, n_lut, dtype=np.float64)

    mu_2b = np.linspace(3.10, max(3.60, r_max_A - 0.70), 7, dtype=np.float64)
    sig_2b = max(0.45, 0.7 * float(mu_2b[1] - mu_2b[0]))
    r_on = max(3.60, r_max_A - 1.5)
    inv_span = 1.0 / max(1e-6, r_max_A - r_on)

    def _rad_basis(r_arr: np.ndarray) -> np.ndarray:
        t_in = np.clip((r_arr - 2.75) / 0.45, 0.0, 1.0)
        f_in = 0.5 * (1.0 - np.cos(math.pi * t_in))
        f_in = np.where(r_arr <= 2.75, 0.0, f_in)
        t_tap = np.clip((r_arr - r_on) * inv_span, 0.0, 1.0)
        f_cut = 0.5 * (1.0 + np.cos(math.pi * t_tap))
        f_cut = np.where(r_arr >= r_max_A, 0.0, f_cut)
        diff = r_arr[:, None] - mu_2b[None, :]
        return np.exp(-(diff * diff) / (2.0 * sig_2b * sig_2b)) * (f_in * f_cut)[:, None]

    if M < 16 or float(np.max(np.abs(u2_true_K))) < 1e-6:
        return {
            "gg_q_scale": 1.0,
            "gg_lj_scale": 1.0,
            "r_min_A": float(r_min_A),
            "r_max_A": float(r_max_A),
            "gg_com_lut_K": np.zeros(n_lut, dtype=np.float32),
            "r2_2b": None,
            "mae_2b_kJ_mol": None,
            "r2_2b_deployed_holdout": None,
            "mae_2b_deployed_holdout_kJ_mol": None,
            "mae_2b_classical_holdout_kJ_mol": None,
            "r2_2b_train": None,
            "mae_2b_train_kJ_mol": None,
            "n_2b_train": 0,
            "n_2b_val": 0,
            "mean_offset_classical_gg_kJ_mol": None,
            "mean_offset_distilled_2b_kJ_mol": None,
        }

    B_2b = _rad_basis(r_com)
    B_lut = _rad_basis(lut_r)
    X = np.column_stack([u_lj0, u_q0, B_2b])  # shape (M, 2 + 7)
    n_p = X.shape[1]

    lb = np.full(n_p, -350.0, dtype=np.float64)
    ub = np.full(n_p, +350.0, dtype=np.float64)
    lb[0], ub[0] = 0.65, 1.50  # gg_lj_scale: preserve Pauli core repulsion while avoiding MP0+D3 clipping at 0.85
    lb[1], ub[1] = 0.50, 1.50  # gg_q_scale^2

    # Pull scaling parameters strongly toward 1.0 and radial corrections toward 0.0
    prior = np.zeros(n_p, dtype=np.float64)
    prior[0] = 1.0
    prior[1] = 1.0
    reg_w = np.full(n_p, 10.0, dtype=np.float64)
    reg_w[0] = 2.0e3
    reg_w[1] = 2.0e3

    y_kJ = u2_true_K * K_TO_KJMOL
    sw_all = np.sqrt(np.where(y_kJ <= 10.0, 1.0, 1.0 / (1.0 + np.maximum(0.0, y_kJ - 10.0) / 15.0)))

    def _fit_subset(idx_sub: np.ndarray) -> np.ndarray:
        sw = sw_all[idx_sub]
        A_w = X[idx_sub] * sw[:, None]
        b_w = u2_true_K[idx_sub] * sw
        A_aug = np.vstack([A_w, np.diag(np.sqrt(reg_w))])
        b_aug = np.concatenate([b_w, np.sqrt(reg_w) * prior])
        res = lsq_linear(A_aug, b_aug, bounds=(lb, ub), method="trf", lsmr_tol="auto")
        return res.x

    def _build_lut(w_rad: np.ndarray) -> np.ndarray:
        lut = np.clip((B_lut @ w_rad).astype(np.float64), -400.0, +400.0)
        lut[lut_r <= 2.75] = 0.0
        lut[-1] = 0.0
        return lut.astype(np.float32)

    def _eval_deployed(idx_sub: np.ndarray, lj_sc: float, q_sc: float, lut_f32: np.ndarray) -> np.ndarray:
        return (
            float(lj_sc) * u_lj0[idx_sub]
            + (float(q_sc) ** 2) * u_q0[idx_sub]
            + _eval_deployed_2b_lut(r_com[idx_sub], lut_f32, r_min_A, r_max_A)
        )

    # Final production refit on 100% of samples (matching _solve_1b_model)
    all_idx = np.arange(M, dtype=np.int64)
    theta_all = _fit_subset(all_idx)
    gg_lj_scale = float(theta_all[0])
    gg_q_scale = math.sqrt(max(0.0, float(theta_all[1])))
    lut_K = _build_lut(theta_all[2:])

    pred_all_K = _eval_deployed(all_idx, gg_lj_scale, gg_q_scale, lut_K)
    ss_res_all = float(np.sum((u2_true_K - pred_all_K) ** 2))
    ss_tot_all = float(np.sum((u2_true_K - np.mean(u2_true_K)) ** 2))
    r2_train = 1.0 - ss_res_all / max(1e-12, ss_tot_all)
    mae_train_kJ = float(np.mean(np.abs(u2_true_K - pred_all_K)) * K_TO_KJMOL)

    if M >= 24:
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(M)
        n_val = max(16, int(round(0.20 * M)))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        theta_cv = _fit_subset(tr_idx)
        lj_cv = float(theta_cv[0])
        q_cv = math.sqrt(max(0.0, float(theta_cv[1])))
        lut_K_cv = _build_lut(theta_cv[2:])

        pred_val_K = _eval_deployed(val_idx, lj_cv, q_cv, lut_K_cv)
        true_val_K = u2_true_K[val_idx]
        pred_classical_val_K = u_lj0[val_idx] + u_q0[val_idx]

        ss_res_val = float(np.sum((true_val_K - pred_val_K) ** 2))
        ss_tot_val = float(np.sum((true_val_K - np.mean(true_val_K)) ** 2))
        r2_val = 1.0 - ss_res_val / max(1e-12, ss_tot_val)
        mae_val_kJ = float(np.mean(np.abs(true_val_K - pred_val_K)) * K_TO_KJMOL)
        mae_classical_val_kJ = float(np.mean(np.abs(true_val_K - pred_classical_val_K)) * K_TO_KJMOL)
        offset_raw_kJ = float(np.mean(true_val_K - pred_classical_val_K) * K_TO_KJMOL)
        offset_fit_kJ = float(np.mean(true_val_K - pred_val_K) * K_TO_KJMOL)

        return {
            "gg_q_scale": float(gg_q_scale),
            "gg_lj_scale": float(gg_lj_scale),
            "r_min_A": float(r_min_A),
            "r_max_A": float(r_max_A),
            "gg_com_lut_K": lut_K,
            "r2_2b": float(r2_val),
            "mae_2b_kJ_mol": float(mae_val_kJ),
            "r2_2b_deployed_holdout": float(r2_val),
            "mae_2b_deployed_holdout_kJ_mol": float(mae_val_kJ),
            "mae_2b_classical_holdout_kJ_mol": float(mae_classical_val_kJ),
            "r2_2b_train": float(r2_train),
            "mae_2b_train_kJ_mol": float(mae_train_kJ),
            "n_2b_train": int(len(tr_idx)),
            "n_2b_val": int(n_val),
            "mean_offset_classical_gg_kJ_mol": float(offset_raw_kJ),
            "mean_offset_distilled_2b_kJ_mol": float(offset_fit_kJ),
        }

    # Fallback when 16 <= M < 24: insufficient samples for an honest >=16 hold-out split
    offset_raw_kJ = float(np.mean(u2_true_K - (u_lj0 + u_q0)) * K_TO_KJMOL)
    offset_fit_kJ = float(np.mean(u2_true_K - pred_all_K) * K_TO_KJMOL)
    return {
        "gg_q_scale": float(gg_q_scale),
        "gg_lj_scale": float(gg_lj_scale),
        "r_min_A": float(r_min_A),
        "r_max_A": float(r_max_A),
        "gg_com_lut_K": lut_K,
        "r2_2b": float(r2_train),
        "mae_2b_kJ_mol": float(mae_train_kJ),
        "r2_2b_deployed_holdout": None,
        "mae_2b_deployed_holdout_kJ_mol": None,
        "mae_2b_classical_holdout_kJ_mol": None,
        "r2_2b_train": float(r2_train),
        "mae_2b_train_kJ_mol": float(mae_train_kJ),
        "n_2b_train": int(M),
        "n_2b_val": 0,
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

    total = nx * ny * nz
    grids_flat = np.empty((n_chan, total), dtype=np.float32)
    # Process unit-cell grid voxels in 500k slices to bound host RAM (<250 MB even on 600-atom cells)
    slice_pts = 500_000
    w_sr_flat = [w_sr[c].ravel().astype(np.float64) for c in range(n_chan)]
    for g0 in range(0, total, slice_pts):
        g1 = min(total, g0 + slice_pts)
        rad_pts, phi_pts, dmin_pts = _eval_unit_cell_pts(
            cart_pts[g0:g1], env_info, mu_centers, sigma_rbf, r_cut, device=device
        )
        phi_term = phi_pts.astype(np.float64) @ q_env  # (slice,)
        rad_2d = rad_pts.reshape(g1 - g0, -1).astype(np.float64)
        for c in range(n_chan):
            val = rad_2d @ w_sr_flat[c] + chan_q[c] * phi_term
            # Continuous-coordinate hard-core rejection is enforced directly in C++/CUDA (include/hard_core.hpp);
            # keep all grid voxels finite and bounded so interpolation near legal walls never falsely rejects.
            grids_flat[c, g0:g1] = np.clip(val, -2.0e4, 8.0e4).astype(np.float32)

    return np.ascontiguousarray(grids_flat.reshape(n_chan, nz, ny, nx), dtype=np.float32)


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


def _hash_state_dict(state_dict: Any) -> Optional[str]:
    """Compute a fast deterministic SHA-256 checksum over a model state_dict."""
    if not (hasattr(state_dict, "keys") and hasattr(state_dict, "__getitem__")):
        return None
    keys = sorted(state_dict.keys())
    if not keys:
        return None
    h = hashlib.sha256()
    for name in keys:
        val = state_dict[name]
        h.update(str(name).encode("utf-8"))
        h.update(b"\x00")
        if hasattr(val, "detach") and hasattr(val, "cpu"):
            t = val.detach().cpu().contiguous()
            shape_str = str(tuple(int(d) for d in t.shape))
            dtype_str = str(t.dtype)
            h.update(shape_str.encode("ascii"))
            h.update(b":")
            h.update(dtype_str.encode("ascii"))
            h.update(b":")
            try:
                h.update(t.numpy().tobytes())
            except Exception:
                try:
                    import torch
                    if t.numel() > 0:
                        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
                except Exception:
                    h.update(bytes(t.untyped_storage()))
        elif isinstance(val, np.ndarray):
            arr = np.ascontiguousarray(val)
            h.update(f"{arr.shape}:{arr.dtype}:".encode("ascii"))
            h.update(arr.tobytes())
        else:
            h.update(repr(val).encode("utf-8"))
        h.update(b"\n")
    return f"state_dict:{h.hexdigest()}"


def _resolve_model_sha256(target, model_name: str) -> Optional[str]:
    """Resolve a deterministic SHA-256 identifier for checkpoint files or in-memory MLIP weights."""
    # 1) Direct local model file path
    if model_name:
        try:
            if Path(str(model_name)).is_file():
                return sha256_file(str(model_name))
        except OSError:
            pass

    if target is None:
        return None

    # 2) Explicit model_file_path or precomputed model_sha256 on target
    model_file_path = getattr(target, "model_file_path", None)
    if model_file_path:
        try:
            if Path(str(model_file_path)).is_file():
                return sha256_file(str(model_file_path))
        except OSError:
            pass

    explicit_sha = getattr(target, "model_sha256", None)
    if isinstance(explicit_sha, str) and explicit_sha.strip():
        return explicit_sha.strip()

    # 3) Inspect mace_calc.model_paths / model_path for cached checkpoint files
    mace_calc = getattr(target, "mace_calc", None)
    if mace_calc is not None:
        for attr in ("model_paths", "model_path"):
            raw_paths = getattr(mace_calc, attr, None)
            if not raw_paths:
                continue
            cand_list = list(raw_paths) if isinstance(raw_paths, (list, tuple)) else [raw_paths]
            existing_hashes = []
            for p in cand_list:
                if not p:
                    continue
                try:
                    if Path(str(p)).is_file():
                        d = sha256_file(str(p))
                        if d:
                            existing_hashes.append(d)
                except OSError:
                    continue
            if len(existing_hashes) == 1:
                return existing_hashes[0]
            if len(existing_hashes) > 1:
                return sha256_json_obj(existing_hashes)

    # 4) Deterministic SHA-256 over PyTorch nn.Module / model state_dict()
    candidates = []
    mace_model = getattr(target, "mace_model", None)
    if mace_model is not None:
        candidates.append(mace_model)
    if mace_calc is not None:
        calc_models = getattr(mace_calc, "models", None)
        if isinstance(calc_models, (list, tuple)):
            candidates.extend(m for m in calc_models if m is not None)
        elif calc_models is not None:
            candidates.append(calc_models)
    generic_model = getattr(target, "model", None)
    if generic_model is not None:
        candidates.append(generic_model)

    for mod in candidates:
        sd_fn = getattr(mod, "state_dict", None)
        if callable(sd_fn):
            try:
                sd = sd_fn()
                digest = _hash_state_dict(sd)
                if digest is not None:
                    return digest
            except Exception:
                continue

    return None


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
    only_1b: bool = False,
    model_sha256: Optional[str] = None,
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
    resolved_model_sha = (
        str(model_sha256).strip()
        if isinstance(model_sha256, str) and model_sha256.strip()
        else (sha256_file(model_file_path) if model_file_path else None)
    )
    key_obj = {
        "version": DISTILL_VERSION,
        "hard_core_rule": UNIFIED_HARD_CORE_RULE,
        "cif_sha256": sha256_file(cif_path) or Path(cif_path).name,
        "supercell": [int(x) for x in supercell],
        "guest_real_geom_sha256": sha256_json_obj(real_geom),
        "guest_sites_sha256": sha256_json_obj(sites_all),
        "grid_dims": [int(x) for x in grid_dims],
        "model": str(model_name),
        "model_sha256": resolved_model_sha,
        "dispersion": bool(dispersion),
        "dispersion_xc": str(dispersion_xc),
        "dtype": str(dtype),
        "n_1b": int(n_1b),
        "n_2b": 0 if only_1b else int(n_2b),
        "only_1b": bool(only_1b),
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
    only_1b: bool = False,
) -> dict:
    """Distill target MLIP (1B unit-cell grids + optional 2B effective pore pair LUT) on-the-fly with disk caching."""
    import time

    t0 = time.perf_counter()
    dist_guest = make_distilled_guest_spec(guest)
    key_model_file = str(model_name) if Path(str(model_name)).is_file() else None
    resolved_model_sha = _resolve_model_sha256(target, model_name)
    cache_key = compute_distill_cache_key(
        cif_path=cif_path,
        supercell=tuple(supercell),
        guest=dist_guest,
        grid_dims=tuple(grid_dims),
        model_name=model_name,
        model_file_path=key_model_file,
        dispersion=dispersion,
        dispersion_xc=dispersion_xc,
        dtype=dtype,
        n_1b=n_1b,
        n_2b=n_2b,
        distill_seed=distill_seed,
        only_1b=only_1b,
    )

    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    p_npz = cdir / f"distill_{cache_key[:16]}.npz"
    p_json = cdir / f"distill_{cache_key[:16]}.json"
    if p_npz.exists() and p_json.exists():
        meta_chk = json.loads(p_json.read_text(encoding="utf-8"))
        cached_msha = meta_chk.get("model_sha256")
        model_sha_ok = (
            resolved_model_sha is None
            or cached_msha is None
            or cached_msha == resolved_model_sha
        )
        if (
            meta_chk.get("distill_sha256") == cache_key
            and list(meta_chk.get("grid_dims", [])) == list(grid_dims)
            and bool(meta_chk.get("only_1b", False)) == bool(only_1b)
            and model_sha_ok
        ):
            out = load_distill_cache(str(p_npz))
            out["distill_wall_s"] = float(time.perf_counter() - t0)
            r2_2b_v = meta_chk.get("r2_2b")
            r2_2b_str = (
                "skipped(only_1b)"
                if meta_chk.get("only_1b")
                else ("N/A" if r2_2b_v is None else f"{float(r2_2b_v):.4f}")
            )
            print(f"  [Auto-MLIP Distill CACHE HIT] {p_npz} (R2_1b={meta_chk['r2_1b']:.4f}, R2_2b={r2_2b_str})")
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

    mu_centers = np.array(
        [1.75, 1.98, 2.22, 2.46, 2.72, 3.00, 3.32, 3.68, 4.08, 4.52, 5.00, 5.55, 6.15],
        dtype=np.float64,
    )
    sigma_rbf = 0.34
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
        env_sigs=env_info["unique_sigs"],
    )

    # 2. Pore-confined 2-body sampling (reusing evaluated 1-body poses + supercell lattice translations)
    if only_1b:
        dimer_pairs = []
        t_mace2_s = 0.0
        fit_2b = {
            "gg_q_scale": 1.0,
            "gg_lj_scale": 1.0,
            "r_min_A": 2.0,
            "r_max_A": float(r_cut_gg),
            "gg_com_lut_K": np.zeros(128, dtype=np.float32),
            "r2_2b": None,
            "mae_2b_kJ_mol": None,
            "r2_2b_deployed_holdout": None,
            "mae_2b_deployed_holdout_kJ_mol": None,
            "mae_2b_classical_holdout_kJ_mol": None,
            "r2_2b_train": None,
            "mae_2b_train_kJ_mol": None,
            "n_2b_train": 0,
            "n_2b_val": 0,
            "mean_offset_classical_gg_kJ_mol": None,
            "mean_offset_distilled_2b_kJ_mol": None,
        }
    else:
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
                u_lj0_arr[valid_2b],
                u_q0_arr[valid_2b],
                rcom_arr[valid_2b],
                u2_true[valid_2b],
                r_cut_gg=r_cut_gg,
                seed=int(distill_seed) + 23,
            )
        else:
            fit_2b = _solve_2b_model(
                np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0),
                r_cut_gg=r_cut_gg,
                seed=int(distill_seed) + 23,
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
        "hard_core_rule": dict(UNIFIED_HARD_CORE_RULE),
        "distill_sha256": cache_key,
        "only_1b": bool(only_1b),
        "cif_path": str(cif_path),
        "supercell": [int(x) for x in supercell],
        "guest_name": dist_guest.name,
        "model": str(model_name),
        "model_sha256": resolved_model_sha,
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
        "n_2b_train": int(fit_2b["n_2b_train"]),
        "n_2b_val": int(fit_2b["n_2b_val"]),
        "r2_1b": fit_1b["r2_1b"],
        "mae_1b_kJ_mol": fit_1b["mae_1b_kJ_mol"],
        "mae_pocket_kJ_mol": fit_1b["mae_pocket_kJ_mol"],
        "r2_2b": fit_2b["r2_2b"],
        "mae_2b_kJ_mol": fit_2b["mae_2b_kJ_mol"],
        "r2_2b_deployed_holdout": fit_2b["r2_2b_deployed_holdout"],
        "mae_2b_deployed_holdout_kJ_mol": fit_2b["mae_2b_deployed_holdout_kJ_mol"],
        "mae_2b_classical_holdout_kJ_mol": fit_2b["mae_2b_classical_holdout_kJ_mol"],
        "r2_2b_train": fit_2b["r2_2b_train"],
        "mae_2b_train_kJ_mol": fit_2b["mae_2b_train_kJ_mol"],
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

    if only_1b:
        msg_2b = "2B: skipped (only_1b=True)"
    elif meta["mae_2b_kJ_mol"] is not None:
        tag_2b = f"val={meta['n_2b_val']}" if meta["n_2b_val"] > 0 else f"insample={meta['n_2b_train']}"
        msg_2b = (
            f"2B({tag_2b}): q_scale={meta['gg_q_scale']:.3f}, lj_scale={meta['gg_lj_scale']:.3f}, "
            f"R2={meta['r2_2b']:.4f}, MAE={meta['mae_2b_kJ_mol']:.2f} kJ/mol"
        )
    else:
        msg_2b = f"2B: identity fallback (n_2b={meta['n_2b_evaluated']})"

    print(
        f"  [Auto-MLIP Distill DONE in {total_wall_s:.2f}s] envs={n_env}, chans={n_chan} | "
        f"1B: R2={meta['r2_1b']:.4f}, MAE={meta['mae_1b_kJ_mol']:.2f} (pocket {meta['mae_pocket_kJ_mol']:.2f}) kJ/mol | "
        f"{msg_2b} | saved {p_npz}"
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
