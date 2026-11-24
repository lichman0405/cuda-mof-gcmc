"""Periodic QEq partial charge assignment & ASE CIF framework loader."""

import math
from pathlib import Path
import numpy as np

# Coulomb constant in eV * Angstrom / e^2
KE_EV_A = 14.39964547842567

# Rappe & Goddard (1991) electronegativity chi^0 (eV) and atomic hardness J^0 = 2*Hard (eV/e)
# indexed by atomic number Z = 1..103 (0 is dummy).
QEQ_CHI_J = {
    1: (4.528, 13.8904),   2: (9.660, 29.8400),   3: (3.006, 4.7720),    4: (4.877, 8.8860),
    5: (5.110, 9.5000),    6: (5.343, 10.1260),   7: (6.899, 11.7600),   8: (8.741, 13.3640),
    9: (10.874, 14.9480),  10: (11.040, 21.1000), 11: (2.843, 4.5920),   12: (3.951, 7.3860),
    13: (4.060, 7.1800),   14: (4.168, 6.9740),   15: (5.463, 8.0000),   16: (6.928, 8.9720),
    17: (8.564, 9.8920),   18: (9.465, 12.7100),  19: (2.421, 3.8400),   20: (3.231, 5.7600),
    21: (3.395, 6.1600),   22: (3.470, 6.7600),   23: (3.650, 6.8200),   24: (3.415, 7.7300),
    25: (3.325, 8.2100),   26: (3.760, 8.2800),   27: (4.105, 8.3500),   28: (4.465, 8.4100),
    29: (4.200, 8.4400),   30: (5.106, 8.5700),   31: (3.641, 6.3200),   32: (4.051, 6.8760),
    33: (5.188, 7.6180),   34: (6.428, 8.2620),   35: (7.790, 8.8500),   36: (8.505, 11.4300),
    37: (2.331, 3.6920),   38: (3.024, 4.8800),   39: (3.830, 5.6200),   40: (3.400, 7.1000),
    41: (3.550, 6.7600),   42: (3.465, 7.5100),   43: (3.290, 7.9800),   44: (3.575, 8.0300),
    45: (3.975, 8.0100),   46: (4.320, 8.0000),   47: (4.436, 6.2680),   48: (5.034, 7.9140),
    49: (3.506, 5.7920),   50: (3.987, 6.2480),   51: (4.899, 6.6840),   52: (5.816, 7.0520),
    53: (6.822, 7.5240),   54: (7.595, 9.9500),   55: (2.183, 3.4220),   56: (2.814, 4.7920),
    57: (2.836, 5.4830),   58: (2.774, 5.3840),   59: (2.858, 5.1280),   60: (2.869, 5.2410),
    61: (2.881, 5.3460),   62: (2.912, 5.4390),   63: (2.878, 5.5750),   64: (3.167, 5.9490),
    65: (3.018, 5.6680),   66: (3.055, 5.7430),   67: (3.127, 5.7820),   68: (3.187, 5.8290),
    69: (3.251, 5.8658),   70: (3.289, 5.9300),   71: (2.963, 4.9258),   72: (3.700, 6.8000),
    73: (5.100, 5.7000),   74: (4.630, 6.6200),   75: (3.960, 7.8400),   76: (5.140, 7.2600),
    77: (5.000, 8.0000),   78: (4.790, 8.8600),   79: (4.894, 5.1720),   80: (6.270, 8.3200),
    81: (3.200, 5.8000),   82: (3.900, 7.0600),   83: (4.690, 7.4800),   84: (4.210, 8.4200),
    85: (4.750, 9.5000),   86: (5.370, 10.7400),  87: (2.000, 4.0000),   88: (2.843, 4.8680),
    89: (2.835, 5.6700),   90: (3.175, 5.8100),   91: (2.985, 5.8100),   92: (3.341, 5.7060),
    93: (3.549, 5.4340),   94: (3.243, 5.6380),   95: (2.990, 6.0070),   96: (2.832, 6.3790),
    97: (3.193, 6.0710),   98: (3.197, 6.2020),   99: (3.333, 6.1780),  100: (3.400, 6.2000),
    101: (3.470, 6.2200), 102: (3.475, 6.3500), 103: (3.500, 6.4000),
}

REPEAT_MGMOF74 = {
    "Mg1": +1.566, "Mof_Mg": +1.566,
    "O1": -0.915, "Mof_Oa": -0.915,
    "O2": -0.768, "Mof_Ob": -0.768,
    "O3": -0.907, "Mof_Oc": -0.907,
    "C1": +0.936, "Mof_Ca": +0.936,
    "C2": -0.325, "Mof_Cb": -0.325,
    "C3": +0.457, "Mof_Cc": +0.457,
    "C4": -0.235, "Mof_Cd": -0.235,
    "H1": +0.191, "Mof_H": +0.191,
}

MGMOF74_TYPE_ID = {
    "Mg1": 0, "Mof_Mg": 0,
    "O1": 1,  "Mof_Oa": 1,
    "O2": 2,  "Mof_Ob": 2,
    "O3": 3,  "Mof_Oc": 3,
    "C1": 4,  "Mof_Ca": 4,
    "C2": 5,  "Mof_Cb": 5,
    "C3": 6,  "Mof_Cc": 6,
    "C4": 7,  "Mof_Cd": 7,
    "H1": 8,  "Mof_H": 8,
}


def compute_periodic_qeq_charges(uc_atoms, max_abs_q: float = 3.5, eps_eff: float = 1.50) -> np.ndarray:
    """Fast periodic Ewald Gaussian-screened QEq charge equilibration (net charge == 0)."""
    from scipy.special import erfc

    n = len(uc_atoms)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)

    Zs = uc_atoms.get_atomic_numbers()
    pos = np.array(uc_atoms.get_positions(), dtype=np.float64)
    cell = np.array(uc_atoms.get_cell(), dtype=np.float64)  # rows A, B, C
    vol = abs(float(np.linalg.det(cell)))
    ke = KE_EV_A / float(eps_eff)

    chi = np.array([QEQ_CHI_J.get(int(z), (4.5, 7.0))[0] for z in Zs], dtype=np.float64)
    J_diag = np.array([QEQ_CHI_J.get(int(z), (4.5, 7.0))[1] for z in Zs], dtype=np.float64)

    # Gaussian exponent zeta_i matching point self-energy J_i^0 = k_e * sqrt(2*zeta_i/pi)
    zeta = (0.5 * math.pi) * (J_diag / ke) ** 2
    gamma_mat = np.sqrt((zeta[:, None] * zeta[None, :]) / (zeta[:, None] + zeta[None, :]))

    alpha = 0.25
    r_cut = 14.0
    k_max = 1.8

    inv_cell_T = np.linalg.inv(cell).T  # rows are reciprocal/2pi vectors
    perp_w = 1.0 / np.linalg.norm(inv_cell_T, axis=1)
    n_rep = [int(math.ceil(r_cut / w)) + 1 for w in perp_w]
    shifts_f = np.stack(
        np.meshgrid(
            np.arange(-n_rep[0], n_rep[0] + 1),
            np.arange(-n_rep[1], n_rep[1] + 1),
            np.arange(-n_rep[2], n_rep[2] + 1),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    shift_vecs = shifts_f @ cell

    # Fractional wrapped differences r_j - r_i so all cells near primary cell are covered
    frac = uc_atoms.get_scaled_positions() % 1.0
    df0 = frac[None, :, :] - frac[:, None, :]
    df0 -= np.round(df0)
    dr0 = df0 @ cell  # (N, N, 3)

    J_mat = np.zeros((n, n), dtype=np.float64)
    rc2 = r_cut * r_cut

    # 1. Real-space Ewald + Gaussian screening
    for sv in shift_vecs:
        dr = dr0 + sv.reshape(1, 1, 3)
        r2 = np.sum(dr * dr, axis=-1)
        mask = (r2 > 1.0e-10) & (r2 < rc2)
        if not np.any(mask):
            continue
        r = np.sqrt(r2[mask])
        ii, jj = np.nonzero(mask)
        g_ij = gamma_mat[ii, jj]
        val = ke * (erfc(alpha * r) - erfc(g_ij * r)) / r
        np.add.at(J_mat, (ii, jj), val)

    # 2. Diagonal self-energy + Ewald self subtraction
    np.fill_diagonal(J_mat, J_mat.diagonal() + J_diag - (2.0 * ke * alpha / math.sqrt(math.pi)))

    # 3. Reciprocal-space Ewald (half-sphere symmetry)
    b_vecs = 2.0 * math.pi * inv_cell_T
    nb = np.linalg.norm(b_vecs, axis=1)
    m_max = [int(math.ceil(k_max / w)) + 1 for w in nb]
    k_list = []
    for mc in range(0, m_max[2] + 1):
        mb_min = 0 if mc == 0 else -m_max[1]
        for mb in range(mb_min, m_max[1] + 1):
            ma_min = 1 if (mc == 0 and mb == 0) else -m_max[0]
            for ma in range(ma_min, m_max[0] + 1):
                kv = ma * b_vecs[0] + mb * b_vecs[1] + mc * b_vecs[2]
                k2 = float(np.dot(kv, kv))
                if 1e-12 < k2 <= k_max * k_max:
                    w = 2.0 * ke * (4.0 * math.pi / vol) * math.exp(-k2 / (4.0 * alpha * alpha)) / k2
                    k_list.append((kv, w))

    if k_list:
        pos_wrap = frac @ cell
        k_arr = np.array([x[0] for x in k_list], dtype=np.float64)  # (K, 3)
        w_arr = np.array([x[1] for x in k_list], dtype=np.float64)  # (K,)
        kr = pos_wrap @ k_arr.T  # (N, K)
        ck = np.cos(kr)
        sk = np.sin(kr)
        # J_rec[i, j] = sum_k w_k * cos(k . (r_j - r_i)) = (ck * w) @ ck.T + (sk * w) @ sk.T
        J_mat += (ck * w_arr) @ ck.T + (sk * w_arr) @ sk.T

    # Solve KKT linear system: [J, 1; 1^T, 0] [q; lam] = [-chi; 0]
    A = np.zeros((n + 1, n + 1), dtype=np.float64)
    A[:n, :n] = 0.5 * (J_mat + J_mat.T)
    A[:n, n] = 1.0
    A[n, :n] = 1.0
    rhs = np.zeros(n + 1, dtype=np.float64)
    rhs[:n] = -chi

    sol = np.linalg.solve(A, rhs)
    q = sol[:n]
    q -= np.mean(q)
    qmax = float(np.max(np.abs(q)))
    if qmax > max_abs_q and qmax > 1e-12:
        q = q * (max_abs_q / qmax)
        q -= np.mean(q)
    return q


def _parse_cif_labels_and_charges(cif_path: Path, uc_atoms):
    if cif_path.suffix.lower() != ".cif":
        return None, None
    from ase.io.cif import parse_cif

    try:
        blk = next(parse_cif(str(cif_path)))
    except Exception:
        return None, None

    labels_asym = blk.get("_atom_site_label")
    charges_asym = blk.get("_atom_site_charge")
    kinds = uc_atoms.arrays.get("spacegroup_kinds")

    def _map_per_atom(vals):
        if vals is None:
            return None
        n_uc = len(uc_atoms)
        if kinds is not None and len(kinds) == n_uc and len(vals) > int(np.max(kinds)):
            return [vals[int(k)] for k in kinds]
        if len(vals) == n_uc:
            return list(vals)
        return None

    labels = _map_per_atom(labels_asym)
    charges = _map_per_atom(charges_asym)
    if charges is not None:
        charges = np.array([float(str(v).split("(")[0]) for v in charges], dtype=np.float64)
    return labels, charges


def extract_framework_for_proposer(cif_path: str, charge_mode: str = "auto") -> dict:
    """Read CIF/structure via ASE (expanding any space group) and resolve charges cleanly."""
    from ase.io import read as ase_read

    p = Path(cif_path)
    uc = ase_read(str(p))
    n_uc = len(uc)
    H_unit_cols = np.array(uc.get_cell(), dtype=np.float64).T
    frac = np.mod(np.array(uc.get_scaled_positions(), dtype=np.float64), 1.0)
    Z = np.array(uc.get_atomic_numbers(), dtype=np.int32)

    labels, cif_charges = _parse_cif_labels_and_charges(p, uc)
    if labels is not None:
        type_ids = np.array([MGMOF74_TYPE_ID.get(str(lb), -1) for lb in labels], dtype=np.int32)
    else:
        type_ids = np.full(n_uc, -1, dtype=np.int32)

    is_mg_mof74 = (
        labels is not None
        and np.any(Z == 12)
        and np.all(type_ids >= 0)
    )

    mode = charge_mode.strip().lower()
    if mode not in ("auto", "cif", "qeq", "zero"):
        raise ValueError(f"Unknown charge_mode '{charge_mode}'. Expected auto|cif|qeq|zero.")

    if mode == "zero":
        charges = np.zeros(n_uc, dtype=np.float64)
        source = "zero"
    elif mode == "qeq":
        charges = compute_periodic_qeq_charges(uc)
        source = "qeq"
    elif mode == "cif":
        if cif_charges is not None:
            charges = cif_charges.copy()
            source = "cif"
        elif is_mg_mof74:
            charges = np.array([REPEAT_MGMOF74[str(lb)] for lb in labels], dtype=np.float64)
            source = "mg_mof74_repeat"
        else:
            charges = np.zeros(n_uc, dtype=np.float64)
            source = "cif_missing_zero"
    else:  # auto
        has_nonzero_cif = (cif_charges is not None) and (float(np.max(np.abs(cif_charges))) > 1e-8)
        if has_nonzero_cif:
            charges = cif_charges.copy()
            source = "cif"
        elif is_mg_mof74:
            charges = np.array([REPEAT_MGMOF74[str(lb)] for lb in labels], dtype=np.float64)
            source = "mg_mof74_repeat"
        else:
            charges = compute_periodic_qeq_charges(uc)
            source = "auto_qeq"

    q_sum = float(np.sum(charges))
    if abs(q_sum) > 1e-6:
        if abs(q_sum) < 1e-2 and n_uc > 0:
            charges = charges - (q_sum / n_uc)
        elif mode in ("cif", "auto") and cif_charges is not None:
            raise ValueError(
                f"Framework '{cif_path}' has non-neutral CIF charges (sum={q_sum:.6f} e >= 1e-2 e)."
            )
        else:
            charges = charges - (q_sum / max(1, n_uc))

    return {
        "uc": uc,
        "H_unit_cols": np.ascontiguousarray(H_unit_cols, dtype=np.float64),
        "frac": np.ascontiguousarray(frac, dtype=np.float64),
        "Z": np.ascontiguousarray(Z, dtype=np.int32),
        "charges": np.ascontiguousarray(charges, dtype=np.float64),
        "type_ids": np.ascontiguousarray(type_ids, dtype=np.int32),
        "charge_source": source,
    }
