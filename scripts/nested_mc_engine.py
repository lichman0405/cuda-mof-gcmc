"""Hybrid GPU-GCMC + MLIP Nested Monte Carlo Engine (ctypes wrapper for mof_gcmc_proposer.dll)."""

import ctypes
import hashlib
import math
import os
import time
from pathlib import Path
from typing import Optional
import numpy as np


def _compute_model_sha256(model_file_path: Optional[str], torch_model=None):
    """Return (sha256_hex, source) from checkpoint file on disk or deterministic state_dict fallback."""
    if model_file_path and os.path.isfile(str(model_file_path)):
        h = hashlib.sha256()
        with open(str(model_file_path), "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest(), "file"
    if torch_model is not None and hasattr(torch_model, "state_dict"):
        h = hashlib.sha256()
        sd = torch_model.state_dict()
        for key in sorted(sd.keys()):
            tensor = sd[key]
            h.update(str(key).encode("utf-8"))
            if hasattr(tensor, "detach"):
                t_cpu = tensor.detach().cpu().contiguous()
                h.update(str(t_cpu.dtype).encode("utf-8"))
                h.update(str(tuple(t_cpu.shape)).encode("utf-8"))
                try:
                    h.update(t_cpu.numpy().tobytes())
                except Exception:
                    h.update(t_cpu.float().numpy().tobytes())
            else:
                arr = np.ascontiguousarray(tensor)
                h.update(str(arr.dtype).encode("utf-8"))
                h.update(str(arr.shape).encode("utf-8"))
                h.update(arr.tobytes())
        return h.hexdigest(), "state_dict"
    return None, None


def _mad_filtered_median(values, z_thresh: float = 3.5):
    """Return (robust_median, n_outliers) using MAD outlier rejection."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, 0
    if arr.size <= 2:
        return float(np.median(arr)), 0
    med = float(np.median(arr))
    abs_dev = np.abs(arr - med)
    mad = float(np.median(abs_dev))
    sigma = 1.4826 * mad
    if sigma > 1e-9:
        keep = abs_dev <= (float(z_thresh) * sigma)
        if np.any(keep):
            n_out = int(arr.size - int(np.sum(keep)))
            return float(np.median(arr[keep])), n_out
    # MAD == 0 (>= 50% tied): median itself is immune to <50% arbitrary outliers
    q75, q25 = np.percentile(arr, [75.0, 25.0])
    iqr_scale = max(1e-6, 1.5 * float(q75 - q25))
    n_out = int(np.sum(abs_dev > max(5.0, float(z_thresh) * iqr_scale)))
    return med, n_out


def _compute_adapt_end(eq_cycles: int, adapt_frac: float) -> int:
    """Compute equilibration adaptation stop cycle (strictly <= eq_cycles // 2 when eq_cycles >= 40)."""
    eq_n = int(eq_cycles)
    frac = float(adapt_frac)
    if eq_n <= 0 or frac <= 0.0:
        return 0
    end = int(eq_n * min(1.0, max(0.0, frac)))
    if eq_n >= 40:
        end = min(end, eq_n // 2)
    return max(0, min(eq_n, end))

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
K_TO_KJMOL = 8.31446261815324e-3
EV_TO_K = 11604.51812155008
KB_SI = 1.380649e-23


def _rand_uniform_quat(rng: np.random.Generator):
    """Shoemake (1992) uniform unit quaternion on SO(3)."""
    u1, u2, u3 = rng.random(3)
    u2 *= 2.0 * math.pi
    u3 *= 2.0 * math.pi
    s1 = math.sqrt(max(0.0, 1.0 - u1))
    s2 = math.sqrt(max(0.0, u1))
    return (
        s2 * math.cos(u3),
        s1 * math.sin(u2),
        s1 * math.cos(u2),
        s2 * math.sin(u3),
    )


def _rot_quat_small(rng: np.random.Generator, max_angle: float, qw: float, qx: float, qy: float, qz: float):
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


def _register_dll_dirs():
    if os.name != "nt":
        return
    dirs = [str(BUILD)]
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path:
        dirs.append(str(Path(cuda_path) / "bin"))
    for path_item in os.environ.get("PATH", "").split(os.pathsep):
        if "cuda" in path_item.lower() and os.path.isdir(path_item):
            dirs.append(path_item)
    for d in dict.fromkeys(dirs):
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass


def quat_rotate_batch(qw, qx, qy, qz, local_xyz: np.ndarray) -> np.ndarray:
    """Rotate local_xyz (M, 3) by N unit quaternions -> (N, M, 3)."""
    if len(qw) == 0:
        return np.zeros((0, local_xyz.shape[0], 3), dtype=np.float64)
    qxyz = np.stack([qx, qy, qz], axis=1)[:, None, :]  # (N, 1, 3)
    v = local_xyz[None, :, :]  # (1, M, 3)
    t = 2.0 * np.cross(qxyz, v)
    return v + np.asarray(qw, dtype=np.float64)[:, None, None] * t + np.cross(qxyz, t)


def _find_proposer_lib() -> Path:
    if os.name == "nt":
        lib_path = BUILD / "mof_gcmc_proposer.dll"
        if not lib_path.exists():
            raise FileNotFoundError(f"Missing {lib_path}. Run `cmd /c build.bat proposer` first.")
        return lib_path
    for cand in (
        BUILD / "libmof_gcmc_proposer.so",
        BUILD / "mof_gcmc_proposer.so",
        BUILD / "libmof_gcmc_proposer.dylib",
    ):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"Missing proposer shared library in {BUILD}. Run `./build.sh proposer` first."
    )


_CACHED_PROPOSER_LIB = None


def get_proposer_cdll():
    """Load and bind stateless host hard-core C ABI functions from the proposer shared library."""
    global _CACHED_PROPOSER_LIB
    if _CACHED_PROPOSER_LIB is not None:
        return _CACHED_PROPOSER_LIB
    _register_dll_dirs()
    dll_path = _find_proposer_lib()
    lib = ctypes.CDLL(str(dll_path))
    lib.proposer_check_hard_core_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ]
    lib.proposer_check_hard_core_cpu.restype = ctypes.c_int
    _CACHED_PROPOSER_LIB = lib
    return lib


def check_hard_core_state_cpp(
    H_unit_cols: np.ndarray,
    supercell,
    fw_unit_frac: np.ndarray,
    guest,
    state: dict,
    max_mol: int = 128,
    r_fw_h_A: float = 1.20,
    r_fw_heavy_A: float = 1.48,
    r_gg_A: float = 0.50,
) -> bool:
    """Evaluate exact C++ continuous-coordinate periodic MIC hard-core overlap predicate."""
    lib = get_proposer_cdll()
    h9 = np.ascontiguousarray(H_unit_cols, dtype=np.float64).reshape(9)
    sc_a, sc_b, sc_c = int(supercell[0]), int(supercell[1]), int(supercell[2])
    frac_arr = np.ascontiguousarray(fw_unit_frac, dtype=np.float64).reshape(-1, 3)
    n_fw = int(frac_arr.shape[0])
    g = guest.to_c_arrays()
    n_sites = int(g["n_sites"])
    site_lxyz = np.ascontiguousarray(g["lxyz"], dtype=np.float64).reshape(-1)
    site_z = np.ascontiguousarray(
        g["Z"] if "Z" in g else [int(s.Z) for s in guest.sites],
        dtype=np.int32,
    ).reshape(-1)
    n_mol = int(state["N"])
    if n_mol < 0 or n_mol > int(max_mol):
        return True
    if n_mol == 0:
        return False

    cx = np.ascontiguousarray(state["cx"], dtype=np.float64)
    cy = np.ascontiguousarray(state["cy"], dtype=np.float64)
    cz = np.ascontiguousarray(state["cz"], dtype=np.float64)
    if "qw" in state:
        qw = np.ascontiguousarray(state["qw"], dtype=np.float64)
        qx = np.ascontiguousarray(state["qx"], dtype=np.float64)
        qy = np.ascontiguousarray(state["qy"], dtype=np.float64)
        qz = np.ascontiguousarray(state["qz"], dtype=np.float64)
    else:
        qw = np.ones(n_mol, dtype=np.float64)
        qx = np.zeros(n_mol, dtype=np.float64)
        qy = np.zeros(n_mol, dtype=np.float64)
        qz = np.zeros(n_mol, dtype=np.float64)

    fw_ptr = (
        frac_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        if n_fw > 0 else ctypes.POINTER(ctypes.c_double)()
    )
    rc = lib.proposer_check_hard_core_cpu(
        h9.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sc_a, sc_b, sc_c,
        fw_ptr, n_fw, n_sites,
        site_lxyz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        site_z.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        int(g["is_isotropic"]),
        n_mol, int(max_mol),
        cx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        cy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        cz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        qw.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        qx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        qy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        qz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        float(r_fw_h_A), float(r_fw_heavy_A), float(r_gg_A),
    )
    if rc < 0:
        raise RuntimeError(f"proposer_check_hard_core_cpu failed with code {rc}")
    return bool(rc != 0)


class CudaGcmcProposer:
    """Zero-I/O ctypes wrapper around mof_gcmc_proposer (.dll / .so)."""

    def __init__(
        self,
        cif_path: str,
        supercell=(1, 1, 2),
        T_K: float = 298.0,
        pressure_bar: float = 0.15,
        ff_mode: int = 1,
        pol_mode: int = 0,
        grid_dims=(256, 256, 72),
        n_boxes: int = 1,
        max_mol: int = 128,
        seed: int = 20261123,
    ):
        _register_dll_dirs()
        dll_path = _find_proposer_lib()
        self.lib = ctypes.CDLL(str(dll_path))
        self._bind()

        self.supercell = tuple(int(x) for x in supercell)
        self.n_boxes = int(n_boxes)
        self.max_mol = int(min(256, max(16, max_mol)))
        self.T_K = float(T_K)
        self.delta_eps_K = 0.0
        self.handle = self.lib.proposer_create(
            str(cif_path).encode("utf-8"),
            int(supercell[0]), int(supercell[1]), int(supercell[2]),
            float(T_K), float(pressure_bar),
            int(ff_mode), int(pol_mode),
            int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]),
            self.n_boxes, self.max_mol, int(seed),
        )
        if not self.handle:
            raise RuntimeError("proposer_create returned NULL")
        self._init_after_handle()

    @classmethod
    def from_atoms(
        cls,
        fw_data: dict,
        guest,
        supercell=(1, 1, 2),
        T_K: float = 298.0,
        pressure_bar: float = 0.15,
        ff_mode: int = 0,
        pol_mode: int = 0,
        grid_dims=(256, 256, 72),
        n_boxes: int = 1,
        max_mol: int = 128,
        seed: int = 20261123,
        device_id: int = 0,
        cbmc_k: int = 128,
        p_ins_del: float = 0.50,
    ):
        obj = cls.__new__(cls)
        _register_dll_dirs()
        dll_path = _find_proposer_lib()
        obj.lib = ctypes.CDLL(str(dll_path))
        obj._bind()
        rc = obj.lib.proposer_set_device(int(device_id))
        if rc != 0:
            raise RuntimeError(f"proposer_set_device({device_id}) failed with code {rc}")

        obj.supercell = tuple(int(x) for x in supercell)
        obj.n_boxes = int(n_boxes)
        obj.max_mol = int(min(256, max(16, max_mol)))
        obj.T_K = float(T_K)
        obj.delta_eps_K = 0.0

        H_u = np.ascontiguousarray(fw_data["H_unit_cols"], dtype=np.float64).reshape(9)
        frac = np.ascontiguousarray(fw_data["frac"], dtype=np.float64).reshape(-1)
        Z_arr = np.ascontiguousarray(fw_data["Z"], dtype=np.int32).reshape(-1)
        q_arr = np.ascontiguousarray(fw_data["charges"], dtype=np.float64).reshape(-1)
        tid_arr = np.ascontiguousarray(fw_data["type_ids"], dtype=np.int32).reshape(-1)
        n_fw = int(len(Z_arr))

        g = guest.to_c_arrays()
        obj.handle = obj.lib.proposer_create_from_data(
            H_u.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            int(supercell[0]), int(supercell[1]), int(supercell[2]),
            n_fw,
            frac.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Z_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            q_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            tid_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            int(g["n_sites"]),
            g["lxyz"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            g["q"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            g["eps"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            g["sig"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            g["lj_type"].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            float(g["Tc_K"]),
            float(g["Pc_bar"]),
            float(g["omega"]),
            int(g["is_isotropic"]),
            float(T_K), float(pressure_bar),
            int(ff_mode), int(pol_mode),
            int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]),
            obj.n_boxes, obj.max_mol, int(seed),
        )
        if not obj.handle:
            raise RuntimeError("proposer_create_from_data returned NULL")
        obj._init_after_handle()
        obj.set_mc_params(cbmc_k=int(cbmc_k), p_ins_del_total=float(p_ins_del))
        return obj

    @classmethod
    def from_distilled(
        cls,
        fw_data: dict,
        guest,
        distill_res: dict,
        supercell=(1, 1, 2),
        T_K: float = 298.0,
        pressure_bar: float = 0.15,
        n_boxes: int = 1,
        max_mol: int = 128,
        seed: int = 20261123,
        device_id: int = 0,
        cbmc_k: int = 128,
        p_ins_del: float = 0.50,
    ):
        grids = np.ascontiguousarray(distill_res["grids_u_lj"], dtype=np.float32)
        if grids.ndim != 4:
            raise ValueError(f"Expected 4D grids (n_lj, nz, ny, nx), got shape {grids.shape}")
        grid_dims = (int(grids.shape[3]), int(grids.shape[2]), int(grids.shape[1]))
        obj = cls.from_atoms(
            fw_data=fw_data,
            guest=guest,
            supercell=tuple(supercell),
            T_K=float(T_K),
            pressure_bar=float(pressure_bar),
            ff_mode=2,
            pol_mode=0,
            grid_dims=grid_dims,
            n_boxes=int(n_boxes),
            max_mol=int(max_mol),
            seed=int(seed),
            device_id=int(device_id),
            cbmc_k=int(cbmc_k),
            p_ins_del=float(p_ins_del),
        )
        obj.set_external_grids(grids)
        obj.set_gg_lut(
            gg_q_scale=float(distill_res.get("gg_q_scale", 1.0)),
            gg_lj_scale=float(distill_res.get("gg_lj_scale", 1.0)),
            r_min_A=float(distill_res.get("gg_lut_rmin_A", 2.0)),
            r_max_A=float(distill_res.get("gg_lut_rmax_A", 10.0)),
            gg_com_lut_K=distill_res.get("gg_com_lut_K"),
        )
        site_z = np.ascontiguousarray([int(s.Z) for s in guest.sites], dtype=np.int32)
        fw_frac = np.ascontiguousarray(np.mod(np.asarray(fw_data["frac"], dtype=np.float64), 1.0))
        obj.set_hard_core_config(
            site_z=site_z,
            fw_unit_frac=fw_frac,
            enable_continuous_hc=True,
            r_fw_h_A=1.20,
            r_fw_heavy_A=1.48,
            r_gg_A=0.50,
        )
        return obj

    def _init_after_handle(self):
        self.fw_mass_gmol = float(self.lib.proposer_get_framework_mass(self.handle))
        self.f_Pa = float(self.lib.proposer_get_fugacity_Pa(self.handle))
        self.r_cut_gg = float(self.lib.proposer_get_rcut_gg(self.handle))
        self.min_perp_width_A = float(self.lib.proposer_get_min_perp_width(self.handle))
        h9 = (ctypes.c_double * 9)()
        self.lib.proposer_get_cell_H(self.handle, h9)
        self.H_cols = np.array(list(h9), dtype=np.float64).reshape(3, 3)

        nb, nm = self.n_boxes, self.n_boxes * self.max_mol
        self._buf_N = np.zeros(nb, dtype=np.int32)
        self._buf_U = np.zeros(nb, dtype=np.float64)
        self._buf_dU = np.zeros(nb, dtype=np.float64)
        self._buf_acc = np.zeros(nb, dtype=np.int32)
        self._buf_cx = np.zeros(nm, dtype=np.float64)
        self._buf_cy = np.zeros(nm, dtype=np.float64)
        self._buf_cz = np.zeros(nm, dtype=np.float64)
        self._buf_ux = np.zeros(nm, dtype=np.float64)
        self._buf_uy = np.zeros(nm, dtype=np.float64)
        self._buf_uz = np.zeros(nm, dtype=np.float64)
        self._buf_qw = np.zeros(nm, dtype=np.float64)
        self._buf_qx = np.zeros(nm, dtype=np.float64)
        self._buf_qy = np.zeros(nm, dtype=np.float64)
        self._buf_qz = np.zeros(nm, dtype=np.float64)
        self._mask = np.zeros(nb, dtype=np.int32)

    def _bind(self):
        L = self.lib
        L.proposer_set_device.argtypes = [ctypes.c_int]
        L.proposer_set_device.restype = ctypes.c_int
        L.proposer_create.argtypes = [
            ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        L.proposer_create.restype = ctypes.c_void_p
        L.proposer_create_from_data.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        L.proposer_create_from_data.restype = ctypes.c_void_p
        L.proposer_destroy.argtypes = [ctypes.c_void_p]
        L.proposer_destroy.restype = None
        L.proposer_get_framework_mass.argtypes = [ctypes.c_void_p]
        L.proposer_get_framework_mass.restype = ctypes.c_double
        L.proposer_get_fugacity_Pa.argtypes = [ctypes.c_void_p]
        L.proposer_get_fugacity_Pa.restype = ctypes.c_double
        L.proposer_get_rcut_gg.argtypes = [ctypes.c_void_p]
        L.proposer_get_rcut_gg.restype = ctypes.c_double
        L.proposer_get_cell_H.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        L.proposer_get_cell_H.restype = None
        L.proposer_get_num_unit_atoms.argtypes = [ctypes.c_void_p]
        L.proposer_get_num_unit_atoms.restype = ctypes.c_int
        L.proposer_get_unit_atoms.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
        ]
        L.proposer_get_unit_atoms.restype = ctypes.c_int
        L.proposer_get_min_perp_width.argtypes = [ctypes.c_void_p]
        L.proposer_get_min_perp_width.restype = ctypes.c_double
        L.proposer_set_chem_pot_shift.argtypes = [ctypes.c_void_p, ctypes.c_double]
        L.proposer_set_chem_pot_shift.restype = ctypes.c_int
        L.proposer_equilibrate.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.proposer_equilibrate.restype = ctypes.c_int
        L.proposer_propose_k.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ]
        L.proposer_propose_k.restype = ctypes.c_int
        L.proposer_commit.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        L.proposer_commit.restype = ctypes.c_int
        L.proposer_get_accepted.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ]
        L.proposer_get_accepted.restype = ctypes.c_int
        L.proposer_eval_energy.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
        ]
        L.proposer_eval_energy.restype = ctypes.c_double
        L.proposer_propose_k_quat.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ]
        L.proposer_propose_k_quat.restype = ctypes.c_int
        L.proposer_get_accepted_quat.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ]
        L.proposer_get_accepted_quat.restype = ctypes.c_int
        L.proposer_set_accepted_quat.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ]
        L.proposer_set_accepted_quat.restype = ctypes.c_int
        L.proposer_get_rng_states.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
        L.proposer_get_rng_states.restype = ctypes.c_int
        L.proposer_set_rng_states.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
        L.proposer_set_rng_states.restype = ctypes.c_int
        L.proposer_eval_energy_quat.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
        ]
        L.proposer_eval_energy_quat.restype = ctypes.c_double
        L.proposer_run_direct_benchmark.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        L.proposer_run_direct_benchmark.restype = ctypes.c_int
        L.proposer_set_external_grids.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ]
        L.proposer_set_external_grids.restype = ctypes.c_int
        L.proposer_set_gg_lut.argtypes = [
            ctypes.c_void_p, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        L.proposer_set_gg_lut.restype = ctypes.c_int
        L.proposer_set_mc_params.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
        L.proposer_set_mc_params.restype = ctypes.c_int
        L.proposer_set_hard_core_config.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ]
        L.proposer_set_hard_core_config.restype = ctypes.c_int

    def set_hard_core_config(
        self,
        site_z: Optional[np.ndarray] = None,
        fw_unit_frac: Optional[np.ndarray] = None,
        enable_continuous_hc: bool = True,
        r_fw_h_A: float = 1.20,
        r_fw_heavy_A: float = 1.48,
        r_gg_A: float = 0.50,
    ) -> None:
        z_ptr = ctypes.POINTER(ctypes.c_int)()
        if site_z is not None:
            z_arr = np.ascontiguousarray(site_z, dtype=np.int32).reshape(-1)
            z_ptr = z_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        fw_ptr = ctypes.POINTER(ctypes.c_double)()
        n_fw = 0
        if fw_unit_frac is not None:
            fw_arr = np.ascontiguousarray(fw_unit_frac, dtype=np.float64).reshape(-1, 3)
            n_fw = int(fw_arr.shape[0])
            if n_fw > 0:
                fw_ptr = fw_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        rc = self.lib.proposer_set_hard_core_config(
            self.handle,
            z_ptr,
            fw_ptr,
            n_fw,
            1 if enable_continuous_hc else 0,
            float(r_fw_h_A),
            float(r_fw_heavy_A),
            float(r_gg_A),
        )
        if rc != 0:
            raise RuntimeError(f"proposer_set_hard_core_config failed with code {rc}")

    def set_external_grids(self, grids_u_lj: np.ndarray, phi_elec: Optional[np.ndarray] = None) -> None:
        arr = np.ascontiguousarray(grids_u_lj, dtype=np.float32)
        if arr.ndim != 4:
            raise ValueError(f"grids_u_lj must have shape (n_lj, nz, ny, nx), got {arr.shape}")
        n_lj, nz, ny, nx = [int(x) for x in arr.shape]
        phi_ptr = ctypes.POINTER(ctypes.c_float)()
        if phi_elec is not None:
            phi_arr = np.ascontiguousarray(phi_elec, dtype=np.float32)
            if phi_arr.shape != (nz, ny, nx):
                raise ValueError(f"phi_elec shape {phi_arr.shape} != {(nz, ny, nx)}")
            phi_ptr = phi_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = self.lib.proposer_set_external_grids(
            self.handle, nx, ny, nz, n_lj,
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            phi_ptr,
        )
        if rc != 0:
            raise RuntimeError(f"proposer_set_external_grids failed with code {rc}")

    def set_gg_lut(
        self,
        gg_q_scale: float = 1.0,
        gg_lj_scale: float = 1.0,
        r_min_A: float = 2.0,
        r_max_A: float = 10.0,
        gg_com_lut_K: Optional[np.ndarray] = None,
    ) -> None:
        if gg_com_lut_K is not None and len(gg_com_lut_K) >= 2:
            lut_arr = np.ascontiguousarray(gg_com_lut_K, dtype=np.float32).reshape(-1)
            n_bins = int(len(lut_arr))
            lut_ptr = lut_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            n_bins = 0
            lut_ptr = ctypes.POINTER(ctypes.c_float)()
        rc = self.lib.proposer_set_gg_lut(
            self.handle, float(gg_q_scale), float(gg_lj_scale),
            float(r_min_A), float(r_max_A), n_bins, lut_ptr,
        )
        if rc != 0:
            raise RuntimeError(f"proposer_set_gg_lut failed with code {rc}")

    def set_mc_params(self, cbmc_k: int = 128, p_ins_del_total: float = 0.50) -> None:
        rc = self.lib.proposer_set_mc_params(self.handle, int(cbmc_k), float(p_ins_del_total))
        if rc != 0:
            raise RuntimeError(f"proposer_set_mc_params failed with code {rc}")
        self.cbmc_k = int(cbmc_k)
        self.p_ins_del = float(p_ins_del_total)

    def close(self):
        if getattr(self, "handle", None):
            self.lib.proposer_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()

    def set_chem_pot_shift(self, delta_eps_kJ_mol: float):
        """Set inner-proposer chemical-potential bias delta_eps = <(U_tgt - U_ref)/N> (kJ/mol)."""
        val_kJ = float(delta_eps_kJ_mol)
        if not math.isfinite(val_kJ):
            raise ValueError(f"delta_eps_kJ_mol must be finite, got {delta_eps_kJ_mol!r}")
        delta_eps_K = val_kJ / K_TO_KJMOL
        if self.T_K <= 0.0 or abs(delta_eps_K / self.T_K) > 40.0:
            raise ValueError(
                f"Chemical-potential shift |delta_eps_K / T_K|={abs(delta_eps_K / max(self.T_K, 1e-12)):.3f} "
                f"exceeds safety bound 40.0 (delta_eps_kJ_mol={val_kJ}, T_K={self.T_K})"
            )
        rc = self.lib.proposer_set_chem_pot_shift(self.handle, delta_eps_K)
        if rc != 0:
            raise RuntimeError(f"proposer_set_chem_pot_shift failed with code {rc}")
        self.delta_eps_K = delta_eps_K

    def get_rng_states(self) -> np.ndarray:
        buf = np.zeros(self.n_boxes, dtype=np.uint64)
        rc = self.lib.proposer_get_rng_states(
            self.handle, buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        )
        if rc != 0:
            raise RuntimeError(f"proposer_get_rng_states failed with code {rc}")
        return buf

    def set_rng_states(self, rng_states) -> None:
        arr = np.ascontiguousarray(rng_states, dtype=np.uint64).reshape(-1)
        if len(arr) != self.n_boxes:
            raise ValueError(f"Expected {self.n_boxes} RNG states, got {len(arr)}")
        rc = self.lib.proposer_set_rng_states(
            self.handle, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        )
        if rc != 0:
            raise RuntimeError(f"proposer_set_rng_states failed with code {rc}")

    def equilibrate(self, n_steps: int):
        rc = self.lib.proposer_equilibrate(self.handle, int(n_steps))
        if rc != 0:
            raise RuntimeError(f"proposer_equilibrate failed with code {rc}")

    def _unpack(self, include_dU: bool):
        boxes = []
        for b in range(self.n_boxes):
            n = int(self._buf_N[b])
            s = b * self.max_mol
            e = s + n
            boxes.append({
                "N": n,
                "U_ref_K": float(self._buf_U[b]),
                "dU_ref_K": float(self._buf_dU[b]) if include_dU else 0.0,
                "inner_acc": int(self._buf_acc[b]) if include_dU else 0,
                "cx": self._buf_cx[s:e].copy(),
                "cy": self._buf_cy[s:e].copy(),
                "cz": self._buf_cz[s:e].copy(),
                "ux": self._buf_ux[s:e].copy(),
                "uy": self._buf_uy[s:e].copy(),
                "uz": self._buf_uz[s:e].copy(),
            })
        return boxes

    def get_unit_atoms(self):
        n_unit = int(self.lib.proposer_get_num_unit_atoms(self.handle))
        if n_unit <= 0:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.int32)
        frac = np.zeros((n_unit, 3), dtype=np.float64)
        z_arr = np.zeros(n_unit, dtype=np.int32)
        rc = self.lib.proposer_get_unit_atoms(
            self.handle,
            frac.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            z_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
        if rc < 0:
            raise RuntimeError(f"proposer_get_unit_atoms failed with code {rc}")
        return frac, z_arr

    def get_accepted(self):
        rc = self.lib.proposer_get_accepted(
            self.handle,
            self._buf_N.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self._buf_U.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_cx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_cy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_cz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_ux.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_uy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_uz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc != 0:
            raise RuntimeError(f"proposer_get_accepted failed with code {rc}")
        return self._unpack(False)

    def propose_k(self, k_steps: int):
        rc = self.lib.proposer_propose_k(
            self.handle, int(k_steps),
            self._buf_N.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self._buf_U.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_dU.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_acc.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self._buf_cx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_cy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_cz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_ux.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_uy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_uz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc != 0:
            raise RuntimeError(f"proposer_propose_k failed with code {rc}")
        return self._unpack(True)

    def commit(self, accept_mask):
        self._mask[:] = np.asarray(accept_mask, dtype=np.int32)
        rc = self.lib.proposer_commit(self.handle, self._mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        if rc != 0:
            raise RuntimeError(f"proposer_commit failed with code {rc}")

    def eval_energy(self, state):
        n = int(state["N"])
        if n == 0:
            return 0.0, False
        if "qw" in state:
            return self.eval_energy_quat(state)
        cx = np.ascontiguousarray(state["cx"], dtype=np.float64)
        cy = np.ascontiguousarray(state["cy"], dtype=np.float64)
        cz = np.ascontiguousarray(state["cz"], dtype=np.float64)
        ux = np.ascontiguousarray(state["ux"], dtype=np.float64)
        uy = np.ascontiguousarray(state["uy"], dtype=np.float64)
        uz = np.ascontiguousarray(state["uz"], dtype=np.float64)
        ov = ctypes.c_int(0)
        u = self.lib.proposer_eval_energy(
            self.handle, n,
            cx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ux.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            uy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            uz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(ov),
        )
        return float(u), bool(ov.value != 0)

    def _unpack_quat(self, include_dU: bool):
        boxes = []
        for b in range(self.n_boxes):
            n = int(self._buf_N[b])
            s = b * self.max_mol
            e = s + n
            boxes.append({
                "N": n,
                "U_ref_K": float(self._buf_U[b]),
                "dU_ref_K": float(self._buf_dU[b]) if include_dU else 0.0,
                "inner_acc": int(self._buf_acc[b]) if include_dU else 0,
                "cx": self._buf_cx[s:e].copy(),
                "cy": self._buf_cy[s:e].copy(),
                "cz": self._buf_cz[s:e].copy(),
                "qw": self._buf_qw[s:e].copy(),
                "qx": self._buf_qx[s:e].copy(),
                "qy": self._buf_qy[s:e].copy(),
                "qz": self._buf_qz[s:e].copy(),
            })
        return boxes

    def get_accepted_quat(self):
        rc = self.lib.proposer_get_accepted_quat(
            self.handle,
            self._buf_N.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self._buf_U.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_cx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_cy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_cz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_qw.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_qx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_qy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_qz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc != 0:
            raise RuntimeError(f"proposer_get_accepted_quat failed with code {rc}")
        return self._unpack_quat(False)

    def set_accepted_quat(self, states):
        if len(states) != self.n_boxes:
            raise ValueError(f"Expected {self.n_boxes} box states, got {len(states)}")
        self._buf_cx.fill(0.0)
        self._buf_cy.fill(0.0)
        self._buf_cz.fill(0.0)
        self._buf_qw.fill(1.0)
        self._buf_qx.fill(0.0)
        self._buf_qy.fill(0.0)
        self._buf_qz.fill(0.0)
        for b, st in enumerate(states):
            n = int(st["N"])
            if n < 0 or n > self.max_mol:
                raise ValueError(f"Box {b} N={n} out of allowed [0, {self.max_mol}]")
            self._buf_N[b] = n
            if n > 0:
                s = b * self.max_mol
                e = s + n
                self._buf_cx[s:e] = np.asarray(st["cx"], dtype=np.float64)
                self._buf_cy[s:e] = np.asarray(st["cy"], dtype=np.float64)
                self._buf_cz[s:e] = np.asarray(st["cz"], dtype=np.float64)
                self._buf_qw[s:e] = np.asarray(st["qw"], dtype=np.float64)
                self._buf_qx[s:e] = np.asarray(st["qx"], dtype=np.float64)
                self._buf_qy[s:e] = np.asarray(st["qy"], dtype=np.float64)
                self._buf_qz[s:e] = np.asarray(st["qz"], dtype=np.float64)
        rc = self.lib.proposer_set_accepted_quat(
            self.handle,
            self._buf_N.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self._buf_cx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_cy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_cz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_qw.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_qx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_qy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_qz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc == -2:
            raise ValueError("proposer_set_accepted_quat rejected illegal hard-core state")
        if rc != 0:
            raise RuntimeError(f"proposer_set_accepted_quat failed with code {rc}")

    def propose_k_quat(self, k_steps: int):
        rc = self.lib.proposer_propose_k_quat(
            self.handle, int(k_steps),
            self._buf_N.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self._buf_U.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_dU.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_acc.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self._buf_cx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_cy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_cz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_qw.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_qx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_qy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._buf_qz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc != 0:
            raise RuntimeError(f"proposer_propose_k_quat failed with code {rc}")
        return self._unpack_quat(True)

    def eval_energy_quat(self, state):
        n = int(state["N"])
        if n == 0:
            return 0.0, False
        cx = np.ascontiguousarray(state["cx"], dtype=np.float64)
        cy = np.ascontiguousarray(state["cy"], dtype=np.float64)
        cz = np.ascontiguousarray(state["cz"], dtype=np.float64)
        qw = np.ascontiguousarray(state["qw"], dtype=np.float64)
        qx = np.ascontiguousarray(state["qx"], dtype=np.float64)
        qy = np.ascontiguousarray(state["qy"], dtype=np.float64)
        qz = np.ascontiguousarray(state["qz"], dtype=np.float64)
        ov = ctypes.c_int(0)
        u = self.lib.proposer_eval_energy_quat(
            self.handle, n,
            cx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            qw.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            qx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            qy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            qz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(ov),
        )
        return float(u), bool(ov.value != 0)

    def run_direct_benchmark(self, eq_steps=100000, prod_steps=200000, sample_stride=20):
        u_avg, u_sem = ctypes.c_double(), ctypes.c_double()
        q_avg, q_sem = ctypes.c_double(), ctypes.c_double()
        md = ctypes.c_double()
        rc = self.lib.proposer_run_direct_benchmark(
            self.handle, int(eq_steps), int(prod_steps), int(sample_stride),
            ctypes.byref(u_avg), ctypes.byref(u_sem),
            ctypes.byref(q_avg), ctypes.byref(q_sem), ctypes.byref(md),
        )
        if rc != 0:
            raise RuntimeError(f"proposer_run_direct_benchmark failed: {rc}")
        return {
            "uptake_mmol_g": float(u_avg.value),
            "uptake_sem_mmol_g": float(u_sem.value),
            "Qst_kJ_mol": float(q_avg.value),
            "Qst_sem_kJ_mol": float(q_sem.value),
            "max_drift_K": float(md.value),
        }


class MaceD3Target:
    """MACE-MP-0 + D3 outer target energy evaluator (returns adsorption U in Kelvin)."""

    def __init__(self, cif_path: str, supercell=(1, 1, 2), model="medium", device="cuda"):
        from ase import Atoms
        from ase.io import read as ase_read
        from mace.calculators import mace_mp

        self.Atoms = Atoms
        uc = ase_read(str(cif_path))
        self.uc = uc
        self.fw = uc.repeat(tuple(supercell))
        self.calc = mace_mp(model=model, dispersion=True, default_dtype="float32", device=device)

        fw_copy = self.fw.copy()
        fw_copy.calc = self.calc
        self.e_fw_eV = float(fw_copy.get_potential_energy())

        co2 = Atoms("CO2", positions=[[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]])
        co2.set_cell([30.0, 30.0, 30.0])
        co2.set_pbc(True)
        co2.calc = self.calc
        self.e_co2_eV = float(co2.get_potential_energy())
        self.n_evals = 0

    def __call__(self, state):
        n = int(state["N"])
        if n == 0:
            return 0.0, False
        bond = 1.160
        cx, cy, cz = state["cx"], state["cy"], state["cz"]
        if "qw" in state and "ux" not in state:
            qw = np.asarray(state["qw"], dtype=np.float64)
            qx = np.asarray(state["qx"], dtype=np.float64)
            qy = np.asarray(state["qy"], dtype=np.float64)
            qz = np.asarray(state["qz"], dtype=np.float64)
            ux = 1.0 - 2.0 * (qy * qy + qz * qz)
            uy = 2.0 * (qx * qy + qw * qz)
            uz = 2.0 * (qx * qz - qw * qy)
        else:
            ux, uy, uz = state["ux"], state["uy"], state["uz"]
        guest_pos = np.empty((3 * n, 3), dtype=np.float64)
        for i in range(n):
            c = np.array([cx[i], cy[i], cz[i]], dtype=np.float64)
            u = np.array([ux[i], uy[i], uz[i]], dtype=np.float64)
            guest_pos[3 * i + 0] = c
            guest_pos[3 * i + 1] = c + bond * u
            guest_pos[3 * i + 2] = c - bond * u

        sys = self.fw.copy() + self.Atoms("CO2" * n, positions=guest_pos)
        sys.calc = self.calc
        e_tot = float(sys.get_potential_energy())
        self.n_evals += 1
        u_int_K = (e_tot - self.e_fw_eV - n * self.e_co2_eV) * EV_TO_K
        return float(u_int_K), False


class FastMaceD3Target:
    """Fast batched energy-only MACE + exact GPU-edge DFT-D3(BJ) evaluator for any rigid guest."""

    def __init__(
        self,
        cif_path: str,
        supercell=(1, 1, 2),
        model="medium",
        device="cuda",
        dispersion: bool = True,
        dispersion_xc: str = "pbe",
        guest=None,
        mace_chunk: int = None,
        uc_atoms=None,
        dtype: str = "float32",
    ):
        import torch
        from ase import Atoms
        from ase.io import read as ase_read
        from mace.calculators import mace_mp
        from mace import data as mace_data
        from mace.tools import torch_geometric, torch_tools
        from mace.tools import utils as mace_utils
        from guests import load_guest

        dtype_norm = str(dtype).strip().lower()
        if dtype_norm not in ("float32", "float64"):
            raise ValueError(f"Unsupported FastMaceD3Target dtype={dtype!r}; expected 'float32' or 'float64'.")

        self.torch = torch
        self.Atoms = Atoms
        self.mace_data = mace_data
        self.torch_geometric = torch_geometric
        self.torch_tools = torch_tools
        self.device = torch.device(device)
        self.dtype = dtype_norm
        self.torch_dtype = torch.float64 if dtype_norm == "float64" else torch.float32
        self.np_dtype = np.float64 if dtype_norm == "float64" else np.float32
        self.use_dispersion = bool(dispersion)
        self.dispersion_xc = str(dispersion_xc)
        self.guest = load_guest("co2") if guest is None else guest
        self.guest_local_pos = self.guest.real_local_positions()
        self.guest_symbols_unit = [s.symbol for s in self.guest.real_sites()]
        self.guest_n_real = len(self.guest_symbols_unit)

        uc = ase_read(str(cif_path)) if uc_atoms is None else uc_atoms.copy()
        self.uc = uc
        self.fw = uc.repeat(tuple(supercell))
        self.n_fw = len(self.fw)

        if mace_chunk is not None:
            self.default_mace_chunk = max(1, int(mace_chunk))
        elif self.device.type == "cuda":
            tot_gb = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            if tot_gb >= 20.0 and self.n_fw < 700:
                self.default_mace_chunk = 2
            elif self.n_fw >= 550:
                self.default_mace_chunk = 1
            else:
                self.default_mace_chunk = 2
        else:
            self.default_mace_chunk = 1 if self.n_fw >= 550 else 2

        calc = mace_mp(
            model=model, dispersion=self.use_dispersion, dispersion_xc=self.dispersion_xc,
            default_dtype=self.dtype, device=device
        )
        if self.use_dispersion:
            self.mace_calc = calc.mixer.calcs[0]
            self.d3_calc = calc.mixer.calcs[1]
        else:
            self.mace_calc = calc
            self.d3_calc = None
        self.mace_model = self.mace_calc.models[0]
        self.mace_model.eval()
        self.r_max = float(self.mace_calc.r_max)
        self.model_file_path = None
        if os.path.isfile(str(model)):
            self.model_file_path = str(Path(str(model)).resolve())
        if self.model_file_path is None:
            for obj in (self.mace_calc, calc):
                for attr in ("model_paths", "model_path", "_model_paths", "models_path"):
                    val = getattr(obj, attr, None)
                    if val:
                        cand = str(val[0]) if isinstance(val, (list, tuple)) else str(val)
                        if os.path.isfile(cand):
                            self.model_file_path = str(Path(cand).resolve())
                            break
                if self.model_file_path is not None:
                    break
        if self.model_file_path is None:
            try:
                from mace.calculators.foundations_models import download_mace_mp_checkpoint
                cand = download_mace_mp_checkpoint(str(model))
                if cand and os.path.isfile(str(cand)):
                    self.model_file_path = str(Path(str(cand)).resolve())
            except Exception:
                pass
        self.model_sha256, self.model_sha256_source = _compute_model_sha256(
            self.model_file_path, self.mace_model
        )
        self.keyspec = mace_data.KeySpecification(
            info_keys=self.mace_calc.info_keys,
            arrays_keys=self.mace_calc.arrays_keys,
        )

        # Always precompute static rigid framework tensors & MACE fw-fw periodic graph once on device
        self.cell_np = np.array(self.fw.get_cell(), dtype=self.np_dtype)
        self.cell_t = torch.tensor(self.cell_np, device=self.device, dtype=self.torch_dtype)
        inv_cell = torch.linalg.inv(self.cell_t)
        perp_w = 1.0 / torch.linalg.norm(inv_cell, dim=0)
        self.fw_pos_t = torch.tensor(self.fw.get_positions(), device=self.device, dtype=self.torch_dtype)
        self.fw_Z_t = torch.tensor(self.fw.get_atomic_numbers(), device=self.device, dtype=torch.long)
        self.pbc_t = torch.tensor([True, True, True], device=self.device, dtype=torch.bool)
        self.guest_Z_unit = torch.tensor(self.guest.real_atomic_numbers(), device=self.device, dtype=torch.long)

        mace_n_rep = [int(math.ceil(self.r_max / float(w))) + 1 for w in perp_w]
        m_S = torch.cartesian_prod(
            torch.arange(-mace_n_rep[0], mace_n_rep[0] + 1, device=self.device),
            torch.arange(-mace_n_rep[1], mace_n_rep[1] + 1, device=self.device),
            torch.arange(-mace_n_rep[2], mace_n_rep[2] + 1, device=self.device),
        ).to(self.torch_dtype)
        self.mace_S_all = m_S
        self.mace_shift_vecs = m_S @ self.cell_t
        self.mace_zero_shift_idx = int(torch.nonzero((m_S.abs().sum(dim=1) == 0), as_tuple=False)[0].item())
        self.mace_fw_ei, self.mace_fw_sh, self.mace_fw_ush = self._pair_edges_with_shifts_chunked(
            self.fw_pos_t, self.fw_pos_t, self.r_max, self.mace_shift_vecs, self.mace_zero_shift_idx,
            same_set=True, row_offset=0, col_offset=0, shift_chunk=8, S_all=self.mace_S_all,
        )
        num_z = len(self.mace_calc.z_table)
        fw_idx = mace_utils.atomic_numbers_to_indices(self.fw.get_atomic_numbers(), z_table=self.mace_calc.z_table)
        self.mace_fw_oh = torch.nn.functional.one_hot(
            torch.tensor(fw_idx, dtype=torch.long, device=self.device), num_classes=num_z
        ).to(self.torch_dtype)
        g_idx = mace_utils.atomic_numbers_to_indices(self.guest.real_atomic_numbers(), z_table=self.mace_calc.z_table)
        self.mace_g_oh_unit = torch.nn.functional.one_hot(
            torch.tensor(g_idx, dtype=torch.long, device=self.device), num_classes=num_z
        ).to(self.torch_dtype)
        heads = list(getattr(self.mace_calc, "available_heads", None) or ["Default"])
        cur_head = getattr(self.mace_calc, "head", None)
        self.mace_head_idx = int(heads.index(cur_head)) if cur_head in heads else 0

        # Precompute exact D3 periodic shifts and static rigid framework-framework D3 edges
        if self.use_dispersion:
            self.rc_d3 = float(self.d3_calc.cutoff)
            n_rep = [int(math.ceil(self.rc_d3 / float(w))) + 1 for w in perp_w]
            s0 = torch.arange(-n_rep[0], n_rep[0] + 1, device=self.device)
            s1 = torch.arange(-n_rep[1], n_rep[1] + 1, device=self.device)
            s2 = torch.arange(-n_rep[2], n_rep[2] + 1, device=self.device)
            S_all = torch.cartesian_prod(s0, s1, s2).to(self.torch_dtype)
            self.S_all = S_all
            self.shift_vecs_d3 = S_all @ self.cell_t
            self.zero_shift_idx = int(torch.nonzero((S_all.abs().sum(dim=1) == 0), as_tuple=False)[0].item())

            self.fw_ei, self.fw_sp = self._pair_edges_chunked(
                self.fw_pos_t, self.fw_pos_t, same_set=True, row_offset=0, col_offset=0
            )
        else:
            self.rc_d3 = None

        # Reference isolated framework and isolated guest energies using identical evaluator (FP64 node sum)
        self.e_fw_mace_eV, self.e_fw_d3_eV = self._eval_components_raw_batch(None, [None])
        self.e_fw_mace_eV = float(self.e_fw_mace_eV[0])
        self.e_fw_d3_eV = float(self.e_fw_d3_eV[0])
        self.e_fw_eV = self.e_fw_mace_eV + self.e_fw_d3_eV
        g_iso = self.guest.to_ase_atoms(box_A=30.0)
        self.e_guest_mace_eV = float(self._eval_mace_only_eV([g_iso])[0])
        if self.use_dispersion:
            g_iso.calc = self.d3_calc
            self.e_guest_d3_eV = float(self.d3_calc.get_potential_energy(g_iso))
            self.e_guest_eV = self.e_guest_mace_eV + self.e_guest_d3_eV
        else:
            self.e_guest_d3_eV = 0.0
            self.e_guest_eV = self.e_guest_mace_eV
        if not (math.isfinite(self.e_fw_eV) and math.isfinite(self.e_guest_eV)):
            raise RuntimeError(f"Non-finite reference energies: e_fw={self.e_fw_eV}, e_guest={self.e_guest_eV}")
        self.e_co2_eV = self.e_guest_eV
        self.n_evals = 0

    def _pair_edges_with_shifts_chunked(
        self,
        pos_row,
        pos_col,
        rc: float,
        shift_vecs,
        zero_shift_idx: int,
        same_set: bool,
        row_offset: int,
        col_offset: int,
        shift_chunk: int = 16,
        S_all=None,
    ):
        torch = self.torch
        nr, nc = pos_row.shape[0], pos_col.shape[0]
        if nr == 0 or nc == 0:
            empty_ei = torch.empty((2, 0), device=self.device, dtype=torch.long)
            empty_sh = torch.empty((0, 3), device=self.device, dtype=self.torch_dtype)
            if S_all is not None:
                return empty_ei, empty_sh, torch.empty((0, 3), device=self.device, dtype=self.torch_dtype)
            return empty_ei, empty_sh
        rc2 = float(rc) * float(rc)
        diff_base = pos_col.unsqueeze(0) - pos_row.unsqueeze(1)  # (nr, nc, 3)
        M = shift_vecs.shape[0]
        srcs, dsts, shs = [], [], []
        ushs = [] if S_all is not None else None
        self_diag = None
        if same_set:
            ii = torch.arange(nr, device=self.device).unsqueeze(1).expand(nr, nc)
            jj = torch.arange(nc, device=self.device).unsqueeze(0).expand(nr, nc)
            self_diag = (ii == jj)
        for s0 in range(0, M, shift_chunk):
            s1 = min(M, s0 + shift_chunk)
            sv = shift_vecs[s0:s1]  # (C, 3)
            d = diff_base.unsqueeze(0) + sv.view(-1, 1, 1, 3)
            r2 = (d * d).sum(dim=-1)
            mask = r2 < rc2
            if same_set and s0 <= zero_shift_idx < s1:
                mask[zero_shift_idx - s0] &= (~self_diag)
            mm, rr, cc = torch.nonzero(mask, as_tuple=True)
            if mm.numel() > 0:
                srcs.append(rr + row_offset)
                dsts.append(cc + col_offset)
                shs.append(sv[mm])
                if S_all is not None:
                    ushs.append(S_all[s0:s1][mm])
        if not srcs:
            empty_ei = torch.empty((2, 0), device=self.device, dtype=torch.long)
            empty_sh = torch.empty((0, 3), device=self.device, dtype=self.torch_dtype)
            if S_all is not None:
                return empty_ei, empty_sh, torch.empty((0, 3), device=self.device, dtype=self.torch_dtype)
            return empty_ei, empty_sh
        ei = torch.stack([torch.cat(srcs), torch.cat(dsts)], dim=0)
        sh = torch.cat(shs, dim=0)
        if S_all is not None:
            return ei, sh, torch.cat(ushs, dim=0)
        return ei, sh

    def _pair_edges_chunked(self, pos_row, pos_col, same_set: bool, row_offset: int, col_offset: int, shift_chunk: int = 16):
        return self._pair_edges_with_shifts_chunked(
            pos_row, pos_col, self.rc_d3, self.shift_vecs_d3, self.zero_shift_idx,
            same_set=same_set, row_offset=row_offset, col_offset=col_offset, shift_chunk=shift_chunk,
        )

    def _guest_positions_from_state(self, state):
        n = int(state["N"])
        if n == 0:
            return np.zeros((0, 3), dtype=np.float64)
        c = np.stack([state["cx"], state["cy"], state["cz"]], axis=1)
        if "qw" in state:
            offsets = quat_rotate_batch(
                state["qw"], state["qx"], state["qy"], state["qz"], self.guest_local_pos
            )
        else:
            u = np.stack([state["ux"], state["uy"], state["uz"]], axis=1)
            offsets = u[:, None, :] * self.guest_local_pos[None, :, 0:1]
        gp = (c[:, None, :] + offsets).reshape(n * self.guest_n_real, 3)
        return gp

    def _eval_mace_only_eV(self, atoms_list, mace_chunk: int = None) -> np.ndarray:
        """Evaluate MACE energy using FP64 per-graph summation of `node_energy` to prevent FP32 summation loss."""
        if mace_chunk is None:
            mace_chunk = self.default_mace_chunk
        torch = self.torch
        graphs = []
        with self.torch_tools.default_dtype(self.mace_calc.default_dtype):
            for at in atoms_list:
                cfg = self.mace_data.config_from_atoms(
                    at, key_specification=self.keyspec, head_name=self.mace_calc.head
                )
                graphs.append(
                    self.mace_data.AtomicData.from_config(
                        cfg,
                        z_table=self.mace_calc.z_table,
                        cutoff=self.mace_calc.r_max,
                        heads=self.mace_calc.available_heads,
                    )
                )
        e_mace_parts = []
        with torch.no_grad():
            for i0 in range(0, len(graphs), mace_chunk):
                mb = self.torch_geometric.Batch.from_data_list(graphs[i0:i0 + mace_chunk]).to(self.device)
                out = self.mace_model(mb.to_dict(), compute_force=False, compute_stress=False)
                if out.get("node_energy") is not None:
                    ne = out["node_energy"].to(torch.float64)
                    es = torch.zeros(int(mb.num_graphs), device=self.device, dtype=torch.float64)
                    es.index_add_(0, mb.batch, ne)
                    e_mace_parts.append(es.detach().cpu().numpy())
                else:
                    e_mace_parts.append(out["energy"].detach().cpu().numpy().astype(np.float64))
        return np.concatenate(e_mace_parts, axis=0)

    def _eval_mace_gpu_graphs_eV(self, guest_pos_list, mace_chunk: int = None) -> np.ndarray:
        """Evaluate MACE energy via cached rigid framework graph + GPU dynamic guest edges (FP64 node sum)."""
        if mace_chunk is None:
            mace_chunk = self.default_mace_chunk
        torch = self.torch
        e_mace_parts = []
        with torch.no_grad():
            for i0 in range(0, len(guest_pos_list), mace_chunk):
                chunk_gps = guest_pos_list[i0:i0 + mace_chunk]
                pos_list, na_list, ei_list, sh_list, ush_list, batch_list = [], [], [], [], [], []
                ptr = [0]
                node_off = 0
                for g_idx, gp_np in enumerate(chunk_gps):
                    if gp_np is None or len(gp_np) == 0:
                        pos = self.fw_pos_t
                        na = self.mace_fw_oh
                        ei, sh, ush = self.mace_fw_ei, self.mace_fw_sh, self.mace_fw_ush
                    else:
                        n_mol = len(gp_np) // self.guest_n_real
                        g_pos_t = torch.tensor(gp_np, device=self.device, dtype=self.torch_dtype)
                        pos = torch.cat([self.fw_pos_t, g_pos_t], dim=0)
                        na = torch.cat([self.mace_fw_oh, self.mace_g_oh_unit.repeat(n_mol, 1)], dim=0)
                        ei_fg, sh_fg, ush_fg = self._pair_edges_with_shifts_chunked(
                            self.fw_pos_t, g_pos_t, self.r_max, self.mace_shift_vecs, self.mace_zero_shift_idx,
                            same_set=False, row_offset=0, col_offset=self.n_fw, shift_chunk=8, S_all=self.mace_S_all,
                        )
                        ei_gf = torch.stack([ei_fg[1], ei_fg[0]], dim=0)
                        sh_gf, ush_gf = -sh_fg, -ush_fg
                        ei_gg, sh_gg, ush_gg = self._pair_edges_with_shifts_chunked(
                            g_pos_t, g_pos_t, self.r_max, self.mace_shift_vecs, self.mace_zero_shift_idx,
                            same_set=True, row_offset=self.n_fw, col_offset=self.n_fw, shift_chunk=8, S_all=self.mace_S_all,
                        )
                        ei = torch.cat([self.mace_fw_ei, ei_fg, ei_gf, ei_gg], dim=1)
                        sh = torch.cat([self.mace_fw_sh, sh_fg, sh_gf, sh_gg], dim=0)
                        ush = torch.cat([self.mace_fw_ush, ush_fg, ush_gf, ush_gg], dim=0)
                    n_nodes = pos.shape[0]
                    pos_list.append(pos)
                    na_list.append(na)
                    ei_list.append(ei + node_off)
                    sh_list.append(sh)
                    ush_list.append(ush)
                    batch_list.append(torch.full((n_nodes,), g_idx, device=self.device, dtype=torch.long))
                    node_off += n_nodes
                    ptr.append(node_off)

                n_g = len(chunk_gps)
                batch_t = torch.cat(batch_list, dim=0)
                data_dict = {
                    "positions": torch.cat(pos_list, dim=0),
                    "node_attrs": torch.cat(na_list, dim=0),
                    "edge_index": torch.cat(ei_list, dim=1),
                    "shifts": torch.cat(sh_list, dim=0),
                    "unit_shifts": torch.cat(ush_list, dim=0),
                    "cell": self.cell_t.repeat(n_g, 1),
                    "batch": batch_t,
                    "ptr": torch.tensor(ptr, device=self.device, dtype=torch.long),
                    "head": torch.full((n_g,), self.mace_head_idx, device=self.device, dtype=torch.long),
                    "weight": torch.ones(n_g, device=self.device, dtype=self.torch_dtype),
                    "energy_weight": torch.ones(n_g, device=self.device, dtype=self.torch_dtype),
                    "forces_weight": torch.zeros(n_g, device=self.device, dtype=self.torch_dtype),
                    "stress_weight": torch.zeros(n_g, device=self.device, dtype=self.torch_dtype),
                    "virials_weight": torch.zeros(n_g, device=self.device, dtype=self.torch_dtype),
                }
                out = self.mace_model(data_dict, compute_force=False, compute_stress=False)
                if out.get("node_energy") is not None:
                    ne = out["node_energy"].to(torch.float64)
                    es = torch.zeros(n_g, device=self.device, dtype=torch.float64)
                    es.index_add_(0, batch_t, ne)
                    e_mace_parts.append(es.detach().cpu().numpy())
                else:
                    e_mace_parts.append(out["energy"].detach().cpu().numpy().astype(np.float64))
        return np.concatenate(e_mace_parts, axis=0)

    def _eval_components_raw_batch(self, atoms_list, guest_pos_list, mace_chunk: int = None):
        if mace_chunk is None:
            mace_chunk = self.default_mace_chunk
        torch = self.torch
        # 1. GPU-cached rigid-framework MACE forward (no_grad, compute_force=False, FP64 node-energy accumulation)
        e_mace_np = self._eval_mace_gpu_graphs_eV(guest_pos_list, mace_chunk=mace_chunk)
        if not self.use_dispersion:
            if self.device.type == "cuda":
                tot_mem = float(torch.cuda.get_device_properties(self.device).total_memory)
                res_mem = float(torch.cuda.memory_reserved(self.device))
                alloc_mem = float(torch.cuda.memory_allocated(self.device))
                if res_mem > 0.50 * tot_mem or (res_mem > 3.5e9 and res_mem > 2.5 * alloc_mem):
                    torch.cuda.empty_cache()
            return e_mace_np, np.zeros_like(e_mace_np)

        # 2. Exact GPU D3(BJ) via cached fw-fw edges + vectorized dynamic edges
        e_d3_np = np.zeros(len(guest_pos_list), dtype=np.float64)
        for g_idx, gp_np in enumerate(guest_pos_list):
            if gp_np is None or len(gp_np) == 0:
                Z_tot = self.fw_Z_t
                pos_tot = self.fw_pos_t
                ei_tot = self.fw_ei
                sp_tot = self.fw_sp
            else:
                n_g = gp_np.shape[0]
                n_mol = n_g // self.guest_n_real
                g_pos_t = torch.tensor(gp_np, device=self.device, dtype=self.torch_dtype)
                g_Z_t = self.guest_Z_unit.repeat(n_mol)
                Z_tot = torch.cat([self.fw_Z_t, g_Z_t], dim=0)
                pos_tot = torch.cat([self.fw_pos_t, g_pos_t], dim=0)
                ei_fg, sp_fg = self._pair_edges_chunked(
                    self.fw_pos_t, g_pos_t, same_set=False, row_offset=0, col_offset=self.n_fw
                )
                ei_gf = torch.stack([ei_fg[1], ei_fg[0]], dim=0)
                sp_gf = -sp_fg
                ei_gg, sp_gg = self._pair_edges_chunked(
                    g_pos_t, g_pos_t, same_set=True, row_offset=self.n_fw, col_offset=self.n_fw
                )
                ei_tot = torch.cat([self.fw_ei, ei_fg, ei_gf, ei_gg], dim=1)
                sp_tot = torch.cat([self.fw_sp, sp_fg, sp_gf, sp_gg], dim=0)
            with torch.no_grad():
                ed = self.d3_calc.dftd_module.calc_energy_batch(
                    Z_tot, pos_tot, ei_tot, self.cell_t, self.pbc_t, sp_tot, damping="bj"
                )
            e_d3_np[g_idx] = float(ed[0].item())

        if self.device.type == "cuda":
            tot_mem = float(torch.cuda.get_device_properties(self.device).total_memory)
            res_mem = float(torch.cuda.memory_reserved(self.device))
            alloc_mem = float(torch.cuda.memory_allocated(self.device))
            if res_mem > 0.50 * tot_mem or (res_mem > 3.5e9 and res_mem > 2.5 * alloc_mem):
                torch.cuda.empty_cache()

        return e_mace_np, e_d3_np

    def _eval_raw_eV_batch(self, atoms_list, guest_pos_list, mace_chunk: int = None):
        e_mace_np, e_d3_np = self._eval_components_raw_batch(atoms_list, guest_pos_list, mace_chunk=mace_chunk)
        return e_mace_np + e_d3_np

    def eval_components(self, state, mace_chunk: int = None) -> dict:
        n = int(state["N"])
        if n == 0:
            return {
                "N": 0, "u_int_K": 0.0, "u_mace_int_K": 0.0, "u_d3_int_K": 0.0,
                "e_tot_eV": self.e_fw_eV, "e_mace_eV": self.e_fw_mace_eV, "e_d3_eV": self.e_fw_d3_eV,
            }
        gp = self._guest_positions_from_state(state)
        e_mace, e_d3 = self._eval_components_raw_batch(None, [gp], mace_chunk=mace_chunk)
        em = float(e_mace[0])
        ed = float(e_d3[0])
        if not (math.isfinite(em) and math.isfinite(ed)):
            raise RuntimeError(f"Non-finite component energy: e_mace={em}, e_d3={ed}")
        u_mace_int_K = (em - self.e_fw_mace_eV - n * self.e_guest_mace_eV) * EV_TO_K
        u_d3_int_K = (ed - self.e_fw_d3_eV - n * self.e_guest_d3_eV) * EV_TO_K
        return {
            "N": n,
            "u_int_K": u_mace_int_K + u_d3_int_K,
            "u_mace_int_K": u_mace_int_K,
            "u_d3_int_K": u_d3_int_K,
            "e_tot_eV": em + ed,
            "e_mace_eV": em,
            "e_d3_eV": ed,
        }

    def eval_batch(self, states, mace_chunk: int = None):
        non_empty_idx = []
        guest_pos_list = []
        for idx, st in enumerate(states):
            n = int(st["N"])
            if n == 0:
                continue
            gp = self._guest_positions_from_state(st)
            non_empty_idx.append(idx)
            guest_pos_list.append(gp)

        results = [(0.0, False)] * len(states)
        if not non_empty_idx:
            return results
        e_tot = self._eval_raw_eV_batch(None, guest_pos_list, mace_chunk=mace_chunk)
        self.n_evals += len(non_empty_idx)
        for k, idx in enumerate(non_empty_idx):
            n = int(states[idx]["N"])
            et = float(e_tot[k])
            if not math.isfinite(et):
                raise RuntimeError(f"Non-finite MACE/D3 energy evaluated for batch item {idx} (N={n}): {et}")
            u_int_K = (et - self.e_fw_eV - n * self.e_guest_eV) * EV_TO_K
            results[idx] = (u_int_K, False)
        return results

    def __call__(self, state):
        return self.eval_batch([state])[0]


def verify_proposer_target_consistency(
    proposer: CudaGcmcProposer,
    target,
    pos_tol_A: float = 0.05,
    cell_tol_A: float = 1e-2,
):
    """Verify C++ CIF loader and ASE target see identical unit-cell geometry, atoms, and supercell."""
    H_rows = proposer.H_cols.T
    ase_cell = np.array(target.fw.get_cell(), dtype=np.float64)
    max_cell_err = float(np.max(np.abs(H_rows - ase_cell)))
    if max_cell_err > cell_tol_A:
        raise RuntimeError(
            f"C++/ASE supercell lattice mismatch: max|H_cpp - H_ase|={max_cell_err:.4e} A > {cell_tol_A} A"
        )

    cpp_frac, cpp_Z = proposer.get_unit_atoms()
    n_unit_cpp = int(len(cpp_Z))
    sc_mult = int(proposer.supercell[0] * proposer.supercell[1] * proposer.supercell[2])
    n_super_cpp = n_unit_cpp * sc_mult
    n_super_ase = int(len(target.fw))
    if n_super_cpp != n_super_ase:
        raise RuntimeError(
            f"C++/ASE supercell atom count mismatch: cpp={n_super_cpp} ({n_unit_cpp}*{sc_mult}) vs ase={n_super_ase}"
        )

    uc = getattr(target, "uc", None)
    if uc is None:
        raise RuntimeError("Target evaluator is missing unit-cell `uc` Atoms for consistency check")
    ase_Z = np.array(uc.get_atomic_numbers(), dtype=np.int32)
    ase_frac = np.array(uc.get_scaled_positions(), dtype=np.float64)
    if len(cpp_Z) != len(ase_Z) or not np.array_equal(np.sort(cpp_Z), np.sort(ase_Z)):
        raise RuntimeError(
            f"C++/ASE unit-cell composition mismatch: cpp_len={len(cpp_Z)}, ase_len={len(ase_Z)}"
        )

    uc_cell_rows = np.array(uc.get_cell(), dtype=np.float64)
    from scipy.optimize import linear_sum_assignment

    max_pos_err = 0.0
    for z_val in np.unique(cpp_Z):
        f_c = cpp_frac[cpp_Z == z_val]
        f_a = ase_frac[ase_Z == z_val]
        df = f_c[:, None, :] - f_a[None, :, :]
        df -= np.round(df)
        dr = df @ uc_cell_rows
        dist = np.linalg.norm(dr, axis=-1)
        row_ind, col_ind = linear_sum_assignment(dist)
        z_max = float(np.max(dist[row_ind, col_ind])) if len(row_ind) > 0 else 0.0
        if z_max > max_pos_err:
            max_pos_err = z_max

    if max_pos_err > pos_tol_A:
        raise RuntimeError(
            f"C++/ASE unit-cell atomic coordinate mismatch: max periodic distance={max_pos_err:.4e} A > {pos_tol_A} A"
        )

    r_max = float(getattr(getattr(target, "mace_calc", None), "r_max", 6.0))
    if proposer.min_perp_width_A < 2.0 * r_max - 1e-6:
        print(
            f"  [WARNING] Supercell min perpendicular width ({proposer.min_perp_width_A:.2f} A) "
            f"< 2 * MACE r_max ({2.0 * r_max:.2f} A); periodic minimum-image graph may include self-image edges."
        )

    return {
        "n_unit_atoms": n_unit_cpp,
        "n_super_atoms": n_super_ase,
        "max_cell_err_A": max_cell_err,
        "max_pos_err_A": max_pos_err,
        "min_perp_width_A": float(proposer.min_perp_width_A),
    }


def _qst_from_sums(
    n_samples: int,
    sN: float,
    sN2: float,
    sU: float,
    sUN: float,
    T_K: float,
    allow_molar_fallback: bool = False,
) -> Optional[float]:
    if n_samples <= 0:
        return None if not allow_molar_fallback else 0.0
    inv_m = 1.0 / float(n_samples)
    mN = sN * inv_m
    mN2 = sN2 * inv_m
    mU = sU * inv_m
    mUN = sUN * inv_m
    varN = mN2 - mN * mN
    if varN > 1e-12:
        return float((T_K - (mUN - mU * mN) / varN) * K_TO_KJMOL)
    if allow_molar_fallback and mN > 1e-12:
        return float((T_K - mU / mN) * K_TO_KJMOL)
    return None if not allow_molar_fallback else 0.0


def compute_pooled_jackknife_qst(box_moments, T_K: float, allow_molar_fallback: bool = False) -> dict:
    """Compute pooled fluctuation Qst and leave-one-box-out Jackknife SEM from per-box raw moments."""
    n_b = len(box_moments)
    if n_b == 0:
        val = 0.0 if allow_molar_fallback else None
        return {"Qst_kJ_mol": val, "Qst_sem_kJ_mol": val, "Qst_loo_kJ_mol": []}

    tot_m = sum(int(m["n_samples"]) for m in box_moments)
    tot_sN = sum(float(m["sum_N"]) for m in box_moments)
    tot_sN2 = sum(float(m["sum_N2"]) for m in box_moments)
    tot_sU = sum(float(m["sum_U"]) for m in box_moments)
    tot_sUN = sum(float(m["sum_UN"]) for m in box_moments)

    q_pooled = _qst_from_sums(
        tot_m, tot_sN, tot_sN2, tot_sU, tot_sUN, float(T_K), allow_molar_fallback=allow_molar_fallback
    )
    if q_pooled is None:
        return {"Qst_kJ_mol": None, "Qst_sem_kJ_mol": None, "Qst_loo_kJ_mol": []}
    if n_b <= 1:
        return {"Qst_kJ_mol": q_pooled, "Qst_sem_kJ_mol": 0.0, "Qst_loo_kJ_mol": [q_pooled]}

    q_loo = []
    for m in box_moments:
        q_i = _qst_from_sums(
            tot_m - int(m["n_samples"]),
            tot_sN - float(m["sum_N"]),
            tot_sN2 - float(m["sum_N2"]),
            tot_sU - float(m["sum_U"]),
            tot_sUN - float(m["sum_UN"]),
            float(T_K),
            allow_molar_fallback=allow_molar_fallback,
        )
        q_loo.append(q_i)
    if any(x is None for x in q_loo):
        return {"Qst_kJ_mol": q_pooled, "Qst_sem_kJ_mol": None, "Qst_loo_kJ_mol": q_loo}
    q_arr = np.asarray(q_loo, dtype=np.float64)
    mean_loo = float(np.mean(q_arr))
    var_jack = ((n_b - 1.0) / n_b) * float(np.sum((q_arr - mean_loo) ** 2))
    sem_jack = math.sqrt(max(0.0, var_jack))
    return {"Qst_kJ_mol": q_pooled, "Qst_sem_kJ_mol": sem_jack, "Qst_loo_kJ_mol": q_loo}


def count_oms_occupied(
    state: dict,
    oms_metal_pos: np.ndarray,
    oms_cell_rows: np.ndarray,
    guest_local_pos: Optional[np.ndarray] = None,
    cutoff_A: float = 2.60,
    inv_cell_rows: Optional[np.ndarray] = None,
) -> int:
    """Count guest molecules having at least one real atom within `cutoff_A` of any open metal site."""
    n = int(state["N"])
    if n == 0 or oms_metal_pos is None or len(oms_metal_pos) == 0:
        return 0
    c = np.stack([state["cx"], state["cy"], state["cz"]], axis=1)
    if guest_local_pos is not None and len(guest_local_pos) > 0:
        if "qw" in state:
            offs = quat_rotate_batch(
                state["qw"], state["qx"], state["qy"], state["qz"], guest_local_pos
            )
            pts = (c[:, None, :] + offs).reshape(n, len(guest_local_pos), 3)
        elif "ux" in state:
            u = np.stack([state["ux"], state["uy"], state["uz"]], axis=1)
            offs = u[:, None, :] * guest_local_pos[None, :, 0:1]
            pts = (c[:, None, :] + offs).reshape(n, len(guest_local_pos), 3)
        else:
            pts = c[:, None, :]
    else:
        pts = c[:, None, :]

    inv_h = np.linalg.inv(oms_cell_rows) if inv_cell_rows is None else inv_cell_rows
    df = (pts[:, :, None, :] - oms_metal_pos[None, None, :, :]) @ inv_h
    df -= np.round(df)
    dr = df @ oms_cell_rows
    d2 = np.sum(dr * dr, axis=-1)  # (n, n_sites, n_metals)
    min_d2_per_mol = np.min(d2, axis=(1, 2))
    return int(np.sum(min_d2_per_mol < (cutoff_A * cutoff_A)))


def _serialize_single_box_state(st: dict) -> dict:
    d_st = {
        "N": int(st["N"]),
        "U_ref_K": float(st.get("U_ref_K", 0.0)),
        "cx": np.asarray(st["cx"], dtype=np.float64).tolist(),
        "cy": np.asarray(st["cy"], dtype=np.float64).tolist(),
        "cz": np.asarray(st["cz"], dtype=np.float64).tolist(),
    }
    if "qw" in st:
        d_st["qw"] = np.asarray(st["qw"], dtype=np.float64).tolist()
        d_st["qx"] = np.asarray(st["qx"], dtype=np.float64).tolist()
        d_st["qy"] = np.asarray(st["qy"], dtype=np.float64).tolist()
        d_st["qz"] = np.asarray(st["qz"], dtype=np.float64).tolist()
    elif "ux" in st:
        d_st["ux"] = np.asarray(st["ux"], dtype=np.float64).tolist()
        d_st["uy"] = np.asarray(st["uy"], dtype=np.float64).tolist()
        d_st["uz"] = np.asarray(st["uz"], dtype=np.float64).tolist()
    return d_st


def _deserialize_single_box_state(st: dict) -> dict:
    out = {
        "N": int(st["N"]),
        "U_ref_K": float(st.get("U_ref_K", 0.0)),
        "cx": np.asarray(st["cx"], dtype=np.float64).copy(),
        "cy": np.asarray(st["cy"], dtype=np.float64).copy(),
        "cz": np.asarray(st["cz"], dtype=np.float64).copy(),
    }
    if "qw" in st:
        out["qw"] = np.asarray(st["qw"], dtype=np.float64).copy()
        out["qx"] = np.asarray(st["qx"], dtype=np.float64).copy()
        out["qy"] = np.asarray(st["qy"], dtype=np.float64).copy()
        out["qz"] = np.asarray(st["qz"], dtype=np.float64).copy()
    elif "ux" in st:
        out["ux"] = np.asarray(st["ux"], dtype=np.float64).copy()
        out["uy"] = np.asarray(st["uy"], dtype=np.float64).copy()
        out["uz"] = np.asarray(st["uz"], dtype=np.float64).copy()
    return out


def _serialize_box_states(states_list) -> list:
    return [_serialize_single_box_state(st) for st in states_list]


def _states_equal(a: dict, b: dict, atol: float = 1e-12) -> bool:
    if int(a["N"]) != int(b["N"]):
        return False
    if int(a["N"]) == 0:
        return True
    for k in ("cx", "cy", "cz"):
        if not np.allclose(a[k], b[k], atol=atol, rtol=0.0):
            return False
    if "qw" in a and "qw" in b:
        for k in ("qw", "qx", "qy", "qz"):
            if not np.allclose(a[k], b[k], atol=atol, rtol=0.0):
                return False
    elif "ux" in a and "ux" in b:
        for k in ("ux", "uy", "uz"):
            if not np.allclose(a[k], b[k], atol=atol, rtol=0.0):
                return False
    return True


def _integrated_autocorr_time(series_list) -> float:
    """Pooled FFT integrated autocorrelation time tau_int >= 1.0 (Geyer initial positive cutoff)."""
    valid = []
    for arr in series_list:
        x = np.asarray(arr, dtype=np.float64)
        if len(x) >= 4 and np.var(x) > 1e-12:
            valid.append(x - np.mean(x))
    if not valid:
        return 1.0
    m = min(len(x) for x in valid)
    n_fft = 1 << (2 * m - 1).bit_length()
    acfs = []
    for x in valid:
        xm = x[:m]
        f = np.fft.rfft(xm, n=n_fft)
        c = np.fft.irfft(f * np.conjugate(f), n=n_fft)[:m]
        c /= (np.arange(m, 0, -1, dtype=np.float64))
        if c[0] > 1e-12:
            acfs.append(c / c[0])
    if not acfs:
        return 1.0
    rho = np.mean(np.stack(acfs, axis=0), axis=0)
    tau = 1.0
    for k in range(1, min(len(rho), m // 2)):
        if rho[k] <= 0.0:
            break
        tau += 2.0 * float(rho[k]) * (1.0 - k / float(m))
    return max(1.0, tau)


def compute_ess_stats(prod_N_history, prod_U_history, elapsed_s: float) -> dict:
    """Compute integrated autocorrelation time, ESS, ESS/GPU-s, and jump-rate stats from production traces."""
    if not prod_N_history:
        return {
            "tau_int_N": 1.0, "ess_N": 0.0, "ess_N_per_s": 0.0,
            "tau_int_U": 1.0, "ess_U": 0.0, "ess_U_per_s": 0.0,
            "esjd_N": 0.0, "sjd_N_per_s": 0.0, "dN_moves_per_s": 0.0,
        }
    n_arr = np.asarray(prod_N_history, dtype=np.float64)  # (prod_cycles, n_boxes)
    u_arr = np.asarray(prod_U_history, dtype=np.float64)
    n_cyc, n_b = n_arr.shape
    n_tot = float(n_cyc * n_b)
    dt = max(float(elapsed_s), 1e-6)

    tau_n = _integrated_autocorr_time([n_arr[:, b] for b in range(n_b)])
    tau_u = _integrated_autocorr_time([u_arr[:, b] for b in range(n_b)])
    ess_n = max(1.0, min(n_tot, n_tot / tau_n))
    ess_u = max(1.0, min(n_tot, n_tot / tau_u))
    if n_cyc >= 2:
        diffs = np.diff(n_arr, axis=0)
        esjd_n = float(np.mean(diffs * diffs))
        sjd_n_per_s = float(np.sum(diffs * diffs)) / dt
        dn_moves_per_s = float(np.sum(diffs != 0)) / dt
    else:
        esjd_n = 0.0
        sjd_n_per_s = 0.0
        dn_moves_per_s = 0.0
    return {
        "tau_int_N": float(tau_n),
        "ess_N": float(ess_n),
        "ess_N_per_s": float(ess_n / dt),
        "tau_int_U": float(tau_u),
        "ess_U": float(ess_u),
        "ess_U_per_s": float(ess_u / dt),
        "esjd_N": float(esjd_n),
        "sjd_N_per_s": float(sjd_n_per_s),
        "dN_moves_per_s": float(dn_moves_per_s),
    }


class NestedMCChain:
    """Stateful Grand-Canonical Nested MC chain supporting equilibration-only adaptation, checkpointing, and REMC."""

    def __init__(
        self,
        proposer: CudaGcmcProposer,
        target_fn=None,
        k_steps: int = 10,
        eq_cycles: int = 200,
        prod_cycles: int = 500,
        seed: int = 20261124,
        log_every: int = 0,
        batch_target_fn=None,
        oms_metal_pos: Optional[np.ndarray] = None,
        oms_cell_rows: Optional[np.ndarray] = None,
        oms_cutoff_A: float = 2.60,
        guest_local_pos: Optional[np.ndarray] = None,
        snapshot_stride: int = 0,
        resume_state: Optional[dict] = None,
        checkpoint_fn=None,
        checkpoint_every: int = 0,
        grid_only: bool = False,
        adapt_shift_frac: float = 0.0,
        adapt_k_frac: float = 0.0,
        k_min: int = 2,
        k_max: int = 24,
        target_outer_acc: float = 0.35,
        adapt_window: int = 20,
    ):
        self.proposer = proposer
        self.target_fn = target_fn
        self.batch_target_fn = batch_target_fn
        self.rng = np.random.default_rng(seed)
        self.n_b = int(proposer.n_boxes)
        self.T_K = float(proposer.T_K)
        self.grid_only = bool(grid_only)
        self.delta_eps_K = 0.0 if self.grid_only else float(getattr(proposer, "delta_eps_K", 0.0))
        self.delta_eps_initial_K = float(self.delta_eps_K)
        self.conv_mmol_g = 1000.0 / float(proposer.fw_mass_gmol)
        self.use_quat = hasattr(proposer, "propose_k_quat") and hasattr(proposer, "get_accepted_quat")
        self.eq_cycles = int(eq_cycles)
        self.prod_cycles = int(prod_cycles)
        self.total_cycles = self.eq_cycles + self.prod_cycles
        self.log_every = int(log_every)
        self.oms_cutoff_A = float(oms_cutoff_A)
        self.guest_local_pos = guest_local_pos
        self.snapshot_stride = int(snapshot_stride)
        self.checkpoint_fn = checkpoint_fn
        self.checkpoint_every = int(checkpoint_every)
        self.max_mol = int(getattr(proposer, "max_mol", 0))

        self.k_steps_initial = max(1, int(k_steps))
        self.k_cur = int(self.k_steps_initial)
        self.k_min = max(1, int(k_min))
        self.k_max = max(self.k_min, int(k_max))
        self.target_outer_acc = float(target_outer_acc)
        self.adapt_shift_frac = 0.0 if self.grid_only else float(adapt_shift_frac)
        self.adapt_k_frac = 0.0 if self.grid_only else float(adapt_k_frac)
        self.adapt_shift_end = _compute_adapt_end(self.eq_cycles, self.adapt_shift_frac)
        self.adapt_k_end = _compute_adapt_end(self.eq_cycles, self.adapt_k_frac)
        self.adapt_max_end = max(self.adapt_shift_end, self.adapt_k_end)
        self.adapt_window = max(1, int(adapt_window))
        self.eff_adapt_window = (
            max(1, min(self.adapt_window, self.adapt_max_end)) if self.adapt_max_end > 0 else self.adapt_window
        )

        self.oms_inv_cell = None
        if oms_metal_pos is not None and oms_cell_rows is not None and len(oms_metal_pos) > 0:
            self.oms_metal_pos = np.asarray(oms_metal_pos, dtype=np.float64)
            self.oms_cell_rows = np.asarray(oms_cell_rows, dtype=np.float64)
            self.oms_inv_cell = np.linalg.inv(self.oms_cell_rows)
        else:
            self.oms_metal_pos = None
            self.oms_cell_rows = None

        self._reset_adapt_window()
        self.shift_update_idx = 0
        self.adaptation_history = []
        self.prod_trial_marg_per_box = [[] for _ in range(self.n_b)]

        if resume_state is not None:
            if int(resume_state["n_boxes"]) != self.n_b:
                raise ValueError(f"Resume state n_boxes={resume_state['n_boxes']} != {self.n_b}")
            if abs(float(resume_state["delta_eps_K"]) - self.delta_eps_K) > 1e-9:
                raise ValueError(f"Resume state delta_eps_K={resume_state['delta_eps_K']} != {self.delta_eps_K}")
            if bool(resume_state.get("grid_only", False)) != self.grid_only:
                raise ValueError(f"Resume state grid_only={resume_state.get('grid_only', False)} != {self.grid_only}")
            if int(resume_state["eq_cycles"]) != self.eq_cycles:
                raise ValueError(f"Resume state eq_cycles={resume_state['eq_cycles']} != {self.eq_cycles}")
            if not self.use_quat:
                raise RuntimeError("True checkpoint resume requires quaternion proposer API")
            self.rng.bit_generator.state = resume_state["py_rng_state"]
            restored_states = [_deserialize_single_box_state(st) for st in resume_state["curr_states"]]
            proposer.set_accepted_quat(restored_states)
            proposer.set_rng_states(resume_state["cuda_rng_states"])
            self.curr = proposer.get_accepted_quat()
            self.U_tgt_curr = np.asarray(resume_state["U_tgt_curr"], dtype=np.float64).copy()
            self.U_ref_curr = np.asarray(resume_state["U_ref_curr"], dtype=np.float64).copy()
            self.N_curr = np.asarray(resume_state["N_curr"], dtype=np.int32).copy()
            for b in range(self.n_b):
                if int(self.curr[b]["N"]) != int(self.N_curr[b]):
                    raise RuntimeError(f"Resume N mismatch in box {b}: {self.curr[b]['N']} vs {self.N_curr[b]}")
                if abs(float(self.curr[b]["U_ref_K"]) - float(self.U_ref_curr[b])) > 1e-6:
                    raise RuntimeError(
                        f"Resume U_ref_K mismatch in box {b}: {self.curr[b]['U_ref_K']} vs {self.U_ref_curr[b]}"
                    )
            self.start_cyc = int(resume_state["completed_cycles"])
            self.cyc = self.start_cyc
            self.k_cur = max(1, int(resume_state.get("k_steps", self.k_cur)))
            self.k_steps_initial = max(1, int(resume_state.get("k_steps_initial", self.k_steps_initial)))
            self.delta_eps_initial_K = float(resume_state.get("delta_eps_initial_K", self.delta_eps_K))
            self.adaptation_history = list(resume_state.get("adaptation_history", []))
            self.shift_update_idx = int(resume_state.get("shift_update_idx", len(self.adaptation_history)))
            self.max_N_observed = int(
                resume_state.get("max_N_observed", int(np.max(self.N_curr)) if len(self.N_curr) else 0)
            )
            self.total_inner_steps = int(
                resume_state.get("total_inner_steps", self.start_cyc * self.n_b * self.k_cur)
            )
            self.target_evals = int(resume_state.get("target_evals", 0))
            self.acc_outer = np.asarray(resume_state["acc_outer"], dtype=np.int64).copy()
            self.prod_acc_outer = np.asarray(resume_state["prod_acc_outer"], dtype=np.int64).copy()
            self.inner_acc_tot = np.asarray(resume_state["inner_acc_tot"], dtype=np.int64).copy()
            self.prod_ins_att = int(resume_state["prod_ins_att"])
            self.prod_ins_acc = int(resume_state["prod_ins_acc"])
            self.prod_del_att = int(resume_state["prod_del_att"])
            self.prod_del_acc = int(resume_state["prod_del_acc"])
            self.acc_empty_to_empty = int(resume_state["acc_empty_to_empty"])
            self.acc_identity_nonempty = int(resume_state["acc_identity_nonempty"])
            self.acc_same_N_changed = int(resume_state["acc_same_N_changed"])
            self.acc_dN_changed = int(resume_state["acc_dN_changed"])
            self.prod_rejected = int(resume_state["prod_rejected"])
            self.prod_total_trials = int(resume_state["prod_total_trials"])
            self.nested_trials = int(resume_state.get("nested_trials", self.prod_total_trials))
            self.nested_accepted = int(resume_state.get("nested_accepted", int(np.sum(self.prod_acc_outer))))
            self.sum_N = np.asarray(resume_state["sum_N"], dtype=np.float64).copy()
            self.sum_N2 = np.asarray(resume_state["sum_N2"], dtype=np.float64).copy()
            self.sum_U = np.asarray(resume_state["sum_U"], dtype=np.float64).copy()
            self.sum_UN = np.asarray(resume_state["sum_UN"], dtype=np.float64).copy()
            self.trajectory = list(resume_state.get("trajectory", []))
            self.box_N_history = [list(row) for row in resume_state["box_N_history"]]
            self.prod_N_history = [list(row) for row in resume_state["prod_N_history"]]
            self.prod_U_history = [list(row) for row in resume_state["prod_U_history"]]
            self.prod_U_ref_history = [list(row) for row in resume_state["prod_U_ref_history"]]
            self.prod_accepted_history = [list(row) for row in resume_state["prod_accepted_history"]]
            self.prod_dN_history = [list(row) for row in resume_state["prod_dN_history"]]
            self.prod_proposed_dN_history = [list(row) for row in resume_state["prod_proposed_dN_history"]]
            self.prod_oms_occ_history = [list(row) for row in resume_state.get("prod_oms_occ_history", [])]
            self.prod_snapshots = list(resume_state.get("prod_snapshots", []))
            self.elapsed_prior = float(resume_state.get("elapsed_s", 0.0))
        else:
            self.curr = proposer.get_accepted_quat() if self.use_quat else proposer.get_accepted()
            self.U_tgt_curr = np.zeros(self.n_b, dtype=np.float64)
            self.U_ref_curr = np.zeros(self.n_b, dtype=np.float64)
            self.N_curr = np.zeros(self.n_b, dtype=np.int32)
            if self.grid_only:
                init_evals = [(float(self.curr[b]["U_ref_K"]), False) for b in range(self.n_b)]
            elif batch_target_fn is not None:
                init_evals = batch_target_fn(self.curr)
            else:
                init_evals = [target_fn(self.curr[b]) for b in range(self.n_b)]
            for b in range(self.n_b):
                ut, ov = init_evals[b]
                if not math.isfinite(float(ut)):
                    raise RuntimeError(f"Non-finite initial target energy in box {b}: {ut}")
                if not math.isfinite(float(self.curr[b]["U_ref_K"])):
                    raise RuntimeError(f"Non-finite initial reference energy in box {b}: {self.curr[b]['U_ref_K']}")
                if ov:
                    raise RuntimeError(f"Initial warm-started state of box {b} has target overlap")
                self.U_tgt_curr[b] = float(ut)
                self.U_ref_curr[b] = float(self.curr[b]["U_ref_K"])
                self.N_curr[b] = int(self.curr[b]["N"])

            self.start_cyc = 0
            self.cyc = 0
            self.max_N_observed = int(np.max(self.N_curr)) if len(self.N_curr) else 0
            self.total_inner_steps = 0
            self.target_evals = 0 if self.grid_only else int(np.sum(self.N_curr > 0))
            self.acc_outer = np.zeros(self.n_b, dtype=np.int64)
            self.prod_acc_outer = np.zeros(self.n_b, dtype=np.int64)
            self.inner_acc_tot = np.zeros(self.n_b, dtype=np.int64)
            self.prod_ins_att = 0
            self.prod_ins_acc = 0
            self.prod_del_att = 0
            self.prod_del_acc = 0
            self.acc_empty_to_empty = 0
            self.acc_identity_nonempty = 0
            self.acc_same_N_changed = 0
            self.acc_dN_changed = 0
            self.prod_rejected = 0
            self.prod_total_trials = 0
            self.nested_trials = 0
            self.nested_accepted = 0

            self.sum_N = np.zeros(self.n_b, dtype=np.float64)
            self.sum_N2 = np.zeros(self.n_b, dtype=np.float64)
            self.sum_U = np.zeros(self.n_b, dtype=np.float64)
            self.sum_UN = np.zeros(self.n_b, dtype=np.float64)

            self.trajectory = []
            self.box_N_history = []
            self.prod_N_history = []
            self.prod_U_history = []
            self.prod_U_ref_history = []
            self.prod_accepted_history = []
            self.prod_dN_history = []
            self.prod_proposed_dN_history = []
            self.prod_oms_occ_history = []
            self.prod_snapshots = []
            self.elapsed_prior = 0.0

        self.t0 = time.perf_counter()

    def _reset_adapt_window(self):
        self._win_cycles = 0
        self._win_prop_dN = []
        self._win_nonempty_starts = 0
        self._win_ins_att = 0
        self._win_del_att = 0
        self._win_marg_kJ = []
        self._win_eff_trials = 0
        self._win_eff_acc = 0

    def _apply_adapt_window(self, cyc: int):
        do_shift = cyc < self.adapt_shift_end
        do_k = cyc < self.adapt_k_end
        if not (do_shift or do_k):
            self._reset_adapt_window()
            return

        de_before = float(self.delta_eps_K * K_TO_KJMOL)
        k_before = int(self.k_cur)
        de_marg, _ = _mad_filtered_median(self._win_marg_kJ) if len(self._win_marg_kJ) >= 6 else (None, 0)
        mean_prop_dN = float(np.mean(self._win_prop_dN)) if self._win_prop_dN else 0.0
        eff_acc = (float(self._win_eff_acc) / float(self._win_eff_trials)) if self._win_eff_trials > 0 else None

        if do_shift:
            RT_kJ = self.T_K * K_TO_KJMOL
            step_scale = max(1.5, RT_kJ)
            lr = 0.35 / (1.0 + 0.25 * float(self.shift_update_idx))
            has_marg = (de_marg is not None) and math.isfinite(float(de_marg))
            # Guard drift term when all boxes started empty (N==0) so empty boxes cannot spuriously push de upward
            has_drift = (
                self._win_nonempty_starts > 0
                and (self._win_ins_att + self._win_del_att) > 0
                and len(self._win_prop_dN) > 0
            )
            drift_term = step_scale * float(np.clip(mean_prop_dN, -2.0, 2.0)) if has_drift else 0.0
            if has_marg and has_drift:
                marg_err = float(de_marg) - de_before
                if marg_err * drift_term >= 0.0:
                    delta_raw = lr * (0.80 * marg_err + drift_term)
                else:
                    w_m = 0.35 / (1.0 + 0.5 * float(self.shift_update_idx))
                    delta_raw = lr * (w_m * marg_err + drift_term)
            elif has_marg:
                delta_raw = lr * (float(de_marg) - de_before)
            elif has_drift:
                delta_raw = lr * drift_term
            else:
                delta_raw = 0.0
            delta_step = float(np.clip(delta_raw, -1.5, 1.5))
            new_de_kJ = float(np.clip(de_before + delta_step, -24.0, 24.0))
            if abs(new_de_kJ - de_before) > 1e-12:
                self.proposer.set_chem_pot_shift(new_de_kJ)
                self.delta_eps_K = float(getattr(self.proposer, "delta_eps_K", new_de_kJ / K_TO_KJMOL))
            self.shift_update_idx += 1

        if do_k and eff_acc is not None:
            acc_lo = max(0.05, min(0.22, self.target_outer_acc - 0.13))
            acc_hi = min(0.90, max(0.50, self.target_outer_acc + 0.15))
            if eff_acc < acc_lo:
                k_next = max(self.k_min, int(round(self.k_cur * 0.75)))
                if k_next == self.k_cur and self.k_cur > self.k_min:
                    k_next = self.k_cur - 1
                self.k_cur = k_next
            elif eff_acc > acc_hi:
                k_scaled = int(round(self.k_cur * 1.25))
                k_next = min(self.k_max, k_scaled + (1 if k_scaled == self.k_cur else 0))
                self.k_cur = k_next

        self.adaptation_history.append({
            "cycle": int(cyc + 1),
            "delta_eps_before_kJ_mol": de_before,
            "delta_eps_kJ_mol": float(self.delta_eps_K * K_TO_KJMOL),
            "k_steps_before": k_before,
            "k_steps": int(self.k_cur),
            "mean_prop_dN": float(mean_prop_dN),
            "de_marg_kJ_mol": None if de_marg is None else float(de_marg),
            "n_marg_samples": int(len(self._win_marg_kJ)),
            "eff_acc": None if eff_acc is None else float(eff_acc),
            "win_ins_att": int(self._win_ins_att),
            "win_del_att": int(self._win_del_att),
            "win_nonempty_starts": int(self._win_nonempty_starts),
        })
        self._reset_adapt_window()

    def build_ckpt(self, completed_cyc: int) -> dict:
        cuda_rng = self.proposer.get_rng_states().tolist() if hasattr(self.proposer, "get_rng_states") else []
        return {
            "completed_cycles": int(completed_cyc),
            "eq_cycles": int(self.eq_cycles),
            "prod_cycles": int(self.prod_cycles),
            "n_boxes": int(self.n_b),
            "k_steps": int(self.k_cur),
            "k_steps_initial": int(self.k_steps_initial),
            "delta_eps_K": float(self.delta_eps_K),
            "delta_eps_initial_K": float(self.delta_eps_initial_K),
            "grid_only": bool(self.grid_only),
            "adaptation_history": self.adaptation_history,
            "shift_update_idx": int(self.shift_update_idx),
            "max_N_observed": int(self.max_N_observed),
            "total_inner_steps": int(self.total_inner_steps),
            "target_evals": int(self.target_evals),
            "py_rng_state": self.rng.bit_generator.state,
            "cuda_rng_states": [int(x) for x in cuda_rng],
            "curr_states": _serialize_box_states(self.curr),
            "U_tgt_curr": [float(x) for x in self.U_tgt_curr],
            "U_ref_curr": [float(x) for x in self.U_ref_curr],
            "N_curr": [int(x) for x in self.N_curr],
            "acc_outer": [int(x) for x in self.acc_outer],
            "prod_acc_outer": [int(x) for x in self.prod_acc_outer],
            "inner_acc_tot": [int(x) for x in self.inner_acc_tot],
            "prod_ins_att": int(self.prod_ins_att),
            "prod_ins_acc": int(self.prod_ins_acc),
            "prod_del_att": int(self.prod_del_att),
            "prod_del_acc": int(self.prod_del_acc),
            "acc_empty_to_empty": int(self.acc_empty_to_empty),
            "acc_identity_nonempty": int(self.acc_identity_nonempty),
            "acc_same_N_changed": int(self.acc_same_N_changed),
            "acc_dN_changed": int(self.acc_dN_changed),
            "prod_rejected": int(self.prod_rejected),
            "prod_total_trials": int(self.prod_total_trials),
            "nested_trials": int(self.nested_trials),
            "nested_accepted": int(self.nested_accepted),
            "sum_N": [float(x) for x in self.sum_N],
            "sum_N2": [float(x) for x in self.sum_N2],
            "sum_U": [float(x) for x in self.sum_U],
            "sum_UN": [float(x) for x in self.sum_UN],
            "box_N_history": self.box_N_history,
            "prod_N_history": self.prod_N_history,
            "prod_U_history": self.prod_U_history,
            "prod_U_ref_history": self.prod_U_ref_history,
            "prod_accepted_history": self.prod_accepted_history,
            "prod_dN_history": self.prod_dN_history,
            "prod_proposed_dN_history": self.prod_proposed_dN_history,
            "prod_oms_occ_history": self.prod_oms_occ_history,
            "prod_snapshots": self.prod_snapshots,
            "trajectory": self.trajectory,
            "elapsed_s": float(self.elapsed_prior + (time.perf_counter() - self.t0)),
        }

    def apply_swapped_boxes(self, box_updates: dict) -> None:
        """Install REMC-swapped box states, validate hard-core legality, and resync proposer U_ref_K cleanly."""
        if not box_updates:
            return
        for b, (st_dict, n_val, u_tgt_val) in box_updates.items():
            bi = int(b)
            self.curr[bi] = _deserialize_single_box_state(st_dict)
            self.N_curr[bi] = int(n_val)
            self.U_tgt_curr[bi] = float(u_tgt_val)
        if self.use_quat and hasattr(self.proposer, "set_accepted_quat"):
            self.proposer.set_accepted_quat(self.curr)
            refreshed = self.proposer.get_accepted_quat()
            for b in range(self.n_b):
                if int(refreshed[b]["N"]) != int(self.N_curr[b]):
                    raise RuntimeError(
                        f"REMC swap N mismatch after proposer sync in box {b}: "
                        f"{refreshed[b]['N']} vs {self.N_curr[b]}"
                    )
                u_ref_new = float(refreshed[b]["U_ref_K"])
                if not math.isfinite(u_ref_new):
                    raise RuntimeError(f"Non-finite U_ref_K after REMC swap sync in box {b}: {u_ref_new}")
                if b in box_updates and hasattr(self.proposer, "eval_energy_quat"):
                    u_chk, ov_chk = self.proposer.eval_energy_quat(refreshed[b])
                    if ov_chk or not math.isfinite(float(u_chk)):
                        raise RuntimeError(f"REMC swap box {b} failed independent hard-core/energy verification (ov={ov_chk}, u={u_chk})")
                    if abs(float(u_chk) - u_ref_new) > 1e-6:
                        raise RuntimeError(f"REMC swap U_ref mismatch in box {b}: {u_chk} vs {u_ref_new}")
                self.U_ref_curr[b] = u_ref_new
            self.curr = refreshed
        else:
            for b in box_updates:
                bi = int(b)
                self.U_ref_curr[bi] = float(self.curr[bi].get("U_ref_K", 0.0))
        if len(self.N_curr) > 0:
            self.max_N_observed = max(self.max_N_observed, int(np.max(self.N_curr)))

    def partial_snapshot(self) -> dict:
        """Return lightweight current post-equilibration traces and counters for periodic checkpoint evaluation."""
        return {
            "completed_cycles": int(self.cyc),
            "eq_cycles": int(self.eq_cycles),
            "prod_cycles_done": int(len(self.prod_N_history)),
            "k_steps": int(self.k_cur),
            "delta_eps_kJ_mol": float(self.delta_eps_K * K_TO_KJMOL),
            "target_evals": int(self.target_evals),
            "total_inner_steps": int(self.total_inner_steps),
            "prod_ins_att": int(self.prod_ins_att),
            "prod_ins_acc": int(self.prod_ins_acc),
            "prod_del_att": int(self.prod_del_att),
            "prod_del_acc": int(self.prod_del_acc),
            "prod_acc_outer": [int(x) for x in self.prod_acc_outer],
            "prod_total_trials": int(self.prod_total_trials),
            "acc_empty_to_empty": int(self.acc_empty_to_empty),
            "acc_identity_nonempty": int(self.acc_identity_nonempty),
            "acc_same_N_changed": int(self.acc_same_N_changed),
            "acc_dN_changed": int(self.acc_dN_changed),
            "prod_N_history": [list(r) for r in self.prod_N_history],
            "prod_U_history": [list(r) for r in self.prod_U_history],
            "prod_oms_occ_history": [list(r) for r in self.prod_oms_occ_history],
        }

    def step_cycles(self, n_steps: int) -> None:
        end_cyc = min(self.total_cycles, self.cyc + max(0, int(n_steps)))
        for cyc in range(self.cyc, end_cyc):
            mask = np.zeros(self.n_b, dtype=np.int32)
            cyc_prop_dN = [0] * self.n_b
            cyc_net_dN = [0] * self.n_b

            trials = self.proposer.propose_k_quat(self.k_cur) if self.use_quat else self.proposer.propose_k(self.k_cur)
            self.total_inner_steps += self.n_b * self.k_cur
            if self.grid_only:
                tri_evals = [(float(trials[b]["U_ref_K"]), False) for b in range(self.n_b)]
            else:
                self.target_evals += sum(1 for t in trials if int(t["N"]) > 0)
                if self.batch_target_fn is not None:
                    tri_evals = self.batch_target_fn(trials)
                else:
                    tri_evals = [self.target_fn(trials[b]) for b in range(self.n_b)]

            in_adapt = (not self.grid_only) and (cyc < self.adapt_max_end)
            for b in range(self.n_b):
                self.inner_acc_tot[b] += trials[b]["inner_acc"]
                if not math.isfinite(float(trials[b]["U_ref_K"])):
                    raise RuntimeError(f"Non-finite trial reference energy at cycle {cyc}, box {b}: {trials[b]['U_ref_K']}")
                n_prev = int(self.N_curr[b])
                dU_ref = float(trials[b]["U_ref_K"]) - float(self.U_ref_curr[b])
                dN = int(trials[b]["N"]) - n_prev
                cyc_prop_dN[b] = dN
                u_tgt_tri, ov = tri_evals[b]
                if not ov and not math.isfinite(float(u_tgt_tri)):
                    raise RuntimeError(f"Non-finite trial target energy at cycle {cyc}, box {b}: {u_tgt_tri}")
                accepted = False
                if self.grid_only:
                    accepted = True
                elif not ov:
                    dU_tgt = float(u_tgt_tri) - float(self.U_tgt_curr[b])
                    ddU = dU_tgt - dU_ref - self.delta_eps_K * float(dN)
                    if not math.isfinite(float(ddU)):
                        raise RuntimeError(f"Non-finite Nested MC ddU at cycle {cyc}, box {b}: {ddU}")
                    if ddU <= 0.0:
                        accepted = True
                    else:
                        arg = -ddU / self.T_K
                        if arg > -50.0 and self.rng.random() < math.exp(arg):
                            accepted = True
                    if dN != 0:
                        marg_kJ = (dU_tgt - dU_ref) * K_TO_KJMOL / float(dN)
                        if math.isfinite(marg_kJ):
                            if in_adapt:
                                self._win_marg_kJ.append(marg_kJ)
                            elif cyc >= self.eq_cycles:
                                self.prod_trial_marg_per_box[b].append(marg_kJ)

                if in_adapt:
                    if n_prev > 0:
                        self._win_nonempty_starts += 1
                    self._win_prop_dN.append(float(dN))
                    if dN > 0:
                        self._win_ins_att += 1
                    elif dN < 0:
                        self._win_del_att += 1
                    if not (n_prev == 0 and int(trials[b]["N"]) == 0):
                        self._win_eff_trials += 1
                        if accepted:
                            self._win_eff_acc += 1

                if cyc >= self.eq_cycles:
                    self.nested_trials += 1
                    same_cfg = _states_equal(trials[b], self.curr[b])
                    self.prod_total_trials += 1
                    if dN > 0:
                        self.prod_ins_att += 1
                        if accepted:
                            self.prod_ins_acc += 1
                    elif dN < 0:
                        self.prod_del_att += 1
                        if accepted:
                            self.prod_del_acc += 1
                    if not accepted:
                        self.prod_rejected += 1
                    elif n_prev == 0 and int(trials[b]["N"]) == 0:
                        self.acc_empty_to_empty += 1
                    elif same_cfg:
                        self.acc_identity_nonempty += 1
                    elif dN == 0:
                        self.acc_same_N_changed += 1
                    else:
                        self.acc_dN_changed += 1

                if accepted:
                    mask[b] = 1
                    self.U_tgt_curr[b] = float(u_tgt_tri)
                    self.U_ref_curr[b] = float(trials[b]["U_ref_K"])
                    self.N_curr[b] = int(trials[b]["N"])
                    self.curr[b] = trials[b]
                    cyc_net_dN[b] = dN
                    self.acc_outer[b] += 1
                    if cyc >= self.eq_cycles:
                        self.prod_acc_outer[b] += 1
                        self.nested_accepted += 1

            self.proposer.commit(mask)
            if len(self.N_curr) > 0:
                self.max_N_observed = max(self.max_N_observed, int(np.max(self.N_curr)))

            if in_adapt:
                self._win_cycles += 1
                if self._win_cycles >= self.eff_adapt_window:
                    self._apply_adapt_window(cyc)

            if cyc >= self.eq_cycles:
                self.prod_N_history.append([int(x) for x in self.N_curr])
                self.prod_U_history.append([float(self.U_tgt_curr[b] * K_TO_KJMOL) for b in range(self.n_b)])
                self.prod_U_ref_history.append([float(self.U_ref_curr[b] * K_TO_KJMOL) for b in range(self.n_b)])
                self.prod_accepted_history.append([int(mask[b]) for b in range(self.n_b)])
                self.prod_dN_history.append(cyc_net_dN)
                self.prod_proposed_dN_history.append(cyc_prop_dN)
                if self.oms_metal_pos is not None:
                    self.prod_oms_occ_history.append([
                        count_oms_occupied(
                            self.curr[b], self.oms_metal_pos, self.oms_cell_rows,
                            guest_local_pos=self.guest_local_pos, cutoff_A=self.oms_cutoff_A,
                            inv_cell_rows=self.oms_inv_cell,
                        )
                        for b in range(self.n_b)
                    ])
                prod_step = cyc - self.eq_cycles + 1
                if self.snapshot_stride > 0 and (prod_step % self.snapshot_stride == 0 or cyc == self.total_cycles - 1):
                    self.prod_snapshots.append({
                        "prod_cycle": int(prod_step),
                        "states": _serialize_box_states(self.curr),
                    })
                for b in range(self.n_b):
                    dn = float(self.N_curr[b])
                    du = float(self.U_tgt_curr[b])
                    self.sum_N[b] += dn
                    self.sum_N2[b] += dn * dn
                    self.sum_U[b] += du
                    self.sum_UN[b] += du * dn

            self.box_N_history.append([int(x) for x in self.N_curr])
            if self.n_b == 1:
                self.trajectory.append({
                    "cycle": cyc,
                    "in_prod": cyc >= self.eq_cycles,
                    "N": int(self.N_curr[0]),
                    "uptake_mmol_g": float(self.N_curr[0] * self.conv_mmol_g),
                    "U_tgt_kJ": float(self.U_tgt_curr[0] * K_TO_KJMOL),
                    "U_ref_kJ": float(self.U_ref_curr[0] * K_TO_KJMOL),
                    "accepted": bool(mask[0]),
                })
            if self.checkpoint_fn is not None and self.checkpoint_every > 0 and ((cyc + 1) % self.checkpoint_every == 0):
                self.checkpoint_fn(self.build_ckpt(cyc + 1))
            if self.log_every > 0 and ((cyc + 1) % self.log_every == 0):
                elapsed = self.elapsed_prior + (time.perf_counter() - self.t0)
                mean_up_now = float(np.mean(self.N_curr) * self.conv_mmol_g)
                print(
                    f"  [NestedMC cyc {cyc+1:4d}/{self.total_cycles}] N={self.N_curr.tolist()} ({mean_up_now:.2f} mmol/g) "
                    f"Acc={np.mean(self.acc_outer)/(cyc+1)*100:5.1f}% k={self.k_cur} "
                    f"dEps={self.delta_eps_K*K_TO_KJMOL:+.2f} ({elapsed:.1f}s)"
                )
            self.cyc = cyc + 1

    def finalize(self, elapsed_s: Optional[float] = None) -> dict:
        dt = float(elapsed_s) if elapsed_s is not None else float(self.elapsed_prior + (time.perf_counter() - self.t0))
        final_ckpt = self.build_ckpt(self.cyc)
        if self.checkpoint_fn is not None and self.cyc > self.start_cyc:
            self.checkpoint_fn(final_ckpt)

        actual_prod = len(self.prod_N_history)
        actual_tot = len(self.box_N_history)
        prod_denom = max(1, actual_prod)
        tot_denom = max(1, actual_tot)
        box_uptakes = (self.sum_N / prod_denom) * self.conv_mmol_g if actual_prod > 0 else np.zeros(self.n_b, dtype=np.float64)
        box_qst = []
        box_molar_bind = []
        box_molar_qst = []
        for b in range(self.n_b):
            mN = self.sum_N[b] / prod_denom if actual_prod > 0 else 0.0
            mN2 = self.sum_N2[b] / prod_denom if actual_prod > 0 else 0.0
            mU = self.sum_U[b] / prod_denom if actual_prod > 0 else 0.0
            mUN = self.sum_UN[b] / prod_denom if actual_prod > 0 else 0.0
            varN = mN2 - mN * mN
            mb = (-mU / mN) * K_TO_KJMOL if mN > 1e-12 else 0.0
            mq = (self.T_K - mU / mN) * K_TO_KJMOL if mN > 1e-12 else 0.0
            box_molar_bind.append(mb)
            box_molar_qst.append(mq)
            if varN > 1e-12:
                q_K = self.T_K - (mUN - mU * mN) / varN
                box_qst.append(float(q_K * K_TO_KJMOL))
            else:
                box_qst.append(None)

        mean_up = float(np.mean(box_uptakes))
        sem_up = float(np.std(box_uptakes, ddof=1) / math.sqrt(self.n_b)) if self.n_b > 1 else 0.0
        valid_box_qst = [q for q in box_qst if q is not None]
        per_box_mean_qst = float(np.mean(valid_box_qst)) if valid_box_qst else None
        per_box_sem_qst = (
            float(np.std(valid_box_qst, ddof=1) / math.sqrt(len(valid_box_qst)))
            if len(valid_box_qst) > 1 else (0.0 if len(valid_box_qst) == 1 else None)
        )
        box_moments = [
            {
                "n_samples": int(actual_prod),
                "sum_N": float(self.sum_N[b]),
                "sum_N2": float(self.sum_N2[b]),
                "sum_U": float(self.sum_U[b]),
                "sum_UN": float(self.sum_UN[b]),
            }
            for b in range(self.n_b)
        ]

        from mcmc_diagnostics import compute_independent_qst
        qst_res = compute_independent_qst(
            self.prod_N_history, self.prod_U_history, T_K=self.T_K,
            prod_ins_acc=self.prod_ins_acc, prod_del_acc=self.prod_del_acc,
        )
        mean_qst = qst_res["qst_fluctuation_kJ_mol"]
        sem_qst = qst_res["qst_fluctuation_sem_kJ_mol"]
        mean_mb = float(np.mean(box_molar_bind))
        mean_mq = float(np.mean(box_molar_qst))
        sem_mq = float(np.std(box_molar_qst, ddof=1) / math.sqrt(self.n_b)) if self.n_b > 1 else 0.0

        acceptance_breakdown = {
            "total_trials": int(self.prod_total_trials),
            "rejected": int(self.prod_rejected),
            "acc_empty_to_empty": int(self.acc_empty_to_empty),
            "acc_identity_nonempty": int(self.acc_identity_nonempty),
            "acc_same_N_changed": int(self.acc_same_N_changed),
            "acc_dN_changed": int(self.acc_dN_changed),
            "effective_move_acc_rate": float((self.acc_same_N_changed + self.acc_dN_changed) / max(1, self.prod_total_trials)),
            "nested_trials": int(self.nested_trials),
            "nested_accepted": int(self.nested_accepted),
        }

        prod_trial_marg_meds = []
        prod_trial_marg_counts = []
        for b in range(self.n_b):
            vals = self.prod_trial_marg_per_box[b]
            prod_trial_marg_counts.append(int(len(vals)))
            if len(vals) >= 2:
                m_val, _ = _mad_filtered_median(vals)
                prod_trial_marg_meds.append(m_val)
            else:
                prod_trial_marg_meds.append(None)

        ess_stats = compute_ess_stats(self.prod_N_history, self.prod_U_history, dt)
        final_acc = self.proposer.get_accepted_quat() if self.use_quat else self.proposer.get_accepted()
        final_states = _serialize_box_states(final_acc)

        return {
            "n_boxes": self.n_b,
            "k_steps": int(self.k_cur),
            "k_steps_initial": int(self.k_steps_initial),
            "grid_only": bool(self.grid_only),
            "eq_cycles": int(self.eq_cycles),
            "prod_cycles": int(actual_prod),
            "prod_cycles_requested": int(self.prod_cycles),
            "completed_cycles": int(self.cyc),
            "delta_eps_initial_kJ_mol": float(self.delta_eps_initial_K * K_TO_KJMOL),
            "delta_eps_kJ_mol": float(self.delta_eps_K * K_TO_KJMOL),
            "adapt_shift_end_cycle": int(self.adapt_shift_end),
            "adapt_k_end_cycle": int(self.adapt_k_end),
            "adaptation_history": self.adaptation_history,
            "max_N_observed": int(self.max_N_observed),
            "max_mol": int(self.max_mol),
            "total_inner_steps": int(self.total_inner_steps),
            "target_evals": int(self.target_evals),
            "prod_trial_marginal_per_box_kJ_mol": prod_trial_marg_meds,
            "prod_trial_marginal_counts": prod_trial_marg_counts,
            "elapsed_s": dt,
            "uptake_mmol_g": mean_up,
            "uptake_sem_mmol_g": sem_up,
            "box_uptakes_mmol_g": box_uptakes.tolist(),
            "Qst_estimator": "independent_fluctuation",
            "Qst_kJ_mol": mean_qst,
            "Qst_sem_kJ_mol": sem_qst,
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
            "outer_acc_rate": float(np.mean(self.acc_outer) / tot_denom),
            "prod_outer_acc_rate": float(np.mean(self.prod_acc_outer) / prod_denom) if actual_prod > 0 else 0.0,
            "prod_ins_att": int(self.prod_ins_att),
            "prod_ins_acc": int(self.prod_ins_acc),
            "prod_del_att": int(self.prod_del_att),
            "prod_del_acc": int(self.prod_del_acc),
            "acceptance_breakdown": acceptance_breakdown,
            "mean_inner_acc_per_burst": float(np.mean(self.inner_acc_tot) / tot_denom),
            "effective_mc_steps_per_target_eval": float(self.k_cur * np.mean(self.acc_outer) / tot_denom),
            "box_N_history": self.box_N_history,
            "prod_N_history": self.prod_N_history,
            "prod_U_history": self.prod_U_history,
            "prod_U_ref_history": self.prod_U_ref_history,
            "prod_accepted_history": self.prod_accepted_history,
            "prod_dN_history": self.prod_dN_history,
            "prod_proposed_dN_history": self.prod_proposed_dN_history,
            "prod_oms_occ_history": self.prod_oms_occ_history,
            "prod_snapshots": self.prod_snapshots,
            "trajectory": self.trajectory,
            "final_states": final_states,
            "final_cuda_rng_states": final_ckpt["cuda_rng_states"],
        }


def run_nested_mc(
    proposer: CudaGcmcProposer,
    target_fn,
    k_steps: int = 10,
    eq_cycles: int = 200,
    prod_cycles: int = 500,
    seed: int = 20261124,
    log_every: int = 0,
    batch_target_fn=None,
    oms_metal_pos: Optional[np.ndarray] = None,
    oms_cell_rows: Optional[np.ndarray] = None,
    oms_cutoff_A: float = 2.60,
    guest_local_pos: Optional[np.ndarray] = None,
    snapshot_stride: int = 0,
    resume_state: Optional[dict] = None,
    checkpoint_fn=None,
    checkpoint_every: int = 0,
    grid_only: bool = False,
    adapt_shift_frac: float = 0.0,
    adapt_k_frac: float = 0.0,
    k_min: int = 2,
    k_max: int = 24,
    target_outer_acc: float = 0.35,
    adapt_window: int = 20,
):
    """Exact Grand-Canonical Nested MC loop (with equilibration-only adaptation, grid-only mode, and bit-exact resume)."""
    chain = NestedMCChain(
        proposer=proposer,
        target_fn=target_fn,
        k_steps=k_steps,
        eq_cycles=eq_cycles,
        prod_cycles=prod_cycles,
        seed=seed,
        log_every=log_every,
        batch_target_fn=batch_target_fn,
        oms_metal_pos=oms_metal_pos,
        oms_cell_rows=oms_cell_rows,
        oms_cutoff_A=oms_cutoff_A,
        guest_local_pos=guest_local_pos,
        snapshot_stride=snapshot_stride,
        resume_state=resume_state,
        checkpoint_fn=checkpoint_fn,
        checkpoint_every=checkpoint_every,
        grid_only=grid_only,
        adapt_shift_frac=adapt_shift_frac,
        adapt_k_frac=adapt_k_frac,
        k_min=k_min,
        k_max=k_max,
        target_outer_acc=target_outer_acc,
        adapt_window=adapt_window,
    )
    chain.step_cycles(chain.total_cycles - chain.cyc)
    return chain.finalize()


def attempt_remc_swaps(N_matrix, fugacities_Pa, parity: int, ladder_rngs):
    """Exact zero-MLIP-cost Grand Canonical pressure-REMC adjacent swap decisions on independent ladders.

    Ladder b at pressure m ONLY attempts swaps with ladder b at pressure m+1:
        log alpha(m, m+1; b) = (N_{m+1,b} - N_{m,b}) * ln(f_m / f_{m+1})
    """
    M = len(N_matrix)
    if M < 2:
        raise ValueError(f"REMC requires at least 2 pressures, got {M}")
    B = len(N_matrix[0])
    if len(fugacities_Pa) != M:
        raise ValueError(f"fugacities_Pa length {len(fugacities_Pa)} != M={M}")
    if len(ladder_rngs) != B:
        raise ValueError(f"ladder_rngs length {len(ladder_rngs)} != B={B}")

    att_up = np.zeros(M, dtype=np.int64)
    acc_up = np.zeros(M, dtype=np.int64)
    accepted_pairs = []

    for m in range(int(parity) % 2, M - 1, 2):
        f_lo = float(fugacities_Pa[m])
        f_hi = float(fugacities_Pa[m + 1])
        if f_lo <= 0.0 or f_hi <= 0.0 or not (math.isfinite(f_lo) and math.isfinite(f_hi)):
            raise ValueError(f"Invalid fugacities for REMC swap at pair ({m}, {m+1}): f_lo={f_lo}, f_hi={f_hi}")
        log_f_ratio = math.log(f_lo / f_hi)
        row_lo = N_matrix[m]
        row_hi = N_matrix[m + 1]
        for b in range(B):
            att_up[m] += 1
            dN_pair = int(row_hi[b]) - int(row_lo[b])
            log_acc = float(dN_pair) * log_f_ratio
            if log_acc >= 0.0 or (log_acc > -50.0 and ladder_rngs[b].random() < math.exp(log_acc)):
                acc_up[m] += 1
                accepted_pairs.append((m, m + 1, b))

    return accepted_pairs, att_up, acc_up


def run_remc_isotherm(
    proposers: list,
    pressures_bar: list,
    target_fn=None,
    batch_target_fn=None,
    k_steps: int = 10,
    eq_cycles: int = 200,
    prod_cycles: int = 500,
    swap_every: int = 2,
    seed: int = 20261124,
    log_every: int = 0,
    oms_metal_pos: Optional[np.ndarray] = None,
    oms_cell_rows: Optional[np.ndarray] = None,
    oms_cutoff_A: float = 2.60,
    guest_local_pos: Optional[np.ndarray] = None,
    snapshot_stride: int = 0,
    grid_only: bool = False,
    adapt_shift_frac: float = 0.0,
    adapt_k_frac: float = 0.0,
    k_min: int = 2,
    k_max: int = 24,
    target_outer_acc: float = 0.35,
    adapt_window: int = 20,
) -> list:
    """Zero-MLIP-cost Grand Canonical Parallel Tempering across M sorted pressures on B independent ladders."""
    from mcmc_diagnostics import evaluate_v2_convergence

    M = len(proposers)
    if M < 2 or len(pressures_bar) != M:
        raise ValueError(f"run_remc_isotherm requires >=2 matching proposers and pressures (got {M}, {len(pressures_bar)})")

    order = [int(i) for i in np.argsort([float(p) for p in pressures_bar])]
    props_sorted = [proposers[i] for i in order]
    p_sorted = [float(pressures_bar[i]) for i in order]

    n_b = int(props_sorted[0].n_boxes)
    T_K = float(props_sorted[0].T_K)
    fugacities_Pa = []
    for m, prop in enumerate(props_sorted):
        if int(prop.n_boxes) != n_b:
            raise ValueError(f"All REMC proposers must share n_boxes={n_b}, got {prop.n_boxes} at m={m}")
        if abs(float(prop.T_K) - T_K) > 1e-6:
            raise ValueError(f"All REMC proposers must share T_K={T_K}, got {prop.T_K} at m={m}")
        fugacities_Pa.append(float(prop.f_Pa))

    chains = [
        NestedMCChain(
            proposer=props_sorted[m],
            target_fn=target_fn,
            batch_target_fn=batch_target_fn,
            k_steps=k_steps,
            eq_cycles=eq_cycles,
            prod_cycles=prod_cycles,
            seed=int(seed) + m * 1009,
            log_every=0,
            oms_metal_pos=oms_metal_pos,
            oms_cell_rows=oms_cell_rows,
            oms_cutoff_A=oms_cutoff_A,
            guest_local_pos=guest_local_pos,
            snapshot_stride=snapshot_stride,
            grid_only=grid_only,
            adapt_shift_frac=adapt_shift_frac,
            adapt_k_frac=adapt_k_frac,
            k_min=k_min,
            k_max=k_max,
            target_outer_acc=target_outer_acc,
            adapt_window=adapt_window,
        )
        for m in range(M)
    ]

    ladder_rngs = [np.random.default_rng(int(seed) + 7919 * (b + 1)) for b in range(n_b)]
    total_cycles = int(eq_cycles) + int(prod_cycles)
    swap_stride = max(1, int(swap_every))
    prod_swap_att_up = np.zeros(M, dtype=np.int64)
    prod_swap_acc_up = np.zeros(M, dtype=np.int64)
    tot_swap_att_up = np.zeros(M, dtype=np.int64)
    tot_swap_acc_up = np.zeros(M, dtype=np.int64)

    t0 = time.perf_counter()
    cyc_done = 0
    swap_round = 0
    next_log = int(log_every) if int(log_every) > 0 else 0

    while cyc_done < total_cycles:
        batch = min(swap_stride, total_cycles - cyc_done)
        if cyc_done < int(eq_cycles) and cyc_done + batch > int(eq_cycles):
            batch = int(eq_cycles) - cyc_done
        for m in range(M):
            chains[m].step_cycles(batch)
        cyc_done += batch

        if cyc_done < total_cycles:
            parity = swap_round % 2
            swap_round += 1
            N_mat = [chains[m].N_curr.tolist() for m in range(M)]
            accepted_pairs, att_up, acc_up = attempt_remc_swaps(
                N_mat, fugacities_Pa, parity=parity, ladder_rngs=ladder_rngs
            )
            tot_swap_att_up += att_up
            tot_swap_acc_up += acc_up
            if cyc_done > int(eq_cycles):
                prod_swap_att_up += att_up
                prod_swap_acc_up += acc_up

            if accepted_pairs:
                updates_by_m = {}
                for m_lo, m_hi, b in accepted_pairs:
                    st_lo = _serialize_single_box_state(chains[m_lo].curr[b])
                    st_hi = _serialize_single_box_state(chains[m_hi].curr[b])
                    n_lo, n_hi = int(chains[m_lo].N_curr[b]), int(chains[m_hi].N_curr[b])
                    u_lo, u_hi = float(chains[m_lo].U_tgt_curr[b]), float(chains[m_hi].U_tgt_curr[b])
                    updates_by_m.setdefault(m_lo, {})[b] = (st_hi, n_hi, u_hi)
                    updates_by_m.setdefault(m_hi, {})[b] = (st_lo, n_lo, u_lo)
                for m_upd, box_map in updates_by_m.items():
                    chains[m_upd].apply_swapped_boxes(box_map)

        if next_log > 0 and cyc_done >= next_log:
            elapsed = time.perf_counter() - t0
            rates = [
                f"{tot_swap_acc_up[m] / max(1, tot_swap_att_up[m]) * 100:.0f}%"
                for m in range(M - 1)
            ]
            means = [f"{float(np.mean(chains[m].N_curr)):.1f}" for m in range(M)]
            print(
                f"  [REMC cyc {cyc_done:4d}/{total_cycles}] <N>={means} | "
                f"SwapRates={rates} ({elapsed:.1f}s)"
            )
            while next_log <= cyc_done:
                next_log += int(log_every)

    wall_s = float(time.perf_counter() - t0)
    results = []
    for m in range(M):
        res = chains[m].finalize(elapsed_s=wall_s)
        conv_mmol_g = 1000.0 / float(props_sorted[m].fw_mass_gmol)
        hist = np.asarray(res["box_N_history"], dtype=np.float64) * conv_mmol_g
        prod_hist = hist[int(eq_cycles):]
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
            mean_h1 = 0.0
            mean_h2 = 0.0
            drift = 0.0

        att_p = int(prod_swap_att_up[m])
        acc_p = int(prod_swap_acc_up[m])
        att_t = int(tot_swap_att_up[m])
        acc_t = int(tot_swap_acc_up[m])
        res["P_bar"] = float(p_sorted[m])
        res["T_K"] = float(T_K)
        res["fugacity_Pa"] = float(fugacities_Pa[m])
        res["fw_mass_gmol"] = float(props_sorted[m].fw_mass_gmol)
        res["first_half_uptake_mmol_g"] = mean_h1
        res["second_half_uptake_mmol_g"] = mean_h2
        res["stationarity_drift_mmol_g"] = drift
        res["mlip_evals"] = 0 if grid_only else int(res["target_evals"])
        res["parallel_mode"] = "remc"
        res["remc_swap_every"] = int(swap_stride)
        res["remc_n_pressures"] = int(M)
        res["remc_isotherm_wall_s"] = wall_s
        res["remc_swap_att_up"] = att_p
        res["remc_swap_acc_up"] = acc_p
        res["remc_swap_rate_up"] = (float(acc_p) / float(att_p)) if att_p > 0 else (0.0 if m < M - 1 else None)
        res["remc_total_swap_att_up"] = att_t
        res["remc_total_swap_acc_up"] = acc_t
        res["remc_total_swap_rate_up"] = (float(acc_t) / float(att_t)) if att_t > 0 else (0.0 if m < M - 1 else None)
        res["diagnostics"] = evaluate_v2_convergence(
            prod_N_history=res["prod_N_history"],
            prod_U_history_kJ=res["prod_U_history"],
            conv_mmol_g=conv_mmol_g,
            T_K=float(T_K),
            prod_ins_acc=res["prod_ins_acc"],
            prod_del_acc=res["prod_del_acc"],
            prod_ins_att=res["prod_ins_att"],
            prod_del_att=res["prod_del_att"],
            prod_outer_acc_rate=res["prod_outer_acc_rate"],
            uptake_sem_mmol_g=res["uptake_sem_mmol_g"],
            acceptance_breakdown=res.get("acceptance_breakdown"),
            oms_occ_history=res.get("prod_oms_occ_history") or None,
        )
        results.append(res)
    return results
