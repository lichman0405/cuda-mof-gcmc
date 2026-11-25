"""Hybrid GPU-GCMC + MLIP Nested Monte Carlo Engine (ctypes wrapper for mof_gcmc_proposer.dll)."""

import ctypes
import math
import os
import time
from pathlib import Path
from typing import Optional
import numpy as np

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
        p_ins_del: float = 0.60,
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
        p_ins_del: float = 0.60,
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

    def set_mc_params(self, cbmc_k: int = 128, p_ins_del_total: float = 0.60) -> None:
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
        ov = ctypes.c_int(0)
        u = self.lib.proposer_eval_energy(
            self.handle, n,
            state["cx"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            state["cy"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            state["cz"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            state["ux"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            state["uy"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            state["uz"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
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
        ov = ctypes.c_int(0)
        u = self.lib.proposer_eval_energy_quat(
            self.handle, n,
            np.asarray(state["cx"], dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            np.asarray(state["cy"], dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            np.asarray(state["cz"], dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            np.asarray(state["qw"], dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            np.asarray(state["qx"], dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            np.asarray(state["qy"], dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            np.asarray(state["qz"], dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
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
        for attr in ("model_paths", "model_path"):
            val = getattr(self.mace_calc, attr, None)
            if val:
                self.model_file_path = str(val[0]) if isinstance(val, (list, tuple)) else str(val)
                break
        if self.model_file_path is None and os.path.exists(str(model)):
            self.model_file_path = str(model)
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


def _serialize_box_states(states_list) -> list:
    out = []
    for st in states_list:
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
        out.append(d_st)
    return out


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
):
    """Exact Grand-Canonical Nested MC loop (with optional grid-only ablation mode) and bit-exact resume."""
    rng = np.random.default_rng(seed)
    n_b = proposer.n_boxes
    T_K = proposer.T_K
    delta_eps_K = 0.0 if grid_only else float(getattr(proposer, "delta_eps_K", 0.0))
    conv_mmol_g = 1000.0 / proposer.fw_mass_gmol
    use_quat = hasattr(proposer, "propose_k_quat") and hasattr(proposer, "get_accepted_quat")
    total_cycles = int(eq_cycles) + int(prod_cycles)

    oms_inv_cell = None
    if oms_metal_pos is not None and oms_cell_rows is not None and len(oms_metal_pos) > 0:
        oms_metal_pos = np.asarray(oms_metal_pos, dtype=np.float64)
        oms_cell_rows = np.asarray(oms_cell_rows, dtype=np.float64)
        oms_inv_cell = np.linalg.inv(oms_cell_rows)
    else:
        oms_metal_pos = None

    if resume_state is not None:
        if int(resume_state["n_boxes"]) != n_b:
            raise ValueError(f"Resume state n_boxes={resume_state['n_boxes']} != {n_b}")
        if abs(float(resume_state["delta_eps_K"]) - delta_eps_K) > 1e-9:
            raise ValueError(f"Resume state delta_eps_K={resume_state['delta_eps_K']} != {delta_eps_K}")
        if bool(resume_state.get("grid_only", False)) != bool(grid_only):
            raise ValueError(f"Resume state grid_only={resume_state.get('grid_only', False)} != {grid_only}")
        if int(resume_state["eq_cycles"]) != int(eq_cycles):
            raise ValueError(f"Resume state eq_cycles={resume_state['eq_cycles']} != {eq_cycles}")
        if not use_quat:
            raise RuntimeError("True checkpoint resume requires quaternion proposer API")
        rng.bit_generator.state = resume_state["py_rng_state"]
        restored_states = []
        for st in resume_state["curr_states"]:
            restored_states.append({
                "N": int(st["N"]),
                "cx": np.asarray(st["cx"], dtype=np.float64),
                "cy": np.asarray(st["cy"], dtype=np.float64),
                "cz": np.asarray(st["cz"], dtype=np.float64),
                "qw": np.asarray(st["qw"], dtype=np.float64),
                "qx": np.asarray(st["qx"], dtype=np.float64),
                "qy": np.asarray(st["qy"], dtype=np.float64),
                "qz": np.asarray(st["qz"], dtype=np.float64),
            })
        proposer.set_accepted_quat(restored_states)
        proposer.set_rng_states(resume_state["cuda_rng_states"])
        curr = proposer.get_accepted_quat()
        U_tgt_curr = np.asarray(resume_state["U_tgt_curr"], dtype=np.float64).copy()
        U_ref_curr = np.asarray(resume_state["U_ref_curr"], dtype=np.float64).copy()
        N_curr = np.asarray(resume_state["N_curr"], dtype=np.int32).copy()
        for b in range(n_b):
            if int(curr[b]["N"]) != int(N_curr[b]):
                raise RuntimeError(f"Resume N mismatch in box {b}: {curr[b]['N']} vs {N_curr[b]}")
            if abs(float(curr[b]["U_ref_K"]) - float(U_ref_curr[b])) > 1e-6:
                raise RuntimeError(
                    f"Resume U_ref_K mismatch in box {b}: {curr[b]['U_ref_K']} vs {U_ref_curr[b]}"
                )
        start_cyc = int(resume_state["completed_cycles"])
        acc_outer = np.asarray(resume_state["acc_outer"], dtype=np.int64).copy()
        prod_acc_outer = np.asarray(resume_state["prod_acc_outer"], dtype=np.int64).copy()
        inner_acc_tot = np.asarray(resume_state["inner_acc_tot"], dtype=np.int64).copy()
        prod_ins_att = int(resume_state["prod_ins_att"])
        prod_ins_acc = int(resume_state["prod_ins_acc"])
        prod_del_att = int(resume_state["prod_del_att"])
        prod_del_acc = int(resume_state["prod_del_acc"])
        acc_empty_to_empty = int(resume_state["acc_empty_to_empty"])
        acc_identity_nonempty = int(resume_state["acc_identity_nonempty"])
        acc_same_N_changed = int(resume_state["acc_same_N_changed"])
        acc_dN_changed = int(resume_state["acc_dN_changed"])
        prod_rejected = int(resume_state["prod_rejected"])
        prod_total_trials = int(resume_state["prod_total_trials"])
        nested_trials = int(resume_state.get("nested_trials", prod_total_trials))
        nested_accepted = int(resume_state.get("nested_accepted", int(np.sum(prod_acc_outer))))
        sum_N = np.asarray(resume_state["sum_N"], dtype=np.float64).copy()
        sum_N2 = np.asarray(resume_state["sum_N2"], dtype=np.float64).copy()
        sum_U = np.asarray(resume_state["sum_U"], dtype=np.float64).copy()
        sum_UN = np.asarray(resume_state["sum_UN"], dtype=np.float64).copy()
        trajectory = list(resume_state.get("trajectory", []))
        box_N_history = [list(row) for row in resume_state["box_N_history"]]
        prod_N_history = [list(row) for row in resume_state["prod_N_history"]]
        prod_U_history = [list(row) for row in resume_state["prod_U_history"]]
        prod_U_ref_history = [list(row) for row in resume_state["prod_U_ref_history"]]
        prod_accepted_history = [list(row) for row in resume_state["prod_accepted_history"]]
        prod_dN_history = [list(row) for row in resume_state["prod_dN_history"]]
        prod_proposed_dN_history = [list(row) for row in resume_state["prod_proposed_dN_history"]]
        prod_oms_occ_history = [list(row) for row in resume_state.get("prod_oms_occ_history", [])]
        prod_snapshots = list(resume_state.get("prod_snapshots", []))
        elapsed_prior = float(resume_state.get("elapsed_s", 0.0))
    else:
        curr = proposer.get_accepted_quat() if use_quat else proposer.get_accepted()
        U_tgt_curr = np.zeros(n_b, dtype=np.float64)
        U_ref_curr = np.zeros(n_b, dtype=np.float64)
        N_curr = np.zeros(n_b, dtype=np.int32)
        if grid_only:
            init_evals = [(float(curr[b]["U_ref_K"]), False) for b in range(n_b)]
        elif batch_target_fn is not None:
            init_evals = batch_target_fn(curr)
        else:
            init_evals = [target_fn(curr[b]) for b in range(n_b)]
        for b in range(n_b):
            ut, ov = init_evals[b]
            if not math.isfinite(float(ut)):
                raise RuntimeError(f"Non-finite initial target energy in box {b}: {ut}")
            if not math.isfinite(float(curr[b]["U_ref_K"])):
                raise RuntimeError(f"Non-finite initial reference energy in box {b}: {curr[b]['U_ref_K']}")
            if ov:
                raise RuntimeError(f"Initial warm-started state of box {b} has target overlap")
            U_tgt_curr[b] = float(ut)
            U_ref_curr[b] = float(curr[b]["U_ref_K"])
            N_curr[b] = curr[b]["N"]

        start_cyc = 0
        acc_outer = np.zeros(n_b, dtype=np.int64)
        prod_acc_outer = np.zeros(n_b, dtype=np.int64)
        inner_acc_tot = np.zeros(n_b, dtype=np.int64)
        prod_ins_att = 0
        prod_ins_acc = 0
        prod_del_att = 0
        prod_del_acc = 0
        acc_empty_to_empty = 0
        acc_identity_nonempty = 0
        acc_same_N_changed = 0
        acc_dN_changed = 0
        prod_rejected = 0
        prod_total_trials = 0
        nested_trials = 0
        nested_accepted = 0

        sum_N = np.zeros(n_b, dtype=np.float64)
        sum_N2 = np.zeros(n_b, dtype=np.float64)
        sum_U = np.zeros(n_b, dtype=np.float64)
        sum_UN = np.zeros(n_b, dtype=np.float64)

        trajectory = []
        box_N_history = []
        prod_N_history = []
        prod_U_history = []
        prod_U_ref_history = []
        prod_accepted_history = []
        prod_dN_history = []
        prod_proposed_dN_history = []
        prod_oms_occ_history = []
        prod_snapshots = []
        elapsed_prior = 0.0

    t0 = time.perf_counter()

    def _build_ckpt(completed_cyc: int) -> dict:
        cuda_rng = proposer.get_rng_states().tolist() if hasattr(proposer, "get_rng_states") else []
        return {
            "completed_cycles": int(completed_cyc),
            "eq_cycles": int(eq_cycles),
            "prod_cycles": int(prod_cycles),
            "n_boxes": int(n_b),
            "delta_eps_K": float(delta_eps_K),
            "grid_only": bool(grid_only),
            "py_rng_state": rng.bit_generator.state,
            "cuda_rng_states": [int(x) for x in cuda_rng],
            "curr_states": _serialize_box_states(curr),
            "U_tgt_curr": [float(x) for x in U_tgt_curr],
            "U_ref_curr": [float(x) for x in U_ref_curr],
            "N_curr": [int(x) for x in N_curr],
            "acc_outer": [int(x) for x in acc_outer],
            "prod_acc_outer": [int(x) for x in prod_acc_outer],
            "inner_acc_tot": [int(x) for x in inner_acc_tot],
            "prod_ins_att": int(prod_ins_att),
            "prod_ins_acc": int(prod_ins_acc),
            "prod_del_att": int(prod_del_att),
            "prod_del_acc": int(prod_del_acc),
            "acc_empty_to_empty": int(acc_empty_to_empty),
            "acc_identity_nonempty": int(acc_identity_nonempty),
            "acc_same_N_changed": int(acc_same_N_changed),
            "acc_dN_changed": int(acc_dN_changed),
            "prod_rejected": int(prod_rejected),
            "prod_total_trials": int(prod_total_trials),
            "nested_trials": int(nested_trials),
            "nested_accepted": int(nested_accepted),
            "sum_N": [float(x) for x in sum_N],
            "sum_N2": [float(x) for x in sum_N2],
            "sum_U": [float(x) for x in sum_U],
            "sum_UN": [float(x) for x in sum_UN],
            "box_N_history": box_N_history,
            "prod_N_history": prod_N_history,
            "prod_U_history": prod_U_history,
            "prod_U_ref_history": prod_U_ref_history,
            "prod_accepted_history": prod_accepted_history,
            "prod_dN_history": prod_dN_history,
            "prod_proposed_dN_history": prod_proposed_dN_history,
            "prod_oms_occ_history": prod_oms_occ_history,
            "prod_snapshots": prod_snapshots,
            "trajectory": trajectory,
            "elapsed_s": float(elapsed_prior + (time.perf_counter() - t0)),
        }

    for cyc in range(start_cyc, total_cycles):
        mask = np.zeros(n_b, dtype=np.int32)
        cyc_prop_dN = [0] * n_b
        cyc_net_dN = [0] * n_b

        trials = proposer.propose_k_quat(k_steps) if use_quat else proposer.propose_k(k_steps)
        if grid_only:
            tri_evals = [(float(trials[b]["U_ref_K"]), False) for b in range(n_b)]
        elif batch_target_fn is not None:
            tri_evals = batch_target_fn(trials)
        else:
            tri_evals = [target_fn(trials[b]) for b in range(n_b)]

        for b in range(n_b):
            inner_acc_tot[b] += trials[b]["inner_acc"]
            if not math.isfinite(float(trials[b]["U_ref_K"])):
                raise RuntimeError(f"Non-finite trial reference energy at cycle {cyc}, box {b}: {trials[b]['U_ref_K']}")
            n_prev = int(N_curr[b])
            dU_ref = trials[b]["U_ref_K"] - U_ref_curr[b]
            dN = int(trials[b]["N"]) - n_prev
            cyc_prop_dN[b] = dN
            u_tgt_tri, ov = tri_evals[b]
            if not ov and not math.isfinite(float(u_tgt_tri)):
                raise RuntimeError(f"Non-finite trial target energy at cycle {cyc}, box {b}: {u_tgt_tri}")
            accepted = False
            if grid_only:
                accepted = True
            elif not ov:
                dU_tgt = u_tgt_tri - U_tgt_curr[b]
                ddU = dU_tgt - dU_ref - delta_eps_K * float(dN)
                if not math.isfinite(float(ddU)):
                    raise RuntimeError(f"Non-finite Nested MC ddU at cycle {cyc}, box {b}: {ddU}")
                if ddU <= 0.0:
                    accepted = True
                else:
                    arg = -ddU / T_K
                    if arg > -50.0 and rng.random() < math.exp(arg):
                        accepted = True
            if cyc >= eq_cycles:
                nested_trials += 1
                same_cfg = _states_equal(trials[b], curr[b])
                prod_total_trials += 1
                if dN > 0:
                    prod_ins_att += 1
                    if accepted:
                        prod_ins_acc += 1
                elif dN < 0:
                    prod_del_att += 1
                    if accepted:
                        prod_del_acc += 1
                if not accepted:
                    prod_rejected += 1
                elif n_prev == 0 and int(trials[b]["N"]) == 0:
                    acc_empty_to_empty += 1
                elif same_cfg:
                    acc_identity_nonempty += 1
                elif dN == 0:
                    acc_same_N_changed += 1
                else:
                    acc_dN_changed += 1

            if accepted:
                mask[b] = 1
                U_tgt_curr[b] = u_tgt_tri
                U_ref_curr[b] = trials[b]["U_ref_K"]
                N_curr[b] = trials[b]["N"]
                curr[b] = trials[b]
                cyc_net_dN[b] = dN
                acc_outer[b] += 1
                if cyc >= eq_cycles:
                    prod_acc_outer[b] += 1
                    nested_accepted += 1

        proposer.commit(mask)

        if cyc >= eq_cycles:
            prod_N_history.append([int(x) for x in N_curr])
            prod_U_history.append([float(U_tgt_curr[b] * K_TO_KJMOL) for b in range(n_b)])
            prod_U_ref_history.append([float(U_ref_curr[b] * K_TO_KJMOL) for b in range(n_b)])
            prod_accepted_history.append([int(mask[b]) for b in range(n_b)])
            prod_dN_history.append(cyc_net_dN)
            prod_proposed_dN_history.append(cyc_prop_dN)
            if oms_metal_pos is not None:
                prod_oms_occ_history.append([
                    count_oms_occupied(
                        curr[b], oms_metal_pos, oms_cell_rows,
                        guest_local_pos=guest_local_pos, cutoff_A=oms_cutoff_A, inv_cell_rows=oms_inv_cell,
                    )
                    for b in range(n_b)
                ])
            prod_step = cyc - eq_cycles + 1
            if snapshot_stride > 0 and (prod_step % snapshot_stride == 0 or cyc == total_cycles - 1):
                prod_snapshots.append({
                    "prod_cycle": int(prod_step),
                    "states": _serialize_box_states(curr),
                })
            for b in range(n_b):
                dn = float(N_curr[b])
                du = float(U_tgt_curr[b])
                sum_N[b] += dn
                sum_N2[b] += dn * dn
                sum_U[b] += du
                sum_UN[b] += du * dn

        box_N_history.append([int(x) for x in N_curr])
        if n_b == 1:
            trajectory.append({
                "cycle": cyc,
                "in_prod": cyc >= eq_cycles,
                "N": int(N_curr[0]),
                "uptake_mmol_g": float(N_curr[0] * conv_mmol_g),
                "U_tgt_kJ": float(U_tgt_curr[0] * K_TO_KJMOL),
                "U_ref_kJ": float(U_ref_curr[0] * K_TO_KJMOL),
                "accepted": bool(mask[0]),
            })
        if checkpoint_fn is not None and checkpoint_every > 0 and ((cyc + 1) % checkpoint_every == 0):
            checkpoint_fn(_build_ckpt(cyc + 1))
        if log_every > 0 and ((cyc + 1) % log_every == 0):
            elapsed = elapsed_prior + (time.perf_counter() - t0)
            mean_up_now = float(np.mean(N_curr) * conv_mmol_g)
            print(
                f"  [NestedMC cyc {cyc+1:4d}/{total_cycles}] N={N_curr.tolist()} ({mean_up_now:.2f} mmol/g) "
                f"Acc={np.mean(acc_outer)/(cyc+1)*100:5.1f}% ({elapsed:.1f}s)"
            )

    dt = elapsed_prior + (time.perf_counter() - t0)
    final_ckpt = _build_ckpt(total_cycles)
    if checkpoint_fn is not None and total_cycles > start_cyc:
        checkpoint_fn(final_ckpt)

    prod_denom = max(1, prod_cycles)
    tot_denom = max(1, total_cycles)
    box_uptakes = (sum_N / prod_denom) * conv_mmol_g if prod_cycles > 0 else np.zeros(n_b, dtype=np.float64)
    box_qst = []
    box_molar_bind = []
    box_molar_qst = []
    for b in range(n_b):
        mN = sum_N[b] / prod_denom if prod_cycles > 0 else 0.0
        mN2 = sum_N2[b] / prod_denom if prod_cycles > 0 else 0.0
        mU = sum_U[b] / prod_denom if prod_cycles > 0 else 0.0
        mUN = sum_UN[b] / prod_denom if prod_cycles > 0 else 0.0
        varN = mN2 - mN * mN
        mb = (-mU / mN) * K_TO_KJMOL if mN > 1e-12 else 0.0
        mq = (T_K - mU / mN) * K_TO_KJMOL if mN > 1e-12 else 0.0
        box_molar_bind.append(mb)
        box_molar_qst.append(mq)
        if varN > 1e-12:
            q_K = T_K - (mUN - mU * mN) / varN
            box_qst.append(float(q_K * K_TO_KJMOL))
        else:
            box_qst.append(None)

    mean_up = float(np.mean(box_uptakes))
    sem_up = float(np.std(box_uptakes, ddof=1) / math.sqrt(n_b)) if n_b > 1 else 0.0
    valid_box_qst = [q for q in box_qst if q is not None]
    per_box_mean_qst = float(np.mean(valid_box_qst)) if valid_box_qst else None
    per_box_sem_qst = (
        float(np.std(valid_box_qst, ddof=1) / math.sqrt(len(valid_box_qst)))
        if len(valid_box_qst) > 1 else (0.0 if len(valid_box_qst) == 1 else None)
    )
    box_moments = [
        {
            "n_samples": int(prod_cycles),
            "sum_N": float(sum_N[b]),
            "sum_N2": float(sum_N2[b]),
            "sum_U": float(sum_U[b]),
            "sum_UN": float(sum_UN[b]),
        }
        for b in range(n_b)
    ]

    from mcmc_diagnostics import compute_independent_qst
    qst_res = compute_independent_qst(
        prod_N_history, prod_U_history, T_K=T_K,
        prod_ins_acc=prod_ins_acc, prod_del_acc=prod_del_acc,
    )
    mean_qst = qst_res["qst_fluctuation_kJ_mol"]
    sem_qst = qst_res["qst_fluctuation_sem_kJ_mol"]
    mean_mb = float(np.mean(box_molar_bind))
    mean_mq = float(np.mean(box_molar_qst))
    sem_mq = float(np.std(box_molar_qst, ddof=1) / math.sqrt(n_b)) if n_b > 1 else 0.0

    acceptance_breakdown = {
        "total_trials": int(prod_total_trials),
        "rejected": int(prod_rejected),
        "acc_empty_to_empty": int(acc_empty_to_empty),
        "acc_identity_nonempty": int(acc_identity_nonempty),
        "acc_same_N_changed": int(acc_same_N_changed),
        "acc_dN_changed": int(acc_dN_changed),
        "effective_move_acc_rate": float((acc_same_N_changed + acc_dN_changed) / max(1, prod_total_trials)),
        "nested_trials": int(nested_trials),
        "nested_accepted": int(nested_accepted),
    }

    ess_stats = compute_ess_stats(prod_N_history, prod_U_history, dt)
    final_acc = proposer.get_accepted_quat() if use_quat else proposer.get_accepted()
    final_states = _serialize_box_states(final_acc)

    return {
        "n_boxes": n_b,
        "k_steps": k_steps,
        "grid_only": bool(grid_only),
        "eq_cycles": eq_cycles,
        "prod_cycles": prod_cycles,
        "delta_eps_kJ_mol": float(delta_eps_K * K_TO_KJMOL),
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
        "outer_acc_rate": float(np.mean(acc_outer) / tot_denom),
        "prod_outer_acc_rate": float(np.mean(prod_acc_outer) / prod_denom) if prod_cycles > 0 else 0.0,
        "prod_ins_att": int(prod_ins_att),
        "prod_ins_acc": int(prod_ins_acc),
        "prod_del_att": int(prod_del_att),
        "prod_del_acc": int(prod_del_acc),
        "acceptance_breakdown": acceptance_breakdown,
        "mean_inner_acc_per_burst": float(np.mean(inner_acc_tot) / tot_denom),
        "effective_mc_steps_per_target_eval": float(k_steps * np.mean(acc_outer) / tot_denom),
        "box_N_history": box_N_history,
        "prod_N_history": prod_N_history,
        "prod_U_history": prod_U_history,
        "prod_U_ref_history": prod_U_ref_history,
        "prod_accepted_history": prod_accepted_history,
        "prod_dN_history": prod_dN_history,
        "prod_proposed_dN_history": prod_proposed_dN_history,
        "prod_oms_occ_history": prod_oms_occ_history,
        "prod_snapshots": prod_snapshots,
        "trajectory": trajectory,
        "final_states": final_states,
        "final_cuda_rng_states": final_ckpt["cuda_rng_states"],
    }
