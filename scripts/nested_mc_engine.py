"""Hybrid GPU-GCMC + MLIP Nested Monte Carlo Engine (ctypes wrapper for mof_gcmc_proposer.dll)."""

import ctypes
import math
import os
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
K_TO_KJMOL = 8.31446261815324e-3
EV_TO_K = 11604.51812155008


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


class CudaGcmcProposer:
    """Zero-I/O ctypes wrapper around build/mof_gcmc_proposer.dll."""

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
        dll_path = BUILD / "mof_gcmc_proposer.dll"
        if not dll_path.exists():
            raise FileNotFoundError(f"Missing {dll_path}. Run `cmd /c build.bat proposer` first.")
        self.lib = ctypes.CDLL(str(dll_path))
        self._bind()

        self.n_boxes = int(n_boxes)
        self.max_mol = int(min(256, max(16, max_mol)))
        self.T_K = float(T_K)
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

        self.fw_mass_gmol = float(self.lib.proposer_get_framework_mass(self.handle))
        self.f_Pa = float(self.lib.proposer_get_fugacity_Pa(self.handle))
        self.r_cut_gg = float(self.lib.proposer_get_rcut_gg(self.handle))
        h9 = (ctypes.c_double * 9)()
        self.lib.proposer_get_cell_H(self.handle, h9)
        self.H_cols = np.array(list(h9), dtype=np.float64).reshape(3, 3)

        # Reusable host transfer buffers
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
        self._mask = np.zeros(nb, dtype=np.int32)

    def _bind(self):
        L = self.lib
        L.proposer_create.argtypes = [
            ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        L.proposer_create.restype = ctypes.c_void_p
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
        L.proposer_run_direct_benchmark.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        L.proposer_run_direct_benchmark.restype = ctypes.c_int

    def close(self):
        if getattr(self, "handle", None):
            self.lib.proposer_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()

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
        return u_int_K, False


def run_nested_mc(
    proposer: CudaGcmcProposer,
    target_fn,
    k_steps: int = 10,
    eq_cycles: int = 200,
    prod_cycles: int = 500,
    seed: int = 20261124,
    log_every: int = 0,
):
    """Exact Grand-Canonical Nested Monte Carlo loop."""
    rng = np.random.default_rng(seed)
    n_b = proposer.n_boxes
    T_K = proposer.T_K
    conv_mmol_g = 1000.0 / proposer.fw_mass_gmol

    curr = proposer.get_accepted()
    U_tgt_curr = np.zeros(n_b, dtype=np.float64)
    U_ref_curr = np.zeros(n_b, dtype=np.float64)
    N_curr = np.zeros(n_b, dtype=np.int32)
    for b in range(n_b):
        ut, ov = target_fn(curr[b])
        if ov:
            raise RuntimeError(f"Initial warm-started state of box {b} has target overlap")
        U_tgt_curr[b] = ut
        U_ref_curr[b] = curr[b]["U_ref_K"]
        N_curr[b] = curr[b]["N"]

    total_cycles = eq_cycles + prod_cycles
    acc_outer = np.zeros(n_b, dtype=np.int64)
    prod_acc_outer = np.zeros(n_b, dtype=np.int64)
    inner_acc_tot = np.zeros(n_b, dtype=np.int64)

    sum_N = np.zeros(n_b, dtype=np.float64)
    sum_N2 = np.zeros(n_b, dtype=np.float64)
    sum_U = np.zeros(n_b, dtype=np.float64)
    sum_UN = np.zeros(n_b, dtype=np.float64)

    trajectory = []
    t0 = time.perf_counter()

    for cyc in range(total_cycles):
        trials = proposer.propose_k(k_steps)
        mask = np.zeros(n_b, dtype=np.int32)
        for b in range(n_b):
            inner_acc_tot[b] += trials[b]["inner_acc"]
            dU_ref = trials[b]["U_ref_K"] - U_ref_curr[b]
            u_tgt_tri, ov = target_fn(trials[b])
            accepted = False
            if not ov:
                dU_tgt = u_tgt_tri - U_tgt_curr[b]
                ddU = dU_tgt - dU_ref
                if ddU <= 0.0:
                    accepted = True
                else:
                    arg = -ddU / T_K
                    if arg > -50.0 and rng.random() < math.exp(arg):
                        accepted = True
            if accepted:
                mask[b] = 1
                U_tgt_curr[b] = u_tgt_tri
                U_ref_curr[b] = trials[b]["U_ref_K"]
                N_curr[b] = trials[b]["N"]
                acc_outer[b] += 1
                if cyc >= eq_cycles:
                    prod_acc_outer[b] += 1

        proposer.commit(mask)

        if cyc >= eq_cycles:
            for b in range(n_b):
                dn = float(N_curr[b])
                du = float(U_tgt_curr[b])
                sum_N[b] += dn
                sum_N2[b] += dn * dn
                sum_U[b] += du
                sum_UN[b] += du * dn

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
        if log_every > 0 and ((cyc + 1) % log_every == 0):
            elapsed = time.perf_counter() - t0
            print(
                f"  [NestedMC cyc {cyc+1:4d}/{total_cycles}] N={N_curr.tolist()} "
                f"Acc={np.mean(acc_outer)/(cyc+1)*100:5.1f}% U_tgt={U_tgt_curr[0]*K_TO_KJMOL:7.2f} kJ/mol ({elapsed:.1f}s)"
            )

    dt = time.perf_counter() - t0
    box_uptakes = (sum_N / prod_cycles) * conv_mmol_g
    box_qst = []
    for b in range(n_b):
        mN = sum_N[b] / prod_cycles
        mN2 = sum_N2[b] / prod_cycles
        mU = sum_U[b] / prod_cycles
        mUN = sum_UN[b] / prod_cycles
        varN = mN2 - mN * mN
        if varN > 1e-12:
            q_K = T_K - (mUN - mU * mN) / varN
            box_qst.append(q_K * K_TO_KJMOL)
        elif mN > 0:
            box_qst.append((T_K - mU / mN) * K_TO_KJMOL)
        else:
            box_qst.append(0.0)

    mean_up = float(np.mean(box_uptakes))
    sem_up = float(np.std(box_uptakes, ddof=1) / math.sqrt(n_b)) if n_b > 1 else 0.0
    mean_qst = float(np.mean(box_qst))
    sem_qst = float(np.std(box_qst, ddof=1) / math.sqrt(n_b)) if n_b > 1 else 0.0

    return {
        "n_boxes": n_b,
        "k_steps": k_steps,
        "eq_cycles": eq_cycles,
        "prod_cycles": prod_cycles,
        "elapsed_s": dt,
        "uptake_mmol_g": mean_up,
        "uptake_sem_mmol_g": sem_up,
        "box_uptakes_mmol_g": box_uptakes.tolist(),
        "Qst_kJ_mol": mean_qst,
        "Qst_sem_kJ_mol": sem_qst,
        "box_qst_kJ_mol": box_qst,
        "outer_acc_rate": float(np.mean(acc_outer) / total_cycles),
        "prod_outer_acc_rate": float(np.mean(prod_acc_outer) / prod_cycles),
        "mean_inner_acc_per_burst": float(np.mean(inner_acc_tot) / total_cycles),
        "effective_mc_steps_per_target_eval": float(k_steps * np.mean(acc_outer) / total_cycles),
        "trajectory": trajectory,
    }
