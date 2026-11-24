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

        self.fw_mass_gmol = float(self.lib.proposer_get_framework_mass(self.handle))
        self.f_Pa = float(self.lib.proposer_get_fugacity_Pa(self.handle))
        self.r_cut_gg = float(self.lib.proposer_get_rcut_gg(self.handle))
        self.min_perp_width_A = float(self.lib.proposer_get_min_perp_width(self.handle))
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

    def set_chem_pot_shift(self, delta_eps_kJ_mol: float):
        """Set inner-proposer chemical-potential bias delta_eps = <(U_tgt - U_ref)/N> (kJ/mol)."""
        self.delta_eps_K = float(delta_eps_kJ_mol) / K_TO_KJMOL
        rc = self.lib.proposer_set_chem_pot_shift(self.handle, self.delta_eps_K)
        if rc != 0:
            raise RuntimeError(f"proposer_set_chem_pot_shift failed with code {rc}")

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
    """Fast batched energy-only MACE-MP-0 + exact GPU-edge DFT-D3(BJ) evaluator."""

    def __init__(self, cif_path: str, supercell=(1, 1, 2), model="medium", device="cuda", dispersion: bool = True, dispersion_xc: str = "pbe"):
        import torch
        from ase import Atoms
        from ase.io import read as ase_read
        from mace.calculators import mace_mp
        from mace import data as mace_data
        from mace.tools import torch_geometric, torch_tools

        self.torch = torch
        self.Atoms = Atoms
        self.mace_data = mace_data
        self.torch_geometric = torch_geometric
        self.torch_tools = torch_tools
        self.device = torch.device(device)
        self.use_dispersion = bool(dispersion)
        self.dispersion_xc = str(dispersion_xc)

        uc = ase_read(str(cif_path))
        self.uc = uc
        self.fw = uc.repeat(tuple(supercell))
        self.n_fw = len(self.fw)
        calc = mace_mp(
            model=model, dispersion=self.use_dispersion, dispersion_xc=self.dispersion_xc,
            default_dtype="float32", device=device
        )
        if self.use_dispersion:
            self.mace_calc = calc.mixer.calcs[0]
            self.d3_calc = calc.mixer.calcs[1]
        else:
            self.mace_calc = calc
            self.d3_calc = None
        self.mace_model = self.mace_calc.models[0]
        self.mace_model.eval()
        self.keyspec = mace_data.KeySpecification(
            info_keys=self.mace_calc.info_keys,
            arrays_keys=self.mace_calc.arrays_keys,
        )

        # Precompute exact D3 periodic shifts and static rigid framework-framework edges once on GPU
        if self.use_dispersion:
            self.rc_d3 = float(self.d3_calc.cutoff)
            self.cell_np = np.array(self.fw.get_cell(), dtype=np.float32)
            self.cell_t = torch.tensor(self.cell_np, device=self.device, dtype=torch.float32)
            inv_cell = torch.linalg.inv(self.cell_t)
            perp_w = 1.0 / torch.linalg.norm(inv_cell, dim=0)
            n_rep = [int(math.ceil(self.rc_d3 / float(w))) + 1 for w in perp_w]
            s0 = torch.arange(-n_rep[0], n_rep[0] + 1, device=self.device)
            s1 = torch.arange(-n_rep[1], n_rep[1] + 1, device=self.device)
            s2 = torch.arange(-n_rep[2], n_rep[2] + 1, device=self.device)
            S_all = torch.cartesian_prod(s0, s1, s2).to(torch.float32)
            self.S_all = S_all
            self.shift_vecs_d3 = S_all @ self.cell_t
            self.zero_shift_idx = int(torch.nonzero((S_all.abs().sum(dim=1) == 0), as_tuple=False)[0].item())

            self.fw_pos_t = torch.tensor(self.fw.get_positions(), device=self.device, dtype=torch.float32)
            self.fw_Z_t = torch.tensor(self.fw.get_atomic_numbers(), device=self.device, dtype=torch.long)
            self.pbc_t = torch.tensor([True, True, True], device=self.device, dtype=torch.bool)
            self.co2_Z_unit = torch.tensor([6, 8, 8], device=self.device, dtype=torch.long)

            self.fw_ei, self.fw_sp = self._pair_edges_chunked(
                self.fw_pos_t, self.fw_pos_t, same_set=True, row_offset=0, col_offset=0
            )

        # Reference isolated framework and isolated CO2 energies using identical fast evaluator
        self.e_fw_eV = self._eval_raw_eV_batch([self.fw.copy()], [None])[0]
        co2 = Atoms("CO2", positions=[[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]])
        co2.set_cell([30.0, 30.0, 30.0])
        co2.set_pbc(True)
        co2.calc = calc
        self.e_co2_eV = float(co2.get_potential_energy())
        self.n_evals = 0

    def _pair_edges_chunked(self, pos_row, pos_col, same_set: bool, row_offset: int, col_offset: int, shift_chunk: int = 16):
        torch = self.torch
        nr, nc = pos_row.shape[0], pos_col.shape[0]
        if nr == 0 or nc == 0:
            return (
                torch.empty((2, 0), device=self.device, dtype=torch.long),
                torch.empty((0, 3), device=self.device, dtype=torch.float32),
            )
        rc2 = self.rc_d3 * self.rc_d3
        diff_base = pos_col.unsqueeze(0) - pos_row.unsqueeze(1)  # (nr, nc, 3)
        M = self.shift_vecs_d3.shape[0]
        srcs, dsts, shs = [], [], []
        self_diag = None
        if same_set:
            ii = torch.arange(nr, device=self.device).unsqueeze(1).expand(nr, nc)
            jj = torch.arange(nc, device=self.device).unsqueeze(0).expand(nr, nc)
            self_diag = (ii == jj)
        for s0 in range(0, M, shift_chunk):
            s1 = min(M, s0 + shift_chunk)
            sv = self.shift_vecs_d3[s0:s1]  # (C, 3)
            d = diff_base.unsqueeze(0) + sv.view(-1, 1, 1, 3)
            r2 = (d * d).sum(dim=-1)
            mask = r2 < rc2
            if same_set and s0 <= self.zero_shift_idx < s1:
                mask[self.zero_shift_idx - s0] &= (~self_diag)
            mm, rr, cc = torch.nonzero(mask, as_tuple=True)
            if mm.numel() > 0:
                srcs.append(rr + row_offset)
                dsts.append(cc + col_offset)
                shs.append(sv[mm])
        if not srcs:
            return (
                torch.empty((2, 0), device=self.device, dtype=torch.long),
                torch.empty((0, 3), device=self.device, dtype=torch.float32),
            )
        return torch.stack([torch.cat(srcs), torch.cat(dsts)], dim=0), torch.cat(shs, dim=0)

    def _guest_positions_from_state(self, state):
        n = int(state["N"])
        if n == 0:
            return np.zeros((0, 3), dtype=np.float64)
        bond = 1.160
        cx, cy, cz = state["cx"], state["cy"], state["cz"]
        ux, uy, uz = state["ux"], state["uy"], state["uz"]
        c = np.stack([cx, cy, cz], axis=1)
        u = np.stack([ux, uy, uz], axis=1)
        gp = np.empty((3 * n, 3), dtype=np.float64)
        gp[0::3] = c
        gp[1::3] = c + bond * u
        gp[2::3] = c - bond * u
        return gp

    def _eval_raw_eV_batch(self, atoms_list, guest_pos_list, mace_chunk: int = None):
        if mace_chunk is None:
            mace_chunk = 1 if self.n_fw >= 550 else 2
        torch = self.torch
        # 1. Micro-batched MACE forward (no_grad, compute_force=False) to keep peak VRAM < 3.5 GB
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
                em = self.mace_model(mb.to_dict(), compute_force=False, compute_stress=False)["energy"]
                e_mace_parts.append(em.detach().cpu().numpy().astype(np.float64))
        e_mace_np = np.concatenate(e_mace_parts, axis=0)
        if not self.use_dispersion:
            return e_mace_np

        # 2. Exact GPU PBE-D3(BJ) via cached fw-fw edges + vectorized dynamic edges
        e_d3_np = np.zeros(len(atoms_list), dtype=np.float64)
        for g_idx, gp_np in enumerate(guest_pos_list):
            if gp_np is None or len(gp_np) == 0:
                Z_tot = self.fw_Z_t
                pos_tot = self.fw_pos_t
                ei_tot = self.fw_ei
                sp_tot = self.fw_sp
            else:
                n_g = gp_np.shape[0]
                n_mol = n_g // 3
                g_pos_t = torch.tensor(gp_np, device=self.device, dtype=torch.float32)
                g_Z_t = self.co2_Z_unit.repeat(n_mol)
                Z_tot = torch.cat([self.fw_Z_t, g_Z_t], dim=0)
                pos_tot = torch.cat([self.fw_pos_t, g_pos_t], dim=0)
                # fw -> guest (and bidirectional transpose guest -> fw with negated shift!)
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

        return e_mace_np + e_d3_np

    def eval_batch(self, states):
        non_empty_idx = []
        atoms_list = []
        guest_pos_list = []
        for idx, st in enumerate(states):
            n = int(st["N"])
            if n == 0:
                continue
            gp = self._guest_positions_from_state(st)
            sys_at = self.fw.copy() + self.Atoms("CO2" * n, positions=gp)
            non_empty_idx.append(idx)
            atoms_list.append(sys_at)
            guest_pos_list.append(gp)

        results = [(0.0, False)] * len(states)
        if not non_empty_idx:
            return results
        e_tot = self._eval_raw_eV_batch(atoms_list, guest_pos_list)
        self.n_evals += len(non_empty_idx)
        for k, idx in enumerate(non_empty_idx):
            n = int(states[idx]["N"])
            u_int_K = (float(e_tot[k]) - self.e_fw_eV - n * self.e_co2_eV) * EV_TO_K
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


def _qst_from_sums(n_samples: int, sN: float, sN2: float, sU: float, sUN: float, T_K: float) -> float:
    if n_samples <= 0:
        return 0.0
    inv_m = 1.0 / float(n_samples)
    mN = sN * inv_m
    mN2 = sN2 * inv_m
    mU = sU * inv_m
    mUN = sUN * inv_m
    varN = mN2 - mN * mN
    if varN > 1e-12:
        return float((T_K - (mUN - mU * mN) / varN) * K_TO_KJMOL)
    if mN > 1e-12:
        return float((T_K - mU / mN) * K_TO_KJMOL)
    return 0.0


def compute_pooled_jackknife_qst(box_moments, T_K: float) -> dict:
    """Compute pooled fluctuation Qst and leave-one-box-out Jackknife SEM from per-box raw moments."""
    n_b = len(box_moments)
    if n_b == 0:
        return {"Qst_kJ_mol": 0.0, "Qst_sem_kJ_mol": 0.0, "Qst_loo_kJ_mol": []}

    tot_m = sum(int(m["n_samples"]) for m in box_moments)
    tot_sN = sum(float(m["sum_N"]) for m in box_moments)
    tot_sN2 = sum(float(m["sum_N2"]) for m in box_moments)
    tot_sU = sum(float(m["sum_U"]) for m in box_moments)
    tot_sUN = sum(float(m["sum_UN"]) for m in box_moments)

    q_pooled = _qst_from_sums(tot_m, tot_sN, tot_sN2, tot_sU, tot_sUN, float(T_K))
    if n_b <= 1:
        return {"Qst_kJ_mol": q_pooled, "Qst_sem_kJ_mol": 0.0, "Qst_loo_kJ_mol": [q_pooled] if n_b == 1 else []}

    q_loo = []
    for m in box_moments:
        q_loo.append(
            _qst_from_sums(
                tot_m - int(m["n_samples"]),
                tot_sN - float(m["sum_N"]),
                tot_sN2 - float(m["sum_N2"]),
                tot_sU - float(m["sum_U"]),
                tot_sUN - float(m["sum_UN"]),
                float(T_K),
            )
        )
    q_arr = np.asarray(q_loo, dtype=np.float64)
    mean_loo = float(np.mean(q_arr))
    var_jack = ((n_b - 1.0) / n_b) * float(np.sum((q_arr - mean_loo) ** 2))
    sem_jack = math.sqrt(max(0.0, var_jack))
    return {"Qst_kJ_mol": q_pooled, "Qst_sem_kJ_mol": sem_jack, "Qst_loo_kJ_mol": q_loo}


def run_nested_mc(
    proposer: CudaGcmcProposer,
    target_fn,
    k_steps: int = 10,
    eq_cycles: int = 200,
    prod_cycles: int = 500,
    seed: int = 20261124,
    log_every: int = 0,
    batch_target_fn=None,
):
    """Exact Grand-Canonical Nested Monte Carlo loop (with optional chemical-potential shift & batching)."""
    rng = np.random.default_rng(seed)
    n_b = proposer.n_boxes
    T_K = proposer.T_K
    delta_eps_K = float(getattr(proposer, "delta_eps_K", 0.0))
    conv_mmol_g = 1000.0 / proposer.fw_mass_gmol

    curr = proposer.get_accepted()
    U_tgt_curr = np.zeros(n_b, dtype=np.float64)
    U_ref_curr = np.zeros(n_b, dtype=np.float64)
    N_curr = np.zeros(n_b, dtype=np.int32)
    if batch_target_fn is not None:
        init_evals = batch_target_fn(curr)
    else:
        init_evals = [target_fn(curr[b]) for b in range(n_b)]
    for b in range(n_b):
        ut, ov = init_evals[b]
        if ov:
            raise RuntimeError(f"Initial warm-started state of box {b} has target overlap")
        U_tgt_curr[b] = ut
        U_ref_curr[b] = curr[b]["U_ref_K"]
        N_curr[b] = curr[b]["N"]

    total_cycles = eq_cycles + prod_cycles
    acc_outer = np.zeros(n_b, dtype=np.int64)
    prod_acc_outer = np.zeros(n_b, dtype=np.int64)
    inner_acc_tot = np.zeros(n_b, dtype=np.int64)
    prod_ins_att = 0
    prod_ins_acc = 0
    prod_del_att = 0
    prod_del_acc = 0

    sum_N = np.zeros(n_b, dtype=np.float64)
    sum_N2 = np.zeros(n_b, dtype=np.float64)
    sum_U = np.zeros(n_b, dtype=np.float64)
    sum_UN = np.zeros(n_b, dtype=np.float64)

    trajectory = []
    box_N_history = []
    t0 = time.perf_counter()

    for cyc in range(total_cycles):
        trials = proposer.propose_k(k_steps)
        if batch_target_fn is not None:
            tri_evals = batch_target_fn(trials)
        else:
            tri_evals = [target_fn(trials[b]) for b in range(n_b)]

        mask = np.zeros(n_b, dtype=np.int32)
        for b in range(n_b):
            inner_acc_tot[b] += trials[b]["inner_acc"]
            dU_ref = trials[b]["U_ref_K"] - U_ref_curr[b]
            dN = int(trials[b]["N"]) - int(N_curr[b])
            u_tgt_tri, ov = tri_evals[b]
            accepted = False
            if not ov:
                dU_tgt = u_tgt_tri - U_tgt_curr[b]
                ddU = dU_tgt - dU_ref - delta_eps_K * float(dN)
                if ddU <= 0.0:
                    accepted = True
                else:
                    arg = -ddU / T_K
                    if arg > -50.0 and rng.random() < math.exp(arg):
                        accepted = True
            if cyc >= eq_cycles:
                if dN > 0:
                    prod_ins_att += 1
                    if accepted:
                        prod_ins_acc += 1
                elif dN < 0:
                    prod_del_att += 1
                    if accepted:
                        prod_del_acc += 1
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
        if log_every > 0 and ((cyc + 1) % log_every == 0):
            elapsed = time.perf_counter() - t0
            mean_up_now = float(np.mean(N_curr) * conv_mmol_g)
            print(
                f"  [NestedMC cyc {cyc+1:4d}/{total_cycles}] N={N_curr.tolist()} ({mean_up_now:.2f} mmol/g) "
                f"Acc={np.mean(acc_outer)/(cyc+1)*100:5.1f}% ({elapsed:.1f}s)"
            )

    dt = time.perf_counter() - t0
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
            box_qst.append(q_K * K_TO_KJMOL)
        elif mN > 0:
            box_qst.append((T_K - mU / mN) * K_TO_KJMOL)
        else:
            box_qst.append(0.0)

    mean_up = float(np.mean(box_uptakes))
    sem_up = float(np.std(box_uptakes, ddof=1) / math.sqrt(n_b)) if n_b > 1 else 0.0
    per_box_mean_qst = float(np.mean(box_qst))
    per_box_sem_qst = float(np.std(box_qst, ddof=1) / math.sqrt(n_b)) if n_b > 1 else 0.0
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
    jack = compute_pooled_jackknife_qst(box_moments, T_K)
    mean_qst = float(jack["Qst_kJ_mol"])
    sem_qst = float(jack["Qst_sem_kJ_mol"])
    mean_mb = float(np.mean(box_molar_bind))
    mean_mq = float(np.mean(box_molar_qst))
    sem_mq = float(np.std(box_molar_qst, ddof=1) / math.sqrt(n_b)) if n_b > 1 else 0.0

    final_acc = proposer.get_accepted()
    final_states = [
        {
            "N": int(st["N"]),
            "U_ref_K": float(st["U_ref_K"]),
            "cx": st["cx"].tolist(),
            "cy": st["cy"].tolist(),
            "cz": st["cz"].tolist(),
            "ux": st["ux"].tolist(),
            "uy": st["uy"].tolist(),
            "uz": st["uz"].tolist(),
        }
        for st in final_acc
    ]

    return {
        "n_boxes": n_b,
        "k_steps": k_steps,
        "eq_cycles": eq_cycles,
        "prod_cycles": prod_cycles,
        "delta_eps_kJ_mol": float(delta_eps_K * K_TO_KJMOL),
        "elapsed_s": dt,
        "uptake_mmol_g": mean_up,
        "uptake_sem_mmol_g": sem_up,
        "box_uptakes_mmol_g": box_uptakes.tolist(),
        "Qst_estimator": "pooled_jackknife",
        "Qst_kJ_mol": mean_qst,
        "Qst_sem_kJ_mol": sem_qst,
        "Qst_per_box_mean_kJ_mol": per_box_mean_qst,
        "Qst_per_box_sem_kJ_mol": per_box_sem_qst,
        "box_qst_kJ_mol": box_qst,
        "box_moments": box_moments,
        "mean_molar_binding_kJ_mol": mean_mb,
        "mean_molar_qst_kJ_mol": mean_mq,
        "mean_molar_qst_sem_kJ_mol": sem_mq,
        "outer_acc_rate": float(np.mean(acc_outer) / tot_denom),
        "prod_outer_acc_rate": float(np.mean(prod_acc_outer) / prod_denom) if prod_cycles > 0 else 0.0,
        "prod_ins_att": int(prod_ins_att),
        "prod_ins_acc": int(prod_ins_acc),
        "prod_del_att": int(prod_del_att),
        "prod_del_acc": int(prod_del_acc),
        "mean_inner_acc_per_burst": float(np.mean(inner_acc_tot) / tot_denom),
        "effective_mc_steps_per_target_eval": float(k_steps * np.mean(acc_outer) / tot_denom),
        "box_N_history": box_N_history,
        "trajectory": trajectory,
        "final_states": final_states,
    }
