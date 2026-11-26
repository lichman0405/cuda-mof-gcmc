"""Rigid guest molecule definitions and presets for GPU-GCMC + MLIP Nested MC."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict, List
import numpy as np


@dataclass(frozen=True)
class GuestSite:
    symbol: str
    Z: int
    pos: tuple  # (x, y, z) in local frame, Angstrom
    q: float
    eps_K: float
    sigma_A: float
    lj_type: int  # -1 for massless non-LJ site (e.g. N2 COM charge site)


@dataclass(frozen=True)
class GuestSpec:
    name: str
    formula: str
    is_isotropic: bool
    Tc_K: float
    Pc_bar: float
    omega: float
    sites: List[GuestSite]

    def real_sites(self) -> List[GuestSite]:
        return [s for s in self.sites if s.Z > 0]

    def real_local_positions(self) -> np.ndarray:
        pts = [s.pos for s in self.sites if s.Z > 0]
        return np.array(pts, dtype=np.float64)  # (N_real, 3)

    def real_atomic_numbers(self) -> np.ndarray:
        return np.array([s.Z for s in self.sites if s.Z > 0], dtype=np.int64)

    def real_symbols_formula(self) -> str:
        return "".join(s.symbol for s in self.sites if s.Z > 0)

    def to_ase_atoms(self, box_A: float = 30.0):
        from ase import Atoms

        rs = self.real_sites()
        syms = [s.symbol for s in rs]
        pos = np.array([s.pos for s in rs], dtype=np.float64) + 0.5 * box_A
        at = Atoms(syms, positions=pos)
        at.set_cell([box_A, box_A, box_A])
        at.set_pbc(True)
        return at

    def to_c_arrays(self) -> dict:
        n = len(self.sites)
        lxyz = np.array([s.pos for s in self.sites], dtype=np.float64).reshape(n, 3)
        q = np.array([s.q for s in self.sites], dtype=np.float64)
        eps = np.array([s.eps_K for s in self.sites], dtype=np.float64)
        sig = np.array([s.sigma_A for s in self.sites], dtype=np.float64)
        lj_type = np.array([s.lj_type for s in self.sites], dtype=np.int32)
        z_arr = np.array([int(s.Z) for s in self.sites], dtype=np.int32)
        return {
            "n_sites": n,
            "lxyz": np.ascontiguousarray(lxyz),
            "q": np.ascontiguousarray(q),
            "eps": np.ascontiguousarray(eps),
            "sig": np.ascontiguousarray(sig),
            "lj_type": np.ascontiguousarray(lj_type),
            "Z": np.ascontiguousarray(z_arr),
            "Tc_K": float(self.Tc_K),
            "Pc_bar": float(self.Pc_bar),
            "omega": float(self.omega),
            "is_isotropic": 1 if self.is_isotropic else 0,
        }


def _centered_sites(raw_sites: List[GuestSite], masses: List[float]) -> List[GuestSite]:
    tot_m = sum(masses)
    if tot_m <= 0.0:
        return raw_sites
    com = np.sum([m * np.array(s.pos, dtype=np.float64) for s, m in zip(raw_sites, masses)], axis=0) / tot_m
    out = []
    for s in raw_sites:
        p = tuple((np.array(s.pos, dtype=np.float64) - com).tolist())
        out.append(GuestSite(s.symbol, s.Z, p, s.q, s.eps_K, s.sigma_A, s.lj_type))
    return out


def _make_spce_h2o() -> GuestSpec:
    r_oh = 1.000
    half = math.radians(109.47 * 0.5)
    raw = [
        GuestSite("O", 8, (0.0, 0.0, 0.0), -0.8476, 78.197, 3.1656, 0),
        GuestSite("H", 1, (r_oh * math.sin(half), 0.0, r_oh * math.cos(half)), 0.4238, 0.0, 0.0, -1),
        GuestSite("H", 1, (-r_oh * math.sin(half), 0.0, r_oh * math.cos(half)), 0.4238, 0.0, 0.0, -1),
    ]
    sites = _centered_sites(raw, [15.999, 1.008, 1.008])
    return GuestSpec("h2o", "H2O", False, 647.096, 220.64, 0.344, sites)


def _make_trappe_nh3() -> GuestSpec:
    r_nh = 1.012
    sin_beta = (2.0 / math.sqrt(3.0)) * math.sin(math.radians(106.7 * 0.5))
    cos_beta = math.sqrt(max(0.0, 1.0 - sin_beta * sin_beta))
    raw = [GuestSite("N", 7, (0.0, 0.0, 0.0), -0.999, 185.0, 3.420, 0)]
    for k in range(3):
        phi = 2.0 * math.pi * k / 3.0
        raw.append(
            GuestSite(
                "H", 1,
                (r_nh * sin_beta * math.cos(phi), r_nh * sin_beta * math.sin(phi), -r_nh * cos_beta),
                0.333, 0.0, 0.0, -1,
            )
        )
    sites = _centered_sites(raw, [14.007, 1.008, 1.008, 1.008])
    return GuestSpec("nh3", "NH3", False, 405.4, 113.33, 0.256, sites)


def _make_ch4_allatom() -> GuestSpec:
    d = 1.090 / math.sqrt(3.0)
    sites = [
        GuestSite("C", 6, (0.0, 0.0, 0.0), -0.24, 33.21, 3.500, 0),
        GuestSite("H", 1, (+d, +d, +d), +0.06, 15.10, 2.500, 1),
        GuestSite("H", 1, (+d, -d, -d), +0.06, 15.10, 2.500, 1),
        GuestSite("H", 1, (-d, +d, -d), +0.06, 15.10, 2.500, 1),
        GuestSite("H", 1, (-d, -d, +d), +0.06, 15.10, 2.500, 1),
    ]
    return GuestSpec("ch4", "CH4", False, 190.564, 45.992, 0.01142, sites)


PRESET_GUESTS: Dict[str, GuestSpec] = {
    "co2": GuestSpec(
        "co2", "CO2", False, 304.1282, 73.773, 0.22394,
        [
            GuestSite("C", 6, (0.0, 0.0, 0.0), +0.70, 27.0, 2.800, 0),
            GuestSite("O", 8, (+1.160, 0.0, 0.0), -0.35, 79.0, 3.050, 1),
            GuestSite("O", 8, (-1.160, 0.0, 0.0), -0.35, 79.0, 3.050, 1),
        ],
    ),
    "co2_epm2": GuestSpec(
        "co2_epm2", "CO2", False, 304.1282, 73.773, 0.22394,
        [
            GuestSite("C", 6, (0.0, 0.0, 0.0), +0.6512, 28.129, 2.757, 0),
            GuestSite("O", 8, (+1.160, 0.0, 0.0), -0.3256, 80.507, 3.033, 1),
            GuestSite("O", 8, (-1.160, 0.0, 0.0), -0.3256, 80.507, 3.033, 1),
        ],
    ),
    "ch4": _make_ch4_allatom(),
    "n2": GuestSpec(
        "n2", "N2", False, 126.192, 33.958, 0.0372,
        [
            GuestSite("X", 0, (0.0, 0.0, 0.0), +0.964, 0.0, 0.0, -1),
            GuestSite("N", 7, (+0.550, 0.0, 0.0), -0.482, 36.0, 3.310, 0),
            GuestSite("N", 7, (-0.550, 0.0, 0.0), -0.482, 36.0, 3.310, 0),
        ],
    ),
    "h2": GuestSpec(
        "h2", "H2", False, 33.145, 12.964, -0.219,
        [
            GuestSite("X", 0, (0.0, 0.0, 0.0), -0.936, 36.7, 2.958, 0),
            GuestSite("H", 1, (+0.370, 0.0, 0.0), +0.468, 0.0, 0.0, -1),
            GuestSite("H", 1, (-0.370, 0.0, 0.0), +0.468, 0.0, 0.0, -1),
        ],
    ),
    "ar": GuestSpec(
        "ar", "Ar", True, 150.687, 48.63, -0.00219,
        [GuestSite("Ar", 18, (0.0, 0.0, 0.0), 0.0, 119.8, 3.405, 0)],
    ),
    "h2o": _make_spce_h2o(),
    "nh3": _make_trappe_nh3(),
}


def load_guest(name: str = "co2", guest_json: str = "", proposer: str = "uff") -> GuestSpec:
    if guest_json:
        p = Path(guest_json)
        if not p.exists():
            raise FileNotFoundError(f"Guest JSON not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        raw_sites = data["sites"]
        lj_map = {}
        sites = []
        for s in raw_sites:
            eps = float(s.get("eps_K", 0.0))
            sig = float(s.get("sigma_A", 0.0))
            if eps > 0.0 and sig > 0.0:
                key = (round(eps, 8), round(sig, 8))
                if key not in lj_map:
                    lj_map[key] = len(lj_map)
                lj_t = lj_map[key]
            else:
                lj_t = -1
            sites.append(
                GuestSite(
                    symbol=str(s.get("symbol", "X")),
                    Z=int(s.get("Z", 0)),
                    pos=tuple(float(v) for v in s["pos"]),
                    q=float(s.get("q", 0.0)),
                    eps_K=eps,
                    sigma_A=sig,
                    lj_type=lj_t,
                )
            )
        return GuestSpec(
            name=str(data.get("name", p.stem)),
            formula=str(data.get("formula", p.stem)),
            is_isotropic=bool(data.get("is_isotropic", len(sites) == 1)),
            Tc_K=float(data.get("Tc_K", 0.0)),
            Pc_bar=float(data.get("Pc_bar", 0.0)),
            omega=float(data.get("omega", 0.0)),
            sites=sites,
        )

    key = name.strip().lower()
    if key == "co2" and proposer == "mercado":
        return PRESET_GUESTS["co2_epm2"]
    if key not in PRESET_GUESTS:
        raise ValueError(f"Unknown guest preset '{name}'. Available: {sorted(PRESET_GUESTS.keys())}")
    return PRESET_GUESTS[key]
