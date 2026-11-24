"""Unit tests for rigid guest definitions, quaternion rotation, CIF expansion, and periodic QEq."""

import json
import math
from pathlib import Path
import numpy as np
import pytest

from guests import PRESET_GUESTS, load_guest
from nested_mc_engine import quat_rotate_batch
from qeq_charges import compute_periodic_qeq_charges, extract_framework_for_proposer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def test_all_preset_guests_neutral_and_valid():
    assert set(PRESET_GUESTS.keys()) >= {"co2", "co2_epm2", "ch4", "n2", "h2", "ar", "h2o", "nh3"}
    for name, g in PRESET_GUESTS.items():
        q_sum = sum(s.q for s in g.sites)
        assert abs(q_sum) < 1e-10, f"{name} net charge {q_sum}"
        assert len(g.real_sites()) >= 1
        arrs = g.to_c_arrays()
        assert arrs["lxyz"].shape == (len(g.sites), 3)
        at = g.to_ase_atoms(box_A=30.0)
        assert len(at) == len(g.real_sites())


def test_load_custom_guest_json(tmp_path):
    custom = {
        "name": "custom_co",
        "formula": "CO",
        "is_isotropic": False,
        "Tc_K": 132.92,
        "Pc_bar": 34.99,
        "omega": 0.048,
        "sites": [
            {"symbol": "C", "Z": 6, "pos": [-0.646, 0.0, 0.0], "q": -0.24, "eps_K": 36.4, "sigma_A": 3.55},
            {"symbol": "O", "Z": 8, "pos": [0.482, 0.0, 0.0], "q": -0.27, "eps_K": 51.0, "sigma_A": 2.95},
            {"symbol": "X", "Z": 0, "pos": [0.0, 0.0, 0.0], "q": 0.51, "eps_K": 0.0, "sigma_A": 0.0},
        ],
    }
    p = tmp_path / "co.json"
    p.write_text(json.dumps(custom), encoding="utf-8")
    g = load_guest(guest_json=str(p))
    assert g.formula == "CO"
    assert len(g.sites) == 3
    assert len(g.real_sites()) == 2
    assert g.sites[2].lj_type == -1
    assert g.sites[0].lj_type == 0 and g.sites[1].lj_type == 1


def test_quat_rotate_preserves_rigid_geometry():
    rng = np.random.default_rng(12345)
    raw = rng.normal(size=(32, 4))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    qw, qx, qy, qz = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]

    # H2O bond lengths and angle
    h2o = load_guest("h2o")
    rot_h2o = quat_rotate_batch(qw, qx, qy, qz, h2o.real_local_positions())
    v1 = rot_h2o[:, 1, :] - rot_h2o[:, 0, :]
    v2 = rot_h2o[:, 2, :] - rot_h2o[:, 0, :]
    r1 = np.linalg.norm(v1, axis=1)
    r2 = np.linalg.norm(v2, axis=1)
    ang = np.degrees(np.arccos(np.clip(np.sum(v1 * v2, axis=1) / (r1 * r2), -1.0, 1.0)))
    assert np.max(np.abs(r1 - 1.0)) < 1e-12
    assert np.max(np.abs(r2 - 1.0)) < 1e-12
    assert np.max(np.abs(ang - 109.47)) < 1e-10

    # CH4 tetrahedral C-H distances and H-C-H angles (109.4712 deg)
    ch4 = load_guest("ch4")
    rot_ch4 = quat_rotate_batch(qw, qx, qy, qz, ch4.real_local_positions())
    ch_vecs = rot_ch4[:, 1:, :] - rot_ch4[:, 0:1, :]
    ch_len = np.linalg.norm(ch_vecs, axis=2)
    assert np.max(np.abs(ch_len - 1.090)) < 1e-12
    cos_hch = np.sum(ch_vecs[:, 0, :] * ch_vecs[:, 1, :], axis=1) / (ch_len[:, 0] * ch_len[:, 1])
    assert np.max(np.abs(cos_hch + 1.0 / 3.0)) < 1e-12


def test_periodic_qeq_and_cif_extraction():
    d_mg_cif = extract_framework_for_proposer(str(DATA / "MgMOF-74.cif"), charge_mode="cif")
    d_mg_qeq = extract_framework_for_proposer(str(DATA / "MgMOF-74.cif"), charge_mode="qeq")
    assert d_mg_cif["charge_source"] == "mg_mof74_repeat"
    assert abs(float(np.sum(d_mg_qeq["charges"]))) < 1e-12
    mae_mg = float(np.mean(np.abs(d_mg_qeq["charges"] - d_mg_cif["charges"])))
    assert mae_mg < 0.25

    d_cu_cif = extract_framework_for_proposer(str(DATA / "Cu-BTC.cif"), charge_mode="cif")
    d_cu_qeq = extract_framework_for_proposer(str(DATA / "Cu-BTC.cif"), charge_mode="qeq")
    assert d_cu_cif["charge_source"] == "cif"
    assert abs(float(np.sum(d_cu_qeq["charges"]))) < 1e-12
    mae_cu = float(np.mean(np.abs(d_cu_qeq["charges"] - d_cu_cif["charges"])))
    assert mae_cu < 0.20

    d_ir_auto = extract_framework_for_proposer(str(DATA / "IRMOF-1.cif"), charge_mode="auto")
    assert d_ir_auto["charge_source"] == "auto_qeq"
    assert abs(float(np.sum(d_ir_auto["charges"]))) < 1e-12
    assert np.max(np.abs(d_ir_auto["charges"])) > 0.3

    d_ir_zero = extract_framework_for_proposer(str(DATA / "IRMOF-1.cif"), charge_mode="zero")
    assert d_ir_zero["charge_source"] == "zero"
    assert np.all(d_ir_zero["charges"] == 0.0)


def test_spacegroup_only_cif_expansion(tmp_path):
    cif_text = """data_test_cubic
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'F m -3 m'
_symmetry_Int_Tables_number 225
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 Si 0.0 0.0 0.0
O1 O 0.25 0.25 0.25
"""
    p = tmp_path / "sg225.cif"
    p.write_text(cif_text, encoding="utf-8")
    d = extract_framework_for_proposer(str(p), charge_mode="auto")
    assert len(d["Z"]) == 12
    assert d["charge_source"] == "auto_qeq"
    assert abs(float(np.sum(d["charges"]))) < 1e-12
    si_q = float(np.mean(d["charges"][d["Z"] == 14]))
    o_q = float(np.mean(d["charges"][d["Z"] == 8]))
    assert si_q > 0.0 and o_q < 0.0
