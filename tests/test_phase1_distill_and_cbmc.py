"""Phase 1 Mathematical & Bit-Exact Verification Suite for Auto-MLIP + 128-Thread GPU-CBMC."""

from collections import OrderedDict
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from guests import load_guest
from mlip_distill import (
    DISTILL_VERSION,
    _eval_deployed_2b_lut,
    _resolve_model_sha256,
    _solve_2b_model,
    _unscaled_gg_pair_energy_K,
    assign_framework_environments,
    compute_distill_cache_key,
    distill_mlip_surrogate,
    load_distill_cache,
    make_distilled_guest_spec,
)
from nested_mc_engine import CudaGcmcProposer, K_TO_KJMOL, run_nested_mc
from qeq_charges import extract_framework_for_proposer
from run_metadata import sha256_file
from run_mlip_gcmc import compute_grid_dims

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _make_synthetic_oms_distill_res(fw_data: dict, guest, grid_dims=(64, 64, 24), well_depth_K: float = -4200.0):
    """Build deterministic physical 3D unit-cell external grids + 2B LUT for classical verification."""
    nx, ny, nz = [int(d) for d in grid_dims]
    uc = fw_data["uc"]
    H = np.asarray(uc.get_cell(), dtype=np.float64)
    inv_H = np.linalg.inv(H)
    fw_frac = np.mod(np.asarray(uc.get_scaled_positions(), dtype=np.float64), 1.0)
    Z = np.asarray(uc.get_atomic_numbers(), dtype=np.int32)

    sa = np.arange(nx, dtype=np.float64) / float(nx)
    sb = np.arange(ny, dtype=np.float64) / float(ny)
    sc = np.arange(nz, dtype=np.float64) / float(nz)
    kz, ky, kx = np.meshgrid(sc, sb, sa, indexing="ij")
    f_pts = np.stack([kx.ravel(), ky.ravel(), kz.ravel()], axis=1)

    df = f_pts[:, None, :] - fw_frac[None, :, :]
    df -= np.round(df)
    dr = df @ H
    d_mat = np.sqrt(np.sum(dr * dr, axis=-1))

    is_mg = (Z == 12)
    d_nonm = np.min(d_mat[:, ~is_mg], axis=1) if np.any(is_mg) else np.min(d_mat, axis=1)
    d_mg = np.min(d_mat[:, is_mg], axis=1) if np.any(is_mg) else np.full(len(f_pts), 99.0)

    # Smooth potential well around non-metal wall (3.3 A) and OMS Mg2+ (2.35 A)
    v_o = (
        600.0 * np.exp(-((np.maximum(d_nonm, 1.5) - 2.6) / 0.25) ** 2)
        - 1600.0 * np.exp(-((d_nonm - 3.35) / 0.55) ** 2)
        + 0.45 * well_depth_K * np.exp(-((d_mg - 2.35) / 0.30) ** 2)
        + 25.0 * np.power(2.20 / np.maximum(d_mg, 1.4), 12.0) * (d_mg < 3.0)
        + 25.0 * np.power(2.35 / np.maximum(d_nonm, 1.4), 12.0) * (d_nonm < 3.2)
    )
    v_c = (
        -1200.0 * np.exp(-((d_nonm - 3.45) / 0.55) ** 2)
        + 25.0 * np.power(2.20 / np.maximum(d_nonm, 1.4), 12.0) * (d_nonm < 3.2)
    )
    v_o = np.where((d_nonm < 1.8) | (d_mg < 1.8), 1.0e6, np.clip(v_o, -2.0e5, 1.0e5))
    v_c = np.where((d_nonm < 1.8) | (d_mg < 1.8), 1.0e6, np.clip(v_c, -2.0e5, 1.0e5))

    n_chan = max(1, max(s.lj_type for s in guest.sites) + 1)
    grids = np.zeros((n_chan, nz, ny, nx), dtype=np.float32)
    for c in range(n_chan):
        grids[c] = (v_c if c == 0 else v_o).astype(np.float32).reshape(nz, ny, nx)

    r_lut = np.linspace(2.0, 10.0, 128, dtype=np.float64)
    lut_K = (-120.0 * np.exp(-((r_lut - 3.8) / 0.9) ** 2) * (r_lut < 9.5)).astype(np.float32)
    lut_K[-1] = 0.0
    return {
        "grids_u_lj": np.ascontiguousarray(grids, dtype=np.float32),
        "gg_com_lut_K": lut_K,
        "gg_q_scale": 0.92,
        "gg_lj_scale": 1.08,
        "gg_lut_rmin_A": 2.0,
        "gg_lut_rmax_A": 10.0,
    }


def test_env_classification_and_guest_channels():
    for cif_name, min_envs in [("MgMOF-74.cif", 6), ("Cu-BTC.cif", 4), ("IRMOF-1.cif", 4)]:
        fw = extract_framework_for_proposer(str(DATA / cif_name), charge_mode="auto")
        env = assign_framework_environments(fw["uc"], charges=fw["charges"])
        assert len(env["unique_sigs"]) >= min_envs
        assert len(env["env_ids"]) == len(fw["uc"])
        assert np.sum(env["env_counts"]) == len(fw["uc"])
        if cif_name == "MgMOF-74.cif":
            assert any(sig[0] == 12 and len(sig[1]) == 5 for sig in env["unique_sigs"])

    for g_name, exp_chan in [("co2", 2), ("h2o", 2), ("ch4", 2), ("n2", 1), ("ar", 1)]:
        dg = make_distilled_guest_spec(load_guest(g_name))
        real_chans = {s.lj_type for s in dg.sites if s.Z > 0}
        assert len(real_chans) == exp_chan
        assert all(s.eps_K > 0.0 and s.sigma_A > 0.0 for s in dg.sites if s.Z > 0)


def test_2b_model_preserves_short_range_repulsion():
    """Verify _solve_2b_model preserves Pauli LJ core repulsion within [0.65, 1.50] and allows soft-LJ below 0.85."""
    rng = np.random.default_rng(20261125)
    r_com = rng.uniform(2.85, 6.20, size=120)
    u_lj0 = 120.0 * ((3.1 / r_com) ** 12 - (3.1 / r_com) ** 6)
    u_q0 = -180.0 * np.exp(-((r_com - 3.6) / 0.8) ** 2)
    # Deep artificial over-attractive target around 3.2 A to stress-test constraints
    u2_true = 0.3 * u_lj0 + 0.2 * u_q0 - 2500.0 * np.exp(-((r_com - 3.15) / 0.35) ** 2)
    res = _solve_2b_model(u_lj0, u_q0, r_com, u2_true, r_cut_gg=6.6, seed=20261125)
    assert 0.65 <= res["gg_lj_scale"] <= 1.50
    assert 0.50 <= res["gg_q_scale"] ** 2 <= 1.50
    assert res["n_2b_val"] == 24
    assert res["n_2b_train"] == 96
    assert res["r2_2b"] == res["r2_2b_deployed_holdout"]
    assert res["mae_2b_kJ_mol"] == res["mae_2b_deployed_holdout_kJ_mol"]
    lut = res["gg_com_lut_K"]
    lut_r = np.linspace(res["r_min_A"], res["r_max_A"], len(lut))
    assert np.all(lut[lut_r <= 2.75] == 0.0)
    assert float(np.min(lut)) >= -400.0
    assert abs(float(lut[-1])) < 1e-6

    # Relaxed lower bound [0.65, 1.50]: soft-LJ target (true scale = 0.72) should move below 0.85 without clipping
    u_lj_strong = 600.0 * ((3.2 / r_com) ** 12 - (3.2 / r_com) ** 6)
    u2_soft = 0.72 * u_lj_strong + 0.95 * u_q0
    res_soft = _solve_2b_model(u_lj_strong, u_q0, r_com, u2_soft, r_cut_gg=6.6, seed=20261126)
    assert 0.65 <= res_soft["gg_lj_scale"] < 0.85
    assert res_soft["r2_2b_deployed_holdout"] > 0.99

    # Extreme negative-LJ target must still be clamped at 0.65 to keep Pauli repulsion positive
    u2_crush = -2.0 * u_lj_strong + u_q0
    res_crush = _solve_2b_model(u_lj_strong, u_q0, r_com, u2_crush, r_cut_gg=6.6, seed=20261127)
    assert res_crush["gg_lj_scale"] >= 0.65 - 1e-9


def test_2b_deployed_lut_eval_matches_cuda_formula():
    """Verify _eval_deployed_2b_lut matches src/proposer_dll.cu:356-376 across interior and boundary regimes."""
    r_min_A, r_max_A, n_bins = 2.0, 10.0, 128
    lut_r = np.linspace(r_min_A, r_max_A, n_bins, dtype=np.float64)

    def _cuda_ref(r_vals, lut_arr):
        lut_f32 = np.asarray(lut_arr, dtype=np.float32)
        inv_dr = float(len(lut_f32) - 1) / (r_max_A - r_min_A)
        out = []
        for rc_com in r_vals:
            if rc_com >= r_max_A:
                out.append(0.0)
                continue
            u_pos = (rc_com - r_min_A) * inv_dr
            if u_pos <= 0.0:
                slope = float(lut_f32[1]) - float(lut_f32[0])
                if slope > 0.0:
                    slope = 0.0
                out.append(float(lut_f32[0]) + slope * u_pos)
            else:
                idx0 = int(math.floor(u_pos))
                if idx0 >= len(lut_f32) - 1:
                    out.append(float(lut_f32[-1]))
                else:
                    frac = u_pos - float(idx0)
                    v0 = float(lut_f32[idx0])
                    v1 = float(lut_f32[idx0 + 1])
                    out.append(v0 + frac * (v1 - v0))
        return np.asarray(out, dtype=np.float64)

    # Case 1: Deployed smooth LUT (zero for r <= 2.75 and at r_max_A)
    lut_deployed = np.clip(-150.0 * np.exp(-((lut_r - 3.9) / 0.85) ** 2), -400.0, 400.0)
    lut_deployed[lut_r <= 2.75] = 0.0
    lut_deployed[-1] = 0.0
    r_test = np.array([1.2, 2.0, 2.5, 2.75, 3.123, 3.9, 5.55, 9.999, 10.0, 12.5], dtype=np.float64)
    got = _eval_deployed_2b_lut(r_test, lut_deployed, r_min_A, r_max_A)
    exp = _cuda_ref(r_test, lut_deployed)
    assert np.allclose(got, exp, atol=1e-13, rtol=0.0)
    assert got[0] == 0.0 and got[1] == 0.0 and got[-2] == 0.0 and got[-1] == 0.0

    # Case 2: Repulsive downward slope at r_min_A (lut[1] < lut[0]) extrapolates upward for r < r_min_A
    lut_rep = np.linspace(200.0, 0.0, n_bins, dtype=np.float32)
    got_rep = _eval_deployed_2b_lut(r_test, lut_rep, r_min_A, r_max_A)
    exp_rep = _cuda_ref(r_test, lut_rep)
    assert np.allclose(got_rep, exp_rep, atol=1e-13, rtol=0.0)
    assert got_rep[0] > float(lut_rep[0])

    # Case 3: Increasing slope at r_min_A (lut[1] > lut[0]) clamps slope to 0 for r < r_min_A
    lut_inc = np.linspace(-100.0, 0.0, n_bins, dtype=np.float32)
    got_inc = _eval_deployed_2b_lut(r_test, lut_inc, r_min_A, r_max_A)
    exp_inc = _cuda_ref(r_test, lut_inc)
    assert np.allclose(got_inc, exp_inc, atol=1e-13, rtol=0.0)
    assert got_inc[0] == float(lut_inc[0])


def test_2b_honest_holdout_split_and_no_leakage():
    """Verify honest 80/20 2B deployed LUT validation, zero val leakage, and clean small-N fallback."""
    rng = np.random.default_rng(20260777)
    M = 100
    r_com = rng.uniform(2.85, 6.30, size=M)
    u_lj0 = 150.0 * ((3.15 / r_com) ** 12 - (3.15 / r_com) ** 6)
    u_q0 = -160.0 * np.exp(-((r_com - 3.7) / 0.85) ** 2)
    u2_clean = 0.90 * u_lj0 + 0.85 * u_q0 - 95.0 * np.exp(-((r_com - 3.95) / 0.70) ** 2)

    split_seed = 20268888
    res = _solve_2b_model(u_lj0, u_q0, r_com, u2_clean, r_cut_gg=6.8, seed=split_seed)
    assert res["n_2b_val"] == 20
    assert res["n_2b_train"] == 80
    assert res["r2_2b"] == res["r2_2b_deployed_holdout"]
    assert res["mae_2b_kJ_mol"] == res["mae_2b_deployed_holdout_kJ_mol"]
    assert res["r2_2b_deployed_holdout"] > 0.95
    assert res["mae_2b_deployed_holdout_kJ_mol"] < res["mae_2b_classical_holdout_kJ_mol"]
    assert res["r2_2b_train"] is not None and res["mae_2b_train_kJ_mol"] is not None

    # Exact zero-leakage proof: perturbing a validation sample must NOT change CV predictions on val_idx.
    # Shifting u2[val_idx[0]] from +5000 K to +10000 K must increase hold-out MAE by exactly 5000/n_val * K_TO_KJMOL.
    perm = np.random.default_rng(split_seed).permutation(M)
    val_idx = perm[:20]
    tr_idx = perm[20:]
    u2_val_5k = u2_clean.copy()
    u2_val_10k = u2_clean.copy()
    u2_val_5k[val_idx[0]] += 5000.0
    u2_val_10k[val_idx[0]] += 10000.0
    r_5k = _solve_2b_model(u_lj0, u_q0, r_com, u2_val_5k, r_cut_gg=6.8, seed=split_seed)
    r_10k = _solve_2b_model(u_lj0, u_q0, r_com, u2_val_10k, r_cut_gg=6.8, seed=split_seed)
    expected_delta_mae = (5000.0 / 20.0) * K_TO_KJMOL
    actual_delta_mae = r_10k["mae_2b_deployed_holdout_kJ_mol"] - r_5k["mae_2b_deployed_holdout_kJ_mol"]
    assert abs(actual_delta_mae - expected_delta_mae) < 1e-12

    # Overfitting noisy data / train-only spiky outlier must NOT produce artificially optimistic hold-out error
    noise = rng.normal(0.0, 220.0, size=M)
    u2_base = u_lj0 + u_q0 + noise
    u2_noisy = u2_base.copy()
    u2_noisy[tr_idx[:3]] -= 1800.0  # strong localized train-only outliers
    r_base = _solve_2b_model(u_lj0, u_q0, r_com, u2_base, r_cut_gg=6.8, seed=split_seed)
    r_noisy = _solve_2b_model(u_lj0, u_q0, r_com, u2_noisy, r_cut_gg=6.8, seed=split_seed)
    assert r_noisy["n_2b_val"] == 20
    assert r_noisy["mae_2b_train_kJ_mol"] > r_base["mae_2b_train_kJ_mol"]
    assert r_noisy["mae_2b_deployed_holdout_kJ_mol"] >= r_base["mae_2b_deployed_holdout_kJ_mol"] - 0.15
    # On i.i.d. noisy data without outliers, honest hold-out R^2 cannot exceed in-sample overfit R^2
    u2_iid_noise = u_lj0 + u_q0 + rng.normal(0.0, 250.0, size=M)
    r_iid = _solve_2b_model(u_lj0, u_q0, r_com, u2_iid_noise, r_cut_gg=6.8, seed=split_seed)
    assert r_iid["r2_2b_deployed_holdout"] < r_iid["r2_2b_train"]

    # Clean fallback when 16 <= M < 24 (n_2b_val = 0, hold-out metrics explicitly None)
    r_small = _solve_2b_model(u_lj0[:20], u_q0[:20], r_com[:20], u2_clean[:20], r_cut_gg=6.8, seed=split_seed)
    assert r_small["n_2b_val"] == 0
    assert r_small["n_2b_train"] == 20
    assert r_small["r2_2b_deployed_holdout"] is None
    assert r_small["mae_2b_deployed_holdout_kJ_mol"] is None
    assert r_small["mae_2b_classical_holdout_kJ_mol"] is None
    assert r_small["r2_2b"] == r_small["r2_2b_train"]

    # Clean fallback when M < 16 (identity scaling + zero LUT)
    r_tiny = _solve_2b_model(u_lj0[:10], u_q0[:10], r_com[:10], u2_clean[:10], r_cut_gg=6.8, seed=split_seed)
    assert r_tiny["n_2b_val"] == 0
    assert r_tiny["n_2b_train"] == 0
    assert r_tiny["gg_lj_scale"] == 1.0
    assert r_tiny["gg_q_scale"] == 1.0
    assert np.all(r_tiny["gg_com_lut_K"] == 0.0)
    assert r_tiny["r2_2b"] is None and r_tiny["mae_2b_kJ_mol"] is None


def test_only_1b_cache_key_separation_and_model_sha256_in_key():
    """Verify DISTILL_VERSION bump, only_1b cache key separation, and backward-compatible model_sha256."""
    assert DISTILL_VERSION == "v1.2"
    cif = str(DATA / "MgMOF-74.cif")
    guest = make_distilled_guest_spec(load_guest("co2"))

    k_default = compute_distill_cache_key(
        cif_path=cif, supercell=(1, 1, 2), guest=guest, grid_dims=(32, 32, 16),
        model_name="medium", model_file_path=None, dispersion=True, dispersion_xc="pbe",
        dtype="float32", n_1b=960, n_2b=192, distill_seed=20260001,
    )
    k_1b2b = compute_distill_cache_key(
        cif_path=cif, supercell=(1, 1, 2), guest=guest, grid_dims=(32, 32, 16),
        model_name="medium", model_file_path=None, dispersion=True, dispersion_xc="pbe",
        dtype="float32", n_1b=960, n_2b=192, distill_seed=20260001, only_1b=False,
    )
    k_only1b = compute_distill_cache_key(
        cif_path=cif, supercell=(1, 1, 2), guest=guest, grid_dims=(32, 32, 16),
        model_name="medium", model_file_path=None, dispersion=True, dispersion_xc="pbe",
        dtype="float32", n_1b=960, n_2b=192, distill_seed=20260001, only_1b=True,
    )
    k_n2b_zero_full = compute_distill_cache_key(
        cif_path=cif, supercell=(1, 1, 2), guest=guest, grid_dims=(32, 32, 16),
        model_name="medium", model_file_path=None, dispersion=True, dispersion_xc="pbe",
        dtype="float32", n_1b=960, n_2b=0, distill_seed=20260001, only_1b=False,
    )
    assert k_default == k_1b2b
    assert k_only1b != k_1b2b
    assert k_only1b != k_n2b_zero_full

    # Explicit model_sha256 distinguishes weights even when model_file_path=None
    k_sha_a = compute_distill_cache_key(
        cif_path=cif, supercell=(1, 1, 2), guest=guest, grid_dims=(32, 32, 16),
        model_name="medium", model_file_path=None, dispersion=True, dispersion_xc="pbe",
        dtype="float32", n_1b=960, n_2b=192, distill_seed=20260001,
        model_sha256="state_dict:" + "a" * 64,
    )
    k_sha_b = compute_distill_cache_key(
        cif_path=cif, supercell=(1, 1, 2), guest=guest, grid_dims=(32, 32, 16),
        model_name="medium", model_file_path=None, dispersion=True, dispersion_xc="pbe",
        dtype="float32", n_1b=960, n_2b=192, distill_seed=20260001,
        model_sha256="state_dict:" + "b" * 64,
    )
    assert k_sha_a != k_sha_b
    assert k_sha_a != k_default


def test_resolve_model_sha256_deterministic_and_sensitive(tmp_path):
    """Verify _resolve_model_sha256 handles files, target attrs, and in-memory state_dict deterministically."""
    # 1. Existing file passed as model_name
    ckpt = tmp_path / "dummy.model"
    ckpt.write_bytes(b"mace-dummy-weights-v1")
    expected_file_sha = sha256_file(str(ckpt))
    assert _resolve_model_sha256(None, str(ckpt)) == expected_file_sha

    # 2. target.model_file_path and target.model_sha256
    t_path = SimpleNamespace(model_file_path=str(ckpt))
    assert _resolve_model_sha256(t_path, "medium") == expected_file_sha
    t_explicit = SimpleNamespace(model_file_path=None, model_sha256="custom_sha_1234")
    assert _resolve_model_sha256(t_explicit, "medium") == "custom_sha_1234"

    # 3. target.mace_calc.model_paths
    t_calc_path = SimpleNamespace(mace_calc=SimpleNamespace(model_paths=[str(ckpt)]))
    assert _resolve_model_sha256(t_calc_path, "medium") == expected_file_sha

    # 4. In-memory dummy model state_dict (order-independent, deterministic, sensitive to weights & dtype)
    class _DummyModule:
        def __init__(self, sd):
            self._sd = sd

        def state_dict(self):
            return self._sd

    w1 = np.array([[1.0, -2.5], [3.25, 0.5]], dtype=np.float32)
    b1 = np.array([0.1, -0.2], dtype=np.float32)
    t_sd_a = SimpleNamespace(mace_model=_DummyModule(OrderedDict([("b", b1.copy()), ("w", w1.copy())])))
    t_sd_b = SimpleNamespace(mace_model=_DummyModule(OrderedDict([("w", w1.copy()), ("b", b1.copy())])))
    sha_a = _resolve_model_sha256(t_sd_a, "mace-matpes-r2scan-0")
    sha_b = _resolve_model_sha256(t_sd_b, "mace-matpes-r2scan-0")
    assert sha_a is not None
    assert sha_a.startswith("state_dict:")
    assert len(sha_a) == len("state_dict:") + 64
    assert sha_a == sha_b

    w1_pert = w1.copy()
    w1_pert[0, 1] += 1e-5
    t_sd_pert = SimpleNamespace(mace_model=_DummyModule({"w": w1_pert, "b": b1.copy()}))
    assert _resolve_model_sha256(t_sd_pert, "mace-matpes-r2scan-0") != sha_a

    t_sd_f64 = SimpleNamespace(mace_model=_DummyModule({"w": w1.astype(np.float64), "b": b1.copy()}))
    assert _resolve_model_sha256(t_sd_f64, "mace-matpes-r2scan-0") != sha_a

    # 5. Real torch.nn.Module if PyTorch is installed locally (pure CPU)
    try:
        import torch
        torch.manual_seed(42)
        m1 = torch.nn.Linear(4, 2)
        torch.manual_seed(42)
        m2 = torch.nn.Linear(4, 2)
        h1 = _resolve_model_sha256(SimpleNamespace(mace_model=m1), "medium")
        h2 = _resolve_model_sha256(SimpleNamespace(mace_calc=SimpleNamespace(models=[m2])), "medium")
        assert h1 is not None and h1.startswith("state_dict:") and h1 == h2
        with torch.no_grad():
            m2.weight[0, 0] += 1e-4
        assert _resolve_model_sha256(SimpleNamespace(mace_model=m2), "medium") != h1
    except ImportError:
        pass

    # 6. None target with built-in model name -> None
    assert _resolve_model_sha256(None, "medium") is None


def test_distill_mlip_surrogate_only_1b_and_2b_holdout_pure_numpy(tmp_path, monkeypatch):
    """Pure-CPU (no torch/ASE required) end-to-end test of distill_mlip_surrogate only_1b, 2B hold-out, and cache."""
    import mlip_distill

    H = np.diag([14.0, 14.0, 14.0])
    inv_H = np.linalg.inv(H)
    fw_frac = np.array([[0.0, 0.0, 0.0], [0.15, 0.15, 0.15]], dtype=np.float64)

    class _FakeUnitCell:
        def repeat(self, reps):
            assert tuple(reps) == (1, 1, 1)
            return self

        def get_cell(self):
            return H

        def get_scaled_positions(self):
            return fw_frac

    def _fake_assign_envs(uc, charges=None):
        return {
            "env_ids": np.array([0, 1], dtype=np.int32),
            "unique_sigs": [(12, (8,)), (8, (12,))],
            "env_counts": np.array([1.0, 1.0], dtype=np.float64),
            "frac": fw_frac,
            "pos": fw_frac @ H,
            "H_unit": H,
            "inv_H_unit": inv_H,
            "Z": np.array([12, 8], dtype=np.int32),
            "active_mask": np.array([True, True], dtype=bool),
            "is_metal_mask": np.array([True, False], dtype=bool),
        }

    def _fake_eval_basis(
        pts_cart, site_chan_idx, n_pts_per_mol, n_chan, guest_chan_q, env_info,
        r_cut=8.0, ewald_alpha=0.30, k_max=2.0, mu_centers=None, sigma_rbf=0.45,
        device=None, chunk_pts=8192,
    ):
        n_env = len(env_info["unique_sigs"])
        n_rad = 1 + len(mu_centers)
        n_mol = len(pts_cart) // n_pts_per_mol
        pts_f = pts_cart @ inv_H
        df = pts_f[:, None, :] - fw_frac[None, :, :]
        df -= np.round(df)
        d = np.sqrt(np.sum((df @ H) ** 2, axis=-1))  # (M, 2)
        out_rad = np.zeros((n_mol, n_chan, n_env, n_rad), dtype=np.float64)
        d_mol = d.reshape(n_mol, n_pts_per_mol, n_env)
        for s_idx in range(n_pts_per_mol):
            ch = int(site_chan_idx[s_idx])
            r_s = np.maximum(d_mol[:, s_idx, :], 1.25)
            out_rad[:, ch, :, 0] += (2.20 / r_s) ** 12
            for k, mu in enumerate(mu_centers):
                out_rad[:, ch, :, 1 + k] += np.exp(-((r_s - mu) ** 2) / (2.0 * sigma_rbf * sigma_rbf))
        out_phi = np.zeros((n_mol, n_env), dtype=np.float64)
        mol_dmin = np.min(d_mol, axis=(1, 2))
        return out_rad, out_phi, mol_dmin

    def _fake_render_grids(grid_dims, env_info, dist_guest, fit_1b, mu_centers, sigma_rbf, r_cut, device=None):
        nx, ny, nz = [int(x) for x in grid_dims]
        n_chan = fit_1b["w_sr"].shape[0]
        return np.zeros((n_chan, nz, ny, nx), dtype=np.float32)

    monkeypatch.setattr(mlip_distill, "assign_framework_environments", _fake_assign_envs)
    monkeypatch.setattr(mlip_distill, "_eval_unit_basis_torch", _fake_eval_basis)
    monkeypatch.setattr(mlip_distill, "_render_unit_cell_grids", _fake_render_grids)

    cif_stub = tmp_path / "mini_fake.cif"
    cif_stub.write_text("# pure-numpy mock cif\n", encoding="utf-8")
    fw_data = {"uc": _FakeUnitCell(), "charges": np.array([1.0, -1.0], dtype=np.float64)}
    guest = make_distilled_guest_spec(load_guest("ar"))

    class _PureNumpyOracle:
        def __init__(self, weight_val: float = 1.0, forbid_dimers: bool = False):
            self.device = None
            self.model_file_path = None
            self.mace_model = SimpleNamespace(
                state_dict=lambda: {"w": np.array([weight_val, -0.5], dtype=np.float32)}
            )
            self.forbid_dimers = forbid_dimers
            self.n_1b_seen = 0
            self.n_2b_seen = 0

        def _e1(self, x, y, z):
            pf = np.array([x, y, z], dtype=np.float64) @ inv_H
            df = pf[None, :] - fw_frac
            df -= np.round(df)
            d = np.sqrt(np.sum((df @ H) ** 2, axis=-1))
            r = np.maximum(d, 1.5)
            return float(np.sum(35.0 * ((2.6 / r) ** 12 - (2.6 / r) ** 6)))

        def eval_batch(self, states, mace_chunk=None):
            out = []
            for st in states:
                n = int(st["N"])
                if n == 1:
                    self.n_1b_seen += 1
                    out.append((self._e1(st["cx"][0], st["cy"][0], st["cz"][0]), False))
                elif n == 2:
                    if self.forbid_dimers:
                        raise AssertionError("only_1b=True must never evaluate 2B dimer states!")
                    self.n_2b_seen += 1
                    e1a = self._e1(st["cx"][0], st["cy"][0], st["cz"][0])
                    e1b = self._e1(st["cx"][1], st["cy"][1], st["cz"][1])
                    sa = {"N": 1, "cx": st["cx"][:1], "cy": st["cy"][:1], "cz": st["cz"][:1],
                          "qw": st["qw"][:1], "qx": st["qx"][:1], "qy": st["qy"][:1], "qz": st["qz"][:1]}
                    sb = {"N": 1, "cx": st["cx"][1:], "cy": st["cy"][1:], "cz": st["cz"][1:],
                          "qw": st["qw"][1:], "qx": st["qx"][1:], "qy": st["qy"][1:], "qz": st["qz"][1:]}
                    ulj0, uq0, rcom, _ = _unscaled_gg_pair_energy_K(sa, sb, H, inv_H, guest, r_cut_gg=6.72)
                    u2 = 0.80 * ulj0 + uq0 - 65.0 * math.exp(-((rcom - 3.9) / 0.8) ** 2)
                    out.append((float(e1a + e1b + u2), False))
            return out

    cache_dir = tmp_path / "distill_cache"

    # 1. only_1b=True: skips 2B completely, outputs zero 128-pt LUT and non-null state_dict model_sha256
    oracle_1b = _PureNumpyOracle(weight_val=1.0, forbid_dimers=True)
    res_1b = distill_mlip_surrogate(
        cif_path=str(cif_stub), fw_data=fw_data, guest=guest, supercell=(1, 1, 1),
        target=oracle_1b, grid_dims=(8, 8, 8), n_1b=60, n_2b=64,
        distill_seed=20269101, cache_dir=str(cache_dir), model_name="medium",
        dispersion=True, dispersion_xc="pbe", dtype="float32", only_1b=True,
    )
    s1 = res_1b["summary"]
    assert res_1b["cache_hit"] is False
    assert s1["version"] == "v1.2"
    assert s1["only_1b"] is True
    assert s1["n_2b_evaluated"] == 0 and s1["n_2b_train"] == 0 and s1["n_2b_val"] == 0
    assert s1["t_mace2_s"] == 0.0
    assert s1["r2_2b"] is None and s1["mae_2b_kJ_mol"] is None
    assert s1["model_sha256"] is not None and s1["model_sha256"].startswith("state_dict:")
    assert res_1b["gg_lj_scale"] == 1.0 and res_1b["gg_q_scale"] == 1.0
    assert res_1b["gg_com_lut_K"].shape == (128,) and np.all(res_1b["gg_com_lut_K"] == 0.0)
    assert oracle_1b.n_2b_seen == 0 and oracle_1b.n_1b_seen == s1["n_1b_evaluated"]

    # 2. Cache hit with same weights (and with target=None), then cache invalidation if weights change
    hit_1b = distill_mlip_surrogate(
        cif_path=str(cif_stub), fw_data=fw_data, guest=guest, supercell=(1, 1, 1),
        target=None, grid_dims=(8, 8, 8), n_1b=60, n_2b=64,
        distill_seed=20269101, cache_dir=str(cache_dir), model_name="medium",
        dispersion=True, dispersion_xc="pbe", dtype="float32", only_1b=True,
    )
    assert hit_1b["cache_hit"] is True
    assert hit_1b["distill_sha256"] == res_1b["distill_sha256"]

    oracle_1b_v2 = _PureNumpyOracle(weight_val=2.0, forbid_dimers=True)
    res_1b_v2 = distill_mlip_surrogate(
        cif_path=str(cif_stub), fw_data=fw_data, guest=guest, supercell=(1, 1, 1),
        target=oracle_1b_v2, grid_dims=(8, 8, 8), n_1b=60, n_2b=64,
        distill_seed=20269101, cache_dir=str(cache_dir), model_name="medium",
        dispersion=True, dispersion_xc="pbe", dtype="float32", only_1b=True,
    )
    assert res_1b_v2["cache_hit"] is False
    assert res_1b_v2["summary"]["model_sha256"] != s1["model_sha256"]

    # 3. only_1b=False uses a separate cache key and records honest 2B hold-out metrics
    oracle_full = _PureNumpyOracle(weight_val=1.0, forbid_dimers=False)
    res_full = distill_mlip_surrogate(
        cif_path=str(cif_stub), fw_data=fw_data, guest=guest, supercell=(1, 1, 1),
        target=oracle_full, grid_dims=(8, 8, 8), n_1b=60, n_2b=48,
        distill_seed=20269101, cache_dir=str(cache_dir), model_name="medium",
        dispersion=True, dispersion_xc="pbe", dtype="float32", only_1b=False,
    )
    sf = res_full["summary"]
    assert res_full["distill_sha256"] != res_1b["distill_sha256"]
    assert sf["only_1b"] is False
    assert oracle_full.n_2b_seen >= 24
    assert sf["n_2b_val"] >= 16 and sf["n_2b_train"] > 0
    assert sf["r2_2b"] == sf["r2_2b_deployed_holdout"]
    assert sf["mae_2b_kJ_mol"] == sf["mae_2b_deployed_holdout_kJ_mol"]
    assert sf["r2_2b_train"] is not None and sf["mae_2b_train_kJ_mol"] is not None
    assert sf["mae_2b_classical_holdout_kJ_mol"] is not None
    assert sf["model_sha256"] == s1["model_sha256"]


def test_only_1b_distill_end_to_end_cpu(tmp_path):
    """Pure-CPU end-to-end test of distill_mlip_surrogate(..., only_1b=True) with real torch + ASE."""
    pytest.importorskip("torch")
    pytest.importorskip("ase")
    from ase import Atoms

    # Minimal cubic 2-atom framework for fast (<1 s) pure-CPU distillation
    uc = Atoms("MgO", positions=[[0.0, 0.0, 0.0], [2.1, 2.1, 2.1]], cell=[10.0, 10.0, 10.0], pbc=True)
    fw_data = {"uc": uc, "charges": np.array([1.2, -1.2], dtype=np.float64)}
    cif_stub = tmp_path / "mini_mof.cif"
    cif_stub.write_text("# synthetic cubic framework\n", encoding="utf-8")
    guest = make_distilled_guest_spec(load_guest("ar"))

    class _CpuMockTarget:
        def __init__(self):
            self.device = None
            self.model_file_path = None
            self.mace_model = SimpleNamespace(
                state_dict=lambda: {"layer.weight": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
            )
            self.n_calls = 0

        def eval_batch(self, states, mace_chunk=None):
            self.n_calls += len(states)
            out = []
            for st in states:
                assert int(st["N"]) == 1, "only_1b=True must never evaluate 2B dimer states!"
                r = np.array([st["cx"][0], st["cy"][0], st["cz"][0]], dtype=np.float64)
                d = np.linalg.norm(((r / 10.0 + 0.5) % 1.0 - 0.5) * 10.0)
                u_K = 40.0 * ((2.6 / max(d, 1.5)) ** 12 - (2.6 / max(d, 1.5)) ** 6)
                out.append((float(u_K), False))
            return out

    oracle = _CpuMockTarget()
    cache_dir = tmp_path / "distill_cache"
    res = distill_mlip_surrogate(
        cif_path=str(cif_stub),
        fw_data=fw_data,
        guest=guest,
        supercell=(1, 1, 1),
        target=oracle,
        grid_dims=(8, 8, 8),
        n_1b=48,
        n_2b=64,
        distill_seed=20269001,
        cache_dir=str(cache_dir),
        model_name="medium",
        dispersion=True,
        dispersion_xc="pbe",
        dtype="float32",
        only_1b=True,
    )
    s = res["summary"]
    assert s["only_1b"] is True
    assert s["version"] == "v1.2"
    assert s["n_2b_evaluated"] == 0
    assert s["n_2b_train"] == 0
    assert s["n_2b_val"] == 0
    assert s["t_mace2_s"] == 0.0
    assert s["r2_2b"] is None and s["mae_2b_kJ_mol"] is None
    assert s["model_sha256"] is not None and s["model_sha256"].startswith("state_dict:")
    assert res["gg_lj_scale"] == 1.0
    assert res["gg_q_scale"] == 1.0
    assert res["gg_com_lut_K"].shape == (128,)
    assert np.all(res["gg_com_lut_K"] == 0.0)
    assert oracle.n_calls == s["n_1b_evaluated"]

    # Cache hit works without loading target
    hit = distill_mlip_surrogate(
        cif_path=str(cif_stub),
        fw_data=fw_data,
        guest=guest,
        supercell=(1, 1, 1),
        target=None,
        grid_dims=(8, 8, 8),
        n_1b=48,
        n_2b=64,
        distill_seed=20269001,
        cache_dir=str(cache_dir),
        model_name="medium",
        dispersion=True,
        dispersion_xc="pbe",
        dtype="float32",
        only_1b=True,
    )
    assert hit["cache_hit"] is True
    assert hit["summary"]["only_1b"] is True
    assert hit["summary"]["model_sha256"] == s["model_sha256"]
    assert np.all(hit["gg_com_lut_K"] == 0.0)


def test_external_grid_and_gg_lut_parity_and_reject_restore():
    """Gate 1: External 3D grid + 2B LUT energy parity (<1e-8 K) and outer-reject bit-exact restore."""
    cif = str(DATA / "MgMOF-74.cif")
    fw_data = extract_framework_for_proposer(cif, charge_mode="auto")
    guest = make_distilled_guest_spec(load_guest("co2"))
    distill_res = _make_synthetic_oms_distill_res(fw_data, guest, grid_dims=(64, 64, 24))

    prop = CudaGcmcProposer.from_distilled(
        fw_data=fw_data, guest=guest, distill_res=distill_res,
        supercell=(1, 1, 2), T_K=298.0, pressure_bar=0.28,
        n_boxes=4, max_mol=64, seed=20261101, cbmc_k=128, p_ins_del=0.60,
    )
    try:
        prop.equilibrate(400)
        st_acc = prop.get_accepted_quat()
        assert sum(s["N"] for s in st_acc) > 0

        # 1. Incremental U_ref vs fresh eval_energy_quat (<1e-8 K)
        for s in st_acc:
            u_eval, ov = prop.eval_energy_quat(s)
            assert not ov
            assert abs(u_eval - s["U_ref_K"]) < 1e-8

        # 2. Outer reject preserves state bit-for-bit while RNG strictly advances
        rng0 = prop.get_rng_states()
        _ = prop.propose_k_quat(15)
        prop.commit([0, 0, 0, 0])
        rng1 = prop.get_rng_states()
        st_after = prop.get_accepted_quat()
        assert not np.array_equal(rng0, rng1)
        for a, b in zip(st_acc, st_after):
            assert a["N"] == b["N"]
            assert a["U_ref_K"] == b["U_ref_K"]
            assert np.array_equal(a["cx"], b["cx"])
            assert np.array_equal(a["qw"], b["qw"])

        # 3. Verify Python 2-body formula matches CUDA pair energy for a dimer in open space
        H = np.asarray(prop.H_cols.T, dtype=np.float64)
        inv_H = np.linalg.inv(H)
        sa = {"N": 1, "cx": np.array([1.0]), "cy": np.array([1.0]), "cz": np.array([1.0]),
              "qw": np.array([1.0]), "qx": np.array([0.0]), "qy": np.array([0.0]), "qz": np.array([0.0])}
        sb = {"N": 1, "cx": np.array([4.8]), "cy": np.array([1.2]), "cz": np.array([1.0]),
              "qw": np.array([0.8]), "qx": np.array([0.6]), "qy": np.array([0.0]), "qz": np.array([0.0])}
        sab = {"N": 2, "cx": np.array([1.0, 4.8]), "cy": np.array([1.0, 1.2]), "cz": np.array([1.0, 1.0]),
               "qw": np.array([1.0, 0.8]), "qx": np.array([0.0, 0.6]), "qy": np.array([0.0, 0.0]), "qz": np.array([0.0, 0.0])}
        ua, _ = prop.eval_energy_quat(sa)
        ub, _ = prop.eval_energy_quat(sb)
        uab, _ = prop.eval_energy_quat(sab)
        u2_cuda = uab - ua - ub
        ulj0, uq0, rc_com, _ = _unscaled_gg_pair_energy_K(
            sa, sb, H, inv_H, guest, r_cut_gg=prop.r_cut_gg, dsf_alpha=0.20
        )
        lut_arr = distill_res["gg_com_lut_K"]
        u_pos = (rc_com - 2.0) * (len(lut_arr) - 1) / (10.0 - 2.0)
        i0 = int(math.floor(u_pos))
        f = u_pos - i0
        lut_val = float(lut_arr[i0]) * (1.0 - f) + float(lut_arr[i0 + 1]) * f
        u2_py = distill_res["gg_lj_scale"] * ulj0 + (distill_res["gg_q_scale"] ** 2) * uq0 + lut_val
        assert abs(u2_cuda - u2_py) < 1e-6, f"CUDA vs Python 2B LUT mismatch: {u2_cuda} vs {u2_py}"
    finally:
        prop.close()


@pytest.mark.slow
def test_gpu_cbmc_128_vs_1_zero_bias_from_empty_and_high_loading():
    """Gate 2: 128-thread GPU-CBMC must sample the exact same grand-canonical distribution as cbmc_k=1."""
    cif = str(DATA / "MgMOF-74.cif")
    fw_data = extract_framework_for_proposer(cif, charge_mode="auto")
    guest = load_guest("co2_epm2")
    grid_dims = compute_grid_dims(fw_data["uc"], 0.14)
    T_K, P_bar = 298.0, 0.02

    p_ref = CudaGcmcProposer.from_atoms(
        fw_data=fw_data, guest=guest, supercell=(1, 1, 2),
        T_K=T_K, pressure_bar=P_bar, ff_mode=1, pol_mode=0,
        grid_dims=grid_dims, n_boxes=16, max_mol=128, seed=20262001, cbmc_k=1, p_ins_del=0.50,
    )
    p_cbmc_empty = CudaGcmcProposer.from_atoms(
        fw_data=fw_data, guest=guest, supercell=(1, 1, 2),
        T_K=T_K, pressure_bar=P_bar, ff_mode=1, pol_mode=0,
        grid_dims=grid_dims, n_boxes=16, max_mol=128, seed=20262002, cbmc_k=128, p_ins_del=0.50,
    )
    p_cbmc_high = CudaGcmcProposer.from_atoms(
        fw_data=fw_data, guest=guest, supercell=(1, 1, 2),
        T_K=T_K, pressure_bar=P_bar, ff_mode=1, pol_mode=0,
        grid_dims=grid_dims, n_boxes=16, max_mol=128, seed=20262003, cbmc_k=128, p_ins_del=0.50,
    )
    try:
        p_cbmc_high.set_chem_pot_shift(-12.0)
        p_cbmc_high.equilibrate(1500)
        hl_N = [s["N"] for s in p_cbmc_high.get_accepted_quat()]
        assert min(hl_N) >= 35
        p_cbmc_high.set_chem_pot_shift(0.0)

        p_ref.equilibrate(35000)
        p_cbmc_empty.equilibrate(6000)
        p_cbmc_high.equilibrate(6000)

        def _sample(prop_obj, stride, n_draws):
            ns = [[] for _ in range(16)]
            for _ in range(n_draws):
                prop_obj.equilibrate(stride)
                for b, s in enumerate(prop_obj.get_accepted_quat()):
                    ns[b].append(float(s["N"]))
            bmeans = np.array([np.mean(x) for x in ns])
            return float(np.mean(bmeans)), float(np.std(bmeans, ddof=1) / math.sqrt(16)), float(np.var(np.asarray(ns), ddof=1))

        m_ref, se_ref, v_ref = _sample(p_ref, 150, 120)
        m_emp, se_emp, v_emp = _sample(p_cbmc_empty, 35, 120)
        m_hi, se_hi, v_hi = _sample(p_cbmc_high, 35, 120)

        for tag, m_val, se_val, v_val in [("empty", m_emp, se_emp, v_emp), ("high", m_hi, se_hi, v_hi)]:
            se_c = math.sqrt(se_ref ** 2 + se_val ** 2 + 1e-6)
            assert abs(m_val - m_ref) < max(1.0, 3.0 * se_c), (
                f"GPU-CBMC ({tag}) mean {m_val:.3f}+-{se_val:.3f} vs k=1 ref {m_ref:.3f}+-{se_ref:.3f}"
            )
            assert abs(v_val - v_ref) < max(2.5, 0.45 * v_ref), (
                f"GPU-CBMC ({tag}) var {v_val:.2f} vs k=1 ref {v_ref:.2f}"
            )
    finally:
        p_ref.close()
        p_cbmc_empty.close()
        p_cbmc_high.close()


@pytest.fixture(scope="module")
def mercado_distill_bundle(tmp_path_factory):
    """Run real distill_mlip_surrogate + cache round-trip once against classical Mercado oracle."""
    cif = str(DATA / "MgMOF-74.cif")
    fw_data = extract_framework_for_proposer(cif, charge_mode="auto")
    guest_tgt = load_guest("co2_epm2")
    guest_dist = make_distilled_guest_spec(guest_tgt)
    grid_dims = compute_grid_dims(fw_data["uc"], 0.14)
    tgt = CudaGcmcProposer.from_atoms(
        fw_data=fw_data, guest=guest_tgt, supercell=(1, 1, 2),
        T_K=298.0, pressure_bar=0.02, ff_mode=1, pol_mode=0,
        grid_dims=grid_dims, n_boxes=1, max_mol=128, seed=20263001, cbmc_k=128, p_ins_del=0.50,
    )

    class _MockTargetOracle:
        def __init__(self, prop_obj, fw_uc):
            import torch
            self._prop = prop_obj
            self.uc = fw_uc
            self.fw = fw_uc.repeat((1, 1, 2))
            self.r_max = 6.0
            self.dispersion = False
            self.d3_xc = ""
            self.dtype = "float32"
            self.model_source = "mock_mercado"
            self.device = torch.device("cuda:0") if torch.cuda.is_available() else None
            self.guest_local_pos = guest_dist.real_local_positions()
            self.eval_calls = 0

        def eval_batch(self, states, mace_chunk=None):
            self.eval_calls += len(states)
            return [self._prop.eval_energy_quat(s) for s in states]

    oracle = _MockTargetOracle(tgt, fw_data["uc"])
    cache_dir = tmp_path_factory.mktemp("distill_cache_gate3")
    distill_res = distill_mlip_surrogate(
        cif_path=cif,
        fw_data=fw_data,
        guest=guest_dist,
        supercell=(1, 1, 2),
        target=oracle,
        grid_dims=(48, 48, 20),
        n_1b=320,
        n_2b=80,
        distill_seed=20263055,
        cache_dir=str(cache_dir),
        model_name="mock_mercado",
        dispersion=False,
        dispersion_xc="",
        dtype="float32",
    )
    assert distill_res["summary"]["r2_1b"] > 0.85
    reloaded = load_distill_cache(distill_res["cache_npz_path"])
    assert np.allclose(reloaded["grids_u_lj"], distill_res["grids_u_lj"], atol=1e-6)
    try:
        yield cif, fw_data, guest_tgt, guest_dist, grid_dims, reloaded
    finally:
        tgt.close()


@pytest.mark.slow
@pytest.mark.parametrize("shift_kJ", [-3.0, 0.0, +3.0])
def test_cross_potential_nested_mc_zero_bias_with_cbmc_128(mercado_distill_bundle, shift_kJ):
    """Gate 3: Distilled external grid + 2B LUT with cbmc_k=128 samples exact Mercado target with zero bias."""
    _, fw_data, guest_tgt, guest_dist, grid_dims, distill_res = mercado_distill_bundle
    T_K, P_bar = 298.0, 0.02

    tgt = CudaGcmcProposer.from_atoms(
        fw_data=fw_data, guest=guest_tgt, supercell=(1, 1, 2),
        T_K=T_K, pressure_bar=P_bar, ff_mode=1, pol_mode=0,
        grid_dims=grid_dims, n_boxes=8, max_mol=128, seed=20263001, cbmc_k=128, p_ins_del=0.50,
    )
    prop = CudaGcmcProposer.from_distilled(
        fw_data=fw_data, guest=guest_dist, distill_res=distill_res,
        supercell=(1, 1, 2), T_K=T_K, pressure_bar=P_bar,
        n_boxes=8, max_mol=128, seed=20263002 + int(shift_kJ * 10), cbmc_k=128, p_ins_del=0.50,
    )
    try:
        tgt.equilibrate(5000)
        box_n = []
        for _ in range(150):
            tgt.equilibrate(25)
            box_n.append([float(s["N"]) for s in tgt.get_accepted_quat()])
        arr_ref = np.asarray(box_n, dtype=np.float64)
        mean_ref = float(np.mean(arr_ref))
        sem_ref = float(np.std(np.mean(arr_ref, axis=0), ddof=1) / math.sqrt(8))

        prop.set_chem_pot_shift(shift_kJ)
        prop.equilibrate(4000)
        res = run_nested_mc(
            prop, None, k_steps=12, eq_cycles=150, prod_cycles=350,
            seed=20263100 + int(shift_kJ * 10),
            batch_target_fn=lambda sts: [tgt.eval_energy_quat(s) for s in sts],
        )
        mean_nest = float(np.mean(res["prod_N_history"]))
        sem_nest = float(res["uptake_sem_mmol_g"] * prop.fw_mass_gmol / 1000.0)
        se_c = math.sqrt(sem_ref ** 2 + sem_nest ** 2 + 1e-6)
        assert res["prod_ins_acc"] >= 25
        assert res["prod_del_acc"] >= 25
        assert abs(mean_nest - mean_ref) < max(1.5, 3.2 * se_c), (
            f"Nested (cbmc_k=128, shift={shift_kJ}) mean {mean_nest:.3f}+-{sem_nest:.3f} vs {mean_ref:.3f}+-{sem_ref:.3f}"
        )
    finally:
        prop.close()
        tgt.close()


def test_cbmc_128_external_grid_bit_exact_checkpoint_resume():
    """Gate 4: Bit-exact checkpoint save and extend resume under cbmc_k=128 + external grids + 2B LUT."""
    cif = str(DATA / "MgMOF-74.cif")
    fw_data = extract_framework_for_proposer(cif, charge_mode="auto")
    guest = make_distilled_guest_spec(load_guest("co2"))
    distill_res = _make_synthetic_oms_distill_res(fw_data, guest, grid_dims=(64, 64, 24))
    T_K, P_bar = 298.0, 0.28

    prop_tgt = CudaGcmcProposer.from_distilled(
        fw_data=fw_data, guest=guest, distill_res=distill_res,
        supercell=(1, 1, 2), T_K=T_K, pressure_bar=P_bar,
        n_boxes=1, max_mol=128, seed=20264001, cbmc_k=128, p_ins_del=0.60,
    )
    prop_cont = CudaGcmcProposer.from_distilled(
        fw_data=fw_data, guest=guest, distill_res=distill_res,
        supercell=(1, 1, 2), T_K=T_K, pressure_bar=P_bar,
        n_boxes=4, max_mol=128, seed=20264002, cbmc_k=128, p_ins_del=0.60,
    )
    try:
        prop_cont.set_chem_pot_shift(-1.5)
        res_cont = run_nested_mc(
            prop_cont, None, k_steps=10, eq_cycles=40, prod_cycles=80,
            seed=20264099, batch_target_fn=lambda sts: [prop_tgt.eval_energy_quat(s) for s in sts],
        )
    finally:
        prop_cont.close()

    ckpt_store = {}
    prop_p1 = CudaGcmcProposer.from_distilled(
        fw_data=fw_data, guest=guest, distill_res=distill_res,
        supercell=(1, 1, 2), T_K=T_K, pressure_bar=P_bar,
        n_boxes=4, max_mol=128, seed=20264002, cbmc_k=128, p_ins_del=0.60,
    )
    try:
        prop_p1.set_chem_pot_shift(-1.5)
        _ = run_nested_mc(
            prop_p1, None, k_steps=10, eq_cycles=40, prod_cycles=40,
            seed=20264099, batch_target_fn=lambda sts: [prop_tgt.eval_energy_quat(s) for s in sts],
            checkpoint_fn=lambda c: ckpt_store.__setitem__("c", json.loads(json.dumps(c))),
            checkpoint_every=80,
        )
    finally:
        prop_p1.close()

    prop_p2 = CudaGcmcProposer.from_distilled(
        fw_data=fw_data, guest=guest, distill_res=distill_res,
        supercell=(1, 1, 2), T_K=T_K, pressure_bar=P_bar,
        n_boxes=4, max_mol=128, seed=99999999, cbmc_k=128, p_ins_del=0.60,
    )
    try:
        prop_p2.set_chem_pot_shift(-1.5)
        res_resumed = run_nested_mc(
            prop_p2, None, k_steps=10, eq_cycles=40, prod_cycles=80,
            seed=20264099, batch_target_fn=lambda sts: [prop_tgt.eval_energy_quat(s) for s in sts],
            resume_state=ckpt_store["c"],
        )
    finally:
        prop_p2.close()
        prop_tgt.close()

    assert res_resumed["prod_N_history"] == res_cont["prod_N_history"]
    assert np.allclose(res_resumed["prod_U_history"], res_cont["prod_U_history"], atol=1e-12)
    assert np.allclose(res_resumed["prod_U_ref_history"], res_cont["prod_U_ref_history"], atol=1e-12)
    assert res_resumed["acceptance_breakdown"] == res_cont["acceptance_breakdown"]
    assert res_resumed["final_cuda_rng_states"] == res_cont["final_cuda_rng_states"]
    for b in range(4):
        assert np.array_equal(res_resumed["final_states"][b]["cx"], res_cont["final_states"][b]["cx"])
