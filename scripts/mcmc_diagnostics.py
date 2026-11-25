"""Stan/Vehtari (2021) MCMC convergence diagnostics, chain stall/drift audit, and independent Qst validation."""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.stats import norm

K_TO_KJMOL = 8.31446261815324e-3


def _split_chains(arr_2d: np.ndarray) -> np.ndarray:
    """Split (n_cyc, n_chains) array into (2 * n_chains, half_cyc) segments."""
    n_cyc, n_c = arr_2d.shape
    half = n_cyc // 2
    if half < 4:
        return np.empty((2 * n_c, 0), dtype=np.float64)
    # Use tail 2 * half cycles so halves have equal length
    sub = arr_2d[n_cyc - 2 * half:, :]
    c1 = sub[:half, :].T
    c2 = sub[half:, :].T
    return np.concatenate([c1, c2], axis=0)


def _rank_normalize(chains_2d: np.ndarray) -> np.ndarray:
    """Blom rank-normalization across pooled split draws (Vehtari et al. 2021)."""
    M, S = chains_2d.shape
    flat = chains_2d.reshape(-1)
    # Average ranks for ties via argsort trick
    order = np.argsort(flat, kind="mergesort")
    sorted_vals = flat[order]
    ranks_sorted = np.empty_like(sorted_vals, dtype=np.float64)
    n_tot = len(flat)
    i = 0
    while i < n_tot:
        j = i + 1
        while j < n_tot and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_r = 0.5 * ((i + 1) + j)  # 1-based average rank
        ranks_sorted[i:j] = avg_r
        i = j
    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    u = (ranks - 0.375) / (n_tot + 0.25)
    z = norm.ppf(np.clip(u, 1e-12, 1.0 - 1e-12))
    return z.reshape(M, S)


def _rhat_and_ess_from_split(chains_ms: np.ndarray) -> Tuple[Optional[float], float]:
    """Compute Stan multi-chain R-hat and Vehtari et al. (2021) Geyer IPS ESS on (M, S) split chains."""
    M, S = chains_ms.shape
    if M < 2 or S < 4:
        return None, 0.0
    # Zero variance overall or all chains internally zero variance with identical means
    chain_means = np.mean(chains_ms, axis=1)
    chain_vars = np.var(chains_ms, axis=1, ddof=1)
    W = float(np.mean(chain_vars))
    if np.var(chains_ms) <= 1e-15 or W <= 1e-15:
        # Constant chains (or piecewise constant plateaus): cannot compute valid R-hat/ESS
        return None, 0.0

    B = float(S * np.var(chain_means, ddof=1))
    var_hat = ((S - 1.0) / S) * W + (1.0 / S) * B
    if var_hat <= 1e-15:
        return None, 0.0
    rhat = math.sqrt(var_hat / W)

    # Autocorrelation per chain via FFT
    n_fft = 1 << (2 * S - 1).bit_length()
    acov_chains = np.zeros((M, S), dtype=np.float64)
    for m in range(M):
        xm = chains_ms[m] - chain_means[m]
        f = np.fft.rfft(xm, n=n_fft)
        c = np.fft.irfft(f * np.conjugate(f), n=n_fft)[:S]
        c /= float(S)  # biased FFT autocovariance matches Stan Geyer variogram normalization
        acov_chains[m] = c

    # Geyer combined lag correlation rho_k = 1 - (W - mean_m acov_{m,k}) / var_hat
    # acov_chains[m, k] is (1/S) sum (x_t - mean)(x_{t+k} - mean); at k=0 it equals var_m * (S-1)/S.
    # Multiply by S/(S-1) so k=0 equals W (exact Stan/Vehtari 2021 positive-definite FFT normalization).
    mean_acov_k = np.mean(acov_chains, axis=0) * (float(S) / float(S - 1))
    rho = 1.0 - (W - mean_acov_k) / var_hat
    rho[0] = 1.0

    # Geyer initial positive monotone sequence over pairs P_t = rho_{2t} + rho_{2t+1}
    p_vals = []
    max_t = (S // 2) - 1
    for t in range(max_t):
        p_t = float(rho[2 * t] + rho[2 * t + 1])
        if p_t <= 0.0:
            break
        if p_vals and p_t > p_vals[-1]:
            p_t = p_vals[-1]
        p_vals.append(p_t)
    # Note P_0 = rho_0 + rho_1, so tau = -1 + 2 * sum(P_t) when t starts at 0
    tau = max(1.0, -1.0 + 2.0 * float(sum(p_vals))) if p_vals else 1.0
    ess = min(float(M * S), float(M * S) / tau)
    return float(rhat), float(ess)


def compute_vehtari_mcmc_stats(trace_2d) -> Dict[str, Any]:
    """Compute Stan rank-normalized split/folded R-hat, bulk/tail ESS, and MCSE for (n_cyc, n_chains)."""
    arr = np.asarray(trace_2d, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D (n_cyc, n_chains) array, got shape {arr.shape}")
    n_cyc, n_chains = arr.shape
    if n_cyc < 8 or n_chains < 2:
        return {
            "n_cycles": int(n_cyc),
            "n_chains": int(n_chains),
            "rhat_rank": None,
            "rhat_folded": None,
            "rhat_max": None,
            "ess_bulk": 0.0,
            "ess_tail": 0.0,
            "ess_mean": 0.0,
            "mcse": None,
            "constant_chains": int(sum(1 for c in range(n_chains) if np.var(arr[:, c]) <= 1e-15)),
            "all_constant": bool(np.var(arr) <= 1e-15),
            "estimable": False,
            "estimable_reason": "trace_too_short_or_few_chains",
        }

    const_chains = int(sum(1 for c in range(n_chains) if np.var(arr[:, c]) <= 1e-15))
    all_const = bool(np.var(arr) <= 1e-15)
    split_raw = _split_chains(arr)
    if all_const or const_chains == n_chains:
        # Either completely flat or chains stuck on distinct constant plateaus
        distinct_plateaus = len({round(float(np.mean(arr[:, c])), 9) for c in range(n_chains)}) > 1
        return {
            "n_cycles": int(n_cyc),
            "n_chains": int(n_chains),
            "rhat_rank": None,
            "rhat_folded": None,
            "rhat_max": None,
            "ess_bulk": 0.0,
            "ess_tail": 0.0,
            "ess_mean": 0.0,
            "mcse": None,
            "constant_chains": const_chains,
            "all_constant": all_const,
            "distinct_constant_plateaus": distinct_plateaus,
            "estimable": False,
            "estimable_reason": "distinct_constant_plateaus" if distinct_plateaus else "all_chains_constant",
        }

    z_bulk = _rank_normalize(split_raw)
    rhat_bulk, ess_bulk = _rhat_and_ess_from_split(z_bulk)

    folded_raw = np.abs(split_raw - float(np.median(split_raw)))
    z_folded = _rank_normalize(folded_raw)
    rhat_folded, _ = _rhat_and_ess_from_split(z_folded)

    q05 = float(np.quantile(split_raw, 0.05))
    q95 = float(np.quantile(split_raw, 0.95))
    i05 = (split_raw <= q05).astype(np.float64)
    i95 = (split_raw >= q95).astype(np.float64)
    _, ess_q05 = _rhat_and_ess_from_split(i05)
    _, ess_q95 = _rhat_and_ess_from_split(i95)
    tail_candidates = [v for v in (ess_q05, ess_q95) if v > 0.0]
    ess_tail = min(tail_candidates) if len(tail_candidates) == 2 else 0.0

    _, ess_mean = _rhat_and_ess_from_split(split_raw)
    # Pooled variance estimate var_hat_plus on raw split chains
    M, S = split_raw.shape
    W_raw = float(np.mean(np.var(split_raw, axis=1, ddof=1)))
    B_raw = float(S * np.var(np.mean(split_raw, axis=1), ddof=1))
    var_plus = ((S - 1.0) / S) * W_raw + (1.0 / S) * B_raw
    sd_raw = math.sqrt(max(0.0, var_plus))
    mcse = float(sd_raw / math.sqrt(ess_mean)) if ess_mean > 0.0 else None

    rhat_max = None
    if rhat_bulk is not None and rhat_folded is not None:
        rhat_max = max(rhat_bulk, rhat_folded)
    elif rhat_bulk is not None:
        rhat_max = rhat_bulk
    elif rhat_folded is not None:
        rhat_max = rhat_folded

    estimable = (rhat_max is not None) and (ess_bulk > 0.0) and (mcse is not None) and (const_chains == 0)
    reason = None
    if const_chains > 0:
        reason = f"{const_chains}_constant_chains"
    elif not estimable:
        reason = "non_estimable_variance_or_ess"

    return {
        "n_cycles": int(n_cyc),
        "n_chains": int(n_chains),
        "rhat_rank": rhat_bulk,
        "rhat_folded": rhat_folded,
        "rhat_max": rhat_max,
        "ess_bulk": float(ess_bulk),
        "ess_tail": float(ess_tail),
        "ess_mean": float(ess_mean),
        "mcse": mcse,
        "constant_chains": const_chains,
        "all_constant": all_const,
        "distinct_constant_plateaus": False,
        "estimable": bool(estimable),
        "estimable_reason": reason,
    }


def compute_independent_qst(
    prod_N_history,
    prod_U_history_kJ,
    T_K: float,
    prod_ins_acc: int,
    prod_del_acc: int,
    block_size: Optional[int] = None,
    target_sem_kJ_mol: float = 2.5,
) -> Dict[str, Any]:
    """Independent fluctuation Qst with moving-block / per-chain uncertainty and strict non-fallback guard."""
    n_arr = np.asarray(prod_N_history, dtype=np.float64)
    u_kJ = np.asarray(prod_U_history_kJ, dtype=np.float64)
    if n_arr.ndim != 2 or u_kJ.shape != n_arr.shape or n_arr.shape[0] < 8:
        return {
            "qst_fluctuation_kJ_mol": None,
            "qst_fluctuation_sem_kJ_mol": None,
            "qst_status": "insufficient",
            "qst_reasons": ["insufficient_production_trace"],
        }

    u_K = u_kJ / K_TO_KJMOL
    n_cyc, n_c = n_arr.shape
    flat_n = n_arr.reshape(-1)
    flat_u = u_K.reshape(-1)
    mN = float(np.mean(flat_n))
    mU = float(np.mean(flat_u))
    varN = float(np.mean((flat_n - mN) ** 2))
    covUN = float(np.mean((flat_u - mU) * (flat_n - mN)))

    reasons: List[str] = []
    if varN <= 1e-12:
        reasons.append("varN_near_zero_no_fallback")
    if (int(prod_ins_acc) + int(prod_del_acc)) < 10:
        reasons.append(f"insufficient_particle_exchanges({prod_ins_acc}+{prod_del_acc}<10)")
    if n_c < 4:
        reasons.append(f"n_chains({n_c})<4")

    if varN <= 1e-12:
        return {
            "qst_fluctuation_kJ_mol": None,
            "qst_fluctuation_sem_kJ_mol": None,
            "qst_status": "insufficient",
            "qst_reasons": reasons,
        }

    q_pooled = float((T_K - covUN / varN) * K_TO_KJMOL)

    # Leave-one-chain-out jackknife + autocorrelation-preserving temporal block jackknife within chains
    per_chain_q = []
    for c in range(n_c):
        nc = n_arr[:, c]
        uc = u_K[:, c]
        vn_c = float(np.var(nc))
        if vn_c > 1e-12:
            per_chain_q.append(float((T_K - np.mean((uc - np.mean(uc)) * (nc - np.mean(nc))) / vn_c) * K_TO_KJMOL))

    # Leave-one-chain-out jackknife (accounts for cross-chain dispersion)
    tot_m = float(flat_n.size)
    sN = float(np.sum(flat_n))
    sN2 = float(np.sum(flat_n * flat_n))
    sU = float(np.sum(flat_u))
    sUN = float(np.sum(flat_u * flat_n))
    loo_q = []
    for c in range(n_c):
        m_sub = tot_m - n_cyc
        sN_sub = sN - float(np.sum(n_arr[:, c]))
        sN2_sub = sN2 - float(np.sum(n_arr[:, c] ** 2))
        sU_sub = sU - float(np.sum(u_K[:, c]))
        sUN_sub = sUN - float(np.sum(u_K[:, c] * n_arr[:, c]))
        mn_s = sN_sub / m_sub
        vn_s = (sN2_sub / m_sub) - mn_s * mn_s
        if vn_s > 1e-12:
            cov_s = (sUN_sub / m_sub) - (sU_sub / m_sub) * mn_s
            loo_q.append(float((T_K - cov_s / vn_s) * K_TO_KJMOL))

    sem_chain = None
    if len(loo_q) == n_c and n_c > 1:
        q_arr = np.asarray(loo_q, dtype=np.float64)
        mu_l = float(np.mean(q_arr))
        sem_chain = math.sqrt(max(0.0, ((n_c - 1.0) / n_c) * float(np.sum((q_arr - mu_l) ** 2))))

    # Temporal block jackknife across contiguous blocks (handles correlation within chains too)
    bs = block_size if block_size is not None else max(10, n_cyc // 8)
    n_blk = n_cyc // bs
    sem_block = None
    if n_blk >= 4:
        blk_loo = []
        for k in range(n_blk):
            i0, i1 = k * bs, (k + 1) * bs
            mask = np.ones(n_cyc, dtype=bool)
            mask[i0:i1] = False
            nb = n_arr[mask].reshape(-1)
            ub = u_K[mask].reshape(-1)
            vnb = float(np.var(nb))
            if vnb > 1e-12:
                cb = float(np.mean((ub - np.mean(ub)) * (nb - np.mean(nb))))
                blk_loo.append(float((T_K - cb / vnb) * K_TO_KJMOL))
        if len(blk_loo) == n_blk:
            b_arr = np.asarray(blk_loo, dtype=np.float64)
            mu_b = float(np.mean(b_arr))
            sem_block = math.sqrt(max(0.0, ((n_blk - 1.0) / n_blk) * float(np.sum((b_arr - mu_b) ** 2))))

    cands = [v for v in (sem_chain, sem_block) if v is not None]
    sem_final = max(cands) if cands else None

    if sem_final is None or not math.isfinite(sem_final):
        reasons.append("qst_sem_not_estimable")
    elif sem_final > target_sem_kJ_mol:
        reasons.append(f"qst_sem({sem_final:.2f})>{target_sem_kJ_mol:.2f}_kJ_mol")

    if len(per_chain_q) < n_c:
        reasons.append(f"some_chains_zero_varN({n_c - len(per_chain_q)}/{n_c})")

    if any("insufficient_particle_exchanges" in r or "n_chains" in r or "zero_varN" in r or "not_estimable" in r for r in reasons):
        q_status = "insufficient" if (int(prod_ins_acc) + int(prod_del_acc)) < 10 or len(per_chain_q) < n_c or sem_final is None else "fail"
    elif reasons:
        q_status = "fail"
    else:
        q_status = "pass"

    return {
        "qst_fluctuation_kJ_mol": q_pooled,
        "qst_fluctuation_sem_kJ_mol": sem_final,
        "qst_chain_jackknife_sem_kJ_mol": sem_chain,
        "qst_block_jackknife_sem_kJ_mol": sem_block,
        "per_chain_qst_kJ_mol": per_chain_q,
        "qst_status": q_status,
        "qst_reasons": reasons,
    }


def evaluate_v2_convergence(
    prod_N_history,
    prod_U_history_kJ,
    conv_mmol_g: float,
    T_K: float,
    prod_ins_acc: int,
    prod_del_acc: int,
    prod_ins_att: int,
    prod_del_att: int,
    prod_outer_acc_rate: float,
    uptake_sem_mmol_g: float,
    acceptance_breakdown: Optional[dict] = None,
    oms_occ_history=None,
) -> Dict[str, Any]:
    """Comprehensive v2 diagnostics implementing §4.1 and §4.2 gates (pass / fail / insufficient / unknown)."""
    if not prod_N_history or not prod_U_history_kJ:
        return {"status": "unknown", "reasons": ["missing_production_traces"], "qst_status": "unknown"}

    n_arr = np.asarray(prod_N_history, dtype=np.float64)
    u_arr = np.asarray(prod_U_history_kJ, dtype=np.float64)
    if n_arr.ndim != 2 or n_arr.shape[0] < 8:
        return {"status": "insufficient", "reasons": ["too_few_production_cycles"], "qst_status": "insufficient"}

    n_cyc, n_c = n_arr.shape
    up_arr = n_arr * float(conv_mmol_g)
    mean_up = float(np.mean(up_arr))
    mean_N = float(np.mean(n_arr))

    stats_N = compute_vehtari_mcmc_stats(up_arr)
    stats_U = compute_vehtari_mcmc_stats(u_arr)

    # Per-chain metrics
    per_chain_drifts = []
    empty_fracs = []
    longest_stalls = []
    net_dN = []
    h = n_cyc // 2
    for c in range(n_c):
        nc = n_arr[:, c]
        uc = up_arr[:, c]
        empty_fracs.append(float(np.mean(nc == 0)))
        per_chain_drifts.append(float(np.mean(uc[h:]) - np.mean(uc[:h])))
        net_dN.append(int(nc[-1] - nc[0]))
        max_st, cur_st = 1, 1
        for t in range(1, n_cyc):
            if nc[t] == nc[t - 1]:
                cur_st += 1
                if cur_st > max_st:
                    max_st = cur_st
            else:
                cur_st = 1
        longest_stalls.append(int(max_st))

    mean_empty_frac = float(np.mean(empty_fracs))
    pooled_drift = float(abs(np.mean(per_chain_drifts)))
    max_abs_chain_drift = float(max(abs(x) for x in per_chain_drifts))

    # Dilute / rare-loading detection (§4.1)
    dilute_limit = bool(mean_N < 0.25 or mean_empty_frac > 0.85)
    min_ess_req = float(100 * n_c)  # 400 for 4 chains

    reasons: List[str] = []
    insufficient_flags: List[str] = []

    if n_c < 4:
        insufficient_flags.append(f"n_chains({n_c})<4")
    if (int(prod_ins_acc) + int(prod_del_acc)) < 10:
        insufficient_flags.append(f"low_exchanges(ins={prod_ins_acc},del={prod_del_acc})")
    if dilute_limit:
        insufficient_flags.append(f"dilute_regime(mean_N={mean_N:.3f},empty_frac={mean_empty_frac:.3f})")

    if stats_N["constant_chains"] > 0:
        max_chain_mean_N = float(max(np.mean(n_arr[:, c]) for c in range(n_c)))
        if (
            dilute_limit
            and not stats_N.get("distinct_constant_plateaus", False)
            and max_chain_mean_N < 0.25
        ):
            insufficient_flags.append(f"dilute_constant_N_chains({stats_N['constant_chains']}/{n_c})")
        else:
            reasons.append(f"constant_N_chains({stats_N['constant_chains']}/{n_c})")
    if not stats_N["estimable"]:
        insufficient_flags.append(f"N_stats_unestimable({stats_N['estimable_reason']})")
    else:
        if stats_N["rhat_max"] >= 1.01:
            reasons.append(f"N_rhat_max={stats_N['rhat_max']:.4f}>=1.01")
        if stats_N["ess_bulk"] < min_ess_req:
            reasons.append(f"N_ess_bulk={stats_N['ess_bulk']:.1f}<{min_ess_req:.0f}")
        if stats_N["ess_tail"] < min_ess_req:
            reasons.append(f"N_ess_tail={stats_N['ess_tail']:.1f}<{min_ess_req:.0f}")

    if not stats_U["estimable"]:
        insufficient_flags.append(f"U_stats_unestimable({stats_U['estimable_reason']})")
    else:
        if stats_U["rhat_max"] >= 1.01:
            reasons.append(f"U_rhat_max={stats_U['rhat_max']:.4f}>=1.01")
        if stats_U["ess_bulk"] < min_ess_req:
            reasons.append(f"U_ess_bulk={stats_U['ess_bulk']:.1f}<{min_ess_req:.0f}")
        if stats_U["ess_tail"] < min_ess_req:
            reasons.append(f"U_ess_tail={stats_U['ess_tail']:.1f}<{min_ess_req:.0f}")

    stats_OMS = None
    if oms_occ_history is not None:
        oms_arr = np.asarray(oms_occ_history, dtype=np.float64)
        if oms_arr.shape == n_arr.shape:
            stats_OMS = compute_vehtari_mcmc_stats(oms_arr)
            mean_oms = float(np.mean(oms_arr))
            if not dilute_limit and mean_oms >= 0.25:
                if not stats_OMS["estimable"]:
                    if stats_OMS.get("distinct_constant_plateaus", False) or stats_OMS["constant_chains"] > 0:
                        reasons.append(f"OMS_non_estimable({stats_OMS['estimable_reason']})")
                    else:
                        insufficient_flags.append(f"OMS_unestimable({stats_OMS['estimable_reason']})")
                else:
                    if stats_OMS["rhat_max"] >= 1.01:
                        reasons.append(f"OMS_rhat_max={stats_OMS['rhat_max']:.4f}>=1.01")
                    if stats_OMS["ess_bulk"] < min_ess_req:
                        reasons.append(f"OMS_ess_bulk={stats_OMS['ess_bulk']:.1f}<{min_ess_req:.0f}")

    mcse_val = stats_N["mcse"]
    eff_mcse = None if mcse_val is None else max(float(mcse_val), float(uptake_sem_mmol_g))
    mcse_target = max(0.05, 0.01 * abs(mean_up))
    if not dilute_limit:
        if eff_mcse is None:
            insufficient_flags.append("mcse_unestimable")
        elif eff_mcse > mcse_target:
            reasons.append(f"MCSE={eff_mcse:.4f}>{mcse_target:.4f}_mmol_g")

    if prod_outer_acc_rate < 0.05 and not dilute_limit:
        reasons.append(f"low_prod_outer_acc={prod_outer_acc_rate*100:.1f}%")

    drift_limit = max(0.25, 3.0 * float(eff_mcse if eff_mcse is not None else uptake_sem_mmol_g))
    if not dilute_limit and max_abs_chain_drift > drift_limit:
        reasons.append(f"max_chain_drift={max_abs_chain_drift:.3f}>{drift_limit:.3f}")

    if reasons:
        status = "fail"
    elif insufficient_flags:
        status = "insufficient"
    else:
        status = "pass"

    qst_res = compute_independent_qst(
        prod_N_history, prod_U_history_kJ, T_K=T_K,
        prod_ins_acc=prod_ins_acc, prod_del_acc=prod_del_acc,
    )

    return {
        "status": status,
        "converged": bool(status == "pass"),
        "reasons": reasons + insufficient_flags,
        "dilute_limit": dilute_limit,
        "min_ess_threshold": min_ess_req,
        "mcse_mmol_g": eff_mcse,
        "mcse_autocorr_mmol_g": mcse_val,
        "mcse_target_mmol_g": None if dilute_limit else mcse_target,
        "N_stats": stats_N,
        "U_stats": stats_U,
        "OMS_stats": stats_OMS,
        "per_chain_signed_drift_mmol_g": per_chain_drifts,
        "pooled_abs_drift_mmol_g": pooled_drift,
        "max_abs_chain_drift_mmol_g": max_abs_chain_drift,
        "empty_frac_per_chain": empty_fracs,
        "longest_stall_per_chain": longest_stalls,
        "net_dN_per_chain": net_dN,
        "prod_outer_acc_rate": float(prod_outer_acc_rate),
        "acceptance_breakdown": acceptance_breakdown or {},
        **qst_res,
    }
