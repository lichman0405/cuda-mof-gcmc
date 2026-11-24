#include "v2_gcmc.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace mofgcmc {

namespace {

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA Error at %s:%d: %s\n",                   \
                         __FILE__, __LINE__, cudaGetErrorString(err));          \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

constexpr int BLOCK_THREADS = 128;
constexpr int SMEM_MAX_MOL = 512;

__device__ inline uint64_t xorshift64star(uint64_t& s) {
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 2685821657736338717ULL;
}

__device__ inline double rng_u01(uint64_t& s) {
    return (double)(xorshift64star(s) >> 11) * (1.0 / 9007199254740992.0);
}

__device__ inline void random_unit_sphere(uint64_t& rng, double& ux, double& uy, double& uz) {
    double z1, z2, s2;
    do {
        z1 = 2.0 * rng_u01(rng) - 1.0;
        z2 = 2.0 * rng_u01(rng) - 1.0;
        s2 = z1 * z1 + z2 * z2;
    } while (s2 >= 1.0 || s2 < 1.0e-12);
    double sq = sqrt(1.0 - s2);
    ux = 2.0 * z1 * sq;
    uy = 2.0 * z2 * sq;
    uz = 1.0 - 2.0 * s2;
}

__device__ inline void rotate_rodrigues(
    uint64_t& rng, double max_angle,
    double ux, double uy, double uz,
    double& nx, double& ny, double& nz) {
    double kx, ky, kz;
    random_unit_sphere(rng, kx, ky, kz);
    double theta = (2.0 * rng_u01(rng) - 1.0) * max_angle;
    double s_th, c_th;
    sincos(theta, &s_th, &c_th);
    double k_dot_u = kx * ux + ky * uy + kz * uz;
    double cx = ky * uz - kz * uy;
    double cy = kz * ux - kx * uz;
    double cz = kx * uy - ky * ux;

    nx = ux * c_th + cx * s_th + kx * k_dot_u * (1.0 - c_th);
    ny = uy * c_th + cy * s_th + ky * k_dot_u * (1.0 - c_th);
    nz = uz * c_th + cz * s_th + kz * k_dot_u * (1.0 - c_th);

    double inv_n = rsqrt(nx * nx + ny * ny + nz * nz);
    nx *= inv_n; ny *= inv_n; nz *= inv_n;
}

struct Co2GuestParams {
    double bond;
    double q_c, eps_c, sig_c;
    double q_o, eps_o, sig_o;
    int shifted;
};

__device__ inline double pair_energy_v2(
    const Cell& cell, const Co2GuestParams& gp,
    double c1x, double c1y, double c1z, double u1x, double u1y, double u1z,
    double c2x, double c2y, double c2z, double u2x, double u2y, double u2z,
    double r_cut, double r_cut2, double dsf_alpha, double dsf_shift, double dsf_deriv,
    int& overlap_flag) {

    double dm_x, dm_y, dm_z, dm_r2;
    cell.mic_delta_super(c2x - c1x, c2y - c1y, c2z - c1z, dm_x, dm_y, dm_z, dm_r2);

    double skip_r = r_cut + 2.0 * gp.bond;
    if (dm_r2 > skip_r * skip_r) return 0.0;

    const double off_s[3] = {0.0, gp.bond, -gp.bond};
    const double q_arr[3] = {gp.q_c, gp.q_o, gp.q_o};
    const double eps_arr[3] = {gp.eps_c, gp.eps_o, gp.eps_o};
    const double sig_arr[3] = {gp.sig_c, gp.sig_o, gp.sig_o};

    double energy = 0.0;
    for (int a = 0; a < 3; ++a) {
        double ax = off_s[a] * u1x, ay = off_s[a] * u1y, az = off_s[a] * u1z;
        for (int b = 0; b < 3; ++b) {
            double bx = off_s[b] * u2x, by = off_s[b] * u2y, bz = off_s[b] * u2z;
            double dx = dm_x + bx - ax, dy = dm_y + by - ay, dz = dm_z + bz - az;
            double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= r_cut2) continue;
            if (r2 < 0.25) { overlap_flag = 1; return 0.0; }

            double r = sqrt(r2);
            double inv_r = 1.0 / r, inv_r2 = 1.0 / r2;

            double sig_ab = 0.5 * (sig_arr[a] + sig_arr[b]);
            double eps_ab = sqrt(eps_arr[a] * eps_arr[b]);
            double sr2 = (sig_ab * sig_ab) * inv_r2;
            double sr6 = sr2 * sr2 * sr2;
            double u_lj = 4.0 * eps_ab * (sr6 * sr6 - sr6);
            if (gp.shifted) {
                double sc2 = (sig_ab * sig_ab) / r_cut2;
                double sc6 = sc2 * sc2 * sc2;
                u_lj -= 4.0 * eps_ab * (sc6 * sc6 - sc6);
            }
            energy += u_lj;

            double erfc_ar = erfc(dsf_alpha * r);
            energy += KE_K * q_arr[a] * q_arr[b] *
                      (erfc_ar * inv_r - dsf_shift + dsf_deriv * (r - r_cut));
        }
    }
    return energy;
}

struct V2Proposal {
    int move_type;
    int target_idx;
    int cur_N;
    int fw_overlap;
    int skip_gg;
    double cx, cy, cz, ux, uy, uz;
    double u_fw_new, u_fw_old;
};

__global__ void gcmc_v2_kernel(
    Cell cell,
    Grid3DView grid,
    const double* __restrict__ mg_xyz, // 18 * 3 Cartesian coords of Mg in 1x1x1 unit cell
    int n_mg,
    V2GcmcConfig cfg,
    const V2BoxSpec* __restrict__ specs,
    V2BoxResult* __restrict__ results,
    unsigned long long* __restrict__ rdf_hist,
    int rdf_bins, double rdf_r_min, double rdf_r_max) {

    int box_id = blockIdx.x;
    int tid = threadIdx.x;
    V2BoxSpec spec = specs[box_id];

    // Full double precision in shared memory (512 * 7 * 8 = 28,672 B/block) -> zero FP32 rounding drift
    __shared__ double s_cx[SMEM_MAX_MOL];
    __shared__ double s_cy[SMEM_MAX_MOL];
    __shared__ double s_cz[SMEM_MAX_MOL];
    __shared__ double s_ux[SMEM_MAX_MOL];
    __shared__ double s_uy[SMEM_MAX_MOL];
    __shared__ double s_uz[SMEM_MAX_MOL];
    __shared__ double s_ufw[SMEM_MAX_MOL];

    __shared__ V2Proposal prop;
    __shared__ double warp_u[4];
    __shared__ int warp_ovl[4];
    __shared__ double s_recomp_total;

    Co2GuestParams gp{};
    if (cfg.ff_mode == FF_MERCADO2016_EPM2) {
        gp.bond = mercado2016::BOND_CO_EPM2;
        gp.q_c = mercado2016::Q_C_EPM2; gp.eps_c = mercado2016::EPS_C_EPM2; gp.sig_c = mercado2016::SIG_C_EPM2;
        gp.q_o = mercado2016::Q_O_EPM2; gp.eps_o = mercado2016::EPS_O_EPM2; gp.sig_o = mercado2016::SIG_O_EPM2;
        gp.shifted = 1;
    } else {
        gp.bond = trappe_co2::BOND_CO_A;
        gp.q_c = trappe_co2::Q_C; gp.eps_c = trappe_co2::EPS_C_K; gp.sig_c = trappe_co2::SIG_C_A;
        gp.q_o = trappe_co2::Q_O; gp.eps_o = trappe_co2::EPS_O_K; gp.sig_o = trappe_co2::SIG_O_A;
        gp.shifted = 0;
    }

    uint64_t rng = cfg.base_seed + (uint64_t)box_id * 1000003ULL + 0x9e3779b97f4a7c15ULL;
    for (int k = 0; k < 8; ++k) xorshift64star(rng);

    int cur_N = 0;
    double U_total = 0.0;
    double max_drift = 0.0;
    double delta_trans = 1.2;
    double max_angle = 0.65;

    constexpr double KB_SI = 1.380649e-23;
    double beta_f_V = (spec.fugacity_Pa * (cell.volume_super * 1.0e-30)) / (KB_SI * cfg.T_K);

    double rc = cfg.r_cut_A, rc2 = rc * rc, a_dsf = cfg.dsf_alpha;
    constexpr double INV_SQRT_PI = 0.56418958354775628695;
    double erfc_rc = erfc(a_dsf * rc), exp_rc = exp(-a_dsf * a_dsf * rc2);
    double dsf_shift = erfc_rc / rc;
    double dsf_deriv = erfc_rc / rc2 + (2.0 * a_dsf * INV_SQRT_PI) * exp_rc / rc;

    long long att_t = 0, acc_t = 0, att_r = 0, acc_r = 0, att_i = 0, acc_i = 0, att_d = 0, acc_d = 0;
    long long win_att_t = 0, win_acc_t = 0;
    long long n_samples = 0;
    double sum_N = 0.0, sum_N2 = 0.0, sum_U = 0.0, sum_UN = 0.0;
    double sum_min_mg_o = 0.0;
    long long n_mgo = 0;

    int total_steps = cfg.eq_steps + cfg.prod_steps;

    for (int step = 0; step < total_steps; ++step) {
        bool in_prod = (step >= cfg.eq_steps);

        if (tid == 0) {
            prop.cur_N = cur_N;
            prop.fw_overlap = 0;
            prop.skip_gg = 0;
            prop.u_fw_new = 0.0;
            prop.u_fw_old = 0.0;

            int mv = (int)(rng_u01(rng) * 4.0);
            if (mv > 3) mv = 3;
            prop.move_type = mv;

            if (mv == 0 || mv == 1) {
                if (cur_N == 0) {
                    prop.skip_gg = 1;
                } else {
                    int idx = (int)(rng_u01(rng) * (double)cur_N);
                    if (idx >= cur_N) idx = cur_N - 1;
                    prop.target_idx = idx;
                    prop.u_fw_old = s_ufw[idx];
                    double ox = s_cx[idx], oy = s_cy[idx], oz = s_cz[idx];
                    double oux = s_ux[idx], ouy = s_uy[idx], ouz = s_uz[idx];

                    if (mv == 0) {
                        double dx = (2.0 * rng_u01(rng) - 1.0) * delta_trans;
                        double dy = (2.0 * rng_u01(rng) - 1.0) * delta_trans;
                        double dz = (2.0 * rng_u01(rng) - 1.0) * delta_trans;
                        double nx = ox + dx, ny = oy + dy, nz = oz + dz;
                        cell.wrap_super_cart(nx, ny, nz);
                        prop.cx = nx; prop.cy = ny; prop.cz = nz;
                        prop.ux = oux; prop.uy = ouy; prop.uz = ouz;
                    } else {
                        prop.cx = ox; prop.cy = oy; prop.cz = oz;
                        rotate_rodrigues(rng, max_angle, oux, ouy, ouz, prop.ux, prop.uy, prop.uz);
                    }

                    double ulj = 0.0, uel = 0.0;
                    bool ovl = cfg.use_tricubic
                        ? eval_co2_framework_grid_tricubic(
                            grid, cell, prop.cx, prop.cy, prop.cz, prop.ux, prop.uy, prop.uz,
                            gp.bond, gp.q_c, gp.q_o, ulj, uel)
                        : eval_co2_framework_grid(
                            grid, cell, prop.cx, prop.cy, prop.cz, prop.ux, prop.uy, prop.uz,
                            gp.bond, gp.q_c, gp.q_o, ulj, uel);
                    if (ovl) { prop.fw_overlap = 1; prop.skip_gg = 1; }
                    else     { prop.u_fw_new = ulj + uel; }
                }
            } else if (mv == 2) {
                if (cur_N >= cfg.max_mol || cur_N >= SMEM_MAX_MOL) {
                    prop.skip_gg = 1;
                } else {
                    prop.target_idx = -1;
                    double sa = rng_u01(rng), sb = rng_u01(rng), sc = rng_u01(rng);
                    cell.super_frac_to_cart(sa, sb, sc, prop.cx, prop.cy, prop.cz);
                    random_unit_sphere(rng, prop.ux, prop.uy, prop.uz);

                    double ulj = 0.0, uel = 0.0;
                    bool ovl = cfg.use_tricubic
                        ? eval_co2_framework_grid_tricubic(
                            grid, cell, prop.cx, prop.cy, prop.cz, prop.ux, prop.uy, prop.uz,
                            gp.bond, gp.q_c, gp.q_o, ulj, uel)
                        : eval_co2_framework_grid(
                            grid, cell, prop.cx, prop.cy, prop.cz, prop.ux, prop.uy, prop.uz,
                            gp.bond, gp.q_c, gp.q_o, ulj, uel);
                    if (ovl) { prop.fw_overlap = 1; prop.skip_gg = 1; }
                    else     { prop.u_fw_new = ulj + uel; }
                }
            } else {
                if (cur_N == 0) {
                    prop.skip_gg = 1;
                } else {
                    int idx = (int)(rng_u01(rng) * (double)cur_N);
                    if (idx >= cur_N) idx = cur_N - 1;
                    prop.target_idx = idx;
                    prop.u_fw_old = s_ufw[idx];
                    prop.cx = s_cx[idx]; prop.cy = s_cy[idx]; prop.cz = s_cz[idx];
                    prop.ux = s_ux[idx]; prop.uy = s_uy[idx]; prop.uz = s_uz[idx];
                }
            }
        }
        __syncthreads();

        double local_du_gg = 0.0;
        int local_ovl = 0;

        if (!prop.skip_gg && prop.cur_N > 0) {
            int N = prop.cur_N, mv = prop.move_type, tgt = prop.target_idx;
            if (mv == 0 || mv == 1) {
                double ox = s_cx[tgt], oy = s_cy[tgt], oz = s_cz[tgt];
                double oux = s_ux[tgt], ouy = s_uy[tgt], ouz = s_uz[tgt];
                for (int j = tid; j < N; j += BLOCK_THREADS) {
                    if (j == tgt) continue;
                    int ov1 = 0, ov2 = 0;
                    double enew = pair_energy_v2(
                        cell, gp, prop.cx, prop.cy, prop.cz, prop.ux, prop.uy, prop.uz,
                        s_cx[j], s_cy[j], s_cz[j], s_ux[j], s_uy[j], s_uz[j],
                        rc, rc2, a_dsf, dsf_shift, dsf_deriv, ov1);
                    if (ov1) { local_ovl = 1; break; }
                    double eold = pair_energy_v2(
                        cell, gp, ox, oy, oz, oux, ouy, ouz,
                        s_cx[j], s_cy[j], s_cz[j], s_ux[j], s_uy[j], s_uz[j],
                        rc, rc2, a_dsf, dsf_shift, dsf_deriv, ov2);
                    local_du_gg += (enew - eold);
                }
            } else if (mv == 2) {
                for (int j = tid; j < N; j += BLOCK_THREADS) {
                    int ov1 = 0;
                    double enew = pair_energy_v2(
                        cell, gp, prop.cx, prop.cy, prop.cz, prop.ux, prop.uy, prop.uz,
                        s_cx[j], s_cy[j], s_cz[j], s_ux[j], s_uy[j], s_uz[j],
                        rc, rc2, a_dsf, dsf_shift, dsf_deriv, ov1);
                    if (ov1) { local_ovl = 1; break; }
                    local_du_gg += enew;
                }
            } else {
                for (int j = tid; j < N; j += BLOCK_THREADS) {
                    if (j == tgt) continue;
                    int ov2 = 0;
                    double eold = pair_energy_v2(
                        cell, gp, prop.cx, prop.cy, prop.cz, prop.ux, prop.uy, prop.uz,
                        s_cx[j], s_cy[j], s_cz[j], s_ux[j], s_uy[j], s_uz[j],
                        rc, rc2, a_dsf, dsf_shift, dsf_deriv, ov2);
                    local_du_gg -= eold;
                }
            }
        }

        int lane = tid & 31, warp_id = tid >> 5;
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_du_gg += __shfl_down_sync(0xFFFFFFFFU, local_du_gg, offset);
            local_ovl |= __shfl_down_sync(0xFFFFFFFFU, local_ovl, offset);
        }
        if (lane == 0) { warp_u[warp_id] = local_du_gg; warp_ovl[warp_id] = local_ovl; }
        __syncthreads();

        if (tid == 0) {
            double dU_gg = warp_u[0] + warp_u[1] + warp_u[2] + warp_u[3];
            int gg_ovl = warp_ovl[0] | warp_ovl[1] | warp_ovl[2] | warp_ovl[3];
            int mv = prop.move_type;

            if (mv == 0) {
                if (in_prod) ++att_t; else ++win_att_t;
                if (!prop.skip_gg && !gg_ovl) {
                    double dU = (prop.u_fw_new - prop.u_fw_old) + dU_gg;
                    double arg = -dU / cfg.T_K;
                    if ((dU <= 0.0) || (arg > -50.0 && rng_u01(rng) < exp(arg))) {
                        int tgt = prop.target_idx;
                        s_cx[tgt] = prop.cx; s_cy[tgt] = prop.cy; s_cz[tgt] = prop.cz;
                        s_ufw[tgt] = prop.u_fw_new;
                        U_total += dU;
                        if (in_prod) ++acc_t; else ++win_acc_t;
                    }
                }
            } else if (mv == 1) {
                if (in_prod) ++att_r;
                if (!prop.skip_gg && !gg_ovl) {
                    double dU = (prop.u_fw_new - prop.u_fw_old) + dU_gg;
                    double arg = -dU / cfg.T_K;
                    if ((dU <= 0.0) || (arg > -50.0 && rng_u01(rng) < exp(arg))) {
                        int tgt = prop.target_idx;
                        s_ux[tgt] = prop.ux; s_uy[tgt] = prop.uy; s_uz[tgt] = prop.uz;
                        s_ufw[tgt] = prop.u_fw_new;
                        U_total += dU;
                        if (in_prod) ++acc_r;
                    }
                }
            } else if (mv == 2) {
                if (in_prod) ++att_i;
                if (!prop.skip_gg && !gg_ovl) {
                    double dU = prop.u_fw_new + dU_gg;
                    double arg = -dU / cfg.T_K;
                    double ratio = (beta_f_V / (double)(cur_N + 1)) * exp(arg > 50.0 ? 50.0 : arg);
                    if ((arg > -50.0) && (ratio >= 1.0 || rng_u01(rng) < ratio)) {
                        s_cx[cur_N] = prop.cx; s_cy[cur_N] = prop.cy; s_cz[cur_N] = prop.cz;
                        s_ux[cur_N] = prop.ux; s_uy[cur_N] = prop.uy; s_uz[cur_N] = prop.uz;
                        s_ufw[cur_N] = prop.u_fw_new;
                        ++cur_N;
                        U_total += dU;
                        if (in_prod) ++acc_i;
                    }
                }
            } else {
                if (in_prod) ++att_d;
                if (!prop.skip_gg && cur_N > 0) {
                    double dU = (-prop.u_fw_old) + dU_gg;
                    double arg = -dU / cfg.T_K;
                    double ratio = ((double)cur_N / beta_f_V) * exp(arg > 50.0 ? 50.0 : arg);
                    if ((arg > -50.0) && (ratio >= 1.0 || rng_u01(rng) < ratio)) {
                        int tgt = prop.target_idx, last = cur_N - 1;
                        if (tgt != last) {
                            s_cx[tgt] = s_cx[last]; s_cy[tgt] = s_cy[last]; s_cz[tgt] = s_cz[last];
                            s_ux[tgt] = s_ux[last]; s_uy[tgt] = s_uy[last]; s_uz[tgt] = s_uz[last];
                            s_ufw[tgt] = s_ufw[last];
                        }
                        --cur_N;
                        U_total += dU;
                        if (in_prod) ++acc_d;
                    }
                }
            }

            if (!in_prod && ((step + 1) % 2000 == 0) && win_att_t > 20) {
                double rate = (double)win_acc_t / (double)win_att_t;
                if (rate < 0.30) delta_trans *= 0.90;
                else if (rate > 0.50) delta_trans *= 1.10;
                if (delta_trans < 0.15) delta_trans = 0.15;
                if (delta_trans > 4.0) delta_trans = 4.0;
                win_att_t = 0; win_acc_t = 0;
            }
            prop.cur_N = cur_N;
        }
        __syncthreads();

        // Exact periodic full-box energy recomputation (zero cumulative drift)
        if (((step + 1) % cfg.recompute_stride) == 0) {
            double local_u = 0.0;
            int N = prop.cur_N;
            for (int i = tid; i < N; i += BLOCK_THREADS) {
                double ulj_chk = 0.0, uel_chk = 0.0;
                bool ov_chk = cfg.use_tricubic
                    ? eval_co2_framework_grid_tricubic(
                        grid, cell, s_cx[i], s_cy[i], s_cz[i], s_ux[i], s_uy[i], s_uz[i],
                        gp.bond, gp.q_c, gp.q_o, ulj_chk, uel_chk)
                    : eval_co2_framework_grid(
                        grid, cell, s_cx[i], s_cy[i], s_cz[i], s_ux[i], s_uy[i], s_uz[i],
                        gp.bond, gp.q_c, gp.q_o, ulj_chk, uel_chk);
                double ufw_fresh = ov_chk ? s_ufw[i] : (ulj_chk + uel_chk);
                s_ufw[i] = ufw_fresh;
                local_u += ufw_fresh;
                for (int j = i + 1; j < N; ++j) {
                    int ov = 0;
                    local_u += pair_energy_v2(
                        cell, gp, s_cx[i], s_cy[i], s_cz[i], s_ux[i], s_uy[i], s_uz[i],
                        s_cx[j], s_cy[j], s_cz[j], s_ux[j], s_uy[j], s_uz[j],
                        rc, rc2, a_dsf, dsf_shift, dsf_deriv, ov);
                }
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                local_u += __shfl_down_sync(0xFFFFFFFFU, local_u, offset);
            }
            if (lane == 0) warp_u[warp_id] = local_u;
            __syncthreads();
            if (tid == 0) {
                s_recomp_total = warp_u[0] + warp_u[1] + warp_u[2] + warp_u[3];
                double drift = fabs(U_total - s_recomp_total);
                if (drift > max_drift) max_drift = drift;
                U_total = s_recomp_total;
            }
            __syncthreads();
        }

        if (tid == 0 && in_prod && ((step - cfg.eq_steps) % cfg.sample_stride == 0)) {
            ++n_samples;
            double dN = (double)cur_N;
            sum_N += dN; sum_N2 += dN * dN;
            sum_U += U_total; sum_UN += U_total * dN;

            // Measure Mg - O_CO2 coordination distance distribution at 0.15 bar
            if (spec.record_rdf && cur_N > 0 && (((step - cfg.eq_steps) / cfg.sample_stride) % 5 == 0)) {
                double inv_dr = (double)rdf_bins / (rdf_r_max - rdf_r_min);
                for (int m = 0; m < cur_N; ++m) {
                    for (int side = -1; side <= 1; side += 2) {
                        double ox = s_cx[m] + (double)side * gp.bond * s_ux[m];
                        double oy = s_cy[m] + (double)side * gp.bond * s_uy[m];
                        double oz = s_cz[m] + (double)side * gp.bond * s_uz[m];
                        // Fold into 1x1x1 unit cell to compare against 18 unit-cell Mg atoms
                        double sa, sb, sc;
                        cell.cart_to_unit_frac(ox, oy, oz, sa, sb, sc);
                        sa -= floor(sa); sb -= floor(sb); sc -= floor(sc);
                        double uox, uoy, uoz;
                        cell.unit_frac_to_cart(sa, sb, sc, uox, uoy, uoz);

                        double min_r2 = 1.0e30;
                        for (int k = 0; k < n_mg; ++k) {
                            double dsa, dsb, dsc;
                            double mx = mg_xyz[3 * k + 0], my = mg_xyz[3 * k + 1], mz = mg_xyz[3 * k + 2];
                            cell.cart_to_unit_frac(uox - mx, uoy - my, uoz - mz, dsa, dsb, dsc);
                            dsa -= round(dsa); dsb -= round(dsb); dsc -= round(dsc);
                            for (int na = -1; na <= 1; ++na) {
                                for (int nb = -1; nb <= 1; ++nb) {
                                    for (int nc = -1; nc <= 1; ++nc) {
                                        double dx, dy, dz;
                                        cell.unit_frac_to_cart(dsa + na, dsb + nb, dsc + nc, dx, dy, dz);
                                        double r2 = dx * dx + dy * dy + dz * dz;
                                        if (r2 < min_r2) min_r2 = r2;
                                    }
                                }
                            }
                        }
                        double r_mg_o = sqrt(min_r2);
                        if (r_mg_o < rdf_r_max) {
                            sum_min_mg_o += r_mg_o;
                            ++n_mgo;
                            int bin = (int)((r_mg_o - rdf_r_min) * inv_dr);
                            if (bin >= 0 && bin < rdf_bins) {
                                atomicAdd(&rdf_hist[bin], 1ULL);
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        double ns = (n_samples > 0) ? (double)n_samples : 1.0;
        double mean_N = sum_N / ns, mean_N2 = sum_N2 / ns;
        double mean_U = sum_U / ns, mean_UN = sum_UN / ns;
        double var_N = mean_N2 - mean_N * mean_N;
        if (var_N < 0.0) var_N = 0.0;

        double qst = 0.0;
        if (var_N > 1.0e-9) {
            qst = (cfg.T_K - (mean_UN - mean_U * mean_N) / var_N) * K_TO_KJMOL;
        }

        V2BoxResult r{};
        r.pressure_bar = spec.pressure_bar;
        r.fugacity_Pa = spec.fugacity_Pa;
        r.replica_id = spec.replica_id;
        r.avg_N = mean_N;
        r.std_N = sqrt(var_N);
        r.uptake_mmol_g = mean_N * 1000.0 / cfg.framework_mass_super_gmol;
        r.avg_U_K = mean_U;
        r.Qst_kJ_mol = qst;
        r.max_energy_drift_K = max_drift;
        r.mean_min_mg_o_A = (n_mgo > 0) ? (sum_min_mg_o / (double)n_mgo) : 0.0;
        r.acc_trans = (att_t > 0) ? (double)acc_t / (double)att_t : 0.0;
        r.acc_rot   = (att_r > 0) ? (double)acc_r / (double)att_r : 0.0;
        r.acc_ins   = (att_i > 0) ? (double)acc_i / (double)att_i : 0.0;
        r.acc_del   = (att_d > 0) ? (double)acc_d / (double)att_d : 0.0;
        results[box_id] = r;
    }
}

} // namespace

double run_v2_gcmc_gpu(
    const Cell& cell,
    const Grid3DView& d_grid,
    const std::vector<FrameworkAtom>& mg_unit_atoms,
    const V2GcmcConfig& cfg,
    const std::vector<V2BoxSpec>& boxes,
    std::vector<V2BoxResult>& results_out,
    std::vector<double>& rdf_hist_out,
    int rdf_bins, double rdf_r_min, double rdf_r_max) {

    int num_boxes = (int)boxes.size();
    results_out.resize(num_boxes);

    std::vector<double> h_mg_xyz;
    for (const auto& at : mg_unit_atoms) {
        if (at.elem == ELEM_MG) {
            h_mg_xyz.push_back((double)at.x);
            h_mg_xyz.push_back((double)at.y);
            h_mg_xyz.push_back((double)at.z);
        }
    }
    int n_mg = (int)h_mg_xyz.size() / 3;

    double* d_mg_xyz = nullptr;
    V2BoxSpec* d_specs = nullptr;
    V2BoxResult* d_results = nullptr;
    unsigned long long* d_rdf = nullptr;

    CUDA_CHECK(cudaMalloc(&d_mg_xyz, h_mg_xyz.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_specs, num_boxes * sizeof(V2BoxSpec)));
    CUDA_CHECK(cudaMalloc(&d_results, num_boxes * sizeof(V2BoxResult)));
    CUDA_CHECK(cudaMalloc(&d_rdf, rdf_bins * sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemcpy(d_mg_xyz, h_mg_xyz.data(), h_mg_xyz.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_specs, boxes.data(), num_boxes * sizeof(V2BoxSpec), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_rdf, 0, rdf_bins * sizeof(unsigned long long)));

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));

    gcmc_v2_kernel<<<num_boxes, BLOCK_THREADS>>>(
        cell, d_grid, d_mg_xyz, n_mg, cfg, d_specs, d_results, d_rdf,
        rdf_bins, rdf_r_min, rdf_r_max);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));

    CUDA_CHECK(cudaMemcpy(results_out.data(), d_results, num_boxes * sizeof(V2BoxResult), cudaMemcpyDeviceToHost));
    std::vector<unsigned long long> h_rdf(rdf_bins);
    CUDA_CHECK(cudaMemcpy(h_rdf.data(), d_rdf, rdf_bins * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    rdf_hist_out.resize(rdf_bins);
    for (int i = 0; i < rdf_bins; ++i) rdf_hist_out[i] = (double)h_rdf[i];

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaFree(d_mg_xyz));
    CUDA_CHECK(cudaFree(d_specs));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_rdf));

    return (double)ms;
}

} // namespace mofgcmc
