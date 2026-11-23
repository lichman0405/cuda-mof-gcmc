#include "unified_gcmc.hpp"

#include <cmath>
#include <cstdint>
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

__device__ inline double pair_energy_unified(
    const Cell& cell,
    const SpeciesDef& sp1, const SpeciesDef& sp2,
    double c1x, double c1y, double c1z, double u1x, double u1y, double u1z,
    double mu1x, double mu1y, double mu1z,
    double c2x, double c2y, double c2z, double u2x, double u2y, double u2z,
    double mu2x, double mu2y, double mu2z,
    int shifted_lj, int pol_mode, double r_cut, double r_cut2,
    double dsf_alpha, double dsf_shift, double dsf_deriv,
    int& overlap_flag) {

    double dm_x, dm_y, dm_z, dm_r2;
    cell.mic_delta_super(c2x - c1x, c2y - c1y, c2z - c1z, dm_x, dm_y, dm_z, dm_r2);

    double skip_r = r_cut + sp1.bond_A + sp2.bond_A;
    if (dm_r2 > skip_r * skip_r) return 0.0;

    const double off1_s[3] = {0.0, sp1.bond_A, -sp1.bond_A};
    const double off2_s[3] = {0.0, sp2.bond_A, -sp2.bond_A};

    double energy = 0.0;
    for (int a = 0; a < sp1.n_sites; ++a) {
        double ax = off1_s[a] * u1x, ay = off1_s[a] * u1y, az = off1_s[a] * u1z;
        bool lj1 = (sp1.lj_grid_idx[a] >= 0);
        double q1 = sp1.q[a];

        for (int b = 0; b < sp2.n_sites; ++b) {
            bool lj2 = (sp2.lj_grid_idx[b] >= 0);
            double q2 = sp2.q[b];
            bool do_lj = lj1 && lj2;
            bool do_q  = (q1 != 0.0 && q2 != 0.0);
            if (!do_lj && !do_q) continue;

            double bx = off2_s[b] * u2x, by = off2_s[b] * u2y, bz = off2_s[b] * u2z;
            double dx = dm_x + bx - ax, dy = dm_y + by - ay, dz = dm_z + bz - az;
            double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= r_cut2) continue;
            if (r2 < 0.25) { overlap_flag = 1; return 0.0; }

            double r = sqrt(r2);
            double inv_r = 1.0 / r, inv_r2 = 1.0 / r2;

            if (do_lj) {
                double sig_ab = 0.5 * (sp1.sigma_A[a] + sp2.sigma_A[b]);
                double eps_ab = sqrt(sp1.eps_K[a] * sp2.eps_K[b]);
                double sr2 = (sig_ab * sig_ab) * inv_r2;
                double sr6 = sr2 * sr2 * sr2;
                double ulj = 4.0 * eps_ab * (sr6 * sr6 - sr6);
                if (shifted_lj) {
                    double sc2 = (sig_ab * sig_ab) / r_cut2;
                    double sc6 = sc2 * sc2 * sc2;
                    ulj -= 4.0 * eps_ab * (sc6 * sc6 - sc6);
                }
                energy += ulj;
            }
            if (do_q) {
                double erfc_ar = erfc(dsf_alpha * r);
                energy += KE_K * q1 * q2 *
                          (erfc_ar * inv_r - dsf_shift + dsf_deriv * (r - r_cut));
            }
        }
    }

    if (pol_mode != 0 && dm_r2 < r_cut2 && dm_r2 >= 0.25) {
        double r_m = sqrt(dm_r2);
        double inv_r = 1.0 / r_m;
        double inv_r3 = inv_r / dm_r2;
        double inv_r5 = inv_r3 / dm_r2;
        double inv_s = 1.0 / cbrt(sqrt(sp1.alpha_iso * sp2.alpha_iso));
        double u_thole = r_m * inv_s;
        double au3 = trappe_co2::THOLE_A * u_thole * u_thole * u_thole;
        double exp_t = (au3 < 50.0) ? exp(-au3) : 0.0;
        double fe = 1.0 - exp_t;
        double ft = 1.0 - (1.0 + au3) * exp_t;
        double mu1_dot_mu2 = mu1x * mu2x + mu1y * mu2y + mu1z * mu2z;
        double mu1_dot_r = mu1x * dm_x + mu1y * dm_y + mu1z * dm_z;
        double mu2_dot_r = mu2x * dm_x + mu2y * dm_y + mu2z * dm_z;
        energy += KE_K * (fe * mu1_dot_mu2 * inv_r3 - 3.0 * ft * mu1_dot_r * mu2_dot_r * inv_r5);
    }

    return energy;
}

struct UnifiedProposal {
    int move_type; // 0=trans, 1=rot, 2=ins, 3=del, 4=swap
    int target_idx;
    int sp_new;
    int sp_old;
    int cur_N;
    int fw_overlap;
    int skip_gg;
    double cx, cy, cz, ux, uy, uz;
    double mux, muy, muz;
    double u_fw_new, u_fw_old;
    double u_pol_new, u_pol_old;
};

__global__ void gcmc_unified_kernel(
    Cell cell,
    MixGrid3DView grid,
    SpeciesDef sp0,
    SpeciesDef sp1,
    UnifiedGcmcConfig cfg,
    const UnifiedBoxSpec* __restrict__ specs,
    UnifiedBoxResult* __restrict__ results) {

    int box_id = blockIdx.x;
    int tid = threadIdx.x;
    UnifiedBoxSpec spec = specs[box_id];

    // Full double-precision coords, framework energies & dipoles (34,816 B) + upol & species (~2,560 B) -> < 48 KB
    __shared__ double s_cx[SMEM_MAX_MOL];
    __shared__ double s_cy[SMEM_MAX_MOL];
    __shared__ double s_cz[SMEM_MAX_MOL];
    __shared__ double s_ux[SMEM_MAX_MOL];
    __shared__ double s_uy[SMEM_MAX_MOL];
    __shared__ double s_uz[SMEM_MAX_MOL];
    __shared__ double s_ufw[SMEM_MAX_MOL];
    __shared__ double s_mux[SMEM_MAX_MOL];
    __shared__ double s_muy[SMEM_MAX_MOL];
    __shared__ double s_muz[SMEM_MAX_MOL];
    __shared__ float s_upol[SMEM_MAX_MOL];
    __shared__ uint8_t s_sp[SMEM_MAX_MOL];

    __shared__ UnifiedProposal prop;
    __shared__ double warp_u[4];
    __shared__ int warp_ovl[4];
    __shared__ double s_recomp_total;

    uint64_t rng = cfg.base_seed + (uint64_t)box_id * 1000003ULL + 0x9e3779b97f4a7c15ULL;
    for (int k = 0; k < 8; ++k) xorshift64star(rng);

    int cur_N = 0, cur_N0 = 0, cur_N1 = 0;
    double U_total = 0.0;
    double cur_Upol0 = 0.0, cur_Upol1 = 0.0;
    double max_drift = 0.0;
    double delta_trans = 1.2;
    double max_angle = 0.65;

    constexpr double KB_SI = 1.380649e-23;
    double vol_m3 = cell.volume_super * 1.0e-30;
    double beta_f_V[2] = {
        (spec.f0_Pa * vol_m3) / (KB_SI * cfg.T_K),
        (spec.f1_Pa * vol_m3) / (KB_SI * cfg.T_K)
    };

    double rc = cfg.r_cut_A, rc2 = rc * rc, a_dsf = cfg.dsf_alpha;
    constexpr double INV_SQRT_PI = 0.56418958354775628695;
    double erfc_rc = erfc(a_dsf * rc), exp_rc = exp(-a_dsf * a_dsf * rc2);
    double dsf_shift = erfc_rc / rc;
    double dsf_deriv = erfc_rc / rc2 + (2.0 * a_dsf * INV_SQRT_PI) * exp_rc / rc;
    int shifted_lj = (cfg.ff_mode == 1) ? 1 : 0;

    long long att_t = 0, acc_t = 0, att_r = 0, acc_r = 0;
    long long att_i[2] = {0, 0}, acc_i[2] = {0, 0}, att_d[2] = {0, 0}, acc_d[2] = {0, 0};
    long long att_sw01 = 0, acc_sw01 = 0, att_sw10 = 0, acc_sw10 = 0;
    long long win_att_t = 0, win_acc_t = 0;

    long long n_samples = 0;
    double sum_N = 0.0, sum_N2 = 0.0, sum_U = 0.0, sum_UN = 0.0;
    double sum_N0 = 0.0, sum_N0_2 = 0.0, sum_N1 = 0.0, sum_N1_2 = 0.0;
    double sum_Upol0 = 0.0, sum_Upol1 = 0.0;

    int total_steps = cfg.eq_steps + cfg.prod_steps;

    for (int step = 0; step < total_steps; ++step) {
        bool in_prod = (step >= cfg.eq_steps);

        if (tid == 0) {
            prop.cur_N = cur_N;
            prop.fw_overlap = 0;
            prop.skip_gg = 0;
            prop.u_fw_new = prop.u_fw_old = 0.0;
            prop.u_pol_new = prop.u_pol_old = 0.0;
            prop.mux = prop.muy = prop.muz = 0.0;

            int n_moves = (cfg.n_comp == 2 && cfg.enable_swap) ? 5 : 4;
            int mv = (int)(rng_u01(rng) * (double)n_moves);
            if (mv >= n_moves) mv = n_moves - 1;
            prop.move_type = mv;

            auto eval_fw = [&](int sp_id, double cx, double cy, double cz,
                               double ux, double uy, double uz,
                               double& ufw, double& upol,
                               double& mx, double& my, double& mz) -> bool {
                const SpeciesDef& sp = (sp_id == 0) ? sp0 : sp1;
                double ulj = 0.0, uel = 0.0;
                bool ovl = eval_species_framework_grid_unified(
                    grid, cell, sp, cfg.use_tricubic, cfg.pol_mode, KE_K,
                    cx, cy, cz, ux, uy, uz,
                    ulj, uel, upol, mx, my, mz);
                ufw = ulj + uel + upol;
                return ovl;
            };

            if (mv == 0 || mv == 1) {
                if (cur_N == 0) {
                    prop.skip_gg = 1;
                } else {
                    int idx = (int)(rng_u01(rng) * (double)cur_N);
                    if (idx >= cur_N) idx = cur_N - 1;
                    int sp_id = (int)s_sp[idx];
                    prop.target_idx = idx;
                    prop.sp_old = sp_id;
                    prop.sp_new = sp_id;
                    prop.u_fw_old = s_ufw[idx];
                    prop.u_pol_old = (double)s_upol[idx];

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

                    bool ovl = eval_fw(sp_id, prop.cx, prop.cy, prop.cz,
                                       prop.ux, prop.uy, prop.uz,
                                       prop.u_fw_new, prop.u_pol_new,
                                       prop.mux, prop.muy, prop.muz);
                    if (ovl) { prop.fw_overlap = 1; prop.skip_gg = 1; }
                }
            } else if (mv == 2) {
                int sp_id = (cfg.n_comp == 2) ? ((rng_u01(rng) < 0.5) ? 0 : 1) : 0;
                prop.sp_new = sp_id;
                prop.sp_old = sp_id;
                if (cur_N >= cfg.max_mol || cur_N >= SMEM_MAX_MOL) {
                    prop.skip_gg = 1;
                } else {
                    prop.target_idx = -1;
                    double sa = rng_u01(rng), sb = rng_u01(rng), sc = rng_u01(rng);
                    cell.super_frac_to_cart(sa, sb, sc, prop.cx, prop.cy, prop.cz);
                    random_unit_sphere(rng, prop.ux, prop.uy, prop.uz);

                    bool ovl = eval_fw(sp_id, prop.cx, prop.cy, prop.cz,
                                       prop.ux, prop.uy, prop.uz,
                                       prop.u_fw_new, prop.u_pol_new,
                                       prop.mux, prop.muy, prop.muz);
                    if (ovl) { prop.fw_overlap = 1; prop.skip_gg = 1; }
                }
            } else if (mv == 3) {
                int sp_id = (cfg.n_comp == 2) ? ((rng_u01(rng) < 0.5) ? 0 : 1) : 0;
                prop.sp_old = sp_id;
                prop.sp_new = sp_id;
                int count_sp = (sp_id == 0) ? cur_N0 : cur_N1;
                if (count_sp == 0) {
                    prop.skip_gg = 1;
                } else {
                    int idx = -1;
                    if (cfg.n_comp == 1) {
                        idx = (int)(rng_u01(rng) * (double)cur_N);
                        if (idx >= cur_N) idx = cur_N - 1;
                    } else {
                        int nth = (int)(rng_u01(rng) * (double)count_sp);
                        if (nth >= count_sp) nth = count_sp - 1;
                        int seen = 0;
                        for (int k = 0; k < cur_N; ++k) {
                            if ((int)s_sp[k] == sp_id) {
                                if (seen == nth) { idx = k; break; }
                                ++seen;
                            }
                        }
                    }
                    prop.target_idx = idx;
                    prop.u_fw_old = s_ufw[idx];
                    prop.u_pol_old = (double)s_upol[idx];
                    prop.cx = s_cx[idx]; prop.cy = s_cy[idx]; prop.cz = s_cz[idx];
                    prop.ux = s_ux[idx]; prop.uy = s_uy[idx]; prop.uz = s_uz[idx];
                    prop.mux = (double)s_mux[idx]; prop.muy = (double)s_muy[idx]; prop.muz = (double)s_muz[idx];
                }
            } else { // Identity Swap (0 <-> 1)
                int src_sp = (rng_u01(rng) < 0.5) ? 0 : 1;
                int dst_sp = 1 - src_sp;
                prop.sp_old = src_sp;
                prop.sp_new = dst_sp;
                int count_src = (src_sp == 0) ? cur_N0 : cur_N1;
                if (count_src == 0) {
                    prop.skip_gg = 1;
                } else {
                    int nth = (int)(rng_u01(rng) * (double)count_src);
                    if (nth >= count_src) nth = count_src - 1;
                    int idx = -1, seen = 0;
                    for (int k = 0; k < cur_N; ++k) {
                        if ((int)s_sp[k] == src_sp) {
                            if (seen == nth) { idx = k; break; }
                            ++seen;
                        }
                    }
                    prop.target_idx = idx;
                    prop.u_fw_old = s_ufw[idx];
                    prop.u_pol_old = (double)s_upol[idx];
                    prop.cx = s_cx[idx]; prop.cy = s_cy[idx]; prop.cz = s_cz[idx];
                    prop.ux = s_ux[idx]; prop.uy = s_uy[idx]; prop.uz = s_uz[idx];

                    bool ovl = eval_fw(dst_sp, prop.cx, prop.cy, prop.cz,
                                       prop.ux, prop.uy, prop.uz,
                                       prop.u_fw_new, prop.u_pol_new,
                                       prop.mux, prop.muy, prop.muz);
                    if (ovl) { prop.fw_overlap = 1; prop.skip_gg = 1; }
                }
            }
        }
        __syncthreads();

        double local_du_gg = 0.0;
        int local_ovl = 0;

        if (!prop.skip_gg && prop.cur_N > 0) {
            int N = prop.cur_N, mv = prop.move_type, tgt = prop.target_idx;
            const SpeciesDef& sp_new_def = (prop.sp_new == 0) ? sp0 : sp1;
            const SpeciesDef& sp_old_def = (prop.sp_old == 0) ? sp0 : sp1;

            if (mv == 0 || mv == 1 || mv == 4) {
                double ox = s_cx[tgt], oy = s_cy[tgt], oz = s_cz[tgt];
                double oux = s_ux[tgt], ouy = s_uy[tgt], ouz = s_uz[tgt];
                double omx = (double)s_mux[tgt], omy = (double)s_muy[tgt], omz = (double)s_muz[tgt];
                for (int j = tid; j < N; j += BLOCK_THREADS) {
                    if (j == tgt) continue;
                    const SpeciesDef& sp_j = ((int)s_sp[j] == 0) ? sp0 : sp1;
                    int ov1 = 0, ov2 = 0;
                    double enew = pair_energy_unified(
                        cell, sp_new_def, sp_j,
                        prop.cx, prop.cy, prop.cz, prop.ux, prop.uy, prop.uz, prop.mux, prop.muy, prop.muz,
                        s_cx[j], s_cy[j], s_cz[j], s_ux[j], s_uy[j], s_uz[j],
                        (double)s_mux[j], (double)s_muy[j], (double)s_muz[j],
                        shifted_lj, cfg.pol_mode, rc, rc2, a_dsf, dsf_shift, dsf_deriv, ov1);
                    if (ov1) { local_ovl = 1; break; }
                    double eold = pair_energy_unified(
                        cell, sp_old_def, sp_j,
                        ox, oy, oz, oux, ouy, ouz, omx, omy, omz,
                        s_cx[j], s_cy[j], s_cz[j], s_ux[j], s_uy[j], s_uz[j],
                        (double)s_mux[j], (double)s_muy[j], (double)s_muz[j],
                        shifted_lj, cfg.pol_mode, rc, rc2, a_dsf, dsf_shift, dsf_deriv, ov2);
                    local_du_gg += (enew - eold);
                }
            } else if (mv == 2) {
                for (int j = tid; j < N; j += BLOCK_THREADS) {
                    const SpeciesDef& sp_j = ((int)s_sp[j] == 0) ? sp0 : sp1;
                    int ov1 = 0;
                    double enew = pair_energy_unified(
                        cell, sp_new_def, sp_j,
                        prop.cx, prop.cy, prop.cz, prop.ux, prop.uy, prop.uz, prop.mux, prop.muy, prop.muz,
                        s_cx[j], s_cy[j], s_cz[j], s_ux[j], s_uy[j], s_uz[j],
                        (double)s_mux[j], (double)s_muy[j], (double)s_muz[j],
                        shifted_lj, cfg.pol_mode, rc, rc2, a_dsf, dsf_shift, dsf_deriv, ov1);
                    if (ov1) { local_ovl = 1; break; }
                    local_du_gg += enew;
                }
            } else {
                for (int j = tid; j < N; j += BLOCK_THREADS) {
                    if (j == tgt) continue;
                    const SpeciesDef& sp_j = ((int)s_sp[j] == 0) ? sp0 : sp1;
                    int ov2 = 0;
                    double eold = pair_energy_unified(
                        cell, sp_old_def, sp_j,
                        prop.cx, prop.cy, prop.cz, prop.ux, prop.uy, prop.uz, prop.mux, prop.muy, prop.muz,
                        s_cx[j], s_cy[j], s_cz[j], s_ux[j], s_uy[j], s_uz[j],
                        (double)s_mux[j], (double)s_muy[j], (double)s_muz[j],
                        shifted_lj, cfg.pol_mode, rc, rc2, a_dsf, dsf_shift, dsf_deriv, ov2);
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
                if (cur_N > 0 && !prop.fw_overlap && !gg_ovl) {
                    double dU = (prop.u_fw_new - prop.u_fw_old) + dU_gg;
                    double arg = -dU / cfg.T_K;
                    if ((dU <= 0.0) || (arg > -50.0 && rng_u01(rng) < exp(arg))) {
                        int tgt = prop.target_idx;
                        s_cx[tgt] = prop.cx; s_cy[tgt] = prop.cy; s_cz[tgt] = prop.cz;
                        s_mux[tgt] = prop.mux; s_muy[tgt] = prop.muy; s_muz[tgt] = prop.muz;
                        s_ufw[tgt] = prop.u_fw_new; s_upol[tgt] = (float)prop.u_pol_new;
                        U_total += dU;
                        if (prop.sp_new == 0) cur_Upol0 += (prop.u_pol_new - prop.u_pol_old);
                        else                  cur_Upol1 += (prop.u_pol_new - prop.u_pol_old);
                        if (in_prod) ++acc_t; else ++win_acc_t;
                    }
                }
            } else if (mv == 1) {
                if (in_prod) ++att_r;
                if (cur_N > 0 && !prop.fw_overlap && !gg_ovl) {
                    double dU = (prop.u_fw_new - prop.u_fw_old) + dU_gg;
                    double arg = -dU / cfg.T_K;
                    if ((dU <= 0.0) || (arg > -50.0 && rng_u01(rng) < exp(arg))) {
                        int tgt = prop.target_idx;
                        s_ux[tgt] = prop.ux; s_uy[tgt] = prop.uy; s_uz[tgt] = prop.uz;
                        s_mux[tgt] = prop.mux; s_muy[tgt] = prop.muy; s_muz[tgt] = prop.muz;
                        s_ufw[tgt] = prop.u_fw_new; s_upol[tgt] = (float)prop.u_pol_new;
                        U_total += dU;
                        if (prop.sp_new == 0) cur_Upol0 += (prop.u_pol_new - prop.u_pol_old);
                        else                  cur_Upol1 += (prop.u_pol_new - prop.u_pol_old);
                        if (in_prod) ++acc_r;
                    }
                }
            } else if (mv == 2) {
                int s = prop.sp_new;
                if (in_prod) ++att_i[s];
                if (cur_N < cfg.max_mol && cur_N < SMEM_MAX_MOL && !prop.fw_overlap && !gg_ovl) {
                    int n_s = (s == 0) ? cur_N0 : cur_N1;
                    double dU = prop.u_fw_new + dU_gg;
                    double arg = -dU / cfg.T_K;
                    double ratio = (beta_f_V[s] / (double)(n_s + 1)) * exp(arg > 50.0 ? 50.0 : arg);
                    if ((arg > -50.0) && (ratio >= 1.0 || rng_u01(rng) < ratio)) {
                        s_cx[cur_N] = prop.cx; s_cy[cur_N] = prop.cy; s_cz[cur_N] = prop.cz;
                        s_ux[cur_N] = prop.ux; s_uy[cur_N] = prop.uy; s_uz[cur_N] = prop.uz;
                        s_mux[cur_N] = prop.mux; s_muy[cur_N] = prop.muy; s_muz[cur_N] = prop.muz;
                        s_ufw[cur_N] = prop.u_fw_new; s_upol[cur_N] = (float)prop.u_pol_new;
                        s_sp[cur_N] = (uint8_t)s;
                        ++cur_N;
                        U_total += dU;
                        if (s == 0) { ++cur_N0; cur_Upol0 += prop.u_pol_new; }
                        else        { ++cur_N1; cur_Upol1 += prop.u_pol_new; }
                        if (in_prod) ++acc_i[s];
                    }
                }
            } else if (mv == 3) {
                int s = prop.sp_old;
                if (in_prod) ++att_d[s];
                int n_s = (s == 0) ? cur_N0 : cur_N1;
                if (n_s > 0 && !gg_ovl) {
                    double dU = (-prop.u_fw_old) + dU_gg;
                    double arg = -dU / cfg.T_K;
                    double ratio = ((double)n_s / beta_f_V[s]) * exp(arg > 50.0 ? 50.0 : arg);
                    if ((arg > -50.0) && (ratio >= 1.0 || rng_u01(rng) < ratio)) {
                        int tgt = prop.target_idx, last = cur_N - 1;
                        if (tgt != last) {
                            s_cx[tgt] = s_cx[last]; s_cy[tgt] = s_cy[last]; s_cz[tgt] = s_cz[last];
                            s_ux[tgt] = s_ux[last]; s_uy[tgt] = s_uy[last]; s_uz[tgt] = s_uz[last];
                            s_mux[tgt] = s_mux[last]; s_muy[tgt] = s_muy[last]; s_muz[tgt] = s_muz[last];
                            s_ufw[tgt] = s_ufw[last]; s_upol[tgt] = s_upol[last]; s_sp[tgt] = s_sp[last];
                        }
                        --cur_N;
                        U_total += dU;
                        if (s == 0) { --cur_N0; cur_Upol0 -= prop.u_pol_old; }
                        else        { --cur_N1; cur_Upol1 -= prop.u_pol_old; }
                        if (in_prod) ++acc_d[s];
                    }
                }
            } else {
                int a = prop.sp_old, b = prop.sp_new;
                if (in_prod) { if (a == 0) ++att_sw01; else ++att_sw10; }
                int n_a = (a == 0) ? cur_N0 : cur_N1;
                int n_b = (b == 0) ? cur_N0 : cur_N1;
                if (n_a > 0 && !prop.fw_overlap && !gg_ovl) {
                    double dU = (prop.u_fw_new - prop.u_fw_old) + dU_gg;
                    double arg = -dU / cfg.T_K;
                    double ratio = (beta_f_V[b] / beta_f_V[a]) * ((double)n_a / (double)(n_b + 1))
                                 * exp(arg > 50.0 ? 50.0 : arg);
                    if ((arg > -50.0) && (ratio >= 1.0 || rng_u01(rng) < ratio)) {
                        int tgt = prop.target_idx;
                        s_mux[tgt] = prop.mux; s_muy[tgt] = prop.muy; s_muz[tgt] = prop.muz;
                        s_ufw[tgt] = prop.u_fw_new; s_upol[tgt] = (float)prop.u_pol_new;
                        s_sp[tgt] = (uint8_t)b;
                        U_total += dU;
                        if (a == 0) {
                            --cur_N0; ++cur_N1;
                            cur_Upol0 -= prop.u_pol_old; cur_Upol1 += prop.u_pol_new;
                            if (in_prod) ++acc_sw01;
                        } else {
                            --cur_N1; ++cur_N0;
                            cur_Upol1 -= prop.u_pol_old; cur_Upol0 += prop.u_pol_new;
                            if (in_prod) ++acc_sw10;
                        }
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

        // Exact periodic full-box recomputation of BOTH u_fw (including dipoles) and pair_energy
        if (((step + 1) % cfg.recompute_stride) == 0) {
            int N = prop.cur_N;
            double local_u = 0.0;
            for (int i = tid; i < N; i += BLOCK_THREADS) {
                const SpeciesDef& sp_i = ((int)s_sp[i] == 0) ? sp0 : sp1;
                double ulj_c = 0.0, uel_c = 0.0, upol_c = 0.0, mx_c = 0.0, my_c = 0.0, mz_c = 0.0;
                bool ov_c = eval_species_framework_grid_unified(
                    grid, cell, sp_i, cfg.use_tricubic, cfg.pol_mode, KE_K,
                    s_cx[i], s_cy[i], s_cz[i], s_ux[i], s_uy[i], s_uz[i],
                    ulj_c, uel_c, upol_c, mx_c, my_c, mz_c);
                double ufw_c = ov_c ? s_ufw[i] : (ulj_c + uel_c + upol_c);
                s_ufw[i] = ufw_c;
                if (!ov_c) {
                    s_upol[i] = (float)upol_c;
                    s_mux[i] = mx_c; s_muy[i] = my_c; s_muz[i] = mz_c;
                }
                local_u += ufw_c;
            }
            __syncthreads();

            for (int i = tid; i < N; i += BLOCK_THREADS) {
                const SpeciesDef& sp_i = ((int)s_sp[i] == 0) ? sp0 : sp1;
                for (int j = i + 1; j < N; ++j) {
                    const SpeciesDef& sp_j = ((int)s_sp[j] == 0) ? sp0 : sp1;
                    int ov = 0;
                    local_u += pair_energy_unified(
                        cell, sp_i, sp_j,
                        s_cx[i], s_cy[i], s_cz[i], s_ux[i], s_uy[i], s_uz[i],
                        (double)s_mux[i], (double)s_muy[i], (double)s_muz[i],
                        s_cx[j], s_cy[j], s_cz[j], s_ux[j], s_uy[j], s_uz[j],
                        (double)s_mux[j], (double)s_muy[j], (double)s_muz[j],
                        shifted_lj, cfg.pol_mode, rc, rc2, a_dsf, dsf_shift, dsf_deriv, ov);
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
            double dN = (double)cur_N, dN0 = (double)cur_N0, dN1 = (double)cur_N1;
            sum_N += dN; sum_N2 += dN * dN;
            sum_U += U_total; sum_UN += U_total * dN;
            sum_N0 += dN0; sum_N0_2 += dN0 * dN0;
            sum_N1 += dN1; sum_N1_2 += dN1 * dN1;
            sum_Upol0 += cur_Upol0;
            sum_Upol1 += cur_Upol1;
        }
        __syncthreads();
    }

    if (tid == 0) {
        double ns = (n_samples > 0) ? (double)n_samples : 1.0;
        double mean_N = sum_N / ns, mean_N2 = sum_N2 / ns;
        double mean_U = sum_U / ns, mean_UN = sum_UN / ns;
        double var_N = fmax(0.0, mean_N2 - mean_N * mean_N);

        double qst = 0.0;
        if (var_N > 1.0e-9) {
            qst = (cfg.T_K - (mean_UN - mean_U * mean_N) / var_N) * K_TO_KJMOL;
        }

        double m0 = sum_N0 / ns, m1 = sum_N1 / ns;
        double v0 = fmax(0.0, sum_N0_2 / ns - m0 * m0);
        double v1 = fmax(0.0, sum_N1_2 / ns - m1 * m1);

        UnifiedBoxResult r{};
        r.pressure_bar = spec.pressure_bar;
        r.y0 = spec.y0; r.y1 = spec.y1;
        r.f0_Pa = spec.f0_Pa; r.f1_Pa = spec.f1_Pa;
        r.replica_id = spec.replica_id;
        r.avg_N0 = m0; r.avg_N1 = m1;
        r.std_N0 = sqrt(v0); r.std_N1 = sqrt(v1);
        r.uptake0_mmol_g = m0 * 1000.0 / cfg.framework_mass_super_gmol;
        r.uptake1_mmol_g = m1 * 1000.0 / cfg.framework_mass_super_gmol;
        r.selectivity_0_over_1 = (cfg.n_comp == 2 && m1 > 1.0e-12 && spec.y0 > 0.0)
            ? (m0 / m1) * (spec.y1 / spec.y0) : 0.0;
        r.Qst_kJ_mol = qst;
        r.avg_Upol0_kJ = (sum_N0 > 0.0) ? (sum_Upol0 / sum_N0) * K_TO_KJMOL : 0.0;
        r.avg_Upol1_kJ = (sum_N1 > 0.0) ? (sum_Upol1 / sum_N1) * K_TO_KJMOL : 0.0;
        r.max_energy_drift_K = max_drift;
        r.acc_t  = (att_t > 0) ? (double)acc_t / (double)att_t : 0.0;
        r.acc_r  = (att_r > 0) ? (double)acc_r / (double)att_r : 0.0;
        r.acc_i0 = (att_i[0] > 0) ? (double)acc_i[0] / (double)att_i[0] : 0.0;
        r.acc_d0 = (att_d[0] > 0) ? (double)acc_d[0] / (double)att_d[0] : 0.0;
        r.acc_i1 = (att_i[1] > 0) ? (double)acc_i[1] / (double)att_i[1] : 0.0;
        r.acc_d1 = (att_d[1] > 0) ? (double)acc_d[1] / (double)att_d[1] : 0.0;
        r.acc_sw01 = (att_sw01 > 0) ? (double)acc_sw01 / (double)att_sw01 : 0.0;
        r.acc_sw10 = (att_sw10 > 0) ? (double)acc_sw10 / (double)att_sw10 : 0.0;
        results[box_id] = r;
    }
}

} // namespace

double run_unified_gcmc_gpu(
    const Cell& cell,
    const MixGrid3DView& d_grid,
    const SpeciesDef& sp0,
    const SpeciesDef& sp1,
    const UnifiedGcmcConfig& cfg,
    const std::vector<UnifiedBoxSpec>& boxes,
    std::vector<UnifiedBoxResult>& results_out) {

    int num_boxes = (int)boxes.size();
    results_out.resize(num_boxes);

    UnifiedBoxSpec* d_specs = nullptr;
    UnifiedBoxResult* d_results = nullptr;
    CUDA_CHECK(cudaMalloc(&d_specs, num_boxes * sizeof(UnifiedBoxSpec)));
    CUDA_CHECK(cudaMalloc(&d_results, num_boxes * sizeof(UnifiedBoxResult)));
    CUDA_CHECK(cudaMemcpy(d_specs, boxes.data(), num_boxes * sizeof(UnifiedBoxSpec), cudaMemcpyHostToDevice));

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));

    gcmc_unified_kernel<<<num_boxes, BLOCK_THREADS>>>(cell, d_grid, sp0, sp1, cfg, d_specs, d_results);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));

    CUDA_CHECK(cudaMemcpy(results_out.data(), d_results, num_boxes * sizeof(UnifiedBoxResult), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaFree(d_specs));
    CUDA_CHECK(cudaFree(d_results));

    return (double)ms;
}

} // namespace mofgcmc
