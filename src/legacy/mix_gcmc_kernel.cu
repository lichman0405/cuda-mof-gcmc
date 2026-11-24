#include "mix_gcmc.hpp"

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
    nx *= inv_n;
    ny *= inv_n;
    nz *= inv_n;
}

__device__ inline double pair_energy_mix(
    const Cell& cell,
    const SpeciesDef& sp1, const SpeciesDef& sp2,
    double c1x, double c1y, double c1z, double u1x, double u1y, double u1z,
    double mu1x, double mu1y, double mu1z,
    double c2x, double c2y, double c2z, double u2x, double u2y, double u2z,
    double mu2x, double mu2y, double mu2z,
    int pol_mode, double r_cut, double r_cut2,
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
        double ax = off1_s[a] * u1x;
        double ay = off1_s[a] * u1y;
        double az = off1_s[a] * u1z;
        bool lj1 = (sp1.lj_grid_idx[a] >= 0);
        double q1 = sp1.q[a];

        for (int b = 0; b < sp2.n_sites; ++b) {
            bool lj2 = (sp2.lj_grid_idx[b] >= 0);
            double q2 = sp2.q[b];
            bool do_lj = lj1 && lj2;
            bool do_q  = (q1 != 0.0 && q2 != 0.0);
            if (!do_lj && !do_q) continue;

            double bx = off2_s[b] * u2x;
            double by = off2_s[b] * u2y;
            double bz = off2_s[b] * u2z;

            double dx = dm_x + bx - ax;
            double dy = dm_y + by - ay;
            double dz = dm_z + bz - az;
            double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= r_cut2) continue;
            if (r2 < 0.25) { // r < 0.5 A hard overlap
                overlap_flag = 1;
                return 0.0;
            }

            double r = sqrt(r2);
            double inv_r = 1.0 / r;
            double inv_r2 = 1.0 / r2;

            if (do_lj) {
                double sig_ab = 0.5 * (sp1.sigma_A[a] + sp2.sigma_A[b]);
                double eps_ab = sqrt(sp1.eps_K[a] * sp2.eps_K[b]);
                double sr2 = (sig_ab * sig_ab) * inv_r2;
                double sr6 = sr2 * sr2 * sr2;
                energy += 4.0 * eps_ab * (sr6 * sr6 - sr6);
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
        // Standard Thole reduced distance u = r / (alpha1 * alpha2)^(1/6)
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

struct MixSharedProposal {
    int move_type;   // 0=trans, 1=rot, 2=ins, 3=del, 4=swap
    int target_idx;
    int sp_new;      // 0 or 1
    int sp_old;      // 0 or 1
    int cur_N;
    int fw_overlap;
    int skip_gg;
    double cx, cy, cz, ux, uy, uz;
    double mux, muy, muz;
    double u_fw_new;
    double u_fw_old;
    double u_pol_new;
    double u_pol_old;
};

__global__ void gcmc_binary_kernel(
    Cell cell,
    MixGrid3DView grid,
    SpeciesDef sp0,
    SpeciesDef sp1,
    MixGcmcConfig cfg,
    const MixBoxSpec* __restrict__ specs,
    MixBoxResult* __restrict__ results,
    unsigned long long* __restrict__ hist0_out,
    unsigned long long* __restrict__ hist1_out,
    int hist_na, int hist_nb) {

    int box_id = blockIdx.x;
    int tid = threadIdx.x;
    MixBoxSpec spec = specs[box_id];

    __shared__ float s_cx[SMEM_MAX_MOL];
    __shared__ float s_cy[SMEM_MAX_MOL];
    __shared__ float s_cz[SMEM_MAX_MOL];
    __shared__ float s_ux[SMEM_MAX_MOL];
    __shared__ float s_uy[SMEM_MAX_MOL];
    __shared__ float s_uz[SMEM_MAX_MOL];
    __shared__ float s_mux[SMEM_MAX_MOL];
    __shared__ float s_muy[SMEM_MAX_MOL];
    __shared__ float s_muz[SMEM_MAX_MOL];
    __shared__ float s_ufw[SMEM_MAX_MOL];
    __shared__ float s_upol[SMEM_MAX_MOL];
    __shared__ int   s_sp[SMEM_MAX_MOL];

    __shared__ MixSharedProposal prop;
    __shared__ double warp_u[4];
    __shared__ int warp_ovl[4];

    uint64_t rng = cfg.base_seed + (uint64_t)box_id * 1000003ULL + 0x9e3779b97f4a7c15ULL;
    for (int k = 0; k < 8; ++k) xorshift64star(rng);

    int cur_N = 0;
    int cur_N0 = 0;
    int cur_N1 = 0;
    double delta_trans = 1.2;
    double max_angle = 0.65;

    constexpr double KB_SI = 1.380649e-23;
    double beta_f_V[2] = {
        (spec.f0_Pa * (cell.volume_super * 1.0e-30)) / (KB_SI * cfg.T_K),
        (spec.f1_Pa * (cell.volume_super * 1.0e-30)) / (KB_SI * cfg.T_K)
    };

    double rc = cfg.r_cut_A;
    double rc2 = rc * rc;
    double a_dsf = cfg.dsf_alpha;
    constexpr double INV_SQRT_PI = 0.56418958354775628695;
    double erfc_rc = erfc(a_dsf * rc);
    double exp_rc = exp(-a_dsf * a_dsf * rc2);
    double dsf_shift = erfc_rc / rc;
    double dsf_deriv = erfc_rc / rc2 + (2.0 * a_dsf * INV_SQRT_PI) * exp_rc / rc;

    long long att_t = 0, acc_t = 0, att_r = 0, acc_r = 0;
    long long att_i[2] = {0, 0}, acc_i[2] = {0, 0};
    long long att_d[2] = {0, 0}, acc_d[2] = {0, 0};
    long long att_sw01 = 0, acc_sw01 = 0, att_sw10 = 0, acc_sw10 = 0;
    long long win_att_t = 0, win_acc_t = 0;

    long long n_samples = 0;
    double sum_N0 = 0.0, sum_N0_2 = 0.0, sum_N1 = 0.0, sum_N1_2 = 0.0;
    double sum_Upol0 = 0.0, sum_Upol1 = 0.0;
    double cur_Upol0 = 0.0, cur_Upol1 = 0.0;

    int total_steps = cfg.eq_steps + cfg.prod_steps;

    for (int step = 0; step < total_steps; ++step) {
        bool in_prod = (step >= cfg.eq_steps);

        if (tid == 0) {
            prop.cur_N = cur_N;
            prop.fw_overlap = 0;
            prop.skip_gg = 0;
            prop.u_fw_new = 0.0;
            prop.u_fw_old = 0.0;
            prop.u_pol_new = 0.0;
            prop.u_pol_old = 0.0;
            prop.mux = prop.muy = prop.muz = 0.0;

            int n_moves = cfg.enable_swap ? 5 : 4;
            int mv = (int)(rng_u01(rng) * (double)n_moves);
            if (mv >= n_moves) mv = n_moves - 1;
            prop.move_type = mv;

            auto eval_fw = [&](int sp_id, double cx, double cy, double cz,
                               double ux, double uy, double uz,
                               double& ufw, double& upol,
                               double& mx, double& my, double& mz) -> bool {
                if (cfg.ideal_gas_test) {
                    ufw = 0.0; upol = 0.0; mx = my = mz = 0.0;
                    return false;
                }
                const SpeciesDef& sp = (sp_id == 0) ? sp0 : sp1;
                double ulj = 0.0, uel = 0.0;
                bool ovl = eval_species_framework_grid(
                    grid, cell, sp, cfg.pol_mode, KE_K,
                    cx, cy, cz, ux, uy, uz,
                    ulj, uel, upol, mx, my, mz);
                ufw = ulj + uel + upol;
                return ovl;
            };

            if (mv == 0 || mv == 1) { // Trans or Rot
                if (cur_N == 0) {
                    prop.skip_gg = 1;
                } else {
                    int idx = (int)(rng_u01(rng) * (double)cur_N);
                    if (idx >= cur_N) idx = cur_N - 1;
                    int sp_id = s_sp[idx];
                    prop.target_idx = idx;
                    prop.sp_old = sp_id;
                    prop.sp_new = sp_id;
                    prop.u_fw_old = (double)s_ufw[idx];
                    prop.u_pol_old = (double)s_upol[idx];

                    double ox = (double)s_cx[idx], oy = (double)s_cy[idx], oz = (double)s_cz[idx];
                    double oux = (double)s_ux[idx], ouy = (double)s_uy[idx], ouz = (double)s_uz[idx];

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
                    if (ovl) {
                        prop.fw_overlap = 1;
                        prop.skip_gg = 1;
                    } else if (cfg.ideal_gas_test) {
                        prop.skip_gg = 1;
                    }
                }
            } else if (mv == 2) { // Insertion
                int sp_id = (rng_u01(rng) < 0.5) ? 0 : 1;
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
                    if (ovl) {
                        prop.fw_overlap = 1;
                        prop.skip_gg = 1;
                    } else if (cfg.ideal_gas_test) {
                        prop.skip_gg = 1;
                    }
                }
            } else if (mv == 3) { // Deletion
                int sp_id = (rng_u01(rng) < 0.5) ? 0 : 1;
                prop.sp_old = sp_id;
                prop.sp_new = sp_id;
                int count_sp = (sp_id == 0) ? cur_N0 : cur_N1;
                if (count_sp == 0) {
                    prop.skip_gg = 1;
                } else {
                    // Pick uniformly among molecules of species sp_id
                    int nth = (int)(rng_u01(rng) * (double)count_sp);
                    if (nth >= count_sp) nth = count_sp - 1;
                    int idx = -1, seen = 0;
                    for (int k = 0; k < cur_N; ++k) {
                        if (s_sp[k] == sp_id) {
                            if (seen == nth) { idx = k; break; }
                            ++seen;
                        }
                    }
                    prop.target_idx = idx;
                    prop.u_fw_old = (double)s_ufw[idx];
                    prop.u_pol_old = (double)s_upol[idx];
                    prop.cx = (double)s_cx[idx]; prop.cy = (double)s_cy[idx]; prop.cz = (double)s_cz[idx];
                    prop.ux = (double)s_ux[idx]; prop.uy = (double)s_uy[idx]; prop.uz = (double)s_uz[idx];
                    prop.mux = (double)s_mux[idx]; prop.muy = (double)s_muy[idx]; prop.muz = (double)s_muz[idx];
                    if (cfg.ideal_gas_test) prop.skip_gg = 1;
                }
            } else { // Move 4: Identity Swap (0->1 or 1->0)
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
                        if (s_sp[k] == src_sp) {
                            if (seen == nth) { idx = k; break; }
                            ++seen;
                        }
                    }
                    prop.target_idx = idx;
                    prop.u_fw_old = (double)s_ufw[idx];
                    prop.u_pol_old = (double)s_upol[idx];
                    prop.cx = (double)s_cx[idx]; prop.cy = (double)s_cy[idx]; prop.cz = (double)s_cz[idx];
                    prop.ux = (double)s_ux[idx]; prop.uy = (double)s_uy[idx]; prop.uz = (double)s_uz[idx];

                    bool ovl = eval_fw(dst_sp, prop.cx, prop.cy, prop.cz,
                                       prop.ux, prop.uy, prop.uz,
                                       prop.u_fw_new, prop.u_pol_new,
                                       prop.mux, prop.muy, prop.muz);
                    if (ovl) {
                        prop.fw_overlap = 1;
                        prop.skip_gg = 1;
                    } else if (cfg.ideal_gas_test) {
                        prop.skip_gg = 1;
                    }
                }
            }
        }
        __syncthreads();

        double local_du_gg = 0.0;
        int local_ovl = 0;

        if (!prop.skip_gg && prop.cur_N > 0) {
            int N = prop.cur_N;
            int mv = prop.move_type;
            int tgt = prop.target_idx;
            const SpeciesDef& sp_new_def = (prop.sp_new == 0) ? sp0 : sp1;
            const SpeciesDef& sp_old_def = (prop.sp_old == 0) ? sp0 : sp1;

            if (mv == 0 || mv == 1 || mv == 4) {
                double ox = (double)s_cx[tgt], oy = (double)s_cy[tgt], oz = (double)s_cz[tgt];
                double oux = (double)s_ux[tgt], ouy = (double)s_uy[tgt], ouz = (double)s_uz[tgt];
                double omux = (double)s_mux[tgt], omuy = (double)s_muy[tgt], omuz = (double)s_muz[tgt];

                for (int j = tid; j < N; j += BLOCK_THREADS) {
                    if (j == tgt) continue;
                    const SpeciesDef& sp_j = (s_sp[j] == 0) ? sp0 : sp1;
                    double jx = (double)s_cx[j], jy = (double)s_cy[j], jz = (double)s_cz[j];
                    double jux = (double)s_ux[j], juy = (double)s_uy[j], juz = (double)s_uz[j];
                    double jmux = (double)s_mux[j], jmuy = (double)s_muy[j], jmuz = (double)s_muz[j];

                    int ov_new = 0, ov_old = 0;
                    double enew = pair_energy_mix(
                        cell, sp_new_def, sp_j,
                        prop.cx, prop.cy, prop.cz, prop.ux, prop.uy, prop.uz,
                        prop.mux, prop.muy, prop.muz,
                        jx, jy, jz, jux, juy, juz, jmux, jmuy, jmuz,
                        cfg.pol_mode, rc, rc2, a_dsf, dsf_shift, dsf_deriv, ov_new);
                    if (ov_new) { local_ovl = 1; break; }
                    double eold = pair_energy_mix(
                        cell, sp_old_def, sp_j,
                        ox, oy, oz, oux, ouy, ouz, omux, omuy, omuz,
                        jx, jy, jz, jux, juy, juz, jmux, jmuy, jmuz,
                        cfg.pol_mode, rc, rc2, a_dsf, dsf_shift, dsf_deriv, ov_old);
                    local_du_gg += (enew - eold);
                }
            } else if (mv == 2) {
                for (int j = tid; j < N; j += BLOCK_THREADS) {
                    const SpeciesDef& sp_j = (s_sp[j] == 0) ? sp0 : sp1;
                    int ov_new = 0;
                    double enew = pair_energy_mix(
                        cell, sp_new_def, sp_j,
                        prop.cx, prop.cy, prop.cz, prop.ux, prop.uy, prop.uz,
                        prop.mux, prop.muy, prop.muz,
                        (double)s_cx[j], (double)s_cy[j], (double)s_cz[j],
                        (double)s_ux[j], (double)s_uy[j], (double)s_uz[j],
                        (double)s_mux[j], (double)s_muy[j], (double)s_muz[j],
                        cfg.pol_mode, rc, rc2, a_dsf, dsf_shift, dsf_deriv, ov_new);
                    if (ov_new) { local_ovl = 1; break; }
                    local_du_gg += enew;
                }
            } else { // Deletion
                for (int j = tid; j < N; j += BLOCK_THREADS) {
                    if (j == tgt) continue;
                    const SpeciesDef& sp_j = (s_sp[j] == 0) ? sp0 : sp1;
                    int ov_old = 0;
                    double eold = pair_energy_mix(
                        cell, sp_old_def, sp_j,
                        prop.cx, prop.cy, prop.cz, prop.ux, prop.uy, prop.uz,
                        prop.mux, prop.muy, prop.muz,
                        (double)s_cx[j], (double)s_cy[j], (double)s_cz[j],
                        (double)s_ux[j], (double)s_uy[j], (double)s_uz[j],
                        (double)s_mux[j], (double)s_muy[j], (double)s_muz[j],
                        cfg.pol_mode, rc, rc2, a_dsf, dsf_shift, dsf_deriv, ov_old);
                    local_du_gg -= eold;
                }
            }
        }

        int lane = tid & 31;
        int warp_id = tid >> 5;
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_du_gg += __shfl_down_sync(0xFFFFFFFFU, local_du_gg, offset);
            local_ovl |= __shfl_down_sync(0xFFFFFFFFU, local_ovl, offset);
        }
        if (lane == 0) {
            warp_u[warp_id] = local_du_gg;
            warp_ovl[warp_id] = local_ovl;
        }
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
                        s_cx[tgt] = (float)prop.cx; s_cy[tgt] = (float)prop.cy; s_cz[tgt] = (float)prop.cz;
                        s_mux[tgt] = (float)prop.mux; s_muy[tgt] = (float)prop.muy; s_muz[tgt] = (float)prop.muz;
                        s_ufw[tgt] = (float)prop.u_fw_new; s_upol[tgt] = (float)prop.u_pol_new;
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
                        s_ux[tgt] = (float)prop.ux; s_uy[tgt] = (float)prop.uy; s_uz[tgt] = (float)prop.uz;
                        s_mux[tgt] = (float)prop.mux; s_muy[tgt] = (float)prop.muy; s_muz[tgt] = (float)prop.muz;
                        s_ufw[tgt] = (float)prop.u_fw_new; s_upol[tgt] = (float)prop.u_pol_new;
                        if (prop.sp_new == 0) cur_Upol0 += (prop.u_pol_new - prop.u_pol_old);
                        else                  cur_Upol1 += (prop.u_pol_new - prop.u_pol_old);
                        if (in_prod) ++acc_r;
                    }
                }
            } else if (mv == 2) { // Insertion
                int s = prop.sp_new;
                if (in_prod) ++att_i[s];
                if (cur_N < cfg.max_mol && cur_N < SMEM_MAX_MOL && !prop.fw_overlap && !gg_ovl) {
                    int n_s = (s == 0) ? cur_N0 : cur_N1;
                    double dU = prop.u_fw_new + dU_gg;
                    double arg = -dU / cfg.T_K;
                    double ratio = (beta_f_V[s] / (double)(n_s + 1)) * exp(arg > 50.0 ? 50.0 : arg);
                    if ((arg > -50.0) && (ratio >= 1.0 || rng_u01(rng) < ratio)) {
                        s_cx[cur_N] = (float)prop.cx; s_cy[cur_N] = (float)prop.cy; s_cz[cur_N] = (float)prop.cz;
                        s_ux[cur_N] = (float)prop.ux; s_uy[cur_N] = (float)prop.uy; s_uz[cur_N] = (float)prop.uz;
                        s_mux[cur_N] = (float)prop.mux; s_muy[cur_N] = (float)prop.muy; s_muz[cur_N] = (float)prop.muz;
                        s_ufw[cur_N] = (float)prop.u_fw_new; s_upol[cur_N] = (float)prop.u_pol_new;
                        s_sp[cur_N] = s;
                        ++cur_N;
                        if (s == 0) { ++cur_N0; cur_Upol0 += prop.u_pol_new; }
                        else        { ++cur_N1; cur_Upol1 += prop.u_pol_new; }
                        if (in_prod) ++acc_i[s];
                    }
                }
            } else if (mv == 3) { // Deletion
                int s = prop.sp_old;
                if (in_prod) ++att_d[s];
                int n_s = (s == 0) ? cur_N0 : cur_N1;
                if (n_s > 0 && !gg_ovl) {
                    double dU = (-prop.u_fw_old) + dU_gg;
                    double arg = -dU / cfg.T_K;
                    double ratio = ((double)n_s / beta_f_V[s]) * exp(arg > 50.0 ? 50.0 : arg);
                    if ((arg > -50.0) && (ratio >= 1.0 || rng_u01(rng) < ratio)) {
                        int tgt = prop.target_idx;
                        int last = cur_N - 1;
                        if (tgt != last) {
                            s_cx[tgt] = s_cx[last]; s_cy[tgt] = s_cy[last]; s_cz[tgt] = s_cz[last];
                            s_ux[tgt] = s_ux[last]; s_uy[tgt] = s_uy[last]; s_uz[tgt] = s_uz[last];
                            s_mux[tgt] = s_mux[last]; s_muy[tgt] = s_muy[last]; s_muz[tgt] = s_muz[last];
                            s_ufw[tgt] = s_ufw[last]; s_upol[tgt] = s_upol[last]; s_sp[tgt] = s_sp[last];
                        }
                        --cur_N;
                        if (s == 0) { --cur_N0; cur_Upol0 -= prop.u_pol_old; }
                        else        { --cur_N1; cur_Upol1 -= prop.u_pol_old; }
                        if (in_prod) ++acc_d[s];
                    }
                }
            } else { // Identity Swap (src_sp -> dst_sp)
                int a = prop.sp_old;
                int b = prop.sp_new;
                if (in_prod) {
                    if (a == 0) ++att_sw01; else ++att_sw10;
                }
                int n_a = (a == 0) ? cur_N0 : cur_N1;
                int n_b = (b == 0) ? cur_N0 : cur_N1;
                if (n_a > 0 && !prop.fw_overlap && !gg_ovl) {
                    double dU = (prop.u_fw_new - prop.u_fw_old) + dU_gg;
                    double arg = -dU / cfg.T_K;
                    double ratio = (beta_f_V[b] / beta_f_V[a]) * ((double)n_a / (double)(n_b + 1))
                                 * exp(arg > 50.0 ? 50.0 : arg);
                    if ((arg > -50.0) && (ratio >= 1.0 || rng_u01(rng) < ratio)) {
                        int tgt = prop.target_idx;
                        s_mux[tgt] = (float)prop.mux; s_muy[tgt] = (float)prop.muy; s_muz[tgt] = (float)prop.muz;
                        s_ufw[tgt] = (float)prop.u_fw_new; s_upol[tgt] = (float)prop.u_pol_new;
                        s_sp[tgt] = b;
                        if (a == 0) {
                            --cur_N0; ++cur_N1;
                            cur_Upol0 -= prop.u_pol_old;
                            cur_Upol1 += prop.u_pol_new;
                            if (in_prod) ++acc_sw01;
                        } else {
                            --cur_N1; ++cur_N0;
                            cur_Upol1 -= prop.u_pol_old;
                            cur_Upol0 += prop.u_pol_new;
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
                win_att_t = 0;
                win_acc_t = 0;
            }

            if (in_prod && ((step - cfg.eq_steps) % cfg.sample_stride == 0)) {
                ++n_samples;
                double d0 = (double)cur_N0, d1 = (double)cur_N1;
                sum_N0 += d0; sum_N0_2 += d0 * d0;
                sum_N1 += d1; sum_N1_2 += d1 * d1;
                sum_Upol0 += cur_Upol0;
                sum_Upol1 += cur_Upol1;

                if (spec.record_hist && hist0_out != nullptr && hist1_out != nullptr && cur_N > 0) {
                    if (((step - cfg.eq_steps) / cfg.sample_stride) % 5 == 0) {
                        for (int m = 0; m < cur_N; ++m) {
                            double sa, sb, sc;
                            cell.cart_to_unit_frac((double)s_cx[m], (double)s_cy[m], (double)s_cz[m],
                                                   sa, sb, sc);
                            sa -= floor(sa);
                            sb -= floor(sb);
                            int ia = (int)(sa * hist_na);
                            int ib = (int)(sb * hist_nb);
                            if (ia >= hist_na) ia = hist_na - 1;
                            if (ib >= hist_nb) ib = hist_nb - 1;
                            if (s_sp[m] == 0) atomicAdd(&hist0_out[ib * hist_na + ia], 1ULL);
                            else              atomicAdd(&hist1_out[ib * hist_na + ia], 1ULL);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        double ns = (n_samples > 0) ? (double)n_samples : 1.0;
        double m0 = sum_N0 / ns, m1 = sum_N1 / ns;
        double v0 = sum_N0_2 / ns - m0 * m0; if (v0 < 0.0) v0 = 0.0;
        double v1 = sum_N1_2 / ns - m1 * m1; if (v1 < 0.0) v1 = 0.0;

        double up0 = m0 * 1000.0 / cfg.framework_mass_super_gmol;
        double up1 = m1 * 1000.0 / cfg.framework_mass_super_gmol;
        double sel = 0.0;
        if (m1 > 1.0e-9 && spec.y0 > 0.0) {
            sel = (m0 / m1) * (spec.y1 / spec.y0);
        }

        MixBoxResult r{};
        r.pressure_bar = spec.pressure_bar;
        r.y0 = spec.y0; r.y1 = spec.y1;
        r.f0_Pa = spec.f0_Pa; r.f1_Pa = spec.f1_Pa;
        r.replica_id = spec.replica_id;
        r.avg_N0 = m0; r.avg_N1 = m1;
        r.std_N0 = sqrt(v0); r.std_N1 = sqrt(v1);
        r.uptake0_mmol_g = up0; r.uptake1_mmol_g = up1;
        r.selectivity_0_over_1 = sel;
        r.avg_Upol0_kJ = (m0 > 1.0e-6) ? ((sum_Upol0 / ns / m0) * K_TO_KJMOL) : 0.0;
        r.avg_Upol1_kJ = (m1 > 1.0e-6) ? ((sum_Upol1 / ns / m1) * K_TO_KJMOL) : 0.0;
        r.acc_t = (att_t > 0) ? (double)acc_t / (double)att_t : 0.0;
        r.acc_r = (att_r > 0) ? (double)acc_r / (double)att_r : 0.0;
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

double run_binary_gcmc_gpu(
    const Cell& cell,
    const MixGrid3DView& d_grid,
    const SpeciesDef& sp0,
    const SpeciesDef& sp1,
    const MixGcmcConfig& cfg,
    const std::vector<MixBoxSpec>& boxes,
    std::vector<MixBoxResult>& results_out,
    std::vector<double>& hist0_out,
    std::vector<double>& hist1_out,
    int hist_na, int hist_nb) {

    int num_boxes = (int)boxes.size();
    results_out.resize(num_boxes);

    MixBoxSpec* d_specs = nullptr;
    MixBoxResult* d_results = nullptr;
    unsigned long long *d_hist0 = nullptr, *d_hist1 = nullptr;

    CUDA_CHECK(cudaMalloc(&d_specs, num_boxes * sizeof(MixBoxSpec)));
    CUDA_CHECK(cudaMalloc(&d_results, num_boxes * sizeof(MixBoxResult)));
    CUDA_CHECK(cudaMemcpy(d_specs, boxes.data(),
                          num_boxes * sizeof(MixBoxSpec), cudaMemcpyHostToDevice));

    size_t hist_size = (size_t)hist_na * (size_t)hist_nb;
    CUDA_CHECK(cudaMalloc(&d_hist0, hist_size * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_hist1, hist_size * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_hist0, 0, hist_size * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_hist1, 0, hist_size * sizeof(unsigned long long)));

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));

    gcmc_binary_kernel<<<num_boxes, BLOCK_THREADS>>>(
        cell, d_grid, sp0, sp1, cfg, d_specs, d_results, d_hist0, d_hist1, hist_na, hist_nb);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));

    CUDA_CHECK(cudaMemcpy(results_out.data(), d_results,
                          num_boxes * sizeof(MixBoxResult), cudaMemcpyDeviceToHost));

    std::vector<unsigned long long> h0(hist_size), h1(hist_size);
    CUDA_CHECK(cudaMemcpy(h0.data(), d_hist0, hist_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h1.data(), d_hist1, hist_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    hist0_out.resize(hist_size);
    hist1_out.resize(hist_size);
    for (size_t i = 0; i < hist_size; ++i) {
        hist0_out[i] = (double)h0[i];
        hist1_out[i] = (double)h1[i];
    }

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaFree(d_specs));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_hist0));
    CUDA_CHECK(cudaFree(d_hist1));

    return (double)ms;
}

} // namespace mofgcmc
