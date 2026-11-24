#include "proposer_api.h"
#include "cell.hpp"
#include "cif_loader.hpp"
#include "forcefield.hpp"
#include "unified_gcmc.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using namespace mofgcmc;

#define CUDA_CHECK_RET(call, ret)                                               \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA Error at %s:%d: %s\n",                   \
                         __FILE__, __LINE__, cudaGetErrorString(err));          \
            return (ret);                                                       \
        }                                                                       \
    } while (0)

constexpr int PROP_THREADS = 128;
constexpr int PROP_SMEM_MAX = 256;

__device__ inline uint64_t xs64(uint64_t& s) {
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 2685821657736338717ULL;
}

__device__ inline double u01(uint64_t& s) {
    return (double)(xs64(s) >> 11) * (1.0 / 9007199254740992.0);
}

__device__ inline void rand_unit_vec(uint64_t& rng, double& ux, double& uy, double& uz) {
    double z1, z2, s2;
    do {
        z1 = 2.0 * u01(rng) - 1.0;
        z2 = 2.0 * u01(rng) - 1.0;
        s2 = z1 * z1 + z2 * z2;
    } while (s2 >= 1.0 || s2 < 1.0e-12);
    double sq = sqrt(1.0 - s2);
    ux = 2.0 * z1 * sq;
    uy = 2.0 * z2 * sq;
    uz = 1.0 - 2.0 * s2;
}

__device__ inline void rot_rod(
    uint64_t& rng, double max_angle,
    double ux, double uy, double uz,
    double& nx, double& ny, double& nz) {
    double kx, ky, kz;
    rand_unit_vec(rng, kx, ky, kz);
    double theta = (2.0 * u01(rng) - 1.0) * max_angle;
    double st, ct;
    sincos(theta, &st, &ct);
    double kd = kx * ux + ky * uy + kz * uz;
    double cx = ky * uz - kz * uy;
    double cy = kz * ux - kx * uz;
    double cz = kx * uy - ky * ux;
    nx = ux * ct + cx * st + kx * kd * (1.0 - ct);
    ny = uy * ct + cy * st + ky * kd * (1.0 - ct);
    nz = uz * ct + cz * st + kz * kd * (1.0 - ct);
    double inv = rsqrt(nx * nx + ny * ny + nz * nz);
    nx *= inv; ny *= inv; nz *= inv;
}

__device__ inline double pair_co2(
    const Cell& cell, const SpeciesDef& sp,
    double c1x, double c1y, double c1z, double u1x, double u1y, double u1z,
    double m1x, double m1y, double m1z,
    double c2x, double c2y, double c2z, double u2x, double u2y, double u2z,
    double m2x, double m2y, double m2z,
    int shifted_lj, int pol_mode,
    double r_cut, double r_cut2, double dsf_alpha, double dsf_shift, double dsf_deriv,
    int& ovl) {

    double dmx, dmy, dmz, r2c;
    cell.mic_delta_super(c2x - c1x, c2y - c1y, c2z - c1z, dmx, dmy, dmz, r2c);
    double skip = r_cut + 2.0 * sp.bond_A;
    if (r2c > skip * skip) return 0.0;

    const double off[3] = {0.0, sp.bond_A, -sp.bond_A};
    double e = 0.0;
    for (int a = 0; a < 3; ++a) {
        double ax = off[a] * u1x, ay = off[a] * u1y, az = off[a] * u1z;
        for (int b = 0; b < 3; ++b) {
            double dx = dmx + off[b] * u2x - ax;
            double dy = dmy + off[b] * u2y - ay;
            double dz = dmz + off[b] * u2z - az;
            double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= r_cut2) continue;
            if (r2 < 0.25) { ovl = 1; return 0.0; }

            double sig = 0.5 * (sp.sigma_A[a] + sp.sigma_A[b]);
            double eps = sqrt(sp.eps_K[a] * sp.eps_K[b]);
            double sr2 = (sig * sig) / r2;
            double sr6 = sr2 * sr2 * sr2;
            double ulj = 4.0 * eps * (sr6 * sr6 - sr6);
            if (shifted_lj) {
                double sc2 = (sig * sig) / r_cut2;
                double sc6 = sc2 * sc2 * sc2;
                ulj -= 4.0 * eps * (sc6 * sc6 - sc6);
            }
            double r = sqrt(r2);
            double uq = KE_K * sp.q[a] * sp.q[b] *
                        (erfc(dsf_alpha * r) / r - dsf_shift + dsf_deriv * (r - r_cut));
            e += ulj + uq;
        }
    }

    if (pol_mode != 0 && r2c < r_cut2 && r2c >= 0.25) {
        double rm = sqrt(r2c), invr = 1.0 / rm;
        double invr3 = invr / r2c, invr5 = invr3 / r2c;
        double invs = 1.0 / cbrt(sp.alpha_iso);
        double ut = rm * invs;
        double au3 = trappe_co2::THOLE_A * ut * ut * ut;
        double et = (au3 < 50.0) ? exp(-au3) : 0.0;
        double fe = 1.0 - et;
        double ft = 1.0 - (1.0 + au3) * et;
        double m1m2 = m1x * m2x + m1y * m2y + m1z * m2z;
        double m1r = m1x * dmx + m1y * dmy + m1z * dmz;
        double m2r = m2x * dmx + m2y * dmy + m2z * dmz;
        e += KE_K * (fe * m1m2 * invr3 - 3.0 * ft * m1r * m2r * invr5);
    }
    return e;
}

struct ProposerStateDev {
    int* d_acc_N = nullptr;
    double* d_acc_U = nullptr;
    double* d_acc_cx = nullptr;
    double* d_acc_cy = nullptr;
    double* d_acc_cz = nullptr;
    double* d_acc_ux = nullptr;
    double* d_acc_uy = nullptr;
    double* d_acc_uz = nullptr;

    int* d_tri_N = nullptr;
    double* d_tri_U = nullptr;
    double* d_tri_dU = nullptr;
    int* d_tri_acc = nullptr;
    double* d_tri_cx = nullptr;
    double* d_tri_cy = nullptr;
    double* d_tri_cz = nullptr;
    double* d_tri_ux = nullptr;
    double* d_tri_uy = nullptr;
    double* d_tri_uz = nullptr;

    int* d_commit_mask = nullptr;
    uint64_t* d_rng = nullptr;
};

struct PropMove {
    int mv;
    int tgt;
    int cur_N;
    int fw_ovl;
    int skip;
    double cx, cy, cz, ux, uy, uz;
    double mux, muy, muz;
    double u_fw_new, u_fw_old;
};

__global__ void propose_burst_kernel(
    Cell cell, MixGrid3DView grid, SpeciesDef sp,
    double T_K, double beta_f_V, double r_cut_gg, double dsf_alpha,
    int ff_mode, int pol_mode, int max_mol,
    int k_steps, int commit_to_acc,
    ProposerStateDev st) {

    int b = blockIdx.x;
    int tid = threadIdx.x;
    int off = b * max_mol;

    __shared__ double s_cx[PROP_SMEM_MAX], s_cy[PROP_SMEM_MAX], s_cz[PROP_SMEM_MAX];
    __shared__ double s_ux[PROP_SMEM_MAX], s_uy[PROP_SMEM_MAX], s_uz[PROP_SMEM_MAX];
    __shared__ double s_mux[PROP_SMEM_MAX], s_muy[PROP_SMEM_MAX], s_muz[PROP_SMEM_MAX];
    __shared__ double s_ufw[PROP_SMEM_MAX];
    __shared__ PropMove prop;
    __shared__ double warp_u[4];
    __shared__ int warp_ovl[4];
    __shared__ double s_u_recomp;

    int init_N = st.d_acc_N[b];
    if (init_N > max_mol) init_N = max_mol;
    for (int i = tid; i < init_N; i += PROP_THREADS) {
        double cx = st.d_acc_cx[off + i], cy = st.d_acc_cy[off + i], cz = st.d_acc_cz[off + i];
        double ux = st.d_acc_ux[off + i], uy = st.d_acc_uy[off + i], uz = st.d_acc_uz[off + i];
        s_cx[i] = cx; s_cy[i] = cy; s_cz[i] = cz;
        s_ux[i] = ux; s_uy[i] = uy; s_uz[i] = uz;
        double ulj = 0.0, uel = 0.0, up = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
        eval_species_framework_grid_unified(grid, cell, sp, 1, pol_mode, KE_K, cx, cy, cz, ux, uy, uz,
                                            ulj, uel, up, mx, my, mz);
        s_ufw[i] = ulj + uel + up;
        s_mux[i] = mx; s_muy[i] = my; s_muz[i] = mz;
    }
    __syncthreads();

    uint64_t rng = st.d_rng[b];
    int cur_N = init_N;
    double U_init = st.d_acc_U[b];
    double U_total = U_init;
    int inner_acc = 0;

    // Strictly fixed proposal step widths for exact detailed balance in Nested MC
    constexpr double DELTA_TRANS = 1.0;
    constexpr double MAX_ANGLE = 0.55;
    int shifted_lj = (ff_mode == 1) ? 1 : 0;
    double rc = r_cut_gg, rc2 = rc * rc;
    constexpr double INV_SQRT_PI = 0.56418958354775628695;
    double erfc_rc = erfc(dsf_alpha * rc), exp_rc = exp(-dsf_alpha * dsf_alpha * rc2);
    double dsf_shift = erfc_rc / rc;
    double dsf_deriv = erfc_rc / rc2 + (2.0 * dsf_alpha * INV_SQRT_PI) * exp_rc / rc;

    for (int step = 0; step < k_steps; ++step) {
        if (tid == 0) {
            prop.cur_N = cur_N;
            prop.fw_ovl = 0;
            prop.skip = 0;
            prop.u_fw_new = prop.u_fw_old = 0.0;
            prop.mux = prop.muy = prop.muz = 0.0;

            int mv = (int)(u01(rng) * 4.0);
            if (mv > 3) mv = 3;
            prop.mv = mv;

            if (mv == 0 || mv == 1) {
                if (cur_N == 0) {
                    prop.skip = 1;
                } else {
                    int idx = (int)(u01(rng) * (double)cur_N);
                    if (idx >= cur_N) idx = cur_N - 1;
                    prop.tgt = idx;
                    prop.u_fw_old = s_ufw[idx];
                    double ox = s_cx[idx], oy = s_cy[idx], oz = s_cz[idx];
                    double oux = s_ux[idx], ouy = s_uy[idx], ouz = s_uz[idx];
                    if (mv == 0) {
                        double nx = ox + (2.0 * u01(rng) - 1.0) * DELTA_TRANS;
                        double ny = oy + (2.0 * u01(rng) - 1.0) * DELTA_TRANS;
                        double nz = oz + (2.0 * u01(rng) - 1.0) * DELTA_TRANS;
                        cell.wrap_super_cart(nx, ny, nz);
                        prop.cx = nx; prop.cy = ny; prop.cz = nz;
                        prop.ux = oux; prop.uy = ouy; prop.uz = ouz;
                    } else {
                        prop.cx = ox; prop.cy = oy; prop.cz = oz;
                        rot_rod(rng, MAX_ANGLE, oux, ouy, ouz, prop.ux, prop.uy, prop.uz);
                    }
                    double ulj = 0.0, uel = 0.0, up = 0.0;
                    bool ovl = eval_species_framework_grid_unified(
                        grid, cell, sp, 1, pol_mode, KE_K,
                        prop.cx, prop.cy, prop.cz, prop.ux, prop.uy, prop.uz,
                        ulj, uel, up, prop.mux, prop.muy, prop.muz);
                    prop.u_fw_new = ulj + uel + up;
                    if (ovl) { prop.fw_ovl = 1; prop.skip = 1; }
                }
            } else if (mv == 2) {
                if (cur_N >= max_mol || cur_N >= PROP_SMEM_MAX) {
                    prop.skip = 1;
                } else {
                    prop.tgt = -1;
                    double sa = u01(rng), sb = u01(rng), sc = u01(rng);
                    cell.super_frac_to_cart(sa, sb, sc, prop.cx, prop.cy, prop.cz);
                    rand_unit_vec(rng, prop.ux, prop.uy, prop.uz);
                    double ulj = 0.0, uel = 0.0, up = 0.0;
                    bool ovl = eval_species_framework_grid_unified(
                        grid, cell, sp, 1, pol_mode, KE_K,
                        prop.cx, prop.cy, prop.cz, prop.ux, prop.uy, prop.uz,
                        ulj, uel, up, prop.mux, prop.muy, prop.muz);
                    prop.u_fw_new = ulj + uel + up;
                    if (ovl) { prop.fw_ovl = 1; prop.skip = 1; }
                }
            } else {
                if (cur_N == 0) {
                    prop.skip = 1;
                } else {
                    int idx = (int)(u01(rng) * (double)cur_N);
                    if (idx >= cur_N) idx = cur_N - 1;
                    prop.tgt = idx;
                    prop.u_fw_old = s_ufw[idx];
                    prop.cx = s_cx[idx]; prop.cy = s_cy[idx]; prop.cz = s_cz[idx];
                    prop.ux = s_ux[idx]; prop.uy = s_uy[idx]; prop.uz = s_uz[idx];
                    prop.mux = s_mux[idx]; prop.muy = s_muy[idx]; prop.muz = s_muz[idx];
                }
            }
        }
        __syncthreads();

        double loc_du = 0.0;
        int loc_ovl = 0;
        if (!prop.skip && prop.cur_N > 0) {
            int N = prop.cur_N, mv = prop.mv, tgt = prop.tgt;
            if (mv == 0 || mv == 1) {
                double ox = s_cx[tgt], oy = s_cy[tgt], oz = s_cz[tgt];
                double oux = s_ux[tgt], ouy = s_uy[tgt], ouz = s_uz[tgt];
                double omx = s_mux[tgt], omy = s_muy[tgt], omz = s_muz[tgt];
                for (int j = tid; j < N; j += PROP_THREADS) {
                    if (j == tgt) continue;
                    int o1 = 0, o2 = 0;
                    double enew = pair_co2(cell, sp, prop.cx, prop.cy, prop.cz, prop.ux, prop.uy, prop.uz,
                                           prop.mux, prop.muy, prop.muz,
                                           s_cx[j], s_cy[j], s_cz[j], s_ux[j], s_uy[j], s_uz[j],
                                           s_mux[j], s_muy[j], s_muz[j],
                                           shifted_lj, pol_mode, rc, rc2, dsf_alpha, dsf_shift, dsf_deriv, o1);
                    if (o1) { loc_ovl = 1; break; }
                    double eold = pair_co2(cell, sp, ox, oy, oz, oux, ouy, ouz, omx, omy, omz,
                                           s_cx[j], s_cy[j], s_cz[j], s_ux[j], s_uy[j], s_uz[j],
                                           s_mux[j], s_muy[j], s_muz[j],
                                           shifted_lj, pol_mode, rc, rc2, dsf_alpha, dsf_shift, dsf_deriv, o2);
                    loc_du += (enew - eold);
                }
            } else if (mv == 2) {
                for (int j = tid; j < N; j += PROP_THREADS) {
                    int o1 = 0;
                    double enew = pair_co2(cell, sp, prop.cx, prop.cy, prop.cz, prop.ux, prop.uy, prop.uz,
                                           prop.mux, prop.muy, prop.muz,
                                           s_cx[j], s_cy[j], s_cz[j], s_ux[j], s_uy[j], s_uz[j],
                                           s_mux[j], s_muy[j], s_muz[j],
                                           shifted_lj, pol_mode, rc, rc2, dsf_alpha, dsf_shift, dsf_deriv, o1);
                    if (o1) { loc_ovl = 1; break; }
                    loc_du += enew;
                }
            } else {
                for (int j = tid; j < N; j += PROP_THREADS) {
                    if (j == tgt) continue;
                    int o2 = 0;
                    double eold = pair_co2(cell, sp, prop.cx, prop.cy, prop.cz, prop.ux, prop.uy, prop.uz,
                                           prop.mux, prop.muy, prop.muz,
                                           s_cx[j], s_cy[j], s_cz[j], s_ux[j], s_uy[j], s_uz[j],
                                           s_mux[j], s_muy[j], s_muz[j],
                                           shifted_lj, pol_mode, rc, rc2, dsf_alpha, dsf_shift, dsf_deriv, o2);
                    loc_du -= eold;
                }
            }
        }
        int lane = tid & 31, wid = tid >> 5;
        for (int s = 16; s > 0; s >>= 1) {
            loc_du += __shfl_down_sync(0xFFFFFFFFU, loc_du, s);
            loc_ovl |= __shfl_down_sync(0xFFFFFFFFU, loc_ovl, s);
        }
        if (lane == 0) { warp_u[wid] = loc_du; warp_ovl[wid] = loc_ovl; }
        __syncthreads();

        if (tid == 0) {
            double dU_gg = warp_u[0] + warp_u[1] + warp_u[2] + warp_u[3];
            int ovl = warp_ovl[0] | warp_ovl[1] | warp_ovl[2] | warp_ovl[3];
            int mv = prop.mv;
            if (mv == 0 && cur_N > 0 && !prop.fw_ovl && !ovl) {
                double dU = (prop.u_fw_new - prop.u_fw_old) + dU_gg;
                double arg = -dU / T_K;
                if ((dU <= 0.0) || (arg > -50.0 && u01(rng) < exp(arg))) {
                    int t = prop.tgt;
                    s_cx[t] = prop.cx; s_cy[t] = prop.cy; s_cz[t] = prop.cz;
                    s_mux[t] = prop.mux; s_muy[t] = prop.muy; s_muz[t] = prop.muz;
                    s_ufw[t] = prop.u_fw_new;
                    U_total += dU;
                    ++inner_acc;
                }
            } else if (mv == 1 && cur_N > 0 && !prop.fw_ovl && !ovl) {
                double dU = (prop.u_fw_new - prop.u_fw_old) + dU_gg;
                double arg = -dU / T_K;
                if ((dU <= 0.0) || (arg > -50.0 && u01(rng) < exp(arg))) {
                    int t = prop.tgt;
                    s_ux[t] = prop.ux; s_uy[t] = prop.uy; s_uz[t] = prop.uz;
                    s_mux[t] = prop.mux; s_muy[t] = prop.muy; s_muz[t] = prop.muz;
                    s_ufw[t] = prop.u_fw_new;
                    U_total += dU;
                    ++inner_acc;
                }
            } else if (mv == 2 && cur_N < max_mol && cur_N < PROP_SMEM_MAX && !prop.fw_ovl && !ovl) {
                double dU = prop.u_fw_new + dU_gg;
                double arg = -dU / T_K;
                double ratio = (beta_f_V / (double)(cur_N + 1)) * exp(arg > 50.0 ? 50.0 : arg);
                if ((arg > -50.0) && (ratio >= 1.0 || u01(rng) < ratio)) {
                    s_cx[cur_N] = prop.cx; s_cy[cur_N] = prop.cy; s_cz[cur_N] = prop.cz;
                    s_ux[cur_N] = prop.ux; s_uy[cur_N] = prop.uy; s_uz[cur_N] = prop.uz;
                    s_mux[cur_N] = prop.mux; s_muy[cur_N] = prop.muy; s_muz[cur_N] = prop.muz;
                    s_ufw[cur_N] = prop.u_fw_new;
                    ++cur_N;
                    U_total += dU;
                    ++inner_acc;
                }
            } else if (mv == 3 && cur_N > 0 && !ovl) {
                double dU = (-prop.u_fw_old) + dU_gg;
                double arg = -dU / T_K;
                double ratio = ((double)cur_N / beta_f_V) * exp(arg > 50.0 ? 50.0 : arg);
                if ((arg > -50.0) && (ratio >= 1.0 || u01(rng) < ratio)) {
                    int t = prop.tgt, last = cur_N - 1;
                    if (t != last) {
                        s_cx[t] = s_cx[last]; s_cy[t] = s_cy[last]; s_cz[t] = s_cz[last];
                        s_ux[t] = s_ux[last]; s_uy[t] = s_uy[last]; s_uz[t] = s_uz[last];
                        s_mux[t] = s_mux[last]; s_muy[t] = s_muy[last]; s_muz[t] = s_muz[last];
                        s_ufw[t] = s_ufw[last];
                    }
                    --cur_N;
                    U_total += dU;
                    ++inner_acc;
                }
            }
            prop.cur_N = cur_N;
        }
        __syncthreads();
    }

    // Exact full recomputation at end of burst (guarantees zero drift in dU_ref)
    int final_N = prop.cur_N;
    double loc_re = 0.0;
    for (int i = tid; i < final_N; i += PROP_THREADS) {
        double ulj = 0.0, uel = 0.0, up = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
        eval_species_framework_grid_unified(grid, cell, sp, 1, pol_mode, KE_K,
                                            s_cx[i], s_cy[i], s_cz[i], s_ux[i], s_uy[i], s_uz[i],
                                            ulj, uel, up, mx, my, mz);
        loc_re += (ulj + uel + up);
        s_mux[i] = mx; s_muy[i] = my; s_muz[i] = mz;
    }
    __syncthreads();
    for (int i = tid; i < final_N; i += PROP_THREADS) {
        for (int j = i + 1; j < final_N; ++j) {
            int ov = 0;
            loc_re += pair_co2(cell, sp, s_cx[i], s_cy[i], s_cz[i], s_ux[i], s_uy[i], s_uz[i],
                               s_mux[i], s_muy[i], s_muz[i],
                               s_cx[j], s_cy[j], s_cz[j], s_ux[j], s_uy[j], s_uz[j],
                               s_mux[j], s_muy[j], s_muz[j],
                               shifted_lj, pol_mode, rc, rc2, dsf_alpha, dsf_shift, dsf_deriv, ov);
        }
    }
    int lane = tid & 31, wid = tid >> 5;
    for (int s = 16; s > 0; s >>= 1) loc_re += __shfl_down_sync(0xFFFFFFFFU, loc_re, s);
    if (lane == 0) warp_u[wid] = loc_re;
    __syncthreads();
    if (tid == 0) {
        s_u_recomp = warp_u[0] + warp_u[1] + warp_u[2] + warp_u[3];
        st.d_rng[b] = rng;
        if (commit_to_acc) {
            st.d_acc_N[b] = final_N;
            st.d_acc_U[b] = s_u_recomp;
        } else {
            st.d_tri_N[b] = final_N;
            st.d_tri_U[b] = s_u_recomp;
            st.d_tri_dU[b] = s_u_recomp - U_init;
            st.d_tri_acc[b] = inner_acc;
        }
    }
    __syncthreads();

    double* ocx = commit_to_acc ? st.d_acc_cx : st.d_tri_cx;
    double* ocy = commit_to_acc ? st.d_acc_cy : st.d_tri_cy;
    double* ocz = commit_to_acc ? st.d_acc_cz : st.d_tri_cz;
    double* oux = commit_to_acc ? st.d_acc_ux : st.d_tri_ux;
    double* ouy = commit_to_acc ? st.d_acc_uy : st.d_tri_uy;
    double* ouz = commit_to_acc ? st.d_acc_uz : st.d_tri_uz;
    for (int i = tid; i < final_N; i += PROP_THREADS) {
        ocx[off + i] = s_cx[i]; ocy[off + i] = s_cy[i]; ocz[off + i] = s_cz[i];
        oux[off + i] = s_ux[i]; ouy[off + i] = s_uy[i]; ouz[off + i] = s_uz[i];
    }
}

__global__ void commit_mask_kernel(int n_boxes, int max_mol, const int* mask, ProposerStateDev st) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    if (b >= n_boxes || mask[b] == 0) return;
    int N = st.d_tri_N[b];
    if (tid == 0) {
        st.d_acc_N[b] = N;
        st.d_acc_U[b] = st.d_tri_U[b];
    }
    int off = b * max_mol;
    for (int i = tid; i < N; i += PROP_THREADS) {
        st.d_acc_cx[off + i] = st.d_tri_cx[off + i];
        st.d_acc_cy[off + i] = st.d_tri_cy[off + i];
        st.d_acc_cz[off + i] = st.d_tri_cz[off + i];
        st.d_acc_ux[off + i] = st.d_tri_ux[off + i];
        st.d_acc_uy[off + i] = st.d_tri_uy[off + i];
        st.d_acc_uz[off + i] = st.d_tri_uz[off + i];
    }
}

__global__ void eval_single_state_kernel(
    Cell cell, MixGrid3DView grid, SpeciesDef sp,
    double r_cut_gg, double dsf_alpha, int ff_mode, int pol_mode, int N,
    const double* cx, const double* cy, const double* cz,
    const double* ux, const double* uy, const double* uz,
    double* out_U, int* out_ovl) {

    int tid = threadIdx.x;
    __shared__ double s_cx[PROP_SMEM_MAX], s_cy[PROP_SMEM_MAX], s_cz[PROP_SMEM_MAX];
    __shared__ double s_ux[PROP_SMEM_MAX], s_uy[PROP_SMEM_MAX], s_uz[PROP_SMEM_MAX];
    __shared__ double s_mux[PROP_SMEM_MAX], s_muy[PROP_SMEM_MAX], s_muz[PROP_SMEM_MAX];
    __shared__ double warp_u[4];
    __shared__ int warp_ovl[4];

    double loc_u = 0.0;
    int loc_ov = 0;
    for (int i = tid; i < N; i += PROP_THREADS) {
        double x = cx[i], y = cy[i], z = cz[i], ex = ux[i], ey = uy[i], ez = uz[i];
        s_cx[i] = x; s_cy[i] = y; s_cz[i] = z;
        s_ux[i] = ex; s_uy[i] = ey; s_uz[i] = ez;
        double ulj = 0.0, uel = 0.0, up = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
        bool ov = eval_species_framework_grid_unified(grid, cell, sp, 1, pol_mode, KE_K,
                                                      x, y, z, ex, ey, ez, ulj, uel, up, mx, my, mz);
        if (ov) loc_ov = 1;
        loc_u += (ulj + uel + up);
        s_mux[i] = mx; s_muy[i] = my; s_muz[i] = mz;
    }
    __syncthreads();

    int shifted_lj = (ff_mode == 1) ? 1 : 0;
    double rc = r_cut_gg, rc2 = rc * rc;
    constexpr double INV_SQRT_PI = 0.56418958354775628695;
    double erfc_rc = erfc(dsf_alpha * rc), exp_rc = exp(-dsf_alpha * dsf_alpha * rc2);
    double dsf_shift = erfc_rc / rc;
    double dsf_deriv = erfc_rc / rc2 + (2.0 * dsf_alpha * INV_SQRT_PI) * exp_rc / rc;

    for (int i = tid; i < N; i += PROP_THREADS) {
        for (int j = i + 1; j < N; ++j) {
            int ov = 0;
            double e = pair_co2(cell, sp, s_cx[i], s_cy[i], s_cz[i], s_ux[i], s_uy[i], s_uz[i],
                                s_mux[i], s_muy[i], s_muz[i],
                                s_cx[j], s_cy[j], s_cz[j], s_ux[j], s_uy[j], s_uz[j],
                                s_mux[j], s_muy[j], s_muz[j],
                                shifted_lj, pol_mode, rc, rc2, dsf_alpha, dsf_shift, dsf_deriv, ov);
            if (ov) loc_ov = 1;
            loc_u += e;
        }
    }
    int lane = tid & 31, wid = tid >> 5;
    for (int s = 16; s > 0; s >>= 1) {
        loc_u += __shfl_down_sync(0xFFFFFFFFU, loc_u, s);
        loc_ov |= __shfl_down_sync(0xFFFFFFFFU, loc_ov, s);
    }
    if (lane == 0) { warp_u[wid] = loc_u; warp_ovl[wid] = loc_ov; }
    __syncthreads();
    if (tid == 0) {
        *out_U = warp_u[0] + warp_u[1] + warp_u[2] + warp_u[3];
        *out_ovl = warp_ovl[0] | warp_ovl[1] | warp_ovl[2] | warp_ovl[3];
    }
}

struct ProposerContext {
    Cell cell{};
    MixGrid3DView d_grid{};
    SpeciesDef sp{};
    std::vector<double> unit_frac; // size 3 * n_unit (sa, sb, sc)
    std::vector<int> unit_Z;       // size n_unit (atomic numbers)
    double fw_mass_super = 0.0;
    double T_K = 298.0;
    double pressure_bar = 0.15;
    double f_Pa = 0.0;
    double beta_f_V = 0.0;
    double delta_eps_K = 0.0;
    double beta_f_V_ref = 0.0;
    double r_cut_gg = 6.5;
    double dsf_alpha = 0.20;
    int ff_mode = 0;
    int pol_mode = 0;
    int n_boxes = 1;
    int max_mol = 128;
    ProposerStateDev dev{};
};

} // namespace

extern "C" {

void* proposer_create(
    const char* cif_path,
    int sc_a, int sc_b, int sc_c,
    double T_K, double pressure_bar,
    int ff_mode, int pol_mode,
    int nx, int ny, int nz,
    int n_boxes, int max_mol,
    uint64_t seed) {

    auto* ctx = new ProposerContext();
    ctx->T_K = T_K;
    ctx->pressure_bar = pressure_bar;
    ctx->ff_mode = ff_mode;
    ctx->pol_mode = pol_mode;
    ctx->n_boxes = std::max(1, n_boxes);
    ctx->max_mol = std::min(PROP_SMEM_MAX, std::max(16, max_mol));

    std::vector<FrameworkAtom> unit_atoms;
    CifStructureInfo info{};
    if (!load_cif_framework(cif_path, sc_a, sc_b, sc_c, ctx->cell, unit_atoms, info)) {
        delete ctx;
        return nullptr;
    }

    if (ff_mode == 1) {
        bool valid_mercado = (info.count_mg > 0);
        for (const auto& fa : unit_atoms) {
            if (fa.atom_type_id < 0 || fa.atom_type_id >= 9) {
                valid_mercado = false;
                break;
            }
        }
        if (!valid_mercado) {
            std::fprintf(stderr,
                         "Error: ff_mode=1 (Mercado 2016) only supports Mg-MOF-74 atom types (Mg1, O1-O3, C1-C4, H1); "
                         "cannot use it with '%s'. Use ff_mode=0 (UFF) instead.\n",
                         cif_path ? cif_path : "<null>");
            delete ctx;
            return nullptr;
        }
    }

    ctx->unit_frac.reserve(unit_atoms.size() * 3);
    ctx->unit_Z.reserve(unit_atoms.size());
    for (const auto& fa : unit_atoms) {
        ctx->unit_frac.push_back((double)fa.sa);
        ctx->unit_frac.push_back((double)fa.sb);
        ctx->unit_frac.push_back((double)fa.sc);
        ctx->unit_Z.push_back(get_atomic_number(fa.elem));
    }

    ctx->fw_mass_super = info.unit_mass_g_mol * (double)(sc_a * sc_b * sc_c);
    ctx->sp = (ff_mode == 1) ? make_species_co2_epm2() : make_species_co2();

    FugacityResult fug = compute_co2_fugacity_pr(T_K, pressure_bar);
    ctx->f_Pa = fug.f_Pa;
    constexpr double KB_SI = 1.380649e-23;
    ctx->beta_f_V = (ctx->f_Pa * ctx->cell.volume_super * 1.0e-30) / (KB_SI * T_K);
    ctx->delta_eps_K = 0.0;
    ctx->beta_f_V_ref = ctx->beta_f_V;
    ctx->r_cut_gg = std::min(11.0, 0.48 * ctx->cell.min_perp_width_super);

    UnifiedGridStats stats{};
    build_unified_grids_gpu(ctx->cell, unit_atoms, nx, ny, nz, 11.0, 0.30, ff_mode, ctx->d_grid, stats);

#define CUDA_CHECK_FAIL(call)                                                   \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA Error at %s:%d: %s\n",                   \
                         __FILE__, __LINE__, cudaGetErrorString(err));          \
            proposer_destroy(ctx);                                              \
            return nullptr;                                                     \
        }                                                                       \
    } while (0)

    size_t n_b = (size_t)ctx->n_boxes;
    size_t n_m = n_b * (size_t)ctx->max_mol;
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_acc_N, n_b * sizeof(int)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_acc_U, n_b * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_acc_cx, n_m * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_acc_cy, n_m * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_acc_cz, n_m * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_acc_ux, n_m * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_acc_uy, n_m * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_acc_uz, n_m * sizeof(double)));

    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_tri_N, n_b * sizeof(int)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_tri_U, n_b * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_tri_dU, n_b * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_tri_acc, n_b * sizeof(int)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_tri_cx, n_m * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_tri_cy, n_m * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_tri_cz, n_m * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_tri_ux, n_m * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_tri_uy, n_m * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_tri_uz, n_m * sizeof(double)));

    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_commit_mask, n_b * sizeof(int)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_rng, n_b * sizeof(uint64_t)));

    CUDA_CHECK_FAIL(cudaMemset(ctx->dev.d_acc_N, 0, n_b * sizeof(int)));
    CUDA_CHECK_FAIL(cudaMemset(ctx->dev.d_acc_U, 0, n_b * sizeof(double)));
    std::vector<uint64_t> seeds(n_b);
    for (int b = 0; b < ctx->n_boxes; ++b) {
        uint64_t s = seed + (uint64_t)b * 1000003ULL + 0x9e3779b97f4a7c15ULL;
        for (int k = 0; k < 8; ++k) { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; s *= 2685821657736338717ULL; }
        seeds[b] = s;
    }
    CUDA_CHECK_FAIL(cudaMemcpy(ctx->dev.d_rng, seeds.data(), n_b * sizeof(uint64_t), cudaMemcpyHostToDevice));
#undef CUDA_CHECK_FAIL
    return ctx;
}

void proposer_destroy(void* handle) {
    if (!handle) return;
    auto* ctx = static_cast<ProposerContext*>(handle);
    free_unified_grids_gpu(ctx->d_grid);
    if (ctx->dev.d_acc_N) cudaFree(ctx->dev.d_acc_N);
    if (ctx->dev.d_acc_U) cudaFree(ctx->dev.d_acc_U);
    if (ctx->dev.d_acc_cx) cudaFree(ctx->dev.d_acc_cx);
    if (ctx->dev.d_acc_cy) cudaFree(ctx->dev.d_acc_cy);
    if (ctx->dev.d_acc_cz) cudaFree(ctx->dev.d_acc_cz);
    if (ctx->dev.d_acc_ux) cudaFree(ctx->dev.d_acc_ux);
    if (ctx->dev.d_acc_uy) cudaFree(ctx->dev.d_acc_uy);
    if (ctx->dev.d_acc_uz) cudaFree(ctx->dev.d_acc_uz);
    if (ctx->dev.d_tri_N) cudaFree(ctx->dev.d_tri_N);
    if (ctx->dev.d_tri_U) cudaFree(ctx->dev.d_tri_U);
    if (ctx->dev.d_tri_dU) cudaFree(ctx->dev.d_tri_dU);
    if (ctx->dev.d_tri_acc) cudaFree(ctx->dev.d_tri_acc);
    if (ctx->dev.d_tri_cx) cudaFree(ctx->dev.d_tri_cx);
    if (ctx->dev.d_tri_cy) cudaFree(ctx->dev.d_tri_cy);
    if (ctx->dev.d_tri_cz) cudaFree(ctx->dev.d_tri_cz);
    if (ctx->dev.d_tri_ux) cudaFree(ctx->dev.d_tri_ux);
    if (ctx->dev.d_tri_uy) cudaFree(ctx->dev.d_tri_uy);
    if (ctx->dev.d_tri_uz) cudaFree(ctx->dev.d_tri_uz);
    if (ctx->dev.d_commit_mask) cudaFree(ctx->dev.d_commit_mask);
    if (ctx->dev.d_rng) cudaFree(ctx->dev.d_rng);
    delete ctx;
}

double proposer_get_framework_mass(void* handle) {
    return handle ? static_cast<ProposerContext*>(handle)->fw_mass_super : 0.0;
}

double proposer_get_fugacity_Pa(void* handle) {
    return handle ? static_cast<ProposerContext*>(handle)->f_Pa : 0.0;
}

double proposer_get_rcut_gg(void* handle) {
    return handle ? static_cast<ProposerContext*>(handle)->r_cut_gg : 0.0;
}

void proposer_get_cell_H(void* handle, double out_H9[9]) {
    if (!handle || !out_H9) return;
    auto* ctx = static_cast<ProposerContext*>(handle);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            out_H9[r * 3 + c] = ctx->cell.H[r][c];
}

int proposer_get_num_unit_atoms(void* handle) {
    return handle ? (int)static_cast<ProposerContext*>(handle)->unit_Z.size() : 0;
}

int proposer_get_unit_atoms(void* handle, double* out_frac_3n, int* out_Z_n) {
    if (!handle || !out_frac_3n || !out_Z_n) return -1;
    auto* ctx = static_cast<ProposerContext*>(handle);
    if (!ctx->unit_frac.empty()) {
        std::memcpy(out_frac_3n, ctx->unit_frac.data(), ctx->unit_frac.size() * sizeof(double));
    }
    if (!ctx->unit_Z.empty()) {
        std::memcpy(out_Z_n, ctx->unit_Z.data(), ctx->unit_Z.size() * sizeof(int));
    }
    return (int)ctx->unit_Z.size();
}

double proposer_get_min_perp_width(void* handle) {
    return handle ? static_cast<ProposerContext*>(handle)->cell.min_perp_width_super : 0.0;
}

int proposer_set_chem_pot_shift(void* handle, double delta_eps_K) {
    if (!handle) return -1;
    auto* ctx = static_cast<ProposerContext*>(handle);
    ctx->delta_eps_K = delta_eps_K;
    double exponent = std::max(-40.0, std::min(40.0, -delta_eps_K / ctx->T_K));
    ctx->beta_f_V_ref = ctx->beta_f_V * std::exp(exponent);
    return 0;
}

int proposer_equilibrate(void* handle, int n_steps) {
    if (!handle || n_steps <= 0) return 0;
    auto* ctx = static_cast<ProposerContext*>(handle);
    propose_burst_kernel<<<ctx->n_boxes, PROP_THREADS>>>(
        ctx->cell, ctx->d_grid, ctx->sp,
        ctx->T_K, ctx->beta_f_V_ref, ctx->r_cut_gg, ctx->dsf_alpha,
        ctx->ff_mode, ctx->pol_mode, ctx->max_mol, n_steps, 1, ctx->dev);
    CUDA_CHECK_RET(cudaGetLastError(), -1);
    CUDA_CHECK_RET(cudaDeviceSynchronize(), -1);
    return 0;
}

int proposer_propose_k(
    void* handle, int k_steps,
    int* out_N, double* out_U_ref_K, double* out_dU_ref_K, int* out_inner_acc,
    double* out_cx, double* out_cy, double* out_cz,
    double* out_ux, double* out_uy, double* out_uz) {

    if (!handle) return -1;
    auto* ctx = static_cast<ProposerContext*>(handle);
    propose_burst_kernel<<<ctx->n_boxes, PROP_THREADS>>>(
        ctx->cell, ctx->d_grid, ctx->sp,
        ctx->T_K, ctx->beta_f_V_ref, ctx->r_cut_gg, ctx->dsf_alpha,
        ctx->ff_mode, ctx->pol_mode, ctx->max_mol, k_steps, 0, ctx->dev);
    CUDA_CHECK_RET(cudaGetLastError(), -1);
    CUDA_CHECK_RET(cudaDeviceSynchronize(), -1);

    size_t n_b = (size_t)ctx->n_boxes;
    size_t n_m = n_b * (size_t)ctx->max_mol;
    CUDA_CHECK_RET(cudaMemcpy(out_N, ctx->dev.d_tri_N, n_b * sizeof(int), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_U_ref_K, ctx->dev.d_tri_U, n_b * sizeof(double), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_dU_ref_K, ctx->dev.d_tri_dU, n_b * sizeof(double), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_inner_acc, ctx->dev.d_tri_acc, n_b * sizeof(int), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_cx, ctx->dev.d_tri_cx, n_m * sizeof(double), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_cy, ctx->dev.d_tri_cy, n_m * sizeof(double), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_cz, ctx->dev.d_tri_cz, n_m * sizeof(double), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_ux, ctx->dev.d_tri_ux, n_m * sizeof(double), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_uy, ctx->dev.d_tri_uy, n_m * sizeof(double), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_uz, ctx->dev.d_tri_uz, n_m * sizeof(double), cudaMemcpyDeviceToHost), -1);
    return 0;
}

int proposer_commit(void* handle, const int* accept_mask) {
    if (!handle || !accept_mask) return -1;
    auto* ctx = static_cast<ProposerContext*>(handle);
    CUDA_CHECK_RET(cudaMemcpy(ctx->dev.d_commit_mask, accept_mask, (size_t)ctx->n_boxes * sizeof(int), cudaMemcpyHostToDevice), -1);
    commit_mask_kernel<<<ctx->n_boxes, PROP_THREADS>>>(ctx->n_boxes, ctx->max_mol, ctx->dev.d_commit_mask, ctx->dev);
    CUDA_CHECK_RET(cudaGetLastError(), -1);
    CUDA_CHECK_RET(cudaDeviceSynchronize(), -1);
    return 0;
}

int proposer_get_accepted(
    void* handle,
    int* out_N, double* out_U_ref_K,
    double* out_cx, double* out_cy, double* out_cz,
    double* out_ux, double* out_uy, double* out_uz) {

    if (!handle) return -1;
    auto* ctx = static_cast<ProposerContext*>(handle);
    size_t n_b = (size_t)ctx->n_boxes;
    size_t n_m = n_b * (size_t)ctx->max_mol;
    CUDA_CHECK_RET(cudaMemcpy(out_N, ctx->dev.d_acc_N, n_b * sizeof(int), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_U_ref_K, ctx->dev.d_acc_U, n_b * sizeof(double), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_cx, ctx->dev.d_acc_cx, n_m * sizeof(double), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_cy, ctx->dev.d_acc_cy, n_m * sizeof(double), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_cz, ctx->dev.d_acc_cz, n_m * sizeof(double), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_ux, ctx->dev.d_acc_ux, n_m * sizeof(double), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_uy, ctx->dev.d_acc_uy, n_m * sizeof(double), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_uz, ctx->dev.d_acc_uz, n_m * sizeof(double), cudaMemcpyDeviceToHost), -1);
    return 0;
}

double proposer_eval_energy(
    void* handle, int N,
    const double* cx, const double* cy, const double* cz,
    const double* ux, const double* uy, const double* uz,
    int* out_overlap) {

    if (!handle) return 0.0;
    if (out_overlap) *out_overlap = 0;
    if (N <= 0) return 0.0;
    auto* ctx = static_cast<ProposerContext*>(handle);
    if (N > PROP_SMEM_MAX) {
        if (out_overlap) *out_overlap = 1;
        return 1.0e9;
    }

    size_t sz = (size_t)N * sizeof(double);
    double *d_cx = nullptr, *d_cy = nullptr, *d_cz = nullptr, *d_ux = nullptr, *d_uy = nullptr, *d_uz = nullptr, *d_U = nullptr;
    int *d_ov = nullptr;
    auto cleanup = [&]() {
        if (d_cx) cudaFree(d_cx);
        if (d_cy) cudaFree(d_cy);
        if (d_cz) cudaFree(d_cz);
        if (d_ux) cudaFree(d_ux);
        if (d_uy) cudaFree(d_uy);
        if (d_uz) cudaFree(d_uz);
        if (d_U) cudaFree(d_U);
        if (d_ov) cudaFree(d_ov);
    };
#define EVAL_CUDA_CHECK(call)                                                   \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA Error at %s:%d: %s\n",                   \
                         __FILE__, __LINE__, cudaGetErrorString(err));          \
            cleanup();                                                          \
            if (out_overlap) *out_overlap = 1;                                  \
            return 1.0e9;                                                       \
        }                                                                       \
    } while (0)

    EVAL_CUDA_CHECK(cudaMalloc(&d_cx, sz));
    EVAL_CUDA_CHECK(cudaMalloc(&d_cy, sz));
    EVAL_CUDA_CHECK(cudaMalloc(&d_cz, sz));
    EVAL_CUDA_CHECK(cudaMalloc(&d_ux, sz));
    EVAL_CUDA_CHECK(cudaMalloc(&d_uy, sz));
    EVAL_CUDA_CHECK(cudaMalloc(&d_uz, sz));
    EVAL_CUDA_CHECK(cudaMalloc(&d_U, sizeof(double)));
    EVAL_CUDA_CHECK(cudaMalloc(&d_ov, sizeof(int)));
    EVAL_CUDA_CHECK(cudaMemcpy(d_cx, cx, sz, cudaMemcpyHostToDevice));
    EVAL_CUDA_CHECK(cudaMemcpy(d_cy, cy, sz, cudaMemcpyHostToDevice));
    EVAL_CUDA_CHECK(cudaMemcpy(d_cz, cz, sz, cudaMemcpyHostToDevice));
    EVAL_CUDA_CHECK(cudaMemcpy(d_ux, ux, sz, cudaMemcpyHostToDevice));
    EVAL_CUDA_CHECK(cudaMemcpy(d_uy, uy, sz, cudaMemcpyHostToDevice));
    EVAL_CUDA_CHECK(cudaMemcpy(d_uz, uz, sz, cudaMemcpyHostToDevice));

    eval_single_state_kernel<<<1, PROP_THREADS>>>(
        ctx->cell, ctx->d_grid, ctx->sp,
        ctx->r_cut_gg, ctx->dsf_alpha, ctx->ff_mode, ctx->pol_mode, N,
        d_cx, d_cy, d_cz, d_ux, d_uy, d_uz, d_U, d_ov);
    EVAL_CUDA_CHECK(cudaGetLastError());
    EVAL_CUDA_CHECK(cudaDeviceSynchronize());

    double h_U = 0.0;
    int h_ov = 0;
    EVAL_CUDA_CHECK(cudaMemcpy(&h_U, d_U, sizeof(double), cudaMemcpyDeviceToHost));
    EVAL_CUDA_CHECK(cudaMemcpy(&h_ov, d_ov, sizeof(int), cudaMemcpyDeviceToHost));
#undef EVAL_CUDA_CHECK

    cleanup();
    if (out_overlap) *out_overlap = h_ov;
    return h_U;
}

int proposer_run_direct_benchmark(
    void* handle, int eq_steps, int prod_steps, int sample_stride,
    double* out_avg_uptake_mmol_g, double* out_sem_uptake_mmol_g,
    double* out_Qst_kJ_mol, double* out_sem_Qst_kJ_mol,
    double* out_max_drift_K) {

    if (!handle) return -1;
    auto* ctx = static_cast<ProposerContext*>(handle);
    int n_b = ctx->n_boxes;
    UnifiedGcmcConfig cfg{};
    cfg.T_K = ctx->T_K;
    cfg.r_cut_A = ctx->r_cut_gg;
    cfg.ewald_alpha = 0.30;
    cfg.dsf_alpha = ctx->dsf_alpha;
    cfg.ff_mode = ctx->ff_mode;
    cfg.use_tricubic = 1;
    cfg.pol_mode = ctx->pol_mode;
    cfg.n_comp = 1;
    cfg.enable_swap = 0;
    cfg.eq_steps = eq_steps;
    cfg.prod_steps = prod_steps;
    cfg.sample_stride = sample_stride;
    cfg.recompute_stride = 25000;
    cfg.max_mol = ctx->max_mol;
    cfg.framework_mass_super_gmol = ctx->fw_mass_super;
    cfg.base_seed = 20261199ULL;

    std::vector<UnifiedBoxSpec> specs(n_b);
    for (int b = 0; b < n_b; ++b) {
        specs[b] = {ctx->pressure_bar, 1.0, 0.0, ctx->f_Pa, 0.0, b};
    }
    std::vector<UnifiedBoxResult> res;
    run_unified_gcmc_gpu(ctx->cell, ctx->d_grid, ctx->sp, make_species_n2(), cfg, specs, res);

    double sum_u = 0.0, sum_u2 = 0.0, sum_q = 0.0, sum_q2 = 0.0, md = 0.0;
    for (int b = 0; b < n_b; ++b) {
        sum_u += res[b].uptake0_mmol_g;
        sum_u2 += res[b].uptake0_mmol_g * res[b].uptake0_mmol_g;
        sum_q += res[b].Qst_kJ_mol;
        sum_q2 += res[b].Qst_kJ_mol * res[b].Qst_kJ_mol;
        if (res[b].max_energy_drift_K > md) md = res[b].max_energy_drift_K;
    }
    double mu_u = sum_u / n_b;
    double mu_q = sum_q / n_b;
    double var_u = (n_b > 1) ? std::max(0.0, (sum_u2 - n_b * mu_u * mu_u) / (n_b - 1)) : 0.0;
    double var_q = (n_b > 1) ? std::max(0.0, (sum_q2 - n_b * mu_q * mu_q) / (n_b - 1)) : 0.0;
    if (out_avg_uptake_mmol_g) *out_avg_uptake_mmol_g = mu_u;
    if (out_sem_uptake_mmol_g) *out_sem_uptake_mmol_g = std::sqrt(var_u / n_b);
    if (out_Qst_kJ_mol) *out_Qst_kJ_mol = mu_q;
    if (out_sem_Qst_kJ_mol) *out_sem_Qst_kJ_mol = std::sqrt(var_q / n_b);
    if (out_max_drift_K) *out_max_drift_K = md;
    return 0;
}

} // extern "C"
