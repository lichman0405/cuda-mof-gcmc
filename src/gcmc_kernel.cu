#include "gcmc_gpu.hpp"
#include "forcefield.hpp"

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
    // Upper 53 bits mapped to [0, 1)
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

// Rodrigues rotation around uniform random unit axis k by angle theta in [-max_angle, +max_angle]
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

__device__ inline double pair_energy_co2(
    const Cell& cell,
    double c1x, double c1y, double c1z, double u1x, double u1y, double u1z,
    double mu1x, double mu1y, double mu1z,
    double c2x, double c2y, double c2z, double u2x, double u2y, double u2z,
    double mu2x, double mu2y, double mu2z,
    int pol_mode, double inv_alpha_cbrt,
    double r_cut, double r_cut2, double dsf_alpha, double dsf_shift, double dsf_deriv,
    int& overlap_flag) {

    double dm_x, dm_y, dm_z, dm_r2;
    cell.mic_delta_super(c2x - c1x, c2y - c1y, c2z - c1z, dm_x, dm_y, dm_z, dm_r2);

    // Quick COM skip: no site pair can be closer than r_cut if COM separation > r_cut + 2*bond_co
    double skip_r = r_cut + 2.0 * trappe_co2::BOND_CO_A;
    if (dm_r2 > skip_r * skip_r) return 0.0;

    const double off1_scale[3] = {0.0, trappe_co2::BOND_CO_A, -trappe_co2::BOND_CO_A};
    const double q_arr[3] = {trappe_co2::Q_C, trappe_co2::Q_O, trappe_co2::Q_O};
    const double eps_arr[3] = {trappe_co2::EPS_C_K, trappe_co2::EPS_O_K, trappe_co2::EPS_O_K};
    const double sig_arr[3] = {trappe_co2::SIG_C_A, trappe_co2::SIG_O_A, trappe_co2::SIG_O_A};

    double energy = 0.0;
    for (int a = 0; a < 3; ++a) {
        double ax = off1_scale[a] * u1x;
        double ay = off1_scale[a] * u1y;
        double az = off1_scale[a] * u1z;
        for (int b = 0; b < 3; ++b) {
            double bx = off1_scale[b] * u2x;
            double by = off1_scale[b] * u2y;
            double bz = off1_scale[b] * u2z;

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

            // TraPPE LJ
            double sig_ab = 0.5 * (sig_arr[a] + sig_arr[b]);
            double eps_ab = sqrt(eps_arr[a] * eps_arr[b]);
            double sr2 = (sig_ab * sig_ab) * inv_r2;
            double sr6 = sr2 * sr2 * sr2;
            energy += 4.0 * eps_ab * (sr6 * sr6 - sr6);

            // DSF Coulomb
            double erfc_ar = erfc(dsf_alpha * r);
            double u_coul = KE_K * q_arr[a] * q_arr[b] *
                            (erfc_ar * inv_r - dsf_shift + dsf_deriv * (r - r_cut));
            energy += u_coul;
        }
    }

    // Thole-damped induced-dipole - induced-dipole interaction between framework-induced dipoles
    if (pol_mode != POL_NONE && dm_r2 < r_cut2 && dm_r2 >= 0.25) {
        double r_m = sqrt(dm_r2);
        double inv_r = 1.0 / r_m;
        double inv_r3 = inv_r / dm_r2;
        double inv_r5 = inv_r3 / dm_r2;
        double u_thole = r_m * inv_alpha_cbrt;
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

struct SharedProposal {
    int move_type; // 0=trans, 1=rot, 2=ins, 3=del
    int target_idx;
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

__global__ void gcmc_batched_kernel(
    Cell cell,
    Grid3DView grid,
    BatchedGcmcConfig cfg,
    const BoxSpec* __restrict__ specs,
    BoxResult* __restrict__ results,
    unsigned long long* __restrict__ hist_out,
    int hist_na, int hist_nb) {

    int box_id = blockIdx.x;
    int tid = threadIdx.x;
    BoxSpec spec = specs[box_id];

    // Molecule positions & framework-induced dipoles stored in shared memory
    __shared__ float s_cx[SMEM_MAX_MOL];
    __shared__ float s_cy[SMEM_MAX_MOL];
    __shared__ float s_cz[SMEM_MAX_MOL];
    __shared__ float s_ux[SMEM_MAX_MOL];
    __shared__ float s_uy[SMEM_MAX_MOL];
    __shared__ float s_uz[SMEM_MAX_MOL];
    __shared__ float s_mux[SMEM_MAX_MOL];
    __shared__ float s_muy[SMEM_MAX_MOL];
    __shared__ float s_muz[SMEM_MAX_MOL];
    __shared__ float s_ufw[SMEM_MAX_MOL];  // cached total framework energy per molecule
    __shared__ float s_upol[SMEM_MAX_MOL]; // cached framework polarization energy per molecule

    __shared__ SharedProposal prop;
    __shared__ double warp_u[4];
    __shared__ int warp_ovl[4];

    // Per-box persistent state managed by thread 0
    uint64_t rng = cfg.base_seed + (uint64_t)box_id * 1000003ULL + 0x9e3779b97f4a7c15ULL;
    // Warm up RNG
    for (int k = 0; k < 8; ++k) xorshift64star(rng);

    int cur_N = 0;
    double U_total = 0.0;
    double U_pol_total = 0.0;
    double delta_trans = 1.2; // Angstrom, adapted during equilibration
    double max_angle = 0.65;  // radians

    // Precompute beta * f * V_super (dimensionless)
    constexpr double KB_SI = 1.380649e-23;
    double beta_f_V = (spec.fugacity_Pa * (cell.volume_super * 1.0e-30)) / (KB_SI * cfg.T_K);

    // Precompute DSF shift & Thole constants
    double rc = cfg.r_cut_A;
    double rc2 = rc * rc;
    double a_dsf = cfg.dsf_alpha;
    constexpr double INV_SQRT_PI = 0.56418958354775628695;
    double erfc_rc = erfc(a_dsf * rc);
    double exp_rc = exp(-a_dsf * a_dsf * rc2);
    double dsf_shift = erfc_rc / rc;
    double dsf_deriv = erfc_rc / rc2 + (2.0 * a_dsf * INV_SQRT_PI) * exp_rc / rc;
    double inv_alpha_cbrt = 1.0 / cbrt(trappe_co2::ALPHA_ISO_A3);

    // Stats accumulators (thread 0)
    long long att_t = 0, acc_t = 0;
    long long att_r = 0, acc_r = 0;
    long long att_i = 0, acc_i = 0;
    long long att_d = 0, acc_d = 0;
    long long win_att_t = 0, win_acc_t = 0;

    long long n_samples = 0;
    double sum_N = 0.0, sum_N2 = 0.0, sum_U = 0.0, sum_UN = 0.0, sum_Upol = 0.0;

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

            double r_mv = rng_u01(rng);
            int mv = (int)(r_mv * 4.0);
            if (mv > 3) mv = 3;
            prop.move_type = mv;

            if (mv == 0 || mv == 1) {
                if (cur_N == 0) {
                    prop.skip_gg = 1;
                } else {
                    int idx = (int)(rng_u01(rng) * (double)cur_N);
                    if (idx >= cur_N) idx = cur_N - 1;
                    prop.target_idx = idx;
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

                    double ulj = 0.0, uel = 0.0, upol = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
                    bool ovl = eval_co2_framework_pol_grid(
                        grid, cell, prop.cx, prop.cy, prop.cz, prop.ux, prop.uy, prop.uz,
                        trappe_co2::BOND_CO_A, trappe_co2::Q_C, trappe_co2::Q_O,
                        cfg.pol_mode, trappe_co2::ALPHA_PAR_A3, trappe_co2::ALPHA_PERP_A3,
                        trappe_co2::W_POL_C, trappe_co2::W_POL_O, KE_K, cfg.pol_scale,
                        ulj, uel, upol, mx, my, mz);
                    if (ovl) {
                        prop.fw_overlap = 1;
                        prop.skip_gg = 1;
                    } else {
                        prop.u_fw_new = ulj + uel + upol;
                        prop.u_pol_new = upol;
                        prop.mux = mx; prop.muy = my; prop.muz = mz;
                    }
                }
            } else if (mv == 2) { // Insertion
                if (cur_N >= cfg.max_mol || cur_N >= SMEM_MAX_MOL) {
                    prop.skip_gg = 1;
                } else {
                    prop.target_idx = -1;
                    double sa = rng_u01(rng);
                    double sb = rng_u01(rng);
                    double sc = rng_u01(rng);
                    cell.super_frac_to_cart(sa, sb, sc, prop.cx, prop.cy, prop.cz);
                    random_unit_sphere(rng, prop.ux, prop.uy, prop.uz);

                    double ulj = 0.0, uel = 0.0, upol = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
                    bool ovl = eval_co2_framework_pol_grid(
                        grid, cell, prop.cx, prop.cy, prop.cz, prop.ux, prop.uy, prop.uz,
                        trappe_co2::BOND_CO_A, trappe_co2::Q_C, trappe_co2::Q_O,
                        cfg.pol_mode, trappe_co2::ALPHA_PAR_A3, trappe_co2::ALPHA_PERP_A3,
                        trappe_co2::W_POL_C, trappe_co2::W_POL_O, KE_K, cfg.pol_scale,
                        ulj, uel, upol, mx, my, mz);
                    if (ovl) {
                        prop.fw_overlap = 1;
                        prop.skip_gg = 1;
                    } else {
                        prop.u_fw_new = ulj + uel + upol;
                        prop.u_pol_new = upol;
                        prop.mux = mx; prop.muy = my; prop.muz = mz;
                    }
                }
            } else { // Deletion
                if (cur_N == 0) {
                    prop.skip_gg = 1;
                } else {
                    int idx = (int)(rng_u01(rng) * (double)cur_N);
                    if (idx >= cur_N) idx = cur_N - 1;
                    prop.target_idx = idx;
                    prop.u_fw_old = (double)s_ufw[idx];
                    prop.u_pol_old = (double)s_upol[idx];
                    prop.cx = (double)s_cx[idx];
                    prop.cy = (double)s_cy[idx];
                    prop.cz = (double)s_cz[idx];
                    prop.ux = (double)s_ux[idx];
                    prop.uy = (double)s_uy[idx];
                    prop.uz = (double)s_uz[idx];
                    prop.mux = (double)s_mux[idx];
                    prop.muy = (double)s_muy[idx];
                    prop.muz = (double)s_muz[idx];
                }
            }
        }
        __syncthreads();

        // Parallel guest-guest energy evaluation across all 128 threads
        double local_du_gg = 0.0;
        int local_ovl = 0;

        if (!prop.skip_gg && prop.cur_N > 0) {
            int N = prop.cur_N;
            int mv = prop.move_type;
            int tgt = prop.target_idx;

            if (mv == 0 || mv == 1) {
                double ox = (double)s_cx[tgt], oy = (double)s_cy[tgt], oz = (double)s_cz[tgt];
                double oux = (double)s_ux[tgt], ouy = (double)s_uy[tgt], ouz = (double)s_uz[tgt];
                double omux = (double)s_mux[tgt], omuy = (double)s_muy[tgt], omuz = (double)s_muz[tgt];
                for (int j = tid; j < N; j += BLOCK_THREADS) {
                    if (j == tgt) continue;
                    double jx = (double)s_cx[j], jy = (double)s_cy[j], jz = (double)s_cz[j];
                    double jux = (double)s_ux[j], juy = (double)s_uy[j], juz = (double)s_uz[j];
                    double jmux = (double)s_mux[j], jmuy = (double)s_muy[j], jmuz = (double)s_muz[j];

                    int ov_new = 0, ov_old = 0;
                    double enew = pair_energy_co2(
                        cell, prop.cx, prop.cy, prop.cz, prop.ux, prop.uy, prop.uz,
                        prop.mux, prop.muy, prop.muz,
                        jx, jy, jz, jux, juy, juz, jmux, jmuy, jmuz,
                        cfg.pol_mode, inv_alpha_cbrt,
                        rc, rc2, a_dsf, dsf_shift, dsf_deriv, ov_new);
                    if (ov_new) {
                        local_ovl = 1;
                        break;
                    }
                    double eold = pair_energy_co2(
                        cell, ox, oy, oz, oux, ouy, ouz, omux, omuy, omuz,
                        jx, jy, jz, jux, juy, juz, jmux, jmuy, jmuz,
                        cfg.pol_mode, inv_alpha_cbrt,
                        rc, rc2, a_dsf, dsf_shift, dsf_deriv, ov_old);
                    local_du_gg += (enew - eold);
                }
            } else if (mv == 2) { // Insertion: new mol vs existing 0..N-1
                for (int j = tid; j < N; j += BLOCK_THREADS) {
                    int ov_new = 0;
                    double enew = pair_energy_co2(
                        cell, prop.cx, prop.cy, prop.cz, prop.ux, prop.uy, prop.uz,
                        prop.mux, prop.muy, prop.muz,
                        (double)s_cx[j], (double)s_cy[j], (double)s_cz[j],
                        (double)s_ux[j], (double)s_uy[j], (double)s_uz[j],
                        (double)s_mux[j], (double)s_muy[j], (double)s_muz[j],
                        cfg.pol_mode, inv_alpha_cbrt,
                        rc, rc2, a_dsf, dsf_shift, dsf_deriv, ov_new);
                    if (ov_new) {
                        local_ovl = 1;
                        break;
                    }
                    local_du_gg += enew;
                }
            } else { // Deletion: removing existing tgt vs others
                for (int j = tid; j < N; j += BLOCK_THREADS) {
                    if (j == tgt) continue;
                    int ov_old = 0;
                    double eold = pair_energy_co2(
                        cell, prop.cx, prop.cy, prop.cz, prop.ux, prop.uy, prop.uz,
                        prop.mux, prop.muy, prop.muz,
                        (double)s_cx[j], (double)s_cy[j], (double)s_cz[j],
                        (double)s_ux[j], (double)s_uy[j], (double)s_uz[j],
                        (double)s_mux[j], (double)s_muy[j], (double)s_muz[j],
                        cfg.pol_mode, inv_alpha_cbrt,
                        rc, rc2, a_dsf, dsf_shift, dsf_deriv, ov_old);
                    local_du_gg -= eold;
                }
            }
        }

        // Warp + Block reduction
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
                if (!prop.skip_gg && !gg_ovl) {
                    double dU = (prop.u_fw_new - prop.u_fw_old) + dU_gg;
                    double arg = -dU / cfg.T_K;
                    bool accept = (dU <= 0.0) || (arg > -50.0 && rng_u01(rng) < exp(arg));
                    if (accept) {
                        int tgt = prop.target_idx;
                        s_cx[tgt] = (float)prop.cx;
                        s_cy[tgt] = (float)prop.cy;
                        s_cz[tgt] = (float)prop.cz;
                        s_mux[tgt] = (float)prop.mux;
                        s_muy[tgt] = (float)prop.muy;
                        s_muz[tgt] = (float)prop.muz;
                        s_ufw[tgt] = (float)prop.u_fw_new;
                        s_upol[tgt] = (float)prop.u_pol_new;
                        U_total += dU;
                        U_pol_total += (prop.u_pol_new - prop.u_pol_old);
                        if (in_prod) ++acc_t; else ++win_acc_t;
                    }
                }
            } else if (mv == 1) {
                if (in_prod) ++att_r;
                if (!prop.skip_gg && !gg_ovl) {
                    double dU = (prop.u_fw_new - prop.u_fw_old) + dU_gg;
                    double arg = -dU / cfg.T_K;
                    bool accept = (dU <= 0.0) || (arg > -50.0 && rng_u01(rng) < exp(arg));
                    if (accept) {
                        int tgt = prop.target_idx;
                        s_ux[tgt] = (float)prop.ux;
                        s_uy[tgt] = (float)prop.uy;
                        s_uz[tgt] = (float)prop.uz;
                        s_mux[tgt] = (float)prop.mux;
                        s_muy[tgt] = (float)prop.muy;
                        s_muz[tgt] = (float)prop.muz;
                        s_ufw[tgt] = (float)prop.u_fw_new;
                        s_upol[tgt] = (float)prop.u_pol_new;
                        U_total += dU;
                        U_pol_total += (prop.u_pol_new - prop.u_pol_old);
                        if (in_prod) ++acc_r;
                    }
                }
            } else if (mv == 2) {
                if (in_prod) ++att_i;
                if (!prop.skip_gg && !gg_ovl) {
                    double dU = prop.u_fw_new + dU_gg;
                    double arg = -dU / cfg.T_K;
                    double ratio = (beta_f_V / (double)(cur_N + 1)) * exp(arg > 50.0 ? 50.0 : arg);
                    bool accept = (arg > -50.0) && (ratio >= 1.0 || rng_u01(rng) < ratio);
                    if (accept) {
                        s_cx[cur_N] = (float)prop.cx;
                        s_cy[cur_N] = (float)prop.cy;
                        s_cz[cur_N] = (float)prop.cz;
                        s_ux[cur_N] = (float)prop.ux;
                        s_uy[cur_N] = (float)prop.uy;
                        s_uz[cur_N] = (float)prop.uz;
                        s_mux[cur_N] = (float)prop.mux;
                        s_muy[cur_N] = (float)prop.muy;
                        s_muz[cur_N] = (float)prop.muz;
                        s_ufw[cur_N] = (float)prop.u_fw_new;
                        s_upol[cur_N] = (float)prop.u_pol_new;
                        ++cur_N;
                        U_total += dU;
                        U_pol_total += prop.u_pol_new;
                        if (in_prod) ++acc_i;
                    }
                }
            } else { // Deletion
                if (in_prod) ++att_d;
                if (!prop.skip_gg && cur_N > 0) {
                    double dU = (-prop.u_fw_old) + dU_gg;
                    double arg = -dU / cfg.T_K;
                    double ratio = ((double)cur_N / beta_f_V) * exp(arg > 50.0 ? 50.0 : arg);
                    bool accept = (arg > -50.0) && (ratio >= 1.0 || rng_u01(rng) < ratio);
                    if (accept) {
                        int tgt = prop.target_idx;
                        int last = cur_N - 1;
                        if (tgt != last) {
                            s_cx[tgt] = s_cx[last];
                            s_cy[tgt] = s_cy[last];
                            s_cz[tgt] = s_cz[last];
                            s_ux[tgt] = s_ux[last];
                            s_uy[tgt] = s_uy[last];
                            s_uz[tgt] = s_uz[last];
                            s_mux[tgt] = s_mux[last];
                            s_muy[tgt] = s_muy[last];
                            s_muz[tgt] = s_muz[last];
                            s_ufw[tgt] = s_ufw[last];
                            s_upol[tgt] = s_upol[last];
                        }
                        --cur_N;
                        U_total += dU;
                        U_pol_total -= prop.u_pol_old;
                        if (in_prod) ++acc_d;
                    }
                }
            }

            // Adaptive step adjustment during equilibration
            if (!in_prod && ((step + 1) % 2000 == 0) && win_att_t > 20) {
                double rate = (double)win_acc_t / (double)win_att_t;
                if (rate < 0.30) delta_trans *= 0.90;
                else if (rate > 0.50) delta_trans *= 1.10;
                if (delta_trans < 0.15) delta_trans = 0.15;
                if (delta_trans > 4.0) delta_trans = 4.0;
                win_att_t = 0;
                win_acc_t = 0;
            }

            // Production sampling
            if (in_prod && ((step - cfg.eq_steps) % cfg.sample_stride == 0)) {
                ++n_samples;
                double dN = (double)cur_N;
                sum_N += dN;
                sum_N2 += dN * dN;
                sum_U += U_total;
                sum_UN += U_total * dN;
                sum_Upol += U_pol_total;

                if (spec.record_density_hist && hist_out != nullptr && cur_N > 0) {
                    // Record every 5th sample across existing molecules for smooth 2D pore density
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
                            atomicAdd(&hist_out[ib * hist_na + ia], 1ULL);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        double ns = (n_samples > 0) ? (double)n_samples : 1.0;
        double mean_N = sum_N / ns;
        double mean_N2 = sum_N2 / ns;
        double mean_U = sum_U / ns;
        double mean_UN = sum_UN / ns;
        double mean_Upol = sum_Upol / ns;

        double var_N = mean_N2 - mean_N * mean_N;
        if (var_N < 0.0) var_N = 0.0;
        double std_N = sqrt(var_N);

        double qst_kj_mol = 0.0;
        if (var_N > 1.0e-9) {
            double cov_UN = mean_UN - mean_U * mean_N;
            qst_kj_mol = (cfg.T_K - cov_UN / var_N) * K_TO_KJMOL;
        }

        BoxResult r{};
        r.pressure_bar = spec.pressure_bar;
        r.fugacity_Pa = spec.fugacity_Pa;
        r.replica_id = spec.replica_id;
        r.avg_N = mean_N;
        r.std_N = std_N;
        r.uptake_mmol_g = mean_N * 1000.0 / cfg.framework_mass_super_gmol;
        r.avg_U_K = mean_U;
        r.avg_U_pol_per_mol_kJ = (mean_N > 1.0e-6) ? ((mean_Upol / mean_N) * K_TO_KJMOL) : 0.0;
        r.Qst_kJ_mol = qst_kj_mol;
        r.acc_trans = (att_t > 0) ? (double)acc_t / (double)att_t : 0.0;
        r.acc_rot   = (att_r > 0) ? (double)acc_r / (double)att_r : 0.0;
        r.acc_ins   = (att_i > 0) ? (double)acc_i / (double)att_i : 0.0;
        r.acc_del   = (att_d > 0) ? (double)acc_d / (double)att_d : 0.0;
        r.final_N   = cur_N;
        results[box_id] = r;
    }
}

} // namespace

double run_batched_gcmc_gpu(
    const Cell& cell,
    const Grid3DView& d_grid,
    const BatchedGcmcConfig& cfg,
    const std::vector<BoxSpec>& boxes,
    std::vector<BoxResult>& results_out,
    std::vector<double>& density_hist_out,
    int hist_na, int hist_nb) {

    int num_boxes = (int)boxes.size();
    results_out.resize(num_boxes);

    BoxSpec* d_specs = nullptr;
    BoxResult* d_results = nullptr;
    unsigned long long* d_hist = nullptr;

    CUDA_CHECK(cudaMalloc(&d_specs, num_boxes * sizeof(BoxSpec)));
    CUDA_CHECK(cudaMalloc(&d_results, num_boxes * sizeof(BoxResult)));
    CUDA_CHECK(cudaMemcpy(d_specs, boxes.data(),
                          num_boxes * sizeof(BoxSpec), cudaMemcpyHostToDevice));

    size_t hist_size = (size_t)hist_na * (size_t)hist_nb;
    CUDA_CHECK(cudaMalloc(&d_hist, hist_size * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_hist, 0, hist_size * sizeof(unsigned long long)));

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));

    gcmc_batched_kernel<<<num_boxes, BLOCK_THREADS>>>(
        cell, d_grid, cfg, d_specs, d_results, d_hist, hist_na, hist_nb);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));

    CUDA_CHECK(cudaMemcpy(results_out.data(), d_results,
                          num_boxes * sizeof(BoxResult), cudaMemcpyDeviceToHost));

    std::vector<unsigned long long> h_hist(hist_size);
    CUDA_CHECK(cudaMemcpy(h_hist.data(), d_hist,
                          hist_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    density_hist_out.resize(hist_size);
    for (size_t i = 0; i < hist_size; ++i) {
        density_hist_out[i] = (double)h_hist[i];
    }

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaFree(d_specs));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_hist));

    return (double)ms;
}

} // namespace mofgcmc
