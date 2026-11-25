#include "proposer_api.h"
#include "cell.hpp"
#include "cif_loader.hpp"
#include "forcefield.hpp"
#include "grid3d.hpp"
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
constexpr int PROP_MAX_SITES = 12;
constexpr int PROP_MAX_LJ = 6;

struct RigidGuestDef {
    int n_sites;
    int n_lj_types;
    int is_isotropic;
    int has_charge;
    double max_radius_A;
    double lx[PROP_MAX_SITES];
    double ly[PROP_MAX_SITES];
    double lz[PROP_MAX_SITES];
    double q[PROP_MAX_SITES];
    double eps_K[PROP_MAX_SITES];
    double sigma_A[PROP_MAX_SITES];
    double w_pol[PROP_MAX_SITES];
    int lj_type[PROP_MAX_SITES];
    double lj_eps_K[PROP_MAX_LJ];
    double lj_sigma_A[PROP_MAX_LJ];
    double alpha_par;
    double alpha_perp;
    double alpha_iso;
    double Tc_K;
    double Pc_bar;
    double omega;
    double gg_q_scale;
    double gg_lj_scale;
    int gg_lut_n;
    double gg_lut_rmin;
    double gg_lut_rmax;
    double gg_lut_inv_dr;
    float gg_lut_K[128];
};

struct ProposerGridView {
    int nx, ny, nz;
    int n_lj;
    int has_phi;
    int has_efield;
    const float* U_lj[PROP_MAX_LJ];
    const float* Phi_elec;
    const float* Ex;
    const float* Ey;
    const float* Ez;
};

struct ProposerKVec {
    double kx, ky, kz;
    double W_C;
    double W_S;
};

__host__ __device__ inline void quat_rotate(
    double qw, double qx, double qy, double qz,
    double vx, double vy, double vz,
    double& ox, double& oy, double& oz) {
    double tx = 2.0 * (qy * vz - qz * vy);
    double ty = 2.0 * (qz * vx - qx * vz);
    double tz = 2.0 * (qx * vy - qy * vx);
    ox = vx + qw * tx + (qy * tz - qz * ty);
    oy = vy + qw * ty + (qz * tx - qx * tz);
    oz = vz + qw * tz + (qx * ty - qy * tx);
}

__host__ __device__ inline void quat_to_xaxis(
    double qw, double qx, double qy, double qz,
    double& ux, double& uy, double& uz) {
    ux = 1.0 - 2.0 * (qy * qy + qz * qz);
    uy = 2.0 * (qx * qy + qw * qz);
    uz = 2.0 * (qx * qz - qw * qy);
}

__host__ __device__ inline void xaxis_to_quat(
    double ux, double uy, double uz,
    double& qw, double& qx, double& qy, double& qz) {
    double n = sqrt(ux * ux + uy * uy + uz * uz);
    if (n < 1.0e-12) {
        qw = 1.0; qx = qy = qz = 0.0;
        return;
    }
    double inv_n = 1.0 / n;
    ux *= inv_n; uy *= inv_n; uz *= inv_n;
    if (ux <= -0.999999) {
        qw = 0.0; qx = 0.0; qy = 1.0; qz = 0.0;
        return;
    }
    double w = 1.0 + ux;
    double y = -uz;
    double z = uy;
    double qn = 1.0 / sqrt(w * w + y * y + z * z);
    qw = w * qn;
    qx = 0.0;
    qy = y * qn;
    qz = z * qn;
}

__device__ inline uint64_t xs64(uint64_t& s) {
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 2685821657736338717ULL;
}

__device__ inline double u01(uint64_t& s) {
    return (double)(xs64(s) >> 11) * (1.0 / 9007199254740992.0);
}

__device__ inline uint64_t splitmix64_next(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__device__ inline double sm_u01(uint64_t& s) {
    return (double)(splitmix64_next(s) >> 11) * (1.0 / 9007199254740992.0);
}

__device__ inline void rand_uniform_quat_sm(
    uint64_t& rng, double& qw, double& qx, double& qy, double& qz) {
    constexpr double TWO_PI = 6.28318530717958647692;
    double u1 = sm_u01(rng);
    double u2 = TWO_PI * sm_u01(rng);
    double u3 = TWO_PI * sm_u01(rng);
    double s1 = sqrt(1.0 - u1);
    double s2 = sqrt(u1);
    double su2, cu2, su3, cu3;
    sincos(u2, &su2, &cu2);
    sincos(u3, &su3, &cu3);
    qw = s2 * cu3;
    qx = s1 * su2;
    qy = s1 * cu2;
    qz = s2 * su3;
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

__device__ inline void rand_uniform_quat(
    uint64_t& rng, double& qw, double& qx, double& qy, double& qz) {
    constexpr double TWO_PI = 6.28318530717958647692;
    double u1 = u01(rng);
    double u2 = TWO_PI * u01(rng);
    double u3 = TWO_PI * u01(rng);
    double s1 = sqrt(1.0 - u1);
    double s2 = sqrt(u1);
    double su2, cu2, su3, cu3;
    sincos(u2, &su2, &cu2);
    sincos(u3, &su3, &cu3);
    qw = s2 * cu3;
    qx = s1 * su2;
    qy = s1 * cu2;
    qz = s2 * su3;
}

__device__ inline void rot_quat_small(
    uint64_t& rng, double max_angle,
    double qw, double qx, double qy, double qz,
    double& nqw, double& nqx, double& nqy, double& nqz) {
    double kx, ky, kz;
    rand_unit_vec(rng, kx, ky, kz);
    double half = 0.5 * (2.0 * u01(rng) - 1.0) * max_angle;
    double sh, ch;
    sincos(half, &sh, &ch);
    double dqw = ch, dqx = sh * kx, dqy = sh * ky, dqz = sh * kz;
    nqw = dqw * qw - dqx * qx - dqy * qy - dqz * qz;
    nqx = dqw * qx + dqx * qw + dqy * qz - dqz * qy;
    nqy = dqw * qy - dqx * qz + dqy * qw + dqz * qx;
    nqz = dqw * qz + dqx * qy - dqy * qx + dqz * qw;
    double inv = rsqrt(nqw * nqw + nqx * nqx + nqy * nqy + nqz * nqz);
    nqw *= inv; nqx *= inv; nqy *= inv; nqz *= inv;
}

__device__ inline bool eval_rigid_guest_framework_grid(
    const ProposerGridView& g, const Cell& cell,
    const RigidGuestDef& guest, int pol_mode, double ke_k,
    double cx, double cy, double cz,
    double qw, double qx, double qy, double qz,
    double& u_lj_out, double& u_elec_out, double& u_pol_out,
    double& mux_out, double& muy_out, double& muz_out) {

    double sa[PROP_MAX_SITES], sb[PROP_MAX_SITES], sc[PROP_MAX_SITES];
    double u_lj = 0.0;

    // Pass 1: VDW overlap check across all sites with LJ grids
    for (int i = 0; i < guest.n_sites; ++i) {
        double dx, dy, dz;
        quat_rotate(qw, qx, qy, qz, guest.lx[i], guest.ly[i], guest.lz[i], dx, dy, dz);
        cell.cart_to_unit_frac(cx + dx, cy + dy, cz + dz, sa[i], sb[i], sc[i]);

        int lg = guest.lj_type[i];
        if (lg >= 0 && lg < g.n_lj && g.U_lj[lg] != nullptr) {
            double ulj = tricubic_periodic(g.U_lj[lg], g.nx, g.ny, g.nz, sa[i], sb[i], sc[i], true, 5.0e4f);
            if (ulj > 1.0e5) {
                u_lj_out = 1.0e6;
                u_elec_out = 0.0;
                u_pol_out = 0.0;
                mux_out = muy_out = muz_out = 0.0;
                return true;
            }
            u_lj += ulj;
        }
    }

    // Pass 2: Electrostatics + optional anisotropic polarization
    double u_el = 0.0, u_pol = 0.0;
    double mux = 0.0, muy = 0.0, muz = 0.0;
    double ux = 1.0, uy = 0.0, uz = 0.0;
    if (pol_mode != 0 && g.has_efield) {
        quat_to_xaxis(qw, qx, qy, qz, ux, uy, uz);
    }

    for (int i = 0; i < guest.n_sites; ++i) {
        if (g.has_phi && guest.q[i] != 0.0 && g.Phi_elec != nullptr) {
            double phi = tricubic_periodic(g.Phi_elec, g.nx, g.ny, g.nz, sa[i], sb[i], sc[i], false, 5.0e5f);
            u_el += guest.q[i] * phi;
        }

        double wa = 0.0;
        if (pol_mode == 1 && i == 0) wa = 1.0;
        else if (pol_mode == 2) wa = guest.w_pol[i];

        if (wa > 0.0 && g.has_efield && g.Ex != nullptr && g.Ey != nullptr && g.Ez != nullptr) {
            double ex = tricubic_periodic(g.Ex, g.nx, g.ny, g.nz, sa[i], sb[i], sc[i], false, 5.0e5f);
            double ey = tricubic_periodic(g.Ey, g.nx, g.ny, g.nz, sa[i], sb[i], sc[i], false, 5.0e5f);
            double ez = tricubic_periodic(g.Ez, g.nx, g.ny, g.nz, sa[i], sb[i], sc[i], false, 5.0e5f);

            double e2 = ex * ex + ey * ey + ez * ez;
            double e_par = ex * ux + ey * uy + ez * uz;
            double e_perp2 = e2 - e_par * e_par;
            if (e_perp2 < 0.0) e_perp2 = 0.0;

            double e_perpx = ex - e_par * ux;
            double e_perpy = ey - e_par * uy;
            double e_perpz = ez - e_par * uz;

            double inv_ke = 1.0 / ke_k;
            u_pol += -0.5 * inv_ke * wa * (guest.alpha_par * e_par * e_par + guest.alpha_perp * e_perp2);
            mux += (wa * inv_ke) * (guest.alpha_par * e_par * ux + guest.alpha_perp * e_perpx);
            muy += (wa * inv_ke) * (guest.alpha_par * e_par * uy + guest.alpha_perp * e_perpy);
            muz += (wa * inv_ke) * (guest.alpha_par * e_par * uz + guest.alpha_perp * e_perpz);
        }
    }

    u_lj_out = u_lj;
    u_elec_out = u_el;
    u_pol_out = u_pol;
    mux_out = mux;
    muy_out = muy;
    muz_out = muz;
    return false;
}

__device__ inline double pair_rigid_guest(
    const Cell& cell, const RigidGuestDef& g,
    double c1x, double c1y, double c1z, double q1w, double q1x, double q1y, double q1z,
    double m1x, double m1y, double m1z,
    double c2x, double c2y, double c2z, double q2w, double q2x, double q2y, double q2z,
    double m2x, double m2y, double m2z,
    int shifted_lj, int pol_mode,
    double r_cut, double r_cut2, double dsf_alpha, double dsf_shift, double dsf_deriv,
    int& ovl) {

    double dmx, dmy, dmz, r2c;
    cell.mic_delta_super(c2x - c1x, c2y - c1y, c2z - c1z, dmx, dmy, dmz, r2c);
    double skip = r_cut + 2.0 * g.max_radius_A;
    if (r2c > skip * skip) return 0.0;

    double a_off[PROP_MAX_SITES][3];
    double b_off[PROP_MAX_SITES][3];
    for (int i = 0; i < g.n_sites; ++i) {
        if (g.is_isotropic) {
            a_off[i][0] = g.lx[i]; a_off[i][1] = g.ly[i]; a_off[i][2] = g.lz[i];
            b_off[i][0] = g.lx[i]; b_off[i][1] = g.ly[i]; b_off[i][2] = g.lz[i];
        } else {
            quat_rotate(q1w, q1x, q1y, q1z, g.lx[i], g.ly[i], g.lz[i], a_off[i][0], a_off[i][1], a_off[i][2]);
            quat_rotate(q2w, q2x, q2y, q2z, g.lx[i], g.ly[i], g.lz[i], b_off[i][0], b_off[i][1], b_off[i][2]);
        }
    }

    double e = 0.0;
    for (int a = 0; a < g.n_sites; ++a) {
        for (int b = 0; b < g.n_sites; ++b) {
            bool has_lj = (g.eps_K[a] > 0.0 && g.eps_K[b] > 0.0 && g.sigma_A[a] > 0.0 && g.sigma_A[b] > 0.0);
            bool has_q  = (g.q[a] != 0.0 && g.q[b] != 0.0);
            if (!has_lj && !has_q) continue;

            double dx = dmx + b_off[b][0] - a_off[a][0];
            double dy = dmy + b_off[b][1] - a_off[a][1];
            double dz = dmz + b_off[b][2] - a_off[a][2];
            double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= r_cut2) continue;
            if (r2 < 0.25) { ovl = 1; return 0.0; }

            if (has_lj) {
                double sig = 0.5 * (g.sigma_A[a] + g.sigma_A[b]);
                double eps = sqrt(g.eps_K[a] * g.eps_K[b]);
                double sr2 = (sig * sig) / r2;
                double sr6 = sr2 * sr2 * sr2;
                double ulj = 4.0 * eps * (sr6 * sr6 - sr6);
                if (shifted_lj) {
                    double sc2 = (sig * sig) / r_cut2;
                    double sc6 = sc2 * sc2 * sc2;
                    ulj -= 4.0 * eps * (sc6 * sc6 - sc6);
                }
                e += g.gg_lj_scale * ulj;
            }
            if (has_q) {
                double r = sqrt(r2);
                double uq = KE_K * g.q[a] * g.q[b] *
                            (erfc(dsf_alpha * r) / r - dsf_shift + dsf_deriv * (r - r_cut));
                e += (g.gg_q_scale * g.gg_q_scale) * uq;
            }
        }
    }

    if (g.gg_lut_n >= 2 && r2c < r_cut2) {
        double rc_com = sqrt(r2c);
        if (rc_com < g.gg_lut_rmax) {
            double u_pos = (rc_com - g.gg_lut_rmin) * g.gg_lut_inv_dr;
            if (u_pos <= 0.0) {
                double slope = (double)g.gg_lut_K[1] - (double)g.gg_lut_K[0];
                if (slope > 0.0) slope = 0.0;
                e += (double)g.gg_lut_K[0] + slope * u_pos;
            } else {
                int idx0 = (int)floor(u_pos);
                if (idx0 >= g.gg_lut_n - 1) {
                    e += (double)g.gg_lut_K[g.gg_lut_n - 1];
                } else {
                    double frac = u_pos - (double)idx0;
                    double v0 = (double)g.gg_lut_K[idx0];
                    double v1 = (double)g.gg_lut_K[idx0 + 1];
                    e += v0 + frac * (v1 - v0);
                }
            }
        }
    }

    if (pol_mode != 0 && g.alpha_iso > 0.0 && r2c < r_cut2 && r2c >= 0.25) {
        double rm = sqrt(r2c), invr = 1.0 / rm;
        double invr3 = invr / r2c, invr5 = invr3 / r2c;
        double invs = 1.0 / cbrt(g.alpha_iso);
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

__global__ void build_proposer_uff_grids_kernel(
    Cell cell,
    const FrameworkAtom* __restrict__ atoms,
    int n_atoms,
    const ProposerKVec* __restrict__ kvecs,
    int n_kvecs,
    RigidGuestDef guest,
    int nx, int ny, int nz,
    int max_na, int max_nb, int max_nc,
    double r_cut_A, double ewald_alpha,
    int need_phi, int need_efield,
    float* U_lj0, float* U_lj1, float* U_lj2, float* U_lj3, float* U_lj4, float* U_lj5,
    float* Phi_elec, float* Ex, float* Ey, float* Ez) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny * nz;
    if (idx >= total) return;

    int ix = idx % nx;
    int iy = (idx / nx) % ny;
    int iz = idx / (nx * ny);

    double sa = (double)ix / (double)nx;
    double sb = (double)iy / (double)ny;
    double sc = (double)iz / (double)nz;

    double rx, ry, rz;
    cell.unit_frac_to_cart(sa, sb, sc, rx, ry, rz);

    double r_cut2 = r_cut_A * r_cut_A;
    constexpr double INV_SQRT_PI = 0.56418958354775628695;
    double two_a = 2.0 * ewald_alpha * INV_SQRT_PI;
    double alpha2 = ewald_alpha * ewald_alpha;

    int n_lj = guest.n_lj_types;
    double u[PROP_MAX_LJ] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    bool ovl[PROP_MAX_LJ] = {false, false, false, false, false, false};
    double phi_r = 0.0, ex_r = 0.0, ey_r = 0.0, ez_r = 0.0;

    for (int j = 0; j < n_atoms; ++j) {
        FrameworkAtom at = atoms[j];
        double eps_fw = (double)at.eps_K;
        double sig_fw = (double)at.sigma_A;
        double qj = at.q;

        double dsa = sa - (double)at.sa; dsa -= round(dsa);
        double dsb = sb - (double)at.sb; dsb -= round(dsb);
        double dsc = sc - (double)at.sc; dsc -= round(dsc);

        for (int na = -max_na; na <= max_na; ++na) {
            for (int nb = -max_nb; nb <= max_nb; ++nb) {
                for (int nc = -max_nc; nc <= max_nc; ++nc) {
                    double dx, dy, dz;
                    cell.unit_frac_to_cart(dsa + na, dsb + nb, dsc + nc, dx, dy, dz);
                    double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 >= r_cut2) continue;

                    double r2_safe = (r2 < 0.0625) ? 0.0625 : r2;
                    double r = sqrt(r2_safe);
                    double inv_r = 1.0 / r;
                    double inv_r2 = 1.0 / r2_safe;

                    for (int s = 0; s < n_lj; ++s) {
                        if (ovl[s]) continue;
                        if (r < 0.4) { ovl[s] = true; continue; }
                        double sig_ij = 0.5 * (sig_fw + guest.lj_sigma_A[s]);
                        double eps_ij = sqrt(eps_fw * guest.lj_eps_K[s]);
                        double sr2 = (sig_ij * sig_ij) * inv_r2;
                        double sr6 = sr2 * sr2 * sr2;
                        u[s] += 4.0 * eps_ij * (sr6 * sr6 - sr6);
                    }

                    if ((need_phi || need_efield) && qj != 0.0) {
                        double ar = ewald_alpha * r;
                        double erfc_ar = erfc(ar);
                        phi_r += KE_K * qj * erfc_ar * inv_r;
                        if (need_efield) {
                            double exp_ar2 = exp(-alpha2 * r2_safe);
                            double e_fac = KE_K * qj * inv_r2 * (erfc_ar * inv_r + two_a * exp_ar2);
                            ex_r += e_fac * dx;
                            ey_r += e_fac * dy;
                            ez_r += e_fac * dz;
                        }
                    }
                }
            }
        }
    }

    for (int s = 0; s < n_lj; ++s) {
        if (ovl[s] || u[s] > 1.0e6) u[s] = 1.0e6;
    }

    if (n_lj > 0 && U_lj0) U_lj0[idx] = (float)u[0];
    if (n_lj > 1 && U_lj1) U_lj1[idx] = (float)u[1];
    if (n_lj > 2 && U_lj2) U_lj2[idx] = (float)u[2];
    if (n_lj > 3 && U_lj3) U_lj3[idx] = (float)u[3];
    if (n_lj > 4 && U_lj4) U_lj4[idx] = (float)u[4];
    if (n_lj > 5 && U_lj5) U_lj5[idx] = (float)u[5];

    if (need_phi || need_efield) {
        double phi_k = 0.0, ex_k = 0.0, ey_k = 0.0, ez_k = 0.0;
        for (int m = 0; m < n_kvecs; ++m) {
            ProposerKVec kv = kvecs[m];
            double kr = kv.kx * rx + kv.ky * ry + kv.kz * rz;
            double s_kr, c_kr;
            sincos(kr, &s_kr, &c_kr);
            phi_k += kv.W_C * c_kr + kv.W_S * s_kr;
            if (need_efield) {
                double e_amp = kv.W_C * s_kr - kv.W_S * c_kr;
                ex_k += e_amp * kv.kx;
                ey_k += e_amp * kv.ky;
                ez_k += e_amp * kv.kz;
            }
        }
        if (need_phi && Phi_elec) Phi_elec[idx] = (float)(phi_r + phi_k);
        if (need_efield && Ex && Ey && Ez) {
            Ex[idx] = (float)(ex_r + ex_k);
            Ey[idx] = (float)(ey_r + ey_k);
            Ez[idx] = (float)(ez_r + ez_k);
        }
    }
}

std::vector<ProposerKVec> build_proposer_kvecs(
    const Cell& cell,
    const std::vector<FrameworkAtom>& atoms,
    double alpha, double k_max_Ainv) {

    constexpr double PI = 3.14159265358979323846;
    double b1[3] = {2.0 * PI * cell.H_unit_inv[0][0], 2.0 * PI * cell.H_unit_inv[0][1], 2.0 * PI * cell.H_unit_inv[0][2]};
    double b2[3] = {2.0 * PI * cell.H_unit_inv[1][0], 2.0 * PI * cell.H_unit_inv[1][1], 2.0 * PI * cell.H_unit_inv[1][2]};
    double b3[3] = {2.0 * PI * cell.H_unit_inv[2][0], 2.0 * PI * cell.H_unit_inv[2][1], 2.0 * PI * cell.H_unit_inv[2][2]};

    double nb1 = std::sqrt(b1[0]*b1[0] + b1[1]*b1[1] + b1[2]*b1[2]);
    double nb2 = std::sqrt(b2[0]*b2[0] + b2[1]*b2[1] + b2[2]*b2[2]);
    double nb3 = std::sqrt(b3[0]*b3[0] + b3[1]*b3[1] + b3[2]*b3[2]);

    int max_m1 = (int)std::ceil(k_max_Ainv / (std::max(1.0e-3, nb1) * 0.5)) + 2;
    int max_m2 = (int)std::ceil(k_max_Ainv / (std::max(1.0e-3, nb2) * 0.5)) + 2;
    int max_m3 = (int)std::ceil(k_max_Ainv / std::max(1.0e-3, nb3)) + 2;

    double kmax2 = k_max_Ainv * k_max_Ainv;
    double inv_4alpha2 = 1.0 / (4.0 * alpha * alpha);
    double prefactor = 2.0 * KE_K * (4.0 * PI / cell.volume_unit);

    std::vector<ProposerKVec> kvecs;
    for (int mc = 0; mc <= max_m3; ++mc) {
        int mb_min = (mc == 0) ? 0 : -max_m2;
        for (int mb = mb_min; mb <= max_m2; ++mb) {
            int ma_min = (mc == 0 && mb == 0) ? 1 : -max_m1;
            for (int ma = ma_min; ma <= max_m1; ++ma) {
                double kx = ma*b1[0] + mb*b2[0] + mc*b3[0];
                double ky = ma*b1[1] + mb*b2[1] + mc*b3[1];
                double kz = ma*b1[2] + mb*b2[2] + mc*b3[2];
                double k2 = kx*kx + ky*ky + kz*kz;
                if (k2 <= 1.0e-12 || k2 > kmax2) continue;

                double w = prefactor * std::exp(-k2 * inv_4alpha2) / k2;
                double Ck = 0.0, Sk = 0.0;
                for (const auto& at : atoms) {
                    if (at.q == 0.0) continue;
                    double krj = kx * at.x + ky * at.y + kz * at.z;
                    Ck += at.q * std::cos(krj);
                    Sk += at.q * std::sin(krj);
                }
                kvecs.push_back({kx, ky, kz, w * Ck, w * Sk});
            }
        }
    }
    return kvecs;
}

void free_proposer_uff_grids_gpu(ProposerGridView& g) {
    for (int s = 0; s < PROP_MAX_LJ; ++s) {
        if (g.U_lj[s]) cudaFree((void*)g.U_lj[s]);
        g.U_lj[s] = nullptr;
    }
    if (g.Phi_elec) cudaFree((void*)g.Phi_elec);
    if (g.Ex) cudaFree((void*)g.Ex);
    if (g.Ey) cudaFree((void*)g.Ey);
    if (g.Ez) cudaFree((void*)g.Ez);
    g.Phi_elec = g.Ex = g.Ey = g.Ez = nullptr;
}

int build_proposer_uff_grids_gpu(
    const Cell& cell,
    const std::vector<FrameworkAtom>& atoms,
    const RigidGuestDef& guest,
    int nx, int ny, int nz,
    double r_cut_A, double ewald_alpha, int pol_mode,
    ProposerGridView& grid_out) {

    grid_out = ProposerGridView{};
    grid_out.nx = nx;
    grid_out.ny = ny;
    grid_out.nz = nz;
    grid_out.n_lj = guest.n_lj_types;

    bool fw_has_charge = false;
    for (const auto& at : atoms) {
        if (std::fabs(at.q) > 1.0e-12) {
            fw_has_charge = true;
            break;
        }
    }
    int need_phi = (fw_has_charge && (guest.has_charge || pol_mode != 0)) ? 1 : 0;
    int need_efield = (fw_has_charge && pol_mode != 0) ? 1 : 0;
    grid_out.has_phi = need_phi;
    grid_out.has_efield = need_efield;

    size_t total = (size_t)nx * (size_t)ny * (size_t)nz;
    size_t bytes = total * sizeof(float);

    float* d_u[PROP_MAX_LJ] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    float *d_phi = nullptr, *d_ex = nullptr, *d_ey = nullptr, *d_ez = nullptr;
    FrameworkAtom* d_atoms = nullptr;
    ProposerKVec* d_kvecs = nullptr;

    auto cleanup_err = [&]() {
        for (int s = 0; s < PROP_MAX_LJ; ++s) {
            if (d_u[s]) cudaFree(d_u[s]);
        }
        if (d_phi) cudaFree(d_phi);
        if (d_ex) cudaFree(d_ex);
        if (d_ey) cudaFree(d_ey);
        if (d_ez) cudaFree(d_ez);
        if (d_atoms) cudaFree(d_atoms);
        if (d_kvecs) cudaFree(d_kvecs);
    };

#define UFF_GRID_CUDA(call)                                                     \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA Error at %s:%d: %s\n",                   \
                         __FILE__, __LINE__, cudaGetErrorString(err));          \
            cleanup_err();                                                      \
            return -1;                                                          \
        }                                                                       \
    } while (0)

    for (int s = 0; s < guest.n_lj_types; ++s) {
        UFF_GRID_CUDA(cudaMalloc(&d_u[s], bytes));
    }
    if (need_phi) UFF_GRID_CUDA(cudaMalloc(&d_phi, bytes));
    if (need_efield) {
        UFF_GRID_CUDA(cudaMalloc(&d_ex, bytes));
        UFF_GRID_CUDA(cudaMalloc(&d_ey, bytes));
        UFF_GRID_CUDA(cudaMalloc(&d_ez, bytes));
    }

    std::vector<ProposerKVec> kvecs;
    if (need_phi || need_efield) {
        kvecs = build_proposer_kvecs(cell, atoms, ewald_alpha, 2.6);
    }

    UFF_GRID_CUDA(cudaMalloc(&d_atoms, std::max((size_t)1, atoms.size()) * sizeof(FrameworkAtom)));
    if (!atoms.empty()) {
        UFF_GRID_CUDA(cudaMemcpy(d_atoms, atoms.data(), atoms.size() * sizeof(FrameworkAtom), cudaMemcpyHostToDevice));
    }
    if (!kvecs.empty()) {
        UFF_GRID_CUDA(cudaMalloc(&d_kvecs, kvecs.size() * sizeof(ProposerKVec)));
        UFF_GRID_CUDA(cudaMemcpy(d_kvecs, kvecs.data(), kvecs.size() * sizeof(ProposerKVec), cudaMemcpyHostToDevice));
    }

    int max_na = std::max(1, (int)std::ceil(r_cut_A / std::max(1.0e-3, cell.perp_width_unit[0])) + 1);
    int max_nb = std::max(1, (int)std::ceil(r_cut_A / std::max(1.0e-3, cell.perp_width_unit[1])) + 1);
    int max_nc = std::max(1, (int)std::ceil(r_cut_A / std::max(1.0e-3, cell.perp_width_unit[2])) + 1);

    int threads = 256;
    int blocks = ((int)total + threads - 1) / threads;
    build_proposer_uff_grids_kernel<<<blocks, threads>>>(
        cell, d_atoms, (int)atoms.size(), d_kvecs, (int)kvecs.size(), guest,
        nx, ny, nz, max_na, max_nb, max_nc, r_cut_A, ewald_alpha, need_phi, need_efield,
        d_u[0], d_u[1], d_u[2], d_u[3], d_u[4], d_u[5],
        d_phi, d_ex, d_ey, d_ez);
    UFF_GRID_CUDA(cudaGetLastError());
    UFF_GRID_CUDA(cudaDeviceSynchronize());

    if (d_atoms) cudaFree(d_atoms);
    if (d_kvecs) cudaFree(d_kvecs);
#undef UFF_GRID_CUDA

    for (int s = 0; s < guest.n_lj_types; ++s) grid_out.U_lj[s] = d_u[s];
    grid_out.Phi_elec = d_phi;
    grid_out.Ex = d_ex;
    grid_out.Ey = d_ey;
    grid_out.Ez = d_ez;
    return 0;
}

struct ProposerStateDev {
    int* d_acc_N = nullptr;
    double* d_acc_U = nullptr;
    double* d_acc_cx = nullptr;
    double* d_acc_cy = nullptr;
    double* d_acc_cz = nullptr;
    double* d_acc_qw = nullptr;
    double* d_acc_qx = nullptr;
    double* d_acc_qy = nullptr;
    double* d_acc_qz = nullptr;

    int* d_tri_N = nullptr;
    double* d_tri_U = nullptr;
    double* d_tri_dU = nullptr;
    int* d_tri_acc = nullptr;
    double* d_tri_cx = nullptr;
    double* d_tri_cy = nullptr;
    double* d_tri_cz = nullptr;
    double* d_tri_qw = nullptr;
    double* d_tri_qx = nullptr;
    double* d_tri_qy = nullptr;
    double* d_tri_qz = nullptr;

    int* d_commit_mask = nullptr;
    uint64_t* d_rng = nullptr;

    // Pre-allocated single-state evaluation buffers (avoids dynamic cudaMalloc/Free per call)
    double* d_eval_cx = nullptr;
    double* d_eval_cy = nullptr;
    double* d_eval_cz = nullptr;
    double* d_eval_qw = nullptr;
    double* d_eval_qx = nullptr;
    double* d_eval_qy = nullptr;
    double* d_eval_qz = nullptr;
    double* d_eval_U = nullptr;
    int* d_eval_ov = nullptr;
};

struct PropMove {
    int mv;
    int tgt;
    int cur_N;
    int fw_ovl;
    int skip;
    double cx, cy, cz, qw, qx, qy, qz;
    double mux, muy, muz;
    double u_fw_new, u_fw_old;
};

__global__ void propose_burst_kernel(
    Cell cell, ProposerGridView grid, RigidGuestDef guest,
    double T_K, double beta_f_V, double r_cut_gg, double dsf_alpha,
    int ff_mode, int pol_mode, int max_mol,
    int k_steps, int commit_to_acc,
    int cbmc_k, double p_ins_del_total,
    ProposerStateDev st) {

    int b = blockIdx.x;
    int tid = threadIdx.x;
    int off = b * max_mol;
    int k_cbmc = (cbmc_k < 1) ? 1 : ((cbmc_k > PROP_THREADS) ? PROP_THREADS : cbmc_k);

    __shared__ double s_cx[PROP_SMEM_MAX], s_cy[PROP_SMEM_MAX], s_cz[PROP_SMEM_MAX];
    __shared__ double s_qw[PROP_SMEM_MAX], s_qx[PROP_SMEM_MAX], s_qy[PROP_SMEM_MAX], s_qz[PROP_SMEM_MAX];
    __shared__ double s_mux[PROP_SMEM_MAX], s_muy[PROP_SMEM_MAX], s_muz[PROP_SMEM_MAX];
    __shared__ double s_ufw[PROP_SMEM_MAX];
    __shared__ PropMove prop;
    __shared__ double warp_u[4];
    __shared__ int warp_ovl[4];
    __shared__ double s_u_recomp;
    __shared__ int s_cbmc_valid[PROP_THREADS];
    __shared__ double s_cbmc_logw[PROP_THREADS];
    __shared__ int s_cbmc_chosen;
    __shared__ uint64_t s_step_seed;

    int init_N = st.d_acc_N[b];
    if (init_N > max_mol) init_N = max_mol;
    for (int i = tid; i < init_N; i += PROP_THREADS) {
        double cx = st.d_acc_cx[off + i], cy = st.d_acc_cy[off + i], cz = st.d_acc_cz[off + i];
        double qw = st.d_acc_qw[off + i], qx = st.d_acc_qx[off + i], qy = st.d_acc_qy[off + i], qz = st.d_acc_qz[off + i];
        s_cx[i] = cx; s_cy[i] = cy; s_cz[i] = cz;
        s_qw[i] = qw; s_qx[i] = qx; s_qy[i] = qy; s_qz[i] = qz;
        double ulj = 0.0, uel = 0.0, up = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
        eval_rigid_guest_framework_grid(grid, cell, guest, pol_mode, KE_K, cx, cy, cz, qw, qx, qy, qz,
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
    int shifted_lj = (ff_mode == 1 || ff_mode == 2) ? 1 : 0;
    double rc = r_cut_gg, rc2 = rc * rc;
    constexpr double INV_SQRT_PI = 0.56418958354775628695;
    double erfc_rc = erfc(dsf_alpha * rc), exp_rc = exp(-dsf_alpha * dsf_alpha * rc2);
    double dsf_shift = erfc_rc / rc;
    double dsf_deriv = erfc_rc / rc2 + (2.0 * dsf_alpha * INV_SQRT_PI) * exp_rc / rc;

    double p_ins = 0.5 * p_ins_del_total;
    if (p_ins < 0.05) p_ins = 0.05;
    if (p_ins > 0.48) p_ins = 0.48;
    double p_del = p_ins;
    double p_exch = p_ins + p_del;

    if (tid == 0) prop.cur_N = cur_N;
    __syncthreads();

    for (int step = 0; step < k_steps; ++step) {
        if (tid == 0) {
            prop.cur_N = cur_N;
            prop.fw_ovl = 0;
            prop.skip = 0;
            prop.u_fw_new = prop.u_fw_old = 0.0;
            prop.mux = prop.muy = prop.muz = 0.0;
            s_cbmc_chosen = -1;
            s_step_seed = xs64(rng);

            double u_mv = u01(rng);
            int mv;
            if (u_mv < p_ins) {
                mv = 2;
            } else if (u_mv < p_exch) {
                mv = 3;
            } else if (guest.is_isotropic) {
                mv = 0;
            } else {
                double mid = p_exch + 0.5 * (1.0 - p_exch);
                mv = (u_mv < mid) ? 0 : 1;
            }
            prop.mv = mv;

            if (grid.n_lj == 0 && grid.has_phi == 0) {
                prop.skip = 1;
            } else if (mv == 0 || mv == 1) {
                if (cur_N == 0) {
                    prop.skip = 1;
                } else {
                    int idx = (int)(u01(rng) * (double)cur_N);
                    if (idx >= cur_N) idx = cur_N - 1;
                    prop.tgt = idx;
                    prop.u_fw_old = s_ufw[idx];
                    double ox = s_cx[idx], oy = s_cy[idx], oz = s_cz[idx];
                    double oqw = s_qw[idx], oqx = s_qx[idx], oqy = s_qy[idx], oqz = s_qz[idx];
                    if (mv == 0) {
                        double nx = ox + (2.0 * u01(rng) - 1.0) * DELTA_TRANS;
                        double ny = oy + (2.0 * u01(rng) - 1.0) * DELTA_TRANS;
                        double nz = oz + (2.0 * u01(rng) - 1.0) * DELTA_TRANS;
                        cell.wrap_super_cart(nx, ny, nz);
                        prop.cx = nx; prop.cy = ny; prop.cz = nz;
                        prop.qw = oqw; prop.qx = oqx; prop.qy = oqy; prop.qz = oqz;
                    } else {
                        prop.cx = ox; prop.cy = oy; prop.cz = oz;
                        rot_quat_small(rng, MAX_ANGLE, oqw, oqx, oqy, oqz, prop.qw, prop.qx, prop.qy, prop.qz);
                    }
                    double ulj = 0.0, uel = 0.0, up = 0.0;
                    bool ovl = eval_rigid_guest_framework_grid(
                        grid, cell, guest, pol_mode, KE_K,
                        prop.cx, prop.cy, prop.cz, prop.qw, prop.qx, prop.qy, prop.qz,
                        ulj, uel, up, prop.mux, prop.muy, prop.muz);
                    prop.u_fw_new = ulj + uel + up;
                    if (ovl) { prop.fw_ovl = 1; prop.skip = 1; }
                }
            } else if (mv == 2) {
                if (cur_N >= max_mol || cur_N >= PROP_SMEM_MAX || beta_f_V <= 1.0e-300) {
                    prop.skip = 1;
                } else {
                    prop.tgt = -1;
                }
            } else {
                if (cur_N == 0 || beta_f_V <= 1.0e-300) {
                    prop.skip = 1;
                } else {
                    int idx = (int)(u01(rng) * (double)cur_N);
                    if (idx >= cur_N) idx = cur_N - 1;
                    prop.tgt = idx;
                }
            }
        }
        __syncthreads();

        double tcx = 0.0, tcy = 0.0, tcz = 0.0;
        double tqw = 1.0, tqx = 0.0, tqy = 0.0, tqz = 0.0;
        double tmux = 0.0, tmuy = 0.0, tmuz = 0.0, tufw = 0.0, tdU = 0.0;

        if (!prop.skip && (prop.mv == 2 || prop.mv == 3)) {
            int N_now = prop.cur_N;
            int idx_del = prop.tgt;
            int tvalid = 0;
            if (tid < k_cbmc) {
                if (prop.mv == 3 && tid == 0) {
                    tcx = s_cx[idx_del]; tcy = s_cy[idx_del]; tcz = s_cz[idx_del];
                    tqw = s_qw[idx_del]; tqx = s_qx[idx_del]; tqy = s_qy[idx_del]; tqz = s_qz[idx_del];
                } else {
                    uint64_t trng = s_step_seed ^ ((uint64_t)(tid + 1) * 0x9e3779b97f4a7c15ULL);
                    splitmix64_next(trng);
                    double sa = sm_u01(trng), sb = sm_u01(trng), sc = sm_u01(trng);
                    cell.super_frac_to_cart(sa, sb, sc, tcx, tcy, tcz);
                    if (!guest.is_isotropic) {
                        rand_uniform_quat_sm(trng, tqw, tqx, tqy, tqz);
                    }
                }
                double ulj = 0.0, uel = 0.0, up = 0.0;
                bool ov_fw = eval_rigid_guest_framework_grid(
                    grid, cell, guest, pol_mode, KE_K,
                    tcx, tcy, tcz, tqw, tqx, tqy, tqz,
                    ulj, uel, up, tmux, tmuy, tmuz);
                if (!ov_fw) {
                    tufw = ulj + uel + up;
                    double dugg = 0.0;
                    int ov_gg = 0;
                    for (int j = 0; j < N_now; ++j) {
                        if (prop.mv == 3 && j == idx_del) continue;
                        dugg += pair_rigid_guest(
                            cell, guest,
                            tcx, tcy, tcz, tqw, tqx, tqy, tqz, tmux, tmuy, tmuz,
                            s_cx[j], s_cy[j], s_cz[j], s_qw[j], s_qx[j], s_qy[j], s_qz[j], s_mux[j], s_muy[j], s_muz[j],
                            shifted_lj, pol_mode, rc, rc2, dsf_alpha, dsf_shift, dsf_deriv, ov_gg);
                        if (ov_gg) break;
                    }
                    if (!ov_gg) {
                        tdU = tufw + dugg;
                        tvalid = 1;
                    }
                }
                if (prop.mv == 3 && tid == 0) {
                    prop.u_fw_old = tdU;
                }
            }
            s_cbmc_valid[tid] = tvalid;
            s_cbmc_logw[tid] = tvalid ? (-tdU / T_K) : -1.0e300;
        }

        double loc_du = 0.0;
        int loc_ovl = 0;
        if (!prop.skip && (prop.mv == 0 || prop.mv == 1) && prop.cur_N > 0) {
            int N = prop.cur_N, tgt = prop.tgt;
            double ox = s_cx[tgt], oy = s_cy[tgt], oz = s_cz[tgt];
            double oqw = s_qw[tgt], oqx = s_qx[tgt], oqy = s_qy[tgt], oqz = s_qz[tgt];
            double omx = s_mux[tgt], omy = s_muy[tgt], omz = s_muz[tgt];
            for (int j = tid; j < N; j += PROP_THREADS) {
                if (j == tgt) continue;
                int o1 = 0, o2 = 0;
                double enew = pair_rigid_guest(
                    cell, guest,
                    prop.cx, prop.cy, prop.cz, prop.qw, prop.qx, prop.qy, prop.qz, prop.mux, prop.muy, prop.muz,
                    s_cx[j], s_cy[j], s_cz[j], s_qw[j], s_qx[j], s_qy[j], s_qz[j], s_mux[j], s_muy[j], s_muz[j],
                    shifted_lj, pol_mode, rc, rc2, dsf_alpha, dsf_shift, dsf_deriv, o1);
                if (o1) { loc_ovl = 1; break; }
                double eold = pair_rigid_guest(
                    cell, guest,
                    ox, oy, oz, oqw, oqx, oqy, oqz, omx, omy, omz,
                    s_cx[j], s_cy[j], s_cz[j], s_qw[j], s_qx[j], s_qy[j], s_qz[j], s_mux[j], s_muy[j], s_muz[j],
                    shifted_lj, pol_mode, rc, rc2, dsf_alpha, dsf_shift, dsf_deriv, o2);
                loc_du += (enew - eold);
            }
            int lane = tid & 31, wid = tid >> 5;
            for (int s = 16; s > 0; s >>= 1) {
                loc_du += __shfl_down_sync(0xFFFFFFFFU, loc_du, s);
                loc_ovl |= __shfl_down_sync(0xFFFFFFFFU, loc_ovl, s);
            }
            if (lane == 0) { warp_u[wid] = loc_du; warp_ovl[wid] = loc_ovl; }
        }
        __syncthreads();

        if (tid == 0) {
            int mv = prop.mv;
            if (!prop.skip && (mv == 0 || mv == 1) && cur_N > 0 && !prop.fw_ovl) {
                double dU_gg = warp_u[0] + warp_u[1] + warp_u[2] + warp_u[3];
                int ovl = warp_ovl[0] | warp_ovl[1] | warp_ovl[2] | warp_ovl[3];
                if (!ovl) {
                    double dU = (prop.u_fw_new - prop.u_fw_old) + dU_gg;
                    double arg = -dU / T_K;
                    if ((dU <= 0.0) || (arg > -50.0 && u01(rng) < exp(arg))) {
                        int t = prop.tgt;
                        if (mv == 0) {
                            s_cx[t] = prop.cx; s_cy[t] = prop.cy; s_cz[t] = prop.cz;
                        } else {
                            s_qw[t] = prop.qw; s_qx[t] = prop.qx; s_qy[t] = prop.qy; s_qz[t] = prop.qz;
                        }
                        s_mux[t] = prop.mux; s_muy[t] = prop.muy; s_muz[t] = prop.muz;
                        s_ufw[t] = prop.u_fw_new;
                        U_total += dU;
                        ++inner_acc;
                    }
                }
            } else if (!prop.skip && mv == 2 && cur_N < max_mol && cur_N < PROP_SMEM_MAX) {
                double u_sel = u01(rng);
                double u_acc = u01(rng);
                double max_lw = -1.0e300;
                int first_valid = -1;
                for (int t = 0; t < k_cbmc; ++t) {
                    if (s_cbmc_valid[t]) {
                        if (first_valid < 0 || s_cbmc_logw[t] > max_lw) max_lw = s_cbmc_logw[t];
                        if (first_valid < 0) first_valid = t;
                    }
                }
                if (first_valid >= 0) {
                    double sum_w = 0.0;
                    for (int t = 0; t < k_cbmc; ++t) {
                        if (s_cbmc_valid[t]) sum_w += exp(s_cbmc_logw[t] - max_lw);
                    }
                    double log_W = max_lw + log(sum_w / (double)k_cbmc);
                    double log_r = log(beta_f_V / (double)(cur_N + 1)) + log_W;
                    if (log_r >= 0.0 || (log_r > -60.0 && u_acc < exp(log_r))) {
                        double tgt_w = u_sel * sum_w;
                        double run_w = 0.0;
                        int chosen = first_valid;
                        for (int t = 0; t < k_cbmc; ++t) {
                            if (!s_cbmc_valid[t]) continue;
                            chosen = t;
                            run_w += exp(s_cbmc_logw[t] - max_lw);
                            if (run_w >= tgt_w) break;
                        }
                        s_cbmc_chosen = chosen;
                    }
                }
            } else if (!prop.skip && mv == 3 && cur_N > 0) {
                double u_acc = u01(rng);
                bool del_ok = false;
                if (!s_cbmc_valid[0]) {
                    del_ok = true;
                } else {
                    double max_lw = s_cbmc_logw[0];
                    for (int t = 1; t < k_cbmc; ++t) {
                        if (s_cbmc_valid[t] && s_cbmc_logw[t] > max_lw) max_lw = s_cbmc_logw[t];
                    }
                    double sum_w = 0.0;
                    for (int t = 0; t < k_cbmc; ++t) {
                        if (s_cbmc_valid[t]) sum_w += exp(s_cbmc_logw[t] - max_lw);
                    }
                    double log_W = max_lw + log(sum_w / (double)k_cbmc);
                    double log_r = log((double)cur_N / beta_f_V) - log_W;
                    if (log_r >= 0.0 || (log_r > -60.0 && u_acc < exp(log_r))) {
                        del_ok = true;
                    }
                }
                if (del_ok) {
                    int t = prop.tgt, last = cur_N - 1;
                    if (t != last) {
                        s_cx[t] = s_cx[last]; s_cy[t] = s_cy[last]; s_cz[t] = s_cz[last];
                        s_qw[t] = s_qw[last]; s_qx[t] = s_qx[last]; s_qy[t] = s_qy[last]; s_qz[t] = s_qz[last];
                        s_mux[t] = s_mux[last]; s_muy[t] = s_muy[last]; s_muz[t] = s_muz[last];
                        s_ufw[t] = s_ufw[last];
                    }
                    --cur_N;
                    prop.cur_N = cur_N;
                    U_total -= prop.u_fw_old;
                    ++inner_acc;
                }
            }
        }
        __syncthreads();

        if (!prop.skip && prop.mv == 2 && s_cbmc_chosen >= 0 && tid == s_cbmc_chosen) {
            int slot = prop.cur_N;
            s_cx[slot] = tcx; s_cy[slot] = tcy; s_cz[slot] = tcz;
            s_qw[slot] = tqw; s_qx[slot] = tqx; s_qy[slot] = tqy; s_qz[slot] = tqz;
            s_mux[slot] = tmux; s_muy[slot] = tmuy; s_muz[slot] = tmuz;
            s_ufw[slot] = tufw;
            prop.u_fw_new = tdU;
        }
        __syncthreads();
        if (!prop.skip && prop.mv == 2 && s_cbmc_chosen >= 0 && tid == 0) {
            ++cur_N;
            prop.cur_N = cur_N;
            U_total += prop.u_fw_new;
            ++inner_acc;
        }
        __syncthreads();
    }

    // Exact full recomputation at end of burst (guarantees zero drift in dU_ref)
    int final_N = prop.cur_N;
    double loc_re = 0.0;
    for (int i = tid; i < final_N; i += PROP_THREADS) {
        double ulj = 0.0, uel = 0.0, up = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
        eval_rigid_guest_framework_grid(grid, cell, guest, pol_mode, KE_K,
                                        s_cx[i], s_cy[i], s_cz[i], s_qw[i], s_qx[i], s_qy[i], s_qz[i],
                                        ulj, uel, up, mx, my, mz);
        loc_re += (ulj + uel + up);
        s_mux[i] = mx; s_muy[i] = my; s_muz[i] = mz;
    }
    __syncthreads();
    for (int i = tid; i < final_N; i += PROP_THREADS) {
        for (int j = i + 1; j < final_N; ++j) {
            int ov = 0;
            loc_re += pair_rigid_guest(
                cell, guest,
                s_cx[i], s_cy[i], s_cz[i], s_qw[i], s_qx[i], s_qy[i], s_qz[i], s_mux[i], s_muy[i], s_muz[i],
                s_cx[j], s_cy[j], s_cz[j], s_qw[j], s_qx[j], s_qy[j], s_qz[j], s_mux[j], s_muy[j], s_muz[j],
                shifted_lj, pol_mode, rc, rc2, dsf_alpha, dsf_shift, dsf_deriv, ov);
        }
    }
    int lane = tid & 31, wid = tid >> 5;
    for (int s = 16; s > 0; s >>= 1) loc_re += __shfl_down_sync(0xFFFFFFFFU, loc_re, s);
    if (lane == 0) warp_u[wid] = loc_re;
    __syncthreads();
    if (tid == 0) {
        s_u_recomp = warp_u[0] + warp_u[1] + warp_u[2] + warp_u[3];
        if (k_steps > 0) st.d_rng[b] = rng;
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
    double* oqw = commit_to_acc ? st.d_acc_qw : st.d_tri_qw;
    double* oqx = commit_to_acc ? st.d_acc_qx : st.d_tri_qx;
    double* oqy = commit_to_acc ? st.d_acc_qy : st.d_tri_qy;
    double* oqz = commit_to_acc ? st.d_acc_qz : st.d_tri_qz;
    for (int i = tid; i < final_N; i += PROP_THREADS) {
        ocx[off + i] = s_cx[i]; ocy[off + i] = s_cy[i]; ocz[off + i] = s_cz[i];
        oqw[off + i] = s_qw[i]; oqx[off + i] = s_qx[i]; oqy[off + i] = s_qy[i]; oqz[off + i] = s_qz[i];
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
        st.d_acc_qw[off + i] = st.d_tri_qw[off + i];
        st.d_acc_qx[off + i] = st.d_tri_qx[off + i];
        st.d_acc_qy[off + i] = st.d_tri_qy[off + i];
        st.d_acc_qz[off + i] = st.d_tri_qz[off + i];
    }
}

__global__ void eval_single_state_kernel(
    Cell cell, ProposerGridView grid, RigidGuestDef guest,
    double r_cut_gg, double dsf_alpha, int ff_mode, int pol_mode, int N,
    const double* cx, const double* cy, const double* cz,
    const double* qw, const double* qx, const double* qy, const double* qz,
    double* out_U, int* out_ovl) {

    int tid = threadIdx.x;
    __shared__ double s_cx[PROP_SMEM_MAX], s_cy[PROP_SMEM_MAX], s_cz[PROP_SMEM_MAX];
    __shared__ double s_qw[PROP_SMEM_MAX], s_qx[PROP_SMEM_MAX], s_qy[PROP_SMEM_MAX], s_qz[PROP_SMEM_MAX];
    __shared__ double s_mux[PROP_SMEM_MAX], s_muy[PROP_SMEM_MAX], s_muz[PROP_SMEM_MAX];
    __shared__ double warp_u[4];
    __shared__ int warp_ovl[4];

    double loc_u = 0.0;
    int loc_ov = 0;
    for (int i = tid; i < N; i += PROP_THREADS) {
        double x = cx[i], y = cy[i], z = cz[i];
        double rw = qw[i], rx = qx[i], ry = qy[i], rz = qz[i];
        s_cx[i] = x; s_cy[i] = y; s_cz[i] = z;
        s_qw[i] = rw; s_qx[i] = rx; s_qy[i] = ry; s_qz[i] = rz;
        double ulj = 0.0, uel = 0.0, up = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
        bool ov = eval_rigid_guest_framework_grid(
            grid, cell, guest, pol_mode, KE_K,
            x, y, z, rw, rx, ry, rz, ulj, uel, up, mx, my, mz);
        if (ov) loc_ov = 1;
        loc_u += (ulj + uel + up);
        s_mux[i] = mx; s_muy[i] = my; s_muz[i] = mz;
    }
    __syncthreads();

    int shifted_lj = (ff_mode == 1 || ff_mode == 2) ? 1 : 0;
    double rc = r_cut_gg, rc2 = rc * rc;
    constexpr double INV_SQRT_PI = 0.56418958354775628695;
    double erfc_rc = erfc(dsf_alpha * rc), exp_rc = exp(-dsf_alpha * dsf_alpha * rc2);
    double dsf_shift = erfc_rc / rc;
    double dsf_deriv = erfc_rc / rc2 + (2.0 * dsf_alpha * INV_SQRT_PI) * exp_rc / rc;

    for (int i = tid; i < N; i += PROP_THREADS) {
        for (int j = i + 1; j < N; ++j) {
            int ov = 0;
            double e = pair_rigid_guest(
                cell, guest,
                s_cx[i], s_cy[i], s_cz[i], s_qw[i], s_qx[i], s_qy[i], s_qz[i], s_mux[i], s_muy[i], s_muz[i],
                s_cx[j], s_cy[j], s_cz[j], s_qw[j], s_qx[j], s_qy[j], s_qz[j], s_mux[j], s_muy[j], s_muz[j],
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
    MixGrid3DView unified_grid{};
    bool owns_unified_grid = false;
    ProposerGridView grid{};
    RigidGuestDef guest{};
    SpeciesDef sp{};
    std::vector<double> unit_frac;
    std::vector<int> unit_Z;
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
    int cbmc_k = 128;
    double p_ins_del_total = 0.60;
    ProposerStateDev dev{};
};

RigidGuestDef make_default_co2_guest(int ff_mode) {
    SpeciesDef sp = (ff_mode == 1) ? make_species_co2_epm2() : make_species_co2();
    RigidGuestDef g{};
    g.n_sites = sp.n_sites;
    g.n_lj_types = 2;
    g.is_isotropic = 0;
    g.has_charge = 1;
    g.max_radius_A = sp.bond_A;
    g.lx[0] = 0.0; g.ly[0] = 0.0; g.lz[0] = 0.0;
    g.lx[1] = sp.bond_A; g.ly[1] = 0.0; g.lz[1] = 0.0;
    g.lx[2] = -sp.bond_A; g.ly[2] = 0.0; g.lz[2] = 0.0;
    for (int i = 0; i < 3; ++i) {
        g.q[i] = sp.q[i];
        g.eps_K[i] = sp.eps_K[i];
        g.sigma_A[i] = sp.sigma_A[i];
        g.w_pol[i] = sp.w_pol[i];
        g.lj_type[i] = (i == 0) ? 0 : 1;
    }
    g.lj_eps_K[0] = sp.eps_K[0];   g.lj_sigma_A[0] = sp.sigma_A[0];
    g.lj_eps_K[1] = sp.eps_K[1];   g.lj_sigma_A[1] = sp.sigma_A[1];
    g.alpha_par = sp.alpha_par;
    g.alpha_perp = sp.alpha_perp;
    g.alpha_iso = sp.alpha_iso;
    g.Tc_K = sp.Tc_K;
    g.Pc_bar = sp.Pc_bar;
    g.omega = sp.omega;
    g.gg_q_scale = 1.0;
    g.gg_lj_scale = 1.0;
    g.gg_lut_n = 0;
    g.gg_lut_rmin = 2.0;
    g.gg_lut_rmax = 10.0;
    g.gg_lut_inv_dr = 0.0;
    return g;
}

void* finalize_proposer_context(
    ProposerContext* ctx,
    const std::vector<FrameworkAtom>& unit_atoms,
    int nx, int ny, int nz,
    uint64_t seed) {

    FugacityResult fug = compute_pure_fugacity_pr(
        ctx->T_K, ctx->pressure_bar, ctx->guest.Tc_K, ctx->guest.Pc_bar, ctx->guest.omega);
    ctx->f_Pa = fug.f_Pa;
    constexpr double KB_SI = 1.380649e-23;
    ctx->beta_f_V = (ctx->f_Pa * ctx->cell.volume_super * 1.0e-30) / (KB_SI * ctx->T_K);
    ctx->delta_eps_K = 0.0;
    ctx->beta_f_V_ref = ctx->beta_f_V;
    ctx->r_cut_gg = std::min(11.0, 0.48 * ctx->cell.min_perp_width_super);

    if (ctx->ff_mode == 1) {
        UnifiedGridStats stats{};
        build_unified_grids_gpu(ctx->cell, unit_atoms, nx, ny, nz, 11.0, 0.30, 1, ctx->unified_grid, stats);
        ctx->owns_unified_grid = true;
        ctx->grid = ProposerGridView{};
        ctx->grid.nx = ctx->unified_grid.nx;
        ctx->grid.ny = ctx->unified_grid.ny;
        ctx->grid.nz = ctx->unified_grid.nz;
        ctx->grid.n_lj = 2;
        ctx->grid.has_phi = 1;
        ctx->grid.has_efield = 1;
        ctx->grid.U_lj[0] = ctx->unified_grid.U_lj[0];
        ctx->grid.U_lj[1] = ctx->unified_grid.U_lj[1];
        ctx->grid.Phi_elec = ctx->unified_grid.Phi_elec;
        ctx->grid.Ex = ctx->unified_grid.Ex;
        ctx->grid.Ey = ctx->unified_grid.Ey;
        ctx->grid.Ez = ctx->unified_grid.Ez;
    } else if (ctx->ff_mode == 2) {
        // External distilled grids will be uploaded via proposer_set_external_grids
        ctx->owns_unified_grid = false;
        ctx->grid = ProposerGridView{};
        ctx->grid.nx = nx;
        ctx->grid.ny = ny;
        ctx->grid.nz = nz;
    } else {
        ctx->owns_unified_grid = false;
        if (build_proposer_uff_grids_gpu(
                ctx->cell, unit_atoms, ctx->guest, nx, ny, nz, 11.0, 0.30, ctx->pol_mode, ctx->grid) != 0) {
            proposer_destroy(ctx);
            return nullptr;
        }
    }

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
    size_t eval_bytes = (size_t)PROP_SMEM_MAX * sizeof(double);
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_acc_N, n_b * sizeof(int)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_acc_U, n_b * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_acc_cx, n_m * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_acc_cy, n_m * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_acc_cz, n_m * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_acc_qw, n_m * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_acc_qx, n_m * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_acc_qy, n_m * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_acc_qz, n_m * sizeof(double)));

    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_tri_N, n_b * sizeof(int)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_tri_U, n_b * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_tri_dU, n_b * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_tri_acc, n_b * sizeof(int)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_tri_cx, n_m * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_tri_cy, n_m * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_tri_cz, n_m * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_tri_qw, n_m * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_tri_qx, n_m * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_tri_qy, n_m * sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_tri_qz, n_m * sizeof(double)));

    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_commit_mask, n_b * sizeof(int)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_rng, n_b * sizeof(uint64_t)));

    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_eval_cx, eval_bytes));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_eval_cy, eval_bytes));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_eval_cz, eval_bytes));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_eval_qw, eval_bytes));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_eval_qx, eval_bytes));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_eval_qy, eval_bytes));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_eval_qz, eval_bytes));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_eval_U, sizeof(double)));
    CUDA_CHECK_FAIL(cudaMalloc(&ctx->dev.d_eval_ov, sizeof(int)));

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

} // namespace

extern "C" {

int proposer_set_device(int device_id) {
    if (device_id < 0) return 0;
    CUDA_CHECK_RET(cudaSetDevice(device_id), -1);
    return 0;
}

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
    ctx->guest = make_default_co2_guest(ff_mode);

    return finalize_proposer_context(ctx, unit_atoms, nx, ny, nz, seed);
}

void* proposer_create_from_data(
    const double H_unit_cols9[9],
    int sc_a, int sc_b, int sc_c,
    int n_fw_unit,
    const double* fw_frac_3n,
    const int* fw_Z_n,
    const double* fw_q_n,
    const int* fw_type_id_n,
    int n_guest_sites,
    const double* site_lxyz_3s,
    const double* site_q_s,
    const double* site_eps_s,
    const double* site_sig_s,
    const int* site_lj_type_s,
    double guest_Tc,
    double guest_Pc,
    double guest_omega,
    int guest_is_isotropic,
    double T_K, double pressure_bar,
    int ff_mode, int pol_mode,
    int nx, int ny, int nz,
    int n_boxes, int max_mol,
    uint64_t seed) {

    if (!H_unit_cols9 || n_fw_unit <= 0 || !fw_frac_3n || !fw_Z_n || !fw_q_n ||
        n_guest_sites <= 0 || n_guest_sites > PROP_MAX_SITES ||
        !site_lxyz_3s || !site_q_s || !site_eps_s || !site_sig_s || !site_lj_type_s) {
        std::fprintf(stderr, "Error: invalid arguments to proposer_create_from_data.\n");
        return nullptr;
    }

    auto* ctx = new ProposerContext();
    ctx->T_K = T_K;
    ctx->pressure_bar = pressure_bar;
    ctx->ff_mode = ff_mode;
    ctx->pol_mode = pol_mode;
    ctx->n_boxes = std::max(1, n_boxes);
    ctx->max_mol = std::min(PROP_SMEM_MAX, std::max(16, max_mol));

    double H_u[3][3];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            H_u[r][c] = H_unit_cols9[r * 3 + c];
        }
    }
    ctx->cell.init_from_H_unit(H_u, sc_a, sc_b, sc_c);

    std::vector<FrameworkAtom> unit_atoms((size_t)n_fw_unit);
    ctx->unit_frac.resize((size_t)n_fw_unit * 3);
    ctx->unit_Z.resize((size_t)n_fw_unit);

    double unit_mass = 0.0;
    double q_sum = 0.0;
    int count_mg = 0;
    for (int i = 0; i < n_fw_unit; ++i) {
        int Z = fw_Z_n[i];
        UffParam lj = get_uff_param_by_Z(Z);
        if (Z < 1 || Z > 103) {
            std::fprintf(stderr, "Error: unsupported atomic number Z=%d in proposer_create_from_data.\n", Z);
            delete ctx;
            return nullptr;
        }
        double sa = fw_frac_3n[3 * i + 0]; sa -= std::floor(sa);
        double sb = fw_frac_3n[3 * i + 1]; sb -= std::floor(sb);
        double sc = fw_frac_3n[3 * i + 2]; sc -= std::floor(sc);
        double cx = 0.0, cy = 0.0, cz = 0.0;
        ctx->cell.unit_frac_to_cart(sa, sb, sc, cx, cy, cz);

        FrameworkAtom fa{};
        fa.x = (float)cx; fa.y = (float)cy; fa.z = (float)cz;
        fa.sa = (float)sa; fa.sb = (float)sb; fa.sc = (float)sc;
        fa.eps_K = (float)lj.eps_K;
        fa.sigma_A = (float)lj.sigma_A;
        fa.q = fw_q_n[i];
        fa.elem = atomic_number_to_elem(Z);
        fa.atom_type_id = fw_type_id_n ? fw_type_id_n[i] : -1;
        unit_atoms[i] = fa;

        ctx->unit_frac[3 * i + 0] = sa;
        ctx->unit_frac[3 * i + 1] = sb;
        ctx->unit_frac[3 * i + 2] = sc;
        ctx->unit_Z[i] = Z;

        unit_mass += get_atomic_mass_by_Z(Z);
        q_sum += fa.q;
        if (Z == 12) ++count_mg;
    }

    if (std::fabs(q_sum) > 1.0e-5) {
        if (std::fabs(q_sum) < 1.0e-2) {
            double corr = -q_sum / (double)n_fw_unit;
            for (auto& fa : unit_atoms) fa.q += corr;
        } else {
            std::fprintf(stderr, "Error: non-neutral framework charges in proposer_create_from_data: sum=%.8f e\n", q_sum);
            delete ctx;
            return nullptr;
        }
    }

    if (ff_mode == 1) {
        bool valid_mercado = (count_mg > 0) && (n_guest_sites == 3);
        for (const auto& fa : unit_atoms) {
            if (fa.atom_type_id < 0 || fa.atom_type_id >= 9) {
                valid_mercado = false;
                break;
            }
        }
        if (!valid_mercado) {
            std::fprintf(stderr, "Error: ff_mode=1 (Mercado 2016) only supports Mg-MOF-74 + 3-site CO2.\n");
            delete ctx;
            return nullptr;
        }
    }

    ctx->fw_mass_super = unit_mass * (double)(sc_a * sc_b * sc_c);
    ctx->sp = (ff_mode == 1) ? make_species_co2_epm2() : make_species_co2();

    RigidGuestDef g{};
    g.n_sites = n_guest_sites;
    g.is_isotropic = guest_is_isotropic ? 1 : 0;
    g.has_charge = 0;
    g.max_radius_A = 0.0;
    g.n_lj_types = 0;
    for (int s = 0; s < n_guest_sites; ++s) {
        double lx = site_lxyz_3s[3 * s + 0];
        double ly = site_lxyz_3s[3 * s + 1];
        double lz = site_lxyz_3s[3 * s + 2];
        double rad = std::sqrt(lx * lx + ly * ly + lz * lz);
        if (rad > g.max_radius_A) g.max_radius_A = rad;

        g.lx[s] = lx; g.ly[s] = ly; g.lz[s] = lz;
        g.q[s] = site_q_s[s];
        g.eps_K[s] = site_eps_s[s];
        g.sigma_A[s] = site_sig_s[s];
        g.lj_type[s] = site_lj_type_s[s];
        if (std::fabs(g.q[s]) > 1.0e-12) g.has_charge = 1;
        if (g.lj_type[s] >= 0) {
            if (g.lj_type[s] >= PROP_MAX_LJ) {
                std::fprintf(stderr, "Error: guest lj_type %d exceeds max %d.\n", g.lj_type[s], PROP_MAX_LJ - 1);
                delete ctx;
                return nullptr;
            }
            if (g.lj_type[s] + 1 > g.n_lj_types) g.n_lj_types = g.lj_type[s] + 1;
            g.lj_eps_K[g.lj_type[s]] = g.eps_K[s];
            g.lj_sigma_A[g.lj_type[s]] = g.sigma_A[s];
        }
    }
    if (n_guest_sites == 3 && pol_mode != 0) {
        g.w_pol[0] = trappe_co2::W_POL_C;
        g.w_pol[1] = trappe_co2::W_POL_O;
        g.w_pol[2] = trappe_co2::W_POL_O;
        g.alpha_par = trappe_co2::ALPHA_PAR_A3;
        g.alpha_perp = trappe_co2::ALPHA_PERP_A3;
        g.alpha_iso = trappe_co2::ALPHA_ISO_A3;
    }
    g.Tc_K = guest_Tc;
    g.Pc_bar = guest_Pc;
    g.omega = guest_omega;
    g.gg_q_scale = 1.0;
    g.gg_lj_scale = 1.0;
    g.gg_lut_n = 0;
    g.gg_lut_rmin = 2.0;
    g.gg_lut_rmax = 10.0;
    g.gg_lut_inv_dr = 0.0;
    ctx->guest = g;

    return finalize_proposer_context(ctx, unit_atoms, nx, ny, nz, seed);
}

void proposer_destroy(void* handle) {
    if (!handle) return;
    auto* ctx = static_cast<ProposerContext*>(handle);
    if (ctx->owns_unified_grid) {
        free_unified_grids_gpu(ctx->unified_grid);
    } else {
        free_proposer_uff_grids_gpu(ctx->grid);
    }
    if (ctx->dev.d_acc_N) cudaFree(ctx->dev.d_acc_N);
    if (ctx->dev.d_acc_U) cudaFree(ctx->dev.d_acc_U);
    if (ctx->dev.d_acc_cx) cudaFree(ctx->dev.d_acc_cx);
    if (ctx->dev.d_acc_cy) cudaFree(ctx->dev.d_acc_cy);
    if (ctx->dev.d_acc_cz) cudaFree(ctx->dev.d_acc_cz);
    if (ctx->dev.d_acc_qw) cudaFree(ctx->dev.d_acc_qw);
    if (ctx->dev.d_acc_qx) cudaFree(ctx->dev.d_acc_qx);
    if (ctx->dev.d_acc_qy) cudaFree(ctx->dev.d_acc_qy);
    if (ctx->dev.d_acc_qz) cudaFree(ctx->dev.d_acc_qz);
    if (ctx->dev.d_tri_N) cudaFree(ctx->dev.d_tri_N);
    if (ctx->dev.d_tri_U) cudaFree(ctx->dev.d_tri_U);
    if (ctx->dev.d_tri_dU) cudaFree(ctx->dev.d_tri_dU);
    if (ctx->dev.d_tri_acc) cudaFree(ctx->dev.d_tri_acc);
    if (ctx->dev.d_tri_cx) cudaFree(ctx->dev.d_tri_cx);
    if (ctx->dev.d_tri_cy) cudaFree(ctx->dev.d_tri_cy);
    if (ctx->dev.d_tri_cz) cudaFree(ctx->dev.d_tri_cz);
    if (ctx->dev.d_tri_qw) cudaFree(ctx->dev.d_tri_qw);
    if (ctx->dev.d_tri_qx) cudaFree(ctx->dev.d_tri_qx);
    if (ctx->dev.d_tri_qy) cudaFree(ctx->dev.d_tri_qy);
    if (ctx->dev.d_tri_qz) cudaFree(ctx->dev.d_tri_qz);
    if (ctx->dev.d_commit_mask) cudaFree(ctx->dev.d_commit_mask);
    if (ctx->dev.d_rng) cudaFree(ctx->dev.d_rng);
    if (ctx->dev.d_eval_cx) cudaFree(ctx->dev.d_eval_cx);
    if (ctx->dev.d_eval_cy) cudaFree(ctx->dev.d_eval_cy);
    if (ctx->dev.d_eval_cz) cudaFree(ctx->dev.d_eval_cz);
    if (ctx->dev.d_eval_qw) cudaFree(ctx->dev.d_eval_qw);
    if (ctx->dev.d_eval_qx) cudaFree(ctx->dev.d_eval_qx);
    if (ctx->dev.d_eval_qy) cudaFree(ctx->dev.d_eval_qy);
    if (ctx->dev.d_eval_qz) cudaFree(ctx->dev.d_eval_qz);
    if (ctx->dev.d_eval_U) cudaFree(ctx->dev.d_eval_U);
    if (ctx->dev.d_eval_ov) cudaFree(ctx->dev.d_eval_ov);
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
    if (!std::isfinite(delta_eps_K) || ctx->T_K <= 0.0) {
        std::fprintf(stderr, "Error: non-finite delta_eps_K=%g or invalid T_K=%g.\n", delta_eps_K, ctx->T_K);
        return -2;
    }
    double exponent = -delta_eps_K / ctx->T_K;
    if (std::abs(exponent) > 40.0) {
        std::fprintf(stderr,
                     "Error: chemical-potential shift |delta_eps_K / T_K|=%.3f > 40.0 (delta_eps_K=%.3f, T_K=%.2f).\n",
                     std::abs(exponent), delta_eps_K, ctx->T_K);
        return -2;
    }
    ctx->delta_eps_K = delta_eps_K;
    ctx->beta_f_V_ref = ctx->beta_f_V * std::exp(exponent);
    return 0;
}

int proposer_equilibrate(void* handle, int n_steps) {
    if (!handle || n_steps <= 0) return 0;
    auto* ctx = static_cast<ProposerContext*>(handle);
    propose_burst_kernel<<<ctx->n_boxes, PROP_THREADS>>>(
        ctx->cell, ctx->grid, ctx->guest,
        ctx->T_K, ctx->beta_f_V_ref, ctx->r_cut_gg, ctx->dsf_alpha,
        ctx->ff_mode, ctx->pol_mode, ctx->max_mol, n_steps, 1,
        ctx->cbmc_k, ctx->p_ins_del_total, ctx->dev);
    CUDA_CHECK_RET(cudaGetLastError(), -1);
    CUDA_CHECK_RET(cudaDeviceSynchronize(), -1);
    return 0;
}

int proposer_propose_k_quat(
    void* handle, int k_steps,
    int* out_N, double* out_U_ref_K, double* out_dU_ref_K, int* out_inner_acc,
    double* out_cx, double* out_cy, double* out_cz,
    double* out_qw, double* out_qx, double* out_qy, double* out_qz) {

    if (!handle) return -1;
    auto* ctx = static_cast<ProposerContext*>(handle);
    propose_burst_kernel<<<ctx->n_boxes, PROP_THREADS>>>(
        ctx->cell, ctx->grid, ctx->guest,
        ctx->T_K, ctx->beta_f_V_ref, ctx->r_cut_gg, ctx->dsf_alpha,
        ctx->ff_mode, ctx->pol_mode, ctx->max_mol, k_steps, 0,
        ctx->cbmc_k, ctx->p_ins_del_total, ctx->dev);
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
    CUDA_CHECK_RET(cudaMemcpy(out_qw, ctx->dev.d_tri_qw, n_m * sizeof(double), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_qx, ctx->dev.d_tri_qx, n_m * sizeof(double), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_qy, ctx->dev.d_tri_qy, n_m * sizeof(double), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_qz, ctx->dev.d_tri_qz, n_m * sizeof(double), cudaMemcpyDeviceToHost), -1);
    return 0;
}

int proposer_propose_k(
    void* handle, int k_steps,
    int* out_N, double* out_U_ref_K, double* out_dU_ref_K, int* out_inner_acc,
    double* out_cx, double* out_cy, double* out_cz,
    double* out_ux, double* out_uy, double* out_uz) {

    if (!handle) return -1;
    auto* ctx = static_cast<ProposerContext*>(handle);
    size_t n_m = (size_t)ctx->n_boxes * (size_t)ctx->max_mol;
    std::vector<double> qw(n_m), qx(n_m), qy(n_m), qz(n_m);
    int rc = proposer_propose_k_quat(
        handle, k_steps, out_N, out_U_ref_K, out_dU_ref_K, out_inner_acc,
        out_cx, out_cy, out_cz, qw.data(), qx.data(), qy.data(), qz.data());
    if (rc != 0) return rc;
    for (size_t i = 0; i < n_m; ++i) {
        quat_to_xaxis(qw[i], qx[i], qy[i], qz[i], out_ux[i], out_uy[i], out_uz[i]);
    }
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

int proposer_get_accepted_quat(
    void* handle,
    int* out_N, double* out_U_ref_K,
    double* out_cx, double* out_cy, double* out_cz,
    double* out_qw, double* out_qx, double* out_qy, double* out_qz) {

    if (!handle) return -1;
    auto* ctx = static_cast<ProposerContext*>(handle);
    size_t n_b = (size_t)ctx->n_boxes;
    size_t n_m = n_b * (size_t)ctx->max_mol;
    CUDA_CHECK_RET(cudaMemcpy(out_N, ctx->dev.d_acc_N, n_b * sizeof(int), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_U_ref_K, ctx->dev.d_acc_U, n_b * sizeof(double), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_cx, ctx->dev.d_acc_cx, n_m * sizeof(double), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_cy, ctx->dev.d_acc_cy, n_m * sizeof(double), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_cz, ctx->dev.d_acc_cz, n_m * sizeof(double), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_qw, ctx->dev.d_acc_qw, n_m * sizeof(double), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_qx, ctx->dev.d_acc_qx, n_m * sizeof(double), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_qy, ctx->dev.d_acc_qy, n_m * sizeof(double), cudaMemcpyDeviceToHost), -1);
    CUDA_CHECK_RET(cudaMemcpy(out_qz, ctx->dev.d_acc_qz, n_m * sizeof(double), cudaMemcpyDeviceToHost), -1);
    return 0;
}

int proposer_get_accepted(
    void* handle,
    int* out_N, double* out_U_ref_K,
    double* out_cx, double* out_cy, double* out_cz,
    double* out_ux, double* out_uy, double* out_uz) {

    if (!handle) return -1;
    auto* ctx = static_cast<ProposerContext*>(handle);
    size_t n_m = (size_t)ctx->n_boxes * (size_t)ctx->max_mol;
    std::vector<double> qw(n_m), qx(n_m), qy(n_m), qz(n_m);
    int rc = proposer_get_accepted_quat(
        handle, out_N, out_U_ref_K,
        out_cx, out_cy, out_cz, qw.data(), qx.data(), qy.data(), qz.data());
    if (rc != 0) return rc;
    for (size_t i = 0; i < n_m; ++i) {
        quat_to_xaxis(qw[i], qx[i], qy[i], qz[i], out_ux[i], out_uy[i], out_uz[i]);
    }
    return 0;
}

int proposer_set_accepted_quat(
    void* handle,
    const int* in_N,
    const double* in_cx, const double* in_cy, const double* in_cz,
    const double* in_qw, const double* in_qx, const double* in_qy, const double* in_qz) {

    if (!handle || !in_N) return -1;
    auto* ctx = static_cast<ProposerContext*>(handle);
    size_t n_b = (size_t)ctx->n_boxes;
    size_t n_m = n_b * (size_t)ctx->max_mol;

    std::vector<int> h_N(n_b);
    for (size_t b = 0; b < n_b; ++b) {
        if (in_N[b] < 0 || in_N[b] > ctx->max_mol) {
            std::fprintf(stderr, "Error: proposer_set_accepted_quat box %zu N=%d out of [0, %d].\n",
                         b, in_N[b], ctx->max_mol);
            return -1;
        }
        h_N[b] = in_N[b];
    }

    CUDA_CHECK_RET(cudaMemcpy(ctx->dev.d_acc_N, h_N.data(), n_b * sizeof(int), cudaMemcpyHostToDevice), -1);
    if (in_cx && in_cy && in_cz && in_qw && in_qx && in_qy && in_qz) {
        CUDA_CHECK_RET(cudaMemcpy(ctx->dev.d_acc_cx, in_cx, n_m * sizeof(double), cudaMemcpyHostToDevice), -1);
        CUDA_CHECK_RET(cudaMemcpy(ctx->dev.d_acc_cy, in_cy, n_m * sizeof(double), cudaMemcpyHostToDevice), -1);
        CUDA_CHECK_RET(cudaMemcpy(ctx->dev.d_acc_cz, in_cz, n_m * sizeof(double), cudaMemcpyHostToDevice), -1);
        CUDA_CHECK_RET(cudaMemcpy(ctx->dev.d_acc_qw, in_qw, n_m * sizeof(double), cudaMemcpyHostToDevice), -1);
        CUDA_CHECK_RET(cudaMemcpy(ctx->dev.d_acc_qx, in_qx, n_m * sizeof(double), cudaMemcpyHostToDevice), -1);
        CUDA_CHECK_RET(cudaMemcpy(ctx->dev.d_acc_qy, in_qy, n_m * sizeof(double), cudaMemcpyHostToDevice), -1);
        CUDA_CHECK_RET(cudaMemcpy(ctx->dev.d_acc_qz, in_qz, n_m * sizeof(double), cudaMemcpyHostToDevice), -1);
    }

    // k_steps = 0, commit_to_acc = 1 recomputes exact d_acc_U on GPU
    propose_burst_kernel<<<ctx->n_boxes, PROP_THREADS>>>(
        ctx->cell, ctx->grid, ctx->guest,
        ctx->T_K, ctx->beta_f_V_ref, ctx->r_cut_gg, ctx->dsf_alpha,
        ctx->ff_mode, ctx->pol_mode, ctx->max_mol, 0, 1,
        ctx->cbmc_k, ctx->p_ins_del_total, ctx->dev);
    CUDA_CHECK_RET(cudaGetLastError(), -1);
    CUDA_CHECK_RET(cudaDeviceSynchronize(), -1);
    return 0;
}

int proposer_get_rng_states(void* handle, uint64_t* out_rng_n) {
    if (!handle || !out_rng_n) return -1;
    auto* ctx = static_cast<ProposerContext*>(handle);
    CUDA_CHECK_RET(cudaMemcpy(out_rng_n, ctx->dev.d_rng, (size_t)ctx->n_boxes * sizeof(uint64_t), cudaMemcpyDeviceToHost), -1);
    return 0;
}

int proposer_set_rng_states(void* handle, const uint64_t* in_rng_n) {
    if (!handle || !in_rng_n) return -1;
    auto* ctx = static_cast<ProposerContext*>(handle);
    CUDA_CHECK_RET(cudaMemcpy(ctx->dev.d_rng, in_rng_n, (size_t)ctx->n_boxes * sizeof(uint64_t), cudaMemcpyHostToDevice), -1);
    return 0;
}

double proposer_eval_energy_quat(
    void* handle, int N,
    const double* cx, const double* cy, const double* cz,
    const double* qw, const double* qx, const double* qy, const double* qz,
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
#define EVAL_CUDA_CHECK(call)                                                   \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA Error at %s:%d: %s\n",                   \
                         __FILE__, __LINE__, cudaGetErrorString(err));          \
            if (out_overlap) *out_overlap = 1;                                  \
            return 1.0e9;                                                       \
        }                                                                       \
    } while (0)

    EVAL_CUDA_CHECK(cudaMemcpy(ctx->dev.d_eval_cx, cx, sz, cudaMemcpyHostToDevice));
    EVAL_CUDA_CHECK(cudaMemcpy(ctx->dev.d_eval_cy, cy, sz, cudaMemcpyHostToDevice));
    EVAL_CUDA_CHECK(cudaMemcpy(ctx->dev.d_eval_cz, cz, sz, cudaMemcpyHostToDevice));
    EVAL_CUDA_CHECK(cudaMemcpy(ctx->dev.d_eval_qw, qw, sz, cudaMemcpyHostToDevice));
    EVAL_CUDA_CHECK(cudaMemcpy(ctx->dev.d_eval_qx, qx, sz, cudaMemcpyHostToDevice));
    EVAL_CUDA_CHECK(cudaMemcpy(ctx->dev.d_eval_qy, qy, sz, cudaMemcpyHostToDevice));
    EVAL_CUDA_CHECK(cudaMemcpy(ctx->dev.d_eval_qz, qz, sz, cudaMemcpyHostToDevice));

    eval_single_state_kernel<<<1, PROP_THREADS>>>(
        ctx->cell, ctx->grid, ctx->guest,
        ctx->r_cut_gg, ctx->dsf_alpha, ctx->ff_mode, ctx->pol_mode, N,
        ctx->dev.d_eval_cx, ctx->dev.d_eval_cy, ctx->dev.d_eval_cz,
        ctx->dev.d_eval_qw, ctx->dev.d_eval_qx, ctx->dev.d_eval_qy, ctx->dev.d_eval_qz,
        ctx->dev.d_eval_U, ctx->dev.d_eval_ov);
    EVAL_CUDA_CHECK(cudaGetLastError());
    EVAL_CUDA_CHECK(cudaDeviceSynchronize());

    double h_U = 0.0;
    int h_ov = 0;
    EVAL_CUDA_CHECK(cudaMemcpy(&h_U, ctx->dev.d_eval_U, sizeof(double), cudaMemcpyDeviceToHost));
    EVAL_CUDA_CHECK(cudaMemcpy(&h_ov, ctx->dev.d_eval_ov, sizeof(int), cudaMemcpyDeviceToHost));
#undef EVAL_CUDA_CHECK

    if (out_overlap) *out_overlap = h_ov;
    return h_U;
}

double proposer_eval_energy(
    void* handle, int N,
    const double* cx, const double* cy, const double* cz,
    const double* ux, const double* uy, const double* uz,
    int* out_overlap) {

    if (!handle) return 0.0;
    if (N <= 0) {
        if (out_overlap) *out_overlap = 0;
        return 0.0;
    }
    std::vector<double> qw(N), qx(N), qy(N), qz(N);
    for (int i = 0; i < N; ++i) {
        xaxis_to_quat(ux[i], uy[i], uz[i], qw[i], qx[i], qy[i], qz[i]);
    }
    return proposer_eval_energy_quat(
        handle, N, cx, cy, cz, qw.data(), qx.data(), qy.data(), qz.data(), out_overlap);
}

int proposer_run_direct_benchmark(
    void* handle, int eq_steps, int prod_steps, int sample_stride,
    double* out_avg_uptake_mmol_g, double* out_sem_uptake_mmol_g,
    double* out_Qst_kJ_mol, double* out_sem_Qst_kJ_mol,
    double* out_max_drift_K) {

    if (!handle) return -1;
    auto* ctx = static_cast<ProposerContext*>(handle);
    if (!ctx->owns_unified_grid) {
        std::fprintf(stderr, "Error: proposer_run_direct_benchmark requires unified grid (ff_mode=1).\n");
        return -1;
    }
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
    run_unified_gcmc_gpu(ctx->cell, ctx->unified_grid, ctx->sp, make_species_n2(), cfg, specs, res);

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

int proposer_set_external_grids(
    void* handle,
    int nx, int ny, int nz,
    int n_lj,
    const float* h_u_lj_flat,
    const float* h_phi_elec) {

    if (!handle || nx <= 0 || ny <= 0 || nz <= 0 || n_lj <= 0 || n_lj > PROP_MAX_LJ || !h_u_lj_flat) {
        std::fprintf(stderr, "Error: invalid arguments to proposer_set_external_grids.\n");
        return -1;
    }
    auto* ctx = static_cast<ProposerContext*>(handle);
    if (ctx->owns_unified_grid) {
        free_unified_grids_gpu(ctx->unified_grid);
        ctx->owns_unified_grid = false;
    } else {
        free_proposer_uff_grids_gpu(ctx->grid);
    }

    ctx->grid = ProposerGridView{};
    ctx->grid.nx = nx;
    ctx->grid.ny = ny;
    ctx->grid.nz = nz;
    ctx->grid.n_lj = n_lj;
    ctx->grid.has_phi = (h_phi_elec != nullptr) ? 1 : 0;
    ctx->grid.has_efield = 0;

    size_t total = (size_t)nx * (size_t)ny * (size_t)nz;
    size_t bytes = total * sizeof(float);
    for (int s = 0; s < n_lj; ++s) {
        float* d_u = nullptr;
        CUDA_CHECK_RET(cudaMalloc(&d_u, bytes), -1);
        CUDA_CHECK_RET(cudaMemcpy(d_u, h_u_lj_flat + (size_t)s * total, bytes, cudaMemcpyHostToDevice), -1);
        ctx->grid.U_lj[s] = d_u;
    }
    if (h_phi_elec != nullptr) {
        float* d_phi = nullptr;
        CUDA_CHECK_RET(cudaMalloc(&d_phi, bytes), -1);
        CUDA_CHECK_RET(cudaMemcpy(d_phi, h_phi_elec, bytes, cudaMemcpyHostToDevice), -1);
        ctx->grid.Phi_elec = d_phi;
    }

    propose_burst_kernel<<<ctx->n_boxes, PROP_THREADS>>>(
        ctx->cell, ctx->grid, ctx->guest,
        ctx->T_K, ctx->beta_f_V_ref, ctx->r_cut_gg, ctx->dsf_alpha,
        ctx->ff_mode, ctx->pol_mode, ctx->max_mol, 0, 1,
        ctx->cbmc_k, ctx->p_ins_del_total, ctx->dev);
    CUDA_CHECK_RET(cudaGetLastError(), -1);
    CUDA_CHECK_RET(cudaDeviceSynchronize(), -1);
    return 0;
}

int proposer_set_gg_lut(
    void* handle,
    double gg_q_scale,
    double gg_lj_scale,
    double r_min_A,
    double r_max_A,
    int n_bins,
    const float* h_gg_com_lut_K) {

    if (!handle || !std::isfinite(gg_q_scale) || !std::isfinite(gg_lj_scale) ||
        gg_q_scale < 0.0 || gg_lj_scale < 0.0) {
        std::fprintf(stderr, "Error: invalid scaling arguments to proposer_set_gg_lut.\n");
        return -1;
    }
    auto* ctx = static_cast<ProposerContext*>(handle);
    ctx->guest.gg_q_scale = gg_q_scale;
    ctx->guest.gg_lj_scale = gg_lj_scale;
    if (h_gg_com_lut_K != nullptr && n_bins >= 2) {
        if (n_bins > 128 || !std::isfinite(r_min_A) || !std::isfinite(r_max_A) || r_max_A <= r_min_A) {
            std::fprintf(stderr, "Error: invalid LUT bounds/bins in proposer_set_gg_lut.\n");
            return -1;
        }
        ctx->guest.gg_lut_n = n_bins;
        ctx->guest.gg_lut_rmin = r_min_A;
        ctx->guest.gg_lut_rmax = r_max_A;
        ctx->guest.gg_lut_inv_dr = (double)(n_bins - 1) / (r_max_A - r_min_A);
        for (int i = 0; i < n_bins; ++i) {
            if (!std::isfinite((double)h_gg_com_lut_K[i])) {
                std::fprintf(stderr, "Error: non-finite entry in proposer_set_gg_lut.\n");
                return -1;
            }
            ctx->guest.gg_lut_K[i] = h_gg_com_lut_K[i];
        }
    } else {
        ctx->guest.gg_lut_n = 0;
        ctx->guest.gg_lut_inv_dr = 0.0;
    }

    propose_burst_kernel<<<ctx->n_boxes, PROP_THREADS>>>(
        ctx->cell, ctx->grid, ctx->guest,
        ctx->T_K, ctx->beta_f_V_ref, ctx->r_cut_gg, ctx->dsf_alpha,
        ctx->ff_mode, ctx->pol_mode, ctx->max_mol, 0, 1,
        ctx->cbmc_k, ctx->p_ins_del_total, ctx->dev);
    CUDA_CHECK_RET(cudaGetLastError(), -1);
    CUDA_CHECK_RET(cudaDeviceSynchronize(), -1);
    return 0;
}

int proposer_set_mc_params(
    void* handle,
    int cbmc_k,
    double p_ins_del_total) {

    if (!handle || cbmc_k < 1 || cbmc_k > PROP_THREADS ||
        !std::isfinite(p_ins_del_total) || p_ins_del_total < 0.10 || p_ins_del_total > 0.96) {
        std::fprintf(stderr, "Error: invalid cbmc_k=%d or p_ins_del_total=%g in proposer_set_mc_params.\n",
                     cbmc_k, p_ins_del_total);
        return -1;
    }
    auto* ctx = static_cast<ProposerContext*>(handle);
    ctx->cbmc_k = cbmc_k;
    ctx->p_ins_del_total = p_ins_del_total;
    return 0;
}

} // extern "C"
