#include "unified_gcmc.hpp"
#include "v2_gcmc.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace mofgcmc {

namespace {

struct UnifiedKVec {
    double kx, ky, kz;
    double W_C;
    double W_S;
};

struct MercadoSitePairs {
    BuckOrLjPair pair[4][9]; // [lj_grid_idx: 0=C_co2, 1=O_co2, 2=N_n2, 3=CH4][type_id: 0..8]
};

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA Error at %s:%d: %s\n",                   \
                         __FILE__, __LINE__, cudaGetErrorString(err));          \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

double find_buck_r_peak_u(double A, double B, double C) {
    double r_hi = 7.0 / B;
    double r_lo = 0.05;
    auto f = [&](double r) {
        return A * B * std::pow(r, 7) * std::exp(-B * r) - 6.0 * C;
    };
    if (f(r_lo) >= 0.0 || f(r_hi) <= 0.0) return 1.25;
    for (int it = 0; it < 80; ++it) {
        double mid = 0.5 * (r_lo + r_hi);
        if (f(mid) < 0.0) r_lo = mid;
        else r_hi = mid;
    }
    double rp = 0.5 * (r_lo + r_hi);
    return (rp > 1.25) ? rp : 1.25;
}

MercadoSitePairs build_mercado_pairs(double r_cut_A) {
    MercadoSitePairs tbl{};
    auto make_lj_shifted = [&](double eps, double sig) -> BuckOrLjPair {
        double sr2 = (sig * sig) / (r_cut_A * r_cut_A);
        double sr6 = sr2 * sr2 * sr2;
        double us = 4.0 * eps * (sr6 * sr6 - sr6);
        return {0, eps, sig, 0.0, 0.4, us};
    };
    auto make_buck_shifted = [&](double C, double A, double B) -> BuckOrLjPair {
        double rp = find_buck_r_peak_u(A, B, C);
        double rc6 = std::pow(r_cut_A, 6);
        double us = A * std::exp(-B * r_cut_A) - C / rc6;
        return {1, A, B, C, rp, us};
    };

    // 0: C_co2 (LJ)
    tbl.pair[0][0] = make_lj_shifted(190.6212, 2.816);
    tbl.pair[0][1] = make_lj_shifted(69.958,   2.794);
    tbl.pair[0][2] = make_lj_shifted(69.958,   2.794);
    tbl.pair[0][3] = make_lj_shifted(69.958,   2.794);
    tbl.pair[0][4] = make_lj_shifted(87.738,   2.904);
    tbl.pair[0][5] = make_lj_shifted(87.738,   2.904);
    tbl.pair[0][6] = make_lj_shifted(87.738,   2.904);
    tbl.pair[0][7] = make_lj_shifted(87.738,   2.904);
    tbl.pair[0][8] = make_lj_shifted(68.317,   2.453);

    // 1: O_co2 (Buckingham for Mg,O,C + LJ for H)
    tbl.pair[1][0] = make_buck_shifted(4.08795E5,      2.47320E7,      3.965);
    tbl.pair[1][1] = make_buck_shifted(1.4313206634E5, 3.3788254669E7, 3.805);
    tbl.pair[1][2] = make_buck_shifted(1.4313206634E5, 2.6778651722E7, 3.780);
    tbl.pair[1][3] = make_buck_shifted(1.4313206634E5, 2.634324168E7,  3.705);
    tbl.pair[1][4] = make_buck_shifted(2.2631166802E5, 2.765001156E7,  3.840);
    tbl.pair[1][5] = make_buck_shifted(2.2631166802E5, 2.1483617388E7, 3.515);
    tbl.pair[1][6] = make_buck_shifted(2.2631166802E5, 2.992168789E7,  3.840);
    tbl.pair[1][7] = make_buck_shifted(2.2631166802E5, 1.314693952E7,  3.315);
    tbl.pair[1][8] = make_lj_shifted(56.900, 2.343);

    // 2: N_n2 (shifted Lorentz-Berthelot with Mercado mixing rules)
    const double eps_fw[9] = {55.919, 30.2267, 30.2267, 30.2267, 52.8734, 52.8734, 52.8734, 52.8734, 22.1662};
    const double sig_fw[9] = {2.691,  3.118145, 3.118145, 3.118145, 3.430851, 3.430851, 3.430851, 3.430851, 2.571134};
    for (int t = 0; t < 9; ++t) {
        tbl.pair[2][t] = make_lj_shifted(std::sqrt(eps_fw[t] * trappe_n2::EPS_N_K),
                                         0.5 * (sig_fw[t] + trappe_n2::SIG_N_A));
    }

    // 3: CH4_sp3 (Buckingham for Mg,O,C + LJ for H from data/mercado2016/force_field.def)
    tbl.pair[3][0] = make_buck_shifted(437687.550998677, 253493890.9495950000, 4.915);
    tbl.pair[3][1] = make_buck_shifted(474764.836775231,  18427418.1646751000, 2.965);
    tbl.pair[3][2] = make_buck_shifted(474764.836775231,  20298567.6591117000, 3.265);
    tbl.pair[3][3] = make_buck_shifted(474764.836775231,  10069809.8255734000, 2.790);
    tbl.pair[3][4] = make_buck_shifted(819149.364225640,  76229282.7237860000, 3.995);
    tbl.pair[3][5] = make_buck_shifted(819149.364225640,  35513432.4787367000, 3.280);
    tbl.pair[3][6] = make_buck_shifted(819149.364225640, 231595395.4975810000, 4.370);
    tbl.pair[3][7] = make_buck_shifted(819149.364225640,   2559350.2565099000, 2.315);
    tbl.pair[3][8] = make_lj_shifted(57.24264100, 3.15);

    return tbl;
}

__host__ __device__ inline double eval_buck_or_lj(const BuckOrLjPair& p, double r, double r2, bool& ovl) {
    if (p.is_buckingham) {
        if (r <= p.r_peak) { ovl = true; return 1.0e6; }
        double r6 = r2 * r2 * r2;
        return (p.A * exp(-p.B * r) - p.C / r6) - p.u_shift;
    }
    if (r < 0.4) { ovl = true; return 1.0e6; }
    double sr2 = (p.B * p.B) / r2;
    double sr6 = sr2 * sr2 * sr2;
    return 4.0 * p.A * (sr6 * sr6 - sr6) - p.u_shift;
}

__global__ void build_unified_grids_kernel(
    Cell cell,
    const FrameworkAtom* __restrict__ atoms,
    int n_atoms,
    const UnifiedKVec* __restrict__ kvecs,
    int n_kvecs,
    MercadoSitePairs merc_tbl,
    int ff_mode,
    int nx, int ny, int nz,
    double r_cut_A, double ewald_alpha,
    float* __restrict__ U_lj0,
    float* __restrict__ U_lj1,
    float* __restrict__ U_lj2,
    float* __restrict__ U_lj3,
    float* __restrict__ Phi_elec,
    float* __restrict__ Ex,
    float* __restrict__ Ey,
    float* __restrict__ Ez) {

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

    const double eps_g[4] = {
        trappe_co2::EPS_C_K, trappe_co2::EPS_O_K, trappe_n2::EPS_N_K, trappe_ch4::EPS_CH4_K
    };
    const double sig_g[4] = {
        trappe_co2::SIG_C_A, trappe_co2::SIG_O_A, trappe_n2::SIG_N_A, trappe_ch4::SIG_CH4_A
    };

    double u[4] = {0.0, 0.0, 0.0, 0.0};
    bool ovl[4] = {false, false, false, false};
    double phi_r = 0.0, ex_r = 0.0, ey_r = 0.0, ez_r = 0.0;

    for (int j = 0; j < n_atoms; ++j) {
        FrameworkAtom at = atoms[j];
        int tid = at.atom_type_id;
        double eps_fw = (double)at.eps_K;
        double sig_fw = (double)at.sigma_A;
        double qj = at.q;

        double dsa = sa - (double)at.sa; dsa -= round(dsa);
        double dsb = sb - (double)at.sb; dsb -= round(dsb);
        double dsc = sc - (double)at.sc; dsc -= round(dsc);

        for (int na = -1; na <= 1; ++na) {
            for (int nb = -1; nb <= 1; ++nb) {
                for (int nc = -2; nc <= 2; ++nc) {
                    double dx, dy, dz;
                    cell.unit_frac_to_cart(dsa + na, dsb + nb, dsc + nc, dx, dy, dz);
                    double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 >= r_cut2) continue;

                    double r2_safe = (r2 < 0.0625) ? 0.0625 : r2;
                    double r = sqrt(r2_safe);
                    double inv_r = 1.0 / r;
                    double inv_r2 = 1.0 / r2_safe;

                    if (ff_mode == 1 && tid >= 0 && tid < 9) {
                        for (int s = 0; s < 4; ++s) {
                            if (!ovl[s]) u[s] += eval_buck_or_lj(merc_tbl.pair[s][tid], r, r2_safe, ovl[s]);
                        }
                    } else {
                        for (int s = 0; s < 4; ++s) {
                            if (ovl[s]) continue;
                            if (r < 0.4) { ovl[s] = true; continue; }
                            double sig_ij = 0.5 * (sig_fw + sig_g[s]);
                            double eps_ij = sqrt(eps_fw * eps_g[s]);
                            double sr2 = (sig_ij * sig_ij) * inv_r2;
                            double sr6 = sr2 * sr2 * sr2;
                            u[s] += 4.0 * eps_ij * (sr6 * sr6 - sr6);
                        }
                    }

                    double ar = ewald_alpha * r;
                    double erfc_ar = erfc(ar);
                    double exp_ar2 = exp(-alpha2 * r2_safe);
                    phi_r += KE_K * qj * erfc_ar * inv_r;
                    double e_fac = KE_K * qj * inv_r2 * (erfc_ar * inv_r + two_a * exp_ar2);
                    ex_r += e_fac * dx;
                    ey_r += e_fac * dy;
                    ez_r += e_fac * dz;
                }
            }
        }
    }

    for (int s = 0; s < 4; ++s) {
        if (ovl[s] || u[s] > 1.0e6) u[s] = 1.0e6;
    }

    double phi_k = 0.0, ex_k = 0.0, ey_k = 0.0, ez_k = 0.0;
    for (int m = 0; m < n_kvecs; ++m) {
        UnifiedKVec kv = kvecs[m];
        double kr = kv.kx * rx + kv.ky * ry + kv.kz * rz;
        double s_kr, c_kr;
        sincos(kr, &s_kr, &c_kr);
        phi_k += kv.W_C * c_kr + kv.W_S * s_kr;
        double e_amp = kv.W_C * s_kr - kv.W_S * c_kr;
        ex_k += e_amp * kv.kx;
        ey_k += e_amp * kv.ky;
        ez_k += e_amp * kv.kz;
    }

    U_lj0[idx] = (float)u[0];
    U_lj1[idx] = (float)u[1];
    U_lj2[idx] = (float)u[2];
    U_lj3[idx] = (float)u[3];
    Phi_elec[idx] = (float)(phi_r + phi_k);
    Ex[idx] = (float)(ex_r + ex_k);
    Ey[idx] = (float)(ey_r + ey_k);
    Ez[idx] = (float)(ez_r + ez_k);
}

std::vector<UnifiedKVec> build_unified_kvecs(
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

    int max_m1 = (int)std::ceil(k_max_Ainv / (nb1 * 0.5)) + 2;
    int max_m2 = (int)std::ceil(k_max_Ainv / (nb2 * 0.5)) + 2;
    int max_m3 = (int)std::ceil(k_max_Ainv / nb3) + 2;

    double kmax2 = k_max_Ainv * k_max_Ainv;
    double inv_4alpha2 = 1.0 / (4.0 * alpha * alpha);
    double prefactor = 2.0 * KE_K * (4.0 * PI / cell.volume_unit);

    std::vector<UnifiedKVec> kvecs;
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

void eval_unified_direct(
    const Cell& cell,
    const std::vector<FrameworkAtom>& atoms,
    const std::vector<UnifiedKVec>& kvecs,
    const MercadoSitePairs& merc_tbl,
    int ff_mode,
    double sa, double sb, double sc,
    double r_cut_A, double alpha,
    double u_out[4], double& phi_out, double e_out[3]) {

    double rx, ry, rz;
    cell.unit_frac_to_cart(sa, sb, sc, rx, ry, rz);

    double r_cut2 = r_cut_A * r_cut_A;
    constexpr double INV_SQRT_PI = 0.56418958354775628695;
    double two_a = 2.0 * alpha * INV_SQRT_PI;
    double alpha2 = alpha * alpha;

    const double eps_g[4] = {
        trappe_co2::EPS_C_K, trappe_co2::EPS_O_K, trappe_n2::EPS_N_K, trappe_ch4::EPS_CH4_K
    };
    const double sig_g[4] = {
        trappe_co2::SIG_C_A, trappe_co2::SIG_O_A, trappe_n2::SIG_N_A, trappe_ch4::SIG_CH4_A
    };

    for (int s = 0; s < 4; ++s) u_out[s] = 0.0;
    bool ovl[4] = {false, false, false, false};
    double phi_r = 0.0, ex_r = 0.0, ey_r = 0.0, ez_r = 0.0;

    for (const auto& at : atoms) {
        int tid = at.atom_type_id;
        double eps_fw = at.eps_K, sig_fw = at.sigma_A, qj = at.q;
        double dsa = sa - (double)at.sa; dsa -= std::round(dsa);
        double dsb = sb - (double)at.sb; dsb -= std::round(dsb);
        double dsc = sc - (double)at.sc; dsc -= std::round(dsc);

        for (int na = -1; na <= 1; ++na) {
            for (int nb = -1; nb <= 1; ++nb) {
                for (int nc = -2; nc <= 2; ++nc) {
                    double dx, dy, dz;
                    cell.unit_frac_to_cart(dsa + na, dsb + nb, dsc + nc, dx, dy, dz);
                    double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 >= r_cut2 || r2 < 1.0e-12) continue;

                    double r = std::sqrt(r2);
                    double inv_r = 1.0 / r, inv_r2 = 1.0 / r2;

                    if (ff_mode == 1 && tid >= 0 && tid < 9) {
                        for (int s = 0; s < 4; ++s) {
                            if (!ovl[s]) u_out[s] += eval_buck_or_lj(merc_tbl.pair[s][tid], r, r2, ovl[s]);
                        }
                    } else {
                        for (int s = 0; s < 4; ++s) {
                            if (ovl[s]) continue;
                            if (r < 0.4) { ovl[s] = true; continue; }
                            double sig_ij = 0.5 * (sig_fw + sig_g[s]);
                            double eps_ij = std::sqrt(eps_fw * eps_g[s]);
                            double sr2 = (sig_ij * sig_ij) * inv_r2;
                            double sr6 = sr2 * sr2 * sr2;
                            u_out[s] += 4.0 * eps_ij * (sr6 * sr6 - sr6);
                        }
                    }

                    double ar = alpha * r;
                    double erfc_ar = std::erfc(ar);
                    double exp_ar2 = std::exp(-alpha2 * r2);
                    phi_r += KE_K * qj * erfc_ar * inv_r;
                    double ef = KE_K * qj * inv_r2 * (erfc_ar * inv_r + two_a * exp_ar2);
                    ex_r += ef * dx; ey_r += ef * dy; ez_r += ef * dz;
                }
            }
        }
    }

    for (int s = 0; s < 4; ++s) if (ovl[s]) u_out[s] = 1.0e6;

    double phi_k = 0.0, ex_k = 0.0, ey_k = 0.0, ez_k = 0.0;
    for (const auto& kv : kvecs) {
        double kr = kv.kx * rx + kv.ky * ry + kv.kz * rz;
        double sk = std::sin(kr), ck = std::cos(kr);
        phi_k += kv.W_C * ck + kv.W_S * sk;
        double ea = kv.W_C * sk - kv.W_S * ck;
        ex_k += ea * kv.kx; ey_k += ea * kv.ky; ez_k += ea * kv.kz;
    }

    phi_out = phi_r + phi_k;
    e_out[0] = ex_r + ex_k; e_out[1] = ey_r + ey_k; e_out[2] = ez_r + ez_k;
}

} // namespace

void build_unified_grids_gpu(
    const Cell& cell,
    const std::vector<FrameworkAtom>& unit_atoms_in,
    int nx, int ny, int nz,
    double r_cut_A, double ewald_alpha, int ff_mode,
    MixGrid3DView& d_grid_out,
    UnifiedGridStats& stats_out) {

    std::vector<FrameworkAtom> atoms = unit_atoms_in;
    if (ff_mode == 1) {
        for (auto& at : atoms) {
            if (at.atom_type_id >= 0 && at.atom_type_id < 9) {
                at.q = mercado2016::get_framework_charge(at.atom_type_id);
            }
        }
    }

    MercadoSitePairs merc_tbl = build_mercado_pairs(r_cut_A);
    auto kvecs = build_unified_kvecs(cell, atoms, ewald_alpha, 2.6);

    size_t total = (size_t)nx * (size_t)ny * (size_t)nz;
    size_t bytes = total * sizeof(float);

    float *d_u[4] = {nullptr, nullptr, nullptr, nullptr};
    float *d_phi = nullptr, *d_ex = nullptr, *d_ey = nullptr, *d_ez = nullptr;
    for (int s = 0; s < 4; ++s) CUDA_CHECK(cudaMalloc(&d_u[s], bytes));
    CUDA_CHECK(cudaMalloc(&d_phi, bytes));
    CUDA_CHECK(cudaMalloc(&d_ex, bytes));
    CUDA_CHECK(cudaMalloc(&d_ey, bytes));
    CUDA_CHECK(cudaMalloc(&d_ez, bytes));

    FrameworkAtom* d_atoms = nullptr;
    UnifiedKVec* d_kvecs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_atoms, atoms.size() * sizeof(FrameworkAtom)));
    CUDA_CHECK(cudaMalloc(&d_kvecs, kvecs.size() * sizeof(UnifiedKVec)));
    CUDA_CHECK(cudaMemcpy(d_atoms, atoms.data(), atoms.size() * sizeof(FrameworkAtom), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kvecs, kvecs.data(), kvecs.size() * sizeof(UnifiedKVec), cudaMemcpyHostToDevice));

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));

    int threads = 256;
    int blocks = ((int)total + threads - 1) / threads;
    build_unified_grids_kernel<<<blocks, threads>>>(
        cell, d_atoms, (int)atoms.size(), d_kvecs, (int)kvecs.size(), merc_tbl, ff_mode,
        nx, ny, nz, r_cut_A, ewald_alpha,
        d_u[0], d_u[1], d_u[2], d_u[3], d_phi, d_ex, d_ey, d_ez);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaFree(d_atoms));
    CUDA_CHECK(cudaFree(d_kvecs));

    d_grid_out.nx = nx;
    d_grid_out.ny = ny;
    d_grid_out.nz = nz;
    for (int s = 0; s < 4; ++s) d_grid_out.U_lj[s] = d_u[s];
    d_grid_out.Phi_elec = d_phi;
    d_grid_out.Ex = d_ex;
    d_grid_out.Ey = d_ey;
    d_grid_out.Ez = d_ez;

    std::vector<float> h_u[4], h_phi(total), h_ex(total), h_ey(total), h_ez(total);
    for (int s = 0; s < 4; ++s) {
        h_u[s].resize(total);
        CUDA_CHECK(cudaMemcpy(h_u[s].data(), d_u[s], bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaMemcpy(h_phi.data(), d_phi, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ex.data(),  d_ex,  bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ey.data(),  d_ey,  bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ez.data(),  d_ez,  bytes, cudaMemcpyDeviceToHost));

    double ref_alpha = 0.25;
    auto kvecs_ref = build_unified_kvecs(cell, atoms, ref_alpha, 2.8);

    std::mt19937_64 rng(777);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double sum_lj[4] = {0, 0, 0, 0}, sum_phi = 0.0, sum_e = 0.0;
    int valid = 0, tries = 0;
    while (valid < 500 && tries < 50000) {
        ++tries;
        double sa = uni(rng), sb = uni(rng), sc = uni(rng);
        double tc_u[4];
        bool bad = false;
        for (int s = 0; s < 4; ++s) {
            tc_u[s] = tricubic_periodic(h_u[s].data(), nx, ny, nz, sa, sb, sc, true, 5.0e4f);
            if (tc_u[s] > 100.0) { bad = true; break; }
        }
        if (bad) continue;

        double du[4], dphi, de[3];
        eval_unified_direct(cell, atoms, kvecs_ref, merc_tbl, ff_mode, sa, sb, sc, r_cut_A, ref_alpha, du, dphi, de);
        if (du[0] > 100.0 || du[1] > 100.0 || du[2] > 100.0 || du[3] > 100.0) continue;

        double tp = tricubic_periodic(h_phi.data(), nx, ny, nz, sa, sb, sc, false, 5.0e5f);
        double tx = tricubic_periodic(h_ex.data(),  nx, ny, nz, sa, sb, sc, false, 5.0e5f);
        double ty = tricubic_periodic(h_ey.data(),  nx, ny, nz, sa, sb, sc, false, 5.0e5f);
        double tz = tricubic_periodic(h_ez.data(),  nx, ny, nz, sa, sb, sc, false, 5.0e5f);

        for (int s = 0; s < 4; ++s) sum_lj[s] += std::fabs(tc_u[s] - du[s]);
        sum_phi += std::fabs(tp - dphi);
        double dex = tx - de[0], dey = ty - de[1], dez = tz - de[2];
        sum_e += std::sqrt(dex * dex + dey * dey + dez * dez);
        ++valid;
    }

    stats_out.build_ms = (double)ms;
    stats_out.verify_samples = valid;
    for (int s = 0; s < 4; ++s) {
        stats_out.pore_tricubic_mae_lj_K[s] = (valid > 0) ? (sum_lj[s] / valid) : 0.0;
    }
    stats_out.pore_tricubic_mae_phi_Ke = (valid > 0) ? (sum_phi / valid) : 0.0;
    stats_out.pore_tricubic_mae_efield_KeA = (valid > 0) ? (sum_e / valid) : 0.0;
}

void free_unified_grids_gpu(MixGrid3DView& d_grid) {
    for (int s = 0; s < 4; ++s) {
        if (d_grid.U_lj[s]) cudaFree((void*)d_grid.U_lj[s]);
        d_grid.U_lj[s] = nullptr;
    }
    if (d_grid.Phi_elec) cudaFree((void*)d_grid.Phi_elec);
    if (d_grid.Ex) cudaFree((void*)d_grid.Ex);
    if (d_grid.Ey) cudaFree((void*)d_grid.Ey);
    if (d_grid.Ez) cudaFree((void*)d_grid.Ez);
    d_grid.Phi_elec = d_grid.Ex = d_grid.Ey = d_grid.Ez = nullptr;
}

} // namespace mofgcmc
