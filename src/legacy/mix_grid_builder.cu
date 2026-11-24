#include "mix_gcmc.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace mofgcmc {

namespace {

struct MixKVec {
    double kx, ky, kz;
    double W_C;
    double W_S;
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

__global__ void build_mix_grids_kernel(
    Cell cell,
    const FrameworkAtom* __restrict__ atoms,
    int n_atoms,
    const MixKVec* __restrict__ kvecs,
    int n_kvecs,
    int nx, int ny, int nz,
    double r_cut_A,
    double ewald_alpha,
    float* __restrict__ U_lj0, // C_CO2
    float* __restrict__ U_lj1, // O_CO2
    float* __restrict__ U_lj2, // N_N2
    float* __restrict__ U_lj3, // CH4
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
    double two_alpha_inv_sqrt_pi = 2.0 * ewald_alpha * INV_SQRT_PI;
    double alpha2 = ewald_alpha * ewald_alpha;

    const double probe_eps[4] = {
        trappe_co2::EPS_C_K, trappe_co2::EPS_O_K, trappe_n2::EPS_N_K, trappe_ch4::EPS_CH4_K
    };
    const double probe_sig[4] = {
        trappe_co2::SIG_C_A, trappe_co2::SIG_O_A, trappe_n2::SIG_N_A, trappe_ch4::SIG_CH4_A
    };

    double ulj[4] = {0.0, 0.0, 0.0, 0.0};
    double phi_real = 0.0;
    double ex_real = 0.0, ey_real = 0.0, ez_real = 0.0;
    bool hard_overlap = false;

    for (int j = 0; j < n_atoms; ++j) {
        FrameworkAtom at = atoms[j];
        double dsa = sa - (double)at.sa;
        double dsb = sb - (double)at.sb;
        double dsc = sc - (double)at.sc;
        dsa -= round(dsa);
        dsb -= round(dsb);
        dsc -= round(dsc);

        double eps_ij[4], sig2_ij[4];
        for (int p = 0; p < 4; ++p) {
            eps_ij[p] = sqrt((double)at.eps_K * probe_eps[p]);
            double s = 0.5 * ((double)at.sigma_A + probe_sig[p]);
            sig2_ij[p] = s * s;
        }
        double qj = at.q;

        for (int na = -1; na <= 1; ++na) {
            for (int nb = -1; nb <= 1; ++nb) {
                for (int nc = -2; nc <= 2; ++nc) {
                    double dx, dy, dz;
                    cell.unit_frac_to_cart(dsa + na, dsb + nb, dsc + nc, dx, dy, dz);
                    double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 >= r_cut2) continue;

                    if (r2 < 0.16) hard_overlap = true;
                    double r2_safe = (r2 < 0.0625) ? 0.0625 : r2;
                    double r = sqrt(r2_safe);
                    double inv_r = 1.0 / r;
                    double inv_r2 = 1.0 / r2_safe;

                    if (!hard_overlap) {
                        for (int p = 0; p < 4; ++p) {
                            double sr2 = sig2_ij[p] * inv_r2;
                            double sr6 = sr2 * sr2 * sr2;
                            ulj[p] += 4.0 * eps_ij[p] * (sr6 * sr6 - sr6);
                        }
                    }

                    double ar = ewald_alpha * r;
                    double erfc_ar = erfc(ar);
                    double exp_ar2 = exp(-alpha2 * r2_safe);
                    phi_real += KE_K * qj * erfc_ar * inv_r;

                    double e_fac = KE_K * qj * inv_r2 * (erfc_ar * inv_r + two_alpha_inv_sqrt_pi * exp_ar2);
                    ex_real += e_fac * dx;
                    ey_real += e_fac * dy;
                    ez_real += e_fac * dz;
                }
            }
        }
    }

    for (int p = 0; p < 4; ++p) {
        if (hard_overlap || ulj[p] > 1.0e6) ulj[p] = 1.0e6;
    }

    double phi_recip = 0.0;
    double ex_recip = 0.0, ey_recip = 0.0, ez_recip = 0.0;
    for (int m = 0; m < n_kvecs; ++m) {
        MixKVec kv = kvecs[m];
        double kr = kv.kx * rx + kv.ky * ry + kv.kz * rz;
        double s_kr, c_kr;
        sincos(kr, &s_kr, &c_kr);

        phi_recip += kv.W_C * c_kr + kv.W_S * s_kr;
        double e_amp = kv.W_C * s_kr - kv.W_S * c_kr;
        ex_recip += e_amp * kv.kx;
        ey_recip += e_amp * kv.ky;
        ez_recip += e_amp * kv.kz;
    }

    U_lj0[idx]    = (float)ulj[0];
    U_lj1[idx]    = (float)ulj[1];
    U_lj2[idx]    = (float)ulj[2];
    U_lj3[idx]    = (float)ulj[3];
    Phi_elec[idx] = (float)(phi_real + phi_recip);
    Ex[idx]       = (float)(ex_real + ex_recip);
    Ey[idx]       = (float)(ey_real + ey_recip);
    Ez[idx]       = (float)(ez_real + ez_recip);
}

std::vector<MixKVec> build_mix_kvecs(
    const Cell& cell,
    const std::vector<FrameworkAtom>& unit_atoms,
    double alpha, double k_max_Ainv) {

    constexpr double PI = 3.14159265358979323846;
    double b1[3] = {2.0 * PI * cell.H_unit_inv[0][0], 2.0 * PI * cell.H_unit_inv[0][1], 2.0 * PI * cell.H_unit_inv[0][2]};
    double b2[3] = {2.0 * PI * cell.H_unit_inv[1][0], 2.0 * PI * cell.H_unit_inv[1][1], 2.0 * PI * cell.H_unit_inv[1][2]};
    double b3[3] = {2.0 * PI * cell.H_unit_inv[2][0], 2.0 * PI * cell.H_unit_inv[2][1], 2.0 * PI * cell.H_unit_inv[2][2]};

    double nb1 = std::sqrt(b1[0] * b1[0] + b1[1] * b1[1] + b1[2] * b1[2]);
    double nb2 = std::sqrt(b2[0] * b2[0] + b2[1] * b2[1] + b2[2] * b2[2]);
    double nb3 = std::sqrt(b3[0] * b3[0] + b3[1] * b3[1] + b3[2] * b3[2]);

    int max_m1 = (int)std::ceil(k_max_Ainv / (nb1 * 0.5)) + 2;
    int max_m2 = (int)std::ceil(k_max_Ainv / (nb2 * 0.5)) + 2;
    int max_m3 = (int)std::ceil(k_max_Ainv / nb3) + 2;

    double kmax2 = k_max_Ainv * k_max_Ainv;
    double inv_4alpha2 = 1.0 / (4.0 * alpha * alpha);
    double prefactor = 2.0 * KE_K * (4.0 * PI / cell.volume_unit);

    std::vector<MixKVec> kvecs;
    for (int mc = 0; mc <= max_m3; ++mc) {
        int mb_min = (mc == 0) ? 0 : -max_m2;
        for (int mb = mb_min; mb <= max_m2; ++mb) {
            int ma_min = (mc == 0 && mb == 0) ? 1 : -max_m1;
            for (int ma = ma_min; ma <= max_m1; ++ma) {
                double kx = ma * b1[0] + mb * b2[0] + mc * b3[0];
                double ky = ma * b1[1] + mb * b2[1] + mc * b3[1];
                double kz = ma * b1[2] + mb * b2[2] + mc * b3[2];
                double k2 = kx * kx + ky * ky + kz * kz;
                if (k2 <= 1.0e-12 || k2 > kmax2) continue;

                double w = prefactor * std::exp(-k2 * inv_4alpha2) / k2;
                double Ck = 0.0, Sk = 0.0;
                for (const auto& at : unit_atoms) {
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

void eval_mix_direct_point(
    const Cell& cell,
    const std::vector<FrameworkAtom>& unit_atoms,
    const std::vector<MixKVec>& kvecs,
    double sa, double sb, double sc,
    double r_cut_A, double alpha,
    double ulj_out[4], double& phi_out, double e_out[3]) {

    double rx, ry, rz;
    cell.unit_frac_to_cart(sa, sb, sc, rx, ry, rz);
    double r_cut2 = r_cut_A * r_cut_A;
    constexpr double INV_SQRT_PI = 0.56418958354775628695;
    double two_a = 2.0 * alpha * INV_SQRT_PI;
    double alpha2 = alpha * alpha;

    const double probe_eps[4] = {
        trappe_co2::EPS_C_K, trappe_co2::EPS_O_K, trappe_n2::EPS_N_K, trappe_ch4::EPS_CH4_K
    };
    const double probe_sig[4] = {
        trappe_co2::SIG_C_A, trappe_co2::SIG_O_A, trappe_n2::SIG_N_A, trappe_ch4::SIG_CH4_A
    };

    for (int p = 0; p < 4; ++p) ulj_out[p] = 0.0;
    double phi_r = 0.0, ex_r = 0.0, ey_r = 0.0, ez_r = 0.0;

    for (const auto& at : unit_atoms) {
        double dsa = sa - (double)at.sa; dsa -= std::round(dsa);
        double dsb = sb - (double)at.sb; dsb -= std::round(dsb);
        double dsc = sc - (double)at.sc; dsc -= std::round(dsc);

        double eps_ij[4], sig2_ij[4];
        for (int p = 0; p < 4; ++p) {
            eps_ij[p] = std::sqrt((double)at.eps_K * probe_eps[p]);
            double s = 0.5 * ((double)at.sigma_A + probe_sig[p]);
            sig2_ij[p] = s * s;
        }

        for (int na = -1; na <= 1; ++na) {
            for (int nb = -1; nb <= 1; ++nb) {
                for (int nc = -2; nc <= 2; ++nc) {
                    double dx, dy, dz;
                    cell.unit_frac_to_cart(dsa + na, dsb + nb, dsc + nc, dx, dy, dz);
                    double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 >= r_cut2 || r2 < 1.0e-12) continue;
                    double inv_r2 = 1.0 / r2;
                    double r = std::sqrt(r2);
                    double inv_r = 1.0 / r;

                    for (int p = 0; p < 4; ++p) {
                        double sr2 = sig2_ij[p] * inv_r2;
                        double sr6 = sr2 * sr2 * sr2;
                        ulj_out[p] += 4.0 * eps_ij[p] * (sr6 * sr6 - sr6);
                    }

                    double erfc_ar = std::erfc(alpha * r);
                    double exp_ar2 = std::exp(-alpha2 * r2);
                    phi_r += KE_K * at.q * erfc_ar * inv_r;
                    double e_fac = KE_K * at.q * inv_r2 * (erfc_ar * inv_r + two_a * exp_ar2);
                    ex_r += e_fac * dx;
                    ey_r += e_fac * dy;
                    ez_r += e_fac * dz;
                }
            }
        }
    }

    double phi_k = 0.0, ex_k = 0.0, ey_k = 0.0, ez_k = 0.0;
    for (const auto& kv : kvecs) {
        double kr = kv.kx * rx + kv.ky * ry + kv.kz * rz;
        double c_kr = std::cos(kr), s_kr = std::sin(kr);
        phi_k += kv.W_C * c_kr + kv.W_S * s_kr;
        double e_amp = kv.W_C * s_kr - kv.W_S * c_kr;
        ex_k += e_amp * kv.kx;
        ey_k += e_amp * kv.ky;
        ez_k += e_amp * kv.kz;
    }

    phi_out = phi_r + phi_k;
    e_out[0] = ex_r + ex_k;
    e_out[1] = ey_r + ey_k;
    e_out[2] = ez_r + ez_k;
}

} // namespace

void build_mix_framework_grids_gpu(
    const Cell& cell,
    const std::vector<FrameworkAtom>& unit_atoms,
    int nx, int ny, int nz,
    double r_cut_A, double ewald_alpha,
    MixGrid3DView& d_grid_out,
    MixGridStats& stats_out) {

    size_t total = (size_t)nx * (size_t)ny * (size_t)nz;
    size_t bytes = total * sizeof(float);

    float* d_ulj[4] = {nullptr, nullptr, nullptr, nullptr};
    float *d_phi = nullptr, *d_ex = nullptr, *d_ey = nullptr, *d_ez = nullptr;
    for (int p = 0; p < 4; ++p) CUDA_CHECK(cudaMalloc(&d_ulj[p], bytes));
    CUDA_CHECK(cudaMalloc(&d_phi, bytes));
    CUDA_CHECK(cudaMalloc(&d_ex, bytes));
    CUDA_CHECK(cudaMalloc(&d_ey, bytes));
    CUDA_CHECK(cudaMalloc(&d_ez, bytes));

    auto kvecs = build_mix_kvecs(cell, unit_atoms, ewald_alpha, 2.6);
    FrameworkAtom* d_atoms = nullptr;
    MixKVec* d_kvecs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_atoms, unit_atoms.size() * sizeof(FrameworkAtom)));
    CUDA_CHECK(cudaMalloc(&d_kvecs, kvecs.size() * sizeof(MixKVec)));
    CUDA_CHECK(cudaMemcpy(d_atoms, unit_atoms.data(),
                          unit_atoms.size() * sizeof(FrameworkAtom), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kvecs, kvecs.data(),
                          kvecs.size() * sizeof(MixKVec), cudaMemcpyHostToDevice));

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));

    int threads = 256;
    int blocks = ((int)total + threads - 1) / threads;
    build_mix_grids_kernel<<<blocks, threads>>>(
        cell, d_atoms, (int)unit_atoms.size(),
        d_kvecs, (int)kvecs.size(),
        nx, ny, nz, r_cut_A, ewald_alpha,
        d_ulj[0], d_ulj[1], d_ulj[2], d_ulj[3], d_phi, d_ex, d_ey, d_ez);
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
    for (int p = 0; p < 4; ++p) d_grid_out.U_lj[p] = d_ulj[p];
    d_grid_out.Phi_elec = d_phi;
    d_grid_out.Ex = d_ex;
    d_grid_out.Ey = d_ey;
    d_grid_out.Ez = d_ez;

    // Verify all 4 LJ grids + Phi + E-field against direct alpha_ref=0.25 at 1,000 pore pts
    std::vector<float> h_ulj[4], h_phi(total), h_ex(total), h_ey(total), h_ez(total);
    for (int p = 0; p < 4; ++p) {
        h_ulj[p].resize(total);
        CUDA_CHECK(cudaMemcpy(h_ulj[p].data(), d_ulj[p], bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaMemcpy(h_phi.data(), d_phi, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ex.data(), d_ex, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ey.data(), d_ey, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ez.data(), d_ez, bytes, cudaMemcpyDeviceToHost));

    double ref_alpha = 0.25;
    auto kvecs_ref = build_mix_kvecs(cell, unit_atoms, ref_alpha, 2.8);

    std::mt19937_64 rng(777);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double sum_lj[4] = {0.0, 0.0, 0.0, 0.0};
    double sum_phi = 0.0, sum_e = 0.0;
    int valid = 0, tries = 0;
    while (valid < 1000 && tries < 100000) {
        ++tries;
        double sa = uni(rng), sb = uni(rng), sc = uni(rng);
        bool skip = false;
        float g_lj[4];
        for (int p = 0; p < 4; ++p) {
            g_lj[p] = trilinear_periodic(h_ulj[p].data(), nx, ny, nz, sa, sb, sc);
            if (g_lj[p] > 100.0f) skip = true;
        }
        if (skip) continue;

        double d_lj[4], d_phi, d_e[3];
        eval_mix_direct_point(cell, unit_atoms, kvecs_ref, sa, sb, sc, r_cut_A, ref_alpha,
                              d_lj, d_phi, d_e);
        for (int p = 0; p < 4; ++p) {
            if (d_lj[p] > 100.0) skip = true;
        }
        if (skip) continue;

        for (int p = 0; p < 4; ++p) sum_lj[p] += std::fabs((double)g_lj[p] - d_lj[p]);
        float g_phi = trilinear_periodic(h_phi.data(), nx, ny, nz, sa, sb, sc);
        float g_ex = trilinear_periodic(h_ex.data(), nx, ny, nz, sa, sb, sc);
        float g_ey = trilinear_periodic(h_ey.data(), nx, ny, nz, sa, sb, sc);
        float g_ez = trilinear_periodic(h_ez.data(), nx, ny, nz, sa, sb, sc);

        sum_phi += std::fabs((double)g_phi - d_phi);
        double dx = (double)g_ex - d_e[0], dy = (double)g_ey - d_e[1], dz = (double)g_ez - d_e[2];
        sum_e += std::sqrt(dx * dx + dy * dy + dz * dz);
        ++valid;
    }

    stats_out.build_ms = (double)ms;
    stats_out.verify_samples = valid;
    double inv_n = (valid > 0) ? (1.0 / valid) : 0.0;
    for (int p = 0; p < 4; ++p) stats_out.mae_lj_K[p] = sum_lj[p] * inv_n;
    stats_out.mae_phi_Ke = sum_phi * inv_n;
    stats_out.mae_efield_KeA = sum_e * inv_n;
}

void free_mix_framework_grids_gpu(MixGrid3DView& d_grid) {
    for (int p = 0; p < 4; ++p) {
        if (d_grid.U_lj[p]) cudaFree((void*)d_grid.U_lj[p]);
    }
    if (d_grid.Phi_elec) cudaFree((void*)d_grid.Phi_elec);
    if (d_grid.Ex)       cudaFree((void*)d_grid.Ex);
    if (d_grid.Ey)       cudaFree((void*)d_grid.Ey);
    if (d_grid.Ez)       cudaFree((void*)d_grid.Ez);
    d_grid = MixGrid3DView{};
}

} // namespace mofgcmc
