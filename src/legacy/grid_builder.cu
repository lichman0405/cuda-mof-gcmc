#include "gcmc_gpu.hpp"
#include "forcefield.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace mofgcmc {

namespace {

struct KVec {
    double kx, ky, kz;
    double W_C; // W(k) * sum_j q_j cos(k.r_j)
    double W_S; // W(k) * sum_j q_j sin(k.r_j)
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

__global__ void build_grids_kernel(
    Cell cell,
    const FrameworkAtom* __restrict__ atoms,
    int n_atoms,
    const KVec* __restrict__ kvecs,
    int n_kvecs,
    int nx, int ny, int nz,
    double r_cut_A,
    double ewald_alpha,
    float* __restrict__ U_lj_C,
    float* __restrict__ U_lj_O,
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

    double ulj_c = 0.0;
    double ulj_o = 0.0;
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

        double eps_c = sqrt((double)at.eps_K * trappe_co2::EPS_C_K);
        double sig_c = 0.5 * ((double)at.sigma_A + trappe_co2::SIG_C_A);
        double sig_c2 = sig_c * sig_c;

        double eps_o = sqrt((double)at.eps_K * trappe_co2::EPS_O_K);
        double sig_o = 0.5 * ((double)at.sigma_A + trappe_co2::SIG_O_A);
        double sig_o2 = sig_o * sig_o;

        double qj = (double)at.q;

        for (int na = -1; na <= 1; ++na) {
            for (int nb = -1; nb <= 1; ++nb) {
                for (int nc = -2; nc <= 2; ++nc) {
                    double dx, dy, dz;
                    cell.unit_frac_to_cart(dsa + na, dsb + nb, dsc + nc, dx, dy, dz);
                    double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 >= r_cut2) continue;

                    if (r2 < 0.16) { // r < 0.4 A: deep atomic core
                        hard_overlap = true;
                    }
                    double r2_safe = (r2 < 0.0625) ? 0.0625 : r2; // r >= 0.25 A for Ewald stability
                    double r = sqrt(r2_safe);
                    double inv_r = 1.0 / r;
                    double inv_r2 = 1.0 / r2_safe;

                    // LJ contribution (capped at core)
                    if (!hard_overlap) {
                        double sr2_c = sig_c2 * inv_r2;
                        double sr6_c = sr2_c * sr2_c * sr2_c;
                        ulj_c += 4.0 * eps_c * (sr6_c * sr6_c - sr6_c);

                        double sr2_o = sig_o2 * inv_r2;
                        double sr6_o = sr2_o * sr2_o * sr2_o;
                        ulj_o += 4.0 * eps_o * (sr6_o * sr6_o - sr6_o);
                    }

                    // Real-space Ewald potential & E-field
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

    if (hard_overlap || ulj_c > 1.0e6) ulj_c = 1.0e6;
    if (hard_overlap || ulj_o > 1.0e6) ulj_o = 1.0e6;

    // Reciprocal-space Ewald summation
    double phi_recip = 0.0;
    double ex_recip = 0.0, ey_recip = 0.0, ez_recip = 0.0;
    for (int m = 0; m < n_kvecs; ++m) {
        KVec kv = kvecs[m];
        double kr = kv.kx * rx + kv.ky * ry + kv.kz * rz;
        double s_kr, c_kr;
        sincos(kr, &s_kr, &c_kr);

        phi_recip += kv.W_C * c_kr + kv.W_S * s_kr;
        double e_amp = kv.W_C * s_kr - kv.W_S * c_kr;
        ex_recip += e_amp * kv.kx;
        ey_recip += e_amp * kv.ky;
        ez_recip += e_amp * kv.kz;
    }

    U_lj_C[idx]   = (float)ulj_c;
    U_lj_O[idx]   = (float)ulj_o;
    Phi_elec[idx] = (float)(phi_real + phi_recip);
    Ex[idx]       = (float)(ex_real + ex_recip);
    Ey[idx]       = (float)(ey_real + ey_recip);
    Ez[idx]       = (float)(ez_real + ez_recip);
}

std::vector<KVec> build_ewald_kvecs(
    const Cell& cell,
    const std::vector<FrameworkAtom>& unit_atoms,
    double alpha,
    double k_max_Ainv) {

    constexpr double PI = 3.14159265358979323846;
    // Reciprocal lattice basis vectors = 2*pi * rows of H_unit_inv
    double b1[3] = {2.0 * PI * cell.H_unit_inv[0][0],
                    2.0 * PI * cell.H_unit_inv[0][1],
                    2.0 * PI * cell.H_unit_inv[0][2]};
    double b2[3] = {2.0 * PI * cell.H_unit_inv[1][0],
                    2.0 * PI * cell.H_unit_inv[1][1],
                    2.0 * PI * cell.H_unit_inv[1][2]};
    double b3[3] = {2.0 * PI * cell.H_unit_inv[2][0],
                    2.0 * PI * cell.H_unit_inv[2][1],
                    2.0 * PI * cell.H_unit_inv[2][2]};

    double nb1 = std::sqrt(b1[0] * b1[0] + b1[1] * b1[1] + b1[2] * b1[2]);
    double nb2 = std::sqrt(b2[0] * b2[0] + b2[1] * b2[1] + b2[2] * b2[2]);
    double nb3 = std::sqrt(b3[0] * b3[0] + b3[1] * b3[1] + b3[2] * b3[2]);

    // Conservative integer bounds for skewed 120-deg reciprocal cell
    int max_m1 = (int)std::ceil(k_max_Ainv / (nb1 * 0.5)) + 2;
    int max_m2 = (int)std::ceil(k_max_Ainv / (nb2 * 0.5)) + 2;
    int max_m3 = (int)std::ceil(k_max_Ainv / nb3) + 2;

    double kmax2 = k_max_Ainv * k_max_Ainv;
    double inv_4alpha2 = 1.0 / (4.0 * alpha * alpha);
    double prefactor = 2.0 * KE_K * (4.0 * PI / cell.volume_unit); // factor 2 for +k/-k half-space

    std::vector<KVec> kvecs;
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
                    Ck += (double)at.q * std::cos(krj);
                    Sk += (double)at.q * std::sin(krj);
                }
                KVec kv{kx, ky, kz, w * Ck, w * Sk};
                kvecs.push_back(kv);
            }
        }
    }
    return kvecs;
}

void eval_direct_point(
    const Cell& cell,
    const std::vector<FrameworkAtom>& unit_atoms,
    const std::vector<KVec>& kvecs,
    double sa, double sb, double sc,
    double r_cut_A, double alpha,
    double& ulj_c_out, double& ulj_o_out, double& phi_out,
    double& ex_out, double& ey_out, double& ez_out) {

    double rx, ry, rz;
    cell.unit_frac_to_cart(sa, sb, sc, rx, ry, rz);
    double r_cut2 = r_cut_A * r_cut_A;
    constexpr double INV_SQRT_PI = 0.56418958354775628695;
    double two_alpha_inv_sqrt_pi = 2.0 * alpha * INV_SQRT_PI;
    double alpha2 = alpha * alpha;

    double ulj_c = 0.0, ulj_o = 0.0, phi_r = 0.0;
    double ex_r = 0.0, ey_r = 0.0, ez_r = 0.0;
    for (const auto& at : unit_atoms) {
        double dsa = sa - (double)at.sa;
        double dsb = sb - (double)at.sb;
        double dsc = sc - (double)at.sc;
        dsa -= std::round(dsa);
        dsb -= std::round(dsb);
        dsc -= std::round(dsc);

        double eps_c = std::sqrt((double)at.eps_K * trappe_co2::EPS_C_K);
        double sig_c = 0.5 * ((double)at.sigma_A + trappe_co2::SIG_C_A);
        double sig_c2 = sig_c * sig_c;

        double eps_o = std::sqrt((double)at.eps_K * trappe_co2::EPS_O_K);
        double sig_o = 0.5 * ((double)at.sigma_A + trappe_co2::SIG_O_A);
        double sig_o2 = sig_o * sig_o;

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

                    double sr2_c = sig_c2 * inv_r2;
                    double sr6_c = sr2_c * sr2_c * sr2_c;
                    ulj_c += 4.0 * eps_c * (sr6_c * sr6_c - sr6_c);

                    double sr2_o = sig_o2 * inv_r2;
                    double sr6_o = sr2_o * sr2_o * sr2_o;
                    ulj_o += 4.0 * eps_o * (sr6_o * sr6_o - sr6_o);

                    double erfc_ar = std::erfc(alpha * r);
                    double exp_ar2 = std::exp(-alpha2 * r2);
                    phi_r += KE_K * at.q * erfc_ar * inv_r;

                    double e_fac = KE_K * at.q * inv_r2 * (erfc_ar * inv_r + two_alpha_inv_sqrt_pi * exp_ar2);
                    ex_r += e_fac * dx;
                    ey_r += e_fac * dy;
                    ez_r += e_fac * dz;
                }
            }
        }
    }

    double phi_k = 0.0;
    double ex_k = 0.0, ey_k = 0.0, ez_k = 0.0;
    for (const auto& kv : kvecs) {
        double kr = kv.kx * rx + kv.ky * ry + kv.kz * rz;
        double c_kr = std::cos(kr);
        double s_kr = std::sin(kr);
        phi_k += kv.W_C * c_kr + kv.W_S * s_kr;
        double e_amp = kv.W_C * s_kr - kv.W_S * c_kr;
        ex_k += e_amp * kv.kx;
        ey_k += e_amp * kv.ky;
        ez_k += e_amp * kv.kz;
    }

    ulj_c_out = ulj_c;
    ulj_o_out = ulj_o;
    phi_out = phi_r + phi_k;
    ex_out = ex_r + ex_k;
    ey_out = ey_r + ey_k;
    ez_out = ez_r + ez_k;
}

} // namespace

void build_framework_grids_gpu(
    const Cell& cell,
    const std::vector<FrameworkAtom>& unit_atoms,
    int nx, int ny, int nz,
    double r_cut_A,
    double ewald_alpha,
    Grid3DView& d_grid_out,
    GridBuildStats& stats_out) {

    size_t total = (size_t)nx * (size_t)ny * (size_t)nz;
    size_t bytes = total * sizeof(float);

    float *d_uc = nullptr, *d_uo = nullptr, *d_phi = nullptr;
    float *d_ex = nullptr, *d_ey = nullptr, *d_ez = nullptr;
    CUDA_CHECK(cudaMalloc(&d_uc, bytes));
    CUDA_CHECK(cudaMalloc(&d_uo, bytes));
    CUDA_CHECK(cudaMalloc(&d_phi, bytes));
    CUDA_CHECK(cudaMalloc(&d_ex, bytes));
    CUDA_CHECK(cudaMalloc(&d_ey, bytes));
    CUDA_CHECK(cudaMalloc(&d_ez, bytes));

    // Build Ewald k-vectors covering |k| <= 2.6 A^-1 (exp(-k^2/(4*alpha^2)) < 1e-8 for alpha=0.30)
    auto kvecs = build_ewald_kvecs(cell, unit_atoms, ewald_alpha, 2.6);

    FrameworkAtom* d_atoms = nullptr;
    KVec* d_kvecs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_atoms, unit_atoms.size() * sizeof(FrameworkAtom)));
    CUDA_CHECK(cudaMalloc(&d_kvecs, kvecs.size() * sizeof(KVec)));
    CUDA_CHECK(cudaMemcpy(d_atoms, unit_atoms.data(),
                          unit_atoms.size() * sizeof(FrameworkAtom), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kvecs, kvecs.data(),
                          kvecs.size() * sizeof(KVec), cudaMemcpyHostToDevice));

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));

    int threads = 256;
    int blocks = ((int)total + threads - 1) / threads;
    build_grids_kernel<<<blocks, threads>>>(
        cell, d_atoms, (int)unit_atoms.size(),
        d_kvecs, (int)kvecs.size(),
        nx, ny, nz, r_cut_A, ewald_alpha,
        d_uc, d_uo, d_phi, d_ex, d_ey, d_ez);
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
    d_grid_out.U_lj_C = d_uc;
    d_grid_out.U_lj_O = d_uo;
    d_grid_out.Phi_elec = d_phi;
    d_grid_out.Ex = d_ex;
    d_grid_out.Ey = d_ey;
    d_grid_out.Ez = d_ez;

    // Cross-check trilinear grid against independent direct Ewald (alpha=0.25 A^-1, kmax=2.8)
    // at 1,000 random non-overlapping pore points.
    std::vector<float> h_uc(total), h_uo(total), h_phi(total);
    std::vector<float> h_ex(total), h_ey(total), h_ez(total);
    CUDA_CHECK(cudaMemcpy(h_uc.data(), d_uc, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_uo.data(), d_uo, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_phi.data(), d_phi, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ex.data(), d_ex, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ey.data(), d_ey, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ez.data(), d_ez, bytes, cudaMemcpyDeviceToHost));

    double ref_alpha = 0.25;
    auto kvecs_ref = build_ewald_kvecs(cell, unit_atoms, ref_alpha, 2.8);

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double sum_err_c = 0.0, sum_err_o = 0.0, sum_err_phi = 0.0;
    double sum_err_e = 0.0;
    std::vector<double> fd_errs;
    int valid = 0, tries = 0;
    while (valid < 1000 && tries < 100000) {
        ++tries;
        double sa = uni(rng), sb = uni(rng), sc = uni(rng);
        float gc = trilinear_periodic(h_uc.data(), nx, ny, nz, sa, sb, sc);
        float go = trilinear_periodic(h_uo.data(), nx, ny, nz, sa, sb, sc);
        if (gc > 100.0f || go > 100.0f) continue;

        double dc, do_, dphi, dex, dey, dez;
        eval_direct_point(cell, unit_atoms, kvecs_ref, sa, sb, sc, r_cut_A, ref_alpha,
                          dc, do_, dphi, dex, dey, dez);
        if (dc > 100.0 || do_ > 100.0) continue;

        float gphi = trilinear_periodic(h_phi.data(), nx, ny, nz, sa, sb, sc);
        float gex  = trilinear_periodic(h_ex.data(), nx, ny, nz, sa, sb, sc);
        float gey  = trilinear_periodic(h_ey.data(), nx, ny, nz, sa, sb, sc);
        float gez  = trilinear_periodic(h_ez.data(), nx, ny, nz, sa, sb, sc);

        sum_err_c += std::fabs((double)gc - dc);
        sum_err_o += std::fabs((double)go - do_);
        sum_err_phi += std::fabs((double)gphi - dphi);
        double d_ex = (double)gex - dex, d_ey = (double)gey - dey, d_ez = (double)gez - dez;
        sum_err_e += std::sqrt(d_ex * d_ex + d_ey * d_ey + d_ez * d_ez);

        // Verify analytical Ewald E(r) vs central finite difference -grad(Phi) on 100 pore points
        // (using smooth full-shell summation fd_rc=1e9 with ewald_alpha=0.30 so no sphere-cutoff jump occurs)
        if ((int)fd_errs.size() < 100) {
            constexpr double h_fd = 1.0e-4;
            constexpr double fd_rc = 1.0e9;
            double fd_u1, fd_u2, fd_p0, fd_ex0, fd_ey0, fd_ez0;
            eval_direct_point(cell, unit_atoms, kvecs, sa, sb, sc, fd_rc, ewald_alpha,
                              fd_u1, fd_u2, fd_p0, fd_ex0, fd_ey0, fd_ez0);
            double rx, ry, rz;
            cell.unit_frac_to_cart(sa, sb, sc, rx, ry, rz);
            double e_fd[3] = {0.0, 0.0, 0.0};
            for (int axis = 0; axis < 3; ++axis) {
                double dx_p[3] = {rx, ry, rz};
                double dx_m[3] = {rx, ry, rz};
                dx_p[axis] += h_fd;
                dx_m[axis] -= h_fd;
                double sap, sbp, scp, sam, sbm, scm;
                cell.cart_to_unit_frac(dx_p[0], dx_p[1], dx_p[2], sap, sbp, scp);
                cell.cart_to_unit_frac(dx_m[0], dx_m[1], dx_m[2], sam, sbm, scm);
                double u1, u2, pp, pm, e1, e2, e3;
                eval_direct_point(cell, unit_atoms, kvecs, sap, sbp, scp, fd_rc, ewald_alpha,
                                  u1, u2, pp, e1, e2, e3);
                eval_direct_point(cell, unit_atoms, kvecs, sam, sbm, scm, fd_rc, ewald_alpha,
                                  u1, u2, pm, e1, e2, e3);
                e_fd[axis] = -(pp - pm) / (2.0 * h_fd);
            }
            double fx = e_fd[0] - fd_ex0, fy = e_fd[1] - fd_ey0, fz = e_fd[2] - fd_ez0;
            fd_errs.push_back(std::sqrt(fx * fx + fy * fy + fz * fz));
        }
        ++valid;
    }

    double sum_err_fd = 0.0;
    for (double v : fd_errs) sum_err_fd += v;

    stats_out.build_ms = (double)ms;
    stats_out.verify_samples = valid;
    stats_out.verify_lj_c_mae_K = (valid > 0) ? (sum_err_c / valid) : 0.0;
    stats_out.verify_lj_o_mae_K = (valid > 0) ? (sum_err_o / valid) : 0.0;
    stats_out.verify_phi_mae_Ke = (valid > 0) ? (sum_err_phi / valid) : 0.0;
    stats_out.verify_efield_mae_KeA = (valid > 0) ? (sum_err_e / valid) : 0.0;
    stats_out.verify_efield_fd_mae_KeA = (!fd_errs.empty()) ? (sum_err_fd / (double)fd_errs.size()) : 0.0;
}

void free_framework_grids_gpu(Grid3DView& d_grid) {
    if (d_grid.U_lj_C)   cudaFree((void*)d_grid.U_lj_C);
    if (d_grid.U_lj_O)   cudaFree((void*)d_grid.U_lj_O);
    if (d_grid.Phi_elec) cudaFree((void*)d_grid.Phi_elec);
    if (d_grid.Ex)       cudaFree((void*)d_grid.Ex);
    if (d_grid.Ey)       cudaFree((void*)d_grid.Ey);
    if (d_grid.Ez)       cudaFree((void*)d_grid.Ez);
    d_grid = Grid3DView{};
}

} // namespace mofgcmc
