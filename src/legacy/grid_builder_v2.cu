#include "v2_gcmc.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace mofgcmc {

namespace {

struct V2KVec {
    double kx, ky, kz;
    double W_C;
    double W_S;
};

struct V2PairTable {
    BuckOrLjPair c_pair[9]; // type_id 0..8 with C_co2
    BuckOrLjPair o_pair[9]; // type_id 0..8 with O_co2
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

double find_buck_r_peak(double A, double B, double C) {
    // Solve A*B*r^7*exp(-B*r) = 6*C on left branch of r^7*exp(-B*r) [0.05, 7/B]
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

V2PairTable build_pair_table(int ff_mode, double r_cut_A) {
    V2PairTable tbl{};
    if (ff_mode == FF_UFF_REPEAT_TRAPPE) {
        double eps_u[9] = {
            get_uff_param(ELEM_MG).eps_K,
            get_uff_param(ELEM_O).eps_K, get_uff_param(ELEM_O).eps_K, get_uff_param(ELEM_O).eps_K,
            get_uff_param(ELEM_C).eps_K, get_uff_param(ELEM_C).eps_K, get_uff_param(ELEM_C).eps_K, get_uff_param(ELEM_C).eps_K,
            get_uff_param(ELEM_H).eps_K
        };
        double sig_u[9] = {
            get_uff_param(ELEM_MG).sigma_A,
            get_uff_param(ELEM_O).sigma_A, get_uff_param(ELEM_O).sigma_A, get_uff_param(ELEM_O).sigma_A,
            get_uff_param(ELEM_C).sigma_A, get_uff_param(ELEM_C).sigma_A, get_uff_param(ELEM_C).sigma_A, get_uff_param(ELEM_C).sigma_A,
            get_uff_param(ELEM_H).sigma_A
        };
        for (int t = 0; t < 9; ++t) {
            tbl.c_pair[t] = {0, std::sqrt(eps_u[t] * trappe_co2::EPS_C_K), 0.5 * (sig_u[t] + trappe_co2::SIG_C_A), 0.0, 0.4, 0.0};
            tbl.o_pair[t] = {0, std::sqrt(eps_u[t] * trappe_co2::EPS_O_K), 0.5 * (sig_u[t] + trappe_co2::SIG_O_A), 0.0, 0.4, 0.0};
        }
        return tbl;
    }

    // FF_MERCADO2016_EPM2 (shifted at r_cut_A)
    auto make_lj_shifted = [&](double eps, double sig) -> BuckOrLjPair {
        double sr2 = (sig * sig) / (r_cut_A * r_cut_A);
        double sr6 = sr2 * sr2 * sr2;
        double us = 4.0 * eps * (sr6 * sr6 - sr6);
        return {0, eps, sig, 0.0, 0.4, us};
    };
    auto make_buck_shifted = [&](double C, double A, double B) -> BuckOrLjPair {
        double rp = find_buck_r_peak(A, B, C);
        double rc6 = std::pow(r_cut_A, 6);
        double us = A * std::exp(-B * r_cut_A) - C / rc6;
        return {1, A, B, C, rp, us};
    };

    // C_co2 with Mof_Mg..Mof_H (all Lennard-Jones)
    tbl.c_pair[0] = make_lj_shifted(190.6212, 2.816);
    tbl.c_pair[1] = make_lj_shifted(69.958, 2.794);
    tbl.c_pair[2] = make_lj_shifted(69.958, 2.794);
    tbl.c_pair[3] = make_lj_shifted(69.958, 2.794);
    tbl.c_pair[4] = make_lj_shifted(87.738, 2.904);
    tbl.c_pair[5] = make_lj_shifted(87.738, 2.904);
    tbl.c_pair[6] = make_lj_shifted(87.738, 2.904);
    tbl.c_pair[7] = make_lj_shifted(87.738, 2.904);
    tbl.c_pair[8] = make_lj_shifted(68.317, 2.453);

    // O_co2 with Mof_Mg..Mof_Cd (Buckingham: col1=C, col2=A, col3=B) & Mof_H (LJ)
    tbl.o_pair[0] = make_buck_shifted(4.08795E5,      2.47320E7,      3.965);
    tbl.o_pair[1] = make_buck_shifted(1.4313206634E5, 3.3788254669E7, 3.805);
    tbl.o_pair[2] = make_buck_shifted(1.4313206634E5, 2.6778651722E7, 3.780);
    tbl.o_pair[3] = make_buck_shifted(1.4313206634E5, 2.634324168E7,  3.705);
    tbl.o_pair[4] = make_buck_shifted(2.2631166802E5, 2.765001156E7,  3.840);
    tbl.o_pair[5] = make_buck_shifted(2.2631166802E5, 2.1483617388E7, 3.515);
    tbl.o_pair[6] = make_buck_shifted(2.2631166802E5, 2.992168789E7,  3.840);
    tbl.o_pair[7] = make_buck_shifted(2.2631166802E5, 1.314693952E7,  3.315);
    tbl.o_pair[8] = make_lj_shifted(56.900, 2.343);

    return tbl;
}

__host__ __device__ inline double eval_pair_vdw(const BuckOrLjPair& p, double r, double r2, bool& hard_ovl) {
    if (p.is_buckingham) {
        if (r <= p.r_peak) {
            hard_ovl = true;
            return 1.0e6;
        }
        double r6 = r2 * r2 * r2;
        return (p.A * exp(-p.B * r) - p.C / r6) - p.u_shift;
    } else {
        if (r < 0.4) {
            hard_ovl = true;
            return 1.0e6;
        }
        double sr2 = (p.B * p.B) / r2;
        double sr6 = sr2 * sr2 * sr2;
        return 4.0 * p.A * (sr6 * sr6 - sr6) - p.u_shift;
    }
}

__global__ void build_v2_grids_kernel(
    Cell cell,
    const FrameworkAtom* __restrict__ atoms,
    int n_atoms,
    const V2KVec* __restrict__ kvecs,
    int n_kvecs,
    V2PairTable tbl,
    int nx, int ny, int nz,
    double r_cut_A, double ewald_alpha,
    float* __restrict__ U_vdw_C,
    float* __restrict__ U_vdw_O,
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

    double uc = 0.0, uo = 0.0;
    double phi_r = 0.0, ex_r = 0.0, ey_r = 0.0, ez_r = 0.0;
    bool ovl_c = false, ovl_o = false;

    for (int j = 0; j < n_atoms; ++j) {
        FrameworkAtom at = atoms[j];
        int t_id = at.atom_type_id;
        double dsa = sa - (double)at.sa; dsa -= round(dsa);
        double dsb = sb - (double)at.sb; dsb -= round(dsb);
        double dsc = sc - (double)at.sc; dsc -= round(dsc);
        double qj = at.q;

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

                    if (!ovl_c) uc += eval_pair_vdw(tbl.c_pair[t_id], r, r2_safe, ovl_c);
                    if (!ovl_o) uo += eval_pair_vdw(tbl.o_pair[t_id], r, r2_safe, ovl_o);

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

    if (ovl_c || uc > 1.0e6) uc = 1.0e6;
    if (ovl_o || uo > 1.0e6) uo = 1.0e6;

    double phi_k = 0.0, ex_k = 0.0, ey_k = 0.0, ez_k = 0.0;
    for (int m = 0; m < n_kvecs; ++m) {
        V2KVec kv = kvecs[m];
        double kr = kv.kx * rx + kv.ky * ry + kv.kz * rz;
        double s_kr, c_kr;
        sincos(kr, &s_kr, &c_kr);
        phi_k += kv.W_C * c_kr + kv.W_S * s_kr;
        double e_amp = kv.W_C * s_kr - kv.W_S * c_kr;
        ex_k += e_amp * kv.kx;
        ey_k += e_amp * kv.ky;
        ez_k += e_amp * kv.kz;
    }

    U_vdw_C[idx]  = (float)uc;
    U_vdw_O[idx]  = (float)uo;
    Phi_elec[idx] = (float)(phi_r + phi_k);
    Ex[idx]       = (float)(ex_r + ex_k);
    Ey[idx]       = (float)(ey_r + ey_k);
    Ez[idx]       = (float)(ez_r + ez_k);
}

std::vector<V2KVec> build_v2_kvecs(
    const Cell& cell,
    const std::vector<FrameworkAtom>& unit_atoms,
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

    std::vector<V2KVec> kvecs;
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

void eval_v2_direct_point(
    const Cell& cell,
    const std::vector<FrameworkAtom>& unit_atoms,
    const std::vector<V2KVec>& kvecs,
    const V2PairTable& tbl,
    double sa, double sb, double sc,
    double r_cut_A, double alpha,
    double& uc_out, double& uo_out, double& phi_out) {

    double rx, ry, rz;
    cell.unit_frac_to_cart(sa, sb, sc, rx, ry, rz);
    double r_cut2 = r_cut_A * r_cut_A;
    double uc = 0.0, uo = 0.0, phi_r = 0.0;
    bool oc = false, oo = false;

    for (const auto& at : unit_atoms) {
        int t_id = at.atom_type_id;
        double dsa = sa - (double)at.sa; dsa -= std::round(dsa);
        double dsb = sb - (double)at.sb; dsb -= std::round(dsb);
        double dsc = sc - (double)at.sc; dsc -= std::round(dsc);

        for (int na = -1; na <= 1; ++na) {
            for (int nb = -1; nb <= 1; ++nb) {
                for (int nc = -2; nc <= 2; ++nc) {
                    double dx, dy, dz;
                    cell.unit_frac_to_cart(dsa + na, dsb + nb, dsc + nc, dx, dy, dz);
                    double r2 = dx*dx + dy*dy + dz*dz;
                    if (r2 >= r_cut2 || r2 < 1.0e-12) continue;
                    double r = std::sqrt(r2);
                    if (!oc) uc += eval_pair_vdw(tbl.c_pair[t_id], r, r2, oc);
                    if (!oo) uo += eval_pair_vdw(tbl.o_pair[t_id], r, r2, oo);
                    phi_r += KE_K * at.q * std::erfc(alpha * r) / r;
                }
            }
        }
    }
    double phi_k = 0.0;
    for (const auto& kv : kvecs) {
        double kr = kv.kx * rx + kv.ky * ry + kv.kz * rz;
        phi_k += kv.W_C * std::cos(kr) + kv.W_S * std::sin(kr);
    }
    uc_out = oc ? 1.0e6 : uc;
    uo_out = oo ? 1.0e6 : uo;
    phi_out = phi_r + phi_k;
}

} // namespace

void build_v2_grids_gpu(
    const Cell& cell,
    const std::vector<FrameworkAtom>& unit_atoms_in,
    int nx, int ny, int nz,
    double r_cut_A, double ewald_alpha, int ff_mode,
    Grid3DView& d_grid_out,
    V2GridStats& stats_out) {

    std::vector<FrameworkAtom> atoms = unit_atoms_in;
    if (ff_mode == FF_MERCADO2016_EPM2) {
        for (auto& at : atoms) {
            at.q = mercado2016::get_framework_charge(at.atom_type_id);
        }
    }

    V2PairTable tbl = build_pair_table(ff_mode, r_cut_A);
    auto kvecs = build_v2_kvecs(cell, atoms, ewald_alpha, 2.6);

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

    FrameworkAtom* d_atoms = nullptr;
    V2KVec* d_kvecs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_atoms, atoms.size() * sizeof(FrameworkAtom)));
    CUDA_CHECK(cudaMalloc(&d_kvecs, kvecs.size() * sizeof(V2KVec)));
    CUDA_CHECK(cudaMemcpy(d_atoms, atoms.data(), atoms.size() * sizeof(FrameworkAtom), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kvecs, kvecs.data(), kvecs.size() * sizeof(V2KVec), cudaMemcpyHostToDevice));

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));

    int threads = 256;
    int blocks = ((int)total + threads - 1) / threads;
    build_v2_grids_kernel<<<blocks, threads>>>(
        cell, d_atoms, (int)atoms.size(),
        d_kvecs, (int)kvecs.size(), tbl,
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

    std::vector<float> h_uc(total), h_uo(total), h_phi(total);
    CUDA_CHECK(cudaMemcpy(h_uc.data(), d_uc, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_uo.data(), d_uo, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_phi.data(), d_phi, bytes, cudaMemcpyDeviceToHost));

    double ref_alpha = 0.25;
    auto kvecs_ref = build_v2_kvecs(cell, atoms, ref_alpha, 2.8);

    std::mt19937_64 rng(999);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double sum_c = 0.0, sum_o = 0.0, sum_phi = 0.0;
    double sum_tc_c = 0.0, sum_tc_o = 0.0, sum_tc_phi = 0.0;
    int valid = 0, tries = 0;
    while (valid < 1000 && tries < 100000) {
        ++tries;
        double sa = uni(rng), sb = uni(rng), sc = uni(rng);
        float gc = trilinear_periodic(h_uc.data(), nx, ny, nz, sa, sb, sc);
        float go = trilinear_periodic(h_uo.data(), nx, ny, nz, sa, sb, sc);
        if (gc > 100.0f || go > 100.0f) continue;

        double dc, do_, dphi;
        eval_v2_direct_point(cell, atoms, kvecs_ref, tbl, sa, sb, sc, r_cut_A, ref_alpha, dc, do_, dphi);
        if (dc > 100.0 || do_ > 100.0) continue;

        float gp = trilinear_periodic(h_phi.data(), nx, ny, nz, sa, sb, sc);
        double tc_c = tricubic_periodic(h_uc.data(), nx, ny, nz, sa, sb, sc, true, 5.0e4f);
        double tc_o = tricubic_periodic(h_uo.data(), nx, ny, nz, sa, sb, sc, true, 5.0e4f);
        double tc_p = tricubic_periodic(h_phi.data(), nx, ny, nz, sa, sb, sc, false, 5.0e5f);

        sum_c += std::fabs((double)gc - dc);
        sum_o += std::fabs((double)go - do_);
        sum_phi += std::fabs((double)gp - dphi);
        sum_tc_c += std::fabs(tc_c - dc);
        sum_tc_o += std::fabs(tc_o - do_);
        sum_tc_phi += std::fabs(tc_p - dphi);
        ++valid;
    }

    // Radial check from Mg1 atom 0 toward closest pore center (r = 2.3..3.8 A)
    double rmg[3] = {atoms[0].x, atoms[0].y, atoms[0].z};
    const double pc_frac[6][2] = {{0,0}, {1.0/3.0, 2.0/3.0}, {2.0/3.0, 1.0/3.0}, {1,1}, {1.0/3.0, -1.0/3.0}, {2.0/3.0, 4.0/3.0}};
    double dir[3] = {1.0, 0.0, 0.0}, best_d = 1e9;
    for (int k = 0; k < 6; ++k) {
        double rx, ry, rz;
        cell.unit_frac_to_cart(pc_frac[k][0], pc_frac[k][1], atoms[0].sc, rx, ry, rz);
        double vx = rx - rmg[0], vy = ry - rmg[1], vz = rz - rmg[2];
        double d = std::sqrt(vx*vx + vy*vy + vz*vz);
        if (d < best_d) { best_d = d; dir[0] = vx/d; dir[1] = vy/d; dir[2] = vz/d; }
    }

    const double r_probes[7] = {2.3, 2.5, 2.7, 2.9, 3.1, 3.4, 3.8};
    double rad_sum = 0.0, rad_tc_sum = 0.0;
    for (int k = 0; k < 7; ++k) {
        double px = rmg[0] + r_probes[k]*dir[0], py = rmg[1] + r_probes[k]*dir[1], pz = rmg[2] + r_probes[k]*dir[2];
        double sa, sb, sc;
        cell.cart_to_unit_frac(px, py, pz, sa, sb, sc);
        sa -= std::floor(sa); sb -= std::floor(sb); sc -= std::floor(sc);
        float go_tri = trilinear_periodic(h_uo.data(), nx, ny, nz, sa, sb, sc);
        double go_tc = tricubic_periodic(h_uo.data(), nx, ny, nz, sa, sb, sc, true, 5.0e4f);
        double dc, do_ex, dphi;
        eval_v2_direct_point(cell, atoms, kvecs, tbl, sa, sb, sc, r_cut_A, ewald_alpha, dc, do_ex, dphi);
        rad_sum += std::fabs((double)go_tri - do_ex);
        rad_tc_sum += std::fabs(go_tc - do_ex);
    }

    stats_out.build_ms = (double)ms;
    stats_out.verify_samples = valid;
    stats_out.pore_mae_c_K = (valid > 0) ? (sum_c / valid) : 0.0;
    stats_out.pore_mae_o_K = (valid > 0) ? (sum_o / valid) : 0.0;
    stats_out.pore_mae_phi_Ke = (valid > 0) ? (sum_phi / valid) : 0.0;
    stats_out.pore_tricubic_mae_c_K = (valid > 0) ? (sum_tc_c / valid) : 0.0;
    stats_out.pore_tricubic_mae_o_K = (valid > 0) ? (sum_tc_o / valid) : 0.0;
    stats_out.pore_tricubic_mae_phi_Ke = (valid > 0) ? (sum_tc_phi / valid) : 0.0;
    stats_out.radial_256_mae_K = rad_sum / 7.0;
    stats_out.radial_256_tricubic_mae_K = rad_tc_sum / 7.0;
    stats_out.radial_128_mae_K = 0.0;
}

} // namespace mofgcmc
