#pragma once

#include "cell.hpp"
#include "forcefield.hpp"
#include "grid3d.hpp"
#include <vector>

namespace mofgcmc {

// 8 precomputed unit-cell periodic grids for binary mixtures (CO2, N2, CH4):
// U_lj[0] = C_CO2, U_lj[1] = O_CO2, U_lj[2] = N_N2, U_lj[3] = CH4_UA
// Phi_elec, Ex, Ey, Ez shared across all charged/polarizable species.
struct MixGrid3DView {
    int nx, ny, nz;
    const float* U_lj[4];
    const float* Phi_elec;
    const float* Ex;
    const float* Ey;
    const float* Ez;
};

struct MixGridStats {
    double build_ms;
    double mae_lj_K[4];
    double mae_phi_Ke;
    double mae_efield_KeA;
    int verify_samples;
};

struct MixGcmcConfig {
    double T_K = 298.0;
    double r_cut_A = 11.0;
    double ewald_alpha = 0.30;
    double dsf_alpha = 0.20;
    int eq_steps = 100000;
    int prod_steps = 250000;
    int sample_stride = 20;
    int max_mol = 512;
    int pol_mode = 2; // 0 = POL_NONE, 2 = POL_3SITE
    int enable_swap = 1;
    int ideal_gas_test = 0;
    double framework_mass_super_gmol = 8737.488;
    uint64_t base_seed = 20261123ULL;
};

struct MixBoxSpec {
    double pressure_bar;
    double y0;
    double y1;
    double f0_Pa;
    double f1_Pa;
    int replica_id;
    int record_hist; // 1 to accumulate 2D pore density histograms for sp0 & sp1
};

struct MixBoxResult {
    double pressure_bar;
    double y0, y1;
    double f0_Pa, f1_Pa;
    int replica_id;
    double avg_N0;
    double avg_N1;
    double std_N0;
    double std_N1;
    double uptake0_mmol_g;
    double uptake1_mmol_g;
    double selectivity_0_over_1;
    double avg_Upol0_kJ;
    double avg_Upol1_kJ;
    double acc_t;
    double acc_r;
    double acc_i0;
    double acc_d0;
    double acc_i1;
    double acc_d1;
    double acc_sw01;
    double acc_sw10;
};

__host__ __device__ inline bool eval_species_framework_grid(
    const MixGrid3DView& g, const Cell& cell,
    const SpeciesDef& sp, int pol_mode, double ke_k,
    double cx, double cy, double cz,
    double ux, double uy, double uz,
    double& u_lj_out, double& u_elec_out, double& u_pol_out,
    double& mux_out, double& muy_out, double& muz_out) {

    const double off_scale[3] = {0.0, sp.bond_A, -sp.bond_A};
    double u_lj = 0.0, u_el = 0.0, u_pol = 0.0;
    double mux = 0.0, muy = 0.0, muz = 0.0;

    for (int i = 0; i < sp.n_sites; ++i) {
        double sx = cx + off_scale[i] * ux;
        double sy = cy + off_scale[i] * uy;
        double sz = cz + off_scale[i] * uz;

        double sa, sb, sc;
        cell.cart_to_unit_frac(sx, sy, sz, sa, sb, sc);

        int lg = sp.lj_grid_idx[i];
        if (lg >= 0) {
            float ulj = trilinear_periodic(g.U_lj[lg], g.nx, g.ny, g.nz, sa, sb, sc);
            if (ulj > 1.0e5f) {
                u_lj_out = 1.0e6;
                u_elec_out = 0.0;
                u_pol_out = 0.0;
                mux_out = muy_out = muz_out = 0.0;
                return true;
            }
            u_lj += (double)ulj;
        }

        if (sp.q[i] != 0.0) {
            float phi = trilinear_periodic(g.Phi_elec, g.nx, g.ny, g.nz, sa, sb, sc);
            u_el += sp.q[i] * (double)phi;
        }

        if (pol_mode != 0 && sp.w_pol[i] > 0.0) {
            double wa = sp.w_pol[i];
            double ex = (double)trilinear_periodic(g.Ex, g.nx, g.ny, g.nz, sa, sb, sc);
            double ey = (double)trilinear_periodic(g.Ey, g.nx, g.ny, g.nz, sa, sb, sc);
            double ez = (double)trilinear_periodic(g.Ez, g.nx, g.ny, g.nz, sa, sb, sc);

            double e2 = ex * ex + ey * ey + ez * ez;
            double e_par = ex * ux + ey * uy + ez * uz;
            double e_perp2 = e2 - e_par * e_par;
            if (e_perp2 < 0.0) e_perp2 = 0.0;

            double e_perpx = ex - e_par * ux;
            double e_perpy = ey - e_par * uy;
            double e_perpz = ez - e_par * uz;

            double inv_ke = 1.0 / ke_k;
            u_pol += -0.5 * inv_ke * wa * (sp.alpha_par * e_par * e_par + sp.alpha_perp * e_perp2);
            mux += (wa * inv_ke) * (sp.alpha_par * e_par * ux + sp.alpha_perp * e_perpx);
            muy += (wa * inv_ke) * (sp.alpha_par * e_par * uy + sp.alpha_perp * e_perpy);
            muz += (wa * inv_ke) * (sp.alpha_par * e_par * uz + sp.alpha_perp * e_perpz);
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

void build_mix_framework_grids_gpu(
    const Cell& cell,
    const std::vector<FrameworkAtom>& unit_atoms,
    int nx, int ny, int nz,
    double r_cut_A, double ewald_alpha,
    MixGrid3DView& d_grid_out,
    MixGridStats& stats_out);

void free_mix_framework_grids_gpu(MixGrid3DView& d_grid);

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
    int hist_na, int hist_nb);

} // namespace mofgcmc
