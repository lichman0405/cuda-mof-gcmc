#pragma once

#include "cell.hpp"
#include "forcefield.hpp"
#include "grid3d.hpp"
#include "mix_gcmc.hpp"
#include <vector>

namespace mofgcmc {

struct UnifiedGridStats {
    double build_ms;
    double pore_tricubic_mae_lj_K[4]; // 0=C_CO2, 1=O_CO2, 2=N_N2, 3=CH4
    double pore_tricubic_mae_phi_Ke;
    double pore_tricubic_mae_efield_KeA;
    int verify_samples;
};

struct UnifiedGcmcConfig {
    double T_K = 298.0;
    double r_cut_A = 11.0;
    double ewald_alpha = 0.30;
    double dsf_alpha = 0.20;
    int ff_mode = 0;            // 0 = FF_UFF_REPEAT (unshifted), 1 = FF_MERCADO2016 (shifted Buckingham/LJ)
    int use_tricubic = 1;       // 1 = O(h^4) Lagrange tricubic, 0 = trilinear
    int pol_mode = 0;           // 0 = POL_NONE, 1 = POL_COM, 2 = POL_3SITE
    int n_comp = 1;             // 1 = single-component, 2 = binary mixture
    int enable_swap = 1;        // Binary identity swap flag
    int eq_steps = 200000;
    int prod_steps = 800000;
    int sample_stride = 20;
    int recompute_stride = 50000;
    int max_mol = 512;
    double framework_mass_super_gmol = 8737.488;
    uint64_t base_seed = 20261123ULL;
};

struct UnifiedBoxSpec {
    double pressure_bar;
    double y0;
    double y1;
    double f0_Pa;
    double f1_Pa;
    int replica_id;
};

struct UnifiedBoxResult {
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
    double Qst_kJ_mol;
    double avg_Upol0_kJ;
    double avg_Upol1_kJ;
    double max_energy_drift_K;
    double acc_t;
    double acc_r;
    double acc_i0;
    double acc_d0;
    double acc_i1;
    double acc_d1;
    double acc_sw01;
    double acc_sw10;
};

__host__ __device__ inline bool eval_species_framework_grid_unified(
    const MixGrid3DView& g, const Cell& cell,
    const SpeciesDef& sp, int use_tricubic, int pol_mode, double ke_k,
    double cx, double cy, double cz,
    double ux, double uy, double uz,
    double& u_lj_out, double& u_elec_out, double& u_pol_out,
    double& mux_out, double& muy_out, double& muz_out) {

    const double off_scale[3] = {0.0, sp.bond_A, -sp.bond_A};
    double sa[3], sb[3], sc[3];
    double u_lj = 0.0;

    // Pass 1: VDW overlap check across all sites before touching electrostatic/E-field grids
    for (int i = 0; i < sp.n_sites; ++i) {
        double sx = cx + off_scale[i] * ux;
        double sy = cy + off_scale[i] * uy;
        double sz = cz + off_scale[i] * uz;
        cell.cart_to_unit_frac(sx, sy, sz, sa[i], sb[i], sc[i]);

        int lg = sp.lj_grid_idx[i];
        if (lg >= 0) {
            double ulj = use_tricubic
                ? tricubic_periodic(g.U_lj[lg], g.nx, g.ny, g.nz, sa[i], sb[i], sc[i], true, 5.0e4f)
                : (double)trilinear_periodic(g.U_lj[lg], g.nx, g.ny, g.nz, sa[i], sb[i], sc[i]);
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

    // Pass 2: Electrostatics + optional COM / distributed anisotropic polarization
    double u_el = 0.0, u_pol = 0.0;
    double mux = 0.0, muy = 0.0, muz = 0.0;
    for (int i = 0; i < sp.n_sites; ++i) {
        if (sp.q[i] != 0.0) {
            double phi = use_tricubic
                ? tricubic_periodic(g.Phi_elec, g.nx, g.ny, g.nz, sa[i], sb[i], sc[i], false, 5.0e5f)
                : (double)trilinear_periodic(g.Phi_elec, g.nx, g.ny, g.nz, sa[i], sb[i], sc[i]);
            u_el += sp.q[i] * phi;
        }

        double wa = 0.0;
        if (pol_mode == 1 && i == 0) wa = 1.0;
        else if (pol_mode == 2) wa = sp.w_pol[i];

        if (wa > 0.0) {
            double ex = use_tricubic
                ? tricubic_periodic(g.Ex, g.nx, g.ny, g.nz, sa[i], sb[i], sc[i], false, 5.0e5f)
                : (double)trilinear_periodic(g.Ex, g.nx, g.ny, g.nz, sa[i], sb[i], sc[i]);
            double ey = use_tricubic
                ? tricubic_periodic(g.Ey, g.nx, g.ny, g.nz, sa[i], sb[i], sc[i], false, 5.0e5f)
                : (double)trilinear_periodic(g.Ey, g.nx, g.ny, g.nz, sa[i], sb[i], sc[i]);
            double ez = use_tricubic
                ? tricubic_periodic(g.Ez, g.nx, g.ny, g.nz, sa[i], sb[i], sc[i], false, 5.0e5f)
                : (double)trilinear_periodic(g.Ez, g.nx, g.ny, g.nz, sa[i], sb[i], sc[i]);

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

void build_unified_grids_gpu(
    const Cell& cell,
    const std::vector<FrameworkAtom>& unit_atoms_in,
    int nx, int ny, int nz,
    double r_cut_A, double ewald_alpha, int ff_mode,
    MixGrid3DView& d_grid_out,
    UnifiedGridStats& stats_out);

void free_unified_grids_gpu(MixGrid3DView& d_grid);

double run_unified_gcmc_gpu(
    const Cell& cell,
    const MixGrid3DView& d_grid,
    const SpeciesDef& sp0,
    const SpeciesDef& sp1,
    const UnifiedGcmcConfig& cfg,
    const std::vector<UnifiedBoxSpec>& boxes,
    std::vector<UnifiedBoxResult>& results_out);

} // namespace mofgcmc
