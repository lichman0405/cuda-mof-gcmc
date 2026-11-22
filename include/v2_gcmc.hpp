#pragma once

#include "cell.hpp"
#include "forcefield.hpp"
#include "grid3d.hpp"
#include <vector>

namespace mofgcmc {

enum V2ForceFieldMode {
    FF_UFF_REPEAT_TRAPPE = 0, // Unshifted 12-6 LJ (r_cut = 11 A) + REPEAT + TraPPE CO2
    FF_MERCADO2016_EPM2  = 1  // Shifted Mercado 2016 DFT Buckingham/LJ + REPEAT + EPM2 CO2
};

struct BuckOrLjPair {
    int is_buckingham; // 1 = Buckingham (A*exp(-B*r) - C/r^6), 0 = Lennard-Jones 12-6
    double A;          // K (or eps_K if LJ)
    double B;          // 1/A (or sigma_A if LJ)
    double C;          // K * A^6 (Buckingham)
    double r_peak;     // Exact turnover radius for hacked Buckingham clamp
    double u_shift;    // U(r_cut) subtraction when shifted == 1
};

struct V2GridStats {
    double build_ms;
    double radial_128_mae_K; // Trilinear vs exact along Mg1->pore (2.3-3.8 A) on 128x128x36
    double radial_256_mae_K; // Trilinear vs exact along Mg1->pore (2.3-3.8 A) on 256x256x72
    double pore_mae_c_K;
    double pore_mae_o_K;
    double pore_mae_phi_Ke;
    int verify_samples;
};

struct V2GcmcConfig {
    double T_K = 298.0;
    double r_cut_A = 11.0;
    double ewald_alpha = 0.30;
    double dsf_alpha = 0.20;
    int ff_mode = FF_UFF_REPEAT_TRAPPE;
    int eq_steps = 200000;
    int prod_steps = 800000;
    int sample_stride = 20;
    int recompute_stride = 50000;
    int max_mol = 512;
    double framework_mass_super_gmol = 8737.488;
    uint64_t base_seed = 20261123ULL;
};

struct V2BoxSpec {
    double pressure_bar;
    double fugacity_Pa;
    int replica_id;
    int record_rdf; // 1 for 0.15 bar boxes
};

struct V2BoxResult {
    double pressure_bar;
    double fugacity_Pa;
    int replica_id;
    double avg_N;
    double std_N;
    double uptake_mmol_g;
    double avg_U_K;
    double Qst_kJ_mol;
    double max_energy_drift_K;
    double mean_min_mg_o_A;
    double acc_trans;
    double acc_rot;
    double acc_ins;
    double acc_del;
};

void build_v2_grids_gpu(
    const Cell& cell,
    const std::vector<FrameworkAtom>& unit_atoms_in,
    int nx, int ny, int nz,
    double r_cut_A, double ewald_alpha, int ff_mode,
    Grid3DView& d_grid_out,
    V2GridStats& stats_out);

double run_v2_gcmc_gpu(
    const Cell& cell,
    const Grid3DView& d_grid,
    const std::vector<FrameworkAtom>& mg_unit_atoms,
    const V2GcmcConfig& cfg,
    const std::vector<V2BoxSpec>& boxes,
    std::vector<V2BoxResult>& results_out,
    std::vector<double>& rdf_hist_out,
    int rdf_bins, double rdf_r_min, double rdf_r_max);

} // namespace mofgcmc
