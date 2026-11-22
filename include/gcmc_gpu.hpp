#pragma once

#include "cell.hpp"
#include "grid3d.hpp"
#include <vector>

namespace mofgcmc {

struct GridBuildStats {
    double build_ms;
    double verify_lj_c_mae_K;
    double verify_lj_o_mae_K;
    double verify_phi_mae_Ke;
    double verify_efield_mae_KeA;
    double verify_efield_fd_mae_KeA;
    int verify_samples;
};

enum PolMode {
    POL_NONE  = 0,
    POL_COM   = 1,
    POL_3SITE = 2
};

struct BatchedGcmcConfig {
    double T_K = 298.0;
    double r_cut_A = 11.0;
    double ewald_alpha = 0.30;
    double dsf_alpha = 0.20;
    int eq_steps = 100000;
    int prod_steps = 200000;
    int sample_stride = 20;
    int max_mol = 512;
    int pol_mode = POL_NONE;
    double pol_scale = 1.0;
    double framework_mass_super_gmol = 8738.46;
    uint64_t base_seed = 20261122ULL;
};

struct BoxSpec {
    double pressure_bar;
    double fugacity_Pa;
    int replica_id;
    int record_density_hist; // 1 for 1.0 bar box(es), 0 otherwise
};

struct BoxResult {
    double pressure_bar;
    double fugacity_Pa;
    int replica_id;
    double avg_N;
    double std_N;
    double uptake_mmol_g;
    double avg_U_K;
    double avg_U_pol_per_mol_kJ;
    double Qst_kJ_mol;
    double acc_trans;
    double acc_rot;
    double acc_ins;
    double acc_del;
    int final_N;
};

// Grid builder & verifier (src/grid_builder.cu)
void build_framework_grids_gpu(
    const Cell& cell,
    const std::vector<FrameworkAtom>& unit_atoms,
    int nx, int ny, int nz,
    double r_cut_A,
    double ewald_alpha,
    Grid3DView& d_grid_out,
    GridBuildStats& stats_out);

void free_framework_grids_gpu(Grid3DView& d_grid);

// Batched GCMC runner (src/gcmc_kernel.cu)
double run_batched_gcmc_gpu(
    const Cell& cell,
    const Grid3DView& d_grid,
    const BatchedGcmcConfig& cfg,
    const std::vector<BoxSpec>& boxes,
    std::vector<BoxResult>& results_out,
    std::vector<double>& density_hist_out,
    int hist_na, int hist_nb);

} // namespace mofgcmc
