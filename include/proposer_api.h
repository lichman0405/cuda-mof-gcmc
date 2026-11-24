#pragma once

#include <cstdint>

#ifdef _WIN32
#define PROPOSER_API __declspec(dllexport)
#else
#define PROPOSER_API __attribute__((visibility("default")))
#endif

extern "C" {

// Create a GPU batched GCMC proposal / evaluation context.
// ff_mode : 0 = UFF + REPEAT (TraPPE CO2), 1 = Mercado 2016 DFT-FF (EPM2 CO2)
// pol_mode: 0 = NonPol, 2 = Pol-3Site (1st-order framework E-field + Thole coupling)
PROPOSER_API void* proposer_create(
    const char* cif_path,
    int sc_a, int sc_b, int sc_c,
    double T_K, double pressure_bar,
    int ff_mode, int pol_mode,
    int nx, int ny, int nz,
    int n_boxes, int max_mol,
    uint64_t seed);

PROPOSER_API void proposer_destroy(void* handle);

PROPOSER_API double proposer_get_framework_mass(void* handle);
PROPOSER_API double proposer_get_fugacity_Pa(void* handle);
PROPOSER_API double proposer_get_rcut_gg(void* handle);
PROPOSER_API void proposer_get_cell_H(void* handle, double out_H9[9]);
PROPOSER_API int proposer_get_num_unit_atoms(void* handle);
PROPOSER_API int proposer_get_unit_atoms(void* handle, double* out_frac_3n, int* out_Z_n);
PROPOSER_API double proposer_get_min_perp_width(void* handle);

// Set chemical-potential shift for inner proposer sampling:
// f_ref = f_tgt * exp(-delta_eps_K / T_K), where delta_eps_K = <(U_tgt - U_ref) / N> in Kelvin
// (negative when target binds stronger than reference, raising f_ref to match target loading).
PROPOSER_API int proposer_set_chem_pot_shift(void* handle, double delta_eps_K);

// Run classical GCMC steps directly on accepted state (for warm-start equilibration).
PROPOSER_API int proposer_equilibrate(void* handle, int n_steps);

// Copy accepted -> trial on GPU, execute k_steps reversible classical moves on trial,
// recompute exact U_ref, and download trial state to host buffers (stride = max_mol per box).
PROPOSER_API int proposer_propose_k(
    void* handle, int k_steps,
    int* out_N, double* out_U_ref_K, double* out_dU_ref_K, int* out_inner_acc,
    double* out_cx, double* out_cy, double* out_cz,
    double* out_ux, double* out_uy, double* out_uz);

// Copy trial[b] -> accepted[b] on GPU wherever accept_mask[b] != 0.
PROPOSER_API int proposer_commit(void* handle, const int* accept_mask);

// Download current accepted state to host buffers (stride = max_mol per box).
PROPOSER_API int proposer_get_accepted(
    void* handle,
    int* out_N, double* out_U_ref_K,
    double* out_cx, double* out_cy, double* out_cz,
    double* out_ux, double* out_uy, double* out_uz);

// Evaluate exact classical U_ref (in K) for an arbitrary host configuration (0 if overlap, out_overlap=1).
PROPOSER_API double proposer_eval_energy(
    void* handle, int N,
    const double* cx, const double* cy, const double* cz,
    const double* ux, const double* uy, const double* uz,
    int* out_overlap);

// Run direct classical GPU GCMC benchmark (same grids/supercell/r_cut_gg) for null-bias verification.
PROPOSER_API int proposer_run_direct_benchmark(
    void* handle, int eq_steps, int prod_steps, int sample_stride,
    double* out_avg_uptake_mmol_g, double* out_sem_uptake_mmol_g,
    double* out_Qst_kJ_mol, double* out_sem_Qst_kJ_mol,
    double* out_max_drift_K);

} // extern "C"
