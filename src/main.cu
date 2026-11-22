#include "cell.hpp"
#include "cif_loader.hpp"
#include "forcefield.hpp"
#include "gcmc_gpu.hpp"
#include "grid3d.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>
#include <vector>

using namespace mofgcmc;

static int verify_hex_mic(const Cell& cell, int n_tests) {
    std::mt19937_64 rng(1234567);
    std::uniform_real_distribution<double> uni(-2.0, 2.0);
    int mismatches = 0;

    for (int t = 0; t < n_tests; ++t) {
        double sa1 = uni(rng), sb1 = uni(rng), sc1 = uni(rng);
        double sa2 = uni(rng), sb2 = uni(rng), sc2 = uni(rng);
        double r1x, r1y, r1z, r2x, r2y, r2z;
        cell.super_frac_to_cart(sa1, sb1, sc1, r1x, r1y, r1z);
        cell.super_frac_to_cart(sa2, sb2, sc2, r2x, r2y, r2z);

        double dx, dy, dz, r2_fast;
        cell.mic_delta_super(r2x - r1x, r2y - r1y, r2z - r1z, dx, dy, dz, r2_fast);

        // Brute-force 3x3x3 over wrapped fractional difference
        double dsa = sa2 - sa1; dsa -= std::round(dsa);
        double dsb = sb2 - sb1; dsb -= std::round(dsb);
        double dsc = sc2 - sc1; dsc -= std::round(dsc);
        double r2_brute = 1.0e30;
        for (int na = -1; na <= 1; ++na) {
            for (int nb = -1; nb <= 1; ++nb) {
                for (int nc = -1; nc <= 1; ++nc) {
                    double bx, by, bz;
                    cell.super_frac_to_cart(dsa + na, dsb + nb, dsc + nc, bx, by, bz);
                    double r2 = bx * bx + by * by + bz * bz;
                    if (r2 < r2_brute) r2_brute = r2;
                }
            }
        }

        if (std::fabs(r2_fast - r2_brute) > 1.0e-8) {
            ++mismatches;
        }
    }
    return mismatches;
}

int main() {
    std::printf("=== GPU Batched GCMC (Step 1): Mg-MOF-74 + CO2 @ 298 K (UFF + REPEAT + TraPPE) ===\n\n");

    Cell cell;
    std::vector<FrameworkAtom> unit_atoms;
    CifStructureInfo info;
    const std::string cif_path = "data/MgMOF-74.cif";

    if (!load_mg_mof74_cif(cif_path, 1, 1, 4, cell, unit_atoms, info)) {
        return 1;
    }

    double m_super_gmol = info.unit_mass_g_mol * 4.0;
    std::printf("[1] CIF & Supercell Verification (%s):\n", cif_path.c_str());
    std::printf("    Unit cell   : a=b=%.3f A, c=%.3f A, alpha=beta=%.1f, gamma=%.1f deg\n",
                info.a, info.c, info.alpha_deg, info.gamma_deg);
    std::printf("    1x1x4 cell  : V=%.2f A^3, min_perp_width=%.3f A (> 2*11.0=%.1f A OK)\n",
                cell.volume_super, cell.min_perp_width_super, 22.0);
    std::printf("    Atoms (1x1x1): %d total (Mg=%d, O=%d, C=%d, H=%d), charge_sum=%.6f e\n",
                info.num_atoms, info.count_mg, info.count_o, info.count_c, info.count_h,
                info.total_charge_e);
    std::printf("    Molar mass  : unit=%.3f g/mol, 1x1x4 super=%.3f g/mol\n",
                info.unit_mass_g_mol, m_super_gmol);
    std::printf("    Min bonds   : Mg-O=%.3f A, C-O=%.3f A, C-C=%.3f A, C-H=%.3f A\n",
                info.min_mg_o_A, info.min_c_o_A, info.min_c_c_A, info.min_c_h_A);

    int mic_mismatches = verify_hex_mic(cell, 10000);
    std::printf("    Hex MIC test: %d / 10000 mismatches vs 3x3x3 brute-force\n\n", mic_mismatches);
    if (mic_mismatches != 0) {
        std::fprintf(stderr, "Hexagonal MIC test failed!\n");
        return 1;
    }

    // Save framework unit atoms for plotting Mg OMS sites on pore density heatmap
    {
        std::ofstream fout("build/framework_unit_atoms.csv");
        fout << "x,y,z,sa,sb,sc,elem,q\n";
        for (const auto& at : unit_atoms) {
            fout << std::fixed << std::setprecision(6)
                 << at.x << "," << at.y << "," << at.z << ","
                 << at.sa << "," << at.sb << "," << at.sc << ","
                 << at.elem << "," << at.q << "\n";
        }
    }

    // Build 6 3D framework grids (128 x 128 x 36)
    const int gx = 128, gy = 128, gz = 36;
    const double r_cut = 11.0;
    const double ewald_alpha = 0.30;
    Grid3DView d_grid{};
    GridBuildStats gstats{};

    std::printf("[2] Building 6 Precomputed 3D Unit-Cell Grids (%dx%dx%d: U_lj_C, U_lj_O, Phi, Ex, Ey, Ez)...\n",
                gx, gy, gz);
    build_framework_grids_gpu(cell, unit_atoms, gx, gy, gz, r_cut, ewald_alpha, d_grid, gstats);
    std::printf("    GPU build time : %.2f ms\n", gstats.build_ms);
    std::printf("    Grid vs direct (alpha_ref=0.25, %d pore pts): MAE(U_lj_C)=%.3f K, MAE(U_lj_O)=%.3f K, MAE(Phi)=%.4f K/e\n\n",
                gstats.verify_samples, gstats.verify_lj_c_mae_K,
                gstats.verify_lj_o_mae_K, gstats.verify_phi_mae_Ke);

    // Prepare 16 pressures x 2 replicas = 32 parallel GCMC boxes at 298 K
    const double T_K = 298.0;
    const double pressures[] = {
        0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.25, 0.50,
        0.75, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 20.0
    };
    const int n_pressures = (int)(sizeof(pressures) / sizeof(pressures[0]));
    const int n_replicas = 2;

    std::vector<FugacityResult> fugs(n_pressures);
    std::vector<BoxSpec> boxes;
    for (int ip = 0; ip < n_pressures; ++ip) {
        fugs[ip] = compute_co2_fugacity_pr(T_K, pressures[ip]);
        int rec = (std::fabs(pressures[ip] - 1.0) < 1.0e-6) ? 1 : 0;
        for (int rep = 0; rep < n_replicas; ++rep) {
            boxes.push_back({pressures[ip], fugs[ip].f_Pa, rep, rec});
        }
    }

    BatchedGcmcConfig cfg;
    cfg.T_K = T_K;
    cfg.r_cut_A = r_cut;
    cfg.ewald_alpha = ewald_alpha;
    cfg.dsf_alpha = 0.20;
    cfg.eq_steps = 100000;
    cfg.prod_steps = 200000;
    cfg.sample_stride = 20;
    cfg.max_mol = 512;
    cfg.framework_mass_super_gmol = m_super_gmol;
    cfg.base_seed = 20261122ULL;

    const int hist_na = 128, hist_nb = 128;
    std::vector<BoxResult> results;
    std::vector<double> density_hist;

    std::printf("[3] Running Batched GPU GCMC (%d boxes = %d pressures x %d replicas, %dk eq + %dk prod)...\n",
                (int)boxes.size(), n_pressures, n_replicas, cfg.eq_steps / 1000, cfg.prod_steps / 1000);
    double gcmc_ms = run_batched_gcmc_gpu(cell, d_grid, cfg, boxes, results, density_hist, hist_na, hist_nb);
    double total_mc_steps = (double)boxes.size() * (double)(cfg.eq_steps + cfg.prod_steps);
    std::printf("    GPU GCMC time  : %.2f ms (%.3f s) -> %.2f M MC-steps/sec\n\n",
                gcmc_ms, gcmc_ms * 1.0e-3, total_mc_steps / (gcmc_ms * 1.0e3));

    std::printf("  P (bar)   phi_PR   <N>_rep0   <N>_rep1  Uptake(mmol/g)    std_rep  Qst(kJ/mol)  accT/R/I/D\n");
    std::printf("  -----------------------------------------------------------------------------------------------\n");

    std::ofstream iso_out("build/isotherm_298K.csv");
    iso_out << "P_bar,f_bar,phi_PR,Z_PR,N_rep0,N_rep1,N_mean,uptake_mmol_g,uptake_err_mmol_g,avg_U_K,Qst_kJ_mol,acc_trans,acc_rot,acc_ins,acc_del\n";

    for (int ip = 0; ip < n_pressures; ++ip) {
        const BoxResult& r0 = results[ip * 2 + 0];
        const BoxResult& r1 = results[ip * 2 + 1];
        double n_mean = 0.5 * (r0.avg_N + r1.avg_N);
        double up_mean = 0.5 * (r0.uptake_mmol_g + r1.uptake_mmol_g);
        double up_err = 0.5 * std::fabs(r0.uptake_mmol_g - r1.uptake_mmol_g);
        double u_mean = 0.5 * (r0.avg_U_K + r1.avg_U_K);

        // Weighted/mean Qst across replicas when var(N) is well-sampled
        double qst_mean = 0.5 * (r0.Qst_kJ_mol + r1.Qst_kJ_mol);
        double acc_t = 0.5 * (r0.acc_trans + r1.acc_trans);
        double acc_r = 0.5 * (r0.acc_rot + r1.acc_rot);
        double acc_i = 0.5 * (r0.acc_ins + r1.acc_ins);
        double acc_d = 0.5 * (r0.acc_del + r1.acc_del);

        std::printf("  %7.3f  %6.4f  %9.3f  %9.3f     %8.4f      %7.4f    %7.2f      %.2f/%.2f/%.2f/%.2f\n",
                    pressures[ip], fugs[ip].phi, r0.avg_N, r1.avg_N,
                    up_mean, up_err, qst_mean, acc_t, acc_r, acc_i, acc_d);

        iso_out << std::fixed << std::setprecision(6)
                << pressures[ip] << "," << fugs[ip].f_bar << "," << fugs[ip].phi << "," << fugs[ip].Z << ","
                << r0.avg_N << "," << r1.avg_N << "," << n_mean << ","
                << up_mean << "," << up_err << "," << u_mean << "," << qst_mean << ","
                << acc_t << "," << acc_r << "," << acc_i << "," << acc_d << "\n";
    }

    // Export 2D pore density histogram (at 1.0 bar)
    {
        std::ofstream hout("build/pore_density_1bar.csv");
        for (int ib = 0; ib < hist_nb; ++ib) {
            for (int ia = 0; ia < hist_na; ++ia) {
                if (ia > 0) hout << ",";
                hout << density_hist[ib * hist_na + ia];
            }
            hout << "\n";
        }
    }

    free_framework_grids_gpu(d_grid);
    std::printf("\nExported build/isotherm_298K.csv, build/pore_density_1bar.csv, build/framework_unit_atoms.csv\n");
    return 0;
}
