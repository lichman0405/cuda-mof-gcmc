#include "cell.hpp"
#include "cif_loader.hpp"
#include "forcefield.hpp"
#include "gcmc_gpu.hpp"
#include "v2_gcmc.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

using namespace mofgcmc;

static void export_v2_csv(
    const std::string& path,
    const std::vector<double>& pressures,
    const std::vector<FugacityResult>& fugs,
    const std::vector<V2BoxResult>& results,
    int n_rep) {

    std::ofstream out(path);
    out << "P_bar,f_bar,phi_PR,Z_PR,N_mean,uptake_mmol_g,uptake_sem_mmol_g,Qst_kJ_mol,Qst_sem_kJ_mol,max_drift_K,acc_ins,acc_del\n";

    int n_p = (int)pressures.size();
    for (int ip = 0; ip < n_p; ++ip) {
        double sum_n = 0.0, sum_u = 0.0, sum_q = 0.0, max_d = 0.0, ai = 0.0, ad = 0.0;
        for (int r = 0; r < n_rep; ++r) {
            const V2BoxResult& br = results[ip * n_rep + r];
            sum_n += br.avg_N;
            sum_u += br.uptake_mmol_g;
            sum_q += br.Qst_kJ_mol;
            if (br.max_energy_drift_K > max_d) max_d = br.max_energy_drift_K;
            ai += br.acc_ins;
            ad += br.acc_del;
        }
        double mean_n = sum_n / n_rep;
        double mean_u = sum_u / n_rep;
        double mean_q = sum_q / n_rep;
        double var_u = 0.0, var_q = 0.0;
        for (int r = 0; r < n_rep; ++r) {
            double du = results[ip * n_rep + r].uptake_mmol_g - mean_u;
            double dq = results[ip * n_rep + r].Qst_kJ_mol - mean_q;
            var_u += du * du;
            var_q += dq * dq;
        }
        double sem_u = (n_rep > 1) ? std::sqrt(var_u / (n_rep * (n_rep - 1))) : 0.0;
        double sem_q = (n_rep > 1) ? std::sqrt(var_q / (n_rep * (n_rep - 1))) : 0.0;

        out << std::fixed << std::setprecision(6)
            << pressures[ip] << "," << fugs[ip].f_bar << "," << fugs[ip].phi << "," << fugs[ip].Z << ","
            << mean_n << "," << mean_u << "," << sem_u << "," << mean_q << "," << sem_q << ","
            << std::scientific << max_d << ","
            << std::fixed << (ai / n_rep) << "," << (ad / n_rep) << "\n";
    }
}

int main() {
    std::printf("=== GPU High-Precision Validation (0.10 A Grid, Double SMEM + Recompute, 64 Boxes) ===\n\n");

    Cell cell;
    std::vector<FrameworkAtom> unit_atoms;
    CifStructureInfo info;
    if (!load_mg_mof74_cif("data/MgMOF-74.cif", 1, 1, 4, cell, unit_atoms, info)) {
        return 1;
    }
    double m_super_gmol = info.unit_mass_g_mol * 4.0;
    const double T_K = 298.0;
    const double r_cut = 11.0;
    const double ewald_alpha = 0.30;

    // 1. Compare radial Mg-O accuracy: 128x128x36 (0.20 A) vs 256x256x72 (0.10 A)
    Grid3DView d_g128{}, d_g256_uff{}, d_g256_merc{};
    V2GridStats st128{}, st256_uff{}, st256_merc{};

    build_v2_grids_gpu(cell, unit_atoms, 128, 128, 36, r_cut, ewald_alpha, FF_UFF_REPEAT_TRAPPE, d_g128, st128);
    free_framework_grids_gpu(d_g128);

    build_v2_grids_gpu(cell, unit_atoms, 256, 256, 72, r_cut, ewald_alpha, FF_UFF_REPEAT_TRAPPE, d_g256_uff, st256_uff);
    std::printf("[1] Grid Resolution & Interpolation Upgrade (UFF+REPEAT, 256x256x72 = 0.10 A):\n");
    std::printf("    Build time 256x256x72 : %.2f ms (6 grids = 113 MB VRAM)\n", st256_uff.build_ms);
    std::printf("    Radial Mg-O MAE (2.3-3.8 A): 128-Trilinear = %.2f K -> 256-Trilinear = %.2f K -> 256-Tricubic = %.3f K (%.1fx finer vs 128)\n",
                st128.radial_256_mae_K, st256_uff.radial_256_mae_K, st256_uff.radial_256_tricubic_mae_K,
                st128.radial_256_mae_K / std::max(st256_uff.radial_256_tricubic_mae_K, 1.0e-6));
    std::printf("    Pore MAE Trilinear (%d pts): U_lj_C=%.3f K, U_lj_O=%.3f K, Phi=%.4f K/e\n",
                st256_uff.verify_samples, st256_uff.pore_mae_c_K, st256_uff.pore_mae_o_K, st256_uff.pore_mae_phi_Ke);
    std::printf("    Pore MAE Tricubic  (%d pts): U_lj_C=%.3f K, U_lj_O=%.3f K, Phi=%.4f K/e\n\n",
                st256_uff.verify_samples, st256_uff.pore_tricubic_mae_c_K, st256_uff.pore_tricubic_mae_o_K, st256_uff.pore_tricubic_mae_phi_Ke);

    const std::vector<double> pressures = {
        0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.25, 0.50,
        0.75, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 20.0
    };
    const int n_p = (int)pressures.size();
    const int n_rep = 4;

    std::vector<FugacityResult> fugs(n_p);
    std::vector<V2BoxSpec> boxes;
    for (int ip = 0; ip < n_p; ++ip) {
        fugs[ip] = compute_co2_fugacity_pr(T_K, pressures[ip]);
        int rec_rdf = (std::fabs(pressures[ip] - 0.15) < 1.0e-6) ? 1 : 0;
        for (int r = 0; r < n_rep; ++r) {
            boxes.push_back({pressures[ip], fugs[ip].f_Pa, r, rec_rdf});
        }
    }

    const int rdf_bins = 80;
    const double rdf_min = 1.5, rdf_max = 5.5;

    // 2. Run high-precision UFF + REPEAT + TraPPE Tricubic (64 boxes x 1.0M steps = 64M steps)
    V2GcmcConfig cfg_uff;
    cfg_uff.T_K = T_K; cfg_uff.r_cut_A = r_cut; cfg_uff.ff_mode = FF_UFF_REPEAT_TRAPPE;
    cfg_uff.use_tricubic = 1;
    cfg_uff.eq_steps = 200000; cfg_uff.prod_steps = 800000; cfg_uff.sample_stride = 20;
    cfg_uff.recompute_stride = 50000; cfg_uff.framework_mass_super_gmol = m_super_gmol;
    cfg_uff.base_seed = 20261123ULL;

    std::vector<V2BoxResult> res_uff;
    std::vector<double> rdf_uff;
    double ms_uff = run_v2_gcmc_gpu(cell, d_g256_uff, unit_atoms, cfg_uff, boxes, res_uff,
                                    rdf_uff, rdf_bins, rdf_min, rdf_max);
    double steps_tot = (double)boxes.size() * (double)(cfg_uff.eq_steps + cfg_uff.prod_steps);
    std::printf("[2] UFF + REPEAT + TraPPE (256x256x72 Tricubic, 64 boxes x 1M steps = 64M steps):\n");
    std::printf("    GPU Time = %.2f ms (%.2f s) -> %.2f M MC-steps/sec\n",
                ms_uff, ms_uff * 1e-3, steps_tot / (ms_uff * 1e3));
    export_v2_csv("build/v2_uff_010A.csv", pressures, fugs, res_uff, n_rep);
    free_framework_grids_gpu(d_g256_uff);

    // 3. Build 256x256x72 Mercado 2016 DFT-Derived FF & run 64 boxes x 1.0M steps
    build_v2_grids_gpu(cell, unit_atoms, 256, 256, 72, r_cut, ewald_alpha, FF_MERCADO2016_EPM2, d_g256_merc, st256_merc);
    std::printf("\n[3] Mercado 2016 DFT-Derived Buckingham/LJ + Mercado2016-Charges + EPM2 (256x256x72 Tricubic):\n");
    std::printf("    Grid build = %.2f ms | Radial Mg-O MAE: Trilin=%.2f K -> Tricubic=%.3f K\n",
                st256_merc.build_ms, st256_merc.radial_256_mae_K, st256_merc.radial_256_tricubic_mae_K);
    std::printf("    Pore MAE Tricubic (%d pts): C=%.3f K, O=%.3f K, Phi=%.4f K/e\n",
                st256_merc.verify_samples, st256_merc.pore_tricubic_mae_c_K, st256_merc.pore_tricubic_mae_o_K, st256_merc.pore_tricubic_mae_phi_Ke);

    V2GcmcConfig cfg_merc = cfg_uff;
    cfg_merc.ff_mode = FF_MERCADO2016_EPM2;
    cfg_merc.base_seed = 20261124ULL;

    std::vector<V2BoxResult> res_merc;
    std::vector<double> rdf_merc;
    double ms_merc = run_v2_gcmc_gpu(cell, d_g256_merc, unit_atoms, cfg_merc, boxes, res_merc,
                                     rdf_merc, rdf_bins, rdf_min, rdf_max);
    std::printf("    GPU Time = %.2f ms (%.2f s) -> %.2f M MC-steps/sec\n\n",
                ms_merc, ms_merc * 1e-3, steps_tot / (ms_merc * 1e3));
    export_v2_csv("build/v2_mercado2016_010A.csv", pressures, fugs, res_merc, n_rep);
    free_framework_grids_gpu(d_g256_merc);

    // Export normalized radial Mg - O_CO2 histogram @ 0.15 bar
    {
        double sum_u = 0.0, sum_m = 0.0;
        for (int i = 0; i < rdf_bins; ++i) { sum_u += rdf_uff[i]; sum_m += rdf_merc[i]; }
        std::ofstream rout("build/v2_mg_o_rdf.csv");
        rout << "r_A,prob_uff,prob_mercado2016\n";
        double dr = (rdf_max - rdf_min) / rdf_bins;
        for (int i = 0; i < rdf_bins; ++i) {
            double r_mid = rdf_min + (i + 0.5) * dr;
            rout << std::fixed << std::setprecision(4) << r_mid << ","
                 << std::setprecision(6) << (sum_u > 0 ? rdf_uff[i] / (sum_u * dr) : 0.0) << ","
                 << (sum_m > 0 ? rdf_merc[i] / (sum_m * dr) : 0.0) << "\n";
        }
    }

    std::printf("  P(bar) | UFF 0.10A (mmol/g)  Qst    Drift(K) | Mercado2016 (mmol/g)   Qst    Drift(K)\n");
    std::printf("  --------------------------------------------------------------------------------------\n");
    double global_max_drift = 0.0;
    for (int ip = 0; ip < n_p; ++ip) {
        auto stat = [&](const std::vector<V2BoxResult>& v, double& mu, double& mq, double& md) {
            mu = mq = md = 0.0;
            for (int r = 0; r < n_rep; ++r) {
                mu += v[ip * n_rep + r].uptake_mmol_g;
                mq += v[ip * n_rep + r].Qst_kJ_mol;
                if (v[ip * n_rep + r].max_energy_drift_K > md) md = v[ip * n_rep + r].max_energy_drift_K;
            }
            mu /= n_rep; mq /= n_rep;
            if (md > global_max_drift) global_max_drift = md;
        };
        double u1, q1, d1, u2, q2, d2;
        stat(res_uff, u1, q1, d1);
        stat(res_merc, u2, q2, d2);
        std::printf("  %6.3f |    %8.4f         %5.2f  %.2e |     %8.4f          %5.2f  %.2e\n",
                    pressures[ip], u1, q1, d1, u2, q2, d2);
    }

    double mgo_uff = 0.0, mgo_merc = 0.0;
    for (int r = 0; r < n_rep; ++r) {
        mgo_uff  += res_uff[5 * n_rep + r].mean_min_mg_o_A;
        mgo_merc += res_merc[5 * n_rep + r].mean_min_mg_o_A;
    }
    std::printf("\n[4] Diagnostics Summary:\n");
    std::printf("    Max energy drift across all 128 GPU boxes (64M steps each): %.3e K\n", global_max_drift);
    std::printf("    Mean min(Mg - O_CO2) @ 0.15 bar: UFF = %.3f A  vs  Mercado 2016 = %.3f A\n",
                mgo_uff / n_rep, mgo_merc / n_rep);
    return 0;
}
