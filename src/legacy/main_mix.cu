#include "cell.hpp"
#include "cif_loader.hpp"
#include "forcefield.hpp"
#include "mix_gcmc.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

using namespace mofgcmc;

static void export_mix_csv(
    const std::string& path,
    const std::vector<double>& pressures,
    const std::vector<BinaryFugacityResult>& fugs,
    const std::vector<MixBoxResult>& results) {

    std::ofstream out(path);
    out << "P_bar,y0,y1,phi0,phi1,Z_PR,f0_bar,f1_bar,"
           "N0_mean,N1_mean,uptake0_mmol_g,uptake0_err,uptake1_mmol_g,uptake1_err,"
           "selectivity,Upol0_kJ,Upol1_kJ,acc_i0,acc_d0,acc_i1,acc_d1,acc_sw01,acc_sw10\n";

    int np = (int)pressures.size();
    for (int ip = 0; ip < np; ++ip) {
        const MixBoxResult& r0 = results[2 * ip + 0];
        const MixBoxResult& r1 = results[2 * ip + 1];
        double n0 = 0.5 * (r0.avg_N0 + r1.avg_N0);
        double n1 = 0.5 * (r0.avg_N1 + r1.avg_N1);
        double up0 = 0.5 * (r0.uptake0_mmol_g + r1.uptake0_mmol_g);
        double up0_err = 0.5 * std::fabs(r0.uptake0_mmol_g - r1.uptake0_mmol_g);
        double up1 = 0.5 * (r0.uptake1_mmol_g + r1.uptake1_mmol_g);
        double up1_err = 0.5 * std::fabs(r0.uptake1_mmol_g - r1.uptake1_mmol_g);

        double sel = 0.0;
        if (n1 > 1.0e-9 && r0.y0 > 0.0) {
            sel = (n0 / n1) * (r0.y1 / r0.y0);
        }

        auto avg = [&](double a, double b) { return 0.5 * (a + b); };
        out << std::fixed << std::setprecision(6)
            << pressures[ip] << "," << r0.y0 << "," << r0.y1 << ","
            << fugs[ip].phi0 << "," << fugs[ip].phi1 << "," << fugs[ip].Z << ","
            << fugs[ip].f0_bar << "," << fugs[ip].f1_bar << ","
            << n0 << "," << n1 << ","
            << up0 << "," << up0_err << "," << up1 << "," << up1_err << ","
            << sel << "," << avg(r0.avg_Upol0_kJ, r1.avg_Upol0_kJ) << ","
            << avg(r0.avg_Upol1_kJ, r1.avg_Upol1_kJ) << ","
            << avg(r0.acc_i0, r1.acc_i0) << "," << avg(r0.acc_d0, r1.acc_d0) << ","
            << avg(r0.acc_i1, r1.acc_i1) << "," << avg(r0.acc_d1, r1.acc_d1) << ","
            << avg(r0.acc_sw01, r1.acc_sw01) << "," << avg(r0.acc_sw10, r1.acc_sw10) << "\n";
    }
}

static void export_hist(const std::string& path, const std::vector<double>& h, int na, int nb) {
    std::ofstream out(path);
    for (int ib = 0; ib < nb; ++ib) {
        for (int ia = 0; ia < na; ++ia) {
            if (ia > 0) out << ",";
            out << h[ib * na + ia];
        }
        out << "\n";
    }
}

int main() {
    std::printf("=== GPU Batched Binary GCMC (Step 3): CO2/N2 & CO2/CH4 in Mg-MOF-74 @ 298 K ===\n\n");

    Cell cell;
    std::vector<FrameworkAtom> unit_atoms;
    CifStructureInfo info;
    if (!load_mg_mof74_cif("data/MgMOF-74.cif", 1, 1, 4, cell, unit_atoms, info)) {
        return 1;
    }
    double m_super_gmol = info.unit_mass_g_mol * 4.0;

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

    const int gx = 128, gy = 128, gz = 36;
    const double r_cut = 11.0;
    const double ewald_alpha = 0.30;
    MixGrid3DView d_grid{};
    MixGridStats gstats{};

    std::printf("[1] Building 8 Multi-Species 3D Grids (%dx%dx%d: C_CO2, O_CO2, N_N2, CH4, Phi, Ex, Ey, Ez)...\n",
                gx, gy, gz);
    build_mix_framework_grids_gpu(cell, unit_atoms, gx, gy, gz, r_cut, ewald_alpha, d_grid, gstats);
    std::printf("    GPU build time  : %.2f ms\n", gstats.build_ms);
    std::printf("    MAE vs direct   : LJ(C)=%.3f, LJ(O)=%.3f, LJ(N)=%.3f, LJ(CH4)=%.3f K | Phi=%.4f K/e | |E|=%.3f K/(e*A)\n\n",
                gstats.mae_lj_K[0], gstats.mae_lj_K[1], gstats.mae_lj_K[2], gstats.mae_lj_K[3],
                gstats.mae_phi_Ke, gstats.mae_efield_KeA);

    SpeciesDef sp_co2 = make_species_co2();
    SpeciesDef sp_n2  = make_species_n2();
    SpeciesDef sp_ch4 = make_species_ch4();

    const double T_K = 298.0;
    const int hist_na = 128, hist_nb = 128;

    // -------------------------------------------------------------------------
    // Test A: Ideal-Gas Detailed Balance Self-Test with Identity Swap ON
    // -------------------------------------------------------------------------
    {
        constexpr double KB_SI = 1.380649e-23;
        double target0 = 12.0, target1 = 40.0;
        double f0_Pa = target0 * KB_SI * T_K / (cell.volume_super * 1.0e-30);
        double f1_Pa = target1 * KB_SI * T_K / (cell.volume_super * 1.0e-30);

        MixGcmcConfig cfg;
        cfg.T_K = T_K;
        cfg.eq_steps = 50000;
        cfg.prod_steps = 250000;
        cfg.sample_stride = 10;
        cfg.pol_mode = 0;
        cfg.enable_swap = 1;
        cfg.ideal_gas_test = 1;
        cfg.framework_mass_super_gmol = m_super_gmol;
        cfg.base_seed = 9001ULL;

        std::vector<MixBoxSpec> boxes(4, {1.0, 0.25, 0.75, f0_Pa, f1_Pa, 0, 0});
        for (int i = 0; i < 4; ++i) boxes[i].replica_id = i;
        std::vector<MixBoxResult> res;
        std::vector<double> h0, h1;
        run_binary_gcmc_gpu(cell, d_grid, sp_co2, sp_n2, cfg, boxes, res, h0, h1, 4, 4);

        double m0 = 0.0, m1 = 0.0;
        for (const auto& r : res) { m0 += r.avg_N0; m1 += r.avg_N1; }
        m0 *= 0.25; m1 *= 0.25;
        std::printf("[Test A] Ideal-Gas Swap ON (4 boxes): <N0>=%.4f (exact %.1f), <N1>=%.4f (exact %.1f)\n",
                    m0, target0, m1, target1);
        if (std::fabs(m0 - target0) > 0.3 || std::fabs(m1 - target1) > 0.5) {
            std::fprintf(stderr, "Error: Ideal-Gas Swap detailed balance test failed!\n");
            free_mix_framework_grids_gpu(d_grid);
            return 1;
        }
    }

    // -------------------------------------------------------------------------
    // Test B: Real Pore Swap-ON vs Swap-OFF Consistency @ 1.0 bar CO2/N2 (15:85), Pol-3Site
    // -------------------------------------------------------------------------
    {
        auto fug = compute_binary_fugacity_pr(T_K, 1.0, sp_co2, sp_n2, 0.15, 0.85, -0.017);
        MixGcmcConfig cfg;
        cfg.T_K = T_K; cfg.eq_steps = 100000; cfg.prod_steps = 250000; cfg.sample_stride = 20;
        cfg.pol_mode = 2; cfg.framework_mass_super_gmol = m_super_gmol;

        std::vector<MixBoxSpec> b4(4, {1.0, 0.15, 0.85, fug.f0_Pa, fug.f1_Pa, 0, 0});
        for (int i = 0; i < 4; ++i) b4[i].replica_id = i;

        std::vector<MixBoxResult> r_on, r_off;
        std::vector<double> h0, h1;
        cfg.enable_swap = 1; cfg.base_seed = 1111ULL;
        run_binary_gcmc_gpu(cell, d_grid, sp_co2, sp_n2, cfg, b4, r_on, h0, h1, 4, 4);
        cfg.enable_swap = 0; cfg.base_seed = 2222ULL;
        run_binary_gcmc_gpu(cell, d_grid, sp_co2, sp_n2, cfg, b4, r_off, h0, h1, 4, 4);

        auto stats_4 = [](const std::vector<MixBoxResult>& v, double& m0, double& m1) {
            m0 = m1 = 0.0;
            for (const auto& r : v) { m0 += r.uptake0_mmol_g; m1 += r.uptake1_mmol_g; }
            m0 *= 0.25; m1 *= 0.25;
        };
        double on0, on1, off0, off1;
        stats_4(r_on, on0, on1);
        stats_4(r_off, off0, off1);
        std::printf("[Test B] 1 bar CO2/N2 (15:85) Pol-3Site Swap Consistency (4 replicas each):\n");
        std::printf("         Swap-ON : CO2=%.4f mmol/g, N2=%.4f mmol/g\n", on0, on1);
        std::printf("         Swap-OFF: CO2=%.4f mmol/g, N2=%.4f mmol/g\n\n", off0, off1);
    }

    const std::vector<double> pressures = {
        0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0
    };
    const int n_p = (int)pressures.size();

    auto run_binary_sweep = [&](const char* title, const SpeciesDef& s0, const SpeciesDef& s1,
                                double y0, double y1, double kij, int pol_mode,
                                const char* csv_path, const char* h0_path, const char* h1_path) {
        std::vector<BinaryFugacityResult> fugs(n_p);
        std::vector<MixBoxSpec> boxes;
        for (int ip = 0; ip < n_p; ++ip) {
            fugs[ip] = compute_binary_fugacity_pr(T_K, pressures[ip], s0, s1, y0, y1, kij);
            int rec = (std::fabs(pressures[ip] - 1.0) < 1.0e-6) ? 1 : 0;
            for (int rep = 0; rep < 2; ++rep) {
                boxes.push_back({pressures[ip], y0, y1, fugs[ip].f0_Pa, fugs[ip].f1_Pa, rep, rec});
            }
        }

        MixGcmcConfig cfg;
        cfg.T_K = T_K;
        cfg.r_cut_A = r_cut;
        cfg.ewald_alpha = ewald_alpha;
        cfg.dsf_alpha = 0.20;
        cfg.eq_steps = 100000;
        cfg.prod_steps = 250000;
        cfg.sample_stride = 20;
        cfg.max_mol = 512;
        cfg.pol_mode = pol_mode;
        cfg.enable_swap = 1;
        cfg.ideal_gas_test = 0;
        cfg.framework_mass_super_gmol = m_super_gmol;
        cfg.base_seed = 20261123ULL + (uint64_t)(pol_mode * 100 + (int)(y0 * 1000));

        std::vector<MixBoxResult> res;
        std::vector<double> hist0, hist1;
        double ms = run_binary_gcmc_gpu(cell, d_grid, s0, s1, cfg, boxes, res, hist0, hist1, hist_na, hist_nb);
        double total_mc = (double)boxes.size() * (double)(cfg.eq_steps + cfg.prod_steps);
        std::printf("[Sweep] %s: %.2f ms (%.3f s) -> %.2f M MC-steps/sec\n",
                    title, ms, ms * 1.0e-3, total_mc / (ms * 1.0e3));
        export_mix_csv(csv_path, pressures, fugs, res);
        if (h0_path && h1_path) {
            export_hist(h0_path, hist0, hist_na, hist_nb);
            export_hist(h1_path, hist1, hist_na, hist_nb);
        }
        return res;
    };

    // Sweep 1: Flue Gas CO2/N2 (15:85), kij = -0.017
    auto n2_non = run_binary_sweep("CO2/N2 (15:85) NonPol   ", sp_co2, sp_n2, 0.15, 0.85, -0.017, 0,
                                   "build/mix_co2_n2_nonpol.csv", nullptr, nullptr);
    auto n2_pol = run_binary_sweep("CO2/N2 (15:85) Pol-3Site", sp_co2, sp_n2, 0.15, 0.85, -0.017, 2,
                                   "build/mix_co2_n2_pol3site.csv", "build/hist_mix_co2_1bar.csv", "build/hist_mix_n2_1bar.csv");

    // Sweep 2: Biogas CO2/CH4 (50:50), kij = +0.092
    auto ch4_non = run_binary_sweep("CO2/CH4 (50:50) NonPol   ", sp_co2, sp_ch4, 0.50, 0.50, 0.092, 0,
                                    "build/mix_co2_ch4_nonpol.csv", nullptr, nullptr);
    auto ch4_pol = run_binary_sweep("CO2/CH4 (50:50) Pol-3Site", sp_co2, sp_ch4, 0.50, 0.50, 0.092, 2,
                                    "build/mix_co2_ch4_pol3site.csv", nullptr, nullptr);

    std::printf("\n=== CO2/N2 (15:85) Summary @ 298 K ===\n");
    std::printf("  P(bar) | NonPol CO2 / N2 (S_CO2/N2)      | Pol-3Site CO2 / N2 (S_CO2/N2)\n");
    std::printf("  -------------------------------------------------------------------------\n");
    for (int ip = 0; ip < n_p; ++ip) {
        auto m0 = [](const std::vector<MixBoxResult>& v, int i) { return 0.5 * (v[2 * i].uptake0_mmol_g + v[2 * i + 1].uptake0_mmol_g); };
        auto m1 = [](const std::vector<MixBoxResult>& v, int i) { return 0.5 * (v[2 * i].uptake1_mmol_g + v[2 * i + 1].uptake1_mmol_g); };
        auto ms = [](const std::vector<MixBoxResult>& v, int i, double y0, double y1) {
            double n0 = 0.5 * (v[2 * i].avg_N0 + v[2 * i + 1].avg_N0);
            double n1 = 0.5 * (v[2 * i].avg_N1 + v[2 * i + 1].avg_N1);
            return (n1 > 1e-9) ? (n0 / n1) * (y1 / y0) : 0.0;
        };
        std::printf("  %6.2f | %6.3f / %5.3f (%6.1f)         | %6.3f / %5.3f (%6.1f)\n",
                    pressures[ip],
                    m0(n2_non, ip), m1(n2_non, ip), ms(n2_non, ip, 0.15, 0.85),
                    m0(n2_pol, ip), m1(n2_pol, ip), ms(n2_pol, ip, 0.15, 0.85));
    }

    std::printf("\n=== CO2/CH4 (50:50) Summary @ 298 K ===\n");
    std::printf("  P(bar) | NonPol CO2 / CH4 (S_CO2/CH4)    | Pol-3Site CO2 / CH4 (S_CO2/CH4)\n");
    std::printf("  -------------------------------------------------------------------------\n");
    for (int ip = 0; ip < n_p; ++ip) {
        auto m0 = [](const std::vector<MixBoxResult>& v, int i) { return 0.5 * (v[2 * i].uptake0_mmol_g + v[2 * i + 1].uptake0_mmol_g); };
        auto m1 = [](const std::vector<MixBoxResult>& v, int i) { return 0.5 * (v[2 * i].uptake1_mmol_g + v[2 * i + 1].uptake1_mmol_g); };
        auto ms = [](const std::vector<MixBoxResult>& v, int i) {
            double n0 = 0.5 * (v[2 * i].avg_N0 + v[2 * i + 1].avg_N0);
            double n1 = 0.5 * (v[2 * i].avg_N1 + v[2 * i + 1].avg_N1);
            return (n1 > 1e-9) ? (n0 / n1) : 0.0;
        };
        std::printf("  %6.2f | %6.3f / %5.3f (%6.1f)         | %6.3f / %5.3f (%6.1f)\n",
                    pressures[ip],
                    m0(ch4_non, ip), m1(ch4_non, ip), ms(ch4_non, ip),
                    m0(ch4_pol, ip), m1(ch4_pol, ip), ms(ch4_pol, ip));
    }

    free_mix_framework_grids_gpu(d_grid);
    std::printf("\nExported Step 3 mixture CSVs & pore histograms to build/.\n");
    return 0;
}
