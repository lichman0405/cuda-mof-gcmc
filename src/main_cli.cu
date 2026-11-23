#include "cell.hpp"
#include "cif_loader.hpp"
#include "forcefield.hpp"
#include "simple_json.hpp"
#include "unified_gcmc.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

using namespace mofgcmc;

namespace {

std::string to_lower_str(std::string s) {
    for (char& ch : s) ch = (char)std::tolower((unsigned char)ch);
    return s;
}

int round_mult8_clamp(double x, int lo, int hi) {
    int n = (int)std::ceil(x);
    n = ((n + 7) / 8) * 8;
    if (n < lo) n = lo;
    if (n > hi) n = hi;
    return n;
}

bool make_species_by_name(const std::string& name_raw, int ff_mode, SpeciesDef& out) {
    std::string name = to_lower_str(name_raw);
    if (name == "co2") {
        out = (ff_mode == 1) ? make_species_co2_epm2() : make_species_co2();
        return true;
    }
    if (name == "n2") {
        out = make_species_n2();
        return true;
    }
    if (name == "ch4") {
        out = make_species_ch4();
        return true;
    }
    return false;
}

void print_usage() {
    std::printf(
        "Usage: mof_gcmc_cli --config <path.json> [--temp <K>] [--out <path.csv>] [--test-scf]\n"
        "Options:\n"
        "  --config <path>   Path to JSON simulation config\n"
        "  --temp   <K>      Override temperature_K from JSON\n"
        "  --out    <csv>    Override output_csv from JSON\n"
        "  --test-scf        Run GPU vs matrix-inversion SCF polarization unit test\n"
        "  --help            Show this message\n");
}

} // namespace

int main(int argc, char** argv) {
    std::string cfg_path;
    double temp_override = -1.0;
    std::string out_override;
    bool test_scf = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else if (arg == "--test-scf") {
            test_scf = true;
        } else if (arg == "--config" && i + 1 < argc) {
            cfg_path = argv[++i];
        } else if (arg == "--temp" && i + 1 < argc) {
            temp_override = std::stod(argv[++i]);
        } else if (arg == "--out" && i + 1 < argc) {
            out_override = argv[++i];
        } else {
            std::fprintf(stderr, "Unknown or incomplete argument: %s\n", arg.c_str());
            print_usage();
            return 1;
        }
    }

    if (test_scf) {
        Cell cell;
        std::vector<FrameworkAtom> unit_atoms;
        CifStructureInfo info;
        if (!load_cif_framework("data/MgMOF-74.cif", 1, 1, 4, cell, unit_atoms, info)) return 1;
        MixGrid3DView d_grid{};
        UnifiedGridStats gst{};
        build_unified_grids_gpu(cell, unit_atoms, 256, 256, 72, 11.0, 0.30, 0, d_grid, gst);
        SpeciesDef co2 = make_species_co2();
        bool ok = run_scf_matrix_unit_test_gpu(cell, d_grid, co2, 11.0, 0.20, "build/scf_test_dump.json");
        free_unified_grids_gpu(d_grid);
        return ok ? 0 : 1;
    }

    if (cfg_path.empty()) {
        print_usage();
        return 1;
    }

    JsonValue root;
    std::string err;
    if (!parse_json_file(cfg_path, root, err)) {
        std::fprintf(stderr, "JSON Parse Error: %s\n", err.c_str());
        return 1;
    }
    if (!root.is_object()) {
        std::fprintf(stderr, "JSON root must be an object.\n");
        return 1;
    }

    std::string cif_path = root.get_string("cif_path", "data/MgMOF-74.cif");
    int rep_a = 1, rep_b = 1, rep_c = 4;
    if (root.has("supercell") && root.at("supercell").is_array() && root.at("supercell").arr.size() >= 3) {
        rep_a = (int)std::llround(root.at("supercell").arr[0].dval);
        rep_b = (int)std::llround(root.at("supercell").arr[1].dval);
        rep_c = (int)std::llround(root.at("supercell").arr[2].dval);
    }

    double T_K = (temp_override > 0.0) ? temp_override : root.get_double("temperature_K", 298.0);
    std::string ff_str = to_lower_str(root.get_string("forcefield", "uff"));
    int ff_mode = (ff_str == "mercado2016" || ff_str == "mercado") ? 1 : 0;

    std::string interp_str = to_lower_str(root.get_string("interpolation", "tricubic"));
    int use_tricubic = (interp_str == "trilinear") ? 0 : 1;

    std::string pol_str = to_lower_str(root.get_string("polarization", "none"));
    int pol_mode = 0;
    if (pol_str == "com" || pol_str == "pol-com") pol_mode = 1;
    else if (pol_str == "3site" || pol_str == "pol-3site") pol_mode = 2;
    else if (pol_str == "scf" || pol_str == "pol-scf") pol_mode = 3;

    double r_cut_A = root.get_double("r_cut_A", 11.0);
    double ewald_alpha = root.get_double("ewald_alpha", 0.30);
    double dsf_alpha = root.get_double("dsf_alpha", 0.20);
    double grid_spacing_A = root.get_double("grid_spacing_A", 0.10);
    double kij = root.get_double("kij", 0.0);
    int replicas = std::max(1, root.get_int("replicas", 4));
    int eq_steps = std::max(1000, root.get_int("eq_steps", 200000));
    int prod_steps = std::max(1000, root.get_int("prod_steps", 800000));
    int sample_stride = std::max(1, root.get_int("sample_stride", 20));
    int recompute_stride = std::max(1000, root.get_int("recompute_stride", 50000));
    int max_mol = std::clamp(root.get_int("max_mol", 256), 16, 256);
    int enable_swap = root.get_bool("enable_swap", true) ? 1 : 0;
    uint64_t seed = (uint64_t)std::llround(root.get_double("seed", 20261123.0));
    std::string output_csv = !out_override.empty() ? out_override : root.get_string("output_csv", "build/cli_out.csv");

    Cell cell;
    std::vector<FrameworkAtom> unit_atoms;
    CifStructureInfo info;
    if (!load_cif_framework(cif_path, rep_a, rep_b, rep_c, cell, unit_atoms, info)) {
        return 1;
    }

    int nx = 0, ny = 0, nz = 0;
    if (root.has("grid_dims") && root.at("grid_dims").is_array() && root.at("grid_dims").arr.size() >= 3) {
        nx = (int)std::llround(root.at("grid_dims").arr[0].dval);
        ny = (int)std::llround(root.at("grid_dims").arr[1].dval);
        nz = (int)std::llround(root.at("grid_dims").arr[2].dval);
    } else {
        nx = round_mult8_clamp(info.a / grid_spacing_A, 32, 320);
        ny = round_mult8_clamp(info.b / grid_spacing_A, 32, 320);
        nz = round_mult8_clamp(info.c / grid_spacing_A, 32, 320);
    }

    // Parse 1 or 2 guest species
    std::vector<std::string> sp_names;
    std::vector<double> sp_y;
    if (root.has("species") && root.at("species").is_array() && !root.at("species").arr.empty()) {
        for (const auto& item : root.at("species").arr) {
            if (sp_names.size() >= 2) break;
            std::string nm = item.get_string("name", "CO2");
            double y = item.get_double("mole_fraction", 1.0);
            sp_names.push_back(nm);
            sp_y.push_back(y);
        }
    } else {
        sp_names = {"CO2"};
        sp_y = {1.0};
    }

    int n_comp = (int)sp_names.size();
    double y0 = 1.0, y1 = 0.0;
    SpeciesDef sp0{}, sp1{};
    if (!make_species_by_name(sp_names[0], ff_mode, sp0)) {
        std::fprintf(stderr, "Unsupported species[0]: %s\n", sp_names[0].c_str());
        return 1;
    }
    if (n_comp == 2) {
        if (!make_species_by_name(sp_names[1], ff_mode, sp1)) {
            std::fprintf(stderr, "Unsupported species[1]: %s\n", sp_names[1].c_str());
            return 1;
        }
        double ysum = sp_y[0] + sp_y[1];
        if (ysum <= 0.0) { std::fprintf(stderr, "Invalid mole fractions.\n"); return 1; }
        y0 = sp_y[0] / ysum;
        y1 = sp_y[1] / ysum;
    } else {
        sp1 = sp0;
    }

    std::vector<double> pressures;
    if (root.has("pressures_bar") && root.at("pressures_bar").is_array()) {
        for (const auto& pv : root.at("pressures_bar").arr) {
            pressures.push_back(pv.dval);
        }
    }
    if (pressures.empty()) pressures = {0.15, 1.0, 5.0};

    double mass_super_gmol = info.unit_mass_g_mol * (double)(rep_a * rep_b * rep_c);

    std::printf("=== Unified High-Precision GPU GCMC CLI ===\n");
    std::printf("CIF        : %s (%dx%dx%d supercell, %d atoms/uc, mass_super=%.3f g/mol)\n",
                cif_path.c_str(), rep_a, rep_b, rep_c, info.num_atoms, mass_super_gmol);
    std::printf("Physics    : T=%.2f K | FF=%s | Interp=%s (%dx%dx%d) | Pol=%s | r_cut=%.2f A\n",
                T_K, ff_mode == 1 ? "Mercado2016" : "UFF+REPEAT",
                use_tricubic ? "Tricubic" : "Trilinear", nx, ny, nz,
                pol_mode == 3 ? "Pol-SCF" : (pol_mode == 2 ? "Pol-3Site" : (pol_mode == 1 ? "Pol-COM" : "None")),
                r_cut_A);
    if (n_comp == 1) {
        std::printf("Component  : %s (y=1.00)\n", sp_names[0].c_str());
    } else {
        std::printf("Components : %s (y=%.3f) / %s (y=%.3f), Swap=%s\n",
                    sp_names[0].c_str(), y0, sp_names[1].c_str(), y1, enable_swap ? "ON" : "OFF");
    }

    MixGrid3DView d_grid{};
    UnifiedGridStats gst{};
    build_unified_grids_gpu(cell, unit_atoms, nx, ny, nz, r_cut_A, ewald_alpha, ff_mode, d_grid, gst);
    std::printf("Grid Build : %.2f ms (8 grids) | 500-pt Tricubic MAE: C_CO2=%.3f K, O_CO2=%.3f K, N2=%.3f K, CH4=%.3f K, Phi=%.4f K/e\n",
                gst.build_ms, gst.pore_tricubic_mae_lj_K[0], gst.pore_tricubic_mae_lj_K[1],
                gst.pore_tricubic_mae_lj_K[2], gst.pore_tricubic_mae_lj_K[3], gst.pore_tricubic_mae_phi_Ke);

    std::vector<BinaryFugacityResult> fugs(pressures.size());
    std::vector<UnifiedBoxSpec> boxes;
    for (size_t ip = 0; ip < pressures.size(); ++ip) {
        fugs[ip] = compute_binary_fugacity_pr(T_K, pressures[ip], sp0, sp1, y0, y1, kij);
        for (int r = 0; r < replicas; ++r) {
            boxes.push_back({pressures[ip], y0, y1, fugs[ip].f0_Pa, fugs[ip].f1_Pa, r});
        }
    }

    UnifiedGcmcConfig gcfg;
    gcfg.T_K = T_K;
    gcfg.r_cut_A = r_cut_A;
    gcfg.ewald_alpha = ewald_alpha;
    gcfg.dsf_alpha = dsf_alpha;
    gcfg.ff_mode = ff_mode;
    gcfg.use_tricubic = use_tricubic;
    gcfg.pol_mode = pol_mode;
    gcfg.n_comp = n_comp;
    gcfg.enable_swap = enable_swap;
    gcfg.eq_steps = eq_steps;
    gcfg.prod_steps = prod_steps;
    gcfg.sample_stride = sample_stride;
    gcfg.recompute_stride = recompute_stride;
    gcfg.max_mol = max_mol;
    gcfg.framework_mass_super_gmol = mass_super_gmol;
    gcfg.base_seed = seed;

    std::vector<UnifiedBoxResult> results;
    double gpu_ms = run_unified_gcmc_gpu(cell, d_grid, sp0, sp1, gcfg, boxes, results);
    free_unified_grids_gpu(d_grid);

    double total_steps = (double)boxes.size() * (double)(eq_steps + prod_steps);
    double avg_scf = 0.0;
    for (const auto& r : results) avg_scf += r.avg_scf_iters;
    if (!results.empty()) avg_scf /= (double)results.size();
    std::printf("GCMC Run   : %d boxes x %.2fM (%d eq + %d prod) in %.2f ms (%.2f M steps/sec)%s\n",
                (int)boxes.size(), (double)(eq_steps + prod_steps) * 1.0e-6, eq_steps, prod_steps,
                gpu_ms, total_steps / (gpu_ms * 1e3),
                pol_mode == 3 ? "" : "\n");
    if (pol_mode == 3) {
        std::printf("SCF Conv   : mean %.2f iterations / trial move (tol = 1e-10 e*A)\n\n", avg_scf);
    }

    std::filesystem::path out_p(output_csv);
    if (out_p.has_parent_path()) {
        std::filesystem::create_directories(out_p.parent_path());
    }
    std::ofstream fout(output_csv);
    if (!fout) {
        std::fprintf(stderr, "Error: cannot write output CSV: %s\n", output_csv.c_str());
        return 1;
    }

    if (n_comp == 1) {
        fout << "P_bar,f_bar,phi,N_mean,uptake_mmol_g,uptake_sem_mmol_g,Qst_kJ_mol,Qst_sem_kJ_mol,avg_Upol_kJ,max_drift_K,acc_ins,acc_del\n";
        std::printf("  P(bar) | Uptake (mmol/g)      | Qst (kJ/mol)   | Upol(kJ) | Drift(K) | acc_i / acc_d\n");
        std::printf("  --------------------------------------------------------------------------------------\n");
    } else {
        fout << "P_bar,y0,y1,f0_bar,f1_bar,N0_mean,N1_mean,uptake0_mmol_g,uptake0_sem_mmol_g,uptake1_mmol_g,uptake1_sem_mmol_g,selectivity,Qst_kJ_mol,avg_Upol0_kJ,avg_Upol1_kJ,max_drift_K,acc_i0,acc_d0,acc_i1,acc_d1,acc_sw01,acc_sw10\n";
        std::printf("  P(bar) | Sp0 (mmol/g)     | Sp1 (mmol/g)     | S(0/1) | Qst   | Drift(K) | acc_i0/d0\n");
        std::printf("  -------------------------------------------------------------------------------------\n");
    }

    for (size_t ip = 0; ip < pressures.size(); ++ip) {
        double m0 = 0, m1 = 0, n0 = 0, n1 = 0, qst = 0, sel = 0, up0 = 0, up1 = 0, md = 0;
        double ai0 = 0, ad0 = 0, ai1 = 0, ad1 = 0, asw01 = 0, asw10 = 0;
        for (int r = 0; r < replicas; ++r) {
            const auto& br = results[ip * replicas + r];
            n0 += br.avg_N0; n1 += br.avg_N1;
            m0 += br.uptake0_mmol_g; m1 += br.uptake1_mmol_g;
            qst += br.Qst_kJ_mol; sel += br.selectivity_0_over_1;
            up0 += br.avg_Upol0_kJ; up1 += br.avg_Upol1_kJ;
            if (br.max_energy_drift_K > md) md = br.max_energy_drift_K;
            ai0 += br.acc_i0; ad0 += br.acc_d0; ai1 += br.acc_i1; ad1 += br.acc_d1;
            asw01 += br.acc_sw01; asw10 += br.acc_sw10;
        }
        n0 /= replicas; n1 /= replicas; m0 /= replicas; m1 /= replicas;
        qst /= replicas; sel /= replicas; up0 /= replicas; up1 /= replicas;
        ai0 /= replicas; ad0 /= replicas; ai1 /= replicas; ad1 /= replicas;
        asw01 /= replicas; asw10 /= replicas;

        double v0 = 0, v1 = 0, vq = 0;
        for (int r = 0; r < replicas; ++r) {
            const auto& br = results[ip * replicas + r];
            v0 += (br.uptake0_mmol_g - m0) * (br.uptake0_mmol_g - m0);
            v1 += (br.uptake1_mmol_g - m1) * (br.uptake1_mmol_g - m1);
            vq += (br.Qst_kJ_mol - qst) * (br.Qst_kJ_mol - qst);
        }
        double sem0 = (replicas > 1) ? std::sqrt(v0 / (replicas * (replicas - 1))) : 0.0;
        double sem1 = (replicas > 1) ? std::sqrt(v1 / (replicas * (replicas - 1))) : 0.0;
        double semq = (replicas > 1) ? std::sqrt(vq / (replicas * (replicas - 1))) : 0.0;

        if (n_comp == 1) {
            std::printf("  %6.3f | %8.4f +- %6.4f | %6.2f +- %4.2f | %8.3f | %.2e | %.4f / %.4f\n",
                        pressures[ip], m0, sem0, qst, semq, up0, md, ai0, ad0);
            fout << std::fixed << std::setprecision(6)
                 << pressures[ip] << "," << fugs[ip].f0_bar << "," << fugs[ip].phi0 << ","
                 << n0 << "," << m0 << "," << sem0 << "," << qst << "," << semq << ","
                 << up0 << "," << std::scientific << md << ","
                 << std::fixed << ai0 << "," << ad0 << "\n";
        } else {
            std::printf("  %6.3f | %7.4f +- %5.4f | %7.4f +- %5.4f | %6.2f | %5.2f | %.2e | %.4f/%.4f\n",
                        pressures[ip], m0, sem0, m1, sem1, sel, qst, md, ai0, ad0);
            fout << std::fixed << std::setprecision(6)
                 << pressures[ip] << "," << y0 << "," << y1 << "," << fugs[ip].f0_bar << "," << fugs[ip].f1_bar << ","
                 << n0 << "," << n1 << "," << m0 << "," << sem0 << "," << m1 << "," << sem1 << ","
                 << sel << "," << qst << "," << up0 << "," << up1 << ","
                 << std::scientific << md << ","
                 << std::fixed << ai0 << "," << ad0 << "," << ai1 << "," << ad1 << ","
                 << asw01 << "," << asw10 << "\n";
        }
    }
    std::printf("\nSaved CSV to %s\n", output_csv.c_str());
    return 0;
}
