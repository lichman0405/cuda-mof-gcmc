#pragma once

#include "cell.hpp"
#include "grid3d.hpp"
#include <string>
#include <vector>

namespace mofgcmc {

struct CifStructureInfo {
    double a, b, c;
    double alpha_deg, beta_deg, gamma_deg;
    int num_atoms;
    int count_mg, count_o, count_c, count_h;
    double total_charge_e;
    double unit_mass_g_mol;
    double min_mg_o_A;
    double min_c_o_A;
    double min_c_c_A;
    double min_c_h_A;
};

bool load_mg_mof74_cif(
    const std::string& cif_path,
    int rep_a, int rep_b, int rep_c,
    Cell& cell_out,
    std::vector<FrameworkAtom>& unit_atoms_out,
    CifStructureInfo& info_out);

} // namespace mofgcmc
