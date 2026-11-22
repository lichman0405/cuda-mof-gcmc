#include "cif_loader.hpp"
#include "forcefield.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mofgcmc {

namespace {

std::string trim(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace((unsigned char)s[b])) ++b;
    size_t e = s.size();
    while (e > b && std::isspace((unsigned char)s[e - 1])) --e;
    return s.substr(b, e - b);
}

std::string strip_quotes(const std::string& s) {
    std::string t = trim(s);
    if (t.size() >= 2 &&
        ((t.front() == '\'' && t.back() == '\'') ||
         (t.front() == '"' && t.back() == '"'))) {
        return trim(t.substr(1, t.size() - 2));
    }
    return t;
}

double wrap01(double v) {
    v = v - std::floor(v);
    if (v >= 1.0 - 1.0e-12) v = 0.0;
    return v;
}

double frac_abs_diff(double a, double b) {
    double d = a - b;
    d -= std::round(d);
    return std::fabs(d);
}

// Parse a linear symmetry component like "-x+y+2/3" -> cx*x + cy*y + cz*z + shift
void parse_symop_expr(const std::string& expr_raw,
                      double& cx, double& cy, double& cz, double& shift) {
    cx = cy = cz = shift = 0.0;
    std::string expr;
    for (char ch : expr_raw) {
        if (!std::isspace((unsigned char)ch)) expr.push_back((char)std::tolower((unsigned char)ch));
    }

    size_t i = 0;
    while (i < expr.size()) {
        double sign = 1.0;
        if (expr[i] == '+') { sign = 1.0; ++i; }
        else if (expr[i] == '-') { sign = -1.0; ++i; }

        size_t start = i;
        while (i < expr.size() && expr[i] != '+' && expr[i] != '-') ++i;
        std::string tok = expr.substr(start, i - start);
        if (tok.empty()) continue;

        if (tok == "x") cx += sign;
        else if (tok == "y") cy += sign;
        else if (tok == "z") cz += sign;
        else {
            auto slash = tok.find('/');
            if (slash != std::string::npos) {
                double num = std::stod(tok.substr(0, slash));
                double den = std::stod(tok.substr(slash + 1));
                shift += sign * (num / den);
            } else {
                shift += sign * std::stod(tok);
            }
        }
    }
}

struct SymOp {
    double m[3][3];
    double t[3];
};

SymOp parse_symop(const std::string& line) {
    std::string clean = strip_quotes(line);
    std::stringstream ss(clean);
    std::string part;
    SymOp op{};
    int row = 0;
    while (std::getline(ss, part, ',') && row < 3) {
        parse_symop_expr(part, op.m[row][0], op.m[row][1], op.m[row][2], op.t[row]);
        ++row;
    }
    return op;
}

struct AsymAtom {
    std::string label;
    std::string symbol;
    double fx, fy, fz;
};

} // namespace

bool load_mg_mof74_cif(
    const std::string& cif_path,
    int rep_a, int rep_b, int rep_c,
    Cell& cell_out,
    std::vector<FrameworkAtom>& unit_atoms_out,
    CifStructureInfo& info_out) {

    std::ifstream fin(cif_path);
    if (!fin) {
        std::fprintf(stderr, "Error: cannot open CIF file: %s\n", cif_path.c_str());
        return false;
    }

    info_out = CifStructureInfo{};
    std::vector<SymOp> symops;
    std::vector<AsymAtom> asym_atoms;

    enum Section { SEC_NONE, SEC_SYM, SEC_ATOM };
    Section sec = SEC_NONE;
    std::vector<std::string> atom_cols;

    std::string line;
    while (std::getline(fin, line)) {
        std::string t = trim(line);
        if (t.empty() || t[0] == '#') continue;

        if (t == "loop_") {
            sec = SEC_NONE;
            atom_cols.clear();
            continue;
        }

        if (t[0] == '_') {
            std::istringstream iss(t);
            std::string key;
            iss >> key;
            if (key == "_cell_length_a") { iss >> info_out.a; continue; }
            if (key == "_cell_length_b") { iss >> info_out.b; continue; }
            if (key == "_cell_length_c") { iss >> info_out.c; continue; }
            if (key == "_cell_angle_alpha") { iss >> info_out.alpha_deg; continue; }
            if (key == "_cell_angle_beta")  { iss >> info_out.beta_deg; continue; }
            if (key == "_cell_angle_gamma") { iss >> info_out.gamma_deg; continue; }
            if (key == "_symmetry_equiv_pos_as_xyz" || key == "_space_group_symop_operation_xyz") {
                sec = SEC_SYM;
                continue;
            }
            if (key.rfind("_atom_site_", 0) == 0) {
                sec = SEC_ATOM;
                atom_cols.push_back(key);
                continue;
            }
            sec = SEC_NONE;
            continue;
        }

        if (sec == SEC_SYM) {
            symops.push_back(parse_symop(t));
        } else if (sec == SEC_ATOM) {
            std::istringstream iss(t);
            std::vector<std::string> tokens;
            std::string tok;
            while (iss >> tok) tokens.push_back(tok);
            if (tokens.size() < atom_cols.size()) continue;

            AsymAtom a{};
            for (size_t c = 0; c < atom_cols.size(); ++c) {
                if (atom_cols[c] == "_atom_site_label") a.label = tokens[c];
                else if (atom_cols[c] == "_atom_site_type_symbol") a.symbol = tokens[c];
                else if (atom_cols[c] == "_atom_site_fract_x") a.fx = std::stod(tokens[c]);
                else if (atom_cols[c] == "_atom_site_fract_y") a.fy = std::stod(tokens[c]);
                else if (atom_cols[c] == "_atom_site_fract_z") a.fz = std::stod(tokens[c]);
            }
            asym_atoms.push_back(a);
        }
    }

    if (symops.empty()) {
        std::fprintf(stderr, "Error: no symmetry operations parsed from %s\n", cif_path.c_str());
        return false;
    }

    cell_out.init(info_out.a, info_out.b, info_out.c,
                  info_out.alpha_deg, info_out.beta_deg, info_out.gamma_deg,
                  rep_a, rep_b, rep_c);

    unit_atoms_out.clear();
    for (const auto& aa : asym_atoms) {
        int elem = element_from_symbol(aa.symbol);
        if (elem < 0) {
            std::fprintf(stderr, "Error: unknown element '%s'\n", aa.symbol.c_str());
            return false;
        }
        UffParam lj = get_uff_param(elem);
        double q = get_repeat_charge(aa.label);
        int type_id = atom_type_from_label(aa.label);

        for (const auto& op : symops) {
            double sa = wrap01(op.m[0][0] * aa.fx + op.m[0][1] * aa.fy + op.m[0][2] * aa.fz + op.t[0]);
            double sb = wrap01(op.m[1][0] * aa.fx + op.m[1][1] * aa.fy + op.m[1][2] * aa.fz + op.t[1]);
            double sc = wrap01(op.m[2][0] * aa.fx + op.m[2][1] * aa.fy + op.m[2][2] * aa.fz + op.t[2]);

            bool dup = false;
            for (const auto& ex : unit_atoms_out) {
                if (frac_abs_diff(sa, ex.sa) < 1.0e-3 &&
                    frac_abs_diff(sb, ex.sb) < 1.0e-3 &&
                    frac_abs_diff(sc, ex.sc) < 1.0e-3) {
                    dup = true;
                    break;
                }
            }
            if (!dup) {
                double cx, cy, cz;
                cell_out.unit_frac_to_cart(sa, sb, sc, cx, cy, cz);
                FrameworkAtom fa{};
                fa.x = (float)cx;
                fa.y = (float)cy;
                fa.z = (float)cz;
                fa.sa = (float)sa;
                fa.sb = (float)sb;
                fa.sc = (float)sc;
                fa.eps_K = (float)lj.eps_K;
                fa.sigma_A = (float)lj.sigma_A;
                fa.q = q;
                fa.elem = elem;
                fa.atom_type_id = type_id;
                unit_atoms_out.push_back(fa);
            }
        }
    }

    info_out.num_atoms = (int)unit_atoms_out.size();
    info_out.total_charge_e = 0.0;
    for (const auto& fa : unit_atoms_out) {
        info_out.total_charge_e += fa.q;
        if (fa.elem == ELEM_MG) ++info_out.count_mg;
        else if (fa.elem == ELEM_O) ++info_out.count_o;
        else if (fa.elem == ELEM_C) ++info_out.count_c;
        else if (fa.elem == ELEM_H) ++info_out.count_h;
    }

    info_out.unit_mass_g_mol =
        info_out.count_mg * 24.305 +
        info_out.count_o  * 15.999 +
        info_out.count_c  * 12.011 +
        info_out.count_h  * 1.008;

    // Verify shortest interatomic bond lengths in the 1x1x1 unit cell
    info_out.min_mg_o_A = 1e9;
    info_out.min_c_o_A = 1e9;
    info_out.min_c_c_A = 1e9;
    info_out.min_c_h_A = 1e9;

    const int N = (int)unit_atoms_out.size();
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            int e1 = unit_atoms_out[i].elem;
            int e2 = unit_atoms_out[j].elem;
            double dsa = unit_atoms_out[i].sa - unit_atoms_out[j].sa;
            double dsb = unit_atoms_out[i].sb - unit_atoms_out[j].sb;
            double dsc = unit_atoms_out[i].sc - unit_atoms_out[j].sc;
            dsa -= std::round(dsa);
            dsb -= std::round(dsb);
            dsc -= std::round(dsc);

            double min_r2 = 1e30;
            for (int na = -1; na <= 1; ++na) {
                for (int nb = -1; nb <= 1; ++nb) {
                    for (int nc = -1; nc <= 1; ++nc) {
                        double dx, dy, dz;
                        cell_out.unit_frac_to_cart(dsa + na, dsb + nb, dsc + nc, dx, dy, dz);
                        double r2 = dx * dx + dy * dy + dz * dz;
                        if (r2 < min_r2) min_r2 = r2;
                    }
                }
            }
            double r = std::sqrt(min_r2);
            if ((e1 == ELEM_MG && e2 == ELEM_O) || (e1 == ELEM_O && e2 == ELEM_MG)) {
                if (r < info_out.min_mg_o_A) info_out.min_mg_o_A = r;
            } else if ((e1 == ELEM_C && e2 == ELEM_O) || (e1 == ELEM_O && e2 == ELEM_C)) {
                if (r < info_out.min_c_o_A) info_out.min_c_o_A = r;
            } else if (e1 == ELEM_C && e2 == ELEM_C) {
                if (r < info_out.min_c_c_A) info_out.min_c_c_A = r;
            } else if ((e1 == ELEM_C && e2 == ELEM_H) || (e1 == ELEM_H && e2 == ELEM_C)) {
                if (r < info_out.min_c_h_A) info_out.min_c_h_A = r;
            }
        }
    }

    if (info_out.num_atoms != 162 ||
        info_out.count_mg != 18 || info_out.count_o != 54 ||
        info_out.count_c != 72 || info_out.count_h != 18) {
        std::fprintf(stderr, "Error: unexpected Mg-MOF-74 atom count: %d (Mg=%d O=%d C=%d H=%d)\n",
                     info_out.num_atoms, info_out.count_mg, info_out.count_o,
                     info_out.count_c, info_out.count_h);
        return false;
    }
    if (std::fabs(info_out.total_charge_e) > 1.0e-6) {
        std::fprintf(stderr, "Error: non-neutral framework charge sum = %.8f e\n",
                     info_out.total_charge_e);
        return false;
    }
    return true;
}

} // namespace mofgcmc
