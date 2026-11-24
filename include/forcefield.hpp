#pragma once

#include <cctype>
#include <cmath>
#include <string>

namespace mofgcmc {

// Electrostatic constant e^2 / (4 * pi * eps0 * k_B) in (K * Angstrom / e^2)
constexpr double KE_K = 167100.9558738398;
// Boltzmann constant conversion: 1 kJ/mol = 120.2723625 K (R = 8.314462618e-3 kJ/(mol*K))
constexpr double K_TO_KJMOL = 8.31446261815324e-3;
constexpr double AVOGADRO = 6.02214076e23;

enum ElementId {
    ELEM_MG = 0,
    ELEM_O  = 1,
    ELEM_C  = 2,
    ELEM_H  = 3,
    ELEM_N  = 4,
    ELEM_S  = 5,
    ELEM_F  = 6,
    ELEM_CL = 7,
    ELEM_ZN = 8,
    ELEM_CU = 9,
    ELEM_FE = 10,
    ELEM_CO = 11,
    ELEM_NI = 12,
    ELEM_MN = 13,
    ELEM_AL = 14,
    ELEM_CR = 15,
    ELEM_ZR = 16,
    ELEM_TI = 17,
    ELEM_V  = 18
};

struct UffParam {
    double eps_K;
    double sigma_A;
};

struct PeriodicElementData {
    double uff_eps_K;
    double uff_sigma_A;
    double mass_g_mol;
    const char* symbol;
};

// Complete 103-element UFF table (Rappe et al., J. Am. Chem. Soc. 1992, 114, 10024-10035)
// Index Z = 0..103; eps_K = D_i * 503.21959899, sigma_A = x_i / 2^(1/6).
inline const PeriodicElementData& get_periodic_element_data(int Z) {
    static const PeriodicElementData TABLE[104] = {
        /*  0 Du*/ {   0.0000, 1.0000,   0.0000, ""   },
        /*  1 H */ {  22.1417, 2.5711,   1.0080, "H"  },
        /*  2 He*/ {  28.1803, 2.1043,   4.0026, "He" },
        /*  3 Li*/ {  12.5805, 2.1836,   6.9400, "Li" },
        /*  4 Be*/ {  42.7737, 2.4455,   9.0122, "Be" },
        /*  5 B */ {  90.5795, 3.6375,  10.8100, "B"  },
        /*  6 C */ {  52.8381, 3.4309,  12.0110, "C"  },
        /*  7 N */ {  34.7221, 3.2607,  14.0070, "N"  },
        /*  8 O */ {  30.1932, 3.1181,  15.9990, "O"  },
        /*  9 F */ {  25.1610, 2.9970,  18.9980, "F"  },
        /* 10 Ne*/ {  21.1352, 2.8892,  20.1797, "Ne" },
        /* 11 Na*/ {  15.0966, 2.6576,  22.9898, "Na" },
        /* 12 Mg*/ {  55.8574, 2.6914,  24.3050, "Mg" },
        /* 13 Al*/ { 254.1260, 4.0082,  26.9820, "Al" },
        /* 14 Si*/ { 202.2943, 3.8264,  28.0850, "Si" },
        /* 15 P */ { 153.4820, 3.6946,  30.9738, "P"  },
        /* 16 S */ { 137.8820, 3.5948,  32.0600, "S"  },
        /* 17 Cl*/ { 114.2310, 3.5164,  35.4500, "Cl" },
        /* 18 Ar*/ {  93.0956, 3.4460,  39.9480, "Ar" },
        /* 19 K */ {  17.6127, 3.3961,  39.0983, "K"  },
        /* 20 Ca*/ { 119.7663, 3.0282,  40.0780, "Ca" },
        /* 21 Sc*/ {   9.5612, 2.9355,  44.9559, "Sc" },
        /* 22 Ti*/ {   8.5547, 2.8286,  47.8670, "Ti" },
        /* 23 V */ {   8.0515, 2.8009,  50.9420, "V"  },
        /* 24 Cr*/ {   7.5483, 2.6932,  51.9960, "Cr" },
        /* 25 Mn*/ {   6.5419, 2.6380,  54.9380, "Mn" },
        /* 26 Fe*/ {   6.5419, 2.5943,  55.8450, "Fe" },
        /* 27 Co*/ {   7.0451, 2.5587,  58.9330, "Co" },
        /* 28 Ni*/ {   7.5483, 2.5248,  58.6930, "Ni" },
        /* 29 Cu*/ {   2.5161, 3.1137,  63.5460, "Cu" },
        /* 30 Zn*/ {  62.3992, 2.4616,  65.3800, "Zn" },
        /* 31 Ga*/ { 208.8361, 3.9048,  69.7230, "Ga" },
        /* 32 Ge*/ { 190.7202, 3.8130,  72.6300, "Ge" },
        /* 33 As*/ { 155.4949, 3.7685,  74.9216, "As" },
        /* 34 Se*/ { 146.4369, 3.7462,  78.9710, "Se" },
        /* 35 Br*/ { 126.3081, 3.7320,  79.9040, "Br" },
        /* 36 Kr*/ { 110.7083, 3.6892,  83.7980, "Kr" },
        /* 37 Rb*/ {  20.1288, 3.6652,  85.4678, "Rb" },
        /* 38 Sr*/ { 118.2566, 3.2438,  87.6200, "Sr" },
        /* 39 Y */ {  36.2318, 2.9801,  88.9058, "Y"  },
        /* 40 Zr*/ {  34.7221, 2.7832,  91.2240, "Zr" },
        /* 41 Nb*/ {  29.6900, 2.8197,  92.9064, "Nb" },
        /* 42 Mo*/ {  28.1803, 2.7190,  95.9500, "Mo" },
        /* 43 Tc*/ {  24.1545, 2.6709,  97.9072, "Tc" },
        /* 44 Ru*/ {  28.1803, 2.6397, 101.0700, "Ru" },
        /* 45 Rh*/ {  26.6706, 2.6094, 102.9055, "Rh" },
        /* 46 Pd*/ {  24.1545, 2.5827, 106.4200, "Pd" },
        /* 47 Ag*/ {  18.1159, 2.8045, 107.8682, "Ag" },
        /* 48 Cd*/ { 114.7341, 2.5373, 112.4140, "Cd" },
        /* 49 In*/ { 301.4285, 3.9761, 114.8180, "In" },
        /* 50 Sn*/ { 285.3255, 3.9128, 118.7100, "Sn" },
        /* 51 Sb*/ { 225.9456, 3.9378, 121.7600, "Sb" },
        /* 52 Te*/ { 200.2814, 3.9823, 127.6000, "Te" },
        /* 53 I */ { 170.5914, 4.0090, 126.9045, "I"  },
        /* 54 Xe*/ { 167.0689, 3.9235, 131.2930, "Xe" },
        /* 55 Cs*/ {  22.6449, 4.0242, 132.9055, "Cs" },
        /* 56 Ba*/ { 183.1719, 3.2990, 137.3270, "Ba" },
        /* 57 La*/ {   8.5547, 3.1377, 138.9055, "La" },
        /* 58 Ce*/ {   6.5419, 3.1680, 140.1160, "Ce" },
        /* 59 Pr*/ {   5.0322, 3.2126, 140.9077, "Pr" },
        /* 60 Nd*/ {   5.0322, 3.1850, 144.2420, "Nd" },
        /* 61 Pm*/ {   4.5290, 3.1600, 144.9128, "Pm" },
        /* 62 Sm*/ {   4.0258, 3.1360, 150.3600, "Sm" },
        /* 63 Eu*/ {   4.0258, 3.1119, 151.9640, "Eu" },
        /* 64 Gd*/ {   4.5290, 3.0005, 157.2500, "Gd" },
        /* 65 Tb*/ {   3.5225, 3.0745, 158.9254, "Tb" },
        /* 66 Dy*/ {   3.5225, 3.0540, 162.5000, "Dy" },
        /* 67 Ho*/ {   3.5225, 3.0371, 164.9303, "Ho" },
        /* 68 Er*/ {   3.5225, 3.0210, 167.2590, "Er" },
        /* 69 Tm*/ {   3.0193, 3.0059, 168.9342, "Tm" },
        /* 70 Yb*/ { 114.7341, 2.9890, 173.0540, "Yb" },
        /* 71 Lu*/ {  20.6320, 3.2429, 174.9668, "Lu" },
        /* 72 Hf*/ {  36.2318, 2.7983, 178.4900, "Hf" },
        /* 73 Ta*/ {  40.7608, 2.8241, 180.9479, "Ta" },
        /* 74 W */ {  33.7157, 2.7342, 183.8400, "W"  },
        /* 75 Re*/ {  33.2125, 2.6317, 186.2070, "Re" },
        /* 76 Os*/ {  18.6191, 2.7796, 190.2300, "Os" },
        /* 77 Ir*/ {  36.7350, 2.5302, 192.2170, "Ir" },
        /* 78 Pt*/ {  40.2576, 2.4535, 195.0840, "Pt" },
        /* 79 Au*/ {  19.6256, 2.9337, 196.9666, "Au" },
        /* 80 Hg*/ { 193.7395, 2.4099, 200.5920, "Hg" },
        /* 81 Tl*/ { 342.1893, 3.8727, 204.3800, "Tl" },
        /* 82 Pb*/ { 333.6346, 3.8282, 207.2000, "Pb" },
        /* 83 Bi*/ { 260.6678, 3.8932, 208.9804, "Bi" },
        /* 84 Po*/ { 163.5464, 4.1952, 208.9824, "Po" },
        /* 85 At*/ { 142.9144, 4.2318, 209.9872, "At" },
        /* 86 Rn*/ { 124.7985, 4.2451, 222.0176, "Rn" },
        /* 87 Fr*/ {  25.1610, 4.3654, 223.0197, "Fr" },
        /* 88 Ra*/ { 203.3007, 3.2758, 226.0254, "Ra" },
        /* 89 Ac*/ {  16.6062, 3.0985, 227.0277, "Ac" },
        /* 90 Th*/ {  13.0837, 3.0255, 232.0377, "Th" },
        /* 91 Pa*/ {  11.0708, 3.0504, 231.0359, "Pa" },
        /* 92 U */ {  11.0708, 3.0246, 238.0289, "U"  },
        /* 93 Np*/ {   9.5612, 3.0504, 237.0482, "Np" },
        /* 94 Pu*/ {   8.0515, 3.0504, 244.0642, "Pu" },
        /* 95 Am*/ {   7.0451, 3.0121, 243.0614, "Am" },
        /* 96 Cm*/ {   6.5419, 2.9631, 247.0703, "Cm" },
        /* 97 Bk*/ {   6.5419, 2.9747, 247.0703, "Bk" },
        /* 98 Cf*/ {   6.5419, 2.9515, 251.0796, "Cf" },
        /* 99 Es*/ {   6.0386, 2.9391, 252.0830, "Es" },
        /*100 Fm*/ {   6.0386, 2.9275, 257.0951, "Fm" },
        /*101 Md*/ {   5.5354, 2.9168, 258.0984, "Md" },
        /*102 No*/ {   5.5354, 2.8936, 259.1010, "No" },
        /*103 Lr*/ {   5.5354, 2.8829, 262.1100, "Lr" }
    };
    if (Z >= 1 && Z <= 103) return TABLE[Z];
    return TABLE[0];
}

inline UffParam get_uff_param_by_Z(int Z) {
    const auto& d = get_periodic_element_data(Z);
    return {d.uff_eps_K, d.uff_sigma_A};
}

inline double get_atomic_mass_by_Z(int Z) {
    return get_periodic_element_data(Z).mass_g_mol;
}

inline int atomic_number_from_symbol(const std::string& sym) {
    if (sym.empty()) return 0;
    std::string norm;
    for (char ch : sym) {
        if (std::isalpha((unsigned char)ch)) {
            if (norm.empty()) norm.push_back((char)std::toupper((unsigned char)ch));
            else norm.push_back((char)std::tolower((unsigned char)ch));
        } else {
            break;
        }
    }
    if (norm == "D") return 1;
    if (norm == "Lw") return 103;
    for (int z = 1; z <= 103; ++z) {
        if (norm == get_periodic_element_data(z).symbol) return z;
    }
    return 0;
}

inline int atomic_number_to_elem(int Z) {
    switch (Z) {
        case 12: return ELEM_MG;
        case 8:  return ELEM_O;
        case 6:  return ELEM_C;
        case 1:  return ELEM_H;
        case 7:  return ELEM_N;
        case 16: return ELEM_S;
        case 9:  return ELEM_F;
        case 17: return ELEM_CL;
        case 30: return ELEM_ZN;
        case 29: return ELEM_CU;
        case 26: return ELEM_FE;
        case 27: return ELEM_CO;
        case 28: return ELEM_NI;
        case 25: return ELEM_MN;
        case 13: return ELEM_AL;
        case 24: return ELEM_CR;
        case 40: return ELEM_ZR;
        case 22: return ELEM_TI;
        case 23: return ELEM_V;
        default: return (Z >= 1 && Z <= 103) ? (100 + Z) : -1;
    }
}

inline int get_atomic_number(int elem) {
    switch (elem) {
        case ELEM_MG: return 12;
        case ELEM_O:  return 8;
        case ELEM_C:  return 6;
        case ELEM_H:  return 1;
        case ELEM_N:  return 7;
        case ELEM_S:  return 16;
        case ELEM_F:  return 9;
        case ELEM_CL: return 17;
        case ELEM_ZN: return 30;
        case ELEM_CU: return 29;
        case ELEM_FE: return 26;
        case ELEM_CO: return 27;
        case ELEM_NI: return 28;
        case ELEM_MN: return 25;
        case ELEM_AL: return 13;
        case ELEM_CR: return 24;
        case ELEM_ZR: return 40;
        case ELEM_TI: return 22;
        case ELEM_V:  return 23;
        default:      return (elem >= 101 && elem <= 203) ? (elem - 100) : 0;
    }
}

// UFF parameters (Rappe et al. JACS 1992): eps in Kelvin, sigma in Angstrom
inline UffParam get_uff_param(int elem) {
    int Z = get_atomic_number(elem);
    return (Z > 0) ? get_uff_param_by_Z(Z) : UffParam{0.0, 1.0};
}

inline double get_atomic_mass(int elem) {
    int Z = get_atomic_number(elem);
    return (Z > 0) ? get_atomic_mass_by_Z(Z) : 0.0;
}

inline int element_from_symbol(const std::string& sym) {
    int Z = atomic_number_from_symbol(sym);
    return (Z > 0) ? atomic_number_to_elem(Z) : -1;
}

// REPEAT partial atomic charges for Mg-MOF-74 (exact sum = 0.000000 e)
inline double get_repeat_charge(const std::string& label) {
    if (label == "Mg1") return +1.566;
    if (label == "O1")  return -0.915;
    if (label == "O2")  return -0.768;
    if (label == "O3")  return -0.907;
    if (label == "C1")  return +0.936;
    if (label == "C2")  return -0.325;
    if (label == "C3")  return +0.457;
    if (label == "C4")  return -0.235;
    if (label == "H1")  return +0.191;
    return 0.0;
}

inline int atom_type_from_label(const std::string& label) {
    if (label == "Mg1" || label == "Mof_Mg") return 0;
    if (label == "O1"  || label == "Mof_Oa") return 1;
    if (label == "O2"  || label == "Mof_Ob") return 2;
    if (label == "O3"  || label == "Mof_Oc") return 3;
    if (label == "C1"  || label == "Mof_Ca") return 4;
    if (label == "C2"  || label == "Mof_Cb") return 5;
    if (label == "C3"  || label == "Mof_Cc") return 6;
    if (label == "C4"  || label == "Mof_Cd") return 7;
    if (label == "H1"  || label == "Mof_H")  return 8;
    return -1;
}

// Mercado et al. (J. Phys. Chem. C 2016, 120, 12590-12604) Mg-MOF-74 charges + EPM2 CO2
namespace mercado2016 {
    // Charges for type_id 0..8 (exact sum = 0.000000 e)
    inline double get_framework_charge(int type_id) {
        static const double Q[9] = {
            +1.560, // 0: Mof_Mg (Mg1)
            -0.899, // 1: Mof_Oa (O1)
            -0.752, // 2: Mof_Ob (O2)
            -0.903, // 3: Mof_Oc (O3)
            +0.900, // 4: Mof_Ca (C1)
            -0.314, // 5: Mof_Cb (C2)
            +0.456, // 6: Mof_Cc (C3)
            -0.234, // 7: Mof_Cd (C4)
            +0.186  // 8: Mof_H  (H1)
        };
        return (type_id >= 0 && type_id < 9) ? Q[type_id] : 0.0;
    }

    // EPM2 CO2 parameters used with Mercado 2016 FF (r_cut shifted)
    constexpr double Q_C_EPM2   = +0.6512;
    constexpr double EPS_C_EPM2 = 28.129;
    constexpr double SIG_C_EPM2 = 2.757;
    constexpr double Q_O_EPM2   = -0.3256;
    constexpr double EPS_O_EPM2 = 80.507;
    constexpr double SIG_O_EPM2 = 3.033;
    constexpr double BOND_CO_EPM2 = 1.160;
}

// TraPPE rigid 3-site CO2 parameters
namespace trappe_co2 {
    constexpr double Q_C      = +0.70;
    constexpr double EPS_C_K  = 27.0;
    constexpr double SIG_C_A  = 2.800;

    constexpr double Q_O      = -0.35;
    constexpr double EPS_O_K  = 79.0;
    constexpr double SIG_O_A  = 3.050;

    constexpr double BOND_CO_A = 1.160;

    // Experimental & ab-initio anisotropic CO2 polarizability tensor (Angstrom^3)
    constexpr double ALPHA_PAR_A3  = 4.05;
    constexpr double ALPHA_PERP_A3 = 1.95;
    constexpr double ALPHA_ISO_A3  = (ALPHA_PAR_A3 + 2.0 * ALPHA_PERP_A3) / 3.0; // 2.65 A^3

    // 3-site distributed polarizability weights (sum = 1.0)
    constexpr double W_POL_C = 0.40;
    constexpr double W_POL_O = 0.30;

    // Standard Thole damping parameter for induced-dipole-induced-dipole interaction
    constexpr double THOLE_A = 2.1304;
}

// TraPPE rigid 3-site N2 parameters (Potoff & Siepmann, AIChE J. 2001)
namespace trappe_n2 {
    constexpr double Q_M      = +0.964;
    constexpr double Q_N      = -0.482;
    constexpr double EPS_N_K  = 36.0;
    constexpr double SIG_N_A  = 3.310;
    constexpr double BOND_NM_A = 0.550;

    constexpr double ALPHA_PAR_A3  = 2.38;
    constexpr double ALPHA_PERP_A3 = 1.45;
    constexpr double ALPHA_ISO_A3  = (ALPHA_PAR_A3 + 2.0 * ALPHA_PERP_A3) / 3.0; // 1.76 A^3
    constexpr double W_POL_N = 0.50;
}

// TraPPE-UA single-site CH4 parameters (Martin & Siepmann, JPCB 1998)
namespace trappe_ch4 {
    constexpr double Q_CH4     = 0.0;
    constexpr double EPS_CH4_K = 148.0;
    constexpr double SIG_CH4_A = 3.730;

    constexpr double ALPHA_PAR_A3  = 2.59;
    constexpr double ALPHA_PERP_A3 = 2.59;
    constexpr double ALPHA_ISO_A3  = 2.59;
    constexpr double W_POL_CH4 = 1.00;
}

// Unified rigid linear/point species descriptor (up to 3 sites: 0 at COM, 1 at +bond, 2 at -bond)
struct SpeciesDef {
    int n_sites;
    double bond_A;
    double q[3];
    double eps_K[3];
    double sigma_A[3];
    int lj_grid_idx[3]; // -1 for massless non-LJ site (e.g., N2 COM M), else 0:C_CO2, 1:O_CO2, 2:N_N2, 3:CH4
    double alpha_par;
    double alpha_perp;
    double alpha_iso;
    double w_pol[3];
    double Tc_K;
    double Pc_bar;
    double omega;
};

inline SpeciesDef make_species_co2() {
    SpeciesDef s{};
    s.n_sites = 3;
    s.bond_A = trappe_co2::BOND_CO_A;
    s.q[0] = trappe_co2::Q_C;   s.q[1] = trappe_co2::Q_O;   s.q[2] = trappe_co2::Q_O;
    s.eps_K[0] = trappe_co2::EPS_C_K; s.eps_K[1] = trappe_co2::EPS_O_K; s.eps_K[2] = trappe_co2::EPS_O_K;
    s.sigma_A[0] = trappe_co2::SIG_C_A; s.sigma_A[1] = trappe_co2::SIG_O_A; s.sigma_A[2] = trappe_co2::SIG_O_A;
    s.lj_grid_idx[0] = 0; s.lj_grid_idx[1] = 1; s.lj_grid_idx[2] = 1;
    s.alpha_par = trappe_co2::ALPHA_PAR_A3;
    s.alpha_perp = trappe_co2::ALPHA_PERP_A3;
    s.alpha_iso = trappe_co2::ALPHA_ISO_A3;
    s.w_pol[0] = trappe_co2::W_POL_C; s.w_pol[1] = trappe_co2::W_POL_O; s.w_pol[2] = trappe_co2::W_POL_O;
    s.Tc_K = 304.1282; s.Pc_bar = 73.773; s.omega = 0.22394;
    return s;
}

inline SpeciesDef make_species_co2_epm2() {
    SpeciesDef s = make_species_co2();
    s.bond_A = mercado2016::BOND_CO_EPM2;
    s.q[0] = mercado2016::Q_C_EPM2; s.q[1] = mercado2016::Q_O_EPM2; s.q[2] = mercado2016::Q_O_EPM2;
    s.eps_K[0] = mercado2016::EPS_C_EPM2; s.eps_K[1] = mercado2016::EPS_O_EPM2; s.eps_K[2] = mercado2016::EPS_O_EPM2;
    s.sigma_A[0] = mercado2016::SIG_C_EPM2; s.sigma_A[1] = mercado2016::SIG_O_EPM2; s.sigma_A[2] = mercado2016::SIG_O_EPM2;
    return s;
}

inline SpeciesDef make_species_n2() {
    SpeciesDef s{};
    s.n_sites = 3;
    s.bond_A = trappe_n2::BOND_NM_A;
    s.q[0] = trappe_n2::Q_M; s.q[1] = trappe_n2::Q_N; s.q[2] = trappe_n2::Q_N;
    s.eps_K[0] = 0.0; s.eps_K[1] = trappe_n2::EPS_N_K; s.eps_K[2] = trappe_n2::EPS_N_K;
    s.sigma_A[0] = 0.0; s.sigma_A[1] = trappe_n2::SIG_N_A; s.sigma_A[2] = trappe_n2::SIG_N_A;
    s.lj_grid_idx[0] = -1; s.lj_grid_idx[1] = 2; s.lj_grid_idx[2] = 2;
    s.alpha_par = trappe_n2::ALPHA_PAR_A3;
    s.alpha_perp = trappe_n2::ALPHA_PERP_A3;
    s.alpha_iso = trappe_n2::ALPHA_ISO_A3;
    s.w_pol[0] = 0.0; s.w_pol[1] = trappe_n2::W_POL_N; s.w_pol[2] = trappe_n2::W_POL_N;
    s.Tc_K = 126.192; s.Pc_bar = 33.958; s.omega = 0.0372;
    return s;
}

inline SpeciesDef make_species_ch4() {
    SpeciesDef s{};
    s.n_sites = 1;
    s.bond_A = 0.0;
    s.q[0] = trappe_ch4::Q_CH4; s.q[1] = 0.0; s.q[2] = 0.0;
    s.eps_K[0] = trappe_ch4::EPS_CH4_K; s.eps_K[1] = 0.0; s.eps_K[2] = 0.0;
    s.sigma_A[0] = trappe_ch4::SIG_CH4_A; s.sigma_A[1] = 0.0; s.sigma_A[2] = 0.0;
    s.lj_grid_idx[0] = 3; s.lj_grid_idx[1] = -1; s.lj_grid_idx[2] = -1;
    s.alpha_par = trappe_ch4::ALPHA_PAR_A3;
    s.alpha_perp = trappe_ch4::ALPHA_PERP_A3;
    s.alpha_iso = trappe_ch4::ALPHA_ISO_A3;
    s.w_pol[0] = trappe_ch4::W_POL_CH4; s.w_pol[1] = 0.0; s.w_pol[2] = 0.0;
    s.Tc_K = 190.564; s.Pc_bar = 45.992; s.omega = 0.01142;
    return s;
}

// Lorentz-Berthelot combining rules
inline void lorentz_berthelot(double eps1, double sig1, double eps2, double sig2,
                              double& eps_ij, double& sig_ij) {
    eps_ij = std::sqrt(eps1 * eps2);
    sig_ij = 0.5 * (sig1 + sig2);
}

struct FugacityResult {
    double P_bar;
    double Z;
    double phi;
    double f_bar;
    double f_Pa;
};

// Generic pure-component Peng-Robinson EOS: computes compressibility factor Z and fugacity f
inline FugacityResult compute_pure_fugacity_pr(
    double T_K, double P_bar, double Tc, double Pc, double omega) {
    if (Tc <= 0.0 || Pc <= 0.0 || P_bar <= 0.0) {
        FugacityResult res{P_bar, 1.0, 1.0, P_bar, P_bar * 1.0e5};
        return res;
    }
    constexpr double R_BAR_L = 0.08314462618; // bar * L / (mol * K)

    double Tr = T_K / Tc;
    double kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega * omega;
    double sqrt_alpha = 1.0 + kappa * (1.0 - std::sqrt(Tr));
    double alpha = sqrt_alpha * sqrt_alpha;

    double a = 0.4572355289213822 * R_BAR_L * R_BAR_L * Tc * Tc / Pc * alpha;
    double b = 0.07779607390388846 * R_BAR_L * Tc / Pc;

    double A = a * P_bar / (R_BAR_L * T_K * R_BAR_L * T_K);
    double B = b * P_bar / (R_BAR_L * T_K);

    // Z^3 + c2*Z^2 + c1*Z + c0 = 0
    double c2 = -(1.0 - B);
    double c1 = A - 3.0 * B * B - 2.0 * B;
    double c0 = -(A * B - B * B - B * B * B);

    double p = c1 - c2 * c2 / 3.0;
    double q = 2.0 * c2 * c2 * c2 / 27.0 - c2 * c1 / 3.0 + c0;
    double disc = 0.25 * q * q + p * p * p / 27.0;

    auto eval_phi = [&](double Z) -> double {
        double arg1 = Z - B;
        double term2 = (Z + (1.0 + std::sqrt(2.0)) * B) / (Z + (1.0 - std::sqrt(2.0)) * B);
        if (arg1 <= 0.0 || term2 <= 0.0) return 1e30;
        double ln_phi = (Z - 1.0) - std::log(arg1)
                        - (A / (2.0 * std::sqrt(2.0) * B)) * std::log(term2);
        return std::exp(ln_phi);
    };

    double best_Z = 1.0;
    double best_phi = 1.0;

    if (disc >= 0.0) {
        double sqrt_d = std::sqrt(disc);
        double u = std::cbrt(-0.5 * q + sqrt_d);
        double v = std::cbrt(-0.5 * q - sqrt_d);
        best_Z = u + v - c2 / 3.0;
        best_phi = eval_phi(best_Z);
    } else {
        constexpr double PI = 3.14159265358979323846;
        double r = std::sqrt(-p * p * p / 27.0);
        double theta = std::acos(-0.5 * q / r);
        double m = 2.0 * std::sqrt(-p / 3.0);
        best_phi = 1e30;
        for (int k = 0; k < 3; ++k) {
            double zk = m * std::cos((theta + 2.0 * PI * k) / 3.0) - c2 / 3.0;
            if (zk > B) {
                double phik = eval_phi(zk);
                if (phik < best_phi) {
                    best_phi = phik;
                    best_Z = zk;
                }
            }
        }
    }

    FugacityResult res;
    res.P_bar = P_bar;
    res.Z = best_Z;
    res.phi = best_phi;
    res.f_bar = best_phi * P_bar;
    res.f_Pa = res.f_bar * 1.0e5;
    return res;
}

inline FugacityResult compute_co2_fugacity_pr(double T_K, double P_bar) {
    return compute_pure_fugacity_pr(T_K, P_bar, 304.1282, 73.773, 0.22394);
}

struct BinaryFugacityResult {
    double P_bar;
    double Z;
    double phi0;
    double phi1;
    double f0_bar;
    double f1_bar;
    double f0_Pa;
    double f1_Pa;
};

// Binary-mixture Peng-Robinson EOS with van der Waals 1-fluid mixing & binary kij
inline BinaryFugacityResult compute_binary_fugacity_pr(
    double T_K, double P_bar,
    const SpeciesDef& sp0, const SpeciesDef& sp1,
    double y0, double y1, double kij) {

    constexpr double R_BAR_L = 0.08314462618;
    auto calc_ab = [&](const SpeciesDef& sp, double& a_i, double& b_i) {
        double Tr = T_K / sp.Tc_K;
        double kappa = 0.37464 + 1.54226 * sp.omega - 0.26992 * sp.omega * sp.omega;
        double sa = 1.0 + kappa * (1.0 - std::sqrt(Tr));
        a_i = 0.4572355289213822 * R_BAR_L * R_BAR_L * sp.Tc_K * sp.Tc_K / sp.Pc_bar * sa * sa;
        b_i = 0.07779607390388846 * R_BAR_L * sp.Tc_K / sp.Pc_bar;
    };

    double a0, b0, a1, b1;
    calc_ab(sp0, a0, b0);
    calc_ab(sp1, a1, b1);

    double a01 = std::sqrt(a0 * a1) * (1.0 - kij);
    double am = y0 * y0 * a0 + 2.0 * y0 * y1 * a01 + y1 * y1 * a1;
    double bm = y0 * b0 + y1 * b1;

    double A = am * P_bar / (R_BAR_L * T_K * R_BAR_L * T_K);
    double B = bm * P_bar / (R_BAR_L * T_K);

    double c2 = -(1.0 - B);
    double c1 = A - 3.0 * B * B - 2.0 * B;
    double c0 = -(A * B - B * B - B * B * B);

    double p = c1 - c2 * c2 / 3.0;
    double q = 2.0 * c2 * c2 * c2 / 27.0 - c2 * c1 / 3.0 + c0;
    double disc = 0.25 * q * q + p * p * p / 27.0;

    auto eval_ln_phis = [&](double Z, double& ln_p0, double& ln_p1) -> bool {
        double arg1 = Z - B;
        double s2 = std::sqrt(2.0);
        double term2 = (Z + (1.0 + s2) * B) / (Z + (1.0 - s2) * B);
        if (arg1 <= 0.0 || term2 <= 0.0) return false;
        double log_t2 = std::log(term2);
        double log_zmb = std::log(arg1);
        double sum_ya0 = y0 * a0 + y1 * a01;
        double sum_ya1 = y0 * a01 + y1 * a1;
        ln_p0 = (b0 / bm) * (Z - 1.0) - log_zmb
              - (A / (2.0 * s2 * B)) * (2.0 * sum_ya0 / am - b0 / bm) * log_t2;
        ln_p1 = (b1 / bm) * (Z - 1.0) - log_zmb
              - (A / (2.0 * s2 * B)) * (2.0 * sum_ya1 / am - b1 / bm) * log_t2;
        return true;
    };

    double best_Z = 1.0, best_phi0 = 1.0, best_phi1 = 1.0;
    if (disc >= 0.0) {
        double sqrt_d = std::sqrt(disc);
        best_Z = std::cbrt(-0.5 * q + sqrt_d) + std::cbrt(-0.5 * q - sqrt_d) - c2 / 3.0;
        double lp0 = 0.0, lp1 = 0.0;
        if (eval_ln_phis(best_Z, lp0, lp1)) {
            best_phi0 = std::exp(lp0);
            best_phi1 = std::exp(lp1);
        }
    } else {
        constexpr double PI = 3.14159265358979323846;
        double r = std::sqrt(-p * p * p / 27.0);
        double theta = std::acos(-0.5 * q / r);
        double m = 2.0 * std::sqrt(-p / 3.0);
        double best_g = 1.0e30;
        for (int k = 0; k < 3; ++k) {
            double zk = m * std::cos((theta + 2.0 * PI * k) / 3.0) - c2 / 3.0;
            double lp0 = 0.0, lp1 = 0.0;
            if (zk > B && eval_ln_phis(zk, lp0, lp1)) {
                double g = y0 * lp0 + y1 * lp1;
                if (g < best_g) {
                    best_g = g;
                    best_Z = zk;
                    best_phi0 = std::exp(lp0);
                    best_phi1 = std::exp(lp1);
                }
            }
        }
    }

    BinaryFugacityResult out{};
    out.P_bar = P_bar;
    out.Z = best_Z;
    out.phi0 = best_phi0;
    out.phi1 = best_phi1;
    out.f0_bar = best_phi0 * y0 * P_bar;
    out.f1_bar = best_phi1 * y1 * P_bar;
    out.f0_Pa = out.f0_bar * 1.0e5;
    out.f1_Pa = out.f1_bar * 1.0e5;
    return out;
}

} // namespace mofgcmc
