#pragma once

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
    ELEM_H  = 3
};

struct UffParam {
    double eps_K;
    double sigma_A;
};

// UFF parameters (Rappe et al. JACS 1992): eps in Kelvin, sigma in Angstrom
inline UffParam get_uff_param(int elem) {
    switch (elem) {
        case ELEM_MG: return {55.8574, 2.6914};
        case ELEM_O:  return {30.1932, 3.1181};
        case ELEM_C:  return {52.8381, 3.4309};
        case ELEM_H:  return {22.1417, 2.5711};
        default:      return {0.0, 1.0};
    }
}

inline int element_from_symbol(const std::string& sym) {
    if (sym == "Mg") return ELEM_MG;
    if (sym == "O")  return ELEM_O;
    if (sym == "C")  return ELEM_C;
    if (sym == "H")  return ELEM_H;
    return -1;
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

// Peng-Robinson EOS for CO2: computes compressibility factor Z and fugacity f
inline FugacityResult compute_co2_fugacity_pr(double T_K, double P_bar) {
    constexpr double R_BAR_L = 0.08314462618; // bar * L / (mol * K)
    constexpr double Tc = 304.1282;
    constexpr double Pc = 73.773;
    constexpr double omega = 0.22394;

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

} // namespace mofgcmc
