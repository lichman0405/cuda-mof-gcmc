#pragma once

#include <cmath>
#include <cuda_runtime.h>

#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

namespace mofgcmc {

struct Vec3 {
    double x, y, z;
};

struct Cell {
    // Unit-cell dimensions (Angstrom, degrees)
    double a, b, c;
    double alpha_deg, beta_deg, gamma_deg;
    int rep_a, rep_b, rep_c;

    // Supercell edge matrix H (columns are supercell lattice vectors A, B, C in Angstrom)
    // and inverse H_inv such that r = H * s_super, s_super = H_inv * r
    double H[3][3];
    double H_inv[3][3];

    // Unit-cell (1x1x1) edge matrix and inverse for folding into 1x1x1 periodic grids
    double H_unit[3][3];
    double H_unit_inv[3][3];

    double volume_unit;
    double volume_super;
    double perp_width_unit[3];
    double min_perp_width_super;
    int mic_mode; // 0 = orthorhombic, 1 = 2D oblique (alpha=beta=90), 2 = general 3D triclinic

    static void invert_3x3(const double M[3][3], double inv[3][3]) {
        double c00 = M[1][1] * M[2][2] - M[1][2] * M[2][1];
        double c01 = -(M[1][0] * M[2][2] - M[1][2] * M[2][0]);
        double c02 = M[1][0] * M[2][1] - M[1][1] * M[2][0];

        double c10 = -(M[0][1] * M[2][2] - M[0][2] * M[2][1]);
        double c11 = M[0][0] * M[2][2] - M[0][2] * M[2][0];
        double c12 = -(M[0][0] * M[2][1] - M[0][1] * M[2][0]);

        double c20 = M[0][1] * M[1][2] - M[0][2] * M[1][1];
        double c21 = -(M[0][0] * M[1][2] - M[0][2] * M[1][0]);
        double c22 = M[0][0] * M[1][1] - M[0][1] * M[1][0];

        double det = M[0][0] * c00 + M[0][1] * c01 + M[0][2] * c02;
        double inv_det = 1.0 / det;

        // Transpose of cofactor matrix * (1 / det)
        inv[0][0] = c00 * inv_det;
        inv[0][1] = c10 * inv_det;
        inv[0][2] = c20 * inv_det;
        inv[1][0] = c01 * inv_det;
        inv[1][1] = c11 * inv_det;
        inv[1][2] = c21 * inv_det;
        inv[2][0] = c02 * inv_det;
        inv[2][1] = c12 * inv_det;
        inv[2][2] = c22 * inv_det;
    }

    static double cross_norm(const double v1[3], const double v2[3]) {
        double cx = v1[1] * v2[2] - v1[2] * v2[1];
        double cy = v1[2] * v2[0] - v1[0] * v2[2];
        double cz = v1[0] * v2[1] - v1[1] * v2[0];
        return std::sqrt(cx * cx + cy * cy + cz * cz);
    }

    void init_from_H_unit(const double H_u[3][3], int ra = 1, int rb = 1, int rc = 4) {
        rep_a = ra;
        rep_b = rb;
        rep_c = rc;

        for (int r = 0; r < 3; ++r) {
            H_unit[r][0] = H_u[r][0];
            H_unit[r][1] = H_u[r][1];
            H_unit[r][2] = H_u[r][2];
            H[r][0] = H_u[r][0] * (double)rep_a;
            H[r][1] = H_u[r][1] * (double)rep_b;
            H[r][2] = H_u[r][2] * (double)rep_c;
        }

        invert_3x3(H_unit, H_unit_inv);
        invert_3x3(H, H_inv);

        double uA[3] = {H_unit[0][0], H_unit[1][0], H_unit[2][0]};
        double uB[3] = {H_unit[0][1], H_unit[1][1], H_unit[2][1]};
        double uC[3] = {H_unit[0][2], H_unit[1][2], H_unit[2][2]};
        a = std::sqrt(uA[0] * uA[0] + uA[1] * uA[1] + uA[2] * uA[2]);
        b = std::sqrt(uB[0] * uB[0] + uB[1] * uB[1] + uB[2] * uB[2]);
        c = std::sqrt(uC[0] * uC[0] + uC[1] * uC[1] + uC[2] * uC[2]);

        constexpr double RAD2DEG = 180.0 / 3.14159265358979323846;
        auto clamp_cos = [](double v) { return v < -1.0 ? -1.0 : (v > 1.0 ? 1.0 : v); };
        double dot_bc = uB[0] * uC[0] + uB[1] * uC[1] + uB[2] * uC[2];
        double dot_ac = uA[0] * uC[0] + uA[1] * uC[1] + uA[2] * uC[2];
        double dot_ab = uA[0] * uB[0] + uA[1] * uB[1] + uA[2] * uB[2];
        alpha_deg = (b > 0.0 && c > 0.0) ? std::acos(clamp_cos(dot_bc / (b * c))) * RAD2DEG : 90.0;
        beta_deg  = (a > 0.0 && c > 0.0) ? std::acos(clamp_cos(dot_ac / (a * c))) * RAD2DEG : 90.0;
        gamma_deg = (a > 0.0 && b > 0.0) ? std::acos(clamp_cos(dot_ab / (a * b))) * RAD2DEG : 90.0;

        volume_unit = std::fabs(
            H_unit[0][0] * (H_unit[1][1] * H_unit[2][2] - H_unit[1][2] * H_unit[2][1]) -
            H_unit[0][1] * (H_unit[1][0] * H_unit[2][2] - H_unit[1][2] * H_unit[2][0]) +
            H_unit[0][2] * (H_unit[1][0] * H_unit[2][1] - H_unit[1][1] * H_unit[2][0]));
        volume_super = volume_unit * (double)(rep_a * rep_b * rep_c);

        perp_width_unit[0] = volume_unit / cross_norm(uB, uC);
        perp_width_unit[1] = volume_unit / cross_norm(uC, uA);
        perp_width_unit[2] = volume_unit / cross_norm(uA, uB);

        double A[3] = {H[0][0], H[1][0], H[2][0]};
        double B[3] = {H[0][1], H[1][1], H[2][1]};
        double C[3] = {H[0][2], H[1][2], H[2][2]};
        double wa = volume_super / cross_norm(B, C);
        double wb = volume_super / cross_norm(C, A);
        double wc = volume_super / cross_norm(A, B);
        min_perp_width_super = wa;
        if (wb < min_perp_width_super) min_perp_width_super = wb;
        if (wc < min_perp_width_super) min_perp_width_super = wc;

        bool ortho_ab_c = (std::fabs(alpha_deg - 90.0) < 1.0e-4) && (std::fabs(beta_deg - 90.0) < 1.0e-4);
        bool ortho_ab   = (std::fabs(gamma_deg - 90.0) < 1.0e-4);
        if (ortho_ab_c && ortho_ab) mic_mode = 0;
        else if (ortho_ab_c) mic_mode = 1;
        else mic_mode = 2;
    }

    void init(double a_, double b_, double c_,
              double alpha_, double beta_, double gamma_,
              int ra = 1, int rb = 1, int rc = 4) {
        constexpr double PI = 3.14159265358979323846;
        double ar = alpha_ * PI / 180.0;
        double br = beta_ * PI / 180.0;
        double gr = gamma_ * PI / 180.0;

        double cos_a = std::cos(ar);
        double cos_b = std::cos(br);
        double cos_g = std::cos(gr);
        double sin_g = std::sin(gr);

        double ax = a_, ay = 0.0, az = 0.0;
        double bx = b_ * cos_g, by = b_ * sin_g, bz = 0.0;
        double cx = c_ * cos_b;
        double cy = c_ * (cos_a - cos_b * cos_g) / sin_g;
        double cz_sq = c_ * c_ - cx * cx - cy * cy;
        double cz = std::sqrt(cz_sq > 0.0 ? cz_sq : 0.0);

        double H_u[3][3] = {
            {ax, bx, cx},
            {ay, by, cy},
            {az, bz, cz}
        };
        init_from_H_unit(H_u, ra, rb, rc);
        a = a_; b = b_; c = c_;
        alpha_deg = alpha_; beta_deg = beta_; gamma_deg = gamma_;
    }

    __host__ __device__ inline void unit_frac_to_cart(double sa, double sb, double sc,
                                                      double& x, double& y, double& z) const {
        x = H_unit[0][0] * sa + H_unit[0][1] * sb + H_unit[0][2] * sc;
        y = H_unit[1][0] * sa + H_unit[1][1] * sb + H_unit[1][2] * sc;
        z = H_unit[2][0] * sa + H_unit[2][1] * sb + H_unit[2][2] * sc;
    }

    __host__ __device__ inline void super_frac_to_cart(double sa, double sb, double sc,
                                                       double& x, double& y, double& z) const {
        x = H[0][0] * sa + H[0][1] * sb + H[0][2] * sc;
        y = H[1][0] * sa + H[1][1] * sb + H[1][2] * sc;
        z = H[2][0] * sa + H[2][1] * sb + H[2][2] * sc;
    }

    __host__ __device__ inline void cart_to_unit_frac(double x, double y, double z,
                                                      double& sa, double& sb, double& sc) const {
        sa = H_unit_inv[0][0] * x + H_unit_inv[0][1] * y + H_unit_inv[0][2] * z;
        sb = H_unit_inv[1][0] * x + H_unit_inv[1][1] * y + H_unit_inv[1][2] * z;
        sc = H_unit_inv[2][0] * x + H_unit_inv[2][1] * y + H_unit_inv[2][2] * z;
    }

    __host__ __device__ inline void cart_to_super_frac(double x, double y, double z,
                                                       double& sa, double& sb, double& sc) const {
        sa = H_inv[0][0] * x + H_inv[0][1] * y + H_inv[0][2] * z;
        sb = H_inv[1][0] * x + H_inv[1][1] * y + H_inv[1][2] * z;
        sc = H_inv[2][0] * x + H_inv[2][1] * y + H_inv[2][2] * z;
    }

    __host__ __device__ inline void wrap_super_cart(double& x, double& y, double& z) const {
        double sa, sb, sc;
        cart_to_super_frac(x, y, z, sa, sb, sc);
        sa -= floor(sa);
        sb -= floor(sb);
        sc -= floor(sc);
        super_frac_to_cart(sa, sb, sc, x, y, z);
    }

    // Exact Minimum Image Convention in supercell coordinates:
    // mic_mode == 0 (orthorhombic): direct wrapped vector
    // mic_mode == 1 (2D oblique/hexagonal): 9 candidate in-plane images
    // mic_mode == 2 (general 3D triclinic): 27 candidate images
    __host__ __device__ inline void mic_delta_super(
        double dx_raw, double dy_raw, double dz_raw,
        double& best_dx, double& best_dy, double& best_dz, double& best_r2) const {
        double sa, sb, sc;
        cart_to_super_frac(dx_raw, dy_raw, dz_raw, sa, sb, sc);
        sa -= floor(sa + 0.5);
        sb -= floor(sb + 0.5);
        sc -= floor(sc + 0.5);

        if (mic_mode == 0) {
            super_frac_to_cart(sa, sb, sc, best_dx, best_dy, best_dz);
            best_r2 = best_dx * best_dx + best_dy * best_dy + best_dz * best_dz;
            return;
        }

        best_r2 = 1.0e30;
        best_dx = best_dy = best_dz = 0.0;
        int nc_max = (mic_mode == 2) ? 1 : 0;
        for (int nc = -nc_max; nc <= nc_max; ++nc) {
            double tc = sc + (double)nc;
            for (int nb = -1; nb <= 1; ++nb) {
                double tb = sb + (double)nb;
                for (int na = -1; na <= 1; ++na) {
                    double ta = sa + (double)na;
                    double dx, dy, dz;
                    super_frac_to_cart(ta, tb, tc, dx, dy, dz);
                    double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 < best_r2) {
                        best_r2 = r2;
                        best_dx = dx;
                        best_dy = dy;
                        best_dz = dz;
                    }
                }
            }
        }
    }
};

} // namespace mofgcmc
