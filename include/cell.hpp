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
    double min_perp_width_super;

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

    void init(double a_, double b_, double c_,
              double alpha_, double beta_, double gamma_,
              int ra = 1, int rb = 1, int rc = 4) {
        a = a_;
        b = b_;
        c = c_;
        alpha_deg = alpha_;
        beta_deg = beta_;
        gamma_deg = gamma_;
        rep_a = ra;
        rep_b = rb;
        rep_c = rc;

        constexpr double PI = 3.14159265358979323846;
        double ar = alpha_deg * PI / 180.0;
        double br = beta_deg * PI / 180.0;
        double gr = gamma_deg * PI / 180.0;

        double cos_a = std::cos(ar);
        double cos_b = std::cos(br);
        double cos_g = std::cos(gr);
        double sin_g = std::sin(gr);

        double ax = a, ay = 0.0, az = 0.0;
        double bx = b * cos_g, by = b * sin_g, bz = 0.0;
        double cx = c * cos_b;
        double cy = c * (cos_a - cos_b * cos_g) / sin_g;
        double cz_sq = c * c - cx * cx - cy * cy;
        double cz = std::sqrt(cz_sq > 0.0 ? cz_sq : 0.0);

        H_unit[0][0] = ax; H_unit[0][1] = bx; H_unit[0][2] = cx;
        H_unit[1][0] = ay; H_unit[1][1] = by; H_unit[1][2] = cy;
        H_unit[2][0] = az; H_unit[2][1] = bz; H_unit[2][2] = cz;

        H[0][0] = ax * rep_a; H[0][1] = bx * rep_b; H[0][2] = cx * rep_c;
        H[1][0] = ay * rep_a; H[1][1] = by * rep_b; H[1][2] = cy * rep_c;
        H[2][0] = az * rep_a; H[2][1] = bz * rep_b; H[2][2] = cz * rep_c;

        invert_3x3(H_unit, H_unit_inv);
        invert_3x3(H, H_inv);

        volume_unit = std::fabs(
            H_unit[0][0] * (H_unit[1][1] * H_unit[2][2] - H_unit[1][2] * H_unit[2][1]) -
            H_unit[0][1] * (H_unit[1][0] * H_unit[2][2] - H_unit[1][2] * H_unit[2][0]) +
            H_unit[0][2] * (H_unit[1][0] * H_unit[2][1] - H_unit[1][1] * H_unit[2][0]));
        volume_super = volume_unit * (double)(rep_a * rep_b * rep_c);

        double A[3] = {H[0][0], H[1][0], H[2][0]};
        double B[3] = {H[0][1], H[1][1], H[2][1]};
        double C[3] = {H[0][2], H[1][2], H[2][2]};
        double wa = volume_super / cross_norm(B, C);
        double wb = volume_super / cross_norm(C, A);
        double wc = volume_super / cross_norm(A, B);
        min_perp_width_super = wa;
        if (wb < min_perp_width_super) min_perp_width_super = wb;
        if (wc < min_perp_width_super) min_perp_width_super = wc;
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

    // Exact hexagonal Minimum Image Convention in supercell coordinates:
    // wraps ds into [-0.5, 0.5) and tests the 5 2D hexagonal image candidates
    // (0,0), (+1,0), (-1,0), (0,+1), (0,-1) in (a,b) to find the exact minimum r^2.
    __host__ __device__ inline void mic_delta_super(
        double dx_raw, double dy_raw, double dz_raw,
        double& best_dx, double& best_dy, double& best_dz, double& best_r2) const {
        double sa, sb, sc;
        cart_to_super_frac(dx_raw, dy_raw, dz_raw, sa, sb, sc);
        sa -= floor(sa + 0.5);
        sb -= floor(sb + 0.5);
        sc -= floor(sc + 0.5);

        const int na[5] = {0, 1, -1, 0, 0};
        const int nb[5] = {0, 0, 0, 1, -1};

        best_r2 = 1.0e30;
        best_dx = best_dy = best_dz = 0.0;
        for (int k = 0; k < 5; ++k) {
            double ta = sa + na[k];
            double tb = sb + nb[k];
            double dx, dy, dz;
            super_frac_to_cart(ta, tb, sc, dx, dy, dz);
            double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < best_r2) {
                best_r2 = r2;
                best_dx = dx;
                best_dy = dy;
                best_dz = dz;
            }
        }
    }
};

} // namespace mofgcmc
