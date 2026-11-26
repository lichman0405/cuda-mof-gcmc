#pragma once

#include "cell.hpp"
#include <cmath>

namespace mofgcmc {

struct HardCoreConfig {
    int enable_continuous_hc = 0;  // 1 for auto_mlip / unified hard-core; 0 for legacy ff_mode=0/1 grids
    double r_fw_h = 1.20;          // Guest Z == 1 vs framework atom cutoff (Angstrom)
    double r_fw_heavy = 1.48;      // Guest Z > 1 vs framework atom cutoff (Angstrom)
    double r_gg = 0.50;            // Distinct guest real-atom (Z > 0) pair cutoff (Angstrom)
    double r_fw_h2 = 1.44;
    double r_fw_heavy2 = 2.1904;
    double r_gg2 = 0.25;

    __host__ __device__ inline void set_radii(double h_A, double heavy_A, double gg_A) {
        r_fw_h = h_A;
        r_fw_heavy = heavy_A;
        r_gg = gg_A;
        r_fw_h2 = h_A * h_A;
        r_fw_heavy2 = heavy_A * heavy_A;
        r_gg2 = gg_A * gg_A;
    }
};

__host__ __device__ inline void hc_quat_rotate(
    double qw, double qx, double qy, double qz,
    double vx, double vy, double vz,
    double& ox, double& oy, double& oz) {
    double tx = 2.0 * (qy * vz - qz * vy);
    double ty = 2.0 * (qz * vx - qx * vz);
    double tz = 2.0 * (qx * vy - qy * vx);
    ox = vx + qw * tx + (qy * tz - qz * ty);
    oy = vy + qw * ty + (qz * tx - qx * tz);
    oz = vz + qw * tz + (qx * ty - qy * tx);
}

// Compute exact periodic minimum squared distance between two unit-cell fractional points
// up to cutoff r_cut (strictly exact for ANY orthorhombic, hexagonal, or triclinic cell).
// Let n_a be the unit normal to face (B_u, C_u). Then |r_vec| >= |da + na| * perp_width_unit[0],
// so any image with |na| >= 1 + floor(0.5 + r_cut_eff / perp_width_unit[0]) has |r_vec| > r_cut.
__host__ __device__ inline double mic_min_r2_unit_frac(
    const Cell& cell,
    double dsa_raw, double dsb_raw, double dsc_raw,
    double r_cut) {

    double dsa = dsa_raw - floor(dsa_raw + 0.5);
    double dsb = dsb_raw - floor(dsb_raw + 0.5);
    double dsc = dsc_raw - floor(dsc_raw + 0.5);

    double r_eff = r_cut * 1.01 + 1.0e-9;
    double p0 = (cell.perp_width_unit[0] > 1.0e-6) ? cell.perp_width_unit[0] : 1.0e-6;
    double p1 = (cell.perp_width_unit[1] > 1.0e-6) ? cell.perp_width_unit[1] : 1.0e-6;
    double p2 = (cell.perp_width_unit[2] > 1.0e-6) ? cell.perp_width_unit[2] : 1.0e-6;

    int na_max = (int)floor(0.5 + r_eff / p0);
    int nb_max = (int)floor(0.5 + r_eff / p1);
    int nc_max = (int)floor(0.5 + r_eff / p2);

    if (na_max == 0 && nb_max == 0 && nc_max == 0) {
        double dx, dy, dz;
        cell.unit_frac_to_cart(dsa, dsb, dsc, dx, dy, dz);
        return dx * dx + dy * dy + dz * dz;
    }

    double best_r2 = 1.0e30;
    for (int nc = -nc_max; nc <= nc_max; ++nc) {
        double tc = dsc + (double)nc;
        for (int nb = -nb_max; nb <= nb_max; ++nb) {
            double tb = dsb + (double)nb;
            for (int na = -na_max; na <= na_max; ++na) {
                double ta = dsa + (double)na;
                double dx, dy, dz;
                cell.unit_frac_to_cart(ta, tb, tc, dx, dy, dz);
                double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < best_r2) best_r2 = r2;
            }
        }
    }
    return best_r2;
}

// Compute exact supercell periodic minimum squared distance on Cartesian difference delta
// up to cutoff r_cut (strictly exact for ANY orthorhombic, hexagonal, or triclinic supercell).
__host__ __device__ inline double mic_min_r2_super_cart(
    const Cell& cell,
    double dx_raw, double dy_raw, double dz_raw,
    double r_cut) {

    double sa, sb, sc;
    cell.cart_to_super_frac(dx_raw, dy_raw, dz_raw, sa, sb, sc);
    sa -= floor(sa + 0.5);
    sb -= floor(sb + 0.5);
    sc -= floor(sc + 0.5);

    double r_eff = r_cut * 1.01 + 1.0e-9;
    double p0 = cell.perp_width_unit[0] * (double)(cell.rep_a > 0 ? cell.rep_a : 1);
    double p1 = cell.perp_width_unit[1] * (double)(cell.rep_b > 0 ? cell.rep_b : 1);
    double p2 = cell.perp_width_unit[2] * (double)(cell.rep_c > 0 ? cell.rep_c : 1);
    if (p0 < 1.0e-6) p0 = 1.0e-6;
    if (p1 < 1.0e-6) p1 = 1.0e-6;
    if (p2 < 1.0e-6) p2 = 1.0e-6;

    int na_max = (int)floor(0.5 + r_eff / p0);
    int nb_max = (int)floor(0.5 + r_eff / p1);
    int nc_max = (int)floor(0.5 + r_eff / p2);

    if (na_max == 0 && nb_max == 0 && nc_max == 0) {
        double dx, dy, dz;
        cell.super_frac_to_cart(sa, sb, sc, dx, dy, dz);
        return dx * dx + dy * dy + dz * dz;
    }

    double best_r2 = 1.0e30;
    for (int nc = -nc_max; nc <= nc_max; ++nc) {
        double tc = sc + (double)nc;
        for (int nb = -nb_max; nb <= nb_max; ++nb) {
            double tb = sb + (double)nb;
            for (int na = -na_max; na <= na_max; ++na) {
                double ta = sa + (double)na;
                double dx, dy, dz;
                cell.super_frac_to_cart(ta, tb, tc, dx, dy, dz);
                double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < best_r2) best_r2 = r2;
            }
        }
    }
    return best_r2;
}

// Check single molecule against periodic unit-cell framework atoms on continuous coordinates.
// Returns true iff any real guest atom (site_Z > 0) has r^2 < r_cut^2.
__host__ __device__ inline bool check_mol_fw_hard_core(
    const Cell& cell,
    const double* __restrict__ fw_unit_frac,
    int n_fw_unit,
    int n_sites,
    const double* __restrict__ lx,
    const double* __restrict__ ly,
    const double* __restrict__ lz,
    const int* __restrict__ site_Z,
    int is_isotropic,
    double cx, double cy, double cz,
    double qw, double qx, double qy, double qz,
    const HardCoreConfig& hc) {

    if (n_fw_unit <= 0 || !fw_unit_frac) return false;
    for (int a = 0; a < n_sites; ++a) {
        int z = site_Z ? site_Z[a] : 6;
        if (z <= 0) continue;  // Virtual sites excluded from hard-core checks
        double r_cut = (z == 1) ? hc.r_fw_h : hc.r_fw_heavy;
        double r_cut2 = (z == 1) ? hc.r_fw_h2 : hc.r_fw_heavy2;
        if (r_cut2 <= 0.0) continue;

        double ox = lx[a], oy = ly[a], oz = lz[a];
        if (!is_isotropic) {
            hc_quat_rotate(qw, qx, qy, qz, lx[a], ly[a], lz[a], ox, oy, oz);
        }
        double sa, sb, sc;
        cell.cart_to_unit_frac(cx + ox, cy + oy, cz + oz, sa, sb, sc);

        for (int j = 0; j < n_fw_unit; ++j) {
            double r2 = mic_min_r2_unit_frac(
                cell,
                sa - fw_unit_frac[3 * j + 0],
                sb - fw_unit_frac[3 * j + 1],
                sc - fw_unit_frac[3 * j + 2],
                r_cut);
            if (r2 < r_cut2) return true;
        }
    }
    return false;
}

// Check a pair of distinct guest molecules against guest-guest real-atom hard-core cutoff.
// Uses exact site-to-site supercell periodic MIC squared distance (< hc.r_gg2).
__host__ __device__ inline bool check_pair_gg_hard_core(
    const Cell& cell,
    int n_sites,
    const double* __restrict__ a_off_x,
    const double* __restrict__ a_off_y,
    const double* __restrict__ a_off_z,
    const double* __restrict__ b_off_x,
    const double* __restrict__ b_off_y,
    const double* __restrict__ b_off_z,
    const int* __restrict__ site_Z,
    double c1x, double c1y, double c1z,
    double c2x, double c2y, double c2z,
    const HardCoreConfig& hc) {

    if (hc.r_gg2 <= 0.0) return false;
    for (int a = 0; a < n_sites; ++a) {
        if (site_Z && site_Z[a] <= 0) continue;
        double s1x = c1x + a_off_x[a];
        double s1y = c1y + a_off_y[a];
        double s1z = c1z + a_off_z[a];
        for (int b = 0; b < n_sites; ++b) {
            if (site_Z && site_Z[b] <= 0) continue;
            double s2x = c2x + b_off_x[b];
            double s2y = c2y + b_off_y[b];
            double s2z = c2z + b_off_z[b];
            double r2 = mic_min_r2_super_cart(cell, s2x - s1x, s2y - s1y, s2z - s1z, hc.r_gg);
            if (r2 < hc.r_gg2) return true;
        }
    }
    return false;
}

} // namespace mofgcmc
