#pragma once

#include "cell.hpp"

namespace mofgcmc {

struct FrameworkAtom {
    float x, y, z;          // Cartesian coords in 1x1x1 unit cell (Angstrom)
    float sa, sb, sc;       // Fractional coords in 1x1x1 unit cell [0, 1)
    float eps_K;            // UFF epsilon (K)
    float sigma_A;          // UFF sigma (Angstrom)
    double q;               // Partial charge (e)
    int elem;               // ElementId
    int atom_type_id;       // 0=Mg1/Mof_Mg, 1=O1/Oa, 2=O2/Ob, 3=O3/Oc, 4=C1/Ca, 5=C2/Cb, 6=C3/Cc, 7=C4/Cd, 8=H1/H
};

// Precomputed periodic 3D grid over the 1x1x1 unit cell (fractional [0, 1)^3).
// Includes U_lj_C, U_lj_O, Phi_elec, plus E-field components (Ex, Ey, Ez) ready for Step 2 B1.
struct Grid3DView {
    int nx, ny, nz;
    const float* U_lj_C;    // Kelvin
    const float* U_lj_O;    // Kelvin
    const float* Phi_elec;  // K / e  (so E_elec = q * Phi_elec in Kelvin)
    const float* Ex;        // K / (e * Angstrom)
    const float* Ey;        // K / (e * Angstrom)
    const float* Ez;        // K / (e * Angstrom)
};

__host__ __device__ inline float trilinear_periodic(
    const float* __restrict__ data, int nx, int ny, int nz,
    double sa, double sb, double sc) {
    sa -= floor(sa);
    sb -= floor(sb);
    sc -= floor(sc);

    double fx = sa * (double)nx;
    double fy = sb * (double)ny;
    double fz = sc * (double)nz;

    int x0 = (int)floor(fx);
    int y0 = (int)floor(fy);
    int z0 = (int)floor(fz);

    double tx = fx - (double)x0;
    double ty = fy - (double)y0;
    double tz = fz - (double)z0;

    x0 = ((x0 % nx) + nx) % nx;
    y0 = ((y0 % ny) + ny) % ny;
    z0 = ((z0 % nz) + nz) % nz;

    int x1 = (x0 + 1 == nx) ? 0 : (x0 + 1);
    int y1 = (y0 + 1 == ny) ? 0 : (y0 + 1);
    int z1 = (z0 + 1 == nz) ? 0 : (z0 + 1);

    int stride_y = nx;
    int stride_z = nx * ny;

    float c000 = data[z0 * stride_z + y0 * stride_y + x0];
    float c100 = data[z0 * stride_z + y0 * stride_y + x1];
    float c010 = data[z0 * stride_z + y1 * stride_y + x0];
    float c110 = data[z0 * stride_z + y1 * stride_y + x1];
    float c001 = data[z1 * stride_z + y0 * stride_y + x0];
    float c101 = data[z1 * stride_z + y0 * stride_y + x1];
    float c011 = data[z1 * stride_z + y1 * stride_y + x0];
    float c111 = data[z1 * stride_z + y1 * stride_y + x1];

    double c00 = c000 * (1.0 - tx) + c100 * tx;
    double c10 = c010 * (1.0 - tx) + c110 * tx;
    double c01 = c001 * (1.0 - tx) + c101 * tx;
    double c11 = c011 * (1.0 - tx) + c111 * tx;

    double c0 = c00 * (1.0 - ty) + c10 * ty;
    double c1 = c01 * (1.0 - ty) + c11 * ty;

    return (float)(c0 * (1.0 - tz) + c1 * tz);
}

// 4-point O(h^4) periodic 3D Lagrange tricubic interpolation with Gibbs-ringing fallback near 1e6 K wall.
__host__ __device__ inline double tricubic_periodic(
    const float* __restrict__ data, int nx, int ny, int nz,
    double sa, double sb, double sc,
    bool is_vdw, float smooth_limit) {

    sa -= floor(sa);
    sb -= floor(sb);
    sc -= floor(sc);

    double fx = sa * (double)nx;
    double fy = sb * (double)ny;
    double fz = sc * (double)nz;

    int x0 = (int)floor(fx);
    int y0 = (int)floor(fy);
    int z0 = (int)floor(fz);

    double tx = fx - (double)x0;
    double ty = fy - (double)y0;
    double tz = fz - (double)z0;

    x0 = ((x0 % nx) + nx) % nx;
    y0 = ((y0 % ny) + ny) % ny;
    z0 = ((z0 % nz) + nz) % nz;

    int ix[4] = {(x0 - 1 + nx) % nx, x0, (x0 + 1) % nx, (x0 + 2) % nx};
    int iy[4] = {(y0 - 1 + ny) % ny, y0, (y0 + 1) % ny, (y0 + 2) % ny};
    int iz[4] = {(z0 - 1 + nz) % nz, z0, (z0 + 1) % nz, (z0 + 2) % nz};

    int stride_y = nx;
    int stride_z = nx * ny;

    // 8 inner corners for fast overlap guard and smooth-wall fallback
    float c000 = data[iz[1] * stride_z + iy[1] * stride_y + ix[1]];
    float c100 = data[iz[1] * stride_z + iy[1] * stride_y + ix[2]];
    float c010 = data[iz[1] * stride_z + iy[2] * stride_y + ix[1]];
    float c110 = data[iz[1] * stride_z + iy[2] * stride_y + ix[2]];
    float c001 = data[iz[2] * stride_z + iy[1] * stride_y + ix[1]];
    float c101 = data[iz[2] * stride_z + iy[1] * stride_y + ix[2]];
    float c011 = data[iz[2] * stride_z + iy[2] * stride_y + ix[1]];
    float c111 = data[iz[2] * stride_z + iy[2] * stride_y + ix[2]];

    if (is_vdw && (c000 >= 1.0e5f || c100 >= 1.0e5f || c010 >= 1.0e5f || c110 >= 1.0e5f ||
                   c001 >= 1.0e5f || c101 >= 1.0e5f || c011 >= 1.0e5f || c111 >= 1.0e5f)) {
        return 1.0e6;
    }

    auto trilin = [&]() -> double {
        double d00 = c000 * (1.0 - tx) + c100 * tx;
        double d10 = c010 * (1.0 - tx) + c110 * tx;
        double d01 = c001 * (1.0 - tx) + c101 * tx;
        double d11 = c011 * (1.0 - tx) + c111 * tx;
        double d0  = d00 * (1.0 - ty) + d10 * ty;
        double d1  = d01 * (1.0 - ty) + d11 * ty;
        return d0 * (1.0 - tz) + d1 * tz;
    };

    double wx[4] = {
        (-1.0 / 6.0) * tx * (tx - 1.0) * (tx - 2.0),
         0.5         * (tx + 1.0) * (tx - 1.0) * (tx - 2.0),
        -0.5         * (tx + 1.0) * tx * (tx - 2.0),
        ( 1.0 / 6.0) * (tx + 1.0) * tx * (tx - 1.0)
    };
    double wy[4] = {
        (-1.0 / 6.0) * ty * (ty - 1.0) * (ty - 2.0),
         0.5         * (ty + 1.0) * (ty - 1.0) * (ty - 2.0),
        -0.5         * (ty + 1.0) * ty * (ty - 2.0),
        ( 1.0 / 6.0) * (ty + 1.0) * ty * (ty - 1.0)
    };
    double wz[4] = {
        (-1.0 / 6.0) * tz * (tz - 1.0) * (tz - 2.0),
         0.5         * (tz + 1.0) * (tz - 1.0) * (tz - 2.0),
        -0.5         * (tz + 1.0) * tz * (tz - 2.0),
        ( 1.0 / 6.0) * (tz + 1.0) * tz * (tz - 1.0)
    };

    double sum = 0.0;
    for (int kz = 0; kz < 4; ++kz) {
        int zoff = iz[kz] * stride_z;
        double wz_v = wz[kz];
        for (int ky = 0; ky < 4; ++ky) {
            int yoff = zoff + iy[ky] * stride_y;
            double wyz = wz_v * wy[ky];
            double row = 0.0;
            for (int kx = 0; kx < 4; ++kx) {
                float v = data[yoff + ix[kx]];
                if (fabsf(v) >= smooth_limit) {
                    double tval = trilin();
                    return (is_vdw && tval > 1.0e5) ? 1.0e6 : tval;
                }
                row += wx[kx] * (double)v;
            }
            sum += wyz * row;
        }
    }
    if (is_vdw && sum > 1.0e5) return 1.0e6;
    return sum;
}

__host__ __device__ inline bool eval_co2_framework_grid_tricubic(
    const Grid3DView& g, const Cell& cell,
    double cx, double cy, double cz,
    double ux, double uy, double uz,
    double bond_co, double q_c, double q_o,
    double& u_lj_out, double& u_elec_out) {

    const double sx_arr[3] = {cx, cx + bond_co * ux, cx - bond_co * ux};
    const double sy_arr[3] = {cy, cy + bond_co * uy, cy - bond_co * uy};
    const double sz_arr[3] = {cz, cz + bond_co * uz, cz - bond_co * uz};

    double sa[3], sb[3], sc[3];
    double u_lj = 0.0;
    for (int i = 0; i < 3; ++i) {
        cell.cart_to_unit_frac(sx_arr[i], sy_arr[i], sz_arr[i], sa[i], sb[i], sc[i]);
        double ulj = (i == 0)
            ? tricubic_periodic(g.U_lj_C, g.nx, g.ny, g.nz, sa[i], sb[i], sc[i], true, 5.0e4f)
            : tricubic_periodic(g.U_lj_O, g.nx, g.ny, g.nz, sa[i], sb[i], sc[i], true, 5.0e4f);
        if (ulj > 1.0e5) {
            u_lj_out = 1.0e6;
            u_elec_out = 0.0;
            return true;
        }
        u_lj += ulj;
    }

    double u_el = 0.0;
    for (int i = 0; i < 3; ++i) {
        double phi = tricubic_periodic(g.Phi_elec, g.nx, g.ny, g.nz, sa[i], sb[i], sc[i], false, 5.0e5f);
        u_el += (i == 0 ? q_c : q_o) * phi;
    }
    u_lj_out = u_lj;
    u_elec_out = u_el;
    return false;
}

// Evaluate framework energy (K) of a single rigid CO2 molecule using trilinear grid lookup.
// Returns true if hard-core overlap occurs (U_lj > 1.0e5 K).
__host__ __device__ inline bool eval_co2_framework_grid(
    const Grid3DView& g, const Cell& cell,
    double cx, double cy, double cz,
    double ux, double uy, double uz,
    double bond_co, double q_c, double q_o,
    double& u_lj_out, double& u_elec_out) {

    const double sx_arr[3] = {cx, cx + bond_co * ux, cx - bond_co * ux};
    const double sy_arr[3] = {cy, cy + bond_co * uy, cy - bond_co * uy};
    const double sz_arr[3] = {cz, cz + bond_co * uz, cz - bond_co * uz};

    double u_lj = 0.0;
    double u_el = 0.0;
    for (int i = 0; i < 3; ++i) {
        double sa, sb, sc;
        cell.cart_to_unit_frac(sx_arr[i], sy_arr[i], sz_arr[i], sa, sb, sc);
        float ulj = (i == 0)
            ? trilinear_periodic(g.U_lj_C, g.nx, g.ny, g.nz, sa, sb, sc)
            : trilinear_periodic(g.U_lj_O, g.nx, g.ny, g.nz, sa, sb, sc);
        if (ulj > 1.0e5f) {
            u_lj_out = 1.0e6;
            u_elec_out = 0.0;
            return true;
        }
        float phi = trilinear_periodic(g.Phi_elec, g.nx, g.ny, g.nz, sa, sb, sc);
        u_lj += (double)ulj;
        u_el += (i == 0 ? q_c : q_o) * (double)phi;
    }
    u_lj_out = u_lj;
    u_elec_out = u_el;
    return false;
}

// Evaluate framework LJ, permanent electrostatic, and anisotropic polarization energy + induced dipole.
// pol_mode: 0 = POL_NONE, 1 = POL_COM (C site weight 1.0), 2 = POL_3SITE (weights 0.40, 0.30, 0.30).
__host__ __device__ inline bool eval_co2_framework_pol_grid(
    const Grid3DView& g, const Cell& cell,
    double cx, double cy, double cz,
    double ux, double uy, double uz,
    double bond_co, double q_c, double q_o,
    int pol_mode, double alpha_par, double alpha_perp,
    double w_pol_c, double w_pol_o, double ke_k, double pol_scale,
    double& u_lj_out, double& u_elec_out, double& u_pol_out,
    double& mux_out, double& muy_out, double& muz_out) {

    const double sx_arr[3] = {cx, cx + bond_co * ux, cx - bond_co * ux};
    const double sy_arr[3] = {cy, cy + bond_co * uy, cy - bond_co * uy};
    const double sz_arr[3] = {cz, cz + bond_co * uz, cz - bond_co * uz};

    double u_lj = 0.0;
    double u_el = 0.0;
    double u_pol = 0.0;
    double mux = 0.0, muy = 0.0, muz = 0.0;

    for (int i = 0; i < 3; ++i) {
        double sa, sb, sc;
        cell.cart_to_unit_frac(sx_arr[i], sy_arr[i], sz_arr[i], sa, sb, sc);
        float ulj = (i == 0)
            ? trilinear_periodic(g.U_lj_C, g.nx, g.ny, g.nz, sa, sb, sc)
            : trilinear_periodic(g.U_lj_O, g.nx, g.ny, g.nz, sa, sb, sc);
        if (ulj > 1.0e5f) {
            u_lj_out = 1.0e6;
            u_elec_out = 0.0;
            u_pol_out = 0.0;
            mux_out = muy_out = muz_out = 0.0;
            return true;
        }
        float phi = trilinear_periodic(g.Phi_elec, g.nx, g.ny, g.nz, sa, sb, sc);
        u_lj += (double)ulj;
        u_el += (i == 0 ? q_c : q_o) * (double)phi;

        double wa = 0.0;
        if (pol_mode == 1 && i == 0) wa = 1.0;
        else if (pol_mode == 2) wa = (i == 0) ? w_pol_c : w_pol_o;

        if (wa > 0.0) {
            double ex = (double)trilinear_periodic(g.Ex, g.nx, g.ny, g.nz, sa, sb, sc) * pol_scale;
            double ey = (double)trilinear_periodic(g.Ey, g.nx, g.ny, g.nz, sa, sb, sc) * pol_scale;
            double ez = (double)trilinear_periodic(g.Ez, g.nx, g.ny, g.nz, sa, sb, sc) * pol_scale;

            double e2 = ex * ex + ey * ey + ez * ez;
            double e_par = ex * ux + ey * uy + ez * uz;
            double e_perp2 = e2 - e_par * e_par;
            if (e_perp2 < 0.0) e_perp2 = 0.0;

            double e_perpx = ex - e_par * ux;
            double e_perpy = ey - e_par * uy;
            double e_perpz = ez - e_par * uz;

            double inv_ke = 1.0 / ke_k;
            u_pol += -0.5 * inv_ke * wa * (alpha_par * e_par * e_par + alpha_perp * e_perp2);
            mux += (wa * inv_ke) * (alpha_par * e_par * ux + alpha_perp * e_perpx);
            muy += (wa * inv_ke) * (alpha_par * e_par * uy + alpha_perp * e_perpy);
            muz += (wa * inv_ke) * (alpha_par * e_par * uz + alpha_perp * e_perpz);
        }
    }

    u_lj_out = u_lj;
    u_elec_out = u_el;
    u_pol_out = u_pol;
    mux_out = mux;
    muy_out = muy;
    muz_out = muz;
    return false;
}

} // namespace mofgcmc
