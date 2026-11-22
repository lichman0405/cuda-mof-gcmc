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

} // namespace mofgcmc
