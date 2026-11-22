"""Independent physics checks & 300 DPI plotting for Step 1 (Mg-MOF-74 + CO2 @ 298 K)."""

import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
ASSETS = ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def make_hex_matrix(a=26.136, c=6.942, gamma_deg=120.0, rep_c=4):
    gr = math.radians(gamma_deg)
    H = np.array([
        [a, a * math.cos(gr), 0.0],
        [0.0, a * math.sin(gr), 0.0],
        [0.0, 0.0, c * rep_c],
    ])
    return H, np.linalg.inv(H)


def verify_hex_mic_python(n_tests=10000):
    rng = np.random.default_rng(2026)
    H, H_inv = make_hex_matrix()
    s1 = rng.uniform(-2.0, 2.0, size=(n_tests, 3))
    s2 = rng.uniform(-2.0, 2.0, size=(n_tests, 3))
    ds = (s2 - s1)
    ds -= np.round(ds)

    cands5 = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=float)
    shifts3 = np.array([[na, nb, nc] for na in (-1, 0, 1) for nb in (-1, 0, 1) for nc in (-1, 0, 1)], dtype=float)

    # 5-candidate distance^2
    cand_ds = ds[:, None, :] + cands5[None, :, :]
    cand_dr = cand_ds @ H.T
    r2_5 = np.min(np.sum(cand_dr ** 2, axis=-1), axis=1)

    # Brute-force 3x3x3 distance^2
    bf_ds = ds[:, None, :] + shifts3[None, :, :]
    bf_dr = bf_ds @ H.T
    r2_27 = np.min(np.sum(bf_dr ** 2, axis=-1), axis=1)

    mismatches = int(np.sum(np.abs(r2_5 - r2_27) > 1e-8))
    print(f"[PY CHECK] Hexagonal MIC 5-cand vs 3x3x3 brute-force: {mismatches} / {n_tests} mismatches")
    assert mismatches == 0


def peng_robinson_co2_phi(T_K, P_bar):
    R = 0.08314462618
    Tc, Pc, omega = 304.1282, 73.773, 0.22394
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega ** 2
    alpha = (1.0 + kappa * (1.0 - math.sqrt(T_K / Tc))) ** 2
    a = 0.4572355289213822 * (R * Tc) ** 2 / Pc * alpha
    b = 0.07779607390388846 * R * Tc / Pc
    A = a * P_bar / (R * T_K) ** 2
    B = b * P_bar / (R * T_K)
    roots = np.roots([1.0, -(1.0 - B), A - 3.0 * B ** 2 - 2.0 * B, -(A * B - B ** 2 - B ** 3)])
    real_roots = [float(np.real(r)) for r in roots if abs(np.imag(r)) < 1e-9 and np.real(r) > B]
    best_phi, best_Z = 1e30, 1.0
    s2 = math.sqrt(2.0)
    for Z in real_roots:
        ln_phi = (Z - 1.0) - math.log(Z - B) - (A / (2.0 * s2 * B)) * math.log((Z + (1.0 + s2) * B) / (Z + (1.0 - s2) * B))
        phi = math.exp(ln_phi)
        if phi < best_phi:
            best_phi, best_Z = phi, Z
    return best_Z, best_phi


def load_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def main():
    verify_hex_mic_python()
    iso = load_csv(BUILD / "isotherm_298K.csv")

    # Independent Peng-Robinson check against CSV at 1, 10, 20 bar
    for p_check in (1.0, 10.0, 20.0):
        idx = int(np.argmin(np.abs(iso["P_bar"] - p_check)))
        z_py, phi_py = peng_robinson_co2_phi(298.0, float(iso["P_bar"][idx]))
        diff_phi = abs(phi_py - float(iso["phi_PR"][idx]))
        print(f"[PY CHECK] PR EOS @ {p_check:4.1f} bar: Z={z_py:.6f}, phi={phi_py:.6f}, |dphi|={diff_phi:.2e}")
        assert diff_phi < 1e-5

    # Load pore density histogram (ib, ia) -> shape (128, 128)
    hist = np.loadtxt(BUILD / "pore_density_1bar.csv", delimiter=",")
    atoms = load_csv(BUILD / "framework_unit_atoms.csv")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=300)
    fig.patch.set_facecolor("white")

    # ------------------------------------------------------------------
    # Panel (a): 298 K CO2 Isotherm (log-P scale + Experiment gap callout)
    # ------------------------------------------------------------------
    ax = axes[0]
    P = iso["P_bar"]
    up = iso["uptake_mmol_g"]
    up_err = iso["uptake_err_mmol_g"]

    ax.errorbar(
        P, up, yerr=up_err, fmt="-o", color="#1f77b4", lw=2.2, ms=5.2, capsize=3,
        label="GPU Batched GCMC\n(UFF + REPEAT + TraPPE)"
    )

    # Literature experimental ranges at 298 K (Mason/Long; Dietzel)
    ax.axvspan(0.12, 0.18, color="#ff7f0e", alpha=0.10)
    ax.errorbar([0.15], [4.1], yerr=[[0.5], [0.5]], fmt="s", color="#d62728", ms=7, capsize=4, lw=1.8,
                label="Exp. Ref @ 0.15 bar (3.6-4.6)\n& 1.0 bar (7.5-8.6 mmol/g)")
    ax.errorbar([1.00], [8.05], yerr=[[0.55], [0.55]], fmt="s", color="#d62728", ms=7, capsize=4, lw=1.8)

    # Annotate the OMS polarization gap at 0.15 bar
    idx_015 = int(np.argmin(np.abs(P - 0.15)))
    ax.annotate(
        f"OMS Polarization Gap\n({up[idx_015]:.2f} vs ~4.1 mmol/g)\n-> Target of Step 2 (B1)",
        xy=(0.15, up[idx_015]), xytext=(0.015, 6.0),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.3),
        fontsize=8.5, color="#9e1b1b",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff5f5", ec="#f5c2c2", lw=0.8)
    )

    ax.set_xscale("log")
    ax.set_xlim(0.004, 25.0)
    ax.set_ylim(0.0, 16.5)
    ax.set_xlabel("Pressure P (bar, log scale)", fontweight="bold")
    ax.set_ylabel("CO₂ Uptake (mmol / g)", fontweight="bold")
    ax.set_title("(a) Mg-MOF-74 CO₂ Isotherm @ 298 K", fontweight="bold", pad=10)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(loc="upper left", fontsize=8.2, frameon=True, facecolor="white", edgecolor="#dddddd")

    # ------------------------------------------------------------------
    # Panel (b): Isosteric Heat of Adsorption Qst vs Loading
    # ------------------------------------------------------------------
    ax = axes[1]
    qst = iso["Qst_kJ_mol"]
    ax.plot(up, qst, "-^", color="#2ca02c", lw=2.1, ms=5.5, label="GPU GCMC Qst (Fluctuation)")
    ax.axhspan(40.0, 47.0, color="#d62728", alpha=0.12, label="Exp. Zero-Loading Qst (40-47 kJ/mol)")
    ax.axhline(26.0, color="#2ca02c", ls="--", lw=1.0, alpha=0.55)

    ax.set_xlim(-0.3, 15.8)
    ax.set_ylim(18.0, 52.0)
    ax.set_xlabel("CO₂ Loading (mmol / g)", fontweight="bold")
    ax.set_ylabel("Isosteric Heat Qst (kJ / mol)", fontweight="bold")
    ax.set_title("(b) Isosteric Heat of Adsorption", fontweight="bold", pad=10)
    ax.grid(True, ls=":", alpha=0.35)
    ax.legend(loc="upper right", fontsize=8.2, frameon=True, facecolor="white", edgecolor="#dddddd")

    # ------------------------------------------------------------------
    # Panel (c): 2D Hexagonal Pore CO2 Density @ 1.0 bar + Mg OMS Sites
    # ------------------------------------------------------------------
    ax = axes[2]
    na, nb = hist.shape[1], hist.shape[0]
    a_len = 26.136
    # Tile 2x2 unit cells in (sa, sb) around [0, 2)^2 so full hexagonal pore is centered cleanly
    sa_edges = np.linspace(0.0, 1.0, na + 1)
    sb_edges = np.linspace(0.0, 1.0, nb + 1)
    # Light 2-pass periodic smoothing over 128x128 fractional bins for smooth density contours
    sm = hist.astype(float)
    for _ in range(2):
        sm = (
            4.0 * sm
            + 2.0 * (np.roll(sm, 1, 0) + np.roll(sm, -1, 0) + np.roll(sm, 1, 1) + np.roll(sm, -1, 1))
            + (np.roll(np.roll(sm, 1, 0), 1, 1) + np.roll(np.roll(sm, 1, 0), -1, 1)
               + np.roll(np.roll(sm, -1, 0), 1, 1) + np.roll(np.roll(sm, -1, 0), -1, 1))
        ) / 16.0
    density = sm / max(float(np.max(sm)), 1.0)

    # Smooth slightly visually? Keep raw exact histogram normalized cleanly
    for ia_tile in (-1, 0, 1):
        for ib_tile in (-1, 0, 1):
            SA, SB = np.meshgrid(sa_edges + ia_tile, sb_edges + ib_tile)
            X = a_len * (SA - 0.5 * SB)
            Y = a_len * (math.sqrt(3.0) / 2.0) * SB
            ax.pcolormesh(X, Y, density, cmap="YlGnBu", vmin=0.0, vmax=0.85, shading="flat", rasterized=True)

    # Overlay framework atoms tiled across (-1..1, -1..1)
    elem_colors = {0: "#e63946", 1: "#ff9f1c", 2: "#4a4e69", 3: "#adb5bd"}
    elem_sizes = {0: 42, 1: 12, 2: 8, 3: 5}
    for el in (2, 3, 1, 0):
        mask = atoms["elem"] == el
        if not np.any(mask):
            continue
        sa_a = atoms["sa"][mask]
        sb_a = atoms["sb"][mask]
        xs, ys = [], []
        for ta in (-1, 0, 1):
            for tb in (-1, 0, 1):
                xa = a_len * ((sa_a + ta) - 0.5 * (sb_a + tb))
                ya = a_len * (math.sqrt(3.0) / 2.0) * (sb_a + tb)
                xs.append(xa)
                ys.append(ya)
        xs = np.concatenate(xs)
        ys = np.concatenate(ys)
        lbl = "Mg²⁺ OMS" if el == 0 else ("Framework (O/C/H)" if el == 1 else None)
        ax.scatter(xs, ys, s=elem_sizes[el], c=elem_colors[el],
                   edgecolors="black" if el == 0 else "none", linewidths=0.6,
                   alpha=0.95 if el == 0 else 0.70, label=lbl, zorder=4 if el == 0 else 3)

    # Center view on the main hexagonal channel centered near (0, a/sqrt(3)) ~ (0, 15.09)
    cx0, cy0 = 0.0, a_len / math.sqrt(3.0)
    span = 11.5
    ax.set_xlim(cx0 - span, cx0 + span)
    ax.set_ylim(cy0 - span, cy0 + span)
    ax.set_aspect("equal")
    ax.set_xlabel("x (Å, ab-plane projection)", fontweight="bold")
    ax.set_ylabel("y (Å, ab-plane projection)", fontweight="bold")
    ax.set_title("(c) 1.0 bar Pore CO₂ Density & Mg²⁺ OMS", fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=8.0, frameon=True, facecolor="white", edgecolor="#dddddd")

    fig.suptitle(
        "GPU Batched GCMC (RTX 5070, 13.91 s for 9.6M Steps): Mg-MOF-74 + CO₂ @ 298 K [UFF + REPEAT + TraPPE]",
        fontsize=11.8, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    out_png = ASSETS / "step1_mof74_co2_298K.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[PLOT] Saved 300 DPI figure to {out_png}")


if __name__ == "__main__":
    main()
