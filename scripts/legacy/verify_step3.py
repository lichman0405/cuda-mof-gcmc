"""Independent mixture PR EOS verification & 300 DPI plotting for Step 3."""

import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
ASSETS = ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def load_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


SPECIES_PR = {
    "CO2": (304.1282, 73.773, 0.22394),
    "N2":  (126.192,  33.958, 0.0372),
    "CH4": (190.564,  45.992, 0.01142),
}


def binary_pr_phi(T_K, P_bar, name0, name1, y0, y1, kij):
    R = 0.08314462618
    def ab(name):
        Tc, Pc, om = SPECIES_PR[name]
        k = 0.37464 + 1.54226 * om - 0.26992 * om ** 2
        al = (1.0 + k * (1.0 - math.sqrt(T_K / Tc))) ** 2
        a = 0.4572355289213822 * (R * Tc) ** 2 / Pc * al
        b = 0.07779607390388846 * R * Tc / Pc
        return a, b

    a0, b0 = ab(name0)
    a1, b1 = ab(name1)
    a01 = math.sqrt(a0 * a1) * (1.0 - kij)
    am = y0 * y0 * a0 + 2.0 * y0 * y1 * a01 + y1 * y1 * a1
    bm = y0 * b0 + y1 * b1
    A = am * P_bar / (R * T_K) ** 2
    B = bm * P_bar / (R * T_K)

    roots = np.roots([1.0, -(1.0 - B), A - 3.0 * B ** 2 - 2.0 * B, -(A * B - B ** 2 - B ** 3)])
    real_roots = [float(np.real(r)) for r in roots if abs(np.imag(r)) < 1e-9 and np.real(r) > B]
    s2 = math.sqrt(2.0)
    best_g, best_phi0, best_phi1 = 1e30, 1.0, 1.0
    for Z in real_roots:
        t2 = math.log((Z + (1.0 + s2) * B) / (Z + (1.0 - s2) * B))
        lz = math.log(Z - B)
        lp0 = (b0 / bm) * (Z - 1.0) - lz - (A / (2.0 * s2 * B)) * (2.0 * (y0 * a0 + y1 * a01) / am - b0 / bm) * t2
        lp1 = (b1 / bm) * (Z - 1.0) - lz - (A / (2.0 * s2 * B)) * (2.0 * (y0 * a01 + y1 * a1) / am - b1 / bm) * t2
        g = y0 * lp0 + y1 * lp1
        if g < best_g:
            best_g = g
            best_phi0 = math.exp(lp0)
            best_phi1 = math.exp(lp1)
    return best_phi0, best_phi1


def check_mix(name, data, s0_name, s1_name, y0, y1, kij):
    for col in data.dtype.names:
        arr = data[col]
        assert not np.any(np.isnan(arr)), f"{name}: NaN in {col}"
        assert not np.any(np.isinf(arr)), f"{name}: Inf in {col}"
    for p_chk in (1.0, 10.0, 30.0):
        idx = int(np.argmin(np.abs(data["P_bar"] - p_chk)))
        phi0, phi1 = binary_pr_phi(298.0, float(data["P_bar"][idx]), s0_name, s1_name, y0, y1, kij)
        d0 = abs(phi0 - float(data["phi0"][idx]))
        d1 = abs(phi1 - float(data["phi1"][idx]))
        assert d0 < 1e-5 and d1 < 1e-5, f"{name} PR mismatch at {p_chk}: {d0}, {d1}"
    di0 = float(np.max(np.abs(data["acc_i0"] - data["acc_d0"])))
    di1 = float(np.max(np.abs(data["acc_i1"] - data["acc_d1"])))
    idx1 = int(np.argmin(np.abs(data["P_bar"] - 1.0)))
    print(f"[CHECK {name:16s}] @1 bar: sp0={data['uptake0_mmol_g'][idx1]:.3f}, "
          f"sp1={data['uptake1_mmol_g'][idx1]:.3f}, S={data['selectivity'][idx1]:.1f} | "
          f"max|i0-d0|={di0:.4f}, max|i1-d1|={di1:.4f}")


def smooth_periodic(hist):
    sm = hist.astype(float)
    for _ in range(2):
        sm = (
            4.0 * sm
            + 2.0 * (np.roll(sm, 1, 0) + np.roll(sm, -1, 0) + np.roll(sm, 1, 1) + np.roll(sm, -1, 1))
            + (np.roll(np.roll(sm, 1, 0), 1, 1) + np.roll(np.roll(sm, 1, 0), -1, 1)
               + np.roll(np.roll(sm, -1, 0), 1, 1) + np.roll(np.roll(sm, -1, 0), -1, 1))
        ) / 16.0
    return sm / max(float(np.max(sm)), 1.0)


def main():
    n2_non = load_csv(BUILD / "mix_co2_n2_nonpol.csv")
    n2_pol = load_csv(BUILD / "mix_co2_n2_pol3site.csv")
    ch4_non = load_csv(BUILD / "mix_co2_ch4_nonpol.csv")
    ch4_pol = load_csv(BUILD / "mix_co2_ch4_pol3site.csv")

    check_mix("CO2/N2 NonPol", n2_non, "CO2", "N2", 0.15, 0.85, -0.017)
    check_mix("CO2/N2 Pol-3S", n2_pol, "CO2", "N2", 0.15, 0.85, -0.017)
    check_mix("CO2/CH4 NonPol", ch4_non, "CO2", "CH4", 0.50, 0.50, 0.092)
    check_mix("CO2/CH4 Pol-3S", ch4_pol, "CO2", "CH4", 0.50, 0.50, 0.092)

    h_co2 = np.loadtxt(BUILD / "hist_mix_co2_1bar.csv", delimiter=",")
    h_n2  = np.loadtxt(BUILD / "hist_mix_n2_1bar.csv", delimiter=",")
    atoms = load_csv(BUILD / "framework_unit_atoms.csv")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
    })

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.5), dpi=300)
    fig.patch.set_facecolor("white")

    # ------------------------------------------------------------------
    # Panel (a): Flue Gas CO2/N2 (15:85)
    # ------------------------------------------------------------------
    ax = axes[0]
    P = n2_pol["P_bar"]
    l1, = ax.plot(P, n2_pol["uptake0_mmol_g"], "-D", color="#1f77b4", lw=2.2, ms=4.8, label="CO₂ (Pol-3Site)")
    l2, = ax.plot(P, n2_non["uptake0_mmol_g"], "--o", color="#6baed6", lw=1.7, ms=4.2, label="CO₂ (NonPol)")
    l3, = ax.plot(P, n2_pol["uptake1_mmol_g"], "-s", color="#ff7f0e", lw=1.8, ms=4.2, label="N₂ (Pol-3Site)")
    l4, = ax.plot(P, n2_non["uptake1_mmol_g"], ":^", color="#fdae6b", lw=1.6, ms=4.2, label="N₂ (NonPol)")

    ax.set_xscale("log")
    ax.set_xlim(0.08, 36.0)
    ax.set_ylim(0.0, 13.5)
    ax.set_xlabel("Total Pressure P (bar, log scale)", fontweight="bold")
    ax.set_ylabel("Competitive Uptake (mmol / g)", fontweight="bold")
    ax.set_title("(a) Flue Gas CO₂/N₂ (15:85) @ 298 K", fontweight="bold", pad=10)
    ax.grid(True, which="both", ls=":", alpha=0.35)

    ax_s = ax.twinx()
    s1, = ax_s.plot(P, n2_pol["selectivity"], "-*", color="#d62728", lw=2.0, ms=6.0, label="S(CO₂/N₂) Pol-3Site")
    s2, = ax_s.plot(P, n2_non["selectivity"], "--x", color="#e377c2", lw=1.5, ms=4.8, label="S(CO₂/N₂) NonPol")
    ax_s.set_ylim(0.0, 82.0)
    ax_s.set_ylabel("Selectivity S(CO₂ / N₂)", fontweight="bold", color="#b2182b")
    ax_s.tick_params(axis="y", colors="#b2182b")

    lines = [l1, l2, l3, l4, s1, s2]
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=7.5,
              frameon=True, facecolor="white", edgecolor="#dddddd")

    # ------------------------------------------------------------------
    # Panel (b): Biogas CO2/CH4 (50:50)
    # ------------------------------------------------------------------
    ax = axes[1]
    P = ch4_pol["P_bar"]
    l1, = ax.plot(P, ch4_pol["uptake0_mmol_g"], "-D", color="#1f77b4", lw=2.2, ms=4.8, label="CO₂ (Pol-3Site)")
    l2, = ax.plot(P, ch4_non["uptake0_mmol_g"], "--o", color="#6baed6", lw=1.7, ms=4.2, label="CO₂ (NonPol)")
    l3, = ax.plot(P, ch4_pol["uptake1_mmol_g"], "-s", color="#2ca02c", lw=1.8, ms=4.2, label="CH₄ (Pol-3Site)")
    l4, = ax.plot(P, ch4_non["uptake1_mmol_g"], ":^", color="#98df8a", lw=1.6, ms=4.2, label="CH₄ (NonPol)")

    ax.set_xscale("log")
    ax.set_xlim(0.08, 36.0)
    ax.set_ylim(0.0, 16.0)
    ax.set_xlabel("Total Pressure P (bar, log scale)", fontweight="bold")
    ax.set_ylabel("Competitive Uptake (mmol / g)", fontweight="bold")
    ax.set_title("(b) Biogas CO₂/CH₄ (50:50) @ 298 K", fontweight="bold", pad=10)
    ax.grid(True, which="both", ls=":", alpha=0.35)

    ax_s2 = ax.twinx()
    s1, = ax_s2.plot(P, ch4_pol["selectivity"], "-*", color="#d62728", lw=2.0, ms=6.0, label="S(CO₂/CH₄) Pol-3Site")
    s2, = ax_s2.plot(P, ch4_non["selectivity"], "--x", color="#e377c2", lw=1.5, ms=4.8, label="S(CO₂/CH₄) NonPol")
    ax_s2.set_ylim(0.0, 30.0)
    ax_s2.set_ylabel("Selectivity S(CO₂ / CH₄)", fontweight="bold", color="#b2182b")
    ax_s2.tick_params(axis="y", colors="#b2182b")

    lines = [l1, l2, l3, l4, s1, s2]
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=7.5,
              frameon=True, facecolor="white", edgecolor="#dddddd")

    # ------------------------------------------------------------------
    # Panel (c): 1.0 bar Competitive Pore Localization (CO2 heatmap + N2 contours)
    # ------------------------------------------------------------------
    ax = axes[2]
    ax.spines["right"].set_visible(False)
    na, nb = h_co2.shape[1], h_co2.shape[0]
    a_len = 26.136
    sa_edges = np.linspace(0.0, 1.0, na + 1)
    sb_edges = np.linspace(0.0, 1.0, nb + 1)
    sa_centers = 0.5 * (sa_edges[:-1] + sa_edges[1:])
    sb_centers = 0.5 * (sb_edges[:-1] + sb_edges[1:])

    d_co2 = smooth_periodic(h_co2)
    d_n2  = smooth_periodic(h_n2)

    for ia_tile in (-1, 0, 1):
        for ib_tile in (-1, 0, 1):
            SA, SB = np.meshgrid(sa_edges + ia_tile, sb_edges + ib_tile)
            X = a_len * (SA - 0.5 * SB)
            Y = a_len * (math.sqrt(3.0) / 2.0) * SB
            ax.pcolormesh(X, Y, d_co2, cmap="YlGnBu", vmin=0.0, vmax=0.80, shading="flat", rasterized=True)

            SAc, SBc = np.meshgrid(sa_centers + ia_tile, sb_centers + ib_tile)
            Xc = a_len * (SAc - 0.5 * SBc)
            Yc = a_len * (math.sqrt(3.0) / 2.0) * SBc
            ax.contour(Xc, Yc, d_n2, levels=[0.25, 0.45, 0.65], colors=["#ff7f0e"],
                       linewidths=[1.0, 1.4, 1.8], alpha=0.85, zorder=3)

    # Dummy artist for N2 contour legend
    ax.plot([], [], color="#ff7f0e", lw=1.6, label="N₂ Contours (Displaced)")

    elem_colors = {0: "#e63946", 1: "#ff9f1c", 2: "#4a4e69", 3: "#adb5bd"}
    elem_sizes = {0: 42, 1: 12, 2: 8, 3: 5}
    for el in (2, 3, 1, 0):
        mask = atoms["elem"] == el
        if not np.any(mask):
            continue
        sa_a, sb_a = atoms["sa"][mask], atoms["sb"][mask]
        xs, ys = [], []
        for ta in (-1, 0, 1):
            for tb in (-1, 0, 1):
                xs.append(a_len * ((sa_a + ta) - 0.5 * (sb_a + tb)))
                ys.append(a_len * (math.sqrt(3.0) / 2.0) * (sb_a + tb))
        xs, ys = np.concatenate(xs), np.concatenate(ys)
        lbl = "Mg²⁺ OMS (CO₂ Locks Here)" if el == 0 else None
        ax.scatter(xs, ys, s=elem_sizes[el], c=elem_colors[el],
                   edgecolors="black" if el == 0 else "none", linewidths=0.6,
                   alpha=0.95 if el == 0 else 0.65, label=lbl, zorder=4 if el == 0 else 3)

    cx0, cy0 = 0.0, a_len / math.sqrt(3.0)
    span = 11.5
    ax.set_xlim(cx0 - span, cx0 + span)
    ax.set_ylim(cy0 - span, cy0 + span)
    ax.set_aspect("equal")
    ax.set_xlabel("x (Å, ab-plane projection)", fontweight="bold")
    ax.set_ylabel("y (Å, ab-plane projection)", fontweight="bold")
    ax.set_title("(c) 1.0 bar Flue Gas: CO₂ vs N₂ in Pore", fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=7.8, frameon=True, facecolor="white", edgecolor="#dddddd")

    fig.suptitle(
        "Step 3: GPU Competitive Co-Adsorption in Mg-MOF-74 @ 298 K (Identity-Swap GCMC + Pol-3Site)",
        fontsize=12.0, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    out_png = ASSETS / "step3_mof74_binary_coadsorption.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[PLOT] Saved 300 DPI figure to {out_png}")


if __name__ == "__main__":
    main()
