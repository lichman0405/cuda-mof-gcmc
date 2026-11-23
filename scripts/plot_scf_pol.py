"""Publication-quality 300 DPI visualization for Level 2 Full Self-Consistent Many-Body SCF Polarization."""

import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
ASSETS = ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

KE_K = 167100.9558738398
THOLE_A = 2.1304


def load_csv(path: Path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def mic_super(H, H_inv, dv):
    df = H_inv @ dv
    df = df - np.round(df)
    return H @ df


def compute_exact_matrix_check():
    data = json.loads((BUILD / "scf_test_dump.json").read_text(encoding="utf-8"))
    H = np.array(data["H_super"], dtype=np.float64)
    H_inv = np.array(data["H_super_inv"], dtype=np.float64)
    rc = float(data["r_cut_A"])
    rc2 = rc * rc
    alpha = float(data["dsf_alpha"])
    bond = float(data["bond_A"])
    q_sites = data["q_sites"]
    a_par = float(data["alpha_par"])
    a_perp = float(data["alpha_perp"])
    a_iso = float(data["alpha_iso"])
    mols = data["molecules"]
    N = len(mols)

    erfc_rc = math.erfc(alpha * rc)
    exp_rc = math.exp(-alpha * alpha * rc2)
    dsf_deriv = erfc_rc / rc2 + (2.0 * alpha / math.sqrt(math.pi)) * exp_rc / rc

    c_arr = np.array([m["c"] for m in mols], dtype=np.float64)
    u_arr = np.array([m["u"] for m in mols], dtype=np.float64)
    efw_arr = np.array([m["efw"] for m in mols], dtype=np.float64)
    e0_gpu = np.array([m["e0"] for m in mols], dtype=np.float64)
    mu_gpu = np.array([m["mu"] for m in mols], dtype=np.float64)

    off_s = [0.0, bond, -bond]
    e0_py = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        e = efw_arr[i].copy()
        for j in range(N):
            if i == j:
                continue
            dm = mic_super(H, H_inv, c_arr[i] - c_arr[j])
            if np.dot(dm, dm) > (rc + bond) ** 2:
                continue
            for b in range(3):
                qb = q_sites[b]
                if qb == 0.0:
                    continue
                d = dm - off_s[b] * u_arr[j]
                r2 = float(np.dot(d, d))
                if r2 >= rc2 or r2 < 0.25:
                    continue
                r = math.sqrt(r2)
                fac = KE_K * qb * (
                    math.erfc(alpha * r) / r2
                    + (2.0 * alpha / math.sqrt(math.pi)) * math.exp(-alpha * alpha * r2) / r
                    - dsf_deriv
                ) / r
                e += fac * d
        e0_py[i] = e

    M = np.eye(3 * N, dtype=np.float64)
    rhs = np.zeros(3 * N, dtype=np.float64)
    A_blocks = []
    for i in range(N):
        ui = u_arr[i].reshape(3, 1)
        Ai = (1.0 / KE_K) * (a_perp * np.eye(3) + (a_par - a_perp) * (ui @ ui.T))
        A_blocks.append(Ai)
        rhs[3 * i:3 * i + 3] = Ai @ e0_py[i]

    inv_s = 1.0 / ((a_iso * a_iso) ** (1.0 / 6.0))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = mic_super(H, H_inv, c_arr[i] - c_arr[j])
            r2 = float(np.dot(d, d))
            if r2 >= rc2 or r2 < 0.25:
                continue
            r = math.sqrt(r2)
            u_t = r * inv_s
            au3 = THOLE_A * (u_t ** 3)
            exp_t = math.exp(-au3) if au3 < 50.0 else 0.0
            fe = 1.0 - exp_t
            ft = 1.0 - (1.0 + au3) * exp_t
            Tij = KE_K * (3.0 * ft / (r2 * r2 * r) * np.outer(d, d) - fe / (r2 * r) * np.eye(3))
            M[3 * i:3 * i + 3, 3 * j:3 * j + 3] = -(A_blocks[i] @ Tij)

    mu_exact = np.linalg.solve(M, rhs).reshape(N, 3)
    u_pol_exact = -0.5 * float(np.sum(mu_exact * e0_py))
    u_gpu = float(data["u_pol_scf"])

    return {
        "mu_gpu": mu_gpu,
        "mu_exact": mu_exact,
        "max_e0_err": float(np.max(np.abs(e0_py - e0_gpu))),
        "max_mu_err": float(np.max(np.abs(mu_gpu - mu_exact))),
        "du_err": abs(u_gpu - u_pol_exact),
        "iters": int(data["iters"]),
        "u_gpu": u_gpu,
    }


def main():
    nonpol = load_csv(BUILD / "cli_uff_298K.csv")
    scf_co2 = load_csv(BUILD / "cli_uff_298K_scf.csv")
    pol3_flue = load_csv(BUILD / "cli_fluegas_298K_pol3site.csv")
    scf_flue = load_csv(BUILD / "cli_fluegas_298K_scf.csv")
    mat = compute_exact_matrix_check()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.8,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.8,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 8.4,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.3), dpi=300)
    fig.subplots_adjust(left=0.055, right=0.965, top=0.84, bottom=0.14, wspace=0.33)

    c_nonpol = "#64748B"
    c_pol3 = "#D97706"
    c_scf = "#0F766E"
    c_upol = "#DC2626"
    c_blue = "#2563EB"

    # ------------------------------------------------------------------
    # Panel (a): Pure CO2 298 K (NonPol vs Pol-SCF) + |U_pol / N|
    # ------------------------------------------------------------------
    ax1 = axes[0]
    pressures = np.array([float(p) for p in scf_co2["P_bar"]])
    u_nonpol = np.array([float(x) for x in nonpol["uptake_mmol_g"]])
    sem_nonpol = np.array([float(x) for x in nonpol["uptake_sem_mmol_g"]])
    u_scf = np.array([float(x) for x in scf_co2["uptake_mmol_g"]])
    sem_scf = np.array([float(x) for x in scf_co2["uptake_sem_mmol_g"]])
    upol_abs = np.abs(np.array([float(x) for x in scf_co2["avg_Upol_kJ"]]))

    x = np.arange(len(pressures))
    w = 0.34
    b1 = ax1.bar(
        x - w / 2, u_nonpol, width=w, yerr=sem_nonpol, capsize=3,
        color=c_nonpol, alpha=0.88, edgecolor="#334155", linewidth=0.8,
        label="NonPol (TraPPE/UFF)", zorder=3
    )
    b2 = ax1.bar(
        x + w / 2, u_scf, width=w, yerr=sem_scf, capsize=3,
        color=c_scf, alpha=0.92, edgecolor="#115E59", linewidth=0.8,
        label="Level 2 Pol-SCF", zorder=3
    )

    deltas = (u_scf - u_nonpol) / u_nonpol * 100.0
    for idx in range(len(pressures)):
        top_y = max(u_nonpol[idx] + sem_nonpol[idx], u_scf[idx] + sem_scf[idx])
        sign_str = "+" if deltas[idx] >= 0 else ""
        tag = f"{sign_str}{deltas[idx]:.1f}%"
        if idx == 2:
            tag += "\n(sat.)"
        ax1.text(
            x[idx] + w / 2, u_scf[idx] + sem_scf[idx] + 0.35, tag,
            ha="center", va="bottom", fontsize=8.3, fontweight="bold",
            color="#0F766E" if deltas[idx] >= 0 else "#475569", zorder=6
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{p:.2f} bar" for p in pressures])
    ax1.set_ylabel(r"$\mathrm{CO_2}$ Uptake (mmol/g)")
    ax1.set_ylim(0, 16.5)
    ax1.grid(axis="y", linestyle=":", alpha=0.45, zorder=0)
    ax1.spines["top"].set_visible(False)

    ax1_r = ax1.twinx()
    ax1_r.plot(
        x, upol_abs, color=c_upol, marker="D", markersize=6.0,
        markeredgecolor="white", markeredgewidth=0.9, linewidth=1.8, linestyle="--",
        label=r"$|U_{\mathrm{pol}}^{\mathrm{SCF}}/N|$ (kJ/mol)", zorder=7
    )
    for idx in range(len(pressures)):
        ax1_r.text(
            x[idx], upol_abs[idx] + 0.085, f"{upol_abs[idx]:.2f}",
            ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=c_upol,
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.82),
            zorder=8
        )
    ax1_r.set_ylabel(r"SCF Polarization $|U_{\mathrm{pol}}/N|$ (kJ/mol)", color=c_upol)
    ax1_r.tick_params(axis="y", colors=c_upol)
    ax1_r.set_ylim(0.0, 1.48)
    ax1_r.spines["top"].set_visible(False)

    h1, l1 = ax1.get_legend_handles_labels()
    h1r, l1r = ax1_r.get_legend_handles_labels()
    ax1.legend(h1 + h1r, l1 + l1r, loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1")
    ax1.set_title("(a) Pure CO$_2$ (298 K): OMS Boost & Screening")

    # ------------------------------------------------------------------
    # Panel (b): Flue Gas (15:85 CO2/N2, 298 K) Level 1 vs Level 2
    # ------------------------------------------------------------------
    ax2 = axes[1]
    p_flue = np.array([1.0, 2.0])
    s_pol3, s_scf = [], []
    up_pol3, up_scf = [], []
    for p in p_flue:
        i3 = int(np.argmin(np.abs(pol3_flue["P_bar"] - p)))
        i2 = int(np.argmin(np.abs(scf_flue["P_bar"] - p)))
        s_pol3.append(float(pol3_flue["selectivity"][i3]))
        s_scf.append(float(scf_flue["selectivity"][i2]))
        up_pol3.append(abs(float(pol3_flue["avg_Upol0_kJ"][i3])))
        up_scf.append(abs(float(scf_flue["avg_Upol0_kJ"][i2])))

    xf = np.arange(len(p_flue))
    wf = 0.32
    ax2.bar(
        xf - wf / 2, s_pol3, width=wf, color=c_pol3, alpha=0.90,
        edgecolor="#92400E", linewidth=0.8, label="Level 1 Pol-3Site (FW only)", zorder=3
    )
    ax2.bar(
        xf + wf / 2, s_scf, width=wf, color=c_scf, alpha=0.92,
        edgecolor="#115E59", linewidth=0.8, label="Level 2 Pol-SCF (Many-Body)", zorder=3
    )
    for idx in range(len(p_flue)):
        ax2.text(
            xf[idx] - wf / 2, s_pol3[idx] + 1.4, f"{s_pol3[idx]:.1f}",
            ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#92400E", zorder=6
        )
        ax2.text(
            xf[idx] + wf / 2, s_scf[idx] + 1.4, f"{s_scf[idx]:.1f}",
            ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#0F766E", zorder=6
        )

    ax2.set_xticks(xf)
    ax2.set_xticklabels([f"{p:.1f} bar (15:85)" for p in p_flue])
    ax2.set_xlim(-0.55, 1.55)
    ax2.set_ylim(0, 92)
    ax2.set_ylabel(r"Selectivity $S(\mathrm{CO_2/N_2})$")
    ax2.grid(axis="y", linestyle=":", alpha=0.45, zorder=0)
    ax2.spines["top"].set_visible(False)

    ax2_r = ax2.twinx()
    x_pts = np.array([xf[0] - wf / 2, xf[0] + wf / 2, xf[1] - wf / 2, xf[1] + wf / 2])
    y_pts = np.array([up_pol3[0], up_scf[0], up_pol3[1], up_scf[1]])
    for g in range(2):
        ax2_r.plot(
            x_pts[2 * g:2 * g + 2], y_pts[2 * g:2 * g + 2],
            color=c_upol, linestyle="--", linewidth=1.6, marker="D", markersize=6.2,
            markeredgecolor="white", markeredgewidth=0.9,
            label=r"$|U_{\mathrm{pol}}(\mathrm{CO_2})|$ (kJ/mol)" if g == 0 else None,
            zorder=7
        )
    for xp, yp in zip(x_pts, y_pts):
        dy = -0.22 if yp > 2.0 else 0.16
        va_str = "top" if yp > 2.0 else "bottom"
        ax2_r.text(
            xp, yp + dy, f"{yp:.2f}",
            ha="center", va=va_str, fontsize=8.2, fontweight="bold", color=c_upol,
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.88),
            zorder=8
        )
    ax2_r.set_ylabel(r"$\mathrm{CO_2}$ $|U_{\mathrm{pol}}/N|$ (kJ/mol)", color=c_upol)
    ax2_r.tick_params(axis="y", colors=c_upol)
    ax2_r.set_ylim(0.0, 4.7)
    ax2_r.spines["top"].set_visible(False)

    h2, l2 = ax2.get_legend_handles_labels()
    h2r, l2r = ax2_r.get_legend_handles_labels()
    ax2.legend(h2 + h2r, l2 + l2r, loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1")
    ax2.set_title(r"(b) Flue Gas Screening: Pol-3Site vs. Pol-SCF")

    # ------------------------------------------------------------------
    # Panel (c): Exact 36x36 Applequist-Thole Matrix Inversion Parity
    # ------------------------------------------------------------------
    ax3 = axes[2]
    mu_g = mat["mu_gpu"].ravel()
    mu_e = mat["mu_exact"].ravel()
    lim = float(max(np.max(np.abs(mu_g)), np.max(np.abs(mu_e)))) * 1.15

    ax3.plot([-lim, lim], [-lim, lim], linestyle="--", color="#94A3B8", linewidth=1.3, label=r"Exact $y = x$", zorder=1)
    ax3.scatter(
        mu_e, mu_g, s=42, color=c_blue, alpha=0.88,
        edgecolors="white", linewidths=0.7,
        label=r"36 dipole components ($12\ \mathrm{CO_2}$)", zorder=4
    )
    ax3.set_xlim(-lim, lim)
    ax3.set_ylim(-lim, lim)
    ax3.set_xlabel(r"Exact Matrix $\mu_{\alpha}^{\mathrm{Inv}}$ ($e\cdot\mathrm{\AA}$)")
    ax3.set_ylabel(r"GPU SCF $\mu_{\alpha}^{\mathrm{GPU}}$ ($e\cdot\mathrm{\AA}$)")
    ax3.grid(True, linestyle=":", alpha=0.45, zorder=0)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    md_co2 = float(np.max(scf_co2["max_drift_K"]))
    md_flue = float(np.max(scf_flue["max_drift_K"]))
    max_drift_all = max(md_co2, md_flue)
    metrics_txt = (
        f"Converged: {mat['iters']} SCF iters\n"
        f"max|$\\Delta E_0$|: {mat['max_e0_err']:.2e} K/(e·Å)\n"
        f"max|$\\Delta\\mu$|: {mat['max_mu_err']:.2e} e·Å\n"
        f"|$\\Delta U_{{SCF}}$|: {mat['du_err']:.2e} K\n"
        f"GCMC Drift ≤ {max_drift_all:.2e} K"
    )
    ax3.text(
        0.04, 0.96, metrics_txt, transform=ax3.transAxes,
        va="top", ha="left", fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.35", fc="#F8FAFC", ec="#CBD5E1", lw=0.9),
        zorder=7
    )

    # Inset: Per-molecule L2 dipole error on log10 scale
    ax_ins = inset_axes(ax3, width="45%", height="34%", loc="lower right", borderpad=1.2)
    mol_err = np.maximum(np.linalg.norm(mat["mu_gpu"] - mat["mu_exact"], axis=1), 1.0e-16)
    mol_ids = np.arange(1, len(mol_err) + 1)
    ax_ins.semilogy(mol_ids, mol_err, marker="o", markersize=3.2, color=c_scf, linewidth=1.2)
    ax_ins.axhline(1.0e-7, color=c_upol, linestyle=":", linewidth=1.0)
    ax_ins.text(1.3, 1.8e-7, "tol 1e-7", fontsize=6.8, color=c_upol, fontweight="bold")
    ax_ins.set_ylim(1.0e-17, 2.0e-6)
    ax_ins.set_xticks([1, 4, 8, 12])
    ax_ins.set_yticks([1e-16, 1e-12, 1e-8])
    ax_ins.tick_params(labelsize=7.0)
    ax_ins.set_title(r"Per-Mol $\|\Delta\mu\|_2$ ($e\cdot\mathrm{\AA}$)", fontsize=7.6, pad=2)
    ax_ins.grid(True, which="major", linestyle=":", alpha=0.4)

    ax3.set_title(r"(c) Exact $36\times 36$ Matrix Solve Check")

    fig.suptitle(
        "Level 2 Full Self-Consistent Many-Body SCF Polarization (Mg-MOF-74, 298 K, 0.10 Å Tricubic UFF)",
        fontsize=12.6, fontweight="bold", y=0.96
    )

    out_png = ASSETS / "step5_level2_scf_polarization.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved {out_png} ({out_png.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
