"""Render clean academic monochrome-slate architecture diagrams (EN & ZH) for cuda-mof-gcmc."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT_EN = ROOT / "assets" / "architecture.png"
OUT_ZH = ROOT / "assets" / "architecture_zh.png"

ACCENT = "#1E3A5F"
ACCENT_BG = "#EFF4FA"
ACCENT_BORDER = "#B9CDE3"
ZONE_BG = "#F8FAFC"
ZONE_BORDER = "#CBD5E1"
CARD_BG = "#FFFFFF"
CARD_BORDER = "#CBD5E1"
TEXT_MAIN = "#0F172A"
TEXT_MUTED = "#475569"
TEXT_SUB = "#334155"


def add_round_box(
    ax, x, y, w, h, *,
    fc=CARD_BG, ec=CARD_BORDER, lw=1.1, ls="-", r=0.9, zorder=2, shadow=False
):
    if shadow:
        sh = FancyBboxPatch(
            (x + 0.18, y - 0.20), w, h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor="#0F172A", alpha=0.04, edgecolor="none", zorder=zorder - 0.5,
        )
        ax.add_patch(sh)
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder,
    )
    ax.add_patch(box)
    return box


def add_pill(ax, cx, cy, text, *, fc="#FFFFFF", ec=ACCENT, tc=ACCENT, fs=8.4, zorder=6):
    return ax.text(
        cx, cy, text, ha="center", va="center",
        fontsize=fs, fontweight="bold", color=tc, zorder=zorder + 1,
        bbox=dict(boxstyle="round,pad=0.28", facecolor=fc, edgecolor=ec, linewidth=1.0),
    )


def add_arrow(ax, p1, p2, *, color="#64748B", lw=1.5, rad=0.0, ls="-", ms=11, zorder=5):
    arr = FancyArrowPatch(
        p1, p2,
        arrowstyle="-|>", mutation_scale=ms,
        linewidth=lw, linestyle=ls, color=color,
        connectionstyle=f"arc3,rad={rad}", zorder=zorder,
    )
    ax.add_patch(arr)
    return arr


TEXTS = {
    "zh": {
        "fonts": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "title": "cuda-mof-gcmc 系统架构图",
        "badge_x": 26.2,
        "subtitle": "GPU 经典内层提议 + 外层 MLIP 无偏校正（零磁盘 I/O 内存直连闭环）",
        "l1_hdr": "01 · 输入与结构前处理层 (Python / ASE)",
        "c1a_t": "qeq_charges.py · MOF 结构与电荷",
        "c1a_1": "• ASE 展开全部 230 空间群 · 读取 CIF / REPEAT",
        "c1a_2": "• 无电荷骨架自动周期 QEq (MAE 0.11–0.15 e)",
        "c1b_t": "guests.py · 多位点刚体客体库",
        "c1b_1": "• 8 种预设 (CO2/CH4/N2/H2/Ar/H2O/NH3) + JSON",
        "c1b_2": "• 支持 Z=0 虚位点与 Peng-Robinson 临界参数",
        "c1c_t": "后续扩展位 · 多势函数与 8 卡 Linux",
        "c1c_1": "• MatterSim / SevenNet / ORB / FAIRChem UMA",
        "c1c_2": "• Linux .so 构建 + 按压强调度至多张 GPU",
        "l2_hdr": "02 · 调度与 Nested MC 核心",
        "c2a_t": "run_mlip_gcmc.py · 任务入口",
        "c2a_1": "• 自适应 3D 网格密度与分块大小",
        "c2a_2": "• 3 阶段化学势偏移 Δε 自适应校准",
        "c2a_3": "• 断点续跑 --resume 与收敛诊断",
        "c2a_4": "输出 JSON / CSV 等温线与诊断",
        "c2b_t": "nested_mc_engine.py · 核心循环",
        "c2b_1": "• 批量评估与精确 Metropolis 判据",
        "c2b_2": "• 零偏置: ΔΔU = ΔU_MLIP − ΔU_ref − Δε·ΔN",
        "c2b_3": "• Pooled-Jackknife Qst 吸附热",
        "c2b_4": "• FFT τ_int、ESS/GPU-s 与 SJD_N/s",
        "c2b_pill": "Python <-> CUDA ctypes 内存直连",
        "l3_hdr": "03 · 内层 GPU 经典提议器 (CUDA DLL)",
        "c3a_t": "mof_gcmc_proposer.dll",
        "c3a_1": "proposer_api.h / proposer_dll.cu",
        "c3a_2": "内存直传建网格 · 多卡 device_id 绑定",
        "c3b_t": "通用物理内核与 3D 三次样条网格",
        "c3b_1": "• cell.hpp: 任意三斜晶胞精确 MIC",
        "c3b_2": "• forcefield.hpp: 103 元素 UFF + PR",
        "c3b_3": "• unified_grid_builder.cu: LJ / Phi / E",
        "c3c_t": "GPU Burst Kernel (四元数刚体)",
        "c3c_1": "• 多副本并行连跑 K 步刚体 GCMC",
        "c3c_2": "• ≤12 位点 / 6 LJ · 末步重算零漂移",
        "c3c_3": "提议耗时仅 ~1.8 ms / 轮 (4 副本 × K 步)",
        "l4_hdr": "04 · 外层 MLIP 能量评估 (PyTorch)",
        "c4a_t": "FastMaceD3Target · 批量推理",
        "c4a_1": "• 四元数展开真实原子 (跳过 Z=0)",
        "c4a_2": "• 无梯度 no_grad 分块前向 (chunk)",
        "c4a_3": "• GPU DFT-D3(BJ): 缓存静态骨架边表",
        "c4a_4": "• 与 Serial ASE FP32 误差 < 0.1%",
        "c4a_pill": "相比串行 ASE 推理提速 4–5×",
        "c4b_t": "显存防碎片保护",
        "c4b_1": "• reserved 超限自动 empty_cache()",
        "c4b_2": "• 避免 Windows WDDM 内存换出",
        "c4b_3": "Mg-MOF-74 0.05 bar: 355.8s → 90.3s (3.9×)",
        "l5_hdr": "05 · 公平对照基准、回归测试与独立经典引擎",
        "c5a_t": "对照基准 · direct + benchmark_speedup",
        "c5a_1": "• Serial ASE & 硬球(1.20 Å)批量 GPU 基线",
        "c5a_2": "同卡 SJD_N/s: Mg-MOF-74 31.5× · IRMOF-1 7.8×",
        "c5b_t": "回归套件 · pyproject.toml + tests/",
        "c5b_1": "• 14 项 pytest (103 元素 / QEq / 多客体 / Direct)",
        "c5b_2": "pytest 14 passed + verify_cli.py vs RASPA3",
        "c5c_t": "独立经典引擎 · mof_gcmc_cli.exe",
        "c5c_1": "• 与 DLL 共享 cell / forcefield / grid3d 源码",
        "c5c_2": "12 盒百万步 ~65s · 等温线 / 烟道气 / SCF 极化",
        "p1": "① K步提议",
        "p2": "② 候选构型",
        "p3": "③ 返回 U_MLIP → 外层精确 Metropolis 判据 (ΔΔU = ΔU_MLIP − ΔU_ref − Δε·ΔN)",
        "p4": "④ commit",
    },
    "en": {
        "fonts": ["Segoe UI", "DejaVu Sans", "Arial"],
        "title": "cuda-mof-gcmc System Architecture",
        "badge_x": 31.5,
        "subtitle": "GPU Classical Inner Proposer + Unbiased Outer MLIP Reweighting (Zero-I/O In-Memory Loop)",
        "l1_hdr": "01 · Input & Preprocessing Layer (Python / ASE)",
        "c1a_t": "qeq_charges.py · MOF & Charges",
        "c1a_1": "• ASE expands all 230 space groups (CIF / REPEAT)",
        "c1a_2": "• Screened periodic QEq fallback (MAE 0.11–0.15 e)",
        "c1b_t": "guests.py · Rigid Multi-Site Guests",
        "c1b_1": "• 8 presets (CO2/CH4/N2/H2/Ar/H2O/NH3) + JSON",
        "c1b_2": "• Massless Z=0 dummy sites & PR critical params",
        "c1c_t": "Planned Extensions · Multi-MLIP & 8-GPU",
        "c1c_1": "• MatterSim / SevenNet / ORB / FAIRChem UMA",
        "c1c_2": "• Linux .so build + multi-GPU pressure dispatch",
        "l2_hdr": "02 · Outer Orchestrator & Nested MC",
        "c2a_t": "run_mlip_gcmc.py · CLI Runner",
        "c2a_1": "• Adaptive 3D grid dims & chunk sizing",
        "c2a_2": "• 3-stage chemical-potential shift Δε",
        "c2a_3": "• Atomic checkpoint --resume & checks",
        "c2a_4": "Exports JSON / CSV isotherms & ESS",
        "c2b_t": "nested_mc_engine.py · Core Loop",
        "c2b_1": "• Exact Metropolis accept / rollback",
        "c2b_2": "• ΔΔU = ΔU_MLIP − ΔU_ref − Δε·ΔN",
        "c2b_3": "• Pooled-Jackknife fluctuation Qst",
        "c2b_4": "• FFT τ_int, ESS/GPU-s & SJD_N/GPU-s",
        "c2b_pill": "Python <-> CUDA ctypes In-Memory Bridge",
        "l3_hdr": "03 · Inner GPU Proposer (CUDA DLL)",
        "c3a_t": "mof_gcmc_proposer.dll",
        "c3a_1": "proposer_api.h / proposer_dll.cu",
        "c3a_2": "In-memory grid setup · multi-GPU device_id",
        "c3b_t": "Universal Physics & 3D Tricubic Grids",
        "c3b_1": "• cell.hpp: general triclinic 1/9/27 MIC",
        "c3b_2": "• forcefield.hpp: 103-elem UFF + PR EOS",
        "c3b_3": "• unified_grid_builder.cu: LJ / Phi / E",
        "c3c_t": "GPU Burst Kernel (Quaternion)",
        "c3c_1": "• N_boxes parallel K-step rigid GCMC",
        "c3c_2": "• <=12 sites / 6 LJ · full U recompute",
        "c3c_3": "~1.8 ms / burst (4 boxes × K steps)",
        "l4_hdr": "04 · Outer MLIP Target (PyTorch CUDA)",
        "c4a_t": "FastMaceD3Target · Batched Eval",
        "c4a_1": "• Quaternion-rotated real atoms (Z > 0)",
        "c4a_2": "• Chunked no_grad MACE inference",
        "c4a_3": "• GPU DFT-D3(BJ) cached fw-fw edges",
        "c4a_4": "• < 0.1% FP32 parity vs. Serial ASE",
        "c4a_pill": "4–5× faster than serial ASE inference",
        "c4b_t": "VRAM Fragmentation Guard",
        "c4b_1": "• Auto empty_cache() on reserved bloat",
        "c4b_2": "• Prevents WDDM host spillover slowdown",
        "c4b_3": "Mg-MOF-74 0.05 bar: 355.8s -> 90.3s (3.9×)",
        "l5_hdr": "05 · Direct Baselines, Pytest Suite & Classical CLI",
        "c5a_t": "Benchmarks · direct + benchmark_speedup",
        "c5a_1": "• Serial ASE & HS-screened Batched GPU Direct",
        "c5a_2": "Same-GPU SJD_N/s: Mg-MOF-74 31.5× · IRMOF-1 7.8×",
        "c5b_t": "Regression · pyproject.toml + tests/",
        "c5b_1": "• 14 pytest checks (103-elem / QEq / guests / parity)",
        "c5b_2": "pytest 14 passed + verify_cli.py vs RASPA3",
        "c5c_t": "Classical Engine · mof_gcmc_cli.exe",
        "c5c_1": "• Shares cell / forcefield / grid3d C++ headers",
        "c5c_2": "12 boxes × 1M steps ~65s · Flue gas & Pol-SCF",
        "p1": "1. K-step burst",
        "p2": "2. Trial batch",
        "p3": "3. Return U_MLIP -> Exact Outer Metropolis (ΔΔU = ΔU_MLIP − ΔU_ref − Δε·ΔN)",
        "p4": "4. commit",
    },
}


def render_diagram(lang: str, out_path: Path):
    T = TEXTS[lang]
    plt.rcParams.update({
        "font.sans-serif": T["fonts"],
        "font.monospace": ["Consolas", "DejaVu Sans Mono"],
        "axes.unicode_minus": False,
    })

    fig = plt.figure(figsize=(15.5, 9.6), dpi=300, facecolor="#FFFFFF")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor="#FFFFFF")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    # ==================== HEADER ====================
    ax.text(2.0, 59.7, T["title"], fontsize=16.5, fontweight="bold", color=TEXT_MAIN, va="center")
    ax.text(T["badge_x"], 59.65, "v0.2", fontsize=9.5, fontweight="bold", color=ACCENT, va="center",
            bbox=dict(boxstyle="round,pad=0.28", fc=ACCENT_BG, ec=ACCENT_BORDER, lw=1.0))
    ax.text(2.0, 57.6, T["subtitle"], fontsize=10.2, color=TEXT_MUTED, va="center")

    # ==================== LAYER 01 ====================
    add_round_box(ax, 1.8, 47.0, 96.4, 9.4, fc=ZONE_BG, ec=ZONE_BORDER, lw=1.1, r=1.1, zorder=1)
    ax.text(3.2, 55.0, T["l1_hdr"], fontsize=10.2, fontweight="bold", color=ACCENT, va="center")

    add_round_box(ax, 3.2, 47.9, 30.2, 5.9, fc=CARD_BG, ec="#94A3B8", lw=1.1, r=0.8, zorder=2, shadow=True)
    ax.text(4.4, 52.3, T["c1a_t"], fontsize=10.0, fontweight="bold", color=TEXT_MAIN, va="center")
    ax.text(4.4, 50.4, T["c1a_1"], fontsize=8.4, color=TEXT_MUTED, va="center")
    ax.text(4.4, 48.9, T["c1a_2"], fontsize=8.4, color=TEXT_MUTED, va="center")

    add_round_box(ax, 35.0, 47.9, 30.2, 5.9, fc=CARD_BG, ec="#94A3B8", lw=1.1, r=0.8, zorder=2, shadow=True)
    ax.text(36.2, 52.3, T["c1b_t"], fontsize=10.0, fontweight="bold", color=TEXT_MAIN, va="center")
    ax.text(36.2, 50.4, T["c1b_1"], fontsize=8.4, color=TEXT_MUTED, va="center")
    ax.text(36.2, 48.9, T["c1b_2"], fontsize=8.4, color=TEXT_MUTED, va="center")

    add_round_box(ax, 66.8, 47.9, 30.0, 5.9, fc=ZONE_BG, ec="#94A3B8", lw=1.1, ls="--", r=0.8, zorder=2)
    ax.text(68.0, 52.3, T["c1c_t"], fontsize=10.0, fontweight="bold", color=TEXT_SUB, va="center")
    ax.text(68.0, 50.4, T["c1c_1"], fontsize=8.4, color=TEXT_MUTED, va="center")
    ax.text(68.0, 48.9, T["c1c_2"], fontsize=8.4, color=TEXT_MUTED, va="center")

    # ==================== MIDDLE TIER (02, 03, 04) ====================
    # Zone 02: Outer Control & Nested MC
    add_round_box(ax, 1.8, 17.6, 28.4, 26.8, fc=ZONE_BG, ec=ZONE_BORDER, lw=1.1, r=1.1, zorder=1)
    ax.text(3.2, 42.9, T["l2_hdr"], fontsize=10.2, fontweight="bold", color=ACCENT, va="center")

    add_round_box(ax, 3.2, 32.4, 25.6, 9.2, fc=CARD_BG, ec="#94A3B8", lw=1.1, r=0.8, zorder=2, shadow=True)
    ax.text(4.3, 40.1, T["c2a_t"], fontsize=9.8, fontweight="bold", color=TEXT_MAIN, va="center")
    ax.text(4.3, 38.2, T["c2a_1"], fontsize=8.4, color=TEXT_MUTED, va="center")
    ax.text(4.3, 36.6, T["c2a_2"], fontsize=8.4, color=TEXT_MUTED, va="center")
    ax.text(4.3, 35.0, T["c2a_3"], fontsize=8.4, color=TEXT_MUTED, va="center")
    ax.text(4.3, 33.4, T["c2a_4"], fontsize=8.2, fontweight="bold", color=ACCENT, va="center")

    add_round_box(ax, 3.2, 18.8, 25.6, 12.2, fc=CARD_BG, ec=ACCENT, lw=1.5, r=0.8, zorder=2, shadow=True)
    ax.text(4.3, 29.5, T["c2b_t"], fontsize=9.8, fontweight="bold", color=TEXT_MAIN, va="center")
    ax.text(4.3, 27.6, T["c2b_1"], fontsize=8.4, color=TEXT_MUTED, va="center")
    ax.text(4.3, 26.0, T["c2b_2"], fontsize=8.2, color=TEXT_SUB, va="center")
    ax.text(4.3, 24.4, T["c2b_3"], fontsize=8.4, color=TEXT_MUTED, va="center")
    ax.text(4.3, 22.8, T["c2b_4"], fontsize=8.4, color=TEXT_MUTED, va="center")
    add_round_box(ax, 4.3, 19.6, 23.4, 2.1, fc=ACCENT_BG, ec=ACCENT_BORDER, lw=0.9, r=0.4, zorder=3)
    ax.text(16.0, 20.65, T["c2b_pill"], fontsize=8.1, fontweight="bold", color=ACCENT, ha="center", va="center", zorder=4)

    # Zone 03: Inner GPU Proposer (CUDA DLL)
    add_round_box(ax, 35.6, 17.6, 28.8, 26.8, fc=ZONE_BG, ec=ZONE_BORDER, lw=1.1, r=1.1, zorder=1)
    ax.text(37.0, 42.9, T["l3_hdr"], fontsize=10.2, fontweight="bold", color=ACCENT, va="center")

    add_round_box(ax, 37.0, 35.8, 26.0, 5.8, fc=CARD_BG, ec="#94A3B8", lw=1.1, r=0.8, zorder=2, shadow=True)
    ax.text(38.1, 40.1, T["c3a_t"], fontsize=9.8, fontweight="bold", color=TEXT_MAIN, va="center")
    ax.text(38.1, 38.4, T["c3a_1"], fontsize=8.3, color=TEXT_MUTED, va="center")
    ax.text(38.1, 36.8, T["c3a_2"], fontsize=8.2, fontweight="bold", color=ACCENT, va="center")

    add_round_box(ax, 37.0, 27.2, 26.0, 7.6, fc=CARD_BG, ec="#94A3B8", lw=1.1, r=0.8, zorder=2, shadow=True)
    ax.text(38.1, 33.4, T["c3b_t"], fontsize=9.6, fontweight="bold", color=TEXT_MAIN, va="center")
    ax.text(38.1, 31.6, T["c3b_1"], fontsize=8.3, color=TEXT_MUTED, va="center")
    ax.text(38.1, 30.0, T["c3b_2"], fontsize=8.3, color=TEXT_MUTED, va="center")
    ax.text(38.1, 28.4, T["c3b_3"], fontsize=8.3, color=TEXT_MUTED, va="center")

    add_round_box(ax, 37.0, 18.8, 26.0, 7.4, fc=CARD_BG, ec=ACCENT, lw=1.5, r=0.8, zorder=2, shadow=True)
    ax.text(38.1, 24.8, T["c3c_t"], fontsize=9.7, fontweight="bold", color=TEXT_MAIN, va="center")
    ax.text(38.1, 23.0, T["c3c_1"], fontsize=8.3, color=TEXT_MUTED, va="center")
    ax.text(38.1, 21.4, T["c3c_2"], fontsize=8.3, color=TEXT_MUTED, va="center")
    ax.text(38.1, 19.7, T["c3c_3"], fontsize=8.1, fontweight="bold", color=ACCENT, va="center")

    # Zone 04: Outer MLIP Target
    add_round_box(ax, 69.8, 17.6, 28.4, 26.8, fc=ZONE_BG, ec=ZONE_BORDER, lw=1.1, r=1.1, zorder=1)
    ax.text(71.2, 42.9, T["l4_hdr"], fontsize=10.2, fontweight="bold", color=ACCENT, va="center")

    add_round_box(ax, 71.2, 29.0, 25.6, 12.6, fc=CARD_BG, ec=ACCENT, lw=1.5, r=0.8, zorder=2, shadow=True)
    ax.text(72.3, 40.0, T["c4a_t"], fontsize=9.8, fontweight="bold", color=TEXT_MAIN, va="center")
    ax.text(72.3, 37.9, T["c4a_1"], fontsize=8.4, color=TEXT_MUTED, va="center")
    ax.text(72.3, 36.1, T["c4a_2"], fontsize=8.4, color=TEXT_MUTED, va="center")
    ax.text(72.3, 34.3, T["c4a_3"], fontsize=8.4, color=TEXT_MUTED, va="center")
    ax.text(72.3, 32.5, T["c4a_4"], fontsize=8.4, color=TEXT_MUTED, va="center")
    add_round_box(ax, 72.3, 29.7, 23.4, 2.0, fc=ACCENT_BG, ec=ACCENT_BORDER, lw=0.9, r=0.4, zorder=3)
    ax.text(84.0, 30.7, T["c4a_pill"], fontsize=8.1, fontweight="bold", color=ACCENT, ha="center", va="center", zorder=4)

    add_round_box(ax, 71.2, 18.8, 25.6, 8.8, fc=CARD_BG, ec="#94A3B8", lw=1.1, r=0.8, zorder=2, shadow=True)
    ax.text(72.3, 26.0, T["c4b_t"], fontsize=9.7, fontweight="bold", color=TEXT_MAIN, va="center")
    ax.text(72.3, 24.0, T["c4b_1"], fontsize=8.4, color=TEXT_MUTED, va="center")
    ax.text(72.3, 22.3, T["c4b_2"], fontsize=8.4, color=TEXT_MUTED, va="center")
    ax.text(72.3, 20.1, T["c4b_3"], fontsize=8.0, fontweight="bold", color=ACCENT, va="center")

    # ==================== LAYER 05 ====================
    add_round_box(ax, 1.8, 1.6, 96.4, 10.6, fc=ZONE_BG, ec=ZONE_BORDER, lw=1.1, r=1.1, zorder=1)
    ax.text(3.2, 10.7, T["l5_hdr"], fontsize=10.2, fontweight="bold", color=ACCENT, va="center")

    add_round_box(ax, 3.2, 2.5, 30.2, 6.8, fc=CARD_BG, ec="#94A3B8", lw=1.1, r=0.8, zorder=2, shadow=True)
    ax.text(4.4, 7.8, T["c5a_t"], fontsize=9.5, fontweight="bold", color=TEXT_MAIN, va="center")
    ax.text(4.4, 5.8, T["c5a_1"], fontsize=8.3, color=TEXT_MUTED, va="center")
    ax.text(4.4, 4.0, T["c5a_2"], fontsize=8.1, fontweight="bold", color=ACCENT, va="center")

    add_round_box(ax, 35.0, 2.5, 30.2, 6.8, fc=CARD_BG, ec="#94A3B8", lw=1.1, r=0.8, zorder=2, shadow=True)
    ax.text(36.2, 7.8, T["c5b_t"], fontsize=9.5, fontweight="bold", color=TEXT_MAIN, va="center")
    ax.text(36.2, 5.8, T["c5b_1"], fontsize=8.3, color=TEXT_MUTED, va="center")
    ax.text(36.2, 4.0, T["c5b_2"], fontsize=8.1, fontweight="bold", color=ACCENT, va="center")

    add_round_box(ax, 66.8, 2.5, 30.0, 6.8, fc=CARD_BG, ec="#94A3B8", lw=1.1, r=0.8, zorder=2, shadow=True)
    ax.text(68.0, 7.8, T["c5c_t"], fontsize=9.5, fontweight="bold", color=TEXT_MAIN, va="center")
    ax.text(68.0, 5.8, T["c5c_1"], fontsize=8.3, color=TEXT_MUTED, va="center")
    ax.text(68.0, 4.0, T["c5c_2"], fontsize=8.1, fontweight="bold", color=ACCENT, va="center")

    # ==================== ARROWS & LOOP LABELS ====================
    add_arrow(ax, (16.0, 47.9), (16.0, 41.6), color="#64748B", lw=1.5)
    add_arrow(ax, (46.0, 47.9), (26.0, 41.6), color="#64748B", lw=1.5, rad=0.10)

    add_arrow(ax, (16.0, 32.4), (16.0, 31.0), color=ACCENT, lw=1.6, ms=10)
    add_arrow(ax, (50.0, 35.8), (50.0, 34.8), color="#64748B", lw=1.4, ms=10)
    add_arrow(ax, (50.0, 27.2), (50.0, 26.2), color="#64748B", lw=1.4, ms=10)
    add_arrow(ax, (84.0, 29.0), (84.0, 27.6), color="#64748B", lw=1.4, ms=10)

    add_arrow(ax, (28.8, 24.8), (37.0, 24.8), color=ACCENT, lw=1.9)
    add_pill(ax, 32.9, 26.8, T["p1"], fc=CARD_BG, ec=ACCENT, tc=ACCENT, fs=8.2)

    add_arrow(ax, (28.8, 20.8), (37.0, 20.8), color=ACCENT, lw=1.9)
    add_pill(ax, 32.9, 18.8, T["p4"], fc=CARD_BG, ec=ACCENT, tc=ACCENT, fs=8.2)

    add_arrow(ax, (63.0, 22.8), (71.2, 35.0), color=ACCENT, lw=1.9, rad=-0.22)
    add_pill(ax, 67.1, 29.2, T["p2"], fc=CARD_BG, ec=ACCENT, tc=ACCENT, fs=8.2)

    bus_x = [84.0, 84.0, 16.0]
    bus_y = [18.8, 14.9, 14.9]
    ax.plot(bus_x, bus_y, color=ACCENT, linewidth=1.9, zorder=5)
    add_arrow(ax, (16.0, 14.9), (16.0, 18.8), color=ACCENT, lw=1.9, ms=12)
    add_pill(ax, 50.0, 14.9, T["p3"], fc=CARD_BG, ec=ACCENT, tc=ACCENT, fs=8.5)

    add_arrow(ax, (9.0, 18.8), (9.0, 9.3), color="#94A3B8", lw=1.2, ls="--", ms=10)
    add_arrow(ax, (60.0, 18.8), (76.0, 9.3), color="#94A3B8", lw=1.2, ls="--", rad=-0.08, ms=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    render_diagram("en", OUT_EN)
    render_diagram("zh", OUT_ZH)


if __name__ == "__main__":
    main()
