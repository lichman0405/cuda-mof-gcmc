[English](README.md) | **简体中文**

# `cuda-mof-gcmc`

![CUDA C++](https://img.shields.io/badge/CUDA-C%2B%2B17-76B900?style=flat-square&logo=nvidia)
![Target GPU](https://img.shields.io/badge/RTX%205070-sm__120-2563EB?style=flat-square)
![RASPA3 Verified](https://img.shields.io/badge/RASPA3%203.0.29-%E2%89%A4%200.54%CF%83-0F766E?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-64748B?style=flat-square)

面向 **NVIDIA RTX 5070**（`sm_120`）的纯 CUDA/C++17 金属有机框架（MOF）刚性吸附巨正则蒙特卡洛（GCMC）引擎。项目将经 `RASPA3` 对标的 $O(h^4)$ 三维 Tricubic 经典/极化 GPU 内核（`598,669 MC-steps/s`）与带化学势偏置的零 I/O 动态库相结合，用于驱动大加速比的机器学习势（`MACE + TorchDFTD3`）嵌套蒙特卡洛模拟。

![Step 4 Mg-MOF-74 Tricubic Validation](assets/step4_validated_mercado_raspa3.png)

---

## 核心特性与已验证基准

- **与原生 `RASPA3 (3.0.29)` 误差内逐点对齐（`≤ 0.54σ`）**：在 `Mg-MOF-74`（`298 K`，UFF+REPEAT）基准上，`0.10 Å` $O(h^4)$ 三维 Tricubic GPU 内核在 `0.15 / 1.0 / 5.0 bar` 与单核 RASPA3 网格版偏差仅 **`+2.32% / +1.39% / -0.41%`**（$\Delta Q_{\text{st}} \le 0.54\ \text{kJ/mol}$），全双精度状态叠加每 `50k` 步重算将 `6,400 万` 步最大能量漂移控制在 **`≤ 1.02e-08 K`**。
- **高吞吐 CUDA 内核**：实测吞吐 **`598,669 MC-steps/s`**（较单核 RASPA3 网格版 `3,908/s` 提速 **`153.2×`**，较精确 Ewald `1,215/s` 提速 **`492.6×`**）；`256×256×72` 三维势能网格构建仅需 `3.03 s`（`38.9×`）。
- **各向异性与全自洽多体极化**：支持框架场 `Pol-COM` / `Pol-3Site` 以及 Block 级共享内存 Applequist-Thole 自洽偶极求解器 `Pol-SCF`（`--test-scf` 与 Python `36×36` 精确矩阵求逆对齐至 **`1.83e-08 K`** 与 **`1.13e-11 e·Å`**）。
- **统一 JSON/CLI 求解器（[`src/main_cli.cu`](src/main_cli.cu)）**：单一可执行文件集成通用 CIF 展开（支持 `19` 种 MOF 元素与可选 `_atom_site_charge` 列）、`UFF` / `Mercado 2016` DFT 力场、二元 `CO₂/N₂` 与 `CO₂/CH₄` Identity Swap 及 Peng–Robinson 混合逸度。
- **零 I/O 嵌套 MLIP-GCMC 引擎（[`scripts/run_mlip_gcmc.py`](scripts/run_mlip_gcmc.py)）**：
  - CUDA 提议器动态库（[`proposer_set_chem_pot_shift`](include/proposer_api.h)）与核内能量一致至 **`0.000e+00 K`**，并通过 `Pol-3Site -> Mercado` 零偏差统计校验（`z = 0.27σ`）。
  - 批处理 `FastMaceD3Target` 较原生 ASE 实现 **`4.21×`** 加速（`157.7 → 37.5 ms/box`，能量误差 `≤ 0.0082 kJ/mol/CO2`），且在 `1×1×2` 与 `1×1×4` 超胞间保持有限尺寸一致性（经典 `z = 0.04σ`、MLIP `z = 0.63σ`）。

> **适用范围与当前局限**：经典/极化求解器针对**刚性 MOF 骨架与刚性小分子**（`CO₂`、`N₂`、`CH₄`，暂不支持柔性长链 CBMC 或骨架呼吸）。`Pol-SCF` 每步需在 FP64 下求解 $O(N^2)$ 自洽迭代且基于常规 TraPPE/UFF 参数。MLIP 嵌套采样仍处于**科研迭代阶段（WIP）**：物理预测精度强依赖所选通用机器学习势、色散搭配、DFT 参考标定与采样链长（详见 [MLIP 技术记录与已知问题](docs/MLIP_NESTED_MC_NOTES.zh-CN.md)）。

---

## 快速开始

### 环境要求
- Windows 10/11 + NVIDIA CUDA Toolkit（`sm_120`, RTX 5070）+ MSVC（Visual Studio 2022 生成工具）
- 可选（用于 MLIP Nested MC 与自动校验绘图）：Python 3.11+（PyTorch CUDA、`mace-torch`、`torch-dftd`、`ase`、`numpy`、`matplotlib`）

### 编译与运行

```powershell
# 全量编译全部 5 个目标（mof_gcmc, mof_gcmc_mix, mof_gcmc_v2, mof_gcmc_cli, mof_gcmc_proposer.dll）
cmd /c build.bat all

# 经典与极化统一 CLI 预设
.\build\mof_gcmc_cli.exe --test-scf
.\build\mof_gcmc_cli.exe --config configs\mg_mof74_co2_298K_uff.json
.\build\mof_gcmc_cli.exe --config configs\mg_mof74_fluegas_298K_pol3site.json
.\build\mof_gcmc_cli.exe --config configs\mg_mof74_co2_298K_scf.json
uv run --with numpy python scripts\verify_cli.py

# GPU-GCMC + MLIP 嵌套蒙特卡洛统一脚本（科研开发中）
.venv-mlip\Scripts\python.exe scripts\verify_nested_mc.py
.venv-mlip\Scripts\python.exe -u scripts\run_mlip_gcmc.py `
  --cif data\MgMOF-74.cif --pressures 0.01 0.05 0.15 `
  --proposer mercado --model mace-matpes-r2scan-0 --dispersion on --xc r2scan
```

---

## 详细文档

- [**经典与极化 CUDA 引擎基准校验（第一至五步）**](docs/CLASSICAL_VALIDATION.zh-CN.md) — 晶体几何校验、Tricubic 网格精度、RASPA3 双重对标表、二元选择性与 Level-2 `Pol-SCF` 核验。
- [**MLIP 嵌套蒙特卡洛：已验证机制与 WIP 研究记录**](docs/MLIP_NESTED_MC_NOTES.zh-CN.md) — 零偏差重加权证明、化学势偏置机制、MACE 批处理加速、基础模型短程过结合诊断与待解决物理问题。

---

## 目录结构

```text
cuda-mof-gcmc/
├── configs/   # mof_gcmc_cli.exe 预设配置（UFF、Mercado 2016、313K PSA、烟道气、Pol-SCF）
├── data/      # 晶体结构 CIF（Mg-MOF-74、Ni-MOF-74、Cu-BTC）与 Mercado 2016 参数文件
├── docs/      # 详细基准测试文档、公式推导与科研进展/待办记录（中/英文）
├── include/   # C++17 / CUDA 头文件：最小镜像、力场参数、三维网格、CIF 解析与提议器 C ABI
├── src/       # CUDA 内核与驱动源码（mof_gcmc、mix、v2、cli、mof_gcmc_proposer.dll）
├── scripts/   # Python 自动校验套件、RASPA3 对标脚本与 MLIP Nested MC 引擎
└── assets/    # 已生成的基准对标与验证图（300 DPI）
```
