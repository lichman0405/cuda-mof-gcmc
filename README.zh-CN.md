[English](README.md) | **简体中文**

# `cuda-mof-gcmc`

![CUDA C++](https://img.shields.io/badge/CUDA-C%2B%2B17-76B900?style=flat-square&logo=nvidia)
![Multi-GPU Verified](https://img.shields.io/badge/Multi--GPU-sm__86%20%2F%20sm__120-2563EB?style=flat-square)
![RASPA3 Verified](https://img.shields.io/badge/RASPA3%203.0.29-%E2%89%A4%200.54%CF%83-0F766E?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-64748B?style=flat-square)

面向 **Windows（RTX 5070 `sm_120`）与多卡 Linux（RTX 3090 `sm_86`）** 的纯 CUDA/C++17 **任意刚性 MOF** 气体吸附 GCMC 引擎。项目包含两部分能力：**（1）经过 `RASPA3` 对标验证的 GPU 经典及多体全自洽极化（`Pol-SCF`）求解器**；**（2）将通用 3D 四元数 CUDA 提议器封装为零磁盘 I/O 动态库（[`mof_gcmc_proposer`](include/proposer_api.h)），与批处理机器学习势（`MACE + TorchDFTD3`）直连并支持多卡并行（`--gpus`）的嵌套蒙特卡洛（Nested MC）引擎**。

> **架构重构进行中**：新一代 **1B+2B 在线自蒸馏 MLIP + CUDA 块并行 GPU-CBMC 嵌套蒙特卡洛** 方案与阶段门禁见 [**`PLAN.zh-CN.md`**](PLAN.zh-CN.md)。旧版未完全收敛的 MLIP 报告与图表已清理，待新引擎通过全部严格 $\hat{R} / \text{ESS} / \text{MCSE}$ 收敛门禁后重编。

---

## 核心架构模块

1. **通用晶体与多位点刚体客体前处理**（[`scripts/qeq_charges.py`](scripts/qeq_charges.py)、[`scripts/guests.py`](scripts/guests.py)）：由 ASE 展开全部 `230` 个空间群，利用 `spacegroup_kinds` 保留 CIF 电荷与原子标签，并对无电荷骨架自动计算带介电屏蔽（`eps_eff = 1.50`）的周期性 Ewald QEq；客体支持最多 `12` 个位点（含 `Z = 0` 无质量虚位点）、四元数三维刚体旋转及通用 Peng–Robinson 状态方程逸度。
2. **零 I/O CUDA 提议器动态库**（[`src/proposer_dll.cu`](src/proposer_dll.cu)、[`include/cell.hpp`](include/cell.hpp)、[`include/grid3d.hpp`](include/grid3d.hpp)）：在 GPU 显存驻留三维 $O(h^4)$ Tricubic 网格与共享内存客体对势，直接通过 `ctypes` 指针推进内层可逆 GCMC 并将无重叠终点构型与精确 `U_ref` 回传外层重加权。
3. **批处理 MLIP 打分、严格收敛诊断与多卡调度**（[`scripts/nested_mc_engine.py`](scripts/nested_mc_engine.py)、[`scripts/mcmc_diagnostics.py`](scripts/mcmc_diagnostics.py)、[`scripts/multi_gpu.py`](scripts/multi_gpu.py)）：`FastMaceD3Target` 负责无梯度微批 MACE + 缓存静态骨架边表的 GPU DFT-D3 计算；集成秩归一化 $\hat{R}$、bulk/tail ESS、独立涨落 $Q_{\text{st}}$、SHA-256 元数据指纹断点续跑（[`scripts/run_metadata.py`](scripts/run_metadata.py)）以及多卡 `pressure` / `replica` 并行调度。

---

## 与 `RASPA3` / `gRASPA` 的差异定位

| 对比维度 | CPU `RASPA3` / GPU `gRASPA` | `cuda-mof-gcmc` |
| :--- | :--- | :--- |
| **核心定位** | 面向成熟经典力场的完整模拟生态（力场齐全、支持柔性长链 CBMC 与大规模高通量筛选） | **GPU 原生多体全自洽极化（`Pol-SCF`）** + **耦合机器学习势的零 I/O Nested MC 架构** |
| **极化物理** | 主打固定点电荷模型（LJ + 静电） | 支持三维 Tricubic 电场骨架极化（`Pol-3Site`）及每步 Block 级 FP64 Applequist–Thole **多体自洽 `Pol-SCF`** |
| **MLIP 嵌套支持** | 无针对 PyTorch MLIP 重加权的零 I/O CUDA 提议器动态库 | 原生 C ABI（[`proposer_api.h`](include/proposer_api.h)）直连 `FastMaceD3Target`、化学势偏置标定、$\hat{R}/\text{ESS}$ 诊断与断点续跑 |
| **选型边界** | 若仅做标准经典力场筛选建议直接使用 `RASPA3` / `gRASPA` | 聚焦任意刚性 MOF 与刚性小分子（`CO₂`、`CH₄`、`N₂`、`H₂`、`Ar`、`H₂O`、`NH₃`） |

---

## 已验证的经典与 `Pol-SCF` 基准

![Step 4 Mg-MOF-74 Tricubic Validation](assets/step4_validated_mercado_raspa3.png)

- **`RASPA3 3.0.29` 对齐精度（`Mg-MOF-74`, `298 K`, UFF+REPEAT）**：与单核 `RASPA3` Tricubic 网格版在 `0.15 / 1.0 / 5.0 bar` 偏差仅 **`+2.32% / +1.39% / -0.41%`**（$\Delta Q_{\text{st}} \le 0.54\ \text{kJ/mol}$），全 FP64 重算使 `6,400 万` 步最大能量漂移 `≤ 1.02e-08 K`。
- **经典内核吞吐**：实测 **`598,669 MC-steps/s`**（较单核 CPU `RASPA3` Tricubic 网格 `3,908/s` 快 **`153.2×`**，较精确 Ewald `1,215/s` 快 **`492.6×`**）；`256×256×72` 网格构建耗时 `3.03 s`（`38.9×`）。
- **`Pol-SCF` 核验**：`--test-scf` 与 Python `36×36` 精确矩阵求逆偏差仅 **`1.83e-08 K`** 与 **`1.13e-11 e·Å`**。
- **物理边界**：本引擎假设**刚性 MOF 骨架与刚性多位点客体分子**（`≤ 12` 位点、`≤ 6` 种 LJ 类型，不支持柔性长链 CBMC 与骨架呼吸）。

---

## 快速开始（Windows 与 Linux）

### 一键检测、缺啥装啥与编译自检（`fast_start.bat` / `fast_start.sh`）

- **Windows**：在仓库根目录执行 [`fast_start.bat`](fast_start.bat)（内部调用 [`scripts/fast_start_win.ps1`](scripts/fast_start_win.ps1)），自动检测 NVIDIA 显卡架构（`sm_XX`）、CUDA Toolkit（`nvcc`）、MSVC 编译环境（VS 2022/2019）、Python 3.10+、CUDA PyTorch 及依赖库，随后编译 [`build.bat`](build.bat) 并运行自检：

```powershell
# 仅检查当前电脑缺哪些依赖，不执行安装和编译
cmd /c fast_start.bat -CheckOnly

# 自动补齐缺失依赖、编译核心 DLL/EXE，并运行 Pol-SCF + 非 MLIP pytest 自检
cmd /c fast_start.bat
```

- **Linux**：在仓库根目录执行 [`fast_start.sh`](fast_start.sh)（优先使用 `uv`，未装时自动回退 `python3 -m venv`；多卡共享服务器可用 `--gpu <id>` 指定空闲卡）：

```bash
# 仅检查环境与依赖状态
./fast_start.sh --check-only

# 自动创建 .venv-mlip、编译核心目标并运行基础自检
./fast_start.sh --gpu 0
```

### 手动编译与基础校验

```powershell
# Windows 编译核心目标（mof_gcmc_cli.exe + mof_gcmc_proposer.dll）
cmd /c build.bat

# Linux 编译核心目标（mof_gcmc_cli + libmof_gcmc_proposer.so）
./build.sh
```

```powershell
# 经典与极化统一 CLI 预设及回归测试（Linux 下把 `.\build\mof_gcmc_cli.exe` 替换为 `./build/mof_gcmc_cli`）
.\build\mof_gcmc_cli.exe --test-scf
.\build\mof_gcmc_cli.exe --config configs\mg_mof74_co2_298K_uff.json
.\build\mof_gcmc_cli.exe --config configs\mg_mof74_fluegas_298K_pol3site.json
.\.venv-mlip\Scripts\python.exe -u scripts\verify_cli.py
.\.venv-mlip\Scripts\python.exe -u scripts\verify_scf_pol.py
.\.venv-mlip\Scripts\python.exe -m pytest -v -m "not mlip"
```

### Nested MLIP-GCMC CLI 入口（`run_mlip_gcmc.py`）

```powershell
.\.venv-mlip\Scripts\python.exe -u scripts\run_mlip_gcmc.py `
  --cif data\IRMOF-1.cif --supercell 1 1 1 --temp 298 --pressures 0.5 1.0 5.0 `
  --guest ch4 --proposer uff --charge-mode auto `
  --model medium --dispersion on --xc pbe `
  --boxes 4 --k-steps 25 --eq-cycles 120 --prod-cycles 320 `
  --out-json build\irmof1_ch4.json --out-csv build\irmof1_ch4.csv
```

- 多卡运行可附加 `--gpus 0 1 2 3 --parallel-mode auto`（在 `--checkpoint-dir` 与 `--resume-mode {none,skip,extend,fresh}` 支持下可严格校验元数据并位精确断点续跑）。

---

## 文档与目录结构

- [**架构重构与验证方案**](PLAN.zh-CN.md) — 1B+2B 在线自蒸馏 MLIP + CUDA 块并行 GPU-CBMC Nested MC 设计方案与阶段门禁。
- [**经典与极化 CUDA 引擎基准校验（第一至五步）**](docs/CLASSICAL_VALIDATION.zh-CN.md) — 晶体几何校验、Tricubic 精度、`RASPA3` 对标表、共吸附选择性与 `Pol-SCF` 核验。

```text
cuda-mof-gcmc/
├── PLAN.zh-CN.md   # 当前主线方案：1B+2B 在线自蒸馏 MLIP + GPU-CBMC Nested MC 设计与门禁
├── fast_start.bat  # Windows 一键检测、补齐依赖、编译与自检入口
├── fast_start.sh   # Linux 一键检测（uv/venv）、补齐依赖、编译与自检入口
├── build.bat       # Windows MSVC + CUDA 编译脚本
├── build.sh        # Linux nvcc + g++ 编译脚本
├── pyproject.toml  # 项目打包配置与 pytest 标记（mlip, slow）
├── configs/        # mof_gcmc_cli 预设配置（UFF、Mercado 2016、313K PSA、烟道气、Pol-SCF）
├── data/           # 晶体 CIF（Mg-MOF-74、Ni-MOF-74、Cu-BTC、IRMOF-1）、Mercado 力场与 NIST ISODB 实验表
├── docs/           # 经典与 Pol-SCF 基准校验文档（CLASSICAL_VALIDATION*.md）
├── include/        # C++17 / CUDA 头文件：三斜 MIC、103 元素 UFF + PR、三维网格与提议器 C ABI
├── src/            # 现役 CUDA 核与驱动源码（mof_gcmc_cli、mof_gcmc_proposer.dll / .so）+ src/legacy/
├── scripts/        # 核心引擎（run_mlip_gcmc、nested_mc_engine、multi_gpu、mcmc_diagnostics、qeq_charges、guests 等）
├── tests/          # Pytest 单元测试与零偏差回归套件
└── assets/         # 经典步骤 1–5 基准对比图
```
