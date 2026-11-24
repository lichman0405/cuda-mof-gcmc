[English](README.md) | **简体中文**

# `cuda-mof-gcmc`

![CUDA C++](https://img.shields.io/badge/CUDA-C%2B%2B17-76B900?style=flat-square&logo=nvidia)
![Target GPU](https://img.shields.io/badge/RTX%205070-sm__120-2563EB?style=flat-square)
![RASPA3 Verified](https://img.shields.io/badge/RASPA3%203.0.29-%E2%89%A4%200.54%CF%83-0F766E?style=flat-square)
![Pytest](https://img.shields.io/badge/pytest-14%20passed-059669?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-64748B?style=flat-square)

面向 **NVIDIA RTX 5070**（`sm_120`）的纯 CUDA/C++17 **任意刚性 MOF** 气体吸附 GCMC 引擎。项目包含两部分能力：**（1）经过 `RASPA3` 对标验证的 GPU 经典及多体全自洽极化（`Pol-SCF`）求解器**；**（2）将通用 3D 四元数 CUDA 内核（103 号元素 UFF + 周期性 QEq，或 `Mercado 2016`）封装为零磁盘 I/O 动态库、通过化学势偏置与批处理机器学习势（`MACE + TorchDFTD3`）直连的嵌套蒙特卡洛（Nested MC）引擎**。

---

## 系统架构

![cuda-mof-gcmc 系统架构图](assets/architecture_zh.png)

1. **通用晶体与多位点刚体客体前处理**（[`scripts/qeq_charges.py`](scripts/qeq_charges.py)、[`scripts/guests.py`](scripts/guests.py)）：由 ASE 展开全部 `230` 个空间群，利用 `spacegroup_kinds` 保留 CIF 标签与电荷，并对无电荷骨架自动计算带介电屏蔽（`eps_eff = 1.50`）的周期性 Ewald QEq；客体支持最多 `12` 个位点（含 `Z = 0` 无质量虚位点）、四元数三维刚体旋转及通用 Peng-Robinson 状态方程逸度。
2. **内层 GPU 经典提议（`K` 步突发采样）**（[`src/proposer_dll.cu`](src/proposer_dll.cu)、[`include/cell.hpp`](include/cell.hpp)、[`include/forcefield.hpp`](include/forcefield.hpp)）：在 `~1.8 ms/轮` 内并行推进 `N_boxes` 个盒子连续跑 $K$ 步可逆经典 GCMC，先把撞墙重叠构型挡在神经网络之外。
3. **零 I/O 内存批处理打分与精确重加权**（[`scripts/nested_mc_engine.py`](scripts/nested_mc_engine.py)）：仅把每轮终点构型通过 `ctypes` 指针送入 `FastMaceD3Target`（关闭梯度的分块 MACE + 缓存静态骨架边表的 GPU DFT-D3 + 显存防碎片保护）。采样前 3 阶段自动校准常数偏置 $\Delta\varepsilon$ 保证增删均衡，正式阶段外层严格抵消 $\Delta\varepsilon\,\Delta N$ 并在拒绝时整盒回滚，保持目标 MLIP 系综严格无偏。

---

## 同卡采样速度对比（vs. Direct MLIP-GCMC）

![同卡采样加速比对比](assets/benchmark_ess_speedup.png)

在同一张 **NVIDIA RTX 5070** 上，从完全相同的 MLIP 平衡态起步，对比两条精确巨正则 `Direct MLIP-GCMC` 基线（[`scripts/direct_mlip_gcmc.py`](scripts/direct_mlip_gcmc.py)——常规 **Serial ASE** 与硬球预筛 `r_hs = 1.20 Å` 的 **Batched GPU** 基线，测评脚本 [`scripts/benchmark_speedup.py`](scripts/benchmark_speedup.py)）：

| 体系 (298 K) | 方法 | 吸附量 `(mmol/g)` | 接受率 | `ESS_N / GPU-s` | `SJD_N / GPU-s` | 相对 `Direct Serial ASE` (`SJD_N` / `ESS_N`) | 相对 `Direct Batched GPU` (`SJD_N` / `ESS_N`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Mg-MOF-74 + CO₂`** (`0.05 bar`, `r2SCAN+D3`) | `Direct Serial ASE` (`r_hs=1.20 Å`) | `10.14` | `20.0%` | `0.197` | `0.080` | `1.0×` / `1.0×` | — |
| | `Direct Batched GPU` (`4 盒, r_hs=1.20 Å`) | `9.79 ± 0.27` | `17.5%` | `0.337` | `0.314` | `3.9×` / `1.7×` | `1.0×` / `1.0×` |
| | `Nested GPU (Mercado K=25)` | `9.94 ± 0.40` | `51.6%` | **`0.725`** | `1.735` | `21.6×` / **`3.7×`** | `5.5×` / **`2.1×`** |
| | `Nested GPU (Mercado K=50)` | `9.98 ± 0.32` | `33.3%` | `0.681` | **`2.534`** | **`31.5×`** / `3.5×` | **`8.1×`** / `2.0×` |
| **`IRMOF-1 + CH₄`** (`5.0 bar`, `MACE-MP0+D3`) | `Direct Serial ASE` (`r_hs=1.20 Å`) | `9.04` | `47.0%` | `0.077` | `0.903` | `1.0×` / `1.0×` | — |
| | `Direct Batched GPU` (`4 盒, r_hs=1.20 Å`) | `8.19 ± 0.15` | `45.6%` | `0.301` | `3.176` | `3.5×` / `3.9×` | `1.0×` / `1.0×` |
| | `Nested GPU (通用 UFF+QEq K=25)` | `8.43 ± 0.32` | `28.1%` | `0.300` | **`7.063`** | **`7.8×`** / `3.9×` | **`2.2×`** / `1.0×` |
| | `Nested GPU (通用 UFF+QEq K=50)` | `8.70 ± 0.08` | `17.0%` | **`0.322`** | `6.833` | `7.6×` / **`4.2×`** | **`2.2×`** / `1.1×` |

> **关键发现**：（1）在 `Mg-MOF-74` 上，无电荷 UFF 的外层接受率只有 `6.2%`，加入周期性 QEq 后可提升到 `38.8%`（`5.45 ± 0.30 mmol/g`）；但要顺畅进出开放 Mg²⁺ 约 `2.25 Å` 的强配位势阱（`9.2–10.0 mmol/g`），仍需要像 `Mercado 2016` 这样贴合短程势阱的内层势。（2）加入 PyTorch CUDA 显存防碎片清理（`memory_reserved` 超阈值自动 `empty_cache()`）后，12 GB 显存下 `Mg-MOF-74 0.05 bar` 全程运行时间从 `355.8 s` 降至 **`90.3 s`（`3.9×`）**。

---

## 与 `RASPA3` / `gRASPA` 的差异定位

| 对比维度 | CPU `RASPA3` / GPU `gRASPA` | `cuda-mof-gcmc` |
| :--- | :--- | :--- |
| **核心定位** | 面向成熟经典力场的完整模拟生态（力场最全、支持柔性长链 CBMC 与大规模高通量筛选） | **GPU 原生多体全自洽极化（`Pol-SCF`）** + **耦合机器学习势的零 I/O Nested MC 架构** |
| **极化物理** | 主打固定点电荷模型（LJ + 静电） | 支持三维 Tricubic 电场骨架极化（`Pol-3Site`）及每步 Block 级 FP64 Applequist-Thole **多体自洽 `Pol-SCF`** |
| **MLIP 嵌套支持** | 无针对 PyTorch MLIP 重加权的零 I/O CUDA 提议器动态库 | 原生 C ABI（[`proposer_api.h`](include/proposer_api.h)）直连 `FastMaceD3Target`、3 阶段 $\Delta\mu_{\text{ref}}$、Pooled-Jackknife $Q_{\text{st}}$ 与 `ESS` 诊断 |
| **对标与选型边界** | 实测对标对象为 **CPU 版 `RASPA3 3.0.29`**（未同卡对比 `gRASPA`）；若仅做标准经典力场筛选建议直接用 `gRASPA` | 聚焦单张 RTX 5070 下的任意刚性 MOF 与刚性小分子（`CO₂`、`CH₄`、`N₂`、`H₂`、`Ar`、`H₂O`、`NH₃`） |

---

## 功能模块与成熟度一览

| 模块 | 功能说明 | 状态 |
| :--- | :--- | :---: |
| **Tricubic 经典 CUDA GCMC**（[`src/main_cli.cu`](src/main_cli.cu)） | `0.10 Å` $O(h^4)$ 三维 Tricubic 网格、`UFF`（103 元素）/ `Mercado 2016`、任意三斜 MIC、`Identity Swap`、PR 逸度 | **Verified**（对标 `RASPA3 3.0.29 <= 0.54σ`） |
| **各向异性与全自洽多体极化** | 骨架电场 `Pol-COM` / `Pol-3Site` + Block 级共享内存 FP64 `Pol-SCF` | **Verified**（`--test-scf` 对齐 `36×36` 逆矩阵至 `1.83e-08 K`） |
| **通用 Nested MLIP-GCMC 引擎**（[`src/proposer_dll.cu`](src/proposer_dll.cu), [`scripts/nested_mc_engine.py`](scripts/nested_mc_engine.py)） | 3D 四元数 CUDA 提议器 + `qeq_charges.py` + `guests.py` + 批处理 `FastMaceD3Target` + 显存防碎片 | **Verified**（`pytest 14 passed`；`Pol3->Merc` `z=1.15`；四元数 `0.000 K`） |
| **对照基准与后续扩展**（[`scripts/benchmark_speedup.py`](scripts/benchmark_speedup.py)） | 同卡对比 `ESS / GPU-s` 与 `SJD_N / GPU-s`（`Mg-MOF-74`、`IRMOF-1`） | **持续推进**（下一步接入多 MLIP 后端与 8 卡 Linux 调度） |

---

## 已验证基准与适用边界

![Step 4 Mg-MOF-74 Tricubic Validation](assets/step4_validated_mercado_raspa3.png)

- **`RASPA3 3.0.29` 对齐精度（`Mg-MOF-74`, `298 K`, UFF+REPEAT）**：与单核 RASPA3 Tricubic 网格版在 `0.15 / 1.0 / 5.0 bar` 偏差仅 **`+2.32% / +1.39% / -0.41%`**（$\Delta Q_{\text{st}} \le 0.54\ \text{kJ/mol}$），全 FP64 重算使 `6,400 万` 步最大能量漂移 `≤ 1.02e-08 K`。
- **经典内核吞吐**：实测 **`598,669 MC-steps/s`**（较单核 CPU `RASPA3` Tricubic 网格 `3,908/s` 快 **`153.2×`**，较精确 Ewald `1,215/s` 快 **`492.6×`**）；`256×256×72` 网格构建耗时 `3.03 s`（`38.9×`）。
- **`Pol-SCF` 核验**：`--test-scf` 与 Python `36×36` 精确矩阵求逆偏差仅 **`1.83e-08 K`** 与 **`1.13e-11 e·Å`**。
- **通用提议器与 QEq 精度**：`Pol-3Site -> Mercado` 零偏置对拍吻合在 **`z = 1.15`**（`|ΔQst| = 0.43 kJ/mol`，能量误差 `0.000e+00 K`）；`SPC/E H₂O` 四元数 `eval_energy_quat` 误差 `0.000e+00 K`；周期性 QEq 在 `Mg-MOF-74` 与 `Cu-BTC` 上的 MAE 分别为 `0.153 e` 与 `0.110 e`。

> **适用范围与当前局限**：本引擎假设**刚性 MOF 骨架与刚性多位点分子**（`≤ 12` 位点、`≤ 6` 种 LJ 类型，不支持柔性长链 CBMC 与骨架呼吸）。MLIP 模拟的物理准确性仍取决于底层基础模型与色散选择——例如 `mace-matpes-r2scan-0` 在 `IRMOF-1 + CH₄` 上存在严重过结合（`-30.6` 至 `-36.4 kJ/mol`），而 `MACE-MP-0 medium + D3(pbe)` 为 `-12.03 kJ/mol`。详见 [MLIP Nested MC 研究记录](docs/MLIP_NESTED_MC_NOTES.zh-CN.md)。

---

## 快速开始

### 环境要求
- Windows 10/11 + NVIDIA CUDA Toolkit（`sm_120`, RTX 5070）+ MSVC（Visual Studio 2022 生成工具）
- 可选（用于 MLIP Nested MC 与 pytest）：Python 3.11+（先按本地 CUDA 版本安装 PyTorch，再执行 `pip install -r requirements.txt`，详见 [requirements.txt](requirements.txt) 与 [pyproject.toml](pyproject.toml)）

### 编译与运行

```powershell
# 默认编译核心目标（mof_gcmc_cli.exe + mof_gcmc_proposer.dll）；若需归档的旧版可执行文件可执行 `build.bat all`
cmd /c build.bat

# 经典与极化统一 CLI 预设及回归校验
.\build\mof_gcmc_cli.exe --test-scf
.\build\mof_gcmc_cli.exe --config configs\mg_mof74_co2_298K_uff.json
.\build\mof_gcmc_cli.exe --config configs\mg_mof74_fluegas_298K_pol3site.json
.\.venv-mlip\Scripts\python.exe -u scripts\verify_cli.py

# Pytest 自动化回归套件（13 项经典 + 1 项 MLIP 一致性）
.\.venv-mlip\Scripts\python.exe -m pytest -v

# 通用 MLIP Nested MC 示例（Mg-MOF-74 + CO2 / IRMOF-1 + CH4 自动 QEq）
.\.venv-mlip\Scripts\python.exe -u scripts\run_mlip_gcmc.py `
  --cif data\MgMOF-74.cif --pressures 0.01 0.05 0.15 `
  --guest co2 --proposer mercado --model mace-matpes-r2scan-0 --dispersion on --xc r2scan `
  --out-json build\mg_r2scan.json --out-csv build\mg_r2scan.csv

.\.venv-mlip\Scripts\python.exe -u scripts\run_mlip_gcmc.py `
  --cif data\IRMOF-1.cif --supercell 1 1 1 --pressures 5.0 `
  --guest ch4 --proposer uff --charge-mode auto --model medium --dispersion on --xc pbe `
  --out-json build\irmof1_ch4.json

# 同卡 ESS / GPU-s 与 SJD_N / GPU-s 加速比对比基准（vs. Direct MLIP-GCMC）
.\.venv-mlip\Scripts\python.exe -u scripts\benchmark_speedup.py
```

---

## 详细文档与目录结构

- [**项目技术报告（预印本风格）**](docs/TECHNICAL_REPORT.zh-CN.md) — 统一汇总理论方法、`RASPA3` / `Pol-SCF` / Nested MC 校验、基础模型短程诊断及四类 MOF NIST ISODB 基准评测。
- [**经典与极化 CUDA 引擎基准校验（第一至五步）**](docs/CLASSICAL_VALIDATION.zh-CN.md) — 晶体几何校验、Tricubic 精度、RASPA3 对标表、共吸附选择性与 `Pol-SCF` 核验。
- [**MLIP 嵌套蒙特卡洛：已验证机制与 WIP 研究记录**](docs/MLIP_NESTED_MC_NOTES.zh-CN.md) — 零偏差重加权证明、$\Delta\mu_{\text{ref}}$ 标定、MACE 批处理加速、基础模型诊断与待解决物理问题。

```text
cuda-mof-gcmc/
├── pyproject.toml  # 项目打包配置与 pytest 标记（mlip, slow）
├── configs/        # mof_gcmc_cli.exe 预设配置（UFF、Mercado 2016、313K PSA、烟道气、Pol-SCF）
├── data/           # 晶体 CIF（Mg-MOF-74、Ni-MOF-74、Cu-BTC、IRMOF-1）、Mercado 力场与 NIST ISODB 实验表
├── docs/           # 详细基准测试文档、公式推导与科研进展/待办记录（中/英文）
├── include/        # C++17 / CUDA 头文件：三斜最小镜像、103 元素 UFF + PR、三维网格与提议器 C ABI
├── src/            # 现役 CUDA 核与驱动源码（mof_gcmc_cli、mof_gcmc_proposer.dll）+ src/legacy/（v1–v3 归档）
├── scripts/        # 核心引擎、QEq、刚体客体与基准脚本 + scripts/experiments/（论文复现）& scripts/legacy/（归档）
├── tests/          # Pytest 回归测试（103 元素斜晶胞、QEq、四元数多客体、Jackknife/ESS、Direct 零偏置）
└── assets/         # 中/英文系统架构图与基准对标图（300 DPI）
```
