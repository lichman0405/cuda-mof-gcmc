[English](README.md) | **简体中文**

# `cuda-mof-gcmc`

![CUDA C++](https://img.shields.io/badge/CUDA-C%2B%2B17-76B900?style=flat-square&logo=nvidia)
![Target GPU](https://img.shields.io/badge/RTX%205070-sm__120-2563EB?style=flat-square)
![RASPA3 Verified](https://img.shields.io/badge/RASPA3%203.0.29-%E2%89%A4%200.54%CF%83-0F766E?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-64748B?style=flat-square)

面向 **NVIDIA RTX 5070**（`sm_120`）的纯 CUDA/C++17 刚性金属有机框架（MOF）气体吸附 GCMC 引擎。项目包含两条互补能力：**（1）经过 `RASPA3` 对标验证的 GPU 经典及多体全自洽极化（`Pol-SCF`）求解器**；**（2）将高性能 CUDA 内核封装为零磁盘 I/O 动态库、通过化学势偏置与批处理机器学习势（`MACE + TorchDFTD3`）直连的嵌套蒙特卡洛（Nested MC）引擎**（$\Delta\mu_{\text{ref}}$ 自动标定）。

---

## 核心架构：带化学势偏置的零 I/O Nested MLIP-GCMC

直接拿 MLIP 跑纯 GCMC 时，约 95% 的神经网络推理会浪费在撞墙重叠的插入试探上；而当 MLIP 与经典提议势的结合能差距较大时，不加偏置会陷入“只进不出”的删除死锁。本引擎通过如下内存直连闭环同时解决这两个瓶颈：

```text
  [1. 底层 CUDA 构型提议器 DLL (build\mof_gcmc_proposer.dll)]
     在偏置逸度 f_ref = f_tgt * exp(-beta * delta_eps) 下连续跑 K 步可逆经典 GCMC
     （含平移、旋转、插入、删除），过滤撞墙重叠废步 -> 输出 K 步后的无重叠整盒构型 x'
                                 |
                       (零磁盘 I/O ctypes 内存直传)
                                 v
  [2. 批处理 PyTorch 目标势 (FastMaceD3Target)]
     4 副本合并微批评估 MACE + TorchDFTD3 精确结合能 U_tgt(x')（37.5 ms/box）
                                 |
                                 v
  [3. 外层严格重加权接收 / 全盒回滚]
     按 min{1, exp[-beta * (dU_tgt - dU_ref - delta_eps * dN)]} 决定接收或拒绝
     若拒绝 -> CUDA 动态库直接将整个模拟盒状态回滚至 K 步之前 x
```

1. **底层构型提议器（`K` 步经典快筛）**：CUDA 动态库在微秒级时间内连续推进 $K$ 步完整可逆经典 GCMC（不只是插入，涵盖增删改位），把 ~95% 的重叠废构型挡在神经网络之外。
2. **零 I/O 内存直传打分**：只将每轮 $K$ 步结束时的整盒最新坐标 $\mathbf{x}'$ 通过 `ctypes` 指针直接送入 GPU 批处理 `MACE + TorchDFTD3`，全程不写任何临时文件。
3. **化学势偏置与外层严格纠偏**：采样前先标定好常数偏置 $\Delta\varepsilon$，使内层经典提议的孔道负载贴近 MLIP 区间以保证增删提议均衡；进入正式采样后 $\Delta\varepsilon$ 锁死不变，外层判据精准抵消 $\Delta\varepsilon\,\Delta N$ 并在拒绝时整盒回滚，严格保持目标 MLIP 巨正则系综无偏。

---

## 与 `RASPA3` / `gRASPA` 的差异定位

| 对比维度 | CPU `RASPA3` / GPU `gRASPA` | `cuda-mof-gcmc` |
| :--- | :--- | :--- |
| **核心定位** | 面向成熟经典力场的完整模拟生态（力场最全、支持柔性长链 CBMC 与大规模高通量筛选） | **GPU 原生多体全自洽极化（`Pol-SCF`）** + **耦合机器学习势的零 I/O Nested MC 架构** |
| **极化物理** | 主打固定点电荷模型（LJ + 静电） | 支持三维 Tricubic 电场骨架极化（`Pol-3Site`）及每步 Block 级 FP64 Applequist-Thole **多体自洽 `Pol-SCF`** |
| **MLIP 嵌套支持** | 无针对 PyTorch MLIP 重加权的零 I/O CUDA 提议器动态库 | 原生 C ABI（[`proposer_api.h`](include/proposer_api.h)）直连 `FastMaceD3Target` + 三阶段 $\Delta\mu_{\text{ref}}$ 自动标定 |
| **对标与选型边界** | 本项目实测对标对象为 **CPU 版 `RASPA3 3.0.29`**（未同卡对比 `gRASPA`）；若仅做大规模标准经典力场筛选建议直接用 `gRASPA` | 聚焦单张 RTX 5070 下的刚性 MOF + 刚性小分子（`CO₂ / N₂ / CH₄`），用于运行开放金属位点多体极化与 MLIP 巨正则模拟 |

---

## 功能模块与成熟度一览

| 模块 | 功能说明 | 状态 |
| :--- | :--- | :---: |
| **Tricubic 经典 CUDA GCMC**（[`src/main_cli.cu`](src/main_cli.cu)） | `0.10 Å` $O(h^4)$ 三维 Tricubic 网格、`UFF` / `Mercado 2016` DFT 力场、二元 `Identity Swap`、PR 逸度 | **Verified**（对标 `RASPA3 3.0.29 <= 0.54σ`） |
| **各向异性与全自洽多体极化** | 骨架电场 `Pol-COM` / `Pol-3Site` + Block 级共享内存 FP64 `Pol-SCF` | **Verified**（`--test-scf` 对齐 `36×36` 逆矩阵至 `1.83e-08 K`） |
| **Nested MC 提议器 DLL 与 `FastMaceD3Target`** | 零 I/O CUDA 提议器（[`src/proposer_dll.cu`](src/proposer_dll.cu)）+ 4 副本批处理 MLIP 引擎（[`scripts/nested_mc_engine.py`](scripts/nested_mc_engine.py)） | **Verified**（零偏差检验 `0.27σ`、较 ASE 提速 `4.21×`） |
| **MLIP 吸附等温线与吸附热研究**（[`scripts/run_mlip_gcmc.py`](scripts/run_mlip_gcmc.py)） | 跨材料 MLIP 等温线计算、基础模型短程过结合诊断与采样收敛研究 | **WIP**（DFT 真值对照与采样优化推进中，见 [WIP 记录](docs/MLIP_NESTED_MC_NOTES.zh-CN.md)） |

---

## 已验证基准与适用边界

![Step 4 Mg-MOF-74 Tricubic Validation](assets/step4_validated_mercado_raspa3.png)

- **`RASPA3 3.0.29` 对齐精度（`Mg-MOF-74`, `298 K`, UFF+REPEAT）**：与单核 RASPA3 Tricubic 网格版在 `0.15 / 1.0 / 5.0 bar` 偏差仅 **`+2.32% / +1.39% / -0.41%`**（$\Delta Q_{\text{st}} \le 0.54\ \text{kJ/mol}$），全 FP64 重算使 `6,400 万` 步最大能量漂移 `≤ 1.02e-08 K`。
- **经典内核吞吐**：实测 **`598,669 MC-steps/s`**（较单核 CPU `RASPA3` Tricubic 网格 `3,908/s` 快 **`153.2×`**，较精确 Ewald `1,215/s` 快 **`492.6×`**）；`256×256×72` 网格构建耗时 `3.03 s`（`38.9×`）。
- **`Pol-SCF` 核验**：`--test-scf` 与 Python `36×36` 精确矩阵求逆偏差仅 **`1.83e-08 K`** 与 **`1.13e-11 e·Å`**。
- **Nested MC 机制核验**：`proposer_eval_energy` 与核内误差 **`0.000e+00 K`**；`Pol-3Site -> Mercado` 零偏差对拍吻合在 **`z = 0.27σ`**；`FastMaceD3Target` 较原生 ASE 提速 **`4.21×`**（`157.7 → 37.5 ms/box`）；`1×1×2` 与 `1×1×4` 超胞一致性偏差分别为 `0.04σ`（经典）与 `0.63σ`（MLIP）。

> **适用范围与当前局限**：本引擎假设**刚性 MOF 骨架与刚性小分子**（`CO₂`、`N₂`、`CH₄`，不支持柔性长链 CBMC 与骨架呼吸）。`Pol-SCF` 每步试探需求解 $O(N^2)$ FP64 自洽迭代且沿用非极化 TraPPE/UFF 基础参数（存在诱导双重计数）。MLIP 嵌套模拟的物理预测精度取决于底层基础模型、色散参数、DFT 标定及 MCMC 采样轮数，当前已知问题详见 [已知问题与待办工作](docs/MLIP_NESTED_MC_NOTES.zh-CN.md#4-已知问题与待办工作)。

---

## 快速开始

### 环境要求
- Windows 10/11 + NVIDIA CUDA Toolkit（`sm_120`, RTX 5070）+ MSVC（Visual Studio 2022 生成工具）
- 可选（用于 MLIP Nested MC）：Python 3.11+（先按本地 CUDA 版本安装 PyTorch，再执行 `pip install -r requirements.txt`，详见 [requirements.txt](requirements.txt)）

### 编译与运行

```powershell
# 默认编译核心目标（mof_gcmc_cli.exe + mof_gcmc_proposer.dll）；若需归档的旧版可执行文件可执行 `build.bat all`
cmd /c build.bat

# 经典与极化统一 CLI 预设
.\build\mof_gcmc_cli.exe --test-scf
.\build\mof_gcmc_cli.exe --config configs\mg_mof74_co2_298K_uff.json
.\build\mof_gcmc_cli.exe --config configs\mg_mof74_fluegas_298K_pol3site.json
.\build\mof_gcmc_cli.exe --config configs\mg_mof74_co2_298K_scf.json
uv run --with numpy python scripts\verify_cli.py

# 零 I/O MLIP Nested MC 机制测试与统一脚本（WIP）
.venv-mlip\Scripts\python.exe scripts\verify_nested_mc.py
.venv-mlip\Scripts\python.exe -u scripts\run_mlip_gcmc.py `
  --cif data\MgMOF-74.cif --pressures 0.01 0.05 0.15 `
  --proposer mercado --model mace-matpes-r2scan-0 --dispersion on --xc r2scan `
  --out-json build\mg_r2scan.json
```

---

## 详细文档与目录结构

- [**项目技术报告（预印本风格）**](docs/TECHNICAL_REPORT.zh-CN.md) — 统一汇总理论方法、`RASPA3` / `Pol-SCF` / Nested MC 校验、基础模型短程诊断及四类 MOF NIST ISODB 基准评测。
- [**经典与极化 CUDA 引擎基准校验（第一至五步）**](docs/CLASSICAL_VALIDATION.zh-CN.md) — 晶体几何校验、Tricubic 精度、RASPA3 对标表、共吸附选择性与 `Pol-SCF` 核验。
- [**MLIP 嵌套蒙特卡洛：已验证机制与 WIP 研究记录**](docs/MLIP_NESTED_MC_NOTES.zh-CN.md) — 零偏差重加权证明、$\Delta\mu_{\text{ref}}$ 标定、MACE 批处理加速、基础模型诊断与待解决物理问题。

```text
cuda-mof-gcmc/
├── configs/   # mof_gcmc_cli.exe 预设配置（UFF、Mercado 2016、313K PSA、烟道气、Pol-SCF）
├── data/      # 晶体 CIF（Mg-MOF-74、Ni-MOF-74、Cu-BTC、IRMOF-1）、Mercado 力场与 NIST ISODB 实验表
├── docs/      # 详细基准测试文档、公式推导与科研进展/待办记录（中/英文）
├── include/   # C++17 / CUDA 头文件：最小镜像、力场参数、三维网格、CIF 解析与提议器 C ABI
├── src/       # 现役 CUDA 核与驱动源码（mof_gcmc_cli、mof_gcmc_proposer.dll）+ src/legacy/（v1–v3 归档）
├── scripts/   # Python 自动校验套件、RASPA3 对标脚本与 MLIP Nested MC 引擎
└── assets/    # 基准对标与验证图（300 DPI）
```
