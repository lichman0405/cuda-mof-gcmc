[English](README.md) | **简体中文**

# `cuda-mof-gcmc`

![CUDA C++](https://img.shields.io/badge/CUDA-C%2B%2B17-76B900?style=flat-square&logo=nvidia)
![Multi-GPU Verified](https://img.shields.io/badge/Multi--GPU-sm__86%20%2F%20sm__120-2563EB?style=flat-square)
![RASPA3 Verified](https://img.shields.io/badge/RASPA3%203.0.29-%E2%89%A4%200.54%CF%83-0F766E?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-64748B?style=flat-square)

面向 **Windows（RTX 5070 `sm_120`）与多卡 Linux（RTX 3090 `sm_86`）** 的纯 CUDA/C++17 **任意刚性 MOF** 气体吸附 GCMC 引擎。项目集成了两大核心能力：**（1）通过 `RASPA3` 对标验证的 GPU 经典及多体全自洽极化（`Pol-SCF`）求解器**；**（2）零手工力场 `Auto-MLIP`（在线 `1B+2B` 自蒸馏 + 128 线程块并行 CUDA Rosenbluth GPU-CBMC）与批处理机器学习势（`MACE + TorchDFTD3`）直连、支持多卡并行的精确嵌套蒙特卡洛（Nested MC）引擎**。

---

## 核心技术成果摘要（详见 [中文技术报告](docs/TECHNICAL_REPORT.zh-CN.md)）

![Auto-MLIP + GPU-CBMC Nested MC 架构图](assets/architecture_auto_mlip_cbmc.png)

- **在线 `1B+2B` 自蒸馏 + 128 线程块并行 GPU-CBMC（零手工原子型力场）**：自动从 `1,200` 个单分子与 `192` 个孔内二聚体 MLIP 采样中蒸馏环境感知 `1B` 网格（15% 留出集 $R^2 = \mathbf{0.985\text{–}0.991}$，MAE `1.27–2.29 kJ/mol`）与孔道 `2B` LUT（MAE `0.10–1.65 kJ/mol`，单次 `52.8–209.5 s` SHA-256 缓存复用）。
- **`Mg-MOF-74 + CO₂ @ 298 K` 方法学压测与实验对齐**：
  ![Mg-MOF-74 方法对比与双向收敛](assets/fig1_mgmof74_method_comparison.png)
  - 相比 `Direct MLIP-GCMC`（插入接收率 `3.65%`），`Auto-MLIP + CBMC`（`k=16`）将 `0.28 bar` **插入接收率提升 `15.8×`（`57.5%`）**、$\text{ESJD}_N$ 提升 **`8.0×`**（`0.1314` vs `0.0164`），`dN_moves/s` 提升 **`2.6×`**；
  - `empty` 与 `high_loading` 双向初始化收敛至 **`6.322 ± 0.233`** 与 **`6.314 ± 0.171 mmol/g`**（差异仅 **`0.0074 mmol/g / 0.12%`**）；在 `1.0 bar`（`r2SCAN No-D3`）实测 **`8.148 ± 0.025 mmol/g`**，与 NIST 实验值 `8.170 mmol/g` 仅差 **`0.26%`**。
- **跨 MOF / 跨客体迁移与基础模型物理诊断**：横跨 **`Mg-MOF-74`、`Ni-MOF-74`、`Cu-BTC`、`IRMOF-1`** 四种骨架与 **`CO₂ / CH₄`** 双客体，配合 **Widom 插入** 揭示了 `MP0+D3` 因 `D3(BJ)` 贡献 `51%–57%` 吸引能导致的过结合机制，以及 `mace-matpes-r2scan-0` 在 `IRMOF-1 + CH₄` 的分布外吸引伪影。
- **`4× RTX 3090` 并行可扩展性**：`replica` 分片模式实测纯 MC 核加速比 **`3.26×`**（`51.10 s → 15.69 s`，并行效率 **`81.4%`**；含多进程权重加载端到端 `1.90×`）；`pressure` 队列模式端到端 **`2.19×`**（`131.23 s → 59.79 s`）。
- **经典与 `Pol-SCF` 基线**（见 [经典基准校验](docs/CLASSICAL_VALIDATION.zh-CN.md)）：`Mg-MOF-74` 对比单核 `RASPA3 3.0.29` Tricubic 误差 `≤ 0.54σ`，经典核吞吐达 `598,669 MC-steps/s`（`153.2×`），`Pol-SCF` 与精确 `36×36` 矩阵求逆偏差 `1.83e-08 K`。

---

## 核心架构模块

1. **通用晶体与多位点刚体客体前处理**（[`scripts/qeq_charges.py`](scripts/qeq_charges.py)、[`scripts/guests.py`](scripts/guests.py)）：由 ASE 展开全部 `230` 个空间群，利用 `spacegroup_kinds` 保留 CIF 电荷与原子标签，无电荷骨架自动计算带介电屏蔽（`eps_eff = 1.50`）的周期性 Ewald QEq；客体支持最多 `12` 个位点及 Peng–Robinson 状态方程逸度。
2. **在线 `1B+2B` MLIP 自蒸馏与零 I/O CUDA CBMC 提议器**（[`scripts/mlip_distill.py`](scripts/mlip_distill.py)、[`src/proposer_dll.cu`](src/proposer_dll.cu)、[`include/proposer_api.h`](include/proposer_api.h)）：在 GPU 显存驻留 `O(h^4)` Tricubic 网格与 `2B` LUT，Block 内 128 线程并发 Rosenbluth 试探推进内层可逆 GCMC。
3. **精确 Nested MC、严格统计审计与多卡调度**（[`scripts/nested_mc_engine.py`](scripts/nested_mc_engine.py)、[`scripts/mcmc_diagnostics.py`](scripts/mcmc_diagnostics.py)、[`scripts/multi_gpu.py`](scripts/multi_gpu.py)）：`FastMaceD3Target` 结合中位数 $\Delta\varepsilon$ 校准、rank-normalized split/folded $\hat{R}$、bulk/tail ESS、独立涨落 $Q_{\text{st}}$ 估计、SHA-256 断点续跑（[`scripts/run_metadata.py`](scripts/run_metadata.py)）以及 `pressure` / `replica` 多卡扩展。

---

## 快速开始（Windows 与 Linux）

### 一键检测、安装、编译与自检（`fast_start.bat` / `fast_start.sh`）

- **Windows**：在仓库根目录执行 [`fast_start.bat`](fast_start.bat)（内部调用 [`scripts/fast_start_win.ps1`](scripts/fast_start_win.ps1)）：

```powershell
# 仅检查依赖状态
cmd /c fast_start.bat -CheckOnly

# 自动补齐缺失依赖、调用 build.bat 编译并运行自检
cmd /c fast_start.bat
```

- **Linux**：在仓库根目录执行 [`fast_start.sh`](fast_start.sh)：

```bash
./fast_start.sh --check-only
./fast_start.sh --gpu 0
```

### `Auto-MLIP + GPU-CBMC` Nested MC 实跑命令（[`scripts/run_mlip_gcmc.py`](scripts/run_mlip_gcmc.py)）

```bash
# Mg-MOF-74 + CO2 @ 298 K (r2SCAN No-D3, 128-trial GPU-CBMC, 多卡压力队列 + 断点续跑)
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=scripts .venv-mlip/bin/python -u scripts/run_mlip_gcmc.py \
  --cif data/MgMOF-74.cif --supercell 1 1 2 --guest co2 --temp 298 \
  --pressures 0.012 0.15 0.28 1.0 --proposer auto_mlip --cbmc-k 128 \
  --model mace-matpes-r2scan-0 --dispersion off \
  --boxes 4 --k-steps 16 --pilot-cycles 40 --eq-cycles 1200 --prod-cycles 4800 \
  --distill-cache-dir build/distill_cache --gpus 0 1 2 3 --parallel-mode auto \
  --checkpoint-dir build/ckpt_mg74 --checkpoint-every 200 --resume-mode extend \
  --out-json build/mgmof74_co2_r2scan.json

# Widom 零负载 Henry 常数与 MACE/D3 能量分解
PYTHONPATH=scripts .venv-mlip/bin/python -u scripts/widom_mlip.py \
  --cif data/MgMOF-74.cif --supercell 1 1 2 --guest co2 --temp 298 \
  --model mace-matpes-r2scan-0 --dispersion off --n-samples 6000 \
  --out-json build/widom_mgmof74_r2scan.json
```

### 经典与 `Pol-SCF` CLI 回归校验

```powershell
.\build\mof_gcmc_cli.exe --test-scf
.\build\mof_gcmc_cli.exe --config configs\mg_mof74_co2_298K_uff.json
.\.venv-mlip\Scripts\python.exe -m pytest -v -m "not mlip"
```

---

## 文档与目录导航

- [**MLIP Nested MC 技术报告（中文）**](docs/TECHNICAL_REPORT.zh-CN.md) | [**Technical Report (EN)**](docs/TECHNICAL_REPORT.md)
- [**经典与全自洽极化 CUDA 引擎基准校验**](docs/CLASSICAL_VALIDATION.zh-CN.md) | [**English**](docs/CLASSICAL_VALIDATION.md)
- [**架构重构与阶段门禁方案**](PLAN.zh-CN.md) | [**结构化数据归档 JSON**](docs/results/phase2_phase3_summary.json)

```text
cuda-mof-gcmc/
├── fast_start.bat / .sh  # 一键环境检测、依赖安装、编译与自检入口
├── build.bat / .sh       # MSVC+CUDA / nvcc+g++ 编译脚本
├── configs/              # mof_gcmc_cli 经典与 Pol-SCF 预设配置
├── data/                 # MOF CIF 晶体、Mercado 力场与 NIST ISODB 实验等温线 CSV
├── docs/                 # 中英文技术报告、经典验证文档及 docs/results/phase2_phase3_summary.json
├── include/              # CUDA/C++17 头文件与零 I/O proposer C ABI (proposer_api.h)
├── src/                  # GPU GCMC 与 128 线程块并行 GPU-CBMC 核源码 + src/legacy/
├── scripts/              # run_mlip_gcmc, mlip_distill, widom_mlip, nested_mc_engine, multi_gpu, plot_phase4_figures
├── tests/                # Pytest 回归测试集（含 test_phase1_distill_and_cbmc.py）
└── assets/               # Phase 1–4 出版级图表与架构图
```
