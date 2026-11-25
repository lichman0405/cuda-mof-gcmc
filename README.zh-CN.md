[English](README.md) | **简体中文**

# `cuda-mof-gcmc`

![CUDA C++](https://img.shields.io/badge/CUDA-C%2B%2B17-76B900?style=flat-square&logo=nvidia)
![Multi-GPU](https://img.shields.io/badge/Multi--GPU-sm__86%20%2F%20sm__120-2563EB?style=flat-square)
![RASPA3 Verified](https://img.shields.io/badge/RASPA3%203.0.29-%E2%89%A4%200.54%CF%83-0F766E?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-64748B?style=flat-square)

刚性 MOF 气体吸附的 CUDA/C++17 GCMC 引擎，支持 Windows（RTX 5070，`sm_120`）和多卡 Linux（RTX 3090，`sm_86`）。项目包含两部分：

1. 经典与全自洽极化（`Pol-SCF`）GCMC 求解器，已与 `RASPA3` 对标，详见[经典基准文档](docs/CLASSICAL_VALIDATION.zh-CN.md)。
2. 以机器学习势（MACE，可选 D3）为目标的嵌套蒙特卡洛引擎。参考势从目标 MLIP 在线蒸馏得到，不需要手工力场；内层在 GPU 上做 128 路 Rosenbluth 构型偏倚移动，外层做批量 MLIP 评估和精确 Metropolis 修正。方法与结果见[技术报告](docs/TECHNICAL_REPORT.zh-CN.md)。

![方法示意](assets/fig1_architecture.png)

## 主要结果

以下数字均来自[技术报告](docs/TECHNICAL_REPORT.zh-CN.md)与 [`docs/results/phase2_phase3_summary.json`](docs/results/phase2_phase3_summary.json)。

- **代理势构建**：每个（框架、客体、MLIP）组合用 1,200 个单分子构型和 192 个孔内二聚体拟合，单卡耗时 56–210 s。在生产使用的 7 个缓存上，1B 项 15% 留出集 R² 为 0.975–0.990，MAE 为 1.27–2.44 kJ/mol。
- **采样效率（Mg-MOF-74 + CO₂，0.28 bar，同一目标势 r2SCAN）**：直接 MLIP-GCMC 运行 300 s 后箱平均吸附量约 1.8 mmol/g；UFF 提议器外层接受率 2.9%；本方法从空箱和预装载两个方向分别收敛到 6.32 ± 0.23 和 6.31 ± 0.17 mmol/g，效率与专门拟合的 Mercado-2016 力场相当。
- **与实验比较**：吻合程度主要由目标 MLIP 决定。r2SCAN 在 Mg-MOF-74 1 bar（8.15 vs. 实验 7.02–8.16 mmol/g）和 IRMOF-1 5 bar（3.74 vs. 4.70）附近接近实验，但在 Mg-MOF-74 0.012 bar 低估 8–14 倍，对 Ni-MOF-74、Cu-BTC 低估约一个数量级；MP0+D3 在 Mg-MOF-74 上高估约 2.4 倍，其结合能中 51%–57% 来自 D3。
- **收敛性**：16 个生产压力点中 6 个通过全部判据（$\hat{R}<1.01$、ESS ≥ 400 等）；未通过的点（主要是等温线陡峭段和高装载）在报告中逐一列出。
- **多卡**：4 张 RTX 3090 上，副本分片的纯 MC 时间加速 3.26 倍，压力队列端到端加速 2.19 倍。
- **经典与 `Pol-SCF` 基线**：Mg-MOF-74 与单核 `RASPA3 3.0.29` Tricubic 结果偏差 ≤ 0.54σ；经典内核 598,669 MC-steps/s；`Pol-SCF` 与 36×36 精确矩阵求逆偏差 1.83×10⁻⁸ K。

![实验一：Mg-MOF-74 + CO₂](assets/fig2_exp1_mgmof74.png)

## 代码结构

1. **结构与客体前处理**（[`scripts/qeq_charges.py`](scripts/qeq_charges.py)、[`scripts/guests.py`](scripts/guests.py)）：ASE 展开任意空间群并保留 CIF 电荷；无电荷框架使用带屏蔽的周期性 Ewald QEq；刚性客体最多 12 个位点，逸度由 Peng–Robinson 方程计算。
2. **代理势蒸馏与 GPU 提议器**（[`scripts/mlip_distill.py`](scripts/mlip_distill.py)、[`src/proposer_dll.cu`](src/proposer_dll.cu)、[`include/proposer_api.h`](include/proposer_api.h)）：1B 三维网格与 2B 残差表驻留显存，每个线程块对应一个模拟箱，通过 C ABI 与 Python 零拷贝交换构型。
3. **嵌套 MC、收敛诊断与多卡调度**（[`scripts/nested_mc_engine.py`](scripts/nested_mc_engine.py)、[`scripts/mcmc_diagnostics.py`](scripts/mcmc_diagnostics.py)、[`scripts/multi_gpu.py`](scripts/multi_gpu.py)、[`scripts/run_metadata.py`](scripts/run_metadata.py)）：批量 MACE/D3 评估、Δε 校准、rank-normalized $\hat{R}$ 与 bulk/tail ESS、带 SHA-256 元数据校验的断点续跑，以及 `pressure` / `replica` 两种多卡模式。

## 快速开始

### 环境检测与编译

```powershell
# Windows：只检查依赖
cmd /c fast_start.bat -CheckOnly
# Windows：补齐依赖、编译并自检
cmd /c fast_start.bat
```

```bash
# Linux
./fast_start.sh --check-only
./fast_start.sh --gpu 0
```

### 运行嵌套 MC（[`scripts/run_mlip_gcmc.py`](scripts/run_mlip_gcmc.py)）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=scripts .venv-mlip/bin/python -u scripts/run_mlip_gcmc.py \
  --cif data/MgMOF-74.cif --supercell 1 1 2 --guest co2 --temp 298 \
  --pressures 0.012 0.15 0.28 1.0 --proposer auto_mlip --cbmc-k 128 \
  --model mace-matpes-r2scan-0 --dispersion off \
  --boxes 4 --k-steps 16 --pilot-cycles 40 --eq-cycles 1200 --prod-cycles 4800 \
  --distill-cache-dir build/distill_cache --gpus 0 1 2 3 --parallel-mode auto \
  --checkpoint-dir build/ckpt_mg74 --checkpoint-every 200 --resume-mode extend \
  --out-json build/mgmof74_co2_r2scan.json
```

Widom 插入与能量分解：

```bash
PYTHONPATH=scripts .venv-mlip/bin/python -u scripts/widom_mlip.py \
  --cif data/MgMOF-74.cif --supercell 1 1 2 --guest co2 --temp 298 \
  --model mace-matpes-r2scan-0 --dispersion off --n-samples 6000 \
  --out-json build/widom_mgmof74_r2scan.json
```

重新生成报告中的图：

```bash
.venv-mlip/bin/python scripts/plot_phase4_figures.py
```

### 经典与 `Pol-SCF` 回归测试

```powershell
.\build\mof_gcmc_cli.exe --test-scf
.\build\mof_gcmc_cli.exe --config configs\mg_mof74_co2_298K_uff.json
.\.venv-mlip\Scripts\python.exe -m pytest -v -m "not mlip"
```

## 文档

- [技术报告（中文）](docs/TECHNICAL_REPORT.zh-CN.md) · [Technical report (English)](docs/TECHNICAL_REPORT.md)
- [经典与极化基准](docs/CLASSICAL_VALIDATION.zh-CN.md) · [English](docs/CLASSICAL_VALIDATION.md)
- [设计方案与阶段门禁](PLAN.zh-CN.md)
- [结果汇总 JSON](docs/results/phase2_phase3_summary.json)

```text
cuda-mof-gcmc/
├── fast_start.bat / .sh  # 环境检测、依赖安装、编译与自检
├── build.bat / .sh       # MSVC+CUDA / nvcc+g++ 编译脚本
├── configs/              # mof_gcmc_cli 预设（UFF、Mercado、烟道气、Pol-SCF）
├── data/                 # CIF、Mercado 力场参数、实验等温线
├── docs/                 # 技术报告、经典基准文档、results/
├── include/              # CUDA/C++17 头文件与 proposer C ABI
├── src/                  # CUDA 内核与驱动；src/legacy/ 为旧版
├── scripts/              # run_mlip_gcmc、mlip_distill、widom_mlip、nested_mc_engine、multi_gpu、plot_phase4_figures 等
├── tests/                # pytest 测试
└── assets/               # 报告与文档中的图
```
