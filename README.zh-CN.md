[English](README.md) | **简体中文**

# `cuda-mof-gcmc`

![CUDA C++](https://img.shields.io/badge/CUDA-C%2B%2B17-76B900?style=flat-square&logo=nvidia)
![Multi-GPU](https://img.shields.io/badge/Multi--GPU-sm__86%20%2F%20sm__120-2563EB?style=flat-square)
![RASPA3 Verified](https://img.shields.io/badge/RASPA3%203.0.29-%E2%89%A4%200.54%CF%83-0F766E?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-64748B?style=flat-square)

刚性 MOF 气体吸附的 CUDA/C++17 巨正则蒙特卡洛（GCMC）引擎，支持 Windows（RTX 5070，`sm_120`）与多卡 Linux 节点（8×RTX 3090，`sm_86`）。项目包含两部分核心能力：

1. 经 `RASPA3 3.0.29` 严格对标的经典及全自洽极化（`Pol-SCF`）GCMC 求解器，详见[经典验证文档](docs/CLASSICAL_VALIDATION.zh-CN.md)。
2. 面向全系统机器学习原子间势（`mace-matpes-r2scan-0`、`MACE-MP-0`、`MACE-DAC-1`，可选 D3(BJ) 色散）的自动化 **嵌套蒙特卡洛 + 独立副本梯子 $\mu$-REMC** 引擎。系统按 `(框架, 客体, 目标 MLIP)` 按需自动蒸馏一体（1B 径向基+电中性 Ewald 三次插值网格）与二体（2B 孔内二聚体 LUT）参考势，无需手工指定框架原子类型；内层在 GPU 上执行 128 路 Rosenbluth 构型偏倚移动，平衡期冻结边际漂移 $(\Delta\varepsilon, k)$ 校准，外层采用 FP64 节点能量累加的批量 MLIP Metropolis 修正保证严格无偏。完整逐点收敛审计与汇总数据见 [`docs/results/jctc_audit_report.md`](docs/results/jctc_audit_report.md) 与 [`docs/results/jctc_summary.json`](docs/results/jctc_summary.json)。

![方法架构总览](assets/fig1_architecture.png)

## 主要研究结果

以下全部数据均直接来自统一硬核同预算 2×2 归档（`build/jctc_presubmission/matched_budget_audit.md` 与 `matched_budget_summary.json`，`v1.4` 缓存）以及多 MOF 归档（[`docs/results/jctc_audit_report.md`](docs/results/jctc_audit_report.md)、[`docs/results/jctc_summary.json`](docs/results/jctc_summary.json)、[`docs/results/precision_hc_audit.json`](docs/results/precision_hc_audit.json)、[`docs/results/phase2_phase3_summary.json`](docs/results/phase2_phase3_summary.json)）。

- **连续坐标周期最小镜像（MIC）硬核统一（Gate 1）：** 直接 MLIP-GCMC 与 CUDA `auto_mlip` 共用同一连续坐标周期最小镜像硬核判定（`include/hard_core.hpp`：客体 H–框架 `1.20 Å`、客体重原子–框架 `1.48 Å`、不同客体实原子对 `0.50 Å`，排除虚拟位点，$N_{\max}=128$，取消网格角点与代理能量截断拒判）。在 `1,040` 个边界/截断/近壁构型上与 27 镜像暴力搜索达成 **`0 / 1,040` 不一致**，非 MLIP 回归测试 **`58 / 58` 全通过**。
- **同预算 2×2 消融实验（Mg-MOF-74 + CO₂，`1×1×2`，298 K，`0.28 bar`，`mace-matpes-r2scan-0` 关闭 D3，8×RTX 3090）：** 四组 × 3 次独立重复共用 8 档压力（`0.012–1.0 bar`）、4 条独立梯子、逐位一致的 `v1.4` 1B 网格与固定 `7,200 s` 采样壁钟预算（每组每次 `57,600` 全梯子 GPU-s，另记 `39.7–43.7 s` 单卡建势与 `88.9–93.3 s` 八卡偏移校准）。12 次运行一致采样同一平衡吸附量（`6.16–6.30 mmol/g`，总中位数 `6.27 mmol/g`），**$\mu$-REMC 是混合与收敛的主导驱动力**：
  - **关闭 REMC（严格验证 `0` 次交换尝试）：** A 组（`1B-only`，中位 $\mathrm{ESS}_{\text{bulk}}(N) = 239.6$，$\hat{R}_N = 1.0156$）`0/3` 达标，B 组（`1B+2B`，中位 $\mathrm{ESS}_{\text{bulk}}(N) = 231.9$，$\hat{R}_N = 1.0189$）`0/3` 达标，6 次运行全部在 `7,200 s` 预算处右删失。
  - **开启 REMC（生产期梯子往返 `29–43` 次）：** 全梯子采样效率 $\mathrm{ESS}_{\text{bulk}}(N)/\text{GPU-s}$ 在 1B-only 上提升 **`2.26×`**（`C/A`：`0.00942` vs. `0.00416`），在 1B+2B 上提升 **`2.29×`**（`D/B`：`0.00921` vs. `0.00402`）；预算终点全部 11 项判据达标次数为 **C 组 `1/3`**（连续两检查点确认 `1/3`，`4,800 s` 达标、`5,100 s` 确认）和 **D 组 `2/3`**（连续两检查点确认 `1/3`，`6,600 s` 达标、`6,900 s` 确认，另 1 次在 `7,200 s` 终点达标）。
  - **2B 拟合精度与采样效率分离：** 2B 残差 LUT 拟合将 80/20 线上留出集二聚体 MAE 从 `2.60` 降至 `1.51 kJ/mol`（$R^2 = 0.818$）并将 0.28 bar 校准偏移减半（`-3.05 -> -1.42 kJ/mol`），但在同预算下未带来中位 ESS 提速（`B/A = 0.97×`，`D/C = 0.98×`），但开启 REMC 时收紧了重复间诊断离散度（D 组 `3/3` 重复均实现 $\hat{R}_{\max} \le 1.0089$ 且 $N/U/\text{OMS}$ 体相 ESS 均 $\ge 418.8$）。
- **自动化按需 1B+2B 蒸馏与跨 MOF 迁移：** 覆盖 4 种 MOF（Mg-MOF-74、Ni-MOF-74、Cu-BTC、IRMOF-1）、3 种客体（$\text{CO}_2$、$\text{CH}_4$、$\text{H}_2\text{O}$）与 4 种 MLIP 设置（单卡建势 `39.7–212.7 s`），1B 留出集 $R^2$ 为 `0.9236–0.9942`（MAE `0.89–2.85 kJ/mol`），2B 线上 LUT 留出集 MAE 为 `0.04–1.54 kJ/mol`；领域微调的 **`MACE-DAC-1+PBE-D3` $\text{CO}_2$ 等温线跨全部 4 种 MOF 实现 18/18（100%）全收敛**。Widom 与平衡态分解显示 D3(BJ) 贡献了 Mg-MOF-74 结合能的 `47.9%–53.0%`（`MACE-DAC-1`）与 `45.8%–54.1%`（`MP0+D3`）。
- **数值精度、多卡扩展与经典基线：** `FP32` 前向配合 `FP64` 节点累加与全 `FP64` 对比在 64 组外层转移对上达成 **`0%` 接受/拒绝判决翻转**（RMS 偏差 `0.00044–0.00130 kJ/mol`，推理提速 `5.93×–9.49×`）；4×RTX 3090 上副本分片纯 MC 加速 **`3.26×`**，压力队列端到端加速 `2.19×`；经典三维插值内核与单核 `RASPA3 3.0.29` 偏差 `≤ 0.54σ`（`598,669 MC-steps/s`），`Pol-SCF` 极化求解与精确矩阵求逆偏差 `1.83×10⁻⁸ K`。

````carousel
![Mg-MOF-74 0.28 bar 同预算 2×2 消融实验结果](assets/fig2_parity_efficiency.png)
<!-- slide -->
![跨 MOF 迁移性与 MACE-DAC-1 全压力等温线](assets/fig3_transferability_finetuned.png)
<!-- slide -->
![泛函敏感性、Widom 基准、MACE/D3 分解与探索性 1x1x4 检验](assets/fig4_physics_diagnostics.png)
<!-- slide -->
![76 点 Vehtari 收敛审计、REMC 交换率与蒸馏/采样 GPU 开销对比](assets/fig5_convergence_scaling.png)
````

## 代码结构

1. **结构与客体前处理**（[`scripts/qeq_charges.py`](scripts/qeq_charges.py)、[`scripts/guests.py`](scripts/guests.py)）：ASE 展开任意空间群并保留 CIF 电荷；无电荷框架采用带屏蔽的周期性 Ewald QEq；刚性客体支持最多 12 个位点，逸度由 Peng–Robinson 方程计算。
2. **代理势蒸馏与 GPU 提议器**（[`scripts/mlip_distill.py`](scripts/mlip_distill.py)、[`src/proposer_dll.cu`](src/proposer_dll.cu)、[`include/proposer_api.h`](include/proposer_api.h)）：1B 三维网格与截断 2B LUT 驻留显存，每模拟箱独占 1 个 CUDA 线程块（128 路并行 CBMC），经 C ABI 零开销交互并周期性全 FP64 重算消除累加漂移。
3. **嵌套 MC、$\mu$-REMC、收敛诊断与全轨迹归档**（[`scripts/nested_mc_engine.py`](scripts/nested_mc_engine.py)、[`scripts/mcmc_diagnostics.py`](scripts/mcmc_diagnostics.py)、[`scripts/multi_gpu.py`](scripts/multi_gpu.py)、[`scripts/archive_jctc_results.py`](scripts/archive_jctc_results.py)）：批量 MACE/D3 评估（FP32 + FP64 节点累加）、平衡期冻结边际漂移 $(\Delta\varepsilon, k)$ 校准、独立梯子 $\mu$-REMC、rank-normalized $\hat{R}$ / 体尾 ESS 判据及带 SHA-256 溯源的全轨迹归档校验。

## 快速开始

### 环境检测与编译

```powershell
# Windows：只检查依赖
cmd /c fast_start.bat -CheckOnly
# Windows：补齐依赖、编译并运行快速自检（加 -FullTest 跑完整经典采样分布对比）
cmd /c fast_start.bat
```

```bash
# Linux
./fast_start.sh --check-only
./fast_start.sh --gpu 0
```

### 运行嵌套 MC 与独立梯子 $\mu$-REMC（[`scripts/run_mlip_gcmc.py`](scripts/run_mlip_gcmc.py)）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv-mlip/bin/python -u scripts/run_mlip_gcmc.py \
  --cif data/IRMOF-1.cif --supercell 1 1 1 --guest co2 --temp 298 \
  --pressures 0.5 1.0 2.5 5.0 --proposer auto_mlip --cbmc-k 128 \
  --model build/models/mace-dac-1.model --dispersion on --dispersion-xc pbe \
  --boxes 4 --k-steps 24 --eq-cycles 800 --prod-cycles 16000 \
  --distill-cache-dir build/distill_cache --gpus 0 1 2 3 \
  --parallel-mode remc --remc-swap on --remc-swap-every 2 \
  --out-json build/irmof1_co2_macedac_remc.json
```

同预算 2×2 消融（`v1.4` 统一硬核，8 卡）与确定性审计 / Figure 2 复现命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv-mlip/bin/python -u scripts/run_jctc_presubmission.py prepare
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv-mlip/bin/python -u scripts/run_jctc_presubmission.py run-formal --budget-wall-s 7200 --ckpt-every-s 300 --skip-done
.venv-mlip/bin/python scripts/audit_jctc_presubmission.py --root build/jctc_presubmission
.venv-mlip/bin/python scripts/plot_phase4_figures.py --mode presubmission
```

Widom 插入与能量分解：

```bash
.venv-mlip/bin/python -u scripts/widom_mlip.py \
  --cif data/MgMOF-74.cif --supercell 1 1 2 --guest co2 --temp 298 \
  --model mace-matpes-r2scan-0 --dispersion off --n-samples 6000 \
  --out-json build/widom_mgmof74_r2scan.json
```

校验历史多 MOF 离线/Release 完整轨迹包（存在 `jctc_traces.npz` 时）并重新生成全部 5 张 JCTC 论文插图：

```bash
.venv-mlip/bin/python scripts/archive_jctc_results.py \
  --verify-summary docs/results/jctc_summary.json --traces-npz docs/results/jctc_traces.npz
.venv-mlip/bin/python scripts/plot_phase4_figures.py --mode jctc
```

### 经典与 `Pol-SCF` 回归测试

```powershell
.\build\mof_gcmc_cli.exe --test-scf
.\build\mof_gcmc_cli.exe --config configs\mg_mof74_co2_298K_uff.json
.\.venv-mlip\Scripts\python.exe -m pytest -v -m "not mlip and not slow"
```

## 文档索引

- **JCTC 审计与精度校验数据：** [`docs/results/jctc_audit_report.md`](docs/results/jctc_audit_report.md) · [`docs/results/jctc_summary.json`](docs/results/jctc_summary.json) · [`docs/results/precision_hc_audit.json`](docs/results/precision_hc_audit.json) · [`docs/results/phase2_phase3_summary.json`](docs/results/phase2_phase3_summary.json)
- **经典与极化基准验证（`RASPA3 3.0.29`）：** [经典与极化基准（中文）](docs/CLASSICAL_VALIDATION.zh-CN.md) · [English](docs/CLASSICAL_VALIDATION.md)

```text
cuda-mof-gcmc/
├── fast_start.bat / .sh  # 环境检测、依赖安装、编译与自检
├── build.bat / .sh       # MSVC+CUDA / nvcc+g++ 编译脚本
├── configs/              # mof_gcmc_cli 预设（UFF、Mercado、烟道气、Pol-SCF）
├── data/                 # CIF、Mercado 力场参数、实验等温线
├── docs/                 # 经典与极化验证文档及 results/ 审计汇总数据
├── include/              # CUDA/C++17 头文件与 proposer C ABI
├── src/                  # CUDA 内核与驱动；src/legacy/ 为旧版
├── scripts/              # run_mlip_gcmc、mlip_distill、widom_mlip、nested_mc_engine、multi_gpu、archive_jctc_results、plot_phase4_figures
├── tests/                # pytest 测试
└── assets/               # Fig. 1-5 论文插图（300 DPI）
```
