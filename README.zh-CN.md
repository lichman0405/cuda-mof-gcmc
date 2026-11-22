[English](README.md) | **简体中文**

# `cuda-mof-gcmc`: 面向开放金属位点 MOF 的 GPU 并行 GCMC 模拟引擎

面向 **NVIDIA GeForce RTX 5070**（`sm_120`）从零构建的纯 CUDA 金属有机框架（MOF）气体吸附巨正则蒙特卡洛（GCMC）引擎，核心聚焦开放金属位点（OMS）的物理吸附建模（典型基准体系：**Mg-MOF-74 / $\text{Mg}_2(\text{dobdc})$ 吸附 $\text{CO}_2$**）。

![Step 1 Mg-MOF-74 CO2 298K Results](assets/step1_mof74_co2_298K.png)

---

## 整体研发路线（三步走规划）

- **[已完成] 第一步：纯 CUDA 刚性 3D 骨架网格 + Batched GCMC 底座（UFF + REPEAT + TraPPE）**
  - 在 GPU 上一次性预计算 6 张周期 3D 单胞网格（`U_lj_C`、`U_lj_O`、`Phi_elec`、`Ex`、`Ey`、`Ez`；包含精确 12-6 LJ 与 3D Ewald 求和），随后以单 Block 128 线程并行跑 32 个独立压力/副本盒子（共享内存存客体分子、精确六方 MIC、Marsaglia/Rodrigues 球面无偏旋转、DSF 静电、Peng–Robinson 逸度换算），并校验细致平衡与副本一致性。
- **[已完成] 第二步（支线 B1）：$\text{Mg}^{2+}$ 开放金属位点显式静电场极化 + Thole 诱导偶极耦合**
  - 正式激活预计算的 `(Ex, Ey, Ez)` 三维电场网格与 $\text{CO}_2$ 各向异性极化率张量（$\alpha_\parallel = 4.05\ \text{\AA}^3,\ \alpha_\perp = 1.95\ \text{\AA}^3,\ \bar{\alpha} = 2.65\ \text{\AA}^3$），对比质心极化（`Pol-COM`）与三点分布极化（`Pol-3Site`，$w_C = 0.40, w_O = 0.30$）及客体间 Thole 阻尼诱导偶极作用（$a = 2.1304$）。
- **[规划中] 第三步：多组分竞争共吸附**
  - 加入组分身份交换（Identity Swap）动作，扩展至 $\text{CO}_2/\text{N}_2$、$\text{CO}_2/\text{CH}_4$ 与 $\text{CO}_2/\text{H}_2\text{O}$ 共吸附筛选。

---

## 第一步实测校验与物理结果（RTX 5070）

### 1. 晶体结构与几何校验（[`data/MgMOF-74.cif`](data/MgMOF-74.cif)）
- **单胞晶格（`R -3` 六方晶系）**：$a = b = 26.136\ \text{\AA},\ c = 6.942\ \text{\AA},\ \alpha = \beta = 90^\circ,\ \gamma = 120^\circ$
- **对称展开后单胞（`1x1x1`）**：`162` 原子（`18 Mg`、`54 O`、`72 C`、`18 H`），REPEAT 总电荷 = `0.000000 e`，单胞摩尔质量 = `2184.372 g/mol`
- **最短成键距离自检**：`Mg–O = 1.948 Å`、`C–O = 1.337 Å`、`C–C = 1.420 Å`、`C–H = 1.123 Å`
- **GCMC 超胞（`1x1x4`）**：$V = 16,426.82\ \text{\AA}^3$，质量 = `8737.488 g/mol`，最小垂直晶面间距 = `22.634 Å > 2 * 11.0 Å`
- **精确六方最小镜像（MIC）**：与 `3x3x3` 暴力穷举对比 **`0 / 10,000` 偏差**（C++ 与 Python 双重通过）

### 2. GPU 三维势能与电场网格预计算（`128 x 128 x 36`）
- **生成耗时**：**`356.79 ms`** 完成全部 6 张网格（`U_lj_C`、`U_lj_O`、`Phi_elec`、`Ex`、`Ey`、`Ez`，Ewald $\alpha = 0.30\ \text{\AA}^{-1},\ r_{\text{cut}} = 11.0\ \text{\AA}$）
- **独立对拍验证**（与不同 Ewald 参数 $\alpha_{\text{ref}} = 0.25\ \text{\AA}^{-1}$ 在 `1,000` 个孔道随机采样点直接求和对比）：
  - `MAE(U_lj_C) = 3.491 K`、`MAE(U_lj_O) = 10.482 K`、`MAE(Phi) = 4.4038 K/e`

### 3. Batched GCMC 性能与 298 K $\text{CO}_2$ 吸附等温线
- **运行配置**：`32` 个并行盒子（`0.005–20.0 bar` 共 `16` 个压力点 × `2` 独立副本），每盒 `100k` 平衡步 + `200k` 采样步（累计 **`960 万` MC 步**）
- **GPU 总耗时**：**`13.906 s`**（`0.69 M MC-steps/sec`）
- **典型压力点结果（UFF + REPEAT + TraPPE 基线）**：
  - `0.150 bar`（$\phi = 0.9992$）：**`1.1724 ± 0.0301 mmol/g`**，$Q_{\text{st}} = \mathbf{26.07\ \text{kJ/mol}}$（`acc_ins = acc_del = 0.09`）
  - `1.000 bar`（$\phi = 0.9945$）：**`6.7019 ± 0.0769 mmol/g`**，$Q_{\text{st}} = \mathbf{27.49\ \text{kJ/mol}}$（`acc_ins = acc_del = 0.05`）
  - `20.000 bar`（$\phi = 0.8934$）：**`14.7214 ± 0.1619 mmol/g`**，$Q_{\text{st}} = \mathbf{36.46\ \text{kJ/mol}}$
- **物理诊断**：与 MOF 经典文献完全一致，不带极化的通用 UFF+REPEAT 力场会显著低估 `0.15 bar` 低压吸附量（`1.17` vs 实验 `~3.6–4.6 mmol/g`）与零覆盖度吸附热（`~25.6` vs 实验 `~40–47 kJ/mol`）——这正是**第二步（支线 B1）引入极化效应**要解决的核心物理差异。

---

## 第二步（支线 B1）：$\text{Mg}^{2+}$ 开放金属位点各向异性极化（RTX 5070）

![Step 2 Mg-MOF-74 Polarization Comparison](assets/step2_mof74_pol_comparison.png)

### 1. 三维电场有限差分与插值精度校验
- **解析 Ewald 梯度校验**：在 `100` 个孔道点上对比解析 `E_direct` 与中心有限差分 $-\nabla\Phi_{\text{FD}}$（$h = 10^{-4}\ \text{\AA}$），误差低至 **`MAE = 2.21e-06 K/(e·Å)`**。
- **三维插值精度**：`128x128x36` 网格 `(Ex, Ey, Ez)` 与独立 $\alpha_{\text{ref}} = 0.25\ \text{\AA}^{-1}$ 在 `1,000` 个孔道点对比得 **`MAE(|E|) = 8.909 K/(e·Å)`**。

### 2. 三种模型同台实测对比（每组在 RTX 5070 上跑 `960 万` MC 步）

| 模型 | GPU 耗时 | `0.15 bar` 吸附量 (mmol/g) | `1.00 bar` 吸附量 (mmol/g) | `20.0 bar` 吸附量 (mmol/g) | 低压 $Q_{\text{st}}$ (kJ/mol) | 平均极化能 $U_{\text{pol}}/N$ (kJ/mol) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`NonPol`（第一步基线）** | `14.210 s` | `1.1724 ± 0.0301` | `6.7019 ± 0.0769` | `14.7214 ± 0.1619` | `25.5–26.1` | `0.00` |
| **`Pol-COM`（质心各向异性极化）** | `16.627 s` | `1.5406 ± 0.0125` | **`7.6421 ± 0.0038`** | `14.9957 ± 0.15` | `26.5–27.6` | `-0.67 → -0.45` |
| **`Pol-3Site`（第二步 B1 分布极化）** | `18.014 s` | **`2.3890 ± 0.0392`** (`+104%`) | `9.1446 ± 0.1418` | `15.3591 ± 0.43` | **`29.2–31.4`** | **`-2.50 → -1.58`** |

- **核心物理发现**：三点分布极化（`Pol-3Site`）能准确捕捉配位氧原子靠近 $\text{Mg}^{2+}$ 时的强烈 $r^{-4}$ 非均匀电场梯度，将平均极化能提升至质心模型的 **3.7 倍**（`-2.50` vs `-0.67 kJ/mol`），使 `0.15 bar` 烟道气分压下 $\text{CO}_2$ 吸附量直接翻倍（`1.17 → 2.39 mmol/g`），并在高压饱和过程中清晰展现 OMS 饱和特征（`-U_pol/N` 从 `2.50` 降至 `1.58 kJ/mol`）。全压力范围严格满足细致平衡（`max|acc_ins - acc_del| <= 0.0007`）。而距离实验 `40–47 kJ/mol` 剩余约 `10 kJ/mol` 的差距，则对应通用 UFF `sigma_Mg` 无法描述的近程轨道电荷转移与极化耦合（需结合 DFT 拟合 OMS 短程力场或 MLIP Nested MC）。

---

## 编译与运行命令

```powershell
# 针对 RTX 5070 (sm_120) 编译
cmd /c build.bat

# 运行第一步 + 第二步全部三组对比（约 49 秒，CSV 导出至 build/）
.\build\mof_gcmc.exe

# 运行第一步校验与出图
uv run --with matplotlib --with numpy python scripts\verify_physics.py

# 运行第二步校验并绘制 300 DPI 对比三联图
uv run --with matplotlib --with numpy python scripts\verify_step2.py
```

---

## 目录结构

```text
cuda-mof-gcmc/
├── data/
│   └── MgMOF-74.cif              # RASPA2 官方 R-3 Mg-MOF-74 晶体结构文件
├── include/
│   ├── cell.hpp                  # 六方/三斜晶胞坐标变换与精确 5 候选 MIC
│   ├── forcefield.hpp            # UFF、REPEAT、TraPPE CO2 参数与 Peng-Robinson 逸度计算
│   ├── grid3d.hpp                # 周期性三维网格视图（U_lj, Phi, Ex, Ey, Ez）与三线性插值
│   ├── cif_loader.hpp            # CIF 对称展开与几何检查接口
│   └── gcmc_gpu.hpp              # 网格构建与 Batched GCMC 调用接口
├── src/
│   ├── cif_loader.cpp            # 18 对称操作展开 -> 162 原子与键长/电中性校验
│   ├── grid_builder.cu           # GPU 3D LJ + Ewald 静电势与电场网格预计算
│   ├── gcmc_kernel.cu            # 多盒子 GPU GCMC 主核函数（128 线程/盒，DSF + 细致平衡）
│   └── main.cu                   # 主程序、单元测试、32 副本等温线模拟与 CSV 导出
├── scripts/
│   ├── verify_physics.py         # 独立 MIC/PR 校验与第一步 300 DPI 出图脚本
│   └── verify_step2.py           # 第二步极化校验与 300 DPI 对比三联图脚本
├── assets/
│   ├── step1_mof74_co2_298K.png  # 第一步：298 K 等温线、Qst 吸附热与孔道密度图
│   └── step2_mof74_pol_comparison.png  # 第二步：NonPol / Pol-COM / Pol-3Site 对比三联图
├── build.bat                     # VS2019 + nvcc (sm_120) 编译脚本
├── README.md                     # 英文项目文档
└── README.zh-CN.md               # 中文项目文档
```
