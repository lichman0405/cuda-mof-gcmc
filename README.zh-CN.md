[English](README.md) | **简体中文**

# `cuda-mof-gcmc`: 面向开放金属位点 MOF 的 GPU 并行 GCMC 模拟引擎

面向 **NVIDIA GeForce RTX 5070**（`sm_120`）从零构建的纯 CUDA 金属有机框架（MOF）气体吸附巨正则蒙特卡洛（GCMC）引擎，核心聚焦开放金属位点（OMS）的物理吸附建模（典型基准体系：**Mg-MOF-74 / $\text{Mg}_2(\text{dobdc})$ 吸附 $\text{CO}_2$**）。

![Step 1 Mg-MOF-74 CO2 298K Results](assets/step1_mof74_co2_298K.png)

---

## 整体研发路线（三步走规划）

- **[已完成] 第一步：纯 CUDA 刚性 3D 骨架网格 + Batched GCMC 底座（UFF + REPEAT + TraPPE）**
  - 在 GPU 上一次性预计算 6 张周期 3D 单胞网格（`U_lj_C`、`U_lj_O`、`Phi_elec`、`Ex`、`Ey`、`Ez`；包含精确 12-6 LJ 与 3D Ewald 求和），随后以单 Block 128 线程并行跑 32 个独立压力/副本盒子（共享内存存客体分子、精确六方 MIC、Marsaglia/Rodrigues 球面无偏旋转、DSF 静电、Peng–Robinson 逸度换算），并校验细致平衡与副本一致性。
- **[下一步] 第二步（支线 B1）：$\text{Mg}^{2+}$ 开放金属位点显式静电场极化**
  - 直接读取第一步已预留好的 `(Ex, Ey, Ez)` 三维电场网格，接入 $\text{CO}_2$ 各向异性极化率张量（$\alpha_\parallel, \alpha_\perp$）与客体诱导偶极自洽/OMS 修正参数，捕获低压区 $\text{Mg}^{2+}$ 强吸附陡升行为。
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

## 编译与运行命令

```powershell
# 针对 RTX 5070 (sm_120) 编译
cmd /c build.bat

# 运行第一步 GPU Batched GCMC（约 14 秒，结果导出至 build/）
.\build\mof_gcmc.exe

# 运行 Python 独立物理校验并绘制 300 DPI 三联图
uv run --with matplotlib --with numpy python scripts\verify_physics.py
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
│   └── verify_physics.py         # 独立 MIC/PR 校验与 300 DPI 科研出图脚本
├── assets/
│   └── step1_mof74_co2_298K.png  # 298 K 等温线、Qst 吸附热与 1 bar 六方孔道密度分布图
├── build.bat                     # VS2019 + nvcc (sm_120) 编译脚本
├── README.md                     # 英文项目文档
└── README.zh-CN.md               # 中文项目文档
```
