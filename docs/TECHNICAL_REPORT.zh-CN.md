[English](TECHNICAL_REPORT.md) | **简体中文** | [返回 README](../README.zh-CN.md)

# 基于零 I/O CUDA 提议与化学势偏置嵌套蒙特卡洛的金属有机框架 CO₂ 吸附混合模拟框架及通用机器学习势评估

**摘要**

通用机器学习原子间势（MLIP）为摆脱金属有机框架（MOF）气体吸附模拟中逐材料拟合第一性原理力场的局限提供了可行路径，但将其直接耦合至巨正则蒙特卡洛（GCMC）系综时，通常受限于高昂的神经网络推理开销与大量骨架重叠试探，而传统嵌套蒙特卡洛（Nested MC）在能量分布差异较大的体系中又容易出现插入与删除不对称停滞。针对上述问题，本文在消费级单卡 GPU（NVIDIA GeForce RTX 5070）上构建了 `cuda-mof-gcmc` 混合模拟引擎：一方面实现了支持 $O(h^4)$ 三维 Tricubic 势能/电场插值与 Block 级共享内存 Applequist–Thole 多体自洽极化（`Pol-SCF`）的高吞吐经典 CUDA GCMC 求解器；另一方面将该求解器封装为零磁盘 I/O 动态库，与批处理 `MACE + TorchDFTD3` 推理端在内存中直连，并通过采样前两阶段经典初估加目标系综预热修正（`PilotStage3`）标定常数化学势偏置，在严格保留目标 MLIP 巨正则分布的前提下恢复了插入与删除试探的双向平衡。数值核验表明，经典 Tricubic 内核与单核 `RASPA3 3.0.29` 在 `Mg-MOF-74`（298 K）上的统计偏差不超过 $0.54\sigma$，单卡吞吐达到 $598,669\ \text{MC-steps/s}$（较同算法单核 CPU 网格版加速 $153.2\times$）；`Pol-SCF` 与精确 `36×36` 矩阵求逆的误差分别为 $1.83 \times 10^{-8}\ \text{K}$ 与 $1.13 \times 10^{-11}$ e·Å。在此基础上，本文系统检验了跨力场零偏差重加权（$z = 0.27\sigma$）、`float32` 推理噪声上界（成对转移误差 $\le 0.190\ k_B T$）与超胞有限尺寸收敛性（$z = 0.04\sigma$ 与 $0.63\sigma$），并在巨正则系综下评估了四代通用 MACE 模型及四类典型 MOF（`Mg-MOF-74`、`Ni-MOF-74`、`HKUST-1` 与无开放金属位点对照 `IRMOF-1`）。结果表明：`MACE-MPA-0` 与 `MACE-MP-0b3` 在孔道远场（7.31 Å）误差可忽略，但在接近 Mg²⁺ 开放金属位点（2.20–2.35 Å）时存在非物理短程过结合塌陷（$-93.98$ 与 $-105.17\ \text{kJ/mol}$）；即使单点势阱相对合理的 `MACE-MatPES-r2SCAN-0 + D3(r2scan)`，在对比 NIST ISODB 原始实验数据时仍于 `Mg-MOF-74`（1.8–3.0 倍）及无开放金属位点的 `IRMOF-1`（2.5–2.9 倍）上均呈现系统性偏高，而 `HKUST-1` 的表观接近（298 K, 1.0 bar 下 $4.098\ \text{mmol/g}$ 对实验 $4.530\ \text{mmol/g}$）则伴随 Cu²⁺ 桨轮轴向结合偏弱（$-10.07\ \text{kJ/mol}$）与有机骨架/小笼色散偏强之间的部分抵消。

**关键词**：金属有机框架；巨正则蒙特卡洛；机器学习原子间势；嵌套蒙特卡洛；开放金属位点；CUDA

---

## 1. 引言

高比表面积金属有机框架（MOF）材料在烟道气碳捕集与温室气体分离方面具有重要应用价值，其中含配位不饱和开放金属位点（OMS）的体系（如 M-MOF-74 系列和 HKUST-1）因局部静电场梯度与金属–CO₂ 轨道相互作用显著增强，在低分压区表现出突出的选择性吸附能力[1, 2]。然而，通用固定电荷经典力场（如 UFF 结合 REPEAT/DDEC 电荷及 TraPPE 客体模型）往往难以定量刻画强局域诱导极化及配位作用，常低估 OMS-MOF 的低压吸附量与等量吸附热；而依赖周期性密度泛函理论（DFT）拟合特定金属位点势能面的方法（如 Mercado 等针对 Mg-MOF-74 提出的 DFT-FF[3]）虽精度较高，却缺乏跨金属与跨拓扑的可迁移性。

近年来，基于图神经网络的通用机器学习原子间势（MLIP，如 MACE 系列基础模型[4]）覆盖了周期表大多数主族与过渡金属元素，有望在无需逐一拟合参数的前提下用于全晶胞吸附模拟。然而，直接将 MLIP 接入常规 GCMC 存在两项实际困难：其一，常规 GCMC 随机插入在致密微孔材料中约 90%–95% 会与框架原子发生空间重叠，直接调用神经网络推理将耗费大量无效算力；其二，若采用经典参考势作为内层采样器的嵌套蒙特卡洛（Nested MC）框架，当经典势与目标 MLIP 的单分子结合能差异达到数个 $k_B T$ 以上时，内外层在相同外压下的平衡吸附量相差悬殊，导致外层只接收插入而不接收删除，马尔可夫采样无法遍历目标巨正则分布。此外，面向纯经典固定电荷力场的 GPU 模拟器（如 `gRASPA`[5]）虽然具备优异的高通量筛选性能与柔性链分子 CBMC 支持，但其设计重心在于经典力场加速，并未针对耦合 PyTorch MLIP 推理提供零磁盘 I/O 内存直连提议接口与全自洽多体极化求解内核。

针对上述问题，本文构建并验证了一套基于单卡 GPU 的 `cuda-mof-gcmc` 混合模拟框架，核心工作如下：

1. 在经典模拟端，实现了具备 $O(h^4)$ 三维 Tricubic 插值、双精度周期性全盒重算及 Block 级共享内存 Applequist–Thole 多体自洽极化（`Pol-SCF`）的 CUDA 求解器，并完成与原生 CPU `RASPA3 3.0.29` 的背靠背精度及吞吐对比；
2. 将高速经典 CUDA 内核封装为无磁盘 I/O 动态库作为底层构型提议器，结合经三阶段化学势偏置（$\Delta\mu_{\text{ref}}$）标定与微批图推理优化的 `FastMaceD3Target` 引擎，完成跨力场零偏差、浮点推理精度边界与超胞有限尺寸收敛性验证；
3. 借助该引擎对四代 MACE 基础模型开展机理剖析与完整巨正则系综采样，并以 NIST ISODB 归档的原始实验等温线为参照，系统对比了 `Mg-MOF-74`、`Ni-MOF-74`、`HKUST-1` 以及无开放金属位点对照骨架 `IRMOF-1`，揭示了新版基础模型的短程塌陷行为与背景色散/过结合误差来源。

---

## 2. 理论方法与软件实现

### 2.1 三维 $O(h^4)$ Tricubic 势能网格与零漂移经典内核

刚性框架近似下，单分子与周期性 MOF 骨架之间的 Lennard-Jones（或 Buckingham）短程作用及长程 Ewald 静电势 $U_{\text{fw}}(\mathbf{r})$ 与静电场 $\mathbf{E}_{\text{fw}}(\mathbf{r}) = -\nabla \Phi_{\text{elec}}(\mathbf{r})$ 均仅依赖于客体原子在单胞内的分数坐标。`cuda-mof-gcmc` 在 GPU 初始化阶段于细密三维规则网格（典型分辨率 $0.10$ Å，对应 `Mg-MOF-74` 为 $256 \times 256 \times 72$）上预计算各客体原子通道与静电势/电场分量，并在 GCMC 步进过程中通过四点三次拉格朗日模板执行 $O(h^4)$ 精度三维 Tricubic 插值。客体分子间的短程截断与 Damped Shifted Force（DSF）静电作用在 GPU 共享内存（Shared Memory）内按双精度（FP64）在线累积，坐标亦保持双精度表示；在六方或三斜晶胞下，内核采用严格的 5 候选最小镜像约定（MIC）。为防止增量能量累积在长程运行中产生舍入漂移，系统每推进 $50,000$ 步即依据当前坐标对 `u_fw + u_gg` 执行一次完整从头重算（full recompute）。外压与气相逸度的换算由单组分及多组分 Peng–Robinson（PR）状态方程实时给出。

### 2.2 各向异性极化与多体自洽 `Pol-SCF` 求解

围绕 OMS 孔道强局部电场梯度，经典求解器除支持非极化（`NonPol`）外，还实现了两层级显式极化：

1. **一阶骨架静电场极化（`Pol-COM` 与 `Pol-3Site`）**：将分子各向异性极化率张量分别置于质心或分散至客体三个原子位点（捕获配位氧原子在 $r^{-4}$ 陡峭梯度下的极化增强），并保留一阶极化与客体永久电荷间的静态交互；
2. **Level-2 多体全自洽偶极极化（`polarization: "scf"`）**：在每次 GCMC 移动、插入或删除试探中，联合计算所有框架诱导、客体永久电荷 DSF 电场以及 Thole 短程阻尼偶极–偶极交互张量 $\mathbf{T}_{ij}^{\text{Thole}}$，通过 Block 级共享内存迭代求解如下线性自洽方程直至收敛容差 $|\Delta \boldsymbol{\mu}| < 10^{-10}$ e·Å：

$$
\boldsymbol{\mu}_i = K_E^{-1}\boldsymbol{\alpha}_i\left(\mathbf{E}_i^{\text{fw}} + \sum_{j \neq i}\mathbf{E}_{j \to i}^{\text{DSF,perm}} + \sum_{j \neq i}\mathbf{T}_{ij}^{\text{Thole}}\boldsymbol{\mu}_j\right),\qquad U_{\text{pol}}^{\text{SCF}} = -\frac{1}{2}\sum_i \boldsymbol{\mu}_i \cdot \mathbf{E}_i^0
$$

内核同时采用 $O(N)$ 增量永久电场更新与历史偶极子热启动以降低单步迭代次数，并引入试探前近距排斥预筛与 `s_scf_conv` 收敛守卫保证细致平衡不被局部奇异构型破坏。

### 2.3 带化学势偏置的嵌套蒙特卡洛与三阶段偏置标定

为使经典 CUDA 提议势 $U_{\text{ref}}$ 严格采样目标机器学习势 $U_{\text{tgt}}$ 对应的巨正则系综，本框架采用固定常数化学势偏置的 Nested MC 架构。设目标化学势对应气相逸度为 $f_{\text{tgt}}$，在进入正式平衡与生产采样前，通过如下机制将内层 CUDA 提议器的有效逸度预设为：

$$
f_{\text{ref}} = f_{\text{tgt}} \exp\left(-\frac{\Delta\varepsilon}{k_B T}\right)
$$

其中 $\Delta\varepsilon$ 为内层相对目标势的单分子平均能量偏移估计值。每个外层周期内，CUDA 提议器以完全可逆的移动概率、固定步宽（$\Delta_{\text{trans}} = 1.0$ Å、$\Delta_{\text{rot}} = 0.55\ \text{rad}$）与末尾全双精度能量重算连续推进 $K$ 步经典 GCMC（包含平移、旋转、插入与删除，而非仅限于单一移动类型），随后仅将第 $K$ 步的整盒终点构型 $\mathbf{x}'$ 交付目标 MLIP 计算 $U_{\text{tgt}}(\mathbf{x}')$。外层 Metropolis 接收概率定义为：

$$
P_{\text{acc}}(\mathbf{x} \to \mathbf{x}') = \min\left\lbrace 1,\ \exp\left[-\beta\left((U_{\text{tgt}}(\mathbf{x}') - U_{\text{tgt}}(\mathbf{x})) - (U_{\text{ref}}(\mathbf{x}') - U_{\text{ref}}(\mathbf{x})) - \Delta\varepsilon\,\Delta N\right)\right]\right\rbrace
$$

若外层拒绝，则由 CUDA 动态库通过状态缓冲区将该模拟盒的完整坐标、分子数与能量回滚至本次 $K$ 步脉冲之前的状态 $\mathbf{x}$。由于 $\Delta\varepsilon$ 在进入正式平衡与生产采样周期之前即固定为常数，内层提议偏置项 $\exp(-\beta\,\Delta\varepsilon\,\Delta N)$ 与外层判据中的修正因子相互抵消，因此在理论极限下该马尔可夫链满足相对于真实目标系综 $\Pi(\mathbf{x}) \propto (z_{\text{tgt}}^N / N!) \exp[-\beta U_{\text{tgt}}(\mathbf{x})]$ 的严格细致平衡。

实际应用中，若直接依赖初始经典孔道负载估计 $\Delta\varepsilon$，当经典力场与目标 MLIP 孔道优势位点不一致时（例如 `HKUST-1` 中经典电荷势高估 Cu²⁺ 作用而 MLIP 偏好小笼位点），经典初估可能产生过高的正偏置并压缩内层有效逸度，导致正式采样阶段爬升缓慢。为此，[`scripts/run_mlip_gcmc.py`](../scripts/run_mlip_gcmc.py) 实现了三阶段采样前标定流程（`Stage 1 -> Stage 2 -> PilotStage3`）：

1. **Stage 1 & Stage 2（经典预估）**：在经典提议器上分别运行 `20,000` 与 `30,000` 步（若初始负载 $N < 4$ 则施加临时负偏移充孔），基于抽样构型的平均单分子能量差更新 $\Delta\varepsilon_1$ 与 $\Delta\varepsilon_2$，并将纯经典正偏移上限钳位在 $+2.0\ \text{kJ/mol}$ 以避免过度抑制内层插入；
2. **PilotStage3（目标系综预热修正）**：以 $\Delta\varepsilon_2$ 预执行 `60–80` 轮实景 MLIP Nested MC（不计入生产统计），使分子弛豫至目标 MLIP 实际偏好的热力学位点，据此采集真实系综均值并作半阻尼更新 $\Delta\varepsilon_{\text{final}} = 0.5\,\Delta\varepsilon_2 + 0.5\,\Delta\varepsilon_{\text{live}}$；随后锁定 $\Delta\varepsilon_{\text{final}}$ 不再变动，重新运行 $10,000$ 步内层预热并进入正式 `eq_cycles + prod_cycles` 采样。

### 2.4 零 I/O CUDA 提议动态库与批处理 MLIP 推理

工程架构由两层零磁盘交互接口构成（详见 [`include/proposer_api.h`](../include/proposer_api.h)、[`src/proposer_dll.cu`](../src/proposer_dll.cu) 与 [`scripts/nested_mc_engine.py`](../scripts/nested_mc_engine.py)）：

- **底层提议动态库（`mof_gcmc_proposer.dll`）**：常驻显存维护 $B$ 个独立副本盒子的完整网格、坐标、内部能量与状态快照，提供 `proposer_create`、`proposer_set_chem_pot_shift`、`proposer_equilibrate`、`proposer_propose_k`、`proposer_commit` 与 `proposer_eval_energy` 等 C ABI 接口，经由 Python `ctypes` 直接交换双精度坐标与能量数组，彻底消除逐帧导出 CIF/XYZ 文件及重复构建势能网格的开销。
- **批处理目标势求值器（`FastMaceD3Target`）**：针对原生 ASE 接口逐盒重建计算图与关闭批处理的延迟瓶颈，直接调用关闭梯度与力反向传播（`torch.no_grad()`, `compute_force=False`）的 MACE 图前向接口对多副本构型作微批打包（常规超胞 `mace_chunk=2`，当框架原子数 $\ge 550$ 时自动切换为 `1` 以控制单卡 12 GB 显存峰值）；对于 DFT-D3(BJ) 色散项，因刚性框架原子在全部采样周期内保持静止，引擎在初始化时即在 GPU 上预计算并缓存周期性框架–框架配对边表与偏移向量，每轮仅在线增量构建框架–客体与客体–客体配对距离，并将所有周期计算置于统一 `TorchDFTD3Calculator` 参数下执行。

### 2.5 微分等量吸附热与积分摩尔焓的热力学定义

在 GCMC 系综中，两种能量统计量具有明确的物理区分，本文均予以分别报告而不作混淆：

- **微分等量吸附热**（$Q_{\text{st}}^{\text{diff}}$）：由粒子数–能量协同涨落关系计算，表征恒温下再吸附第 $(N+1)$ 个分子时的偏摩尔焓变：

$$
Q_{\text{st}}^{\text{diff}} = RT - \frac{\langle U N \rangle - \langle U \rangle \langle N \rangle}{\langle N^2 \rangle - \langle N \rangle^2}
$$

- **平均积分摩尔吸附焓**（$Q_{\text{int}}^{\text{molar}}$）：反映当前负载下全部已吸附分子的平均结合强度，即 $-\langle U / N \rangle + RT$（实际实现为 $RT - \langle U \rangle / \langle N \rangle$）。在零覆盖度极限或单一位点均匀体系中，两者数值接近；但当超过 1:1 开放金属位点饱和容量（例如 `Mg-MOF-74` 对应 $8.24\ \text{mmol/g}$）后，新增客体分子被迫占据弱吸附配体/次级孔道位点，微分热 $Q_{\text{st}}^{\text{diff}}$ 将必然下降，而积分摩尔焓仍维持于较高均值，此时两者并非同一物理量的替代估计量。

---

## 3. 数值检验与计算性能评估

### 3.1 经典内核与 `RASPA3 3.0.29` 对标及 `Pol-SCF` 核验

在单张 RTX 5070 GPU 上对 `Mg-MOF-74`（298 K，UFF + REPEAT + TraPPE 基线及 Mercado 2016 DFT-FF）开展网格精度、长时漂移、多体极化矩阵对齐与原生 CPU `RASPA3 3.0.29` 并行副本对标测试，核心量化结果如下（见图 1 与表 1）：

- **网格预计算与插值误差**：构建 `0.10 Å` 分辨率 $256 \times 256 \times 72$ 六通道三维势能/电场网格耗时 `3.03 s`（113 MB 显存），较单核 CPU `RASPA3` 构建 $151 \times 151 \times 46$ 网格耗时（`117.98 s`）加速 **`38.9×`**；近程 `Mg–O` 径向能量插值平均绝对误差（MAE）较 $0.20$ Å 三线性插值下降 **`290.3×`**（由 `190.46 K` 降至 **`0.656 K`**，Mercado 2016 降至 **`0.092 K`**）。
- **零能量漂移与细致平衡**：$128$ 盒累计 `6,400 万` MC 步下，周期性全盒重算将最大绝对能量漂移抑制在 **`5.879e-09 K`**（UFF）和 **`1.024e-08 K`**（Mercado 2016），全工况增删接受率差值 `max|acc_ins - acc_del| <= 0.000253`。
- **吞吐性能**：CUDA `v2` Tricubic 内核达到 **`598,669 MC-steps/s`**（`6,400 万`步耗时 `106.90 s`），对比执行同 Tricubic 网格算法的单核 CPU `RASPA3 3.0.29`（`3,908 MC-steps/s`）加速 **`153.2×`**，对比无网格单核 `RASPA3` 精确 Ewald 算法（`1,215 MC-steps/s`）加速 **`492.6×`**。
- **Level-2 `Pol-SCF` 矩阵求逆与筛选精度**：`--test-scf` 内置核验显示，GPU Block 级共享内存 Applequist–Thole 自洽求解器与 Python 端精确 `36×36` 矩阵求逆的极化能偏差为 **`1.83e-08 K`**、诱导偶极最大偏差为 **`1.13e-11 e·Å`**；在 298 K、`15:85` CO₂/N₂ 烟道气中，多体相互退极化屏蔽使 CO₂/N₂ 吸附选择性由单骨架场 `Pol-3Site` 的 `64.58`（`1.0 bar`）与 `67.38`（`2.0 bar`）回落至 **`37.28`** 与 **`43.24`**（见图 2）。

**表 1** `Mg-MOF-74`（298 K, UFF+REPEAT）基准下 CUDA `v2` Tricubic 内核与 `RASPA3 3.0.29` 对齐检验结果

| 压力 | `v2` `0.10 Å` Tricubic (mmol/g) | `RASPA3` 精确 Ewald (mmol/g) | `RASPA3` Tricubic 网格 (mmol/g) | `v2` vs. 网格相对偏差 ($z$) | $\Delta Q_{\text{st}}$ (kJ/mol) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **`0.15 bar`** | **`1.4276 ± 0.0123`** | `1.4087 ± 0.0328` | `1.3952 ± 0.0719` | `+2.32%` (`0.44σ`) | `+0.14` |
| **`1.00 bar`** | **`7.6349 ± 0.1576`** | `7.3384 ± 0.1329` | `7.5303 ± 0.1156` | `+1.39%` (`0.54σ`) | `-0.38` |
| **`5.00 bar`** | **`13.1862 ± 0.0221`** | `13.3048 ± 0.2118` | `13.2402 ± 0.1871` | `-0.41%` (`0.29σ`) | `+0.54` |

![Step 4 Validation: 0.10 A Tricubic Grid, RASPA3 & Mercado 2016 DFT-FF](../assets/step4_validated_mercado_raspa3.png)

*图 1 `Mg-MOF-74`（298 K）下 `0.10 Å` 三维 Tricubic CUDA 内核、`RASPA3 3.0.29` 基准与 `Mercado 2016` DFT 力场对标结果。*

![Level 2 Pol-SCF Visualization](../assets/step5_level2_scf_polarization.png)

*图 2 Level-2 全自洽多体极化（`Pol-SCF`）纯组分等温线、CO₂/N₂ 相互退极化屏蔽效应及 `36×36` 矩阵求逆对齐校验。*

### 3.2 Nested MC 零偏差证明、化学势偏置与推理加速

在进一步引入神经网络势之前，先在精确已知的解析力场间及批处理接口上验证重加权统计无偏性与加速效率（图 3）：

1. **DLL 算能一致性与零偏差证明**：`proposer_eval_energy` 返回的参考势能与 CUDA 内核内部累计 `U_ref_K` 差异为 **`0.000e+00 K`**。在 `Mg-MOF-74`（`1×1×2`、`298 K, 0.15 bar`、4 副本）上以 `Pol-3Site` 为提议势、以 `Mercado 2016` 为目标势运行无偏置 Nested MC（$K = 15$，`2,000` 平衡 + `10,000` 生产轮），测得吸附量为 **`5.9668 ± 0.2072 mmol/g`**、$Q_{\text{st}}^{\text{diff}} = 35.05 \pm 1.27\ \text{kJ/mol}$，与直接 `Mercado 2016` GPU GCMC 参考值（`6.0230 ± 0.0406 mmol/g`、$Q_{\text{st}}^{\text{diff}} = 36.28 \pm 0.25\ \text{kJ/mol}$）完全吻合（**`z = 0.27σ`**）。
2. **偏置条件下无偏性及死锁解除**：在同一验证组中施加化学势偏移 `Δε = -2.57 kJ/mol`（$2.82\times$ 内层逸度），Nested MC 给出 **`5.8723 ± 0.0526 mmol/g`**（$Q_{\text{st}}^{\text{diff}} = 35.55 \pm 0.63\ \text{kJ/mol}$），且外层插入与删除实现完全对称采样（`Ins = 1,288 / 4,274`、`Del = 1,262 / 3,511`、总外层接受率 `57.5%`）。相比之下，直接以未加偏置的 `Mercado 2016` 提议实景 `MACE-MP-0 + D3`（`0.15 bar`）在 450 轮内产生 **`0` 次成功删除**（分子数自 `30` 单调增至 `41`）；引入两阶段偏移 `Δε = -9.19 kJ/mol`（`40.84×` 逸度）后，在 `0.01 bar` 下恢复了双向可逆交换（`Ins = 64 / 82`、`Del = 40 / 436`、外层接受率 `50.6%`），4 副本稳态收敛至 `9.7771 ± 0.2617 mmol/g`。
3. **批处理推理吞吐**：在 4 副本 `Mg-MOF-74 1×1×2` 上，`FastMaceD3Target` 将平均每副本推理耗时由 ASE 单盒顺序调用的 `157.7 ms/box` 降低至 **`37.5 ms/box`**（整批 `149.9 ms`，加速 **`4.21×`**），与完整 ASE 计算的总能量偏差仅 **`0.0025–0.0082 kJ/mol/CO2`**。

![Phase 2.5 Shifted & Batched Nested MC](../assets/phase25_shifted_batched_mace.png)

*图 3 化学势偏置（$\Delta\mu_{\text{ref}}$）零偏差重加权校验、`FastMaceD3Target` 四副本批处理加速及 `MACE-MP-0 + D3` 双向平衡轨迹。*

### 3.3 `float32` 推理误差边界与超胞有限尺寸收敛性

针对实际运行采用 `float32` 精度（以兼顾显存与推理吞吐）及截断半径受超胞尺寸限制两项数值隐患，分别开展了定量控制实验（图 4 与表 2）：

- **`float32` vs. `float64` 精度误差边界**：基于 `CudaGcmcProposer` 在 `Mg-MOF-74`（`1×1×2`）三个典型负载区间（`N = 1`、`13–16` 与 `33–37` 个 CO₂，总原子数 327–435）生成 15 个非重叠构型，分别以在线 `float32 FastMaceD3Target` 与全 `float64 MACE + TorchDFTD3` 求值。结果显示，`MACE-MatPES-r2SCAN-0 + D3(r2scan)` 的总结合能最大绝对偏差为 **`0.2058 kJ/mol`**（`RMS = 0.0951 kJ/mol`），单分子偏差上限 **`0.0940 kJ/mol/CO2`**，任意两构型间相对能量差 $|\Delta\Delta U|$ 最大值 **`0.3500 kJ/mol`**（即 `0.141 kB*T @ 298 K`）；`MACE-MP-0 (Medium) + D3(pbe)` 相应上限分别为 **`0.4650 kJ/mol`**、**`0.3130 kJ/mol/CO2`** 与 **`0.4712 kJ/mol`**（`0.190 kB*T @ 298 K`）。单精度引入的能量涨落远低于室温热能尺度，不会破坏外层 Metropolis 采样权重。
- **超胞有限尺寸收敛性**：在 `Mg-MOF-74`（`298 K, 0.01 bar`）下对比 `1×1×2`（324 骨架原子，$c = 13.884$ Å，等价客体–客体最小镜像截断 $6.66$ Å）与倍增超胞 `1×1×4`（648 骨架原子，$c = 27.768$ Å，截断 $10.86$ Å）。直接 `Mercado 2016` 给出 `0.8018 ± 0.0115` vs. `0.8030 ± 0.0337 mmol/g`（**`z = 0.04σ`**）；实时 `MACE-MP-0 + D3` Nested MC 给出 **`9.3398 ± 0.3610`** vs. **`9.0615 ± 0.2539 mmol/g`**（**`z = 0.63σ`**），积分摩尔焓分别为 `44.99 ± 0.54` 与 `45.01 ± 0.21 kJ/mol`，证实 `1×1×2` 超胞未引入显著的有限尺寸截断伪影。

**表 2** 在线 `float32 FastMaceD3Target` 与 `float64` 全精度参考对比（`Mg-MOF-74 1×1×2`, 15 构型）

| 模型组合 | 最大总能量差 (kJ/mol) | 总能量 RMS 误差 (kJ/mol) | 最大单分子差 (kJ/mol/CO2) | 最大成对转移误差 (kJ/mol) | 对应 298 K 热能尺度 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`MACE-MatPES-r2SCAN-0 + D3(r2scan)`** | `0.2058` | `0.0951` | `0.0940` | `0.3500` | `0.141 kB*T` |
| **`MACE-MP-0 (Medium) + D3(pbe)`** | `0.4650` | `0.2528` | `0.3130` | `0.4712` | `0.190 kB*T` |

![Stage-1 FP32 Parity, Differential Qst Convergence & Thermal Desorption](../assets/stage1_qst_fp32_diagnostics.png)

*图 4 在线 `float32` vs. `float64` 能量误差上界、`Mg-MOF-74` 1,200 轮涨落微分热收敛对比及 298 K / 313 K 热脱附检查。*

---

## 4. 通用 MACE 基础模型诊断与四类 MOF 吸附基准评估

### 4.1 远场零偏移校验与四代 MACE 短程过结合行为

在 `Mg-MOF-74` 上横向对比了 `MACE-MP-0 (Medium)`、`MACE-MPA-0`、`MACE-MP-0b3` 与 `MACE-MatPES-r2SCAN-0` 四代通用模型在加载态集合（`N ≈ 25`）及静力构型上的物理表现（图 5 与表 3）。为排除长程基线漂移或分子孤立参考能误差的干扰，首先计算了处于一维孔道正中、距离最近框架原子 $r_{\min} = 7.31$ Å（显著大于 MACE $6.0$ Å 消息传递截断半径）的单分子 CO₂ 结合能：关闭 D3 时，四个模型的远场残余交互分别为 **`+0.01`**（`MP-0`）、**`+0.00`**（`MPA-0`）、**`-0.21`**（`MP-0b3`）与 **`-0.03 kJ/mol`**（`MatPES-r2SCAN-0`），而开启 D3 均给出物理一致的长程色散尾项 `-2.38` 至 `-2.56 kJ/mol`。这直接证明各个模型之间的巨大能标差异并非源于孤立参考能或远场截断基线偏移。

然而，沿无遮挡轴向 `Mg²⁺···O=C=O` 接近曲线作静态扫描后发现，版本更晚近的 `MACE-MPA-0` 与 `MACE-MP-0b3` 在靠近配位中心（$r_{\min} = 2.20$–$2.35$ Å）时表现出严重短程塌陷：即便完全关闭 D3 色散修正，二者势阱深度仍分别高达 **`-75.26`** 与 **`-85.48 kJ/mol`**，叠加 D3(PBE) 后更深达 **`-93.98`** 与 **`-105.17 kJ/mol`**，对应负载集合平均结合强度达 `-82.84` 与 `-92.15 kJ/mol`，远高于真实物理吸附窗口，直接造成 GCMC 孔道满饱和锁死。相比之下，较早的 `MACE-MP-0 + D3(PBE)` 轴向势阱为 `-48.74 kJ/mol`，而元广义梯度近似（meta-GGA）训练的 `MACE-MatPES-r2SCAN-0` 在无 D3 时势阱为 `-35.39 kJ/mol`，搭配 `D3(xc="r2scan")` 后位于 **`-46.45 kJ/mol`**（加载态 `-U/N = 40.13 kJ/mol`），单分子静态特征最接近通常认为的物理吸附范围。

**表 3** 四代 MACE 基础模型在 `Mg-MOF-74` 孔道中心远场（`7.31 Å`）、无遮挡 `Mg²⁺` 径向极值与加载集合下的诊断对比

| 模型配置 | 远场能量 `7.31 Å` (kJ/mol) | 轴向 `Mg²⁺` 极小值 (kJ/mol) | 极小距离 (`Å`) | 加载态平均 `-U/N` (kJ/mol) |
| :--- | :---: | :---: | :---: | :---: |
| **`MACE-MP-0` (No D3 / `+D3(PBE)`)** | `+0.01` / `-2.48` | `-29.31` / `-48.74` | `2.35` / `2.20` | `—` / `41.63` |
| **`MACE-MPA-0` (No D3 / `+D3(PBE)`)** | `+0.00` / `-2.38` | **`-75.26`** / **`-93.98`** | `2.35` / `2.35` | `—` / **`82.84`** |
| **`MACE-MP-0b3` (No D3 / `+D3(PBE)`)** | `-0.21` / `-2.56` | **`-85.48`** / **`-105.17`** | `2.20` / `2.20` | `—` / **`92.15`** |
| **`MACE-MatPES-r2SCAN-0` (No D3)** | `-0.03` | `-35.39` | `2.35` | `25.47` |
| **`MACE-MatPES-r2SCAN-0 + D3(r2scan)`** | `-2.40` | **`-46.45`** | `2.20` | **`40.13`** |

![MLIP Mechanism & Overbinding Audit](../assets/paper_fig3_mlip_mechanism_audit.png)

*图 5 四代 MACE 模型在 `Mg-MOF-74` 无遮挡轴向径向扫描、7.31 Å 远场截止检验与加载态平均结合能诊断。*

### 4.2 四类代表性 MOF 与 NIST ISODB 实验数据对比分析

采用静态极小值相对合理的 `MACE-MatPES-r2SCAN-0 + D3(r2scan)`，结合 NIST ISODB 归档的纯净未经改性样品 298 K 实验等温线（见图 6 及表 4），对跨化学环境与跨拓扑的四类代表性 MOF 开展了完整 GCMC 模拟与热力学剖析：

![Stage-1 Four-MOF Benchmark vs. NIST ISODB](../assets/stage1_four_mof_nist_isodb.png)

*图 6 四类代表性 MOF（`Mg-MOF-74`、`Ni-MOF-74`、`HKUST-1` 与 `IRMOF-1`）在 298 K 下的 GCMC 模拟等温线与 NIST ISODB 实验数据对比。*

**表 4** `MACE-MatPES-r2SCAN-0 + D3(r2scan)` 与经典基线在四类 MOF 上的平衡 GCMC 结果及 NIST ISODB 对照

| 体系与工况 | 外层生产步数 | 经典基线 (mmol/g) | MLIP 吸附量 (mmol/g) | 微分热 $Q_{\text{st}}^{\text{diff}}$ (kJ/mol) | 积分焓 $Q_{\text{int}}^{\text{molar}}$ (kJ/mol) | 生产期增/删 (4 盒) | NIST ISODB `298 K` 实验对照 (DOI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Mg-MOF-74`, `298 K, 0.01 bar` | `1200` | `0.802` (`Mercado`) | `6.3407 ± 0.4486` | `50.28 ± 1.81` | `48.07 ± 0.25` | `133/590`, `112/435` | `2.102` @ `0.0101 bar` (`10.1021/acs.jpcc.7b08239`) |
| `Mg-MOF-74`, `298 K, 0.05 bar` | `1200` | `3.312` (`Mercado`) | `9.4714 ± 0.2595` | `36.31 ± 1.97` | `45.77 ± 0.27` | `182/573`, `171/823` | `4.410` @ `0.0315 bar` (`10.1021/acs.jpcc.7b08239`) |
| `Mg-MOF-74`, `298 K, 0.15 bar` | `480` | `6.023` (`Mercado`) | `11.4721 ± 0.1519` | `31.97 ± 4.23` | `43.17 ± 0.32` | `71/199`, `72/302` | `5.867–7.044` @ `0.099–0.311 bar` |
| `Mg-MOF-74`, `313 K, 0.01 bar` | `480` | `—` | `3.9595 ± 0.6894` | `42.59 ± 6.03` | `46.81 ± 0.44` | `46/338`, `39/135` | 相对 `298 K` 热脱附 `−2.38 mmol/g` |
| `Ni-MOF-74`, `298 K, 0.01 bar` | `260` | `0.249` (`Pol-3Site`) | `0.5779 ± 0.1437` | `33.83 ± 1.26` | `—` | `35/67`, `27/367` | `0.398` @ `0.0101 bar` (`10.1021/acs.jpcc.5b03700`) |
| `Ni-MOF-74`, `298 K, 0.05 bar` | `480` | `1.199` (`Pol-3Site`) | `1.4671 ± 0.1705` | `35.57 ± 1.30` | `36.48 ± 0.66` | `75/182`, `55/757` | `2.091` @ `0.0512 bar` (`10.1021/acs.jpcc.5b03700`) |
| `Ni-MOF-74`, `298 K, 0.15 bar` | `480` | `2.984` (`Pol-3Site`) | `4.2921 ± 0.3457` | `35.23 ± 2.74` | `36.98 ± 0.50` | `79/170`, `59/876` | `4.456` @ `0.1512 bar` (`10.1021/acs.jpcc.5b03700`) |
| `Ni-MOF-74`, `313 K, 0.05 bar` | `480` | `—` | `1.1449 ± 0.2693` | `38.55 ± 1.71` | `37.14 ± 0.39` | `74/270`, `60/641` | 相对 `298 K` 热脱附 `−0.32 mmol/g` |
| `HKUST-1`, `298 K, 0.15 bar` | `480` | `0.835` (`UFF`) | `1.2199 ± 0.1482` | `25.49 ± 3.67` | `25.51 ± 0.49` | `69/197`, `56/1271` | `0.880` @ `0.147 bar` (`10.1002/aic.15837`) |
| `HKUST-1`, `298 K, 0.50 bar` | `480` | `2.126` (`UFF`) | `2.3279 ± 0.1678` | `21.46 ± 3.61` | `23.91 ± 0.33` | `79/220`, `60/1263` | `2.670` @ `0.505 bar` (`10.1002/aic.15837`) |
| `HKUST-1`, `298 K, 1.00 bar` | `480` | `3.676` (`UFF`) | `4.0984 ± 0.0657` | `24.34 ± 0.62` | `23.53 ± 0.32` | `76/182`, `52/1303` | `4.530` @ `0.998 bar` (`10.1002/aic.15837`) |
| `IRMOF-1`, `298 K, 2.00 bar` | `360` | `1.546` (`UFF`) | `5.2688 ± 0.5154` | `18.67 ± 1.54` | `17.90 ± 0.51` | `50/229`, `35/960` | `1.838` @ `2.074 bar` (`10.1021/acs.jctc.5b00093`) |
| `IRMOF-1`, `298 K, 5.00 bar` | `360` | `4.539` (`UFF`) | `12.2190 ± 0.5134` | `26.51 ± 2.68` | `19.01 ± 0.31` | `103/295`, `72/653` | `4.917` @ `5.240 bar` (`10.1021/acs.jctc.5b00093`) |

综合表 4 与各构型探测结果，可得出以下三项明确机理判断：

1. **高统计采样分辨超饱和区微分热行为（`Mg-MOF-74`）**：将 `Mg-MOF-74` 外层生产周期由 `480` 轮延长至 `1,200` 轮后，4 个副本在 `0.01 bar` 与 `0.05 bar` 均实现充分双向交换（`Ins/Del = 133/112` 与 `182/171`），涨落法微分热标准误由 `±10.56 kJ/mol` 降至 **`±1.97 kJ/mol`**（降幅 $5.4\times$）。在突破 1:1 `Mg²⁺` 配位饱和（$8.24\ \text{mmol/g}$）到达 $9.471\ \text{mmol/g}$ 时，微分热 $Q_{\text{st}}^{\text{diff}}$ 清晰地自 $50.28\ \text{kJ/mol}$ 回落至 **`36.31 kJ/mol`**，而全孔积分摩尔焓仍维持在 $45.77\ \text{kJ/mol}$。此外，`Ni-MOF-74`（480 轮）与 `Mg-MOF-74` 在 `313 K` 下均呈现与物理预期一致的热脱附（分别减少 $0.32$ 与 $2.38\ \text{mmol/g}$），且模型跨材料重现了实验 `Mg-MOF-74 > Ni-MOF-74` 吸附序。
2. **`PilotStage3` 偏置修正消除非平衡滞后与 `HKUST-1` 误差抵消分析**：早期使用 `200` 轮测试 `HKUST-1` 时，因纯经典 Stage-2 计算出过高正偏置（`+3.71 / +4.66 kJ/mol`），将内层提议逸度压缩至 `0.22× / 0.15×`，引起热启动过低与 `0.5–1.0 bar` 伪平台（`1.34 -> 1.27 mmol/g`）。引入正偏移钳位（$\le +2.0\ \text{kJ/mol}$）与 `PilotStage3` 目标系综重标定后（最终偏移 `-0.28 / +0.90 / +1.34 kJ/mol`），`HKUST-1` 顺利平衡并随压力单调上升（`1.220 -> 2.328 -> 4.098 mmol/g`），`0.5` 与 `1.0 bar` 接近 NIST ISODB 实验（偏差约 `10%–13%`）。值得注意的是，针对 `HKUST-1` Cu²⁺–Cu²⁺ 双核反铁磁桨轮轴向开展的独立静态扫描表明，`r2SCAN-0+D3` 与 `MP-0+D3` 给出仅 `-10.07` 与 `-7.72 kJ/mol`（$2.40$ Å）的弱极小值；因此 `HKUST-1` 等温线与实验数值接近，极可能反映了自旋无关模型低估 Cu²⁺ 化学配位与高估框架/小笼背景色散之间的部分抵消。
3. **无开放金属位点对照组 `IRMOF-1` 隔离出系统性背景过结合**：在不含开放金属位点的 `IRMOF-1` 上，经典零电荷 `UFF` 基线（`1.546 / 4.539 mmol/g`）与 NIST ISODB 实验（`1.838 / 4.917 mmol/g`）吻合良好，但 `MACE-MatPES-r2SCAN-0 + D3(r2scan)` 仍高出实验 **`2.5–2.9×`**（积分热 `~18.4–19.0 kJ/mol` vs. 经典 `13.2–14.4 kJ/mol`），与 `Mg-MOF-74` 高出实验 `1.8–3.0×` 具有一致方向性（整体结合强约 $4–5\ \text{kJ/mol/CO}_2$）。经直接检查 `torch-dftd 0.5.3` 源码参数确认，`xc="r2scan"` 确实加载了标准的 Grimme/Ehlert `r2SCAN-D3(BJ)` 阻尼系数（$s_6=1.0000,\ a_1=0.4948,\ s_8=0.7898,\ a_2=5.7308\ a_0$）[6]，从而排除了色散参数误回退至 PBE 的软件缺陷，将系统性过结合的可能根源聚焦于 `r2SCAN-D3(BJ)` 泛函本征精度及/或 MACE 与外加 D3 项之间的微弱色散重复计数。

---

## 5. 局限性与后续工作

尽管混合引擎的统计无偏性、数值稳定性及四类 MOF 采样收敛性已获充分验证，当前工作仍存在以下明确边界与后续改进空间：

1. **周期性 DFT 单点参考拆解误差（Stage 2）**：上述 `IRMOF-1` 与 `Mg-MOF-74` 中观察到的约 `4–5 kJ/mol/CO2` 背景过结合以及 `MPA-0 / MP-0b3` 短程塌陷，尚缺乏针对抽样构型计算的严格周期性 DFT 单点能对照（`r2SCAN`、`r2SCAN+D3(BJ)`、`PBE+D3` 及自旋极化双核 `Cu²⁺` 校验），因此目前尚未定量拆分“MLIP 神经网络拟合残差”、“DFT 泛函本征误差”与“D3 色散重复计数”三者各自占比。
2. **模型刚性与经典极化力场假设**：当前体系均假设 MOF 骨架及小分子客体严格刚性（忽略骨架呼吸与 CO₂ 弯曲振动）；同时经典 `Pol-SCF` 模块直接将多体 Thole 极化叠加于原非极化 TraPPE/UFF 参数之上，与专门按极化模型重新优化参数的力场相比存在诱导能双重计数效应。
3. **微调误差修正与软件横评**：在取得周期性 DFT 单点数据集后，可通过轻量级微调或残差 $\Delta$-learning 模型（Stage 3）修正基准模型在特定孔道环境下的过结合；此外，经典提议端虽已完成与 CPU `RASPA3 3.0.29` 的严格对标，尚待在同一块 RTX 5070 上补充与 GPU `gRASPA` 的直接纯经典核速度对照。

---

## 6. 参考文献与数据来源

[1] Boyd, P. G.; et al. Data-driven design of metal–organic frameworks for wet flue gas CO₂ capture. *Nature* **2019**, *576*, 253–256.  
[2] Queen, W. L.; Hudson, M. R.; Bloch, E. D.; et al. Comprehensive study of carbon dioxide adsorption in the metal–organic frameworks $\text{M}_2(\text{dobdc})$ ($\text{M} = \text{Mg, Mn, Fe, Co, Ni, Cu, Zn}$). *Chem. Sci.* **2014**, *5*, 4569–4581.  
[3] Mercado, R.; Vlaisavljevich, B.; Lin, L.-C.; et al. Force field development from periodic density functional theory calculations for gas separation applications using metal–organic frameworks. *J. Phys. Chem. C* **2016**, *120*, 12590–12604.  
[4] Batatia, I.; et al. A foundation model for atomistic materials chemistry. *arXiv:2401.00096* **2024**; MatPES / `MACE-MatPES-r2SCAN-0` foundational potentials.  
[5] Dubbeldam, D.; Torres-Knoop, A.; Walton, K. S. On the inner workings of Monte Carlo codes / `RASPA3` (`3.0.29`); Zhou, J.; et al. `gRASPA`: GPU-accelerated Monte Carlo simulation of adsorption in porous materials. *J. Chem. Theory Comput.* **2024**.  
[6] Ehlert, S.; Huniar, U.; Orio, M.; Grimme, S. $\text{r}^2\text{SCAN-3c}$: A “Swiss army knife” composite electronic-structure method. *J. Chem. Phys.* **2021**, *154*, 061101.  
[7] NIST/ARPA-E Database of Novel and Emerging Adsorbent Materials (NIST ISODB) 实验数据集归档于 `data/exp_isotherms/manifest.json`：  

- `Mg-MOF-74` (298 K CO₂): DOI `10.1021/acs.jpcc.7b08239`, `10.1039/c5dt03137k`  
- `Ni-MOF-74` (298 K CO₂): DOI `10.1021/acs.jpcc.5b03700`, `10.1080/15422119.2014.884507`  
- `HKUST-1` (298 K CO₂): DOI `10.1002/aic.15837`, `10.1007/s10450-013-9527-2`  
- `IRMOF-1` (298 K CO₂): DOI `10.1021/acs.jctc.5b00093`, `10.1038/srep12045`  
