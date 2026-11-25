# `cuda-mof-gcmc` 架构重构与验证方案

## 一、问题根源与方法定位

### 1.1 当前代码的四个实际缺陷

1. **内层势能面与 MLIP 严重失配**：通用 `UFF+QEq` 的 `12-6` 排斥墙在开放金属位点真实配位距离（如 `Mg²⁺–O = 2.15–2.45 Å`）处高达 `+20 ~ +55 kJ/mol`，内层采样无法进入真实配位势阱；而手工拟合力场（`Mercado 2016`）仅适用于单一 `Mg-MOF-74 + CO₂`，无法迁移至其它 MOF 与客体，违背通用 MLIP 的目标。
2. **CUDA 核内均匀随机插入效率过低**：[proposer_dll.cu](file:///c:/Users/lishi/code/cuda-mof-gcmc/src/proposer_dll.cu) 中 `propose_burst_kernel` 的插入步（`mv == 2`）仅由 `tid == 0` 单线程在全晶胞均匀投点，其余 127 个线程在准备试探阶段闲置。在中高载量或强配位口袋下有效命中率低于 `0.5%`，单次外层 MLIP 调用很难产生多分子粒子数跳变。
3. **不带嵌套修正的纯 MLIP 网格近似存在显著多体误差**：仅预扫单分子 3D MLIP 网格并搭配经典 `TraPPE/EPM2` 客体对势直接跑 GCMC，会丢失孔道屏蔽、多分子协同极化及模型色散项的差异（`Mg-MOF-74` 在 `N = 27 → 26` 删分子测试中，单分子 MLIP + `EPM2` 与完整多体 MACE 之间存在 `+6.09 kJ/mol` 的系统偏移），直接导致中高载量等温线压力偏离数倍。
4. **辅助工程缺陷**：
   - [run_mlip_gcmc.py](file:///c:/Users/lishi/code/cuda-mof-gcmc/scripts/run_mlip_gcmc.py) 中 `--shift-mode auto` 原有的 `[-24, +2/+4] kJ/mol` 非对称正上限截断，在目标结合弱于提议势（如 `No-D3`）时会锁死化学势偏置。
   - [proposer_dll.cu](file:///c:/Users/lishi/code/cuda-mof-gcmc/src/proposer_dll.cu) 中 `proposer_eval_energy_quat` 每次调用都重复执行 9 次 `cudaMalloc` 与 9 次 `cudaFree`，应改为复用 `ProposerContext` 预分配缓冲。

### 1.2 前人工作边界与本项目明确创新点

- **已有工作（文献如实引用，绝不虚报）**：
  - 经典力场预采样 Nested MC / 双势场重加权（Gelb, *J. Chem. Phys.* 2003；Iftimie et al., *J. Chem. Phys.* 2000；Goeminne & Van Speybroeck, *J. Chem. Theory Comput.* 2023）。
  - 预计算单分子 3D MLIP 宿主–客体网格 + 经典客体势的近似 GCMC。
- **本项目真正站得住脚的三项组合创新**：
  1. **零人工力场的在线 1B+2B MLIP 自蒸馏代理势**：仅输入任意刚性 MOF CIF、刚体客体与目标 MLIP，自动按周期性配位连接拓扑识别框架原子化学环境，用约 30 秒的目标 MLIP 单分子（`~1,000` 构型）+ 孔内双分子（`~200` 构型）评估，在线蒸馏出**单分子三维网格 $u_1^{\text{distill}}$** 与**孔内等效两体客体势 $u_{2,\text{pore}}^{\text{distill}}$**。彻底摆脱 `UFF` 排斥墙与 `Mercado` 专用力场，开箱适配任意刚性 MOF（含各类开放金属位点）及任意 MLIP / 色散组合。
  2. **CUDA 线程块级 128 线程并行 Rosenbluth CBMC 内层采样**：在 `propose_burst_kernel` 内让一个线程块的 128 个线程并行生成并评估最多 128 个试插/试删位姿（三维 $O(h^4)$ 三三次插值网格 + 共享内存客体对势），在百微秒级完成高接受率多步粒子交换，使每次外层 MLIP 推理可跨出 `|ΔN| = 2~5` 的多分子跳变。
  3. **精确多体 MLIP 嵌套修正闭环（`direct_prob = 0.0`）**：将完整多体目标势严格分解为
     $$U_{\text{tgt}}(X^N) = \sum_{i=1}^N u_1^{\text{distill}}(x_i) + \sum_{1 \le i < j \le N} u_{2,\text{pore}}^{\text{distill}}(x_i, x_j) + \Delta U_{\ge 3}^{\text{many-body}}(X^N)$$
     前两项由 GPU 微秒级内层 CBMC 链承载，三体及以上多体协同能与蒸馏残差由外层完整多体 MLIP 通过精确 Metropolis 判据严格修正，收敛至严格无偏的目标 MLIP 巨正则系综。

---

## 二、数学公式与细致平衡严格性

### 2.1 自动化学环境分型与 1B+2B 在线蒸馏

1. **周期性配位环境自动分型**：
   按共价半径阈值
   $$r_{jk} < r_{\text{cov}}(Z_j) + r_{\text{cov}}(Z_k) + 0.35\text{ Å}$$
   构建周期性最近邻邻接表，赋予每个框架原子 $j$ 唯一的化学环境签名：
   $$\tau_j = \bigl(Z_j,\ \text{sorted}(Z_k : k \in \mathcal{N}_j)\bigr)$$
   完全不依赖人工力场标签或 CIF 残基命名。

2. **单分子 1B 势能分解与 GPU 网格生成**：
   对客体每个真实原子通道 $c \in \{0, \dots, C-1\}$，在每个化学环境 $\alpha$ 上分解为电中性约束周期性 Ewald 静电基 + 非负单调短程排斥核 + 平滑径向基函数（覆盖 $1.65\text{–}6.5\text{ Å}$，截断 $r_{\text{cut}} = 8.0\text{ Å}$）：
   $$V_c(\mathbf{r}) = q_c \sum_{\alpha} q_{\alpha}^{\text{eff}} \Phi_{\text{ewald}}^{(\alpha)}(\mathbf{r}) + \sum_{\alpha} \left[ w_{c,\alpha}^{\text{core}} \left(\frac{r_0}{r}\right)^{12} + \sum_{b=1}^{B} w_{c,\alpha,b}\,\phi_b(r) \right]$$
   受电中性与非负核约束：
   $$\sum_{\alpha} N_{\alpha} q_{\alpha}^{\text{eff}} = 0, \qquad w_{c,\alpha}^{\text{core}} \ge 0$$
   拟合采用带约束岭回归（毫秒级完成），随后在 GPU 上直接生成单胞周期性 `float32` 网格 `U_lj[c]`（最近框架原子距离 $d_{\min} < r_{\text{hard}}$ 处统一封顶 `1.0e6f` 拦截硬核重叠），内层评价每个客体分子只需对每个真实原子做一次 $O(h^4)$ 三三次插值查表。

3. **孔内双分子 2B 等效客体对势**：
   在孔道内采样的无重叠双分子样本上计算精确孔内二体相互作用：
   $$u_{2,\text{pore}}^{\text{MLIP}}(x_a, x_b) = U_{\text{tgt}}(\{x_a, x_b\}) - U_{\text{tgt}}(\{x_a\}) - U_{\text{tgt}}(\{x_b\})$$
   据此拟合客体有效电荷缩放因子与位点对径向残差查找表 `gg_corr_lut[pair_type][0..127]`（覆盖 $r \in [2.0, r_{\text{cut}}]\text{ Å}$，并在 $r_{\text{cut}}$ 处平滑归零），自动吸收目标模型有无 D3 色散带来的二体强度变化以及框架介质屏蔽效应。

4. **元数据指纹缓存与跨压强复用**：
   蒸馏结果按 `(cif_sha256, guest_sha256, model_sha256, dispersion, xc, dtype, grid_dims, distill_seed)` 的 SHA-256 哈希缓存至磁盘。整条等温线（如 20 个压强点）或多卡作业仅在首个任务蒸馏一次（约 25–30 秒），其余压强点或工作卡在 `< 0.2 s` 内直接挂载网格。

### 2.2 CUDA 块并行 $K$ 路 Rosenbluth CBMC 超细致平衡

设单步插入与删除试探概率严格相等（$p_{\text{ins}} = p_{\text{del}}$；默认各向异性客体 `p_trans = 0.20, p_rot = 0.20, p_ins = 0.30, p_del = 0.30`；各向同性客体 `p_trans = 0.30, p_ins = 0.35, p_del = 0.35`），块内 CBMC 试探路数 $K = \text{cbmc\_k} \in [1, 128]$（默认 `128`）。

- **插入步（`mv == 2`, $N \to N+1$）**：
  由主线程 RNG 状态 `d_rng[b]` 派生本步 `step_seed`，线程 `tid = 0..K-1` 并行独立从 $V \times SO(3)$ 均匀生成候选位姿 $y_t$，并基于三维网格与共享内存客体对势评估完整内层增量能量 $\Delta U_t = u_1(y_t) + \sum_{j=0}^{N-1} u_2(y_t, x_j)$（重叠时 $w_t = 0$）。计算平均 Rosenbluth 权重：
  $$\overline{W}_{\text{ins}} = \frac{1}{K}\sum_{t=0}^{K-1} w_t, \qquad w_t = \exp\!\left(-\frac{\Delta U_t}{T_K}\right)$$
  按概率 $p_t = w_t / \sum_{t'=0}^{K-1} w_{t'}$ 选中 $y_{t^*}$，插入接受概率为：
  $$P_{\text{acc}}^{\text{ins}} = \min\!\left(1,\ \frac{\beta f_{\text{ref}} V}{N + 1}\,\overline{W}_{\text{ins}}\right)$$

- **删除步（`mv == 3`, $N \to N-1$）**：
  均匀随机选取当前存在的第 `idx` 个分子作为 $y_0 = x_{\text{idx}}$，计算其与其余 $N-1$ 个分子的能量 $\Delta U_0$；同时线程 `tid = 1..K-1` 独立均匀生成 $K-1$ 个诱饵位姿 $y_t$，计算各自相对同一组剩余 $N-1$ 个分子的 $\Delta U_t$，规约得到：
  $$\overline{W}_{\text{del}} = \frac{1}{K}\sum_{t=0}^{K-1} w_t, \qquad w_t = \exp\!\left(-\frac{\Delta U_t}{T_K}\right)$$
  删除接受概率为：
  $$P_{\text{acc}}^{\text{del}} = \min\!\left(1,\ \frac{N}{\beta f_{\text{ref}} V}\,\frac{1}{\overline{W}_{\text{del}}}\right)$$

- **细致平衡严格性**：
  当 $K = 1$ 时上述公式精确退化为标准 Metropolis GCMC；对任意 $K \ge 1$，选中位姿的 Boltzmann 因子 $w_{t^*} = e^{-\beta \Delta U_{t^*}}$ 在正向与逆向转移概率比中严格对消，单步严格保持内层巨正则分布 $\pi_{\text{ref}}(X^N) \propto \frac{(\beta f_{\text{ref}} V)^N}{N!} e^{-\beta U_{\text{ref}}(X^N)}$ 的超细致平衡。因此 $k_{\text{steps}}$ 步内层链整体满足无偏条件，外层接受率保持简洁精确形式：
  $$\alpha_{\text{outer}}(X \to X') = \min\!\left(1,\ \exp\!\left[-\beta\left((U_{\text{tgt}}(X') - U_{\text{tgt}}(X)) - (U_{\text{ref}}(X') - U_{\text{ref}}(X)) - \Delta\varepsilon\,\Delta N\right)\right]\right)$$

---

## 三、工程实现拆解

### 3.1 模块 A：CUDA 提议器内核升级（`include/proposer_api.h`, `src/proposer_dll.cu`）

1. **外部蒸馏网格注入接口**：新增 `proposer_set_external_grids(...)`，支持将蒸馏生成的 `n_lj` 通道 `float32` 单胞网格直接写入 GPU 显存，复用现有 `tricubic_periodic` 插值通道，避免在 CUDA 核内硬编码特定力场拟合形式。
2. **二体客体修正 LUT 接口**：新增 `proposer_set_gg_lut(...)`，将位点对径向残差查找表与有效电荷缩放因子写入 `RigidGuestDef`，保证 `propose_burst_kernel` 与 `eval_single_state_kernel` 使用完全一致的客体间能量公式。
3. **块并行 Rosenbluth CBMC 与参数接口**：新增 `proposer_set_mc_params(void* handle, int cbmc_k, double p_ins_del)`，重写 `propose_burst_kernel` 的插入与删除路径，使用 warp/block 规约与确定性 `step_seed` 派生保持单状态 RNG 可序列化与断点逐位一致。
4. **单态求值缓冲复用**：在 `ProposerContext` 中预分配 `d_eval_cx..qz, d_eval_U, d_eval_ov`，移除 `proposer_eval_energy_quat` 内的动态 `cudaMalloc` / `cudaFree`。

### 3.2 模块 B：在线 MLIP 自蒸馏器（`scripts/mlip_distill.py`）

1. 基于框架晶胞与共价半径自动构建化学环境分类，导出环境清单与近邻拓扑摘要。
2. 执行分层无重叠孔道单分子采样（极性/开放位点近壁区 40%、中等距离过渡区 40%、孔道中心区 20%）及孔道内双分子二聚体采样（`~1,000` 单分子 + `~200` 双分子，单卡 3090 约耗时 25–30 秒）。
3. 批处理调用 `FastMaceD3Target` 计算真实标签，带电中性及非负短程核约束求解 1B + 2B 代理模型，在留出验证集上输出全域与近金属位点 `R²` / `MAE`。
4. 利用 GPU 向量化评估生成单胞 3D 网格，并按全配置 SHA-256 哈希持久化缓存以便跨压强、跨卡秒级复用。

### 3.3 模块 C：主采样器与多卡编排整合（`nested_mc_engine.py`, `run_mlip_gcmc.py`, `multi_gpu.py`, `run_metadata.py`）

1. `--proposer` 新增 `auto_mlip` 并设为默认选项（保留 `uff`, `pol3site`, `mercado` 用于对照消融），新增 `--cbmc-k`（默认 `128`）、`--distill-cache-dir` 及 `--grid-only`（纯自蒸馏网格无嵌套重加权对照模式）。
2. 将 `--shift-mode auto` 统一修正为对称范围 `[-24.0, +24.0] kJ/mol` 与双向载量探测，正式平衡与生产前锁定 $\Delta\varepsilon$。
3. 多卡 `pressure` / `replica` 模式下，在调度分发前先由主进程或首卡完成一次自蒸馏落盘，其余进程只读加载缓存，并将蒸馏缓存摘要写入 `run_metadata.py` 防止错配续跑。

---

## 四、分阶段执行顺序与硬门禁

### Phase 1：CUDA 并行 CBMC + 自蒸馏模块实现与零偏差单元测试（本地 + `g0048`）

- **门禁 1（能量一致性）**：`proposer_eval_energy_quat` 与 `propose_burst_kernel` 末尾能量在 `auto_mlip` 外部网格 + 2B LUT 模式下最大差值 `< 1e-8 K`。
- **门禁 2（GPU-CBMC 零偏差）**：经典势与外部蒸馏网格势下，`cbmc_k = 128` 与 `cbmc_k = 1` 及 `RASPA3` 参考结果一致（吸附量均值、粒子数方差与 $Q_{\text{st}}$ 偏差 `< 2.0σ`）。
- **门禁 3（Nested MC 跨势场零偏差）**：分别从 `empty` 和 `high_loading` 初态出发，覆盖正/负 $\Delta\varepsilon$，以 `auto_mlip` / `UFF` 提议器嵌套采样已知目标势场，全部通过严格无偏校验。
- **门禁 4（断点逐位复现）**：`cbmc_k = 128` 下断点保存与 `--resume-mode extend` 续跑与连续单次运行在轨迹、矩统计量与 RNG 状态上 100% 逐位一致。

### Phase 2：方法学三大核心对照实验（仅使用 `g0048` 空闲 `GPUs 0–3`）

- **统一硬收敛标准**（任意指标未达标均判定为未收敛并延长采样，严禁放宽门槛）：
  - 每工况至少 `4` 条独立链，关键压强同时验证 `empty` 与 `high_loading` 双向初态；
  - 秩归一化 split/folded $\hat{R}(N) < 1.01$、$\hat{R}(U) < 1.01$（含 OMS 体系同时检查 $\hat{R}(N_{\text{OMS}}) < 1.01$）；
  - $\text{ESS}_{\text{bulk}}(N) \ge 100 \times n_{\text{chains}}$，$\text{ESS}_{\text{tail}}(N) \ge 50 \times n_{\text{chains}}$；
  - 吸附量 $\text{MCSE} \le \max(0.05\text{ mmol/g},\ 1\%)$，双向初态均值差 95% CI 落在 $\pm \max(0.10\text{ mmol/g},\ 2\%)$ 以内；
  - $Q_{\text{st}}$ 严格采用多链独立涨落估计（`Var(N)` 不足则标注不可估计，绝不静默回退积分摩尔焓）。
- **实验组 1（`Mg-MOF-74 + CO₂`, `298 K`, `0.012 / 0.28 / 1.0 bar`, `No-D3` 与 `+D3` 四方同台对比）**：
  1. `Auto-MLIP (1B+2B) + GPU-CBMC Nested MC`（零人工力场主方法）；
  2. `Direct MLIP-GCMC`（同卡达成相同 MCSE 的总墙时与 `ESS/GPU-s` 加速比，显式计入一次性蒸馏耗时）；
  3. `UFF+QEq Nested MC` 与 `Mercado Nested MC`（量化突破 UFF 开放金属位点排斥壁垒的效果，并与手工定制力场对比）；
  4. `Unreweighted Auto-MLIP Grid GCMC`（`1B + EPM2` 与 `1B + 2B` 不做外层多体修正，定量揭示近似网格在中高载量的系统偏差）。
- **实验组 2（跨金属、跨拓扑、跨客体零调参泛化验证）**：
  - `Ni-MOF-74 + CO₂`（开放 $\text{Ni}^{2+}$ 位点，`Mercado` 力场不适用，`Auto-MLIP` 零参数运行）；
  - `HKUST-1 (Cu-BTC) + CO₂`（双核铜桨轮与多级孔笼）；
  - `IRMOF-1 + CO₂ / CH₄`（非开放位点大孔基准）。

### Phase 3：物理机理核验与多卡扩展测试

1. 独立零载量 Widom 插入法（亨利系数 $K_H$ 与零载吸附热 $Q_{\text{st}}(0)$）对拍极低压 GCMC。
2. 固定构型 MACE 本征项 vs `TorchDFTD3` 色散项定量拆解，及 `1×1×2` vs `1×1×4` 超胞有限尺寸检查。
3. 与 `data/exp_isotherms/` 纯净样品实验数据客观对照（明确绝对/过剩吸附量与温度换算，禁止使用任何经验乘数或凑数色散权重）。
4. `1×` vs `4× RTX 3090` 在 `pressure` 与 `replica` 模式下的端到端吞吐加速实测。

### Phase 4：全收敛数据驱动的文档与图表重编

全部门禁与实验收敛通过后，从零撰写中英文技术报告，重绘全新系统架构图与核心实验对比图，并同步更新中英文 README。
