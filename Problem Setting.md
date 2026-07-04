

---

# 项目文档：基于频率感知隐空间扩散模型的物理场重构

# (Freq-aware Latent Diffusion for Physical Field Reconstruction)

## 1. 问题动机 (Problem Motivation)

在计算物理和科学机器学习领域，从稀疏观测中重建完整的连续物理场（如声场、电磁场）是一个经典的逆问题。传统的算子学习（如 FNO）和图像生成模型在处理该问题时面临以下挑战：

- **频率敏感性与相位漂移 (Frequency Sensitivity & Phase Drift):** 物理场（如亥姆霍兹方程的解）随频率 $\omega$ 呈非线性演化。频率升高会导致波长缩短和复杂的干涉图样 。现有模型在处理训练集未覆盖的频率（外推测试）时，往往无法准确捕捉波动相位，导致重构失效 。
    
- **高维空间的自由度灾难 (Curse of Dimensionality):** 直接在原始高维网格空间（如 $H \times W \times D$）进行扩散生成，计算成本极高，且难以保证长距离的物理空间连贯性 。
    
- **极端稀疏观测下的欠定性 (Extreme Sparsity):** 当空间采样点极度稀疏（如 $\rho < 1\%$）时，缺乏有效的物理先验会导致模型生成的解违背基本的物理拓扑 。
    

**核心初衷：** 物理场虽然随频率变化剧烈，但其**空间模态组合的演化（即张量核心 $G$ 的轨迹）**在隐空间中具有更强的规律性。我们旨在通过频率感知的隐空间扩散模型，锁定物理场的模态演变规律，实现鲁棒的频率外推与重构。


## 1.2 任务设定 (Task Setting)

本研究关注一类**频率驱动的二维物理场重构问题**。我们的目标是在给定稀疏空间观测和目标频率的条件下，利用从历史数据中学习到的隐空间先验，恢复高保度的完整场。

### 2.1 输入与输出定义

- **输入 (Inputs):**
    
    - **目标频率 ($\omega_h$):** 需要进行物理场重构的特定频率点 。
        
    - **稀疏观测 ($y_{\omega_h}$):** 在目标频率 $\omega_h$ 下，通过测量算子 $\mathcal{M}$ 获得的少量、离散且可能非网格化的空间采样点 。其测量模型表示为：$y_{\omega_h} = \mathcal{M}(u_{\omega_h}) + \varepsilon$ 。
        
    - **条件信息 (Conditioning):** 包含特定样本的物理背景，如初始条件（IC）或边界条件（BC）的隐式线索（通过稀疏观测间接提供） 。
        
- **输出 (Outputs):**
    
    - **完整物理场 ($\hat{u}_{\omega_h}$):** 在目标频率 $\omega_h$ 下，覆盖全空间网格或连续坐标的重构物理场切片 。
        

### 2.2 数据集构建与联合概率建模

为了捕捉物理场随频率演化的动力学规律，我们构建了如下结构的训练集：

1. **轨迹生成 (Trajectory Generation):** 针对同一类物理系统（如特定几何布局下的波动方程），通过改变 $N$ 组初始/边界条件（IC/BC）并在 $M$ 个频率点 $\{\omega_1, \dots, \omega_M\}$ 上进行数值求解，得到 $N \times M$ 个样本 。
    
2. **张量因子解耦 (Tensor Factorization):** 基于 **Functional Tucker Model (FTM)**，假设每个样本均可分解为共享空间基与频率相关系数的乘积：$u(x, \omega) \approx f(x) \cdot G(\omega)$ 。
    
    - **$f(x)$:** 所有样本共享的空间潜函数（Spatial Basis），代表系统的本征物理模态 。
        
    - **$G(\omega)$:** Tucker Core，代表特定样本在特定频率下的模态权重，是频率感知的核心隐变量 。
        
3. **概率分布建模:** 扩散模型通过建模条件分布 $p(G \mid \omega)$ 来学习物理系统的**频率响应流形** 。在推理阶段，通过联合先验分布与观测似然 $p(y_{\omega_h} \mid G, \omega_h)$，利用 **DPS (Diffusion Posterior Sampling)** 引导生成过程，使隐变量 $G$ 从通用的频率先验塌缩为符合当前观测证据的唯一解 。
---

## 2. 方法论 (Methodology)

本方法采用“空间基萃取—频率感知生成—证据驱动修正”的三阶段框架。


###  2.1 数据集构建 (Dataset Construction):

为了支撑频率感知的隐空间建模，数据集通过对同一类物理系统进行多维度参数采样构建：

- **样本生成轨迹 (Trajectories):** 针对特定的 PDE 家族（如亥姆霍兹方程），选取 $N$ 组不同的初始条件/边界条件 (IC/BC) 以及特定的频率序列 $\{\omega_1, \omega_2, \dots, \omega_M\}$ 。对于每个特定样本，其在不同频率下的物理场切片共同构成该样本沿频率轴变化的场轨迹轨迹 $\mathcal{U}^{(i)} = \{u_{\omega_1}, u_{\omega_2}, \dots, u_{\omega_M}\}$ 。
    
- **张量因子分解:** 假设每个物理场切片均可分解为张量因子形式：$u(x, \omega) \approx f(x) \cdot G(\omega)$ 。其中，$f(x)$ 是所有样本和频率点共享的空间基表征（由 $K$ 组连续空间潜函数组成），而 Tucker Core $G(\omega)$ 则包含了特定样本在特定频率下的系数表征 。
    
- **联合概率分布 (Joint Probability):** 整体数据被视为从联合分布 $p(G, \omega, \text{BC/IC})$ 中抽取的样本对 。扩散模型的核心任务是学习条件先验 $p(G \mid \omega)$，即在已知激励频率的情况下，模态权重 $G$ 的概率分布 。
    
- **观测似然 (Likelihood):** 在推理阶段，稀疏观测 $y_{\omega}$ 满足测量模型 $y_{\omega} = \mathcal{M}(u_{\omega}) + \varepsilon \approx \mathcal{M}(f(x) \cdot G(\omega)) + \varepsilon$ 。在高斯噪声假设下，对应的观测似然为：
    
    $$p(y_{\omega} \mid G, \omega) \propto \exp \left( -\frac{1}{2\sigma^2} \| \mathcal{M}(f(x) \cdot G) - y_{\omega} \|^2 \right)$$
    

该设定将稀疏观测作为间接证据，通过似然梯度引导 $G$ 的生成过程，使其从通用的频率先验塌缩到符合特定 BC/IC 约束的解上 。

### 2.2 频率感知隐空间扩散 (Freq-aware Latent Diffusion)

不同于传统的无条件生成，我们通过构建条件扩散模型学习分布 $p(\mathcal{G} \mid \omega)$ ：

- **条件建模:** 将频率 $\omega$ 进行傅里叶特征编码（Fourier Embedding），作为扩散模型的强约束条件。
    
- **隐空间规律:** 扩散模型在低维的核心张量空间（Core Space）运行，专门学习物理系统的**频率响应流形** 。这使得模型能够理解频率如何驱动模态权重的重新分配，从而具备更强的频率外推能力。
    

### 2.3 基于 DPS 的后验证据修正 (Posterior Guidance via DPS)

在推理阶段，给定目标频率 $\omega_{target}$ 和稀疏观测 $y_{\omega_{target}}$，利用 **Diffusion Posterior Sampling (DPS)** 进行解的塌缩 ：

- **先验提供:** 扩散模型生成符合频率 $\omega_{target}$ 物理特征的候选核心 $\hat{\mathcal{G}}$ 。
    
- **似然修正:** 稀疏观测作为证据，计算重构场与观测点之间的 MSE 损失，通过梯度回传修正 $\hat{\mathcal{G}}$ ：
    
    $$\nabla_{\mathcal{G}} \log p(y \mid \mathcal{G}) \propto \nabla_{\mathcal{G}} \| \mathcal{M}(f(x) \cdot \mathcal{G}) - y \|^2$$
    
- **高效重构:** 由于 FTM 的多线性结构，该梯度计算表现为简单的矩阵乘法，无需复杂的自动微分，极大地提升了重构速度 。
    

---

## 3. 实验设定 (Experiment Setting)

### 3.1 数据构造

- **物理方程:** 2D 亥姆霍兹方程 (Helmholtz Equation)。
    
- **样本集 ($N \times M$):** 改变 $N$ 种边界条件（如随机点源位置/障碍物布局）和 $M$ 种频率序列 $\{\omega_1, \dots, \omega_M\}$ 。
    
- **隐变量提取:** 预训练 FTM 得到所有样本对应的 $\{ \omega, \mathcal{G}_\omega \}$ 样本对。

#### 3.1.1 与当前代码一致的数据生成流程

结合 `Generate_dataset.py` 与 `Helmholtz_Solver.py`，当前可复现实验的数据准备流程如下：

1. **频率轴采样:** 给定 $M$ 与频段 $[\omega_{min}, \omega_{max}]$，按 `linear` 或 `log` 方式生成频率网格 `omega`。
2. **样本条件采样（对应 BC/IC 随机性）:** 每个样本随机采样 $K \in [K_{min}, K_{max}]$ 个点源，源位置在非 PML 区域内均匀采样，源幅值取单位模复数（随机相位）。
3. **PDE 离散求解:** 对每个频率构建并因式分解一次稀疏线性系统矩阵，随后复用该分解求解所有样本右端项（显著降低总计算量）。
4. **复场保存:** 解场 $u$ 按实部/虚部分别存储为 `float32`，用于后续频率条件建模与 FTM 训练。
5. **FTM 训练导出（可选）:** 额外导出 `data` 与 `mask_tr`，其中 `data` 形状为 $(N,M,H,W,2)$，最后一维为实虚通道。

#### 3.1.2 数值求解器与离散设置（实现细节）

- **方程形式:** $\nabla^2 u + k^2 u = -f,\; k=\omega/c$。
- **空间离散:** 正方形区域 $[0,L] \times [0,L]$，均匀网格 $N_g \times N_g$，网格步长 $h=L/N_g$。
- **吸收层:** 四周采用 PML 复拉伸（与频率耦合），用于减少反射。
- **边界处理（代码当前实现）:** 在最外层网格行列上强制 Dirichlet 行覆盖（矩阵对角置 1）。这意味着实现是“PML + 外边界Dirichlet”的组合，而非纯连续意义下无限域。
- **默认参数（脚本）:**
    - 数据脚本默认：`N=200, M=50, grid=128, omega_min=2.0, omega_max=51.0, pml_width=0.12, sigma_max=50.0, seed=42`
    - 频率最高点分辨率检查：$\mathrm{PPW}=\frac{2\pi c N_g}{\omega_{max}L}$，默认阈值 `min_ppw=8.0`，仅告警不强制。

#### 3.1.3 数据组织（HDF5 + FTM元数据）

主数据文件（如 `helmholtz_dataset_42.h5`）包含：

- `metadata`: JSON 字符串，记录生成参数与耗时统计
- `omega`: `(M,) float64`
- `sources`: `(N, K_max, 2) float64`，不足 `K_max` 的位置填 `NaN`
- `amplitudes`: `(N, K_max, 2) float64`，最后一维为 `(real, imag)`
- `fields_real`: `(N, M, N_g, N_g) float32`
- `fields_imag`: `(N, M, N_g, N_g) float32`
- `grid_x`, `grid_y`: `(N_g,) float64`

当 `export_ftm=True` 时，额外包含：

- `data`: `(N, M, N_g, N_g, 2) float32`，由实虚场堆叠而成
- `mask_tr`: `(M, N_g, N_g, 2) float32`，训练观测掩膜

并额外输出一个 `*_metadata.npy`，包含：

- `u_ind_uni`, `v_ind_uni`: 空间坐标轴
- `w_ind_uni`: 通道索引（实/虚）
- `t_ind_uni`: 频率坐标轴
- `mask_tr`, `data_scale`, `obs_ratio`, `mask_mode`

#### 3.1.4 数据规模与存储开销（按当前实现）

记网格边长为 $N_g$，则：

- **PDE 求解次数:** $N \times M$
- **复场自由度:** $2 \times N \times M \times N_g^2$（实虚两通道）
- **仅 `fields_real` + `fields_imag` 的原始体积（不计压缩）:**
    $$8 \cdot N \cdot M \cdot N_g^2 \text{ bytes}$$
- **若同时导出 `data`（FTM）再增加同量级体积**（压缩后体积依赖场复杂度与频段）

以默认参数 $N=500, M=11, N_g=128$ 粗估：

- `fields_real` + `fields_imag` 原始约 $1.31$ GB
- 若包含 `data`，总原始体积约再增加 $1.31$ GB（总约 $2.62$ GB）

#### 3.1.5 稀疏观测掩膜策略

- `obs_ratio` 控制观测比例（默认 `0.01`）。
- `mask_mode=fixed`: 全频率共享同一空间采样位置。
- `mask_mode=per_freq`: 每个频率独立采样掩膜（默认）。

该设计可用于模拟“传感器固定阵列”与“频率扫描时可变观测”两种测量机制。

#### 3.1.6 推荐数据生成命令

```bash
# 默认（生成HDF5 + FTM导出）
python Generate_dataset.py

# 中等规模快速实验
python Generate_dataset.py --N 100 --M 30 --grid 96 --omega_min 2 --omega_max 40

# 极稀疏观测，固定测点
python Generate_dataset.py --obs_ratio 0.01 --mask_mode fixed
```

#### 3.1.7 当前实现与问题设定的差异与建议

下面这些点建议在文档中明确为“当前版本约束”，避免与总目标设定混淆：

1. **当前只覆盖 Helmholtz 单一 PDE。** 文档中的弹性波、变系数扩散属于后续扩展，不是当前数据脚本已实现内容。
2. **几何障碍物/复杂介质尚未进入生成脚本。** 当前随机性主要来自多点源位置与相位，而非障碍物 Mask 或空间变系数。
3. **观测建模目前是规则网格二值掩膜。** 文档中“非网格化采样”尚未在该脚本中实现。
4. **边界条件描述建议更精确。** 代码是“PML + 外边界Dirichlet离散约束”，建议在论文/文档中明确该实现细节。
5. **文件名建议统一。** 当前文件为 `Generate_dataset.py` 与 `Helmholtz_Solver.py`，文档中若引用 `generate_dataset.py`、`Helmholtz_Sover.py` 建议统一拼写，减少复现实验歧义。
    

### 3.2 频率感知测试 (Freq-aware Test)

- **频率外推能力:** 在训练频率范围 $[f_{min}, f_{max}]$ 之外的未知频率点上进行重构测试，评估模型对相位和波动模式的还原度。
    
- **稀疏重构鲁棒性:** 在极低采样率（$\rho = 1\%$）下，对比有无频率感知先验的重构误差 。
    

### 3.3 对比基准 (Baselines)

- **FNO (Fourier Neural Operator):** 评估其在未见频率和稀疏数据下的表现 。
    
- **CoNFILD:** 评估在原始坐标空间做扩散重构的效率与精度 。
    
- **LRTFR:** 传统的低秩张量分解方法，验证引入扩散先验的必要性 。
    
### 3.4 仿真实验与数据泛化 (Simulation & Generalization)

为了验证模型在复杂物理约束下的重构能力，我们设计了跨物理场景的仿真实验：

1. **场景扩展 (PDE Selection):** 除标准的亥姆霍兹方程外，本框架进一步应用于**弹性波方程 (Elastic Wave)** 与 **变系数扩散方程**。这些场景涵盖了波模态转换、色散效应等强频率感知特性。
2. **参数空间采样:** - **几何多样性:** 随机生成 $N$ 组非规则障碍物 Mask 集合 $\mathcal{S}$。
   - **频谱覆盖:** 在连续频段 $[\omega_{min}, \omega_{max}]$ 内均匀采样 $M$ 个频率点。
3. [cite_start]**评价维度:** - **重构精度:** 评估全场 VRMSE [cite: 589]。
   - **物理一致性:** 评估重构场对原始 PDE 算子的残差满足程度。
   - **频率外推鲁棒性:** 测试模型在训练频段之外（Out-of-distribution frequency）的重构稳定性。
---


