

---

# 复数隐空间 Transport：方法设想、形式化、验证与实验设计

## 1. 问题背景与目标

我们当前的模型针对频率驱动的复数物理场进行重构。对每个频率 $\omega$，场表示为

$$
u(x,\omega)=u_R(x,\omega)+i\,u_I(x,\omega)
$$

其中 $(u_R,u_I)$ 分别为实部和虚部。

现有表示采用**共享空间基** $\Phi(x)$，并分别学习实部和虚部对应的 latent core：

$$
u_R(x,\omega)\approx \langle G_R(\omega),\Phi(x)\rangle,\qquad
u_I(x,\omega)\approx \langle G_I(\omega),\Phi(x)\rangle.
$$

将二者合并，可写为复数 latent core 的形式：

$$
u(x,\omega)\approx \langle G^{(c)}(\omega),\Phi(x)\rangle,
\qquad
G^{(c)}(\omega):=G_R(\omega)+i\,G_I(\omega).
$$

核心问题是：

> 当频率从参考频率 $\omega_{\mathrm{ref}}$ 变化到目标频率 $\omega^\star$ 时，复数 latent core $G^{(c)}$ 是否存在一种具有物理含义的、可学习的、结构化的变化规律？

我们的目标是提出并验证一个 **complex transport** 假设，即：

- 目标频率下的复数 latent core，不是完全独立生成的；
    
- 它可以由参考频率下的复数 latent core 经过某种**复数旋转—缩放—残差修正**得到；
    
- 这种结构可作为：
    
    1. 一个独立的可解释模块；
        
    2. 一个 sampling-time guidance；
        
    3. 后续条件生成框架的物理先验。
        

---

## 2. 核心思想：Complex Transport

## 2.1 基本形式

我们希望用一个结构化映射 $T$ 描述从参考频率到目标频率的 latent 迁移：

$$
G^{(c)}(\omega^\star)\approx
T_{\omega_{\mathrm{ref}}\to\omega^\star}\big(G^{(c)}(\omega_{\mathrm{ref}})\big).
$$

为了使该映射具有"复数变换"的物理形式，我们不直接用任意黑箱网络预测 $G^{(c)}(\omega^\star)$，而是定义：

$$
\hat G^{(c)}(\omega^\star) =
\alpha_\psi(\omega_{\mathrm{ref}},\omega^\star,z_{\mathrm{ref}})
\odot
G^{(c)}(\omega_{\mathrm{ref}})
+
\beta_\psi(\omega_{\mathrm{ref}},\omega^\star,z_{\mathrm{ref}})
$$

其中：

- $\odot$ 表示逐元素乘法，或 block-wise/mode-wise 广播乘法；
    
- $\alpha_\psi \in \mathbb C^d$ 是复数 transport 系数；
    
- $\beta_\psi \in \mathbb C^d$ 是残差偏置项；
    
- $z_{\mathrm{ref}}$ 是从参考 latent 中抽取的低维 summary；
    
- $d$ 为 transport 作用维度，可为 element-wise、block-wise 或 mode-wise。
    

其中：

- 主项 $\alpha_\psi \odot G^{(c)}_{\mathrm{ref}}$ 表示**复数旋转—缩放 transport**；
    
- 残差项 $\beta_\psi$ 用来吸收无法由纯复乘法解释的部分。
    

---

## 2.2 极坐标形式

为增强物理解释性，我们将 $\alpha_\psi$ 参数化为：

$$
\alpha_\psi = \rho_\psi \odot e^{i\theta_\psi}
$$

其中：

- $\rho_\psi \in \mathbb R_+^d$：幅值增益；
    
- $\theta_\psi \in \mathbb R^d$：相位旋转量。
    

因此：

$$
\hat G^{(c)}(\omega^\star) =
\big(\rho_\psi \odot e^{i\theta_\psi}\big)\odot G^{(c)}(\omega_{\mathrm{ref}})
+
\beta_\psi.
$$

这对应于：

- 每个 latent mode 的幅值随频率发生重标定；
    
- 每个 latent mode 的相位随频率发生推进或旋转；
    
- 额外残差处理模态切换、复杂非线性、局部失配。
    

---

## 2.3 实虚部展开形式

若记

$$
\alpha_\psi = a_\psi + i b_\psi,\qquad
\beta_\psi = \beta_R + i\beta_I,
$$

则：

$$
\hat G_R^\star = a_\psi \odot G_R^{\mathrm{ref}} - b_\psi \odot G_I^{\mathrm{ref}} + \beta_R,
$$

$$
\hat G_I^\star = b_\psi \odot G_R^{\mathrm{ref}} + a_\psi \odot G_I^{\mathrm{ref}} + \beta_I.
$$

这对应标准的二维旋转—缩放结构：

$$
\begin{pmatrix}
\hat G_R^\star \\
\hat G_I^\star
\end{pmatrix}
=
\begin{pmatrix}
a_\psi & -b_\psi \\
b_\psi & a_\psi
\end{pmatrix}
\begin{pmatrix}
G_R^{\mathrm{ref}} \\
G_I^{\mathrm{ref}}
\end{pmatrix}
+
\begin{pmatrix}
\beta_R \\
\beta_I
\end{pmatrix}.
$$

因此，它不是普通 MLP 回归，而是受限于复乘法结构的 learnable transport。上述的变换矩阵只有两个参数，如果写成另一种形式：
$$
\begin{pmatrix}
\hat G_R^\star \\
\hat G_I^\star
\end{pmatrix}
=
\begin{pmatrix}
w_{11} & w_{12} \\
w_{21} & w_{22}
\end{pmatrix}
\begin{pmatrix}
G_R^{\mathrm{ref}} \\
G_I^{\mathrm{ref}}
\end{pmatrix}
+
\begin{pmatrix}
\beta_R \\
\beta_I
\end{pmatrix}.
$$

这种形式把两个参数变成了四个参数，可能有更好的表征能力。

---

## 3. 为什么这种形式是合理的

## 3.1 物理直觉

对于线性波动/亥姆霍兹类问题，频率变化本质上改变的是波数 $k$。对于固定介质和几何，解在共享模态上展开时，频率变化主要体现为模态系数的变化：

$$
u(x,\omega)\approx \sum_{r=1}^R g_r(\omega)\phi_r(x).
$$

此时，模态系数 $g_r(\omega)$ 是复数。  
当频率变化时，最自然的局部变化机制就是：

- 幅值增强/衰减；
    
- 相位推进/旋转。
    

这正对应复乘法：

$$
g_r(\omega^\star)\approx \rho_r e^{i\theta_r}g_r(\omega_{\mathrm{ref}}).
$$

因此，在共享空间基下，对 latent coefficient 使用 complex transport 是合理的。

---

## 3.2 局部频率展开支持

若 $g_r(\omega)\neq 0$ 且在 $\omega$ 上可微，则对小的 $\Delta\omega$ 有：

$$
g_r(\omega+\Delta\omega) =
g_r(\omega)+\Delta\omega\,g_r'(\omega)+O(\Delta\omega^2).
$$

若定义：

$$
c_r(\omega):=\frac{g_r'(\omega)}{g_r(\omega)}\in\mathbb C,
$$

则：

$$
g_r(\omega+\Delta\omega)
\approx
\big(1+\Delta\omega\,c_r(\omega)\big)g_r(\omega)
\approx
e^{\Delta\omega\,c_r(\omega)}g_r(\omega).
$$

设

$$
c_r(\omega)=\eta_r(\omega)+i\xi_r(\omega),
$$

则：

$$
e^{\Delta\omega\,c_r(\omega)} =
e^{\Delta\omega\,\eta_r(\omega)}
e^{i\Delta\omega\,\xi_r(\omega)}.
$$

这正是：

- 幅值项 $e^{\Delta\omega\,\eta_r}$
    
- 相位旋转项 $e^{i\Delta\omega\,\xi_r}$
    

这提供了一个局部理论支持：

> 在局部频率邻域内，复数模态系数的变化天然具有 rotation-scaling 的一阶近似形式。

---

## 4. Predictor 的设计

## 4.1 设计原则

predictor 不应直接输出 target core：

$$
\hat G^\star = \mathrm{MLP}(G_{\mathrm{ref}},\omega_{\mathrm{ref}},\omega^\star),
$$

因为这只是普通黑箱回归。

更合理的做法是让 predictor 输出**transport 参数**，而不是直接输出答案。

---

## 4.2 输入设计

推荐输入：

$$
[\Delta\omega,\ \omega_{\mathrm{ref}},\ \omega^\star,\ z_{\mathrm{ref}}],
\qquad
\Delta\omega:=\omega^\star-\omega_{\mathrm{ref}}.
$$

其中 $z_{\mathrm{ref}}$ 是参考 latent 的 summary，例如：

- $\mathrm{mean}(|G_{\mathrm{ref}}|)$
    
- pooled amplitude statistics
    
- PCA/MLP bottleneck
    
- mode-wise energy summary
    

不建议一开始直接输入完整 $G_{\mathrm{ref}}$，以免 predictor 黑箱化过强。

---

## 4.3 输出设计：Generator Form

最推荐形式是让 predictor 输出一个复 generator：

$$
\gamma_\psi = \eta_\psi + i\xi_\psi,
$$

并定义：

$$
\alpha_\psi = \exp(\Delta\omega\,\gamma_\psi).
$$

于是：

$$
\alpha_\psi =
e^{\Delta\omega\,\eta_\psi}
e^{i\Delta\omega\,\xi_\psi}.
$$

这意味着 predictor 学到的是：

- $\eta_\psi$：幅值增长/衰减率
    
- $\xi_\psi$：相位推进率
    

最终 transport 为：

$$
\hat G^{(c)}(\omega^\star) =
\exp(\Delta\omega\,\gamma_\psi)\odot G^{(c)}(\omega_{\mathrm{ref}})
+
\beta_\psi.
$$

这是最漂亮、最有理论味的一种形式。

---

## 4.4 粒度设计

transport 不一定要逐元素作用，可按以下粒度设计。

### A. Global scalar

$$
\alpha\in\mathbb C
$$

整个 core 一个复数 transport。

作用：作为最简单 baseline。

### B. Mode-wise

$$
\alpha\in\mathbb C^{R}
$$

每个主要 latent mode 一个复系数。

作用：推荐主模型。

### C. Block-wise

按 Tucker core 的某些 rank block 输出 transport 参数。

作用：在表达力与可解释性之间折中。

### D. Element-wise

$$
\alpha\in\mathbb C^{d}
$$

每个 latent 元素一个参数。

作用：作为最强上界，但不宜作为主版本。

---

## 5. 训练目标

## 5.1 基本 transport 监督损失

对训练样本中的频率对 $(\omega_{\mathrm{ref}},\omega^\star)$，有真实 latent：

$$
G^{(c)}_{\mathrm{ref}},\qquad G^{(c)}_{\mathrm{tar}}.
$$

定义预测：

$$
\hat G^{(c)}_{\mathrm{tar}} =
\alpha_\psi \odot G^{(c)}_{\mathrm{ref}} + \beta_\psi.
$$

基本损失：

$$
\mathcal L_{\mathrm{trans}} =
\left|
G^{(c)}_{\mathrm{tar}}-\hat G^{(c)}_{\mathrm{tar}}
\right|_2^2.
$$

实际实现中可用实部/虚部分别写：

$$
\mathcal L_{\mathrm{trans}} =
|G_{R,\mathrm{tar}}-\hat G_{R,\mathrm{tar}}|_2^2
+
|G_{I,\mathrm{tar}}-\hat G_{I,\mathrm{tar}}|_2^2.
$$

---

## 5.2 Identity consistency

当 $\omega^\star=\omega_{\mathrm{ref}}$ 时，应满足：

$$
\alpha(\omega,\omega)=1,\qquad \beta(\omega,\omega)=0.
$$

因此：

$$
\mathcal L_{\mathrm{id}} =
|\alpha(\omega,\omega)-1|_2^2
+
|\beta(\omega,\omega)|_2^2.
$$

---

## 5.3 Composition consistency

希望近似满足组合律：

$$
T_{\omega_2\to\omega_3}\circ T_{\omega_1\to\omega_2}
\approx
T_{\omega_1\to\omega_3}.
$$

对于纯 transport 系数，这意味着：

$$
\alpha_{13}
\approx
\alpha_{23}\odot\alpha_{12},
$$

以及

$$
\beta_{13}
\approx
\alpha_{23}\odot\beta_{12}+\beta_{23}.
$$

因此定义：

$$
\mathcal L_{\mathrm{comp}} =
\left|
\alpha_{13} - \alpha_{23}\odot\alpha_{12}
\right|_2^2
+
\left|
\beta_{13} - (\alpha_{23}\odot\beta_{12}+\beta_{23})
\right|_2^2.
$$

如果第一版不想太复杂，可先只约束 $\alpha$。

---

## 5.4 Smoothness regularization

对小的频率步长 $\delta\omega$，希望 transport 不要过于剧烈：

$$
\mathcal L_{\mathrm{smooth}} =
\left|
\alpha(\omega,\omega+\delta\omega)-1
\right|_2^2.
$$

或直接对 generator 做平滑约束：

$$
\mathcal L_{\mathrm{gen-smooth}} =
|\gamma(\omega+\delta)-\gamma(\omega)|_2^2.
$$

---

## 5.5 总损失

$$
\mathcal L =
\mathcal L_{\mathrm{trans}}
+
\lambda_{\mathrm{id}}\mathcal L_{\mathrm{id}}
+
\lambda_{\mathrm{comp}}\mathcal L_{\mathrm{comp}}
+
\lambda_{\mathrm{smooth}}\mathcal L_{\mathrm{smooth}}.
$$

---

## 6. 这个模块如何使用

## 6.1 作为独立可解释模块

最先做的版本应是**独立训练与验证**。  
即：

1. 先训练好主模型，得到稳定的 shared basis 与每个频率的 latent core。
    
2. 冻结 latent 表示。
    
3. 单独训练 transport predictor。
    

这样最干净，便于解释。

---

## 6.2 作为 sampling-time guidance

若主扩散模型当前仍训练为：

$$
p_\theta(G\mid \omega^\star),
$$

即没有 reference conditioning，那么 transport 模块可仅在采样时使用。

### Step 1: reference-induced guess

$$
\tilde G^{(c)}_{\mathrm{ref\to tar}} =
\alpha_\psi \odot G^{(c)}_{\mathrm{ref}} + \beta_\psi.
$$

### Step 2: 定义 reference prior

$$
p_{\mathrm{ref}}(G\mid G_{\mathrm{ref}},\omega_{\mathrm{ref}},\omega^\star)
\propto
\exp\left(
-\frac{1}{2\tau^2}
|G-\tilde G^{(c)}_{\mathrm{ref\to tar}}|_2^2
\right).
$$

### Step 3: 采样时加入额外 guidance

若原采样中已有：

- generative prior score
    
- target observation likelihood score
    

则现在总梯度为：

$$
\nabla_G \log p_{\mathrm{total}} =
\nabla_G \log p_\theta(G\mid \omega^\star)
+
\lambda_{\mathrm{obs}}\nabla_G \log p(y^\star\mid G)
+
\lambda_{\mathrm{ref}}\nabla_G \log p_{\mathrm{ref}}(G\mid G_{\mathrm{ref}}).
$$

其中：

$$
\nabla_G \log p_{\mathrm{ref}} =
-\frac{1}{\tau^2}(G-\tilde G^{(c)}_{\mathrm{ref\to tar}}).
$$

这是一条非常推荐的路线，因为：

- 不需要重训 diffusion prior；
    
- reference 只在采样时提供结构性引导；
    
- 物理解释强。
    

---

## 7. 如何验证这个想法是否成立

这里分为**可解释性验证**与**任务有效性验证**两类。

---

## 7.1 可解释性验证

### 实验 A：Complex ratio 的统计分析

对于样本 $s$ 和频率对 $(\omega_a,\omega_b)$，定义逐元素复比值：

$$
R_s(\omega_a,\omega_b) =
\frac{G_s^{(c)}(\omega_b)}
{G_s^{(c)}(\omega_a)+\epsilon}.
$$

写成极坐标：

$$
R_s=\rho_s e^{i\Delta\phi_s}.
$$

分析：

- $|R_s|$ 的分布
    
- $\arg(R_s)$ 的分布
    
- 分布随 $|\omega_b-\omega_a|$ 的变化
    

目标：

- 观察是否存在明显的幅值重标定与相位推进模式；
    
- 观察小频率差下是否更稳定。
    

### 实验 B：拟合最佳 diagonal complex transport

对固定频率对 $(\omega_a,\omega_b)$，在训练样本上求解最优 diagonal transport：

$$
\alpha^\star =
\arg\min_\alpha
\sum_s
|G_s^{(c)}(\omega_b)-\alpha\odot G_s^{(c)}(\omega_a)|_2^2.
$$

对于每个元素有闭式解：

$$
\alpha_j^\star =
\frac{\sum_s G_{s,j}^{(c)}(\omega_b)\overline{G_{s,j}^{(c)}(\omega_a)}}
{\sum_s |G_{s,j}^{(c)}(\omega_a)|^2}.
$$

然后在测试样本上评估：

$$
E_{\mathrm{diag-cplx}} =
\frac{|G_s^{(c)}(\omega_b)-\alpha^\star\odot G_s^{(c)}(\omega_a)|}
{|G_s^{(c)}(\omega_b)|}.
$$

并与以下 baseline 比较：

1. Identity:
    $$
    G(\omega_b)\approx G(\omega_a)
    $$
    
2. Real scalar scaling:
    $$
    G(\omega_b)\approx cG(\omega_a),\quad c\in\mathbb R
    $$
    
3. Complex scalar scaling:
    $$
    G(\omega_b)\approx cG(\omega_a),\quad c\in\mathbb C
    $$
    
4. Diagonal complex transport
    
5. Full linear map（作为上界）
    

目标：

- 证明 diagonal complex transport 在 latent space 中已经具有明显解释力；
    
- 证明它优于简单缩放或恒等映射。
    

### 实验 C：误差随频率差变化

定义：

$$
E(\Delta\omega)=
\mathbb E_{s,\omega}
\left[
\frac{
|G_s^{(c)}(\omega+\Delta\omega)-\hat T_{\Delta\omega}(G_s^{(c)}(\omega))|
}{
|G_s^{(c)}(\omega+\Delta\omega)|
}
\right].
$$

画：

- diagonal complex transport error vs $\Delta\omega$
    
- real scaling error vs $\Delta\omega$
    
- full linear transport vs $\Delta\omega$
    

目标：

- 证明 complex transport 是一个局部结构先验；
    
- 小 $\Delta\omega$ 下最成立；
    
- 随频率差增大逐渐变弱，但仍优于简单 baseline。
    

### 实验 D：field-space vs latent-space transport

直接在 field space 中做同样的拟合：

$$
u(\omega_b)\approx \beta\odot u(\omega_a)
$$

并与 latent-space transport 比较。

目标：

- 证明复数 rotation-scaling 不是 field space 的显然规律；
    
- 它是在共享模态 latent space 中显著变得可组织。
    

这是非常关键的一组证据。

---

## 7.2 任务有效性验证

### 实验 E：transport predictor 直接预测 target latent

比较：

1. No transport
    
2. Real scalar transport
    
3. Complex scalar transport
    
4. Diagonal complex transport
    
5. Learnable generator-based transport
    
6. Learnable transport + residual $\beta$
    

指标：

- latent reconstruction error
    
- decode 后 field reconstruction error
    
- phase error
    
- amplitude error
    

目标：

- 证明 learnable structured transport 优于简单手工 baseline；
    
- 证明 residual 项是否必要。
    

### 实验 F：sampling-time guidance ablation

比较：

1. No reference
    
2. Reference as initialization only
    
3. Reference as guidance only
    
4. Reference as init + guidance
    

具体地：

- Init only:
    $$
    G_T = \tilde G_{\mathrm{ref\to tar}} + \sigma_T \epsilon
    $$
    
- Guidance only:  
    reference 只出现在 $\nabla_G \log p_{\mathrm{ref}}$ 中
    
- Init + guidance:  
    二者都用
    

目标：

- 验证 reference 放在采样时是否有效；
    
- 判断是否有必要未来再把 reference 放进训练条件。
    

### 实验 G：transport 粒度 ablation

比较：

- global scalar
    
- mode-wise
    
- block-wise
    
- element-wise
    

目标：

- 判断最合理的表达粒度；
    
- 平衡可解释性与性能。
    

### 实验 H：是否需要 residual term $\beta$

比较：

- pure transport:
    $$
    \hat G = \alpha\odot G_{\mathrm{ref}}
    $$
    
- transport + residual:
    $$
    \hat G = \alpha\odot G_{\mathrm{ref}}+\beta
    $$
    

目标：

- 判断 rotation-scaling 是否已足够；
    
- 评估 residual 吸收复杂现象的价值。
    

---

## 8. 推荐的实施顺序

为了避免一次性改动太多，建议按以下顺序推进。

### Phase 1：可解释性阶段

目标：先证明 latent 里确实存在可用的 complex transport 规律。

1. 抽取所有样本、所有频率的 latent core
    
2. 做 complex ratio 统计分析
    
3. 做 diagonal complex transport 拟合
    
4. 画 error vs frequency gap
    
5. 比较 latent space 与 field space
    

这一步不改主模型。

---

### Phase 2：独立 predictor 阶段

目标：训练一个结构化 transport predictor。

1. 设计 $z_{\mathrm{ref}}$
    
2. 先做 simplest version：
    
    - mode-wise
        
    - no residual
        
    - generator form
        
3. 训练 predictor
    
4. 做 transport reconstruction 实验与 ablation
    

这一步仍不动 diffusion 主体。

---

### Phase 3：sampling-time integration

目标：将 transport 模块接入当前采样过程。

1. 先试 init only
    
2. 再试 guidance only
    
3. 再试 init + guidance
    
4. 调 $\lambda_{\mathrm{ref}},\tau$
    

看 reference 是否能在不改训练的前提下带来收益。

---

### Phase 4：更完整系统

若前三步验证成功，再考虑：

- 多参考频率融合
    
- 将 reference 放进 diffusion training
    
- complex transport + PDE residual 的联合版本
    

---

## 9. 推荐主结果图表

### 图 1

Complex ratio 的幅值与相位分布图。

### 图 2

Different transport assumptions 的 latent error vs frequency gap。

### 图 3

Field-space transport vs latent-space transport 对比图。

### 图 4

Transport predictor 的 ablation：

- real scalar
    
- complex scalar
    
- diagonal
    
- generator-based
    
- + residual
    

### 图 5

Sampling-time reference 使用方式对比：

- no ref
    
- init only
    
- guidance only
    
- init + guidance
    

### 表 1

不同 transport 粒度的性能表。

### 表 2

不同频率 regime 下的性能表：

- nearby frequency
    
- medium gap
    
- far gap
    
- low-to-high
    
- high-to-low
    

---

## 10. 需要特别避免的错误

### 错误 1

直接用 MLP 从 $(G_{\mathrm{ref}},\omega_{\mathrm{ref}},\omega^\star)$ 回归 $G_{\mathrm{tar}}$，再称其为 complex transport。  
这不成立，因为没有结构保证。

### 错误 2

一开始就把 transport predictor 和 diffusion 主模型端到端一起训。  
这样很难解释，也很难定位问题。

### 错误 3

只报最终重构误差，不做 latent transport 可解释性分析。  
这样 reviewer 会觉得 transport 只是新的网络部件，而不是有物理意义的结构。

---

## 11. 最终建议

这条线最好的推进方式不是"直接把它变成主模型"，而是：

1. **先做成一个可解释的结构发现与验证模块**
    
2. **再做成一个独立 predictor**
    
3. **再把它作为 sampling-time guidance 加入当前系统**
    

这样每一步都能单独回答一个问题：

- latent 里有没有这种规律？
    
- 这种规律能不能被 learnable predictor 捕捉？
    
- 这种规律能不能真正提升目标任务？
    

如果这三步都成立，这个 transport 就不再只是一个想法，而会成为整篇 paper 里非常有辨识度的一条主线。

---

## 12. 多 Reference Transport 的主设定

在实际 inference 时，不应只取一个参考频率做 transport，而应取**最近的若干个 reference**，各自 transport 到目标频率后再做加权融合。这比单 reference 更自然、更稳，也更具物理合理性。**推荐将多 reference 版本作为主设定，单 reference 退为 ablation。**

---

### 12.1 为什么多 reference 更合理

只取最近单个参考频率存在以下问题：

1. **信息利用不充分**：两侧邻近频率通常都带有信息，尤其当 $\omega^\star$ 落在它们之间时。
2. **单 reference 偏差更大**：一个 reference 的 transport 误差可能较大，尤其当频率差稍大时。
3. **物理上更像局部插值/外推**：对平滑的频率响应族，目标频率应由附近多个频率共同约束，而非由单点”跳过去”。

更合理的 inference 逻辑是：

> 先从多个参考频率分别 transport 到目标频率，再把这些 transported guesses 做加权叠加，形成目标频率 latent 的 reference-induced estimate。

---

### 12.2 正式形式

设有 $K$ 个邻近参考频率：

$$
\mathcal{N}(\omega^\star) = \{\omega_{(1)},\dots,\omega_{(K)}\}.
$$

每个参考频率对应 latent core $G^{(c)}(\omega_{(k)})$。对每个 reference 做单独 transport：

$$
\tilde G_k^{(c)}(\omega^\star) = \alpha_k \odot G^{(c)}(\omega_{(k)}) + \beta_k,
$$

其中：

$$
\alpha_k = \alpha_\psi(\omega_{(k)},\omega^\star,z_k),\qquad
\beta_k = \beta_\psi(\omega_{(k)},\omega^\star,z_k).
$$

然后做加权叠加，得到 **reference-induced target estimate**：

$$
\tilde G^{(c)}_{\mathrm{multi}}(\omega^\star) =
\sum_{k=1}^{K} w_k\,\tilde G_k^{(c)}(\omega^\star),
\qquad \sum_{k=1}^{K} w_k = 1.
$$

---

### 12.3 权重设计

**方案 A：按频率距离的固定权重（推荐主版本）**

$$
w_k = \frac{\exp(-\lambda |\omega^\star-\omega_{(k)}|)}{\sum_{j=1}^{K}\exp(-\lambda |\omega^\star-\omega_{(j)}|)},
$$

或更简单的倒数距离形式：

$$
w_k \propto \frac{1}{|\omega^\star-\omega_{(k)}|+\epsilon}.
$$

无额外参数，物理直觉直接。

**方案 B：线性插值权重（$K=2$ 最简起点）**

取左右最近两个频率 $\omega_l < \omega^\star < \omega_r$：

$$
w_l = \frac{\omega_r-\omega^\star}{\omega_r-\omega_l},\qquad
w_r = \frac{\omega^\star-\omega_l}{\omega_r-\omega_l}.
$$

则：

$$
\tilde G^{(c)}_{\mathrm{multi}}(\omega^\star) =
w_l\,\tilde G_l^{(c)}(\omega^\star) + w_r\,\tilde G_r^{(c)}(\omega^\star).
$$

这可称为 **two-sided transported interpolation**，结构最干净，适合作为第一版实现。

**方案 C：Learnable attention weights（仅作 ablation）**

$$
w_k = \mathrm{softmax}(f_{\mathrm{attn}}(\omega^\star,\omega_{(k)},z_k)).
$$

权重完全可学会使整体退化为黑箱融合，建议仅作为对比实验。

---

### 12.4 “先 transport 再融合”的合理性

有两种可能的顺序：

**路线 1（推荐）：先各自 transport，再融合**

$$
\tilde G_k^\star = T_{k\to\star}(G_k),\qquad
\tilde G^\star = \sum_k w_k\,\tilde G_k^\star.
$$

**路线 2：先融合参考 latent，再统一 transport**

$$
\bar G_{\mathrm{ref}} = \sum_k w_k G_k,\qquad
\tilde G^\star = T_{\mathrm{fused}\to\star}(\bar G_{\mathrm{ref}}).
$$

推荐路线 1：每个 reference 到目标频率的频率差不同，对应 transport 参数也不同。正确的逻辑是各参考频率先按自己的频率 gap 被搬运到目标频率坐标系，再在那里融合，而非在 source 空间提前混合。

---

### 12.5 两段式系统结构

整体采用**单 reference transport predictor + 多 reference aggregator** 的两段式结构。

**模块 A：单 reference transport predictor**

- 输入：$(\omega_k,\omega^\star,z_k)$
- 输出：$(\alpha_k,\beta_k)$
- 计算：$\tilde G_k^\star = \alpha_k\odot G_k+\beta_k$

**模块 B：多 reference aggregator**

- 输入：$\{\tilde G_k^\star,\omega_k,\omega^\star\}_{k=1}^{K}$
- 输出：$\tilde G^\star_{\mathrm{multi}} = \sum_k w_k\,\tilde G_k^\star$

transport predictor 负责”如何从某个 reference 搬运到目标频率”，aggregator 负责”多个邻近 reference 如何融合”，职责清晰，便于独立分析与消融。

---

### 12.6 Sampling-time Guidance 的完整公式

将单 reference 的 guidance center 替换为多 reference 融合结果：

$$
\tilde G_{\mathrm{multi}} = \sum_{k=1}^{K} w_k \big(\alpha_k\odot G_k+\beta_k\big).
$$

Reference prior 定义为：

$$
p_{\mathrm{ref}}(G) \propto \exp\left(-\frac{1}{2\tau^2}|G-\tilde G_{\mathrm{multi}}|^2\right).
$$

采样时 guidance 梯度：

$$
\nabla_G \log p_{\mathrm{ref}}(G) = -\frac{1}{\tau^2}(G-\tilde G_{\mathrm{multi}}).
$$

该改动与单 reference 框架完全兼容，仅替换 guidance center 的来源，无需修改 diffusion 训练过程。

---

### 12.7 补充实验设计

**实验 I：单 reference vs 多 reference**

| 设置 | 权重方式 |
|------|---------|
| 最近单个 reference | — |
| 最近两个 reference | 线性插值权重 |
| 最近两个 reference | 距离 softmax 权重 |
| 最近 $K$ 个 reference | 距离衰减权重 |

指标：latent transport error、decode 后 field error、phase error。

**实验 J：融合顺序 ablation**

比较路线 1（先 transport 再融合）与路线 2（先融合再 transport）。若路线 1 明显更优，则为所提 formulation 提供直接支撑。

**实验 K：权重策略 ablation**

比较固定物理权重（线性插值、倒数距离）与可学习权重的差异。若固定权重已足够好，则叙事更简洁且物理解释更强。