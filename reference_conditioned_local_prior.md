# Reference-Conditioned Local Prior
## 基于 local weighted mean + diagonal uncertainty 的正式说明
## 以及其 Gaussian Process（对角形式）高级版本

---

# 1. 背景与目标

当前系统中，目标频率 $$\omega^\star$$ 下的 latent core $$G^\star$$ 是通过以下三类信息共同约束得到的：

1. 生成先验 $$p_\theta(G^\star \mid \omega^\star)$$
2. 目标频率稀疏观测的 likelihood
3. 目标频率对应的 PDE / wave-physics guidance

在这个框架下，**same-sample reference cores** 的作用并不是直接替代 likelihood，也不应简单地额外增加很多 heuristic guidance 项，而更适合作为一个：

$$
\textbf{reference-conditioned local prior}
$$

也就是说，多个 reference cores 为 target core 提供：

- 一个 **局部中心**
- 一个 **局部不确定度**
- 一个 **same-sample family support 的近似几何**

从而在 posterior sampling 中塑造更合理的局部几何。

本文档分为两个层次：

1. **基础版**：local weighted mean + diagonal uncertainty  
2. **高级版**：Gaussian Process（对角形式）的 reference-conditioned local prior

---

# 2. 基本设定

设目标频率为：

$$
\omega^\star
$$

存在同一样本下的多个 reference frequencies：

$$
\Omega_{\mathrm{ref}} = \{\omega_1, \omega_2, \dots, \omega_K\}
$$

对应的 latent cores 为：

$$
\mathcal G_{\mathrm{ref}} = \{G_1, G_2, \dots, G_K\}, \qquad G_k \in \mathbb R^d
$$

其中 $$d$$ 是 latent core 展平后的维度，通常可能在几十到几百之间。

我们的目标是基于 $$\{(\omega_k, G_k)\}_{k=1}^K$$，为 target core $$G^\star$$ 构造一个 local prior：

$$
q_{\mathrm{ref}}(G^\star \mid \omega^\star, \mathcal G_{\mathrm{ref}})
$$

并将其与已有的：

- learned prior
- likelihood
- PDE guidance

融合起来。

---

# 3. 基础版：local weighted mean + diagonal uncertainty

## 3.1 Local weighted mean

定义：

$$
\mu_{\mathrm{ref}} = \sum_{k=1}^K w_k G_k,
\qquad
\sum_{k=1}^K w_k = 1,
\qquad
w_k \ge 0
$$

这里 $$w_k$$ 表示 reference $$k$$ 对目标频率局部先验中心的贡献程度。

### 物理解释

$$
\mu_{\mathrm{ref}}
$$

不是 target core 的精确预测，而是：

- 一个 **same-sample family anchor**
- 一个由邻近 reference frequencies 定义出的 **局部 prior center**
- 一个对 target latent 的保守、局部、可解释的估计

它体现的不是“target = interpolation”，而是：

> 在没有 target direct evidence 之前，多个 same-sample reference cores 共同定义了 target core 所在的局部 family support 中心。

---

## 3.2 权重 $$w_k$$ 的设计

---

### 版本 A：两 reference 的线性插值权重

若只使用左右最近两个 reference：

$$
\omega_l < \omega^\star < \omega_r
$$

对应 cores：

$$
G_l,\qquad G_r
$$

则定义：

$$
w_l = \frac{\omega_r - \omega^\star}{\omega_r - \omega_l},
\qquad
w_r = \frac{\omega^\star - \omega_l}{\omega_r - \omega_l}
$$

于是：

$$
\mu_{\mathrm{ref}} = w_l G_l + w_r G_r
$$

#### 特点
- 无额外超参数
- 最容易解释
- 非常适合作为最干净的起步版本

---

### 版本 B：softmax 距离衰减权重

若使用最近 $$K$$ 个 reference frequencies，则定义频率距离：

$$
d_k := |\omega^\star - \omega_k|
$$

并设：

$$
w_k=
\frac{\exp(-\lambda d_k)}
{\sum_{j=1}^K \exp(-\lambda d_j)}
$$

其中 $$\lambda > 0$$ 控制局部性。

#### 参数建议

可设：

$$
\lambda = \frac{c}{\Delta\omega_{\mathrm{avg}}}
$$

其中 $$\Delta\omega_{\mathrm{avg}}$$ 是频率网格平均间隔，$$c \in \{0.5,1,2,3\}$$ 可用于调参。

---

### 版本 C：频率距离 + PDE-confidence 权重

若参考频率上的重构场 $$\hat u_k$$ 可获得，并可计算其 PDE residual：

$$
r_k := \|\mathcal R_{\omega_k}(\hat u_k)\|^2
$$

则可定义 reference 置信度：

$$
s_k = \exp(-\gamma r_k)
$$

最终权重为：

$$
w_k=
\frac{\exp(-\lambda d_k)\exp(-\gamma r_k)}
{\sum_{j=1}^K \exp(-\lambda d_j)\exp(-\gamma r_j)}
$$

这是一种更“物理”的增强版，但建议先作为 ablation，而不是第一版主模型。

---

## 3.3 Diagonal uncertainty

在 local weighted mean 基础上，定义每个 latent 维度的加权方差：

$$
\sigma_{{\rm ref},j}^2=
\sum_{k=1}^K w_k \big(G_{k,j} - \mu_{{\rm ref},j}\big)^2 + \epsilon
$$

其中：

- $$G_{k,j}$$ 表示第 $$k$$ 个 reference core 的第 $$j$$ 个维度
- $$\mu_{{\rm ref},j}$$ 是 local mean 的第 $$j$$ 个分量
- $$\epsilon > 0$$ 是数值稳定项

于是：

$$
\Sigma_{\mathrm{ref}}=
\mathrm{diag}(\sigma_{{\rm ref},1}^2,\dots,\sigma_{{\rm ref},d}^2)
$$

### 为什么用 diagonal，而不用 full covariance

若 latent core 维度较高（例如几百维），而 reference frequency 数量较少（例如 2 到 8 个），直接估 full covariance 会出现：

- 协方差估计病态
- 数值不稳
- 统计不可靠
- 引入不必要复杂度

因此 diagonal uncertainty 是一个非常合理且稳健的第一版选择。

---

## 3.4 基础版 local Gaussian prior

基于上述 $$\mu_{\mathrm{ref}}$$ 和 $$\Sigma_{\mathrm{ref}}$$，定义：

$$
q_{\mathrm{ref}}(G^\star \mid \omega^\star,\mathcal G_{\mathrm{ref}})=
\mathcal N(\mu_{\mathrm{ref}}, \Sigma_{\mathrm{ref}})
$$

即：

$$
q_{\mathrm{ref}}(G^\star)
\propto
\exp\left(
-\frac{1}{2}(G^\star-\mu_{\mathrm{ref}})^\top
\Sigma_{\mathrm{ref}}^{-1}
(G^\star-\mu_{\mathrm{ref}})
\right)
$$

由于 $$\Sigma_{\mathrm{ref}}$$ 是 diagonal，其逆易于计算：

$$
\Sigma_{\mathrm{ref}}^{-1}=
\mathrm{diag}\left(
\frac{1}{\sigma_{{\rm ref},1}^2},
\dots,
\frac{1}{\sigma_{{\rm ref},d}^2}
\right)
$$

因此 Gaussian local prior 的 score 为：

$$
\nabla_{G^\star}\log q_{\mathrm{ref}}(G^\star)=
-\Sigma_{\mathrm{ref}}^{-1}(G^\star-\mu_{\mathrm{ref}})
$$

逐维写为：

$$
[\nabla_{G^\star}\log q_{\mathrm{ref}}]_j=
-\frac{G_j^\star-\mu_{{\rm ref},j}}{\sigma_{{\rm ref},j}^2}
$$

---

# 4. 高级版：Gaussian Process（对角形式）的 local prior

基础版实际上可以看成一种非常简单的局部高斯近似。  
如果想进一步增强“reference-conditioned local prior”的统计结构，可以采用 **Gaussian Process（对角形式）**。

这里“对角形式”指的是：

> 对每个 latent 维度独立地在频率轴上做一个 1D Gaussian Process 回归，而不试图估计所有 latent 维度之间的 full covariance。

这对于高维 latent core 是最稳妥、最自然的高级版本。

---

## 4.1 建模形式

对每个 latent 维度 $$j=1,\dots,d$$，建立一维函数：

$$
g_j(\omega)
$$

并假设：

$$
g_j(\omega) \sim \mathrm{GP}(m_j(\omega), k_j(\omega,\omega'))
$$

给定 reference data：

$$
\{(\omega_k, G_{k,j})\}_{k=1}^K
$$

即可对目标频率 $$\omega^\star$$ 的该维度做 GP posterior inference，得到：

$$
g_j(\omega^\star) \mid \{(\omega_k, G_{k,j})\}
\sim
\mathcal N(\mu_{{\rm GP},j}, \sigma_{{\rm GP},j}^2)
$$

最终：

$$
q_{\mathrm{GP}}(G^\star \mid \omega^\star,\mathcal G_{\mathrm{ref}})=
\mathcal N(\mu_{\mathrm{GP}}, \Sigma_{\mathrm{GP}})
$$

其中：

$$
\mu_{\mathrm{GP}} = (\mu_{{\rm GP},1},\dots,\mu_{{\rm GP},d})
$$

$$
\Sigma_{\mathrm{GP}}=
\mathrm{diag}(\sigma_{{\rm GP},1}^2,\dots,\sigma_{{\rm GP},d}^2)
$$

---

## 4.2 GP 的均值函数选择

### 版本 A：zero mean

$$
m_j(\omega)=0
$$

最标准，但在 same-sample reference 很 informative 的情况下，通常不够好。

---

### 版本 B：reference weighted average mean（推荐）

定义：

$$
m_j(\omega^\star)=\sum_{k=1}^K w_k G_{k,j}
$$

然后 GP 实际上建模的是 residual：

$$
G_{k,j} = m_j(\omega_k) + \delta_{k,j}
$$

这意味着：

- 同一样本 reference cores 先给出一个 local family center
- GP 只补偿 target frequency 下相对于这个中心的局部变化

这是非常推荐的版本。

---

### 版本 C：two-sided linear mean（推荐于最近两 reference）

若只使用左右最近两个参考频率，则：

$$
m_j(\omega^\star)=
\frac{\omega_r-\omega^\star}{\omega_r-\omega_l}G_{l,j}
+
\frac{\omega^\star-\omega_l}{\omega_r-\omega_l}G_{r,j}
$$

这会使 GP 看起来像是对线性局部插值的 refinement，而不是主预测器本身。

---

## 4.3 Kernel 的建议

### 第一推荐：RBF kernel

$$
k(\omega,\omega')=
\sigma_f^2 \exp\left(
-\frac{(\omega-\omega')^2}{2\ell^2}
\right)
$$

#### 优点
- 最稳
- 参数少
- 非常适合“局部频率邻域内的平滑 prior”

#### lengthscale 建议
可设：

$$
\ell = c \cdot \Delta\omega_{\mathrm{avg}}
$$

其中 $$c \in \{1,2,3\}$$。

---

### 第二推荐：Matérn-\(3/2\) kernel

$$
k(\omega,\omega')=
\sigma_f^2
\left(
1+\frac{\sqrt{3}|\omega-\omega'|}{\ell}
\right)
\exp\left(
-\frac{\sqrt{3}|\omega-\omega'|}{\ell}
\right)
$$

#### 适用
若发现 RBF 过于平滑，可尝试 Matérn-\(3/2\)。

---

### 建议暂不考虑
- full multitask GP kernel
- deep kernel
- spectral mixture kernel

原因：这些会显著增加复杂度，而你当前目标只是构造 local prior，而不是做高维频谱建模。

---

## 4.4 对角 GP 的 posterior 公式

以某个维度 $$j$$ 为例。  
记：

- 参考频率向量：
$$
\omega_{\rm ref} = [\omega_1,\dots,\omega_K]^\top
$$
- 观测值向量：
$$
\mathbf g_j = [G_{1,j},\dots,G_{K,j}]^\top
$$

定义 kernel matrix：

$$
K_j = \big[k_j(\omega_a,\omega_b)\big]_{a,b=1}^K + \sigma_n^2 I
$$

定义 target-to-reference kernel 向量：

$$
k_{\star,j} =
\big[
k_j(\omega^\star,\omega_1),\dots,k_j(\omega^\star,\omega_K)
\big]^\top
$$

则 posterior mean 与 variance 为：

$$
\mu_{{\rm GP},j}=
m_j(\omega^\star)
+
k_{\star,j}^\top K_j^{-1}
\big(\mathbf g_j - m_j(\omega_{\rm ref})\big)
$$

$$
\sigma_{{\rm GP},j}^2=k_j(\omega^\star,\omega^\star)-k_{\star,j}^\top K_j^{-1} k_{\star,j}
$$

其中：

$$
m_j(\omega_{\rm ref})=
\big[m_j(\omega_1),\dots,m_j(\omega_K)\big]^\top
$$

---

## 4.5 对角 GP 的最终 local prior

将所有维度拼接：

$$
\mu_{\mathrm{GP}} = (\mu_{{\rm GP},1},\dots,\mu_{{\rm GP},d})
$$

$$
\Sigma_{\mathrm{GP}}=
\mathrm{diag}(\sigma_{{\rm GP},1}^2,\dots,\sigma_{{\rm GP},d}^2)
$$

于是：

$$
q_{\mathrm{GP}}(G^\star \mid \omega^\star,\mathcal G_{\mathrm{ref}})=
\mathcal N(\mu_{\mathrm{GP}}, \Sigma_{\mathrm{GP}})
$$

其 score 为：

$$
\nabla_{G^\star}\log q_{\mathrm{GP}}(G^\star)=
-\Sigma_{\mathrm{GP}}^{-1}(G^\star-\mu_{\mathrm{GP}})
$$

逐维写为：

$$
[\nabla_{G^\star}\log q_{\mathrm{GP}}]_j=
-\frac{G_j^\star-\mu_{{\rm GP},j}}{\sigma_{{\rm GP},j}^2}
$$

---

# 5. 与已有主系统的融合方式

无论是基础版 local Gaussian prior，还是高级版 diagonal GP local prior，都可以作为 learned prior 的补充：

$$
p_{\mathrm{fused}}(G^\star \mid \omega^\star,\mathcal G_{\mathrm{ref}})
\propto
p_\theta(G^\star \mid \omega^\star)\,
q_{\mathrm{ref}}(G^\star \mid \omega^\star,\mathcal G_{\mathrm{ref}})
$$

或者在高级版中：

$$
p_{\mathrm{fused}}(G^\star \mid \omega^\star,\mathcal G_{\mathrm{ref}})
\propto
p_\theta(G^\star \mid \omega^\star)\,
q_{\mathrm{GP}}(G^\star \mid \omega^\star,\mathcal G_{\mathrm{ref}})
$$

最终 posterior 为：

$$
p(G^\star \mid y^\star,\omega^\star,\mathcal G_{\mathrm{ref}})
\propto
p_\theta(G^\star \mid \omega^\star)\,
q_{\mathrm{ref/GP}}(G^\star \mid \omega^\star,\mathcal G_{\mathrm{ref}})\,
p(y^\star \mid G^\star,\omega^\star)\,
p_{\mathrm{pde}}(G^\star \mid \omega^\star)
$$

对应总 score：

$$
\nabla_{G^\star}\log p_{\mathrm{total}}=
\nabla_{G^\star}\log p_\theta(G^\star \mid \omega^\star)
+
\nabla_{G^\star}\log q_{\mathrm{ref/GP}}(G^\star)
+
\lambda_{\mathrm{obs}}\nabla_{G^\star}\log p(y^\star \mid G^\star,\omega^\star)
+
\lambda_{\mathrm{pde}}\nabla_{G^\star}\log p_{\mathrm{pde}}(G^\star \mid \omega^\star)
$$

---

# 6. 基础版与高级版的区别

## 基础版：local weighted mean + diagonal uncertainty
- 最简单
- 最稳健
- 几乎不需要统计建模
- 非常适合先验证“reference-conditioned local prior”是否有效

## 高级版：diagonal GP local prior
- 在频率轴上引入更明确的局部回归结构
- 均值和不确定度更有统计解释
- 更适合写成“reference-conditioned local surrogate prior”
- 但实现和调参会复杂一些

---

# 7. 推荐的实验设计

## 7.1 基础版 vs 高级版
比较：

1. no reference prior
2. local weighted mean + diagonal uncertainty
3. diagonal GP local prior

---

## 7.2 均值函数 ablation
比较：

1. zero mean
2. weighted-average mean
3. two-sided linear mean

---

## 7.3 kernel ablation（高级版）
比较：

1. RBF
2. Matérn-\(3/2\)

---

## 7.4 reference 数量 ablation
比较：
- nearest 2 references
- nearest 3 references
- nearest 4 references

---

## 7.5 与主系统结合效果
比较：

1. prior only
2. prior + likelihood
3. prior + likelihood + PDE
4. fused prior + likelihood + PDE

---

# 8. 推荐的实施顺序

## Step 1
先做最小版本：

- nearest-2 references
- two-sided linear mean
- diagonal uncertainty

验证 local prior 这个想法本身是否有价值。

## Step 2
若有效，再做高级版：

- diagonal GP
- weighted-average mean 或 linear mean
- RBF kernel

## Step 3
最后再尝试：
- softmax distance weights
- PDE-confidence weighting
- Matérn kernel

---

# 9. 最后总结

这个 reference-conditioned local prior 的设计逻辑是：

1. likelihood 负责注入 **target-specific sample evidence**
2. PDE 负责注入 **target-frequency wave physics**
3. reference-conditioned local prior 负责注入 **same-sample reference context**

其中：

- **基础版** 用 local weighted mean + diagonal uncertainty 构造一个稳健的 local Gaussian prior
- **高级版** 用 diagonal Gaussian Process 在频率轴上构造一个更有统计解释的 local prior

一句话概括：

$$
\text{multiple same-sample reference cores define a local probabilistic support for target-frequency latent inference}
$$

而不是直接去“预测” target core。
