# ocean_data — 水声复声压场实验流程

把 `ocean-acoustic-agent`（UnderwaterAcoustics.jl / AcousticsToolbox.jl）生成的
**2D 复声压场** 打包成与 Helmholtz / 弹性数据完全相同的 FTM 格式（C=2，`[real, imag]`），
直接复用现有的 FTM → 频率感知扩散 → DPS 重建 流水线。

## 物理映射

| 本项目 | 水声 | 说明 |
|--------|------|------|
| 频率 $\omega$（$M$ 个） | `frequency_hz` 扫频 | Hz 升高→波长缩短、干涉复杂 |
| 样本 $N$（BC/IC） | 不同 environment | 见下方采样开关 |
| 2D 平面 $(x,y)$ | (深度 $z$, 距离 $r$) 网格 H×W | `ReceiverGrid` |
| 复场 $(Re, Im)$ | `pressure_real/imag` | C=2 |

## 两个 conda 环境（重要）

- **数据生成**：`ocean-acoustic` 环境（含 `juliacall` + agent + Julia 后端）。
- **训练/评测**：`ACM-FD` 环境（torch 2.9 + CUDA）。两个阶段分开运行。

```bash
# 生成阶段
conda activate ocean-acoustic
export PATH="$HOME/.juliaup/bin:$PATH"

# 训练阶段
PYBIN=~/anaconda3/envs/ACM-FD/bin/python
```

> Bellhop/Kraken（变地形障碍物、温跃层 SSP）需要 `AcousticsToolbox.jl`，已安装。
> 仅 Pekeris（等速、range-independent）时不需要它。

## 1. 生成数据

每个采样因子都有独立的 **是否启用** 开关；关闭时用对应固定默认值。
生成时可只打开其中几个。

| 开关 | 作用 | 触发的模型 |
|------|------|-----------|
| `--vary_water_depth` | 水深 ~ U[wd_range] | — |
| `--vary_source_depth` | 源深 ~ U[sd_range] | — |
| `--vary_ssp` | 声速：变等速值；加 `--ssp_profile` 则用温跃层剖面 | 温跃层→Bellhop |
| `--vary_seabed` | 海底类型（按模型选兼容池） | — |
| `--vary_obstacle` | range-dependent 海山地形 | Bellhop |

模型自动选择：有障碍物 **或** 温跃层 SSP → `bellhop`，否则 → `pekeris_ray`（快）。
任一 (sample, freq) 求解失败会自动回退到最兼容配置（pekeris + rigid + 等速）并计入 `n_fallbacks`。

```bash
# 冒烟（仅变 ssp / 海底 / 源深 / 水深，pekeris，快）
python Generate_dataset.py --N 12 --grid_h 32 --grid_w 48 \
  --freq_train 100 150 200 250 300 350 400 --freq_extrap 500 600 \
  --vary_ssp --vary_seabed --vary_source_depth --vary_water_depth \
  --out ocean_dataset.h5

# 含障碍物 + 温跃层（Bellhop，较慢）
python Generate_dataset.py --N 50 --vary_obstacle --vary_ssp --ssp_profile \
  --vary_water_depth --vary_seabed --out ocean_dataset_obstacle.h5
```

输出 `ocean_dataset.h5`（键 `data (N,M,H,W,2)` / `mask_tr` / `omega` / `grid_x(深度,len H)` /
`grid_y(距离,len W)` / `metadata` / `data_scale`）+ 旁挂 `*_metadata.npy`。
逐频归一化（与弹性一致），坐标轴归一化到 [0,1]。每个样本的环境参数记录在
`metadata["samples"]` 中以便复现。

运行时说明：agent 每次调用都会向 `OUTPUT_DIR` 写一个 case 目录；脚本默认把它指到临时目录
并即时清理（`--keep_runs` 保留，`--output_dir` 指定位置）。生成默认串行（`--n_jobs>1`
会为每个 worker 各起一个 Julia 运行时，开销大）。

## 2. 训练 FTM → 扩散 → DPS 重建

三个脚本由 `elastic_data/` 复制而来，对通道数通用，C=2 自动识别。

```bash
$PYBIN train_FTM_GPU.py --data_h5 ocean_dataset.h5 --out ckp/ftm_ocean.pt \
  --rank_x 24 --rank_y 24

$PYBIN train_diffusion.py --ftm_ckpt ckp/ftm_ocean.pt --out ckp/diffusion_ocean.pt

$PYBIN test_diffusion.py --diff_ckpt ckp/diffusion_ocean.pt --ftm_ckpt ckp/ftm_ocean.pt \
  --data_h5 ocean_dataset.h5 --out_dir visual_data/ocean_eval \
  --dps_weight 2.0 --guidance_grad_clip 1.0 --phys_weight 0
```

注意：
- **`--phys_weight 0`**（默认）。水声没有像弹性那样可内联的简单 PDE 残差算子，
  `test_diffusion.py` 里的弹性物理残差路径保持关闭（`pde_res` 显示 NaN 属正常）。
- DPS 引导对欠训练模型较敏感，建议从 **较小 `--dps_weight`（如 2）+ `--guidance_grad_clip 1.0`**
  起步，避免引导梯度爆炸出 NaN；模型训练充分后可再调大。

## 3. 评估 FTM 重构质量（test_FTM.py）

用训练好的**共享基函数 + 核心张量**重构全场，计算相对误差并可视化（real / imag / |p| /
phase 的 GT vs Pred vs Err，加上频率-RMSE 曲线、逐样本 RMSE、CSV、summary.json）。

```bash
$PYBIN test_FTM.py --ckpt ckp/ftm_ocean.pt --data_h5 ocean_dataset_dense.h5 \
  --out_dir visual_data/ftm_eval --max_eval_samples 60 --num_visualize 6 \
  --freq_idx 4 --sample_idx 2
```

**关于观测密度（重要）**：`train_FTM_GPU.py` 的核心张量是**只在观测点上拟合**的。
水声场能量高度集中在声源附近、远场近乎为零（动态范围极大），所以：

- 用 **稀疏观测**（如 `obs_ratio=0.1`，每频≈317 点）直接训 FTM，核心张量会过拟合掩膜，
  未观测点重构成椒盐噪声、全场相对 RMSE ≈ 1.5。**这正是本项目引入扩散先验 + DPS 的动机**
  （FTM 单独无法从极稀疏观测恢复全场）。
- 要单独考察 **FTM 的表达/重构能力**，应在**稠密观测**上训练。脚本提供了一个轻量做法：
  `ocean_dataset_dense.h5` = 复制 `data` + 全 1 掩膜（由 h5py `copy` 流式生成，见生成命令注释）。
  在其上训 FTM（rank 32）后，全场相对 RMSE ≈ 0.39，且重构与 GT 视觉吻合；误差随频率升高而
  增大（100 Hz≈0.1 → 600 Hz≈0.6，高频模态更精细，定秩可分基难以表达），远场相位误差大是因为
  那里 |p|≈0、相位本身无意义。这也是给扩散模型提供干净核心张量的推荐做法。

生成稠密掩膜副本：
```python
import h5py, numpy as np
s=h5py.File('ocean_dataset.h5','r'); d=h5py.File('ocean_dataset_dense.h5','w')
for k in ('omega','grid_x','grid_y','metadata','data_scale'): s.copy(k,d)
s.copy('data',d)
M,H,W,C=s['data'].shape[1:]
d.create_dataset('mask_tr',data=np.ones((M,H,W,C),np.uint8),compression='gzip',compression_opts=4)
```

## 已验证

`Generate_dataset.py`（pekeris 与 bellhop/障碍物两条路径，0 回退）→ FTM → 扩散 → DPS
全流程跑通；DPS 相对无引导先验有正向改善。`test_FTM.py` 在稠密观测上重构 real/imag/|p|/phase
与 GT 吻合（相对 RMSE≈0.39，随频率升高而增大）。Baseline（confild/fno/lrtfr）暂未接入。
