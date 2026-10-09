# LBQ 谱多样性正则实验

## 实验边界

本实验检验：对 Video readout 的 LBQ hidden 施加谱多样性约束，能否减少表示集中，并改善 Action 预测及 LIBERO-plus 鲁棒性。

**attention 集中于 q16，不等于 hidden representation 已经发生谱坍塌。** 先用相同配置的零权重对照确认谱指标，再结合完整任务成功率判断。有效秩提高本身不是任务性能改善的证据，也不应强制 32 个 query 均匀获得 attention。

新任务继承主线 32 LBQ、2 层 alternating_cross_self、lbq_only、末层 readout 的 full-finetune 配置，只覆盖 `model.loss` 的两个字段。Video/Action forward、首帧可见性、future_video_reads_lbq、初始化顺序、冻结策略、优化器、原 action/video loss 及其 timestep 权重均不变。没有恢复此前的 inference-only Action feedback。

从主线使用的 Wan/Action 预训练权重重新开始配对实验，不加载已有 LIBERO checkpoint，不设置 resume。原有 Action 层权重映射逻辑原样保留。

## 公式与实现

对每个样本的 `H = auxiliary["lbq_tokens"]` 单独计算，默认形状 `[32,3072]`，取自 Video 指定 readout layer，位于 Action `lbq_embedding` projector 之前：

```text
X_i = H_i / max(||H_i||_2, 1e-6)
G = X X^T                                  # [N,N]
e = clamp_min(eigvalsh(G), 0)
p = e / max(sum(e), 1e-12)
L_spec = mean_batch(log(N) + sum(p * log(max(p, 1e-12))))
L_total = lambda_video * L_video + lambda_action * L_action
          + lambda_lbq_spectral * L_spec
```

行归一化不更改实际送入 Action 的 hidden。完整 hidden 维度参与 Gram 计算，不使用随机投影，不混合 batch 内不同样本，不加对角 jitter。归一化、Gram 和特征值分解都在禁用 autocast 的 FP32 下执行；分解的是 32x32 矩阵，而非 3072x3072 矩阵。

依据 Being-H0.7 §3.3 的负谱熵思想进行适配。论文使用随机子空间投影、另有 norm/alignment loss；本实验不引入这些项目，且增加常数 `log(N)` 方便观察，梯度不受该常数影响。**这是针对 BridgeWAM 的适配，不是完整复现 Being-H0.7 的训练方法。** 原文 post-training 阶段并未继续使用这些 anti-collapse 正则。

参考：[Being-H0.7 §3.3](https://arxiv.org/html/2605.00078v1#S3.SS3)。

边界：N=1 时正则为带零梯度的零；全零 hidden 的 effective rank 报告为 0，loss 为 log(N)，不会通过 jitter 伪造多样性。完全相同/全零 token 的对称状态可能没有有效的分离梯度，本实验不声称单靠该项必然能恢复完全坍塌。接近零范数时归一化 epsilon 会影响尺度不变性。

## 梯度与诊断

`L_spec` 直接回传到 Video readout 之前的 LBQ/Video 路径，以及参与该路径且可训练的 proprio encoder；不会直接回传到下游 Action projector 或 Action blocks。后两者仍由原 action loss 更新，video loss 及原来的跨模块梯度路径未变。若某条路径被冻结，该项不会将其解冻。

配置：

```yaml
model:
  loss:
    lambda_lbq_spectral: 0.0        # 原任务默认：关闭
    lbq_spectral_diagnostics: false
```

独立 spectral task 默认权重 `1e-4`、诊断开启。仅在主线 `bridgewam.models.wan22.factory.create_bridgewam` 实现；在其他 joint/IDM/ablation factory 上开启会明确报错，避免静默忽略配置。

日志新增：

| 指标 | 意义 |
|---|---|
| `loss_lbq_spectral_raw` | 未乘权重的谱损失 |
| `loss_lbq_spectral` | 加入总 loss 的加权项 |
| `lbq_{embedding,readout,context}_effective_rank` | exp(谱熵)，先按样本计算再平均；全零为 0 |
| `lbq_{embedding,readout,context}_top1_mass` | 最大特征值占总谱能量的比例 |
| `lbq_{embedding,readout,context}_top4_mass` | 最大 min(4,N) 个特征值的比例 |
| `lbq_{embedding,readout,context}_token_norm` | token L2 范数的均值 |
| `lbq_{embedding,readout,context}_low_norm_fraction` | 范数小于 1e-6 的 token 比例 |
| `lbq_{embedding,readout,context}_mean_cosine` | 非对角 token 对的平均余弦相似度 |

`embedding` 是可学习的初始 query 参数，`readout` 是 loss 的约束目标，`context` 是经过 Action projector 的表示。前者对 batch 共享；后两者按每个样本独立计算。embedding/context 仅作 detached 诊断。平均余弦可能被正负抵消，不能单独用来判断坍塌。

关闭诊断但权重大于零时，仍记录两项 loss；权重为零且诊断开启时，对 detached readout 计算，绝不向总 loss 加入 `0 * L_spec`。两个选项均关闭时不调用谱函数。诊断开启会在每个训练 step 计算三个小谱，日志频率仍由 `log_every` 控制。

现有 trainer 会自动聚合、记录所有新增标量，不需要修改 trainer。文本日志使用四位小数，小权重项可能显示为 `0.0000`；同时看 raw loss，启用 W&B 时可查看未被文本格式舍入的指标。

## 训练指令

服务器先同步本次修改。以下不实际启动服务器训练；在服务器自行执行。两次任务**串行运行**，不要在同一组 GPU 上同时启动。数据、seed、训练预算必须一致。

```bash
cd /path/to/BridgeWAM
conda activate bridgewam
export PYTHONPATH="$PWD/LIBERO:$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/checkpoints
export DIFFSYNTH_SKIP_DOWNLOAD=true
export BRIDGEWAM_TRAIN_OUTPUT_BASE=/path/to/training-results

# 对照组：同一新 task，正则权重为零，保留诊断。
export RUN_ID=lbq_spectral_control_seed42_$(date +%Y%m%d_%H%M%S)
export BRIDGEWAM_TMUX_SESSION_NAME="$RUN_ID"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/train_zero1.sh 8 \
    task=libero_uncond_2cam224_lbqs_spectral_2layer_fullfinetune_1e-4 \
    seed=42 num_workers=8 save_every=5000 resume=null \
    model.skip_dit_load_from_pretrain=false \
    model.action_dit_pretrained_path="$DIFFSYNTH_MODEL_BASE_PATH/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt" \
    model.loss.lambda_lbq_spectral=0.0 \
    model.loss.lbq_spectral_diagnostics=true \
    data.train.text_embedding_cache_dir=/path/to/text-cache/libero
```

对照组结束后，在同一环境运行实验组：

```bash
export RUN_ID=lbq_spectral_1e-4_seed42_$(date +%Y%m%d_%H%M%S)
export BRIDGEWAM_TMUX_SESSION_NAME="$RUN_ID"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/train_zero1.sh 8 \
    task=libero_uncond_2cam224_lbqs_spectral_2layer_fullfinetune_1e-4 \
    seed=42 num_workers=8 save_every=5000 resume=null \
    model.skip_dit_load_from_pretrain=false \
    model.action_dit_pretrained_path="$DIFFSYNTH_MODEL_BASE_PATH/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt" \
    model.loss.lambda_lbq_spectral=1.0e-4 \
    model.loss.lbq_spectral_diagnostics=true \
    data.train.text_embedding_cache_dir=/path/to/text-cache/libero
```

输出目录为 `$BRIDGEWAM_TRAIN_OUTPUT_BASE/<task>/$RUN_ID`，以脚本实际打印路径为准。没有自动加载历史 LIBERO checkpoint；`skip_dit_load_from_pretrain=false` 保证沿用 Video/Action 预训练初始化。

先进行 `max_steps=10 save_every=10 eval_every=0` smoke test 检查有限 loss/梯度和日志，随后正式训练重新开始，不要把 smoke ckpt 当正式实验起点。正式训练保持原 eval 流程及步数预算。

如 1e-4 约束没有明显影响，可以补充 1e-5/1e-3；这不是保证有效的最佳超参数。建议至少 3 个配对 seed，每个 seed 下 control/treatment 同配置、同初始化及数据划分。不要只挑最高分 checkpoint：比较固定 step 和预先约定的选模标准。

## 验证与解读

本地或服务器依赖环境中执行：

```bash
PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -m unittest -v \
    tests.test_lbq_spectral_regularization \
    tests.test_latent_bridge_queries \
    tests.test_alternating_action_dit
```

测试覆盖数值边界、BF16 输入/FP32 谱计算、真实小型 Video/LBQ/Action 训练前向、梯度路由、零权重一致性、初始化 RNG/权重一致性、checkpoint schema 和十步 Action 推理一致性，以及 Hydra 配置传递。

正式评测仍使用现有 LIBERO/LIBERO-plus manager，指定对应 checkpoint 和训练生成的 dataset_stats；推理不执行正则项，也没有额外 query 或缓存。可使用新 task 或结构完全相同的原主线 task。新增配置不进入 checkpoint 架构元数据，无新增参数或 buffer；训练配置文件用于保留正则超参数，续训时需显式保持这些参数一致。

重点比较：

1. Readout effective rank 是否确实提高、top1/top4 mass 是否降低，且没有范数异常。
2. Action context 是否也保留该变化。如果 readout 变丰富但 context 再度集中，需要调查 projector/任务需求，而不是直接增加权重。
3. 同一批评测样本、同一初始状态下的完整 LIBERO 和 LIBERO-plus 成功率，以及各 suite/扰动类别，不能只看 loss 或单张 attention 图。
4. 若谱指标改善而成功率下降，只能说多样性被提高，不能说“学到更多有用物理信息”；模型可能需要任务特定的低秩瓶颈。

实现位置：`src/bridgewam/models/wan22/lbq/spectral_regularization.py`；接入位于 `BridgeWAM.training_loss` 和模型工厂；默认配置为 `configs/model/bridgewam.yaml`；新任务位于 `configs/task/libero_uncond_2cam224_lbqs_spectral_2layer_fullfinetune_1e-4.yaml`。
