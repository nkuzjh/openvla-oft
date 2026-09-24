# OpenVLA-OFT · CSGO Seen-10 实验设计与验收记录

本文记录已实现的 legacy / aligned v2 的设计、配置依据、实现边界和验收证据。环境、手动运行命令、输出目录和正式结果见 [CSGO_SEEN10.md](CSGO_SEEN10.md)。记录更新于 2026-09-24，代码工作基于 `main@41c7227ce3bdfe48acfef7a6104958836201b0f7` 及工作区中的 aligned 变更；文档整理后已补充跨服务器路径适配，见第 7.3 节；不重新运行模型或正式实验。

文档结构与记录范围参考 [X-VLA 运行说明](../X-VLA/CSGO_SEEN10.md)、[X-VLA 方案](../X-VLA/CSGO_SEEN10_PLAN.md)、[RDT 运行说明](../RoboticsDiffusionTransformer/CSGO_SEEN10.md) 和 [RDT 方案及验收](../RoboticsDiffusionTransformer/CSGO_SEEN10_PLAN.md)。各项目的结构、配置和验收结果分别记录，不相互借用完成状态。

## 1. 范围、比较对象与设计依据

只接入 Benchmark v2 Seen-10 localization。主比较对象为 UniLIP `exp32_loc`，`exp32` 的定位结果用于观察联合生成训练的影响；生成任务、`exp32_gen` 和 CrossMap 不在当前实现范围。

当前批准的主实验是 `openvla_oft_seen10_aligned_v2`：原始 OpenVLA-7B 初始化，冻结 vision 和 VL projector；其余适用模块保留官方 OFT 的 LoRA/全量训练方式及优化配置；使用 train-only Qnorm/clip 和不改变 pose 的颜色增强。有效 batch 128、19,500 updates，保存 4,000 / 8,000 / 12,000 / 16,000 / 19,500，主表使用 final。早期拟议的“projector 全量训练”“排除 lm_head 的七类 LoRA”“关闭 Qnorm/增强”均不是当前执行方案。

公平性分层如下。用户最终批准的 normalization、增强和优化差异必须随论文结果披露，不能称为与 UniLIP 完全同配方。

| 分类 | 对齐或控制内容 | 当前处理 |
| --- | --- | --- |
| A：严格对齐 | manifest、地图/split/ID、信息边界、外部单步 5D 和物理单位 | 固定发布数据；只用 FPV、radar、指令/地图名 |
| A：严格对齐 | state、定位 batch/update/曝光、checkpoint 选择、官方指标 | 无 state token；128 × 19,500；预定 final，best 仅看 validation |
| A：外部契约一致，内部差异披露 | train/test normalization、增强 | 外部统一；内部 Q99/target clip 与训练颜色增强是批准的差异，UniLIP 无这两项 |
| B：原生结构适配 | tokenizer、action 维度/目标函数、视觉 processor、LoRA/优化器 | 原生 OFT 连续 L1、5D×1 head、224、原生优化设置；不强制模拟 π0.5 的 32D |
| C：报告与控制 | 总量/可训练量、预训练来源、计算/显存、时间、推理成本 | 保存参数/来源审计；正式吞吐、延迟和成本待实际运行后记录 |

事实来源按证据层次区分：YAML 声明与 CLI → `runner.py` 解析/校验 → 构建后的 requires_grad / optimizer → checkpoint / scheduler /日志。局部 GPU 报告能证明实际模型参数与两步更新，不能代替尚不存在的 aligned 正式 checkpoint。UniLIP 数值沿用前轮本机审计；本次只复核 final `trainer_state.json` 的 19,500 / 19,550，不把文档整理称作重新完成全量审计。

## 2. 数据、输入与外部坐标协议

使用 [data.py](csgo_seen10/data.py) 的 `Seen10Dataset`、`normalize_pose`、`denormalize_pose`，并复用共享 evaluator 的 `protocol.py`。固定十地图及数据路径见运行说明。train / validation / test 分别为 50,000 / 5,000 / 20,000，每图为 5,000 / 500 / 2,000。前轮数据审计检查了 30 个 split JSON 的 ID 唯一性及跨 split 无交集；本次没有重新遍历计数。

发布 manifest SHA256：

```text
4debad27e0d481d31587a537325d6781247934551a7248885e686ed94046ba47
```

数据流为：原始 JSON `[x,y,z,pitch_rad,yaw_rad]` → 外部 benchmark normalized 5D → aligned 内部 Q99 target → 模型/L1 → 逆 Q99 → 标准预测 JSONL → 官方反归一化及 metric。

```text
b = [x/1024, y/1024, (z-zmin_map)/(zmax_map-zmin_map),
     pitch_rad/(2π), yaw_rad/(2π)]
```

Z 标定来自发布的逐地图 exact min/max，构建 benchmark 时在划分前按 approved corpus 冻结，三 split 共用；本项目不由 validation/test 重算。benchmark 的 quantile coverage bins 是选样机制，与本项目额外采用的 action Qnorm 无关。外部无 epsilon、无 clamp。

`Seen10Dataset` 可保留监督/展示 metadata，但 `collate_samples` 只将双图和指令作为条件；GT 独立作为 loss target。推理读取 `include_targets=False`。动作占位符与 GT 无关，不喂入历史坐标、未来动作、额外地图坐标或真实机器人状态。

统一 metric：XY 欧氏距离、Z 绝对误差、Pitch 绝对角误差、Yaw 模 360° 的最短距离。XY/Z 为 benchmark 坐标单位；角度为度。官方输出逐地图与 equal-map macro；完整测试各图等量，所以这些均值的 pooled = macro，不创建伪造的 pooled 字段。

UniLIP 历史 Z normalization 的分母为 `Δz+1e-6`。重评已有预测时，应保留其物理含义，转为本 evaluator 的 `b_z = unilip_b_z * (Δz+1e-6)/Δz`；其余维度不变，不 clamp。原 UniLIP 历史 evaluator 的角度处理也不能直接等同于共享指标。本项目没有实现独立的 UniLIP 导出工具，统一重评尚未完成。

## 3. Action 路径、normalization、state 与 loss

### 3.1 原版 OpenVLA 与 OFT 的区别

原版 OpenVLA 通过 `lm_head` 在词表尾部的离散 action bins 上解码。这里使用 OpenVLA-OFT 的并行动作路径：读取 action token 的隐藏状态，由独立 [L1RegressionActionHead](prismatic/models/action_heads.py) 输出连续值，训练主目标为 L1；不是继续使用离散 token logits 预测 CSGO pose。

原始 `checkpoints/openvla-7b` 不含预训练好的 OFT 7D 连续 head。官方 OFT 微调入口按任务维度构造 head；本项目同样新建 **5D×1** head，隐藏输入宽度 `5×4096=20480`，输出 `[B,1,5]`。没有保留 7D 再补零、独立 action expert、噪声/timestep 或 32D padding。机器人任务的默认维度不被 CSGO 改写。

原生双图 processor 将每图分别送入 DINOv2/SigLIP，组合输入为 `[B,12,224,224]`。模型保持原生 OFT 双向 action attention、SDPA、BF16 和 gradient checkpointing。连续 L1 不增加 CE、状态预测或额外监督。

### 3.2 train-only Q99 与预测反变换

[action_normalization.py](csgo_seen10/action_normalization.py) 的 `fit_seen_train_stats` 对**完整 50,000 条** train 外部目标拟合 NumPy linear Q01/Q99，不只取 sampler 实际保留的 49,920 条。五维 mask 均为 true；validation/test 不参与。legacy 的 mode 为 `none`，aligned 为 `bounds_q99`。

```text
a_target = clip(2*(b-q01)/(q99-q01+1e-8)-1, -1, 1)
loss = mean(abs(a_pred-a_target))            # 全部 5D，horizon 1
b_pred = 0.5*(a_pred+1)*(q99-q01+1e-8)+q01   # 不裁剪 a_pred 或 b_pred
```

全常数维按 OFT 将训练 target 设为零；非恒定但 q01=q99 时拒绝退化统计。逆变换保留上述 epsilon，不额外强制常数维。归一化和反变换以 FP32 执行；public `predict_normalized` 返回外部 benchmark normalized pose，不能将内部 `[-1,1]` 直接交给 evaluator。

未截断目标在浮点误差内 round-trip，尾部训练 target 的 clip 不可逆。Q99 是本实验选择的 LIBERO-OFT 机制，不代表所有官方 OFT 任务都使用 Q99。best-val 按**逆 Q99 后外部 5D 平均 L1**选择，不能用内部训练 L1 与 UniLIP 的 flow MSE 直接比较。

`action_normalization.json` 保存 schema、source split、样本数、manifest/有序 ID 哈希、q01/q99/min/max/mask 和 stats 哈希；训练启动另写 `normalization_audit.json`。已有局部审计：

| 维度 | x | y | z | pitch | yaw |
| --- | ---: | ---: | ---: | ---: | ---: |
| train 值低于 q01 的比例 | 0.00994 | 0.00960 | 0.00876 | 0.00226 | 0.01000 |
| train 值高于 q99 的比例 | 0.00988 | 0.00994 | 0.00044 | 0.00442 | 0.01000 |

统计哈希为 `3b8f9932e821c2c0ea22ca23b6a379e2564b1613d12c23c4c61294f7142b086e`；有序 train ID 哈希为 `099f3b2f3e91c118e7ee1ddba73396ee5fe9d2704446212df73f535e882a9d66`。这些是已读诊断报告中的统计，未使用测试集拟合。

### 3.3 State 确实省略

`use_proprio=false` 时不构造 proprio projector，也不拼接 state token；传给原生路径的 proprio 为 `None`，不是全零向量。局部真实模型 hook 验证拼接前后视觉 token 数均为 512，隐藏宽度仍为 4096。它减少的是本来可选的 token 序列项，不是缩小 LLM hidden dimension。

## 4. FPV / radar 增强与标签一致性

[augmentations.py](csgo_seen10/augmentations.py) 的 `augment_image` 在 RGB 域执行如下操作，再进入原生 224 resize/rescale/normalize。相同增强图同时供两个视觉编码器使用。

| 操作 | FPV 训练 | radar 训练 | validation / test |
| --- | --- | --- | --- |
| Brightness | `[0,1]` RGB 加性偏移 ±0.2 | 同幅度，独立随机流 | 关闭 |
| Contrast | 因子 [0.8,1.2]，围绕通道均值 | 同幅度，独立随机流 | 关闭 |
| Saturation | HSV 饱和度因子 [0.8,1.2] | 同幅度，独立随机流 | 关闭 |
| Hue | 色相周期偏移 ±0.05 | 同幅度，独立随机流 | 关闭 |
| 随机 crop / flip / rotation | 关闭 | 关闭 | 关闭 |
| Erasing / CoarseDropout / GridDropout | 关闭 | 关闭 | 关闭 |
| 确定性原生 processor | 开启 | 开启 | 开启 |

固定顺序 brightness → contrast → saturation → hue，不修改 pose。随机流由 `(seed, epoch, sample_id, view)` 确定，不依赖 worker/rank；sampler 传递 epoch 使 persistent worker 使用正确增强。RGB 像素合法范围 clamp 与 action target clip 是两件事。

这套 NumPy/PIL 实现保留官方 OFT 的颜色操作种类和幅度，不宣称与其 TensorFlow 管线逐位等价。FPV crop 未必改变外参标签，但会改变有效视野/内参；当前不引入这项变化。颜色增强可能改变 radar 的视觉辨识难度，属于需披露的训练策略差异；推理无 TTA。

## 5. 模块映射、LoRA 与实际优化参数

实际计数来自成功局部 GPU 运行的 [parameter_audit.json](outputs/csgo_aligned_validation/20260924T103648Z/parameter_audit.json)，包含逐 tensor 的名字、shape、requires_grad 和 optimizer 成员。

| 功能角色 / 本项目模块 | aligned 训练方式 | 总参数 | 声明可训练参数 | LR |
| --- | --- | ---: | ---: | ---: |
| Vision：DINOv2 + SigLIP `vision_backbone` | 全冻结，无 LoRA；train 时保持 eval | 730,911,680 | 0 | — |
| VL connector：`projector` | 全冻结，无 LoRA | 71,385,600 | 0 | — |
| LLM：`language_model`，含 lm_head | base 冻结，官方 all-linear 语义 LoRA | 6,820,050,944 | 81,111,040 | 5e-4 |
| 连续 action head：`L1RegressionActionHead` | 新建 5D×1，全量训练 | 117,538,821 | 117,538,821 | 5e-4 |
| 独立 action expert / action connector / timestep / proprio | 不适用或不构造 | — | — | — |
| 合计 | 声明可训练占 **2.5666%** | **7,739,887,045** | **198,649,861** | 单组 |

VL projector 是三层 MLP `2176→8704→4096→4096`，不是 attention 模块，也不是 action head。当前已经按最终要求冻结；不再执行曾拟议的“由 LoRA 改全量训练”。

LoRA 固定 `r=32, alpha=16, dropout=0, bias=none, init=gaussian`。`_aligned_linear_targets` 先按安装的 PEFT 0.11.1 解析整个模型的 `all-linear`，再排除 vision/projector 路径。LLM 内覆盖 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`，且该包装层级下还包含 `lm_head`，因此与仅七类投影、排除 lm_head 的方案不同。

其中 lm_head adapter 的 1,157,120 个参数虽然 requires_grad=true 且进入 optimizer，但连续 L1 不依赖 logits，其梯度为 None。预期在 L1 反向路径上的可训练参数为 **197,492,741**；不能将声明数量等同于实际每步更新量。optimizer 有 466 个 tensor 成员，局部两步后只有 464 个 state 条目，与两项未使用的 lm_head adapter tensor 对应。

官方配方来源为 [vla-scripts/finetune.py](vla-scripts/finetune.py)：

- 单组 AdamW，LR `5e-4`、betas `(0.9,0.999)`、eps `1e-8`、weight_decay `0.01`；后面三项是官方 AdamW 调用的实际默认值，在 aligned 中显式记录。
- `lr_warmup_steps=0`；MultiStepLR milestone `100000`、gamma `0.1`，19,500 步内 LR 始终为 `5e-4`。legacy milestone 为 13,000，二者不能混写。
- 不使用 UniLIP 的多组 LR、weight_decay=0 或 cosine-min schedule，不进行事后 LR sweep。
- 原生 BF16/SDPA 路径，PEFT 可将 adapter 保持 FP32；不能笼统声称所有训练参数同一 dtype。

初始化只用原始 OpenVLA-7B。三个 shard 哈希固定在 aligned YAML，并与 [原始资产核验](outputs/csgo_seen10_validation/weight_integrity.json) 一致；该记录保存了 2026-09-17 官方仓库 `main` 文件的大小/LFS SHA256，而非一个不可变 HF commit。aligned 启动实际重算 shard 哈希，足以识别所用权重；冻结 projector 从原始 base 恢复。

## 6. 预算、checkpoint、恢复与推理约束

`GlobalUpdateSampler` 按 seed+epoch 全局 shuffle 50,000 个 train ID，丢尾 80 后再分给各 rank，不补重复样本，丢弃项不做前后向。每 epoch `49,920/128=390` updates，50 epochs 对应 **19,500 updates / 2,496,000 定位曝光**。改变 GPU 数量不能改变此预算；不承诺与 UniLIP 拥有相同逐 update 样本顺序。

默认 `batch_size=1, grad_accumulation_steps=128`；启动验证 `world_size×batch×accumulation=128`。scheduler、日志、验证和 checkpoint 均以 optimizer update 计数，日志主 loss 平均整个累计窗口。

| 事项 | aligned 约定 |
| --- | --- |
| seed | 手动 CLI 显式 42；兼容默认仍为 0 |
| 保存 / 完整 validation | 4,000 / 8,000 / 12,000 / 16,000 / 19,500；final 即使不整除 4,000 也强制执行 |
| 目录 | `checkpoints/step_00004000` 等，保留五个实际目录 |
| 链接 | checkpoint 根和运行根的 `best` / `late` 均链接到对应 step，不复制权重，无 `last` |
| best | 完整 5,000 条 validation 的外部 normalized L1 最小值；严格小于才更新 |
| 主表 | final / 完成后的 late，step 19,500；best-val 只能附加报告 |
| 保存内容 | PEFT、head、processor、normalization、model recipe、optimizer/scheduler、各 rank RNG、采样位置、provenance |
| 恢复 | 同 recipe/base/stats/数据/seed，且 world size、microbatch、accumulation 不变；拒绝 legacy warm-start |
| 正式推理 | 单进程、batch 1、相同 seed；一次连续前向，无采样迭代、TTA 或输出 clip |
| 推理续写 | 固定 checkpoint/provenance 后，`--resume` 仅补缺失 ID；完整测试需 20,000 唯一 ID |

单个 checkpoint 直接写入 step 目录，不宣称整个目录事务式原子保存。异常中断后需确认保存组件完整，再用恢复入口校验。CPU 测试验证了 sampler 恢复后缀和 best/late 规则；完整真实 aligned 训练中断恢复轨迹尚未实测。

## 7. 配置覆盖与文件级实现边界

### 7.1 配置如何成为运行值

`load_config` 读取 YAML；通用 `_config_value` 顺序为顶层字段 → `train.<key>` → `model.<key>` → 代码默认。地图、路径等字段有各自解析函数；数据路径另受 `CSGO_DATA_ROOT > DATA_ROOT > YAML` 控制。共享 evaluator 和 Python 可由 `SHARED_EVAL_DIR`、`UNILIP_PYTHON` 覆盖。CLI 提供 config、seed、smoke、resume/checkpoint，以及 data-root、eval-root、unilip-python、model-path 路径覆盖和只读 print-paths，不是任意 YAML key 覆盖器。路径 CLI 优先于环境变量，配置与相对路径统一以项目根目录解析；evaluator Python 默认使用项目 .venv，保留旧绝对路径的兼容回退，详见运行说明第 3 节。

aligned 由 `recipe_id` 分派，`_validate_aligned_config` 对批准的固定语义作校验。`_aligned_recipe` / `resolved_recipe.json` 记录配方；`training_config.json` 与 `run_provenance.json` 记录解析结果和实际执行信息；`parameter_audit.json` 记录构建后的 optimizer，而 checkpoint 的 `training_state.json`、`optimizer.pt`、`scheduler.pt` 记录保存时状态。aligned 正式产物当前尚不存在，不以 YAML 代替运行完成的证据。

未指定 aligned profile 的旧配置继续走 legacy：无新增 Qnorm/增强，原 all-linear 范围、batch 32、13,000 衰减及 3,900 保存间隔。既有 CLI 保留；新运行目录独立，旧结果不迁移、不覆盖。

### 7.2 关键配置与实现位置

| 文件 / 对象 | 负责的行为与字段 |
| --- | --- |
| [configs/csgo_seen10.yaml](configs/csgo_seen10.yaml) | legacy 可复现配置，保持原行为 |
| [configs/csgo_seen10_aligned_v2.yaml](configs/csgo_seen10_aligned_v2.yaml) | recipe、base 哈希、冻结、LoRA、Qnorm/clip、逐视图增强、128 batch、19,500 updates、4,000 events |
| [data.py](csgo_seen10/data.py)：Seen10Dataset / normalize_pose | 发布 split/ID、双图、外部 pose、GT 条件边界 |
| [model.py](csgo_seen10/model.py)：create_model / _aligned_linear_targets | 原始 base、模块冻结、LoRA target、原生 5D head、兼容 checkpoint |
| model.py：collate_samples / forward_action / predict_normalized | GT-independent 占位符、无 state、连续 L1、逆 Q99 输出 |
| [action_normalization.py](csgo_seen10/action_normalization.py)：ActionNormalization | none / bounds_q99、完整 train 拟合、统计身份、target clip / inverse |
| [augmentations.py](csgo_seen10/augmentations.py)：augment_image | none / oft_photometric_only，FPV/radar 独立可重复随机流 |
| [sampling.py](csgo_seen10/sampling.py)：GlobalUpdateSampler / event_steps | 全局完整更新批次、rank 划分、epoch/恢复、强制 final |
| [runner.py](csgo_seen10/runner.py)：train / save_checkpoint / inference / eval_command | 配置、实际参数组、预算、验证、保存恢复、推理 provenance、原 evaluator |
| [scripts/check_csgo_aligned.py](scripts/check_csgo_aligned.py)、[tests](tests) | CPU 契约/采样/变换检查，独立有限 GPU 更新与组件重载 |
| [train_seen10.py](train_seen10.py)、[infer_seen10.py](infer_seen10.py)、[eval_seen10.py](eval_seen10.py) | 兼容命令入口；参数和路径用法见运行说明 |
| [scripts/run_csgo_seen10.sh](scripts/run_csgo_seen10.sh) | train/infer 的 torchrun 包装；smoke 会实际串联三个阶段 |

训练/推理/评测支持 `--print-paths`，此分支不导入模型、不执行阶段任务；它不是完整配置或权重验收。入口仍没有 `--dry-run`、`--output-root` 或 `RUN_FULL` 防执行开关。环境与下载脚本的 `--dry-run` 是各自独立功能，不能照搬到训练命令。

### 7.3 跨服务器路径适配（2026-09-24）

目标布局为 `/home/user/yc57963/task/openvla-oft`、同级 `UniLIP/data/csgo_benchmark_v2` 和 `csgo_benchmark_v2_eval_general`；原服务器同级布局继续可用。两份 YAML 仅修改数据/evaluator/Python 路径，不改变冻结、LoRA、Q99、增强、batch、预算、保存步骤或 seed 规则。

- [paths.py](csgo_seen10/paths.py) 统一 CLI→环境变量→YAML→默认路径，文件相对 checkout 根解析，Python symlink 保留 venv 身份；仅旧服务器内置默认在缺失时允许回退，错误自定义路径不隐式替换。
- [cli.py](csgo_seen10/cli.py) 提供轻量配置加载和 `--print-paths`，三个入口在路径检查后才导入训练运行时；wrapper 的路径模式绕开 torchrun。
- 数据加载、原始模型、输出/checkpoint、恢复和 evaluator 使用同一项目根定位规则。旧 checkpoint 的路径/身份检查保留，未实现跨路径恢复转换。
- [setup_csgo_seen10.sh](scripts/setup_csgo_seen10.sh) 去除固定 UniLIP Python 回退，支持指定 Python3.11、只读 dry-run/check 和迁移后环境检查；保持原 CUDA 12.8 依赖。
- [download_csgo_model.py](scripts/download_csgo_model.py) 独立下载/续传官方 base，使用项目目录或 `OPENVLA_MODEL_PATH`；setup 默认仍下载，`--skip-model` 保持兼容。
- evaluator 默认本地 `.venv/bin/python`；定位依赖 PyTorch/PyYAML 已由项目提供，不需引入完整生成评测依赖。原服务器历史指标不因默认解释器改变而重写；新服务器记录自己的依赖与运行 provenance。

验证采用 [test_csgo_paths.py](tests/test_csgo_paths.py)：模拟新服务器根路径、CLI/env优先级、旧默认回退与错误自定义路径、Python symlink，以及外部 cwd 下三个入口不导入模型/不写实验产物。脚本语法、环境/下载 dry-run 与路径打印检查已通过。未在目标服务器实际安装或下载；旧主机上的这些检查不能证明新服务器驱动、网络及资产已经就绪。

## 8. 与原生 OFT、legacy 和 UniLIP 的比较边界

| 对齐项 | 官方 OFT L1 配方 | OpenVLA legacy | OpenVLA aligned v2 | UniLIP exp32_loc | UniLIP exp32 |
| --- | --- | --- | --- | --- | --- |
| 数据/任务 | 机器人任务，按任务配置 | Seen-10 loc | 同一 Seen-10 loc | Seen-10 loc | 同源 balanced loc + gen |
| 外部目标 / 内部 loss | 任务相关维度，连续 L1 | 5D×1 L1 | 5D×1，Q99 空间 L1 | 5D→32D 补零，完整 32D flow MSE | 相同定位路径，另含生成目标 |
| 内部 Qnorm / clip | 任务相关，LIBERO Q99 | 无 | train-only Q99 / target clip | 无 action Qnorm | 无 action Qnorm |
| state | 可选 proprio | 无 token | 无 token | 无 token | 定位无 token |
| 图像 / 增强 | 原生 224；可启用 crop + color | 224，无随机增强 | 224；双图独立 color，无几何增强 | 224，无随机增强 | 定位输入同 loc |
| Vision / VL connector | all-linear LoRA 可覆盖 | all-linear LoRA | 整模块冻结 | 冻结 | 定位所用模块冻结 |
| LoRA | all-linear，r32 时 alpha16/dropout0 | 同左 | 排除两个冻结模块，保留官方解析语义 | LLM/expert 七类，r32/alpha64/dropout.05 | 同 loc |
| 可训练量 | 任务/维度相关 | 228,367,109 声明口径 | 198,649,861 声明；197,492,741 预期 L1 路径 | 55,688,992 | 77,002,784，汇总分区 |
| 总参数口径 | 任务相关 | 7,769,604,293 | 7,739,887,045 | 保存完整模型 2,269,423,363 | 保存完整模型 2,288,441,091 |
| 优化 | AdamW，默认 wd .01 | 单组 5e-4 | 单组 5e-4，wd .01 | 多组 1e-4 / 5e-4，wd0 | 多组 5e-5 / 1e-4 / 5e-4，wd0 |
| LR schedule | MultiStep，默认 milestone100k | step13k 降 10 倍 | milestone100k，预算内不降 | cosine-min，warmup .003 | 同类 cosine-min |
| batch / updates | 任务相关 | 32 / 19,500 | 128 / 19,500 | 128 / 19,500 | nominal128 源样本 / 19,550 |
| 定位曝光 | 任务相关 | 624,000 | 2,496,000 | 2,496,000 | 2,500,000，另有等量 gen |
| checkpoint | 按任务脚本 | best，实际同 final19500 | final19500 主结果；五个预定节点 | final19500；历史每2000保存、保留4 | final19550；历史每2000保存、保留2 |
| 推理 | 原生连续回归一次前向 | 一次前向 | 一次前向，seed42/batch1/单进程 | flow 10步；历史进程数未确认 | flow 10步；历史进程数未确认 |

UniLIP 的参考来源为 [exp32_loc final](../UniLIP/outputs/csgo_1b/exp32_loc/checkpoint-19500)、[exp32 final](../UniLIP/outputs/csgo_1b/exp32/checkpoint-19550) 中的 config、training_args、trainer_state、scheduler 和分区 optimizer，结合 [unified_task_dataset.py](../UniLIP/csgo_datasets/unified_task_dataset.py) 与 [nonmix_trainer.py](../UniLIP/unilip/train/nonmix_trainer.py) 的前轮审计。表中完整保存模型总量不等于单次定位前向激活参数量。

LR 按功能角色细分：exp32_loc 的 LLM/expert LoRA 和 action input/output/timestep 为 1e-4，localization connector/norm/projector 为 5e-4；exp32 的 expert 为 5e-5、connector 为 1e-4，其余同上。运行时 scheduler 对各参数组采用同一最低比例 0.1，因此不能把全局 `min_lr=1e-5` 当成所有参数组的绝对终点；例如 exp32 expert 终点为 5e-6。这些多组配置没有移植给 OFT。

exp32 每 epoch 390 个完整的 128 源样本 update 加 80 条尾批，391×50=19,550；实际 loc/gen 各 2,500,000，不能直接用 19,550×128 计算曝光。exp32_loc 与 aligned 丢尾后均为 390×50 updates。

主论文应描述为“数据、信息边界、定位预算和评测协议对齐，保留批准的模型原生训练差异”。Q99 改变维度权重并截断训练尾部；颜色增强可能改变泛化；OFT 的连续 head、loss、优化和预训练语料不同。这些差异不能由单张结果表归因成单纯架构优劣。当前不新增未经批准的 normalization/state/维度消融；如增加须预先固定独立配置、预算与 validation 选择规则。

## 9. 验收要求与已完成记录

### 9.1 必须维持的验收项

1. 三 split 的地图/ID/数量、重复和交集；完整测试 coverage；无额外 GT/state/history 条件。
2. 外部单位与 round-trip；Q99 只由完整 train 统计；区分 target clip 与预测不裁剪。
3. FPV/radar 增强种类、独立随机流、pose 不变；validation/test 无随机增强。
4. 无 proprio 模块/token、5D×1 target/head/L1；不引入 7D/32D padding 或额外 loss。
5. 实际 requires_grad、LoRA targets、冻结模块、optimizer 成员/LR 与梯度；单独报告 inactive lm_head。
6. 全局128/update、19,500 updates、2,496,000曝光、五个保存节点及 final/best/late；恢复状态一致性。
7. 真实单 batch 前后向、有限局部更新、组件保存重载；与正式 train/infer/eval 入口区分。
8. 标准预测到原 evaluator 的闭环；历史 legacy smoke 不能代替 aligned 正式覆盖率或指标。

### 9.2 2026-09-24 aligned 局部验收

以下是前轮完成的记录，本次只读报告、整理文档，未重跑。实现检查入口 [scripts/check_csgo_aligned.py](scripts/check_csgo_aligned.py) 默认 CPU 检查；显式 `--gpu-smoke` 每次最多两个单样本 optimizer updates，不调用正式训练/批量推理/evaluator 入口。它会加载真实模型并写独立诊断报告，不是 dry-run。

| 检查 | 实际结果与范围 |
| --- | --- |
| CPU 单测 | 7 项通过：模型 3、变换 4；[test_aligned_model.py](tests/test_aligned_model.py)、[test_aligned_transforms.py](tests/test_aligned_transforms.py) |
| CPU 配方/统计 | [20260924T103311Z/report.json](outputs/csgo_aligned_validation/20260924T103311Z/report.json)，完整 train 50k Q99 与哈希 |
| 采样 / 节点 | 1/2/4 world-size 的 CPU 全局顺序哈希一致；390 updates/epoch、丢尾80；恢复后缀与 best/late 通过；不是实际多卡前后向 |
| 真实7B参数 | 总量7,739,887,045，声明可训练198,649,861；optimizer 单组 LR5e-4；见完整参数清单 |
| 冻结 / 梯度 | vision/projector 全部 grad=None；冻结探针不变；LoRA 与 head 有有限梯度和实际权重变化 |
| 无 state | hook 检查 proprio=None；视觉 token 512→512 |
| 有限两步更新 | 成功运行内部 L1 0.57385075→5.31071997；外部 L1 0.18285809→2.03146410；有限但未显示收敛，未改已批准超参数 |
| 保存与实际重载 | 释放旧模型后用 production create_model 重载；同 batch 输出最大差0；stats哈希一致；optimizer464个state，scheduler last_epoch=2、LR5e-4 |
| 显存 | 成功局部 batch peak allocated 15.95 GiB；不能外推正式训练峰值、吞吐或独占硬件成本 |
| 产物清理 | 临时大型 checkpoint 已删除，报告保留；未创建 aligned 正式运行目录 |

成功证据：[20260924T103648Z/report.json](outputs/csgo_aligned_validation/20260924T103648Z/report.json)、[parameter_audit.json](outputs/csgo_aligned_validation/20260924T103648Z/parameter_audit.json)。首次 GPU 检查 [20260924T103341Z/report.json](outputs/csgo_aligned_validation/20260924T103341Z/report.json) 因 BF16 LayerNorm 单参数探针未变化而失败；随后换用具有可辨识更新的 head 输出 bias，并同时检查梯度与模块冻结。两次 GPU 调用各做两步，合计四次诊断更新，不能将成功报告的两步误写成全部历史开销。

### 9.3 首次接入与 legacy 正式运行

2026-09-17 真实原始 OpenVLA-7B smoke：5 updates、5次各100条 validation、5 checkpoints；best=3、late=5；100条预测（每图10）、50张 validation 图、10张 inference 图及 loss 曲线。共享 evaluator smoke 标记 `formal=false`，不是正式结果。

| 历史证据 | 内容 |
| --- | --- |
| [contract_acceptance.json](outputs/csgo_seen10_validation/contract_acceptance.json) | GT不改变输入，target `[2,1,5]`、pixels `[2,12,224,224]`、OFT attention检查 |
| [environment_acceptance.json](outputs/csgo_seen10_validation/environment_acceptance.json)、[weight_integrity.json](outputs/csgo_seen10_validation/weight_integrity.json) | 原始环境和官方资产身份 |
| [smoke_acceptance.json](outputs/csgo_seen10_validation/smoke_acceptance.json)、[shared_evaluator_smoke.json](outputs/csgo_seen10_validation/shared_evaluator_smoke.json) | 真实模型5-step与原evaluator闭环 |
| [training_resume_acceptance.json](outputs/csgo_seen10_validation/training_resume_acceptance.json)、[sdpa_repeatability.json](outputs/csgo_seen10_validation/sdpa_repeatability.json) | step4→5恢复，数据位置/scheduler/RNG一致；head逐位一致；BF16 SDPA导致LoRA最大差4.88e-4，不保证逐位训练复现 |
| [inference_resume_acceptance.json](outputs/csgo_seen10_validation/inference_resume_acceptance.json) | 仅补5条缺失预测，最终100条与原始输出逐条一致 |
| [legacy final training_state.json](outputs/csgo_benchmark_v2_seen10/OpenVLA-OFT/seed_0/checkpoints/step_00019500/training_state.json) | 正式19,500、batch1×acc32、seed0、best L1 .0240991；best/late同final |
| [legacy summary_equal_map.json](outputs/csgo_benchmark_v2_seen10/OpenVLA-OFT/seed_0/evaluation/localization/summary_equal_map.json) | 20,000完整coverage与正式XY/Z/Pitch/Yaw；数值见运行说明 |

legacy 更新曝光为624,000；旧循环完整 epoch 尾部有16个 microbatch 做了反向但不参与更新，12个完整 epoch 共192次无效曝光，总前向曝光624,192。不能用该历史预算宣称已对齐128有效batch。

### 9.4 验收边界与仍未确认项

- aligned 正式训练、完整5k validation、20k推理/evaluator均未启动，尚无五个正式 checkpoint 或正式结果。
- 实际128个microbatch累计的完整训练循环、真实多GPU前后向、完整aligned中断恢复轨迹尚未执行；CPU sampler与组件重载不等于这些检查已通过。
- 两步loss上升只用于功能验收，不能断言可收敛或需要改LR；正式成本、延迟、FLOPs和收敛曲线仍待记录。
- `normalization_audit.json` 启动输出逻辑在GPU检查后补充，仅完成静态检查；不能说成功GPU报告覆盖了这项后续完整入口行为。
- legacy训练 provenance 记录旧commit `e4287e9`，未保存可完全还原的当时dirty源码；已有输出可以核验，但不承诺当前重跑逐位等同历史运行。
- UniLIP保存总量与定位激活量、原始权重完整哈希、部分历史推理拓扑/人工checkpoint决策未全部确认；本项目也未完成其统一evaluator重评。
- Q99/clip、颜色增强、预训练语料、连续L1与flow MSE、内部维度和推理步数的差异已明确保留；结果解释必须受这些边界约束。
- 文档整理后的路径适配仅修改本项目入口、路径配置、环境/下载脚本及显式依赖，不修改 UniLIP、数据或共享 evaluator。未安装/下载或启动训练、推理、评测；后续正式运行仍由用户手动执行。
