# OpenVLA-OFT · CSGO Benchmark v2 Seen-10

本文是 Seen-10 定位任务的运行说明，记录环境、命令、输出和当前结果。实验设计、实现边界、横向比较与验收证据统一维护在 [CSGO_SEEN10_PLAN.md](CSGO_SEEN10_PLAN.md)。两份文档同时覆盖首次接入的 legacy 和当前已实现的 aligned v2；正式训练、全量推理及评测由使用者手动启动。

记录更新：2026-09-24。本次仅整理文档，未运行训练、推理、评测或模型测试。

## 1. 实验范围与数据

只接入 localization，不含生成任务或 CrossMap。输入为当前 FPV RGB、对应地图 radar 和包含地图名的定位指令；不输入 GT 坐标、历史位姿或 robot state。输出为单步 5D pose `[x,y,z,pitch,yaw]`。

- 数据：`/home/jiahao/task/UniLIP/data/csgo_benchmark_v2`，直接读取发布 manifest、split 和 calibration，不扫描图片重新划分。
- 共享评测器：`/home/jiahao/task/csgo_benchmark_v2_eval_general`；数据适配也依赖其纯 Python `protocol.py`。
- 固定地图顺序：`cs_agency, cs_italy, de_ancient, de_anubis, de_dust2, de_inferno, de_mirage, de_nuke, de_overpass, de_train`。
- `seen_train` 50,000 条、`seen_validation` 5,000 条、`seen_discrete_test` 20,000 条；每图分别 5,000 / 500 / 2,000 条。
- 外部标准预测使用 `[x/1024, y/1024, (z-zmin)/(zmax-zmin), pitch_rad/(2π), yaw_rad/(2π)]`。Z 使用发布的固定逐地图范围，分母无 epsilon，不裁剪预测。

## 2. 配置与实验区别

| 项目 | 首次接入 legacy | 当前主实验 aligned v2 |
| --- | --- | --- |
| 配置 | [configs/csgo_seen10.yaml](configs/csgo_seen10.yaml) | [configs/csgo_seen10_aligned_v2.yaml](configs/csgo_seen10_aligned_v2.yaml) |
| 记录 seed | 0 | 42，命令必须显式传入 |
| 正式输出根目录 | `outputs/csgo_benchmark_v2_seen10` | `outputs/csgo_benchmark_v2_seen10_aligned_v2` |
| Vision / VL projector | base 冻结，all-linear LoRA 可训练 | 整个模块冻结，不注入 LoRA |
| LLM / 连续 action head | all-linear LoRA / 新建 5D×1 L1 head 全量训练 | 保留相同类型的适配与 head |
| 内部 target | 外部 normalized 5D | seen_train Q01/Q99 normalization + target clip |
| 图像增强 | 无随机增强 | FPV、radar 各自独立的颜色增强，无几何增强 |
| 有效 batch / updates | 32 / 19,500 | 128 / 19,500 |
| 参与 optimizer 更新的定位曝光 | 624,000 | 2,496,000 |
| LR / scheduler | 5e-4，step 13,000 后为 5e-5 | 5e-4；原生 milestone 100,000，在本预算内不衰减 |
| 验证和保存步骤 | 3,900 / 7,800 / 11,700 / 15,600 / 19,500 | **4,000 / 8,000 / 12,000 / 16,000 / 19,500** |
| 默认推理 checkpoint | validation `best` | `late`；主表使用训练完成后的 final |
| 当前状态 | seed 0 已完成正式训练和评测 | 实现及局部验收完成，正式实验未启动 |

两套运行目录都在各自根目录下追加 `OpenVLA-OFT/seed_<seed>/`，checkpoint 与结果同属该目录。两套实验不能跨 recipe 恢复，legacy 结果不能作为 aligned 主结果。

aligned 与 UniLIP `exp32_loc` 对齐输入信息、split、外部 5D、有效 batch、更新数和定位曝光；保留用户批准的 Qnorm/clip、颜色增强和 OFT 原生优化配置，因此并非逐项相同的训练配方。比较边界见 PLAN。`exp32` 的定位结果仅作联合训练的次要比较对象。

## 3. 环境与原始权重

以下命令均从 `/home/jiahao/task/openvla-oft` 执行。当前本机环境与权重已经准备完成；仅新环境需要执行安装脚本：

```bash
cd /home/jiahao/task/openvla-oft
bash scripts/setup_csgo_seen10.sh
```

脚本默认安装依赖并准备模型；`--skip-model` 只跳过模型下载，仍会安装依赖。项目使用独立 `.venv`：Python 3.11、PyTorch 2.7.1+cu128、Torchvision 0.22.1、PEFT 0.11.1，以及 [requirements-csgo-seen10.txt](requirements-csgo-seen10.txt) 固定的 OFT Transformers fork，不向 UniLIP 环境安装项目依赖。

初始化使用原始 `openvla/openvla-7b`，本地目录 `checkpoints/openvla-7b/`。aligned 配置固定三个权重 shard 的 SHA256，启动时重新校验，不从旧 CSGO adapter 热启动。原始资产核验见 [weight_integrity.json](outputs/csgo_seen10_validation/weight_integrity.json)。该 base 不含一个可直接复用的 OFT 7D 连续 head；本项目新建 5D×1 head。

默认路径已写入 YAML。需要迁移数据或 evaluator 时可设置：

```bash
export CSGO_DATA_ROOT=/home/jiahao/task/UniLIP/data/csgo_benchmark_v2
export SHARED_EVAL_DIR=/home/jiahao/task/csgo_benchmark_v2_eval_general
export UNILIP_PYTHON=/home/jiahao/miniconda3/envs/UniLIP/bin/python
```

数据路径优先级为 `CSGO_DATA_ROOT > DATA_ROOT > YAML`。注意清理不再使用的环境变量，避免覆盖配置中的路径。

## 4. 手动执行与恢复

### 4.1 aligned v2：当前主实验

单 GPU microbatch 1 × 梯度累计 128 = 有效 batch 128。每 epoch shuffle 全部 50,000 条，前向前丢尾 80 条，完成 390 个完整 update；50 epochs 共 19,500 updates。每个保存节点完整验证 5,000 条 validation，以外部 normalized 5D 平均 L1 选择 `best`，平局保留更早步骤；`late` 指向最新保存点，完成后为 `step_00019500`。没有 `last` 别名。

依次手动执行训练、推理、评测；下列命令并未在本次文档整理中执行：

```bash
./.venv/bin/python train_seen10.py --config configs/csgo_seen10_aligned_v2.yaml --seed 42
./.venv/bin/python infer_seen10.py --config configs/csgo_seen10_aligned_v2.yaml --seed 42
./.venv/bin/python eval_seen10.py --config configs/csgo_seen10_aligned_v2.yaml --seed 42
```

CLI 的兼容默认 seed 仍是 0，不能省略 aligned 命令中的 `--seed 42`。训练断点恢复：

```bash
./.venv/bin/python train_seen10.py --config configs/csgo_seen10_aligned_v2.yaml --seed 42 \
  --resume-checkpoint outputs/csgo_benchmark_v2_seen10_aligned_v2/OpenVLA-OFT/seed_42/checkpoints/late
```

恢复要求相同的 recipe、数据身份、base/stats、world size、microbatch 和累计步数。新运行若使用多 GPU，须在单独配置中调整累计步数，使 `GPU 数 × 每卡 batch × accumulation = 128`；不支持断点恢复时切换拓扑。wrapper `scripts/run_csgo_seen10.sh train` 使用 `torchrun`，进程数来自 `NPROC_PER_NODE`，默认 1。

aligned 测试推理固定单进程、batch 1、相同 seed 42、一次连续回归前向，无随机增强或额外预测 clip。默认取 `late`。中断后只补缺失 ID：

```bash
./.venv/bin/python infer_seen10.py --config configs/csgo_seen10_aligned_v2.yaml --seed 42 --resume \
  --checkpoint outputs/csgo_benchmark_v2_seen10_aligned_v2/OpenVLA-OFT/seed_42/checkpoints/late
```

推理会核对 checkpoint 和 provenance，不允许在同一预测目录混用不同模型。附加 best-val 结果可显式指定 `--checkpoint .../checkpoints/best`，但必须使用新的结果目录并明确标注。当前 CLI 没有 `--output-root`：复制 YAML 并修改 `output_root`，推理显式指定原 checkpoint，评测使用同一份新 YAML；不要覆盖 final 的预测。

### 4.2 legacy：复现已有接入

seed 0 已有正式产物。以下使用新 seed 1 保持输出独立；若要相同 seed 重跑，应复制 YAML 并更换 `output_root`。

```bash
bash scripts/run_csgo_seen10.sh train --config configs/csgo_seen10.yaml --seed 1
bash scripts/run_csgo_seen10.sh infer --config configs/csgo_seen10.yaml --seed 1
bash scripts/run_csgo_seen10.sh eval --config configs/csgo_seen10.yaml --seed 1
```

legacy 同样支持训练的 `--resume-checkpoint`，以及推理的 `--checkpoint` / `--resume`；默认推理选择 validation `best`。本项目没有 `RUN_FULL` 执行保护开关，运行 `train` 就会开始训练。

`bash scripts/run_csgo_seen10.sh smoke --config configs/csgo_seen10.yaml --seed 0` 是历史完整 smoke 命令，**会实际串联小规模训练、推理和 evaluator**，不是只读检查。本机 seed 0 smoke 已存在；如需再次运行，先使用不同 `smoke_output_root`，不要覆盖原验收证据。本轮不执行该命令。

### 4.3 官方统一评测

正式 evaluator 要求完整 20,000 条 coverage。仅重评已有 legacy 预测时，可直接调用共享入口，写入未使用的目录：

```bash
/home/jiahao/miniconda3/envs/UniLIP/bin/python \
  /home/jiahao/task/csgo_benchmark_v2_eval_general/run_eval.py localization \
  --pred-root outputs/csgo_benchmark_v2_seen10/OpenVLA-OFT/seed_0/localization \
  --data-root /home/jiahao/task/UniLIP/data/csgo_benchmark_v2 \
  --pose-space normalized \
  --output outputs/csgo_benchmark_v2_seen10/OpenVLA-OFT/seed_0/evaluation/localization_recheck
```

重复评测时更换 `--output`；共享 evaluator 拒绝覆盖已有非空结果目录。UniLIP 历史预测的 Z 分母含 `1e-6`，统一重评前需要显式转换以保持物理 Z；本项目未提供单独的 UniLIP 导出命令，也未完成这项重评，转换口径见 PLAN。

## 5. 输出、可视化与指标

aligned 正式目录为 `outputs/csgo_benchmark_v2_seen10_aligned_v2/OpenVLA-OFT/seed_42/`，主要产物如下；这些是运行后的约定路径，当前尚未生成正式目录。

```text
training_config.json、run_provenance.json    解析配置、数据/源码/原始权重身份
resolved_recipe.json、parameter_audit.json   aligned 配方与实际参数/optimizer 清单
action_normalization.json                   仅 seen_train 拟合的 Q99 统计及哈希
normalization_audit.json                    normalization 与训练 target 截断审计
logs/main_loss.jsonl、main_loss.png          逐 optimizer update 的主 loss
logs/validation.jsonl                       完整 validation loss
logs/validation_predictions.jsonl           validation 预测记录
logs/sampler_epochs.jsonl                   aligned 采样、丢尾与曝光记录
checkpoints/step_00004000/                  adapter、head、processor、stats、recipe、
                                           optimizer/scheduler、RNG、training_state
checkpoints/step_00008000/                  其余保存节点同上
checkpoints/step_00012000/
checkpoints/step_00016000/
checkpoints/step_00019500/
checkpoints/best、checkpoints/late           对应步骤的符号链接，根目录也有同名链接
validation_visualizations/step_00004000/    每节点、每地图的 validation 图
localization/predictions.jsonl              标准 normalized 5D 预测，不含 GT
localization/inference_manifest.json        预测身份与运行信息
localization/inference_provenance.json      推理 checkpoint、seed、来源信息
localization/visualizations/                每地图测试预测图
evaluation/localization/per_map/<map>.json  逐地图指标
evaluation/localization/summary_equal_map.json  equal-map 摘要
```

legacy 采用第 2 节的 3,900 间隔与旧 metadata 格式，不要求具有新增的 aligned 审计文件。正式与 smoke 目录相互独立；局部诊断单独使用 `outputs/csgo_aligned_validation/`。

可视化文件名为 `vis_map_<map>.png`，每地图固定随机选 10 个样本。radar 上显示 GT 实心点、预测空心点及连线，旁边显示 FPV 和反归一化坐标/角度；GT 仅用于监督或预测后的评估展示，不进入模型条件。越界仅影响绘图标记，不修改预测数值。

最终指标读取 `summary_equal_map.json` 的 `metrics_macro_map`：XY_Dist、Z_Dist、Pitch_Dist、Yaw_Dist，越低越好。XY/Z 为 benchmark 坐标单位，不标成米；角度为度，Yaw 使用周期最短距离。完整测试每图 2,000 条，以上均值指标的 pooled 与 equal-map macro 数学上相等；evaluator 没有单独的 pooled 字段。

## 6. 当前结果与执行状态

legacy seed 0 已完成 **19,500 updates**，`best` 和 `late` 都指向 `step_00019500`，保存的 validation L1 为 `0.0240991`。正式测试 coverage 完整，20,000 条、每图 2,000 条。[官方 equal-map 摘要](outputs/csgo_benchmark_v2_seen10/OpenVLA-OFT/seed_0/evaluation/localization/summary_equal_map.json) 为：

| 实验 | XY | Z | Pitch | Yaw |
| --- | ---: | ---: | ---: | ---: |
| legacy seed 0，step 19,500 | 38.4299 | 2.2797 | 2.2817° | 17.7660° |
| aligned v2 seed 42 | 未运行 | 未运行 | 未运行 | 未运行 |

legacy 的 batch、normalization、增强、可训练模块与 aligned 不同，上述分数仅作历史结果，不能用于声称已完成公平主实验。

aligned 已完成 7 项 CPU 单测及真实 7B 的局部 GPU 前后向、冻结检查和 checkpoint 组件重载。成功报告为 [report.json](outputs/csgo_aligned_validation/20260924T103648Z/report.json)，实际参数清单为 [parameter_audit.json](outputs/csgo_aligned_validation/20260924T103648Z/parameter_audit.json)。**正式 19,500-step 训练、20,000 条推理和正式评测均未启动**；局部测试不代表收敛或全量运行通过。失败探针的原因、历史 smoke、已验证范围和未确认事项见 [PLAN 第 9 节](CSGO_SEEN10_PLAN.md#9-验收要求与已完成记录)。
