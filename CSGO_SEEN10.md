# OpenVLA-OFT · CSGO Benchmark v2 Seen-10

本文是 Seen-10 定位任务的运行说明，记录环境、命令、输出和当前结果。实验设计、实现边界、横向比较与验收证据统一维护在 [CSGO_SEEN10_PLAN.md](CSGO_SEEN10_PLAN.md)。两份文档同时覆盖首次接入的 legacy 和当前已实现的 aligned v2；正式训练、全量推理及评测由使用者手动启动。

记录更新：2026-09-25。通用评测改为使用评测器项目自己的统一环境；本次安装独立评测环境并检查依赖/路径，不运行训练、推理或正式评测。

## 1. 实验范围与数据

只接入 localization，不含生成任务或 CrossMap。输入为当前 FPV RGB、对应地图 radar 和包含地图名的定位指令；不输入 GT 坐标、历史位姿或 robot state。输出为单步 5D pose `[x,y,z,pitch,yaw]`。

- 数据：`../UniLIP/data/csgo_benchmark_v2`（相对 OpenVLA 项目根目录），直接读取发布 manifest、split 和 calibration，不扫描图片重新划分。
- 共享评测器：`../csgo_benchmark_v2_eval_general`；数据适配也依赖其纯 Python `protocol.py`。
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

## 3. 环境、模型下载与两台服务器路径

推荐目录布局如下；同样适用于原服务器 `/home/jiahao/task/`。路径由 checkout 所在位置推导，不依赖用户名或启动命令时的工作目录。

```text
/home/user/yc57963/task/
  openvla-oft/
    .venv/
    checkpoints/openvla-7b/
    checkpoints/huggingface/
    outputs/
  UniLIP/data/csgo_benchmark_v2/
  csgo_benchmark_v2_eval_general/
    run_eval.py
    protocol.py
    .venv/bin/python  # 所有定位项目共用的评测解释器
```

### 3.1 新服务器准备

在新服务器项目根目录执行。环境安装与模型下载可分开运行：

```bash
cd /home/user/yc57963/task/openvla-oft
bash scripts/setup_csgo_seen10.sh --dry-run
bash scripts/setup_csgo_seen10.sh --skip-model
./.venv/bin/python scripts/download_csgo_model.py --dry-run
./.venv/bin/python scripts/download_csgo_model.py
```

也可用 `bash scripts/setup_csgo_seen10.sh` 一次完成环境安装、模型下载及 CUDA 检查，保留原命令行为。安装使用 Python 3.11，自动从 PATH 寻找，或由 Conda 创建项目内环境；可通过 `OPENVLA_SETUP_PYTHON=/实际路径/python3.11` 指定创建环境的解释器。不再依赖旧服务器 UniLIP Conda 环境。

依赖保持 PyTorch 2.7.1+cu128、Torchvision 0.22.1+cu128、PEFT 0.11.1 及 [requirements-csgo-seen10.txt](requirements-csgo-seen10.txt) 固定的 OFT Transformers fork。仍要求适配 CUDA 12.8 的驱动；路径适配不自动改变 CUDA/模型配方。OpenVLA 的训练/推理继续使用自身 `.venv`。定位评测默认由通用评测器项目的 `.venv/bin/python` 执行，该环境由用户单独安装，供其他定位项目直接复用；不再默认借用 OpenVLA 或 UniLIP 环境。

模型下载工具使用官方 `openvla/openvla-7b`，支持已有文件校验、断点续传与 SHA256 校验；默认落在本项目 `checkpoints/openvla-7b`，缓存保持在项目内。aligned 启动仍按 YAML 固定的三个原始 shard SHA256 复核，不从旧 CSGO adapter 热启动。该 base 不含一个可直接复用的 OFT 7D 连续 head，本项目新建 5D×1 head。原服务器历史资产核验见 [weight_integrity.json](outputs/csgo_seen10_validation/weight_integrity.json)。

只检查已准备的环境和本地模型文件，不安装、下载或初始化 CUDA：

```bash
bash scripts/setup_csgo_seen10.sh --check
```

`--check` 的模型部分只检查本地文件存在，不替代下载时或 aligned 启动时的完整权重哈希核验。Git 不同步 `.venv`、模型、benchmark 数据、共享 evaluator 和历史 outputs；需在新服务器单独准备完整数据 bundle 与 evaluator 目录。不要直接把原机器 `.venv` 当作可迁移环境；脚本会拒绝检测到的断链或指向旧位置的环境，需在新服务器重建。

### 3.2 安装统一通用评测器环境

同步完整的 `csgo_benchmark_v2_eval_general` 目录后，在目标服务器单独安装一次：

```bash
cd /home/user/yc57963/task/csgo_benchmark_v2_eval_general
bash setup_env.sh
cd /home/user/yc57963/task/openvla-oft
```

安装入口及固定依赖见 [评测器 README](../csgo_benchmark_v2_eval_general/README.md)。 默认安装命令也逐项准备生成评测权重：优先复用 UniLIP 缓存，缺失时下载至评测器 `loaded_models`。仅部署定位评测可用 `bash setup_env.sh --skip-weights`；定位训练/评测本身不需要这些生成指标权重。默认 CPU PyTorch 环境满足定位评测，避免不同模型项目的包版本影响评测；生成任务如需 GPU 可按该 README 显式选择 CUDA 后端。环境和评测器代码由用户自行部署；OpenVLA eval 不检查、不安装、不修复统一环境，也不会回退到模型环境，解释器未安装时直接由进程启动报错。

若此前设置过指向 OpenVLA/UniLIP 的解释器环境变量，先取消覆盖以使用统一默认：

```bash
unset UNILIP_PYTHON CSGO_EVAL_PYTHON
```

### 3.3 路径覆盖与启动前核对

默认 YAML 使用项目相对路径。解析优先级为 **显式路径 CLI → 环境变量 → YAML → 同级目录默认值**。所有相对文件路径以 OpenVLA 项目根目录为基准；Python 可执行文件保留 `.venv/bin/python` 路径，不解析 symlink 到其基础解释器。

| 用途 | 默认 | 环境变量 | CLI |
| --- | --- | --- | --- |
| Benchmark 数据 | `../UniLIP/data/csgo_benchmark_v2` | `CSGO_DATA_ROOT`，兼容 `DATA_ROOT` | `--data-root` |
| 共享 evaluator | `../csgo_benchmark_v2_eval_general` | `SHARED_EVAL_DIR`，兼容 `CSGO_EVAL_ROOT` | `--eval-root` |
| evaluator Python | `<shared_eval_dir>/.venv/bin/python` | `CSGO_EVAL_PYTHON`，兼容 `UNILIP_PYTHON` | `--eval-python`，兼容 `--unilip-python` |
| 原始模型目录 | `checkpoints/openvla-7b` | `OPENVLA_MODEL_PATH`，下载工具也读取 | `--model-path` |

旧 YAML 中原服务器的数据/evaluator 默认绝对路径在存在时沿用，不存在时迁移到同级目录。两份当前 YAML 的 `unilip_python: null` 表示使用解析后的 evaluator 目录内 `.venv/bin/python`；修改 `SHARED_EVAL_DIR` 也会随之切换默认评测环境。Python 显式路径只按优先级选取，不检查其存在性，不回退；旧 YAML 若显式指定其他 Python 则仍视为用户覆盖。两个数据变量同时存在时 `CSGO_DATA_ROOT` 优先；两个 evaluator 变量同时存在时 `SHARED_EVAL_DIR` 优先。清理不再使用的旧环境变量，避免覆盖 YAML。

布局不同时可在同一 shell 设置：

```bash
export CSGO_DATA_ROOT=/home/user/yc57963/task/UniLIP/data/csgo_benchmark_v2
export SHARED_EVAL_DIR=/home/user/yc57963/task/csgo_benchmark_v2_eval_general
export CSGO_EVAL_PYTHON=/home/user/yc57963/task/csgo_benchmark_v2_eval_general/.venv/bin/python
```

训练、推理、评测共用解析规则。只打印实际路径、不加载模型、不创建实验目录：

```bash
bash scripts/run_csgo_seen10.sh train --config configs/csgo_seen10_aligned_v2.yaml --seed 42 --print-paths
bash scripts/run_csgo_seen10.sh infer --config configs/csgo_seen10_aligned_v2.yaml --seed 42 --print-paths
bash scripts/run_csgo_seen10.sh eval --config configs/csgo_seen10_aligned_v2.yaml --seed 42 --print-paths
```

`--print-paths` 只检查解析结果，不承诺文件已齐全。当前改动支持在新服务器从原始 base 新建同配方实验，未放宽旧 checkpoint 中记录的绝对路径/数据身份恢复检查；跨路径迁移旧训练状态不能直接视为已经支持。

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
../csgo_benchmark_v2_eval_general/.venv/bin/python \
  ../csgo_benchmark_v2_eval_general/run_eval.py localization \
  --pred-root outputs/csgo_benchmark_v2_seen10/OpenVLA-OFT/seed_0/localization \
  --data-root ../UniLIP/data/csgo_benchmark_v2 \
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
