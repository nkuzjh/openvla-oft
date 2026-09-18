# OpenVLA-OFT · CSGO Benchmark v2 Seen-10

仅接入 localization：FPV + radar + 固定任务说明/地图名 → 绝对 `[x,y,z,pitch,yaw]`，horizon=1。禁用 proprio，动作占位符与 GT 无关。复用原生双图视觉骨干、OFT 语言模型、L1 action head、all-linear LoRA、AdamW/MultiStepLR 和 PEFT checkpoint 格式。

## 环境

从本项目根目录执行：

```bash
cd /home/jiahao/task/openvla-oft
bash scripts/setup_csgo_seen10.sh
```

训练/推理使用项目自己的 `.venv`（Python 3.11，PyTorch 2.7.1+cu128 / CUDA 12.8）；`requirements-csgo-seen10.txt` 固定本项目的 OFT Transformers fork。环境与权重下载脚本不向 UniLIP 环境安装依赖。基座目录为 `checkpoints/openvla-7b/`。

数据和通用评测器默认位置：

```bash
export DATA_ROOT=/home/jiahao/task/UniLIP/data/csgo_benchmark_v2
export SHARED_EVAL_DIR=/home/jiahao/task/csgo_benchmark_v2_eval_general
export UNILIP_PYTHON=/home/jiahao/miniconda3/envs/UniLIP/bin/python
```

数据适配直接复用通用评测目录的纯 Python `protocol.py`，因此训练时也需要保留该目录。不会扫描图片重新划分、复制数据或修改通用评测器。

## 命令

```bash
# 独立 smoke 输出；真实预训练模型的小规模训练、保存、重载、推理和统一评测
bash scripts/run_csgo_seen10.sh smoke --seed 0

# RUN_FULL=0；本次只接入 localization。正式运行按需执行：
bash scripts/run_csgo_seen10.sh train --seed 0
bash scripts/run_csgo_seen10.sh infer --seed 0
bash scripts/run_csgo_seen10.sh eval --seed 0
```

三个入口也可直接用项目 Python 调用（单进程）：

```bash
./.venv/bin/python train_seen10.py --config configs/csgo_seen10.yaml --seed 0
./.venv/bin/python infer_seen10.py --config configs/csgo_seen10.yaml --seed 0 \
  --checkpoint outputs/csgo_benchmark_v2_seen10/OpenVLA-OFT/seed_0/checkpoints/best
./.venv/bin/python eval_seen10.py --config configs/csgo_seen10.yaml --seed 0
```

追加种子只需将 `--seed 0` 改为 `--seed 1` 或 `--seed 2`。训练恢复和已有预测补齐示例：

```bash
./.venv/bin/python train_seen10.py --config configs/csgo_seen10.yaml --seed 0 \
  --resume-checkpoint outputs/csgo_benchmark_v2_seen10/OpenVLA-OFT/seed_0/checkpoints/late
./.venv/bin/python infer_seen10.py --config configs/csgo_seen10.yaml --seed 0 --resume \
  --checkpoint outputs/csgo_benchmark_v2_seen10/OpenVLA-OFT/seed_0/checkpoints/best
```

配置位于 `configs/csgo_seen10.yaml`。正式训练读取 50,000 个训练样本，`max_steps=19,500`、`event_every=3,900`，只在 `3,900/7,800/11,700/15,600/19,500` 五个步骤完整验证 5,000 个 validation 样本并保存；按验证 L1 loss 选择 `best`。

## 输出

正式 seed 根目录：`outputs/csgo_benchmark_v2_seen10/OpenVLA-OFT/seed_0/`；smoke 根目录：`outputs/csgo_benchmark_v2_seen10_smoke/OpenVLA-OFT/seed_0/`。

- `checkpoints/step_00003900/`、`step_00007800/`、`step_00011700/`、`step_00015600/`、`step_00019500/`：LoRA、原生动作头、processor、optimizer、scheduler、RNG 和数据位置；`checkpoints/late`、`checkpoints/best` 及根目录同名链接指向对应保存结果。
- `late` / `best`：最后一次保存 / 最优验证 checkpoint 的链接。
- `localization/predictions.jsonl`：20,000 行标准化预测，不含 GT。
- `logs/main_loss.jsonl`、`logs/main_loss.png`：主训练 loss 和曲线。
- 训练验证可视化：`validation_visualizations/step_00003900/vis_map_<map>.png`（其余四个保存步骤同样）；推理可视化：`localization/visualizations/vis_map_<map>.png`。每地图固定随机 10 个样本，radar 的 GT 实心点、预测空心点、连线与右侧 FPV；坐标为物理单位，角度为度。越界只影响标记显示，不裁剪预测数值。

归一化：X/Y 除以 1024，Z 使用发布 calibration 的固定范围，pitch/yaw 除以 360°。原始 split 角度为弧度。图像沿用原生 processor 的 224×224 直接缩放，无裁剪或翻转。

正式统一评测也可直接执行：

```bash
"$UNILIP_PYTHON" "$SHARED_EVAL_DIR/run_eval.py" localization \
  --pred-root outputs/csgo_benchmark_v2_seen10/OpenVLA-OFT/seed_0/localization \
  --data-root "$DATA_ROOT" \
  --output outputs/csgo_benchmark_v2_seen10/OpenVLA-OFT/seed_0/evaluation/localization
```

该命令要求完整 coverage，实际写入 `evaluation/localization/per_map/<map>.json` 和 `evaluation/localization/summary_equal_map.json`；指标为 XY_Dist、Z_Dist、Pitch_Dist、Yaw_Dist。smoke 仅生成诊断结果，不能填入 Table 1。已有正式结果不会被覆盖。

2026-09-17 已完成真实 `openvla/openvla-7b` smoke，实际输出位于
`outputs/csgo_benchmark_v2_seen10_smoke/OpenVLA-OFT/seed_0/`：5 optimizer steps、5 次各
100 条 validation、5 个 checkpoints（`best`=step 3，`late`=step 5），100 条标准预测
（每地图 10 条）、50 张 validation 可视化、10 张 inference 可视化和 loss 曲线。统一评测器
smoke 已通过，但仍为 `formal=false`，没有执行正式全量训练。

验收记录见 `outputs/csgo_seen10_validation/smoke_acceptance.json` 和
`outputs/csgo_seen10_validation/shared_evaluator_smoke.json`。该 smoke 输出不会再次直接重跑
覆盖；如需重跑，请配置不同的 `smoke_output_root`。

恢复验收也已完成：从 step 4 恢复后执行 step 5，数据位置、scheduler、CPU/CUDA/NumPy/Python
RNG 状态与原运行一致；动作头权重逐位一致。原生 CUDA BF16 SDPA backward 存在非确定性，
LoRA 权重最大差异为 `4.88e-4`，不保证训练逐位复现。推理恢复仅补算缺失的 5 条，最终
100 条预测与首次完整推理逐条一致。证据见 `outputs/csgo_seen10_validation/` 下的
`training_resume_acceptance.json`、`sdpa_repeatability.json` 和 `inference_resume_acceptance.json`。
