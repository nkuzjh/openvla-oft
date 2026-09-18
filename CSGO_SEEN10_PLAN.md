# OpenVLA-OFT 接入 CSGO Benchmark v2 Seen-10

本次范围：仅 localization；`RUN_FULL=0`，完成真实预训练模型的训练/保存/重载/推理/统一评测 smoke，不生成 Table 1 正式结果。默认 seed 0，命令支持后续 seed 1、2。

## 数据与模型

- `csgo_seen10/data.py`：按 minimal report、manifest、split 和发布 calibration 读取 Seen-10，不扫描图片重新划分。FPV 与 radar 使用发布映射，固定十地图顺序。
- 复用原生 Prismatic 双图视觉骨干、语言骨干、OFT 并行动作解码、L1 action head 和 all-linear LoRA。显式选择 CSGO 的 `horizon=1, action_dim=5`，保持机器人任务默认值。
- 输入只含 FPV、radar、固定定位 instruction 和 map name；禁用 proprio。动作占位符不含 GT，标签单独进入 L1 loss。输出顺序 `[x,y,z,pitch,yaw]`，XY/1024、发布 Z 区间、弧度/(2π)，不裁剪预测。
- 训练复用原生 finetune 的 forward、AdamW、MultiStepLR、torchrun/DDP 和 PEFT adapter/action-head checkpoint 约定；只为 manifest 数据、五次验证保存、恢复训练及本地日志增加必要分支/封装，不引入新训练框架。

## 文件与运行

- `csgo_seen10/`：数据适配、模型/训练推理封装、可视化；必要时对原入口、常量和惰性导入做小改动。
- `configs/csgo_seen10.yaml`、`train_seen10.py`、`infer_seen10.py`、`scripts/run_csgo_seen10.sh`：统一配置及 train/infer/eval/smoke 入口。
- `requirements-csgo-seen10.txt`、`scripts/setup_csgo_seen10.sh`：项目独立 `.venv`，Blackwell 兼容 PyTorch，原 OFT transformers fork。
- `CSGO_SEEN10.md`：最终可复现命令、环境、产物、验收证据。

预期命令：`bash scripts/setup_csgo_seen10.sh`；`bash scripts/run_csgo_seen10.sh smoke --seed 0`；完整运行使用同脚本的 `train`、`infer`、`eval` 子命令。正式输出为 `outputs/csgo_benchmark_v2_seen10/OpenVLA-OFT/seed_0/`，smoke 使用独立路径。

## 验收边界

1. 50,000/5,000/20,000 split 数量和归一化与通用 protocol 一致；推理 dataset 不返回 GT pose。
2. 真实 OpenVLA-7B 双图 batch 完成有限 loss 的 forward/backward 和参数更新；保存后重载生成标准 JSONL。
3. 总训练步数可被 5 整除，仅在 1/5 至 5/5 处验证与保存；保留全部 checkpoint，`late`/`best` 指向最后/最优验证 checkpoint；恢复 optimizer/scheduler/step 状态。
4. validation 选择 checkpoint；每地图固定随机 10 样本的 GT/预测 radar+FPV 可视化；训练主 loss 曲线。
5. 调用已存在的 `/home/jiahao/task/csgo_benchmark_v2_eval_general/run_eval.py smoke localization` 验证标准输出。正式 eval 必须完整覆盖 20,000 样本，不写 partial 正式结果；不改写通用评测器。
6. 不修改 UniLIP 和 DATA_ROOT，不覆盖已有运行结果；不运行完整训练。
