# Resume 权重加载后无进度：检查与修复（2026-09-30）

## 已确认的代码问题与诊断边界

旧训练循环先执行 `enumerate(train_loader)`，再对 `batch_index < next_batch` 执行 `continue`。DataLoader 在进入循环体之前已读取 FPV/radar、解码、增强、processor 预处理并打包，因此恢复位置之前的图片仍被完整处理一次。这不会更新模型，却会在 `num_workers=0` 时占用主进程 CPU、让 GPU 等待，而且旧代码不输出这段进度。

每 rank 无效处理的样本数为 `checkpoint.data_position.next_batch × checkpoint.training_settings.batch_size`。例如 390 updates/epoch、单卡 microbatch128/累计1，在 step4000 保存时通常对应 epoch10、next_batch100，旧代码需要先重处理12,800条样本（FPV和radar共25,600次图像加载）。实际值以 checkpoint JSON 为准，不根据文件夹名称推定。

远程截图只有 shards 加载完成、CPU 忙和 GPU 利用率低，尚不足以确认进程正处于这段循环。shards 之后还可能在执行 LoRA/action head 构造、数据 metadata 检查、optimizer 恢复或 provenance 哈希。若 `next_batch=0`，历史图片重读不能解释该次停顿。没有把本地修复通过描述为远程训练已恢复。

## 本次修改

- `csgo_seen10/sampling.py`：aligned 的 `GlobalUpdateSampler` 在索引层跳过已完成的 microbatch；只允许完整 optimizer update 边界，保留当前 epoch 的原始 shuffle/rank 分配，进入新 epoch 时清除偏移。
- `csgo_seen10/runner.py`：保存完整 epoch 的 batch 数，用绝对 batch 编号更新 checkpoint 位置；不因剩余 loader 长度缩短而提前判定 epoch 结束。legacy sampler 保留原恢复路径。
- 增加带 `flush=True` 的启动阶段、恢复 step/epoch/next_batch、首个有效 batch 和恢复后首个 optimizer update 日志。无需等到下一个50-step日志节点才能看到训练推进。
- 未改变 batch、梯度累计、样本顺序、增强定义、optimizer/scheduler、有效更新预算或 checkpoint 格式；没有删除原有恢复身份检查。aligned 图像增强由 seed/epoch/sample_id/view 确定，DataLoader 使用独立 generator，不需要重读历史图片来推进模型 RNG。

## 远程只读检查

从项目根目录执行，不启动模型：

```bash
./.venv/bin/python - <<'PY'
import json
from pathlib import Path
r = Path('outputs/csgo_benchmark_v2_seen10_aligned_v2/OpenVLA-OFT/seed_42')
p = r / 'checkpoints/late/training_state.json'
s = json.loads(p.read_text())
print('checkpoint:', p.resolve())
print('step:', s.get('step'))
print('data_position:', s.get('data_position'))
print('training_settings:', s.get('training_settings'))
f = r / 'logs/main_loss.jsonl'
if f.exists():
    print('last update:', f.read_text().splitlines()[-1:])
PY
```

`main_loss.jsonl` 每个 optimizer update 都写；stdout 旧代码仅前两步及每50步打印。检查最新 loss step 是否已超过 checkpoint step，可区分“没有推进”和“stdout 尚未到打印间隔”。

同步修复的 `csgo_seen10/sampling.py` 与 `csgo_seen10/runner.py` 后，已在运行的 Python 进程不会自动采用新代码。由用户确认原进程状态并决定停止/重启，不能向同一 run 目录并发启动第二个训练进程。单卡原运行保持 checkpoint 对应的 world size、batch、累计和实验配置，可以直接用 `python train_seen10.py` 恢复。2026-10-09 已另行实现 aligned 单卡→多卡恢复，需同时同步新增的 `csgo_seen10/resume.py`；此时使用 torchrun、保持有效batch相同，详见[扩卡恢复方案](CSGO_SEEN10_MULTI_GPU_RESUME.md)。

原 resume 命令保持兼容，后续手动重启可为 Python 增加 `-u`。这是及时输出日志的选项，不是索引跳过修复的替代品。启动日志依次报告：

```text
[startup] loading base model, LoRA adapter and action head
[startup] model ready; reading seen_train metadata
[startup] reading seen_validation metadata
[startup] restoring optimizer, scheduler and RNG ...
[startup] restored step=..., epoch=..., next_batch=...
[startup] computing provenance hashes and writing run metadata
[startup] ready: optimizer_step=.../19500, full_epoch_batches=...
[train] epoch=... next_batch=...; skip indices before image loading; waiting for first active batch
[train] first active batch ready; starting forward/backward at step=...
[train] optimizer_step=...
```

如果仍长时间停顿，请记录最后一条阶段日志和 elapsed，以及 checkpoint JSON 中的恢复位置。第一批未完成前后是两个不同的排查方向，不能仅凭 GPU 利用率截图归因。

## 验证范围

只执行 CPU 单元测试，不加载原始7B权重、不启动正式训练/推理/评测。测试文件 `tests/test_csgo_resume_sampling.py` 检查恢复后缀、历史数据不再访问、分布式索引、累计边界、epoch 重置和小型模型恢复一致性；这不代替远程数据盘和 GPU 实测。

用户原有 `CSGO_SEEN10.md` 中的未提交恢复命令修改保持原样。

本机实际通过：恢复专项4项、aligned模型适配3项、normalization/增强4项、路径5项，共16项CPU单元测试；另通过Python语法和 `git diff --check` 检查。专项小型模型测试使用每2个microbatch累计一次AdamW更新，并比较恢复后至epoch结束的参数、optimizer、MultiStepLR、Torch RNG和下一batch位置，旧回放与新索引跳过一致。此结论仅覆盖CPU测试，不宣称真实7B模型在不同GPU上的浮点结果逐位一致。
