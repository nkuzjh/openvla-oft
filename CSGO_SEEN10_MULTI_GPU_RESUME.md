# OpenVLA aligned 单卡 checkpoint 扩卡恢复：完整变更方案

日期：2026-10-09。目标是允许用户手动修改 GPU 数、microbatch 和梯度累计，将已有单卡 aligned checkpoint 恢复到双卡或更多卡，同时保留全局更新位置和有效 batch。允许计算和随机执行轨迹发生变化，不要求跨拓扑逐位复现。

## 1. 支持范围与配置入口

| 保存时 → 恢复时 | 行为 |
| --- | --- |
| 单卡 → 单卡 | 原有严格恢复：microbatch、累计、seed等设置分别一致；沿用原位置和本rank RNG文件 |
| 单卡 → 双卡/多卡 | 自动识别扩卡；仅当前 aligned + `global_full_update_batches` 采样路径；允许三个拓扑字段变化，但乘积必须一致 |
| 多卡 → 相同卡数、相同microbatch/累计 | 原有严格恢复；读取各rank分别保存的RNG，支持扩卡后的checkpoint继续恢复 |
| 多卡 → 不同卡数；相同卡数但改变microbatch/累计 | 本轮不新增支持，仍报设置不匹配 |
| legacy 单卡 → 多卡 | 原采样器未提供本方案的全局更新语义，不开放扩卡 |

不增加额外确认开关或强制参数。用户继续使用 `--resume-checkpoint`，在YAML里修改 `batch_size`、`grad_accumulation_steps`，通过torchrun/wrapper指定进程数。`world_size`以实际进程组为准，不以可见GPU数量或YAML声明猜测。

未修改现有YAML的microbatch、累计、验证batch、seed或输出目录。训练目标、normalization/clip、输入、增强、模型/LoRA、优化器、LR和评测协议沿用当前实验。

## 2. 有效batch与实验身份校验

从 `training_state.json.training_settings` 读取旧配置；兼容旧产物从 `provenance.training_settings` 读取。计算：

```text
旧有效batch = 旧world_size × 旧batch_size × 旧grad_accumulation_steps
新有效batch = 当前world_size × 当前batch_size × 当前grad_accumulation_steps
```

单卡扩卡时两个乘积必须相等；不相等直接报错，不自动改用户参数。当前aligned还要求实际有效batch为128。仅在单卡→多卡分支允许这三个字段不一致；seed、learning_rate、scheduler衰减点、checkpoint间隔、smoke标识等继续逐项匹配。

recipe、数据/manifest、base权重、normalization stats和其他 `resume_identity` 校验保留。改变服务器上的数据绝对路径仍可能触发现有身份检查；本次不包含跨路径checkpoint转换。校验在加载大模型前执行，坏的采样位置在数据metadata读取后、optimizer/RNG恢复前拒绝。

## 3. 自动换算恢复位置

已有checkpoint按每rank保存 `data_position={epoch, next_batch}`。`next_batch`是microbatch编号，不能直接当成更新数。

```text
U = floor(训练集大小 / 有效batch)             # 每个完整epoch的optimizer更新数
k = 旧next_batch / 旧grad_accumulation_steps  # 当前epoch已完成的更新数
新next_batch = k × 新grad_accumulation_steps
新epoch = 旧epoch
```

需满足：epoch/step/next_batch为非负整数，旧next_batch处于epoch范围内且可被旧累计次数整除，并且：

```text
checkpoint.step == 旧epoch × U + k
```

如果旧位置恰好处在epoch尾（k=U），归一为下一epoch的 `next_batch=0`；已经归一的旧checkpoint可直接转换。缺失位置、部分梯度累计边界、超过epoch尾或与step冲突均报错。

例如50,000条训练数据、有效batch128时，U=390，丢尾80条的原策略保留。step4000对应epoch10完成100次更新：

| 原配置 | 新配置 | 旧位置 → 新位置 |
| --- | --- | --- |
| 1卡×128×累计1 | 2卡×32×累计2 | `(10,100)` → `(10,200)` |
| 1卡×128×累计1 | 2卡×64×累计1 | `(10,100)` → `(10,100)` |
| 1卡×16×累计8 | 2卡×64×累计1 | `(10,800)` → `(10,100)` |

`GlobalUpdateSampler`仍以seed+epoch生成相同全局排列，再把每个完整update分配给各rank。映射后在索引层跳过已经完成的更新，不读取历史图片。当前epoch余下部分及后续epoch的全局样本组保持一致，不发生因位置换算错误造成的重复/漏样本。

optimizer、scheduler、best validation loss和global step从原checkpoint恢复。训练仍从旧step继续计数到原定max_steps，checkpoint事件仍按原全局step触发；不重新warmup或重置Adam状态。DDP平均rank梯度，microbatch loss仍除累计次数，不额外除world_size。

## 4. RNG复制规则

用户要求新增设备复制原单卡随机种子，本实现复制的是checkpoint中更完整的随机数生成器状态，而不只重新执行 `manual_seed(seed)`。

- 扩卡时，每个rank（包括rank0）均读取旧 `rng_state_rank_0.pt`。
- Python、NumPy、Torch CPU恢复相同的旧rank0状态。
- CUDA从旧训练设备对应的那一个状态取值，写入当前rank的实际训练设备 `ctx.device`。不能直接把旧CUDA列表用于 `set_rng_state_all`，否则新rank1可能没有得到源训练GPU的状态。
- 新保存的RNG文件增加 `cuda_device_index`，标明当时实际训练的逻辑设备。旧文件没有该字段时用索引0，兼容现有单卡入口默认 `local_rank=0` 的历史checkpoint。物理GPU编号经 `CUDA_VISIBLE_DEVICES` 映射后通常是逻辑0。
- 目标为CUDA时，旧CUDA状态缺失、源索引越界或状态为空直接报错。CPU测试只复制CPU相关状态。
- 普通单卡恢复和同配置多卡恢复继续调用原 `_restore_rng_state`，不切换到rank0复制逻辑。
- 扩卡后保存每个rank自己的RNG文件，后续相同多卡配置恢复时各自读取，能够连续续训。

当前aligned的shuffle、按样本确定的增强和dropout=0设置有助于控制随机变化；但改变计算分块、梯度归约顺序等仍可能产生浮点差异。复制RNG状态不等于保证跨卡数训练轨迹逐位相同。

## 5. 代码和日志

| 文件/函数 | 变更 |
| --- | --- |
| `csgo_seen10/resume.py::validate_resume_settings` | 识别1→N分支，校验有效batch及其余设置；其他场景继续strict |
| `csgo_seen10/resume.py::remap_resume_position` | 验证完整update边界和全局step，自动换算epoch/next_batch |
| `runner.py::_restore_single_rank_rng_state` | 复制旧rank0状态，将原active CUDA状态映射到当前rank设备 |
| `runner.py::_save_rng_state` | 新增CUDA源设备索引元数据 |
| `runner.py::train` | 串联校验、位置换算、RNG分支与记录；继续加载原optimizer/scheduler |
| `tests/test_csgo_topology_resume.py` | 配置、位置、采样、RNG和兼容性单元测试 |
| `tests/test_csgo_topology_resume_gloo.py` | 双进程CPU/Gloo恢复与native DDP前向集成测试 |

日志打印旧/新GPU数、microbatch/累计、有效batch、旧/新位置和RNG复制策略。`run_provenance.json`和后续checkpoint的provenance增加 `resume_transition`，记录from/to training_settings、old/new position、global_step、effective_batch_size和rng_strategy。`resume_transition_history`保留转换历史；多卡严格恢复时继续继承它。

源checkpoint不被重写，不预先伪造 `rng_state_rank_1.pt`。后续新checkpoint正常保存当前多卡设置、映射后的新位置以及各rank的RNG。

## 6. 用户手动操作示例

以下例子假定旧checkpoint有效batch为128，继续seed42和同一输出目录。先复制原YAML：

```bash
cp configs/csgo_seen10_aligned_v2.yaml configs/csgo_seen10_aligned_v2_2gpu.yaml
```

手动修改新文件中的训练批次字段：

```yaml
batch_size: 32
grad_accumulation_steps: 2
effective_batch_size: 128
```

验证batch独立于梯度累计，可按显存选择，不改变验证集大小。保留原recipe、seed、学习率、scheduler、数据和输出目录。确认原任务已停止后，由用户启动：

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 PYTHONUNBUFFERED=1 \
nohup bash scripts/run_csgo_seen10.sh train \
  --config configs/csgo_seen10_aligned_v2_2gpu.yaml --seed 42 \
  --resume-checkpoint outputs/csgo_benchmark_v2_seen10_aligned_v2/OpenVLA-OFT/seed_42/checkpoints/late \
  >>openvla_aligned_v2.resume_2gpu.out 2>&1 &
```

也可直接使用 `.venv/bin/python -u -m torch.distributed.run --standalone --nproc_per_node=2 train_seen10.py ...`。普通 `python train_seen10.py` 即使看见两张GPU也只启动一个训练进程，不构成双卡训练。

单卡恢复继续使用原命令及checkpoint保存时的microbatch/累计，无需新增参数。双卡保存新checkpoint后，继续使用同一双卡配置和命令恢复即可。同步至少包含 `csgo_seen10/runner.py` 和新增 `csgo_seen10/resume.py`；此前索引跳过修复的 `sampling.py` 也需已同步。

## 7. 验收标准与执行范围

1. 新旧乘积相同才允许扩卡；seed、LR等变化仍拒绝，legacy和其他未开放的转换继续报错。
2. 旧/新累计增减、累计相同、epoch中间、epoch尾和final位置正确；不完整/损坏位置拒绝。
3. 50,000样本、尾80条、双卡和四卡重组后的每update样本序列等于原单卡后缀，跨epoch同样成立。
4. CPU/Python/NumPy状态确实来自原rank0；CUDA模拟验证只写当前device，并测试旧文件默认索引0和新显式索引。
5. 单卡原函数依然要求对应rank文件并使用原恢复方式；扩卡后多卡严格恢复按各rank自身状态继续。
6. CPU/Gloo真实双进程小模型验证单卡保存→双卡恢复→双卡保存→双卡再次恢复；比较全局样本组、模型参数、AdamW和MultiStepLR状态及RNG。
7. 小VLA返回原生 `PrismaticCausalLMOutputWithPast`（含loss/logits/hidden_states），经生产DDP包装与native forward连续执行两次更新，验证unused lm_head路径及head同步。

本轮只执行CPU单元/小模型分布式测试；不下载/加载原始7B模型，不启动正式训练、推理或评测，不停止已有任务。真实双GPU/NCCL吞吐、显存和长期训练精度仍由远程运行验证。

## 8. 本机实际验收结果

执行命令：

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
git diff --check
```

结果：31项全部通过，约17.6秒。其中新增设置/位置/采样/RNG测试13项，新增真实双进程CPU/Gloo测试2项，既有回归测试16项。小模型跨卡比较采用float64及数值容差，不把结果扩展为真实7B模型的逐位复现承诺；同配置恢复的RNG读取与状态延续另有严格检查。native DDP测试出现 `find_unused_parameters=True` 的性能提示，两个rank均完成连续更新及head同步检查。

Python语法和差异空白检查通过。现有实验YAML未修改；用户暂存的单卡恢复命令仍保留在暂存区，本轮文档增补位于工作区。未访问或重启远程训练任务。
