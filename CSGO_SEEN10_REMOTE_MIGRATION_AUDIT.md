# OpenVLA 远程迁移改动核对与可转发 prompt 的依据

## 审计范围

用户提供的 `be12533eadd19c7abd0783f953ebf1635e07b0e6` 不属于本地 OpenVLA 仓库：该仓库并非浅克隆，`git show` 返回 `bad object`。该提交实际位于 `../RoboticsDiffusionTransformer`，说明为 `@67：cluster缺少依赖`。因此不能宣称已完成该 SHA 到 OpenVLA 当前状态的直接 Git diff。

本次按已确认的 OpenVLA 迁移历史核对：迁移前 `e91cbf581881b760386fbecefe35992eaa8128a2` → 当前 `b49367ebab2cf3d790aa61279a7a85d409e17fbd`，结合当前源码和用户远程报错。审计开始时工作区干净；本轮只新增本报告和可转发 prompt，不修改训练配置或程序，不执行安装、下载、训练、推理、评测。

| 提交 | 实际范围 |
| --- | --- |
| `17d777e` | 环境、模型下载、相对路径、CLI 覆盖、只读路径打印、数据协议与运行入口适配、测试和文档 |
| `3c59c84` | OpenVLA 默认调用共享 evaluator 的独立 `.venv`；保留旧参数别名；明确不做评测环境预检或回退 |
| `85d1003` | OpenVLA 文档补充共享评测器权重查找说明；本仓库没有实现指标权重缓存逻辑 |
| `6373231`、`5059a5d`、`b49367e` | 用户后续调整 aligned 的训练 microbatch、累计和验证 batch；不是迁移所需的通用默认值 |

## 本项目内的具体变化

| 项目 | 修改前 → 当前 | 证据 |
| --- | --- | --- |
| 数据/evaluator 默认路径 | 旧服务器绝对路径 → 相对 checkout 的同级目录 | `configs/csgo_seen10.yaml:6`、`configs/csgo_seen10_aligned_v2.yaml:9` |
| 集中路径解析 | 分散拼接 → `paths.py` 统一项目根目录、优先级和有限旧默认兼容 | `csgo_seen10/paths.py:9`、`:30` |
| 路径覆盖 | 增加 data/eval/model/Python CLI；统一 CLI > env > 配置 > 默认 | `csgo_seen10/cli.py:26`、`:41` |
| 轻量路径检查 | 原入口先导入运行时 → `--print-paths` 先输出后返回，wrapper 绕开 torchrun | `csgo_seen10/cli.py:51`、`train_seen10.py:18`、`scripts/run_csgo_seen10.sh:22` |
| 底层数据协议 | 固定 evaluator 位置 → 使用实际解析目录，CLI/config 值传递到数据加载 | `csgo_seen10/data.py:73`、`csgo_seen10/runner.py:155` |
| 输出和 checkpoint | 相对调用 cwd → 相对项目根目录 | `csgo_seen10/runner.py:161`、`:529` |
| 环境创建 | 原脚本可回退固定 UniLIP Python → 显式 Python3.11 / PATH / 项目内 Conda prefix | `scripts/setup_csgo_seen10.sh:111` |
| 环境检查 | 新增 dry-run/check 与旧位置环境识别，保留现有环境，不自动删除 | `scripts/setup_csgo_seen10.sh:73`、`:148`、`:170` |
| 下载工具 | 从 setup 内嵌逻辑抽出独立下载命令；支持专用模型路径变量和 dry-run/check | `scripts/download_csgo_model.py:14`、`:136` |
| 下载一致性 | 完整文件复用、partial 续传、大小和可获得的 LFS SHA256 校验，完成后形成正式文件 | `scripts/download_csgo_model.py:48`、`:58` |
| 准备依赖 | 保持模型 CUDA/fork 等配方，显式补充 `PyYAML==6.0.3` | `requirements-csgo-seen10.txt` |
| 评测 Python | 默认固定 UniLIP Conda → 解析后的 evaluator 目录 `.venv/bin/python` | `csgo_seen10/paths.py:61`、`csgo_seen10/runner.py:1599` |
| 测试 | 新增路径迁移/覆盖、symlink、无模型导入/无写入、无评测环境探测等 5 项测试 | `tests/test_csgo_paths.py` |
| 文档 | 补充远程准备、共享评测器安装、变量、只读检查、资产同步、续训边界 | `CSGO_SEEN10.md` 第 3 节、`CSGO_SEEN10_PLAN.md` 第 7.3 节 |

上表记录实际新增或重构的迁移功能，不声称下载的每个底层机制都是首次新增；部分机制从旧 setup 脚本抽出保留。本轮通过源码和提交核对，没有重新执行上述历史测试，不能据此宣称目标服务器部署通过。

## 实际覆盖顺序与两个环境

- 数据：`--data-root` > `CSGO_DATA_ROOT` > `DATA_ROOT` > 配置 > `../UniLIP/data/csgo_benchmark_v2`。
- evaluator 源码：`--eval-root` > `SHARED_EVAL_DIR` > `CSGO_EVAL_ROOT` > 配置 > `../csgo_benchmark_v2_eval_general`。
- evaluator Python：`--eval-python`/`--unilip-python`（同一参数的别名）> `CSGO_EVAL_PYTHON` > `UNILIP_PYTHON` > 非空 YAML `unilip_python` > `<实际 evaluator 目录>/.venv/bin/python`。
- base 模型：`--model-path` > `OPENVLA_MODEL_PATH` > 配置 > `checkpoints/openvla-7b`；原官方 Hub ID 的兼容保留。
- wrapper 自身的 Python：`PYTHON` > `<OpenVLA 项目>/.venv/bin/python`。它运行 OpenVLA 入口；eval 再启动独立 evaluator Python。激活 Conda 并不会直接替代这些显式路径。

两份 YAML 的 `unilip_python: null` 都允许自动使用共享环境；不存在时直接由子进程报错，不借用其他环境。Python 可执行路径不解引用 symlink，以保留 venv 身份。

训练仍从共享目录读取仅依赖标准库的 `protocol.py`，用模型环境执行。它并不调用评测器环境或生成指标模型。远程真实报错已显示目录名称为 `csgo_benchmark_v2_evaluator`，默认名称为 `csgo_benchmark_v2_eval_general`：通过 `SHARED_EVAL_DIR` 覆盖即可改变协议位置和默认评测 Python；不能只安装环境而遗漏/错放源码。另一个项目内的 `protocol.py` 不能自动当作同一份共享源码。

## 不能扩大的结论

1. 本范围不包含通用评测器仓库内的 `setup_env.sh`、依赖安装、指标权重下载/缓存和 SHA256 实现。OpenVLA 内只调整调用和文档，不能要求其他 VLA 各自重做一套。
2. 环境 dry-run/check 与训练路径打印不是模型 smoke；后者会运行真实训练/推理/评测，本轮均未启动。
3. OpenVLA 下载工具使用官方仓库的 `main`，非不可变 revision；可获得的 LFS 哈希在下载时核验，aligned 另有三个固定原始 shard 哈希。不能把它描述成全部资产都被固定 revision/哈希锁定。下载 URL 仍固定 Hugging Face 域名，不宣称已有通用 `HF_ENDPOINT` 镜像支持。
4. 下载脚本读取 `OPENVLA_MODEL_PATH`，不读取训练 YAML 或训练 CLI `--model-path`。自定义模型目录要在两个阶段显式保持一致。
5. `setup --check` 的模型部分只验证本地文件存在/非空及索引结构，不等于全量哈希复核；环境部分也不是远程驱动、GPU、网络运行保证。
6. 原有恢复身份约束没有放宽，支持换服务器从 base 新建实验不等于旧 checkpoint 能跨路径续训。
7. 当前 aligned 配置为 `batch_size=128`、`grad_accumulation_steps=1`、`val_batch_size=128`，单卡有效 batch 仍为 128；这不是本轮改动，也未做显存验收。已有运行文档仍有旧 microbatch 1 × 累计 128 的说明，不能把该段当作当前配置；本轮不改实验文档内容。其他模型应保留自己的批准配方。
8. 训练/推理/评测没有通用 `--dry-run`、`--output-root`、`RUN_FULL` 开关；可用的是 `--print-paths`。新增 prompt 不应让接收项目照抄不存在的命令。

## 可转发产物

完整执行 prompt 见 [CSGO_SEEN10_REMOTE_MIGRATION_PROMPT.md](CSGO_SEEN10_REMOTE_MIGRATION_PROMPT.md)。它无需依赖本报告或本次对话即可发送给其他项目；对 OpenVLA 已有实现提炼通用要求，同时要求接收项目保留自身模型配方和历史命令。
