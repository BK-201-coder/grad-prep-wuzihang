# Pathformer：ETTm2 四组预测实验复现

复现范围：多变量 ETTm2 数据集，输入长度 96，预测长度 96、192、336、720。论文对照值来自 [Pathformer（ICLR 2024）表 1](https://proceedings.iclr.cc/paper_files/paper/2024/file/2be6705de7412adf107900add727a795-Paper-Conference.pdf)。数值越低越好。

| 预测长度 | 论文 MSE / MAE | 本机 MSE / MAE | MSE / MAE 相对偏差 | 本次耗时 | 来源 |
|---:|---:|---:|---:|---:|---|
| 96 | 0.170 / 0.248 | 0.170883 / 0.249601 | +0.52% / +0.65% | 未记录 | 服务器上已有的正式运行 |
| 192 | 0.238 / 0.295 | 0.233686 / 0.291941 | −1.81% / −1.04% | 14 分 54 秒 | 2026-10-12 新运行 |
| 336 | 0.293 / 0.331 | 0.295909 / 0.331363 | +0.99% / +0.11% | 13 分 30 秒 | 2026-10-12 新运行 |
| 720 | 0.390 / 0.389 | 0.407154 / 0.398203 | +4.40% / +2.37% | 17 分 32 秒 | 2026-10-12 新运行 |

本次三个新运行均正常结束，退出码为 0，保存了最佳验证损失对应的 checkpoint、完整训练日志和测试图。96→96 的结果、checkpoint 和日志在本次任务开始前已经存在；本报告如实将其标记为既有结果。

## 环境与参数

- 服务器：NVIDIA GeForce RTX 4080 SUPER（32 GB），Python 3.10.8，PyTorch 2.1.2+cu118。
- 代码提交：`ea85d82932215e171357da47b3bc82d502344758`；`run.py` 固定随机种子为 1024。
- ETTm2 数据 SHA-256：`db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd`。
- 输入长度 96；多变量预测；batch size 512；最多 30 个 epoch；patience 10；学习率 0.001；3 层、每层 4 个 patch 备选，Top-K=2。每组的完整参数记录在 `train.log` 首行及对应 `run.sh` 中。
- 仓库 `requirements.txt` 固定 PyTorch 1.10.1+cu111，但当前服务器安装的是 2.1.2+cu118。96→720 的偏差可能受环境、随机性等影响；现有结果不足以确定原因。

## 保存位置

远端项目根目录：`/root/autodl-tmp/pathformer_repro/pathformer/`。

- 新运行：`reproductions/ETTm2_96_{192,336,720}_repro_20261012/`，每个目录有 `train.log`、`metrics.txt`、`run.sh`、依赖快照、数据与 checkpoint 校验值，以及 `checkpoints/`。
- 新运行测试图：`test_results/ETTm2_96_{192,336,720}_repro_20261012_PathFormer_ftETTm2_slM_pl96_{192,336,720}/`。
- 既有 96→96：`logs/LongForecasting/ETTm2_96_96_formal.log`、`checkpoints/ETTm2_96_96_PathFormer_ftETTm2_slM_pl96_96/`、`test_results/ETTm2_96_96_PathFormer_ftETTm2_slM_pl96_96/`。
- 项目汇总：`result.txt`。新运行使用独立名称，没有覆盖既有 checkpoint。

压缩包包含代码、数据、四组实验的日志、指标、checkpoint、测试图和复现脚本。原仓库的 `scripts/multivariate/ETTm2.sh` 有多处反斜杠续行紧贴参数的问题；本次新运行使用包内记录的显式参数调用，避免影响原脚本及既有结果。
