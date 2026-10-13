# Pathformer 在 ETTh2 上的四项长期预测复现

本次在 AutoDL 远程实例上完成 ETTh2 多变量预测：输入长度均为 96，预测长度分别为 96、192、336、720。四项训练、测试均正常结束，退出状态均为 0；原始数据、代码快照、运行脚本、完整日志、依赖清单、模型检查点和预测图已打包为 `ETTh2_reproduction_20261013.tar.gz`。这是 ETTh2 数据集上的四个设置，并非整篇论文所有数据集的复现。

## 结果

| 输入→预测 | 本次 MSE | 本次 MAE | 论文 MSE | 论文 MAE | 实际耗时 |
|---|---:|---:|---:|---:|---:|
| 96→96 | 0.282335 | 0.333468 | 0.279 | 0.331 | 6 分 47 秒 |
| 96→192 | 0.350044 | 0.375966 | 0.349 | 0.380 | 8 分 16 秒 |
| 96→336 | 0.341629 | 0.379706 | 0.348 | 0.382 | 7 分 18 秒 |
| 96→720 | 0.412281 | 0.431181 | 0.398 | 0.424 | 4 分 16 秒 |

论文值见 [Pathformer 论文](https://proceedings.iclr.cc/paper_files/paper/2024/file/2be6705de7412adf107900add727a795-Paper-Conference.pdf) 的长期预测结果表。本次与论文结果接近，但不完全相同；本次每个设置只运行一次（代码固定随机种子 1024），不能据此判断论文值是否可在多次重复后精确达到。

## 数据、代码与环境

- 数据：从 [ETDataset 官方仓库的 ETTh2.csv](https://github.com/zhouhaoyi/ETDataset/blob/main/ETT-small/ETTh2.csv) 获取。共 17,420 条逐小时记录、7 个数值变量及日期列，没有缺失值。SHA-256：`a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b`。
- 代码：[Pathformer 项目](https://github.com/decisionintelligence/pathformer)，提交 `ea85d82932215e171357da47b3bc82d502344758`，代码快照包含在压缩包中。
- GPU：NVIDIA GeForce RTX 4090 D；Python 3.10.8；PyTorch 2.1.2+cu118。每项的完整依赖版本和 GPU 信息见其 `pip_freeze.txt`、`gpu_at_start.txt`。
- 参数遵循仓库的 `scripts/multivariate/ETTh2.sh`：`features=M`、输入长度 96、30 轮上限、patience 10、batch size 512、学习率 0.0005。96/192/336 分别使用 `d_model=4/8/4` 与 `k=2`；720 使用 `d_model=16` 与 `k=3`，并采用脚本所列不同的 patch 列表。原 shell 脚本中有缺少空格的续行写法，因此使用压缩包内的 `reproductions/run_etth2.sh` 显式传参，避免参数粘连。

## 文件说明和校验

压缩包的 `reproductions/ETTh2_96_<预测长度>_repro_20261013/` 中分别有 `train.log`、`metrics.txt`、`config.txt`、开始/结束时间、检查点及其 SHA-256。`test_results/` 中有每项预测示例 PDF；`dataset/ETT/ETTh2.csv` 是训练数据；`reproductions/Pathformer_source_ea85d829.tar.gz` 是对应提交的代码快照。运行脚本会拒绝覆盖同名已有实验；如需再跑一轮，应先在脚本中更换 `run_id`，以免覆盖本次结果。

本地压缩包 SHA-256：`7d8138a4dc28e8975fd5e764c9f21cdbf45862a56eb40d0bbc5be31bd8d72507`，已与远端原包一致，并成功列出压缩包内 84 个条目。

远端保存位置：`/root/autodl-tmp/pathformer_repro/pathformer/reproductions/ETTh2_96_*_repro_20261013/`；远端总包：`/root/autodl-tmp/pathformer_repro/ETTh2_reproduction_20261013.tar.gz`。远端实例重建或停止后是否保留这些文件取决于 AutoDL 的存储策略；本地副本已保存。
