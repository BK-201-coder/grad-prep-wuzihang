# PatchTST/42 监督学习复现：ETTh2 与 ETTm2

本包记录了 8 项多变量长期预测实验：输入长度固定为 336，ETTh2 和 ETTm2 分别预测未来 96、192、336、720 个时间步。它只覆盖论文的一部分，不包括其余数据集、自监督学习或消融实验。每个配置按官方脚本使用随机种子 2021 运行一次，因此这里没有多随机种子统计结果。

论文：Nie 等，*A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*，ICLR 2023，https://arxiv.org/abs/2211.14730 。结果应与论文**表 3 的多变量监督学习 PatchTST/42 列**比较，不要与 PatchTST/64 或表 8 的单变量结果混用。

官方代码：https://github.com/yuqinie98/PatchTST

使用的代码提交：`204c21efe0b39603ad6e2ca640ef5896646ab1a9`

数据来源：https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small

两个 CSV 的 SHA-256 记录在 `environment/data-sha256.txt`。

## 实验设置与复现边界

模型和训练参数遵循官方的 `scripts/PatchTST/etth2.sh` 与 `scripts/PatchTST/ettm2.sh`：批量大小 128，最多训练 100 轮，Adam 学习率 0.0001，patch 长度 16，步幅 8，输入和输出均为 7 个变量，编码器 3 层，不使用混合精度。

- ETTh2：`d_model=16`、`n_heads=4`、`d_ff=128`，`dropout=fc_dropout=0.3`，`type3` 学习率调整，早停耐心值 100。
- ETTm2：`d_model=128`、`n_heads=16`、`d_ff=256`，`dropout=fc_dropout=0.2`，`OneCycle/TST` 学习率调整，`pct_start=0.4`，早停耐心值 20。

每项实验的完整参数保存在对应的 `checkpoints/.../config.json`。运行时明确指定了模型标识，避免原官方 shell 脚本中变量展开的歧义。

保留了 ETT 原始的按时间顺序划分方式（训练、验证、测试约为 12/4/4 个月），并且只用训练集拟合 `StandardScaler`。MSE 和 MAE 在标准化后的 7 个变量上计算。验证集打乱顺序、测试集 `drop_last=True` 也遵循官方实现；每项实际评估的测试窗口数记在结果表中。最佳模型根据验证损失选择。日志仍记录每轮测试损失，但测试损失没有用于选择模型或调整参数。

作者依赖文件固定 `torch==1.11.0`。本次使用远端 GPU 可运行的 PyTorch 2.1.2+cu118、Python 3.10.8、NumPy 1.26.4、pandas 1.5.3、scikit-learn 1.3.2、SciPy 1.11.4。完整安装清单和 GPU 信息位于 `environment/`。数据加载进程数为 4（官方默认 10），OMP/MKL 线程数为 4。环境、硬件和浮点随机性可能造成结果差异，因此报告列出实际结果与论文数值的差值。

对官方源码的改动记录在 `environment/instrumentation.patch`：额外保存完整参数、JSON 指标和真实值数组，并在生成图表后关闭绘图对象。模型、损失函数、优化器、数据划分、批处理和最佳模型选择逻辑未改动。`environment/benchmark.json` 只是训练前的速度和兼容性检查，不是论文实验结果。

## 本轻量包的内容

- `code/`：官方代码及上述结果保存改动。
- `data/`：ETTh2、ETTm2 的原始 CSV。
- `environment/`：依赖、硬件、源码提交、数据哈希与速度检查记录。
- `orchestration/`：实验参数、顺序运行程序、检查及报告生成程序。
- `results/summary.csv`：8 项 MSE、MAE、论文参考值、差值和运行时间。
- `results/REPORT.md`：汇总对照报告。
- `results/ETTh2/` 和 `results/ETTm2/`：按预测长度分开的日志、每轮损失、配置、JSON 指标、完成记录和曲线图。
- `results/progress.log` 与 `results/status.json`：整批进度及最终状态。

**轻量包不含** 8 个模型权重（`checkpoint.pth`）以及逐样本预测值和真实值数组（`pred.npy`、`true.npy`）。需要直接加载已训练模型、重新计算逐样本误差或制作新的逐样本图表时，请使用对应的完整包。轻量包保留了指标和校验结论，但无法仅靠本包重新执行数组级验证。

`archives/SHA256SUMS.txt` 是本轻量包所含文件的校验清单；解压后可在 `patchtst_repro` 目录运行 `sha256sum -c archives/SHA256SUMS.txt`。

