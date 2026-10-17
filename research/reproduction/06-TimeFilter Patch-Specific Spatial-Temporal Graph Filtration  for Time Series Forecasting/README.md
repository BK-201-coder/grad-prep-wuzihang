# TimeFilter：ETTh2 与 ETTm2 复现结果

本包包含 ICML 2025 论文 *TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting* 的一部分复现：两个 ETT 数据集、输入长度 96、多变量预测（M），预测长度 96/192/336/720，共 8 项。每项均按作者对应脚本的超参数运行，随机种子由官方 `run.py` 固定为 2021。这里不是论文全部 13 个数据集的完整复现。

| 数据集 | 预测长度 | 本次 MSE | 本次 MAE | 论文 MSE | 论文 MAE | MSE 差值 |
|---|---:|---:|---:|---:|---:|---:|
| ETTh2 | 96 | 0.283356 | 0.337460 | 0.283 | 0.337 | +0.000356 |
| ETTh2 | 192 | 0.361819 | 0.391522 | 0.362 | 0.392 | -0.000181 |
| ETTh2 | 336 | 0.410070 | 0.431416 | 0.404 | 0.424 | +0.006070 |
| ETTh2 | 720 | 0.406915 | 0.433524 | 0.407 | 0.433 | -0.000085 |
| ETTm2 | 96 | 0.168588 | 0.254954 | 0.169 | 0.255 | -0.000412 |
| ETTm2 | 192 | 0.234703 | 0.298411 | 0.235 | 0.299 | -0.000297 |
| ETTm2 | 336 | 0.292745 | 0.336066 | 0.293 | 0.336 | -0.000255 |
| ETTm2 | 720 | 0.389961 | 0.393275 | 0.390 | 0.393 | -0.000039 |

论文列取自作者论文附录 Table 8 中固定输入长度 96 的 TimeFilter 列：<https://arxiv.org/html/2501.13041>。差值是“本次 − 论文”；论文数值保留 3 位小数，因此极小差值不代表逐位一致。论文结果进行了 3 次重复实验，本次每项只运行 1 次，不能用来复现论文的方差或统计显著性。

本次指标按保存的 `pred.npy` 和 `true.npy` 重新计算，并与 `metrics.npy` 核对。8 项均已验证模型权重、预测数组、真实值和指标。

## 文件说明

- `code/`：作者源码的独立副本；仅启用原本注释掉的预测与指标数组保存语句。
- `data/`：实际使用的 ETTh2.csv、ETTm2.csv；SHA-256 见 `environment/data_sha256.txt`。
- `results/`：各项 `metrics.npy`、`pred.npy`、`true.npy`、`input.npy`。
- `checkpoints/`：各项最佳验证模型权重。
- `runs/`：每项命令、训练日志、退出码与校验记录。
- `figures/`：测试集单个样本的预测示意图。
- `environment/`：Python、GPU、安装包、源码与数据校验记录。
- `progress.log`：队列开始与完成时间。
- `official_source.zip`：下载的原始作者源码快照；`timefilter_save_outputs.patch`：结果保存改动。

轻量包只包含代码、配置、日志、指标记录与示意图，不含完整预测数组、模型权重或数据集。完整包含这些文件，可用于复核和继续分析。

## 复现环境与差异

- 作者源码仓库：<https://github.com/TROUBADOUR000/TimeFilter>；下载时记录的上游提交：`dffde87e4fff0fdeeebbacde03dc1e432e15b3a1`。
- 作者 README 建议 Python 3.10.16，requirements.txt 指定 PyTorch 2.3.1、NumPy 1.26.4、pandas 2.2.3、scikit-learn 1.5.2。
- 本次复用已验证的独立环境副本：Python 3.10.8、PyTorch 2.1.2+cu118、NumPy 1.24.4、pandas 1.5.3、scikit-learn 1.2.2；GPU 为 NVIDIA vGPU-32GB。版本差异可能影响结果，因此不能称为严格的逐位复现。
- 作者脚本写死了 GPU 编号 7/4；本实例仅有 1 张可见 GPU，运行时映射为 GPU 0，未改变模型超参数。
- 不包含论文其他 11 个数据集，也未做多随机种子重复或消融实验。

