# Mesogeos 论文 Track A：三个官方基线的复现

2026-10-15 在 AutoDL 的 NVIDIA vGPU-32GB 上，使用作者提供的 **2023 年原版 Track A 数据**和官方代码，分别完成 LSTM、Transformer、GTN 一次训练与测试。每个模型均按官方训练配置跑满 30 轮，随机种子 12345。2025 年更新版数据**没有**用于这些实验。

这里是适合 GitHub 浏览的轻量版：保留运行脚本、配置、日志、指标清单和三个最佳模型权重。原版 CSV、全部 90 个中间轮次检查点与约 2 GB 的完整压缩包保存在本地交付目录，**没有上传 GitHub**。`code/` 是固定到提交 `828db48231637884a3fde5795760e42338cf780c` 的[官方源码子模块](https://github.com/Orion-AI-Lab/mesogeos)。克隆本仓库时如需源码，请使用 `git clone --recurse-submodules`。

## 测试结果与论文对照

| 模型 | 本次 Precision | 本次 Recall | 本次 F1 | 本次 AUPRC | 论文 Table 1：P / R / F1 / AUPRC |
| --- | ---: | ---: | ---: | ---: | --- |
| LSTM | 0.7595 | 0.8050 | 0.7816 | 0.8494 | 0.763 / 0.812 / 0.786 / 0.853 |
| Transformer | 0.7629 | 0.8203 | 0.7906 | 0.8691 | 0.802 / 0.759 / 0.780 / 0.856 |
| GTN | 0.7818 | 0.7801 | 0.7810 | 0.8565 | 0.781 / 0.790 / 0.786 / 0.858 |

这些是一次运行的测试结果，不应称作与论文完全一致。Transformer 的 Precision 比论文低约 0.0391，Recall 高约 0.0613；LSTM 和 GTN 的 F1 分别比论文低约 0.0044 和 0.0050。官方 `test.py` 在抽取负样本时未先固定 NumPy 随机状态，重复测试可能有小幅波动；本目录保存的是**首次测试**输出，没有为了接近论文而反复挑选结果。

## 文件怎么用

- `code/`：官方源码子模块；模型训练与测试的实现位于 `ml_tracks/a.fire_danger/`。
- `validate_*.py` 与 `preflight_*.py`：训练前核对原版数据哈希、论文样本划分和官方读取器输入。
- `prepare_*.py`、`run_*.sh`：准备三组配置，并调用官方 `train.py`、`test.py`；脚本使用实验服务器上的绝对路径，迁移到新机器前需要修改路径。
- `finalize_*.py`、`capture_mesogeos_environment.py`：检查训练结果并记录运行环境。
- `results/`：顶层的 `COMPLETE.json`、配置和 `run.log` 属于 LSTM；`results/lstm/` 保存其最佳权重与日志；`results/transformer/` 和 `results/gtn/` 分别保存另外两个模型的配置、日志、最佳权重及 `COMPLETE.json`。各 `COMPLETE.json` 记录指标和文件哈希。

这个 GitHub 目录可以查看和核对结果，但**不能单独重新训练**：还需要从作者提供的[官方数据入口](https://drive.google.com/drive/folders/1aRXQXVvw6hz0eYgtJDoixjPQO-_bRKz9)取得原版 CSV，并配置相应环境与路径。完整离线成果另保存在 `D:\CodexDownload\Mesogeos_TrackA_LSTM`。

训练源码提交为 `828db48231637884a3fde5795760e42338cf780c`。环境主要版本：Python 3.10.8、PyTorch 2.1.2+cu118、NumPy 1.24.4、Pandas 1.5.3、scikit-learn 1.2.2。GTN 官方配置指定 GPU 1，而本实例只有 GPU 0，因此仅将其 `gpu_id` 改为 0；三个模型另外只调整数据与成果路径、实验名称，未改官方训练超参数。

数据校验：`positives.csv` SHA-256 为 `c44554e29f0d676ad13d42950500fb5bc6361c5dc5afe77958c59b7cc3c8ed0b`；`negatives.csv` SHA-256 为 `ce5e6fd204d4c26189fe45c2a2d709ccfdead6fa08c0fcbeb5ce5c59d654820b`。原始负样本候选池是 17,342 个样本，官方读取器按论文设置抽出 17,148 个（正负比例 1:2）；最终训练、验证、测试样本数分别是 19,353、2,262、4,107。

论文：[Mesogeos: A Multi-Purpose Dataset for Data-Driven Wildfire Modeling in the Mediterranean](https://proceedings.neurips.cc/paper_files/paper/2023/file/9ee3ed2dd656402f954ef9dc37e39f48-Paper-Datasets_and_Benchmarks.pdf)；[官方源码](https://github.com/Orion-AI-Lab/mesogeos)。
