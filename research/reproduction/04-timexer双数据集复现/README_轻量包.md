# TimeXer ETTh2/ETTm2 轻量复现包

本包来自已完成并验证的完整结果包。实验为官方 TimeXer 外生变量预测（features=MS）：ETTh2、ETTm2 各预测长度 96/192/336/720，输入长度 96，随机种子 2021，共 8 项。

## 保留的内容

- `code/`：本次使用的 TimeXer 源码。
- `dataset/`：ETTh2、ETTm2 数据文件；其 SHA256 见 `environment/dataset_sha256.txt`。
- `environment/`：源码提交、依赖清单、GPU 信息和最小导入调整补丁。
- `orchestration/`：训练队列、单项运行、指标验证及打包脚本。
- `results/`：中文报告、指标汇总 CSV、预测示例图，以及各项命令、训练日志、时间、退出码、指标和 `COMPLETE.json`。

## 为缩小体积而移除的内容

- 每项训练的 `checkpoint.pth`：模型权重。
- 每项训练的 `pred.npy` 和 `true.npy`：完整逐点预测与真实值数组。

因此，本包可以查看八项实验结果并借助代码、数据和配置重新训练，但**不能直接加载已训练模型，也不能仅凭本包重新计算逐点预测指标**。`COMPLETE.json` 的 `verified=true` 是制作轻量包前对完整包进行验证的记录；若要独立复核权重与数组，请使用完整结果包。

`results/REPORT.md` 给出逐项指标和论文对照；`results/summary.csv` 可导入表格软件。实验环境与论文环境可能不同，本次为单一随机种子的一次复现。
