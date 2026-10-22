# t-PatchGNN 源码与数据

本目录保存了 [官方 t-PatchGNN 仓库](https://github.com/usail-hkust/t-PatchGNN)中本次 PhysioNet 实验所需的代码。实验日期：2026-10-22。轻量包不含数据集。

## 目录

- `tPatchGNN/`：未改动的模型和训练入口。
- `tPatchGNN_wo_transformer/`：独立消融副本，跳过 Transformer 和它使用的位置编码，保留后续图网络。
- `lib/`：官方数据读取与指标工具。
- `data/physionet/`：供下载后的数据存放；官方 `processed.zip` 约 22 MB，需从[官方仓库](https://github.com/usail-hkust/t-PatchGNN/tree/main/data/physionet)获取。
- `requirements.txt`：官方依赖列表。服务器实际运行环境为 Python 3.10.8、PyTorch 2.1.2+cu118、RTX 4090 D；官方 README 使用 Python 3.9、CUDA 12.2，版本并不完全相同。

## 在 Linux 服务器上重新运行

先将 `processed.zip` 放在 `data/physionet/`，再在本目录解压数据：

```bash
unzip data/physionet/processed.zip -d data/physionet
```

安装依赖后，从轻量包根目录执行 `bash run_physionet.sh`。原始运行命令在 `reproduction_logs/run_B.sh`、`run_extras.sh` 中，但含服务器绝对路径，仅供核对。

消融副本与原模型的区别只有 `model/tPatchGNN.py` 中的 Transformer 前向调用，原始模型保存在 `tPatchGNN/`。论文未公开对应的消融代码，因此本次是按其文字定义实现的近似对照。

本次保存了日志与测试指标；官方训练脚本没有保存权重。
