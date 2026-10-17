from __future__ import annotations

import hashlib
import json
import os
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/root/autodl-tmp/timefilter_repro")
CODE = ROOT / "code"
ARCHIVES = ROOT / "archives"
FIGURES = ROOT / "figures"
EXPECTED = [f"{dataset}_96_{horizon}" for dataset in ("ETTh2", "ETTm2") for horizon in (96, 192, 336, 720)]
PAPER_FIXED_96 = {
    "ETTh2": {96: (0.283, 0.337), 192: (0.362, 0.392), 336: (0.404, 0.424), 720: (0.407, 0.433)},
    "ETTm2": {96: (0.169, 0.255), 192: (0.235, 0.299), 336: (0.293, 0.336), 720: (0.390, 0.393)},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify() -> list[dict]:
    records = []
    for run_id in EXPECTED:
        marker = ROOT / "runs" / run_id / "COMPLETE.json"
        if not marker.is_file():
            raise RuntimeError(f"Missing verified run: {run_id}")
        record = json.loads(marker.read_text())
        if record.get("verified") is not True:
            raise RuntimeError(f"Unverified run: {run_id}")
        result = Path(record["result_dir"])
        checkpoint = Path(record["checkpoint_dir"]) / "checkpoint.pth"
        assert checkpoint.is_file() and checkpoint.stat().st_size > 0
        metrics = np.load(result / "metrics.npy")
        pred = np.load(result / "pred.npy", mmap_mode="r")
        true = np.load(result / "true.npy", mmap_mode="r")
        assert metrics.shape == (5,) and np.isfinite(metrics).all()
        assert pred.shape == true.shape and pred.size > 0
        mse = float(np.mean((pred - true) ** 2))
        mae = float(np.mean(np.abs(pred - true)))
        if not (np.isclose(mse, metrics[1], rtol=1e-4) and np.isclose(mae, metrics[0], rtol=1e-4)):
            raise RuntimeError(f"Metric mismatch for {run_id}")
        records.append({"run_id": run_id, "mse": mse, "mae": mae, "shape": list(pred.shape), "result": result})
    return records


def make_figures(records: list[dict]) -> None:
    FIGURES.mkdir(exist_ok=True)
    for dataset in ("ETTh2", "ETTm2"):
        record = next(r for r in records if r["run_id"] == f"{dataset}_96_96")
        pred = np.load(record["result"] / "pred.npy", mmap_mode="r")
        true = np.load(record["result"] / "true.npy", mmap_mode="r")
        fig, axis = plt.subplots(figsize=(9, 3.5))
        axis.plot(true[0, :, 0], label="True", linewidth=1.6)
        axis.plot(pred[0, :, 0], label="Prediction", linewidth=1.3)
        axis.set(title=f"TimeFilter {dataset} 96→96 (test sample 0, channel 0)", xlabel="Forecast step", ylabel="Scaled value")
        axis.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / f"{dataset}_96_prediction.png", dpi=150)
        plt.close(fig)


def write_report(records: list[dict]) -> None:
    rows = []
    for record in records:
        dataset, _, horizon_text = record["run_id"].split("_")
        horizon = int(horizon_text)
        paper_mse, paper_mae = PAPER_FIXED_96[dataset][horizon]
        rows.append(f"| {dataset} | {horizon} | {record['mse']:.6f} | {record['mae']:.6f} | {paper_mse:.3f} | {paper_mae:.3f} | {record['mse'] - paper_mse:+.6f} |")
    rows = "\n".join(rows)
    upstream = (ROOT / "environment/upstream_commit.txt").read_text().strip()
    report = f"""# TimeFilter：ETTh2 与 ETTm2 复现结果

本包包含 ICML 2025 论文 *TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting* 的一部分复现：两个 ETT 数据集、输入长度 96、多变量预测（M），预测长度 96/192/336/720，共 8 项。每项均按作者对应脚本的超参数运行，随机种子由官方 `run.py` 固定为 2021。这里不是论文全部 13 个数据集的完整复现。

| 数据集 | 预测长度 | 本次 MSE | 本次 MAE | 论文 MSE | 论文 MAE | MSE 差值 |
|---|---:|---:|---:|---:|---:|---:|
{rows}

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

- 作者源码仓库：<https://github.com/TROUBADOUR000/TimeFilter>；下载时记录的上游提交：`{upstream}`。
- 作者 README 建议 Python 3.10.16，requirements.txt 指定 PyTorch 2.3.1、NumPy 1.26.4、pandas 2.2.3、scikit-learn 1.5.2。
- 本次复用已验证的独立环境副本：Python 3.10.8、PyTorch 2.1.2+cu118、NumPy 1.24.4、pandas 1.5.3、scikit-learn 1.2.2；GPU 为 NVIDIA vGPU-32GB。版本差异可能影响结果，因此不能称为严格的逐位复现。
- 作者脚本写死了 GPU 编号 7/4；本实例仅有 1 张可见 GPU，运行时映射为 GPU 0，未改变模型超参数。
- 不包含论文其他 11 个数据集，也未做多随机种子重复或消融实验。

远程训练时可用 `screen -x timefilter` 查看终端；按 `Ctrl+A`、`D` 退出查看但不停止训练。也可用 `tail -f /root/autodl-tmp/timefilter_repro/progress.log` 查看进度。
"""
    (ROOT / "README.md").write_text(report, encoding="utf-8")


def add_tree(archive: tarfile.TarFile, source: Path, target: str, exclude: set[str] | None = None) -> None:
    exclude = exclude or set()
    archive.add(source, arcname=target, recursive=False)
    for child in sorted(source.iterdir()):
        if child.name in exclude or child.is_symlink():
            continue
        if child.is_dir():
            add_tree(archive, child, f"{target}/{child.name}", exclude={"__pycache__", ".git"})
        else:
            archive.add(child, arcname=f"{target}/{child.name}", recursive=False)


def package(name: str, full: bool) -> Path:
    archive_path = ARCHIVES / name
    with tarfile.open(archive_path, "w:gz", compresslevel=1) as archive:
        for filename in ("README.md", "progress.log", "official_source.zip", "timefilter_save_outputs.patch", "run_timefilter_queue.sh", "finalize_timefilter.py"):
            archive.add(ROOT / filename, arcname=filename)
        add_tree(archive, ROOT / "environment", "environment")
        add_tree(archive, ROOT / "figures", "figures")
        add_tree(archive, CODE, "code", exclude={"data", "results", "checkpoints", "__pycache__", ".git"})
        if full:
            add_tree(archive, CODE / "results", "results")
            add_tree(archive, CODE / "checkpoints", "checkpoints")
            for dataset in ("ETTh2", "ETTm2"):
                archive.add(CODE / "data" / f"{dataset}.csv", arcname=f"data/{dataset}.csv")
        add_tree(archive, ROOT / "runs", "runs")
    return archive_path


def main() -> None:
    records = verify()
    ARCHIVES.mkdir(exist_ok=True)
    make_figures(records)
    write_report(records)
    full = package("TimeFilter_ETTh2_ETTm2_full.tar.gz", True)
    light = package("TimeFilter_ETTh2_ETTm2_light.tar.gz", False)
    checksums = "\n".join(f"{sha256(path)}  {path.name}" for path in (full, light)) + "\n"
    (ARCHIVES / "SHA256SUMS.txt").write_text(checksums)
    (ARCHIVES / "COMPLETE.json").write_text(json.dumps({"verified": True, "runs": EXPECTED, "full": full.name, "light": light.name}, indent=2))
    print(checksums)
    print(f"FULL_SIZE={full.stat().st_size} LIGHT_SIZE={light.stat().st_size}")


if __name__ == "__main__":
    main()
