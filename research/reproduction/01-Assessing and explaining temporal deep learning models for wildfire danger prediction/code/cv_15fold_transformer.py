import argparse
import copy
import glob
import json
import os
import re
import subprocess
from pathlib import Path

import pandas as pd
import torch


# ============================================================
# 基本路径
# ============================================================

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

ALL_YEARS = list(range(2006, 2023))

TRAIN_TEMPLATE = ROOT / "configs/config_transformer/config_train_autodl.json"
TEST_TEMPLATE = ROOT / "configs/config_transformer/config_test_autodl.json"

GENERATED_DIR = ROOT / "configs/generated_cv/transformer"
RESULT_DIR = ROOT / "cv_results"
LOG_DIR = RESULT_DIR / "logs"

DATA_ROOT = "/root/autodl-tmp/wildfire_repro/data"

GENERATED_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 命令执行
# ============================================================

def run_command(cmd, log_path):
    print("\n" + "=" * 100)
    print("RUN:", " ".join(cmd))
    print("LOG:", log_path)
    print("=" * 100)

    output_lines = []

    with open(log_path, "w", buffering=1) as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            output_lines.append(line)

        process.wait()

    if process.returncode != 0:
        raise RuntimeError(
            f"\nCommand failed with code {process.returncode}:\n"
            + " ".join(cmd)
        )

    return "".join(output_lines)


# ============================================================
# 测试指标解析
# ============================================================

def parse_overall_metrics(output):
    pattern = (
        r"\[OVERALL\]\s+"
        r"acc=([0-9.]+)\s+"
        r"prec=([0-9.]+)\s+"
        r"rec=([0-9.]+)\s+"
        r"f1=([0-9.]+)\s+"
        r"auprc=([0-9.]+)"
    )

    match = re.search(pattern, output)

    if not match:
        raise RuntimeError("Cannot find [OVERALL] metrics in test output.")

    accuracy, precision, recall, f1, auprc = map(float, match.groups())

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auprc": auprc,
    }


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--start-fold",
    type=int,
    default=1,
    help="First fold to run (1-15)",
)

parser.add_argument(
    "--end-fold",
    type=int,
    default=15,
    help="Last fold to run (1-15)",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=None,
    help="Temporarily override epochs, e.g. --epochs 1 for smoke test",
)

parser.add_argument(
    "--result-name",
    default="transformer_15fold.csv",
)

args = parser.parse_args()


if not (1 <= args.start_fold <= 15):
    raise ValueError("start-fold must be between 1 and 15")

if not (1 <= args.end_fold <= 15):
    raise ValueError("end-fold must be between 1 and 15")

if args.start_fold > args.end_fold:
    raise ValueError("start-fold cannot be greater than end-fold")


RESULT_CSV = RESULT_DIR / args.result_name


# ============================================================
# 读取模板
# ============================================================

with open(TRAIN_TEMPLATE, "r") as f:
    train_template = json.load(f)

with open(TEST_TEMPLATE, "r") as f:
    test_template = json.load(f)


# ============================================================
# 支持断点继续
# ============================================================

if RESULT_CSV.exists():
    old_df = pd.read_csv(RESULT_CSV)
    results = old_df.to_dict("records")
    completed_folds = set(old_df["fold"].astype(int).tolist())

    print("Existing results found.")
    print("Completed folds:", sorted(completed_folds))
else:
    results = []
    completed_folds = set()


# ============================================================
# 构造并运行15折
# ============================================================

for fold_idx in range(args.start_fold, args.end_fold + 1):

    if fold_idx in completed_folds:
        print(f"\nFold {fold_idx} already completed. Skip.")
        continue

    # Fold 1 -> val=2006
    # Fold 15 -> val=2020
    val_year = 2005 + fold_idx

    test_years_int = [
        val_year + 1,
        val_year + 2,
    ]

    excluded = {
        val_year,
        *test_years_int,
    }

    train_years_int = [
        y for y in ALL_YEARS
        if y not in excluded
    ]

    assert len(train_years_int) == 14

    train_years = [str(y) for y in train_years_int]
    val_years = [str(val_year)]
    test_years = [str(y) for y in test_years_int]

    print("\n\n")
    print("#" * 100)
    print(f"FOLD {fold_idx}/15")
    print("Train:", train_years)
    print("Val  :", val_years)
    print("Test :", test_years)
    print("#" * 100)


    # ========================================================
    # 生成 train config
    # ========================================================

    train_cfg = copy.deepcopy(train_template)

    dargs = train_cfg["dataset"]["args"]

    dargs["dataset_root"] = DATA_ROOT
    dargs["pos_source"] = "positives.csv"
    dargs["neg_source"] = "negatives.csv"

    dargs["train_year"] = train_years
    dargs["val_year"] = val_years
    dargs["test_year"] = test_years

    train_config_path = (
        GENERATED_DIR /
        f"fold_{fold_idx:02d}_train.json"
    )

    with open(train_config_path, "w") as f:
        json.dump(train_cfg, f, indent=2)


    # ========================================================
    # 记录训练前已有 model_best
    # ========================================================

    before_models = set(
        glob.glob(
            str(
                ROOT /
                "saved/models/transformer/*/model_best.pth"
            )
        )
    )


    # ========================================================
    # 训练
    # ========================================================

    train_cmd = [
        "python",
        "train.py",
        "-c",
        str(train_config_path),
    ]

    # 冒烟测试可使用 --epochs 1
    if args.epochs is not None:
        train_cmd.extend(
            ["--ep", str(args.epochs)]
        )

    train_log = (
        LOG_DIR /
        f"fold_{fold_idx:02d}_train.log"
    )

    run_command(
        train_cmd,
        train_log,
    )


    # ========================================================
    # 找本折新产生的 model_best.pth
    # ========================================================

    after_models = set(
        glob.glob(
            str(
                ROOT /
                "saved/models/transformer/*/model_best.pth"
            )
        )
    )

    new_models = list(
        after_models - before_models
    )

    if not new_models:
        raise RuntimeError(
            f"Fold {fold_idx}: no new model_best.pth found."
        )

    model_best = max(
        new_models,
        key=os.path.getmtime,
    )

    model_best = Path(model_best).resolve()

    print("\nBest model:")
    print(model_best)


    # ========================================================
    # 读取 best epoch
    # ========================================================

    checkpoint = torch.load(
        model_best,
        map_location="cpu",
    )

    best_epoch = checkpoint.get("epoch")
    monitor_best = checkpoint.get("monitor_best")

    print("Best epoch:", best_epoch)
    print("Best val loss:", monitor_best)


    # ========================================================
    # 生成 test config
    # ========================================================

    test_cfg = copy.deepcopy(test_template)

    targs = test_cfg["dataset"]["args"]

    targs["dataset_root"] = DATA_ROOT
    targs["pos_source"] = "positives.csv"
    targs["neg_source"] = "negatives.csv"

    targs["train_year"] = train_years
    targs["val_year"] = val_years
    targs["test_year"] = test_years

    test_cfg["model_path"] = str(model_best)

    test_config_path = (
        GENERATED_DIR /
        f"fold_{fold_idx:02d}_test.json"
    )

    with open(test_config_path, "w") as f:
        json.dump(test_cfg, f, indent=2)


    # ========================================================
    # 测试
    # ========================================================

    test_log = (
        LOG_DIR /
        f"fold_{fold_idx:02d}_test.log"
    )

    test_output = run_command(
        [
            "python",
            "test.py",
            "-c",
            str(test_config_path),
        ],
        test_log,
    )


    # ========================================================
    # 提取 OVERALL 指标
    # ========================================================

    metrics = parse_overall_metrics(
        test_output
    )

    row = {
        "fold": fold_idx,
        "train_years": ",".join(train_years),
        "val_year": val_year,
        "test_years": ",".join(test_years),
        "best_epoch": best_epoch,
        "best_val_loss": monitor_best,
        **metrics,
        "checkpoint": str(model_best),
    }

    results.append(row)


    # ========================================================
    # 每完成一折立即保存
    # ========================================================

    df = pd.DataFrame(results)
    df = df.sort_values("fold")

    df.to_csv(
        RESULT_CSV,
        index=False,
    )

    print("\n" + "-" * 100)
    print("FOLD COMPLETE")
    print(row)
    print("Saved to:", RESULT_CSV)
    print("-" * 100)


# ============================================================
# 最终汇总
# ============================================================

df = pd.DataFrame(results).sort_values("fold")

metric_cols = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auprc",
]

print("\n\n")
print("=" * 100)
print("TEMPORAL CROSS-VALIDATION RESULTS")
print("=" * 100)

print(
    df[
        ["fold"] + metric_cols
    ].to_string(index=False)
)

print("\nMEAN:")
print(df[metric_cols].mean())

print("\nSTD:")
print(df[metric_cols].std())

summary = pd.DataFrame(
    {
        "metric": metric_cols,
        "mean": [
            df[c].mean()
            for c in metric_cols
        ],
        "std": [
            df[c].std()
            for c in metric_cols
        ],
    }
)

summary_path = (
    RESULT_DIR /
    "transformer_15fold_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)

print("\nFull fold results:")
print(RESULT_CSV)

print("\nSummary:")
print(summary_path)