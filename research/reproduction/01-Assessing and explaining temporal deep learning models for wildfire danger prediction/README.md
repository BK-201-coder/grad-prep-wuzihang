# 多源数据驱动森林火灾短期风险预测复现

## 1. 项目简介

本项目用于复现多源数据驱动的森林火灾短期风险预测流程，当前已完成以下模型的训练与测试：

- Random Forest（随机森林，RF）
- XGBoost
- LSTM（长短期记忆网络）
- Transformer

在固定时间划分实验之外，本项目还增加了 Transformer 的 **15 折时间交叉验证脚本**，用于自动完成每一折的配置生成、训练、最佳模型选择、测试、指标提取与结果汇总。

当前项目主要目录结构如下：

```text
wildfire-risk-prediction-reproduction/
├── README.md
├── code/
│   ├── train.py
│   ├── test.py
│   ├── cv_15fold_transformer.py
│   ├── test_dataset.py
│   ├── parse_config.py
│   ├── configs/
│   ├── datasets/
│   ├── dataloaders/
│   ├── models/
│   ├── trainer/
│   ├── tester/
│   ├── utils/
│   ├── xai/
│   ├── shap_local/
│   └── integrated_gradients/
├── data/
│   └── README.md
├── environment/
│   ├── requirements_original.txt
│   ├── pip_freeze.txt
│   ├── environment_versions.txt
│   └── gpu_info.txt
└── results/
    ├── single_fold_metrics.csv
    ├── cv/
    │   ├── transformer_15fold.csv
    │   └── transformer_15fold_summary.csv
    └── figures/
```

## 2. 实验环境

本次复现实验主要在 AutoDL 云服务器上完成。

### 2.1 硬件环境

```text
CPU: 18 核
内存: 60 GB
GPU: NVIDIA GeForce RTX 4090 D
```

### 2.2 软件环境

主要运行环境如下：

```text
Python: 3.10
PyTorch: 2.1.2+cu118
CUDA: 11.8
NumPy: 1.26.4
Pandas: 1.5.3
Scikit-learn: 1.2.2
Joblib: 1.2.0
XGBoost: 2.1.4
Optuna: 3.6.1
TensorBoard: 2.17.0
```

完整环境信息可查看：

```text
environment/environment_versions.txt
environment/pip_freeze.txt
environment/gpu_info.txt
```

## 3. 环境配置

建议使用独立 Python 环境运行本项目。

### 3.1 创建 Conda 环境

```bash
conda create -n wildfire python=3.10 -y
conda activate wildfire
```

### 3.2 安装 PyTorch

如果使用 CUDA 11.8，可安装与本次实验一致或兼容的 PyTorch 版本：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3.3 安装主要依赖

优先参考本项目保存的依赖文件：

```bash
pip install -r environment/requirements_original.txt
```

如果原始依赖文件中个别包与当前系统不兼容，也可以按本次复现实验的主要版本安装：

```bash
pip install pandas==1.5.3 scikit-learn==1.2.2 joblib==1.2.0 xgboost==2.1.4 optuna==3.6.1 numpy==1.26.4 matplotlib tensorboard
```

如果需要尽量还原本次 AutoDL 环境，请参考：

```text
environment/pip_freeze.txt
```

## 4. 数据准备

本次实验使用两个预处理 CSV 数据文件：

```text
positives.csv
negatives.csv
```

本次服务器上的实际数据路径为：

```text
/root/autodl-tmp/wildfire_repro/data/
```

为了避免将大文件直接上传至 GitHub，本仓库默认不包含这两个 CSV。

建议在本地建立：

```text
data/
├── positives.csv
└── negatives.csv
```

然后将配置文件中的 `dataset_root` 修改为实际数据目录，例如：

```json
"dataset_root": "/your/path/to/data",
"pos_source": "positives.csv",
"neg_source": "negatives.csv"
```

Windows 路径示例：

```json
"dataset_root": "D:/wildfire_reproduction/data"
```

建议在 JSON 中使用 `/`，避免反斜杠转义问题。

## 5. 输入特征

### 5.1 动态特征

当前模型使用 12 个动态特征：

```text
d2m
lai
lst_day
lst_night
ndvi
rh
smi
sp
ssrd
t2m
tp
wind_speed
```

### 5.2 静态特征

当前模型使用 12 个静态特征：

```text
dem
roads_distance
slope
lc_agriculture
lc_forest
lc_grassland
lc_settlement
lc_shrubland
lc_sparse_vegetation
lc_water_bodies
lc_wetland
population
```

对于 LSTM 和 Transformer，当前时序窗口长度为 30 天，因此典型输入形状为：

```text
dynamic shape = (30, 12)
static shape  = (12,)
```

## 6. 固定时间划分实验

本项目首先使用以下固定时间划分：

```text
训练集：2006–2019
验证集：2020
测试集：2021–2022
```

对应测试集规模：

```text
Positive samples: 2778
Negative samples: 4774
Total samples: 7552
```

## 7. 固定划分下的模型结果

| 模型 | Accuracy | Precision | Recall | F1 | AUPRC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.7839 | 0.7211 | 0.6728 | 0.6961 | 0.7618 |
| XGBoost | 0.7973 | 0.7293 | 0.7138 | 0.7215 | 0.7899 |
| LSTM | 0.7928 | 0.6958 | 0.7757 | 0.7336 | 0.7923 |
| Transformer | 0.8014 | 0.7022 | 0.7988 | 0.7474 | 0.8113 |

完整结果文件：

```text
results/single_fold_metrics.csv
```

## 8. 运行说明

运行前先进入代码目录：

```bash
cd code
```

下面所有命令默认均在 `code/` 目录下执行。

## 9. LSTM 训练与测试

### 9.1 训练

```bash
python train.py -c configs/config_lstm/config_train_autodl.json
```

训练完成后，模型通常保存在：

```text
saved/models/lstm/<时间戳>/
```

其中 `model_best.pth` 为验证集指标最优时保存的模型。

### 9.2 测试

修改：

```text
configs/config_lstm/config_test_autodl.json
```

将其中的：

```json
"model_path": "..."
```

指向实际 `model_best.pth`，然后运行：

```bash
python test.py -c configs/config_lstm/config_test_autodl.json
```

## 10. Random Forest 训练与测试

### 10.1 训练

```bash
python train.py -c configs/config_rf/config_train_autodl.json
```

训练完成后，随机森林模型通常保存为：

```text
saved/log/rf/<时间戳>/random_forest_model_<时间戳>.pkl
```

同时可能生成特征重要性图和混淆矩阵。

### 10.2 测试

修改：

```text
configs/config_rf/config_test_autodl.json
```

将模型路径指向实际生成的 `.pkl` 文件，然后运行：

```bash
python test.py -c configs/config_rf/config_test_autodl.json
```

## 11. XGBoost 训练与测试

### 11.1 训练

```bash
python train.py -c configs/config_xgb/config_train_autodl.json
```

训练完成后，模型通常保存在：

```text
saved/log/xgb/<时间戳>/xgboost_model_<时间戳>.pkl
```

### 11.2 测试

修改：

```text
configs/config_xgb/config_test_autodl.json
```

将模型路径指向实际模型文件，然后运行：

```bash
python test.py -c configs/config_xgb/config_test_autodl.json
```

## 12. Transformer 训练与测试

### 12.1 训练

```bash
python train.py -c configs/config_transformer/config_train_autodl.json
```

训练完成后会生成：

```text
saved/models/transformer/<时间戳>/
├── checkpoint-epoch1.pth
├── checkpoint-epoch2.pth
├── ...
├── checkpoint-epoch30.pth
├── model_best.pth
└── val_transformer_outputs_epoch*.csv
```

测试时建议使用：

```text
model_best.pth
```

### 12.2 测试

修改：

```text
configs/config_transformer/config_test_autodl.json
```

将：

```json
"model_path": "..."
```

修改为实际最佳模型路径，然后执行：

```bash
python test.py -c configs/config_transformer/config_test_autodl.json
```

## 13. Transformer 15 折时间交叉验证

本项目额外增加：

```text
cv_15fold_transformer.py
```

用于自动完成 Transformer 的 15 折时间交叉验证。

### 13.1 时间划分规则

数据年份范围：

```text
2006–2022
```

每一折使用：

```text
14 年：训练
1 年：验证
验证年之后连续 2 年：测试
```

Fold 1：

```text
Train: 2009–2022
Val:   2006
Test:  2007–2008
```

Fold 15：

```text
Train: 2006–2019
Val:   2020
Test:  2021–2022
```

## 14. 15 折脚本使用方法

### 14.1 查看帮助

```bash
python cv_15fold_transformer.py -h
```

### 14.2 冒烟测试

建议第一次运行时只执行 1 折、1 个 epoch：

```bash
python cv_15fold_transformer.py --start-fold 1 --end-fold 1 --epochs 1 --result-name transformer_smoke.csv
```

该步骤用于确认以下流程是否正常：

```text
年份划分
→ 自动生成配置
→ 训练
→ 查找 model_best.pth
→ 测试
→ 提取指标
→ 保存 CSV
```

### 14.3 正式运行完整 15 折

```bash
python cv_15fold_transformer.py --start-fold 1 --end-fold 15
```

不指定 `--epochs` 时，将使用 Transformer 配置文件中的正式训练轮数。

当前配置：

```text
epochs = 30
```

因此完整实验为：

```text
15 folds × 30 epochs
```

## 15. 长时间任务后台运行

在远程服务器中，建议使用 `screen` 避免 SSH 断开导致训练停止。

创建 screen 会话：

```bash
screen -S transformer15
```

进入项目：

```bash
cd /root/autodl-tmp/wildfire_repro/mesogeos/ml_tracks/a_fire_danger
```

启动训练：

```bash
python cv_15fold_transformer.py --start-fold 1 --end-fold 15
```

让任务继续在后台运行：

```text
Ctrl + A
然后按 D
```

重新进入：

```bash
screen -r transformer15
```

查看已有 screen：

```bash
screen -ls
```

## 16. 查看 15 折运行进度

查看已经完成的折：

```bash
cat cv_results/transformer_15fold.csv
```

查看某一折训练日志：

```bash
tail -f cv_results/logs/fold_01_train.log
```

查看测试日志：

```bash
tail -f cv_results/logs/fold_01_test.log
```

检查 Python 任务：

```bash
ps -ef | grep "[p]ython"
```

检查 GPU：

```bash
nvidia-smi
```

## 17. Transformer 15 折实验结果

| 指标 | Mean | Std |
|---|---:|---:|
| Accuracy | 0.821373 | 0.011602 |
| Precision | 0.726247 | 0.014340 |
| Recall | 0.820507 | 0.030851 |
| F1 | 0.770253 | 0.017388 |
| AUPRC | 0.831947 | 0.018018 |

完整结果：

```text
results/cv/transformer_15fold.csv
```

汇总结果：

```text
results/cv/transformer_15fold_summary.csv
```

## 18. 注意事项

### 18.1 数据路径

配置文件中可能仍包含 AutoDL 服务器绝对路径，例如：

```text
/root/autodl-tmp/...
```

如果在其他服务器或本地运行，需要修改：

```json
"dataset_root"
```

以及测试配置中的：

```json
"model_path"
```

### 18.2 数据文件名称

原始部分配置中使用：

```text
positives_binary_coastal.csv
negatives_binary_coastal.csv
```

本次复现实验实际使用：

```text
positives.csv
negatives.csv
```

因此运行前应确认配置中的 `pos_source` 和 `neg_source` 与实际文件名一致。

### 18.3 coastal 字段

当前公开 CSV 中未发现 `coastal` 字段，因此运行过程中可能出现：

```text
'coastal' column not found; defaulting to 0.
```

当前代码对此进行了兼容处理。

## 19. 当前复现结论

固定时间划分：

```text
Train: 2006–2019
Val:   2020
Test:  2021–2022
```

下，Transformer 的测试结果为：

```text
Accuracy = 0.8014
Precision = 0.7022
Recall = 0.7988
F1 = 0.7474
AUPRC = 0.8113
```

完成完整 15 折时间交叉验证后：

```text
Mean Accuracy = 0.8214
Mean Precision = 0.7262
Mean Recall = 0.8205
Mean F1 = 0.7703
Mean AUPRC = 0.8319
```

说明不同年份测试窗口之间存在明显性能差异，完整时间交叉验证能够比单一时间划分更全面地反映模型的跨年份泛化表现。

## 20. 说明

本项目主要用于学术复现、实验记录和森林火灾短期风险预测方法学习。

代码基于原公开项目框架进行环境适配、路径修改、模型复现以及时间交叉验证扩展。

