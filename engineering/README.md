# AI Image Generator

基于 **min-DALLE + Streamlit** 实现的本地文本生成图像应用。用户输入英文 Prompt 后，系统会通过文本分词、BART 文本编码、自回归图像 Token 生成以及 VQGAN 解码完成图像生成，并提供历史记录、提示词收藏、风格预设、生成参数管理和生成统计等功能。

## 1. 项目功能

### 文本生成图像
- 输入英文 Prompt 生成图像
- 提供常用 Quick Prompts，可快速填入示例提示词
- 生成过程显示 4 个阶段的进度：
  1. 文本分词
  2. 文本编码
  3. 图像 Token 生成
  4. 图像解码与增强
- 生成完成后自动进行亮度、对比度和饱和度增强

### 风格预设
支持给 Prompt 自动追加风格描述，可在系统管理页面添加、删除和应用风格。

项目内置/可初始化的风格包括：
- 油画
- 水彩
- 素描
- 赛博朋克
- 蒸汽波
- 卡通
- 写实
- 梵高风格

### 历史记录
- 将 Prompt、生成图像、生成时间等信息保存到 SQLite 数据库
- 查看历史生成记录
- 删除单条历史记录
- 清空全部历史记录

### 提示词收藏
- 收藏常用 Prompt
- 对收藏 Prompt 进行分类
- 记录 Prompt 使用次数
- 支持删除收藏

### 系统管理
管理页面包含：
- 收藏提示词管理
- 风格预设管理
- 生成报告
- 系统日志
- 生成参数设置

### 生成统计与日志
可查看：
- 总生成次数
- 平均生成耗时
- 图像质量评分展示
- 常用提示词
- 生成成功次数与成功率
- 成功/失败日志
- 当前运行设备

同时支持将生成统计导出为文本报告。

> 注意：当前代码中的“图像质量评分”由程序在 75～95 之间随机生成，用于界面演示，并不是真实的图像质量评价模型。

---

## 2. 技术栈

| 模块 | 技术 |
|---|---|
| 编程语言 | Python |
| Web 界面 | Streamlit |
| 深度学习框架 | PyTorch |
| 文本生成图像模型 | min-DALLE |
| 文本编码 | BPE Tokenizer + DALL·E BART Encoder |
| 图像 Token 生成 | DALL·E BART Decoder |
| 图像解码 | VQGAN Detokenizer |
| 数值计算 | NumPy |
| 图像处理 | Pillow |
| 数据存储 | SQLite |
| GPU 加速 | CUDA（若 PyTorch 检测到可用 GPU） |
| 模型文件下载 | Requests / Hugging Face Mirror |

模型的核心生成流程为：

```text
English Prompt
      ↓
BPE Tokenizer
      ↓
DalleBartEncoder
      ↓
Autoregressive DalleBartDecoder
      ↓
256 Image Tokens
      ↓
VQGanDetokenizer
      ↓
256 × 256 RGB Image
      ↓
Pillow Image Enhancement
```

---

## 3. 项目结构

建议项目目录保持如下结构：

```text
engineering/src
├── app.py
├── min-DALLE.py
├── min-DALLE.ipynb
├── ai_image_generator.db
│
├── min_dalle_stub/
│   ├── text_tokenizer.py
│   ├── dalle_bart_encoder.py
│   ├── dalle_bart_decoder.py
│   └── vqgan_detokenizer.py
│
└── files/
    ├── vocab.json
    ├── merges.txt
    ├── encoder.pt
    ├── decoder.pt
    └── detoker.pt
```

各文件作用：

- `app.py`：项目主程序，负责 Streamlit 页面、模型调用、数据库管理和图像生成。
- `min-DALLE.py`：min-DALLE 核心生成流程的 Python 示例，并可在模型文件不存在时下载模型文件。
- `min-DALLE.ipynb`：Notebook 版本，用于分步骤学习和调试 min-DALLE 的生成流程。
- `ai_image_generator.db`：SQLite 数据库，保存历史记录、收藏 Prompt、风格预设、日志和模型参数等数据。
- `min_dalle_stub/`：min-DALLE 模型相关实现。
- `files/`：模型权重、Tokenizer 词表及相关配置文件。

---

## 4. 环境准备

建议使用独立 Python 虚拟环境。

### Conda

```bash
conda create -n min-dalle-app python=3.10 -y
conda activate min-dalle-app
```

也可以使用 `venv`：

```bash
python -m venv .venv
```

Windows：

```bash
.venv\Scripts\activate
```

Linux / macOS：

```bash
source .venv/bin/activate
```

---

## 5. 安装依赖

当前项目文件中没有提供完整的 `requirements.txt`，根据代码中的直接导入项，需要安装以下主要依赖：

```bash
pip install streamlit numpy pillow requests transformers emoji
```

PyTorch 建议根据本机 CUDA 版本单独安装。

最简单的 CPU/默认版本安装方式：

```bash
pip install torch
```

如果使用 NVIDIA GPU，建议安装与本机 CUDA 环境匹配的 PyTorch 版本。

项目还需要根目录中存在：

```text
min_dalle_stub/
```

否则会出现：

```text
ModuleNotFoundError: No module named 'min_dalle_stub'
```

---

## 6. 准备 min-DALLE 模型文件

`app.py` 会直接读取：

```text
files/vocab.json
files/merges.txt
files/encoder.pt
files/decoder.pt
files/detoker.pt
```

如果这些文件还不存在，可以先运行：

```bash
python min-DALLE.py
```

该脚本会创建 `files/` 目录，并在缺少相关文件时下载：
- `vocab.json`
- `merges.txt`
- `encoder.pt`
- `decoder.pt`
- `detoker.pt`

模型权重文件较大，首次运行需要等待下载完成。

---

## 7. 运行项目

进入 `engineering` 项目目录：

```bash
cd engineering
```

启动 Streamlit：

```bash
streamlit run app.py
```

启动成功后，Streamlit 通常会自动打开浏览器。

默认本地访问地址一般为：

```text
http://localhost:8501
```

---

## 8. 使用方法

1. 启动项目。
2. 点击 **Initialize Model** 加载 min-DALLE 模型。
3. 在底部输入英文 Prompt，例如：

```text
a futuristic city skyline at night
```

4. 点击 **Generate**。
5. 等待文本编码、图像 Token 生成和 VQGAN 解码完成。
6. 在页面中查看生成结果。
7. 可点击收藏 Prompt，也可以进入 **系统管理** 调整风格和生成参数。

如果已经选择风格预设，系统会自动把对应的英文风格后缀追加到 Prompt 后再进行生成。

---

## 9. 数据库

项目使用 SQLite 进行本地数据持久化。

主程序实际使用的核心数据表包括：

```text
image_history
prompt_favorites
generation_logs
model_config
style_presets
```

数据库文件：

```text
ai_image_generator.db
```

程序启动时会自动检查并创建主程序所需的数据表，因此不需要额外部署数据库服务。

---

## 10. 生成参数

系统管理页面可以调整生成参数。

默认值：

```text
temperature = 0.3
top_k       = 128
第三采样参数 = 4
```

页面还提供三组快捷配置：

```text
创意模式：0.8 / 256 / 8
平衡模式：0.5 / 128 / 5
稳定模式：0.3 / 64  / 3
```

这些参数会保存在 SQLite 数据库中，并在下一次生成时读取。

> 代码说明：`app.py` 当前把第三个参数命名为 `top_p`，但原始 `min-DALLE.py` 示例中该位置对应 `supercondition_factor`。如果后续继续完善项目，建议统一参数名称和实际含义。

---

## 11. 硬件说明

代码会通过：

```python
torch.cuda.is_available()
```

自动判断是否存在 CUDA GPU。

建议使用 NVIDIA GPU 运行。min-DALLE 的 Encoder、Decoder 和 VQGAN 权重较大，且项目使用 `float16` 推理；虽然代码中保留了 CPU 分支，但 CPU 推理速度会明显更慢，部分环境下的兼容性也可能较差。

---

## 12. 常见问题

### 1. 找不到 `min_dalle_stub`

确认项目根目录中存在：

```text
min_dalle_stub/
```

并且从项目根目录执行：

```bash
streamlit run app.py
```

### 2. 找不到模型文件

先执行：

```bash
python min-DALLE.py
```

或者手动确认以下文件已经存在：

```text
files/vocab.json
files/merges.txt
files/encoder.pt
files/decoder.pt
files/detoker.pt
```

### 3. CUDA Out of Memory

可以尝试：
- 关闭其他占用显存的程序
- 重启 Python/Streamlit
- 确认 PyTorch 与 CUDA 版本匹配
- 使用显存更大的 GPU

### 4. 页面启动但没有生成图像

优先检查：
- 模型是否成功 Initialize
- `files/` 中模型权重是否完整
- `min_dalle_stub/` 是否存在
- 控制台是否有 PyTorch/CUDA 报错
- `ai_image_generator.db` 是否具有读写权限

---

## 13. 项目核心流程

```text
用户输入 Prompt
        ↓
应用风格预设（可选）
        ↓
BPE 文本分词
        ↓
DALL·E BART Encoder 编码文本
        ↓
DALL·E BART Decoder 自回归生成 256 个图像 Token
        ↓
VQGAN Detokenizer 还原 RGB 图像
        ↓
Pillow 图像增强
        ↓
Streamlit 展示结果
        ↓
SQLite 保存历史记录和运行信息
```

---

## 14. 项目说明

本项目主要用于学习和演示 **文本生成图像模型的完整推理流程**，并在 min-DALLE 基础上增加了可视化 Web 界面和本地数据管理功能。

相比单独运行 `min-DALLE.py`，`app.py` 将模型推理封装成了一个完整的小型应用，能够实现：

```text
模型推理
+ Web 交互
+ 参数管理
+ 风格管理
+ Prompt 收藏
+ 历史记录
+ 日志统计
```

因此既可以用于理解 min-DALLE 的工作原理，也可以作为一个本地 AI 图像生成系统进行演示。
