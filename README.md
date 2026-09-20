科研任务：

1.复习了transformer模型。

2.阅读了Detection of Cardiovascular Diseases in ECG Images Using Machine Learning and Deep Learning Methods，了解到可以用深度学习来提取特征用传统机器学习来进行分类。

3.阅读了A self-supervised electrocardiogram foundation model for empowering cardiovascular disease  prediction and genetic factor discovery，了解到ECG 不再只是诊断输入，而可以被基础模型转化成高维数字表型，再用于疾病预测、表型组研究和遗传学研究。对预训练和迁移学习有了更深的理解，并且学习到AUROC和AUPRC的区别，AUROC是整体区分能力，AUPRC是找少数阳性患者的能力。

4.阅读了Foundation models for electrocardiogram interpretation: clinical implications，了解到ECG自监督基础模型在标签有限、任务新颖、临床终点难以获得的数字生物标志物任务中，利用大量无标签ECG预训练获得更强的迁移能力。

5.对于AI-ECG 数字生物标志物方向的论文产出过程有了一定的了解，提出临床问题 → 用 ECG 基础模型得到数字指标 → 证明它和疾病/临床结局有关 → 进一步解释它和哪些超声、MRI、实验室指标有关 → 再定位到 ECG 里的具体导联和波形 → 最后做独立外部验证。

6.对于创新也有了自己的理解，可以先了解不同模型框架，再缝合不同的框架或者只在框架内进行模块的修改，取得好的实验结果之后再做消融实验来验证修改后的模型确实有效。



工程实践：

​	本次工程实践主要完成了一个基于 min-DALLE 的 AI 图像生成系统。在实践过程中，我先从 min-DALLE 的基本生成流程入手，了解了文本提示词经过分词、编码后，如何通过模型生成图像 Token，再利用 VQGAN 将这些 Token 还原成最终图像。通过对 `min-DALLE.py` 和 Notebook 代码的运行和分析，我对文本生成图像模型的基本工作过程有了比较直观的认识。

​	在能够正常生成图像之后，我进一步使用 Streamlit 对原有代码进行了界面化改造。用户可以直接在网页中输入英文提示词，系统会显示图像生成过程，并在生成完成后展示结果。为了让程序使用起来更方便，还加入了快捷提示词、风格选择、历史记录以及生成参数调整等功能。

​	在数据管理方面，项目使用 SQLite 保存生成记录、收藏的提示词、风格配置和运行日志。这样即使程序重新启动，之前的数据也可以继续保留。同时还增加了系统管理页面，可以查看生成情况、删除历史记录、管理提示词和修改相关参数。

​	整个实践基本经历了从模型代码学习、功能调试，到界面开发和系统整合的过程。相比最开始只能通过 Python 脚本生成单张图片，最终完成的程序在操作方式和功能上更加完整。通过这次工程实践，我不仅进一步熟悉了 Python 和 PyTorch 的使用，也对 Streamlit、SQLite 以及一个 AI 项目从模型到实际应用的开发过程有了更具体的认识。



