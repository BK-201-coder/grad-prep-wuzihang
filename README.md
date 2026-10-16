科研任务：

1.复习了transformer模型。

2.阅读了Deep learning for wildfire risk prediction Integrating remote sensing and environmental data，这篇论文系统综述了**多源遥感与环境数据驱动的森林火灾风险预测方法**，重点梳理了风险定义、关键影响因子、深度学习时空建模、特征共线性与可解释性，并指出未来应向**多模态数据融合、物理约束、概率校准和高效时空模型**方向发展。

3.阅读了Integrated wildfire danger models and factors，这篇论文系统综述了森林火灾危险度建模中的 GIS（地理信息系统）、RS（遥感）、统计、机器学习和模拟方法，并指出森林火灾危险度是气象、植被、地形、水文、社会经济、土地利用和气候等多类时空因素共同作用的结果，因此未来应基于高质量**多源数据构建更综合、可验证且具有区域适应性**的森林火灾危险度预测体系。

4.阅读了Wildfire Danger Prediction and Understanding With Deep Learning，这篇论文证明了多源环境数据与时间序列建模能够显著提高次日森林火灾危险预测能力，而下一步值得深入的方向不是简单换更复杂的模型，而是进一步**优化燃料状态表达、多时间尺度建模、空间尺度选择、样本构建和可解释性分析**。

5.阅读并复现了Mesogeos A multi-purpose dataset for data-driven  wildfire modeling in the Mediterranean，多源数据驱动的森林火灾短期风险预测首先要解决“**数据如何统一、标签如何可靠、时间信息如何组织**”的问题，**在此基础上再讨论长短期记忆网络、变换器等模型结构**，往往比单纯追求更复杂的深度学习网络更有研究价值。

6.阅读并复现了 Assessing and explaining temporal deep learning models for wildfire danger prediction，了解到森林火灾短期风险预测不仅要关注模型精度，还应重视**历史时间窗口、空间泛化和物理可解释性**；Transformer 能更好利用长期时序信息，但更复杂的模型不一定更优，因此后续研究应重点放在**多源数据融合、时空特征建模以及预测性能与物理一致性的协同提升**上。

7.阅读了 Current model capabilities and future predictive pathways of wildfires under a changing climate，了解到当前森林火灾预测主要包括 **LSM（陆面模型）、ESM（地球系统模型）和统计/机器学习模型** 三类，其中过程模型更强调物理机制和长期气候响应，机器学习更适合多源数据融合和高效预测；未来的发展重点不是单纯提高模型复杂度，而是加强 **过程模型与AI融合、动态燃料和人类活动表征、多模型集成、高分辨率时空预测、不确定性量化以及物理信息机器学习**，从而提高火灾预测在未来气候和不同区域中的可靠性与泛化能力。

8.阅读了 BCWildfire，了解到多源森林火灾短期风险预测的关键不仅在于选择 Transformer、Mamba 等模型，而在于**构建长期统一的多源时空数据、合理设计负样本和历史时间窗口，并同时利用燃料、气象、空间位置和人类活动信息**；后续研究可重点围绕**多时间尺度、动态/静态特征融合、空间邻域建模以及预测不确定性**展开。

9.阅读了CGFNet: Frequency-Domain Causal Discovery and Dual-Path Spectral Filtering for Wildfire Prediction，了解到**不同火灾驱动因素具有不同的时间变化规律和作用滞后，森林火灾预测不仅需要融合多源数据，还需要考虑变量之间的动态关系及长期、短期时间尺度的差异**；论文提出 CGFNet（因果图滤波网络），通过频域因果发现、时间超图和双路径动态滤波，同时学习变量间的时滞关系与变量自身的时间变化。

10.阅读了 Trustworthy Data-Driven Wildfire Risk Prediction and Understanding in Western Canada，了解到**多源森林火灾短期风险预测不仅需要融合长期与短期环境信息，还需要考虑负样本构建、预测不确定性和模型可解释性**；论文通过 WaveletMixer（小波混合网络）实现多尺度时间建模，并利用 SHAP（夏普利加性解释方法）分析不同环境下的驱动因素差异。

11.阅读并复现了 **Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting**，了解到复杂时序数据预测不仅需要融合**不同时间尺度下的局部变化与长期趋势信息**，还需要根据不同样本的动态特征**自适应选择有效的信息尺度**；论文提出 **Pathformer（多尺度自适应路径 Transformer）**，通过**多尺度 Patch 划分**结合**双重注意力机制（Dual Attention）**，同时建模不同时间分辨率和时间距离下的**局部细节与全局依赖关系**，并利用**多尺度路由器（Multi-scale Router）根据输入数据的趋势与周期特征动态选择关键尺度**，实现**自适应多尺度时序表征**。

12.阅读并复现了A TIME SERIES IS WORTH 64 WORDS LONG-TERM FORECASTING WITH TRANSFORMERS，**这篇论文最重要的贡献不是提出更复杂的 Transformer 结构，而是通过时间片段划分（Patching）和通道独立（Channel Independence）解决长期时间序列预测中“历史窗口受限、计算成本高、不同变量时间规律难以有效学习”的问题。**作者提出 **PatchTST**，将连续时间点划分为 Patch 作为 Transformer 的输入，在保留局部时间变化特征的同时降低注意力计算成本，并通过各变量独立建模、共享网络参数以及掩码自监督学习提高预测性能和泛化能力，为森林火灾预测中**多源驱动因素的长期历史信息提取**提供了重要思路，但尚未解决**不同环境状态下各驱动因素历史窗口自适应选择和复杂负样本识别**的问题。

13.**这篇论文最重要的贡献不是提出全新的 Transformer 注意力机制，而是解决时间序列预测中“目标变量与外部驱动因素难以有效融合”的问题**，作者提出 **TimeXer**，通过**内生变量的 Patch 级自注意力、外生变量的 Variate 级表示以及 Global Token 桥接的交叉注意力**，在保留目标序列时间变化特征的同时引入辅助变量信息，为森林火灾预测中**多源驱动因素交互建模**提供重要思路。



工程实践：

​	本次工程实践主要完成了一个基于 min-DALLE 的 AI 图像生成系统。在实践过程中，我先从 min-DALLE 的基本生成流程入手，了解了文本提示词经过分词、编码后，如何通过模型生成图像 Token，再利用 VQGAN 将这些 Token 还原成最终图像。通过对 `min-DALLE.py` 和 Notebook 代码的运行和分析，我对文本生成图像模型的基本工作过程有了比较直观的认识。

​	在能够正常生成图像之后，我进一步使用 Streamlit 对原有代码进行了界面化改造。用户可以直接在网页中输入英文提示词，系统会显示图像生成过程，并在生成完成后展示结果。为了让程序使用起来更方便，还加入了快捷提示词、风格选择、历史记录以及生成参数调整等功能。

​	在数据管理方面，项目使用 SQLite 保存生成记录、收藏的提示词、风格配置和运行日志。这样即使程序重新启动，之前的数据也可以继续保留。同时还增加了系统管理页面，可以查看生成情况、删除历史记录、管理提示词和修改相关参数。

​	整个实践基本经历了从模型代码学习、功能调试，到界面开发和系统整合的过程。相比最开始只能通过 Python 脚本生成单张图片，最终完成的程序在操作方式和功能上更加完整。通过这次工程实践，我不仅进一步熟悉了 Python 和 PyTorch 的使用，也对 Streamlit、SQLite 以及一个 AI 项目从模型到实际应用的开发过程有了更具体的认识。



