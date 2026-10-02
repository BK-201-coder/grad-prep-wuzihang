1. **这篇论文主要讨论气候变化背景下未来森林火灾如何预测，核心观点是目前不存在一种模型能够解决所有问题，而是需要根据研究目的选择不同建模框架。** 论文将现有方法主要分为 LSM（Land Surface Model，陆面模型）、ESM（Earth System Model，地球系统模型）和 Statistical/ML（Statistical/Machine Learning，统计/机器学习模型）三类。
2. **LSM、ESM 和 ML 的最大区别在于“是否显式描述火灾物理过程”。** LSM 主要模拟植被、燃料、土壤水分、点火和火灾蔓延等陆面过程；ESM 在此基础上进一步模拟大气—陆地—海洋以及火灾排放对气候的反馈；ML 则主要利用历史气象、植被、地形、人类活动和火灾数据学习输入与火灾之间的统计关系。
3. **对于我的“多源数据驱动森林火灾短期风险预测”，ML 路线更加直接，但不能简单认为深度学习一定优于传统机器学习。** 近年的全球火灾预测研究比较 RF（Random Forest，随机森林）、XGBoost（Extreme Gradient Boosting，极端梯度提升）和 Neural Network（神经网络）后发现，XGBoost 已能达到甚至超过更复杂网络的效果，而增加燃料、天气和点火数据带来的收益明显大于单纯增加模型复杂度。
4. **森林火灾预测真正关键的是把“天气—燃料—点火”三个方面的信息同时表达出来。** 只利用气象信息容易把“适合燃烧的天气”误认为“实际会发生火灾”，加入燃料状态和点火信息后可以明显减少高风险误报，因此高质量多源数据可能比复杂网络结构更重要。
5. **过程模型和机器学习各有明显优势，也各有不足。** LSM/ESM 的物理意义和未来气候外推能力较强，但空间分辨率较粗、计算成本高，而且对点火和人类活动描述仍较简单；ML 速度快、适合高维多源数据，但强依赖历史样本，在未来极端气候或训练数据之外的 OOD（Out-of-Distribution，分布外）情景中可靠性可能下降。
6. **未来森林火灾预测不能只研究“会不会发生”，还需要同时关注空间、时间和极端事件。** 当前研究正在从传统表格分类逐渐发展到 Time-series Forecasting（时间序列预测）、Image Segmentation（图像分割）和 Spatiotemporal Prediction（时空预测），最终希望生成连续的未来火灾风险空间图，而不仅是单个样本的0/1分类结果。
7. **预测结果不能只有一个概率，还需要知道模型“有多确定”。** 最新研究开始同时建模 Epistemic Uncertainty（认知/模型不确定性）和 Aleatoric Uncertainty（偶然/数据不确定性），输出风险概率的同时给出不确定性，使低置信度预测能够被识别，这对真正的火灾预警比单纯提高 F1-score 更有实际意义。
8. **论文认为未来比较重要的路线不是“更大的黑箱模型”，而是 Process-based Model（过程模型）与 AI 的融合。** 包括使用 ML 作为计算昂贵物理模块的代理模型、利用残差学习修正过程模型偏差，以及发展 Physics-informed ML（物理信息机器学习），将生物物理约束和守恒关系融入网络训练。
9. **遥感基础模型和自监督学习可能会成为未来多源森林火灾预测的重要技术基础。** CVPR 2025 的 AnySat 和 RobSense、ICCV 2025 的 SkySense V2 已经开始统一处理不同传感器、不同空间分辨率、时间序列以及缺失模态数据，说明遥感研究正在从“针对一个任务训练一个模型”转向 Multi-modal Foundation Model（多模态基础模型）。([CVF开放获取](https://openaccess.thecvf.com/content/CVPR2025/html/Astruc_AnySat_One_Earth_Observation_Model_for_Many_Resolutions_Scales_and_CVPR_2025_paper.html?utm_source=chatgpt.com))

