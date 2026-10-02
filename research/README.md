科研任务：

1.复习了transformer模型。

2.阅读了Deep learning for wildfire risk prediction Integrating remote sensing and environmental data，这篇论文系统综述了**多源遥感与环境数据驱动的森林火灾风险预测方法**，重点梳理了风险定义、关键影响因子、深度学习时空建模、特征共线性与可解释性，并指出未来应向**多模态数据融合、物理约束、概率校准和高效时空模型**方向发展。

3.阅读了Integrated wildfire danger models and factors，这篇论文系统综述了森林火灾危险度建模中的 GIS（地理信息系统）、RS（遥感）、统计、机器学习和模拟方法，并指出森林火灾危险度是气象、植被、地形、水文、社会经济、土地利用和气候等多类时空因素共同作用的结果，因此未来应基于高质量**多源数据构建更综合、可验证且具有区域适应性**的森林火灾危险度预测体系。

4.阅读了Wildfire Danger Prediction and Understanding With Deep Learning，这篇论文证明了多源环境数据与时间序列建模能够显著提高次日森林火灾危险预测能力，而下一步值得深入的方向不是简单换更复杂的模型，而是进一步**优化燃料状态表达、多时间尺度建模、空间尺度选择、样本构建和可解释性分析**。

5.阅读了Mesogeos A multi-purpose dataset for data-driven  wildfire modeling in the Mediterranean，多源数据驱动的森林火灾短期风险预测首先要解决“**数据如何统一、标签如何可靠、时间信息如何组织**”的问题，**在此基础上再讨论长短期记忆网络、变换器等模型结构**，往往比单纯追求更复杂的深度学习网络更有研究价值。

6.阅读了 Assessing and explaining temporal deep learning models for wildfire danger prediction，了解到森林火灾短期风险预测不仅要关注模型精度，还应重视**历史时间窗口、空间泛化和物理可解释性**；Transformer 能更好利用长期时序信息，但更复杂的模型不一定更优，因此后续研究应重点放在**多源数据融合、时空特征建模以及预测性能与物理一致性的协同提升**上。

7.阅读了 **Current model capabilities and future predictive pathways of wildfires under a changing climate**，了解到当前森林火灾预测主要包括 **LSM（陆面模型）、ESM（地球系统模型）和统计/机器学习模型** 三类，其中过程模型更强调物理机制和长期气候响应，机器学习更适合多源数据融合和高效预测；未来的发展重点不是单纯提高模型复杂度，而是加强 **过程模型与AI融合、动态燃料和人类活动表征、多模型集成、高分辨率时空预测、不确定性量化以及物理信息机器学习**，从而提高火灾预测在未来气候和不同区域中的可靠性与泛化能力。

8.对于创新也有了自己的理解，可以先了解不同模型框架，再缝合不同的框架或者只在框架内进行模块的修改，取得好的实验结果之后再做消融实验来验证修改后的模型确实有效。