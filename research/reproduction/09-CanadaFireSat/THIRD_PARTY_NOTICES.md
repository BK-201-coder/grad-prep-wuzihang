# 上游来源与许可

本项目是 CanadaFireSat 的小规模复现记录。`official/` 为作者代码与其 DeepSatModels 子模块的必要源码和配置，不包含上游Git历史及宣传图片。

- CanadaFireSat-Model：https://github.com/eceo-epfl/CanadaFireSat-Model ，提交号见 `configs/provenance.json`；许可见 `official/LICENSE`。
- DeepSatModels：https://github.com/hugoporta/DeepSatModels ，许可见 `official/deepsat/LICENSE.txt`。
- 对上游代码的兼容修改见 `configs/compatibility.patch`，已应用于打包源码。
- 数据与作者权重由官方公开仓库提供下载，本包仅包含下载清单，不包含这些数据或权重文件。

论文：Hugo Porta, Emanuele Dalsasso, Jessica L. McCarty, Devis Tuia. CanadaFireSat: Toward high-resolution wildfire forecasting with multiple modalities. ISPRS Journal of Photogrammetry and Remote Sensing, 2026. DOI: https://doi.org/10.1016/j.isprsjprs.2026.05.050 。
