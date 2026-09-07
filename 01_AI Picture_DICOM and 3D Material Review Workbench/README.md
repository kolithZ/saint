# Chest X-ray Pneumonia Small-Sample Baseline

本项目实现了一个可复现的胸部 X 光二分类验证流程：输入单张影像，输出 `NORMAL` 或 `PNEUMONIA`，并给出肺炎概率。它用于算法研发、工程验证和课程展示，**不得用于临床诊断、治疗或医疗决策**。

虽然项目目录名包含 DICOM/3D，当前实际数据是二维 JPEG 胸片；因此实现采用 2D 模型，不把它伪装成 DICOM、CT/MRI 或 3D 任务。

## 已实现内容

- `inspect_dataset.py`：完整读取影像，统计类别、尺寸、通道、损坏文件和字节级/解码像素级重复项。
- `make_subset.py`：以固定随机种子从原始 `train/` 生成 `300 + 300` 训练集和 `50 + 50` 验证集清单；不复制图片。
- `train.py`：加载 ImageNet 预训练 ResNet18、冻结 backbone，仅训练线性分类层 3 个 epoch。
- `evaluate.py`：在原始 `test/` 的清单上计算 AUROC、PNEUMONIA Sensitivity、Accuracy 和混淆矩阵，并导出少量 TP/TN/FP/FN 供人工复核。
- `tests/test_metrics.py`：覆盖 AUROC 平分规则、Sensitivity 的阳性类定义和跨 split 图像泄漏拦截。

设计依据见 [DESIGN.md](DESIGN.md)。WBC/CRP 多模态融合保留为后续方案；本地数据没有逐影像配对的实验室结果，代码不会伪造此类数据或报告多模态效果。

## 数据集与许可

外部目录 `../chest_xray/` 中的顶层结构、文件命名和实际计数与 Kermany 等人公开的儿科胸片版本一致：

```text
chest_xray/
├── train/  NORMAL: 1,341, PNEUMONIA: 3,875
├── val/    NORMAL: 8,     PNEUMONIA: 8
└── test/   NORMAL: 234,   PNEUMONIA: 390
```

合计为 5,856 张。实际完整审计已成功解码全部 5,856 张（0 张损坏），尺寸范围为 `384–2916 × 127–2713`，其中 5,573 张为灰度 `L`、283 张为 `RGB`。源数据内部有 30 组字节级/解码像素级重复项，但没有任何一组跨越顶层 train、val、test。目录还含有 `chest_xray/chest_xray/` 的重复副本和 `__MACOSX/` 压缩元数据；项目配置只指向顶层三个 split，因此两者不会进入实验。

本地档案没有附带下载记录或患者/检查 ID 表，故无法证明其直接下载渠道，也不能将文件名当作可靠患者 ID。以下给出与该版本相符的正式原始数据与论文来源；如实际下载渠道不同，应在提交时补充该渠道的归档地址和版本号。

- 数据集：Kermany, D.; Zhang, K.; Goldbaum, M. *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*, Mendeley Data, Version 2, DOI [10.17632/rscbjbr9sj.2](https://doi.org/10.17632/rscbjbr9sj.2)。
- 许可：原始 Mendeley 页面标注为 [CC BY 4.0](https://data.mendeley.com/datasets/rscbjbr9sj/2)。使用、再分发或发表时须遵守该许可并正确署名。
- 论文：Kermany DS, et al. *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*. Cell, 2018, 172(5):1122–1131.e9, DOI [10.1016/j.cell.2018.02.010](https://pubmed.ncbi.nlm.nih.gov/29474911/)。
- 数据性质：公开的去标识化研究数据；当前本地副本没有可供本项目核验的 patient ID、study ID、检查时间或 WBC/CRP 记录。

## 环境安装

建议使用 PyTorch 有对应 wheel 的 Python 版本（Python 3.11+）。从本目录执行：

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

如果外部数据不在默认的相邻 `../chest_xray/` 路径，复制并修改 `config.json` 中的 `data_root`。所有相对路径均相对于本项目目录，不需要把 2.4 GB 影像复制进项目。

## 运行流程

依次执行：

```bash
./.venv/bin/python src/inspect_dataset.py
./.venv/bin/python src/make_subset.py
./.venv/bin/python src/train.py
./.venv/bin/python src/evaluate.py
./.venv/bin/python -m unittest discover -s tests -v
```

`inspect_dataset.py` 会遍历所有顶层影像并解码，因此运行时间取决于磁盘性能。`make_subset.py` 以 `config.json` 的 `seed` 固定抽样，并避免训练/验证样本与测试集中解码像素完全相同的影像重合。初始 `val/` 仅有各 8 张，本实现按设计从原始 `train/` 中重建平衡验证集，且不会使用那 16 张作模型选择。

训练第一次运行会由 torchvision 下载 ImageNet 预训练 ResNet18 权重；若下载失败，脚本会停止，不会悄悄改用随机初始化模型。评估读取训练生成的 checkpoint，阈值在训练前固定为 `0.5`，不会使用测试集调阈值。

## 输出与复核

所有可再生结果放在 `outputs/`，并默认不纳入版本控制：

```text
outputs/
├── dataset_summary.json       # 数据集检查结果
├── read_errors.json           # 无法读取的图像（正常情况为空）
├── duplicate_images.json      # 字节/解码像素重复项
├── image_inventory.csv
├── subset_summary.json
├── training_log.csv
├── training_summary.json
├── resnet18_frozen_backbone.pt
├── metrics.json               # AUROC、Sensitivity、Accuracy、混淆矩阵
├── predictions.csv
├── case_review.csv
└── error_cases/               # 每个 TP/TN/FP/FN 最多两张复核副本
```

`splits/manifest.csv` 以及按 split 保存的 CSV 包含每个实验样本的相对源路径和 SHA-256；重新使用同一数据与配置可以复现同一批样本。开发集会去除相同解码像素的重复图像，测试清单保留原始 `test/` 全部 624 个路径，以符合设计中“使用原始 test”的要求。

## 本次实际运行结果

以下结果来自本机的完整审计、固定种子 `20260907`、600 张训练图、100 张验证图、ImageNet `IMAGENET1K_V1` 预训练 ResNet18、冻结 backbone 和 3 个 epoch。没有使用 WBC/CRP，也没有在测试集上选阈值。

| 集合 | 样本数 | AUROC | Sensitivity（PNEUMONIA） | Accuracy |
|---|---:|---:|---:|---:|
| Validation | 100 | 0.9940 | 0.8800 | 0.9300 |
| Test（原始 test） | 624 | 0.9441 | 0.8205 | 0.8574 |

测试集固定阈值为 0.5，对应混淆矩阵：TN=215、FP=19、FN=70、TP=320。完整的机器可读结果在 `outputs/metrics.json`，每个预测在 `outputs/predictions.csv`，并导出了每个 TP/TN/FP/FN 最多两张图片供复核。

## 模型和评价

```text
JPEG chest X-ray
  → Resize 224×224、3 通道、ImageNet normalization
  → ImageNet-pretrained ResNet18（backbone frozen）
  → trainable linear classifier
  → P(NORMAL), P(PNEUMONIA)
```

训练时仅使用小角度仿射变换和轻微亮度/对比度扰动。默认 batch size 为 16、学习率为 0.001、epoch 为 3。正类固定为 `PNEUMONIA`，核心指标是：

- **AUROC**：以概率排序评价区分能力；实现对分数平局给予 0.5 分。
- **Sensitivity / Recall**：`TP / (TP + FN)`，反映肺炎样本被识别出的比例。
- **Accuracy**：附加统计，不作为本项目的主结论。

## 限制与后续多模态方向

1. 小样本训练仅验证工程链路，不能代表全量数据、外部数据或临床场景中的性能。
2. 缺少权威 patient/study ID；尽管代码阻止完全相同的解码影像跨 split，仍无法独立证明患者级隔离。
3. 公开标签可能有噪声，模型分数不是临床确诊概率。应检查导出的 FP/FN 是否与曝光、裁剪、文字标记或其他采集伪特征相关。
4. 原始测试集类别不平衡（234 NORMAL、390 PNEUMONIA），且其内部有少量重复图像路径；因此同时报告阈值无关的 AUROC 与 PNEUMONIA Sensitivity，并不将本次数值视为独立临床性能证据。
5. WBC 与 CRP 没有合法、逐影像配对的本地记录，当前只能运行 image-only baseline。真实多模态方案应使用 Patient/Encounter/Study ID 和检查时间，且只采用预测时已经可获得的检验结果，以避免时间泄漏；缺失时应回退至 image-only 模型。
6. 当前数据为二维 JPEG，不适用于 DICOM 元数据解析、三维体数据建模或跨医院泛化结论。

## AI 辅助说明

AI 辅助用于方案梳理、脚本和文档编写。数据来源由正式数据页和论文交叉核对；实际样本数由本地审计脚本生成；指标只来自本地运行产物；脚本与单元测试用于检查评价公式和最基本的泄漏防护。
