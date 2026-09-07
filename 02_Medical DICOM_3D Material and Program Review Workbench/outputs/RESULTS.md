# 本次实验与案例检查

本报告对应 2026-09-07 默认配置的一次固定种子实验，数据来自项目外部的 `../chest_xray`。后续新实验写入其他目录，不会自动改写本文。

## 数据与方法

原始 train/val/test 共 5,856 张图片，均可解码。分别为 5,216 / 16 / 624 张，训练从 train 固定选择 NORMAL 和 PNEUMONIA 各 300 张。原始文件、标签与划分未修改。

文件内容哈希和解码像素哈希均检出 30 组完全重复，但没有跨划分完全重复；患者级隔离因缺少可靠 ID 未核验。来源和许可记录仍需依据真实下载来源补充。

输入为 RGB 224×224，经 ImageNet 标准化后输入预训练 ResNet18。Baseline 仅训练 fc；对照组额外训练 layer4，早期层与其 BatchNorm 统计保持冻结。两组均训练 3 epoch，batch size 16，随机种子 42，使用 CPU。最佳 checkpoint 根据 val 平均交叉熵选择，与 test 结果无关。

| 实验 | 选择 epoch | val loss | Sensitivity | ROC-AUC | Specificity | Accuracy | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 3 | 0.3114 | 0.9385 | 0.9373 | 0.6880 | 0.8446 | 24 | 73 |
| Layer4 对照 | 2 | 0.1473 | 0.9872 | 0.9420 | 0.6538 | 0.8622 | 5 | 81 |

Baseline 训练阶段约 39.4 秒，对照组约 51.2 秒；时间包含训练准备与验证，不包含此前全量数据审计和之后独立 test 评估，且受机器负载影响。

对照组的 Sensitivity 提升 4.87 个百分点，ROC-AUC 提升 0.0046，但误报增加 8 张、Specificity 下降 3.42 个百分点。这里只能说明此次固定实验的点估计变化，尚未进行多随机种子、置信区间、外部数据或临床验证。

## 曲线与预测产物

Baseline：

![Baseline training](baseline/training_curve.png)

![Baseline confusion matrix](baseline/confusion_matrix.png)

Layer4 对照：

![Layer4 training](improved/training_curve.png)

![Layer4 confusion matrix](improved/confusion_matrix.png)

对照组第 3 轮训练损失进一步下降，但 val 损失从 0.1473 上升到 0.2433，因此保留第 2 轮 checkpoint。验证集只有 8 张 NORMAL 和 8 张 PNEUMONIA，不能从 val ROC-AUC=1 推断稳定泛化性能。

连续概率见 [Baseline predictions](baseline/predictions.csv) 与 [Layer4 predictions](improved/predictions.csv)，每份 624 行。ROC 图分别为 [Baseline ROC](baseline/roc_curve.png) 和 [Layer4 ROC](improved/roc_curve.png)。

## 成功与失败案例

以下为按固定种子抽取的案例，每种最多 3 张。标签均来自原始目录；这里检查的是文件、图表与预测记录能否对应，不重新判断医学标签。

正确预测 NORMAL（TN）：

![Baseline TN](baseline/cases/TN.png)

正确预测 PNEUMONIA（TP）：

![Baseline TP](baseline/cases/TP.png)

真实 PNEUMONIA 预测成 NORMAL（FN，漏报）：

![Baseline FN](baseline/cases/FN.png)

真实 NORMAL 预测成 PNEUMONIA（FP，误报）：

![Baseline FP](baseline/cases/FP.png)

Baseline 漏报示例 `person117_bacteria_557.jpeg` 的 P(PNEUMONIA) 约为 0.072；误报示例 `NORMAL2-IM-0246-0001.jpeg` 的 P(PNEUMONIA) 约为 0.912。这说明预测错误也可能伴随较高置信程度，不能将 Softmax 数值解释为可靠临床风险。

在导出的错误案例中可见黑边、文字侧标及取景范围差异；这是 AI 辅助的图像工程观察，不构成误判原因的证明或医学判断。尚未开展系统质量标注，因此不能断言错误集中于模糊、曝光或某种体位。

对照组的 [TN](improved/cases/TN.png)、[TP](improved/cases/TP.png)、[FN](improved/cases/FN.png)、[FP](improved/cases/FP.png) 同样各导出 3 张。用于人工记录的 [Baseline 复核表](baseline/case_review.csv) 与 [Layer4 复核表](improved/case_review.csv) 包含曝光、模糊、裁剪、采集差异及备注栏，初始留空；医学专家复核尚未完成。

## 可追溯性与边界

- 两组训练清单 SHA-256：`291305837f671a6c50cf6263c2562614ee12198812f3b629df27846d48bbc3ba`。
- ImageNet 权重 SHA-256：`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。
- 各实验的配置、环境、训练/验证数据指纹与 checkpoint 摘要保存在 `training_summary.json`；test 指纹与完整指标保存在 `metrics.json`。
- 自动测试覆盖数据隔离、抽样、指标和冻结训练行为。图表已进行工程显示检查，但没有进行诊断性审阅。
- 数据来源许可、患者级隔离、AP/PA 配对信息、外部验证和概率校准尚不具备。本项目只证明算法工程流程可运行，不用于患者决策。
