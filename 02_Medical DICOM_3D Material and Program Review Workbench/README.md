# 胸部 X 光肺炎异常检测：小样本迁移学习

根据 [DESIGN.md](DESIGN.md) 实现的可复现实验项目。使用 ImageNet 预训练 ResNet18 对胸片进行 `NORMAL / PNEUMONIA` 图像级二分类，提供数据审计、固定小样本抽取、CPU 训练、独立测试、案例检查、单图推理和实验对比。

**本项目只用于算法研发、工程验证和课程展示，不用于临床诊断、排除肺炎、治疗或患者分诊。** 文件夹名称沿用工作台命名；本次设计与实现的输入是二维胸片图片，未实现 DICOM 查看器或 3D 重建。

## 1. 已完成的实际实验

2026-09-07 在 macOS arm64、Python 3.14.6、PyTorch 2.14.0、torchvision 0.29.0 上，使用 CPU、4 个计算线程，完成两组预先指定的 3 epoch 实验。两组使用同一份 600 张训练清单、相同预处理与随机种子 42；测试集共 624 张，没有抽样缩减。

| 实验 | 可训练层 | 最佳 epoch | Sensitivity | ROC-AUC | Specificity | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | fc | 3 | 0.9385 | 0.9373 | 0.6880 | 0.8446 |
| Layer4 对照 | layer4 + fc | 2 | 0.9872 | 0.9420 | 0.6538 | 0.8622 |

| 实验 | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 161 | 73 | 24 | 366 |
| Layer4 对照 | 153 | 81 | 5 | 385 |

解冻 layer4 后，本次实验的 Sensitivity 增加 **4.87 个百分点**，ROC-AUC 增加 **0.0046**，同时 Specificity 下降 **3.42 个百分点**，误报从 73 增至 81。结果体现漏报与误报的取舍，不能据此宣称临床效果或统计显著性。验证集仅有 16 张，模型选择具有较大不确定性。

可直接查看 [实验对比 CSV](outputs/comparison.csv)、[完整对比 JSON](outputs/comparison.json)、[实验与案例说明](outputs/RESULTS.md)。这些数值来自实际运行，完整精度保存在各实验的 `metrics.json` 中。

## 2. 数据位置与来源

数据位于此项目文件夹**外部**，默认路径为 `../chest_xray`：

```text
saint/
├── chest_xray/
│   ├── train/{NORMAL,PNEUMONIA}/
│   ├── val/{NORMAL,PNEUMONIA}/
│   ├── test/{NORMAL,PNEUMONIA}/
│   ├── chest_xray/                 # 解压后的嵌套副本，忽略
│   └── __MACOSX/                   # 归档元数据，忽略
└── 02_Medical DICOM_3D Material and Program Review Workbench/
    ├── configs/
    ├── src/
    └── outputs/
```

不移动、复制或重新划分原始图片，只保存数据集相对路径。读取每个类别目录的直接子文件，支持 JPEG、PNG、BMP、TIFF；忽略隐藏文件、嵌套副本及不支持的扩展名。

实际审计结果：

| 原始划分 | NORMAL | PNEUMONIA | 总数 | 无法读取 |
| --- | ---: | ---: | ---: | ---: |
| train | 1,341 | 3,875 | 5,216 | 0 |
| val | 8 | 8 | 16 | 0 |
| test | 234 | 390 | 624 | 0 |
| 合计 | 1,583 | 4,273 | 5,856 | 0 |

训练仅使用 train 中每类 300 张。验证和测试保留原始全部可读图像。文件哈希与解码后的 RGB 像素哈希均发现 30 组重复图像，跨划分完全重复组数均为 0。重复图像只记录、不自动删除；这项检查不能发现所有近似重复，也不能证明患者级隔离。

设计文件将数据描述为公开、脱敏胸片，但当前没有实际下载记录、原始发布方、许可及独立脱敏核验材料。请依据真实来源补充 [configs/data_source.yaml](configs/data_source.yaml)；程序不会根据目录名或文件名推测来源或许可。

## 3. 环境安装

以下命令在本项目目录执行：

```bash
cd "02_Medical DICOM_3D Material and Program Review Workbench"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

可在兼容的 Python/PyTorch 平台运行，默认不要求 GPU。[requirements-lock.txt](requirements-lock.txt) 记录了本次实际安装的精确依赖；复用本次环境版本可执行 `python -m pip install -r requirements-lock.txt`。其他操作系统应选择其支持的 PyTorch 安装包。

首次训练需要 ImageNet 权重 `resnet18-f37072fd.pth`，会从 torchvision 官方权重地址下载到本项目 `.torch/checkpoints/`。也可在配置的 `pretrained_weights` 填入该官方文件的本地路径；程序检查 SHA-256 前缀并记录完整摘要。本次复用了相邻项目已有的官方权重缓存，通过校验后完成训练。

权重下载失败时训练会明确报错，**不会静默改用随机权重**。保存的 `best.pt` 已包含完整网络参数，评估和单图推理无需联网。

## 4. 运行方式

### 一键重新运行

当前目录已包含本次实际产物，建议用新输出目录重跑：

```bash
python src/run_pipeline.py --run-dir outputs/run_02
```

执行顺序是：全量数据审计 → 固定训练子集 → baseline 训练 → layer4 对照训练 → 两组 test 评估 → 对比。两组的超参数预先确定，最佳模型仅按 **val 平均交叉熵最小值** 选择；同值保留较早 epoch。

在没有既有实验产物的新副本中，也可直接执行 `python src/run_pipeline.py`，写入 `outputs/baseline`、`outputs/improved` 和 `outputs/audit`。程序拒绝覆盖已有训练记录或权重，避免混合两次实验；请更换 `--run-dir` 或配置中的 `output_dir`。

数据放在其他位置时：

```bash
python src/run_pipeline.py --data-root /absolute/path/chest_xray --run-dir outputs/run_03
```

**所有配置文件名、配置内路径和命令行相对路径都以本项目目录为基准**，不受 shell 当前工作目录影响。绝对路径保持原意。

### 分步骤执行

以下训练命令适用于尚未生成对应输出的新实验；已有本次权重时，可直接从评估步骤开始。

```bash
# 审计原始三个划分，输出数量、尺寸、读取错误、重复检查和示例图
python src/check_dataset.py

# 生成或校验复用 configs/train_subset.csv
python src/prepare_train_subset.py

# 两组训练：使用同一份固定训练清单
python src/train.py --config configs/baseline.yaml
python src/train.py --config configs/improved.yaml

# 分别对完整原始 test 评估
python src/evaluate.py --config configs/baseline.yaml
python src/evaluate.py --config configs/improved.yaml

# 校验数据指纹与关键实验设置一致后对比
python src/compare.py
```

单独重训时可以指定新目录，然后将该权重及输出目录传给评估：

```bash
python src/train.py --output-dir outputs/baseline_run_02
python src/evaluate.py --checkpoint outputs/baseline_run_02/best.pt --output-dir outputs/baseline_run_02
```

### 单张图片推理

```bash
python src/predict.py --image ../chest_xray/test/NORMAL/IM-0001-0001.jpeg --checkpoint outputs/baseline/best.pt --output outputs/single_prediction.json
```

输出 `predicted_label`、`p_normal`、`p_pneumonia`，无须 AP/PA 信息。本次示例的肺炎概率约为 0.5509，预测为 `PNEUMONIA`，而目录标签是 `NORMAL`，是一个实际误报案例。概率未经临床校准，不能当作患病风险。图像预处理参数从 checkpoint 读取。

## 5. 配置与可复现性

| 配置项 | 默认值 / 含义 |
| --- | --- |
| `data_root` | `../chest_xray`，外部数据集根目录 |
| `subset_path` | `configs/train_subset.csv`，两组共用 |
| `train_samples_per_class` | 300，可降低到 200 |
| `image_size` | 224，转换 RGB 后直接缩放到 224×224 |
| `batch_size` / `epochs` | 16 / 3 |
| `learning_rate` | fc 的 Adam 学习率 0.001 |
| `backbone_learning_rate` | 对照组 layer4 的 Adam 学习率 0.0001 |
| `freeze_backbone` | baseline 为 true；对照组为 false，仅额外解冻 layer4 |
| `random_seed` | 42，同时设置 Python、NumPy、PyTorch、DataLoader 随机源 |
| `device` | cpu；也支持 auto、cuda、mps，不可用时明确报错 |
| `torch_threads` / `num_workers` | 4 / 0 |
| `metadata_csv` | null，可选可靠的图片—患者/检查配对表 |
| `pretrained_weights` | null，使用本地缓存或下载官方 ImageNet 权重 |

两组均使用 `ResNet18 → Linear(512, 2)`、无额外图像增强、ImageNet 标准化均值 `(0.485, 0.456, 0.406)` 和标准差 `(0.229, 0.224, 0.225)`、无类别权重的交叉熵。灰度图先复制成三通道。类别顺序固定为 `NORMAL=0, PNEUMONIA=1`；经 Softmax 后选择较高概率的类别，相等时选择 NORMAL。

Baseline 只有 **1,026** 个可训练参数，主干保持 eval 模式，BatchNorm 运行统计也冻结。对照组有 **8,394,754** 个可训练参数，只有 layer4 和 fc 处于训练模式，其余层及其 BatchNorm 统计保持冻结。

训练子集按排序后的原始路径、固定随机种子无放回等量抽取，保存 `path,label` CSV。已有清单会校验并复用；校验包括类别数量、目录标签一致性、训练划分、重复路径、越界路径和可读性。改变样本数量时，建议给两组配置指定同一个新 `subset_path`；如确需替换原清单，显式使用 `python src/prepare_train_subset.py --overwrite`。

每个实验保存自己的训练清单副本、实际配置、环境版本、权重摘要、训练/验证数据内容摘要和选模依据。评估记录 test 内容摘要。对比工具会拒绝训练清单、图像内容、预训练权重或关键设置不一致的实验。确定性设置便于在相同环境复现，不保证不同硬件或依赖版本逐位一致。

## 6. 产物与代码结构

```text
configs/
  baseline.yaml / improved.yaml    # 两组实验配置
  train_subset.csv / .json         # 600 张相对路径及抽样摘要
  data_source.yaml                 # 实际来源与许可信息待补充
src/
  common.py                       # 路径、配置、随机种子、摘要
  check_dataset.py                # 读取、尺寸、重复及可选患者 ID 审计
  prepare_train_subset.py          # 仅抽取 train，保存固定清单
  dataset.py / model.py            # 预处理、模型与冻结策略
  train.py / evaluate.py           # 训练选模与独立测试
  metrics.py / plots.py            # 指标与无界面 PNG 图表
  predict.py / compare.py          # 单图推理与公平比较
  run_pipeline.py                  # 一键运行
tests/test_pipeline.py             # 数据与训练行为回归测试
outputs/
  audit/
    dataset_stats.csv             # 六组类别统计
    image_inventory.csv           # 每图尺寸、模式、文件/像素摘要
    dataset_summary.json          # 审计汇总与患者隔离限制
    read_errors.json / duplicates.json
    samples.png
  baseline/                       # improved/ 具有相同结构
    best.pt                       # 以 val 损失选出的完整模型
    resolved_config.json
    train_subset.csv
    training_summary.json / history.csv / training_curve.png
    validation_read_errors.json / test_read_errors.json
    metrics.json / predictions.csv
    confusion_matrix.png / roc_curve.png
    cases/{TN,TP,FN,FP}.png        # 每类最多 3 张，无案例则标注空类别
    case_review.csv               # 人工质量复核记录表
  comparison.csv / comparison.json
  single_prediction.json
  RESULTS.md                      # 本次固定实验说明，不由后续重跑自动覆盖
```

`predictions.csv` 字段是 `image_path,true_label,predicted_label,p_normal,p_pneumonia`。混淆矩阵的行是真实类别，列是预测类别，顺序均为 NORMAL、PNEUMONIA。Sensitivity 为 `TP/(TP+FN)`；ROC-AUC 使用连续的 `p_pneumonia`。测试数据缺少某类时，不可定义指标写为 JSON `null`，不伪造为 0。

默认案例抽样每类最多 3 张，固定随机种子；FN 表示真实 PNEUMONIA 预测成 NORMAL，FP 表示真实 NORMAL 预测成 PNEUMONIA。`case_review.csv` 留有模糊、曝光、裁剪、采集差异和备注列；同一评估目录重跑时保留已选图片的已有备注。图表标题使用英文以避免无中文字体时缺字。

模型权重、PNG、环境和缓存已在项目 `.gitignore` 中排除，当前本地文件仍可查看。CSV/JSON、代码、测试及文档可纳入版本控制；不提交原始影像。仅从 Git 获取代码时，需要运行脚本重新生成被忽略的图表和模型。

## 7. 数据质量、患者配对与限制

- 图像损坏会在审计中记录。抽样时排除不可读 train 图像；已经进入固定清单的图像若随后损坏，训练报错，避免悄悄改变训练样本。val/test 不可读图像单独记录排除路径及总数，本次均为 0。
- 患者级隔离当前为 **unverified**。文件名中的 `person...` 等字符串不自动作为已核验患者 ID；缺少可靠配对资料时，沿用原划分并明确限制。
- 若获得可靠元数据，可提供 UTF-8 CSV：`path,patient_id,study_id,view_position`。`path` 必须是数据集相对路径；ID 可缺失，体位允许 AP、PA 或空值。设置 `metadata_csv` 后重新执行审计，会检查路径唯一性、ID 覆盖率及患者/检查跨集合重复。仅检查 study_id 不能证明 patient_id 隔离。
- AP/PA 只用于数据审计，当前未作为网络输入；不从 JPG 外观臆造该信息。
- 原数据中存在重复图像，本实验不做患者重划分或去重扩展实验。验证集过小、训练样本有限、标签噪声和来源差异都限制结果解释。
- 224×224 直接缩放可能改变纵横比；当前遵循设计的简单预处理，未验证替代缩放策略。
- 未进行外部验证、概率校准、置信区间或临床专家审阅。当前结果不能推广为真实医院性能。

## 8. 验证与 AI 辅助说明

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

本次已通过 10 项回归测试，覆盖固定抽样、坏图排除、训练/验证/测试隔离、路径越界、标签错误、患者元数据重叠、指标边界、冻结层参数及 BatchNorm 行为、实验对比一致性。还实际运行了全量审计、两组训练、两组完整 test 推理、单图预测，并检查了输出图表。

另在临时目录用合成图片完成一键流程冒烟验证，确认新运行目录、两组训练、评估、对比及覆盖保护串联可用；合成图片的指标不进入正式实验结果。最终核对了两份各 624 行预测的概率、混淆矩阵、checkpoint 选择和文档链接。

AI 用于代码实现、测试、文档整理及图表的工程检查。成功/失败案例已导出供人工审阅；CSV 的质量复核字段初始为空，不表示已由医学专家检查。AI 不修改原始标签，不替代数据许可与脱敏核验，也不对图像给出诊断结论。
