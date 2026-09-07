# 胸部 X 光二分类：数据审计、固定特征基线与多模态扩展设计

已依据 `design.md` 完成可运行实现，并于 **2026-09-07** 在外部 `chest_xray` 数据集上完成全量审计、清单锁定、CPU 训练、验证、测试和单图推理。真实测试集 **ROC-AUC=0.941922，PNEUMONIA 召回率=0.979328**（固定阈值 0.5）。患者级独立性仍未核实，结果仅用于研发与工程验证，不用于诊断、治疗或临床决策。

`NORMAL=0`、`PNEUMONIA=1` 是数据集标签；NORMAL 不代表排除所有疾病，预测分数不是经校准的真实患病概率。当前没有真实配对辅助信息，实际模型只使用图像；多模态部分按设计保留为扩展方案，不虚构症状或实验增益。

## 1. 已完成内容

| 内容 | 实现与证据 |
|---|---|
| 全量只读数据审计 | `inspect_chest_xray.py`；`reports/audit/` |
| 排除重复与候选组重叠、锁定清单 | `prepare_manifest.py`；`reports/splits/` |
| 预处理与冻结的 ImageNet ResNet18 特征提取 | `xray/model.py`；512 维图像特征 |
| 训练集标准化与 balanced 逻辑回归 | `run_baseline.py`；`reports/baseline/classifier.npz` |
| 独立于拟合步骤的验证与最终测试 | 样本级预测、指标、模型锁文件与运行日志 |
| 成功、漏报、误报样例复核 | `reports/baseline/error_analysis.md`、`cases.csv`、本地 `cases.png` |
| 离线单图预测 | `predict.py`；已验证 TP/TN/FP/FN 四类样例的类别一致；分数有批大小导致的浮点差异 |
| 工程验证 | 16 项自动化测试通过；模型哈希、仅训练集拟合、指标重算均通过 |
| 多模态接入、配对与缺失处理方案 | `design.md` 第 4 节；尚无配对数据，不进行虚构实验 |

## 2. 目录与外部数据

```text
saint/
├── chest_xray/                         # 外部数据，不复制进项目、不修改
│   ├── chest_xray/                     # 唯一训练数据根目录
│   │   ├── train/{NORMAL,PNEUMONIA}/
│   │   ├── val/{NORMAL,PNEUMONIA}/
│   │   └── test/{NORMAL,PNEUMONIA}/
│   ├── train/、val/、test/             # 另一套副本，仅作比较
│   └── __MACOSX/                       # 不参与读取
└── 10_Medical imaging algorithm and multi-modal scheme design/
    ├── design.md                       # 原始方案与实施状态说明
    ├── readme.md
    ├── data_source.json                # 实际来源记录；未知字段明确为 null
    ├── requirements-audit.txt          # 仅审计：Pillow
    ├── requirements.txt                # 完整实现的兼容依赖范围
    ├── requirements-lock.txt           # 本次实际环境的运行依赖版本
    ├── inspect_chest_xray.py
    ├── prepare_manifest.py
    ├── run_baseline.py
    ├── predict.py
    ├── verify_run.py
    ├── xray/                          # 审计、划分、模型、评估、训练模块
    ├── tests/                         # test_pipeline.py、test_verification.py
    └── reports/
        ├── audit/
        ├── splits/
        ├── baseline/
        └── tests.txt
```

命令均从项目目录执行，数据路径为 `../chest_xray/chest_xray`，不是项目内的 `./chest_xray`。绝不从最外层递归汇总图片，避免把副本重复加入模型。

## 3. 安装与复现

### 环境

完整流程使用 Python **3.11+**；本次实测环境为 Python 3.14.6、macOS arm64、CPU 4 线程，PyTorch 2.14.0、torchvision 0.29.0、scikit-learn 1.9.0、Pillow 12.3.0、NumPy 2.5.3。其他系统请安装对应平台可用的兼容 PyTorch/torchvision 组合。

```bash
cd "10_Medical imaging algorithm and multi-modal scheme design"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

本机项目目录的 `.venv` 已按锁定版本安装完成，可直接运行下方命令；首次克隆或迁移到其他机器时再执行安装步骤。

同平台复现本次依赖版本可改用 `-r requirements-lock.txt`。只运行审计则安装 `-r requirements-audit.txt` 即可，不需要 PyTorch。审计本身不联网；首次安装依赖或没有本地预训练权重时需要联网。

### 顺序执行完整流程

已交付结果保存在 `reports/audit`、`reports/splits`、`reports/baseline`，下面使用新目录，避免覆盖真实记录：

```bash
.venv/bin/python inspect_chest_xray.py \
  --data-root "../chest_xray/chest_xray" \
  --compare-root "../chest_xray" \
  --out reports/audit_reproduce

.venv/bin/python prepare_manifest.py \
  --audit reports/audit_reproduce \
  --out reports/splits_reproduce

.venv/bin/python run_baseline.py \
  --manifest-dir reports/splits_reproduce \
  --out reports/baseline_reproduce \
  --weights reports/baseline/backbone.pth \
  --device cpu --threads 4 --batch-size 32

.venv/bin/python verify_run.py --run reports/baseline_reproduce
.venv/bin/python -m unittest discover -s tests -v
```

本地交付包含 `reports/baseline/backbone.pth`，可以离线复用。Git 默认不追踪权重；如果从 Git 获取项目后没有该文件，训练时省略 `--weights`，torchvision 会下载明确指定的 `IMAGENET1K_V1` 权重。也可以传入官方 `resnet18-f37072fd.pth`，程序检查其 SHA-256 前缀并记录完整校验值。可用 `TORCH_HOME` 指定下载缓存目录。

所有报告输出目录必须是新目录，且不得与输入数据目录重叠；不会覆盖已有输出。失败可能留下不完整目录，只有出现 `run_summary.json` 且 `status=complete` 才表示整次训练与评估完成；重跑请另选目录。每个入口支持 `--help`。

Windows PowerShell 使用 `python -m venv .venv`、`.\.venv\Scripts\python.exe`，将多行命令合成一行执行。默认 CPU；Apple Silicon 可选 `--device mps`，但本次正式记录来自 CPU。不同设备、批大小和依赖版本可能造成浮点差异，不保证逐位相同。

### 单张图片预测

```bash
.venv/bin/python predict.py \
  --run reports/baseline \
  --image "../chest_xray/chest_xray/test/NORMAL/IM-0001-0001.jpeg"
```

输出 JSON 含 `image`、`pneumonia_score`、`threshold`、`predicted_label`、`aux_available`。上述真实样例预测为 PNEUMONIA，分数约 0.96724，是一个**误报示例**。该样例的单图与批量推理分数差约 3.13e-6，见 `inference_check.json`；这不是所有图片的误差上限。中断复查另外检查 TP/TN/FP/FN 各一张，类别全部一致，最大分数差约 2.69e-5。同样的 32 张批次复算可精确重现原有特征，差异来自批大小变化下的浮点计算，详见 `reports/review/batch_precision_check.json`。推理使用已保存权重、数值分类器和锁定配置，不重新训练、调整阈值或联网下载。

## 4. 数据检查与清单处理

### 本地真实数量

| 集合 | 原始 NORMAL | 原始 PNEUMONIA | 原始合计 | 保留 NORMAL | 保留 PNEUMONIA | 实验合计 |
|---|---:|---:|---:|---:|---:|---:|
| train | 1,341 | 3,875 | 5,216 | 1,340 | 3,397 | 4,737 |
| val | 8 | 8 | 16 | 8 | 8 | 16 |
| test | 234 | 390 | 624 | 231 | 387 | 618 |
| 合计 | 1,583 | 4,273 | 5,856 | 1,579 | 3,792 | 5,371 |

依据 `reports/audit/summary.json` 与 `reports/splits/split_summary.json`：

- 5,856 张均可完整解码；L 模式 5,573 张、RGB 283 张。宽度 384–2916，高度 127–2713。
- 4 个隐藏条目被忽略；扫描限于六个类别目录内直接出现的 JPEG/PNG，不读取符号链接或子目录。
- 字节重复和 RGB 像素重复各 30 组；二者可能表示同一批图片，不能相加计数。无跨集合完全重复、无完全重复图片的标签冲突。
- 170 个候选文件名前缀跨集合出现；所有文件名匹配当前规则。前缀不是经过核实的患者编号。
- 内外目录在本次支持图片扫描范围内的相对路径与字节完全一致，不代表整个目录树一致。

### 实验前固定的处理规则

1. 不可读、多帧、非 L/RGB、缺失哈希的样本从实验清单排除。
2. 字节或像素哈希通过连通分组处理；任何完全重复组有标签冲突时整组隔离，不自行改标签。
3. 无冲突完全重复组只保留一张，优先级 `test > val > train`，同集合按相对路径排序。
4. 对剩余样本按候选前缀隔离；跨集合的前缀只保留最高优先级集合中的样本，不将它们移动到其他集合。
5. 没有重新随机划分、没有按模型分数排除样本、没有把测试图片转为训练图片。

本次排除 **32 张完全重复副本 + 453 张与测试集候选前缀重叠的开发图片 = 485 张**。测试集仅因内部去重由 624 变为 618；验证集仍为 16 张。每张原始图片的 `included`、`exclusion_reason`、`final_split` 见 `selection_records.csv`。

最终清单无相同字节、相同解码像素或候选前缀跨集合重叠；**`patient_level_isolation_verified` 仍为 `false`**。文件名冲突、跨命名空间关联、近似重复和不同图片属于同一患者等问题未被排除，因此指标只能称为“启发式隔离下的有限流程实验”，不能称为严格患者独立评估。

### 报告字段

`manifest.csv` 保留设计要求的 `relative_path`、`split`、`label`、`label_id`、`width`、`height`、`mode`、`format`、`frame_count`、`readable`、`file_sha256`、`rgb_pixel_sha256`、`candidate_group_id`、`group_source`、`patient_id_verified`。

像素哈希在原始 L/RGB 模式完整解码后转 RGB，结合宽高计算；没有执行模型预处理。其他模式不静默压缩混入训练。`readable` 只表示当前解码和单帧检查，不代表医学质量合格。

候选规则支持 `person<数字>_virus/bacteria_<数字>`、`IM-<数字>-…`、`NORMAL2-IM-<数字>-…`，保留独立命名空间，不含 split；未知名字标为 `unmatched`。审计只记录问题，不自动删除或移动原图。

| 审计文件 | 用途 |
|---|---|
| `summary.json` | 数量、模式、尺寸范围、异常和重复统计、环境及清单哈希 |
| `manifest.csv` | 原始逐图清单 |
| `issues.csv` | 隐藏、忽略条目、哈希或解码异常、模式问题 |
| `duplicates.csv` | 两类哈希的重复明细与跨集合／标签冲突标记 |
| `candidate_group_overlaps.csv` | 候选前缀跨集合出现的全部成员 |
| `root_comparison.json` | 仅传入 `--compare-root` 时输出；只比较扫描范围内图片 |

## 5. 模型、复现约束与真实结果

处理链为：EXIF 方向处理 → 灰度 → 等比例双线性缩放 → 居中黑色补边至 224×224 → 灰度复制为 RGB → ImageNet 均值／标准差归一化 → 冻结的 ResNet18 → 512 维特征 → StandardScaler → 逻辑回归。

不叠加 torchvision 默认中心裁剪，不做随机增强。权重固定为 `ResNet18_Weights.IMAGENET1K_V1`，移除原 ImageNet 分类头，设为 eval 并禁止参数梯度。均值 `[0.485,0.456,0.406]`、标准差 `[0.229,0.224,0.225]`。几何预处理是本项目约定，并非预训练权重默认整套变换。

标准化器和分类器仅在训练特征上拟合；逻辑回归 `C=1.0`、`class_weight="balanced"`、`solver="lbfgs"`、`max_iter=3000`、种子 42。阈值在评估前固定为 0.5，没有参数搜索或阈值优化。验证预测保存后写入 `model_lock.json`，再提取测试特征并评估。路径、文件名、前缀、标签字段从不进入特征向量。

| 集合 | 数量 | ROC-AUC | PNEUMONIA 召回率 | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| 验证 | 16 | 1.000000 | 1.000000 | 8 | 0 | 8 | 0 |
| 测试 | 618 | 0.941922 | 0.979328 | 379 | 121 | 110 | 8 |

**121 张 NORMAL 被误报**，较高召回率不表示整体结果已经适合使用。验证集只有 16 张，其满分不能说明稳健性；没有可靠患者标识，未另外划验证集，也不编造患者级置信区间。ROC-AUC 对单类集合输出 `null` 并说明原因；没有正类时召回率也为 `null`，不会伪填 0 或 1。

本次基线命令约用时 **84.17 秒**，包含特征提取、分类器拟合和结果导出，不含此前审计、环境准备与权重下载；逻辑回归 66 次迭代收敛。这是本机一次实测，不是运行时承诺。

### 基线输出

| 文件 | 说明 |
|---|---|
| `config.json` | 数据根目录、清单哈希、设备、依赖、种子、预处理、分类器、阈值、案例规则 |
| `manifest.csv` | 本次锁定的实际输入，保留原始 split |
| `backbone.pth`、`classifier.npz` | 离线推理所需权重、标准化均值与尺度、逻辑回归系数与截距 |
| `model_lock.json` | 最终测试前保存的配置、清单、分类器和权重校验值 |
| `features.npz` | 按清单各 split 原顺序保存的 512 维特征，无模拟特征 |
| `validation_predictions.csv`、`test_predictions.csv` | 真实标签、连续分数、阈值、预测标签、TP/TN/FP/FN、`aux_available=false` |
| `validation_metrics.json`、`test_metrics.json` | 两项主要指标和混淆计数 |
| `run_summary.json`、`run.log` | 完成状态、用时、迭代数和实际运行输出 |
| `cases.csv`、`cases.png`、`case_geometry.csv`、`error_analysis.md` | 12 个真实预测案例与原图／处理图的工程复核 |
| `verification.json`、`inference_check.json` | 结果重算、训练集统计校验和单图推理一致性证据 |

训练前检查清单校验值，每张图片在提取特征前重新检查文件哈希；移动数据后可给训练命令传 `--data-root`，但图片内容必须与清单一致。推理校验模型锁，分类器使用 `allow_pickle=False` 的数值文件，PyTorch 权重使用 `weights_only=True` 加载。哈希用于发现意外变化，不是对任意来源模型文件的安全认证。

Git 默认忽略原始数据、权重、特征和案例图片；本地仍保留这些产物，约 60 MB。不要把未核实许可的图片重新发布。仅从 Git 克隆时，按上述流程重新生成被忽略的大文件。

## 6. 测试与可信范围

`python -m unittest discover -s tests -v`：16 项测试通过，覆盖坏图、多帧、高位深、隐藏与符号链接、跨集合哈希重复、传递标签冲突、候选前缀命名空间、确定性隔离、EXIF 方向与完整视野补边、未定义指标、输出覆盖／路径越界、数据变更，以及分类器数值序列化。

`verify_run.py` 在已保存真实实验上确认：完成标记和运行汇总一致、模型锁哈希一致、清单标签与数量正确、无已知重复／候选前缀重叠、特征及分类器数组维度正确且数值有限、标准化器均值和尺度等于**训练特征**统计、验证与测试的全部预测字段及指标可逐项重算。新增 6 项回归测试覆盖缺失完成标记、错误运行状态／汇总、CSV 类别／阈值等元数据错误、特征数量／非有限数值与阈值不一致。另已对 12 组原图／处理图作视觉检查并记录局限。合成测试图片仅用于工程测试，不参与训练、指标或案例展示。

本次测试后的案例分析没有用于重训或调整参数。若后续据此改进，必须说明该测试集已用于开发反馈。当前未进行临床验证、患者身份恢复、病灶定位、分割、报告生成或真实多模态实验。

中断后的完整复查与修复记录见 `reports/review/completion_review.md`，最新 16 项测试日志见 `reports/review/tests.txt`；原始 `reports/tests.txt` 保留首次 10 项测试的历史记录。已核对 5,856 张原图哈希、所有原始源码哈希和报告可读性，未发现交付缺失或实际模型／数据损坏。本次只增强校验与修正文档，没有重训、改阈值或改动原始实验指标。

## 7. 多模态扩展与缺失处理

设计使用检查申请时已经存在的真实结构化症状（如 fever、cough）及缺失标记，以可信匿名患者 ID、exam_id 和时间关系完成配对；标识与时间只用于配对和泄漏检查，不作为预测特征。只使用预测时点之前的记录，不用事后诊断或文件名中的 virus/bacteria。

有可靠配对数据后，将症状编码与图像特征拼接，编码／填补／标准化只在训练部分拟合，并在相同患者、图像、划分、覆盖范围上比较图像与融合模型。无辅助信息或配对不可靠时使用独立图像基线，部分缺失明确编码，不把空值视为“没有症状”。完整方案见 `design.md` 第 4 节。本次所有真实预测均为 `aux_available=false`。

## 8. 来源记录、未知事项与参考

用户提供本地 `chest_xray` 图片包，并说明已脱敏、没有配对辅助信息。本次审计确认了本地文件数量和内容比较；**实际下载页面、下载日期、压缩包版本与校验值、许可与署名依据、原始患者／检查划分说明仍无法从文件目录独立确认**，已在 `data_source.json` 明确保留未知状态。这些是需外部证据才能完成的数据来源事项，不以工程测试结果替代。

原文提供的 [Kaggle 数据集入口](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) 与 [Mendeley Data 参考入口](https://data.mendeley.com/datasets/rscbjbr9sj/3) 仅作为核对线索，不冒充当前压缩包下载记录，也不直接套用其他版本许可。脚本不证明脱敏充分性或医学标签正确性，不识别所有近似副本。

实现参考：[torchvision ResNet18 权重与归一化说明](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)、[scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)。原始方案依据与题目背景保留在 `design.md`；项目目录未提供题目原文件，因此不声称重新核对了缺失的题目附件。

AI 参与代码、方案落地、测试和文档整理。已通过真实数据运行与工程验证核对结果，仍未完成的数据来源核实、患者独立性证明和多模态实验均如实保留状态。
