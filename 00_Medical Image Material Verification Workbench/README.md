# 胸片肺炎标签二分类：400 张小样本研发方案

本项目选择作业中的**检测方向（影像级二分类）**，输出 NORMAL / PNEUMONIA 数据集类别及模型分数。没有病灶框或分割标注，因此不做定位或分割。已提供可独立查看的中文方案、真实抽样图片、数据审计和 CPU 基线；仅用于课程讨论与研发验证，不输出诊断、治疗建议或临床结论。

## 先看哪些材料

1. [DESIGN.md](DESIGN.md)：按任务、数据、模型与辅助信息、验证和风险展开。
2. [reports/RESULTS.md](reports/RESULTS.md)：本机实际运行结果与成功/失败样本讨论。
3. [data/chest_xray_small](../data/chest_xray_small)：400 张原尺寸图片副本及可追溯清单 `manifest.csv`。
4. [reports/data_audit.json](reports/data_audit.json)：全量读取、重复检查、抽样数量和泄漏检查。
5. [reports/test_examples.png](../reports/test_examples.png)：测试集 TP/TN/FP/FN 示例，每类最多 4 张；含义以数据集标签为准。

## 数据来源、许可与获取

公开数据为 Kermany 等人的儿科胸部 X 光数据，常见分发页是 [Kaggle — Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)。原始发布者为 Daniel Kermany、Kang Zhang、Michael Goldbaum，原始存档见 [Mendeley v2](https://data.mendeley.com/datasets/rscbjbr9sj/2) 及 [v3](https://data.mendeley.com/datasets/rscbjbr9sj/3)。数据集论文：Kermany et al., *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*, Cell, 2018，[DOI](https://doi.org/10.1016/j.cell.2018.02.010)。

2026-09-07 核查：Mendeley v2/v3 均标注 **CC BY 4.0**；v3 描述另外写有 “research only”，本项目限定教学研发使用，并保留两项原始说明。使用或分享数据时保留作者署名、来源、[许可链接](https://creativecommons.org/licenses/by/4.0/)和改动说明，不暗示原作者背书。本项目只抽取并重命名副本；图片字节没有修改，可视化另做缩放，模型输入另做灰度化和缩放。

本地 `chest_xray/` 的目录、命名和 5,856 张图片计数与常见 Kaggle 包一致，视为该公开数据的本地副本；**目录中没有下载凭证或许可文件，未与发布端逐文件验真**。上述来源匹配是依据结构的判断，提交时应附你原先的下载网址/记录，不把本地哈希称作发布者提供的官方校验值。公开版本使用匿名化编号，本项目不接入私人病例或可识别信息；未对每张图完成烧录文字去标识化审查，外发图片前须逐张复核。

已有本地数据可直接运行，无需再次下载。其他人复现可从上述公开页面按当时的使用条件下载胸片部分，解压成 `../chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}`；不要下载同一存档的 OCT 部分。若改用 v3 的不同文件名，应先更新患者分组解析规则，当前脚本遇到未知规则会停止。

## 项目目录

```text
saint/
├── chest_xray/                  # 用户原数据，仅被读取
├── data/chest_xray_small/
│   ├── train/{NORMAL,PNEUMONIA}/ # 每类 120 张
│   ├── val/{NORMAL,PNEUMONIA}/   # 每类 40 张
│   ├── test/{NORMAL,PNEUMONIA}/  # 每类 40 张
│   └── manifest.csv
├── reports/
│   └── test_examples.png         # 图片保留在原位置
└── 00_Medical Image Material Verification Workbench/
    ├── README.md / DESIGN.md / config.json / requirements.txt
    ├── scripts/                  # 全量审计、抽样与 CPU 基线
    ├── tests/test_pipeline.py    # 泄漏、指标和优化器测试
    ├── deliverables/胸片分类项目详细说明.docx
    └── reports/
        ├── RESULTS.md
        ├── source_inventory.csv  # 只扫描顶层规范目录的全量清单
        ├── read_errors.json / data_audit.json
        └── baseline_metrics.json / baseline_model.npz / predictions.csv
```

## 如何运行

需要 Python 3.10+、NumPy 和 Pillow，无需 GPU、深度学习框架或预训练权重。在工作台目录执行：

```bash
cd "00_Medical Image Material Verification Workbench"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/prepare_subset.py
python scripts/run_baseline.py
python -m unittest discover -s tests -v
```

本项目已生成小样本；若只复核已有结果，跳过 `prepare_subset.py`，直接运行基线和测试。抽样脚本拒绝覆盖已有输出目录；要重新抽样，复制 `config.json`，为 `output` 和 `reports` 设置新的相对目录，再给两个脚本传入 `--config 新配置.json`。固定种子和不变源文件产生相同清单，不要为了更好分数反复换种子。

当前系统默认 Python 缺少依赖，本次实际使用 Codex 自带 Python：

```bash
/Users/hengning/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/run_baseline.py
```

运行时的 Python/NumPy/Pillow 版本保存在 `reports/baseline_metrics.json`。图片与小样本副本保留在工作台外，模型二进制位于 `reports/`；如作业需交小样本，可单独打包 `../data/chest_xray_small/` 并附本 README 中的来源署名。原数据未移动或删除。

## 已知限制与 AI 辅助说明

400 张为两类平衡的教学子集，不能代表现实患病率。患者级隔离只能核查文件名代理组和重复像素，无法证明真实患者/检查绝对独立；重编码、裁剪后的近重复也不一定能查出。16×16 像素逻辑回归只是工程基线。辅助体位信息全部缺失，多模态改进只有设计、没有实测收益。来源与标签也未完成临床复核。

AI 辅助了公开来源检索、方案组织、脚本编写和结果整理。可靠性判断依靠原始发布页与许可页、实际文件计数/解码/哈希、可重跑脚本、泄漏与指标测试，以及成功/失败图人工式视觉检查；不把 AI 的图像描述当作临床解释，不把未运行的模型或辅助信息扩展写成完成实验。
