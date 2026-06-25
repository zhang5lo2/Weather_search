# AGENTS.md

本文件是本项目的最终工作规范。修改代码、训练模型、记录实验或准备提交前，先按本文件检查。

## 项目目标

本项目用于 2026 睿抗 RAICOM / 智海 Mo 平台天气图片识别任务。目标是训练一个四分类图像模型，并通过 `main.py` 中的 `predict(X)` 完成平台推理。

任务标签固定为：

| 代码标签 | 含义 |
| --- | --- |
| `cloudy` | 多云 |
| `rainy` | 雨天 |
| `snowy` | 雪天 |
| `sunny` | 晴天 |

主指标为 `macro F1`。`accuracy` 只能作为辅助指标，不能单独用于判断模型优劣。

## 关键路径

```text
.
+-- main.py                         # Mo/比赛推理入口，必须暴露 predict(X)
+-- train.py                        # 本地训练入口
+-- evaluate.py                     # 固定验证集评估入口
+-- src/weather_classifier/
+-- data/train/                     # 本地训练图片，目录名即类别标签
+-- results/model_best.pth          # 默认正式模型权重
+-- docs/baselines.md               # baseline 与实验记录
+-- docs/competition.md             # 比赛约束摘要
+-- archive/original_submission/    # 原始脚本和 notebook，仅用于追溯
+-- environment.yml
+-- requirements.txt
```

当前本地训练集数量：

```text
cloudy  2184
rainy    446
snowy    403
sunny   1966
total   4999
```

`rainy` 和 `snowy` 样本明显更少，训练、验证和误差分析时必须关注少数类表现。

## 环境约束

项目使用 Conda 环境：

```powershell
conda env create -f environment.yml
conda activate raicom-weather
```

环境已存在时：

```powershell
conda env update -n raicom-weather -f environment.yml --prune
conda activate raicom-weather
```

比赛平台约束：

```text
torch 2.1.7
CPU 2 核
内存 8 GiB
整体 CPU 推理时间 70 分钟内
```

本地公开源未提供精确 `torch 2.1.7`，当前环境使用 PyTorch `2.1.x` / torchvision `0.16.x` 兼容线。正式提交前以 Mo 平台模板实际版本为准。

本地已验证核心版本：

```text
torch 2.1.2
torchvision 0.16.2
opencv-python 4.10.0
numpy 1.26.4
scikit-learn 1.9.0
```

注意：

- `opencv-python` 固定为 `4.10.0.84`，避免新版 OpenCV 拉高 numpy 到 2.x。
- 不使用 `archive/original_submission/requirements.txt` 重建环境；那是旧平台冻结依赖。
- 提交推理不能依赖联网下载模型、权重或数据。

## 数据与评估协议

比赛当前没有把官方测试集或评分集放到本地。本地阶段只允许从 `data/train/` 内部固定划分验证集，用于模型选择和实验比较。

固定本地验证协议：

```text
split: stratified train/validation
seed: 42
val_ratio: 0.15
train size: 4249
val size: 750
```

验证集类别数量：

```text
cloudy=328
rainy=67
snowy=60
sunny=295
```

执行原则：

- 没有固定验证集，不开始调参。
- 验证集只用于评估和选模，不参与训练更新。
- 不用训练集原图直接报告最终效果；那会高估模型。
- 官方测试/评分集只通过平台评测入口调用 `main.predict(X)`，本地无法提前得到最终分数。
- 如果后续使用 K-fold、多 seed 或全量重训，必须在 `docs/baselines.md` 记录，并说明与固定验证协议的差异。
- 如果人工修正标签或清理数据，应记录标准和样本列表；不要只清理模型预测错的图片，否则会引入偏差。

## Baseline 与当前默认模型

当前 baseline 记录在 `docs/baselines.md`：

```text
name: baseline_simple_cnn_2026-06-25
model: simple_cnn
image_size: 128
epochs: 10
batch_size: 32
device: CPU
checkpoint: results/baseline_simple_cnn_2026-06-25_model_best.pth
val_acc: 0.6253
val_macro_f1: 0.5838
val_weighted_f1: 0.6379
CPU speed: about 255 images/s
```

该 baseline 是最小可用方案，不是最终高分方案。主要问题是 `cloudy` 召回率低，许多 `cloudy` 被误判为 `rainy` 或 `snowy`。

当前默认正式权重已更新为当前最优模型：

```text
name: exp04_resnet18_224_2026-06-25
model: resnet18 pretrained
checkpoint: results/model_best.pth
val_acc: 0.9280
val_macro_f1: 0.9128
val_weighted_f1: 0.9281
CPU speed: about 88 images/s
```

后续实验必须回答：

- 改了什么？
- 是否比 baseline 的 `val_macro_f1=0.5838` 和当前默认模型的 `val_macro_f1=0.8576` 更好？
- 哪些类别的 precision/recall/F1 改善或变差？
- 错误类型是否减少？
- 是否值得保留到最终方案？

## 训练命令

快速烟测：

```powershell
conda run -n raicom-weather python train.py --data-dir data/train --output-dir results/smoke_test --epochs 1 --batch-size 4 --image-size 64 --limit-per-class 2 --val-ratio 0.5 --num-workers 0
```

复现当前 baseline：

```powershell
conda run -n raicom-weather python train.py --data-dir data/train --output-dir results --epochs 10 --batch-size 32 --image-size 128 --model simple_cnn --num-workers 0
```

评估当前 checkpoint：

```powershell
conda run -n raicom-weather python evaluate.py --data-dir data/train --checkpoint results/model_best.pth --batch-size 32 --num-workers 0 --device cpu
```

预训练 ResNet18 方向：

```powershell
conda run -n raicom-weather python train.py --data-dir data/train --epochs 15 --model resnet18 --pretrained --num-workers 0
```

训练要求：

- 以 `val_macro_f1` 保存和比较模型。
- 保留 class-weighted loss 或 balanced sampler 处理类别不均衡，除非实验明确验证替代方案。
- 每轮训练只改变一个主要因素，如模型、分辨率、增强、采样、损失函数或学习率。
- 训练可使用预训练权重，但最终提交包必须包含权重，推理时不能下载。
- 训练输出写入 `results/`；正式默认权重为 `results/model_best.pth`。
- 烟测输出如 `results/smoke_test/` 验证后应删除，避免误当正式模型。

## 实验迭代协议

训练模型的顺序是：先定义目标，再跑 baseline，再做误差分析，最后决定下一步。不要盲目堆模型或随机试参数。

每轮实验按以下流程执行：

1. 明确本轮要验证的问题。
2. 固定数据划分和主指标。
3. 只改变一个主要变量。
4. 训练模型并保存 checkpoint。
5. 在固定验证集上评估 `macro F1`、各类 precision/recall/F1 和混淆矩阵。
6. 抽取并查看错误样本，尤其关注 `cloudy`、`rainy`、`snowy` 的混淆。
7. 判断主要问题属于偏差、方差、数据不匹配、标签问题还是推理约束问题。
8. 在 `docs/baselines.md` 或新的实验记录中写明命令、指标、结论和下一步。

最小实验记录字段：

| 实验名 | 改动 | 主指标 | 训练表现 | 验证表现 | 主要错误 | 结论 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| baseline | simple_cnn | 0.5838 | train_acc 0.7399 | val_acc 0.6253 | cloudy 混淆 | 可用基线 |

## 调优决策规则

先看错误，再决定动作：

- 训练集差、验证集也差：优先降低偏差，尝试更强模型、预训练模型、更高分辨率或训练更久。
- 训练集好、验证集差：优先降低方差，尝试更强增强、正则化、早停、更多数据或清理异常样本。
- 同分布验证好、平台测试差：优先考虑数据不匹配，检查平台输入格式、预处理一致性和真实场景差异。
- 某类 precision 很低：模型把太多其他类预测成该类，优先查混淆矩阵和该类判别边界。
- 某类 recall 很低：该类被大量漏判，优先查样本量、增强策略、类别权重和标签质量。
- 小幅指标提升要谨慎，尤其验证集较小时，应看多 seed 或错例是否真实改善。

优先级清单：

1. 数据划分是否正确。
2. 主指标是否仍是 `macro F1`。
3. 标签和类别顺序是否可靠。
4. 错误主要集中在哪些类别。
5. 当前瓶颈是偏差还是方差。
6. 是否存在平台/真实场景数据不匹配。
7. 是否需要更强模型或预训练权重。
8. 是否需要调整增强、采样、损失或阈值。
9. 是否满足 CPU 2 核、8 GiB、70 分钟推理约束。

## 推理入口

`main.py` 必须保持以下接口：

```python
def predict(X):
    ...
```

推理约定：

- `X` 是 `cv2.imread` 读取的 `numpy.ndarray`，通道顺序是 BGR。
- 推理前必须把 BGR 转 RGB，并使用与训练一致的预处理。
- 返回值必须是 `cloudy`、`rainy`、`snowy`、`sunny` 之一。
- 默认加载 `results/model_best.pth`。
- `import main` 不应触发训练、下载或耗时计算。
- `predict(X)` 内不要重复加载模型，模型应缓存。
- `predict(X)` 内不要访问网络或依赖本机绝对路径。

任何影响 `predict(X)` 的改动，都必须重新做单图预测和 CPU 推理耗时检查。

## 提交前检查

至少完成以下检查：

```powershell
conda run -n raicom-weather python -c "import torch, torchvision, cv2, numpy, sklearn; print(torch.__version__)"
```

```powershell
conda run -n raicom-weather python -m py_compile main.py train.py evaluate.py src/weather_classifier/model.py src/weather_classifier/predictor.py
```

```powershell
conda run -n raicom-weather python -c "import main; print(callable(main.predict))"
```

```powershell
conda run -n raicom-weather python -c "import cv2, main; img=cv2.imread('data/train/cloudy/cloudy_00001.jpg'); print(main.predict(img))"
```

```powershell
conda run -n raicom-weather python evaluate.py --data-dir data/train --checkpoint results/model_best.pth --batch-size 32 --num-workers 0 --device cpu
```

还需确认：

- `results/model_best.pth` 存在。
- `main.py` 能在 Mo 平台模板中导入并加载 checkpoint。
- CPU 推理速度满足平台整体时间约束。
- 提交包包含推理代码和权重文件。
- 提交前不依赖 `archive/original_submission/` 作为运行入口。

## 修改规则

- 不重新引入多层同名目录结构。
- 不把大型模型、训练日志和原始数据纳入版本控制，除非提交包明确需要。
- 修改模型结构后，同步检查 `save_checkpoint`、`load_checkpoint` 和 `Predictor`。
- 修改标签顺序后，同步检查训练集 `class_to_idx`、checkpoint `classes`、`main.py` 输出和平台模板要求。
- 修改数据划分、预处理、增强或损失函数后，必须重新记录实验指标。
- 最终测试集或平台评分集只用于最终确认，不用于反复选择模型。

