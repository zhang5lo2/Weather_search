# Baseline Experiments

本文件记录可复现实验结果。后续模型必须和本 baseline 使用相同的本地验证划分或明确说明差异，否则指标不可直接横向比较。

## baseline_simple_cnn_2026-06-25

### 目标

建立一个能完整完成训练、保存、加载和 CPU 推理的基础模型，作为 2026 睿抗 / 智海 Mo 天气四分类任务的本地 baseline。

### 环境

```text
conda env: raicom-weather
local torch: 2.1.2
local torchvision: 0.16.2
opencv-python: 4.10.0
scikit-learn: 1.9.0
```

比赛 FAQ 给出的最终推理环境约束：

```text
torch: 2.1.7
CPU: 2 核
memory: 8 GiB
overall CPU inference time: 70 分钟内
```

本地公开源未提供精确 `torch 2.1.7`，当前环境使用 PyTorch `2.1.x` / torchvision `0.16.x` 兼容线。正式提交前仍需在 Mo 平台模板中复查依赖和推理入口。

### 数据集处理

本次只使用本地 `data/train/`，目录名即类别标签：

```text
cloudy  2184
rainy    446
snowy    403
sunny   1966
total   4999
```

当前没有官方测试集或评分集落在本地，因此本 baseline 不使用外部测试集，只从训练集内部做分层验证集划分：

```text
split: stratified train/validation
seed: 42
val_ratio: 0.15
train size: 4249
val size: 750
```

划分后的类别数量：

```text
train: cloudy=1856, rainy=379, snowy=343, sunny=1671
val:   cloudy=328,  rainy=67,  snowy=60,  sunny=295
```

训练预处理：

```text
Resize(128, 128)
RandomHorizontalFlip(p=0.5)
ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12)
ToTensor()
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

验证和推理预处理：

```text
Resize(128, 128)
ToTensor()
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

推理入口 `main.predict(X)` 接收 `cv2.imread` 读取的 BGR 图像，内部会先转换为 RGB，再执行与验证一致的预处理。

类别不均衡处理：

```text
CrossEntropyLoss(class_weights)
WeightedRandomSampler
```

### 模型与训练命令

模型：

```text
model: simple_cnn
image_size: 128
epochs: 10
batch_size: 32
optimizer: AdamW
lr: 3e-4
weight_decay: 1e-4
scheduler: CosineAnnealingLR
device: CPU
```

训练命令：

```powershell
conda run -n raicom-weather python train.py --data-dir data/train --output-dir results --epochs 10 --batch-size 32 --image-size 128 --model simple_cnn --num-workers 0
```

评估命令：

```powershell
conda run -n raicom-weather python evaluate.py --data-dir data/train --checkpoint results/model_best.pth --batch-size 32 --num-workers 0 --device cpu
```

模型文件：

```text
results/model_best.pth
results/model_latest.pth
```

### 训练过程

| epoch | train_loss | train_acc | val_loss_weighted | val_acc | val_macro_f1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8188 | 0.5333 | 1.0602 | 0.3907 | 0.3531 |
| 2 | 0.6855 | 0.5879 | 1.0648 | 0.3867 | 0.3625 |
| 3 | 0.5877 | 0.6237 | 1.0477 | 0.4413 | 0.3984 |
| 4 | 0.5711 | 0.6416 | 0.7690 | 0.5693 | 0.5255 |
| 5 | 0.5265 | 0.6717 | 1.4352 | 0.3387 | 0.3445 |
| 6 | 0.4947 | 0.6865 | 0.7636 | 0.5587 | 0.5236 |
| 7 | 0.4588 | 0.7133 | 0.7906 | 0.5653 | 0.5194 |
| 8 | 0.4369 | 0.7141 | 0.7403 | 0.5867 | 0.5396 |
| 9 | 0.4090 | 0.7397 | 0.7605 | 0.5760 | 0.5370 |
| 10 | 0.3997 | 0.7399 | 0.6834 | 0.6253 | 0.5838 |

最佳 checkpoint：

```text
epoch: 10
val_macro_f1: 0.5838
val_acc: 0.6253
```

### 独立评估结果

固定同一验证集重新加载 `results/model_best.pth`，CPU 推理结果：

```text
val_size: 750
val_loss_weighted: 0.6834
val_loss_unweighted: 0.9159
val_acc: 0.6253
val_macro_f1: 0.5838
val_weighted_f1: 0.6379
elapsed_seconds: 2.94
images_per_second: 255.02
```

分类报告：

```text
              precision    recall  f1-score   support

      cloudy     0.9237    0.3323    0.4888       328
       rainy     0.2490    0.8955    0.3896        67
       snowy     0.4474    0.8500    0.5862        60
       sunny     0.8989    0.8441    0.8706       295

    accuracy                         0.6253       750
   macro avg     0.6297    0.7305    0.5838       750
weighted avg     0.8156    0.6253    0.6379       750
```

混淆矩阵，行是真实类别，列是预测类别，顺序为 `cloudy, rainy, snowy, sunny`：

```text
[
  [109, 158, 38, 23],
  [0, 60, 6, 1],
  [1, 4, 51, 4],
  [8, 19, 19, 249]
]
```

入口检查：

```text
data/train/cloudy/cloudy_00001.jpg -> cloudy
```

### 结论

该模型可作为 CPU 友好的最小可用 baseline，但不是最终高分方案。主要问题是 `cloudy` 召回率偏低，大量 `cloudy` 被预测为 `rainy` 或 `snowy`；`rainy`、`snowy` 的召回率高但 precision 偏低。下一步应优先尝试预训练 `resnet18`、更强数据增强、类别不均衡策略调整和多 seed 验证。

### 无官方测试集时的处理原则

比赛本地阶段没有官方测试集时，不应把训练集全部用于训练后再用同一批图片报告效果。正确做法是：

1. 从 `data/train/` 内部固定划分一个分层验证集，用于本地选模型和比较实验。
2. 验证集只用于评估，不参与训练增强采样后的权重更新。
3. 所有本地实验必须记录 `seed`、`val_ratio`、划分方式和类别分布，保证指标可复现。
4. 如果需要更稳的本地估计，可以做多 seed 或 K-fold，但最终仍应保留一个固定验证协议作为对比标准。
5. 官方测试集或评分集只会在平台评测时通过提交入口调用模型，本地无法提前得到最终分数；提交前要确保 `main.predict(X)`、权重文件和依赖在平台 CPU 环境可运行。

## exp01_resnet18_pretrained_128_2026-06-25

### 目标

第一次模型改进只改变一个主要因素：将 baseline 的 `simple_cnn` 替换为 ImageNet 预训练 `resnet18`。验证协议、类别权重、balanced sampler 和 `macro F1` 主指标保持一致。

### 环境与数据

沿用 baseline 环境和固定验证协议：

```text
conda env: raicom-weather
local torch: 2.1.2
local torchvision: 0.16.2
split: stratified train/validation
seed: 42
val_ratio: 0.15
train size: 4249
val size: 750
```

类别数量：

```text
train: cloudy=1856, rainy=379, snowy=343, sunny=1671
val:   cloudy=328,  rainy=67,  snowy=60,  sunny=295
```

### 模型与训练命令

模型：

```text
model: resnet18
pretrained: ImageNet
image_size: 128
epochs: 5
batch_size: 32
optimizer: AdamW
lr: 3e-4
weight_decay: 1e-4
scheduler: CosineAnnealingLR
device: CPU
```

训练命令：

```powershell
conda run -n raicom-weather python train.py --data-dir data/train --output-dir results/exp01_resnet18_pretrained_128 --epochs 5 --batch-size 32 --image-size 128 --model resnet18 --pretrained --num-workers 0 --device cpu
```

评估命令：

```powershell
conda run -n raicom-weather python evaluate.py --data-dir data/train --checkpoint results/model_best.pth --batch-size 32 --num-workers 0 --device cpu --predictions-csv results/exp01_resnet18_pretrained_128/val_predictions.csv
```

模型文件：

```text
results/exp01_resnet18_pretrained_128/model_best.pth
results/exp01_resnet18_pretrained_128/model_latest.pth
results/model_best.pth
results/model_latest.pth
```

原 baseline 权重已备份：

```text
results/baseline_simple_cnn_2026-06-25_model_best.pth
results/baseline_simple_cnn_2026-06-25_model_latest.pth
```

### 训练过程

| epoch | train_loss | train_acc | val_loss_weighted | val_acc | val_macro_f1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3550 | 0.7865 | 0.5254 | 0.7773 | 0.7249 |
| 2 | 0.1637 | 0.8856 | 0.5234 | 0.8147 | 0.7691 |
| 3 | 0.1047 | 0.9270 | 0.4431 | 0.8747 | 0.8348 |
| 4 | 0.0540 | 0.9590 | 0.4247 | 0.8880 | 0.8534 |
| 5 | 0.0330 | 0.9727 | 0.4312 | 0.8973 | 0.8576 |

最佳 checkpoint：

```text
epoch: 5
val_macro_f1: 0.8576
val_acc: 0.8973
```

### 独立评估结果

固定同一验证集重新加载 `results/model_best.pth`，CPU 推理结果：

```text
val_size: 750
val_loss_weighted: 0.4312
val_loss_unweighted: 0.3116
val_acc: 0.8973
val_macro_f1: 0.8576
val_weighted_f1: 0.8975
elapsed_seconds: 3.55
images_per_second: 211.19
```

分类报告：

```text
              precision    recall  f1-score   support

      cloudy     0.9182    0.8902    0.9040       328
       rainy     0.7465    0.7910    0.7681        67
       snowy     0.8305    0.8167    0.8235        60
       sunny     0.9238    0.9458    0.9347       295

    accuracy                         0.8973       750
   macro avg     0.8548    0.8609    0.8576       750
weighted avg     0.8981    0.8973    0.8975       750
```

混淆矩阵，行是真实类别，列是预测类别，顺序为 `cloudy, rainy, snowy, sunny`：

```text
[
  [292, 13, 5, 18],
  [11, 53, 2, 1],
  [5, 2, 49, 4],
  [10, 3, 3, 279]
]
```

### 与 baseline 对比

| 实验 | model | val_acc | val_macro_f1 | val_weighted_f1 | CPU images/s |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline_simple_cnn_2026-06-25 | simple_cnn | 0.6253 | 0.5838 | 0.6379 | 255.02 |
| exp01_resnet18_pretrained_128_2026-06-25 | resnet18 pretrained | 0.8973 | 0.8576 | 0.8975 | 211.19 |

提升：

```text
val_macro_f1: +0.2738
val_acc: +0.2720
```

### 错误分析摘要

验证集预测明细：

```text
results/exp01_resnet18_pretrained_128/val_predictions.csv
```

验证集 750 张中预测正确 673 张，错误 77 张。剩余主要混淆：

```text
cloudy -> sunny: 18
cloudy -> rainy: 13
rainy  -> cloudy: 11
sunny  -> cloudy: 10
cloudy -> snowy: 5
snowy  -> cloudy: 5
snowy  -> sunny: 4
```

### 结论

第一次改进有效，预训练 ResNet18 显著优于小 CNN baseline，且 CPU 推理速度仍然充足。当前默认 `results/model_best.pth` 已更新为本实验的 ResNet18 checkpoint。

下一步优先方向：

1. 围绕 `cloudy` 与 `sunny/rainy` 的混淆做错误样本人工检查。
2. 尝试 160 或 224 输入分辨率，观察 `macro F1` 与 CPU 推理速度的平衡。
3. 尝试更细的数据增强和少数类策略，重点看 `rainy` precision 与 `snowy` F1 是否继续提升。

## exp02_resnet18_aug_128_2026-06-25

### 目标

只改变训练增强强度，观察是否能进一步减少 `cloudy/sunny/rainy` 混淆。

### 训练配置

```text
model: resnet18
pretrained: ImageNet
image_size: 128
epochs: 5
batch_size: 32
optimizer: AdamW
lr: 3e-4
weight_decay: 1e-4
scheduler: CosineAnnealingLR
device: CPU
augmentation: stronger than exp01
```

训练命令：

```powershell
conda run -n raicom-weather python train.py --data-dir data/train --output-dir results/exp02_resnet18_aug_128 --epochs 5 --batch-size 32 --image-size 128 --model resnet18 --pretrained --num-workers 0 --device cpu --random-resized-crop --crop-scale-min 0.75 --brightness 0.25 --contrast 0.25 --saturation 0.20 --hue 0.03 --random-rotation 7
```

### 结果

```text
val_acc: 0.8640
val_macro_f1: 0.8195
val_weighted_f1: 0.8658
elapsed_seconds: 3.21
images_per_second: 233.40
```

分类报告：

```text
              precision    recall  f1-score   support

      cloudy     0.9233    0.8079    0.8618       328
       rainy     0.6136    0.8060    0.6968        67
       snowy     0.7778    0.8167    0.7967        60
       sunny     0.8974    0.9492    0.9226       295
```

### 结论

这组增强过强，整体指标低于 exp01，不应替换默认模型。`cloudy` 召回下降明显。

## exp03_resnet18_160_2026-06-25

### 目标

只提升输入分辨率到 160，观察细节信息是否能继续提升宏 F1 和少数类召回。

### 训练配置

```text
model: resnet18
pretrained: ImageNet
image_size: 160
epochs: 5
batch_size: 32
optimizer: AdamW
lr: 3e-4
weight_decay: 1e-4
scheduler: CosineAnnealingLR
device: CPU
augmentation: same as exp01
```

训练命令：

```powershell
conda run -n raicom-weather python train.py --data-dir data/train --output-dir results/exp03_resnet18_160 --epochs 5 --batch-size 32 --image-size 160 --model resnet18 --pretrained --num-workers 0 --device cpu
```

### 结果

```text
val_acc: 0.8960
val_macro_f1: 0.8688
val_weighted_f1: 0.8966
elapsed_seconds: 5.04
images_per_second: 148.77
```

分类报告：

```text
              precision    recall  f1-score   support

      cloudy     0.9043    0.8933    0.8988       328
       rainy     0.7632    0.8657    0.8112        67
       snowy     0.8475    0.8333    0.8403        60
       sunny     0.9313    0.9186    0.9249       295
```

### 结论

`160` 明显优于 `128` 的 exp01，并提升了 `rainy/snowy` 召回，但还不是当前最优。

## exp04_resnet18_224_2026-06-25

### 目标

继续提高输入分辨率到 224，验证是否还能进一步提升准确率和召回率。

### 训练配置

```text
model: resnet18
pretrained: ImageNet
image_size: 224
epochs: 5
batch_size: 32
optimizer: AdamW
lr: 3e-4
weight_decay: 1e-4
scheduler: CosineAnnealingLR
device: CPU
augmentation: same as exp01
```

训练命令：

```powershell
conda run -n raicom-weather python train.py --data-dir data/train --output-dir results/exp04_resnet18_224 --epochs 5 --batch-size 32 --image-size 224 --model resnet18 --pretrained --num-workers 0 --device cpu
```

### 结果

```text
val_acc: 0.9280
val_macro_f1: 0.9128
val_weighted_f1: 0.9281
elapsed_seconds: 8.79
images_per_second: 85.28
```

分类报告：

```text
              precision    recall  f1-score   support

      cloudy     0.9271    0.9299    0.9285       328
       rainy     0.8406    0.8657    0.8529        67
       snowy     0.9322    0.9167    0.9244        60
       sunny     0.9488    0.9424    0.9456       295
```

### 结论

这是当前最优模型。CPU 推理速度仍满足平台约束，且少数类召回继续提升。

## 当前对比

| 实验名 | 改动 | 主指标 | 训练表现 | 验证表现 | 主要错误 | 结论 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| baseline_simple_cnn_2026-06-25 | simple_cnn | 0.5838 | train_acc 0.7399 | val_acc 0.6253 | cloudy 混淆 | 可用基线 |
| exp01_resnet18_pretrained_128_2026-06-25 | resnet18 pretrained | 0.8576 | train_acc 0.9727 | val_acc 0.8973 | cloudy/sunny | 有效提升 |
| exp02_resnet18_aug_128_2026-06-25 | stronger aug | 0.8195 | train_acc 0.9369 | val_acc 0.8640 | cloudy 召回下降 | 不保留 |
| exp03_resnet18_160_2026-06-25 | image_size 160 | 0.8688 | train_acc 0.9743 | val_acc 0.8960 | cloudy/sunny | 优于 exp01 |
| exp04_resnet18_224_2026-06-25 | image_size 224 | 0.9128 | train_acc 0.9769 | val_acc 0.9280 | cloudy/sunny 少量残余 | 当前最优 |

### 当前默认模型

```text
name: exp04_resnet18_224_2026-06-25
model: resnet18 pretrained
checkpoint: results/model_best.pth
val_acc: 0.9280
val_macro_f1: 0.9128
val_weighted_f1: 0.9281
CPU speed: about 88 images/s
```

### 下一步建议

当前已经达到较高精度和召回，继续单纯增大分辨率的收益开始变小。后续优先考虑：

1. 用 `224` 模型做多 seed 复核，确认结果稳定。
2. 针对 `cloudy` 与 `sunny` 的残余混淆做错例分析。
3. 尝试更轻的增强或学习率策略微调，看看是否能在不损失速度的情况下再挤出一点 `macro F1`。
