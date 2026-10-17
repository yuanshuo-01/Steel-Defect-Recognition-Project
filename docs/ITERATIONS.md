# 迭代记录

每次迭代对应一次提交与一个版本标记，记录"改了什么、为什么改、结果如何"。

## v1.0 —— 基线：重构出可跑通的最小闭环

**起点**：学习阶段留下的三个脚本（`model.py` / `train.py` / `test.py`）是面向 FashionMNIST + LeNet 的，
与钢材缺陷任务不匹配，主要问题：

1. `train.py` / `test.py` 里 `from model import LeNet`，但 `model.py` 中是 `ResNet18`，直接运行必然报错；
2. 数据管线读的是 FashionMNIST（28×28、10 类），与 200×200、6 类的缺陷数据不兼容；
3. 训练与测试没有统一的数据划分（缺少独立的 val / test）；
4. `plt.show()` 会阻塞，且曲线没有落盘，无法作为交付材料。

**改动**：

- 目录重构为 `src/` + `scripts/` + `docs/`，`model.py` 拆分为模型 / 数据 / 工具模块；
- `src/model.py` 保留手写残差块，输入通道改为 1（缺陷是纹理特征，颜色无判别力），输出改为 6 类；
- `src/dataset.py` 新增：按 `images/{train,val,test}` 读取、灰度化、Resize、归一化
  （归一化系数 0.5042 / 0.1707 由 train 划分统计得到）、随机翻转增强；
- `src/train.py` 新增：固定随机种子、每个 epoch 记录 train/val 指标、按验证集最优保存权重、
  曲线与 CSV 落盘；
- `src/test.py` 新增：加载权重评估 test 划分，输出准确率与分类报告；
- `.gitignore` 排除 `data/`（30MB 可再生数据）与 `outputs/`（训练产物）。

**结果**（RTX 5060 Laptop，30 epoch，batch 32，Adam lr=1e-3，耗时 36 分钟）：

| 指标 | 数值 |
| --- | --- |
| 最优验证准确率 | 0.9778 |
| 测试准确率 | **0.9389** |
| macro F1 | 0.9383 |

各类表现（测试集）：crazing 0.968 / inclusion 0.949 / patches 0.921 / pitted_surface 0.877 /
rolled-in_scale 0.968 / scratches 0.947。

**发现的问题**（作为后续迭代的出发点）：

1. 验证 loss 抖动剧烈（第 6 轮 5.14、第 15 轮 2.43、第 30 轮 1.44），固定学习率下训练不稳定；
2. `pitted_surface` 召回仅 0.833，与 `inclusion` / `patches` 混淆；`scratches` 有 3 张被判成 `patches`；
3. 每个 epoch 都要重建 4 个 DataLoader 子进程，单轮 45s 里相当一部分是数据加载开销。

---

## v1.1 —— 训练策略改进：余弦退火 + 强数据增强 + 权重衰减

**改动原因**（对应 v1.0 暴露的三个问题）：

1. 固定 lr=1e-3 时验证 loss 剧烈抖动（最高到 5.14），说明后期学习率过大 →
   加入 `CosineAnnealingLR` 余弦退火，末期 lr 降到 1e-6；
2. `pitted_surface` 召回偏低 → 加入强数据增强：旋转 ±15°、平移缩放、亮度对比度扰动、
   高斯模糊（模拟工业现场的光照差异、拍摄角度与镜头失焦）；
3. Adam 默认无正则 → 加 `weight_decay=1e-4`；
4. 每个 epoch 重建 DataLoader 子进程开销巨大 → `persistent_workers=True` 复用 worker。

**结果**（40 epoch，其余参数同 v1.0）：

| 指标 | v1.0 | v1.1 |
| --- | --- | --- |
| 最优验证准确率 | 0.9778 | **1.0000** |
| 测试准确率 | 0.9389 | **0.9944** |
| macro F1 | 0.9383 | **0.9944** |
| 训练耗时 | 36m13s | **2m55s** |

- 验证 loss 从 1.44 降到 0.0117，训练过程稳定收敛；
- 测试集 180 张仅错 1 张：`patches` → `pitted_surface`；
- 训练耗时下降约 12 倍，主要来自 DataLoader worker 复用。

**改动文件**：`src/train.py`（调度器、权重衰减、强增强开关、实验配置落盘）、
`src/dataset.py`（强增强算子、persistent_workers）。

---

## v1.2 —— 评估体系与可解释性完善

**改动原因**：v1.1 之后准确率已接近上限（测试集只剩 1 个错例），此时"只会报一个准确率"
不足以支撑交付，需要回答两个问题：错在哪儿、模型凭什么判断对。

**改动**：

1. `src/test.py`：混淆矩阵同时输出原始计数图与按行归一化图（召回率视角）；
2. `scripts/visualize_predictions.py`：从测试集抽样 24 张，展示"真实类别 / 预测类别 / 置信度"，
   判错标红、判对标绿；
3. `scripts/grad_cam.py`：手写 Grad-CAM（不依赖第三方 CAM 库），对 b5 最后一层卷积做
   前向取值 + 反向梯度加权，可视化模型判据所在区域；
4. `src/utils.py`：补充 matplotlib 中文字体设置，避免图内标题乱码。

**结果**：

- 混淆矩阵（归一化）确认唯一错例：1 张 `patches` 被判为 `pitted_surface`，两类在纹理上确实高度相似；
- 随机抽样 24 张的抽样准确率 23/24，与全量测试一致；
- Grad-CAM 显示：划痕样本的高亮区正好覆盖纵向缺陷条带，斑块/氧化皮样本高亮区覆盖缺陷主体，
  说明模型学到的判据集中在缺陷区域而非背景，预测具备可解释性。

**改动文件**：`src/test.py`、`src/utils.py`、`scripts/visualize_predictions.py`、`scripts/grad_cam.py`。

---

## v2.0 —— 对比实验：ImageNet 预训练 ResNet-18 微调 vs 手写 ResNet-18

**改动原因**：验证手写 ResNet 的水平——与工业界常规做法（预训练 + 微调）相比差距多大，
这本身就是一个有说服力的实验结论，也是本项目"迁移学习"学习内容的落地。

**改动**：

1. `src/model.py` 新增 `build_transfer_model`：加载 torchvision ImageNet 预训练 ResNet-18，
   首层三通道卷积核沿通道维求和取均值压成单通道核（保留预训练滤波器信息），
   分类头替换为 6 类；
2. `src/train.py` 增加 `--arch {custom,resnet18}` 与 `--ckpt-name`，两条实验线共用同一套
   数据管线与训练策略，保证对比公平；
3. `src/test.py` 增加 `--arch`，评估时与训练时的结构保持一致。

**实验设置**：预训练版微调 lr=1e-4（比从头训练小一个量级），30 epoch，其余与 v1.1 相同。

| 指标 | 手写 ResNet-18（v1.1） | 预训练 ResNet-18 微调 |
| --- | --- | --- |
| 最优验证准确率 | 1.0000 | 1.0000 |
| 测试准确率 | 0.9944（179/180） | **1.0000（180/180）** |
| macro F1 | 0.9944 | 1.0000 |
| 训练耗时 | 2m55s（40 epoch） | 4m10s（30 epoch） |
| 从零训练参数 | 是 | ImageNet 预训练 |

**结论**：

1. 预训练微调在测试集上全对，验证集 loss 收敛到 0.0002（手写版 0.0117），小样本下
   预训练特征仍有优势；
2. 但手写版差距只有 1 张图（0.56 个百分点），说明针对灰度纹理任务、配合强增强与
   余弦退火，从零训练的手写网络已经足够接近常规工业做法；
3. 唯一错例是 `patches` ↔ `pitted_surface`，两类缺陷纹理本身高度相似，属于合理混淆。

**改动文件**：`src/model.py`、`src/train.py`、`src/test.py`、`docs/ITERATIONS.md`。
