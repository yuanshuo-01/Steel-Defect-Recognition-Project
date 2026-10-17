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
