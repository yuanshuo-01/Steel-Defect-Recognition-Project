# Steel-Defect-Recognition-Project

基于**手写 ResNet-18** 的热轧带钢表面缺陷识别（6 分类）项目。

- 任务：把钢材表面图像分到 6 类缺陷（裂纹 crazing / 夹杂 inclusion / 斑块 patches /
  麻点 pitted_surface / 氧化皮 rolled-in_scale / 划痕 scratches）
- 数据集：NEU Surface Defect Database（东北大学 Song Kechen & Yan Yunhui 发布），
  每类约 300 张 200×200 灰度图，使用官方划分 train 1437 / val 180 / test 180
- 核心网络：残差块与 ResNet-18 主干全部**手写实现**（见 `src/model.py`），
  并与 ImageNet 预训练 ResNet-18 微调做了对比实验

## 最终结果

| 指标 | 手写 ResNet-18（本项目主体） | 预训练 ResNet-18 微调（对比） |
| --- | --- | --- |
| 测试准确率 | 0.9944（179/180） | 1.0000（180/180） |
| macro F1 | 0.9944 | 1.0000 |
| 最优验证准确率 | 1.0000 | 1.0000 |

每次迭代的改动原因与实验数据见 [`docs/ITERATIONS.md`](docs/ITERATIONS.md)。

## 1. 环境搭建

```bash
conda create -n envpytorch python=3.10 -y
conda activate envpytorch
pip install -r requirements.txt
```

实测开发环境：

| 组件 | 版本 |
| --- | --- |
| Python | 3.10.20 |
| PyTorch / torchvision | 2.11.0+cu130 / 0.26.0+cu130 |
| CUDA | 13.0（NVIDIA GeForce RTX 5060 Laptop GPU） |
| numpy / pandas / matplotlib / scikit-learn | 2.2.6 / 2.3.3 / 3.10.9 / 1.7.2 |

无 GPU 也可以运行（自动回退 CPU），训练会慢一些。

## 2. 数据准备

数据集约 30MB，不纳入 Git 仓库（见 `.gitignore`），用脚本一键获取：

```bash
python scripts/download_data.py          # 默认下载到 data/NEU-DET
```

也可以手动从 NEU 官方渠道下载数据集，按下面的目录结构放置：

```
data/NEU-DET/
├── classes.txt
├── train.txt / val.txt / test.txt       # 官方划分清单
└── images/
    ├── train/  (1437 张)
    ├── val/    (180 张)
    └── test/   (180 张)
```

图片命名为 `<类别>_<编号>.jpg`（如 `crazing_12.jpg`），标签由文件名前缀解析，
分类任务不需要额外的标注文件。

## 3. 训练

```bash
# 主线：手写 ResNet-18（默认）
python src/train.py --epochs 40 --batch-size 32 --lr 0.001 --scheduler cosine --strong-aug

# 对比线：ImageNet 预训练 ResNet-18 微调（学习率小一个量级）
python src/train.py --arch resnet18 --epochs 30 --batch-size 32 --lr 0.0001 \
    --scheduler cosine --strong-aug --ckpt-name best_model_resnet18_tl.pth
```

常用参数：`--img-size`（默认 224）、`--seed`（默认 42）、`--device cuda|cpu`、
`--weight-decay`（默认 1e-4）。

训练产物（`outputs/`，不入库）：

- `checkpoints/best_model.pth` —— 验证集准确率最优权重
- `logs/train_history.csv`、`logs/train_config.json` —— 逐轮指标与实验配置
- `figures/train_curves.png` —— 训练曲线

## 4. 测试与可视化

```bash
# 测试集评估：准确率、分类报告、混淆矩阵图
python src/test.py --checkpoint outputs/checkpoints/best_model.pth
# 若评估迁移学习模型，需指定结构
python src/test.py --arch resnet18 --checkpoint outputs/checkpoints/best_model_resnet18_tl.pth

# 预测样例可视化：抽样 24 张，显示真实/预测类别与置信度
python scripts/visualize_predictions.py --num 24

# Grad-CAM 可解释性：高亮模型判据所在的图像区域
python scripts/grad_cam.py --num 8
```

## 5. 项目结构

```
src/
├── model.py        手写残差块 + ResNet-18；预训练模型适配（单通道卷积核压缩、6 类分类头）
├── dataset.py      数据读取、灰度化、归一化、翻转/旋转/模糊增强
├── train.py        训练主循环：余弦退火、最优权重保存、曲线与配置落盘
├── test.py         测试评估：分类报告、混淆矩阵
└── utils.py        随机种子、曲线与混淆矩阵绘制、中文字体
scripts/
├── download_data.py           数据集下载
├── visualize_predictions.py   预测样例可视化
└── grad_cam.py                手写 Grad-CAM 可解释性可视化
docs/ITERATIONS.md              每次迭代的改动原因与实验结果
```

## 6. 关键实现说明

- **残差块**：主路径 3×3 conv → BN → ReLU → 3×3 conv → BN，shortcut 在通道变化或下采样时
  用 1×1 卷积对齐并配套 BN，两支相加后再激活；
- **单通道输入**：缺陷是纹理特征，颜色无判别力，输入统一转灰度，预训练模型的首层
  三通道卷积核沿通道维求均值压成单通道，保留预训练信息；
- **归一化系数**：均值 0.5042 / 标准差 0.1707，由 train 划分统计得到；
- **数据增强**：翻转 + 旋转 ±15° + 平移缩放 + 亮度对比度扰动 + 高斯模糊，
  对应工业现场的拍摄角度、光照与失焦差异；
- **可复现**：固定随机种子，实验配置自动写入 `outputs/logs/train_config.json`。
