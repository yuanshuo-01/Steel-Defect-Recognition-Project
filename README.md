# Steel-Defect-Recognition-Project

基于**手写 ResNet-18** 的热轧带钢表面缺陷识别（6 分类）项目。

数据集：NEU Surface Defect Database（东北大学 Song Kechen & Yan Yunhui），
6 类缺陷各约 300 张，使用官方划分 train 1437 / val 180 / test 180。

---

## 1. 环境搭建

```bash
# 使用已有的 Anaconda 环境（实测可用）
conda activate envpytorch

# 或新建环境
conda create -n envpytorch python=3.10 -y
conda activate envpytorch
pip install -r requirements.txt
```

实测环境（本项目开发所用）：

| 组件 | 版本 |
| --- | --- |
| Python | 3.10.20 |
| PyTorch | 2.11.0+cu130 |
| torchvision | 0.26.0+cu130 |
| CUDA | 13.0（NVIDIA GeForce RTX 5060 Laptop GPU） |
| numpy / pandas / matplotlib / scikit-learn | 2.2.6 / 2.3.3 / 3.10.9 / 1.7.2 |

无 GPU 也能跑（自动回退 CPU），只是训练慢一些。

## 2. 数据准备

数据集体积约 30MB，未纳入 Git 仓库（见 `.gitignore`），用脚本一键获取：

```bash
python scripts/download_data.py
```

下载完成后目录结构为：

```
data/NEU-DET/
├── classes.txt
├── train.txt / val.txt / test.txt
└── images/
    ├── train/  (1437 张)
    ├── val/    (180 张)
    └── test/   (180 张)
```

图片命名格式为 `<类别>_<编号>.jpg`，标签由文件名前缀解析，分类任务不需要额外标注文件。

## 3. 训练

```bash
python src/train.py --epochs 30 --batch-size 32 --lr 0.001
```

常用参数：`--img-size`（默认 224）、`--seed`（默认 42）、`--device cuda|cpu`。
训练产物：

- `outputs/checkpoints/best_model.pth` —— 验证集准确率最优权重
- `outputs/logs/train_history.csv` —— 每 epoch 的 loss / accuracy
- `outputs/figures/train_curves.png` —— 训练曲线

## 4. 测试

```bash
python src/test.py --checkpoint outputs/checkpoints/best_model.pth
```

输出测试集准确率、分类报告（precision / recall / f1）与混淆矩阵，
同时写入 `outputs/logs/test_report.txt`。

## 5. 项目结构

```
src/model.py     手写 ResNet-18（残差块 + 4 个 stage + 分类头）
src/dataset.py   数据读取、灰度化、归一化与数据增强
src/train.py     训练主循环，保存最优权重与曲线
src/test.py      测试评估、分类报告、混淆矩阵
src/utils.py     随机种子、曲线与混淆矩阵绘制
scripts/download_data.py   数据集下载
docs/ITERATIONS.md          迭代记录
```

## 6. 结果

见 `docs/ITERATIONS.md`（每次迭代的改动原因与实验结果）。

当前最优版本（v1.0 基线）：测试准确率 **0.9389**，macro F1 0.9383。

| 缺陷类别 | 含义 | F1 |
| --- | --- | --- |
| crazing | 裂纹 | 0.968 |
| inclusion | 夹杂 | 0.949 |
| patches | 斑块 | 0.921 |
| pitted_surface | 麻点 | 0.877 |
| rolled-in_scale | 氧化皮 | 0.968 |
| scratches | 划痕 | 0.947 |
