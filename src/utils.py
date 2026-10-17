"""通用工具：随机种子、目录创建、训练曲线、混淆矩阵。"""

import os
import random

import matplotlib
matplotlib.use("Agg")           # 服务器/后台运行不弹窗，只保存图片
import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed=42):
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def get_device(prefer_cuda=True):
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def plot_curves(history, save_path):
    """绘制 train/val 的 loss 与 accuracy 曲线。"""
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], "ro-", label="train loss")
    axes[0].plot(epochs, history["val_loss"], "bs-", label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "ro-", label="train acc")
    axes[1].plot(epochs, history["val_acc"], "bs-", label="val acc")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("acc")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, save_path, normalize=False):
    """绘制并保存混淆矩阵，同时返回归一化后的矩阵（百分比）。"""
    if normalize:
        cm_plot = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    else:
        cm_plot = cm.astype("float")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion matrix (rows = true)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    thresh = cm_plot.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = "{:.2f}".format(cm_plot[i, j]) if normalize else str(int(cm[i, j]))
            ax.text(j, i, value, ha="center", va="center",
                    color="white" if cm_plot[i, j] > thresh else "black", fontsize=10)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return cm_plot
