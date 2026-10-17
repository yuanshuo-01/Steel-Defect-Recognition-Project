"""
预测结果可视化：从测试集抽样，展示"真实类别 / 预测类别 / 置信度"。

用法：
    python scripts/visualize_predictions.py --num 24
"""

import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# 脚本在 scripts/ 下执行，需要把 src/ 加入模块搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from dataset import CLASSES, CLASS_CN, NORM_MEAN, NORM_STD, SteelDefectDataset  # noqa: E402
from model import build_model  # noqa: E402
from utils import setup_chinese_font  # noqa: E402


def parse_args():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.path.join(root, "data", "NEU-DET"))
    parser.add_argument("--checkpoint", default=os.path.join(root, "outputs", "checkpoints", "best_model.pth"))
    parser.add_argument("--output-dir", default=os.path.join(root, "outputs"))
    parser.add_argument("--num", type=int, default=24)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def denormalize(tensor):
    """把归一化后的张量还原成可显示的灰度图。"""
    arr = tensor.squeeze(0).cpu().numpy()
    arr = arr * NORM_STD[0] + NORM_MEAN[0]
    return np.clip(arr, 0, 1)


def main():
    args = parse_args()
    setup_chinese_font()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=len(CLASSES), in_channels=1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    dataset = SteelDefectDataset(args.data_root, "test", img_size=args.img_size, train=False)
    indices = random.sample(range(len(dataset)), min(args.num, len(dataset)))

    cols = 6
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.6))
    axes = np.array(axes).reshape(rows, cols)

    correct = 0
    for ax, idx in zip(axes.ravel(), indices):
        img, label = dataset[idx]
        with torch.no_grad():
            logits = model(img.unsqueeze(0).to(device))
            prob = F.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(prob))
        conf = float(prob[pred])
        correct += int(pred == label)

        ax.imshow(denormalize(img), cmap="gray")
        ax.set_title("{}\n{} {:.2f}".format(CLASS_CN[label], CLASS_CN[pred], conf),
                     color="green" if pred == label else "red", fontsize=9)
        ax.axis("off")

    for ax in axes.ravel()[len(indices):]:
        ax.axis("off")

    plt.suptitle("预测样例（上：真实，下：预测 + 置信度），抽样 {} 张".format(len(indices)))
    plt.tight_layout()
    out_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "predictions_sample.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print("抽样准确率: {}/{}".format(correct, len(indices)))
    print("已保存", save_path)


if __name__ == "__main__":
    main()
