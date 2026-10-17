"""
测试脚本（v1）：加载最优权重，在 test 划分上评估。

用法（在项目根目录执行）：
    python src/test.py --checkpoint outputs/checkpoints/best_model.pth
"""

import argparse
import os

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from dataset import CLASSES, build_dataloaders
from model import build_model
from utils import ensure_dir, get_device, set_seed


def parse_args():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="钢材表面缺陷分类 - 测试")
    parser.add_argument("--data-root", default=os.path.join(root, "data", "NEU-DET"))
    parser.add_argument("--checkpoint", default=os.path.join(root, "outputs", "checkpoints", "best_model.pth"))
    parser.add_argument("--output-dir", default=os.path.join(root, "outputs"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def predict(model, loader, device):
    """返回全部预测标签与真实标签。"""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            y_pred.extend(preds.cpu().tolist())
            y_true.extend(labels.tolist())
    return y_true, y_pred


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    fig_dir = ensure_dir(os.path.join(args.output_dir, "figures"))

    _, _, test_loader = build_dataloaders(
        args.data_root, img_size=args.img_size, batch_size=args.batch_size,
        num_workers=args.num_workers)

    model = build_model(num_classes=len(CLASSES), in_channels=1, stem_kernel=7).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)

    y_true, y_pred = predict(model, test_loader, device)
    acc = accuracy_score(y_true, y_pred)

    print("测试集样本数:", len(y_true))
    print("测试准确率: {:.4f}".format(acc))
    print()
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("混淆矩阵（行=真实，列=预测）:")
    print(cm)

    report_path = os.path.join(args.output_dir, "logs", "test_report.txt")
    ensure_dir(os.path.dirname(report_path))
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("checkpoint: {}\n".format(args.checkpoint))
        f.write("test accuracy: {:.4f}\n\n".format(acc))
        f.write(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))
        f.write("\nconfusion matrix:\n{}\n".format(cm))
    print("报告已保存至", report_path)


if __name__ == "__main__":
    main()
