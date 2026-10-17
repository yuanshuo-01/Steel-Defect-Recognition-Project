"""
训练脚本（v1 基线）

用法（在项目根目录执行）：
    python src/train.py --epochs 30 --batch-size 32 --lr 0.001

产物：
    outputs/checkpoints/best_model.pth   验证集最优权重
    outputs/logs/train_history.csv       每个 epoch 的 loss / acc
    outputs/figures/train_curves.png     训练曲线
"""

import argparse
import os
import time

import pandas as pd
import torch
import torch.nn as nn

from dataset import CLASSES, build_dataloaders
from model import build_model
from utils import ensure_dir, get_device, plot_curves, set_seed


def parse_args():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="钢材表面缺陷分类 - 训练")
    parser.add_argument("--data-root", default=os.path.join(root, "data", "NEU-DET"))
    parser.add_argument("--output-dir", default=os.path.join(root, "outputs"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto / cuda / cpu")
    return parser.parse_args()


def run_one_epoch(model, loader, criterion, optimizer, device, train=True):
    """跑一个 epoch，返回 (平均 loss, 准确率)。"""
    model.train() if train else model.eval()
    total_loss, total_correct, total_num = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            total_loss += loss.item() * images.size(0)
            total_correct += torch.sum(preds == labels.data).item()
            total_num += images.size(0)

    return total_loss / total_num, total_correct / total_num


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device in ("auto", "cuda"))
    ckpt_dir = ensure_dir(os.path.join(args.output_dir, "checkpoints"))
    log_dir = ensure_dir(os.path.join(args.output_dir, "logs"))
    fig_dir = ensure_dir(os.path.join(args.output_dir, "figures"))

    train_loader, val_loader, _ = build_dataloaders(
        args.data_root, img_size=args.img_size, batch_size=args.batch_size,
        num_workers=args.num_workers)

    model = build_model(num_classes=len(CLASSES), in_channels=1, stem_kernel=7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    print("device: {} | 类别数: {} | train: {} val: {}".format(
        device, len(CLASSES), len(train_loader.dataset), len(val_loader.dataset)))

    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_state = 0.0, None
    since = time.time()

    for epoch in range(args.epochs):
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, False)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print("Epoch {:>2}/{} | train loss {:.4f} acc {:.4f} | val loss {:.4f} acc {:.4f}".format(
            epoch + 1, args.epochs, train_loss, train_acc, val_loss, val_acc))

    # 循环外统一保存最优权重：若放在循环内，会把权重回滚成历史最优，等于破坏训练
    ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
    torch.save(best_state, ckpt_path)
    pd.DataFrame(history).to_csv(os.path.join(log_dir, "train_history.csv"), index=False)
    plot_curves(history, os.path.join(fig_dir, "train_curves.png"))

    print("最优验证准确率: {:.4f}".format(best_acc))
    print("耗时 {:.0f}m{:.0f}s".format((time.time() - since) // 60, (time.time() - since) % 60))
    print("权重已保存至", ckpt_path)


if __name__ == "__main__":
    main()
