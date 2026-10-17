"""
Grad-CAM 可解释性可视化：看模型到底"盯"着图像的哪块区域做判断。

实现要点：
1. 前向时在目标卷积层取值特征图，反向时取该层梯度；
2. 用梯度的全局平均作为通道权重，对特征图加权求和；
3. ReLU 掉负响应，双线性上采样到原图尺寸，叠加显示。

用法：
    python scripts/grad_cam.py --num 8
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from dataset import CLASSES, CLASS_CN, NORM_MEAN, NORM_STD, SteelDefectDataset  # noqa: E402
from model import build_model  # noqa: E402
from utils import setup_chinese_font  # noqa: E402


class GradCAM:
    """手工实现的 Grad-CAM，不依赖第三方库。"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.features = None
        self.grads = None
        self.fwd_handle = target_layer.register_forward_hook(self._save_features)
        self.bwd_handle = target_layer.register_full_backward_hook(self._save_grads)

    def _save_features(self, module, inputs, output):
        self.features = output.detach()

    def _save_grads(self, module, grad_input, grad_output):
        self.grads = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1))
        self.model.zero_grad()
        logits[0, class_idx].backward(retain_graph=False)

        weights = self.grads.mean(dim=(2, 3), keepdim=True)      # 通道权重
        cam = (weights * self.features).sum(dim=1, keepdim=True)  # 加权求和
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / max(cam.max() - cam.min(), 1e-8)
        return cam, class_idx, logits

    def close(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


def denormalize(tensor):
    arr = tensor.squeeze(0).cpu().numpy()
    arr = arr * NORM_STD[0] + NORM_MEAN[0]
    return np.clip(arr, 0, 1)


def parse_args():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.path.join(root, "data", "NEU-DET"))
    parser.add_argument("--checkpoint", default=os.path.join(root, "outputs", "checkpoints", "best_model.pth"))
    parser.add_argument("--output-dir", default=os.path.join(root, "outputs"))
    parser.add_argument("--num", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    setup_chinese_font()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=len(CLASSES), in_channels=1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # 取最后一个 stage 的第二个残差块的最后一层卷积，感受野最大、语义最强
    target_layer = model.b5[-1].conv2
    cam_extractor = GradCAM(model, target_layer)

    dataset = SteelDefectDataset(args.data_root, "test", img_size=args.img_size, train=False)
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(dataset), size=min(args.num, len(dataset)), replace=False)

    cols = 4
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.4))
    axes = np.array(axes).reshape(rows, cols)

    for ax, idx in zip(axes.ravel(), indices):
        img, label = dataset[idx]
        x = img.unsqueeze(0).to(device)
        cam, pred, _ = cam_extractor(x)

        base = denormalize(img)
        ax.imshow(base, cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=0.45)
        ax.set_title("true {} / pred {}".format(CLASS_CN[label], CLASS_CN[pred]),
                     color="green" if pred == label else "red", fontsize=10)
        ax.axis("off")

    for ax in axes.ravel()[len(indices):]:
        ax.axis("off")

    cam_extractor.close()
    plt.suptitle("Grad-CAM：高亮区域为模型判据所在（b5 最后一层卷积）")
    plt.tight_layout()
    out_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "grad_cam.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print("已保存", save_path)


if __name__ == "__main__":
    main()
