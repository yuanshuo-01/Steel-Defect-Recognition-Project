"""
钢材表面缺陷数据集读取与预处理

数据来源：NEU Surface Defect Database（东北大学 Song Kechen & Yan Yunhui），
使用其公开划分：train 1437 / val 180 / test 180，共 6 类、每类数量均衡。
目录结构约定：
    data/NEU-DET/images/{train,val,test}/<类别名>_<编号>.jpg
标签直接由文件名前缀解析，分类任务不需要额外的标注文件。
"""

import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# 6 类缺陷，顺序即标签 id
CLASSES = [
    "crazing",          # 裂纹
    "inclusion",        # 夹杂
    "patches",          # 斑块
    "pitted_surface",   # 麻点
    "rolled-in_scale",  # 氧化皮
    "scratches",        # 划痕
]
CLASS_CN = ["裂纹", "夹杂", "斑块", "麻点", "氧化皮", "划痕"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# 归一化系数：在 train 划分的灰度图上统计得到
NORM_MEAN = [0.5042]
NORM_STD = [0.1707]


def label_from_name(file_name):
    """crazing_12.jpg -> 0"""
    cls_name = "_".join(os.path.splitext(file_name)[0].split("_")[:-1])
    return CLASS_TO_IDX[cls_name]


class SteelDefectDataset(Dataset):
    """按 images/<split>/<class>_<id>.jpg 组织的分类数据集。"""

    def __init__(self, root, split="train", img_size=224, train=False, strong_aug=False):
        self.root = os.path.join(root, "images", split)
        if not os.path.isdir(self.root):
            raise FileNotFoundError("找不到数据目录：{}".format(self.root))
        self.samples = []
        for f in sorted(os.listdir(self.root)):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                self.samples.append((os.path.join(self.root, f), label_from_name(f)))
        if not self.samples:
            raise RuntimeError("{} 下没有图片".format(self.root))

        ops = [
            transforms.Grayscale(num_output_channels=1),   # 缺陷是纹理特征，统一转单通道
            transforms.Resize((img_size, img_size)),
        ]
        if train:
            # 缺陷图像没有"上下左右"的语义约束，翻转与旋转都是安全的增强
            ops += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
            if strong_aug:
                ops += [
                    transforms.RandomRotation(15),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                ]
        ops += [transforms.ToTensor(), transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)]
        self.transform = transforms.Compose(ops)
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")   # 先统一成 RGB，再由 Grayscale 转单通道
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_dataloaders(data_root, img_size=224, batch_size=32, num_workers=2, strong_aug=False):
    """返回 train / val / test 三个 DataLoader。"""
    train_set = SteelDefectDataset(data_root, "train", img_size, train=True, strong_aug=strong_aug)
    val_set = SteelDefectDataset(data_root, "val", img_size, train=False)
    test_set = SteelDefectDataset(data_root, "test", img_size, train=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import collections

    default_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "data", "NEU-DET")
    for split in ["train", "val", "test"]:
        ds = SteelDefectDataset(default_root, split)
        counter = collections.Counter(CLASSES[label] for _, label in ds.samples)
        print(split, len(ds), dict(counter))
