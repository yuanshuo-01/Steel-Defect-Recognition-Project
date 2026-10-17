"""
手写 ResNet-18 网络结构（钢材表面缺陷分类）

设计说明：
1. 残差块沿用学习阶段的 Residual 实现，主路径为 3x3 conv -> BN -> ReLU -> 3x3 conv -> BN，
   shortcut 分支仅在通道数变化或需要下采样时启用 1x1 卷积，并配套 BN 保证两条支路分布一致。
2. 输入通道默认为 1：钢材缺陷是纹理特征，颜色信息没有判别力，转灰度后参数量与计算量都更小。
3. stem 提供 7x7 / 3x3 两种选择，便于后续做结构对比实验。
"""

import torch
from torch import nn


class Residual(nn.Module):
    """残差块：y = F(x) + shortcut(x)，相加后再激活。"""

    def __init__(self, input_channels, num_channels, strides=1, use_1conv=False, dropout=0.0):
        super(Residual, self).__init__()
        self.ReLU = nn.ReLU(inplace=True)
        # 主路径：3x3 conv -> BN -> ReLU -> 3x3 conv -> BN
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels,
                               kernel_size=3, padding=1, stride=strides, bias=False)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # shortcut 分支：需要改变通道数或下采样时才启用，并配套 BN
        if use_1conv:
            self.conv3 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels,
                                   kernel_size=1, stride=strides, bias=False)
            self.bn3 = nn.BatchNorm2d(num_channels)
        else:
            self.conv3 = None
            self.bn3 = None

    def forward(self, x):
        y = self.ReLU(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))          # 接上一层输出 y，而不是原始输入 x
        y = self.drop(y)

        if self.conv3 is not None:           # 显式判断，并把结果赋回 x
            x = self.bn3(self.conv3(x))

        y = self.ReLU(y + x)                 # 相加后再激活
        return y


class ResNet18(nn.Module):
    """手工搭建的 ResNet-18，stage 通道数为 64-64-128-256-512。"""

    def __init__(self, block=Residual, in_channels=1, num_classes=6, stem_kernel=7, dropout=0.0):
        super(ResNet18, self).__init__()
        if stem_kernel == 7:
            # 与标准 ResNet 一致：大卷积 stem + 最大池化，快速降分辨率
            stem = [
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),          # 顺序必须是 Conv -> BN -> ReLU
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        elif stem_kernel == 3:
            # 小卷积 stem：保留更多纹理细节，更适合 200x200 的缺陷图
            stem = [
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        else:
            raise ValueError("stem_kernel 只支持 7 或 3，当前为 {}".format(stem_kernel))

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.b1 = nn.Sequential(*stem)
        self.b2 = nn.Sequential(
            block(64, 64, 1, False, dropout),
            block(64, 64, 1, False, dropout),
        )
        self.b3 = nn.Sequential(
            block(64, 128, 2, True, dropout),
            block(128, 128, 1, False, dropout),
        )
        self.b4 = nn.Sequential(
            block(128, 256, 2, True, dropout),
            block(256, 256, 1, False, dropout),
        )
        self.b5 = nn.Sequential(
            block(256, 512, 2, True, dropout),
            block(512, 512, 1, False, dropout),
        )
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x


def build_model(num_classes=6, in_channels=1, stem_kernel=7, dropout=0.0):
    return ResNet18(in_channels=in_channels, num_classes=num_classes,
                    stem_kernel=stem_kernel, dropout=dropout)


def build_transfer_model(num_classes=6, in_channels=1, pretrained=True):
    """构建 torchvision 预训练 ResNet-18，用于与手写版做对比实验。

    适配两处：
    1. 输入通道：预训练模型首层是 RGB 三通道，缺陷图是单通道灰度。
       这里把预训练的三通道卷积核沿通道维求和取均值，压成单通道核，
       既保留了 ImageNet 预训练的滤波器信息，又匹配了输入格式。
    2. 输出层：1000 类分类头替换为 6 类。
    """
    from torchvision import models

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    net = models.resnet18(weights=weights)

    if in_channels != 3:
        old_conv = net.conv1
        new_conv = nn.Conv2d(in_channels, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                             padding=old_conv.padding, bias=False)
        if pretrained:
            new_conv.weight.data = old_conv.weight.data.sum(dim=1, keepdim=True) / 3.0
        net.conv1 = new_conv

    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


if __name__ == "__main__":
    model = build_model()
    print(model)
    try:
        from torchsummary import summary
        summary(model, (1, 224, 224))
    except ImportError:
        print("未安装 torchsummary，跳过结构摘要")
