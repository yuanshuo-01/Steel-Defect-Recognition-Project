import torch
from torch import nn
from torchsummary import summary


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, strides=1, use_1conv=False):
        super(Residual, self).__init__()
        self.ReLU = nn.ReLU(inplace=True)
        # 主路径：3x3 conv -> BN -> ReLU -> 3x3 conv -> BN
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels,
                               kernel_size=3, padding=1, stride=strides, bias=False)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

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
        y = self.bn2(self.conv2(y))          # 修正：接上一层输出 y，而不是原始输入 x

        if self.conv3 is not None:           # 修正：显式判断，并把结果赋回 x
            x = self.bn3(self.conv3(x))

        y = self.ReLU(y + x)                 # 相加后再激活
        return y


class ResNet18(nn.Module):
    def __init__(self, block=Residual):
        super(ResNet18, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),              # 修正顺序：Conv -> BN -> ReLU
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(
            block(64, 64, 1, False),
            block(64, 64, 1, False)
        )
        self.b3 = nn.Sequential(
            block(64, 128, 2, True),
            block(128, 128, 1, False)
        )
        self.b4 = nn.Sequential(
            block(128, 256, 2, True),
            block(256, 256, 1, False)
        )
        self.b5 = nn.Sequential(
            block(256, 512, 2, True),
            block(512, 512, 1, False)
        )
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18().to(device)
    print(summary(model, (1, 224, 224), device=str(device)))
