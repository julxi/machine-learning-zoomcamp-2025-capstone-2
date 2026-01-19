import torch
import torch.nn as nn
import preprocess


# -------------------------
# 1) FAST: Depthwise-separable small net (very few params, quick on CPU)
# -------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=kernel,
                padding=padding,
                groups=in_ch,
                bias=False,
            ),  # depthwise
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),  # pointwise
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SmallCNNFast(nn.Module):
    """Tiny, very fast model for quick experiments on CPU."""

    def __init__(self, in_channels=preprocess.NUM_CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            DepthwiseSeparableConv(in_channels, 32),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
            DepthwiseSeparableConv(32, 64),
            nn.MaxPool2d(2),  # 4x4 -> 2x2
            DepthwiseSeparableConv(64, 128),  # 2x2
            nn.AdaptiveAvgPool2d(1),  # global pooling -> 1x1
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# -------------------------
# 2) MIDDLE: compact residual network (good accuracy / cost trade-off)
# -------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class MediumResNet(nn.Module):
    """Compact ResNet-like model — decent capacity while still OK on CPU."""

    def __init__(self, in_channels=preprocess.NUM_CHANNELS):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(ResBlock(64, 64), ResBlock(64, 64))
        self.layer2 = nn.Sequential(ResBlock(64, 128, stride=2), ResBlock(128, 128))
        self.layer3 = nn.Sequential(ResBlock(128, 256, stride=2), ResBlock(256, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.head(x).squeeze(-1)


# -------------------------
# 3) SLOWER: deeper bottleneck + lightweight SE (more representational power)
# -------------------------
class SEBlock(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expansion=4):
        super().__init__()
        mid = out_ch // expansion
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.net(x) + self.skip(x))


class LargeResNetSE(nn.Module):
    """Deeper bottleneck-style model with SE modules. More params but still modest."""

    def __init__(self, in_channels=preprocess.NUM_CHANNELS):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(Bottleneck(64, 256), Bottleneck(256, 256))
        self.se1 = SEBlock(256)
        self.layer2 = nn.Sequential(
            Bottleneck(256, 512, stride=2), Bottleneck(512, 512)
        )
        self.se2 = SEBlock(512)
        self.layer3 = nn.Sequential(
            Bottleneck(512, 1024, stride=2), Bottleneck(1024, 1024)
        )
        self.se3 = SEBlock(1024)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.se1(x)
        x = self.layer2(x)
        x = self.se2(x)
        x = self.layer3(x)
        x = self.se3(x)
        x = self.pool(x)
        return self.head(x).squeeze(-1)
