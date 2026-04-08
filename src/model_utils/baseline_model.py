import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

BASE_DIR = Path(__file__).resolve().parent.parent
config_path = BASE_DIR / 'model_utils' / 'model_settings.json'
categories_path = BASE_DIR / 'dataset_utils' / 'category_list.txt'

with open(config_path, 'r') as file:
    data = json.load(file)

HEIGHT = data.get('HEIGHT')
WIDTH = data.get('WIDTH')
FRAMES_COUNT = data.get('FRAMES_COUNT')
FRAME_STEP = data.get('FRAME_STEP')
BATCH_SIZE = data.get('BATCH_SIZE')
EPOCHS = data.get('EPOCHS')

with open(categories_path, 'r') as file:
    category_list = file.read().splitlines()
    CLASS_COUNT = len(category_list)


class Conv2Plus1D(nn.Module):
    """
     Custom convolutional layer that decomposes 3D convolution into:
       - spatial convolution (height, width)
       - temporal convolution (time dimension)
     This reduces computation while preserving performance.
   """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.mid_channels = out_channels // 2 if out_channels >= 2 else 1

        self.spatial = nn.Conv3d(  # Extracts spatial features frame-by-frame.
            in_channels,
            self.mid_channels,
            kernel_size=(1, 3, 3),  # (T, H, W)
            padding=(0, 1, 1)  # padding = 0 in time
        )
        self.bn1 = nn.BatchNorm3d(self.mid_channels)

        self.temporal = nn.Conv3d(  # Captures temporal dynamics across frames.
            self.mid_channels,
            out_channels,
            kernel_size=(3, 1, 1),  # (T, H, W)
            padding=(1, 0, 0)  # padding = 1 in time
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:  # Adjusts input channels for next operations.
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.shortcut(x)

        # Spatial features (Conv2D over H/W)
        x = F.relu(self.bn1(self.spatial(x)))
        # Temporal features (Conv1D over time)
        x = self.bn2(self.temporal(x))

        x += identity

        return F.relu(x)


class ActionRecognition(nn.Module):
    """
        Build the video classification model.

        Architecture:
        - ResNet
        - Initial Conv2Plus1D + normalization + activation
        - Multiple residual blocks with downsampling
        - Global pooling and dense classification layer
    """

    OUT_CHANNELS = {  # Maps ResNet variant to feature dimension.
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048
    }

    def __init__(self, num_classes, backbone_type='resnet18'):
        super().__init__()

        if backbone_type == 'resnet18':
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif backbone_type == 'resnet34':
            backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        elif backbone_type == 'resnet50':
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.backbone = backbone

        self.backbone = nn.Sequential(
            *list(self.backbone.children())[:-2])  # Use ResNet layers, but remove detection head and global pooling.
        self.out_channels = self.OUT_CHANNELS.get(backbone_type, 512)

        for param in self.backbone.parameters():
            param.requires_grad = False  # Freeze backbone - no gradient updates.

        self.conv1 = nn.Conv3d(self.out_channels,
                               16,
                               kernel_size=(3, 7, 7),
                               padding=(1, 3, 3))

        self.bn1 = nn.BatchNorm3d(16)

        self.pool = nn.MaxPool3d((1, 2, 2))  # Downsamples spatial dimensions only.

        self.block1 = Conv2Plus1D(16, 16)
        self.block2 = Conv2Plus1D(16, 32)
        self.block3 = Conv2Plus1D(32, 64)
        self.block4 = Conv2Plus1D(64, 128)

        self.gap = nn.AdaptiveAvgPool3d(1)  # Collapse spatial and temporal dimensions - change to (B, C, 1, 1, 1)
        self.fc = nn.Linear(128, num_classes)  # Final classification layer.

    def forward(self, x):
        B, C, T, H, W = x.shape  # (B - batch, C - channels (RGB), T - number of frames, H, W)

        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)  # (B*T, C, H, W) - Flatten time into batch dimension.

        with torch.no_grad():
            feat = self.backbone(x)  # Extract features without training ResNet.

        _, C2, H2, W2 = feat.shape

        feat = feat.view(B, T, C2, H2, W2)
        x = feat.permute(0, 2, 1, 3, 4)  # (B, C2, T, H, W) - Restore temporal structure.

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # Progressive feature extraction + downsampling
        x = self.block1(x)

        x = self.block2(x)
        x = self.pool(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.gap(x)  # Global pooling - (B, C, T, H, W) -> (B, C, 1, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten for linear layer

        return self.fc(x)


def create_model(num_classes=CLASS_COUNT, backbone_type='resnet18'):
    model = ActionRecognition(
        num_classes=num_classes,
        backbone_type=backbone_type
    )
    return model
