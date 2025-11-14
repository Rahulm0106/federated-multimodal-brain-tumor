# src/models/backbone.py
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MobileNetV3Backbone(nn.Module):
    def __init__(self, pretrained=True, freeze=False):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = mobilenet_v3_small(weights=weights)

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 576

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x