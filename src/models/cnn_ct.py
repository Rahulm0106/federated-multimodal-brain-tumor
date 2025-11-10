import torch
import torch.nn as nn
from torchvision import models

class CNN_CT(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN_CT, self).__init__()
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')  # Pretrained on ImageNet
        self.backbone.fc = nn.Identity()  # Remove final FC layer, output 512 features
        self.proxy_head = nn.Linear(512, num_classes)  # Proxy head for classification

    def forward(self, x):
        features = self.backbone(x)
        logits = self.proxy_head(features)
        return logits

    def extract_features(self, x):
        return self.backbone(x)  # For fusion

# Basic forward test
if __name__ == "__main__":
    model = CNN_CT()
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch=1, RGB, 224x224
    output = model(dummy_input)
    print(output.shape)  # Should be [1, 4]