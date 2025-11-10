import torch
import torch.nn as nn
from torchvision import models

class CNN_MRI(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN_MRI, self).__init__()
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone.fc = nn.Identity()
        self.proxy_head = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.proxy_head(features)
        return logits

    def extract_features(self, x):
        return self.backbone(x)

# Basic forward test (similar to above)
if __name__ == "__main__":
    model = CNN_MRI()
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch=1, RGB, 224x224
    output = model(dummy_input)
    print(output.shape)  # Should be [1, 4]