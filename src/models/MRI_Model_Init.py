import torch
import torch.nn as nn
from torchvision import models


class CNN_MRI(nn.Module):
    def __init__(self, feature_dim=512):
        """
        Initializes the ResNet18-based model for 1-channel MRI images.

        Args:
            feature_dim (int): The dimension of the output feature vector.
                              ResNet18's default is 512.
        """
        super(CNN_MRI, self).__init__()

        # Load a pre-trained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Modify the first convolutional layer for 1-channel (grayscale) input
        # The original conv1 is: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        original_conv1 = resnet.conv1

        # Create a new conv1 with 1 input channel
        self.conv1 = nn.Conv2d(1, 64,
                               kernel_size=original_conv1.kernel_size,
                               stride=original_conv1.stride,
                               padding=original_conv1.padding,
                               bias=original_conv1.bias)

        # To leverage pre-training, we average the weights of the 3 RGB channels
        # and use that as the weight for our new 1-channel layer.
        with torch.no_grad():
            self.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)

        # Replace the original conv1 with our new one
        resnet.conv1 = self.conv1

        # Remove the final fully connected layer (the classifier)
        # We want the 512-dim feature vector *before* the classifier.
        # This list comprehension gets all layers *except* the last one (fc).
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1, H, W]

        Returns:
            torch.Tensor: Output feature vector of shape [batch_size, 512]
        """
        # Pass input through the modified ResNet (all layers except fc)
        x = self.features(x)

        # The output of the ResNet features (after the avgpool) is [batch_size, 512, 1, 1]
        # We flatten it to get our [batch_size, 512] feature vector
        x = torch.flatten(x, 1)

        return x