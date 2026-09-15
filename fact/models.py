"""Models used in the paper: ResNet18 (CIFAR-10), a small CNN (MNIST) and pretrained ResNet50 (HAM10000)."""

import torch
from torch import nn
from torchvision import models


class MnistCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        def block(c_in, c_out, stride=1):
            return [nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1), nn.ReLU(), nn.BatchNorm2d(c_out)]

        self.features = nn.Sequential(
            *block(1, 32), *block(32, 32), *block(32, 32, stride=2), nn.MaxPool2d(2, 2), nn.Dropout(0.25),
            *block(32, 64), *block(64, 64), *block(64, 64, stride=2), nn.MaxPool2d(2, 2), nn.Dropout(0.25),
            *block(64, 128), nn.MaxPool2d(2, 2), nn.Dropout(0.25),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        return torch.log_softmax(self.fc(self.features(x).flatten(1)), dim=1)


def build_model(name, num_classes):
    if name == "cnn":
        return MnistCNN(num_classes)
    if name == "resnet18":
        model = models.resnet18(num_classes=num_classes)
        # CIFAR-sized stem: 3x3 convolution and no max-pooling.
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    if name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V2")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"unknown model {name!r}")
