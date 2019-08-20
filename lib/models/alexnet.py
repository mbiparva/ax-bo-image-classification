from utils.config import cfg

from torchvision.models import AlexNet
import torch.nn as nn


class AlexNetSmall(AlexNet):
    top_fea_size = 4

    def __init__(self):
        super().__init__(num_classes=cfg.NUM_CLASSES)

        self.features[0] = nn.Conv2d(1 if cfg.DATASET_NAME == 'MNIST' else 3, 64, kernel_size=11, stride=1, padding=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier[1] = nn.Linear(256, 4096)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexnet():
    model = AlexNetSmall()
    return model
