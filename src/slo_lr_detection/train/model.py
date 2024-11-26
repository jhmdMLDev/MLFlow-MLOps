import torch
import torch.nn as nn
from torchsummary import summary
import torchvision.models as models


class ResNet(nn.Module):
    """Model"""

    def __init__(self, config) -> None:
        """Init function"""
        super().__init__()
        self.model = models.resnet18(pretrained=config.PRETRAINED)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.model.fc = nn.Linear(in_features=512, out_features=2, bias=True)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward function

        Args:
            x (torch.tensor): input

        Returns:
            torch.tensor: output
        """
        return self.model(x)
