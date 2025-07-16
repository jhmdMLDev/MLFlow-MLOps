"""Model structure for ml left and right detection for SLO.
"""

import os
import sys

import torch
from torch import nn
from torchvision import models

from .config import inf_config
from .train.model import ResNet


def get_traced_model(save_path, model_path):
    """This function saves the model as jit file for cpp use.

    Args:
        save_path (str): save path.
        model_path (str, optional): Model path. Defaults to r"/efs/Model_checks/KD/KD_W15epoch_35_model.pt".
    """
    cfg = inf_config()

    model = ResNet(cfg)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    # An example input you would normally provide to your model's forward() method.
    example = torch.rand(1, 1, 512, 512)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)

    traced_script_module.save(save_path + "/lr_detection.pt")


if __name__ == "__main__":
    get_traced_model(
        r"path/to/save",
        r"path/to/your/model",
    )
