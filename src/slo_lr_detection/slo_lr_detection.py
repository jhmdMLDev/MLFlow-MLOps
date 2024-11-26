"""Main inference function
"""

import torch

from slo_lr_detection.train.model import ResNet
from slo_lr_detection.config import inf_config


def slo_ml_is_left(
    img,
    model_path,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
):
    """The inference funtion for left/right detection

    Args:
        img (np.array): image
        model_path (string): path to model.pt file to load trained weights
        device (_type_, optional): The device depending on the device.
        Defaults to torch.device("cuda:0" if torch.cuda.is_available() else "cpu").

    Returns:
        bool: True for left and False for right.
    """
    # configuration for inference
    cfg = inf_config()
    transform = cfg.TRANSFORM

    # model
    model = ResNet(cfg).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # inference
    x = transform(img).unsqueeze(0)

    # inference
    y_pred = model(x.to(device))
    prediction = torch.argmax(y_pred, dim=-1)

    return bool(prediction == 0)
