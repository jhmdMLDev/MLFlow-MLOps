import os

import numpy as np
import cv2
import torch

from .model import ResNet
from .config import inf_config


def test(
    data_dir,
    is_left=True,
    model_path=r"/slo_lr_detection/checkpoint/Resnet_LR/RESNET_LR_epoch_41_model.pt",
):
    # config
    cfg = inf_config()
    transform = cfg.TRANSFORM

    # model
    model = ResNet(cfg).to(cfg.DEVICE)

    # load weights
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=cfg.DEVICE,
        )
    )

    # data
    img_list = os.listdir(data_dir)

    tp = 0
    for img_path in img_list:
        img = cv2.imread(os.path.join(data_dir, img_path), 0)
        x = transform(img).unsqueeze(0)
        y = model(x.to(cfg.DEVICE))
        is_left_pred = torch.argmax(y, dim=-1) == 0
        if is_left_pred.item() == is_left:
            tp += 1
        else:
            print(img_path + " IS PREDICTED INCORRECTLY")

    return tp / len(img_list)


if __name__ == "__main__":
    acc = test(r"data\val\R", is_left=False)
    print(acc)
