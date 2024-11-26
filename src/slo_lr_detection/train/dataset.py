import os

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from .config import train_config, val_config


def get_lr_dataset(config):
    return LRDataset(config)


class LRDataset(torch.utils.data.Dataset):
    def __init__(self, config) -> None:
        super().__init__()
        self.data_dir = config.DATA_DIR
        self.transform = config.TRANSFORM

        self.folders = os.listdir(self.data_dir)

        self.img_path = []
        self.side = []
        for fold in self.folders:
            if fold == "L":
                left_folder = os.path.join(self.data_dir, fold)
            elif fold == "R":
                right_folder = os.path.join(self.data_dir, fold)

        # append left images
        for img_path in os.listdir(left_folder):
            self.img_path.append(os.path.join(left_folder, img_path))
            self.side.append(True)

        for img_path in os.listdir(right_folder):
            self.img_path.append(os.path.join(right_folder, img_path))
            self.side.append(False)

    def __getitem__(self, idx):
        # load image
        img = cv2.imread(self.img_path[idx], 0)
        is_left = self.side[idx]
        # enhance dataset by inversing the eye
        if np.random.random() > 0.5:
            img = cv2.flip(img, 1)
            is_left = not is_left

        # apply transforms
        input_batch = self.transform(img)

        if is_left:
            output_batch = torch.tensor(np.array([1, 0])).float()
        else:
            output_batch = torch.tensor(np.array([0, 1])).float()

        return input_batch, output_batch

    def __len__(self):
        return len(self.img_path)
