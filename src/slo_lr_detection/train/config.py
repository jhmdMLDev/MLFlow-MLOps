import yaml
import ml_collections
from torchvision import transforms
import torch


def load_config_from_yaml(file_path):
    with open(file_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return config_dict


def train_config(file_path):
    config_data = load_config_from_yaml(file_path)["train"]
    config = ml_collections.ConfigDict()

    # Populate config with values from YAML
    config.DATA_DIR = config_data["DATA_DIR"]
    config.IMG_SIZE = config_data["IMG_SIZE"]
    config.PRETRAINED = config_data["PRETRAINED"]
    config.EPOCHS = config_data["EPOCHS"]
    config.GPU_ID = config_data["GPU_ID"]
    config.DEVICE = torch.device(
        "cuda:" + str(config.GPU_ID) if torch.cuda.is_available() else "cpu"
    )
    config.LEARNING_RATE = config_data["LEARNING_RATE"]
    config.MOMENTUM = config_data["MOMENTUM"]
    config.WEIGHT_DECAY = float(config_data["WEIGHT_DECAY"])
    config.BATCH_SIZE = config_data["BATCH_SIZE"]
    config.NUM_WORKERS = config_data["NUM_WORKERS"]
    config.CHECK_PATH = config_data["CHECK_PATH"]
    config.ID = config_data["ID"]

    # Define transform
    config.TRANSFORM = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transforms.RandomAutocontrast(p=0.5),
            transforms.RandomAffine(degrees=(0, 0), translate=(0, 0), scale=(0.8, 1.2)),
        ]
    )

    return config


def val_config(file_path):
    config_data = load_config_from_yaml(file_path)["val"]
    config = train_config(file_path)  # Start with train config

    # Override with validation-specific settings
    config.DATA_DIR = config_data["DATA_DIR"]
    config.BATCH_SIZE = config_data["BATCH_SIZE"]

    # Define transform for validation
    config.TRANSFORM = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        ]
    )

    return config
