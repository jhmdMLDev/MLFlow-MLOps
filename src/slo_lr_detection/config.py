"""Configuration for the model
"""
import ml_collections
from torchvision import transforms


def inf_config():
    """Inference cfg dictionary

    Returns:
        ml_collections.ConfigDict: Dictionary
    """
    # def
    config = ml_collections.ConfigDict()

    # data
    config.PRETRAINED = False
    config.IMG_SIZE = 1000

    # transform
    config.TRANSFORM = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        ]
    )

    return config
