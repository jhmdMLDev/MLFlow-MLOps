import argparse
from src.slo_lr_detection.train.train import train


def parse_args():
    parser = argparse.ArgumentParser(description="Train a SLOLR model")
    parser.add_argument(
        "--data_path",
        type=str,
        help="Data main directory",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/config.yaml",
        help="config directory",
    )
    parser.add_argument(
        "--check_path",
        type=str,
        help="Result check path",
    )
    # Model and training hyperparameters
    parser.add_argument(
        "--img_size",
        type=int,
        help="Image size for training and validation",
    )
    parser.add_argument(
        "--pretrained",
        type=bool,
        help="Use a pretrained model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        help="GPU ID to use for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        help="Momentum for optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        help="Weight decay for optimizer",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for training",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        help="Batch size for validation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--id",
        type=str,
        help="Experiment ID or model version identifier",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
