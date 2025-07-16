import os
from datetime import datetime

import mlflow
import torch
import torch.optim as optim
from torchsummary import summary
import hypertune

from .config import train_config, val_config
from .model import ResNet
from .dataset import get_lr_dataset
from .engine import train_per_epoch, model_chkpnts, loss_per_epoch
from .logger import setup_logger


def update_config(args, train_cfg, val_cfg):
    """
    Updates train and validation configurations based on parsed arguments.

    Args:
        args: Parsed command-line arguments.
        train_cfg: Configuration dictionary or object for training.
        val_cfg: Configuration dictionary or object for validation.
    """
    # Update paths based on the provided data path
    if args.data_path:
        train_cfg.DATA_DIR = os.path.join(args.data_path, "train")
        val_cfg.DATA_DIR = os.path.join(args.data_path, "val")

    # Update other configurations based on arguments if present
    if args.check_path:
        train_cfg.CHECK_PATH = args.check_path

    if args.img_size:
        train_cfg.IMG_SIZE = args.img_size

    if args.pretrained is not None:
        train_cfg.PRETRAINED = args.pretrained

    if args.epochs:
        train_cfg.EPOCHS = args.epochs

    if args.gpu_id is not None:
        train_cfg.GPU_ID = args.gpu_id

    if args.learning_rate:
        train_cfg.LEARNING_RATE = args.learning_rate

    if args.momentum:
        train_cfg.MOMENTUM = args.momentum

    if args.weight_decay:
        train_cfg.WEIGHT_DECAY = args.weight_decay

    if args.batch_size:
        train_cfg.BATCH_SIZE = args.batch_size

    if args.num_workers:
        train_cfg.NUM_WORKERS = args.num_workers

    if args.id:
        train_cfg.ID = args.id

    # Validation-specific configurations
    if args.val_batch_size:
        val_cfg.BATCH_SIZE = args.val_batch_size

    return train_cfg, val_cfg


def train(args):
    # Load configurations and update with args
    train_cfg = train_config(args.config_path)
    val_cfg = val_config(args.config_path)

    # update args
    train_cfg, val_cfg = update_config(args, train_cfg, val_cfg)

    experiment = datetime.now().strftime("%Y.%m.%d.%H.%M.%S.%f")

    # Set the experiment with a custom artifact location
    ml_flow_path = os.path.join(train_cfg.CHECK_PATH, "mlruns")
    mlflow.set_tracking_uri(ml_flow_path)
    mlflow.set_experiment(experiment_name=experiment)

    # update train check path to the experiment
    train_cfg.CHECK_PATH = os.path.join(train_cfg.CHECK_PATH, experiment)
    os.makedirs(train_cfg.CHECK_PATH, exist_ok=True)

    logger = setup_logger(
        "Performance Logger", os.path.join(train_cfg.CHECK_PATH, "train_log.json")
    )

    logger.info("Logger Created!")

    logger.info(f"DEVICE: {train_cfg.DEVICE}")

    with mlflow.start_run():
        # Log configuration parameters
        mlflow.log_params(
            {
                "epochs": train_cfg.EPOCHS,
                "batch_size": train_cfg.BATCH_SIZE,
                "learning_rate": train_cfg.LEARNING_RATE,
                "momentum": train_cfg.MOMENTUM,
                "weight_decay": train_cfg.WEIGHT_DECAY,
            }
        )

        # load the model
        model = ResNet(train_cfg).to(train_cfg.DEVICE)
        summary(model, (1, train_cfg.IMG_SIZE, train_cfg.IMG_SIZE))

        logger.info("Model Summary Done!")

        # optimizer and loss
        optimizer = optim.SGD(
            model.parameters(),
            lr=train_cfg.LEARNING_RATE,
            momentum=train_cfg.MOMENTUM,
            weight_decay=train_cfg.WEIGHT_DECAY,
        )

        loss_fn = torch.nn.CrossEntropyLoss()

        logger.info("Loss and Optimzers are set!")

        # datasets
        dataset_train = get_lr_dataset(train_cfg)
        dataset_val = get_lr_dataset(val_cfg)

        logger.info("Dataset prepared!")

        # data loader
        train_loader = torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size=train_cfg.BATCH_SIZE,
            num_workers=train_cfg.NUM_WORKERS,
            shuffle=True,
            drop_last=True,
        )

        val_loader = torch.utils.data.DataLoader(
            dataset=dataset_val,
            batch_size=val_cfg.BATCH_SIZE,
            num_workers=val_cfg.NUM_WORKERS,
            shuffle=True,
            drop_last=True,
        )

        logger.info("Data Loader prepared!")

        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []

        for epoch in range(1, train_cfg.EPOCHS + 1):
            logger.info(f"epoch {epoch} started!")
            # train
            train_loss, train_acc, val_loss, val_acc = train_per_epoch(
                train_loader,
                val_loader,
                model,
                optimizer,
                loss_fn,
                epoch,
                train_cfg,
                logger,
            )
            logger.info(f"epoch {epoch} train done!")

            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

            # Log metrics to MLflow
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                step=epoch,
            )

            # save model
            model_chkpnts(model, train_cfg)
            loss_per_epoch(
                train_loss_list,
                train_acc_list,
                val_loss_list,
                val_acc_list,
                epoch,
                train_cfg,
            )
            mlflow.pytorch.log_model(model, "slo_lr_detection")
            logger.info(
                f"epoch {epoch} checkpoints are saved to the {train_cfg.CHECK_PATH}!"
            )

        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag="accuracy",
            metric_value=val_acc,
            global_step=train_cfg.EPOCHS,
        )
