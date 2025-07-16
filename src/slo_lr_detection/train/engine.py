import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def train_per_epoch(
    train_loader, val_loader, model, optimizer, loss_fn, epoch, config, logger
):
    """train pipeline

    Args:
        train_loader (torch.utils.data.DataLoader): train data loader
        val_loader (torch.utils.data.DataLoader):  val data loader
        model (nn.Module): model
        optimizer (torch.optim): optimizer
        loss_fn (function/nn.Module,...): loss function
        epoch (int): epoch number
        config (ml_collection.ConfigDict): configuration dictionary
    """
    loop = tqdm(train_loader, leave=True)
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    print(f" epoch: {epoch} out of {config.EPOCHS}")

    for batch_idx, (x_batch, y_batch) in enumerate(loop):
        # initializations
        model.train()
        optimizer.zero_grad()

        # forward pass
        x_batch, y_batch = x_batch.float().to(config.DEVICE), y_batch.to(config.DEVICE)
        out = model(x_batch)
        loss = loss_fn(out, y_batch)

        # back prop
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        # train accuracy
        out_labels = torch.argmax(out, dim=-1)
        y_batch_labels = torch.argmax(y_batch, dim=-1)

        acc = torch.sum(out_labels == y_batch_labels) / out.shape[0]
        train_acc.append(acc.item())

        # validation
        if batch_idx == len(train_loader) - 1:
            logger.info("Validation loop started!")
            for x_batch_val, y_batch_val in val_loader:
                model.eval()
                x_batch_val, y_batch_val = x_batch_val.float().to(
                    config.DEVICE
                ), y_batch_val.to(config.DEVICE)
                # forward pass
                with torch.no_grad():
                    y_pred = model(x_batch_val)
                    loss_val = loss_fn(y_pred, y_batch_val)

                val_loss.append(loss_val.item())

                # accuracy measurement
                y_pred_labels = torch.argmax(y_pred, dim=-1)
                y_batch_val_labels = torch.argmax(y_batch_val, dim=-1)

                acc = torch.sum(y_pred_labels == y_batch_val_labels) / y_pred.shape[0]
                val_acc.append(acc.item())

                # update progress bar
                loop.set_postfix(
                    train_loss=np.mean(train_loss),
                    train_acc=np.mean(train_acc),
                    val_loss=np.mean(val_loss),
                    val_acc=np.mean(val_acc),
                )
            logger.info("Validation loop ended!")

        else:
            # update progress bar
            loop.set_postfix(
                train_loss=np.mean(train_loss),
                train_acc=np.mean(train_acc),
                val_loss="--",
                val_acc="--",
            )
    return np.mean(train_loss), np.mean(train_acc), np.mean(val_loss), np.mean(val_acc)


def model_chkpnts(model, config):
    """This function saves the trained models

    Args:
        model (nn.Module): _description_
        config (ml_collection.ConfigDict): _description_
        epoch (int): _description_
    """

    check_point = config.ID + "_model.pt"
    save_path = os.path.join(config.CHECK_PATH, check_point)
    output_save = open(save_path, mode="wb")
    torch.save(model.state_dict(), output_save)


def loss_per_epoch(train_loss, train_acc, val_loss, val_acc, epoch, config):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(list(range(1, epoch + 1)), train_loss)
    plt.plot(list(range(1, epoch + 1)), val_loss)
    plt.legend(["train loss", "validation loss"])
    plt.title("loss per epoch")
    plt.xlabel("epochs")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(list(range(1, epoch + 1)), train_acc)
    plt.plot(list(range(1, epoch + 1)), val_acc)
    plt.legend(["train acc", "validation acc"])
    plt.title("acc per epoch")
    plt.xlabel("epochs")
    plt.ylabel("acc")

    plt.savefig(os.path.join(config.CHECK_PATH, config.ID + "_acc_loss_per_epoch.png"))
