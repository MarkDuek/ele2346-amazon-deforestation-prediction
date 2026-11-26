"""Main module for Amazon deforestation prediction training pipeline.

This module provides the main entry point for training deep learning models
on the Amazon deforestation dataset.
"""

import json
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchinfo import summary

from src.data.amazon_dataset import AmazonDataset
from src.loss.bce_plus_iou import WeightedBCEWithLogitsAndIoULoss
from src.models.tiny_3d_unet.model import Tiny3DUNet
from src.training.trainer import Trainer
from src.utils.data_utils import inspect_h5_file
from src.utils.training_utils import create_directories, plot_training_curves
from src.utils.utils import get_device, load_config, parse_args


def main():
    """Main function to run the Amazon deforestation
    prediction training pipeline."""

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    args = parse_args()
    config = load_config(args.config)
    logger.info("Loaded config:\n%s", json.dumps(config, indent=2))
    create_directories(config)

    # Set random seeds for reproducibility
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device = get_device(config["device"])
    val_ratio = config["data"]["val_ratio"]

    inspect_h5_file(config["data"]["h5_paths"]["input"])
    inspect_h5_file(config["data"]["h5_paths"]["target"])

    model = Tiny3DUNet()

    logger.info("Model summary:")
    summary(
        model,
        input_size=(
            2,
            len(config["data"]["paths"]),
            config["data"]["time_slice"] - 1,
            config["data"]["patch_size"],
            config["data"]["patch_size"],
        ),
    )

    amazon_dataset = AmazonDataset(config)
    logger.info("Dataset input shape: %s", amazon_dataset[0][0].shape)
    logger.info("Dataset target shape: %s", amazon_dataset[0][1].shape)

    generator = torch.Generator().manual_seed(seed)
    train_data, val_data = random_split(
        amazon_dataset, [(1 - val_ratio), val_ratio], generator=generator
    )

    logger.info("Creating Train loader")
    train_loader = DataLoader(
        train_data,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,  # Use multiple workers for faster data loading
        pin_memory=True,  # Pin memory for faster GPU transfers
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,  # Prefetch batches
    )
    logger.info("Creating Validation loader")
    val_loader = DataLoader(
        val_data,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,  # Use multiple workers for faster data loading
        pin_memory=True,  # Pin memory for faster GPU transfers
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,  # Prefetch batches
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    logger.info("Optimizer summary: %s", optimizer)

    loss_fn = WeightedBCEWithLogitsAndIoULoss(config)
    logger.info("Loss function summary: %s", loss_fn)

    trainer = Trainer(
        model, train_loader, val_loader, optimizer, loss_fn, device, config
    )

    logger.info("%s Starting training %s", 16 * "=", 16 * "=")
    trained_model, train_losses, val_losses = trainer.train()
    logger.info("%s Training completed %s", 16 * "=", 16 * "=")

    plot_training_curves(
        train_losses, val_losses, config["save_paths"]["plot"]
    )

    logger.info("%s Starting evaluation %s", 16 * "=", 16 * "=")
    _ = trainer.evaluate(trained_model, val_loader)
    logger.info("Validation set evaluation completed.")
    logger.info("%s Evaluation completed %s", 16 * "=", 16 * "=")


if __name__ == "__main__":
    main()
