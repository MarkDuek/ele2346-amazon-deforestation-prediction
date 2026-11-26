"""Small utilities used by the training entry point."""

import logging
import os
from typing import Any, Dict, List


def plot_training_curves(
    train_losses: List[float], val_losses: List[float], save_path: str
) -> None:
    """Save training and validation loss curves."""
    logger = logging.getLogger(__name__)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot generation")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info("Training curves saved to: %s", save_path)


def create_directories(config: Dict[str, Any]) -> None:
    """Create parent directories for configured output files."""
    for path in config.get("save_paths", {}).values():
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
