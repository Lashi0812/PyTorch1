"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path
from typing import Dict
from typing import List
import matplotlib.pyplot as plt


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
      model: A target PyTorch model to save.
      target_dir: A directory for saving the model to.
      model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
      save_model(model=model_0,
                 target_dir="models",
                 model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)


def plot_loss_curve(history: Dict[str, List[float]]):
    epochs_range = range(len(history["train_loss"]))

    plt.figure(figsize=(8, 3))

    # Plot the loss
    plt.subplot(121)
    plt.plot(epochs_range, history["train_loss"], label="Training")
    plt.plot(epochs_range, history["test_loss"], label="Testing")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot the loss
    plt.subplot(122)
    plt.plot(epochs_range, history["train_acc"], label="Training")
    plt.plot(epochs_range, history["test_acc"], label="Testing")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
