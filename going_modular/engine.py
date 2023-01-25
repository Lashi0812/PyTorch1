"""
Contains function for training and testing the PyTorch Model
"""

from typing import List, Dict, Tuple

from collections import defaultdict
import numpy as np

import torch
from tqdm.auto import tqdm
from torchmetrics import Metric
from torch.nn.modules.loss import _Loss


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: _Loss,
               optimizer: torch.optim.Optimizer,
               metric_fn: Metric,
               device: str) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
      model: A PyTorch model to be trained.
      dataloader: A DataLoader instance for the model to be trained on.
      loss_fn: A PyTorch loss function to minimize.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      metric_fn: To measure the performance of madel using metric function
      device: A target device to compute on (e.g. "cuda" or "cpu").


    Returns:
      A tuple of training loss and training accuracy metrics.
      In the form (train_loss, train_accuracy). For example:

      (0.1112, 0.8743)
    """
    # put the model into train mode
    model.train()

    # set up the train loss and train accuracy per batch
    train_batch_loss = []
    train_batch_acc = []

    for batch, (X, y) in enumerate(dataloader):
        # put the data into the target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_logits = model(X)
        y_probs = torch.softmax(y_logits, dim=1)

        # 2. Calculate the loss and accuracy
        loss = loss_fn(y_logits, y)
        acc = metric_fn(y_probs, y)

        train_batch_loss.append(loss.item())
        train_batch_acc.append(acc.item())

        # 3. Optimiser zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. optimizer step
        optimizer.step()

    train_loss = np.array(train_batch_loss).mean()
    train_acc = np.array(train_batch_acc).mean()
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: _Loss,
              metric_fn: Metric,
              device: str) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
      model: A PyTorch model to be tested.
      dataloader: A DataLoader instance for the model to be tested on.
      loss_fn: A PyTorch loss function to calculate loss on the test data.
      device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
      A tuple of testing loss and testing accuracy metrics.
      In the form (test_loss, test_accuracy). For example:

      (0.0223, 0.8985)
    """
    # Put the model into eval mode
    model.eval()

    # set up the train loss and train accuracy per batch
    test_batch_loss = []
    test_batch_acc = []

    # Turn on the inference mode
    with torch.inference_mode():
        # loop through dataloader
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_logits = model(X)
            y_probs = torch.softmax(y_logits, dim=1)

            # 2. Calculate the loss and accuracy
            loss = loss_fn(y_logits, y)
            acc = metric_fn(y_probs, y)

            # accumulate the loss and accuracy
            test_batch_acc.append(acc)
            test_batch_loss.append(loss)

    # adjust to metric to average per batch
    test_loss = np.array(test_batch_loss).mean()
    test_acc = np.array(test_batch_acc).mean()
    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: _Loss,
          metric_fn: Metric,
          epochs: int,
          device: str) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
      model: A PyTorch model to be trained and tested.
      train_dataloader: A DataLoader instance for the model to be trained on.
      test_dataloader: A DataLoader instance for the model to be tested on.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      loss_fn: A PyTorch loss function to calculate loss on both datasets.
      metric_fn: To measure the performance of madel using metric function
      epochs: An integer indicating how many epochs to train for.
      device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
      A dictionary of training and testing loss as well as training and
      testing accuracy metrics. Each metric has a value in a list for
      each epoch.
      In the form: {train_loss: [...],
                    train_acc: [...],
                    test_loss: [...],
                    test_acc: [...]}
      For example if training for epochs=2:
                   {train_loss: [2.0616, 1.0537],
                    train_acc: [0.3945, 0.3945],
                    test_loss: [1.2641, 1.5706],
                    test_acc: [0.3400, 0.2973]}
    """
    history = defaultdict(list)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           metric_fn=metric_fn,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        metric_fn=metric_fn,
                                        device=device)

        # Print out what's happening
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

    return history
