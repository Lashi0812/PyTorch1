"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
from torchvision import transforms
from torchmetrics import Accuracy
from timeit import default_timer as timer

import model
import data_setup
import engine
import utils

if __name__ == '__main__':

    # setup hyperparameter
    NUM_EPOCHS = 5
    BATCH_SIZE = 32
    HIDDEN_UNITS = 10
    LEARNING_RATE = 0.001

    # set the directories
    train_dir = "../data/pizza_steak_sushi/train"
    test_dir = "../data/pizza_steak_sushi/test"

    # set up device-agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create transformer
    data_transformer = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor()
    ])

    # Create dataloader adn get class names
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transformer,
        batch_size=BATCH_SIZE
    )

    # create the model
    model_0 = model.TinyVGG(input_shape=3,
                            hidden_units=HIDDEN_UNITS,
                            output_shape=len(class_names)).to(device)

    # setup loss ,optimizer and Metric
    cross_entropy_loss_fn = torch.nn.CrossEntropyLoss()
    adam_optimizer = torch.optim.Adam(model_0.parameters(),
                                      lr=LEARNING_RATE)
    accuracy_fn = Accuracy(task="multiclass",
                           num_classes=len(class_names))

    # start timer
    start_time = timer()

    # start training using engine module
    engine.train(model=model_0,
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 loss_fn=cross_entropy_loss_fn,
                 optimizer=adam_optimizer,
                 metric_fn=accuracy_fn,
                 epochs=NUM_EPOCHS,
                 device=device)

    # end timer
    end_timer = timer()
    print(f"[INFO] Total training time : {end_timer - start_time:.3f} seconds")

    # save the model to file
    utils.save_model(model=model_0,
                     target_dir="../models",
                     model_name="using_modular.pth")
