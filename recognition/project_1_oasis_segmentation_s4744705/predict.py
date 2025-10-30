# Only to be used as a module!
from train import train, test, create_plots, dice_score
from torch.utils.data import DataLoader, Dataset
from dataset import get_datasets_and_data_loaders
import torch
import torch.nn as nn
from modules import ImprovedUnet
import random
import matplotlib.pyplot as plt

def make_predictions(model: ImprovedUnet, dir: str = "images/", seed: int = 1, device = None) -> None:
    """
    Makes and saves to the specified directory predictions on a random susbet of the test, training, and validation sets
    model: The model to make predictions with
    dir: Directory to save predicted figures to 
    seed: The random seed with which to choose these subset
    device: Device to make predictions on
    """
    train_dataset, _, test_dataset, _, validation_dataset, _ = get_datasets_and_data_loaders(1)

    random.seed(seed)
    samples_train_indices = random.sample(range(len(train_dataset)), 4)
    samples_test_indices = random.sample(range(len(test_dataset)), 4)
    samples_validation_indices = random.sample(range(len(validation_dataset)),4)

    model.eval()  
    save_predictions(model, train_dataset, samples_train_indices, "training_preds_sample.png", dir, seed, device)
    save_predictions(model, test_dataset, samples_test_indices, "test_preds_sample.png", dir, seed, device)
    save_predictions(model, validation_dataset, samples_validation_indices, "validation_preds_sample.png", dir, seed, device)


def save_predictions(model: ImprovedUnet, dataset: Dataset, indices: list[int], filename: str, dir: str = "images/", seed: int = 1, device = None) -> None:
    pass