from train import train, test, create_plots, dice_score
from torch.utils.data import DataLoader, Dataset
from dataset import get_datasets_and_data_loaders
import torch
import torch.nn as nn
from modules import ImprovedUnet

def make_predictions(model: ImprovedUnet, dir: str = "images/", seed: int = 1):
    """
    Makes and saves to the specified directory predictions on a random susbet of the test, training, and validation sets
    model: The model to make predictions with
    dir: Directory to save predicted figures to 
    seed: The random seed with which to choose these subset
    """
    pass