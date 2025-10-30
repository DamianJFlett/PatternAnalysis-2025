"""
In essence, does the same as  running train.py to get the model, and then passes that to predict functions.
"""
from train import train, test, create_plots, dice_score
from torch.utils.data import DataLoader, Dataset
from dataset import get_datasets_and_data_loaders
import torch
import argparse


def main():
    pass

if __name__ == "__main__":
    main()