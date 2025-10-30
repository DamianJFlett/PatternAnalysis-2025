# Can be run as a script or used as a library w/ the train function

import torch
from modules import ImprovedUnet, DiceLoss
from dataset import get_datasets_and_data_loaders



def train(model, train_loader, test_set, epochs = 20, lr = 1e-4, device = None):
    pass

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Beginning training with device {device}...")
    train_set, train_loader, test_set, test_loader, validation_set, validation_loader = get_datasets_and_data_loaders()
    model = ImprovedUnet(dropout_prob=0.3)
    train(model, train_loader, test_set, epochs = 20, lr = 1e-4, device = device)