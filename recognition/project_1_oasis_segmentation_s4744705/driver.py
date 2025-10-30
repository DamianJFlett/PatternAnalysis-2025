"""
In essence, does the same as  running train.py to get the model, and then passes that to predict functions.
"""
from train import train, test
from predict import make_predictions
from dataset import get_datasets_and_data_loaders
import torch
import argparse
from modules import ImprovedUnet


def main():
    """
    Completes the full train, test, predict pipeline. Note that if not train_new, new plots will not be created as those are created during the training process.
    Also worth noting that if the full training pipeline is completed, testing and prediction will be done on the model that had the highest dice score during training, 
    but otherwise will be done on the model after every epoch is run. 
    """
    parser = argparse.ArgumentParser(description = "Train and Test the model")
    parser.add_argument("--batch-size", type = int, default = 2, help = "Batch Size used in training")
    parser.add_argument("--epochs", type = int, default = 20, help = "Number of Epochs to train for")
    parser.add_argument("--lr", type = float, default = 1e-4, help = "Learning Rate used in training")
    parser.add_argument("--dropout-prob", type = float, default = 0.3, help = "Dropout probability in context modules")
    parser.add_argument("--plot", type = int, default = 1, help = "Decides whether to make and save plots or not" )
    parser.add_argument("--dir", type = str, default = "images/", help = "Where to save images for the report" )
    parser.add_argument("--seed", type = int, default = 1, help = "The seed to use when choosing what to make predictions on" )
    parser.add_argument("--train-new", type = int, default = 0, help = "True if you want to train the model, False if you wan tto use the model saved in best_model.pth" )
    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    dropout_prob = args.dropout_prob
    plot = args.plot
    dir = args.dir
    seed = args.seed
    train_new = args.train_new
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, train_loader, test_set, test_loader, _, validation_loader = get_datasets_and_data_loaders(batch_size)
    if train_new:
        model = ImprovedUnet(dropout_prob=dropout_prob).to(device)
        train(model, train_loader, validation_loader, epochs = epochs, lr = lr, batch_size=batch_size, device = device, plot = plot)
    else:
        model = ImprovedUnet(dropout_prob=dropout_prob).to(device)
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        model.eval()
    test(model, test_loader, test_set, device = device)
    make_predictions(model, dir = dir, seed = seed, device = device)

if __name__ == "__main__":
    main()