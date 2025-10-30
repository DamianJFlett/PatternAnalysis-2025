# Can be run as a script (if you just want to train and test the model for hyperparameter tuning etc.) or used as a library w/ the train function

import torch
from modules import ImprovedUnet, DiceLoss
from torch.utils.data import DataLoader, Dataset
from dataset import get_datasets_and_data_loaders
import matplotlib.pyplot as plt
import argparse

# Inspired by lecture code on DICE score
def dice_score(predictions: torch.Tensor, targets: torch.Tensor, smooth = 1e-6):
        """
        Takes a batch of predictions and targets and returns the dice score over those.
        predictions: predicted masks
        targets: actual masks
        smooth: smoothing constant, should be small. Avoids cases where coincidentally might have division by 0
        returns: Dice score
        """
        predictions = torch.sigmoid(predictions)
        # want plain binary prediction, not the probability vector (> 0.5 maps to 1: foreground, <=0.5 maps to 0: background)
        predictions = (predictions > 0.5).float()
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1).float()

        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        dice_coeff = (2.0 * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)

        return dice_coeff.item()

def train(model: ImprovedUnet, train_loader: DataLoader, validation_loader: Dataset, epochs: int = 20, lr: float = 1e-4, batch_size:int = 2, device = None, plot: bool = True) -> ImprovedUnet:
    """
    Trains an improved Unet Model, prints out progress output, validating as we go. Depending on parameters, also plots training loss and dice score over epochs.
    model: Improved UNet to train
    train_loader: data loader for training data
    validation_loader: data loader for validation data
    epochs: number of epochs to run training on
    lr: learning rate
    batch_size: batch size for training
    device: device to conduct training on
    plot: boolean indicating whether to draw and save new plots or not after training
    returns: trained model
    """
    print(f"Beginning training with device {device}...")
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_dice = 0
    epoch_losses, dices = [], []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (images, masks) in enumerate(train_loader):
            # for each batch
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        # Validate
        model.eval()
        dice = 0
        with torch.no_grad(): # Don't train the model here! 
            for images, masks in validation_loader:
                # Same as ab ove, we are just looking through batches, getting the current output, and evaluating it
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                dice += dice_score(outputs, masks)
        dice_avg = dice/len(validation_loader)
        dices.append(dice_avg)
        print(f"Epoch {epoch + 1} Completed! \nTraining Loss: {avg_loss} | Dice Score on Validation Set: {dice_avg}...")

        # Save best model
        if dice_avg > best_dice:
            best_dice = dice_avg
            torch.save(model.state_dict(), "best_model.pth")
    if plot:
        create_plots(epoch_losses, dices)

    print(f"Training complete. Best Dice score: {best_dice}")
    return model

def create_plots(epoch_losses: list[float], dices: list[float]) -> None:
    """
    Creates plots for the dice scores and training losses
    epoch_losses: array of losses during training
    """

    plt.figure()
    plt.plot(epoch_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig('images/Training_loss.png')
    plt.close()

    plt.figure()
    plt.plot(dices, label="Average Dice Score")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Validation Dice Score")
    plt.legend()
    plt.savefig('images/Dice_Score.png')
    plt.close()



def test(model: ImprovedUnet, test_loader: DataLoader, test_set: Dataset, device = None) -> None:
    """
    Tests an improved UNet model and prints out results
    model: Model to test
    test_loader: loader for testing data
    test_set: dataset for testing data
    device: device to create masks on
    """
    print(f"Beginning testing...")
    model.eval()
    dice_total = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            dice_total += dice_score(outputs, masks)
    avg_dice = dice_total/len(test_loader)
    print(f"Average Dice Similarity score in testing was {avg_dice}")


def main():
    parser = argparse.ArgumentParser(description = "Train and Test the model")
    parser.add_argument("--batch-size", type = int, default = 2, help = "Batch Size used in training")
    parser.add_argument("--epochs", type = int, default = 20, help = "Number of Epochs to train for")
    parser.add_argument("--lr", type = float, default = 1e-4, help = "Learning Rate used in training")
    parser.add_argument("--dropout-prob", type = float, default = 0.3, help = "Dropout probability in context modules")
    parser.add_argument("--plot", type = bool, default = True, help = "Decides whether to make and save plots or not" )
    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    dropout_prob = args.dropout_prob
    plot = args.plot
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, train_loader, test_set, test_loader, _, validation_loader = get_datasets_and_data_loaders(batch_size)
    model = ImprovedUnet(dropout_prob=dropout_prob).to(device)
    train(model, train_loader, validation_loader, epochs = epochs, lr = lr, batch_size=batch_size, device = device, plot = plot)

    test(model, test_loader, test_set, device = device)

if __name__ == "__main__":
    main()


