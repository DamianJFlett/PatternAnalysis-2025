# Only to be used as a module!
from torch.utils.data import Dataset
from dataset import get_datasets_and_data_loaders
import torch
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
    save_predictions(model, train_dataset, samples_train_indices, "training_preds_sample.png", dir, device)
    save_predictions(model, test_dataset, samples_test_indices, "test_preds_sample.png", dir, device)
    save_predictions(model, validation_dataset, samples_validation_indices, "validation_preds_sample.png", dir, device)


def save_predictions(model: ImprovedUnet, dataset: Dataset, indices: list[int], filename: str, dir: str = "images/",  device = None) -> None:
    """
    Saves predictions for the particular dataset given compared to original image and ground truth masks. 
    model: model to make predictions on
    dataset: data to make predictions on
    indices: indices in the dataset to make predictions on
    filename: name to save files as EXCLUDING DIRECTORY (will be indexed for multiple files)
    dir: directory to save files to
    device: device to generate masks on
    """
    for i, index in enumerate(indices):
        image, mask = dataset[index]
        image = image.unsqueeze(0).to(device)  # add batch dimension so model accepts it
        with torch.no_grad():
            # make prediction mask
            pred = torch.sigmoid(model(image))
        # move to cpu
        pred = pred.squeeze().cpu().numpy()
        mask = mask.squeeze().cpu().numpy()
        image = image.squeeze().cpu().numpy()

        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.title("Input")
        plt.imshow(image, cmap="gray")
        plt.subplot(1,3,2)
        plt.title("Ground Truth")
        plt.imshow(mask, cmap="gray")
        plt.subplot(1,3,3)
        plt.title("Prediction")
        plt.imshow(pred, cmap="gray")
        plt.savefig(f"{dir}/{filename}_{i}.png")
        plt.close()