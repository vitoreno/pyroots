import argparse
import copy
import datetime
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch import nn
from pathlib import Path

from src.NetworkModels import BinaryNet, Net
from src.RootUtils import *



transform = T.Compose(
        [
            T.ToTensor(),
            T.RandomRotation(45),
            T.RandomVerticalFlip(),
            T.RandomHorizontalFlip(),
            T.RandomAdjustSharpness(2),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            T.RandomAutocontrast(),
            # T.RandomEqualize(),
            T.Normalize((0.5, 0.5, 0.5), (0.1, 0.1, 0.1)),
        ]
    )

def split_data(
        dataset_path: Path,
        train_perc: float = 0.7,
        transform: T.Compose = T.Compose([])):
    """ Suddivide il dataset originario in training e validazione.

    Args:
        dataset_path: path del dataset
        train_perc: percentuale di dati di training
        transform: insieme di augmentation applicate
    Restituisce:
        loader per dati di training e validazione
    """
    orig_set = torchvision.datasets.ImageFolder(dataset_path, transform) # Original dataset con le trasformazioni fatte alle immagini
    train_set, val_set = TrainValSplit(orig_set, train_perc) #Divie l'original dataset in altri due set secondo le percentuali definite prima
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_loader, val_loader


def training_network(
        n_train_epochs: int = 20,
        recap_file: str = "recap.md"):
    """ Crea la rete di addestramento.
    Args:
        n_train_epochs: Numero di epoche
        recap_file: Directory del file di recap da creare
    Restituisce:
        Calcolati e stampati i valori della funzione LogisticLoss li salva nel file recap.md nella directory scelta dall'utente
    """
    
    net = BinaryNet()
    
    if not args.bTrain:
        model_fn = args.model_fn
        net.load_state_dict(torch.load(model_fn))
    net.to(device)

    criterion = nn.BCEWithLogitsLoss()  # nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    if args.bTrain:
        for epoch in range(n_train_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                # get the inputs; data is a list of [inputs, labels]
                # inputs, labels = data
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels.reshape(-1, 1).float())
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                    running_loss = 0.0
        print("Finished Training")

        now = datetime.datetime.now()
        model_fn = now.strftime("%Y%m%d_%H%M%S")
        torch.save(net.state_dict(), f".\\models\\{model_fn}.pth")
        with open(recap_file, "a") as f:
            f.write(f"\nmodel: {model_fn} args: {str(args)}\n") ##Viene salvato il file recap.md con i parametri di addestramento 



def validation_network():
    """ Crea la rete di validazione."""
    
    if args.bValidate:
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        correct_threshold = 0.2
        with torch.no_grad():
            for data in val_loader:
                images, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                # _, predicted = torch.max(outputs.data, 1)
                predicted = torch.sigmoid(outputs)
                total += labels.size(0)
                correct += (
                    (torch.abs(predicted - labels.reshape(-1, 1)) < correct_threshold)
                    .sum()
                    .item()
                )

        print(
            f"Accuracy of the network on the {total} test images: {100 * correct // total} %"
        )







def main(args): #Funzione principale dello script: RICHIAMA TUTTE LE ALTRE
    RNG_SEED = 42
    random.seed(RNG_SEED)

    
    train_loader, val_loader = split_data(args.DATASET_PATH, transform=transform) ##Richiamo i risultati della funzione spit_data

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") ##Controlla che sia disponibile la GPU e la va a richiamare con CUDA

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrainRootClassifier")
    parser.add_argument(
        "--train",
        dest="bTrain",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train model",
    )
    parser.add_argument(
        "--val",
        dest="bValidate",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate model",
    )
    parser.add_argument(
        "--dataset", dest="DATASET_PATH", default=".\\data", help="Dataset path"
    )
    parser.add_argument(
        "--batch_size", dest="batch_size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--model_fn", dest="model_fn", default="model.pth", help="Model filename"
    )
    parser.add_argument(
        '--train_perc', 
        type=float,
        default=0.7 
        help='Percentage of data for training', 
        required=True
    )
    parser.add_argument(
        "--n_train_epochs", 
        type=int, 
        default=20
        help="Number of epochs for training", 
        required=True
    )
    parser.add_argument(
        "--recap_file",
        type=str, 
        default="recap.md", 
        help="Path to the recap file."
    )

    args = parser.parse_args()
    

    main(args)

    



