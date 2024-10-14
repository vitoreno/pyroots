# python3 src/TrainRootClassifier.py --dataset data/processed/cracks_65/ --model_fn cracks_65.pth
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

# from src.NetworkModels import BinaryNet, Net
# from src.RootUtils import *
from torch import nn, flatten
import torch.nn.functional as F

def TrainValSplit(orig_set, TRAIN_PERC=0.7):
    """Suddivide un ImageFolder dataset in training e test set
    mediante sottocampionamento uniforme delle istanze delle
    varie classi

    input
    orig_set -- dataset complessivo
    TRAIN_PERC -- percentuale per il training set

    output
    train_set -- pytorch subset per il training
    val_set -- pytorch subset per il validation
    """
    VAL_PERC = 1 - TRAIN_PERC
    # Ricopio i target in un array numpy
    targets = np.array(copy.deepcopy(orig_set.targets), dtype=np.uint8)
    train_idx = []
    val_idx = []
    for class_idx, _ in enumerate(orig_set.classes):
        first_idx = np.where(targets == class_idx)[0][0]
        last_idx = np.where(targets == class_idx)[0][-1]
        class_shuffled_idx = list(range(first_idx, last_idx + 1))
        random.shuffle(class_shuffled_idx)
        threshold = int(len(class_shuffled_idx) * TRAIN_PERC)
        train_idx.extend(class_shuffled_idx[:threshold])
        val_idx.extend(class_shuffled_idx[threshold:])

    train_set = torch.utils.data.Subset(orig_set, train_idx)
    val_set = torch.utils.data.Subset(orig_set, val_idx)
    return train_set, val_set



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.conv3 = nn.Conv2d(32, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(16 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BinaryNet(nn.Module):
    def __init__(self, img_width_height = 65):
        super().__init__()
        self.conv_layers = 3
        self.img_width_height = img_width_height
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.conv3 = nn.Conv2d(32, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(16 * self.__get_linear_size(), 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def __get_linear_size(self):
        size = self.img_width_height
        for i in range(self.conv_layers):
            size = int((size-2)/2)
        return size**2


def main(args):
    RNG_SEED = 42
    random.seed(RNG_SEED)

    # args.DATASET_PATH = "C:\\Users\\vitor\\Desktop\\bin_class_65"
    TRAIN_PERC = 0.7
    N_TRAIN_EPOCHS = 20

    augment = T.Compose(
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

    transform = T.Compose(
        [
            T.ToTensor(),
            # T.RandomRotation(45),
            # T.RandomVerticalFlip(),
            # T.RandomHorizontalFlip(),
            # T.RandomAdjustSharpness(2),
            # T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            # T.RandomAutocontrast(),
            # T.RandomEqualize(),
            T.Normalize((0.5, 0.5, 0.5), (0.1, 0.1, 0.1)),
        ]
    )

    if args.augmentation:
        orig_set = torchvision.datasets.ImageFolder(
            args.DATASET_PATH, augment
        )  # your dataset
    else:
        orig_set = torchvision.datasets.ImageFolder(
            args.DATASET_PATH, transform
        )

    train_set, val_set = TrainValSplit(orig_set, TRAIN_PERC)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = BinaryNet()
    if not args.bTrain:
        model_fn = args.model_fn  #'.\\models\\20220711_165340.pth'
        net.load_state_dict(torch.load(model_fn))
    net.to(device)

    criterion = nn.BCEWithLogitsLoss()  # nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    if args.bTrain:
        for epoch in range(N_TRAIN_EPOCHS):  # loop over the dataset multiple times
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
        with open("models\\recap.md", "a") as f:
            f.write(f"\nmodel: {model_fn} args: {str(args)}\n")

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
        '--augmentation', default=False, type=bool
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

    args = parser.parse_args()
    main(args)
