import torch
import torchvision
import random
import copy
import datetime

from torch import nn

import torch.optim as optim
import torchvision.transforms as T
from src.NetworkModels import BinaryNet
from src.RootUtils import *

import argparse
from pathlib import Path

import pandas as pd



def main(args):
    RNG_SEED = 42
    random.seed(RNG_SEED)

    TRAIN_PERC = 0.7
    N_TRAIN_EPOCHS = 20

    if args.augment:
        transform = T.Compose([T.ToTensor(),
                                T.RandomRotation(45),
                                T.RandomVerticalFlip(),
                                T.RandomHorizontalFlip(),
                                T.RandomAdjustSharpness(2),
                                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                T.RandomAutocontrast(),
                                T.Normalize((0.5, 0.5, 0.5), (0.1, 0.1, 0.1))])
    else:
        transform = T.Compose([T.ToTensor(),
                               T.Normalize((0.5, 0.5, 0.5), (0.1, 0.1, 0.1))])
    
    orig_set = torchvision.datasets.ImageFolder(args.DATASET_PATH, transform)  # your dataset
    train_set, val_set = TrainValSplit(orig_set, TRAIN_PERC)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = BinaryNet(args.img_width_height)
    if not args.bTrain:
        model_fn = f'.\\models\\{args.model_fn}' #'.\\models\\20220711_165340.pth'
        net.load_state_dict(torch.load(model_fn))
    net.to(device)
    
    criterion = nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    if args.bTrain:
        for epoch in range(N_TRAIN_EPOCHS):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                # get the inputs; data is a list of [inputs, labels]
                #inputs, labels = data
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
                if i % 2000 == 1999: # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
        print('Finished Training')

        now = datetime.datetime.now()
        now_str = now.strftime('%Y%m%d_%H%M%S')
        Path(args.results).mkdir(parents=True, exist_ok=True)
        model_fn = Path(args.results, f'{now_str}.pth')
        torch.save(net.state_dict(), model_fn)
        with open('recap.md', 'a') as f:
            f.write(f"\nmodel: {model_fn} args: {str(args)}\n")

    if args.bValidate:
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        correct_threshold = 0.2
        thr_range = torch.arange(0.1, 1.1, 0.1)
        validation_results = {}
        for idx, correct_threshold in enumerate(thr_range):
            validation_results[idx+1] = {'tn': 0, 'tp': 0, 'fn': 0, 'fp': 0}
        with torch.no_grad():
            for data in val_loader:
                images, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                # _, predicted = torch.max(outputs.data, 1)
                raw_predicted = torch.sigmoid(outputs)
                total += labels.size(0)
                labels = labels.reshape(-1,1)
                #correct += (torch.abs(predicted - labels.reshape(-1,1)) < correct_threshold).sum().item()
                for idx, correct_threshold in enumerate(thr_range):
                    predicted = copy.deepcopy(raw_predicted)
                    predicted[predicted < correct_threshold] = 0
                    predicted[predicted >= correct_threshold] = 1
                    validation_results[idx+1]['tn'] += torch.logical_and(predicted == 0, labels == 0).sum().item()
                    validation_results[idx+1]['tp'] += torch.logical_and(predicted == 1, labels == 1).sum().item()
                    validation_results[idx+1]['fn'] += torch.logical_and(predicted == 0, labels == 1).sum().item()
                    validation_results[idx+1]['fp'] += torch.logical_and(predicted == 1, labels == 0).sum().item()
            
            for idx, correct_threshold in enumerate(thr_range):
                res = validation_results[idx+1]
                try:
                    res['accuracy'] = (res['tn'] + res['tp']) / (res['tn'] + res['tp'] + res['fn'] + res['fp'])
                except Exception:
                    res['accuracy'] = 'N/A'
                try:
                    res['precision'] = res['tp'] / (res['tp'] + res['fp'])
                except Exception:
                    res['precision'] = 'N/A'
                try:
                    res['recall'] = res['tp'] / (res['tp'] + res['fn'])
                except Exception:
                    res['recall'] = 'N/A'
                try:
                    res['F1'] = 2 * (res['precision']*res['recall']) / (res['precision']+res['recall'])
                except Exception:
                    res['F1'] = 'N/A'
        # verificare il funzionamento ed esportare il dataframe su file
        df = pd.DataFrame(validation_results)
        print(df.to_string())
        export_fn = model_fn.replace("models", "validation")[:-4]
        df.to_csv(f'{export_fn}.csv', sep=';', float_format='%.4f')
        df.to_pickle(f'{export_fn}.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TrainRootClassifier')
    parser.add_argument('--train', dest='bTrain', type=bool, action=argparse.BooleanOptionalAction, default=True, help='Train model')
    parser.add_argument('--val', dest='bValidate', type=bool, action=argparse.BooleanOptionalAction, default=True, help='Validate model')
    parser.add_argument('--augment', default=True, type=bool)
    parser.add_argument('--dataset', dest='DATASET_PATH', default='.\\data', help='Dataset path')
    parser.add_argument('--results', '-r', default='results', help='Results path')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model_fn', dest='model_fn', default='model.pth', help='Model filename')
    parser.add_argument('--img_width_height', dest='img_width_height', type=int, default=65, help='Size of width/height of the squared patch')

    args = parser.parse_args()
    main(args)