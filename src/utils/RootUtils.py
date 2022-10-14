import copy
import random

import numpy as np
import torch


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
