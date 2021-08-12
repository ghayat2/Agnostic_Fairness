import torch
import pandas as pd
from model import *

import numpy as np


def demographic_parity(model, device, image_dataset, min_groups=None):
    """
    Evaluates the accuracy of images from the majority and minority groups from each class
    :param model: the trained model
    :param device: the device on which the model was trained
    :param image_dataset: the image data container
    :param min_groups: the path of images in the minority groups from each class
    :return: a table containing the accuracy of each group
    """
    indices = _get_indices_2(image_dataset, min_groups) if min_groups else _get_indices_1(image_dataset)

    class1_majority = torch.utils.data.Subset(image_dataset, indices=indices[0][0])
    class1_minority = torch.utils.data.Subset(image_dataset, indices=indices[0][1])

    class2_majority = torch.utils.data.Subset(image_dataset, indices=indices[1][0])
    class2_minority = torch.utils.data.Subset(image_dataset, indices=indices[1][1])

    dataloaders = [torch.utils.data.DataLoader(x, batch_size=4, shuffle=True, num_workers=4) for x in
                   [class1_majority, class1_minority, class2_minority, class2_majority]]
    accuracies = [[float(accuracy(model, device, dataloader)) for dataloader in dataloaders[:2]],
                  [float(accuracy(model, device, dataloader)) for dataloader in dataloaders[2:]]]

    return pd.DataFrame(accuracies, index=["Class0", "Class1"], columns=["Group0", "Group1"])


def _get_indices_1(image_set, num_labels=2, num_protected=2):
    """
    Gets the indices of the minority group images in the image_set container
    :param image_set: the image container from which to get the indices from
    :param num_labels: the number of labels
    :param num_protected: the number of demographic groups
    :return: the indices corresponding to images in different demographic groups
    """
    indices = [[[] for _ in range(num_protected)] for _ in range(num_labels)]
    for _, label, cluster, index in image_set:
        indices[label][cluster].append(index)

    return indices


def _get_indices_2(image_set, min_groups, num_labels=2, num_protected=2):
    """
    Gets the indices of the minority group images in the image_set container
    :param image_set: the image container from which to get the indices from
    :param min_groups: the paths of the images in the minority group
    :param num_labels: the number of labels
    :param num_protected: the number of demographic groups
    :return: the indices corresponding to images in different demographic groups
    """
    indices = [[[] for _ in range(num_protected)] for _ in range(num_labels)]
    for i, (_, label, _, index) in enumerate(image_set):
        indices[label][int(image_set.samples[i][0].split("/")[-1] in min_groups[label])].append(index)

    return indices


def equalizing_odds(preds, labels, protect):
    """
    Evaluates the accuracy of the predictions for each cluster of each class
    :param preds: the predictions
    :param labels: the true labels
    :param protect: the cluster values
    :return: the accuracy of the predictions from each cluster
    """
    counts = [[[0.0, 0.0] for i in range(len(np.unique(protect)))] for _ in range(len(np.unique(labels)))]
    for pred, label, subgroup in zip(preds, labels, protect):
        counts[int(label)][int(subgroup)][0] += int(pred == label)
        counts[int(label)][int(subgroup)][1] += 1
    return [[round(p[0] / p[1], 3) for p in l] for l in counts]
