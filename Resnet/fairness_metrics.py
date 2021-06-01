import torch
import pandas as pd
from model import accuracy


def demographic_parity(model, device, image_dataset, min_groups):
    """
    Evaluates the accuracy of images from the majority and minority groups from each class
    :param model: the trained model
    :param device: the device on which the model was trained
    :param image_dataset: the image data container
    :param min_groups: the path of images in the minority groups from each class
    :return: a table containing the accuracy of each group
    """
    indices = get_indices(image_dataset, min_groups)

    class1_majority = torch.utils.data.Subset(image_dataset, indices=indices[0][0])
    class1_minority = torch.utils.data.Subset(image_dataset, indices=indices[0][1])

    class2_majority = torch.utils.data.Subset(image_dataset, indices=indices[1][0])
    class2_minority = torch.utils.data.Subset(image_dataset, indices=indices[1][1])

    dataloaders = [torch.utils.data.DataLoader(x, batch_size=4, shuffle=True, num_workers=4) for x in
                   [class1_majority, class1_minority, class2_minority, class2_majority]]
    accuracies = [[float(accuracy(model, device, dataloader)) for dataloader in dataloaders[:2]],
                  [float(accuracy(model, device, dataloader)) for dataloader in dataloaders[2:]]]

    return pd.DataFrame(accuracies, index=["Class0", "Class1"], columns=["Group0", "Group1"])


def get_indices(image_set, min_groups, num_labels=2, num_protected=2):
    """
    Gets the indices of the minority group images in the image_set container
    :param image_set: the image container from which to get the indices from
    :param min_groups: the paths of the images in the minority group
    :param num_labels: the number of labels
    :param num_protected: the number of demographic groups
    :return: the indices corresponding to images in different demographic groups
    """
    indices = [[[] for _ in range(num_protected)] for _ in range(num_labels)]
    for i, ((_, label), _) in enumerate(image_set):
        indices[label][int(image_set.samples[i][0].split("/")[-1] in min_groups[label])].append(i)

    return indices
