"""
This script evaluates a trained model against multiple fairness metrics
"""

from __future__ import print_function, division

import getopt
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import os
import pandas as pd
from math import ceil

plt.ion()  # interactive mode

data_dir = '../Datasets/doctor_nurse/train_test_split'
VAL_MODE, EPOCH, CASE, ID, BIAS = False, 30, "Case_3", 0, 0.8

try:
    opts, args = getopt.getopt(sys.argv[1:], "h",
                               ["bias=", "val_mode=", "epoch=", "id=", "case="])
except getopt.GetoptError:
    print("Wrong format ...")
    print(sys.argv)
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print("Results.py --bias=<bias> --val_mode=<val_mode> --epoch=<epoch> "
              "--id=<id> --case=<case>")
        sys.exit()
    if opt == '--bias':
        BIAS = float(arg)
    if opt == '--val_mode':
        VAL_MODE = int(arg)
    if opt == '--epoch':
        EPOCH = int(arg)
    if opt == '--id':
        ID = int(arg)
    if opt == '--case':
        if arg not in ["Case_1", "Case_2", "Case_3"]:
            print("Case must be one of (Case_1, Case_2, Case_3)")
            sys.exit()
        CASE = arg

print(
    f"RUNNING SCRIPT WITH ARGUMENTS : -bias={BIAS} -val_mode={VAL_MODE}"
    f"-epoch={EPOCH}, -id={ID}, -case={CASE}")

path_dr_f_d = '/home/ghayat/Datasets/doctor_nurse/dr/fem_dr_dark_56/'
path_dr_f_l = '/home/ghayat/Datasets/doctor_nurse/dr/fem_dr_light_256/'
path_dr_m_d = '/home/ghayat/Datasets/doctor_nurse/dr/mal_dr_dark_62/'
path_dr_m_l = '/home/ghayat/Datasets/doctor_nurse/dr/mal_dr_light_308/'

dr_f_d = sorted(os.listdir(path_dr_f_d))
dr_f_l = sorted(os.listdir(path_dr_f_l))
dr_m_d = sorted(os.listdir(path_dr_m_d))
dr_m_l = sorted(os.listdir(path_dr_m_l))

path_nur_f_d = '/home/ghayat/Datasets/doctor_nurse/nurse/fem_nurse_dark_63/'
path_nur_f_l = '/home/ghayat/Datasets/doctor_nurse/nurse/fem_nurse_light_252/'
path_nur_m_d = '/home/ghayat/Datasets/doctor_nurse/nurse/mal_nurse_dark_76/'
path_nur_m_l = '/home/ghayat/Datasets/doctor_nurse/nurse/mal_nurse_light_203/'

nur_f_d = sorted(os.listdir(path_nur_f_d))
nur_f_l = sorted(os.listdir(path_nur_f_l))
nur_m_d = sorted(os.listdir(path_nur_m_d))
nur_m_l = sorted(os.listdir(path_nur_m_l))

dr_m, dr_f = len(dr_m_d) + len(dr_m_l), len(dr_f_d) + len(dr_f_l)

w_protected = 4


def chunks(lst, K):
    """Yield successive K-sized chunks from lst."""
    results, n = [], ceil(len(lst) / K)
    for i in range(0, len(lst), n):
        results.append(lst[i:(i + n) if i + n < len(lst) else -1])
    return results


def make_clusters(sets, protected_groups, K):
    assert len(sets) == len(protected_groups)

    clusters = []
    for i, s in enumerate(sets):
        majority, minority = [], []
        for img in s:
            minority.append(img) if img in protected_groups[i] else majority.append(img)

        clusters.append([minority] + chunks(majority, K - 1))

    return clusters


def accuracy(model, dataloader):
    model.eval()
    corrects, total = 0, 0
    for i, ((inputs, labels), _) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        corrects += torch.sum(preds == labels.data)
        total += inputs.size(0)

    return corrects.double() / total


def demographic_parity(model, test_set):
    dr_path = os.path.join(test_set, "doctors")
    nurs_path = os.path.join(test_set, "nurses")

    dr_m_indices, dr_f_indices = split_gender(dr_path, dr_m_l + dr_m_d)
    nurs_m_indices, nurs_f_indices = split_gender(nurs_path, nur_m_d + nur_m_l)

    dr_m = torch.utils.data.Subset(image_datasets["test"], indices=dr_m_indices)
    dr_f = torch.utils.data.Subset(image_datasets["test"], indices=dr_f_indices)

    nurs_m = torch.utils.data.Subset(image_datasets["test"], indices=[len(dr_m + dr_f) + i for i in nurs_m_indices])
    nurs_f = torch.utils.data.Subset(image_datasets["test"], indices=[len(dr_m + dr_f) + i for i in nurs_f_indices])

    dataloaders = [torch.utils.data.DataLoader(x, batch_size=4, shuffle=True, num_workers=4) for x in
                   [dr_m, dr_f, nurs_m, nurs_f]]
    accuracies = [[float(accuracy(model, dataloader)) for dataloader in dataloaders[:2]],
                  [float(accuracy(model, dataloader)) for dataloader in dataloaders[2:]]]

    return pd.DataFrame(accuracies, index=["Doctor", "Nurse"], columns=["Men", "Women"])


def split_gender(path, male_group):
    s = sorted(os.listdir(path))

    l1, l2 = [], []
    for i, image in enumerate(s):
        if image in male_group:
            l1.append(i)
        else:
            l2.append(i)

    return l1, l2


def weighted_cross_entropy_loss(output, labels, weights):
    cel = -torch.log(torch.exp(output.gather(1, labels.view(-1, 1))) / torch.sum(torch.exp(output), 1).view(-1, 1))
    weighted_cel = weights * cel.view(-1)
    return torch.mean(weighted_cel)


class my_ImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform, clusters):
        super().__init__(root, transform)
        self.clusters = clusters

    def __getitem__(self, index: int):
        img = self.samples[index][0].split("/")[-1]
        group_number = max(
            [[img in c for c in clusters].index(max([img in c for c in clusters])) for clusters in self.clusters])
        return super().__getitem__(index), group_number


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir, BIAS = '/home/ghayat/Datasets/doctor_nurse/train_test_split', 0.8
image_datasets = {x: my_ImageFolder(os.path.join(data_dir, f"train_{BIAS}" if x == "train" and BIAS else x),
                                    data_transforms[x],
                                    make_clusters([os.listdir(os.path.join(data_dir,
                                                                           f"train_{BIAS}/doctors" if x == "train" and BIAS else x)),
                                                   os.listdir(os.path.join(data_dir,
                                                                           f"train_{BIAS}/nurses" if x == "train" and BIAS else x))],
                                                  [set(dr_f_d + dr_f_l), set(nur_m_l + nur_m_d)],
                                                  K=5))
                  for x in ['train', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, len(class_names))

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()  # weighted_cross_entropy_loss

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

train_accs, test_accs, fairness_accs = [], [], []
for i in range(5):
    PATH = f"/home/ghayat/Resnet/{CASE}/checkpoints/" + (
        "w_val" if VAL_MODE else "w.o_val") + f"/Bias_{BIAS}/model_ep_{EPOCH}/Run_{ID}/trial_{i}/checkpoint.pt"
    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
    model_conv.load_state_dict(checkpoint['model_state_dict'])
    optimizer_conv.load_state_dict(checkpoint['optimizer_state_dict'])
    exp_lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    train_accs.append(accuracy(model_conv, dataloaders['train']))
    test_accs.append(accuracy(model_conv, dataloaders['test']))
    fairness_accs.append(demographic_parity(model_conv, os.path.join(data_dir, "test")).to_numpy())

train_accs = np.array(train_accs)
test_accs = np.array(test_accs)
fairness_accs = np.array(fairness_accs)

print(train_accs)
print(test_accs)
print(fairness_accs)
print()

print(f"Training accuracy: {train_accs.mean()} += {train_accs.std()}")
print(f"Test accuracy: {test_accs.mean()} += {test_accs.std()}")
print(f"Fairness accuracy: \n {np.mean(fairness_accs, axis=0)} += {np.std(fairness_accs, axis=0)}")
