#!/usr/bin/env python
# coding: utf-8

"""Resnet random weight allocation

This script implements case 3 of the document: https://mit.zoom.us/j/98140955616
It clusters the training (Bias_0.8) training data into 5 groups, one of them containnig the protceted group (either female for doctors or male for nurses). During model training, we assign the same weight to every group except for one, where the weight is 4 times the others. We hope that this procedure will increase the fairness performance from Case 1 (Normal training on Bias_0.8 dataset)"""

from __future__ import print_function, division

import copy
import getopt
import os
import sys
import time
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import visdom
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
import pandas as pd
from math import ceil

plt.ion()  # interactive mode


class my_ImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform, clusters):
        super().__init__(root, transform)
        self.clusters = clusters

    def __getitem__(self, index: int):
        img = self.samples[index][0].split("/")[-1]
        group_number = max(
            [[img in c for c in clusters].index(max([img in c for c in clusters])) for clusters in self.clusters])
        return super().__getitem__(index), group_number


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


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def train_model(model, criterion, optimizer, scheduler, num_epochs=15, num_clusters=5, start_epoch=0, val_mode=False,
                show_progress=False):
    if show_progress:
        vis = visdom.Visdom()
        loss_window = vis.line(X=np.ones((1)) * start_epoch,
                               Y=np.zeros((1, 2)) if val_mode else np.zeros((1)),
                               opts=dict(xlabel='epoch',
                                         ylabel='Loss',
                                         title='epoch loss',
                                         markers=True,
                                         legend=["Train", "Val"] if val_mode else ["Train"],
                                         ))
        acc_window = vis.line(X=np.ones((1)) * start_epoch,
                              Y=np.zeros((1, 2)) if val_mode else np.zeros((1)),
                              opts=dict(xlabel='epoch',
                                        ylabel='Accuracy',
                                        title='epoch accuracy',
                                        markers=True,
                                        legend=["Train", "Val"] if val_mode else ["Train"],
                                        ))

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(start_epoch, start_epoch + num_epochs):
        losses, accuracies = {}, {}
        print('Epoch {}/{}'.format(epoch, start_epoch + num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        phases = ['train', 'val'] if val_mode else ["train"]
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, ((inputs, labels), groups) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    weights = torch.tensor([W_PROTECTED if g == epoch % num_clusters else 1 for g in groups.numpy()],
                                           device=device)  # iteration "i", epoch "epoch"
                    loss = criterion(outputs, labels, weights)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            losses[phase] = epoch_loss
            accuracies[phase] = epoch_acc

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

        if show_progress:
            X1 = np.ones((1)) * epoch

            Y1 = np.array([losses["train"]])
            Y2 = np.array(losses["val"]) if val_mode else None
            vis.line(X=X1, Y=np.column_stack((Y1, Y2)) if val_mode else Y1, win=loss_window, update='append')

            Y1 = np.array([accuracies["train"]])
            Y2 = np.array([accuracies["val"]]) if val_mode else None
            vis.line(X=X1, Y=np.column_stack((Y1, Y2)) if val_mode else Y1, win=acc_window, update='append')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    if "val" in phases:
        model.load_state_dict(best_model_wts)

    return model


def visualize_model(model, dataloader, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, ((inputs, labels), groups) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}, Actual: {}'.format(class_names[preds[j]], class_names[labels[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def weighted_cross_entropy_loss(output, labels, weights):
    cel = -torch.log(torch.exp(output.gather(1, labels.view(-1, 1))) / torch.sum(torch.exp(output), 1).view(-1, 1))
    weighted_cel = weights * cel.view(-1)
    return torch.mean(weighted_cel)


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
    s = os.listdir(path)

    l1, l2 = [], []
    for i, image in enumerate(s):
        if image in male_group:
            l1.append(i)
        else:
            l2.append(i)

    return l1, l2


path_dr_f_d = '../Datasets/doctor_nurse/dr/fem_dr_dark_56/'
path_dr_f_l = '../Datasets/doctor_nurse/dr/fem_dr_light_256/'
path_dr_m_d = '../Datasets/doctor_nurse/dr/mal_dr_dark_62/'
path_dr_m_l = '../Datasets/doctor_nurse/dr/mal_dr_light_308/'

dr_f_d = os.listdir(path_dr_f_d)
dr_f_l = os.listdir(path_dr_f_l)
dr_m_d = os.listdir(path_dr_m_d)
dr_m_l = os.listdir(path_dr_m_l)

path_nur_f_d = '../Datasets/doctor_nurse/nurse/fem_nurse_dark_63/'
path_nur_f_l = '../Datasets/doctor_nurse/nurse/fem_nurse_light_252/'
path_nur_m_d = '../Datasets/doctor_nurse/nurse/mal_nurse_dark_76/'
path_nur_m_l = '../Datasets/doctor_nurse/nurse/mal_nurse_light_203/'

nur_f_d = os.listdir(path_nur_f_d)
nur_f_l = os.listdir(path_nur_f_l)
nur_m_d = os.listdir(path_nur_m_d)
nur_m_l = os.listdir(path_nur_m_l)

dr_m, dr_f = len(dr_m_d) + len(dr_m_l), len(dr_f_d) + len(dr_f_l)

try:
    opts, args = getopt.getopt(sys.argv[1:], "h",
                               ["w_protected=", "bias=", "val_mode=", "start_epoch=", "num_epoch=", "num_clusters=",
                                "visdom=", "id=", "num_trials="])
except getopt.GetoptError:
    print("Wrong format ...")
    print(sys.argv)
    sys.exit(2)

data_dir = '../Datasets/doctor_nurse/train_test_split'
W_PROTECTED, BIAS, VAL_MODE, START_EPOCH, NUM_EPOCH, NUM_CLUSTERS, SHOW_PROGRESS, ID = 1, 0, False, 0, 15, 5, False, 0

for opt, arg in opts:
    if opt == '-h':
        print("Case_1+2.py --w_protected=<w_protected> --bias=<bias> --val_mode=<val_mode> --start_epoch=<start_epoch>"
              "--num_epoch=<num_epoch> --num_clusters=<num_clusters> --visdom=<visdom> --id=<id> "
              "--num_trials=<num_trials>")
        sys.exit()
    if opt == '--w_protected':
        W_PROTECTED = int(arg)
    if opt == '--bias':
        BIAS = float(arg)
    if opt == '--val_mode':
        VAL_MODE = int(arg)
    if opt == '--start_epoch':
        START_EPOCH = int(arg)
    if opt == '--num_epoch':
        NUM_EPOCH = int(arg)
    if opt == '--num_clusters':
        NUM_CLUSTERS = int(arg)
    if opt == '--visdom':
        SHOW_PROGRESS = int(arg)
    if opt == '--id':
        ID = int(arg)
    if opt == '--num_trials':
        NUM_TRIALS = int(arg)

print(
    f"RUNNING SCRIPT WITH ARGUMENTS : -w_protected={W_PROTECTED} -bias={BIAS} -val_mode={VAL_MODE} -start_epoch={START_EPOCH} "
    f"-num_epoch={NUM_EPOCH}, -num_clusters={NUM_CLUSTERS}, -visdom={SHOW_PROGRESS}, -id={ID}, -num_trials={NUM_TRIALS}")

# ###  Defining dataloaders


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

image_datasets = {x: my_ImageFolder(os.path.join(data_dir, f"train_{BIAS}" if x == "train" and BIAS else x),
                                    data_transforms[x],
                                    make_clusters([os.listdir(os.path.join(data_dir,
                                                                           f"train_{BIAS}/doctors" if x == "train" and BIAS else x)),
                                                   os.listdir(os.path.join(data_dir,
                                                                           f"train_{BIAS}/nurses" if x == "train" and BIAS else x))],
                                                  [set(dr_f_d + dr_f_l), set(nur_m_l + nur_m_d)],
                                                  K=NUM_CLUSTERS))
                  for x in ['train', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# For descirption of architecture of resNet, see: https://arxiv.org/pdf/1512.03385.pdf

train_accs, test_accs, fairness_accs = [], [], []
for trial in range(NUM_TRIALS):
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, len(class_names))

    model_conv = model_conv.to(device)

    criterion = weighted_cross_entropy_loss  # nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    if START_EPOCH:
        PATH = "Case_3/checkpoints/" + (
            "w_val" if VAL_MODE else "w.o_val") + f"/Bias_{BIAS}/model_ep_{START_EPOCH}/Run_{ID}/trial_{trial}/checkpoint.pt"
        checkpoint = torch.load(PATH)
        model_conv.load_state_dict(checkpoint['model_state_dict'])
        optimizer_conv.load_state_dict(checkpoint['optimizer_state_dict'])
        exp_lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, start_epoch=START_EPOCH,
                             num_epochs=NUM_EPOCH, num_clusters=NUM_CLUSTERS,
                             val_mode=VAL_MODE, show_progress=SHOW_PROGRESS)

    train_accs.append(round(float(accuracy(model_conv, dataloaders['train'])), 3))
    test_accs.append(round(float(accuracy(model_conv, dataloaders['test'])), 3))
    fairness_accs.append(demographic_parity(model_conv, os.path.join(data_dir, "test")).to_numpy())

    # #### Saving checkpoint

    PATH = "Case_3/checkpoints/" + (
        "w_val" if VAL_MODE else "w.o_val") + f"/Bias_{BIAS}/model_ep_{START_EPOCH + NUM_EPOCH}/Run_{ID}/trial_{trial}"
    LOSS = "CrossEntropyLoss"

    os.makedirs(PATH, exist_ok=True)
    torch.save({
        'epoch': START_EPOCH + NUM_EPOCH,
        'model_state_dict': model_conv.state_dict(),
        'optimizer_state_dict': optimizer_conv.state_dict(),
        'lr_scheduler_state_dict': exp_lr_scheduler.state_dict(),
        'loss': LOSS,
    }, PATH + "/checkpoint.pt")

train_accs = np.array(train_accs)
test_accs = np.array(test_accs)
fairness_accs = np.array(fairness_accs)
PATH = "Case_3/checkpoints/" + (
    "w_val" if VAL_MODE else "w.o_val") + f"/Bias_{BIAS}/model_ep_{START_EPOCH + NUM_EPOCH}/Run_{ID}/stats.txt"

file = open(PATH, "w")
file.write(f"Training accuracy: {train_accs.mean()} += {train_accs.std()} \n")
file.write(f"Test accuracy: {test_accs.mean()} += {test_accs.std()} \n")
file.write(f"Fairness accuracy: \n {np.mean(fairness_accs, axis=0)} \n += \n {np.std(fairness_accs, axis=0)}")
file.close()
