from __future__ import print_function, division

import sys, getopt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import time
import os
import copy
import visdom
import pandas as pd
from math import ceil


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=15,
                start_epoch=0, val_mode=False,
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
            for (inputs, labels), weights in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                weights = weights.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

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


def train_model_groups(model, criterion, optimizer, scheduler, w_protected, dataloaders, dataset_sizes, device,
                       num_epochs=15, num_clusters=5, start_epoch=0, val_mode=False, show_progress=False):
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

                    weights = torch.tensor([w_protected if g == epoch % num_clusters else 1 for g in groups.numpy()],
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


def visualize_model(model, dataloader, class_names, device, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
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


def accuracy(model, device, dataloader):
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
