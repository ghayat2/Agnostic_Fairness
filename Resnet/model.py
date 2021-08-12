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
    """
    Trains the model according to given arguments
    :param model: the model to train
    :param criterion: the objective of the model
    :param optimizer: the optimizer to use in the training process
    :param scheduler: the scheduler
    :param dataloaders: the data containers both train and test
    :param dataset_sizes: the sizes of the data both train and test
    :param device: the device on which to train the model
    :param num_epochs: the number of epochs for which to train the model
    :param start_epoch: the epoch number to start with
    :param val_mode: whether to validate the model while training
    :param show_progress: whether to show plots showing training progress
    :return: the trained model
    """
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
            for inputs, labels, weights, _ in dataloaders[phase]:
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


def train_cluster_reweight(model, device, train_loader, optimizer, scheduler, epochs, verbose=1, num_clusters=2,
                           num_labels=2,
                           update_lr=10, cluster_weights=None):
    """
    Trains the model in MODE 2. Each cluster has an individual weight that is updating at each epoch depending on
    the cluster's accuracy.
    :param model: The model
    :param device: The device the model is trained on
    :param train_loader: the training set
    :param optimizer: the optimizer
    :param epochs: the number of epochs
    :param verbose: If strictly positive, prints tracking notification during training
    :param num_clusters: the number of cluster per class
    :param num_labels: the number of classes (binary vs multiclass classification)
    :param cluster_lr: The rate at which the cluster weights are being updated
    :param cluster_weights: Values with which to initialize cluster weights
    :return: the performance history of the model during the training process
    """

    if not cluster_weights:
        cluster_weights = [[1.0 for _ in range(num_clusters)] for _ in range(num_labels)]
    history = pd.DataFrame([], columns=["loss", "accuracy"] +
                                       [f"cluster_acc_{t}{s}" for t in range(num_labels) for s in range(num_clusters)] +
                                       [f"cluster_weight_{t}{s}" for t in range(num_labels) for s in
                                        range(num_clusters)] +
                                       [f"cluster_grad_{t}{s}" for t in range(num_labels) for s in range(num_clusters)],
                           index=range(epochs))

    for epoch in range(epochs):
        model.train()
        sum_num_correct, sum_loss = 0, 0
        cluster_counts = [[[0.0, 0.0] for _ in range(num_clusters)] for _ in range(num_labels)]
        cluster_grads = [[[] for _ in range(num_clusters)] for _ in range(num_labels)]

        for batch_idx, (data, target, cluster, _) in enumerate(train_loader):
            data, target, cluster = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long), \
                                    cluster.to(device, dtype=torch.long)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                output = model(data)
                _, preds = torch.max(output, 1)

                weights = torch.tensor([cluster_weights[int(t)][int(c)] for c, t in zip(cluster, target)],
                                       requires_grad=True).type(torch.float)

                loss = weighted_cross_entropy_loss(output, target, weights)
                loss.backward()
                optimizer.step()

            correct = preds.eq(target.view_as(preds)).sum().item()
            sum_num_correct += correct
            sum_loss += loss.item() * train_loader.batch_size

            cluster_counts = cluster_counts_update(cluster_counts, preds, target, cluster)
            cluster_grads = get_cluster_grads(cluster_grads, weights.grad.numpy(), cluster, target)

        scheduler.step()
        sum_loss /= len(train_loader.dataset)
        clusters_accs = [[l[0] / l[1] for l in cluster] for cluster in cluster_counts]
        cluster_grads = [[np.average(l) for l in clusters] for clusters in cluster_grads]

        cluster_new_weights = cluster_weight_updates(cluster_weights, cluster_grads, clusters_accs, update_lr)
        cluster_weights = normalize_clusters(cluster_weights, cluster_new_weights,
                                             [[total for correct, total in counts] for counts in cluster_counts])

        acc = 100. * sum_num_correct / len(train_loader.dataset)
        # cluster_weights = update_weights(cluster_weights, clusters_accs) # weights update by customized heuristic

        history.iloc[epoch] = [sum_loss, acc] + list(np.array(clusters_accs).reshape((-1))) + list(
            np.array(cluster_weights).reshape((-1))) + list(np.array(cluster_grads).reshape((-1)))

        if verbose:
            print(
                '\nEpoch: {} - Train set: Average loss: {:.2e}, Accuracy: {}/{} ({:.0f}%), Cluster accuracies: {}, '
                'weights: {}\n'.format(
                    epoch,
                    sum_loss, sum_num_correct, len(train_loader.dataset),
                    acc,
                    str(clusters_accs),
                    str(cluster_weights)))

        print("Evaluating: Train mode")
        train_pred_labels, train_labels, train_protect, _, train_accuracy, _ = test(model, device,
                                                                                    train_loader, eval=False)
        print(train_accuracy)
        print(equalizing_odds(train_pred_labels, train_labels, train_protect))
        print("Evaluating: Eval mode")
        train_pred_labels, train_labels, train_protect, _, train_accuracy, _ = test(model, device,
                                                                                    train_loader, eval=True)
        print(train_accuracy)
        print(equalizing_odds(train_pred_labels, train_labels, train_protect))

    return history


def train_sample_reweight(model, device, train_loader, optimizer, scheduler, epochs, verbose=1, num_clusters=2,
                          num_labels=2,
                          update_lr=10, init_weights=None):
    """
    Trains the model in MODE 2. Each sample has an individual weight that is updating at each epoch depending on
    the cluster's accuracy.
    Note:
    At each epoch, the weights updated is limited to the correctly classified samples from the better than average
    clusters (decreased) and the incorrectly classified samples from the under average clusters (increased)
    :param model: The model
    :param device: The device the model is trained on
    :param train_loader: the training set
    :param optimizer: the optimizer
    :param epochs: the number of epochs
    :param verbose: If strictly positive, prints tracking notification during training
    :param num_clusters: the number of cluster per class
    :param num_labels: the number of classes (binary vs multiclass classification)
    :param update_lr: The rate at which the samplle weights are being updated
    :param init_weights: cluster weight values to initialize sample weights with
    :return: the performance history of the model during the training process
    """
    model.train()
    sample_weights = cluster_dic_init(train_loader, init_weights if init_weights else 1.0, num_clusters, num_labels)

    history = pd.DataFrame([], columns=["loss", "accuracy"] +
                                       [f"cluster_acc_{t}{s}" for t in range(num_labels) for s in range(num_clusters)] +
                                       [f"cluster_updates_{t}{s}" for t in range(num_labels) for s in
                                        range(num_clusters)],
                           index=range(epochs))

    for epoch in range(epochs):
        sum_num_correct, sum_loss = 0, 0
        correct_classifications = cluster_dic_init(train_loader, 0.0, num_clusters, num_labels)
        sample_grads = cluster_dic_init(train_loader, 0.0, num_clusters, num_labels)

        for batch_idx, (data, target, cluster, indexes) in enumerate(train_loader):
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                output = model(data)
                _, preds = torch.max(output, 1)

                weights = torch.tensor(
                    [sample_weights[int(t)][int(c)][int(i)] for c, t, i in zip(cluster, target, indexes)],
                    requires_grad=True).type(torch.float)

                loss = weighted_cross_entropy_loss(output, target, weights)
                loss.backward()
                optimizer.step()

            correct = preds.eq(target.view_as(preds)).sum().item()
            sum_num_correct += correct
            sum_loss += loss.item() * train_loader.batch_size

            correct_classifications = update_classification(correct_classifications, indexes, preds, target, cluster)
            sample_grads = get_sample_grads(sample_grads, indexes, weights.grad.numpy(), cluster, target)

        scheduler.step()
        sum_loss /= len(train_loader.dataset)
        clusters_accs = [[sum(cluster.values()) / len(cluster) for cluster in clusters]
                         for clusters in correct_classifications]

        new_sample_weights, updates = sample_weight_updates(train_loader, sample_weights, sample_grads,
                                                            correct_classifications,
                                                            clusters_accs, update_lr)

        sample_weights = normalize_samples(sample_weights, new_sample_weights)

        acc = 100. * sum_num_correct / len(train_loader.dataset)
        history.iloc[epoch] = [sum_loss, acc] + list(np.array(clusters_accs).reshape((-1))) \
                              + list(np.array(updates).reshape((-1)))
        if verbose:
            print(
                '\nEpoch: {} - Train set: Average loss: {:.2e}, Accuracy: {}/{} ({:.0f}%), Cluster accuracies: {}, '
                'number of updates {}\n'
                    .format(
                    epoch,
                    sum_loss, sum_num_correct, len(train_loader.dataset),
                    acc,
                    str(clusters_accs),
                    str(updates)))

    return history


def train_individual_reweight(model, device, train_loader, optimizer, scheduler, epochs, verbose=1, num_clusters=2,
                              num_labels=2,
                              update_lr=10):
    """
    Trains the model in MODE 2. Each sample has an individual weight that is updating at each epoch depending on
    the correctness of its classification at the last epoch.
    Note:
    At each epoch, the weight of correctly classified samples is decreased while the weight of the incorrectly
    classified samples is increased.
    :param model: The model
    :param device: The device the model is trained on
    :param train_loader: the training set
    :param optimizer: the optimizer
    :param epochs: the number of epochs
    :param verbose: If strictly positive, prints tracking notification during training
    :param num_clusters: the number of cluster per class
    :param num_labels: the number of classes (binary vs multiclass classification)
    :param sample_lr: The rate at which the samplel weights are being updated
    :return: the performance history of the model during the training process
    """
    model.train()
    sample_weights = dic_init(train_loader, 1.0)

    history = pd.DataFrame([], columns=["loss", "accuracy"] +
                                       [f"cluster_acc_{t}{s}" for t in range(num_labels) for s in range(num_clusters)] +
                                       [f"cluster_balance_{t}{s}" for t in range(num_labels) for s in
                                        range(num_clusters)],
                           index=range(epochs))

    for epoch in range(epochs):
        sum_num_correct, sum_loss = 0, 0
        correct_classifications = dic_init(train_loader, 0.0)
        sample_grads = dic_init(train_loader, 0.0)

        for batch_idx, (data, target, cluster, indexes) in enumerate(train_loader):
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                output = model(data)
                _, preds = torch.max(output, 1)
                weights = torch.tensor([sample_weights[int(i)] for i in indexes], requires_grad=True).type(torch.float)

                loss = weighted_cross_entropy_loss(output, target, weights)
                loss.backward()
                optimizer.step()

            correct = preds.eq(target.view_as(preds)).sum().item()
            sum_num_correct += correct
            sum_loss += loss.item() * train_loader.batch_size

            correct_classifications = individual_classification(correct_classifications, indexes, preds, target)
            sample_grads = get_indivdual_grads(sample_grads, indexes, weights.grad.numpy())

        scheduler.step()
        sum_loss /= len(train_loader.dataset)
        clusters_accs = cluster_accuracies(train_loader, correct_classifications, num_labels, num_clusters)

        sample_weights, updates = individual_weight_updates(train_loader, sample_weights, sample_grads,
                                                            correct_classifications, update_lr, num_clusters,
                                                            num_labels)

        acc = 100. * sum_num_correct / len(train_loader.dataset)
        history.iloc[epoch] = [sum_loss, acc] + list(np.array(clusters_accs).reshape((-1))) \
                              + list(np.array(updates).reshape((-1, 2)))
        if verbose:
            print(
                '\nEpoch: {} - Train set: Average loss: {:.2e}, Accuracy: {}/{} ({:.0f}%), Cluster accuracies: {}, '
                'number of updates {}\n'
                    .format(
                    epoch,
                    sum_loss, sum_num_correct, len(train_loader.dataset),
                    acc,
                    str(clusters_accs),
                    str(updates)))

    return history


def chunks(lst, K):
    """
    Yield successive K-sized chunks from lst.
    :param lst: the python list
    :param K: the number of chunks
    :return: a list split into K chunks
    """
    results, n = [], ceil(len(lst) / K)
    for i in range(0, len(lst), n):
        results.append(lst[i:(i + n) if i + n < len(lst) else -1])
    return results


def make_clusters(sets, protected_groups, K):
    """
    Makes K clusters for each class. One of them contains the minority group and the rest is distributed in K-1
    random groups
    :param sets: the paths of images in each class
    :param protected_groups: the paths of images from the minority group
    :param K: the number of clusters
    :return: A list of dimension (n_class, K)
    """
    assert len(sets) == len(protected_groups)

    clusters = []
    for i, s in enumerate(sets):
        majority, minority = [], []
        for img in s:
            minority.append(img) if img in protected_groups[i] else majority.append(img)

        clusters.append([minority] + chunks(majority, K - 1))

    return clusters


def update_classification(correct_classification, indexes, predictions, labels, clusters):
    """
    Reports whether each sample has been correctly classified and sorts the prediction by class/cluster
    :param correct_classification: the data structure to be updated
    :param indexes: the sample indexes
    :param predictions: the model predictions
    :param labels: the sample labels
    :param clusters: the sample cluster
    :return: the updated data structure
    """
    for i, prediction, label, cluster in zip(indexes, predictions, labels, clusters):
        correct_classification[int(label)][int(cluster)][int(i)] = int(prediction == label)
    return correct_classification


def individual_classification(correct_classification, indexes, predictions, labels):
    """
    Reports whether each sample has been correctly classified
    :param correct_classification: the data structure to be updated
    :param indexes: the sample indexes
    :param predictions: the model predictions
    :param labels: the sample labels
    :return: the updated data structure
    """
    for i, prediction, label in zip(indexes, predictions, labels):
        correct_classification[int(i)] = int(prediction == label)
    return correct_classification


def cluster_accuracies(dataset, correct_classifications, num_labels=2, num_clusters=2):
    """
    Computes the accuracy of every cluster
    :param num_clusters: number of clusters in the dataset
    :param num_labels: number of classes in the dataset
    :param dataset: the training set
    :param correct_classifications: the classification correctness of every sample
    :return: an array containing the accuracy of every cluster
    """
    accuracies = [[[0, 0] for _ in range(num_clusters)] for _ in range(num_labels)]

    for _, labels, clusters, indexes in dataset:
        for i, label, cluster in zip(indexes, labels, clusters):
            accuracies[label][cluster][0] += correct_classifications[int(i)]
            accuracies[label][cluster][1] += 1
    return [[p[0] / p[1] for p in accs] for accs in accuracies]


def get_sample_grads(sample_grads, indexes, grads, clusters, labels):
    """
    Updates the data structure containing all the sample gradients of the epoch
    :param sample_grads: the data structure
    :param indexes: the sample indexes
    :param grads: the sample gradients
    :param clusters: the sample cluster value
    :param labels: the sample labels
    :return: the updated data structure
    """
    for i, grad, cluster, label in zip(indexes, grads, clusters, labels):
        sample_grads[int(label)][int(cluster)][int(i)] = grad
    return sample_grads


def get_indivdual_grads(sample_grads, indexes, grads):
    """
        Updates the data structure containing all the sample gradients of the epoch
        :param sample_grads: the data structure
        :param indexes: the sample indexes
        :param grads: the sample gradients
        :return: the updated data structure
        """
    for i, grad in zip(indexes, grads):
        sample_grads[int(i)] = grad
    return sample_grads


def sample_weight_updates(dataset, sample_weights, sample_grads, correct_classifications, clusters_accs, cluster_lr=10):
    """
    Updates the sample weights depending on the correctness of the model predictions and the sample gradients. The
    weights updated is limited to the correctly classified samples from the better than average clusters (decreased)
    and the incorrectly classified samples from the under average clusters (increased)
    :param dataset: the dataset
    :param sample_weights: the sample weights
    :param sample_grads: the sample gradients
    :param correct_classifications: the structure containing the correctness of predictions
    :param clusters_accs: the accuracy of the clusters
    :param cluster_lr: the learning rate
    :return: the new weights, the number of weights updated
    """
    new_dict = [[{} for _ in range(len(sample_weights[0]))] for _ in range(len(sample_weights))]
    updates = [[0 for _ in range(len(sample_weights[0]))] for _ in range(len(sample_weights))]
    for _, labels, clusters, indexes in dataset:
        for i, label, cluster in zip(indexes, labels, clusters):
            if clusters_accs[int(label)][int(cluster)] < np.average(clusters_accs[int(label)]) \
                    and not correct_classifications[int(label)][int(cluster)][int(i)]:
                new_dict[int(label)][int(cluster)][int(i)] = sample_weights[int(label)][int(cluster)][int(i)] \
                                                             + cluster_lr * sample_grads[int(label)][int(cluster)][
                                                                 int(i)]
                updates[int(label)][int(cluster)] += 1
            elif clusters_accs[int(label)][int(cluster)] > np.average(clusters_accs[int(label)]) \
                    and correct_classifications[int(label)][int(cluster)][int(i)]:
                new_dict[int(label)][int(cluster)][int(i)] = sample_weights[int(label)][int(cluster)][int(i)] \
                                                             - cluster_lr * sample_grads[int(label)][int(cluster)][
                                                                 int(i)]
                updates[int(label)][int(cluster)] -= 1
            else:
                new_dict[int(label)][int(cluster)][int(i)] = sample_weights[int(label)][int(cluster)][int(i)]
    return new_dict, updates


def individual_weight_updates(dataset, sample_weights, sample_grads, correct_classifications, update_lr,
                              num_clusters, num_labels):
    """
        Updates the sample weights depending on the correctness of the model predictions and the sample gradients.
        :param dataset: the dataset
        :param sample_weights: the sample weights
        :param sample_grads: the sample gradients
        :param correct_classifications: the structure containing the correctness of predictions
        :param update_lr: the learning rate
        :return: the new weights, the number of weights updated
        """
    new_dict = {}
    updates = [[[0, 0] for _ in range(num_clusters)] for _ in range(num_labels)]
    for _, labels, clusters, indexes in dataset:
        for i, label, cluster in zip(indexes, labels, clusters):
            if not correct_classifications[int(i)]:
                new_dict[int(i)] = sample_weights[int(i)] + update_lr * sample_grads[int(i)]
                updates[int(label)][int(cluster)][0] += 1
            else:
                new_dict[int(i)] = sample_weights[int(i)] - update_lr * sample_grads[int(i)]
                updates[int(label)][int(cluster)][1] -= 1
    return new_dict, updates


def get_cluster_grads(cluster_grads, new_grads, cluster, target):
    """
    Updates the cluster gradients with computed gradients of new batch
    :param cluster_grads: Array of shape [num_classes, num_clusters] where each element is a list of all gradients
                        from the cluster weight computed so far
    :param new_grads: The new cluster weight gradients from the new batch
    :param cluster: the cluster id of samples from the new batch
    :param target: the target from the samples of the new batch
    :return: the updated cluster gradients
    """
    for i, (c, t) in enumerate(zip(cluster, target)):
        cluster_grads[int(t)][int(c)].append(new_grads[i])
    return cluster_grads


def normalize_samples(sample_old_weights, sample_new_weights):
    """
    Normalizes the new sample weights with respect to the size of cluster.
    :param cluster_old_weights: The old weights of the clusters
    :param cluster_new_weights: The new weights of the clusters
    :param cluster_sizes: the size of the clusters
    :return: The normalized sample weights
    """
    res_dict = [[{} for _ in range(len(sample_new_weights[0]))] for _ in range(len(sample_new_weights))]
    for label, (old_weights, new_weights) in enumerate(zip(sample_old_weights, sample_new_weights)):
        cst_1, cst_2 = find_normalizing_cst(old_weights, new_weights)
        for cluster, new_w_dic in enumerate(new_weights):
            res_dict[label][cluster] = {k: cst_1 * w / cst_2 for k, w in new_w_dic.items()}
    return res_dict


def find_normalizing_cst(old_weights, new_weights):
    """
    Finds the sum of the sample gradients of each cluster
    :param old_weights: old weights
    :param new_weights: new weights
    :return: the normalizing clonstants
    """
    cst_1, cst_2 = 0, 0
    for old_dic, new_dic in zip(old_weights, new_weights):
        cst_1 += np.sum(list(old_dic.values()))
        cst_2 += np.sum(list(new_dic.values()))
    return cst_1, cst_2


def cluster_weight_updates(weights, grads, accs, lr):
    """
     Updates the cluster weights for the next epoch - the direction of the update depends on the cluster accuracy
    :param weights: The current weights of the clusters
    :param grads: The average gradient of the cluster weights from the epoch
    :param accs: The cluster accuracy from the epoch
    :param lr: The rate of the update
    :return: The updated cluster weights
    """
    return [[w + lr * g if a < np.average(cluster_accs) else w - lr * g for w, g, a
             in zip(cluster_weights, cluster_grads, cluster_accs)] for cluster_weights, cluster_grads, cluster_accs
            in zip(weights, grads, accs)]


def normalize_clusters(cluster_old_weights, cluster_new_weights, cluster_sizes):
    """
    Normalizes the new weights with respect to the size of cluster.
    :param cluster_old_weights: The old weights of the clusters
    :param cluster_new_weights: The new weights of the clusters
    :param cluster_sizes: the size of the clusters
    :return: The normalized weights
    """
    cluster_ratios = [[s / np.sum(sizes) for s in sizes] for sizes in cluster_sizes]
    return [[np.sum(np.array(old_weights) * np.array(sizes)) * new_w / np.sum(np.array(new_weights) * np.array(sizes))
             for new_w in new_weights] for old_weights, new_weights, sizes
            in zip(cluster_old_weights, cluster_new_weights, cluster_ratios)]


def visualize_model(model, dataloader, class_names, device, num_images=6):
    """
    Visualize some input and model predictions
    :param model: the trained model
    :param dataloader: the image data container
    :param class_names: the name of each class
    :param device: the device on which the model is trained
    :param num_images: the number of images to display
    :return: Visual representation of input images along with model predictions
    """
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


def cluster_counts_update(counts, preds, targets, clusters):
    """
    Counts the number of correct predictions per cluster
    :param counts: Array of shape (num_classes, num_clusters), where each element is a list [a,b] where
                    a="Cluster samples correctly predicted" and b="Total number of samples in cluster"
    :param preds: Model Predictions
    :param targets: True labels
    :param clusters: Clusters (Subgroups)
    :return: Updated counts
    """
    for pred, target, cluster in zip(preds, targets, clusters):
        counts[int(target)][int(cluster)][0] += int(pred == target)
        counts[int(target)][int(cluster)][1] += 1
    return counts


def cluster_dic_init(dataset, values, num_clusters, num_labels):
    """
    Creates a data structure that contains a dictionary for each cluster with the initial value specified for each
    sample key
    :param dataset: the dataset
    :param value: the initial value to give for each sample key
    :param num_clusters: the number of clusters
    :param num_labels: the number of classes
    :return: the data structure
    """
    if isinstance(values, float):
        values = [[values for _ in range(num_clusters)] for _ in range(num_labels)]

    dicts = [[{} for _ in range(num_clusters)] for _ in range(num_labels)]
    for _, labels, clusters, indexes in dataset:
        for i, label, cluster in zip(indexes, labels, clusters):
            dicts[int(label)][int(cluster)][int(i)] = values[int(label)][int(cluster)]
    return dicts


def dic_init(dataset, value):
    dic = {}
    for _, _, _, indexes in dataset:
        for i in indexes:
            dic[int(i)] = value
    return dic


def weighted_cross_entropy_loss(output, labels, weights):
    """
    The weighted binary cross entropy loss
    :param output: predictions
    :param labels: labels
    :param weights: the weights
    :return: the loss value
    """
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
    """
    Computes the predictive accuracy of the model on the given dataloader
    :param model: the trained model
    :param device: the device on which the model was trained
    :param dataloader: the image data container
    :return: the accuracy of the model on the given data
    """
    model.eval()
    corrects, total = 0, 0
    for i, (inputs, labels, _, _) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        corrects += torch.sum(preds == labels.data)
        total += inputs.size(0)

    return corrects.double() / total


def test(model, device, test_loader, eval=True):
    """
    Evaluates the quality of the predictions of the trained model on a test set
    :param model: the trained model
    :param device: the device the model is evaluated on
    :param test_loader: the test set
    :return: the model predictions, loss values and accuracy
    """
    if eval:
        model.eval()
    else:
        model.train()

    test_loss, correct = 0, 0
    test_pred = torch.zeros(0, 1).to(device)
    test_labels = torch.zeros(0, 1).to(device)
    test_protect = torch.zeros(0, 1).to(device)
    test_probs = torch.zeros(0, 2).to(device)
    with torch.no_grad():
        for data, target, protect, i in test_loader:
            data, target, protect = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long), \
                                    protect.to(device, dtype=torch.long)
            output = model(data)
            test_probs = torch.cat([test_probs, output], 0)
            _, preds = torch.max(output, 1)

            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(output, target)
            test_loss += loss.item() * test_loader.batch_size  # sum up loss for each test sample
            test_pred = torch.cat([test_pred, preds[:, None].type(torch.float)], 0)
            test_labels = torch.cat([test_labels, target[:, None].type(torch.float)], 0)
            test_protect = torch.cat([test_protect, protect[:, None].type(torch.float)], 0)
            correct += preds.eq(target.view_as(preds)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)

    return test_pred, test_labels, test_protect, test_loss, test_accuracy, test_probs


@DeprecationWarning
def train_model_groups(model, criterion, optimizer, scheduler, w_protected, dataloaders, dataset_sizes, device,
                       num_epochs=15, num_clusters=5, start_epoch=0, val_mode=False, show_progress=False):
    """
        Trains the model according to given arguments
        :param model: the model to train
        :param criterion: the objective of the model
        :param optimizer: the optimizer to use in the training process
        :param scheduler: the scheduler
        :param dataloaders: the data containers both train and test
        :param dataset_sizes: the sizes of the data both train and test
        :param device: the device on which to train the model
        :param num_epochs: the number of epochs for which to train the model
        :param start_epoch: the epoch number to start with
        :param val_mode: whether to validate the model while training
        :param show_progress: whether to show plots showing training progress
        :return: the trained model
        """
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
            for i, (inputs, labels, groups, _) in enumerate(dataloaders[phase]):
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
