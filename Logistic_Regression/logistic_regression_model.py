import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


class Predictor(nn.Module):
    """
    Logistic Regression implemented with pyTorch
    """

    def __init__(self, num_predictor_features):
        super(Predictor, self).__init__()
        self.linear = torch.nn.Linear(num_predictor_features, 1)

    def forward(self, x):
        y_logits = self.linear(x)
        y_pred = F.sigmoid(y_logits)
        return y_logits, y_pred


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


@DeprecationWarning
def update_weights(weights, cluster_accs):
    """
    Update the cluster weights for the next epoch
    :param weights: The weights that need to be updated
    :param cluster_accs: The accuracy of each cluster
    :return: the updated weights
    """
    cst = 1
    for i, cluster in enumerate(cluster_accs):
        avg_acc = np.mean(cluster)
        for j, acc in enumerate(cluster):
            if avg_acc > acc:
                weights[i][j] += (avg_acc - acc) * cst
    return weights


def my_BCELoss(preds, labels, weights):
    """
    The weighted binary cross entropy loss
    :param preds: predictions
    :param labels: labels
    :param weights: weights
    :return: the loss value
    """
    return torch.mean(-weights * (labels * torch.log(preds) + (1 - labels) * torch.log(1 - preds)))


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


def gradient_updates(weights, grads, accs, lr):
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


def normalize(cluster_old_weights, cluster_new_weights, cluster_sizes):
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


def train(model, device, train_loader, optimizer, epochs, verbose=1, minority_w=None):
    """
    Trains the model in MODE 0 or 1. When MODE is 1, a minority weight is given for each class and applied to samples
    belonging to thr minority group. If MODE is 0, no weights is applied to the dataset.
    :param model: the model
    :param device: the device on which the model is trained
    :param train_loader: the training set
    :param optimizer: the optimizer
    :param epochs: the number of epochs
    :param verbose: print tracking notifications while training
    :param minority_w: the minority weights to apply the the minority group. If None, then MODE is 0, else MODE is 1
    :return: the performance history of the model during the training process
    """
    model.train()
    history = pd.DataFrame([], columns=["loss", "accuracy"], index=range(epochs))

    for epoch in range(epochs):
        sum_num_correct = 0
        sum_loss = 0

        for batch_idx, (data, target, protect) in enumerate(train_loader):
            data, target, protect = data.to(device, dtype=torch.float), target.to(device,
                                                                                  dtype=torch.float), protect.to(device,
                                                                                                                 dtype=torch.float)
            optimizer.zero_grad()
            logits, output = model(data)

            weights = torch.tensor(
                [minority_w[0] if not target[i] and not protect[i] else 1 for i in range(len(data))]).type(torch.float) \
                      * torch.tensor(
                [minority_w[1] if target[i] and not protect[i] else 1 for i in range(len(data))]).type(
                torch.float) if minority_w else None

            criterion = torch.nn.BCELoss(weight=weights)
            loss = criterion(output.view_as(target), target)
            pred = (output > 0.5) * 1
            pred = pred.float()
            correct = pred.eq(target.view_as(pred)).sum().item()
            sum_num_correct += correct
            sum_loss += loss.item() * train_loader.batch_size
            loss.backward()
            optimizer.step()

        sum_loss /= len(train_loader.dataset)
        acc = 100. * sum_num_correct / len(train_loader.dataset)

        history.iloc[epoch] = [sum_loss, acc]

        if verbose:
            print('\nTrain set: Average loss: {:.2e}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                sum_loss, sum_num_correct, len(train_loader.dataset),
                acc))

    return history


def train_reweight(model, device, train_loader, optimizer, epochs, verbose=1, num_clusters=2, num_labels=2,
                   cluster_lr=10):
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
    :return: the performance history of the model during the training process
    """
    model.train()
    cluster_weights = [[1.0 for _ in range(num_clusters)] for _ in range(num_labels)]
    # cluster_weights = [[-0.0762689, 1.64822], [-0.22587, 1.21449]]
    history = pd.DataFrame([], columns=["loss", "accuracy"] +
                                       [f"cluster_acc_{t}{s}" for t in range(num_labels) for s in range(num_clusters)] +
                                       [f"cluster_weight_{t}{s}" for t in range(num_labels) for s in
                                        range(num_clusters)] +
                                       [f"cluster_grad_{t}{s}" for t in range(num_labels) for s in range(num_clusters)],
                           index=range(epochs))

    for epoch in range(epochs):
        sum_num_correct, sum_loss = 0, 0
        cluster_counts = [[[0.0, 0.0] for _ in range(num_clusters)] for _ in range(num_labels)]
        cluster_grads = [[[] for _ in range(num_clusters)] for _ in range(num_labels)]

        for batch_idx, (data, target, cluster) in enumerate(train_loader):
            data, target, protect = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float), \
                                    cluster.to(device, dtype=torch.float)
            optimizer.zero_grad()
            logits, output = model(data)

            weights = torch.tensor([cluster_weights[int(t)][int(c)] for c, t in zip(cluster, target)],
                                   requires_grad=True).type(torch.float)

            loss = my_BCELoss(output.view_as(target), target, weights)
            pred = (output > 0.5) * 1
            pred = pred.float()

            correct = pred.eq(target.view_as(pred)).sum().item()
            sum_num_correct += correct
            sum_loss += loss.item() * train_loader.batch_size
            loss.backward()
            optimizer.step()

            cluster_counts = cluster_counts_update(cluster_counts, pred, target, cluster)
            cluster_grads = get_cluster_grads(cluster_grads, weights.grad.numpy(), cluster, target)

        sum_loss /= len(train_loader.dataset)
        clusters_accs = [[l[0] / l[1] for l in cluster] for cluster in cluster_counts]
        cluster_grads = [[np.average(l) for l in clusters] for clusters in cluster_grads]
        cluster_new_weights = gradient_updates(cluster_weights, cluster_grads, clusters_accs, cluster_lr)
        cluster_weights = normalize(cluster_weights, cluster_new_weights,
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

    return history


def test(model, device, test_loader):
    """
    Evaluates the quality of the predictions of the trained model on a test set
    :param model: the trained model
    :param device: the device the model is evaluated on
    :param test_loader: the test set
    :return: the model predictions, loss values and accuracy
    """
    model.eval()
    test_loss, correct = 0, 0
    test_pred = torch.zeros(0, 1).to(device)
    with torch.no_grad():
        for data, target, protect in test_loader:
            data, target, protect = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float), \
                                    protect.to(device, dtype=torch.float)
            logit, output = model(data)
            criterion = torch.nn.BCELoss()
            loss = criterion(output, target.view_as(output))
            test_loss += loss.item() * test_loader.batch_size  # sum up loss for each test sample
            pred = (output > 0.5) * 1
            pred = pred.float()
            test_pred = torch.cat([test_pred, pred], 0)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)

    return test_pred, test_loss, test_accuracy
