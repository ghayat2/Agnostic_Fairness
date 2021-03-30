import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


class Predictor(nn.Module):
    def __init__(self, num_predictor_features):
        super(Predictor, self).__init__()
        self.linear = torch.nn.Linear(num_predictor_features, 1)

    def forward(self, x):
        y_logits = self.linear(x)
        y_pred = F.sigmoid(y_logits)
        return y_logits, y_pred


def cluster_counts_update(counts, preds, targets, clusters):
    for pred, target, cluster in zip(preds, targets, clusters):
        counts[int(target)][int(cluster)][0] += int(pred == target)
        counts[int(target)][int(cluster)][1] += 1
    return counts


def update_weights(weights, cluster_accs):
    cst = 1
    for i, cluster in enumerate(cluster_accs):
        avg_acc = np.mean(cluster)
        for j, acc in enumerate(cluster):
            if avg_acc > acc:
                weights[i][j] += (avg_acc - acc) * cst
    return weights


def my_BCELoss(preds, labels, weights):
    return torch.mean(-weights * (labels * torch.log(preds) + (1 - labels) * torch.log(1 - preds)))


def get_cluster_grads(grads, cluster, target, num_clusters=2, num_labels=2):
    cluster_grads = [[[] for _ in range(num_clusters)] for _ in range(num_labels)]
    for i, (c, t) in enumerate(zip(cluster, target)):
        cluster_grads[int(t)][int(c)].append(grads[i])
    return [[np.average(l) for l in clusters] for clusters in cluster_grads]


def gradient_ascent(weights, grads, lr):
    return [[w + lr * g for w, g in zip(cluster_weights, cluster_grads)] for cluster_weights, cluster_grads in
            zip(weights, grads)]


def train(model, device, train_loader, optimizer, epochs, verbose=1, minority_w=(1, 1)):
    model.train()
    history = pd.DataFrame([], columns=["loss", "accuracy"], index=range(epochs))

    for epoch in range(epochs):
        sum_num_correct = 0
        sum_loss = 0

        batches = enumerate(train_loader)

        for batch_idx, (data, target, protect) in batches:
            data, target, protect = data.to(device, dtype=torch.float), target.to(device,
                                                                                  dtype=torch.float), protect.to(
                device, dtype=torch.float)
            optimizer.zero_grad()
            logits, output = model(data)

            weights = torch.tensor(
                [minority_w[0] if not target[i] and not protect[i] else 1 for i in range(len(data))]).type(torch.float) \
                      * torch.tensor(
                [minority_w[1] if target[i] and not protect[i] else 1 for i in range(len(data))]).type(
                torch.float)

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
    model.train()
    cluster_weights = [[1.0 for _ in range(num_clusters)] for _ in range(num_labels)]
    history = pd.DataFrame([], columns=["loss", "accuracy"] +
                                       [f"cluster_acc_{t}{s}" for s in range(num_clusters) for t in range(num_labels)] +
                                       [f"cluster_weight_{t}{s}" for s in range(num_clusters) for t in
                                        range(num_labels)],
                           index=range(epochs))

    for epoch in range(epochs):
        sum_num_correct = 0
        sum_loss = 0

        batches = enumerate(train_loader)
        cluster_counts = [[[0, 0] for _ in range(num_clusters)] for _ in range(num_labels)]

        for batch_idx, (data, target, cluster) in batches:
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
            cluster_grads = get_cluster_grads(weights.grad.numpy(), cluster, target, num_clusters, num_labels)
            cluster_weights = gradient_ascent(cluster_weights, cluster_grads, cluster_lr) # weights update by gradient ascent

        sum_loss /= len(train_loader.dataset)
        clusters_accs = [[l[0] / l[1] for l in cluster] for cluster in cluster_counts]
        acc = 100. * sum_num_correct / len(train_loader.dataset)
        # cluster_weights = update_weights(cluster_weights, clusters_accs) # weights update by customized heuristic

        history.iloc[epoch] = [sum_loss, acc] + list(np.array(clusters_accs).reshape((-1))) + list(
            np.array(cluster_weights).reshape((-1)))

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
    model.eval()
    test_loss = 0
    correct = 0
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
