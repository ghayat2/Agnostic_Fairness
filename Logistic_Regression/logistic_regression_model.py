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


def cluster_weight_updates_adaptive_lr(weights, grads, accs, exp):
    """
     Updates the cluster weights for the next epoch - the learning rate and direction of the update depends on the
      cluster accuracy
    :param weights: The current weights of the clusters
    :param grads: The average gradient of the cluster weights from the epoch
    :param accs: The cluster accuracy from the epoch
    :param exp: controls the rate increase if the learning rate
    :return: The updated cluster weights
    """
    for cluster_grad in grads:
        for g in cluster_grad:
            if g <= 0:
                print("NEGATIVE")
                import sys
                sys.exit(0)
    return [[w - ((np.average(cluster_accs) - a) * 100) ** exp * g if exp % 2 == 0 and a > np.average(
        cluster_accs) else w + ((np.average(cluster_accs) - a) * 100) ** exp * g for w, g, a
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

        for batch_idx, (data, target, protect, _) in enumerate(train_loader):
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
            print('\nEpoch {}: Train set: Average loss: {:.2e}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                epoch, sum_loss, sum_num_correct, len(train_loader.dataset),
                acc))

    return history


def train_cluster_reweight(model, device, train_loader, optimizer, epochs, verbose=1, num_clusters=2, num_labels=2,
                           cluster_lr=10, cluster_weights=None):
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
    model.train()
    if not cluster_weights:
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

        for batch_idx, (data, target, cluster, _) in enumerate(train_loader):
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

        cluster_new_weights = cluster_weight_updates(cluster_weights, cluster_grads, clusters_accs, cluster_lr)
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

    return history


def train_sample_reweight(model, device, train_loader, optimizer, epochs, verbose=1, num_clusters=2, num_labels=2,
                          sample_lr=10, init_weights=None):
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
    :param sample_lr: The rate at which the samplel weights are being updated
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
            data, target, protect = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float), \
                                    cluster.to(device, dtype=torch.float)
            optimizer.zero_grad()
            logits, output = model(data)
            weights = torch.tensor(
                [sample_weights[int(t)][int(c)][int(i)] for c, t, i in zip(cluster, target, indexes)],
                requires_grad=True).type(torch.float)

            loss = my_BCELoss(output.view_as(target), target, weights)
            pred = (output > 0.5) * 1
            pred = pred.float()

            correct = pred.eq(target.view_as(pred)).sum().item()
            sum_num_correct += correct
            sum_loss += loss.item() * train_loader.batch_size
            loss.backward()
            optimizer.step()

            correct_classifications = update_classification(correct_classifications, indexes, pred, target, cluster)
            sample_grads = get_sample_grads(sample_grads, indexes, weights.grad.numpy(), cluster, target)

        sum_loss /= len(train_loader.dataset)
        clusters_accs = [[sum(cluster.values()) / len(cluster) for cluster in clusters]
                         for clusters in correct_classifications]

        new_sample_weights, updates = sample_weight_updates(train_loader, sample_weights, sample_grads,
                                                            correct_classifications,
                                                            clusters_accs, sample_lr)

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
    test_probs = torch.zeros(0, 1).to(device)
    with torch.no_grad():
        for data, target, protect, i in test_loader:

            data, target, protect = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float), \
                                    protect.to(device, dtype=torch.float)
            logit, output = model(data)
            test_probs = torch.cat([test_probs, output], 0)

            criterion = torch.nn.BCELoss()
            loss = criterion(output, target.view_as(output))
            test_loss += loss.item() * test_loader.batch_size  # sum up loss for each test sample
            pred = (output > 0.5) * 1
            pred = pred.float()
            test_pred = torch.cat([test_pred, pred], 0)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)

    return test_pred, test_loss, test_accuracy, test_probs
