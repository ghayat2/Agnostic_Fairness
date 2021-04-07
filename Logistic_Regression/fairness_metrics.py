from sklearn.metrics import *
import numpy as np


def binary_confusion_matrix(true_labels, pred_labels, protect, protect_group):
    indices = np.where(protect == protect_group)
    group_pred_labels = pred_labels[indices]
    group_true_labels = true_labels[indices]

    return confusion_matrix(group_true_labels, group_pred_labels)


def false_positive_rate(group_confusion_matrix):
    return group_confusion_matrix[0][1] / np.sum(group_confusion_matrix[0, :])


def true_negative_rate(group_confusion_matrix):
    return 1 - false_positive_rate(group_confusion_matrix)


def false_negative_rate(group_confusion_matrix):
    return group_confusion_matrix[1][0] / np.sum(group_confusion_matrix[1, :])


def true_positive_rate(group_confusion_matrix):
    return 1 - false_negative_rate(group_confusion_matrix)


def false_positive_rate_difference(confusion_matrix_1, confusion_matrix_2):
    return false_positive_rate(confusion_matrix_1) - false_positive_rate(confusion_matrix_2)


def true_positive_rate_difference(confusion_matrix_1, confusion_matrix_2):
    return true_positive_rate(confusion_matrix_1) - true_positive_rate(confusion_matrix_2)


def false_negative_rate_difference(confusion_matrix_1, confusion_matrix_2):
    return false_negative_rate(confusion_matrix_1) - false_negative_rate(confusion_matrix_2)


def average_odds_difference(confusion_matrix_1, confusion_matrix_2):
    fpr_difference = false_positive_rate_difference(confusion_matrix_1, confusion_matrix_2)
    tpr_difference = true_positive_rate_difference(confusion_matrix_1, confusion_matrix_2)
    return 0.5 * (fpr_difference + tpr_difference)


def frac_predicted_positive(confusion_matrix):
    return np.sum(confusion_matrix[:, 1]) / np.sum(confusion_matrix)


def statistical_parity_difference(confusion_matrix_1, confusion_matrix_2):
    frac_prediced_positive_1 = frac_predicted_positive(confusion_matrix_1)
    frac_prediced_positive_2 = frac_predicted_positive(confusion_matrix_2)
    return frac_prediced_positive_1 - frac_prediced_positive_2


def reweighting_weights(df, label, protect):
    """
    Computes the weights of minority class for each label. Note that this method only supports one protected attribute
    :param df: The dataset
    :param label: The label column
    :param protect: The protected attribute list (of size 1)
    :return: The optimal weights in order to have same distribution for each class
    """
    counts_high = df[df[label] == 1][protect[0]].value_counts()
    counts_low = df[df[label] == 0][protect[0]].value_counts()
    Maj, Min = 1, 0
    return counts_low[Maj] / counts_low[Min], counts_high[Maj] / counts_high[Min]


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
