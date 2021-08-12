from sklearn.metrics import *
import numpy as np
from logistic_regression_model import test
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm


def binary_confusion_matrix(true_labels, pred_labels, protect, protect_group):
    """
    Confusion matrix between labels and predictions
    :param true_labels:  labels
    :param pred_labels: predictions
    :param protect: protected attribute
    :param protect_group: the protected value
    :return: confusion matrix
    """
    indices = np.where(protect == protect_group)
    group_pred_labels = pred_labels[indices]
    group_true_labels = true_labels[indices]

    return confusion_matrix(group_true_labels, group_pred_labels)


def generalized_binary_confusion_matrix(true_labels, pred_scores, protect, protect_group, unfavorable_label=0,
                                        favorable_label=1):
    """
    Confusion matrix with soft predictions (logits are summed up to yield cell value, instead of hard classification)
    :param true_labels: labels
    :param pred_scores:  predictions
    :param protect: protected attribute
    :param protect_group: protected value
    :param unfavorable_label:unfavorable label
    :param favorable_label: favorable label
    :return:
    """
    indices = np.where(protect == protect_group)
    group_pred_scores = pred_scores[indices]
    group_true_labels = true_labels[indices]

    GTN = np.sum((1 - group_pred_scores)[group_true_labels == unfavorable_label])
    GFN = np.sum((1 - group_pred_scores)[group_true_labels == favorable_label])
    GTP = np.sum(group_pred_scores[group_true_labels == favorable_label])
    GFP = np.sum(group_pred_scores[group_true_labels == unfavorable_label])

    return [[GTN, GFP], [GFN, GTP]]


def generalized_false_positive_rate(true_labels, preds, groups, to_keep):
    cm = generalized_binary_confusion_matrix(true_labels, preds, groups, to_keep)
    return cm[0][1] / sum(cm[0])


def generalized_false_negative_rate(true_labels, preds, groups, to_keep):
    cm = generalized_binary_confusion_matrix(true_labels, preds, groups, to_keep)
    return cm[1][0] / sum(cm[1])


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


@DeprecationWarning
def reweighting_weights_(df, label, protect):
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


def reweighting_weights(df, label, protect):
    """
    Returns the weights to apply to the samples from the minority group for the two classes
    Note:
    This method for binary classification when there is one minority group
    :param df: the dataset
    :param label: label == 0 is the unfavorable class (unfav), label == 1 is the favorable class (fav)
    :param protect: protect == 0 is the unprivilege (up) group, protect == 1 is the privilege group (p)
    :return: the weights to apply to the minority group when training
    """
    # This method works only when there is one protected attribute
    protect = protect[0]

    n = len(df)
    n_unfav, n_fav = len(df[df[label] == 0]), len(df[df[label] == 1])
    n_up, n_p = len(df[df[protect] == 0]), len(df[df[protect] == 1]),

    n_up_unfav, n_p_unfav = len(df[np.logical_and(df[label] == 0, df[protect] == 0)]), len(
        df[np.logical_and(df[label] == 0, df[protect] == 1)])
    n_up_fav, n_p_fav = len(df[np.logical_and(df[label] == 1, df[protect] == 0)]), len(
        df[np.logical_and(df[label] == 1, df[protect] == 1)])

    w_p_fav = n_fav * n_p / (n * n_p_fav)
    w_p_unfav = n_unfav * n_p / (n * n_p_unfav)
    w_up_fav = n_fav * n_up / (n * n_up_fav)
    w_up_unfav = n_unfav * n_up / (n * n_up_unfav)

    return w_up_unfav / w_p_unfav, w_up_fav / w_p_fav


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


def scatter_plots(device, predictor_maj, predictor_min, predictor_comb, test_maj_loader,
                  test_min_loader, test_comb_loader):
    """
    Constructs a scatter plot that compares the model's predictions with the base rate models. It points out which
    data samples were misclassified by the model and verifies that these were hard to classify by the base models as
    well. In addition, the plot reflects the level of bias in the predictions by analying the symmetry of the plot.
    :param device: The pytorch device
    :param predictor_maj: the base rate model for the majority data
    :param predictor_min: the base rate model for the minority data
    :param predictor_comb: the model to be evaluated
    :param test_maj_loader: the majority test dataloader
    :param test_min_loader: the minority test dataloader
    :param test_comb_loader: the test dataloader
    :return: a scatter plot illustrating the above.
    """

    maj_pred, _, _, maj_probs = test(predictor_maj, device, test_maj_loader)
    min_pred, _, _, min_probs = test(predictor_min, device, test_min_loader)
    maj_pred, maj_probs = maj_pred.numpy().reshape((-1)), maj_probs.numpy().reshape((-1))
    min_pred, min_probs = min_pred.numpy().reshape((-1)), min_probs.numpy().reshape((-1))

    comb_pred, _, _, comb_probs = test(predictor_comb, device, test_comb_loader)
    comb_pred, comb_probs = comb_pred.numpy().reshape((-1)), comb_probs.numpy().reshape((-1))
    comb_maj_pred, comb_maj_probs = comb_pred[:len(test_maj_loader.dataset)], comb_probs[:len(test_maj_loader.dataset)]
    comb_min_pred, comb_min_probs = comb_pred[len(test_maj_loader.dataset):], comb_probs[len(test_maj_loader.dataset):]

    s1_maj, s2_maj, s3_maj = maj_probs[np.logical_and(maj_pred == test_maj_loader.dataset.label,
                                                      comb_maj_pred == test_maj_loader.dataset.label)], \
                             maj_probs[np.logical_and(maj_pred == test_maj_loader.dataset.label,
                                                      comb_maj_pred != test_maj_loader.dataset.label)], \
                             maj_probs[np.logical_and(maj_pred != test_maj_loader.dataset.label,
                                                      comb_maj_pred == test_maj_loader.dataset.label)]

    s1_min, s2_min, s3_min = min_probs[np.logical_and(min_pred == test_min_loader.dataset.label,
                                                      comb_min_pred == test_min_loader.dataset.label)], \
                             min_probs[np.logical_and(min_pred == test_min_loader.dataset.label,
                                                      comb_min_pred != test_min_loader.dataset.label)], \
                             min_probs[np.logical_and(min_pred != test_min_loader.dataset.label,
                                                      comb_min_pred == test_min_loader.dataset.label)]

    X = list(s1_maj) + list(s2_maj) + list(s1_min) + list(s2_min)
    Y = [0.25] * (len(s1_maj) + len(s2_maj)) + [-0.25] * (len(s1_min) + len(s2_min))
    C = ["Green"] * len(s1_maj) + ["Red"] * len(s2_maj) + ["Green"] * len(s1_min) + ["Red"] * len(s2_min)

    plt.figure(1, figsize=(18, 12))
    plt.title("Accuracy of base models vs model")
    plt.xlabel("Base models' classification accuracies (ie: Pr(Y=1 | X), threshold at 0.5)")
    plt.ylabel("Minority ----- Majority")
    green_patch = mpatches.Patch(color='green', label='corr. class. base model, corr. class. model')
    red_patch = mpatches.Patch(color='red', label='corr. class. base model, incorr. class. model')
    blue_patch = mpatches.Patch(color='blue', label='incorr. class. base model, corr. class. model')
    boundary, = plt.plot([0.5, 0.5], [-0.5, 0.5], label="Decision boundary", linestyle='--', color="y")
    plt.legend(handles=[green_patch, red_patch, blue_patch, boundary], loc="upper right")

    axes = plt.gca()
    axes.set_ylim([-0.5, 0.5])
    axes.hist(s3_maj, weights=[1 / len(s3_maj)] * len(s3_maj), bins=int(len(s3_maj) / 4), color="Blue", alpha=0.7)
    axes.get_yaxis().set_ticks([])

    axes2 = axes.twinx()
    axes2.set_ylim([-0.5, 0.5])
    axes2.invert_yaxis()
    axes2.hist(s3_min, weights=[1 / len(s3_min)] * len(s3_min), bins=int(len(s3_min) / 2), color="darkblue", alpha=0.7)
    axes2.get_yaxis().set_ticks([])

    return axes.scatter(X, Y, s=5, c=C)
