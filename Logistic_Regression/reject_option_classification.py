"""
Code taken from:
https://github.com/Trusted-AI/AIF360/blob/master/aif360/algorithms/postprocessing/reject_option_classification.py
"""

import numpy as np
from warnings import warn

import fairness_metrics
from fairness_metrics import *


class RejectOptionClassification(object):
    """Reject option classification is a postprocessing technique that gives
    favorable outcomes to unpriviliged groups and unfavorable outcomes to
    priviliged groups in a confidence band around the decision boundary with the
    highest uncertainty [10]_.
    References:
        .. [10] F. Kamiran, A. Karim, and X. Zhang, "Decision Theory for
           Discrimination-Aware Classification," IEEE International Conference
           on Data Mining, 2012.
    """

    def __init__(self, unprivileged_groups, privileged_groups,
                 low_class_thresh=0.01, high_class_thresh=0.99,
                 num_class_thresh=100, num_ROC_margin=50,
                 metric_name="Statistical parity difference",
                 metric_ub=0.05, metric_lb=-0.05):
        """
        Args:
            unprivileged_groups (dict or list(dict)): Representation for
                unprivileged group.
            privileged_groups (dict or list(dict)): Representation for
                privileged group.
            low_class_thresh (float): Smallest classification threshold to use
                in the optimization. Should be between 0. and 1.
            high_class_thresh (float): Highest classification threshold to use
                in the optimization. Should be between 0. and 1.
            num_class_thresh (int): Number of classification thresholds between
                low_class_thresh and high_class_thresh for the optimization
                search. Should be > 0.
            num_ROC_margin (int): Number of relevant ROC margins to be used in
                the optimization search. Should be > 0.
            metric_name (str): Name of the metric to use for the optimization.
                Allowed options are "Statistical parity difference",
                "Average odds difference", "Equal opportunity difference".
            metric_ub (float): Upper bound of constraint on the metric value
            metric_lb (float): Lower bound of constraint on the metric value
        """

        allowed_metrics = ["Statistical parity difference",
                           "Average odds difference",
                           "Equal opportunity difference"]

        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups

        self.low_class_thresh = low_class_thresh
        self.high_class_thresh = high_class_thresh
        self.num_class_thresh = num_class_thresh
        self.num_ROC_margin = num_ROC_margin
        self.metric_name = metric_name
        self.metric_ub = metric_ub
        self.metric_lb = metric_lb

        self.classification_threshold = None
        self.ROC_margin = None

        if ((self.low_class_thresh < 0.0) or (self.low_class_thresh > 1.0) or \
                (self.high_class_thresh < 0.0) or (self.high_class_thresh > 1.0) or \
                (self.low_class_thresh >= self.high_class_thresh) or \
                (self.num_class_thresh < 1) or (self.num_ROC_margin < 1)):
            raise ValueError("Input parameter values out of bounds")

        if metric_name not in allowed_metrics:
            raise ValueError("metric name not in the list of allowed metrics")

    def fit(self, dataset_true, dataset_pred, pred_scores):
        """Estimates the optimal classification threshold and margin for reject
        option classification that optimizes the metric provided.
        Note:
            The `fit` function is a no-op for this algorithm.
        Args:
            dataset_true (BinaryLabelDataset): Dataset containing the true
                `labels`.
            dataset_pred (BinaryLabelDataset): Dataset containing the predicted
                `scores`.
        Returns:
            RejectOptionClassification: Returns self.
        """

        fair_metric_arr = np.zeros(self.num_class_thresh * self.num_ROC_margin)
        balanced_acc_arr = np.zeros_like(fair_metric_arr)
        ROC_margin_arr = np.zeros_like(fair_metric_arr)
        class_thresh_arr = np.zeros_like(fair_metric_arr)

        cnt = 0
        # Iterate through class thresholds
        for class_thresh in np.linspace(self.low_class_thresh,
                                        self.high_class_thresh,
                                        self.num_class_thresh):

            self.classification_threshold = class_thresh
            if class_thresh <= 0.5:
                low_ROC_margin = 0.0
                high_ROC_margin = class_thresh
            else:
                low_ROC_margin = 0.0
                high_ROC_margin = (1.0 - class_thresh)

            # Iterate through ROC margins
            for ROC_margin in np.linspace(
                    low_ROC_margin,
                    high_ROC_margin,
                    self.num_ROC_margin):
                self.ROC_margin = ROC_margin

                # Predict using the current threshold and margin
                dataset_transf_pred = self.predict(dataset_pred, pred_scores)

                ROC_margin_arr[cnt] = self.ROC_margin
                class_thresh_arr[cnt] = self.classification_threshold

                # Balanced Accuracy and fairness metric computations
                bcm = confusion_matrix(dataset_true.label, dataset_transf_pred.label)
                balanced_acc_arr[cnt] = 0.5 * (
                        fairness_metrics.true_positive_rate(bcm) + fairness_metrics.true_negative_rate(bcm))

                # balanced_acc_arr[cnt] = np.sum(dataset_true.label == dataset_pred.label)/len(dataset_true.label)
                accs = equalizing_odds(dataset_transf_pred.label, dataset_true.label,
                                                       dataset_true.protect)
                fair_metric_arr[cnt] = np.average([max(acc) - min(acc) for acc in accs])

                cnt += 1

        rel_inds = np.logical_and(fair_metric_arr >= self.metric_lb,
                                  fair_metric_arr <= self.metric_ub)
        if any(rel_inds):
            best_ind = np.where(balanced_acc_arr[rel_inds]
                                == np.max(balanced_acc_arr[rel_inds]))[0][0]
        else:
            warn("Unable to satisy fairness constraints")
            rel_inds = np.ones(len(fair_metric_arr), dtype=bool)
            best_ind = np.where(fair_metric_arr[rel_inds]
                                == np.min(fair_metric_arr[rel_inds]))[0][0]

        self.ROC_margin = ROC_margin_arr[rel_inds][best_ind]
        self.classification_threshold = class_thresh_arr[rel_inds][best_ind]

        return self

    def predict(self, dataset, pred_scores, favorable_label=1, unfavorable_label=0):
        """Obtain fair predictions using the ROC method.
        Args:
            dataset (BinaryLabelDataset): Dataset containing scores that will
                be used to compute predicted labels.
        Returns:
            dataset_pred (BinaryLabelDataset): Output dataset with potentially
            fair predictions obtain using the ROC method.
        """
        dataset_new = dataset.copy()

        fav_pred_inds = (pred_scores > self.classification_threshold)
        unfav_pred_inds = ~fav_pred_inds

        y_pred = np.zeros(pred_scores.shape)
        y_pred[fav_pred_inds] = favorable_label
        y_pred[unfav_pred_inds] = unfavorable_label

        # Indices of critical region around the classification boundary
        crit_region_inds = np.logical_and(
            pred_scores <= self.classification_threshold + self.ROC_margin,
            pred_scores > self.classification_threshold - self.ROC_margin)

        # New, fairer labels
        dataset_new.label = y_pred
        dataset_new.label[np.logical_and(crit_region_inds,
                                         (dataset.protect == self.privileged_groups[0][
                                             "sex"]).reshape((-1, 1)))] = unfavorable_label
        dataset_new.label[np.logical_and(crit_region_inds,
                                         (dataset.protect == self.unprivileged_groups[0][
                                             "sex"]).reshape((-1, 1)))] = favorable_label

        return dataset_new

    def fit_predict(self, dataset_true, dataset_pred):
        """fit and predict methods sequentially."""
        return self.fit(dataset_true, dataset_pred).predict(dataset_pred)


# Function to obtain the pareto frontier
def _get_pareto_frontier(costs, return_mask=True):  # <- Fastest for many points
    """
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask, False to return integer indices of efficient points.
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    adapted from: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for

    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs <= costs[next_point_index], axis=1)
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient
