"""Code taken from:
https://github.com/Trusted-AI/AIF360/blob/master/aif360/algorithms/postprocessing/calibrated_eq_odds_postprocessing.py
"""
import numpy as np

import fairness_metrics


class CalibratedEqOddsPostprocessing(object):
    """Calibrated equalized odds postprocessing is a post-processing technique
    that optimizes over calibrated classifier score outputs to find
    probabilities with which to change output labels with an equalized odds
    objective [7]_.
    References:
        .. [7] G. Pleiss, M. Raghavan, F. Wu, J. Kleinberg, and
           K. Q. Weinberger, "On Fairness and Calibration," Conference on Neural
           Information Processing Systems, 2017
    Adapted from:
    https://github.com/gpleiss/equalized_odds_and_calibration/blob/master/calib_eq_odds.py
    """

    def __init__(self, unprivileged_groups, privileged_groups,
                 cost_constraint='weighted', seed=None):
        """
        Args:
            unprivileged_groups (dict or list(dict)): Representation for
                unprivileged group.
            privileged_groups (dict or list(dict)): Representation for
                privileged group.
            cost_contraint: fpr, fnr or weighted
            seed (int, optional): Seed to make `predict` repeatable.
        """

        self.seed = seed
        self.model_params = None
        self.unprivileged_groups = [unprivileged_groups] \
            if isinstance(unprivileged_groups, dict) else unprivileged_groups
        self.privileged_groups = [privileged_groups] \
            if isinstance(privileged_groups, dict) else privileged_groups
        self.cost_constraint = cost_constraint
        if self.cost_constraint == 'fnr':
            self.fn_rate = 1
            self.fp_rate = 0
        elif self.cost_constraint == 'fpr':
            self.fn_rate = 0
            self.fp_rate = 1
        elif self.cost_constraint == 'weighted':
            self.fn_rate = 1
            self.fp_rate = 1

        self.base_rate_priv = 0.0
        self.base_rate_unpriv = 0.0

    def fit(self, dataset_true, dataset_pred, favorable_label=1):
        """Compute parameters for equalizing generalized odds using true and
        predicted scores, while preserving calibration.
        Args:
            dataset_true (BinaryLabelDataset): Dataset containing true `labels`.
            dataset_pred (BinaryLabelDataset): Dataset containing predicted
                `scores`.
        Returns:
            CalibratedEqOddsPostprocessing: Returns self.
        """

        self.base_rate_priv = self.base_rate(dataset_true.label, dataset_true.protect, favorable_label,
                                             self.privileged_groups[0]["sex"])
        self.base_rate_unpriv = self.base_rate(dataset_true.label, dataset_true.protect, favorable_label,
                                               self.unprivileged_groups[0]["sex"])

        # Create a dataset with "trivial" predictions
        trivial_scores = dataset_pred.copy()
        trivial_scores[(dataset_true.protect == self.privileged_groups[0][
            "sex"]).reshape((-1))] = self.base_rate_priv
        trivial_scores[(dataset_true.protect == self.unprivileged_groups[0][
            "sex"]).reshape((-1))] = self.base_rate_unpriv

        priv_cost = weighted_cost(dataset_true.label, dataset_pred, dataset_true.protect, self.fp_rate, self.fn_rate,
                                  self.privileged_groups[0]["sex"], self.base_rate_priv, self.base_rate_unpriv)
        unpriv_cost = weighted_cost(dataset_true.label, dataset_pred, dataset_true.protect, self.fp_rate, self.fn_rate,
                                  self.unprivileged_groups[0]["sex"], self.base_rate_priv, self.base_rate_unpriv)
        priv_trivial_cost = weighted_cost(dataset_true.label, trivial_scores, dataset_true.protect,
                                                        self.fp_rate, self.fn_rate, self.privileged_groups[0]["sex"],
                                                        self.base_rate_priv, self.base_rate_unpriv)
        unpriv_trivial_cost = weighted_cost(dataset_true.label, trivial_scores, dataset_true.protect,
                                                          self.fp_rate, self.fn_rate,self.unprivileged_groups[0]["sex"],
                                                          self.base_rate_priv, self.base_rate_unpriv)

        unpriv_costs_more = unpriv_cost > priv_cost
        self.priv_mix_rate = (unpriv_cost - priv_cost) / (priv_trivial_cost - priv_cost) if unpriv_costs_more else 0
        self.unpriv_mix_rate = 0 if unpriv_costs_more else (priv_cost - unpriv_cost) / (
                unpriv_trivial_cost - unpriv_cost)

        return self

    def predict(self, dataset, scores, threshold=0.5, favorable_label = 1, unfavorable_label = 0):
        """Perturb the predicted scores to obtain new labels that satisfy
        equalized odds constraints, while preserving calibration.
        Args:
            dataset (BinaryLabelDataset): Dataset containing `scores` that needs
                to be transformed.
            threshold (float): Threshold for converting `scores` to `labels`.
                Values greater than or equal to this threshold are predicted to
                be the `favorable_label`. Default is 0.5.
        Returns:
            dataset (BinaryLabelDataset): transformed dataset.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        cond_vec_priv = (dataset.protect == self.privileged_groups[0]["sex"]).reshape((-1))
        cond_vec_unpriv = (dataset.protect == self.unprivileged_groups[0]["sex"]).reshape((-1))

        unpriv_indices = (np.random.random(sum(cond_vec_unpriv))
                          <= self.unpriv_mix_rate)
        unpriv_new_pred = scores[cond_vec_unpriv].copy()
        unpriv_new_pred[unpriv_indices] = self.base_rate_unpriv

        priv_indices = (np.random.random(sum(cond_vec_priv))
                        <= self.priv_mix_rate)
        priv_new_pred = scores[cond_vec_priv].copy()
        priv_new_pred[priv_indices] = self.base_rate_priv

        dataset_new = dataset.copy()
        new_scores = scores.copy()

        new_scores = np.zeros_like(scores, dtype=np.float64)
        new_scores[cond_vec_priv] = priv_new_pred
        new_scores[cond_vec_unpriv] = unpriv_new_pred

        # Create labels from scores using a default threshold
        dataset_new.label = np.where(new_scores >= threshold,
                                      favorable_label,
                                      unfavorable_label)
        return dataset_new, new_scores

    def fit_predict(self, dataset_true, dataset_pred, threshold=0.5):
        """fit and predict methods sequentially."""
        return self.fit(dataset_true, dataset_pred).predict(
            dataset_pred, threshold=threshold)

    def base_rate(self, labels, groups, favorable_label, to_keep):
        """Compute the base rate, :math:`Pr(Y = 1) = P/(P+N)`, optionally
        conditioned on protected attributes.
        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.
        Returns:
            float: Base rate (optionally conditioned).
        """
        a, b = 0, 0
        for l, g in zip(labels, groups):
            if g == to_keep:
                if l == favorable_label:
                    a += 1
                b += 1
        return a / b


######### SUPPORTING FUNCTIONS ##########

def weighted_cost(labels, preds, groups, fp_rate, fn_rate, to_keep, base_rate_priv, base_rate_unpriv):
    norm_const = float(fp_rate + fn_rate) if \
        (fp_rate != 0 and fn_rate != 0) else 1
    return ((fp_rate / norm_const
             * fairness_metrics.generalized_false_positive_rate(labels, preds, groups, to_keep)
             * (1 - base_rate_priv) +
             (fn_rate / norm_const
              * fairness_metrics.generalized_false_negative_rate(labels, preds, groups, to_keep)
              * base_rate_unpriv)))

