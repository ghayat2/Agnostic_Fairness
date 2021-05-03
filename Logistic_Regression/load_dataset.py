import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn import preprocessing
from sklearn.metrics import *
from sklearn.utils import resample
from sklearn.ensemble import IsolationForest
import numpy as np
from torch.utils import *
import matplotlib.pyplot as plt
from fairness_metrics import reweighting_weights
import itertools
from logistic_regression_model import *


def get_data(filepath):
    return pd.read_csv(filepath)


def minmax_scale(df):
    minmax_scale = preprocessing.MinMaxScaler().fit(df.values)
    scaled_df = minmax_scale.transform(df.values)
    scaled_df = pd.DataFrame(scaled_df, index=df.index, columns=df.columns)
    return scaled_df


def top_k_proxy_features(df, label, protect, k):
    correlations = []
    for feature in df:
        if feature not in [protect, label]:
            correlation_score = normalized_mutual_info_score(df[feature], df[protect], average_method='arithmetic')
            correlations.append((feature, correlation_score))
    top_k = sorted(correlations, key=lambda kv: kv[1], reverse=True)[:k]
    print(top_k[0], "is the most correlacted attribute")
    return top_k


def df_without_k_proxies(df, label, protect, k):
    top_k = top_k_proxy_features(df, protect, label, k)
    top_k_features = set([feature for feature, _ in top_k])
    remaining_features = set(df.columns) - top_k_features
    return df[remaining_features]


def balance_df_label(df, label, downsample=True):
    min_label = df.groupby(label).size().min()
    max_label = df.groupby(label).size().max()
    func = lambda sublabel: resample(sublabel, replace=False,
                                     n_samples=min_label if downsample else max_label, random_state=1)
    df_label_balanced = df.groupby(label).apply(func).reset_index(drop=True)
    return df_label_balanced


def balance_df(df, label_col, protected_cols, label_only=False, downsample=True):
    if label_only:
        df_balanced_label = balance_df_label(df, label_col, downsample=downsample)
    else:
        subgroups = pd.DataFrame([])
        for label in df[label_col].unique():
            df_filtered = df[df[label_col] == label]
            min_subgroup = df_filtered.groupby(protected_cols).size().min()
            max_subgroup = df_filtered.groupby(protected_cols).size().max()
            func = lambda subgroup: resample(subgroup, replace=False,
                                             n_samples=min_subgroup if downsample else max_subgroup, random_state=1)
            subgroups = pd.concat(
                [subgroups, df_filtered.groupby(protected_cols).apply(func).reset_index(drop=True)])

        df_balanced_label = balance_df_label(subgroups, label_col, downsample=True)
    return df_balanced_label


@DeprecationWarning
def balance_df_(df, label, protect, label_only=False, downsample=True):
    if label_only:
        df_balanced_label = balance_df_label(df, label, downsample=downsample)
    else:
        majority_label_class, minority_label_class = df[label].value_counts().index[0], df[label].value_counts().index[
            1]
        majority_protect_class, minority_protect_class = df[protect].value_counts().index[0], \
                                                         df[protect].value_counts().index[1]

        # balancing across demographics group for greater label
        df_minority_greater = df[(df[label] == majority_label_class) & (df[protect] == minority_protect_class)]
        df_majority_greater = df[(df[label] == majority_label_class) & (df[protect] == majority_protect_class)]
        df_majority_downsampled_greater = resample(df_majority_greater, replace=False,
                                                   n_samples=len(df_minority_greater), random_state=1)

        # balancing across demographics group for smaller label
        df_minority_smaller = df[(df[label] == minority_label_class) & (df[protect] == minority_protect_class)]
        df_majority_smaller = df[(df[label] == minority_label_class) & (df[protect] == majority_protect_class)]
        df_majority_downsampled_smaller = resample(df_majority_smaller, replace=False,
                                                   n_samples=len(df_minority_smaller), random_state=1)

        df_protect_balanced = pd.concat(
            [df_majority_downsampled_greater, df_majority_downsampled_smaller, df_minority_greater,
             df_minority_smaller])
        # balancing across labels for all demographic groups after balancing across demographic groups

        df_balanced_label = balance_df_label(df_protect_balanced, label, downsample=True)
    conf = confusion_matrix(df_balanced_label[label].values, df_balanced_label[protect].values)
    print(conf, " balanced", label, protect)
    return df_balanced_label


@DeprecationWarning
def balance_df_label_(df, label, downsample=True):
    majority_label, minority_label = df[label].value_counts().index[0], df[label].value_counts().index[1]
    df_minority = df[df[label] == minority_label]
    df_majority = df[df[label] == majority_label]

    if downsample:
        df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=1)
        df_label_balanced = pd.concat([df_majority_downsampled, df_minority])
    else:
        df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=1)
        df_label_balanced = pd.concat([df_majority, df_minority_upsampled])
    return df_label_balanced


def split_train_test(df, train=0.75):
    np.random.seed(seed=1)
    shuffled = np.random.permutation(df.index)
    n_train = int(len(shuffled) * train)
    i_train, i_test = shuffled[:n_train], shuffled[n_train:]
    return df.loc[i_train], df.loc[i_test]


def statistics(df, label_col, protected_cols, verbose=0):
    stats = {}
    for label in df[label_col].unique():
        stats[label] = df[df[label_col] == label].groupby(protected_cols).apply(
            lambda subgroup: (len(subgroup), round(len(subgroup) / len(df[df[label_col] == label]), 3)))
    if verbose:
        print(stats)
        for label in df[label_col].unique():
            print("label: {}: {} samples ({:.2f}%)".format(label, len(df[df[label_col] == label]),
                                                           len(df[df[label_col] == label]) / len(df) * 100))
    return stats


@DeprecationWarning
def statistics_(df, label_col, protected_cols, verbose=0):
    stats = {"Male": df[df["gender"] == 1]["income"].value_counts() / len(df[df["gender"] == 1]),
             "Female": df[df["gender"] == 0]["income"].value_counts() / len(df[df["gender"] == 0]),
             "<50K": df[df["income"] == 0]["gender"].value_counts() / len(df[df["income"] == 0]),
             ">50K": df[df["income"] == 1]["gender"].value_counts() / len(df[df["income"] == 1]),
             "Income": df["income"].value_counts()}

    if verbose:
        print("-" * 20)
        print("Male \n", stats["Male"])
        print("Female \n", stats["Female"])
        print("<50K \n", stats["<50K"])
        print(">50K \n", stats[">50K"])
        print("Income \n", stats["Income"])

    return stats


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, arg_1, arg_2, arg_3):
        if isinstance(arg_1, pd.DataFrame):
            self.constructor_1(arg_1, arg_2, arg_3)
        else:
            self.constructor_2(arg_1, arg_2, arg_3)

    def constructor_1(self, df, label_column, protect_columns):
        'Initialization'
        # self.features = df.drop([label_column, protect_column], axis=1).values ## Fairness through unawarness
        self.features = df.drop([label_column], axis=1).values
        self.label = df[label_column].values
        self.mapping = {element: i for i, element in enumerate(
            itertools.product(*[list(df[col].unique()) for col in protect_columns]))}
        self.protect = np.array(list(map(lambda a: self.mapping[tuple(a)], list(df[protect_columns].values))))

    def constructor_2(self, features, labels, protect):
        self.features = features
        self.label = labels
        self.protect = protect

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.features)

    def __getitem__(self, index):
        'Generates one sample of data'
        X = self.features[index]
        y = self.label[index]
        z = self.protect[index]
        return X, y, z, index

    def copy(self):
        return Dataset(self.features.copy(), self.label.copy(), self.protect.copy())


def train_test_dataset(filepath, label, protect, is_scaled=True, num_proxy_to_remove=0,
                       balanced=None, reweighting=0, init=0, split=0.75):
    df = get_data(filepath)

    # Scaling the dataset
    if is_scaled:
        df = minmax_scale(df)

    # Removing proxy features
    if num_proxy_to_remove > 0:
        df = df_without_k_proxies(df, label, protect, num_proxy_to_remove)

    train_df, test_df = split_train_test(df, train=split)

    # Balancing the dataset
    if balanced is not None:
        train_df = balance_df(df, label, protect, label_only=balanced["train_label_only"],
                              downsample=balanced["downsample"])
        test_df = balance_df(df, label, protect, label_only=balanced["test_label_only"],
                             downsample=balanced["downsample"])

    # Splitting dataset into train, test features
    print("Statistics")
    statistics(train_df, label, protect, verbose=1)
    statistics(test_df, label, protect, verbose=1)

    train_dataset = Dataset(train_df, label, protect)
    test_dataset = Dataset(test_df, label, protect)

    w_minority_train = reweighting_weights(train_df, label, protect) if reweighting == 1 or init else (1.0, 1.0)

    return train_dataset, test_dataset, w_minority_train


def load_split_dataset(filepath, label, protect, is_scaled=True, num_proxy_to_remove=0, balanced=0, keep=1, verbose=0,
                       filters=None):
    df = get_data(filepath)

    # Scaling the dataset
    if is_scaled:
        df = minmax_scale(df)

    # Removing proxy features
    if num_proxy_to_remove > 0:
        df = df_without_k_proxies(df, label, protect, num_proxy_to_remove)

    df_majority = df[df[protect] == 1]
    df_minority = df[df[protect] == 0]

    if keep < 1:
        if filters[0]:
            df_majority = filter(df_majority, label, protect, improve=filters[0] == 1, epochs=100, verbose=verbose,
                                 keep=keep)
        if filters[1]:
            df_minority = filter(df_minority, label, protect, improve=filters[1] == 1, epochs=100, verbose=verbose,
                                 keep=keep)

    train_maj_df, test_maj_df = split_train_test(df_majority)
    train_min_df, test_min_df = split_train_test(df_minority)

    if balanced:
        train_maj_df, train_min_df = balance_df_label(train_maj_df, label, downsample=True), \
                                     balance_df_label(train_min_df, label, downsample=True)
        test_maj_df, test_min_df = balance_df_label(test_maj_df, label, downsample=True), \
                                   balance_df_label(test_min_df, label, downsample=True)

    # Splitting dataset into train, test features
    print("Statistics - majority")
    statistics(train_maj_df, label, protect, verbose=verbose)
    statistics(test_maj_df, label, protect, verbose=verbose)
    print("Statistics - minority")
    statistics(train_min_df, label, protect, verbose=verbose)
    statistics(test_min_df, label, protect, verbose=verbose)

    train_maj_dataset = Dataset(train_maj_df, label, [protect])
    train_min_dataset = Dataset(train_min_df, label, [protect])
    train_dataset = Dataset(pd.concat([train_maj_df, train_min_df]), label, [protect])

    test_maj_dataset = Dataset(test_maj_df, label, [protect])
    test_min_dataset = Dataset(test_min_df, label, [protect])
    test_dataset = Dataset(pd.concat([test_maj_df, test_min_df]), label, [protect])

    return train_maj_dataset, train_min_dataset, train_dataset, test_maj_dataset, test_min_dataset, test_dataset


def filter_outliers(df, n_estimators=100, proportion=0.1):
    od = IsolationForest(n_estimators=n_estimators, contamination=proportion, random_state=1).fit(df)
    od_pred = od.predict(df)

    return df[od_pred == 1]


def filter(df, label, protect, improve, epochs=40, lr=0.001, verbose=1, keep=0.9):
    device = torch.device("cpu")
    dataset = Dataset(df, label, [protect])
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1000, shuffle=False)

    num_features = dataset[0][0].shape[0]
    predictor = Predictor(num_features).to(device)
    optimizer = optim.Adam(predictor.parameters(), lr=lr)

    train(predictor, device, train_loader, optimizer, epochs, verbose=1)
    test_pred, _, _, probs = test(predictor, device, train_loader)

    p = (1 - keep) / 2

    set_1 = [i for i in range(len(test_pred)) if not test_pred[i] and dataset.label[i]] if improve else \
        [i for i in range(len(test_pred)) if not test_pred[i] and not dataset.label[i]]
    set_2 = [i for i in range(len(test_pred)) if test_pred[i] and not dataset.label[i]] if improve else \
        [i for i in range(len(test_pred)) if test_pred[i] and dataset.label[i]]

    indices_to_remove = list(map(lambda pair: pair[1], sorted([(probs[i], i) for i in set_1]))) \
                            [:int(p * len(dataset))] \
                        + list(map(lambda pair: pair[1], sorted([(probs[i], i) for i in set_2]))) \
                            [-int(p * len(dataset)):]

    indices_to_keep = list(set(range(len(dataset))) - set(indices_to_remove))

    if verbose:
        print("-" * 20)
        print(f"Filter dataset from {len(dataset)} -> {len(indices_to_keep)}")
        print("-" * 20)

    return df.iloc[indices_to_keep]
