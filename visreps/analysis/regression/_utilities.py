from collections import defaultdict
from collections.abc import Collection

import numpy as np
import torch

from visreps.analysis.regression._definition import Regression


def create_splits(
    n: int,
    *,
    n_folds: int,
    shuffle: bool,
    seed: int,
) -> list[np.ndarray]:
    if shuffle:
        rng = np.random.default_rng(seed=seed)
        indices = rng.permutation(n)
    else:
        indices = np.arange(n)

    return np.array_split(indices, n_folds)


def create_stratified_splits(
    y: np.ndarray,
    *,
    n_folds: int,
    shuffle: bool,
    seed: int,
    balance_first_column_only: bool = True,
) -> list[np.ndarray]:
    if not balance_first_column_only:
        assert y.ndim == 1, "y must be a 1D array if balance_first_column_only is False"
    rng = np.random.default_rng(seed=seed)
    if y.ndim > 1:
        y = y[:, 0]

    class_indices = defaultdict(list)
    for idx, label in enumerate(y):
        class_indices[label].append(idx)

    if shuffle:
        for label in class_indices:
            rng.shuffle(class_indices[label])

    folds = [[] for _ in range(n_folds)]
    for label, indices in class_indices.items():
        split_indices = np.array_split(indices, n_folds)
        for fold, split in zip(folds, split_indices, strict=False):
            fold.extend(split)

    return [np.array(fold) for fold in folds]


def regression(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    model: Regression,
    indices_train: Collection[int] | None = None,
    indices_test: Collection[int] | None = None,
    evaluate_only: bool = False,
    train_score: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if indices_train is None and indices_test is not None:
        indices_train = np.setdiff1d(np.arange(x.shape[-2]), np.array(indices_test))
    elif indices_test is None and indices_train is not None:
        indices_test = np.setdiff1d(np.arange(x.shape[-2]), np.array(indices_train))
    elif indices_train is None and indices_test is None:
        indices_train = np.arange(x.shape[-2])
        indices_test = np.arange(x.shape[-2])

    x_train, x_test = x[..., indices_train, :], x[..., indices_test, :]
    y_train, y_test = y[..., indices_train, :], y[..., indices_test, :]

    if not evaluate_only:
        model.fit(x=x_train, y=y_train)

    if train_score:
        y_predicted = model.predict(x_train)
        return y_train, y_predicted

    y_predicted = model.predict(x_test)
    return y_test, y_predicted


def regression_cv(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    model: Regression,
    n_folds: int,
    shuffle: bool,
    seed: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    y_true, y_predicted = [], []

    splits = create_splits(n=y.shape[-2], n_folds=n_folds, shuffle=shuffle, seed=seed)
    for indices_test in splits:
        y_true_, y_predicted_ = regression(
            model=model,
            x=x,
            y=y,
            indices_test=indices_test,
        )
        y_true.append(y_true_)
        y_predicted.append(y_predicted_)

    return y_true, y_predicted
