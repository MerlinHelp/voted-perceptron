import numpy as np
import numpy.typing as npt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import pytest
from typing import cast

import voted_perceptron.perceptron as vp

RANDOM_STATE=42

# -------------------------- FIXTURES --------------------------
@pytest.fixture
def lin_sep_blob_data() -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.int8]
]:
    N_SAMPLES=10
    CLUSTER_STD=0.8

    X_neg, _ = make_blobs( # pyright: ignore
        n_samples=N_SAMPLES,
        centers=[[-4, -4]],
        cluster_std=CLUSTER_STD,
        random_state=RANDOM_STATE,
    )
    y_neg = np.full(N_SAMPLES, -1)

    X_pos, _ = make_blobs( # pyright: ignore
        n_samples=N_SAMPLES,
        centers=[[4, 4]],
        cluster_std=CLUSTER_STD,
        random_state=RANDOM_STATE
    )
    y_pos = np.full(N_SAMPLES, 1)

    X = np.vstack((X_neg, X_pos))
    y = np.concatenate((y_neg, y_pos)).astype(np.int8)

    return (X, y)

@pytest.fixture
def lin_insep_blob_data() -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.int8]
]:

    N_SAMPLES=5
    EASY_CLUSTER_STD=0.8
    HARD_CLUSTER_STD=0.5

    X_easy_neg, _ = make_blobs( # pyright: ignore
        n_samples=N_SAMPLES,
        centers=[[-4, -4]],
        cluster_std=EASY_CLUSTER_STD,
        random_state=RANDOM_STATE,
    )
    y_easy_neg = np.full(N_SAMPLES, -1)

    X_easy_pos, _ = make_blobs( # pyright: ignore
        n_samples=N_SAMPLES,
        centers=[[4, 4]],
        cluster_std=EASY_CLUSTER_STD,
        random_state=RANDOM_STATE
    )
    y_easy_pos = np.full(N_SAMPLES, 1)

    X_hard_neg = make_blobs( # pyright: ignore
        n_samples=N_SAMPLES,
        centers=[[-0.5, 0.5]],
        cluster_std=HARD_CLUSTER_STD,
        random_state=RANDOM_STATE
    )

    y_hard_neg = np.full(N_SAMPLES, 1)

    X_hard_pos, _ = make_blobs( # pyright: ignore
        n_samples=N_SAMPLES,
        centers=[[0.5, -0.5]],
        cluster_std=HARD_CLUSTER_STD,
        random_state=RANDOM_STATE
    )

    y_hard_pos = np.full(N_SAMPLES, 1)

    X = np.vstack((X_easy_neg, X_easy_pos, X_hard_neg, X_hard_pos))
    y = np.concatenate((y_easy_neg, y_easy_pos, y_hard_neg, y_hard_pos))

    return (X, y)

@pytest.fixture
def lin_sep_data() -> tuple[
    npt.NDArray,
    npt.NDArray[np.int8]
]:
    X = np.array([
        [1, 2],
        [2, 1],
        [-1, 0],
        [-1, 1]
    ], dtype=np.float32)

    y = np.array([1, 1, -1, -1], dtype=np.int8)

    return (X, y)

@pytest.fixture
def lin_insep_data() -> tuple[
    npt.NDArray,
    npt.NDArray[np.int8]
]:
    # xor is linearly inseparable
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=np.float32)

    y = np.array([-1, 1, 1, -1], dtype=np.int8)

    return (X, y)

# -------------------------- TESTS --------------------------
def test_linear_perceptron_on_linearly_separable_data(lin_sep_data):
    """
    X_train: array([
        [1, 2],
        [2, 1],
        [-1, 0],
        [-1, 1]
    ])

    y_train: array(
        [1, 1, -1, -1]
    )
    """
    X, y = lin_sep_data 

    M_x, M_y, c_vec, n_support_vectors = vp.train_linear_voted_perceptron(
        X,
        y,
        5
    )

    assert M_x.ndim == 2 
    assert M_y.ndim == 1
    assert c_vec.ndim == 1

    # Recall that c_vec has mistake_count + 1 values
    assert M_x.shape[0] == M_y.shape[0] == c_vec.shape[0] - 1

    res, mistakes = vp.evaluate_model(
        X,
        y,
        M_x,
        M_y,
        c_vec,
        confidence_methods={},
    )

    assert np.sum(mistakes) == 0

def test_linear_perceptron_generalizes_on_linearly_separable_data(lin_sep_blob_data):
    X, y = lin_sep_blob_data

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    M_x, M_y, c_vec, n_support_vectors = vp.train_linear_voted_perceptron(
        X_train,
        y_train,
        5
    )

    res, mistakes = vp.evaluate_model(
        X,
        y,
        M_x,
        M_y,
        c_vec,
        confidence_methods={},
    )

    assert np.sum(mistakes) == 0

def test_kernel_outperforms_linear_on_insep_data(lin_insep_data):
    pass
