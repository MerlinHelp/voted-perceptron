"""<TODO>: A one-line summary of the module or program, terminated by a period.

<TODO>: Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.function_bar()
"""
import numpy as np
import numpy.typing as npt
from collections.abc import Callable

from voted_perceptron.utils import resize_vector, resize_matrix

INITIAL_CAPACITY = 100_000
ConfidenceFn = Callable[..., None]

def train_linear_voted_perceptron(
    X_train: npt.NDArray[np.float32],
    y_train: npt.NDArray[np.float32],
    epochs: int
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.uint32],
    int
]:
    """Trains a linear voted perceptron model.

    <TODO>: Write more detailed description

    Args:
        X_train: Training feature matrix of shape (n_samples, n_features).
        y_train: Binary label (-1 or +1) arr of shape (n_samples,).
        epochs: Number of training passes of the perceptron algorithm over the
            dataset.

    Returns:
        M_x: Misclassified feature vectors of shape (mistake_count, n_features).
        M_y: True labels corresponding to mistakes of shape (mistake_count,).
        c_vec: Vote counts vector of shape (mistake_count + 1,).
        n_support_vectors: Number of unique mistakes stored.
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    capacity = INITIAL_CAPACITY

    # Preallocate perceptrons array to avoid repeated memory allocation
    M_x = np.empty((capacity, X_train.shape[1]), dtype=np.float32)
    M_y = np.empty(capacity, dtype=np.float32)
    c_vec = np.empty(capacity, dtype=np.uint32)
    c_vec[0] = 0
    mistake_count = 0
    supvec_indices = set()

    v_curr = np.zeros(X_train.shape[1], dtype=np.float32)
    for _ in range(epochs):
        for i in range(len(X_train)):
            x_i = X_train[i]
            y_i = y_train[i]

            y_hat = 1 if np.dot(v_curr, x_i) >= 0 else -1

            if y_hat == y_i:
                c_vec[mistake_count] += 1
            else:
                # mistake_count - 1 since c_vec is 1 size greater
                if mistake_count - 1 >= capacity:
                    capacity *= 2
                    M_x = resize_matrix(M_x, capacity)
                    M_y = resize_vector(M_y, capacity)
                    c_vec = resize_vector(c_vec, capacity)

                M_x[mistake_count] = x_i
                M_y[mistake_count] = y_i
                supvec_indices.add(i)
                v_curr += y_i * x_i
                mistake_count += 1
                c_vec[mistake_count] = 1

    return (
        M_x[:mistake_count].copy(),
        M_y[:mistake_count].copy(),
        c_vec[:mistake_count + 1].copy(),
        len(supvec_indices)
    )

def train_kernel_voted_perceptron(
    X_train: npt.NDArray[np.float32],
    y_train: npt.NDArray[np.float32],
    epochs: int,
    d: int
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.uint32],
    int
]:
    """Trains a voted perceptron model using the kernel method.

    <TODO>: Write more detailed description

    Args:
        X_train: Training feature matrix of shape (n_samples, n_features).
        y_train: Binary label (-1 or +1) arr of shape (n_samples,).
        epochs: Number of training passes of the perceptron algorithm over the
            dataset.
        d: Degree used for polynomial expansion used in kernel method.

    Returns:
        M_x: Misclassified feature vectors of shape (mistake_count, n_features).
        M_y: True labels corresponding to mistakes of shape (mistake_count,).
        c_vec: Vote counts vector of shape (mistake_count + 1,).
        n_support_vectors: Number of unique mistakes stored.
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    capacity = INITIAL_CAPACITY

    # Preallocate mistakes array to avoid repeated memory allocation
    M_x = np.empty((capacity, X_train.shape[1]), dtype=np.float32)
    M_y = np.empty(capacity, dtype=np.float32)
    c_vec = np.empty(capacity, dtype=np.uint32)
    c_vec[0] = 0
    mistake_count = 0
    supvec_indices = set()

    for _ in range(epochs):
        for i in range(len(X_train)):
            x_i = X_train[i]
            y_i = y_train[i]

            v_dot_x = np.sum(
                M_y[:mistake_count] *
                (np.dot(M_x[:mistake_count], x_i) + 1.0) ** d
            )

            y_hat = 1 if v_dot_x >= 0 else -1

            if y_hat == y_i:
                c_vec[mistake_count] += 1
            else:
                if mistake_count - 1 >= capacity:
                    capacity *= 2
                    M_x = resize_matrix(M_x, capacity)
                    M_y = resize_vector(M_y, capacity)
                    c_vec = resize_vector(c_vec, capacity)

                M_x[mistake_count] = x_i
                M_y[mistake_count] = y_i
                supvec_indices.add(i)
                mistake_count += 1
                c_vec[mistake_count] = 1

    return (
        M_x[:mistake_count].copy(),
        M_y[:mistake_count].copy(),
        c_vec[:mistake_count + 1].copy(),
        len(supvec_indices)
    )

def evaluate_model(
    X_test: npt.NDArray[np.float32],
    y_test: npt.NDArray[np.float32],
    M_x: npt.NDArray[np.float32],
    M_y: npt.NDArray[np.float32],
    c_vec: npt.NDArray[np.uint32],
    confidence_methods: dict[str, ConfidenceFn],
    batch_size: int = 1000,
    kernel: bool = False,
    d: int = 4
) -> tuple[
    dict[str, npt.NDArray],
    npt.NDArray
]:
    """Evaluates a voted perceptron model using batch inference.

    Computes predictions using either linear or kernel voting. Also evaluates
    multiple confidence metrics via pluggable evaluation functions.

    Each confidence function writes results into a shared output array.

    Args:
        M_x: Misclassified feature vectors of shape (mistake_count, n_features).
        M_y: True labels corresponding to mistakes of shape (mistake_count,).
        c_vec: Vote counts vector of shape (mistake_count + 1,).
        confidence_methods: Mapping of confidence metric functions. For example,
            {"conf_1": Callable[..., None], "conf_2": Callable[..., None], ...}
        batch_size: Batch size for evaluation.
        kernel: Bool deciding whether or not to use kernel trick for prediction.
        d: Polynomial kernel degree.

    Returns:
        A tuple containing a dict, mapping each confidence method name to the
        corresponding confidence arrays for each method, and the mistake
        mask NumPy array indicating where the misclassified examples are.

        For example:
        (
            {"conf_1": array([4, 2, 6, 7]), "conf_2": array([3, 1, 4, 1])},
            [True, True, False, False]
        )
    """
    y_pred_all = np.zeros(len(X_test))
    if not kernel:
       updates = M_x * M_y[:, np.newaxis]
       V = np.vstack([np.zeros(M_x.shape[1]), np.cumsum(updates, axis=0)])

    res = {}

    # Confidence arrays
    for name in confidence_methods:
        res[name] = np.zeros(len(X_test))

    for i in range(0, len(X_test), batch_size):
        X_batch = X_test[i:i+batch_size]
        batch_len = len(X_batch)
        
        if len(M_x) > 0:
            if not kernel:
                dots_u = np.dot(X_batch, V.T)
                signs = np.where(dots_u >= 0, 1, -1).astype(np.int8)
            else:
                if d == 1:
                    K_matrix = (np.dot(X_batch, M_x.T) + 1.0)
                else:
                    K_matrix = (np.dot(X_batch, M_x.T) + 1.0) ** d
                
                K_weighted = K_matrix * M_y
            
                # Tracking incremental votes
                H_vals = np.hstack(
                    [
                        np.zeros((batch_len, 1)),
                        np.cumsum(K_weighted, axis=1)
                    ]
                )

                signs = np.where(H_vals >= 0, 1, -1).astype(np.int8)
            
            batch_scores = np.dot(signs, c_vec)
        else:
            signs = np.ones((batch_len, 1))
            batch_scores = np.ones(batch_len) * c_vec[0]
            
        # Standard Prediction based on weighted votes
        y_pred_batch = np.where(batch_scores >= 0, 1, -1).astype(np.int8)
        y_pred_all[i:i+batch_len] = y_pred_batch
        
        shared = {
            "signs": signs,
            "i": i,
            "batch_len": len(X_batch),
            "batch_scores": batch_scores
        }

        for name, func in confidence_methods.items():
            context = shared.copy()
            context["conf_result"] = res[name]
            func(**context)

    mistakes_mask = (y_pred_all != y_test)
    return (res, mistakes_mask)

def evaluate_conf_streak(
    *,
    signs: npt.NDArray[np.int8],
    i: int,
    batch_len: int,
    conf_result: npt.NDArray,
    **kwargs
):
    """Computes confidence based on consecutive agreement streak length.

    Counts how many of the perceptrons vote the same in a row, counting back
    from the last perceptron.

    Writes the results into the provided conf_result array in-place.

    Args:
        signs: Vote history matrix.
        i: Global test example start index of batch.
        batch_len: Number of samples in batch.
        conf_result: Output array for confidence values.

    Returns:
        None
    """
    for j in range(batch_len):
        global_j = i + j
        signs_sample = signs[j]
        last_vote = signs_sample[-1]
        reversed_signs = signs_sample[::-1]
        matches = (reversed_signs == last_vote)

        streak = len(signs_sample) if np.all(matches) else np.argmin(matches)
        conf_result[global_j] = streak

def evaluate_conf_abs_diff(
    *,
    signs: npt.NDArray[np.int8],
    i: int,
    batch_len: int,
    conf_result: npt.NDArray,
    **kwargs
):
    """Computes confidence based on the voting absolute difference.

    In other words, evaluates confidence using |#(+) and #(-)|.

    Writes results into the provided confidence array in-place.

    Args:
        signs: Vote history matrix for the batch.
        i: Global test example start index of batch.
        batch_len: Number of samples in batch.
        conf_result: Output array for confidence values.

    Returns:
        None
    """
    conf_result[i:i+batch_len] = np.abs(np.sum(signs, axis=1))

def evaluate_conf_weighted(
    *,
    batch_scores: npt.NDArray[np.float32],
    i: int,
    batch_len: int,
    conf_result: npt.NDArray,
    **kwargs
):
    """Computes confidence using weighted vote scores.

    In other words, take into account how many samples were correctly
    classified by each perceptron.

    Writes results into the provided confidence array in-place.

    Args:
        batch_scores: Weighted vote scores for the batch.
        i: Global test example start index of batch.
        batch_len: Number of samples in batch.
        conf_result: Output array for confidence values.

    Returns:
        None
    """
    conf_result[i:i+batch_len] = np.abs(batch_scores)

def get_error_rates(
    confidences: npt.NDArray,
    mistakes: npt.NDArray
) -> tuple[
    npt.NDArray,
    npt.NDArray,
    npt.NDArray
]:
    """Computes error rate and coverage across confidence thresholds.

    Evaluates model performance under different confidence cutoffs.

    Args:
        confidences: Confidence scores for each prediction.
        mistakes: Boolean array indicating misclassifications.

    Returns:
        thresholds: Array of confidence thresholds.
        error_rates: Error rate at each threshold.
        coverages: Coverage percentage at each threshold.
    """
    NUM_THRESHOLDS = 100
    thresholds = np.linspace(
        np.min(confidences),
        np.max(confidences),
        NUM_THRESHOLDS
    )
    error_rates = np.empty(NUM_THRESHOLDS, dtype=np.float32)
    coverages = np.empty(NUM_THRESHOLDS, dtype=np.float32)
    total_samples = len(confidences)
    
    for i in range(NUM_THRESHOLDS):
        t = thresholds[i]
        keep_mask = confidences >= t
        retained = np.sum(keep_mask)
        coverage_pct = (retained / total_samples) * 100
        
        # Enforce minimum 20 samples to prevent random variance artifacts
        if retained > 20: 
            error_rate = np.mean(mistakes[keep_mask]) * 100
            error_rates[i] = error_rate
        else:
            error_rates[i] = np.nan
            
        coverages[i] = coverage_pct
            
    return (thresholds, error_rates, coverages)
