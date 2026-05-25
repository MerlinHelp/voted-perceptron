import numpy as np
import numpy.typing as npt
from collections.abc import Callable

INITIAL_CAPACITY = 100_000
ConfidenceFn = Callable[..., None]

def _resize_mistake_arr(
    M_x: npt.NDArray[np.float32],
    M_y: npt.NDArray[np.float32],
    capacity: int,
    feature_dims: int
) -> (
    tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32]
    ]
):

    new_M_x = np.empty((capacity * 2, feature_dims), dtype=np.float32)
    new_M_x[:capacity] = M_x
    new_M_y = np.empty((capacity, feature_dims), dtype=np.float32)
    new_M_y[:capacity] = M_y

    return (new_M_x, new_M_y)

def train_linear_voted_perceptron(
    X_train: npt.NDArray[np.float32],
    y_train: npt.NDArray[np.float32],
    epochs: int
) -> (
    tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.uint32],
        int
    ]
):
    """Runs Binary Linear Perceptron using dynamically resizing numpy arrays"""
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    capacity = INITIAL_CAPACITY

    # Preallocate mistakes array to avoid repeated memory allocation
    M_x = np.empty((capacity, X_train.shape[1]), dtype=np.float32)
    M_y = np.empty(capacity, dtype=np.float32)
    c_vec = [0]
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
                v_curr += y_i * x_i

                if mistake_count >= capacity:
                    _resize_mistake_arr(M_x, M_y, capacity, X_train.shape[1])

                M_x[mistake_count] = x_i
                M_y[mistake_count] = y_i
                supvec_indices.add(i)
                mistake_count += 1
                c_vec.append(1)

    return (M_x.copy(), M_y.copy(), np.array(c_vec), len(supvec_indices))

def train_kernel_voted_perceptron(
    X_train: npt.NDArray[np.float32],
    y_train: npt.NDArray[np.float32],
    epochs: int,
    d: int
) -> (
    tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.uint32],
        int
    ]
):
    """Runs Binary Kernel Perceptron using dynamically resizing numpy arrays"""
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    capacity = INITIAL_CAPACITY

    # Preallocate mistakes array to avoid repeated memory allocation
    M_x = np.empty((capacity, X_train.shape[1]), dtype=np.float32)
    M_y = np.empty(capacity, dtype=np.float32)
    c_vec = [0]
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
                if mistake_count >= capacity:
                    _resize_mistake_arr(M_x, M_y, capacity, X_train.shape[1])

                M_x[mistake_count] = x_i
                M_y[mistake_count] = y_i
                supvec_indices.add(i)
                mistake_count += 1
                c_vec.append(1)

    return (M_x.copy(), M_y.copy(), np.array(c_vec), len(supvec_indices))

def evaluate_model(
    X_test: npt.NDArray[np.float32],
    y_test: npt.NDArray[np.float32],
    M_x: npt.NDArray[np.float32],
    M_y: npt.NDArray[np.float32],
    c_vec: npt.NDArray[np.uint32],
    confidence_methods: dict[str, ConfidenceFn],
    d: int = 4,
    batch_size: int = 1000
) -> (
    dict[str, npt.NDArray]
):
    """Evaluates the binary model and calculates passed in confidence metrics."""
    
    y_pred_all = np.zeros(len(X_test))

    res = {}

    # Confidence arrays
    for name in confidence_methods:
        res[name] = np.zeros(len(X_test))

    for i in range(0, len(X_test), batch_size):
        X_batch = X_test[i:i+batch_size]
        batch_len = len(X_batch)
        
        if len(M_x) > 0:
            if d == 1:
                K_matrix = (np.dot(X_batch, M_x.T) + 1.0)
            else:
                K_matrix = (np.dot(X_batch, M_x.T) + 1.0) ** d
                
            K_weighted = K_matrix * M_y
            
            # Tracking incremental votes
            H_vals = np.hstack([np.zeros((batch_len, 1)), np.cumsum(K_weighted, axis=1)])
            signs = np.where(H_vals >= 0, 1, -1)
            
            batch_scores = np.dot(signs, c_vec)
        else:
            signs = np.ones((batch_len, 1))
            batch_scores = np.ones(batch_len) * c_vec[0]
            
        # Standard Prediction based on weighted votes
        y_pred_batch = np.where(batch_scores >= 0, 1, -1)
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
    res["mistakes_mask"] = mistakes_mask
    return res

def evaluate_conf_streak(
    *,
    signs: npt.NDArray[np.uint8],
    i: int,
    batch_len: int,
    conf_result: npt.NDArray,
    **kwargs
):
    """Evaluates confidence using METHOD 1: Unbroken Streak"""
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
    signs: npt.NDArray[np.uint8],
    i: int,
    batch_len: int,
    conf_result: npt.NDArray,
    **kwargs
):
    """Evaluates confidence using METHOD 2: Absolute Vote Difference |#+ - #-|"""
    conf_result[i:i+batch_len] = np.abs(np.sum(signs, axis=1))

def evaluate_conf_weighted(
    *,
    i: int,
    batch_len: int,
    batch_scores: npt.NDArray[np.float32],
    conf_result: npt.NDArray,
    **kwargs
):
    """Evaluates confidence using METHOD 3: Survival-Weighted Confidence |Sum(vote * C)|"""
    conf_result[i:i+batch_len] = np.abs(batch_scores)

def get_error_rates(confidences, mistakes):
    """Helper function to calculate error rates and coverages across confidence thresholds."""
    thresholds = np.linspace(np.min(confidences), np.max(confidences), 100)
    error_rates = []
    coverages = []
    total_samples = len(confidences)
    
    for t in thresholds:
        keep_mask = confidences >= t
        retained = np.sum(keep_mask)
        coverage_pct = (retained / total_samples) * 100
        
        # Enforce minimum 20 samples to prevent random variance artifacts
        if retained > 20: 
            error_rate = np.mean(mistakes[keep_mask]) * 100
            error_rates.append(error_rate)
        else:
            error_rates.append(np.nan)
            
        coverages.append(coverage_pct)
            
    return thresholds, error_rates, coverages
