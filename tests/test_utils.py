import numpy as np
import numpy.typing as npt

from voted_perceptron.utils import resize_vector, resize_matrix

NEW_CAPACITY=42

def test_resize_vector_empty():
    empty_vec = np.empty((0,))
    empty_vec = resize_vector(empty_vec, NEW_CAPACITY)

    assert empty_vec.shape == (NEW_CAPACITY,)

def test_resize_vector_normal():
    old_capacity = 2

    normal_vec = np.ones((old_capacity))
    normal_vec = resize_vector(normal_vec, NEW_CAPACITY)

    assert normal_vec.shape == (NEW_CAPACITY,)
    assert np.array_equal(normal_vec[:old_capacity], np.array([1, 1]))
