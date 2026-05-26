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

def resize_vector(
    arr: npt.NDArray,
    new_capacity: int
) -> npt.NDArray:
    """Resizes a 1D vector by increasing its capacity.

    Args:
        arr: 1D vector of shape (k,)
        new_capacity: New size to allocate

    Returns:
        Resized array of shape (new_capacity,)
    """
    new_arr = np.empty((new_capacity,), dtype=arr.dtype)
    new_arr[:arr.shape[0]] = arr
    return new_arr

def resize_matrix(
    arr: npt.NDArray,
    new_capacity: int
) -> npt.NDArray:
    """Resizes a 2D matrix by increasing row capacity.

    Args:
        arr: 2D array of shape (r, c)
        new_capacity: New number of rows

    Returns:
        Resized array of shape (new_capacity, d)
    """
    new_arr = np.empty((new_capacity, arr.shape[1]), dtype=arr.dtype)
    new_arr[:arr.shape[0], :] = arr
    return new_arr

