import numpy as np
from itertools import product
from typing import List, Union, Callable, Optional

def find_nearest_idx(array, value):
    # I hope to the great kookaburra above that this doesn't already exist in numpy...

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx



def dynamic_nested_loops(
    arrays: List[np.ndarray],
    axis_names: Optional[List[str]] = None,
    include_indices: bool = False,
    as_dict: bool = False,
):
    """
    General-purpose n-dimensional nested loop generator.

    Parameters:
    - arrays: List of numpy arrays (length M) to loop over.
    - axis_names: Optional list of names for each axis.
    - include_indices: If True, yield (indices, values)
    - as_dict: If True, values are returned as {axis_name: value}

    Yields:
    - Tuple of values by default.
    - Tuple (index_tuple, value_tuple) if include_indices is True.
    - Dict if as_dict is True and axis_names are provided.

    Example:
        for idx, val in dynamic_nested_loops([np.array([1,2]), np.array([3,4])], include_indices=True):
    ...     print(idx, val)
    """
    num_axes = len(arrays)

    if axis_names is not None and len(axis_names) != num_axes:
        raise ValueError("Length of axis_names must match number of arrays")

    for index_tuple in product(*(range(len(arr)) for arr in arrays)):
        value_tuple = tuple(arrays[dim][i] for dim, i in enumerate(index_tuple))

        if as_dict:
            if axis_names is None:
                raise ValueError("axis_names must be provided when as_dict=True")
            value = {name: val for name, val in zip(axis_names, value_tuple)}
        else:
            value = value_tuple

        if include_indices:
            yield index_tuple, value
        else:
            yield value


