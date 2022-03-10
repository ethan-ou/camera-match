import numpy as np
from scipy.optimize import least_squares
from camera_match.metrics import colour_difference

from typing import Optional, Any, Tuple, Union
from numpy.typing import NDArray
from camera_match.metrics import DifferenceMetric

def solve_fn(x: NDArray[Any], x_shape: tuple[int], fn, source: NDArray[Any], target: NDArray[Any], metric: DifferenceMetric):
    x = np.reshape(x, x_shape)
    return colour_difference(source=fn(source, x), target=target, metric=metric)

def optimise_matrix(fn, x: NDArray[Any], source: NDArray[Any], target: NDArray[Any], metrics: list[DifferenceMetric] = ["MSE", "Weighted Euclidean"]):
    new_x = x

    for metric in metrics:
        new_x = least_squares(solve_fn, new_x.flatten(), ftol=1e-5,
                                args=(x.shape, fn, source, target, metric)).x

    return np.reshape(new_x, x.shape)
