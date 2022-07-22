from colour import sRGB_to_XYZ, XYZ_to_Lab
from colour import delta_E as _delta_E
import numpy as np
from numpy.typing import NDArray

from typing import Union, Literal, Any

DifferenceMetric = Union[Literal["MSE", "RMSE", "Weighted Euclidean", "Delta E", "Delta E Power"], str]

def colour_difference(source: NDArray[Any], target: NDArray[Any], metric: DifferenceMetric) -> Any:
    difference_metrics = {
        "MSE": MSE,
        "RMSE": RMSE,
        "Weighted Euclidean": mean_weighted_euclidean,
        "Delta E": mean_delta_E,
        "Delta E Power": mean_delta_E_power
    }

    return difference_metrics[metric](source, target)

def MSE(source, target):
    return np.mean((source-target)**2)

def RMSE(source, target):
    return np.sqrt(np.mean((source-target)**2))

def mean_weighted_euclidean(source, target):
    return np.mean(weighted_euclidean(source, target))

def mean_delta_E(source, target):
    return np.mean(delta_E(source, target))

def mean_delta_E_power(source, target):
    return np.mean(delta_E_power(source, target))

# Modified Euclidean Distance
# Taken from https://stackoverflow.com/questions/8863810/python-find-similar-colors-best-way
def weighted_euclidean(source, target):
    # Avoid errors with 1D arrays
    if source.ndim < 2:
        rm = 0.5 * (source + target)
    else:
        rm = 0.5 * (source[:, 0] + target[:, 0])
    drgb = (source - target) ** 2
    t = np.array([2 + rm, 4 + 0 * rm, 3 - rm]).T
    return np.sqrt(np.sum(t * drgb, 1))

# Modified Delta E formula with co-efficients from Huang et al. 2015
# Power functions improving the performance of color-difference formulas
# Useful for detecting small differences in colour.
def delta_E_power(source, target):
    return 1.43 * delta_E(source, target) ** 0.7

def delta_E(source, target):
    return _delta_E(XYZ_to_Lab(sRGB_to_XYZ(source)), XYZ_to_Lab(sRGB_to_XYZ(target)))
