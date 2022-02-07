from colour import sRGB_to_XYZ, XYZ_to_Lab, delta_E
import numpy as np

def colour_distance(source, target, metric):
    if metric == "MSE":
        return MSE(source, target)
    elif metric == "Weighted Euclidean":
        return mean_weighted_euclidean(source, target)
    elif metric == "Delta E":
        return mean_delta_E(source, target)
    elif metric == "Delta E Power":
        return mean_delta_E_power(source, target)
    else:
        raise ValueError("Metric not found")

# Use MSE over RMSE due to performance
# Reduces time by 10-15%
def MSE(source, target):
    return np.mean((source-target)**2)

def mean_weighted_euclidean(source, target):
    return np.mean(find_weighted_euclidean(source, target))

def mean_delta_E(source, target):
    return np.mean(find_delta_E(source, target))

def mean_delta_E_power(source, target):
    return np.mean(find_delta_E_power(source, target))

# Modified Euclidean Distance
# Taken from https://stackoverflow.com/questions/8863810/python-find-similar-colors-best-way
def find_weighted_euclidean(source, target):
    rm = 0.5 * (source[:, 0] + target[:, 0])
    drgb = (source - target) ** 2
    t = np.array([2 + rm, 4 + 0 * rm, 3 - rm]).T
    return np.sqrt(np.sum(t * drgb, 1))

# Modified Delta E formula with co-efficients from Huang et al. 2015
# Power functions improving the performance of color-difference formulas
# Useful for detecting small differences in colour.
def find_delta_E_power(source, target):
    return 1.43 * find_delta_E(source, target) ** 0.7

def find_delta_E(source, target):
    return delta_E(XYZ_to_Lab(sRGB_to_XYZ(source)), XYZ_to_Lab(sRGB_to_XYZ(target)))
