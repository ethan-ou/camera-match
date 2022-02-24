import numpy as np
from metrics import find_delta_E

def remove_outlier_patches(source, target, max_delta_E=10):
    delta_E = find_delta_E(source, target)
    allowed = np.where(delta_E < max_delta_E)

    return (source[allowed], target[allowed])

def remove_saturated_patches(source, target, max_value=1):
    source_allowed = np.where(
        ((source > 0) & (source < max_value)).all(axis=1))
    target_allowed = np.where(
        ((target > 0) & (target < max_value)).all(axis=1))

    indicies = np.intersect1d(source_allowed, target_allowed)

    return (np.take(source, indicies, axis=0), np.take(target, indicies, axis=0))
