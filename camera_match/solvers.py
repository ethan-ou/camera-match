import numpy as np
from colour.characterisation import polynomial_expansion_Finlayson2015
from xalglib import xalglib
from scipy.optimize import least_squares
from models import colour_correction, identity_matrix
from metrics import colour_distance

def solve_fn(matrix, matrix_shape, source, target, method, metric):
    matrix = np.reshape(matrix, matrix_shape)
    source_corrected = colour_correction(source, matrix, method)

    return colour_distance(source_corrected, target, metric)

# Add linearise pre-step
def colour_correction_solver(source, target, method):
    matrix = identity_matrix(method)

    # First Stage: MSE
    # Fastest optimisation speed with least accuracy
    solve_RMSE = least_squares(solve_fn, matrix.flatten(), verbose=2, ftol=1e-5, args=(matrix.shape,
        source, target, method, "MSE"))

    # Second Stage: Weighted Euclidean
    # Moderate optimisation speed with good accuracy
    solve_euclidean = least_squares(solve_fn, solve_RMSE.x, verbose=2, args=(matrix.shape,
        source, target, method, "Weighted Euclidean"))

    # Third Stage: Delta E
    # Slowest optimisation speed, used to check results
    solve_Delta_E = least_squares(solve_fn, solve_euclidean.x, verbose=2, args=(matrix.shape,
        source, target, method, "Delta E"))

    return np.reshape(solve_Delta_E.x, matrix.shape)

def rbf_interpolation(source, target, cube_size):
    """
    Takes a 2D array of RGB triplet points as both args.
    Returns an array of interpolated grid values.
    """
    data = np.hstack((source, target))

    model = xalglib.rbfcreate(3, 3)
    xalglib.rbfsetpoints(model, data.tolist())

    xalglib.rbfsetalgohierarchical(model, 5.0, 5, 0.0)
    xalglib.rbfbuildmodel(model)

    grid = np.linspace(0, 1, cube_size)
    values = []

    # Haven't figured out how to get triplets from the
    # official gridcalc3v function, so doing it manually, so it's slower.
    for b in grid:
        for g in grid:
            for r in grid:
                values.append(xalglib.rbfcalc(model, [r, g, b]))

    return values


