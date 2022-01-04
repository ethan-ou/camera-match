import numpy as np
import xalglib
from models import colour_correction_matrix, root_polynomial_colour_correction_matrix, tetrahedral_colour_correction_matrix



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
