import numpy as np
from colour.characterisation import polynomial_expansion_Finlayson2015
import xalglib
from models import colour_correction, root_polynomial_colour_correction, tetrahedral_colour_correction

def solve_fn(matrix, shape, fn, source, target):

    pass


def colour_correction_solver(source, target, degree=1):
    matrix = matrix_colour_correction_Finlayson2015(source, target, degree)


    pass

def root_polynomial_colour_correction_solver(source, target, degree=3):
    matrix = matrix_colour_correction_Finlayson2015(source, target, degree)


    pass

def tetrahedral_colour_correction_solver(source, target):
    matrix = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1]])

    pass

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

MATRIX_METHODS = {
    'CCM': colour_correction_solver,
    'RPCC': root_polynomial_colour_correction_solver,
    'TCC': tetrahedral_colour_correction_solver
}

LUT_METHODS = {
    'RBF': rbf_interpolation
}

def curves_solver(source, target, method='RBF', **kwargs):
    pass

def matrix_solver(source, target, method='CCM', **kwargs):
    pass

def LUT_solver(source, target, method='RBF', **kwargs):
    pass



