import numpy as np
from scipy.optimize import least_squares, minimize
from camera_match.metrics import colour_difference

from typing import Any
from numpy.typing import NDArray
from camera_match.metrics import DifferenceMetric

def optimise_matrix(fn, matrix: NDArray[Any], source: NDArray[Any], target: NDArray[Any], metrics: list[DifferenceMetric] = ["MSE", "Weighted Euclidean"]):
    def solve_fn(flat_matrix: NDArray[Any], matrix_shape: tuple[int], fn, source: NDArray[Any], target: NDArray[Any], metric: DifferenceMetric):
        matrix = np.reshape(flat_matrix, matrix_shape)
        return colour_difference(source=fn(source, matrix), target=target, metric=metric)

    new_matrix = matrix

    for metric in metrics:
        new_matrix = least_squares(solve_fn, new_matrix.flatten(), ftol=1e-5,
                                args=(matrix.shape, fn, source, target, metric)).x

    return np.reshape(new_matrix, matrix.shape)

def optimise_pipeline(nodes, source: NDArray[Any], target: NDArray[Any], metrics: list[DifferenceMetric] = ["MSE", "Weighted Euclidean"]):
    def reshape_array(flat_matrix, matrix_shapes):
        matrix = []
        index = 0
        for shape in matrix_shapes:
            size = np.product(shape)
            matrix.append(flat_matrix[index : index + size].reshape(shape))
            index += size

        return matrix

    def get_matrix_from_nodes(nodes):
        matrix = []
        for node in nodes:
            if hasattr(node, 'matrix'):
                matrix.append(node.matrix)

        return matrix

    def apply_matrix_to_nodes(nodes, matrix):
        matrix_index = 0
        for node in nodes:
            if hasattr(node, 'matrix'):
                node.matrix = matrix[matrix_index]
                matrix_index += 1

        return nodes

    def solve_fn(flat_matrix: NDArray[Any], matrix_shapes: list[tuple[int]], nodes, source: NDArray[Any], target: NDArray[Any], metric: DifferenceMetric):
        matrix = reshape_array(flat_matrix, matrix_shapes)
        nodes = apply_matrix_to_nodes(nodes, matrix)

        for node in nodes:
            source = node.apply(source)

        return colour_difference(source=source, target=target, metric=metric)

    matrix = get_matrix_from_nodes(nodes)

    if not matrix:
        return nodes

    shapes = [a.shape for a in matrix]
    flat_matrix = np.concatenate([a.flatten() for a in matrix])

    for metric in metrics:
        flat_matrix = minimize(solve_fn, flat_matrix, tol=1e-3,
                                args=(shapes, nodes, source, target, metric)).x


    matrix = reshape_array(flat_matrix, shapes)
    return apply_matrix_to_nodes(nodes, matrix)
