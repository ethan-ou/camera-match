import numpy as np
from scipy.optimize import minimize
from camera_match.metrics import colour_difference

from typing import Any
from numpy.typing import NDArray
from camera_match.metrics import DifferenceMetric

# def _reshape_matrix(matrix_flat, matrix_shapes):
#     matrix = []
#     index = 0
#     for shape in matrix_shapes:
#         size = np.product(shape)
#         matrix.append(matrix_flat[index : index + size].reshape(shape))
#         index += size

#     return matrix

# def _get_matrix_from_nodes(nodes):
#     matrix = []
#     for node in nodes:
#         if hasattr(node, 'matrix'):
#             matrix.append(node.matrix)

#     return matrix

# def _apply_matrix_to_nodes(nodes, matrix_list):
#     matrix_index = 0
#     for node in nodes:
#         if hasattr(node, 'matrix'):
#             node.matrix = matrix_list[matrix_index]
#             matrix_index += 1

#     return nodes

class NodeOptimiser:
    def __init__(self, fn, matrix, fn_args=None, metrics=None):
        self.fn = fn
        self.matrix = matrix
        self.fn_args = fn_args

        if isinstance(metrics, str):
            metrics = [metrics]

        if metrics is None:
            metrics = ["MSE", "Weighted Euclidean"]

        self.metrics = metrics

    def solve(self, source, target):
        matrix = self.matrix
        for metric in self.metrics:
            matrix = minimize(self._solve_fn, matrix.flatten(), method='CG', options={'maxiter':32},
                                   args=(self.matrix.shape, self.fn, self.fn_args, source, target, metric)
                                   ).x
        
        return np.reshape(matrix, self.matrix.shape)
    
    @staticmethod
    def _solve_fn(matrix_flat, matrix_shape, fn, fn_args, source, target, metric):
        matrix = np.reshape(matrix_flat, matrix_shape)
        
        if isinstance(fn_args, tuple):
            source = fn(source, matrix, *fn_args)
        elif fn_args:
            source = fn(source, matrix, fn_args)
        else:
            source = fn(source, matrix)

        return colour_difference(source=source, target=target, metric=metric)



# class PipelineOptimiser:
#     def __init__(self, tol=1e-3, metrics=None):
#         self.tol = tol

#         if metrics is None:
#             metrics = ["MSE", "Weighted Euclidean"]

#         self.metrics = metrics
    
#     def solve(self, source, target, nodes):
#         matrix = _get_matrix_from_nodes(nodes)

#         if not matrix:
#             return nodes

#         shapes = [a.shape for a in matrix]
#         flat_matrix = np.concatenate([a.flatten() for a in matrix])

#         for metric in self.metrics:
#             flat_matrix = minimize(self._solve_fn, flat_matrix, tol=self.tol,
#                                     args=(shapes, source, target, nodes, metric)).x

#         matrix = _reshape_matrix(flat_matrix, shapes)
#         return _apply_matrix_to_nodes(nodes, matrix)

#     @staticmethod
#     def _solve_fn(matrix_flat, matrix_shapes, source, target, nodes, metric):
#         matrix = _reshape_matrix(matrix_flat, matrix_shapes)
#         nodes = _apply_matrix_to_nodes(nodes, matrix)

#         for node in nodes:
#             source = node.apply(source)

#         return colour_difference(source=source, target=target, metric=metric)