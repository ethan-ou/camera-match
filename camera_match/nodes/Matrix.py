import numpy as np
from colour.characterisation import polynomial_expansion_Finlayson2015, matrix_colour_correction_Finlayson2015
from .Node import Node

from typing import Optional, Any
from numpy.typing import NDArray
from camera_match.optimise import NodeOptimiser


class LinearMatrix(Node):
    def __init__(self, matrix: Optional[NDArray[Any]] = None):
        self.matrix = matrix

        if self.matrix is None:
            self.matrix = self.identity()

    def solve(self, source: NDArray[Any], target: NDArray[Any]):
        # Setting Matrix with Moore-Penrose solution for speed
        self.matrix = matrix_colour_correction_Finlayson2015(source, target, degree=1)
        
        optimiser = NodeOptimiser(self.apply_matrix, self.matrix)
        self.matrix = optimiser.solve(source, target)

    def apply(self, RGB: NDArray[Any]) -> NDArray[Any]:
        return self.apply_matrix(RGB, self.matrix)

    @staticmethod
    def identity() -> NDArray[Any]:
        return np.identity(3)

    @staticmethod
    def apply_matrix(RGB: NDArray[Any], matrix: NDArray[Any]) -> NDArray[Any]:
        shape = RGB.shape
        RGB = np.reshape(RGB, (-1, 3))
        return np.reshape(np.transpose(np.dot(matrix, np.transpose(RGB))), shape)


class RootPolynomialMatrix(Node):
    def __init__(self, matrix: Optional[NDArray[Any]] = None, degree: int=2):
        if degree > 4 or degree < 1:
            raise ValueError(
                f"Degree for Root Polynomial Matrix must be between 1 and 4."
            )

        self.matrix = matrix
        self.degree = degree

        if self.matrix is None:
            self.matrix = self.identity(self.degree)

    def solve(self, source: NDArray[Any], target: NDArray[Any]):
        # Setting Matrix with Moore-Penrose solution for speed
        self.matrix = matrix_colour_correction_Finlayson2015(source, target, degree=self.degree, root_polynomial_expansion=True)
        
        optimiser = NodeOptimiser(self.apply_matrix, self.matrix, fn_args=(self.degree))
        self.matrix = optimiser.solve(source, target)

    def apply(self, RGB: NDArray[Any]) -> NDArray[Any]:
        return self.apply_matrix(RGB, self.matrix, self.degree)

    @staticmethod
    def identity(degree: int) -> NDArray[Any]:
        polynomial_expansion = {
            1: np.identity(3),
            2: np.hstack((np.identity(3), np.zeros((3, 3)))),
            3: np.hstack((np.identity(3), np.zeros((3, 10)))),
            4: np.hstack((np.identity(3), np.zeros((3, 19)))),
        }

        return polynomial_expansion[degree]

    @staticmethod
    def apply_matrix(RGB: NDArray[Any], matrix: NDArray[Any], degree: int) -> NDArray[Any]:
        shape = RGB.shape
        RGB = np.reshape(RGB, (-1, 3))

        RGB_e = polynomial_expansion_Finlayson2015(RGB, degree, root_polynomial_expansion=True)

        return np.reshape(np.transpose(np.dot(matrix, np.transpose(RGB_e))), shape)


class TetrahedralMatrix(Node):
    def __init__(self, matrix: Optional[NDArray[Any]] = None):
        self.matrix = matrix

        if self.matrix is None:
            self.matrix = self.identity()

    def solve(self, source: NDArray[Any], target: NDArray[Any]):
        optimiser = NodeOptimiser(self.apply_matrix, self.matrix)
        self.matrix = optimiser.solve(source, target)

    def apply(self, RGB: NDArray[Any]) -> NDArray[Any]:
        return self.apply_matrix(RGB, self.matrix)

    @staticmethod
    def identity() -> NDArray[Any]:
        return np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1]])

    @staticmethod
    def apply_matrix(RGB: NDArray[Any], matrix: NDArray[Any]) -> NDArray[Any]:
        def tetra_case(index, r, matrix_r, g, matrix_g, b, matrix_b, constant):
            R = np.multiply.outer(r[index], matrix_r)
            G = np.multiply.outer(g[index], matrix_g)
            B = np.multiply.outer(b[index], matrix_b)
            return R + G + B + constant

        shape = RGB.shape
        r, g, b = np.transpose(np.reshape(RGB, (-1, 3)))
        
        blk = np.array([0, 0, 0])
        wht = np.array([1, 1, 1])
        red, yel, grn, cyn, blu, mag = matrix

        case_1 = np.logical_and(r > g, g > b)
        case_2 = np.logical_and(r > g, np.logical_and(g <= b, r > b))
        case_3 = np.logical_and(r > g, np.logical_and(g <= b, r <= b))
        case_4 = np.logical_and(r <= g, b > g)
        case_5 = np.logical_and(r <= g, np.logical_and(b <= g, b > r))
        case_6 = np.logical_and(r <= g, np.logical_and(b <= g, b <= r))

        n_RGB = np.zeros(RGB.shape)
        n_RGB[case_1] = tetra_case(case_1, r, (red-blk), g, (yel-red), b, (wht-yel), blk)
        n_RGB[case_2] = tetra_case(case_2, r, (red-blk), g, (wht-mag), b, (mag-red), blk)
        n_RGB[case_3] = tetra_case(case_3, r, (mag-blu), g, (wht-mag), b, (blu-blk), blk)
        n_RGB[case_4] = tetra_case(case_4, r, (wht-cyn), g, (cyn-blu), b, (blu-blk), blk)
        n_RGB[case_5] = tetra_case(case_5, r, (wht-cyn), g, (grn-blk), b, (cyn-grn), blk)
        n_RGB[case_6] = tetra_case(case_6, r, (yel-grn), g, (grn-blk), b, (wht-yel), blk)

        return n_RGB.reshape(shape)
