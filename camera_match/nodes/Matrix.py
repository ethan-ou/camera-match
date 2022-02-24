from abc import ABC, abstractmethod
import numpy as np
from colour import polynomial_expansion_Finlayson2015

class LinearMatrix:
    def __init__(self, matrix=None):
        self.matrix = matrix

        if self.matrix is None:
            self.matrix = self.identity()

    def identity(self):
        return np.identity(3)

    def apply(self, RGB):
        shape = RGB.shape
        RGB = np.reshape(RGB, (-1, 3))
        return np.reshape(np.transpose(np.dot(self.matrix, np.transpose(RGB))), shape)


class RootPolynomialMatrix:
    def __init__(self, matrix=None, degree=2):
        if degree > 4 or degree < 1:
            raise ValueError(
                f"Degree for Root Polynomial Matrix must be between 1 and 4."
            )

        self.matrix = matrix
        self.degree = degree

        if self.matrix is None:
            self.matrix = self.identity(degree)

    def identity(self):
        polynomial_expansion = {
            1: np.identity(3),
            2: np.hstack((np.identity(3), np.zeros((3, 3)))),
            3: np.hstack((np.identity(3), np.zeros((3, 10)))),
            4: np.hstack((np.identity(3), np.zeros((3, 19)))),
        }

        return polynomial_expansion[self.degree]

    def apply(self, RGB):
        shape = RGB.shape
        RGB = np.reshape(RGB, (-1, 3))

        RGB_e = polynomial_expansion_Finlayson2015(RGB, self.degree, root_polynomial_expansion=True)

        return np.reshape(np.transpose(np.dot(self.matrix, np.transpose(RGB_e))), shape)


class TetrahedralMatrix:
    def __init__(self, matrix=None):
        self.matrix = matrix

        if self.matrix is None:
            self.matrix = self.identity()

    def identity(self):
        return np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1]])

    def apply(self, RGB):
        # Return indicies of boolean comparison
        # e.g. a = [0, 1, 3], b = [1, 1, 1]
        # i((a > b)) -> [1, 2]
        def i(arr):
            return arr.nonzero()[0]

        # Find and remove existing elements (emulates if statement)
        # e.g. exists([0, 1, 2], [[1], [0]]) -> [2]
        def exists(arr, prev):
            return np.setdiff1d(arr, np.concatenate(prev))

        # RGB Multiplication of Tetra
        def t_matrix(index, r, mult_r, g, mult_g, b, mult_b):
            return np.multiply.outer(r[index], mult_r) + np.multiply.outer(g[index], mult_g) + np.multiply.outer(b[index], mult_b)

        shape = RGB.shape
        RGB = np.reshape(RGB, (-1, 3))
        r, g, b = RGB.T

        wht = np.array([1, 1, 1])
        red, yel, grn, cyn, blu, mag = self.matrix

        base_1 = r > g
        base_2 = ~(r > g)

        case_1 = i(base_1 & (g > b))
        case_2 = exists(i(base_1 & (r > b)), [case_1])
        case_3 = exists(i(base_1), [case_1, case_2])
        case_4 = i(base_2 & (b > g))
        case_5 = exists(i(base_2 & (b > r)), [case_4])
        case_6 = exists(i(base_2), [case_4, case_5])

        n_RGB = np.zeros(RGB.shape)
        n_RGB[case_1] = t_matrix(case_1, r, red, g, (yel-red), b, (wht-yel))
        n_RGB[case_2] = t_matrix(case_2, r, red, g, (wht-mag), b, (mag-red))
        n_RGB[case_3] = t_matrix(case_3, r, (mag-blu), g, (wht-mag), b, blu)
        n_RGB[case_4] = t_matrix(case_4, r, (wht-cyn), g, (cyn-blu), b, blu)
        n_RGB[case_5] = t_matrix(case_5, r, (wht-cyn), g, grn, b, (cyn-grn))
        n_RGB[case_6] = t_matrix(case_6, r, (yel-grn), g, grn, b, (wht-yel))

        return n_RGB.reshape(shape)
