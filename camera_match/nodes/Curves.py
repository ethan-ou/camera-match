import numpy as np
from scipy.interpolate import PchipInterpolator
from .Node import Node
from .datasets import (
    EMOR_MAX_FACTORS,
    EMOR_LENGTH,
    EMOR_X,
    EMOR_F0,
    EMOR_H
)
from camera_match.optimise import optimise_matrix

from typing import Optional, Any, Tuple
from numpy.typing import NDArray


class CurvesInterpolation(Node):
    def __init__(self, interpolator = PchipInterpolator, **kwargs):
        self.interpolator = interpolator
        self.kwargs = kwargs
        self.curve = None

    def solve(self, source: NDArray[Any], target: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
        s_R, s_G, s_B = np.reshape(source, (-1, 3)).T
        t_R, t_G, t_B = np.reshape(target, (-1, 3)).T

        def fit_channel(source, target, interpolator, kwargs):
            sort = source.argsort()
            return interpolator(source[sort], target[sort], **kwargs)

        R_curve = fit_channel(s_R, t_R, self.interpolator, self.kwargs)
        G_curve = fit_channel(s_G, t_G, self.interpolator, self.kwargs)
        B_curve = fit_channel(s_B, t_B, self.interpolator, self.kwargs)

        def apply_curve(RGB: NDArray[Any]) -> NDArray[Any]:
            shape = RGB.shape
            R, G, B = np.reshape(RGB, (-1, 3)).T

            return np.reshape(np.array([R_curve(R), G_curve(G), B_curve(B)]).T, shape)

        self.curve = apply_curve

        return (self.apply(source), target)

    def apply(self, RGB: NDArray[Any]) -> NDArray[Any]:
        if self.curve is None:
            return RGB

        return self.curve(RGB)

class CurvesEMOR(Node):
    def __init__(self, matrix = None, degree: int = 6, interpolator = PchipInterpolator):
        if degree > EMOR_MAX_FACTORS or degree < 1:
            raise ValueError(
                f"degree for EMoR must be between 1 and 11."
            )

        self.degree = degree
        self.interpolator = interpolator
        self.matrix = matrix

        if self.matrix is None:
            self.matrix = self.identity(self.degree)

    # Adapted from: https://github.com/dailerob/BSDF-measurement-device/blob/9948f051f59ce456e2f9c66e5c2a2050832ba1c9/Find_Camera_Response.py
    def solve(self, source: NDArray[Any], target: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
        self.matrix = self._solve_pinv(source, target, self.degree)

        apply_matrix = lambda RGB, matrix: self.apply_matrix(RGB, matrix, self.degree)
        self.matrix = optimise_matrix(apply_matrix, self.matrix, source, target)

        return (self.apply(source), target)

    def apply(self, RGB: NDArray[Any]) -> NDArray[Any]:
        if self.matrix is None:
            return RGB

        return self.apply_matrix(RGB, self.matrix, self.degree)

    @staticmethod
    def identity(degree: int) -> NDArray[Any]:
        return np.zeros((degree, 3))

    @staticmethod
    def apply_matrix(RGB: NDArray[Any], matrix: NDArray[Any], degree: int, interpolator = PchipInterpolator) -> NDArray[Any]:
        m_R, m_G, m_B = np.reshape(matrix, (-1, 3)).T

        h = EMOR_H[0:degree, :]

        curve = lambda x: interpolator(EMOR_X, np.dot(np.transpose(x), h) + EMOR_F0)

        R_curve = curve(m_R)
        G_curve = curve(m_G)
        B_curve = curve(m_B)

        shape = RGB.shape
        R, G, B = np.reshape(RGB, (-1, 3)).T
        return np.reshape(np.array([R_curve(R), G_curve(G), B_curve(B)]).T, shape)

    @staticmethod
    def _solve_pinv(source, target, degree):
        s_R, s_G, s_B = np.reshape(source, (-1, 3)).T
        t_R, t_G, t_B = np.reshape(target, (-1, 3)).T

        def fit_channel(source, target, degree):
            h = EMOR_H[0:degree, :]

            indicies = np.round(source * (EMOR_LENGTH - 1)).astype(np.int32)
            deviation = target - EMOR_F0[indicies]
            model = h[:, indicies]

            return np.dot(np.dot(np.linalg.pinv(np.dot(model, np.transpose(model))), model), deviation)

        R_curve = fit_channel(s_R, t_R, degree)
        G_curve = fit_channel(s_G, t_G, degree)
        B_curve = fit_channel(s_B, t_B, degree)

        return np.array([R_curve, G_curve, B_curve]).T
