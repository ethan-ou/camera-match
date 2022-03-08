import numpy as np
from scipy.interpolate import PchipInterpolator
from .Node import Node
from .datasets import (
    EMOR_MAX_FUNCTIONS,
    EMOR_LENGTH,
    EMOR_X,
    EMOR_F0,
    EMOR_H
)

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
    def __init__(self, num_params: int = 5, interpolator = PchipInterpolator, **kwargs):
        if num_params > EMOR_MAX_FUNCTIONS or num_params < 1:
            raise ValueError(
                f"num_params for EMoR must be between 1 and 11."
            )

        self.num_params = num_params
        self.interpolator = interpolator
        self.kwargs = kwargs
        self.curve = None

    # Adapted from: https://github.com/dailerob/BSDF-measurement-device/blob/9948f051f59ce456e2f9c66e5c2a2050832ba1c9/Find_Camera_Response.py
    def solve(self, source: NDArray[Any], target: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
        s_R, s_G, s_B = np.reshape(source, (-1, 3)).T
        t_R, t_G, t_B = np.reshape(target, (-1, 3)).T

        def fit_channel(source, target, num_params, interpolator, kwargs):
            h = EMOR_H[0:num_params, :]

            indicies = np.round(source * (EMOR_LENGTH - 1)).astype(np.int32)
            deviation = target - EMOR_F0[indicies]
            model = h[:, indicies]

            model_fit = np.dot(np.dot(np.linalg.pinv(np.dot(model, np.transpose(model))), model), deviation)
            curve_fit = np.dot(np.transpose(model_fit), h) + EMOR_F0

            return interpolator(EMOR_X, curve_fit, **kwargs)

        R_curve = fit_channel(s_R, t_R, self.num_params, self.interpolator, self.kwargs)
        G_curve = fit_channel(s_G, t_G, self.num_params, self.interpolator, self.kwargs)
        B_curve = fit_channel(s_B, t_B, self.num_params, self.interpolator, self.kwargs)

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
