from colour import LUT3D
import numpy as np
from colour.algebra import table_interpolation_tetrahedral
import itertools
from scipy.interpolate import PchipInterpolator
from .Node import Node

from typing import Optional, Any, Tuple
from numpy.typing import NDArray

# xalglib only available in Windows & Linux(?)
try:
    from xalglib import xalglib
except ImportError:
    import warnings
    warnings.warn("RBF library cannot be loaded.", ImportWarning)

class RBF(Node):
    def __init__(self, size: int = 33, init_radius: float = 5.0, num_layers: int = 10, penalty: float = 0.0):
        self.size = size
        self.LUT = None

        self.init_radius = init_radius
        self.num_layers = num_layers
        self.penalty = penalty

    def solve(self, source: NDArray[Any], target: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
        data = np.hstack((source, target))

        model = xalglib.rbfcreate(3, 3)
        xalglib.rbfsetpoints(model, data.tolist())
        xalglib.rbfsetalgohierarchical(model, self.init_radius, self.num_layers, self.penalty)
        xalglib.rbfbuildmodel(model)

        LUT_table = LUT3D.linear_table(self.size)

        for x, y, z in itertools.product(range(self.size), range(self.size), range(self.size)):
            LUT_table[x][y][z] = xalglib.rbfcalc(model, [x / self.size, y / self.size, z / self.size])

        self.LUT = LUT3D(table=LUT_table)
        return (self.apply(source), target)

    def apply(self, RGB: NDArray[Any]) -> NDArray[Any]:
        if self.LUT is None:
            return RGB

        return self.LUT.apply(RGB, interpolator=table_interpolation_tetrahedral)

class CurvesInterpolation(Node):
    def __init__(self, interpolator = PchipInterpolator, **kwargs):
        self.interpolator = interpolator
        self.kwargs = kwargs
        self.curve = None

    def solve(self, source: NDArray[Any], target: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
        s_R, s_G, s_B = np.reshape(source, (-1, 3)).T
        t_R, t_G, t_B = np.reshape(target, (-1, 3)).T

        R_sort = s_R.argsort()
        G_sort = s_G.argsort()
        B_sort = s_B.argsort()

        R_curve = self.interpolator(s_R[R_sort], t_R[R_sort], **self.kwargs)
        G_curve = self.interpolator(s_G[G_sort], t_G[G_sort], **self.kwargs)
        B_curve = self.interpolator(s_B[B_sort], t_B[B_sort], **self.kwargs)

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