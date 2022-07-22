import numpy as np
from scipy.spatial.distance import cdist
from .Node import Node

from typing import Optional, Any
from numpy.typing import NDArray

class RBF(Node):
    def __init__(self, radius: float=1):
        self.radius = radius
        self.weights = None
        self.coordinates = None

    def solve(self, source: NDArray[Any], target: NDArray[Any]):
        self.coordinates = source
        self.weights = self._solve_weights(source, target)

    def __call__(self, RGB: NDArray[Any]) -> NDArray[Any]:
        if self.weights is None or self.coordinates is None:
            return RGB

        shape = RGB.shape
        RGB = np.reshape(RGB, (-1, 3))

        points = self.coordinates.shape[0]

        H = np.zeros((RGB.shape[0], points + 3 + 1))
        H[:, :points] = self.basis(cdist(RGB, self.coordinates), self.radius)
        H[:, points] = 1.0
        H[:, -3:] = RGB
        return np.reshape(np.asarray(np.dot(H, self.weights)), shape)

    def _solve_weights(self, X, Y):
        npts, dim = X.shape
        H = np.zeros((npts + 3 + 1, npts + 3 + 1))
        H[:npts, :npts] = self.basis(cdist(X, X), self.radius)
        H[npts, :npts] = 1.0
        H[:npts, npts] = 1.0
        H[:npts, -3:] = X
        H[-3:, :npts] = X.T

        rhs = np.zeros((npts + 3 + 1, dim))
        rhs[:npts, :] = Y
        return np.linalg.solve(H, rhs)

    @staticmethod
    def basis(X, r):
        arg = X / r
        v = 1 - arg / 9
        return np.where(v > 0, np.exp(1 - arg - 1/v), 0)
