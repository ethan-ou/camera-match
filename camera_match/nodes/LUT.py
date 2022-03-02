import numpy as np
from .Node import Node
from xalglib import xalglib

class RBF(Node):
    def __init__(self, size=33, LUT=None, library="alglib"):
        self.size = size
        self.LUT = LUT
        self.library = library

    # Implement RBF from scipy
    def solve(self, source, target):
        """
        Takes a 2D array of RGB triplet points as both args.
        Returns an array of interpolated grid values.
        """
        data = np.hstack((source, target))

        model = xalglib.rbfcreate(3, 3)
        xalglib.rbfsetpoints(model, data.tolist())

        xalglib.rbfsetalgohierarchical(model, 5.0, 5, 0.0)
        xalglib.rbfbuildmodel(model)

        grid = np.linspace(0, 1, self.size)
        values = []

        # Haven't figured out how to get triplets from the
        # official gridcalc3v function, so doing it manually, so it's slower.
        for b in grid:
            for g in grid:
                for r in grid:
                    values.append(xalglib.rbfcalc(model, [r, g, b]))

        # TODO: Add LUT from colour

        self.LUT = values

        return values

    def apply(self, RGB):
        if self.LUT is None:
            return RGB



