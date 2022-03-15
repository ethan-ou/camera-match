"""Top-level package for camera_match."""

__author__ = """Ethan Ou"""
__email__ = 'ethantim@gmail.com'
__version__ = '0.1.0'

from .nodes import (
    Lift,
    Gain,
    CurvesInterpolation,
    CurvesEMOR,
    RBF,
    LinearMatrix,
    RootPolynomialMatrix,
    TetrahedralMatrix,
    Node,
    CST,
)

from .pipeline import (
    Pipeline
)
