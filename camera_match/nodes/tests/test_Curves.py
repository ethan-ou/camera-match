import numpy as np
from camera_match import (
    CurvesInterpolation,
    CurvesEMOR
)

# From https://github.com/jandren/tone-curve-explorer/blob/master/tonecurves/basecurve.py
CURVE_TEST = np.array(
    [
        [0.000000, 0.000000, 0.000000],
        [0.018124, 0.001943, 0.009145],
        [0.143357, 0.019814, 0.026570],
        [0.330116, 0.080784, 0.131526],
        [0.457952, 0.145700, 0.175858],
        [0.734950, 0.295961, 0.350981],
        [0.904758, 0.651915, 0.614997],
        [1.000000, 1.000000, 1.000000]
    ]
)

CURVE_REFERENCE = np.array(
    [
        [0.000000, 0.000000, 0.000000],
        [0.018124, 0.001943, 0.009145],
        [0.143357, 0.019814, 0.026570],
        [0.330116, 0.080784, 0.131526],
        [0.457952, 0.145700, 0.175858],
        [0.734950, 0.295961, 0.350981],
        [0.904758, 0.651915, 0.614997],
        [1.000000, 1.000000, 1.000000]
    ]
)

class TestCurvesInterpolation:
    def test_solve(self):
        curve = CurvesInterpolation()

        source, target = curve.solve(CURVE_TEST, CURVE_REFERENCE)

        np.testing.assert_allclose(
            source,
            target,
            atol=0.001
        )

class TestCurvesEMOR:
    def test_solve(self):
        curve = CurvesEMOR()

        source, target = curve.solve(CURVE_TEST, CURVE_REFERENCE)

        np.testing.assert_allclose(
            source,
            target,
            atol=0.001
        )
