import numpy as np
from camera_match import (
    Lift,
    Gain,
    CurvesInterpolation,
    CurvesEMOR
)

LIFT_TEST = np.array(
    [
        [0.018124, 0.001943, 0.009145]
    ]
)

LIFT_REFERENCE = np.array(
    [
        [0.000000, 0.000000, 0.000000]
    ]
)

GAIN_TEST = np.array(
    [
        [0.904758, 0.651915, 0.614997]
    ]
)

GAIN_REFERENCE = np.array(
    [
        [0.95, 0.95, 0.95]
    ]
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
        [0.006560, 0.012248, 0.012659],
        [0.027310, 0.251013, 0.289973],
        [0.045915, 0.621951, 0.342731],
        [0.206554, 0.771384, 0.510114],
        [0.442337, 0.843079, 0.733820],
        [0.673263, 0.956678, 0.894290],
        [1.000000, 1.000000, 1.000000]
    ]
)

class TestLift:
    def test_solve(self):
        lift = Lift()

        source, target = lift.solve(LIFT_TEST, LIFT_REFERENCE)

        np.testing.assert_allclose(
            source,
            target,
            atol=0.001
        )

class TestGain:
    def test_solve(self):
        gain = Gain()

        source, target = gain.solve(GAIN_TEST, GAIN_REFERENCE)

        np.testing.assert_allclose(
            source,
            target,
            atol=0.001
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
            atol=0.004
        )
