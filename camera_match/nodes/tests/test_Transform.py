import numpy as np
from camera_match import CST

class TestCST:
    def test_gamma_encoding(self):
        RGB_A = np.array([0.092809000000000, 0.391006832034084, 0.570631558120417])
        RGB_B = np.array([0.0, 0.18, 1.0])

        node = CST(source_gamma="ALEXA Log C")

        np.testing.assert_allclose(node.apply(RGB_A), RGB_B)

    def test_gamma_decoding(self):
        RGB_A = np.array([0.0, 0.18, 1.0])
        RGB_B = np.array([0.092809000000000, 0.391006832034084, 0.570631558120417])

        node = CST(target_gamma="ALEXA Log C")

        np.testing.assert_allclose(node.apply(RGB_A), RGB_B)

    def test_gamma_transform(self):
        RGB_A = np.array([0.092809000000000, 0.391006832034084, 0.570631558120417])
        RGB_B = np.array([0.092864125122190, 0.41055718475073, 0.596027343690123])

        node = CST(source_gamma="ALEXA Log C", target_gamma="S-Log3")

        np.testing.assert_allclose(node.apply(RGB_A), RGB_B)

    def test_colourspace_transform(self):
        RGB_A = np.array([0.21931722, 0.06950287, 0.04694832])
        RGB_B = np.array([0.45595289, 0.03040780, 0.04087313])

        node = CST(source_colourspace="ACES2065-1", target_colourspace="sRGB")

        np.testing.assert_allclose(node.apply(RGB_A), RGB_B)

    def test_gamma_colourspace_transform(self):
        RGB_A = np.array([0.21931722, 0.06950287, 0.04694832])
        RGB_B = np.array([0.32882878, 0.00622853, 0.02822571])

        node = CST(source_gamma="ALEXA Log C", target_gamma="S-Log3",
                source_colourspace="ACES2065-1", target_colourspace="sRGB")

        np.testing.assert_allclose(node.apply(RGB_A), RGB_B, rtol=1e-06)