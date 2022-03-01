from colour import cctf_decoding, cctf_encoding, RGB_to_RGB, RGB_COLOURSPACES
from camera_match import Node
from numpy.typing import NDArray
from typing import Any, Tuple, Optional

class CST(Node):
    def __init__(self, source_gamma=None, target_gamma=None,
                source_colourspace=None, target_colourspace=None, apply_to_target: bool=False):
        """
        Convert RGB array from one colourspace to another
        similar to Resolve's Color Space Transform.

        Note: Gamma values set to None are equivalent to
        "Linear" gamma in Resolve.
        """

        self.source_gamma = source_gamma
        self.target_gamma = target_gamma
        self.source_colourspace = source_colourspace
        self.target_colourspace = target_colourspace
        self.apply_to_target = apply_to_target

    def solve(self, source: NDArray[Any], target: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
        source = self.apply(source)

        if self.apply_to_target is True:
            target = self.apply(target)

        return (source, target)


    def apply(self, RGB: NDArray[Any]) -> NDArray[Any]:
        """
        Applies Color Space Transform to an RGB array.
        """

        if self.source_gamma is not None:
            RGB = cctf_decoding(RGB, function=self.source_gamma)

        if self.source_colourspace is not None and self.target_colourspace is not None:
            RGB = RGB_to_RGB(RGB, RGB_COLOURSPACES[self.source_colourspace],
                            RGB_COLOURSPACES[self.target_colourspace])

        if self.target_gamma is not None:
            RGB = cctf_encoding(RGB, function=self.target_gamma)

        return RGB
