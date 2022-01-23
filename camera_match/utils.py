from colour import sRGB_to_XYZ, XYZ_to_Lab, delta_E

# Modified Delta E formula with co-efficients from Huang et al. 2015
# Power functions improving the performance of color-difference formulas
# Useful for detecting small differences in colour.
def find_delta_E_mod(source, target):
    return 1.43 * find_delta_E(source, target) ** 0.7

def find_delta_E(source, target):
    return delta_E(XYZ_to_Lab(sRGB_to_XYZ(source)), XYZ_to_Lab(sRGB_to_XYZ(target)))
