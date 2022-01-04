def find_delta_E(source, target):
    return delta_E(XYZ_to_Lab(sRGB_to_XYZ(source)),
                   XYZ_to_Lab(sRGB_to_XYZ(target)))
