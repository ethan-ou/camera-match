import numpy as np

from colour.characterisation import polynomial_expansion_Finlayson2015

def colour_correction_matrix(RGB, matrix, degree=1):
    shape = RGB.shape

    RGB = np.reshape(RGB, (-1, 3))

    RGB_e = polynomial_expansion_Finlayson2015(RGB, degree,
                                               root_polynomial_expansion=True)

    return np.reshape(np.transpose(np.dot(matrix, np.transpose(RGB_e))), shape)

def root_polynomial_colour_correction_matrix(RGB, matrix, degree=3):
    shape = RGB.shape

    RGB = np.reshape(RGB, (-1, 3))

    RGB_e = polynomial_expansion_Finlayson2015(RGB, degree,
                                               root_polynomial_expansion=True)

    return np.reshape(np.transpose(np.dot(matrix, np.transpose(RGB_e))), shape)

def tetrahedral_colour_correction_matrix(RGB, matrix):
    # Return indicies of boolean comparison
    # e.g. a = [0, 1, 3], b = [1, 1, 1]
    # i((a > b)) -> [1, 2]
    def i(arr):
        return arr.nonzero()[0]

    # Find and remove existing elements (emulates if statement)
    # e.g. exists([0, 1, 2], [[1], [0]]) -> [2]
    def exists(arr, prev):
        return np.setdiff1d(arr, np.concatenate(prev))

    # RGB Multiplication of Tetra
    def t_matrix(index, r, mult_r, g, mult_g, b, mult_b):
        return np.multiply.outer(r[index], mult_r) + np.multiply.outer(g[index], mult_g) + np.multiply.outer(b[index], mult_b)

    shape = RGB.shape
    RGB = np.reshape(RGB, (-1, 3))
    r, g, b = RGB.T

    wht = np.array([1, 1, 1])
    red, yel, grn, cyn, blu, mag = matrix

    base_1 = r > g
    base_2 = ~(r > g)

    case_1 = i(base_1 & (g > b))
    case_2 = exists(i(base_1 & (r > b)), [case_1])
    case_3 = exists(i(base_1), [case_1, case_2])
    case_4 = i(base_2 & (b > g))
    case_5 = exists(i(base_2 & (b > r)), [case_4])
    case_6 = exists(i(base_2), [case_4, case_5])

    n_RGB = np.zeros(RGB.shape)
    n_RGB[case_1] = t_matrix(case_1, r, red, g, (yel-red), b, (wht-yel))
    n_RGB[case_2] = t_matrix(case_2, r, red, g, (wht-mag), b, (mag-red))
    n_RGB[case_3] = t_matrix(case_3, r, (mag-blu), g, (wht-mag), b, blu)
    n_RGB[case_4] = t_matrix(case_4, r, (wht-cyn), g, (cyn-blu), b, blu)
    n_RGB[case_5] = t_matrix(case_5, r, (wht-cyn), g, grn, b, (cyn-grn))
    n_RGB[case_6] = t_matrix(case_6, r, (yel-grn), g, grn, b, (wht-yel))

    return n_RGB.reshape(shape)

