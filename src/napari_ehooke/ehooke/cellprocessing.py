"Module used to encapsulate some functions used in the cells module"

import numpy as np


def rotation_matrices(step):
    """ returns a list of rotation matrixes over 180 deg
    matrixes are transposed to use with 2 column point arrays (x,y),
    multiplying after the array
    TODO: optimize with np vectors
    """

    result = []
    ang = 0

    while ang < 180:
        sa = np.sin(ang / 180.0 * np.pi)
        ca = np.cos(ang / 180.0 * np.pi)
        # note .T, for column points
        result.append(np.matrix([[ca, -sa], [sa, ca]]).T)
        ang = ang + step

    return result


def bounded_value(minval, maxval, currval):
    """ returns the value or the extremes if outside
    """

    if currval < minval:
        return minval

    elif currval > maxval:
        return maxval

    else:
        return currval


def bounded_point(x0, x1, y0, y1, p):
    tx, ty = p
    tx = bounded_value(x0, x1, tx)
    ty = bounded_value(y0, y1, ty)
    return tx, ty


def bound_rectangle(points):
    """ returns a tuple (x0,y0,x1,y1,width) of the bounding rectangle
    points must be a N,2 array of x,y coords
    """

    x0, y0 = np.amin(points, axis=0)
    x1, y1 = np.amax(points, axis=0)
    a = np.min([(x1 - x0), (y1 - y0)])
    return x0, y0, x1, y1, a
