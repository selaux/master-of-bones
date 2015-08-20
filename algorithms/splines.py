import numpy as np
from scipy.interpolate import splprep, splev


def evaluate_spline(ts, tck):
    """
    Evaluate a spline at the parameters ts
    :param ts: spline params
    :param tck: The spline representation of the outline obtained using scipy.interprolate.splrep
    :return:
    """
    coords = splev(ts, tck)

    spline_points = np.zeros((len(coords[0]), 2))
    spline_points[:, 0] = coords[0]
    spline_points[:, 1] = coords[1]

    return spline_points


def get_spline_params(points):
    """
    Extract the spline representation from points
    :param points: Points to represent with a spline (must be in order)
    :return:
    """
    distances = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
    distances = np.cumsum(distances / np.sum(distances))

    params = np.concatenate((
        (distances-1)[-10:],
        distances,
        (distances+1)[:10]
    ))

    y = np.concatenate((points[:, 0][-10:], points[:, 0], points[:, 0][:10]))
    x = np.concatenate((points[:, 1][-10:], points[:, 1], points[:, 1][:10]))

    tck, u = splprep([y, x], s=0.025, u=params)

    return tck, u
