import numpy as np
import landmarks
from sklearn.neighbors import NearestNeighbors


def to_standard_size_and_position(outline):
    centroid = np.mean(outline['points'], axis=0)
    result = outline['points'] - np.tile(centroid, (outline['points'].shape[0], 1))
    scale_factor = np.sqrt(np.sum(np.power(result, 2)) / result.shape[0])
    result = np.divide(result, scale_factor)
    return result


def estimate_transform(bones, estimator, init_reference_estimator, iterations):
    standard_bones = map(to_standard_size_and_position, bones)
    reference = max(standard_bones, key=lambda o: o.shape[0])
    reference_estimator = init_reference_estimator(reference)

    def do_registration(points):
        distance = 0
        if points is reference:
            return points, distance
        for j in range(iterations):
            from_points, to_points = reference_estimator(points)
            points, distance = estimator(to_points, from_points, points)
        return points, distance

    registered_bones = map(do_registration, standard_bones)

    for i, bone in enumerate(bones):
        bone['registered'] = registered_bones[i][0]
        bone['error'] = registered_bones[i][1]

def procrustes(X, Y, points_to_transform, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}
    Z = b * np.dot(points_to_transform, T) + c

    return Z, d

def get_nearest_neighbor_point_estimator(reference):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(reference)
    def get_neighbors(outline):
        distances, indices = nbrs.kneighbors(outline)
        indices = np.array(indices).flatten()
        return outline, reference[indices]
    return get_neighbors


def get_marker_using_angles_estimator(reference):
    reference_landmarks = landmarks.get_using_angles(reference)
    def get_landmarks(outline):
        point_landmarks = landmarks.get_using_angles(outline)
        return point_landmarks, reference_landmarks
    return get_landmarks

