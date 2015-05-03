import numpy as np
import landmarks
from sklearn.neighbors import NearestNeighbors
from skimage import transform as tf
from scipy.interpolate import splprep, splev


def append_standard_size_and_position(bone):
    centroid = np.mean(bone['points'], axis=0)
    result = bone['points'] - np.tile(centroid, (bone['points'].shape[0], 1))
    result_markers = bone['markers'] - np.tile(centroid, (bone['markers'].shape[0], 1))

    scale_factor = np.sqrt(np.sum(np.power(result, 2)) / result.shape[0])
    result = np.divide(result, scale_factor)
    result_markers = np.divide(result_markers, scale_factor)

    bone['registered'] = result
    bone['registered_markers'] = result_markers

    return bone


def estimate_transform(bones, estimator, init_reference_estimator, iterations, progress_callback=None):
    bones = map(append_standard_size_and_position, bones)
    reference = max(bones, key=lambda o: o['registered'].shape[0])
    reference_estimator = init_reference_estimator(reference)
    progress = {
        'value': 0,
        'max': len(bones) * iterations,
        'callback': progress_callback
    }

    def do_registration(bone, progress):
        if bone is reference:
            bone['error'] = 0
        for j in range(iterations):
            from_points, to_points = reference_estimator(bone)
            result = estimator(to_points, from_points, bone, ['registered', 'registered_markers'])
            bone['registered'] = result[0][0]
            bone['registered_markers'] = result[0][1]
            bone['error'] = result[1]
            progress['value'] += 1
            if progress['callback']:
                progress['callback'](progress['value'], progress['max'])

    for bone in bones:
        do_registration(bone, progress)

def get_error(points, reference, tform):
    transformed = tform(points)
    error = np.sum((points - transformed)**2)
    norm = ((reference - reference.mean(0))**2).sum()
    return error / norm


def affine(reference, points, bone, properties_to_transform):
    tform = tf.estimate_transform('affine', points, reference)
    transformed = list(map(tform, [ bone[p] for p in properties_to_transform  ]))
    error = get_error(points, reference, tform)
    return transformed, error


def similarity(reference, points, bone, properties_to_transform):
    tform = tf.estimate_transform('similarity', points, reference)
    transformed = list(map(tform, [ bone[p] for p in properties_to_transform  ]))
    error = get_error(points, reference, tform)
    return transformed, error


def projective(reference, points, bone, properties_to_transform):
    tform = tf.estimate_transform('projective', points, reference)
    transformed = list(map(tform, [ bone[p] for p in properties_to_transform  ]))
    error = get_error(points, reference, tform)
    return transformed, error

def procrustes(reference, points, bone, properties_to_transform, scaling=True, reflection=False):
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

    n,m = reference.shape
    ny,my = points.shape

    muX = reference.mean(0)
    muY = points.mean(0)

    X0 = reference - muX
    Y0 = points - muY

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
    # tform = {'rotation':T, 'scale':b, 'translation':c}
    Z = list(map(lambda p:  (b * np.dot(bone[p], T) + c), properties_to_transform))

    X = b * np.dot(points, T) + c
    error = np.sum((points - X)**2)
    norm = ((reference - reference.mean(0))**2).sum()

    return Z, (error / norm)

def get_nearest_neighbor_point_estimator(reference):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(reference['registered'])
    def get_neighbors(outline):
        distances, indices = nbrs.kneighbors(outline['registered'])
        indices = np.array(indices).flatten()
        return outline['registered'], reference['registered'][indices]
    return get_neighbors


def get_marker_using_angles_estimator(reference):
    reference_landmarks = landmarks.get_using_angles(reference['registered'])
    def get_landmarks(outline):
        point_landmarks = landmarks.get_using_angles(outline['registered'])
        return point_landmarks, reference_landmarks
    return get_landmarks

def get_marker_using_space_partitioning_estimator(reference):
    reference_landmarks = landmarks.get_using_space_partitioning(reference['registered'])
    def get_landmarks(outline):
        point_landmarks = landmarks.get_using_space_partitioning(outline['registered'])
        return point_landmarks, reference_landmarks
    return get_landmarks

def get_manual_markers(reference):
    reference_landmarks = reference['registered_markers']
    def get_landmarks(outline):
        point_landmarks = outline['registered_markers']
        return point_landmarks, reference_landmarks
    return get_landmarks

def get_manual_markers_5_to_9(reference):
    reference_landmarks = reference['registered_markers'][4:8, :]
    def get_landmarks(outline):
        point_landmarks = outline['registered_markers'][4:8, :]
        return point_landmarks, reference_landmarks
    return get_landmarks

def get_spline_points_estimator(reference, num_evaluations=250):
    def extract_spline(points):
        y = points[:, 0].flatten()
        x = points[:, 1].flatten()
        tck, u = splprep([y, x], s=0)
        coords = splev(np.linspace(0, 1, num_evaluations), tck)
        spline_points = np.zeros((num_evaluations, 2))
        spline_points[:, 0] = coords[0]
        spline_points[:, 1] = coords[1]
        return spline_points
    reference_spline = extract_spline(reference['registered'])
    def get_landmarks(outline):
        point_spline = extract_spline(outline['registered'])
        return point_spline, reference_spline
    return get_landmarks


