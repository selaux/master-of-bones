from functools import partial
import numpy as np
import landmarks
from sklearn.neighbors import NearestNeighbors
from skimage import transform as tf
from scipy.interpolate import splprep, splev


def append_standard_size_and_position(source_property, from_properties, to_properties, bone, independent_scaling=False):
    """
    Applies a transformations to all from_properties in bone and puts the results in the to_properties of bone
    The transformation is defined by moving the points of source_property in bone into the center of the coordinate
    system and scaling the result to have a mean square error of 1
    When setting independent_scaling to True, the x and y dimensions are scaled independent of each other, meaning
    they each have a mean square error of 1
    :param source_property: The property that defines the transformation
    :param from_properties: Array of properties that are transfromed
    :param to_properties: Array of properties that the results are stored in (has to have the same length as from_properties)
    :param bone: The bone that is considered
    :param independent_scaling: Defines wether axes are scaled independent of each other
    :return:
    """
    centroid = np.mean(bone[source_property], axis=0)
    moved_src = bone[source_property] - np.tile(centroid, (bone[source_property].shape[0], 1))
    if independent_scaling:
        scale_factor_0 = np.sqrt(np.sum(np.power(moved_src[:, 0], 2)) / moved_src.shape[0])
        scale_factor_1 = np.sqrt(np.sum(np.power(moved_src[:, 1], 2)) / moved_src.shape[0])
    else:
        scale_factor = np.sqrt(np.sum(np.power(moved_src, 2)) / moved_src.shape[0])
        scale_factor_0 = scale_factor
        scale_factor_1 = scale_factor

    for p_from, p_to in zip(from_properties, to_properties):
        if len(bone[p_from]) > 0:
            result = bone[p_from] - np.tile(centroid, (bone[p_from].shape[0], 1))
            result[:, 0] = np.divide(result[:, 0], scale_factor_0)
            result[:, 1] = np.divide(result[:, 1], scale_factor_1)
            bone[p_to] = result
        else:
            bone[p_to] = bone[p_from]

    return bone


def estimate_transform(bones, estimator, init_reference_estimator, iterations, progress_callback=None, independent_scaling=False, continue_registration=False):
    """
    Estimates a transform that fits all {bones} onto each other based on the transformation estimator {estimator}
    (defines a transformation type that is estimated), a point-to-point  reference estimator defined by {init_reference_estimator}
    (defines which points are used to estimate the transformation)
    :param bones: bones that are registered
    :param estimator: defines the type of the transformation that is estimated
    :param init_reference_estimator: defines the points used to estimate the transformation
    :param iterations: defines the number of iterations (1 iteration = estimate_references -> estimate_transform)
    :param progress_callback: progress callback that is called after each iteration
    :param independent_scaling: Wether the initial normalization of the bone should use axis independent scaling (see append_standard_size_and_position)
    :param continue_registration: Wether an already begun registration should be contued ('registered' is then used instead of 'points' in the bone dict)
    :return: Nothing the registerd outlines are stored in bone['registered']
    """
    if not continue_registration:
        bones = map(partial(
            append_standard_size_and_position,
            'points', ['points', 'markers'],
            ['registered', 'registered_markers'],
            independent_scaling=independent_scaling
        ), bones)
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
            #TODO: Fix this (registered_markers does not exist for all bones, because its manually defined)
            #result = estimator(to_points, from_points, bone, ['registered', 'registered_markers'])
            result = estimator(to_points, from_points, bone, ['registered'])
            bone['registered'] = result[0][0]
            #bone['registered_markers'] = result[0][1]
            bone['error'] = result[1]
            progress['value'] += 1
            if progress['callback']:
                progress['callback'](progress['value'], progress['max'])

    for bone in bones:
        do_registration(bone, progress)

def get_error(points, reference, tform):
    """
    Returns the mean square error for the transform tform and the points points with respect to the reference
    """
    transformed = tform(points)
    error = np.sum((points - transformed)**2)
    norm = ((reference - reference.mean(0))**2).sum()
    return error / norm


"""
The following are transformation types that are passed into estimate_transform as the estimator. They all estimate
a transformation for a single point and applies it to properties_to_transform the applied transformations are then
returned, additional to the mean_square_error if the registration
:param reference: A reference that all bones should be transformed onto (currently it's the bone with the most points)
:param points: The points that are transformed onto the reference
:param bone: the bone that is considered
:param properties_to_transform: the properties of the bone the transformation should be applied to
:return:
"""
def affine(reference, points, bone, properties_to_transform):
    """
    Estimates a affine transform
    """
    tform = tf.estimate_transform('affine', points, reference)
    transformed = list(map(tform, [ bone[p] for p in properties_to_transform ]))
    error = get_error(points, reference, tform)
    return transformed, error


def similarity(reference, points, bone, properties_to_transform):
    """
    Estimates a similarity transform
    """
    tform = tf.estimate_transform('similarity', points, reference)
    transformed = list(map(tform, [ bone[p] for p in properties_to_transform  ]))
    error = get_error(points, reference, tform)
    return transformed, error


def projective(reference, points, bone, properties_to_transform):
    """
    Estimates a projective transform
    """
    tform = tf.estimate_transform('projective', points, reference)
    transformed = list(map(tform, [ bone[p] for p in properties_to_transform  ]))
    error = get_error(points, reference, tform)
    return transformed, error

def procrustes(reference, points, bone, properties_to_transform, scaling=True, reflection=False):
    """
    Estimates a procrustes transform
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


"""
The following functions are reference estimators, they are initialized with the reference that all bones are
registered onto and return a function that returns two arrays of points that define the registration (one reference
array and one array of points that are registered to the reference).
"""
def get_nearest_neighbor_point_estimator(reference):
    """
    Returns the nearest neighbors as points to register
    """
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(reference['registered'])
    def get_neighbors(outline):
        distances, indices = nbrs.kneighbors(outline['registered'])
        indices = np.array(indices).flatten()
        return outline['registered'], reference['registered'][indices]
    return get_neighbors


def get_marker_using_angles_estimator(reference):
    """
    Returns the landmarks extracted by angle (see landmarks.py) to register
    """
    reference_landmarks = landmarks.get_using_angles(reference['registered'])
    def get_landmarks(outline):
        point_landmarks = landmarks.get_using_angles(outline['registered'])
        return point_landmarks, reference_landmarks
    return get_landmarks

def get_marker_using_space_partitioning_estimator(reference):
    """
    Returns the landmarks extracted by space partitioning (see landmarks.py) to register
    """
    reference_landmarks = landmarks.get_using_space_partitioning(reference['registered'])
    def get_landmarks(outline):
        point_landmarks = landmarks.get_using_space_partitioning(outline['registered'])
        return point_landmarks, reference_landmarks
    return get_landmarks

def get_manual_markers(reference):
    """
    Returns the manually defined landmarks to register
    """
    reference_landmarks = reference['registered_markers']
    def get_landmarks(outline):
        point_landmarks = outline['registered_markers']
        return point_landmarks, reference_landmarks
    return get_landmarks

def get_manual_markers_5_to_9(reference):
    """
    Returns thhe manually defined landmarks 5-9 out of 11 to register
    """
    reference_landmarks = reference['registered_markers'][4:8, :]
    def get_landmarks(outline):
        point_landmarks = outline['registered_markers'][4:8, :]
        return point_landmarks, reference_landmarks
    return get_landmarks

def get_spline_points_estimator(reference, num_evaluations=250):
    """
    Evaluates 250 spline points and use them to register
    """
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


