from abc import ABCMeta, abstractmethod
from math import radians, floor
import numpy as np
from scipy.interpolate import splev, spalde
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from splines import get_spline_params, evaluate_spline
import helpers.geometry as gh
import helpers.classes as ch
import helpers.features as fh


class WindowExtractor:
    """
    Base class to extract the section that is evaluated around a certain evaluation point with size {window_size}
    and the number of points inside the window being {number_of_evaluations}
    For 2D the evaluation point could be an angle
    """
    __metaclass__ = ABCMeta

    def __init__(self, evaluation_point, window_size, number_of_evaluations):
        self.evaluation_point = evaluation_point
        self.window_size = window_size
        self.number_of_evaluations = number_of_evaluations

    @abstractmethod
    def extract_window_space(self, bone):
        raise NotImplementedError

class WindowExtractor2DByWindowLength(WindowExtractor):
    """
    Get an array of spline parameters that represent a section of the outline around the angle {evaluation_point}
    that has the length {window_size}.
    angle: The angle around which this window is positioned
    window_size: The length of the section that is this window
    number_of_evaluations: The number of spline parameters that should be returned by this function
    """

    def extract_window_space(self, bone):
        tck = bone['spline_params']
        EXTENSION_STEP = 0.0025
        source = np.array([0.0, 0.0])
        ray = np.array(gh.pol2cart(2, radians(self.evaluation_point)))
        total_space = np.linspace(0, 1, self.number_of_evaluations)

        coords = splev(total_space, tck)
        num_total_spline_points = len(coords[0])
        total_spline = np.zeros((num_total_spline_points, 2))
        total_spline[:, 0] = coords[0]
        total_spline[:, 1] = coords[1]

        param_for_spline = None
        for i in range(0, num_total_spline_points):
            j = i+1 if i+1 < num_total_spline_points else 0
            intersect = gh.seg_intersect(source, ray, total_spline[i, :], total_spline[j, :])
            if intersect is not None:
                dist_from_i = np.linalg.norm(intersect - total_spline[i, :])
                norm_dist_from_i = dist_from_i / np.linalg.norm(total_spline[j, :] - total_spline[i, :])

                if total_space[i] == 0:
                    param_for_spline = total_space[j]
                if total_space[j] == 0:
                    param_for_spline = total_space[i]
                else:
                    param_for_spline = total_space[i] * (1-norm_dist_from_i) + total_space[j] * norm_dist_from_i
                break

        if param_for_spline is None:
            raise Exception('No intersection found')

        current_window_size = 0
        window_width = 0
        while window_width < self.window_size:
            current_window_size = current_window_size + EXTENSION_STEP

            window_space = np.linspace(param_for_spline-current_window_size, param_for_spline+current_window_size, self.number_of_evaluations)
            if np.any(window_space < 0):
                window_space[window_space < 0] = window_space[window_space < 0] + 1
            if np.any(window_space > 1):
                window_space[window_space > 1] = window_space[window_space > 1] - 1

            coords = splev(window_space, tck)
            window_spline = np.zeros((len(coords[0]), 2))
            window_spline[:, 0] = coords[0]
            window_spline[:, 1] = coords[1]

            window_width = np.cumsum(np.linalg.norm(window_spline - np.roll(window_spline, -1, axis=0), axis=1))[-2]
        return window_space


"""
The following functions are functions that extract features for all bones at once. They all
have the following signature.
They all return an numpy array with len(bones) observations and a variable number of features per observation
:param outlines: Array of the outlines of all bones (ATM these are dicts, mostly spline_params are used in the feature functions)
:param window_spaces: Array of the windows that are evaluated for each bone
"""
def feature_flatten_splines(outlines, window_spaces):
    """
    Evaluate the spline at all points defined by the respective windows spline parameters
    Then return a flattened version of these points as the feature vector for each bone
    """
    outline_points = [evaluate_spline(w, s['spline_params']) for w, s in zip(window_spaces, outlines)]
    return np.array([s.flatten() for s in outline_points])

def feature_use_curvature_of_dist_from_center(outlines, window_spaces):
    """
    Evaluate the spline at all points defined by the respective windows spline parameters
    Then calculate the distance to the center of the coordinate system.
    After that the resulting curve is transformed into frequency space using fft and parts with high
    frequencies are removed. The result is transformed back into spatial space and the curvature of
    the obtained curve is returned
    """
    BORDER_FREQUENCY = 8
    distances = feature_use_distance_to_center(outlines, window_spaces)

    frequencies = np.fft.fftfreq(distances[0].size, 0.05)
    fourier_transforms = np.array([
                                      np.fft.fft(d) for d in distances
                                      ])
    filtered = np.logical_or(frequencies > BORDER_FREQUENCY, frequencies < -BORDER_FREQUENCY)
    fourier_transforms[:,  filtered] = 0
    inverse_fourier_transforms = np.array([
                                              np.fft.ifft(ft) for ft in fourier_transforms
                                              ]).real

    c = np.array([gh.curvature(w, ift) for w, ift in zip(window_spaces, inverse_fourier_transforms)])
    return c

def feature_use_distance_to_center(outlines, window_spaces):
    """
    Evaluate the spline at all points defined by the respective windows spline parameters
    Then calculate the distance to the center of the coordinate system for each point in each observation
    and return this curve as the features
    """
    outline_points = [evaluate_spline(w, s['spline_params']) for w, s in zip(window_spaces, outlines)]
    return np.array([np.linalg.norm(o, axis=1) for o in outline_points])


def feature_use_dist_center_and_curvature(outlines, window_spaces):
    """
    Evaluate the spline at all points defined by the respective windows spline parameters
    Then calculate the distance to the center of the coordinate system for each point in each observation
    Also calculate the curvature for each point in each observation
    Return [distance_to_center, curvature] for each point in each observation as features
    """
    outline_points = [evaluate_spline(w, s['spline_params']) for w, s in zip(window_spaces, outlines)]

    distances = np.array([ np.array([ np.linalg.norm(p) for p in o ]) for o in outline_points ])
    distances = fh.normalize(distances)
    c = np.array([gh.curvature(o[:, 0], o[:, 1]) for o in outline_points])
    c = fh.normalize(c)

    features = np.hstack((distances, c))

    return features


def feature_use_deviation_from_mean(outlines, window_spaces):
    """
    Evaluate the spline at all points defined by the respective windows spline parameters
    Calculate the mean of all evaluated windows
    Use the flattened vector from point to mean for each point in each observation as features
    """
    outline_points = [evaluate_spline(w, s['spline_params']) for w, s in zip(window_spaces, outlines)]

    mean = np.mean(outline_points, axis=0)
    features = []
    for o in outline_points:
        features.append((o - mean).flatten())
    return features


def feature_use_spline_derivatives(outlines, window_spaces):
    """
    Evaluate the spline derivatives for each point in each observation an return them as features
    """
    spline_derivatives = [np.array(spalde(w, s['spline_params'])).flatten() for w, s in zip(window_spaces, outlines)]
    return spline_derivatives


def feature_use_distances_to_markers(outlines, window_spaces):
    """
    Evaluate the spline at all points defined by the respective windows spline parameters
    Calculate the median point for each window
    Calculate the distance to each manually defined marker from the median for each observation and return it as
    features
    """
    window_centers = [np.median(w) for w in window_spaces]
    spline_centers = [
        np.array(evaluate_spline([w], s['spline_params'])[0, :]) for w, s in zip(window_centers, outlines)
        ]

    vectors_to_markers = np.array(
        [s['markers'] - np.tile(c, (s['markers'].shape[0], 1)) for c, s in zip(spline_centers, outlines)]
    )
    norms = np.linalg.norm(vectors_to_markers, axis=2)

    return norms


def wrap_with_pca(fn, n_components):
    """
    Return a function that represents {fn}, but wrapped by a principal component analysis that is executed
    on the features after they are calculated. The values of {n_components} of the PCA are returned as the new features
    for each observation
    :param fn: The feature function that should be wrapped
    :param n_components: The number of components that should be returned
    :return: function
    """
    def feature_fn(spline_params, window_spaces):
        features = fn(spline_params, window_spaces)
        pca = PCA(n_components)
        reduced = pca.fit_transform(features)
        return reduced
    return feature_fn


"""
The following three functions are functions that return indicators for the separability of the
two classes with respect to the calculated features they all have the following signature
:param svc: The trained svc
:param features: The features passed into training the svc
:param known_classes: The classes passed into training the svc
:return: float
"""
def get_recall(svc, features, known_classes):
    """
    Get the recall of the svc
    """
    predicted_classes = svc.predict(features)
    return float(np.count_nonzero(predicted_classes == known_classes)) / float(known_classes.shape[0])


def get_mean_confidence(svc, features, known_classes):
    """
    Get the mean confidence (accumulated distance of all observations to the maximum-margin hyperplane weighted by
    the correctness of the classification of the respective observation)
    """
    confidences = svc.decision_function(features)
    predicted_classes = svc.predict(features)
    equals_predicted_classes = (predicted_classes == known_classes)
    weights = np.full_like(equals_predicted_classes, -1, dtype=np.float)
    weights[equals_predicted_classes] = 1
    return np.mean(np.abs(confidences) * weights)


def get_margin(svc, features, classes):
    """
    Get the margin of the maximum-margin hyperplane of the svc
    """
    return 2 / np.linalg.norm(svc.coef_)


class ComparisonIterator2D:
    """
    Iterates for the 2D comparison, returns an angle every {step_size} degrees, that needs to be evaluated
    """
    MAX_DEGREES = 360

    def __init__(self, step_size):
        self.current_step = 1
        self.step_size = step_size

    def __iter__(self):
        return self

    def __len__(self):
        return int(floor(self.MAX_DEGREES / float(self.step_size)))

    def next(self):
        current_step = self.current_step
        next_step = current_step + self.step_size
        if current_step > self.MAX_DEGREES:
            raise StopIteration
        else:
            self.current_step = next_step
            return current_step




def do_single_comparison(evaluation_point, outlines, feature_fn, window_extractor, window_size, number_of_evaluations):
    """
    Do a comparison of the two classes at a certain angle
    :param evaluation_point: Angle at which the bones are evaluated
    :param outlines: Array of outlines of all bones (ATM these are dicts)
    :param feature_fn: The function that calculates the features for the extracted windows
    :param extract_window_fn: The function to extract the window around the angle for each bone that is then compared
    :param window_size: The width/size of the window around angle that is compared
    :param number_of_evaluations: The number of spline points that are evaluated to calculate the features for the window
    :return: {
        angle: passed through
        windows: the windows (which are arrays of spline parameters) that were evaluated
        features: the features calculated for these windows
        rm: recall * margin
        recall: see get_recall
        mean_confidence: see get_mean_confidence
        margin: see get_margin
    }
    """
    windows = []
    classes = np.array(map(lambda o: o['class'], outlines))
    window_extractor = window_extractor(evaluation_point, window_size, number_of_evaluations)
    for outline in outlines:
        window = window_extractor.extract_window_space(outline)
        windows.append(window)
    features = feature_fn(outlines, windows)
    windows = np.array(windows)
    features = np.array(features)

    svc = LinearSVC()
    svc.fit(features, classes)

    recall = get_recall(svc, features, classes)
    margin = get_margin(svc, features, classes)
    rm = recall*margin
    mean_confidence = get_mean_confidence(svc, features, classes)

    return {
        'angle': evaluation_point,
        'windows': windows,
        'features': features,
        'rm': rm,
        'recall': recall,
        'mean_confidence': mean_confidence,
        'margin': margin
    }


def do_comparison(outlines, class1, class2, step_size=5, feature_fn=None, window_extractor=None, window_size=.75, number_of_evaluations=25, use_pca=True, pca_components=4, progress_callback=None):
    """
    Does a single comparison every {step_size} degrees for classes {class1} and {class2}
    :param outlines: Array of outlines of all bones (ATM these are dicts)
    :param class1: The first class that is evaluated
    :param class2: The second class that is evaluated
    :param step_size: The step width in degrees that is used
    :param feature_fn: see do_single_comparison
    :param extract_window_fn: see do_single_comparison
    :param window_size: see do_single_comparison
    :param number_of_evaluations: see do_single_comparison
    :param use_pca: Wether all features are passed into a principal component analysis before training the svc
    :param pca_components: The number of pca components used to train the svc when use_pca is True
    :param progress_callback: A progress callback that is called whenever a single comparison at an angle is finished
    :return:
    """
    print(window_size, number_of_evaluations, use_pca, pca_components)
    iterator = ComparisonIterator2D(step_size)
    outlines = ch.filter_by_classes(outlines, [class1, class2])
    for outline in outlines:
        outline['spline_params'] = get_spline_params(outline['points'])[0]

    result = []
    for i, angle in enumerate(iterator):
        if use_pca:
            feature_fn = wrap_with_pca(feature_fn, pca_components)
        result.append(do_single_comparison(angle, outlines, feature_fn=feature_fn, window_extractor=window_extractor, window_size=window_size, number_of_evaluations=number_of_evaluations))
        if progress_callback:
            progress_callback(i, len(iterator)-1)
    return result
