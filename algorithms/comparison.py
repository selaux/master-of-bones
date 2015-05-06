from math import radians
import numpy as np
from scipy.interpolate import splev, spalde
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from splines import get_spline_params, evaluate_spline
import helpers.geometry as gh
import helpers.classes as ch
import helpers.features as fh

def get_recall(svc, features, known_classes):
    predicted_classes = svc.predict(features)
    return float(np.count_nonzero(predicted_classes == known_classes)) / float(known_classes.shape[0])


def get_margin(svc):
    return 2 / np.linalg.norm(svc.coef_)


def do_single_comparison(angle, outlines, feature_fn, extract_window_fn, window_size, number_of_evaluations):
    windows = []
    classes = np.array(map(lambda o: o['class'], outlines))
    for outline in outlines:
        window = extract_window_fn(outline['spline_params'], angle, window_size, number_of_evaluations)
        windows.append(window)
    features = feature_fn(map(lambda o: o['spline_params'], outlines), windows)
    windows = np.array(windows)
    features = np.array(features)

    svc = LinearSVC()
    svc.fit(features, classes)

    recall = get_recall(svc, features, classes)
    margin = get_margin(svc)
    rm = recall*margin

    return {
        'angle': angle,
        'windows': windows,
        'features': features,
        'rm': rm,
        'recall': recall,
        'margin': margin
    }


def do_comparison(outlines, class1, class2, step_size=5, feature_fn=None, extract_window_fn=None, window_size=.75, number_of_evaluations=25, use_pca=True, pca_components=4, progress_callback=None):
    print(window_size, number_of_evaluations, use_pca, pca_components)
    outlines = ch.filter_by_classes(outlines, [class1, class2])
    eval_angles = range(1, 360, step_size)
    for outline in outlines:
        outline['spline_params'] = get_spline_params(outline['points'])[0]

    result = []
    for i, angle in enumerate(eval_angles):
        if use_pca:
            feature_fn = wrap_with_pca(feature_fn, pca_components)
        result.append(do_single_comparison(angle, outlines, feature_fn=feature_fn, extract_window_fn=extract_window_fn, window_size=window_size, number_of_evaluations=number_of_evaluations))
        if progress_callback:
            progress_callback(i, len(eval_angles)-1)
    return result


def extract_window_space_by_length(tck, degrees, window_size, number_of_evaluations):
    EXTENSION_STEP = 0.0025
    source = np.array([0.0, 0.0])
    ray = np.array(gh.pol2cart(2, radians(degrees)))
    total_space = np.linspace(0, 1, number_of_evaluations)

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
                param_for_spline = total_space[i] * (1-dist_from_i) + total_space[j] * dist_from_i
            break

    if param_for_spline is None:
        raise Exception('No intersection found')

    current_window_size = 0
    window_width = 0
    while window_width < window_size:
        current_window_size = current_window_size + EXTENSION_STEP

        window_space = np.linspace(param_for_spline-current_window_size, param_for_spline+current_window_size, number_of_evaluations)
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


def feature_flatten_splines(spline_params, window_spaces):
    outline_points = [evaluate_spline(w, s) for w, s in zip(window_spaces, spline_params)]
    return np.array([s.flatten() for s in outline_points])


def feature_use_distance_to_center(spline_params, window_spaces):
    outline_points = [evaluate_spline(w, s) for w, s in zip(window_spaces, spline_params)]
    return np.array([np.linalg.norm(o, axis=1) for o in outline_points])


def feature_use_dist_center_and_curvature(spline_params, window_spaces):
    outline_points = [evaluate_spline(w, s) for w, s in zip(window_spaces, spline_params)]

    def curvature(x, y):
        dalpha = np.pi/1000
        xd1 = np.gradient(x)
        xd2 = np.gradient(xd1)
        yd1 = np.gradient(y)
        yd2 = np.gradient(yd1)
        return np.abs(xd1*yd2 - yd1*xd2) / np.power(xd1**2 + yd1**2, 3./2)

    distances = np.array([ np.array([ np.linalg.norm(p) for p in o ]) for o in outline_points ])
    distances = fh.normalize(distances)
    curvature = np.array([curvature(o[:, 0], o[:, 1]) for o in outline_points])
    curvature = fh.normalize(curvature)

    features = np.hstack((distances, curvature))

    return features


def feature_use_deviation_from_mean(spline_params, window_spaces):
    outline_points = [evaluate_spline(w, s) for w, s in zip(window_spaces, spline_params)]

    mean = np.mean(outline_points, axis=0)
    features = []
    for o in outline_points:
        features.append((o - mean).flatten())
    return features


def feature_use_spline_derivatives(spline_params, window_spaces):
    spline_derivatives = [ np.array(spalde(w, s)).flatten() for w, s in zip(window_spaces, spline_params) ]
    return spline_derivatives


def wrap_with_pca(fn, n_components):
    def feature_fn(spline_params, window_spaces):
        features = fn(spline_params, window_spaces)
        pca = PCA(n_components)
        reduced = pca.fit_transform(features)
        return reduced
    return feature_fn