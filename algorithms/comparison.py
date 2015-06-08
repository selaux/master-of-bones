from abc import ABCMeta, abstractmethod
from math import radians, floor, degrees, sqrt
import numpy as np
from scipy.interpolate import splev, spalde
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
import vtk
from helpers.to_vtk import get_line_actor
from splines import get_spline_params, evaluate_spline
import helpers.geometry as gh
import helpers.classes as ch
import helpers.features as fh
import matplotlib.cm as cmx


class WindowExtractor:
    """
    Base class to extract the section that is evaluated around a certain evaluation point with size {window_size}
    and the number of points inside the window being {number_of_evaluations}
    For 2D the evaluation point could be an angle
    For 3D it could be a tuple of two angles
    extract_window_space needs to be implemented for this class to work
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


class FeatureCalculation:
    """
    This is the base class for functions that extract features for all bones at once.
    They all return an numpy array with len(bones) observations and a variable number of features per observation
    use_pca: Wether the function should be wrapped inside a pca to reduce the number of dimensions
    number_of_pca_components: How many pca components should be considered
    calculate_raw_features needs to be implemented for this class to work
    """
    __metaclass__ = ABCMeta

    def __init__(self, use_pca, number_of_pca_components):
        self.use_pca = use_pca
        self.number_of_pca_components = number_of_pca_components

    def calculate_features(self, bones, windows):
        features = self.calculate_raw_features(bones, windows)
        if self.use_pca:
            pca = PCA(self.number_of_pca_components)
            reduced = pca.fit_transform(features)
            return reduced
        else:
            return features

    @abstractmethod
    def calculate_raw_features(self, bones, windows):
        raise NotImplementedError

class FeatureFlattenSplines(FeatureCalculation):
    """
    Evaluate the spline at all points defined by the respective windows spline parameters
    Then return a flattened version of these points as the feature vector for each bone
    """
    def calculate_raw_features(self, bones, windows):
        outline_points = [evaluate_spline(w, s['spline_params']) for w, s in zip(windows, bones)]
        return np.array([s.flatten() for s in outline_points])

class FeatureCurvatureOfDistanceFromCenter(FeatureCalculation):
    """
    Evaluate the spline at all points defined by the respective windows spline parameters
    Then calculate the distance to the center of the coordinate system.
    After that the resulting curve is transformed into frequency space using fft and parts with high
    frequencies are removed. The result is transformed back into spatial space and the curvature of
    the obtained curve is returned
    """
    def calculate_raw_features(self, bones, windows):
        distance_to_center_feature = FeatureDistanceToCenter(False, 0)

        BORDER_FREQUENCY = 8
        distances = distance_to_center_feature.calculate_raw_features(bones, windows)

        frequencies = np.fft.fftfreq(distances[0].size, 0.05)
        fourier_transforms = np.array([np.fft.fft(d) for d in distances])
        filtered = np.logical_or(frequencies > BORDER_FREQUENCY, frequencies < -BORDER_FREQUENCY)
        fourier_transforms[:,  filtered] = 0
        inverse_fourier_transforms = np.array([np.fft.ifft(ft) for ft in fourier_transforms]).real

        c = np.array([gh.curvature(w, ift) for w, ift in zip(windows, inverse_fourier_transforms)])
        return c

class FeatureDistanceToCenter(FeatureCalculation):
    """
    Evaluate the spline at all points defined by the respective windows spline parameters
    Then calculate the distance to the center of the coordinate system for each point in each observation
    and return this curve as the features
    """
    def calculate_raw_features(self, bones, windows):
        outline_points = [evaluate_spline(w, s['spline_params']) for w, s in zip(windows, bones)]
        return np.array([np.linalg.norm(o, axis=1) for o in outline_points])

class FeatureDistanceToCenterAndCurvature(FeatureCalculation):
    """
    Evaluate the spline at all points defined by the respective windows spline parameters
    Then calculate the distance to the center of the coordinate system for each point in each observation
    Also calculate the curvature for each point in each observation
    Return [distance_to_center, curvature] for each point in each observation as features
    """
    def calculate_raw_features(self, bones, windows):
        outline_points = [evaluate_spline(w, s['spline_params']) for w, s in zip(windows, bones)]

        distances = np.array([ np.array([ np.linalg.norm(p) for p in o ]) for o in outline_points ])
        distances = fh.normalize(distances)
        c = np.array([gh.curvature(o[:, 0], o[:, 1]) for o in outline_points])
        c = fh.normalize(c)

        features = np.hstack((distances, c))

        return features

class FeatureFlattenedDeviationVectorFromMeanBone(FeatureCalculation):
    """
    Evaluate the spline at all points defined by the respective windows spline parameters
    Calculate the mean of all evaluated windows
    Use the flattened vector from point to mean for each point in each observation as features
    """
    def calculate_raw_features(self, bones, windows):
        outline_points = [evaluate_spline(w, s['spline_params']) for w, s in zip(windows, bones)]

        mean = np.mean(outline_points, axis=0)
        features = []
        for o in outline_points:
            features.append((o - mean).flatten())
        return features

class FeatureSplineDerivatives(FeatureCalculation):
    """
    Evaluate the spline derivatives for each point in each observation an return them as features
    """
    def calculate_raw_features(self, bones, windows):
        spline_derivatives = [np.array(spalde(w, s['spline_params'])).flatten() for w, s in zip(windows, bones)]
        return spline_derivatives

class FeatureDistancesToMarkers(FeatureCalculation):
    """
    Evaluate the spline at all points defined by the respective windows spline parameters
    Calculate the median point for each window
    Calculate the distance to each manually defined marker from the median for each observation and return it as
    features
    """
    def calculate_raw_features(self, bones, windows):
        window_centers = [np.median(w) for w in windows]
        spline_centers = [
            np.array(evaluate_spline([w], s['spline_params'])[0, :]) for w, s in zip(window_centers, bones)
            ]

        vectors_to_markers = np.array(
            [s['markers'] - np.tile(c, (s['markers'].shape[0], 1)) for c, s in zip(spline_centers, bones)]
        )
        norms = np.linalg.norm(vectors_to_markers, axis=2)

        return norms


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


class SingleComparisonResult:
    """
    Class to store the result of a comparison at a single evaluation point.
    This is an abstract class. get_windows_actors needs to be implemented to make this work
    An evaluation point in 2D is an angle
    An evaluation point in 3D is a tuple of angles
    """
    __metaclass__ = ABCMeta

    def __init__(self, evaluation_point, bones, windows, features, svc):
        self.evaluation_point = evaluation_point
        self.bones = bones
        self.windows = windows
        self.features = features
        self.svc = svc

        self.recall = self.get_recall_indicator()
        self.mean_confidence = self.get_mean_confidence_indicator()
        self.margin = self.get_margin_indicator()
        self.rm = self.margin * self.recall

    def get_recall_indicator(self):
        """
        Get the recall of the svc
        """
        classes = np.array(map(lambda o: o['class'], self.bones))
        predicted_classes = self.svc.predict(self.features)
        return float(np.count_nonzero(predicted_classes == classes)) / float(classes.shape[0])

    def get_mean_confidence_indicator(self):
        """
        Get the mean confidence (accumulated distance of all observations to the maximum-margin hyperplane weighted by
        the correctness of the classification of the respective observation)
        """
        classes = np.array(map(lambda o: o['class'], self.bones))
        confidences = self.svc.decision_function(self.features)
        predicted_classes = self.svc.predict(self.features)
        equals_predicted_classes = (predicted_classes == classes)
        weights = np.full_like(equals_predicted_classes, -1, dtype=np.float)
        weights[equals_predicted_classes] = 1
        return np.mean(np.abs(confidences) * weights)

    def get_margin_indicator(self):
        """
        Get the margin of the maximum-margin hyperplane of the svc
        """
        return np.linalg.norm(self.svc.coef_)

    def get_performance_indicators(self):
        """
        Return an array of all performance indicators of the algorithm at this evaluation_point
        """
        return [
            {
                'label': 'Mean Confidence',
                'value': self.mean_confidence,
            },
            {
                'label': 'Margin',
                'value': self.margin,
            },
            {
                'label': 'Precision',
                'value': self.recall,
            },
            {
                'label': 'Precision * Margin',
                'value': self.rm,
            }
        ]

    @abstractmethod
    def get_windows_actors(self):
        raise NotImplementedError

class SingleComparisonResult2D(SingleComparisonResult):
    def get_windows_actors(self):
        actors = []
        for i, bone in enumerate(self.bones):
            spline_params = bone['spline_params']
            window = self.windows[i, :]
            points = evaluate_spline(window, spline_params)

            actor = get_line_actor(points)
            actor.GetProperty().SetRepresentationToWireframe()
            actor.GetProperty().SetLineWidth(1.5)
            actor.GetProperty().SetColor(bone['color'][0], bone['color'][1], bone['color'][2])

            actors.append(actor)
        return actors

def do_single_comparison(evaluation_point, bones, feature_extractor, window_extractor, window_size, number_of_evaluations):
    """
    Do a comparison of the two classes at a certain angle
    :param evaluation_point: Angle at which the bones are evaluated
    :param bones: Array of outlines of all bones (ATM these are dicts)
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
    classes = np.array(map(lambda o: o['class'], bones))
    window_extractor_instance = window_extractor(evaluation_point, window_size, number_of_evaluations)
    for outline in bones:
        window = window_extractor_instance.extract_window_space(outline)
        windows.append(window)
    features = feature_extractor.calculate_features(bones, windows)
    windows = np.array(windows)
    features = np.array(features)

    svc = LinearSVC()
    svc.fit(features, classes)

    return SingleComparisonResult2D(evaluation_point, bones, windows, features, svc)

class ComparisonResult:
    """
    Class to encapsulate the comparison result for all evaluation points it has to implement
    get_closest_single_result: Should return the closest result for a evaluation point
    (used to display the windows for this point)
    update_actor Should update points / edges / faces / colors of the actor that represents this comparison
    """
    __metaclass__ = ABCMeta

    def __init__(self, class1, class2, bones):
        self.class1 = class1
        self.class2 = class2
        self.bones = bones
        self.single_results = []

        self.actor = vtk.vtkActor()
        self.actor_data = vtk.vtkPolyData()
        self.actor_mapper = vtk.vtkPolyDataMapper()
        self.actor_mapper.SetInputData(self.actor_data)
        self.actor.SetMapper(self.actor_mapper)
        self.actor.GetProperty().SetRepresentationToWireframe()
        self.actor.GetProperty().SetLineWidth(3)

    def add_single_result(self, result):
        self.single_results.append(result)

    @abstractmethod
    def get_closest_single_result(self, evaluation_point):
        raise NotImplementedError

    @abstractmethod
    def update_actor(self, ratio, index_of_performance_indicator):
        raise NotImplementedError


class ComparisonResult2D(ComparisonResult):
    def get_morphed_outline(self, ratio):
        space = np.linspace(0, 1, 500)
        class1bones = np.array([evaluate_spline(space, o['spline_params']) for o in self.bones if o['class'] == self.class1])
        class2bones = np.array([evaluate_spline(space, o['spline_params']) for o in self.bones if o['class'] == self.class2])
        class1part = ratio
        class2part = 1 - ratio

        return np.mean(class1bones, axis=0) * class1part + np.mean(class2bones, axis=0) * class2part

    def get_color_for_angle(self, angle, index_of_performance_indicator):
        pi = map(lambda c: c.get_performance_indicators()[index_of_performance_indicator]['value'], self.single_results)
        pi = np.array(pi)
        min_incidator = min(pi)
        pi = pi - min(pi)
        max_indicator = max(pi)
        closest = self.get_closest_single_result(angle)
        indicator = (closest.get_performance_indicators()[index_of_performance_indicator]['value'] - min_incidator) / max_indicator
        raw_color = cmx.gnuplot(sqrt(indicator))
        color = (int(raw_color[0] * 255), int(raw_color[1] * 255), int(raw_color[2] * 255))
        return color

    def get_closest_single_result(self, angle):
        angles = np.array([m.evaluation_point for m in self.single_results])
        closest = np.argmin(np.abs(angles - angle))
        return self.single_results[closest]

    def get_min_and_max_performance_indicators(self, index_of_performance_indicator):
        indicators = np.array([m.get_performance_indicators()[index_of_performance_indicator]['value'] for m in self.single_results])
        min = np.min(indicators)
        max = np.max(indicators)
        return min, max

    def update_actor(self, ratio, index_of_performance_indicator):
        points = vtk.vtkPoints()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        vertices = vtk.vtkCellArray()
        lines = vtk.vtkCellArray()
        mean_outline = self.get_morphed_outline(ratio)

        for i, point in enumerate(mean_outline):
            points.InsertNextPoint([point[1], point[0], 1.0])
            vertex = vtk.vtkVertex()
            vertex.GetPointIds().SetId(0, i)
            vertices.InsertNextCell(vertex)

            rho, phi = gh.cart2pol(point[0], point[1])
            phi = degrees(phi) + 360 if phi < 0 else degrees(phi)
            color = self.get_color_for_angle(phi, index_of_performance_indicator)
            colors.InsertNextTuple3(*color)

        for i in range(0, mean_outline.shape[0]):
            j = i+1 if i+1 < mean_outline.shape[0] else 0
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i)
            line.GetPointIds().SetId(1, j)
            lines.InsertNextCell(line)

        self.actor_data.SetPoints(points)
        self.actor_data.SetLines(lines)
        self.actor_data.SetVerts(vertices)
        self.actor_data.GetPointData().SetScalars(colors)
        self.actor_data.Modified()

def do_comparison(outlines, class1, class2, step_size=5, feature_extractor=None, window_extractor=None, window_size=.75, number_of_evaluations=25, use_pca=True, pca_components=4, progress_callback=None):
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
    :param progress_callback: A progress callback that is called whenever a single comparison at an evaluation_point is finished
    :return:
    """
    print(window_size, number_of_evaluations, use_pca, pca_components)
    iterator = ComparisonIterator2D(step_size)
    outlines = ch.filter_by_classes(outlines, [class1, class2])
    for outline in outlines:
        outline['spline_params'] = get_spline_params(outline['points'])[0]
    feature_extractor_instance = feature_extractor(use_pca, pca_components)

    result = ComparisonResult2D(class1, class2, outlines)
    for i, evaluation_point in enumerate(iterator):
        result.add_single_result(do_single_comparison(
            evaluation_point,
            outlines,
            feature_extractor=feature_extractor_instance,
            window_extractor=window_extractor,
            window_size=window_size,
            number_of_evaluations=number_of_evaluations)
        )
        if progress_callback:
            progress_callback(i, len(iterator)-1)
    return result
