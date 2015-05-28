import numpy as np
from math import radians, degrees
from helpers import geometry as gh
from scipy.signal import argrelextrema

def get_using_angles(outline):
        landmark_definitions = [
            {
                'a_min': 30,
                'a_max': 90,
                'method': 'max'
            },
            {
                'a_min': 80,
                'a_max': 100,
                'method': 'min'
            },
            {
                'a_min': 90,
                'a_max': 150,
                'method': 'max'
            },
            {
                'a_min': 170,
                'a_max': 190,
                'method': 'max'
            },
            {
                'a_min': 210,
                'a_max': 270,
                'method': 'max'
            },
            {
                'a_min': 260,
                'a_max': 280,
                'method': 'min'
            },
            {
                'a_min': 270,
                'a_max': 330,
                'method': 'max'
            }
        ]
        landmarks = []
        rho = 2
        centroid = np.mean(outline, axis=0)

        for definition in landmark_definitions:
            intersects_for_angles = []
            method = np.argmax if definition['method'] == 'max' else np.argmin

            for angle in range(definition['a_min'], definition['a_max']):
                startpoint = centroid
                endpoint = gh.pol2cart(rho, radians(angle)) + centroid

                intersect = None
                for i in range(outline.shape[0]):
                    j = i+1 if i != outline.shape[0]-1 else 0
                    p1 = outline[i, :]
                    p2 = outline[j, :]

                    intersect = gh.seg_intersect(p1, p2, startpoint, endpoint)
                    if intersect is not None:
                        break
                if intersect is None:
                    raise Exception('No Intersect.')
                else:
                    intersects_for_angles.append(intersect)

            intersects_for_angles = np.array(intersects_for_angles)
            distances = np.linalg.norm(intersects_for_angles - np.tile(centroid, (intersects_for_angles.shape[0], 1)), axis=1)

            landmarks.append(intersects_for_angles[method(distances), :])

        return np.array(landmarks)

def get_using_space_partitioning(outline_points):
    landmark_definitions = [
        {
            'x_min': -1,
            'x_max': 0.25,
            'y_min': -1.5,
            'y_max': -0.75,
            'method': 'min',
            'dim': 'y'
        },
        {
            'x_min': -0.5,
            'x_max': 0.5,
            'y_min': -1.5,
            'y_max': -0.75,
            'method': 'max',
            'dim': 'y'
        },
        {
            'x_min': 0.25,
            'x_max': 1,
            'y_min': -1.5,
            'y_max': -0.75,
            'method': 'min',
            'dim': 'y'
        },
        {
            'x_min': -1.5,
            'x_max': 0,
            'y_min': -0.3,
            'y_max': 0.3,
            'method': 'min',
            'dim': 'x'
        },
        {
            'x_min': -1,
            'x_max': -0.25,
            'y_min': 0.75,
            'y_max': 1.5,
            'method': 'max',
            'dim': 'y'
        },
        {
            'x_min': -0.5,
            'x_max': 0.5,
            'y_min': 0.75,
            'y_max': 1.5,
            'method': 'min',
            'dim': 'y'
        },
        {
            'x_min': 0.25,
            'x_max': 1,
            'y_min': 0.75,
            'y_max': 1.5,
            'method': 'max',
            'dim': 'y'
        }
    ]
    landmarks = []

    for definition in landmark_definitions:
        method = np.argmax if definition['method'] == 'max' else np.argmin
        method_extrema = np.greater if definition['method'] == 'max' else np.less
        dim = 0 if definition['dim'] == 'y' else 1
        sort_dim = 1 - dim

        x_min = definition['x_min']
        x_max = definition['x_max']
        y_min = definition['y_min']
        y_max = definition['y_max']
        selector = outline_points[:, 1] >= x_min
        selector = np.logical_and(selector, outline_points[:, 1] <= x_max)
        selector = np.logical_and(selector, outline_points[:, 0] >= y_min)
        selector = np.logical_and(selector, outline_points[:, 0] <= y_max)

        possible_points = outline_points[selector, :]
        sorted_points = possible_points[np.argsort(possible_points[:, sort_dim]), :]
        #print(sorted_points)

        local_extrema = argrelextrema(sorted_points[:, dim], method_extrema, order=10)[0]
        if (len(local_extrema) > 0):
            index_of_landmark = local_extrema[0]
        else:
            index_of_landmark = method(sorted_points[:, dim])

        #print(index_of_landmark)

        landmarks.append(sorted_points[index_of_landmark, :])
    return np.array(landmarks)