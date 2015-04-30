import numpy as np
from math import radians, degrees
from helpers import geometry as gh

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
                'a_min': 160,
                'a_max': 200,
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