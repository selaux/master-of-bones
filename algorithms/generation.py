from functools import partial
from random import randint
import numpy as np
from math import cos, sin, sqrt, pi, exp
from math import radians

def evaluate_gaussian(height, center, std_dev, x):
    return height * exp(-1.0 * ((x-center)**2) / (2 * (std_dev**2)))

def generate_ellipse(ratio, min_number_of_points, max_number_of_points, std_dev_angle, std_dev_radius, std_dev_transformation, special_points):
    number_of_points = randint(min_number_of_points, max_number_of_points)
    step_size = 360.0 / number_of_points
    points = []
    randomized_special_points = []

    for sp in special_points:
        center = np.random.normal(sp['angle'], sp['angle_std_dev']) if sp['angle_std_dev'] != 0.0 else sp['angle']
        height = np.random.normal(sp['height'], sp['height_std_dev']) if sp['height_std_dev'] != 0.0 else sp['height']
        std_dev = np.random.normal(sp['std_dev'], sp['std_dev_std_dev']) if sp['std_dev_std_dev'] != 0.0 else sp['std_dev']
        randomized_special_points.append({
            'center': center,
            'height': height,
            'std_dev': std_dev
        })

    steps = map(lambda x: step_size * x, range(number_of_points))
    randomized_steps = []
    for i, step in enumerate(steps):
        min_step = steps[i-1] if i != 0 else 0
        max_step = steps[i+1] if i != len(steps)-1 else 360

        std_dev_step = 0.5 * step_size * std_dev_angle
        if i != 0 and std_dev_step > 0.0:
            new_step = step + np.random.normal(0.0, std_dev_step)
            while new_step <= min_step or new_step >= max_step:
                new_step = step + np.random.normal(0.0, std_dev_step)
            step = new_step

        randomized_steps.append(step)

    normal_steps = np.array(map(lambda s: [ ratio * sin(radians(s)), cos(radians(s)) ], randomized_steps))
    normals = []
    for i, step in enumerate(randomized_steps):
        after = i+1 if i < len(normal_steps)-1 else 0
        before = i-1 if i != 0 else len(normal_steps)-1

        v = np.roll(normal_steps[before, :] - normal_steps[after, :], 1)
        v /= np.linalg.norm(v)

        normal = np.array([v[0], -v[1]])

        normals.append(normal)
    normals = np.array(normals)

    for i, step in enumerate(randomized_steps):
        a = 1
        b = ratio

        if std_dev_radius > 0.0:
            a += np.random.normal(0.0, std_dev_radius)
            b += np.random.normal(0.0, std_dev_radius)

        x = a * cos(radians(step))
        y = b * sin(radians(step))

        for sp in randomized_special_points:
            gauss1 = partial(evaluate_gaussian, sp['height'], sp['center'], sp['std_dev'])
            gauss2 = partial(evaluate_gaussian, sp['height'], sp['center']+360.0, sp['std_dev'])
            factor1 = gauss1(step)
            factor2 = gauss2(step)
            #print(step, factor1, factor2, normals[i, :])

            evaluated = factor1 * normals[i, :] + factor2 * normals[i, :]
            x += evaluated[1]
            y += evaluated[0]

        points.append([y, x])
    points = np.array(points)

    if std_dev_transformation > 0.0:
        trafo_x = np.random.normal(0.0, std_dev_transformation)
        trafo_y = np.random.normal(0.0, std_dev_transformation)

        points[:, 0] += trafo_y
        points[:, 1] += trafo_x

    return points

