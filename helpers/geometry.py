import math
import numpy as np

def alpha_shape(triangulation, alpha=25):
    triangles_to_delete = []

    for index, triangle in enumerate(triangulation.points[triangulation.simplices]):
        pa = triangle[0]
        pb = triangle[1]
        pc = triangle[2]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)
        # Here's the radius filter.
        if circum_r > alpha:
            triangles_to_delete.append(index)
    triangulation.simplices = np.delete(triangulation.simplices, triangles_to_delete, axis=0)

    return triangulation

def extract_circle(outline_unsorted):
    if len(outline_unsorted) == 0:
        return None

    outline = []

    outline.append([ outline_unsorted[0][0], outline_unsorted[0][1] ])
    del outline_unsorted[0]
    init = outline[0][0]
    while outline[-1][1] != init:
        found = False
        look_for = outline[-1][1]
        for i in range(0, len(outline_unsorted)):
            edge = outline_unsorted[i]
            if edge[0] == look_for:
                found = True
                outline.append([ edge[0], edge[1] ])
            if edge[1] == look_for:
                found = True
                outline.append([ edge[1], edge[0] ])
            if found:
                del outline_unsorted[i]
                break
        if not found:
            return None
    return outline

def extract_outline_edges(simplices):
    edges = np.zeros((simplices.shape[0] * 3, 2), dtype=np.int)
    outline_unsorted = []
    circles = []

    for i, triangle in enumerate(simplices):
        edges[3*i] = np.sort([ triangle[0], triangle[1] ])
        edges[3*i+1] = np.sort([ triangle[0], triangle[2] ])
        edges[3*i+2] = np.sort([ triangle[1], triangle[2] ])

    cont_edges = np.ascontiguousarray(edges).view(np.dtype((np.void, edges.dtype.itemsize * edges.shape[1])))
    unique_edges, idx, inv, counts = np.unique(cont_edges, True, True, True)
    unique_edges = edges[idx]

    for i in range(0, unique_edges.shape[0]):
        if counts[i] == 1:
            outline_unsorted.append(edges[idx[i], :])

    circle = extract_circle(outline_unsorted)
    while circle:
        circles.append(circle)
        circle = extract_circle(outline_unsorted)

    if len(circles) == 0:
        raise Exception('No outline found!')
    outline = max(circles, key=lambda c: len(c))

    return np.array(outline, dtype=np.int)

def get_centroid_and_scale(points):
    centroid = np.mean(points, axis=0)
    result = points - np.tile(centroid, (points.shape[0], 1))
    scale_factor = np.sqrt(np.sum(np.power(result, 2)) / result.shape[0])
    return centroid, scale_factor

def normalize_outline_with_markers(outline):
    centroid, scale = get_centroid_and_scale(outline['points'])
    markers = map(lambda i: outline['markers'][i] if i in outline['markers'] else None, range(1, 12))

    outline['points'], outline['edges'] = normalize_outline(outline['points'], outline['edges'])
    if not None in markers:
        markers = np.array(markers) - np.tile(centroid, (len(markers), 1))
        outline['markers'] = np.divide(markers, scale)

def normalize_outline(points, ordered_edges):
    centroid = np.mean(points, axis=0)
    result = points - np.tile(centroid, (points.shape[0], 1))
    scale_factor = np.sqrt(np.sum(np.power(result, 2)) / result.shape[0])
    points = np.divide(result, scale_factor)

    root = np.array([0, 0])
    raypoint = np.array(pol2cart(2, 0))
    intersect = None
    start_index = None
    start_point = None

    for i, s in enumerate(ordered_edges):
        intersect = seg_intersect(root, raypoint, points[s[0], :], points[s[1], :])
        if intersect is not None:
            start_index = i
            start_point = intersect
            break

    if intersect is None:
        raise Exception('No intersect found at {0} degrees'.format(0))

    intersect[0] = 0
    new_points = np.roll(points, -start_index-1, axis=0)
    if new_points[-1, 0] > new_points[0, 0]:
        # Invert direction
        new_points = np.flipud(new_points)

    new_points = np.vstack((intersect, new_points))
    mask = ((np.diff(new_points[:, 0]) != 0) & (np.diff(new_points[:, 0]) != 0))
    new_points = new_points[mask, :]

    new_edges = np.zeros((new_points.shape[0], 2), dtype=np.int)
    new_edges[:, 0] = range(0, len(new_points))
    new_edges[:, 1] = range(1, len(new_points) + 1)
    new_edges[-1, 1] = 0

    return new_points, new_edges


def extract_outline(points, simplices):
    outline = extract_outline_edges(simplices)
    points_in_outline = np.array(np.concatenate((outline[:, 0], [outline[-1, 1]])))
    new_places = range(0, points_in_outline.shape[0])

    new_outline = np.copy(outline)
    for new_place, old_place in enumerate(points_in_outline):
        new_outline[outline == old_place] = new_place

    return points[points_in_outline, :].copy(), new_outline

def perp(u, v):
    return u[0]*v[1] - u[1]*v[0]

def in_segment(a, b1, b2):
    if b1[0] != b2[0]:
        if b1[0] <= a[0] and a[0] <= b2[0]:
            return True
        if b1[x] >= a[0] and a[0] >= b2[0]:
            return True
    else:
        if b1[1] <= a[1] and a[1] <= b2[1]:
            return True
        if b1[1] >= a[1] and a[1] >= b2[1]:
            return True
    return False

def seg_intersect(a1, a2, b1, b2):
    u = a2 - a1
    v = b2 - b1
    w = a1 - b1
    d = perp(u, v)

    if abs(d) < 0.00000001:
        if perp(u,w) != 0 or perp(v,w) != 0:
            return None

        du = np.dot(u, u)
        dv = np.dot(v, v)
        if du == 0 and dv == 0:
            return a1 if np.array_equal(a1, b1) else None
        if du == 0:
            return a1 if in_segment(a1, b1, b2) else None
        if dv == 0:
            return b1 if in_segment(b1, a1, a2) else None
        w2 = a2 - b1
        if v[0] != 0:
            t0, t1 = w[0] / v[0], w2[0] / v[0]
        else:
            t0, t1 = w[1] / v[1], w2[1] / v[1]
        if t0 > t1:
            t1, t2 = t2, t1
        if t0 > 1 or t1 < 0:
            return None
        t0 = 0 if t0 < 0 else t0
        t1 = 1 if t1 > 1 else t1
        return b1 + t0 * v
    else:
        si = perp(v,w) / d
        if si < 0 or si > 1:
            return None

        ti = perp(u,w) / d
        if ti < 0 or ti > 1:
            return None
        return a1 + si * u

def curvature(x, y):
    """
    Calculate the curvature for the points that are represented by x and y
    :param x: array of x coordinates
    :param y: array of y coordinates
    :return:
    """
    dalpha = np.pi/1000
    xd1 = np.gradient(x, 2)
    xd2 = np.gradient(xd1, 2)
    yd1 = np.gradient(y, 2)
    yd2 = np.gradient(yd1, 2)

    return np.abs(xd1*yd2 - yd1*xd2) / np.power(xd1**2 + yd1**2, 3.0/2.0)

def angle(v1, v2):
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)

def cart2pol(y, x):
    rho = np.linalg.norm([x, y])
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (y, x)
