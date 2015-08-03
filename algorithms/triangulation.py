from math import sqrt
from scipy.spatial.qhull import Delaunay
import numpy as np

def alpha_shape_unique(simplices, alpha=25):
    triangles_to_delete = []
    index = 0
    length = len(simplices)
    while index < length:
        pa = simplices[index, 0, :]
        pb = simplices[index, 1, :]
        pc = simplices[index, 2, :]
        # Lengths of sides of triangle
        a = sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        # Area of triangle by Heronc's formula
        area = sqrt(s*(s-a)*(s-b)*(s-c))
        if area == 0.0:
            triangles_to_delete.append(index)
        else:
            circum_r = a*b*c/(4.0*area)
            # Here's the radius filter.
            if circum_r > alpha:
                triangles_to_delete.append(index)
        index += 1
    return triangles_to_delete

def triangulation(bone_pixels):
    indices_of_bone_pixels = np.nonzero(bone_pixels)
    indices_of_bone_pixels = np.vstack(indices_of_bone_pixels).transpose()
    indices_of_bone_pixels[:, 0] = bone_pixels.shape[0] - indices_of_bone_pixels[:,0]

    #tb = time()
    triangulation = Delaunay(indices_of_bone_pixels)
    #print(time() - tb)
    to_delete = alpha_shape_unique(triangulation.points[triangulation.simplices], alpha=25)
    triangulation.simplices = np.delete(triangulation.simplices, to_delete, axis=0)
    #print(time() - tb)

    return triangulation
