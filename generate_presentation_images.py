from skimage.io import imread
from skimage.morphology import binary_erosion
import matplotlib.pyplot as plt
import algorithms.triangulation as at
import numpy as np

bone_pixels = binary_erosion(imread('thesis/presentation/outline-clustered.png'))

points = np.nonzero(bone_pixels)
points = np.vstack(points).transpose()
points[:, 0] = bone_pixels.shape[0] - points[:,0]

triangulation = at.triangulation(bone_pixels, alpha_shape=False)
alpha_shape = at.triangulation(bone_pixels, alpha_shape=True)

plt.triplot(points[:, 0], points[:, 1], alpha_shape.simplices.copy(), color='white')
plt.gca().set_axis_bgcolor('black')
plt.gcf().set_size_inches(bone_pixels.shape[0] / 100, bone_pixels.shape[1] / 100)
plt.savefig('thesis/presentation/outline-alpha.png', dpi=100)
