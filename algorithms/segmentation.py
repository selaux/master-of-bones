import numpy as np
from skimage import io, color
from skimage.filters.rank import median, mean
from skimage.transform import rescale
from skimage.morphology import disk, watershed
from skimage import morphology
from skimage import segmentation


MU = 1.25
SCALE = 0.1
INITIAL_WINDOW_SIZE = 50
BORDER_SIZE = 100
PADDING = 150

def watershed_cut(image):
    labels = watershed(image)
    bone_pixels = labels == 1

    indices_of_bone_pixels = np.where(bone_pixels == True)
    min_dim_0 = indices_of_bone_pixels[0].min() - PADDING
    max_dim_0 = indices_of_bone_pixels[0].max() + PADDING
    min_dim_1 = indices_of_bone_pixels[1].min() - PADDING
    max_dim_1 = indices_of_bone_pixels[1].max() + PADDING

    min_dim_0 = max(min_dim_0, 0)
    min_dim_1 = max(min_dim_1, 0)
    max_dim_0 = min(max_dim_0, image.shape[0])
    max_dim_1 = min(max_dim_1, image.shape[1])

    return np.copy(image[min_dim_0:max_dim_0,min_dim_1:max_dim_1,:])

def watershed(image):
    hsv_image = color.rgb2hsv(image)

    low_res_image = rescale(hsv_image[:, :, 0], SCALE)
    local_mean = mean(low_res_image, disk(50))
    local_minimum_flat = np.argmin(local_mean)
    local_minimum = np.multiply(np.unravel_index(local_minimum_flat, low_res_image.shape), round(1 / SCALE))

    certain_bone_pixels = np.full_like(hsv_image[:, :, 0], False, bool)
    certain_bone_pixels[
    local_minimum[0] - INITIAL_WINDOW_SIZE/2:local_minimum[0]+INITIAL_WINDOW_SIZE/2,
    local_minimum[1] - INITIAL_WINDOW_SIZE/2:local_minimum[1]+INITIAL_WINDOW_SIZE/2
    ] = True

    certain_non_bone_pixels = np.full_like(hsv_image[:, :, 0], False, bool)
    certain_non_bone_pixels[0:BORDER_SIZE, :] = True
    certain_non_bone_pixels[-BORDER_SIZE:-1, :] = True
    certain_non_bone_pixels[:, 0:BORDER_SIZE] = True
    certain_non_bone_pixels[:, -BORDER_SIZE:-1] = True

    smoothed_hsv = median(hsv_image[:, :, 0], disk(50))
    threshold = MU * np.median(smoothed_hsv[certain_bone_pixels])

    possible_bones = np.zeros_like(hsv_image[:, :, 0])
    possible_bones[smoothed_hsv < threshold] = 1

    markers = np.zeros_like(possible_bones)
    markers[certain_bone_pixels] = 1
    markers[certain_non_bone_pixels] = 2

    labels = morphology.watershed(-possible_bones, markers)

    return labels

def slic(image):
    NUMBER_OF_CLUSTERS = 6
    COMPACTNESS = 15
    return segmentation.slic(image, NUMBER_OF_CLUSTERS, COMPACTNESS)
