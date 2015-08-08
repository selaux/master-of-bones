from math import radians, sqrt
import numpy as np
from scipy.cluster.vq import kmeans2
from skimage import  color, img_as_ubyte
from skimage.filters import gabor_filter, gaussian_filter
from skimage.filters.rank import median, mean
from skimage.transform import rescale
from skimage.morphology import disk, watershed
from skimage import morphology
from skimage import segmentation
import mahotas as mh

from helpers import features as fh
from sklearn.decomposition import PCA

MU = 1.3
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

def threshold_median_filter():
    pass

def felzenszwalb(image):
    SCALE = 250
    SIGMA = 5
    MIN_SIZE = 500
    return segmentation.felzenszwalb(image, SCALE, SIGMA, MIN_SIZE)

def slic(image):
    NUMBER_OF_CLUSTERS = 6
    COMPACTNESS = 15
    return segmentation.slic(image, NUMBER_OF_CLUSTERS, COMPACTNESS)

def textural_edges(image):
    WINDOW_HEIGHT = 0
    WINDOW_WIDTH = 10
    SCALE = .1

    def extract_semi_localities_for_each_pixel_x(image):
        image_with_padded_borders = np.pad(image, ((WINDOW_HEIGHT, WINDOW_HEIGHT), (WINDOW_WIDTH, WINDOW_WIDTH), (0, 0)), mode='symmetric')
        neighborhoods = np.zeros((image.shape[0] * image.shape[1], 2, WINDOW_WIDTH, 3), dtype=np.uint8)

        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                neighborhoods[y*image.shape[1] + x, 0, :, :] = image_with_padded_borders[y:y+2*WINDOW_HEIGHT+1, x:x+WINDOW_WIDTH, :]
                neighborhoods[y*image.shape[1] + x, 1, :, :] = image_with_padded_borders[y:y+2*WINDOW_HEIGHT+1, x+WINDOW_WIDTH:x+2*WINDOW_WIDTH, :]

        return neighborhoods

    def extract_semi_localities_for_each_pixel_y(image):
        image_with_padded_borders = np.pad(image, ((WINDOW_WIDTH, WINDOW_WIDTH), (WINDOW_HEIGHT, WINDOW_HEIGHT), (0, 0)), mode='symmetric')
        neighborhoods = np.zeros((image.shape[0] * image.shape[1], 2, WINDOW_WIDTH, 3), dtype=np.uint8)

        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                neighborhoods[y*image.shape[1] + x, 0, :, :] = image_with_padded_borders[y:y+WINDOW_WIDTH, x:x+2*WINDOW_HEIGHT+1, :].transpose((1, 0 ,2))
                neighborhoods[y*image.shape[1] + x, 1, :, :] = image_with_padded_borders[y+WINDOW_WIDTH:y+2*WINDOW_WIDTH, x:x+2*WINDOW_HEIGHT+1, :].transpose((1, 0 ,2))

        return neighborhoods

    def extract_semi_localities_for_each_pixel(image):
        return np.concatenate((extract_semi_localities_for_each_pixel_x(image), extract_semi_localities_for_each_pixel_y(image)), axis=1)

    def map_feature_for_pixel(neighborhood):
        neighborhood_left = neighborhood[0]
        neighborhood_right = neighborhood[1]
        neighborhood_top = neighborhood[2].transpose()
        neighborhood_bottom = neighborhood[3].transpose()
        har_left = mh.features.pftas(neighborhood_left)
        har_right = mh.features.pftas(neighborhood_right)
        har_top = mh.features.pftas(neighborhood_top)
        har_bottom = mh.features.pftas(neighborhood_bottom)

        dist_x = np.linalg.norm(har_left - har_right)
        dist_y = np.linalg.norm(har_top - har_bottom)

        return [dist_x + dist_y]
        #return list(har_left - har_right) + list(har_top - har_bottom)

    image = img_as_ubyte(rescale(image, SCALE))
    neighborhoods = extract_semi_localities_for_each_pixel(image)
    features = map(map_feature_for_pixel, neighborhoods)
    features = np.array(features).reshape((image.shape[0], image.shape[1], len(features[0])))

    return segmentation_clustering(features, add_locality=False)

def gabor(image):
    ALPHA = 0.25
    BETA = 0.5

    def get_gabor_kernels():
        kernels = []
        angles = [ radians(i) for i in [ 0, 30, 60, 90, 120, 150 ] ]
        #angles = [ radians(i) for i in [ 0, 45, 90, 135 ] ]
        frequencies = [ sqrt(2) * i for i in [ 1, 2, 4, 8, 16, 32, 128, 256 ] ] # Jahn
        base = [ 1, 2, 4, 8, 16, 32, 64, 128 ]
        frequencies = [ 0.25 - (2 ** (i-0.5)) / 1024 for i in base ] + [ 0.25 + (2 ** (i-0.5)) / 1024 for i in base ]# Zhang
        # frequencies = [ 0.1, 0.25, 0.4, 1]
        for angle in angles:
            for frequency in frequencies:
                kernels.append((frequency, angle))
        return kernels

    def apply_gabor_kernels(image, kernels):
        number_of_kernels = len(kernels)
        third_dimension = 1 if len(image.shape) != 3 else image.shape[2]
        features = np.zeros((image.shape[0], image.shape[1], number_of_kernels * third_dimension))
        for i, kernel in enumerate(kernels):
            for j in range(third_dimension):
                feature_index = i*third_dimension + j
                image_dim = image if third_dimension == 1 else image[:, :, j]
                features[:, :, feature_index] = gabor_filter(image_dim, kernel[0], kernel[1])[0]
                features[:, :, feature_index] = fh.normalize(features[:, :, feature_index])
        return features

    def apply_nonlinearity(features):
        return np.tanh(np.multiply(features, ALPHA))

    def apply_blur(features, kernels):
        for i, kernel in enumerate(kernels):
            sigma = (1024 / kernel[0]) * BETA
            features[:, :, i] = gaussian_filter(features[:, :, i], sigma)
            features[:, :, i] = fh.normalize(features[:, :, i])
        return features

    image = img_as_ubyte(rescale(image, 0.25))
    kernels = get_gabor_kernels()
    features = apply_gabor_kernels(image, kernels)
    features = apply_nonlinearity(features)
    features = apply_blur(features, kernels)

    return segmentation_clustering(features, add_locality=False)


def segmentation_clustering(features, normalize_features=True, use_pca=True, add_locality=True):
    def add_locality_info(features):
        locality = np.zeros((features.shape[0], features.shape[1], 2))
        for i in range(features.shape[0]):
            for j in range(features.shape[1]):
                locality[i, j, 0] = i / features.shape[0]
                locality[i, j, 1] = j / features.shape[1]
        return np.concatenate((locality, features), 2)

    def reduce_feature_complexity(features):
        pca = PCA(n_components=6)
        #pca = RandomizedPCA(n_components=6)
        reduced = pca.fit_transform(features)
        return reduced

    def do_clustering(features):
        centroids, labels = kmeans2(features, 8)
        return labels

    if add_locality:
        features = add_locality_info(features)
    flattened = features.reshape((features.shape[0] * features.shape[1], features.shape[2]) )
    if normalize_features:
        flattened = fh.normalize(flattened)
    if use_pca and flattened.shape[1] > 6:
        flattened = reduce_feature_complexity(flattened)

    flat_labels = do_clustering(flattened)

    return flat_labels.reshape((features.shape[0], features.shape[1]))

