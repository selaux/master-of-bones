import numpy as np

def normalize(features):
    minimum = features.min()
    features = features - minimum
    maximum = features.max()
    if maximum != 0:
        return np.divide(features, features.max())
    return features

def normalize_each(features):
    number_of_features = features.shape[1] if len(features.shape) > 1 else 1

    if number_of_features == 1:
        return normalize(features)
    else:
        for i in range(features.shape[1]):
            features[:,i] = normalize(features[:,i])
        return features
