from collections import Counter
import numpy as np


def euclidean_distance(x1, x2):
    """
        Calculate the Euclidean distance between two points
        :param x1: np.array
        :param x2: np.array
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

def inference(X, y, k, x_new):
    """
        Predict the label of a new point
        :param X: np.array
        :param y: np.array
        :param k (odd value): int
        :param x_new: np.array
    """

    distances = {}

    for i in range(len(X)):
        distance = euclidean_distance(X[i], x_new)
        distances[i] = distance #assumes the X and y have the same mapping

    # Select the k-nearest points position
    sorted_distances = sorted(distances.items(), key=lambda x: x[1])
    k_nearest = sorted_distances[:k]

    # Get the labels of the k-nearest points
    k_nearest_labels = [y[i] for i in k_nearest.keys()]

    common_labels = Counter(k_nearest_labels).most_common(1)

    return common_labels[0][0]



