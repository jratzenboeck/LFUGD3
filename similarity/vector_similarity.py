from math import sqrt
import math


# Compute Euclidean distance between two vectors of numbers
def euclidean_distance(x, y):
    if len(x) != len(y):
        return math.inf

    distance = sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))
    return distance


# Compute a similarity score between two vectors of numbers based on the Euclidean distance
def euclidean_similarity(x, y):
    similarity = 1.0 / (1.0 + euclidean_distance(x, y))
    return similarity
