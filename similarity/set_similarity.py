

# Compute Dice's Coefficient between two lists or sets of strings (words)
# dice_coefficient = 2(len(intersection(x, y))) / len(x) + len(y)
def dice_similarity(x, y):
    if not len(x) or not len(y):
        return 0.0

    set_x = set(x)
    set_y = set(y)
    overlap = len(set_x & set_y)
    dice_coefficient = (2.0 * overlap) / (len(set_x) + len(set_y))

    return dice_coefficient

