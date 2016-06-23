import numpy as np
import csv
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

def read_reg_train_data(reg_train_path):
    actual_rating_set = []
    svd_rating_set = []
    cf_item_set = []

    with open(reg_train_path, newline='') as file:
        reader = csv.reader(file, delimiter='\t', quotechar='|')
        next(reader, None)  # skip the header
        for row in reader:
            actual_rating_set.append(float(row[0]))
            svd_rating_set.append(float(row[1]))
            cf_item_set.append(float(row[2]))

    return (actual_rating_set, svd_rating_set, cf_item_set)

def calc_linear_regression(reg_training_path):
    actual_ratings, svd_ratings, cf_item_ratings = read_reg_train_data(reg_training_path)
    rating_sets = [svd_ratings, cf_item_ratings]

    for method in rating_sets:
        X = pd.DataFrame(method)
        neigh = KNeighborsRegressor(n_neighbors=2)
        neigh.fit(X, actual_ratings)
        predicted_values = neigh.predict(X)
        rmse = np.sqrt(np.mean((actual_ratings - predicted_values) ** 2))
        print('RMSE:', rmse)

if __name__ == '__main__':
    reg_training_path = 'data/reg_input/reg.train'
    calc_linear_regression(reg_training_path)