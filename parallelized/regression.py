import numpy as np
import csv
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import time

def read_reg_train_data(reg_train_path):
    dataset = []

    with open(reg_train_path, newline='') as file:
        reader = csv.reader(file, delimiter='\t', quotechar='|')
        next(reader, None)  # skip the header
        for row in reader:
            (cf_item, cf_user, svd_rating, content_item, actual_rating) = \
                (float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[5]))
            dataset.append((cf_item, cf_user, svd_rating, content_item, actual_rating))

    return dataset

def calc_linear_regression(reg_training_path):
    dataset = read_reg_train_data(reg_training_path)
    rmse = 0
    n_folds = 5
    folds = KFold(n=len(dataset), n_folds=n_folds, shuffle=False)

    fold = 0
    for train_indices, test_indices in folds:
        fold += 1
        training_set = [dataset[i] for i in train_indices]
        test_set = [dataset[i] for i in test_indices]
        training_dataframe = get_data_frame(training_set)
        test_dataframe = get_data_frame(test_set)
        column_names = ['cf_item', 'cf_user', 'svd', 'content_item', 'actual_rating']
        training_dataframe.columns = column_names
        test_dataframe.columns = column_names

        actual_rating_training_column = training_dataframe['actual_rating']
        #actual_rating_test_column = test_dataframe['actual_rating']

        training_dataframe = training_dataframe.drop('actual_rating', axis=1)
        test_dataframe = test_dataframe.drop('actual_rating', axis=1)

        neigh = KNeighborsRegressor(n_neighbors=10)
        #print('Initialized k nearest neighbors regressor with k =', i)
        neigh.fit(training_dataframe, actual_rating_training_column)
        #print('Fit data models')
        predict_set = neigh.predict(test_dataframe)
        print(predict_set)
        rmse += mean_squared_error([rec[4] for rec in test_set], [rec for rec in predict_set]) ** 0.5
        print("Fold (%d) finished with accumulated RMSE of (%f) (%s)" % (fold, rmse, time.strftime('%y_%m_%d_%H_%M_%S')))
    return rmse / float(n_folds)

def get_data_frame(dataset):
    frame = pd.DataFrame(dataset)
    return frame

if __name__ == '__main__':
    reg_training_path = 'data/reg_input/reg.train'
    rmse = calc_linear_regression(reg_training_path)
    print(rmse)