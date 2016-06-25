from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
import csv
import time


def load_dataset(path):
    input_attributes = []
    output_attribute = []
    with open(path, newline='') as file:
        reader = csv.reader(file, delimiter='\t', quotechar='|')
        next(reader, None)  # skip the header
        for row in reader:
            (cf_item, cf_user, cf_svd, content_item, actual_rating) = (float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[5]))
            input_attributes.append([cf_item, cf_user, cf_svd, content_item])
            output_attribute.append(actual_rating)

    return input_attributes, output_attribute


def knn_cross_validate(dataset_input, dataset_output, k, n_folds=10):
    rmse = 0
    folds = KFold(n=len(dataset_input), n_folds=n_folds, shuffle=False)

    fold = 0
    for train_indices, test_indices in folds:
        fold += 1
        train_input = [dataset_input[i] for i in train_indices]
        train_output = [dataset_output[i] for i in train_indices]
        test_input = [dataset_input[i] for i in test_indices]
        test_output = [dataset_output[i] for i in test_indices]

        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_input, train_output)
        test_predicted = knn.predict(test_input)

        rmse += mean_squared_error(test_output, test_predicted) ** 0.5
        print("Fold (%d) finished with accumulated RMSE of (%f) (%s)" % (fold, rmse, time.strftime('%y_%m_%d_%H_%M_%S')))

    return rmse / float(n_folds)


if __name__ == '__main__':
    ds_input, ds_output = load_dataset('data/reg_input/reg.train')
    for k in [1, 5, 10, 20, 40, 80, 160]:
        rmse = knn_cross_validate(dataset_input=ds_input, dataset_output=ds_output, k=k, n_folds=10)
        print('RMSE = %.4f for K = %d' % (rmse, k))

