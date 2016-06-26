from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from decimal import Decimal
import csv
import time


def load_training_dataset(path):
    key_attribute = []
    input_attributes = []
    output_attribute = []
    with open(path, newline='') as file:
        reader = csv.reader(file, delimiter='\t', quotechar='|')
        next(reader, None)  # skip the header
        for row in reader:
            (cf_svd, content_item, cf_item, userId_movieId, actual_rating, content_user, cf_user) = \
                (float(row[0]), float(row[1]), float(row[2]), row[3], float(row[4]), float(row[5]), float(row[6]))
            key_attribute.append(userId_movieId)
            input_attributes.append([cf_item, cf_user, cf_svd, content_item])
            output_attribute.append(actual_rating)

    return key_attribute, input_attributes, output_attribute


def load_prediction_dataset(path):
    key_attribute = []
    input_attributes = []
    with open(path, newline='') as file:
        reader = csv.reader(file, delimiter='\t', quotechar='|')
        next(reader, None)  # skip the header
        for row in reader:
            (cf_svd, content_item, cf_item, userId_movieId, content_user, cf_user) = \
                (float(row[0]), float(row[1]), float(row[2]), row[3], float(row[4]), float(row[5]))
            key_attribute.append(userId_movieId)
            input_attributes.append([cf_item, cf_user, cf_svd, content_item])

    return key_attribute, input_attributes


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
    training_key, training_input, training_output = load_training_dataset('data/reg_input/reg.train.dat')
    prediction_key, prediction_input = load_prediction_dataset('data/reg_input/reg.predict.dat')

    # Regression Cross Validation
    # for k in [1, 5, 10, 20, 40, 80, 160, 320, 330, 430, 530, 640, 650, 750, 850, 950, 1280]:
    #     rmse = knn_cross_validate(dataset_input=training_input, dataset_output=training_output, k=k, n_folds=10)
    #     print('RMSE = %.4f for K = %d' % (rmse, k))

    #Regression Prediction
    k = 530
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(training_input, training_output)
    prediction_output = knn.predict(prediction_input)

    # Output
    output_path = '../output/predict_output_%s.csv' % time.strftime('%y_%m_%d_%H_%M_%S')
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(1, len(prediction_key)):
            key = prediction_key[i]
            user_id = key.split('_')[0]
            movie_id = key.split('_')[1]
            predicted_rating = Decimal(prediction_output[i])
            writer.writerow([user_id, movie_id, round(predicted_rating, 3)])

