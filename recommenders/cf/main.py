from .collaborative_filtering import CF
from .collaborative_filtering import similarity_pearson
from .collaborative_filtering import similarity_cosine
from random import shuffle
import csv


# Read dataset
def read_dataset(path):
    dataset = []
    with open(path, newline='') as file:
        reader = csv.reader(file, delimiter='\t', quotechar='|')
        for row in reader:
            (user_id, movie_id, rating) = (row[0], row[1], float(row[2]))
            dataset.append((user_id, movie_id, rating))

    return dataset


# Write dataset
def write_dataset(dataset, path):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for instance in dataset:
            writer.writerow(instance)

    return dataset


# Shuffling training data once to use cross-validation without shuffling
# so that we can use previously generated models and save time
'''dataset = read_dataset('../../data/training.dat')
shuffle(dataset)
write_dataset(dataset, '../../data/shuffled.training.dat')'''


# Cross validation
# k doesn't affect generating the similarity files, so it can be set to any value
rmse = CF.cross_validate('../../data/shuffled.training.dat',
                         k=1,
                         similarity=similarity_cosine,
                         n_folds=10,
                         models_directory='../../models',
                         load_models=True)
print("RMSE: %.3f" % rmse)

# Prediction
'''ibcf = CF(k=2, similarity=similarity_pearson)
ibcf.load_dataset('../../data/lecture.training.dat')
ibcf.train(item_based=True)
ibcf.save_model('../../models/model_sim{}.csv'.format(ibcf.similarity.__name__))
ibcf.load_model('../../models/model_sim{}.csv'.format(ibcf.similarity.__name__))
ibcf.predict_missing_ratings(item_based=True)
predictions = ibcf.predict_for_set_with_path('../../data/lecture.predict.dat')
print(predictions)'''