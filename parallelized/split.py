import csv
from random import shuffle
from sklearn.cross_validation import train_test_split


def read_trainingset(path):
    dataset = []
    with open(path, newline='') as file:
        reader = csv.reader(file, delimiter='\t', quotechar='|')
        for row in reader:
            (user_id, movie_id, rating) = (row[0], row[1], float(row[2]))
            dataset.append((user_id, movie_id, rating))
    return dataset


def get_shuffled_trainingset(path):
    dataset = read_trainingset(path)
    shuffle(dataset)
    return dataset


def get_splitted_datasets(path):
    dataset = read_trainingset(path)
    #count_70_pct = int(len(dataset) * 0.7)
    #rec_train = dataset[:count_70_pct]
    #rec_test = dataset[len(dataset) - count_70_pct:]
    rec_train, rec_test = train_test_split(dataset, test_size=0.3, random_state=42)
    return (rec_train, rec_test)


def write_dataset(path, dataset):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for instance in dataset:
            writer.writerow(instance)
    return dataset


pathToTrainingSet = '../data/training.dat'
number_of_splits = 4
for i in range(1, number_of_splits):
    rec_train, rec_test = get_splitted_datasets(pathToTrainingSet)
    print('Dataset split')
    write_dataset('data/rec_input/' + str(i) + '.rec.train', rec_train)
    write_dataset('data/rec_test/' + str(i) + '.rec.test', rec_test)
    print('Wrote training and test file ' + str(i))
