from math import sqrt
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
import csv


# Return Pearson Correlation Coefficient for items at key1 and key2 in dataset dictionary
def similarity_pearson(dataset, key1, key2):
    # Get mutual items
    mutual_items = {}
    for item in dataset[key1]:
        if item in dataset[key2]:
            mutual_items[item] = 1

    # If there are no ratings in common, return 0
    if len(mutual_items) == 0:
        return 0

    # Sum of the ratings
    sum1 = sum([dataset[key1][item] for item in mutual_items])
    sum2 = sum([dataset[key2][item] for item in mutual_items])

    # Sum of the rating squares
    sum1_squares = sum([pow(dataset[key1][item], 2) for item in mutual_items])
    sum2_squares = sum([pow(dataset[key2][item], 2) for item in mutual_items])

    # Sum of the products
    sum_product = sum([dataset[key1][item] * dataset[key2][item] for item in mutual_items])

    # Calculate r (Pearson score)
    numerator = sum_product - (sum1 * sum2 / len(mutual_items))
    denominator = sqrt(
        (sum1_squares - pow(sum1, 2) / len(mutual_items)) * (sum2_squares - pow(sum2, 2) / len(mutual_items)))
    if denominator == 0:
        return 0

    score = numerator / denominator
    # Normalize score to be between 0 and 1
    score = (score - (-1)) / (1 - (-1))
    return score


# Return Cosine Similarity for items at key1 and key2 in dataset dictionary
def similarity_cosine(dataset, key1, key2):
    # Get mutual items
    mutual_items = {}
    for item in dataset[key1]:
        if item in dataset[key2]:
            mutual_items[item] = 1

    # If there are no ratings in common, return 0
    if len(mutual_items) == 0:
        return 0

    # Sum of the rating squares
    sum1_squares = sum([pow(dataset[key1][item], 2) for item in dataset[key1]])
    sum2_squares = sum([pow(dataset[key2][item], 2) for item in dataset[key2]])

    # Sum of the products
    sum_product = sum([dataset[key1][item] * dataset[key2][item] for item in mutual_items])

    # Calculate score
    numerator = sum_product
    denominator = sqrt(sum1_squares) * sqrt(sum2_squares)
    if denominator == 0:
        return 0

    score = numerator / denominator
    return score


class IBCF:

    # Transform rows into columns and vice versa
    # Transform dataset from user-centric to item-centric and vice-versa
    # Returns the transformed dataset
    @staticmethod
    def __transform_dataset(dataset):
        transformed_dataset = {}
        for item_i in dataset:
            for item_j in dataset[item_i]:
                transformed_dataset.setdefault(item_j, {})
                transformed_dataset[item_j][item_i] = dataset[item_i][item_j]

        return transformed_dataset

    # Cross validate item-based collaborative filtering
    @staticmethod
    def cross_validate(path,
                       k=30,
                       similarity=similarity_cosine,
                       n_folds=10,
                       models_directory='',
                       load_models=False,
                       external_similarities=None,
                       alpha=0.5):
        rmse = 0

        # Read dataset and split it
        dataset = []
        with open(path, newline='') as file:
            reader = csv.reader(file, delimiter='\t', quotechar='|')
            for row in reader:
                (user_id, movie_id, rating) = (row[0], row[1], float(row[2]))
                dataset.append((user_id, movie_id, rating))

        # Use shuffle=True to shuffle the folds selection
        # folds = KFold(n=len(dataset), n_folds=n_folds, shuffle=True)
        folds = KFold(n=len(dataset), n_folds=n_folds, shuffle=False)

        fold = 0
        for train_indices, test_indices in folds:
            fold += 1
            training_set = [dataset[i] for i in train_indices]
            test_set = [dataset[i] for i in test_indices]

            ibcf = IBCF(k, similarity)
            ibcf.set_dataset(training_set)
            if load_models:
                ibcf.load_model(models_directory + '/model_f{}_sim{}.csv'.format(fold, similarity.__name__))
            else:
                ibcf.train()
                ibcf.save_model(models_directory + '/model_f{}_sim{}.csv'.format(fold, similarity.__name__))
            ibcf.predict_missing_ratings()
            # Inject the external similarities if they were provided
            if external_similarities is not None:
                ibcf.modify_items_similarity(external_similarities, alpha=alpha)
            predict_set = ibcf.predict_for_set(test_set)

            rmse += mean_squared_error([rec[2] for rec in test_set], [rec[2] for rec in predict_set]) ** 0.5

        return rmse / float(n_folds)

    # Constructor
    def __init__(self, k=25, similarity=similarity_pearson):
        self.__dataset = {}
        self.__items_similarity = {}
        self.__mean_user_ratings = {}
        self.k = k
        self.similarity = similarity

    # Normalize dataset by subtracting mean user ratings
    def __normalize_dataset(self):
        for user in self.__dataset:
            for item in self.__dataset[user]:
                self.__dataset[user][item] -= self.__mean_user_ratings[user]

    # Denormalize dataset by adding mean user ratings
    def __denormalize_dataset(self):
        for user in self.__dataset:
            for item in self.__dataset[user]:
                self.__dataset[user][item] += self.__mean_user_ratings[user]

    # Set the dataset from a tuples list
    # The tuples must be in the following format (user, item, rating)
    def set_dataset(self, dataset):
        self.__dataset = {}
        for (user_id, movie_id, rating) in dataset:
            self.__dataset.setdefault(user_id, {})
            self.__dataset[user_id][movie_id] = float(rating)

        # Set mean user ratings
        self.__mean_user_ratings = {}
        for user in self.__dataset:
            self.__mean_user_ratings[user] = sum(self.__dataset[user].values()) / len(self.__dataset[user].values())

    # Load dataset from a csv file that is formatted as tuples
    # The tuples must be in the following format (user, item, rating)
    def load_dataset(self, path):
        dataset = []
        with open(path, newline='') as file:
            reader = csv.reader(file, delimiter='\t', quotechar='|')
            for row in reader:
                (user_id, movie_id, rating) = (row[0], row[1], float(row[2]))
                dataset.append((user_id, movie_id, rating))

        self.set_dataset(dataset)
        return dataset

    # Calculate pairwise item-item similarity scores
    def calculate_pairwise_items_similarity(self):
        self.__items_similarity = {}

        # Invert the dataset to be item-centric
        dataset_item_centric = IBCF.__transform_dataset(self.__dataset)
        c = 0
        for item_i in dataset_item_centric:
            # Status updates for large datasets
            c += 1
            if c % 100 == 0:
                print("Items_Similarity: %d / %d" % (c, len(dataset_item_centric)))

            self.__items_similarity.setdefault(item_i, {})
            # Calculate how similar this item to other items
            for item_j in dataset_item_centric:
                # If the similarity is calculated before, don't calculate it again
                if item_j in self.__items_similarity:
                    if item_i in self.__items_similarity[item_j]:
                        self.__items_similarity[item_i][item_j] = self.__items_similarity[item_j][item_i]
                        continue

                # If item_i is item_j set the similarity to one
                if item_i == item_j:
                    self.__items_similarity[item_i][item_j] = 1
                    continue

                self.__items_similarity[item_i][item_j] = self.similarity(dataset_item_centric, item_i, item_j)

    # Train the model
    # This method is simply calling calculate_items_similarity
    def train(self):
        self.calculate_pairwise_items_similarity()

    # Save the trained model into a CSV file as tuples
    # The tuples are in the following format (item01, item02, similarity_score)
    # The trained model is the pairwise items similarity
    def save_model(self, path):
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for item_i in self.__items_similarity:
                for item_j in self.__items_similarity[item_i]:
                    writer.writerow([item_i, item_j, self.__items_similarity[item_i][item_j]])

    # Load a trained model from a CSV file that is formatted as tuples
    # The tuples must be in the following format (item01, item02, similarity_score)
    # The trained model is the pairwise items similarity
    def load_model(self, path):
        self.__items_similarity = {}
        with open(path, newline='') as file:
            reader = csv.reader(file, delimiter='\t', quotechar='|')
            for row in reader:
                item_i = row[0]
                item_j = row[1]
                similarity = float(row[2])

                self.__items_similarity.setdefault(item_i, {})
                self.__items_similarity[item_i][item_j] = similarity

    # Predict missing ratings in the dataset
    def predict_missing_ratings(self):
        # For each item in items_similarity, sort its similar items
        # according to the similarity scores
        items_similarity_sorted = {}
        for item in self.__items_similarity:
            items_similarity_sorted[item] = sorted(self.__items_similarity[item].items(),
                                                   key=lambda rec: rec[1],
                                                   reverse=True)

        # Loop over all users
        c = 0
        for user in self.__dataset:
            # Status updates for large datasets
            c += 1
            if c % 100 == 0:
                print("Missing_Ratings: %d / %d" % (c, len(self.__dataset)))

            # Loop over all items
            for item in items_similarity_sorted:
                # Ignore if this user has already rated this item
                if item in self.__dataset[user]:
                    continue

                neighbours = 0
                weighted_similarity = 0
                similarities_sum = 0
                # Loop over similar items
                for (similar_item, similarity) in items_similarity_sorted[item]:
                    # Check if the similar item is the item itself
                    if similar_item == item:
                        continue

                    # We are only interested in items that have been rated by the user
                    if similar_item not in self.__dataset[user]:
                        continue

                    neighbours += 1
                    # We are only interested in the k nearest neighbours
                    if neighbours > self.k:
                        break

                    weighted_similarity += similarity * self.__dataset[user][similar_item]
                    similarities_sum += similarity

                if similarities_sum > 0:
                    self.__dataset[user][item] = weighted_similarity / similarities_sum

    # Predict how the user would rate the item in each tuple in the list
    # The tuples must be in one of the following formats (user, item) or (user, item, rating)
    # If the rating is provided it will be overwritten
    def predict_for_set(self, predict_set):
        result = []
        # Remove the rating if it is already provided
        predict_set = [(rec[0], rec[1]) for rec in predict_set]
        for (user, item) in predict_set:
            rating = 0
            if user in self.__dataset:
                if item in self.__dataset[user]:
                    rating = self.__dataset[user][item]
                else:
                    # Set average user ratings in case of any problem
                    rating = self.__mean_user_ratings[user]

            # Post-process rating in case of any problems
            if rating < 1:
                rating = 1

            if rating > 5:
                rating = 5

            result.append((user, item, rating))

        return result

    # Load dataset from a csv file and predicts how the user would rate the item in each tuple in the file
    # The tuples must be in the following format (user, item)
    def predict_for_set_with_path(self, path):
        # Read dataset
        dataset = []
        with open(path, newline='') as file:
            reader = csv.reader(file, delimiter='\t', quotechar='|')
            for row in reader:
                (user_id, movie_id) = (row[0], row[1])
                dataset.append((user_id, movie_id))

        # Predict
        return self.predict_for_set(dataset)

    # Predict how a user would rate an item
    def predict(self, user, item):
        rating = 0
        if user in self.__dataset:
            if item in self.__dataset[user]:
                rating = self.__dataset[user][item]
            else:
                # Set average user ratings in case of any problem
                rating = self.__mean_user_ratings[user]

        # Post-process rating in case of any problems
        if rating < 1:
            rating = 1

        if rating > 5:
            rating = 5

        return rating

    # Modify items similarity by external similarities
    # These similarities could be computed using other resources like the text
    # These similarities should be provided in a dictionary of item-item keys
    # The modification is based on the weighted sum
    # sim = ((1 - alpha) * sim) + (alpha * external_sim)
    # items_similarity should be computed before calling this function
    def modify_items_similarity(self, external_similarities, alpha=0.5):
        for item_i in self.__items_similarity:
            # If item_i doesn't have similarity scores in external_similarities, skip it
            if item_i not in external_similarities:
                continue

            for item_j in self.__items_similarity[item_i]:
                # If item_j doesn't have similarity score with item_i in external_similarities, skip it
                if item_j not in external_similarities[item_i]:
                    continue

                self.__items_similarity[item_i][item_j] = ((1 - alpha) * self.__items_similarity[item_i][item_j]) + \
                                                          (alpha * external_similarities[item_i][item_j])
