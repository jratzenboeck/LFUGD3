from recommenders.cf.collaborative_filtering import CF
from recommenders.cf.collaborative_filtering import similarity_cosine
from recommenders.cf.collaborative_filtering import similarity_pearson
from similarity.text_similarity import TextSimilarity
from similarity.movie_similarity import MovieSimilarity
from similarity.user_similarity import UserSimilarity
from data_structure.movie import Movie
from data_structure.user import User
import json
import csv
import time


# Load movies
def load_movies(path):
    data_json = None
    with open(path) as file:
        data_json = json.load(file)
        file.close()

    movies = {}
    for movie_id, movie_data in data_json.items():
        movies[int(movie_id)] = Movie(movie_id=movie_id, dictionary=movie_data)

    return movies


# Load users
def load_users(path):
    users = {}
    with open(path, newline='') as file:
        reader = csv.reader(file, delimiter='\t', quotechar='|')
        for row in reader:
            (user_id, gender, age, job, zip_code) = (int(row[0]), row[1], int(row[2]), int(row[3]), float(row[4]))
            users[user_id] = User(user_id=user_id)

    return users


# Save pairwise similarity into a CSV file as triples
# The triples are in the following format (key01, key02, similarity_score),
# where key01, key02 are user_ids in user models or movie_ids in item models
def save_pairwise_similarity(movies_similarity, path):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for key_i in movies_similarity:
            for key_j in movies_similarity[key_i]:
                writer.writerow([key_i, key_j, movies_similarity[key_i][key_j]])


# Load pairwise similarity from a CSV file that is formatted as triples
# The triples must be in the following format (key01, key02, similarity_score)
# where key01, key02 are user_ids in user models or movie_ids in item models
def load_pairwise_similarity(path):
    pairwise_similarity = {}
    with open(path, newline='') as file:
        reader = csv.reader(file, delimiter='\t', quotechar='|')
        for row in reader:
            key_i = row[0]
            key_j = row[1]
            similarity = float(row[2])

            pairwise_similarity.setdefault(key_i, {})
            pairwise_similarity[key_i][key_j] = similarity

    return pairwise_similarity


# Build movies similarity model
def build_movies_similarity_models(movies,
                                   stopwords_path=None,
                                   index_path=None,
                                   topics_number=50,
                                   year_weight=1,
                                   rating_weight=1,
                                   genres_weight=1,
                                   stakeholders_weight=1,
                                   description_weight=1,
                                   other_features_weight=1):
    stopwords = []
    if stopwords_path is not None:
        stopwords = open(stopwords_path).read().splitlines()

    index = None
    if index_path is not None:
        index = open(index_path).read().splitlines()

    # Get all texts for text models
    texts = []
    texts += [movie.wikiDescription for movie in movies.values()]
    texts += [movie.omdbDescription for movie in movies.values()]
    texts += [movie.tmdbDescription for movie in movies.values()]

    # Build text models
    text_similarity = TextSimilarity(stopwords=stopwords, index=index, topics_number=topics_number)
    text_similarity.set_document_list(texts)
    text_similarity.build_models()

    # Movie Similarities
    movie_similarity = MovieSimilarity(year_weight=year_weight,
                                       rating_weight=rating_weight,
                                       genres_weight=genres_weight,
                                       stakeholders_weight=stakeholders_weight,
                                       description_weight=description_weight,
                                       other_features_weight=other_features_weight)

    return text_similarity, movie_similarity

def main_ibcf():
    # Input Parameters
    input = {}
    input['stopwords_path'] = 'stopwords/english'
    input['index_path'] = None
    input['topics_number'] = 150
    input['year_weight'] = 5
    input['rating_weight'] = 6
    input['genres_weight'] = 7
    input['stakeholders_weight'] = 4
    input['description_weight'] = 6
    input['other_features_weight'] = 2
    # IMPORTANT PARAMETERS
    input['k'] = 15
    # Alpha is for the external similarity, 1.0: only content-based, 0.0: only item-based
    input['alpha'] = 1.0

    # Load movies
    '''movies = load_movies('data/movies.aggregated.info.json')

    # Build similarity models
    text_similarity, movie_similarity = build_similarity_models(movies,
                                                                stopwords_path=input['stopwords_path'],
                                                                index_path=input['index_path'],
                                                                topics_number=input['topics_number'],
                                                                year_weight=input['year_weight'],
                                                                rating_weight=input['rating_weight'],
                                                                genres_weight=input['genres_weight'],
                                                                stakeholders_weight=input['stakeholders_weight'],
                                                                description_weight=input['description_weight'],
                                                                other_features_weight=input['other_features_weight'])

    # Compute pairwise movies similarity
    # ATTENTION!!!: Input Parameter (text similarity method)
    movies_similarity = movie_similarity.pairwise_similarity(movies, text_similarity=text_similarity.similarity_lsi)
    save_pairwise_movies_similarity(movies_similarity, path='models/movies_similarity_{}.csv'.format('lsi'))'''
    movies_similarity = load_pairwise_similarity('models/movies_similarity_{}.csv'.format('lsi'))

    # IBCF Cross Validation
    rmse = CF.cross_validate('data/shuffled.training.dat',
                             item_based=True,
                             k=input['k'],
                             similarity=similarity_cosine,
                             n_folds=10,
                             models_directory='models',
                             load_models=True,
                             external_similarities=movies_similarity,
                             alpha=input['alpha'])

    # Output
    print("RMSE: %.3f" % rmse)

    input['RMSE'] = rmse
    output_path = 'output/output_%s.csv' % time.strftime('%y_%m_%d_%H_%M_%S')
    with open(output_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=input.keys())
        writer.writeheader()
        writer.writerow(input)

    # IBCF Prediction
    '''cf = CF(k=input['k'], similarity=similarity_cosine)
    cf.load_dataset('data/training.dat')
    cf.train(item_based=True)
    cf.save_model('models/items_model_sim{}.csv'.format(cf.similarity.__name__))
    # Inject the external similarities
    cf.modify_items_similarity(movies_similarity, alpha=input['alpha'])
    cf.predict_missing_ratings(item_based=True)
    predictions = cf.predict_for_set_with_path('data/predict.dat')
    print(predictions)

    # Output
    output_path = 'output/predict_output_%s.csv' % time.strftime('%y_%m_%d_%H_%M_%S')
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for (user, item, rating) in predictions:
            writer.writerow([user, item, rating])'''


def main_ubcf():
    # Input Parameters
    input = {}
    input['gender_weight'] = 5
    input['age_weight'] = 4
    input['job_weight'] = 3
    input['zip_code_weight'] = 2
    # IMPORTANT PARAMETERS
    input['k'] = 80
    # Alpha is for the external similarity, 1.0: only content-based, 0.0: only user-based
    input['alpha'] = 1.0

    # Load users
    users = load_users('data/users.dat')

    # Build similarity models
    user_similarity = UserSimilarity(gender_weight=input['gender_weight'],
                                     age_weight=input['age_weight'],
                                     job_weight=input['job_weight'],
                                     zip_code_weight=input['zip_code_weight'])

    # Compute pairwise users similarity
    users_similarity = user_similarity.pairwise_similarity(users)
    save_pairwise_similarity(users_similarity, path='models/users_similarity.csv')
    # users_similarity = load_pairwise_similarity('models/users_similarity.csv')

    # UBCF Cross Validation
    rmse = CF.cross_validate('data/shuffled.training.dat',
                             item_based=False,
                             k=input['k'],
                             similarity=similarity_pearson,
                             n_folds=10,
                             models_directory='models',
                             load_models=True,
                             external_similarities=users_similarity,
                             alpha=input['alpha'])

    # Output
    print("RMSE: %.3f" % rmse)

    input['RMSE'] = rmse
    output_path = 'output/output_%s.csv' % time.strftime('%y_%m_%d_%H_%M_%S')
    with open(output_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=input.keys())
        writer.writeheader()
        writer.writerow(input)

    # UBCF Prediction
    '''cf = CF(k=input['k'], similarity=similarity_cosine)
    cf.load_dataset('data/training.dat')
    cf.train(item_based=False)
    cf.save_model('models/items_model_sim{}.csv'.format(cf.similarity.__name__))
    # Inject the external similarities
    cf.modify_items_similarity(movies_similarity, alpha=input['alpha'])
    cf.predict_missing_ratings(item_based=False)
    predictions = cf.predict_for_set_with_path('data/predict.dat')
    print(predictions)

    # Output
    output_path = 'output/predict_output_%s.csv' % time.strftime('%y_%m_%d_%H_%M_%S')
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for (user, item, rating) in predictions:
            writer.writerow([user, item, rating])'''


if __name__ == '__main__':
    main_ubcf()