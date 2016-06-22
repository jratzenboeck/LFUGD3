from recommenders.cf.collaborative_filtering import CF
from recommenders.cf.collaborative_filtering import similarity_cosine
from recommenders.cf.collaborative_filtering import similarity_pearson
import csv


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


def recommend_item_content(train_path, predict_path, predict_out_path):
    # Input Parameters
    input = {}
    input['stopwords_path'] = '../stopwords/english'
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

    # Read pairwise movies similarity
    movies_similarity = load_pairwise_similarity('../models/movies_similarity_{}.csv'.format('lsi'))

    # IBCF Prediction
    cf = CF(k=input['k'], similarity=similarity_cosine)
    cf.load_dataset(train_path)
    cf.train(item_based=True)
    # cf.save_model('models/items_model_sim{}.csv'.format(cf.similarity.__name__))
    # Inject the external similarities
    cf.modify_pairwise_similarity(movies_similarity, alpha=input['alpha'])
    cf.predict_missing_ratings(item_based=True)
    predictions = cf.predict_for_set_with_path(predict_path)

    # Output
    with open(predict_out_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for (user, item, rating) in predictions:
            writer.writerow([user, item, rating])


def recommend_user_content(train_path, predict_path, predict_out_path):
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

    # Read pairwise users similarity
    users_similarity = load_pairwise_similarity('models/users_similarity.csv')

    # UBCF Prediction
    cf = CF(k=input['k'], similarity=similarity_pearson)
    cf.load_dataset(train_path)
    cf.train(item_based=False)
    # cf.save_model('models/users_model_sim{}.csv'.format(cf.similarity.__name__))
    # Inject the external similarities
    cf.modify_pairwise_similarity(users_similarity, alpha=input['alpha'])
    cf.predict_missing_ratings(item_based=False)
    predictions = cf.predict_for_set_with_path(predict_path)

    # Output
    with open(predict_out_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for (user, item, rating) in predictions:
            writer.writerow([user, item, rating])


if __name__ == '__main__':
    files_number = 3
    for i in range(files_number):
        train_path = 'data/rec_input/%d.rec.train' % i
        predict_path = 'data/rec_test/%d.rec.test' % i
        predict_out_path_item = 'data/rec_output/%d.content.item.out' % i
        predict_out_path_user = 'data/rec_output/%d.content.user.out' % i
        recommend_item_content(train_path=train_path, predict_path=predict_path, predict_out_path=predict_out_path_item)
        recommend_user_content(train_path=train_path, predict_path=predict_path, predict_out_path=predict_out_path_user)
