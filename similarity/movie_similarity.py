from data_structure.movie import Movie
from similarity.vector_similarity import euclidean_similarity
from similarity.set_similarity import dice_similarity


class MovieSimilarity:

    # Normalize weights to sum up to one
    def normalize_weights(self):
        weights_sum = self.__year_weight + \
                      self.__rating_weight + \
                      self.__genres_weight + \
                      self.__stakeholders_weight + \
                      self.__description_weight + \
                      self.__other_features_weight
        if weights_sum == 0:
            weights_sum = 1

        self.__year_weight /= weights_sum
        self.__rating_weight /= weights_sum
        self.__genres_weight /= weights_sum
        self.__stakeholders_weight /= weights_sum
        self.__description_weight /= weights_sum
        self.__other_features_weight /= weights_sum

    def __init__(self,
                 year_weight=1,
                 rating_weight=1,
                 genres_weight=1,
                 stakeholders_weight=1,
                 description_weight=1,
                 other_features_weight=1):

        self.__year_weight = year_weight
        self.__rating_weight = rating_weight
        self.__genres_weight = genres_weight
        self.__stakeholders_weight = stakeholders_weight
        self.__description_weight = description_weight
        self.__other_features_weight = other_features_weight

        self.normalize_weights()

    # Extract rating attributes as a vector (list) of numbers
    @staticmethod
    def __get_rating_vector(movie):
        vector = [movie.imdbRating,
                  movie.tomatoRating,
                  movie.tomatoUserRating,
                  movie.popularity,
                  movie.tomatoMeter,
                  movie.tomatoUserMeter]
        return vector

    # Calculate the similarity between two movies based on the year attribute
    @staticmethod
    def similarity_year(movie_1, movie_2):
        similarity = euclidean_similarity([movie_1.year], [movie_2.year])
        return similarity

    # Calculate the similarity between two movies based on the rating-related attributes
    @staticmethod
    def similarity_rating(movie_1, movie_2):
        similarity = euclidean_similarity(MovieSimilarity.__get_rating_vector(movie_1),
                                          MovieSimilarity.__get_rating_vector(movie_2))
        return similarity

    # Calculate the similarity between two movies based on the genres-related attributes
    @staticmethod
    def similarity_genres(movie_1, movie_2):
        similarity = dice_similarity(movie_1.genres, movie_2.genres)
        return similarity

    # Calculate the similarity between two movies based on the stakeholders-related attributes
    @staticmethod
    def similarity_stakeholders(movie_1, movie_2,
                                actors_weight=6,
                                production_companies_weight=3,
                                directors_weight=4,
                                writers_weight=4):
        # Normalize weights
        weights_sum = actors_weight + production_companies_weight + directors_weight + writers_weight
        if weights_sum == 0:
            weights_sum = 1

        actors_weight /= weights_sum
        production_companies_weight /= weights_sum
        directors_weight /= weights_sum
        writers_weight /= weights_sum

        # Calculate similarity
        similarity = 0
        similarity += actors_weight * dice_similarity(movie_1.actors, movie_2.actors)
        # Measuring similarity as two strings doesn't really make sense here, they should lists
        # !!! similarity += production_companies_weight * dice_similarity(movie_1.productionCompanies, movie_2.productionCompanies)
        similarity += directors_weight * dice_similarity(movie_1.directors, movie_2.directors)
        similarity += writers_weight * dice_similarity(movie_1.writers, movie_2.writers)

        return similarity

    # Calculate the similarity between two movies based on the description-related attributes
    @staticmethod
    def similarity_description(movie_1, movie_2, text_similarity,
                               wiki_weight=1,
                               omdb_weight=1,
                               tmdb_weight=1):
        # Normalize weights
        weights_sum = wiki_weight + omdb_weight + tmdb_weight
        if weights_sum == 0:
            weights_sum = 1

        wiki_weight /= weights_sum
        omdb_weight /= weights_sum
        tmdb_weight /= weights_sum

        # Calculate similarity
        similarity = 0
        similarity += wiki_weight * text_similarity(movie_1.wikiDescription, movie_2.wikiDescription)
        similarity += omdb_weight * text_similarity(movie_1.omdbDescription, movie_2.omdbDescription)
        similarity += tmdb_weight * text_similarity(movie_1.tmdbDescription, movie_2.tmdbDescription)

        return similarity

    # Calculate the similarity between two movies based on other (general) attributes
    @staticmethod
    def similarity_other_features(movie_1, movie_2, text_similarity,
                                  awards_weight=1):
        # Normalize weights
        weights_sum = awards_weight
        if weights_sum == 0:
            weights_sum = 1

        awards_weight /= weights_sum

        # Calculate similarity
        similarity = 0
        # Measuring similarity as two strings doesn't really make sense here, they should lists
        # !!! similarity += awards_weight * text_similarity(movie_1.awards, movie_2.awards)

        return similarity

    # Calculate the similarity between two movies by summing all sub-similarities
    def similarity(self, movie_1, movie_2, text_similarity):
        similarity = 0
        similarity += self.__year_weight * MovieSimilarity.similarity_year(movie_1, movie_2)
        similarity += self.__rating_weight * MovieSimilarity.similarity_rating(movie_1, movie_2)
        similarity += self.__genres_weight * MovieSimilarity.similarity_genres(movie_1, movie_2)
        similarity += self.__stakeholders_weight * MovieSimilarity.similarity_stakeholders(movie_1, movie_2)
        similarity += self.__description_weight * MovieSimilarity.similarity_description(movie_1, movie_2, text_similarity)
        similarity += self.__other_features_weight * MovieSimilarity.similarity_other_features(movie_1, movie_2, text_similarity)

        return similarity

    # Calculate pairwise similarities between each movie and all the others
    # The movies should be passed in a dictionary
    # The result is a two dimensional dictionary, it uses the same keys passed in the input dictionary
    def pairwise_similarity(self, movies, text_similarity):
        result = {}
        c = 0
        for movie_i in movies:
            # Status updates for large datasets
            c += 1
            if c % 100 == 0:
                print("Movies_Similarity: %d / %d" % (c, len(movies)))

            result.setdefault(movie_i, {})
            for movie_j in movies:
                # If the similarity is calculated before, don't calculate it again
                if movie_j in result:
                    if movie_i in result[movie_j]:
                        result[movie_i][movie_j] = result[movie_j][movie_i]
                        continue

                # If movie_i is movie_j, set the similarity to one
                if movie_i == movie_j:
                    result[movie_i][movie_j] = 1
                    continue

                result[movie_i][movie_j] = self.similarity(movies[movie_i], movies[movie_j], text_similarity)

        return result

