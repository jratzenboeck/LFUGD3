from data_structure.user import User
from similarity.vector_similarity import euclidean_similarity


class UserSimilarity:

    # Normalize weights to sum up to one
    def normalize_weights(self):
        weights_sum = self.__gender_weight + \
                      self.__age_weight + \
                      self.__job_weight + \
                      self.__zip_code_weight
        if weights_sum == 0:
            weights_sum = 1

        self.__gender_weight /= weights_sum
        self.__age_weight /= weights_sum
        self.__job_weight /= weights_sum
        self.__zip_code_weight /= weights_sum

    def __init__(self,
                 gender_weight=1,
                 age_weight=1,
                 job_weight=1,
                 zip_code_weight=1):

        self.__gender_weight = gender_weight
        self.__age_weight = age_weight
        self.__job_weight = job_weight
        self.__zip_code_weight = zip_code_weight

        self.normalize_weights()

    # Calculate the similarity between two users based on the gender attribute
    @staticmethod
    def similarity_gender(user_1, user_2):
        similarity = 0
        if user_1.gender == user_2.gender:
            similarity = 1

        return similarity

    # Calculate the similarity between two users based on the age attribute
    @staticmethod
    def similarity_age(user_1, user_2):
        similarity = euclidean_similarity([user_1.age], [user_2.age])
        return similarity

    # Calculate the similarity between two users based on the job attribute
    @staticmethod
    def similarity_job(user_1, user_2):
        similarity = 0
        if user_1.job == user_2.job:
            similarity = 1

        return similarity

    # Calculate the similarity between two users based on the gender attribute
    @staticmethod
    def similarity_zip_code(user_1, user_2):
        similarity = euclidean_similarity([user_1.zip_code], [user_2.zip_code])
        return similarity

    # Calculate the similarity between two movies by summing all sub-similarities
    def similarity(self, user_1, user_2):
        similarity = 0
        similarity += self.__gender_weight * UserSimilarity.similarity_gender(user_1, user_2)
        similarity += self.__age_weight * UserSimilarity.similarity_age(user_1, user_2)
        similarity += self.__job_weight * UserSimilarity.similarity_job(user_1, user_2)
        similarity += self.__zip_code_weight * UserSimilarity.similarity_zip_code(user_1, user_2)

        return similarity

    # Calculate pairwise similarities between each user and all the others
    # The users should be passed in a dictionary
    # The result is a two dimensional dictionary, it uses the same keys passed in the input dictionary
    def pairwise_similarity(self, users):
        result = {}
        c = 0
        for user_i in users:
            # Status updates for large datasets
            c += 1
            if c % 100 == 0:
                print("Users_Similarity: %d / %d" % (c, len(users)))

            result.setdefault(user_i, {})
            for user_j in users:
                # If the similarity is calculated before, don't calculate it again
                if user_j in result:
                    if user_i in result[user_j]:
                        result[user_i][user_j] = result[user_j][user_i]
                        continue

                # If user_i is user_j, set the similarity to one
                if user_i == user_j:
                    result[user_i][user_j] = 1
                    continue

                result[user_i][user_j] = self.similarity(users[user_i], users[user_j])

        return result
