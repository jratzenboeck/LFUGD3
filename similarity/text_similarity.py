from multiprocessing import cpu_count
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.models import LdaMulticore
from gensim.similarities import MatrixSimilarity
from gensim.matutils import cossim


class TextSimilarity:

    # Pre-process a text document
    def __preprocess_text_document(self, text):
        # clean and tokenize document string
        tokens = self.__tokenizer.tokenize(text.lower())

        # remove stop words from tokens
        tokens_stopped = [token for token in tokens if token not in self.stop_words]

        # remove any word that is not present in the index
        if self.index is not None:
            tokens_stopped = [token for token in tokens_stopped if token in self.index]

        # stem tokens
        tokens_stemmed = [self.__stemmer.stem(token) for token in tokens_stopped]

        return tokens_stemmed

    def __init__(self, stopwords=[], index=None, topics_number=200):
        self.__tokenizer = RegexpTokenizer(r'\w+')
        self.__stemmer = PorterStemmer()
        self.__document_list = []
        self.__dictionary = None
        self.__corpus = None
        self.__model_tfidf = None
        self.__model_lsi = None
        self.__model_lda = None
        self.topics_number = topics_number
        self.stop_words = stopwords
        self.index = index

        if self.index is not None:
            self.index = [term.lower() for term in self.index]

    def set_document_list(self, document_list):
        self.__document_list = document_list

    # Build text transformation models, using the documents_list provided
    def build_models(self):
        documents_tokenized = []
        for doc in self.__document_list:
            processed_document = self.__preprocess_text_document(doc)
            if len(processed_document) > 0:
                documents_tokenized.append(processed_document)

        # if the documents get filtered out completely (by the intersection with the index),
        # add some random word to prevent exceptions
        if len(documents_tokenized) <= 0:
            documents_tokenized.append(['None'])

        # turn tokenized documents into a id <-> term dictionary
        self.__dictionary = Dictionary(documents_tokenized)

        # convert tokenized documents into a document-term matrix
        self.__corpus = [self.__dictionary.doc2bow(document) for document in documents_tokenized]

        # generate models
        self.__model_tfidf = TfidfModel(corpus=self.__corpus)
        self.__model_lsi = LsiModel(corpus=self.__corpus,
                                    num_topics=self.topics_number)
        self.__model_lda = LdaMulticore(corpus=self.__corpus,
                                        num_topics=self.topics_number,
                                        id2word=self.__dictionary,
                                        workers=cpu_count() - 1,
                                        chunksize=2000,
                                        passes=1,
                                        batch=False)

    # Calculate the similarity between two texts using TF-IDF model
    def similarity_tfidf(self, text1, text2):
        # convert text into bag of words model
        text1_bow = self.__dictionary.doc2bow(self.__preprocess_text_document(text1))
        text2_bow = self.__dictionary.doc2bow(self.__preprocess_text_document(text2))

        # transform text into the model's domain
        text1_model = self.__model_tfidf[text1_bow]
        text2_model = self.__model_tfidf[text2_bow]

        return cossim(text1_model, text2_model)

    # Calculate the similarity between two texts using LSI model
    def similarity_lsi(self, text1, text2):
        # convert text into bag of words model
        text1_bow = self.__dictionary.doc2bow(self.__preprocess_text_document(text1))
        text2_bow = self.__dictionary.doc2bow(self.__preprocess_text_document(text2))

        # transform text into the model's domain
        text1_model = self.__model_lsi[text1_bow]
        text2_model = self.__model_lsi[text2_bow]

        return cossim(text1_model, text2_model)

    # Calculate the similarity between two texts using LDA model
    def similarity_lda(self, text1, text2):
        # convert text into bag of words model
        text1_bow = self.__dictionary.doc2bow(self.__preprocess_text_document(text1))
        text2_bow = self.__dictionary.doc2bow(self.__preprocess_text_document(text2))

        # transform text into the model's domain
        text1_model = self.__model_lda[text1_bow]
        text2_model = self.__model_lda[text2_bow]

        return cossim(text1_model, text2_model)

    # Calculate pairwise similarities between each text and all the others
    # The texts should be passed in a dictionary
    # The result is a two dimensional dictionary, it uses the same keys passed in the input dictionary
    def pairwise_similarity_tfidf(self, texts):
        result = {}
        for text_i in texts:
            result.setdefault(text_i, {})
            for text_j in texts:
                # If the similarity is calculated before, don't calculate it again
                if text_j in result:
                    if text_i in result[text_j]:
                        result[text_i][text_j] = result[text_j][text_i]
                        continue

                # If text_i is text_j, set the similarity to one
                if text_i == text_j:
                    result[text_i][text_j] = 1
                    continue

                result[text_i][text_j] = self.similarity_tfidf(texts[text_i], texts[text_j])

        return result

    # Calculate pairwise similarities between each text and all the others
    # The texts should be passed in a dictionary
    # The result is a two dimensional dictionary, it uses the same keys passed in the input dictionary
    def pairwise_similarity_lsi(self, texts):
        result = {}
        for text_i in texts:
            result.setdefault(text_i, {})
            for text_j in texts:
                # If the similarity is calculated before, don't calculate it again
                if text_j in result:
                    if text_i in result[text_j]:
                        result[text_i][text_j] = result[text_j][text_i]
                        continue

                # If text_i is text_j, set the similarity to one
                if text_i == text_j:
                    result[text_i][text_j] = 1
                    continue

                result[text_i][text_j] = self.similarity_lsi(texts[text_i], texts[text_j])

        return result

    # Calculate pairwise similarities between each text and all the others
    # The texts should be passed in a dictionary
    # The result is a two dimensional dictionary, it uses the same keys passed in the input dictionary
    def pairwise_similarity_lda(self, texts):
        result = {}
        for text_i in texts:
            result.setdefault(text_i, {})
            for text_j in texts:
                # If the similarity is calculated before, don't calculate it again
                if text_j in result:
                    if text_i in result[text_j]:
                        result[text_i][text_j] = result[text_j][text_i]
                        continue

                # If text_i is text_j, set the similarity to one
                if text_i == text_j:
                    result[text_i][text_j] = 1
                    continue

                result[text_i][text_j] = self.similarity_lda(texts[text_i], texts[text_j])

        return result

    # Calculate the pairwise documents similarity using the documents_list provided
    # The result is a two dimensional array,
    # sorted with the same order of the provided documents_list in each dimension
    def calculate_pairwise_documents_similarity_tfidf(self):
        if self.__model_tfidf is not None:
            return MatrixSimilarity(self.__model_tfidf[self.__corpus])

    # Calculate the pairwise documents similarity using the documents_list provided
    # The result is a two dimensional array,
    # sorted with the same order of the provided documents_list in each dimension
    def calculate_pairwise_documents_similarity_lsi(self):
        if self.__model_lsi is not None:
            return MatrixSimilarity(self.__model_lsi[self.__corpus])

    # Calculate the pairwise documents similarity using the documents_list provided
    # The result is a two dimensional array,
    # sorted with the same order of the provided documents_list in each dimension
    def calculate_pairwise_documents_similarity_lda(self):
        if self.__model_lda is not None:
            return MatrixSimilarity(self.__model_lda[self.__corpus])

