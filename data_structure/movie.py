

# If the param is a string that is not a number, it return 0
# otherwise, it returns the number
def get_certain_number(param):
    number = 0
    if param is not None:
        try:
            number = float(param)
        except ValueError:
            number = 0
    return number


class Movie:

    # Make sure that all number attributes are real numbers or zeros
    def fix_numbers(self):
        self.year = get_certain_number(self.year)
        if self.year == 0:
            self.year = 1982

        self.tomatoMeter = get_certain_number(self.tomatoMeter)
        self.tomatoRating = get_certain_number(self.tomatoRating)
        self.tomatoUserRating = get_certain_number(self.tomatoUserRating)
        self.tomatoUserMeter = get_certain_number(self.tomatoUserMeter)
        self.tomatoReviews = get_certain_number(self.tomatoReviews)
        self.imdbRating = get_certain_number(self.imdbRating)
        self.popularity = get_certain_number(self.popularity)

    def __init__(self, movie_id=None, dictionary=None):
        self.movie_id = movie_id
        self.title = ""
        self.year = None
        self.genres = []
        self.critics = []
        self.actors = []
        self.directors = []
        self.writers = []
        self.productionCompanies = ''
        self.awards = ''
        self.wikiDescription = ""
        self.omdbDescription = ""
        self.tmdbDescription = ""
        self.tomatoMeter = 0
        self.tomatoRating = 0
        self.tomatoUserRating = 0
        self.tomatoUserMeter = 0
        self.tomatoReviews = 0
        self.imdbRating = 0
        self.popularity = 0

        if dictionary is not None:
            for key in dictionary:
                setattr(self, key, dictionary[key])

        self.fix_numbers()
