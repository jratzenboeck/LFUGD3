

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


class User:

    # Make sure that all number attributes are real numbers or zeros
    def fix_numbers(self):
        self.age = get_certain_number(self.age)
        self.zip_code = get_certain_number(self.zip_code)

    def __init__(self, user_id=None, dictionary=None):
        self.user_id = user_id
        self.gender = ''
        self.age = None
        self.job = None
        self.zip_code = None

        if dictionary is not None:
            for key in dictionary:
                setattr(self, key, dictionary[key])

        self.fix_numbers()
