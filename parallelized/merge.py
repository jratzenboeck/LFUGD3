from glob import glob
from os.path import basename
import csv


def read_data(recommenders_output_directory, actual_values_directory):
    data = {}

    # Reading all output files and accumulate ratings in case of duplicates
    files = glob('%s/*.out' % recommenders_output_directory)
    for file_path in files:
        file_name = basename(file_path)
        file_name_parts = file_name.split('.')

        set_number = file_name_parts[0]
        method = '_'.join(file_name_parts[1:(len(file_name_parts) - 1)])

        with open(file_path, newline='') as file:
            reader = csv.reader(file, delimiter='\t', quotechar='|')
            for row in reader:
                (user_id, movie_id, rating) = (row[0], row[1], float(row[2]))
                key = '%s_%s' % (user_id, movie_id)

                if key not in data:
                    data.setdefault(key, {})

                if method in data[key]:
                    data[key][method]['sum'] += rating
                    data[key][method]['count'] += 1
                else:
                    data[key][method] = {'sum': rating, 'count': 1}

    # Flatten the accumulated results by averaging them
    for key in data:
        for method in data[key]:
            data[key][method] = data[key][method]['sum'] / data[key][method]['count']

    # Reading actual values
    files = glob('%s/*.test' % actual_values_directory)
    for file_path in files:
        with open(file_path, newline='') as file:
            reader = csv.reader(file, delimiter='\t', quotechar='|')
            for row in reader:
                (user_id, movie_id, rating) = (row[0], row[1], float(row[2]))
                key = '%s_%s' % (user_id, movie_id)

                if key not in data:
                    data.setdefault(key, {})

                data[key]['userId_movieId'] = key
                data[key]['actual_rating'] = rating

    return data


def write_data(data, regression_input_directory):
    # Get the column names from the first item
    header = []
    for key in data:
        header = data[key].keys()
        break

    output_path = '%s/reg.train' % regression_input_directory
    with open(output_path, 'w', newline='') as file:
        dict_writer = csv.DictWriter(file, header, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        dict_writer.writeheader()
        dict_writer.writerows(data.values())


if __name__ == '__main__':
    rec_output_dir = 'data/rec_output'
    rec_test_dir = 'data/rec_test'
    reg_input_dir = 'data/reg_input'
    data = read_data(recommenders_output_directory=rec_output_dir,
                     actual_values_directory=rec_test_dir)
    write_data(data=data,
               regression_input_directory=reg_input_dir)
