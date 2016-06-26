from glob import glob
from os.path import basename
import csv


def read_data(recommenders_output_directory, actual_values_directory=None):
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
        data[key]['userId_movieId'] = key
        for method in data[key]:
            # Skip the key
            if method == 'userId_movieId':
                continue

            data[key][method] = data[key][method]['sum'] / data[key][method]['count']

    # Reading actual values
    if actual_values_directory is None:
        files = glob('%s/*.test' % actual_values_directory)
        for file_path in files:
            with open(file_path, newline='') as file:
                reader = csv.reader(file, delimiter='\t', quotechar='|')
                for row in reader:
                    (user_id, movie_id, rating) = (row[0], row[1], float(row[2]))
                    key = '%s_%s' % (user_id, movie_id)

                    if key in data:
                        data[key]['actual_rating'] = rating

    return data


def write_data(data, output_file_path):
    # Get the column names from the first item
    header = []
    for key in data:
        header = data[key].keys()
        break

    with open(output_file_path, 'w', newline='') as file:
        dict_writer = csv.DictWriter(file, header, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        dict_writer.writeheader()
        dict_writer.writerows(data.values())


def merge_training_files():
    rec_output_dir = 'data/rec_output/train'
    rec_test_dir = 'data/rec_test'
    reg_input_path = 'data/reg_input/reg.train'
    data = read_data(recommenders_output_directory=rec_output_dir,
                     actual_values_directory=rec_test_dir)
    write_data(data=data,
               output_file_path=reg_input_path)


def merge_prediction_files():
    rec_output_dir = 'data/rec_output/predict'
    rec_test_dir = 'data/rec_test'
    reg_input_path = 'data/reg_input/reg.predict'
    data = read_data(recommenders_output_directory=rec_output_dir,
                     actual_values_directory=rec_test_dir)
    write_data(data=data,
               output_file_path=reg_input_path)


if __name__ == '__main__':
    # merge_training_files()
    merge_prediction_files()

