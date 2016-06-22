import csv

def get_ratings_of_predictset(path):
    dataset = []
    with open(path, newline='') as file:
        reader = csv.reader(file, delimiter='\t', quotechar='|')
        for row in reader:
            rating = float(row[2])
            dataset.append(rating)
    return dataset

def merge_ratings_to_testset(test_set_path, rec_output_paths):
    merged = []
    with open(test_set_path, newline='') as file:
        reader = csv.reader(file, ddelimiter='\t', quotechar='|')
        i = 0
        for row in reader:
            for rec_output_path in rec_output_paths:
                rec_output_ratings = get_ratings_of_predictset(rec_output_path)
                row.append(rec_output_ratings[i])
            merged.append(row)
            i += 1
    return merged

# Merges the results for one split
rec_output_paths = ['1.cf.item.out', '1.cf.svd.out', '1.cf.svd.out', '1.content.item.out', '1.content.user.out']
merged_set = merge_ratings_to_testset('data/1.rec.test', rec_output_paths)
