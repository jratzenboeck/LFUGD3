import sys

#To show some messages:
import recsys.algorithm
recsys.algorithm.VERBOSE = True
from recsys.algorithm.factorize import SVD
from recsys.datamodel.data import Data
from recsys.evaluation.prediction import RMSE, MAE

def svd():
    filename = 'training.dat'
    format = {'col': 0, 'row': 1, 'value': 2, 'ids': int}

    #data = Data()
    #data.load(filename, format = format)
    #train, test = data.split_train_test(percent=80) # 80% train, 20% test

    svd = SVD()
    svd.load_data(filename=filename, format=format)
    #svd.set_data(train)
    #svd.compute(k=50, min_values=10, pre_normalize=None, mean_center=True, post_normalize=True)

