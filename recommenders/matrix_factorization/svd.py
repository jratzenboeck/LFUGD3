import pandas as pd
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import pickle
#from sklearn.metrics import mean_squared_error

def svd_new():
    training, test = read_dataset()
    data_shape = (6040, 3650)

    training[:, 0:2] -= 1
    X_train = scipy.sparse.csr_matrix((training[:, 2], (training[:, 0], training[:, 1])), dtype=np.float, shape=data_shape)
    
    # Compute means of nonzero elements
    X_row_mean = np.zeros(data_shape[0])
    X_row_sum = np.zeros(data_shape[0])
    
    train_rows, train_cols = X_train.nonzero()
    
    # Iterate through nonzero elements to compute sums and counts of rows elements
    for i in range(train_rows.shape[0]):
        X_row_mean[train_rows[i]] += X_train[train_rows[i], train_cols[i]]
        X_row_sum[train_rows[i]] += 1
    
    # Note that (X_row_sum == 0) is required to prevent divide by zero
    X_row_mean /= X_row_sum + (X_row_sum == 0)
    
    # Subtract mean rating for each user
    for i in range(train_rows.shape[0]):
        X_train[train_rows[i], train_cols[i]] -= X_row_mean[train_rows[i]]

    X_train = np.array(X_train.toarray())

    ks = np.arange(2, 50)
    train_mae = np.zeros(ks.shape[0])
    train_scores = X_train[(train_rows, train_cols)]

    # Now take SVD of X_train
    U, s, Vt = np.linalg.svd(X_train, full_matrices=False)
    
    for j, k in enumerate(ks):
        X_pred = U[:, 0:k].dot(np.diag(s[0:k])).dot(Vt[0:k, :])
    
        pred_train_scores = X_pred[(train_rows, train_cols)]

        #train_mae[j] = mean_squared_error(train_scores, pred_train_scores)

        #print(k,  train_mae[j])
        print(pred_train_scores)
        
def svd_old():
    training, test = read_dataset();
    users = np.unique(training[0])
    movies = np.unique(training[1])

    number_of_rows = len(users)
    number_of_columns = len(movies)

    movie_indices = {}
    user_indices = {}

    for i in range(len(movies)):
        movie_indices[movies[i]] = i

    for i in range(len(users)):
        user_indices[users[i]] = i

    # sp sparse matrix to store the values
    V = scipy.lil_matrix((number_of_rows, number_of_columns))

    # adds data into the sparse matrix
    for line in training.values:
        u, i, r = map(int,line)
        V[user_indices[u], movie_indices[i]] = r

    #as these operations consume a lot of time, it's better to save processed data
    with open('matrix.pickle', 'wb') as handle:
        pickle.dump(V, handle)

    # compute the SVD
    u,s, vt = scipy.svds(V, k = 5)

    with open('matrix_u', 'wb') as handle:
        pickle.dump(u, handle)
    with open('matrix_s', 'wb') as handle:
        pickle.dump(s, handle)
    with open('matrix_v^t', 'wb') as handle:
        pickle.dump(vt, handle)

    s_diag_matrix = np.zeros((s.shape[0], s.shape[0]))

    for i in range(s.shape[0]):
        s_diag_matrix[i,i] = s[i]

    X_pred = np.dot(np.dot(u, s_diag_matrix), vt)



def read_dataset():
    dataset = pd.read_csv('../../data/shuffled.training.dat', sep = '\t', header=None)
    msk = np.random.rand(len(dataset)) < 0.8
    training_idx = dataset[msk]
    test_idx = dataset[~msk]

    return (training_idx.values, test_idx.values)





