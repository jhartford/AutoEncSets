from scipy.sparse import csr_matrix
import numpy as np

def df_to_matrix(df, users, movies):
    row = df.user_id - 1
    col = df.movie_id - 1
    data = df.rating
    return csr_matrix((data, (row, col)), shape=(users, movies))

def get_mask(matrix):
    return np.array(1.*(matrix.toarray() > 0.).reshape((matrix.shape[0], 
                                                        matrix.shape[1], 1)), 
                    dtype="float32")


