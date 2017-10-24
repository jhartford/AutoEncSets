from scipy.sparse import csr_matrix
import numpy as np

def df_to_matrix(df, users, movies):
    row = df.user_id - 1
    col = df.movie_id - 1
    data = np.array(df.rating) #- 3.5
    return csr_matrix((data, (row, col)), shape=(users, movies), dtype="float32")

def get_mask(matrix):
    return np.array(1.*(matrix > 0.), dtype="float32")

def to_indicator(mat):
    out = np.zeros((mat.shape[0], mat.shape[1], 5))
    for i in range(1, 6):
        out[:, :, i-1] = (1 * (mat == i)).reshape((mat.shape[0], mat.shape[1]))
    return np.array(out,  dtype="float32")

def to_number(mat):
    out = (np.argmax(mat, axis=2).reshape((mat.shape[0], mat.shape[1], 1)))
    out[mat.sum(axis=2) > 0] += 1
    return np.array(out, dtype="float32")
