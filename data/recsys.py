import pandas as pd
import numpy as np

from data import CompletionDataset

def ml100k(validation=0., seed=None):
    rng = np.random.RandomState(seed)
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    train_df = pd.read_csv("./data/ml-100k/u1.base", sep="\t", names=r_cols, encoding='latin-1')
    validation_df = pd.read_csv("./data/ml-100k/u1.test", sep="\t", names=r_cols, encoding='latin-1')
    
    ratings = np.concatenate([train_df.rating, validation_df.rating])
    mask = np.concatenate([np.array([train_df.user_id, train_df.movie_id]).T,
                           np.array([validation_df.user_id, validation_df.movie_id]).T], axis=0)
     
    if validation > 0.:
        n = train_df.shape[0]
        n_train = int(n * (1-validation))
        ind_tr = rng.permutation(np.concatenate([np.zeros(n_train), np.ones(n - n_train)]))
    else:
        ind_tr = np.zeros_like(train_df.rating)
        
    indicator = np.concatenate([ind_tr, 2 * np.ones_like(validation_df.rating)])
    return CompletionDataset(ratings, mask, indicator)
