
import pandas as pd
import numpy as np

from sklearn.covariance import MinCovDet
from scipy.spatial.distance import cdist

def compute_distance(X):
    
    '''Compute the Mahalanobis distance between pairwise combinations of all samples in X.

    Args:
        X (pd.DataFrame): DataFrame with spatial profile data
    
    Returns:
        dists (pd.DataFrame): All pairwise distances between samples. The index from X will become the index and columns for this DataFrame.

    Note that this function will calculate pairwise distances for all combinations of samples in the index (e.g. it returns an n x n DataFrame, which can become quite large depending on the input data)
    '''

    mincovdet = MinCovDet(random_state=17)
    vi = mincovdet.fit(X.values).covariance_

    dists = cdist(X.values, X.values, 'mahalanobis', VI=np.linalg.inv(vi))
    dists = np.triu(dists, k=1)

    idx = X.index.copy()
    idx.names = ['{}_A'.format(n) for n in X.index.names]

    cols = X.index.copy()
    cols.names = ['{}_B'.format(n) for n in X.index.names]

    dists = pd.DataFrame(dists, index = idx, columns = cols)
    
    return dists.where(dists!=0, np.nan)