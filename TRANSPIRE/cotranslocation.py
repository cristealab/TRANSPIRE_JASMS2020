
import pandas as pd
import numpy as np

from sklearn.covariance import MinCovDet
from scipy.spatial.distance import cdist

def compute_distance(X):
    
    '''
    Compute the Mahalanobis distance between pairwise combinations of all samples in X.

    :param X: pd.DataFrame properly formatted for TRANSPIRE analysis (i.e. has the proper multi index levels)
    :returns dists: pd.DataFrame for all pairwise distances between samples.

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