
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
    mincovdet = sklearn.covariance.MinCovDet(random_state=17)
    vi = mincovdet.fit(X.T.values).covariance_
    dists = cdist(X.values, X.values, 'mahalanobis', VI=vi)
    
    dists = np.triu(dists, k=1)
    dists = pd.DataFrame(dists, index = X.index.get_level_values('accession_A'), columns = X.index.get_level_values('accession_B'))
    
    return dists.where(dists!=0, np.nan)