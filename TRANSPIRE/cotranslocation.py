
import pandas as pd
import numpy as np
import requests
import io 
from sklearn.covariance import MinCovDet
from scipy.spatial.distance import cdist
import os
import time
import itertools

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

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

def extract_true_pos(dists, complex_to_prot, prot_to_complex):
    all_accs = np.unique([dists.index.get_level_values(i).unique().values.tolist() for i in dists.index.names if 'accession' in i])
    corum_in_dataset = prot_to_complex.index.values[prot_to_complex.index.isin(all_accs)]
    corum_pairs = np.array(list(itertools.product(corum_in_dataset, corum_in_dataset)))
    
    corum_dists = dists[dists.index.get_level_values('accession_A').isin(corum_pairs[:, 0])&dists.index.get_level_values('accession_B').isin(corum_pairs[:, 1])]
    
    return corum_dists

def extract_true_neg(dists, df):
    locs = df.reset_index(['gene name', 'localization'])['localization'].dropna()
    locs = locs[~locs.index.duplicated()]
    
    cA = locs.reindex(dists.index, level='accession_A')
    cB = locs.reindex(dists.index, level='accession_B')
    
    return dists[(cA!=cB)&(cA.notnull()&cB.notnull())]

def compute_fpr(x, y):

    res = {}

    for i in np.linspace(min((y.min(), x.min())), max(y.max(), x.max()), 100):
        
        tp = x[x<=i].groupby(['condition_A', 'condition_B']).size()
        tn = y[y>i].groupby(['condition_A', 'condition_B']).size()
        fp = y[y<=i].groupby(['condition_A', 'condition_B']).size()
        fn = x[x>i].groupby(['condition_A', 'condition_B']).size()
        
        res[i] = pd.concat([tp, tn, fp, fn], axis=1, keys = ['tp', 'tn', 'fp', 'fn'])
        
    res = pd.concat(res, names = ['distance']).dropna()
    fpr =(res['fp']/(res['fp']+res['tn']))

    return fpr

class GetSTRINGInteractions:
    def __init__(self):
        pass
    
    def to_query_string(self, mylist, sep): #can also accept arrays
        '''convert a list to a string that can be used as a query string in an http post request'''
        
        l = ''

        for item in mylist:
            try:
                l = l + str(item) + sep
            
            except TypeError: # exception to deal with NaNs in mylist
                pass
        
        return l
    
    def map_identifiers_string(self, proteins, species):
        '''Use STRING's API to retrive the corresponding string identifiers for each protein.
        I highly reccommend using this function prior to querying for interactions. 
        It significantly decreases the request response time'''
        
        # STRING will only let you query 2000 proteins at a time, otherwise you get an error message back
        
        if len(proteins) >= 2000:
            n_chunks = int(np.ceil(len(proteins)/2000))
            dfs = []
            
            for chunk in range(n_chunks):
                ps = proteins[2000*chunk:2000*(chunk+1)]
                
                p = self.to_query_string(ps, '%0D') #each protein on a new line

                url = 'https://string-db.org/api/tsv/get_string_ids'
                params = {'identifiers': p, 'species':species, 'echo_query': 1, 'caller_identity': 'Princeton_University'}

                r = requests.post(url, data = params)
                _df = pd.read_csv(io.StringIO(r.text), sep = '\t', header = 0, index_col = None)
                
                dfs.append(_df)
                time.sleep(1)
                
            df = pd.concat(dfs, axis = 0, join = 'outer')
   
        else:
            ps = proteins
        
            p = self.to_query_string(ps, '%0D') #each protein on a new line

            url = 'https://string-db.org/api/tsv/get_string_ids'
            params = {'identifiers': p, 'species':species, 'echo_query': 1, 'caller_identity': 'Princeton_University'}

            r = requests.post(url, data = params)
            df = pd.read_csv(io.StringIO(r.text), sep = '\t', header = 0, index_col = None)
            
            
        df = df[['stringId', 'queryItem']].set_index('stringId')
        
        return df

    
    def get_interactions(self, IDs, species):
        
        # STRING will only let you query 2000 proteins at a time
        
        if len(IDs) > 2000:
            
            n_chunks = int(np.ceil(len(IDs)/2000))
            
            dfs = []
            
            for chunk in range(n_chunks):
                ID_list = IDs[2000*chunk:2000*(chunk+1)]
                
                p = self.to_query_string(ID_list, '%0D') #each ID on a new line

                url = 'https://string-db.org/api/tsv/network'
                params = {'identifiers': p, 'species':species, 'caller_identity': 'Princeton_University'}

                r = requests.post(url, data = params)
                _df = pd.read_csv(io.StringIO(r.text), sep = '\t', header = 0, index_col = None)
                dfs.append(_df)
                time.sleep(1)
            
            df = pd.concat(dfs, axis = 0, join = 'outer')
                
        else:
            ID_list = IDs
        
            p = self.to_query_string(ID_list, '%0D') #each ID on a new line

            url = 'https://string-db.org/api/tsv/network'
            params = {'identifiers': p, 'species':species, 'caller_identity': 'Princeton_University'}

            r = requests.post(url, data = params)
            df = pd.read_csv(io.StringIO(r.text), sep = '\t', header = 0, index_col = None)
        
        return df

    def query(self, proteins, species):

        string_IDs = self.map_identifiers_string(proteins.tolist(), species)
        string_IDs = string_IDs[~string_IDs.squeeze().index.duplicated()]

        interactions_ = self.get_interactions(string_IDs.index.values.tolist(), species)

        interactions = interactions_.copy()

        interactions['Accession_A'] = string_IDs.loc[species+'.'+interactions_['stringId_A'], 'queryItem'].values
        interactions['Accession_B'] = string_IDs.loc[species+'.'+interactions_['stringId_B'], 'queryItem'].values
        interactions = interactions.set_index(['Accession_A', 'Accession_B'])
        interactions = interactions[~interactions.index.duplicated()]
        interactions = interactions[interactions['score']>=0.4]

        # create a copy of values for when the indices are reversed
        interactions_copy = interactions.copy()
        interactions_copy.index = pd.MultiIndex.from_tuples(list(zip(interactions.index.get_level_values('Accession_B'), interactions.index.get_level_values('Accession_A'))), names = interactions.index.names)

        temp = pd.concat([interactions, interactions_copy])
        interactions = temp[~temp.index.duplicated()]

        return interactions

    

