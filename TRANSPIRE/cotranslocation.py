
import pandas as pd
import numpy as np
import requests
import io 
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
    

def load_CORUM():

    prot_to_complex = {}
    complex_to_prot = {}

    for complex_num, accs in zip(corum.index, corum['subunits(UniProt IDs)'].str.split(';')):
        for acc in accs:
            if not acc in prot_to_complex:
                prot_to_complex[acc] = [complex_num]

            else:
                prot_to_complex[acc].append(complex_num)

            if not complex_num in complex_to_prot:
                complex_to_prot[complex_num] = [acc]

            else:
                complex_to_prot[complex_num].append(acc)

    prot_to_complex = pd.Series(prot_to_complex)
    complex_to_prot = pd.Series(complex_to_prot)

    complex_to_prot.index.names = ['complex id']
    complex_to_prot.name = 'subunit accession'

    prot_to_complex.index.names = ['subunit accession']
    prot_to_complex.name = 'complex id'

    return prot_to_complex, complex_to_prot