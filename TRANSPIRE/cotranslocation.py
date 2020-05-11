
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
    '''Extract pairwise distances between CORUM complex proteins as a true positive metric for determining cotranslocation

    Args:
        dists (pd.Series): Pairwise distances between proteins; must have 'accession_A' and 'accession_B' index levels
        prot_to_complex (pd.Series): Series for mapping Uniprot accession numbers to their corresponding CORUM complex IDs (as returned by TRANSPIRE.data.import_data.load_CORUM)
        complex_to_prot (pd.Series): Series for mapping CORUM complex IDs the corresponding Uniprot accession numbers of their subunits (as returned by TRANSPIRE.data.import_data.load_CORUM)

    Returns:
        corum_dists (pd.Series): subset of dists corresponding to distances between members of CORUM complex members

    Raises:
        AssertionError: If dists is not a pd.Series
        AssertionError: If 'accession_A' or 'accession_B' are not levels in the index

    '''

    assert(isinstance(dists, pd.Series))
    assert('accession_A' in dists.index.names)
    assert('accession_B' in dists.index.names)

    all_accs = np.unique([dists.index.get_level_values(i).unique().values.tolist() for i in dists.index.names if 'accession' in i])
    corum_in_dataset = prot_to_complex.index.values[prot_to_complex.index.isin(all_accs)]
    corum_pairs = np.array(list(itertools.product(corum_in_dataset, corum_in_dataset)))
    
    corum_dists = dists[dists.index.get_level_values('accession_A').isin(corum_pairs[:, 0])&dists.index.get_level_values('accession_B').isin(corum_pairs[:, 1])]
    
    return corum_dists

def extract_true_neg(dists, df):
    '''Extract pairwise distances between markers for distinct subcellular organelles as a true negative metric for determining cotranslocation

    Args:
        dists (pd.Series):  Pairwise distances between proteins; must have 'accession_A' and 'accession_B' index levels
        df (pd.DataFrame): Protein profiles DataFrame formatted for TRANSPIRE (e.g. with 'accession', 'gene name' and 'localization' index levels)

    Returns:
        dists (pd.Series): subset of dists corresponding to distances between markers of distinct subcellular organelles

    Raises:
        AssertionError: If dists is not a pd.Series
        AssertionError: If 'accession_A' or 'accession_B' are not levels in the index

    '''

    assert(isinstance(dists, pd.Series))
    assert('accession_A' in dists.index.names)
    assert('accession_B' in dists.index.names)

    locs = df.reset_index(['gene name', 'localization'])['localization'].dropna()
    locs = locs[~locs.index.duplicated()]
    
    cA = locs.reindex(dists.index, level='accession_A')
    cB = locs.reindex(dists.index, level='accession_B')
    
    return dists[(cA!=cB)&(cA.notnull()&cB.notnull())]

def compute_fpr(x, y):
    '''Compute false positive rates using x (true positive) and y (true negative) for values ranging between min(x.min(), y.min()) and max(x.max(), y.max())

    Args:
        x (pd.Series): True postive pairwise distances
        y (pd.Series): True negative pariwise distances

    Returns:
        fpr (pd.Series): false positive rates for an array of distances ranging from min(x.min(), y.min()) to max(x.max(), y.max())

    '''

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
    '''Retrieve known interactions from the STRING database using their REST API

    Attributes:
        None

    '''

    def __init__(self):
        '''Initialize object        
        '''

        pass
    
    def to_query_string(self, mylist, sep): #can also accept arrays
        '''Convert a list to a string that can be used as a query string in an http post request
        
        Args:
            mylist (list): list of values
            sep (str): separator for concatentating the values

        Returns:
            l (str): items in mylist concatenated into a single string

        '''
        
        l = ''

        for item in mylist:
            try:
                l = l + str(item) + sep
            
            except TypeError: # exception to deal with NaNs in mylist
                pass
        
        return l
    
    def map_identifiers_string(self, proteins, species):
        '''Use STRING's API to retrive the corresponding STRING identifiers for each protein
        
        Args:
            proteins (Union(list, np.ndarray)): Uniprot protein accessions to be mapped to StringIDs
            species (str): Taxonomic identifier for the given protein species (e.g. '9606' for Homo Sapiens)
        
        Returns:
            df (pd.DataFrame): Uniprot accessions mapped to their corresponding StringIDs   
        
        '''
        
        # STRING will only let you query 2000 proteins at a time, otherwise you get an error message back
        
        if len(proteins) >= 2000:
            n_chunks = int(np.ceil(len(proteins)/2000))
            dfs = []
            
            for chunk in range(n_chunks):
                ps = proteins[2000*chunk:2000*(chunk+1)]
                
                p = self.to_query_string(ps, '%0D') #each protein on a new line

                url = 'https://string-db.org/api/tsv/get_string_ids'
                params = {'identifiers': p, 'species':species, 'echo_query': 1, 'caller_identity': 'TRANSPIRE'}

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
        '''Query STRING database for known interactions between proteins

        Args:
            IDs (Union(list, np.ndarray)): StringIDs for query proteins
            species (str): Taxonomic identifier for the given protein species (e.g. '9606' for Homo Sapiens)

        Returns:
            df (pd.DataFrame): known interactions between proteins as well as their corresponding STRING data (evidence scores, etc.)

        '''
        
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

    def query(self, proteins, species, score_cutoff):
        '''Perform a STRING database query on a given set of protein accession numbers.
        
        This is a simple wrapper combining several GetSTRINGInteractions methods that returns at DataFrame of known interactions between the input proteins.

        Args:
            proteins (np.ndarray): Uniprot accession numbers for proteins to query for known interactions
            species (str): Taxonomic identifier for the given protein species (e.g. '9606' for Homo Sapiens)
            score_cutoff (float): STRING score cutoff for the returned iteractions

        Returns:
            interactions (pd.DataFrame): Known iteractions bewteen the input proteins

        '''

        string_IDs = self.map_identifiers_string(proteins.tolist(), species)
        string_IDs = string_IDs[~string_IDs.squeeze().index.duplicated()]

        interactions_ = self.get_interactions(string_IDs.index.values.tolist(), species)

        interactions = interactions_.copy()

        interactions['Accession_A'] = string_IDs.loc[species+'.'+interactions_['stringId_A'], 'queryItem'].values
        interactions['Accession_B'] = string_IDs.loc[species+'.'+interactions_['stringId_B'], 'queryItem'].values
        interactions = interactions.set_index(['Accession_A', 'Accession_B'])
        interactions = interactions[~interactions.index.duplicated()]
        interactions = interactions[interactions['score']>=score_cutoff]

        # create a copy of values for when the indices are reversed
        interactions_copy = interactions.copy()
        interactions_copy.index = pd.MultiIndex.from_tuples(list(zip(interactions.index.get_level_values('Accession_B'), interactions.index.get_level_values('Accession_A'))), names = interactions.index.names)

        temp = pd.concat([interactions, interactions_copy])
        interactions = temp[~temp.index.duplicated()]

        return interactions

    

