import pandas as pd
import numpy as np

import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data(f):

    '''
    Load a dataset for analysis.

    :param f: absolute file path for .csv or .txt data file
    :returns df: MultiIndex dataframe (index and columns are both MultiIndexes)
    :raises ValueError: Error is raised when the target file formatting does not match what is required by TRANSPIRE for proper analysis. 

    '''
    
    f_types = {'csv': ',', 'txt': '\t', 'xlsx': ''}
    
    f_type = f.split('.')[-1]

    if not f_type in f_types:
        raise ValueError('File type must be .csv, .txt (tab-separated), or excel')

    if f_type == 'xlsx':
        df = pd.read_excel(f, header=[0, 1])
    else:
        df = pd.read_csv(f, header=[0, 1], sep=f_types[f_type])
    
    if not all([s in df.iloc[0, :].astype(str).str.lower().values for s in ['accession', 'gene name']]):
        raise ValueError('Dataframe is not properly formatted.')

    idx_cols = np.where(df.iloc[0, :].astype(str).str.lower().isin(['accession', 'gene name', 'localization']))[0]

    if f_type == 'xlsx':
        df = pd.read_excel(f, header=[0, 1], index_col = idx_cols.tolist())
    else:
        df = pd.read_csv(f, index_col = idx_cols.tolist(), header=[0, 1], sep=f_types[f_type])

    try:
        df.index.names = [s.lower() for s in df.index.names]
        df.columns.names = [s.lower() for s in df.columns.names]
        
    except AttributeError as _:
        
        raise ValueError('Dataframe index or column names are improperly formatted')

    if not all([s in df.index.names for s in ['accession', 'gene name']])&all([s in df.columns.names for s in ['condition', 'fraction']]):
        raise ValueError('Dataframe is not properly formatted. Check index and column name spelling and structure')
        
    return df

def add_markers(df_, markers_):

    '''
    Append organelle marker localization information to a dataframe.

    :param df_: Pandas dataframe formatted for TRANSPIRE analysis
    :param markers_: String referring to an organelle marker set in external data or a custom set of markers loaded as a Pandas dataframe or series with an "accession" and "localization" column specifying organelle marker Uniprot accession numbers and their corresponding subcellular localization.
    :returns df: a copy of the original input dataframe with organelle localizations appended as an additional index level

    '''

    if isinstance(markers_, str):
        markers_ = load_organelle_markers(markers_)

    elif isinstance(markers_, pd.Series) or isinstance(markers_, pd.DataFrame):
        markers_ = load_organelle_markers('custom', df=markers_)

    else:
        raise ValueError()
    
    df = df_.copy() 
    
    if 'localization' in df.index.names:
        raise ValueError('Index level "localization" already exists. If wanting to over-write these labels, remove them from the dataframe using df.reset_index("localization", drop=True)')
    
    df['localization'] = markers_.reindex(df.index, level='accession')
    
    return df.reset_index().set_index(df.index.names+['localization'])

def load_organelle_markers(marker_set_name, df=None):

    if not isinstance(marker_set_name, str):
        raise ValueError("marker_set_name must be a string")

    if marker_set_name == 'custom':

        df = df.reset_index().copy()
        df.columns =  [n.lower() for n in df.columns]

        if 'accession' in df.columns:
            df = df.set_index('accession')
        else:
            raise ValueError('Marker dataframe does not have an "accession" column.')
            
        if 'localization' in df.columns:
            return df['localization'].squeeze()
        else:
            raise ValueError('Marker dataframe does not have a "localization" column.')

    elif marker_set_name in [f.split('.')[0] for f in os.listdir(os.path.join(THIS_DIR, 'external', 'organelle_markers'))]:
        return pd.read_csv(os.path.join(THIS_DIR, 'external', 'organelle_markers', '{}.csv'.format(marker_set_name)), header=0, index_col=0).squeeze()
    
    else:
        raise ValueError('{} is not a valid marker set name'.format(marker_set_name))


def load_predictions(f):

    df = pd.read_csv(f, header=[0, 1], index_col=[0, 1, 2, 3, 4, 5, 6])

    assert(all([i in ['accession_A', 'Accession_B', 'gene name_A', 'gene name_B', 'condition_A', 'condition_B'] for i in df.index.names]))

    return df

