import pandas as pd
import numpy as np

def load_data(f):

    '''
    Load a dataset for analysis.

    :param f: absolute file path for .csv or .txt data file
    :returns df: MultiIndex dataframe (index and columns are both MultiIndexes)
    :raises ValueError: Error is raised when the target file formatting does not match what is required by TRANSPIRE for proper analysis. 

    '''
    
    df = pd.read_csv(f, header=[0, 1])
    
    if not all([s in df.iloc[0, :].astype(str).str.lower().values for s in ['accession', 'gene name']]):
        raise ValueError('Dataframe is not properly formatted.')

    idx_cols = np.where(df.iloc[0, :].astype(str).str.lower().isin(['accession', 'gene name', 'localization']))[0]

    df = pd.read_csv(f, index_col = idx_cols.tolist(), header=[0, 1])

    try:
        df.index.names = [s.lower() for s in df.index.names]
        df.columns.names = [s.lower() for s in df.columns.names]
        
    except AttributeError as e:
        
        raise ValueError('Dataframe index or column names are improperly formatted')

    if not all([s in df.index.names for s in ['accession', 'gene name']])&all([s in df.columns.names for s in ['condition', 'fraction']]):
        raise ValueError('Dataframe is not properly formatted. Check index and column name spelling and structure')
        
    return df

def add_markers(df_, markers_):

    '''
    Append organelle marker localization information to a dataframe.

    :param df_: Pandas dataframe formatted for TRANSPIRE analysis
    :param markers_: Pandas dataframe or series with an "accession" and "localization" column specifying organelle marker Uniprot accession numbers and their corresponding subcellular localization.
    :returns df: a copy of the original input dataframe with organelle localizations appended as an additional index level

    '''
    
    df = df_.copy()
    markers = markers_.copy().reset_index()   
    
    if 'localization' in df.index.names:
        raise ValueError('Index level "localization" already exists. If wanting to over-write these labels, remove them from the dataframe using df.reset_index("localization", drop=True)')
    
    try:
        
        markers.columns = [s.lower() for s in markers.columns.values.tolist()]
        markers = markers.set_index('accession')
        markers = markers['localization'].squeeze()
        
    except (KeyError, AttributeError):
        
        raise ValueError('Organelle markers dataframe is not properly formatted. Check index and column names')
    
    df['localization'] = markers.reindex(df.index, level='accession')
    
    return df.reset_index().set_index(df.index.names+['localization'])