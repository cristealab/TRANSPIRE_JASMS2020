import pandas as pd
import numpy as np

import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data(f):

    '''Load a dataset for analysis.

    Args:
        f (str): absolute file path for Excel, .csv, or .txt data file
    
    Returns:
        df (pd.DataFrame): MultiIndex dataframe (index and columns are both MultiIndexes)

    Raises:
        ValueError: Error is raised when the target file formatting does not match what is required by TRANSPIRE for proper analysis. 

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

    '''Append organelle marker localization information to a dataframe.

    Args:
        df_ (pd.DataFrame): Pandas dataframe formatted for TRANSPIRE analysis
        markers_(Union(str, pd.DataFrame)): String referring to an organelle marker set in external data or a custom set of markers loaded as a pd.DataFrame or pd.Series with an "accession" and "localization" column specifying organelle marker Uniprot accession numbers and their corresponding subcellular localization.
    
    Returns:
        df(pd.DataFrame): a copy of the original input dataframe with organelle localizations appended as an additional index level

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
    '''Load an organelle marker set from TRANSPIRE.data.external.organelle_markers

    Args:
        marker_set_name (str): Name of marker set to load
        df (pd.DataFrame, optional): DataFrame to coerce into proper formatting for TRANSPIRE

    Returns:
        markers (pd.Series): Marker set loaded as a pd.Series with index and value pairs referring to protein accession number and associated subcellular localization

    Raises:
        ValueError: If marker_set_name is not a valid marker set in TRANSPIRE.data.external.organelle_markers

    ''' 

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
    '''Load TRANSPIRE predictions from a filepath

    Args:
        f (str): valid filepath to .csv or .zip file

    Returns:
        df (pd.DataFrame): DataFrame loaded from filepath

    '''

    df = pd.read_csv(f, header=[0], index_col=[0, 1, 2, 3, 4, 5, 6, 7])

    assert(all([i in ['accession_A', 'accession_B', 'gene name_A', 'gene name_B', 'condition_A', 'condition_B', 'localization_A', 'localization_B'] for i in df.index.names]))

    return df

def load_CORUM():
    '''Load core CORUM complexes

    Args:
        None
    
    Returns:
        corum (pd.DataFrame): DataFrame representation of CORUM core complexes information
        prot_to_complex (pd.Series): Series for mapping Uniprot accession numbers to their corresponding CORUM complex IDs
        complex_to_prot (pd.Series): Series for mapping CORUM complex IDs the corresponding Uniprot accession numbers of their subunits

    '''

    corum = pd.read_csv(os.path.join(THIS_DIR, 'external', 'coreComplexes.txt'), sep='\t', index_col=0)

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

    return corum, prot_to_complex, complex_to_prot

