import pandas as pd
import numpy as np
import itertools

from ..utils import get_mapping

def make_translocations(df, comparisons, synthetic = True):

    '''Generate synthetic translocations between organelles using pre-defined organelle marker proteins or simply generate concatenated protein profiles across the specified comparisons.

    Args:
        df (pd.DataFrame): Pandas dataframe properly formatted for TRANSPIRE analysis
        comparisons (Union(list, np.array)): Pairwise combinations of conditions to make translocations between (list or array of tuples, e.g. [('control', 'treatment_1'), ('control', 'treatment_2'), ('control', 'treatment_3')])
        synthetic (bool): Whether or not to generate synthetic translocation profiles using organelle marker proteins. If False, this function will just return concatenated profiles for all samples accross the provided comparisons

    Returns:
        df_concatenated (pd.DataFrame): Dataframe with concatenated profiles.

    '''

    if not 'condition' in df.index.names:

        df = df.stack('condition')
    
    catted = []

    for cA, cB in comparisons:
        
        if synthetic == True:
            A = df[(~df.index.get_level_values('localization').isnull())&(df.index.get_level_values('condition')==cA)].copy()
            B = df[(~df.index.get_level_values('localization').isnull())&(df.index.get_level_values('condition')==cB)].copy()
        
        else:
            A = df[df.index.get_level_values('condition')==cA].copy()
            B = df[df.index.get_level_values('condition')==cB].copy()

        A = A[A.index.get_level_values('accession').isin(B.index.get_level_values('accession'))]
        B = B[B.index.get_level_values('accession').isin(A.index.get_level_values('accession'))]

        if synthetic == True:
            n_idx = np.array(list(itertools.product(range(A.shape[0]), range(B.shape[0]))))
        else:
            n_idx = np.array(list(zip(range(A.shape[0]), range(B.shape[0]))))

        a = A.iloc[n_idx[:, 0], :]
        b = B.iloc[n_idx[:, 1], :]

        a.index.names = ['{}_A'.format(n) for n in a.index.names]
        b.index.names = ['{}_B'.format(n) for n in b.index.names]

        new_idx = pd.MultiIndex.from_arrays(pd.concat([a.reset_index()[a.index.names], b.reset_index()[b.index.names]], axis=1).values.T, names = a.index.names+b.index.names)

        a.index = new_idx
        b.index = new_idx
        c = pd.concat([a, b], axis=1)

        c.columns = range(1, c.shape[1]+1)
        catted.append(c)

    catted = pd.concat(catted)

    if synthetic == True:
        catted['label'] = catted.index.get_level_values('localization_A').str.cat(catted.index.get_level_values('localization_B').values, sep=' to ')
        catted = catted.reset_index().set_index(catted.index.names+['label'])
        
        return catted

    else:

        return catted
    