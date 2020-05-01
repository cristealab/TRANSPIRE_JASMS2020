import pandas as pd
import numpy as np
import itertools

def make_translocations(df, comparisons, synthetic = True):

    '''
    Generate synthetic translocations between organelles using pre-defined organelle marker proteins.

    :param df: pandas dataframe properly formatted for TRANSPIRE analysis
    :param comparisons: Pairwise combinations of conditions to make translocations between (list of tuples, e.g. [('control', 'treatment_1'), ('control', 'treatment_2'), ('control', 'treatment_3')]).
    
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
        
        mapping = pd.Series(range(0, len(np.unique(catted.index.get_level_values('label')))), index = np.unique(catted.index.get_level_values('label')))
        mapping_r = pd.Series(mapping.index, index=mapping)
        
        return catted, mapping, mapping_r

    else:

        return catted
    