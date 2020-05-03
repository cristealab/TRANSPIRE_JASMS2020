import sklearn.model_selection
import pandas as pd
import numpy as np

def group_organelles(df_, mapping):
    
    '''
    :param df_: Pandas dataframe properly formatted for TRANSPIRE analysis. Must contain a "localization" index level.
    :param mapping: dict of values defining how to combine organelles. 
    Note that organelles not found in mapping, but defined in the "localization" index level will retain their original value.
    :returns df: Copy of the inuput dataframe with the "localization" index level mapped according to mapping
    
    '''
    
    df = df_.copy().reset_index()
    df['localization'] = df['localization'].where(~df['localization'].isin(list(mapping.keys())), df['localization'].map(mapping))
    
    return df.set_index(df_.index.names)


def sample_balanced(X, n_limit, n_folds, random_state = 17):
    def parse_data(x):

        if x.shape[0]>= n_limit:

            return x.sample(n_limit, random_state=17).loc[x.name[0], :].reset_index('label', drop=True)

        else:

            to_samp = n_limit-x.shape[0]

            poss = X[(X.index.get_level_values('label')==x.name[1])&(~X.index.isin(x.loc[x.name[0], :].index))]
            poss_sample = poss.sample(min(poss.shape[0], to_samp), random_state=x.name[0])

            return pd.concat([x.loc[x.name[0]], poss_sample]).reset_index('label', drop=True)
    
    X_train_dfs = []

    skf = sklearn.model_selection.StratifiedKFold(n_splits=n_folds, random_state=17)
    
    print('Creating balanced training partitions (this may take a while)....', end=' ')

    for _, test_idx in skf.split(X, X.index.get_level_values('label')):
        X_train_dfs.append(X.iloc[test_idx, :])

    temp = pd.concat(X_train_dfs, keys=range(1, n_folds+1), names=['fold'])
    
    X_train_dfs = temp.groupby(['fold', 'label']).apply(parse_data)
    
    print('done')

    return X_train_dfs

def train_test_validate_split(X, groupby_levels, f_train = 0.5, f_validate = 0.25, f_test = 0.25, random_state=17):
    
    if not abs(sum([f_train, f_validate, f_test])-1) < 0.05:
        raise ValueError ('Sum of f_train, f_validate, and f_test must equal 1.')
    
    print('Splitting data into training, validation, and testing folds (this may take a while) . . . ', end='')

    X_train_validate = X.groupby(groupby_levels, group_keys=False).apply(lambda x: x.sample(frac=f_train, random_state=random_state))
    X_train_df = X_train_validate.groupby(groupby_levels, group_keys=False).apply(lambda x: x.sample(frac=(2*f_validate)/(f_train+f_validate), random_state=random_state))
    X_validate_df = X_train_validate[~X_train_validate.index.isin(X_train_df.index)].copy()
    X_test_df = X[~X.index.isin(X_train_validate.index)].copy()

    print('done')

    return X_train_validate, X_train_df, X_validate_df, X_test_df


def map_binary(x, mapping):
    
    cols = np.array([np.array(arr) for arr in x[x.columns[~x.isnull().all()]].columns.astype(int).map(mapping).str.split(' to ', expand=True).values])
    
    zero = x[x[x.columns[~x.isnull().all()]].columns[(cols[:, 0]==cols[:, 1])|(cols[:, 0]=='No translocation')]]
    one = x[x[x.columns[~x.isnull().all()]].columns[(cols[:, 0]!=cols[:, 1])&(cols[:, 0]!='No translocation')]]
    
    if 'label' in x.index.names:
        return pd.concat([zero.sum(axis=1), one.sum(axis=1), pd.Series(x.index.get_level_values('localization_A')!=x.index.get_level_values('localization_B'), index=x.index)*1], keys = ['no translocation', 'translocation', 'true label'], axis=1)
    else:
        return pd.concat([zero.sum(axis=1), one.sum(axis=1)], keys = ['no translocation', 'translocation'], axis=1)

        
def lookup(string, df, level):
    return df[df.index.get_level_values(level).str.lower().str.contains(string.lower())]
