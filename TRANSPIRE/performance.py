import pandas as pd
import numpy as np

import sklearn.metrics

from .utils import map_binary

import warnings
warnings.simplefilter("ignore")

def eval_report(means, mapping, mapping_r):
    '''Compute an array of model performance metrics given mean classifer scores across all possible prediction classes

    Computed metrics include binary and multi-class log-loss, macro F1 scores, micro F1 scores, and weighted F1 scores.

    Args:
        means (pd.DataFrame): DataFrame of classifer scores across each possible class
        mapping (pd.Series): Mapping generator used to encode class labels
        mapping_r(pd.Series): Mapping genrator used to decode class labels

    Returns:
        results (pd.DataFrame): DataFrame of computed metrics

    '''

    # log loss
    log_loss = sklearn.metrics.log_loss(means.index.get_level_values('label').map(mapping).astype(int), means)

    # f1 scores (macro, micro, weighted, per-class)
    per_class_f1 = pd.Series(sklearn.metrics.f1_score(means.index.get_level_values('label').map(mapping).astype(int), means.idxmax(axis=1).astype(int), average=None, labels = range(mapping.max())), index = mapping_r[[i for i in range(mapping.max())]])
    macro_f1 = sklearn.metrics.f1_score(means.index.get_level_values('label').map(mapping).astype(int), means.idxmax(axis=1).astype(int), average='macro', labels = range(mapping.max()))
    micro_f1 = sklearn.metrics.f1_score(means.index.get_level_values('label').map(mapping).astype(int), means.idxmax(axis=1).astype(int), average='micro', labels = range(mapping.max()))
    weighted_f1 = sklearn.metrics.f1_score(means.index.get_level_values('label').map(mapping).astype(int), means.idxmax(axis=1).astype(int), average='weighted', labels = range(mapping.max()))
    
    # map results to their binary representation (e.g. translocating v. not translocating) and compute loss and F1 scores
    binary = map_binary(means, mapping_r)
    binary_loss = sklearn.metrics.log_loss(binary['true label'], binary['translocation score'])
    binary_f1 = sklearn.metrics.f1_score(binary['true label'], binary['translocation score']>0.5, average='weighted')
    
    results = {

        'F1 score (per class)': per_class_f1,
        'singular metrics': pd.Series({
                                'loss': log_loss, 
                                'F1 score (micro)': micro_f1,
                                'F1 score (macro)': macro_f1,
                                'F1 score (weighted)': weighted_f1,
                                'loss (binary)': binary_loss, 
                                'F1 score (weighted, binary)': binary_f1
            },)
    }

    return pd.concat(results, names = ['type of metric', 'metric'])

def compute_fpr(x, n=100):
    '''Compute false-positive rates for translocation score cutoffs

    Args:
        x (pd.DataFrame): DataFrame including 'translocation score' and 'true label' columns (as produced by TRANSPIRE.utils.map_binary)
        n (int, optional): number of bins to split the translocation scores into

    Returns:
        fpr (pd.Series): false-positive rates for different translocation score cutoffs given the true binary labels


    '''

    fp = [((x['translocation score'] > i)&(x['true label']==0)).sum() for i in np.linspace(0, 1, n)]
    fpr = fp/((x['true label']==0).sum())
    
    return pd.Series(fpr, index=np.linspace(0, 1, n))

def compute_cutoff(fprs,level, i):
    '''Compute score cutoff based on desired false-positive rate stringency

    Args:
        fprs (pd.DataFrame): calculated false-positive rates (DataFrame columns should correspond to translocation scores)
        level (list): list of multindex levels to groupby (e.g. conditions, folds, etc.)
        i (float): fpr cutoff between 0 and 1

    Returns:
        cutoffs (pd.Series): corresponding score cutoffs for each set of levels as defined by the 'level' grouping

    '''
 
    return fprs[fprs<=i].idxmax(axis=1).groupby(level).mean()
        

