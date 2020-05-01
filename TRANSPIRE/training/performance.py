import pandas as pd
import sklearn.metrics

from ..utils import map_binary

import warnings
warnings.simplefilter("ignore")

def eval_report(means, mapping, mapping_r):
    # log loss
    log_loss = sklearn.metrics.log_loss(means.index.get_level_values('label').map(mapping).astype(int), means)

    # f1 scores (macro, micro, weighted, per-class)
    per_class_f1 = pd.Series(sklearn.metrics.f1_score(means.index.get_level_values('label').map(mapping).astype(int), means.idxmax(axis=1).astype(int), average=None, labels = range(mapping.max())), index = mapping_r[[i for i in range(mapping.max())]])
    macro_f1 = sklearn.metrics.f1_score(means.index.get_level_values('label').map(mapping).astype(int), means.idxmax(axis=1).astype(int), average='macro', labels = range(mapping.max()))
    micro_f1 = sklearn.metrics.f1_score(means.index.get_level_values('label').map(mapping).astype(int), means.idxmax(axis=1).astype(int), average='micro', labels = range(mapping.max()))
    weighted_f1 = sklearn.metrics.f1_score(means.index.get_level_values('label').map(mapping).astype(int), means.idxmax(axis=1).astype(int), average='weighted', labels = range(mapping.max()))
    
    # map results to their binary representation (e.g. translocating v. not translocating) and compute loss and F1 scores
    binary = map_binary(means, mapping_r)
    binary_loss = sklearn.metrics.log_loss(binary['true label'], binary['translocation'])
    binary_f1 = sklearn.metrics.f1_score(binary['true label'], binary['translocation']>0.5, average='weighted')
    
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

    return results