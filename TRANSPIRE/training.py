
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)

import gpflow
    
def compute_inducing(X, n_induce):
    '''Compute inducing points using the K-means approach

    Args:
        X (Union(pd.DataFrame, np.ndarray)): input data to be fit by the K-means algorithm
        n_induce (int): number of inducing points to return (i.e. the number of clusters to be fit by K-means)

    Returns:
        Z_induce (np.ndarray): array of inducing points as determined by K-means clustering of the input data (e.g. the K-means-defined cluster centers)

    '''
    
    return MiniBatchKMeans(n_induce, max_iter=2000).fit(X).cluster_centers_ 

def build_model(X, y, **model_params):

    '''Build a GPFlow SVGP classifier.

    Args:
        X (np.ndarray): training data (n x m)
        y (np.ndarray): encoded training data labels (n x 1)
        model_params (dict): Key, value pairs of model parameters to be passed to gpflow.models.SVGP. Parameters not specified in the dictionary will be chosen from default settings.

    Returns:
        m (gpflow.models.SVGP): GPFlow SVGP model built with the defined model parameters

    '''

    assert(isinstance(X, np.ndarray))
    assert(isinstance(y, np.ndarray))
    assert(X.shape[0]==y.shape[0])
    assert(y.shape[1]==1)

    if not 'Z' in model_params:
        if not 'n_induce' in model_params:
            n_induce = 100
        else:
            n_induce = model_params['n_induce']
        Z = compute_inducing(X, n_induce)
    else:
        Z = model_params['Z']

    if not 'n_latent' in model_params:
        n_latent = np.unique(y).shape[0]
    else:
        n_latent = model_params['n_latent']

    if not 'likelihood_func' in model_params:
        likelihood = gpflow.likelihoods.SoftMax(n_latent)
    else:
        likelihood = model_params['likelihood_func'](n_latent)

    if not 'q_diag' in model_params:
        q_diag = False
    else:
        q_diag = model_params['q_diag']

    if not 'whiten' in model_params:
        whiten = False
    else:
        whiten = model_params['whiten']

    if not 'minibatch_size' in model_params:
        minibatch_size = None
    else:
        minibatch_size = model_params['minibatch_size']
        if minibatch_size < 1:
            minibatch_size = int(X.shape[0]*minibatch_size)

    if not 'kernel' in model_params:
        kernel = gpflow.kernels.SquaredExponential(X.shape[1], ARD=True) + gpflow.kernels.White(X.shape[1])
    else:
        kernel = model_params['kernel']


        
    m = WeightedSVGP(X, y,  
                    Z = Z, 
                    weights = weights,
                    kern = kernel, 
                    likelihood = likelihood, 
                    num_latent = n_latent, 
                    q_diag = q_diag, 
                    whiten = whiten)

    return m