
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)

import gpflow
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def compute_inducing(X, n_induce):
    return MiniBatchKMeans(n_induce, max_iter=2000).fit(X).cluster_centers_ 

def build_model(X, y, **model_params):

    '''
    Build an SVGP classifier.

    :param X: np.array of the training data
    :param y: np.array of the encoded training data labels
    :param model_params: dict of model parameters. Parameters not specified in the dictionary will be chosen from default settings.

    :returns m: gpflow SVGP model

    '''

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

    
    m = gpflow.models.SVGP(X, y,  
                            Z = Z,
                            kern = kernel, 
                            likelihood = likelihood, 
                            num_latent = n_latent, 
                            q_diag=q_diag, 
                            whiten=whiten, 
                            minibatch_size=minibatch_size)

    return m