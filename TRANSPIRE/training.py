
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)

import gpflow
from gpflow.decors import params_as_tensors
from gpflow import settings
from gpflow.params import DataHolder
from gpflow.params import Minibatch

class WeightedSVGP(gpflow.models.SVGP):
    def __init__(self, X, Y, Z, kern, likelihood, weights = None, **kwargs):
        
        if (weights is None) or (weights.shape == Y.shape):
            
            super().__init__(X, Y, Z=Z, kern = kern, likelihood = likelihood, **kwargs)

            if weights is None:
                weights = np.ones(Y.shape)
            
            if 'minibatch_size' in kwargs:
                raise NotImplementedError('Weighted SVGP cannot currently minibatch')
            else:
                self.weights = DataHolder(weights) 

        else:
            raise ValueError('Shape of weights {} must match shape of Y {}'.format(weights.shape, Y.shape))


    @params_as_tensors
    def _build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """

        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean, fvar = self._build_predict(self.X, full_cov=False, full_output_cov=False)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.float_type) / tf.cast(tf.shape(self.X)[0], settings.float_type)

        likelihood = tf.reduce_sum(var_exp*self.weights) * scale - KL

        return likelihood
    
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

    assert(isinstance(X, np.ndarray))
    assert(isinstance(y, np.ndarray))
    assert(X.shape[0]==y.shape[0])
    assert(y.shape[1]==1)

    if 'weights' in model_params:
        assert(model_params['weights'].shape==y.shape)

        if 'minibatch_size' in model_params:
            assert(isinstance(model_params['minibatch_size'], type(None)))

        weights = model_params['weights']

    else:
        weights = None

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

    

    if minibatch_size is not None and weights is None:

        m = gpflow.models.SVGP(X, y,  
                               Z = Z, 
                               kern = kernel, 
                               likelihood = likelihood, 
                               num_latent = n_latent, 
                               q_diag = q_diag, 
                               whiten = whiten, 
                               minibatch_size = minibatch_size)

    elif minibatch_size is None:
        
        m = WeightedSVGP(X, y,  
                        Z = Z, 
                        weights = weights,
                        kern = kernel, 
                        likelihood = likelihood, 
                        num_latent = n_latent, 
                        q_diag = q_diag, 
                        whiten = whiten)

    return m