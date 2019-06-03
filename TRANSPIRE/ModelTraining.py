import pandas as pd
import numpy as np
import itertools
import os

from scipy.cluster.vq import kmeans
from scipy import interp

import sklearn.metrics
from sklearn.model_selection import StratifiedKFold

import gpflow

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

class StratifiedCV(StratifiedKFold): # minimal wrapper for sklearn StratifiedKFold
    def __init__(self, n_folds = 5, shuffle=True, random_state=1, **kwargs):
        super().__init__(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    
    def return_splits(self, X, y):
        # check that there are at least n_folds samples per group
        if (X.groupby(y).size() < self.get_n_splits()).any():
            problems = X.groupby(y).size()[X.groupby(y).size() < self.get_n_splits()]
            
            # this will throw the sklearn warning
            self.split(X, y)
            print('removing training groups that are too small for proper cross-validation')

            # throw out the groups that are too small
            X = X[~y.isin(problems.index)]
            y = y[~y.isin(problems.index)]
              
        res = []

        for train_idx, test_idx in self.split(X, y):
            train = X.iloc[train_idx, :]
            y_train = y.iloc[train_idx]

            test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            res.append([[train, test], [y_train, y_test]])
        
        return res

class Encoder:
    def __init__(self):
        pass

    def fit(self, labels):
        self.classes = np.unique(labels)
        n_classes = len(self.classes)
        self.encode = pd.Series(range(n_classes), index=self.classes, name='encoded')
        self.encode.index.names = ['labels']

    def transform(self, labels):
        return self.encode[labels].values

    def inverse_transform(self, encoded):
        return pd.Series(self.encode.index.values, index=self.encode.values)[encoded].values

class GPFlowGPC:
    def __init__(self, kernel = 'Matern52', root = os.getcwd(), induce = 'kmeans', n_induce = 10, noise=True, save=True, maxiter = 10000, maxfun = 10000, name = '', **kwargs):
        
        self.kernel = kernel
        self.root = root
        self.induce = induce
        self.n_induce = n_induce
        self.noise = noise
        self.save = save
        self.maxiter = maxiter
        self.maxfun = maxfun
        self.name = name
        
        self.test_results = {}
        self.unk_results = {}
        self.fold = 0
      
        if self.n_induce < 1:
            raise ValueError('number of inducing points must be greater than zero')
        
    def train_GPC(self, X, y, minibatch = None):
        
        self.fold = self.fold + 1
            
        n_classes = len(y.unique())
        D = X.shape[1]

        # generate inducing points
        if self.induce == 'kmeans':            
            if minibatch is not None: # mini-batch k-means for better speed
                induce = minibatch.partial_fit(X.values).cluster_centers_
            else:
                if self.n_induce >= X.shape[0]:
                    induce = X.values
                else:
                    induce = kmeans(X, self.n_induce)[0]    
        else:
            if isinstance(self.induce, np.array):
                if self.induce.shape[1] == D:
                    # check that inducing point array exists within X?
                    induce = self.induce
                
                else:
                    raise ValueError('inducing points do not match data dimensions')
            else:
                raise ValueError('please choose or provide a valid inducing point metric')

        self.effective_n_induce = induce.shape[0]

        # define the kernel
        if self.kernel == 'Matern52':
            kernel = gpflow.kernels.Matern52(D, ARD=True)

        elif self.kernel == 'RBF':
            kernel = gpflow.kernels.RBF(D, ARD=True)

        else:
            raise ValueError('please choose a valid kernel')

        if self.noise:
            kernel = kernel + gpflow.kernels.White(D)

        # define the likelihood function
        if n_classes == 2: #binary case
            likelihood = gpflow.likelihoods.Bernoulli()
            latent = 1

        elif n_classes > 2: # multi class case
            invlink = gpflow.likelihoods.RobustMax(n_classes)  # Robustmax inverse link function
            likelihood = gpflow.likelihoods.MultiClass(n_classes, invlink=invlink)
            latent = n_classes

        # make the model
        m = gpflow.models.SVGP(X.values, y.values, kern=kernel, likelihood=likelihood, Z = induce, num_latent=latent)
        m.feature.trainable = False

        # train the model
        gpflow.train.ScipyOptimizer(options={'maxiter': self.maxiter, 'maxfun':self.maxfun}).minimize(m, maxiter=self.maxiter)
        print('fold {}, '.format(self.fold), end='')

        # save the model
        if self.save:
            savedir = os.path.join(self.root, 'Models')

            if not os.path.exists(savedir):
                os.mkdir(savedir)

            path = os.path.join(savedir, '{} fold {}'.format(self.name, self.fold))

            if os.path.exists(path): # if filename already exists, delete the old version
                os.remove(path)

            gpflow.saver.Saver().save(path, m)

        self.curr_model = m
    
    def load_model(self, path):
        print('load fold {}, '.format(self.fold+1), end = '')
        self.curr_model = gpflow.saver.Saver().load(path)
        self.fold = self.fold + 1

    def evaluate_performance(self, X, converter = None, type = None):
        
        pred_means, pred_vars = self.curr_model.predict_y(X.values)        
        pred_means = pd.DataFrame(pred_means, index=X.index)
        pred_vars = pd.DataFrame(pred_vars, index = X.index)
        
        if pred_means.shape[1]>1 and converter is None:
            raise ValueError('you must provide a label converter in the multi class case')
        
        if converter is not None and pred_means.shape[1]>1:
            pred_means.columns = converter.inverse_transform(pred_means.columns.values)
            pred_vars.columns = converter.inverse_transform(pred_vars.columns.values)

        else:
            pred_means.columns = ['Translocation']
            pred_means['No translocation'] = 1-pred_means['Translocation']
            
            pred_vars.columns = ['Translocation']
            pred_vars['No translocation'] = pred_vars['Translocation']
            
        res = pd.concat([pred_means, pred_vars], axis=1, keys=['score', 'uncertainty'])

        if type == 'test':
            self.test_results[self.fold] = res

        elif type == 'unknown':
            res.index.names = res.index.names[:-1]+['protein']
            self.unk_results[self.fold] = res
        
class PerformanceMetrics:
    def __init__(self, true, scores, uncertainty):
        
        if scores.shape != uncertainty.shape:
            raise ValueError('dimensions of scores and probabilities must match')
        
        self.true = true
        self.scores = scores
        self.uncertainty = uncertainty
        
        predicted = scores.idxmax(axis=1)
        vars_ = uncertainty.lookup(scores.index, predicted)

        self.labels = np.unique(true)
        self.predicted = predicted
        self.predicted_uncertainty = vars_
    
    def compute_logloss(self):
        return sklearn.metrics.log_loss(self.true, self.scores)
        
    def compute_f1scores(self):
        
        micro, macro, weighted, none = [sklearn.metrics.f1_score(self.true, self.predicted, average=method, labels=self.labels) for method in ['micro', 'macro', 'weighted', None]]
        none = pd.Series(none, index = self.labels, name = 'class f1 score')
                 
        return [micro, macro, weighted, none]
    
    def compute_ROC(self, fpr_interp = np.linspace(0, 1, 100)):
        def ROC(x):
            
            fpr, tpr, _ = sklearn.metrics.roc_curve((self.true==x.name)*1, x)
            fpr = pd.Series(fpr, name = 'FPR')
            tpr = pd.Series(tpr, name = 'TPR')
           
            return (pd.concat([fpr, tpr], names=[x.name], keys = ['FPR', 'TPR']))
        
        
        rates = self.scores.apply(ROC)
        all_fpr = rates.loc['FPR', :].stack().unique()
        mean_tpr = rates.apply(lambda x: pd.Series(interp(all_fpr, x['FPR'], x['TPR']), index=all_fpr)).sort_index()
        mean_tpr.index.names = ['x']
        
        all_rates = rates.stack().unstack(0).sort_values(['FPR'])
        auc_micro = sklearn.metrics.auc(all_rates['FPR'], all_rates['TPR'])
        
        macro = mean_tpr.mean(axis=1).sort_index()
        auc_macro = sklearn.metrics.auc(macro.index.values, macro)
        auc_none = pd.Series([sklearn.metrics.auc(mean_tpr.reset_index()['x'], mean_tpr[col]) for col in mean_tpr.columns], name = 'class auc', index = mean_tpr.columns)
            
        return [mean_tpr, auc_macro, auc_micro, auc_none]        
    
    def compute_MCC(self):
        return sklearn.metrics.matthews_corrcoef(self.true, self.predicted)
    
    def compute_confusionmatrix(self):
        cm = sklearn.metrics.confusion_matrix(self.true, self.predicted, labels=self.labels)
        cm = pd.DataFrame(cm, index=self.labels, columns=self.labels)
        cm = cm/cm.sum(axis=1)
        
        return cm                                                                 
                                                                     
    def performance_report(self, logloss = True, f1score = True, ROC = True, MCC = True, **kwargs):
        '''compute common ML performance metrics, can be customized to perform only certain metrics calculations'''
        
        float_results = {}
        
        if logloss:
            float_results['log loss'] = self.compute_logloss(**kwargs)
            
        if f1score:
            micro, macro, weighted, none = self.compute_f1scores(**kwargs)
            float_results['micro f1'] = micro
            float_results['macro f1'] = macro
            float_results['weighted f1'] = weighted
            
        if ROC:
            mean_tpr, auc_macro, auc_micro, auc_none = self.compute_ROC(**kwargs)
            float_results['macro auc'] = auc_macro
            float_results['micro auc'] = auc_micro
            
        if MCC:
            mcc = self.compute_MCC(**kwargs)
            float_results['Matthews corellation coefficient'] = mcc
            
        float_results = pd.Series(float_results, name = 'x').sort_index()
        
        a = pd.concat([none, auc_none], axis=1, sort = True).T
        b = pd.concat([a, float_results], sort=True)
        b.rename(columns = {0:'x'}, inplace=True)
        c = pd.concat([mean_tpr.reset_index(level='x'), b], keys=['ROC', 'float values'], sort=True)
        c.index.names = ['metric type', 'index']
        
        return c
