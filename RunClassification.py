import ModelTraining
import pandas as pd
import os
import gpflow
from sklearn.cluster import MiniBatchKMeans

class Classify:
    def __init__(self, x, clf_type = None, **kwargs):

        # reset the session and graph to avoid possible errors
        gpflow.reset_default_graph_and_session()

        self.clf_type = clf_type

        if self.clf_type == 'binary':
            print('classification type is binary')
            self.label = 'binary label'
        
        elif self.clf_type == 'multi class':
            print('classification type is multi class')
            self.label = 'multi class label'

        else:
            raise ValueError('provide a valid classification type')

        self.CV = ModelTraining.StratifiedCV()
        self.clf = self.classify(x, **kwargs)
        print()
        print('\t-----> generating metrics report ')
        self.reports = self.assess_clfs(self.clf)        

    def classify(self, x, unk=None, **kwargs):
        LEC = ModelTraining.Encoder()
        LEC.fit(x.index.get_level_values(self.label).unique())
        y = pd.Series(LEC.transform(x.index.get_level_values(self.label)), index=x.index, name='encoded label')
        
        GPC = ModelTraining.GPFlowGPC(**kwargs)
        if GPC.n_induce > x.shape[0]:
            print('The number of inducing points cannot be greater than the number of samples; setting n_induce to {}\nDepending on the size of the dataset, this may result in extremely long run times or memory exhaustion'.format(x.shape[0]))
            # pre-compute memory requirements and availability?
            GPC = ModelTraining.GPFlowGPC(n_induce = x.shape[0], **kwargs)

        if x.shape[0]>10000:
            print('\t-----> large dataset, pre-computing kmeans for minibatching . . .', end = ' ')
            self.minibatch = MiniBatchKMeans(GPC.n_induce).fit(x.values)
            print('Finished initial kmeans fit')

        else:
            self.minibatch = None
        
        print('\t-----> training model: ', end='')
        for (X_train, X_test), (y_train, y_test) in self.CV.return_splits(x, y):
            
            model_path = os.path.join(GPC.root, 'Models', '{} fold {}'.format(GPC.name, GPC.fold+1))
            
            if os.path.exists(model_path): # model has already been fitted, load it from memory
                GPC.load_model(model_path)
            
            else:
                GPC.train_GPC(X_train, y_train, minibatch = self.minibatch)

            GPC.evaluate_performance(X_test, converter = LEC, type='test')

            if unk is not None:
                GPC.evaluate_performance(unk, converter = LEC, type='unknown')
            
            gpflow.reset_default_graph_and_session()

        return GPC

    def assess_clfs(self, clf):
        res = {}
        for fold in clf.test_results:
            df = clf.test_results[fold]       
            metrics = ModelTraining.PerformanceMetrics(df.reset_index()[self.label], df['score'], df['uncertainty'])
            res[fold] = metrics.performance_report()
        return res

class ExportResults:
    def __init__(self, root = os.getcwd(), **kwargs):
        self.savedir = os.path.join(root, 'Results')
        
        if not os.path.exists(self.savedir):
            os.mkdir(self.savedir)
    
    def save(self, s):
        test_results = pd.concat([pd.concat(clf.clf.test_results, names=['fold']) for clf in s], keys=s.index, names=s.index.names, sort=True)
        test_results.to_csv(os.path.join(self.savedir, 'test_results.zip'))

        unk_results = pd.concat([pd.concat(clf.clf.unk_results, names=['fold']) for clf in s], keys=s.index, names=s.index.names, sort=True)
        unk_results.to_csv(os.path.join(self.savedir, 'unknown_results.zip'))

        report = pd.concat([pd.concat(clf.reports, names=['fold']) for clf in s], keys=s.index, names=s.index.names, sort=True)
        report.to_csv(os.path.join(self.savedir, 'metrics_report.zip'))
