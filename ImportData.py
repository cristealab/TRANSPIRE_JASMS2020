import pandas as pd
import numpy as np

import os
import itertools

class TrainingData:
    def __init__(self, comparisons=None, train_type='subtractive'):
        self.type = train_type
        self.comparisons = comparisons # dict of columns to compare
    
    def generate_training_data(self, df, localizations, interval=0.1, start = 0.1, translocations = None):

        training_data = {}

        if translocations == True:
            for c in self.comparisons:
                data = df.stack()[c]
                from_ = data.iloc[:, 0].dropna().unstack().index
                to_ = data.iloc[:, 1].dropna().unstack().index
                prot_combos = pd.MultiIndex.from_tuples(list(itertools.product(from_, to_)))

                f = data.iloc[:, 0].unstack().loc[prot_combos.get_level_values(0), :].values
                t = data.iloc[:, 1].unstack().loc[prot_combos.get_level_values(1), :].values
                
                if self.type == 'subtractive':
                    delta = pd.DataFrame((t-f), index = prot_combos)
                
                elif self.type == 'ratio':
                    delta = pd.DataFrame((t/f), index = prot_combos)

                elif self.type == 'additive':
                    delta = pd.DataFrame(np.hstack([f, t]), index = prot_combos)
                
                delta.index.names = ['from marker', 'to marker']

                training_data['from {} to {}'.format(c[0], c[1])] = delta
                
            training_data = pd.concat(training_data, names=['comparison'])

            from_ = localizations[training_data.index.get_level_values('from marker')]
            to_ = localizations[training_data.index.get_level_values('to marker')]
            
            training_data['multi class label'] = from_.str.cat(to_.values, sep = ' to ').values
            training_data['binary label'] = np.where(from_.values==to_.values, 'No translocation', 'Translocation')
            training_data = training_data.reset_index().set_index(['comparison', 'binary label', 'multi class label', 'from marker', 'to marker'])

            intervals = np.linspace(start, 1, int(((1-start)/interval)+1))
            dfs = {}
            df = training_data[training_data.index.get_level_values('binary label')=='Translocation']

            for i in intervals:
                if i<1:
                    dfs[i] = df*i

                else:
                    dfs[i] = training_data*i

            training_data = pd.concat(dfs, copy=False, keys=intervals, names=['interval'])
            
            training_data.columns =  range(training_data.shape[1])

            training_data = training_data.reorder_levels(['comparison', 'binary label', 'multi class label', 'interval', 'from marker', 'to marker'])
            
        elif translocations == False : # no need to generate additional training data
    
            for c in self.comparisons:
                data = pd.DataFrame(np.hstack([df.stack()[c[0]].unstack(), df.stack()[c[1]].unstack()]), index = df.index).dropna()

                data['multi class label'] = localizations[data.index]
                data['binary label'] = ''
                data['to marker'] = data.index.values
                data.index.names = ['from marker']
                data['interval'] = 1.0

                training_data['from {} to {}'.format(c[0], c[1])] = data.reset_index().set_index(['binary label', 'multi class label', 'interval', 'from marker', 'to marker'])
        
            training_data = pd.concat(training_data, names=['comparison'])

        training_data.columns = range(training_data.shape[1])

        return training_data.sort_index()

class JeanBeltran2016:
    def __init__(self, interval=0.1, start=0.1, translocation_training=True, localization_training=True, **kwargs):

        self.directory = r'Z:\2 Programming\1 Translocation program\JeanBeltran2016'

        print ('loading data . . .')
        comparisons, df, markers, genenames = self.load_data()
        df.columns.names = ['condition', 'fraction']
        self.localization_data_ = df.copy()
    
        # impute values within lower 1-5% for zeros
        n = df.stack(['condition', 'fraction'])[df.stack(['condition', 'fraction'])!=0].describe([0.01, 0.05])
        np.random.seed(0)
        rands = np.random.uniform(n['1%'], n['5%'], size=df.shape)
        df = df.where(df != 0, rands)

        self.markers = markers
        self.genenames = genenames
        self.comparisons = comparisons
        self.translocation_data = None

        translocation_data = {}
        localization_data = {}

        if not 'train_type' in kwargs:
            train_type = 'subtractive'
        else:
            train_type = kwargs['train_type']
        
        if train_type == 'ratio':
            for c in comparisons:
                translocation_data['from {} to {}'.format(c[0], c[1])] = (df[c[1]]/df[c[0]].values).dropna()

        elif train_type == 'additive':
            for c in comparisons:
                translocation_data['from {} to {}'.format(c[0], c[1])] = pd.concat([df[c[1]], df[c[0]]], axis=1).dropna()
        
        elif train_type == 'subtractive':
            for c in comparisons:
                translocation_data['from {} to {}'.format(c[0], c[1])] = df[c[1]].subtract(df[c[0]].values).dropna()
       
        for c in comparisons:
            localization_data['from {} to {}'.format(c[0], c[1])] = pd.DataFrame(np.hstack([df[c[0]], df[c[1]]]), index = df.index).dropna()

        self.translocation_data = pd.concat(translocation_data, axis=1, names = ['comparison'])
        self.localization_data = pd.concat(localization_data, axis=1, names = ['comparison'])

        markers_df = df[df.index.isin(markers.index)]
        localizations = markers.loc[markers_df.index, 'Subcellular localization']

        trainingdata = TrainingData(comparisons, **kwargs)

        if translocation_training:
            print('generating training data . . .')
            self.translocation_training = trainingdata.generate_training_data(markers_df, localizations, interval=interval, start = start, translocations=True)
            print('\t translocation characterization training data generated')

        if localization_training:
            self.localization_training = trainingdata.generate_training_data(markers_df, localizations, translocations=False)
            print('\t localization assignment training data generated')

        print('Finished.')

    def load_data(self, directory = "Z:\\2 Programming\\1 Translocation program\\Core data\\Pierre's Data"):
    
        profiles = []

        for root, dirs, filenames in os.walk(directory):
            for file in filenames:
                if 'Profiles' in file:
                    title = file.split('.TMT.')[1].split('.csv')[0]
                    df = pd.read_csv(os.path.join(root, file), index_col=0, header=0)
                    t = '{} {}hpi'.format(title.split('.h')[0], title.split('.h')[1])
                    df.columns = pd.MultiIndex.from_tuples([(t, i) for i in range(df.shape[1])])
                    profiles.append(df)

                elif 'localization' in file:
                    loc_df = pd.read_csv(os.path.join(root, file), sep = None, engine = 'python', header = [0, 1], index_col = 0)

        prof_df = pd.concat(profiles, axis = 1, sort=True)

        ##############################################################################################

        # import the organelle markers

        file = "Z:\\2 Programming\\1 Translocation program\\Core data\\markers.csv"

        markers_df = pd.read_csv(file, sep= ',', header = 0, index_col = 0)
        markers_df = markers_df[markers_df.index.isin(prof_df.index)]
        markers_df.loc[markers_df[markers_df['Subcellular localization']=='Cytosol'].index, 'Subcellular localization'] = 'Cytoplasm' 

        matched_combos = [['MOCK 24hpi', 'HCMV 24hpi'], 
                        ['MOCK 48hpi', 'HCMV 48hpi'],
                        ['MOCK 72hpi', 'HCMV 72hpi'],
                        ['MOCK 96hpi', 'HCMV 96hpi'],
                        ['MOCK 120hpi', 'HCMV 120hpi']]

        sequential = [['HCMV 24hpi', 'HCMV 48hpi'],
                    ['HCMV 48hpi', 'HCMV 72hpi'],
                    ['HCMV 72hpi', 'HCMV 96hpi'],
                    ['HCMV 96hpi', 'HCMV 120hpi']]


        _combos = matched_combos + sequential

        # re-normalize the data
        stacked = prof_df.stack(0)
        normed_df = stacked.apply(lambda x: x/stacked.sum(axis=1)).unstack([-1]).swaplevel(axis=1).sort_index(axis=1)

        genename_df = loc_df.swaplevel(axis=1)[['Gene', 'Organism']]
        genename_df.columns = ['gene name', 'organism']

        return [_combos, normed_df, markers_df, genename_df]

class Christoforou2015:
    def __init__(self):
        self.directory = r'Z:\2 Programming\1 Translocation program\Christoforou2015'
        
        datapath = os.path.join(self.directory, 'hyperLOPIT2015_raw.csv')
        markerspath = os.path.join(self.directory, 'hyperLOPIT2015_markers.csv')
        TAGMpath = os.path.join(self.directory, 'hyperLOPIT2015_TAGM.csv')
        
        self.localization_data = pd.read_csv(datapath, index_col=0, header = 0)
        self.TAGM = pd.read_csv(TAGMpath, index_col=0, header = 0)
        
        self.markers = pd.read_csv(markerspath, index_col=0)['markers2015']
        self.markers = self.markers[self.markers!='unknown'].dropna()
        markers_df = self.localization_data.loc[self.markers.index, :].dropna()

        self.localization_training = TrainingData(train_type='localization assignment').generate_training_data(markers_df, self.markers)
        
        