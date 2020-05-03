
import unittest
import os
import itertools
import pandas as pd
import numpy as np

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)

import gpflow

import TRANSPIRE.training

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TestSVGP(unittest.TestCase):
    def setUp(self):

        data = TRANSPIRE.data.import_data.load_data(os.path.join(THIS_DIR, 'test_files/test_Gilbertson2018.csv'))
        comparisons = [('{}_{}'.format(cA, r), '{}_{}'.format(cB, r)) for cA, cB in list(itertools.combinations(['D219A', 'WT'], 2)) for r in [1]]
        
        translocations, mapping, mapping_r = TRANSPIRE.data.generate_translocations.make_translocations(data, comparisons)

        self.X = translocations.values
        self.y = translocations.index.get_level_values('label').map(mapping).values.reshape(-1, 1)

    def test_default_build(self):

        m=TRANSPIRE.training.build_model(self.X, self.y)
        gpflow.training.AdamOptimizer(learning_rate=0.5).minimize(m, maxiter=1)
        gpflow.reset_default_graph_and_session()

    def test_custom_build(self):

        m=TRANSPIRE.training.build_model(self.X, self.y, **{'Z': self.X[:10, :], 'kernel': gpflow.kernels.Matern12(self.X.shape[1], ARD=True)})
        gpflow.training.AdamOptimizer(learning_rate=0.5).minimize(m, maxiter=0)
        gpflow.reset_default_graph_and_session()

    def test_manual_train(self):

        m = TRANSPIRE.training.build_model(self.X, self.y)
        gpflow.training.AdamOptimizer(learning_rate=0.5).minimize(m, maxiter=1)
        gpflow.reset_default_graph_and_session()

    def test_minibatch(self):

        m = TRANSPIRE.training.build_model(self.X, self.y, **{'minibatch_size': 0.1})
        gpflow.training.AdamOptimizer(learning_rate=0.5).minimize(m, maxiter=0)
        gpflow.reset_default_graph_and_session()

    def test_minibatch_SVGP_fail(self):
        
        with self.assertRaises(AssertionError):
            TRANSPIRE.training.build_model(self.X, self.y, **{'minibatch_size': 0.1, 'weights': np.ones(self.y.shape)})
            gpflow.reset_default_graph_and_session()

    def test_bad_weights(self):

        weights = np.ones((10, 1))
        with self.assertRaises(AssertionError):
            TRANSPIRE.training.build_model(self.X, self.y, **{'weights': weights})
            gpflow.reset_default_graph_and_session()

    def test_load_saved_model(self):
        saved_model_path = os.path.join(THIS_DIR, 'test_files', 'test_SVGP')

        m = gpflow.saver.Saver().load(saved_model_path)
        
        self.assertIsInstance(m, TRANSPIRE.training.WeightedSVGP)

        m.predict_y(np.random.rand(10, m.feature.Z.shape[1]))
        gpflow.reset_default_graph_and_session()