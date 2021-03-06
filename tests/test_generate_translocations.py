import unittest
import os
import itertools

import pandas as pd
import numpy as np

import TRANSPIRE.data

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TestMakeTranslocations(unittest.TestCase):
    
    def setUp(self):
        self.data = TRANSPIRE.data.import_data.load_data(os.path.join(THIS_DIR, 'test_files/test_Gilbertson2018.csv'))
        self.comparisons = [('{}_{}'.format(cA, r), '{}_{}'.format(cB, r)) for cA, cB in list(itertools.combinations(['D219A', 'WT'], 2)) for r in [1]]

    def test_make_synthetic(self):
        translocations = TRANSPIRE.data.generate_translocations.make_translocations(self.data, self.comparisons, synthetic = True)
        self.assertIsInstance(translocations, pd.DataFrame)   

    def test_make_actual(self):
        translocations = TRANSPIRE.data.generate_translocations.make_translocations(self.data, self.comparisons, synthetic = False)
        self.assertIsInstance(translocations, pd.DataFrame)
        
 
if __name__ == '__main__':
    unittest.main()
