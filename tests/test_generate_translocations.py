import unittest
import os
import itertools

from TRANSPIRE import data

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TestMakeTranslocations(unittest.TestCase):
    
    def setUp(self):
        self.data = data.import_data.load_data(os.path.join(THIS_DIR, 'test_files/test_Gilbertson2018.csv'))
        self.comparisons = [('{}_{}'.format(cA, r), '{}_{}'.format(cB, r)) for cA, cB in list(itertools.combinations(['D219A', 'WT', 'muSOX'], 2)) for r in [1, 2, 3]]

    def test_make_synthetic(self):
        data.generate_translocations.make_translocations(self.data, self.comparisons, synthetic = True)

    def test_make_actual(self):
        data.generate_translocations.make_translocations(self.data, self.comparisons, synthetic = False)
