import unittest
import os

import TRANSPIRE.data

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TestDataImport(unittest.TestCase):

    def test_load_csv(self):
        TRANSPIRE.data.import_data.load_data(os.path.join(THIS_DIR, 'test_files/test_Gilbertson2018.csv'))
    
    def test_load_txt(self):
        TRANSPIRE.data.import_data.load_data(os.path.join(THIS_DIR, 'test_files/test_Gilbertson2018.txt'))

    def test_load_excel(self):
        TRANSPIRE.data.import_data.load_data(os.path.join(THIS_DIR, 'test_files/test_Gilbertson2018.xlsx'))

    def test_bad_import_structure(self):
        with self.assertRaises(ValueError):
            TRANSPIRE.data.import_data.load_data(os.path.join(THIS_DIR, 'test_files/test_Gilbertson2018_missing_genenames.csv'))

class TestMarkerImport(unittest.TestCase):
    def test_load_marker_sets(self):

        marker_dir = os.path.join(THIS_DIR.split('tests')[0], 'TRANSPIRE', 'data', 'external', 'organelle_markers')

        for marker_set in os.listdir(marker_dir):
            marker_set_name = marker_set.split('.')[0]
            TRANSPIRE.data.import_data.load_organelle_markers(marker_set_name)

class TestAddMarkers(unittest.TestCase):

    def setUp(self):
        self.data = TRANSPIRE.data.import_data.load_data(os.path.join(THIS_DIR, 'test_files/test_Gilbertson2018.csv'))
        self.markers = TRANSPIRE.data.import_data.load_organelle_markers('HEK293T')

    def test_add_redundant_markers(self):
        with self.assertRaises(ValueError):
            self.new_data = TRANSPIRE.data.import_data.add_markers(self.data, self.markers)
        
    def test_add_markers(self):
        TRANSPIRE.data.import_data.add_markers(self.data.reset_index('localization', drop=True), self.markers)

if __name__ == '__main__':
    unittest.main()