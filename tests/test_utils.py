import unittest
import os
import itertools

import pandas as pd
import numpy as np

import TRANSPIRE.utils
import TRANSPIRE.data 

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.data = TRANSPIRE.data.import_data.load_data(os.path.join(THIS_DIR, 'test_files/test_Gilbertson2018.csv'))
        self.comparisons = [('{}_{}'.format(cA, r), '{}_{}'.format(cB, r)) for cA, cB in list(itertools.combinations(['D219A', 'WT'], 2)) for r in [1]]
        self.translocations, self.mapping, self.mapping_r = TRANSPIRE.data.generate_translocations.make_translocations(self.data, self.comparisons)

        self.preds = pd.DataFrame(np.random.rand(self.translocations.shape[0], self.mapping.shape[0]), index = self.translocations.index, columns = range(self.mapping.shape[0]))
        self.preds = self.preds.apply(lambda x: x/self.preds.sum(axis=1))

    def test_organelle_group_mapping(self):

        mapping = {'Nucleus': 'Nucleus/Cytoplasm', 'Cytosol': 'Nucleus/Cytoplasm'}
        TRANSPIRE.utils.group_organelles(self.data, mapping)

    def test_train_test_validate_split(self):

        f_train = 0.7
        f_validate = 0.2
        f_test = 0.1

        TRANSPIRE.utils.train_test_validate_split(self.translocations, ['condition_A', 'condition_B', 'label'], f_train, f_test, f_validate)

    def test_train_test_validate_split_bad_fractions(self):

        f_train = 0.5
        f_validate = 0.2
        f_test = 0.1

        with self.assertRaises(ValueError):
            TRANSPIRE.utils.train_test_validate_split(self.translocations, ['condition_A', 'condition_B', 'label'], f_train, f_test, f_validate)

    def test_map_binary(self):
        TRANSPIRE.utils.map_binary(self.preds, self.mapping_r)

    def test_lookup_success(self):
        in_df = TRANSPIRE.utils.lookup('ACOT7', self.data, 'gene name')

        self.assertIsNotNone(in_df)

    def test_lookup_fail(self):
        not_in_df = TRANSPIRE.utils.lookup('fdahfdas', self.data, 'gene name')

        self.assertIsInstance(not_in_df, pd.DataFrame)
        self.assertEqual(not_in_df.shape[0], 0)

    def test_uniprot_mapping(self):
        df_gene = TRANSPIRE.utils.uniprot_mapping_service(['Q92614'], 'gene')
        df_string = TRANSPIRE.utils.uniprot_mapping_service(['Q92614'], 'string_db')

        self.assertIsInstance(df_string, pd.DataFrame))
        self.assertIsInstance(df_gene, pd.DataFrame))

        self.assertEqual(df_gene.values[0], 'MYO18A')
        self.assertEqual(df_string.values[0], '9606.ENSP00000437073')
