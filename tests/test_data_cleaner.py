import unittest
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join('../scripts/')))

from data_cleaner import Clean_Data
utils = Clean_Data()


class TestClean_Data(unittest.TestCase):
    """
    A class for unit-testing functiosns in the data_cleaner.py file.
s
    """
    def setUp(self):
        self.df = pd.read_csv('../data/store.csv').head(500)
        return self.df
         
    def test_read_data(self):
        self.assertIsInstance( utils.read_data('../data/test.csv'), pd.DataFrame)

    def test_fill_null(self):
        utils.fill_null('Promo2SinceWeek',self.df,0)
        val = self.df['Promo2SinceWeek'].isna().sum()
        self.assertEqual(val,0,msg='Column has null values')


if __name__ == '__main__':
	unittest.main()

    


