import pandas as pd

class Clean_Data:
    """
    The PEP8 Standard AMAZING!!!
    """
    def __init__(self):
        # self.df = df
        print('Utility Functions Imported!!!')

    def read_data(self, path):
        data = pd.read_csv(path)
        return data
        
    def drop_duplicate(self, df:pd.DataFrame)->pd.DataFrame:
        """
        drop duplicate rows
        """
        df.drop_duplicates(inplace=True)
        return df
    
    def fill_null(self,column:str,df:pd.DataFrame,value):
    	"""
    	fill null values with specified value
    	"""
    	df[column] = df[column].fillna(value)

    def to_datetime(self, df:pd.DataFrame, col, format):
        """
        convert column to datetime
        """
        df[col] = pd.to_datetime(df[col],format=format)

    
    def convert_type(self, df:pd.DataFrame,col:str,type)->pd.DataFrame:
        """
        convert column data type to str or int
        """
        df[col] = df[col].astype(type)
    
