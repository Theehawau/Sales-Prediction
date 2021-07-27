import pandas as pd

class Clean_Data:
    """
    The PEP8 Standard AMAZING!!!
    """
    def __init__(self, df:pd.DataFrame):
        self.df = df
        print('Automation in Action...!!!')

    def read_data(self, path):
        data = pd.read_csv(path)
        return data
        
    def drop_duplicate(self, df:pd.DataFrame)->pd.DataFrame:
        """
        drop duplicate rows
        """
        df.drop_duplicates(inplace=True)
        return df

    def to_datetime(self, df:pd.DataFrame, col, format):
        """
        convert column to datetime
        """
        df[col] = pd.to_datetime(df[col],format=format)

    
    def convert_type(self, df:pd.DataFrame,col,type)->pd.DataFrame:
        """
        convert columns like polarity, subjectivity, retweet_count
        favorite_count etc to numbers
        """
        df.astype({col:type})
    
