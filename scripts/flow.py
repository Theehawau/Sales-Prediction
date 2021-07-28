# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Get Data from DVC remote
# import dvc.api


# %%
import numpy as np 
import pandas as pd
#import matplotlib.pyplot as plt 
#import seaborn as sns
import mlflow
import mlflow.sklearn


# %%
# path = 'data/train.csv'
# repo = 'git@github.com:Theehawau/Sales-Prediction.git'
# version = 'v1'
# data_url = dvc.api.get_url(
#     path=path,
#     repo=repo,
#     rev=version
# )


# %%
# Read the train data from remote repo
data_url = '../data/train.csv'
version = 'v1'
train_db = pd.read_csv(data_url, sep=',')
train_db.head(2)


# %%
mlflow.set_experiment('Sales Prediction')


# %%
mlflow.log_param('data_url', data_url)
mlflow.log_param('data_version', version)
mlflow.log_param('input_rows',train_db.shape[0])
mlflow.log_param('input_cols', train_db.shape[1])


# %%
cols = pd.DataFrame(list(train_db.columns))
cols.to_csv('features.csv', header=False, index=False)
mlflow.log_artifact('features.csv')


# %%



# %%



