#!/usr/bin/env python
# coding: utf-8

# # Importing Library

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import mlflow
import mlflow.sklearn


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# # Initiating Logging

# In[3]:


import logging
logging.basicConfig(filename='modellingSteps.log', level=logging.INFO)
logging.info('This log file records the steps for modelling for this project')


# In[4]:


def log(text:str):
    logging.info(text)


# # Importing Scripts

# In[5]:


import sys, os 
sys.path.append(os.path.abspath(os.path.join('../scripts/')))
log('Import utility scripts')
from data_cleaner import *


# In[6]:


util = Clean_Data()


# # Importing Data

# In[7]:


log('Import Train Dataset to db')


# In[8]:


db = util.read_data('../data/trainData.csv')
db.head(2)


# In[9]:


db.isna().sum()


# In[10]:


db.dropna(axis=0,inplace=True)


# In[11]:


log('Import Test Dataset to test_db')


# In[12]:


test_db = util.read_data('../data/testData.csv')
test_db.head(2)


# In[13]:


mlflow.set_experiment('Sales Prediction')


# # Data Preprocesssing:

# In[14]:


log('Commence Data Processing:')


# In[15]:


log('Convert Date columns in db and test_db to Datetime type')
util.to_datetime(test_db,'Date','%Y-%m-%d')
util.to_datetime(db,'Date','%Y-%m-%d')


# In[16]:


def get_date_features(df,col):
    df['Year'] = df[col].dt.year
    df['Month'] = df[col].dt.month
    df['Day'] = df[col].dt.day
    df.drop(columns=[col], axis=1,inplace=True)


# In[17]:


log('Get date features from date column in db and test_db')
get_date_features(test_db,'Date')
get_date_features(db,'Date')
test_db.head(2)


# # Extracting feature columns:

# In[18]:


log('Extract feature columns from db to train_db:')


# In[19]:


log('Drop Columns:"Date,Customers,Sales,PromoInterval,CompetitionOpenSinceYear,CompetitionOpenSinceMonth"')
train_db = db.drop(columns=['Customers','Sales',
                           'PromoInterval'], axis=1)
# 'CompetitionOpenSinceYear','CompetitionOpenSinceMonth'


# In[20]:


train_db.head(2)


# In[21]:


log('Drop columns:"Id,PromoInterval,CompetitionOpenSinceYear,CompetitionOpenSinceMonth" in test_db:')
test_db = test_db.drop(columns=['Id','PromoInterval',
                                'CompetitionOpenSinceYear',
                                'CompetitionOpenSinceMonth'], axis=1)


# In[22]:


test_db.head(2)


# In[23]:


log('Extract Sales column from db into y for target variable')
y = db[['Sales']]
y.head(2)


# In[24]:


data_url = '../data/train.csv'
version = 'v1'
mlflow.log_param('data_url', data_url)
mlflow.log_param('data_version', version)
mlflow.log_param('input_rows',train_db.shape[0])
mlflow.log_param('input_cols', train_db.shape[1])

cols_x = pd.DataFrame(list(train_db.columns))
cols_x.to_csv('features.csv', header=False, index=False)
mlflow.log_artifact('features.csv')


cols_y = pd.DataFrame(list(y.columns))
cols_y.to_csv('target.csv', header=False, index=False)
mlflow.log_artifact('target.csv')


# # Encoding categorical data

# In[25]:


log('Encode Categorical variables in train_db and test_db:')


# In[26]:


def encode(df,columns):
    df = pd.get_dummies(df,columns=columns)
    return df


# In[27]:


log('Encode Columns:"StoreType,Assortment,StateHoliday" in train_db.')
train_db = encode(train_db,['StoreType','Assortment','StateHoliday'])


# In[28]:


log('Encode Columns:"StoreType,Assortment" in test_db.')
test_db = encode(test_db,['StoreType','Assortment'])


# # Scaling

# In[29]:


log('Start Feature Scaling:')
log('Import StandardScaler from sklearn.preprocessing')
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()


# In[30]:


log('Scale train_db into X')
X = sc_X.fit_transform(train_db)


# # Split data into train and test set

# In[31]:


log('Split data into train and test set:')
log('Import train_test_split from sklearn.model_selection')
log('Split X into X_train, X_test and y into Y_train and Y_test in ratio 80:20')
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)


# # Train model

# In[32]:


log('Train model:')
log('Import Linear Regression from sklearn.linear_model')
log('Create instance of Linear Regression in regressor')
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[33]:


log('Fit regressor to X_train,Y_train')
regressor.fit(X_train, Y_train)


# In[34]:


log('Predict Y_pred from X_test using regressor.predict')
Y_pred = regressor.predict(X_test)


# # Check Model Accuracy

# In[35]:


log('Check Model Accuracy:')


# In[36]:


log('Import rmse,mae,rs2 libraries from sklearn.metrics')
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# In[37]:


log('Compute rmse,mae,rs2 values')


# In[38]:


def eval_metrics(actual,pred):
    """"
    Function to obtain rmse,mae,r2 values for actual and predicted values
    """
    rmse = np.sqrt(mean_squared_error(actual,pred))
    mae = mean_absolute_error(actual,pred)
    r2 = r2_score(actual,pred)
    return rmse,mae,r2


# In[39]:


print(f'Sales prediction model: \n \t RMSE:{eval_metrics(Y_test,Y_pred)[0]} \n \t MAE:{eval_metrics(Y_test,Y_pred)[1]} \n \t R2:{eval_metrics(Y_test,Y_pred)[2]}')


# In[40]:


mlflow.log_metrics({'RMSE':eval_metrics(Y_test,Y_pred)[0],                   'MAE':eval_metrics(Y_test,Y_pred)[1],                   'R2':eval_metrics(Y_test,Y_pred)[2]})

