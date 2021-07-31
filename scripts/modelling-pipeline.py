#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np 
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import mlflow


# In[27]:


import warnings
warnings.filterwarnings('ignore')


# In[28]:


import sys, os 
sys.path.append(os.path.abspath(os.path.join('../scripts/')))
# Import utility scripts
from data_cleaner import *


# In[29]:


util = Clean_Data()


# In[ ]:


mlflow.set_experiment('Sales Prediction')


# In[30]:


def get_date_features(df,col):
    df[col] = pd.to_datetime(df[col])
    df['Year'] = df[col].dt.year
    df['Month'] = df[col].dt.month
    df['Day'] = df[col].dt.day
    df['WeekOfYear'] = df[col].dt.weekofyear


# In[31]:


store_db = util.read_data('../data/store.csv')
train_db = util.read_data('../data/train.csv')


# In[35]:


train_db.shape


# In[34]:


store_db.shape


# In[7]:


store_db.fillna(0, inplace=True)


# In[39]:


db = pd.merge(left=train_db,right=store_db,on='Store',how='inner')


# In[9]:


# Process data for dashboard
train = db[(db.Open != 0) & (db.Sales>0)]
def process(df):
    
#     #Replacing null values for CompetitionOpenSinceMonth,CompetitionOpenSinceYear
#     util.fill_null('CompetitionOpenSinceMonth',df,df['CompetitionOpenSinceMonth'].mean())
#     util.fill_null('CompetitionOpenSinceYear',df,df['CompetitionOpenSinceYear'].mean())
#     util.fill_null('CompetitionDistance',df,df['CompetitionDistance'].mean())
#     df.fillna(0, inplace=True)
    
    #label encode categorical_features  
    mapping = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
    df.StoreType.replace(mapping, inplace=True)
    df.Assortment.replace(mapping, inplace=True)
    
    df['IsHoliday'] = df.StateHoliday.map({'0':0,'a':1,'b':1,'c':1,0:0})
    df['IsWeekend'] = df.DayOfWeek.map({6:1,7:1,1:0,2:0,3:0,4:0,5:0})
    
    #Get date features  
    get_date_features(df, 'Date')
    
    #Calculate competitor open time in months
    df['CompetitionOpenMonths'] = 12 * (df.Year - df.CompetitionOpenSinceYear) +     (df.Month - df.CompetitionOpenSinceMonth)
    df['CompetitionOpenMonths'] = df['CompetitionOpenMonths'].apply(lambda x: x if x > 0 else 0)

    # calculate promo2 open time in months
    df['Promo2OpenMonths'] = 12 * (df.Year - df.Promo2SinceYear) +         (df.WeekOfYear - df.Promo2SinceWeek) / 4.0
    df['Promo2OpenMonths'] = df['Promo2OpenMonths'].apply(lambda x: x if x > 0 else 0)
    
    #Check if month in promo2 month
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',              7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    df['month_str'] = df.Month.map(month2str)
    def check(row):
        if isinstance(row['PromoInterval'],str) and row['month_str'] in row['PromoInterval']:
            return 1
        else:
            return 0
        
    df['IsPromoMonth'] =  df.apply(lambda row: check(row),axis=1)  
    # select the features we need
    features = ['Store', 'DayOfWeek', 'Promo', 'IsHoliday', 'SchoolHoliday',
       'StoreType', 'Assortment', 'CompetitionDistance',
       'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
       'Promo2SinceWeek', 'Promo2SinceYear', 'Year', 'Month', 'Day',
       'WeekOfYear', 'CompetitionOpenMonths', 'Promo2OpenMonths', 'IsPromoMonth','IsWeekend']
    X = df[features]
    y = df[['Sales']]
#     feature = df[features]
#     target = df[['Sales','Date']]
#     feature.reset_index().drop(columns=['index'], inplace =True)
#     feature.to_csv('../Data/X.csv',index=False)
#     target.reset_index().drop(columns=['index'], inplace =True)
#     target.to_csv('../Data/Y.csv',index=False)
    return X,y
    
X,y = process(train)


# In[10]:


y.reset_index().drop(columns=['index'], inplace =True)
X.reset_index().drop(columns=['index'], inplace =True)


# In[ ]:


data_url = '../data/X.csv'
version = 'v2'
mlflow.log_param('data_url', data_url)
mlflow.log_param('data_version', version)
mlflow.log_param('input_rows',X.shape[0])
mlflow.log_param('input_cols', X.shape[1])

cols_x = pd.DataFrame(list(X.columns))
cols_x.to_csv('features.csv', header=False, index=False)
mlflow.log_artifact('features.csv')


cols_y = pd.DataFrame(list(y.columns))
cols_y.to_csv('target.csv', header=False, index=False)
mlflow.log_artifact('target.csv')


# # Splitting to test and train sets

# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X = sc_X.fit_transform(X)

# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[12]:


# try random forest
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators = 15)
clf.fit(X_train, y_train)


# In[14]:


# validation
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
y_pred = clf.predict(X_test)

print(np.sqrt(mean_squared_error(y_test, y_pred)))
print(r2_score(y_test, y_pred))


# In[ ]:


mlflow.log_metrics({'RMSE':np.sqrt(mean_squared_error(y_test, y_pred)),
                    'R2':r2_score(y_test, y_pred)})


# In[ ]:


feat_importances = pd.Series(clf.feature_importances_, index=features)
feat_importances.sort_values(ascending = True).plot(kind='barh')
plt.xlabel('importance')
plt.title('Feature Importance')


# # Preprocessor Using Pipeline

# In[ ]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.to_list()

# numeric_transformer = Pipeline(steps=[
#        ('imputer', SimpleImputer(strategy='constant', fill_value=0))
#       ,('scaler', StandardScaler())
# ])
# 

# preprocessor = ColumnTransformer(
#    transformers=[
#     ('numeric', numeric_transformer, numeric_features)
# ]) 
# 

# # Estimator

# from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
# pipeline = Pipeline(steps = [
#                ('preprocessor', preprocessor)
#               ,('regressor',RandomForestRegressor())
#            ])

# rf_model = pipeline.fit(X_train, y_train)
# print (rf_model)

# # Model Accuracy

# MSE is chosen as loss function because the dataset contains outliers that are important to the business

# from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
# predictions = rf_model.predict(X_test)
# print (f'r2_score : {r2_score(y_test, predictions)}\n\
# RMSE:{np.sqrt(mean_squared_error(y_test, predictions))}\
#  ')
