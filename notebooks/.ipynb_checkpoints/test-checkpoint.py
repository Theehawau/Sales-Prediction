{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Importing Library"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np \n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt \n",
    "import seaborn as sns\n",
    "import mlflow\n",
    "import mlflow.sklearn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Initiating Logging"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import logging\n",
    "logging.basicConfig(filename='modellingSteps.log', level=logging.INFO)\n",
    "logging.info('This log file records the steps for modelling for this project')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def log(text:str):\n",
    "    logging.info(text)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Importing Scripts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys, os \n",
    "sys.path.append(os.path.abspath(os.path.join('../scripts/')))\n",
    "log('Import utility scripts')\n",
    "from data_cleaner import *"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Utility Functions Imported!!!\n"
     ]
    }
   ],
   "source": [
    "util = Clean_Data()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Importing Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "log('Import Train Dataset to db')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Store</th>\n",
       "      <th>DayOfWeek</th>\n",
       "      <th>Date</th>\n",
       "      <th>Sales</th>\n",
       "      <th>Customers</th>\n",
       "      <th>Open</th>\n",
       "      <th>Promo</th>\n",
       "      <th>StateHoliday</th>\n",
       "      <th>SchoolHoliday</th>\n",
       "      <th>StoreType</th>\n",
       "      <th>Assortment</th>\n",
       "      <th>CompetitionDistance</th>\n",
       "      <th>CompetitionOpenSinceMonth</th>\n",
       "      <th>CompetitionOpenSinceYear</th>\n",
       "      <th>Promo2</th>\n",
       "      <th>Promo2SinceWeek</th>\n",
       "      <th>Promo2SinceYear</th>\n",
       "      <th>PromoInterval</th>\n",
       "      <th>Month</th>\n",
       "      <th>Day</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>2015-07-31</td>\n",
       "      <td>5263</td>\n",
       "      <td>555</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>c</td>\n",
       "      <td>a</td>\n",
       "      <td>1270.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>2008.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>31</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>2015-07-30</td>\n",
       "      <td>5020</td>\n",
       "      <td>546</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>c</td>\n",
       "      <td>a</td>\n",
       "      <td>1270.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>2008.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Store  DayOfWeek        Date  Sales  Customers  Open  Promo StateHoliday  \\\n",
       "0      1          5  2015-07-31   5263        555     1      1            0   \n",
       "1      1          4  2015-07-30   5020        546     1      1            0   \n",
       "\n",
       "   SchoolHoliday StoreType Assortment  CompetitionDistance  \\\n",
       "0              1         c          a               1270.0   \n",
       "1              1         c          a               1270.0   \n",
       "\n",
       "   CompetitionOpenSinceMonth  CompetitionOpenSinceYear  Promo2  \\\n",
       "0                        9.0                    2008.0       0   \n",
       "1                        9.0                    2008.0       0   \n",
       "\n",
       "   Promo2SinceWeek  Promo2SinceYear PromoInterval  Month  Day  \n",
       "0              0.0                0             0      7   31  \n",
       "1              0.0                0             0      7   30  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "db = util.read_data('../data/trainData.csv')\n",
    "db.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Store                             0\n",
       "DayOfWeek                         0\n",
       "Date                              0\n",
       "Sales                             0\n",
       "Customers                         0\n",
       "Open                              0\n",
       "Promo                             0\n",
       "StateHoliday                      0\n",
       "SchoolHoliday                     0\n",
       "StoreType                         0\n",
       "Assortment                        0\n",
       "CompetitionDistance               0\n",
       "CompetitionOpenSinceMonth    323348\n",
       "CompetitionOpenSinceYear     323348\n",
       "Promo2                            0\n",
       "Promo2SinceWeek                   0\n",
       "Promo2SinceYear                   0\n",
       "PromoInterval                     0\n",
       "Month                             0\n",
       "Day                               0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "db.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "db.dropna(axis=0,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "log('Import Test Dataset to test_db')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Id</th>\n",
       "      <th>Store</th>\n",
       "      <th>DayOfWeek</th>\n",
       "      <th>Date</th>\n",
       "      <th>Open</th>\n",
       "      <th>Promo</th>\n",
       "      <th>StateHoliday</th>\n",
       "      <th>SchoolHoliday</th>\n",
       "      <th>StoreType</th>\n",
       "      <th>Assortment</th>\n",
       "      <th>CompetitionDistance</th>\n",
       "      <th>CompetitionOpenSinceMonth</th>\n",
       "      <th>CompetitionOpenSinceYear</th>\n",
       "      <th>Promo2</th>\n",
       "      <th>Promo2SinceWeek</th>\n",
       "      <th>Promo2SinceYear</th>\n",
       "      <th>PromoInterval</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>2015-09-17</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>c</td>\n",
       "      <td>a</td>\n",
       "      <td>1270.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>2008.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>857</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>2015-09-16</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>c</td>\n",
       "      <td>a</td>\n",
       "      <td>1270.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>2008.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Id  Store  DayOfWeek        Date  Open  Promo StateHoliday  SchoolHoliday  \\\n",
       "0    1      1          4  2015-09-17   1.0      1            0              0   \n",
       "1  857      1          3  2015-09-16   1.0      1            0              0   \n",
       "\n",
       "  StoreType Assortment  CompetitionDistance  CompetitionOpenSinceMonth  \\\n",
       "0         c          a               1270.0                        9.0   \n",
       "1         c          a               1270.0                        9.0   \n",
       "\n",
       "   CompetitionOpenSinceYear  Promo2  Promo2SinceWeek  Promo2SinceYear  \\\n",
       "0                    2008.0       0              0.0                0   \n",
       "1                    2008.0       0              0.0                0   \n",
       "\n",
       "  PromoInterval  \n",
       "0             0  \n",
       "1             0  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_db = util.read_data('../data/testData.csv')\n",
    "test_db.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "mlflow.set_experiment('Sales Prediction')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data Preprocesssing:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "log('Commence Data Processing:')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "log('Convert Date columns in db and test_db to Datetime type')\n",
    "util.to_datetime(test_db,'Date','%Y-%m-%d')\n",
    "util.to_datetime(db,'Date','%Y-%m-%d')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_date_features(df,col):\n",
    "    df['Year'] = df[col].dt.year\n",
    "    df['Month'] = df[col].dt.month\n",
    "    df['Day'] = df[col].dt.day\n",
    "    df.drop(columns=[col], axis=1,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Id</th>\n",
       "      <th>Store</th>\n",
       "      <th>DayOfWeek</th>\n",
       "      <th>Open</th>\n",
       "      <th>Promo</th>\n",
       "      <th>StateHoliday</th>\n",
       "      <th>SchoolHoliday</th>\n",
       "      <th>StoreType</th>\n",
       "      <th>Assortment</th>\n",
       "      <th>CompetitionDistance</th>\n",
       "      <th>CompetitionOpenSinceMonth</th>\n",
       "      <th>CompetitionOpenSinceYear</th>\n",
       "      <th>Promo2</th>\n",
       "      <th>Promo2SinceWeek</th>\n",
       "      <th>Promo2SinceYear</th>\n",
       "      <th>PromoInterval</th>\n",
       "      <th>Year</th>\n",
       "      <th>Month</th>\n",
       "      <th>Day</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>c</td>\n",
       "      <td>a</td>\n",
       "      <td>1270.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>2008.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2015</td>\n",
       "      <td>9</td>\n",
       "      <td>17</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>857</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>c</td>\n",
       "      <td>a</td>\n",
       "      <td>1270.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>2008.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2015</td>\n",
       "      <td>9</td>\n",
       "      <td>16</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Id  Store  DayOfWeek  Open  Promo StateHoliday  SchoolHoliday StoreType  \\\n",
       "0    1      1          4   1.0      1            0              0         c   \n",
       "1  857      1          3   1.0      1            0              0         c   \n",
       "\n",
       "  Assortment  CompetitionDistance  CompetitionOpenSinceMonth  \\\n",
       "0          a               1270.0                        9.0   \n",
       "1          a               1270.0                        9.0   \n",
       "\n",
       "   CompetitionOpenSinceYear  Promo2  Promo2SinceWeek  Promo2SinceYear  \\\n",
       "0                    2008.0       0              0.0                0   \n",
       "1                    2008.0       0              0.0                0   \n",
       "\n",
       "  PromoInterval  Year  Month  Day  \n",
       "0             0  2015      9   17  \n",
       "1             0  2015      9   16  "
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "log('Get date features from date column in db and test_db')\n",
    "get_date_features(test_db,'Date')\n",
    "get_date_features(db,'Date')\n",
    "test_db.head(2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Extracting feature columns:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "log('Extract feature columns from db to train_db:')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "log('Drop Columns:\"Date,Customers,Sales,PromoInterval,CompetitionOpenSinceYear,CompetitionOpenSinceMonth\"')\n",
    "train_db = db.drop(columns=['Customers','Sales',\n",
    "                           'PromoInterval'], axis=1)\n",
    "# 'CompetitionOpenSinceYear','CompetitionOpenSinceMonth'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Store</th>\n",
       "      <th>DayOfWeek</th>\n",
       "      <th>Open</th>\n",
       "      <th>Promo</th>\n",
       "      <th>StateHoliday</th>\n",
       "      <th>SchoolHoliday</th>\n",
       "      <th>StoreType</th>\n",
       "      <th>Assortment</th>\n",
       "      <th>CompetitionDistance</th>\n",
       "      <th>CompetitionOpenSinceMonth</th>\n",
       "      <th>CompetitionOpenSinceYear</th>\n",
       "      <th>Promo2</th>\n",
       "      <th>Promo2SinceWeek</th>\n",
       "      <th>Promo2SinceYear</th>\n",
       "      <th>Month</th>\n",
       "      <th>Day</th>\n",
       "      <th>Year</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>c</td>\n",
       "      <td>a</td>\n",
       "      <td>1270.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>2008.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>31</td>\n",
       "      <td>2015</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>c</td>\n",
       "      <td>a</td>\n",
       "      <td>1270.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>2008.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>30</td>\n",
       "      <td>2015</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Store  DayOfWeek  Open  Promo StateHoliday  SchoolHoliday StoreType  \\\n",
       "0      1          5     1      1            0              1         c   \n",
       "1      1          4     1      1            0              1         c   \n",
       "\n",
       "  Assortment  CompetitionDistance  CompetitionOpenSinceMonth  \\\n",
       "0          a               1270.0                        9.0   \n",
       "1          a               1270.0                        9.0   \n",
       "\n",
       "   CompetitionOpenSinceYear  Promo2  Promo2SinceWeek  Promo2SinceYear  Month  \\\n",
       "0                    2008.0       0              0.0                0      7   \n",
       "1                    2008.0       0              0.0                0      7   \n",
       "\n",
       "   Day  Year  \n",
       "0   31  2015  \n",
       "1   30  2015  "
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_db.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "log('Drop columns:\"Id,PromoInterval,CompetitionOpenSinceYear,CompetitionOpenSinceMonth\" in test_db:')\n",
    "test_db = test_db.drop(columns=['Id','PromoInterval',\n",
    "                                'CompetitionOpenSinceYear',\n",
    "                                'CompetitionOpenSinceMonth'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Store</th>\n",
       "      <th>DayOfWeek</th>\n",
       "      <th>Open</th>\n",
       "      <th>Promo</th>\n",
       "      <th>StateHoliday</th>\n",
       "      <th>SchoolHoliday</th>\n",
       "      <th>StoreType</th>\n",
       "      <th>Assortment</th>\n",
       "      <th>CompetitionDistance</th>\n",
       "      <th>Promo2</th>\n",
       "      <th>Promo2SinceWeek</th>\n",
       "      <th>Promo2SinceYear</th>\n",
       "      <th>Year</th>\n",
       "      <th>Month</th>\n",
       "      <th>Day</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>c</td>\n",
       "      <td>a</td>\n",
       "      <td>1270.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>2015</td>\n",
       "      <td>9</td>\n",
       "      <td>17</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>c</td>\n",
       "      <td>a</td>\n",
       "      <td>1270.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>2015</td>\n",
       "      <td>9</td>\n",
       "      <td>16</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Store  DayOfWeek  Open  Promo StateHoliday  SchoolHoliday StoreType  \\\n",
       "0      1          4   1.0      1            0              0         c   \n",
       "1      1          3   1.0      1            0              0         c   \n",
       "\n",
       "  Assortment  CompetitionDistance  Promo2  Promo2SinceWeek  Promo2SinceYear  \\\n",
       "0          a               1270.0       0              0.0                0   \n",
       "1          a               1270.0       0              0.0                0   \n",
       "\n",
       "   Year  Month  Day  \n",
       "0  2015      9   17  \n",
       "1  2015      9   16  "
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_db.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Sales</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5263</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5020</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Sales\n",
       "0   5263\n",
       "1   5020"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "log('Extract Sales column from db into y for target variable')\n",
    "y = db[['Sales']]\n",
    "y.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_url = '../data/train.csv'\n",
    "version = 'v1'\n",
    "mlflow.log_param('data_url', data_url)\n",
    "mlflow.log_param('data_version', version)\n",
    "mlflow.log_param('input_rows',train_db.shape[0])\n",
    "mlflow.log_param('input_cols', train_db.shape[1])\n",
    "\n",
    "cols_x = pd.DataFrame(list(train_db.columns))\n",
    "cols_x.to_csv('features.csv', header=False, index=False)\n",
    "mlflow.log_artifact('features.csv')\n",
    "\n",
    "\n",
    "cols_y = pd.DataFrame(list(y.columns))\n",
    "cols_y.to_csv('target.csv', header=False, index=False)\n",
    "mlflow.log_artifact('target.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Encoding categorical data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "log('Encode Categorical variables in train_db and test_db:')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "def encode(df,columns):\n",
    "    df = pd.get_dummies(df,columns=columns)\n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "log('Encode Columns:\"StoreType,Assortment,StateHoliday\" in train_db.')\n",
    "train_db = encode(train_db,['StoreType','Assortment','StateHoliday'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "log('Encode Columns:\"StoreType,Assortment\" in test_db.')\n",
    "test_db = encode(test_db,['StoreType','Assortment'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Scaling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "log('Start Feature Scaling:')\n",
    "log('Import StandardScaler from sklearn.preprocessing')\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "sc_X = StandardScaler()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "log('Scale train_db into X')\n",
    "X = sc_X.fit_transform(train_db)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Split data into train and test set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "log('Split data into train and test set:')\n",
    "log('Import train_test_split from sklearn.model_selection')\n",
    "log('Split X into X_train, X_test and y into Y_train and Y_test in ratio 80:20')\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Train model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "log('Train model:')\n",
    "log('Import Linear Regression from sklearn.linear_model')\n",
    "log('Create instance of Linear Regression in regressor')\n",
    "from sklearn.linear_model import LinearRegression\n",
    "regressor = LinearRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "log('Fit regressor to X_train,Y_train')\n",
    "regressor.fit(X_train, Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "log('Predict Y_pred from X_test using regressor.predict')\n",
    "Y_pred = regressor.predict(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Check Model Accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "log('Check Model Accuracy:')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "log('Import rmse,mae,rs2 libraries from sklearn.metrics')\n",
    "from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "log('Compute rmse,mae,rs2 values')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "def eval_metrics(actual,pred):\n",
    "    \"\"\"\"\n",
    "    Function to obtain rmse,mae,r2 values for actual and predicted values\n",
    "    \"\"\"\n",
    "    rmse = np.sqrt(mean_squared_error(actual,pred))\n",
    "    mae = mean_absolute_error(actual,pred)\n",
    "    r2 = r2_score(actual,pred)\n",
    "    return rmse,mae,r2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Sales prediction model: \n",
      " \t RMSE:2559.3755944523796 \n",
      " \t MAE:1787.511786215058 \n",
      " \t R2:0.5629258596707101\n"
     ]
    }
   ],
   "source": [
    "print(f'Sales prediction model: \\n \\t RMSE:{eval_metrics(Y_test,Y_pred)[0]} \\n \\\n",
    "\\t MAE:{eval_metrics(Y_test,Y_pred)[1]} \\n \\\n",
    "\\t R2:{eval_metrics(Y_test,Y_pred)[2]}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "mlflow.log_metrics({'RMSE':eval_metrics(Y_test,Y_pred)[0],\\\n",
    "                   'MAE':eval_metrics(Y_test,Y_pred)[1],\\\n",
    "                   'R2':eval_metrics(Y_test,Y_pred)[2]})"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
