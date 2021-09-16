# Sales Prediction
![workflow](https://github.com/Theehawau/Sales-Prediction/actions/workflows/cml.yml/badge.svg)

### Overview
<p>Sales Prediction / Forecasting will help a business prepare better, plan ahead and the exploratory data analysis will help to determine what factors affect sales and how to exploit those factors and maximize sales.
Sales Prediction is a regression problem as it is about predicting a value. Various Machine Learning Regression algorithms would be used for this project.
Loss function used for the model is Mean Squared Error because of the meaningful outliers in the dataset.</p>
<p>The Datasets for this project can be found <a href="https://drive.google.com/file/d/1Cyru3Q4wyEisnxR2r4K95kYDiETJ9oQK/view?usp=sharing">here</a></p>

### Runtime
python-3.6.4

### Installation
```sh
git clone https://github.com/Theehawau/Sales-Prediction
cd Sales-Prediction
pip -r install requirements.txt
```
### Features
#### Exploratory Data Analysis
The datasets were explored in [this file](../main/notebooks/Exploratory-Analysis.ipynb)

#### Data Modeling and Preprocessing
Modeling and preprocessing python notebook can be found [here](../main/notebooks/Modelling.ipynb)
Sample of Modeling and preprocessing using pipeline can be found [here](../main/notebooks/modelling-pipeline.ipynb)

#### Deep Learning model
Notebook can be found [here](../main/notebooks/deepLearningModel.ipynb)

#### Steps Logging
Log of the steps can be found in:
  * [Modelling](../main/notebooks/modellingSteps.log)

#### Scripts
  * [dashboard.py](../main/scripts/dashboard.py) : Dashboard base code
  * [plot.py](../main/scripts/plot.py) : script with helper functions for plotting 
  * [data_cleaner.py](../main/scripts/data_cleaner.py) : script with helper functions for Data cleaner
  * [modelling.py](../main/scripts/modelling.py) : script to generate mlflow runs
  
#### Tests
Test for helper scripts can be found [here](../main/tests/test_data_cleaner.py)
