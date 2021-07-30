import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os 
sys.path.append(os.path.abspath(os.path.join('../')))
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error

# load data
X = pd.read_csv('data/X.csv')
Y = pd.read_csv('data/Y.csv')

# split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y['Sales'], test_size=0.2, random_state=123)

st.title('Sales Prediction Dashboard')
st.markdown('This helps to predict sales for a Pharmaceutical Company')

st.sidebar.title('Predict')
st.sidebar.markdown('Try Different Models!!')

st.sidebar.subheader("Choose Prediction model")
model = st.sidebar.selectbox("Model", ( "Logistic Regression", "Random Forest Regression"))

if model == "Random Forest Regression":
    st.sidebar.subheader("Features")
    # n_estimators= st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key="n_estimators")
    # max_depth = st.sidebar.number_input("The maximum depth of tree", 1, 20, step =1, key="max_depth")
    # bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ("True", "False"), key="bootstrap")
    store = st.sidebar.number_input('Store Id', min_value=1, key='Store')
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Scatter plot","Show Values", "None"))
    
    if st.sidebar.button("Predict", key="predict"):
        st.subheader("Random Forest Results")
        clf = RandomForestRegressor(n_estimators = 15)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        st.write("RMSE: ", np.sqrt(mean_squared_error(y_test,y_pred)))
        st.write("R2 Score: ", r2_score(y_test,y_pred))

        # plot_metrics(metrics)