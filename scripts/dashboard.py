import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os 
import base64
# sys.path.append(os.path.abspath(os.path.join('../')))
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error

# load data
X = pd.read_csv('https://drive.google.com/file/d/1sgPGD84cNyfm3Sve6WiLTGJgITIhZluJ/view?usp=sharing')
Y = pd.read_csv('https://drive.google.com/file/d/1w8-oe8hcEdu0MAGfvEsxGkPuEaW5SOmn/view?usp=sharing')


# split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y['Sales'], test_size=0.2, random_state=123)


st.set_page_config(page_title='Predict Sales', layout='wide')
st.title('Sales Prediction Dashboard')
st.markdown('This helps to predict sales for a Pharmaceutical Company')

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="Prediction.csv">Download Prediction csv file</a>'
    return href


st.sidebar.title('Predict')
st.sidebar.markdown('You can try Different Models!!')

st.sidebar.subheader("Choose Prediction model")
model = st.sidebar.selectbox("Model", ( "Logistic Regression", "Random Forest Regression"))

if model == "Random Forest Regression":
    st.sidebar.subheader("Features")
    uploaded_file = st.sidebar.file_uploader("Choose Test csv file" , type='csv')
    # store = st.sidebar.number_input('Store Id', min_value=1, key='Store')
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Scatter plot","Show Values", "None"))
    if uploaded_file is not None:
        testdata = pd.read_csv(uploaded_file)
        X_test = testdata
        st.write(testdata)
        if st.sidebar.button("Predict", key="predict"):
            st.subheader("Random Forest Results")
            clf = RandomForestRegressor(n_estimators = 15)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            # st.write("RMSE: ", np.sqrt(mean_squared_error(y_test,y_pred)))
            # st.write("R2 Score: ", r2_score(y_test,y_pred))
            prediction = pd.DataFrame(y_pred,columns=['Predicted Sales'])
            st.write(prediction)
            st.markdown(get_table_download_link(prediction), unsafe_allow_html=True)
    if uploaded_file is None:
        if st.sidebar.button("Predict", key="predict"):
            st.subheader("Random Forest Results")
            clf = RandomForestRegressor(n_estimators = 15)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            st.write("RMSE: ", np.sqrt(mean_squared_error(y_test,y_pred)))
            st.write("R2 Score: ", r2_score(y_test,y_pred))

        # plot_metrics(metrics)