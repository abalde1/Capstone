import streamlit as st
import pandas as pd
import sklearn 
import joblib,os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt 
import seaborn as sns
from multipages import MultiPage
from pages import login, model, report
#from pip._vendor.certifi.__main__ import args


def main():
    
    car_data = pd.read_csv('data/car data.csv')
    car_data.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
    car_data.replace({'Transmission':{'Manual':0, 'Automatic':1}},inplace=True)
    car_data.replace({'Seller_Type':{'Individual':0, 'Dealer':1}},inplace=True)

    st.title("Car Price Prediction")

    st.sidebar.title("Login")
    username = st.sidebar.text_input("User Name")
    password = st.sidebar.text_input("Password", type = 'password')
    login_b = st.sidebar.checkbox("Login")
    if login_b and username=='admin' and password== 'password':
        st.sidebar.success("Your are successfully logged in as Admin, you can now navigate the app.")
        st.write("Use the drop-down menu to select Prediction model or Report.")
        activity = ["Prediction model", "Report"]
        menu = st.selectbox("Menu", activity)
        if menu == "Prediction model":
            st.subheader("Prediction model")
            st.text('Select your car year and fuel type')

            #Column component
            sel_col, disp_col = st.columns(2)
            age = sel_col.slider('What is the year of your car?', min_value = 2006, max_value = 2020, value = 2010)
            fuel = sel_col.selectbox('Fuel type?', options = ['Petrol', 'Diesel'])


            age_reshaped = np.array(age).reshape(-1,1)

            X = car_data[["Year"]]
            y = car_data[["Selling_Price"]]
            act_value = [[age]]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

            lrgr = LinearRegression()
            lrgr.fit(X_train, y_train)

            train_p1 = lrgr.predict(X_train)
            train_p = lrgr.predict(act_value)
            test_p = lrgr.predict(X_test)

            if sel_col.button("Estimate"):
                disp_col.subheader('Your car value is: ')
                c_value=train_p
                disp_col.write(c_value)

        elif menu == "Report":
            st.subheader("Data report")
            st.text("The dataset is available on kaggle.com...")
            st.write(car_data.head())
            st.subheader('Selling price distrubution')
            Price_distrubition = pd.DataFrame(car_data['Selling_Price']).head(50)
            st.bar_chart(Price_distrubition)

            x_value = car_data['Year']
            y_value = car_data['Selling_Price']
            st.subheader('Data relation visualisation')
            numeric_columns = car_data.select_dtypes(['float64', 'float32', 'int32', 'int64']).columns
            select_box1 = st.selectbox(label='X axis', options=numeric_columns,index=1)
            st.write(select_box1)
            select_box2 = st.selectbox(label='Y axis', options=numeric_columns)
            #create scatterplot
            fig=sns.relplot(x=select_box1, y=select_box2, data=car_data)
            st.pyplot(fig)
    
    else:
        st.sidebar.write("Enter valid username and password then check login button.")



if __name__ == '__main__':
    main()







