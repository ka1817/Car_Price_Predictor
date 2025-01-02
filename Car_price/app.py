import streamlit as st
import pandas as pd
import joblib

# Load the cleaned car data from CSV
df = pd.read_csv("C:\\Users\\saipr\\anaconda3\\car_price\\cleaned_car.csv")

st.title('Car Price Prediction')

# Change text_input to selectbox for selecting a car name
name = st.selectbox('Select Car Name', df['name'].unique())

company = st.selectbox('Select Car Company', df['company'].unique())
year = st.number_input('Car Year', min_value=1994, max_value=2025, value=2020)
kms_driven = st.number_input('Kms Driven', min_value=0, value=10000)
fuel_type = st.selectbox('Fuel Type', df['fuel_type'].unique())

# When the user clicks the button, we use the trained model to predict the price
if st.button('Predict Price'):
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'name': [name],
        'company': [company],
        'year': [year],
        'kms_driven': [kms_driven],
        'fuel_type': [fuel_type]
    })
    
    # Load the saved model
    model = joblib.load('model.pkl')
    
    # Make prediction
    predicted_price = model.predict(input_data)
    
    # Display the predicted price
    st.write(f'Predicted Price for {name}: ₹{predicted_price[0]:,.2f}')

