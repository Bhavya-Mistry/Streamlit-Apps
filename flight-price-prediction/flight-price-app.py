import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Set page title
st.set_page_config(page_title="Flight Price Predictor", layout="centered")

st.title("🛫 Flight Price Prediction App")

# Load model and scaler
with open("flight-price-prediction\decision_tree_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("flight-price-prediction\scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# All columns your model expects
input_columns = ['Duration', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi', 'Total_Stops_2 stops',
       'Total_Stops_3 stops', 'Total_Stops_4 stops', 'Total_Stops_non-stop',
       'Additional_Info_1 Short layover', 'Additional_Info_2 Long layover',
       'Additional_Info_Business class', 'Additional_Info_Change airports',
       'Additional_Info_In-flight meal not included', 'Additional_Info_No Info',
       'Additional_Info_No check-in baggage included', 'Additional_Info_No info',
       'Additional_Info_Red-eye flight', 'weekdayornot_Weekend']

# User inputs
with st.form("prediction_form"):
    st.subheader("Enter Flight Details")

    duration = st.number_input("Flight Duration (in minutes)", min_value=30, max_value=1440, value=120)

    airline = st.selectbox("Airline", ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet',
       'Multiple carriers', 'GoAir', 'Vistara', 'Air Asia',
       'Vistara Premium economy', 'Jet Airways Business',
       'Multiple carriers Premium economy', 'Trujet'])

    source = st.selectbox("Source", ['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai'])

    destination = st.selectbox("Destination", ['New Delhi', 'Banglore', 'Cochin', 'Kolkata', 'Delhi', 'Hyderabad'])

    stops = st.selectbox("Total Stops", ['1 stop','2 stops', '3 stops', '4 stops', 'non-stop'])

    additional_info = st.selectbox("Additional Info", ['No info', 'In-flight meal not included',
       'No check-in baggage included', '1 Short layover', 'No Info',
       '1 Long layover', 'Change airports', 'Business class',
       'Red-eye flight', '2 Long layover'])

    journey_date = st.date_input("Date of Journey", min_value=datetime.today())

    submitted = st.form_submit_button("Predict Price")

# Prediction logic
if submitted:
    weekday = journey_date.strftime('%A')
    weekdayornot = 'Weekend' if weekday in ['Saturday', 'Sunday'] else 'Weekday'

    # Create zero-filled feature vector
    input_dict = {col: 0 for col in input_columns}
    input_dict['Duration'] = duration

    # Set corresponding one-hot features
    for category, value in {
        f"Airline_{airline}": airline,
        f"Source_{source}": source,
        f"Destination_{destination}": destination,
        f"Total_Stops_{stops}": stops,
        f"Additional_Info_{additional_info}": additional_info,
        "weekdayornot_Weekend": weekdayornot
    }.items():
        if category in input_dict:
            input_dict[category] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    st.success(f"💰 Predicted Flight Price: ₹{int(prediction):,}")
