import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from statsmodels.tsa.api import VAR

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("powerconsumption.csv")
    df['Temperature'].fillna(df['Temperature'].mean(), inplace=True)
    df['Humidity'].fillna(df['Humidity'].mean(), inplace=True)
    df['DiffuseFlows'].fillna(df['DiffuseFlows'].mean(), inplace=True)
    df["Month"] = pd.to_datetime(df["Datetime"]).dt.month
    df["Year"] = pd.to_datetime(df["Datetime"]).dt.year
    df["Date"] = pd.to_datetime(df["Datetime"]).dt.date
    df["Time"] = pd.to_datetime(df["Datetime"]).dt.time
    df["Week"] = pd.to_datetime(df["Datetime"]).dt.isocalendar().week
    df["Day"] = pd.to_datetime(df["Datetime"]).dt.day_name()
    df = df.set_index("Datetime")
    df.index = pd.to_datetime(df.index)
    df = pd.get_dummies(df, columns=['Day'])
    return df

# Train and cache the models
@st.cache_data
def train_models(df):
    X = df[['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows', 'Month',
            'Day_Friday', 'Day_Monday', 'Day_Saturday', 'Day_Sunday', 'Day_Thursday', 'Day_Tuesday', 'Day_Wednesday']]
    y = df['PowerConsumption']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
    rf_model = RandomForestRegressor(random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    rf_model.fit(X_train_scaled, y_train)
    
    xgb_pred = xgb_model.predict(X_test_scaled)
    rf_pred = rf_model.predict(X_test_scaled)
    stacked_pred = np.column_stack((xgb_pred, rf_pred))
    
    meta_model = XGBRegressor(objective='reg:squarederror', random_state=42)
    meta_model.fit(stacked_pred, y_test)
    
    var_features = ['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows']
    var_data = df[var_features]
    train_size = int(len(var_data) * 0.8)
    var_train = var_data[:train_size]
    var_model = VAR(var_train)
    var_model_fitted = var_model.fit(maxlags=15)
    
    return scaler, xgb_model, rf_model, meta_model, var_model_fitted

df = load_data()
scaler, xgb_model, rf_model, meta_model, var_model_fitted = train_models(df)

# Function to predict future values using VAR model
def predict_var(var_model_fitted, steps):
    lag_order = var_model_fitted.k_ar
    var_pred = var_model_fitted.forecast(var_model_fitted.endog[-lag_order:], steps=steps)
    return var_pred

# Function to create static features for future dates
def create_static_features(future_dates):
    static_features_list = []
    for date in future_dates:
        month = date.month
        day_of_week = date.dayofweek
        static_features = [
            month,
            1 if day_of_week == 4 else 0,  # Friday
            1 if day_of_week == 0 else 0,  # Monday
            1 if day_of_week == 5 else 0,  # Saturday
            1 if day_of_week == 6 else 0,  # Sunday
            1 if day_of_week == 3 else 0,  # Thursday
            1 if day_of_week == 1 else 0,  # Tuesday
            1 if day_of_week == 2 else 0,  # Wednesday
        ]
        static_features_list.append(static_features)
    return pd.DataFrame(static_features_list, index=future_dates, columns=[
        'Month', 'Day_Friday', 'Day_Monday', 'Day_Saturday', 'Day_Sunday', 'Day_Thursday', 'Day_Tuesday', 'Day_Wednesday'
    ])

# Function to predict power consumption for a given date
def predict_power_consumption(date, var_model_fitted, scaler, xgb_model, rf_model, meta_model, df):
    future_date = pd.to_datetime(date)
    
    # Ensure the date is in the future
    if future_date <= df.index[-1]:
        raise ValueError("The date must be in the future.")
    
    # Calculate the number of steps to predict
    steps = (future_date - df.index[-1]).days
    
    # Predict future values using VAR
    var_forecast = predict_var(var_model_fitted, steps=steps)
    future_dates = pd.date_range(start=df.index[-1], periods=steps + 1, freq='D')[1:]
    
    # Create DataFrame for VAR predictions
    var_features = ['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows']
    var_forecast_df = pd.DataFrame(var_forecast, index=future_dates, columns=var_features)
    
    # Create static features for future dates
    future_static_df = create_static_features(future_dates)
    
    # Combine VAR predictions and static features
    future_data = pd.concat([var_forecast_df, future_static_df], axis=1)
    
    # Standardize future data using the scaler fitted on the training data
    X_future_scaled = scaler.transform(future_data)
    
    # Predict power consumption using base models
    xgb_pred_future = xgb_model.predict(X_future_scaled)
    rf_pred_future = rf_model.predict(X_future_scaled)
    
    # Stack predictions
    stacked_pred_future = np.column_stack((xgb_pred_future, rf_pred_future))
    
    # Predict power consumption using meta-model
    meta_pred_future = meta_model.predict(stacked_pred_future)
    
    # Return the prediction for the requested date
    return meta_pred_future[-1]

# Streamlit UI
st.title("Power Consumption Prediction")
date_input = st.text_input("Enter the date (YYYY-MM-DD):")
if st.button("Predict"):
    try:
        predicted_power_consumption = predict_power_consumption(date_input, var_model_fitted, scaler, xgb_model, rf_model, meta_model, df)
        st.success(f"Predicted power consumption for {date_input}: {predicted_power_consumption:.2f}")
    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add static content
st.markdown("""
            
            
            
  ELECTRICITY OPTIMIZATION TIPS :          
            
 Efficient Lighting
- Switch to LED Bulbs: Use LED or CFL bulbs instead of incandescent ones to reduce energy consumption.
- Utilize Natural Light: Maximize the use of natural daylight by arranging furniture and working spaces near windows.

 Smart Thermostat Usage
- Install Programmable Thermostats: Set thermostats to lower temperatures when you're not home or during the night.
- Regular Maintenance: Ensure heating and cooling systems are regularly maintained for optimal efficiency.

 Appliance Management
- Unplug Idle Electronics: Unplug devices when not in use to avoid phantom energy consumption.
- Energy-Efficient Appliances: Invest in appliances with high Energy Star ratings.

 Efficient Water Heating
- Lower Water Heater Temperature: Set your water heater to 120°F (49°C) to save energy.
- Insulate Water Heater and Pipes: Insulating can reduce heat loss and improve efficiency.

 Efficient Cooling and Heating
- Seal Leaks: Ensure windows and doors are properly sealed to prevent energy loss.
- Use Fans Wisely: Use ceiling fans to help circulate air and reduce reliance on air conditioning.

 Smart Home Technology
- Smart Power Strips: Use smart power strips that cut off power to devices in standby mode.
- Home Automation Systems: Implement smart home systems to control lights, thermostats, and appliances remotely.

 Behavior Adjustments
- Turn Off Lights: Always turn off lights when leaving a room.
- Use Energy During Off-Peak Hours: Shift energy-intensive activities, like laundry and dishwashing, to off-peak hours.

 Renewable Energy Solutions
- Solar Panels: Consider installing solar panels to generate your own electricity.
- Energy Storage: Use battery storage systems to store excess energy for later use.
""")
