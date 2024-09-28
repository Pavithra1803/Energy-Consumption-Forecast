import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from statsmodels.tsa.api import VAR
from datetime import datetime

# Read the CSV file
df = pd.read_csv("powerconsumption.csv")

# Display the first five rows
print("="*50)
print("First Five Rows", "\n")
print(df.head(5), "\n")

# Display information about the dataset
print("="*50)
print("Information About Dataset", "\n")
print(df.info(), "\n")

# Describe the dataset
print("="*50)
print("Describe the Dataset", "\n")
print(df.describe(), "\n")

# Check for missing values
print("="*50)
print("Null Values", "\n")
print(df.isnull().sum(), "\n")

# Fill missing values with mean
df['Temperature'].fillna(df['Temperature'].mean(), inplace=True)
df['Humidity'].fillna(df['Humidity'].mean(), inplace=True)
df['DiffuseFlows'].fillna(df['DiffuseFlows'].mean(), inplace=True)

# Check if there are still missing values
print(df.isnull().sum())

# Add new date-related columns
df["Month"] = pd.to_datetime(df["Datetime"]).dt.month
df["Year"] = pd.to_datetime(df["Datetime"]).dt.year
df["Date"] = pd.to_datetime(df["Datetime"]).dt.date
df["Time"] = pd.to_datetime(df["Datetime"]).dt.time
df["Week"] = pd.to_datetime(df["Datetime"]).dt.isocalendar().week
df["Day"] = pd.to_datetime(df["Datetime"]).dt.day_name()

# Set 'Datetime' as the index
df = df.set_index("Datetime")
df.index = pd.to_datetime(df.index)

# Perform one-hot encoding on the 'Day' column
df = pd.get_dummies(df, columns=['Day'])

# Print the updated dataframe
print(df.head())

# Define features and target variable
X = df[['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows', 'Month',
        'Day_Friday', 'Day_Monday', 'Day_Saturday', 'Day_Sunday', 'Day_Thursday', 'Day_Tuesday', 'Day_Wednesday']]
y = df['PowerConsumption']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize base models
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
rf_model = RandomForestRegressor(random_state=42)

# Train base models
xgb_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
xgb_pred = xgb_model.predict(X_test_scaled)
rf_pred = rf_model.predict(X_test_scaled)

# Stack predictions
stacked_pred = np.column_stack((xgb_pred, rf_pred))

# Initialize meta-model (XGBoost)
meta_model = XGBRegressor(objective='reg:squarederror', random_state=42)

# Train meta-model on stacked predictions
meta_model.fit(stacked_pred, y_test)

# Make predictions using base models on training set
xgb_pred_train = xgb_model.predict(X_train_scaled)
rf_pred_train = rf_model.predict(X_train_scaled)

# Stack training predictions
stacked_pred_train = np.column_stack((xgb_pred_train, rf_pred_train))

# Make predictions on training set using meta-model
meta_pred_train = meta_model.predict(stacked_pred_train)

# Calculate R-squared on training set
r2_train = r2_score(y_train, meta_pred_train)
print("R-squared on training set:", r2_train)

# Make predictions on test set using meta-model
meta_pred_test = meta_model.predict(stacked_pred)

# Calculate R-squared on test set
r2_test = r2_score(y_test, meta_pred_test)
print("R-squared on test set:", r2_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, meta_pred_test))
print("Root Mean Squared Error (RMSE) on test set:", rmse)

# Calculate MSE
mse = mean_squared_error(y_test, meta_pred_test)
print("Mean Squared Error (MSE) on test set:", mse)

# Calculate MAE
mae = mean_absolute_error(y_test, meta_pred_test)
print("Mean Absolute Error (MAE) on test set:", mae)

# Plot actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, meta_pred_test, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Actual vs. Predicted Values')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

# Calculate residuals
residuals = y_test - meta_pred_test

# Plot residuals
plt.figure(figsize=(8, 6))
plt.scatter(meta_pred_test, residuals, color='green', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

# Define features for the VAR model
var_features = ['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows']
var_data = df[var_features]

train_size = int(len(var_data) * 0.8)
var_train = var_data[:train_size]

# Train the VAR model
var_model = VAR(var_train)
var_model_fitted = var_model.fit(maxlags=15)

# Function to predict future values using VAR model
def predict_var(var_model_fitted, steps):
    lag_order = var_model_fitted.k_ar
    var_pred = var_model_fitted.forecast(var_data.values[-lag_order:], steps=steps)
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
def predict_power_consumption(date, var_model_fitted, scaler, xgb_model, rf_model, meta_model):
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

# Take the date as input from the user
future_date = input("Enter the future date (YYYY-MM-DD): ")

# Predict power consumption for the input date
try:
    predicted_power_consumption = predict_power_consumption(future_date, var_model_fitted, scaler, xgb_model, rf_model, meta_model)
    print(f"Predicted power consumption for {future_date}: {predicted_power_consumption}")
except ValueError as e:
    print(e)
