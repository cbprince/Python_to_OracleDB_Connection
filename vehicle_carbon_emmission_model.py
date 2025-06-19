# the initial setup and data preprocessing code, along with placeholders for sample data you can adapt to Zambia-specific data.
# Part 1: Setting Up and Importing Necessary Libraries
# First, install and import the libraries needed for data processing, visualization, and modeling.

# Install libraries if not already installed
!pip install pandas numpy scikit-learn matplotlib seaborn

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Set up basic plotting style
sns.set(style="whitegrid")

===========================================================

# Part 2: Sample Data Preparation
# Here’s an example of how the data could be structured. You can replace this with your actual transport and emissions data for Zambia.

# Create a sample dataset of transport data
data = {
    'Vehicle_Type': ['Car', 'Bus', 'Truck', 'Motorbike', 'Taxi'],
    'Fuel_Type': ['Petrol', 'Diesel', 'Diesel', 'Petrol', 'Petrol'],
    'Average_Mileage_km': [15000, 30000, 40000, 10000, 20000],
    'Fuel_Consumption_L_per_100km': [8, 25, 30, 3, 7],
    'Emission_Factor_kg_CO2_per_L': [2.31, 2.68, 2.68, 2.31, 2.31]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the data
print("Sample Transport Data")
print(df)

===========================================================

# Part 3: Calculating Current Emissions
# This code calculates the current annual emissions based on the fuel consumption, mileage, and emission factor for each vehicle type.

# Calculate emissions per vehicle type
df['Annual_Emissions_kg_CO2'] = (df['Average_Mileage_km'] / 100) * df['Fuel_Consumption_L_per_100km'] * df['Emission_Factor_kg_CO2_per_L']

# Display the calculated emissions
print("\nCurrent Emissions by Vehicle Type")
print(df[['Vehicle_Type', 'Annual_Emissions_kg_CO2']])

============================================================

# Explanation of Each Calculation
# Daily Emissions: Divides the annual emissions by 365 (assuming consistent daily emissions throughout the year).
# Monthly Emissions: Divides the annual emissions by 12 to estimate a monthly value.
# Quarterly Emissions: Divides the annual emissions by 4, reflecting emissions every three months.
# Biannual Emissions: Divides the annual emissions by 2, representing emissions every six months.


# Calculate daily, monthly, quarterly, and biannual emissions
df['Daily_Emissions_kg_CO2'] = df['Annual_Emissions_kg_CO2'] / 365
df['Monthly_Emissions_kg_CO2'] = df['Annual_Emissions_kg_CO2'] / 12
df['Quarterly_Emissions_kg_CO2'] = df['Annual_Emissions_kg_CO2'] / 4
df['Biannual_Emissions_kg_CO2'] = df['Annual_Emissions_kg_CO2'] / 2

# Display the results
print("\nEmissions by Time Period for Each Vehicle Type")
print(df[['Vehicle_Type', 'Daily_Emissions_kg_CO2', 'Monthly_Emissions_kg_CO2', 
         'Quarterly_Emissions_kg_CO2', 'Biannual_Emissions_kg_CO2', 'Annual_Emissions_kg_CO2']])

=============================================================================================================

# Predictive Modeling. 
# We’ll build a machine learning model to predict emissions based on features like vehicle type, fuel consumption, and average mileage. 
# For this example, we’ll use a Random Forest Regressor, a flexible and powerful model for handling a mix of numerical and categorical data.

# Part 4: Setting Up Predictive Modeling
# 4.1 Preparing Data for Modeling
# First, we need to encode categorical variables (like Vehicle_Type and Fuel_Type) and split the data into training and test sets.

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Encode categorical variables
label_encoder = LabelEncoder()
df['Vehicle_Type_Encoded'] = label_encoder.fit_transform(df['Vehicle_Type'])
df['Fuel_Type_Encoded'] = label_encoder.fit_transform(df['Fuel_Type'])

# Select features and target variable
X = df[['Vehicle_Type_Encoded', 'Fuel_Type_Encoded', 'Average_Mileage_km', 'Fuel_Consumption_L_per_100km']]
y = df['Annual_Emissions_kg_CO2']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

===========================================================================================================
# 4.2 Building and Training the Model
# Now, we’ll train the Random Forest model on the training data.

# Initialize the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

===========================================================================================================
# 4.3 Evaluating the Model
# Evaluate the model’s performance using metrics like Mean Squared Error (MSE) and R² Score, which indicate how well the model predicts emissions on the test data.

# Predict emissions on the test set
y_pred = rf_model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:\nMean Squared Error: {mse}\nR² Score: {r2}")

========================================================================================================

# Part 5: Scenario Testing and Visualization
# With the model trained, we can create various scenarios by adjusting input values (like vehicle mileage or fuel type) to see how they affect emissions. 
# We’ll also plot the predicted emissions to help visualize trends.

# 5.1 Scenario Testing Code
# For example, we can test a scenario where we increase the adoption of electric vehicles (setting fuel consumption to zero) or improve fuel efficiency.

# Example scenario: Improve fuel efficiency by 20% for all vehicles
X_scenario = X_test.copy()
X_scenario['Fuel_Consumption_L_per_100km'] *= 0.8  # Reduce fuel consumption by 20%

# Predict emissions for the scenario
y_scenario_pred = rf_model.predict(X_scenario)

# Compare original vs. scenario emissions
df_scenario = pd.DataFrame({'Original_Emissions': y_test, 'Scenario_Emissions': y_scenario_pred})
print("\nOriginal vs. Scenario Emissions (first 5 samples)")
print(df_scenario.head())

==========================================================================================================
# 5.2 Visualization of Scenario
# Plotting the scenario emissions alongside the original emissions allows for easy comparison.

import matplotlib.pyplot as plt

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Original Emissions')
plt.scatter(range(len(y_scenario_pred)), y_scenario_pred, color='green', label='Scenario Emissions (20% Fuel Efficiency)')
plt.xlabel("Sample")
plt.ylabel("Annual Emissions (kg CO₂)")
plt.legend()
plt.title("Original vs. Scenario Emissions")
plt.show()

====================================================================================================================================================

# guide on how to implement more advanced scenario testing and interactive visualization to make the model more useful for exploring policy impacts. 
# These steps will include creating multiple scenarios and building a simple dashboard with Streamlit for user-friendly visualization.

# Part 6: Advanced Scenario Testing
# In this part, we’ll create multiple scenarios, such as:

# Scenario 1: Increase the proportion of electric vehicles (EVs) in the transport sector.
# Scenario 2: Implement fuel efficiency improvements.
# Scenario 3: Change in average mileage due to policy shifts (e.g., encouraging public transport).
# These scenarios will allow policymakers to explore potential emissions reductions from various initiatives.

# 6.1 Creating Scenario Functions
# Define functions to simulate each scenario. Each function will adjust the input data and calculate new emissions based on the modified data.

def apply_electric_vehicle_scenario(data, ev_percentage=0.2):
    """Assume a certain percentage of vehicles become electric (zero emissions)."""
    data_scenario = data.copy()
    num_ev = int(ev_percentage * len(data_scenario))
    data_scenario['Fuel_Consumption_L_per_100km'][:num_ev] = 0  # Set fuel consumption to zero for EVs
    return data_scenario

def apply_fuel_efficiency_improvement(data, efficiency_increase=0.15):
    """Increase fuel efficiency by a specified percentage (reduce fuel consumption)."""
    data_scenario = data.copy()
    data_scenario['Fuel_Consumption_L_per_100km'] *= (1 - efficiency_increase)
    return data_scenario

def apply_mileage_change(data, mileage_factor=1.1):
    """Adjust the average mileage for each vehicle to simulate increased/decreased usage."""
    data_scenario = data.copy()
    data_scenario['Average_Mileage_km'] *= mileage_factor
    return data_scenario

=====================================================================================================================

# 6.2 Calculating Emissions for Scenarios
# Now we can use these functions to adjust the data and predict emissions under each scenario.

# Scenario 1: Increase EV Adoption
X_ev_scenario = apply_electric_vehicle_scenario(X_test, ev_percentage=0.3)
y_ev_scenario = rf_model.predict(X_ev_scenario)

# Scenario 2: Fuel Efficiency Improvement
X_efficiency_scenario = apply_fuel_efficiency_improvement(X_test, efficiency_increase=0.2)
y_efficiency_scenario = rf_model.predict(X_efficiency_scenario)

# Scenario 3: Increased Mileage Usage
X_mileage_scenario = apply_mileage_change(X_test, mileage_factor=1.2)
y_mileage_scenario = rf_model.predict(X_mileage_scenario)

==========================================================================================================================================

# Part 7: Interactive Visualization with Streamlit
# To create an interactive dashboard, we’ll use Streamlit, a Python library that makes it easy to create data web apps with minimal code. 
# If you haven’t installed it, use:

!pip install streamlit

=========================================================================================================================

# 7.1 Streamlit Dashboard Code
# Here’s a simple code snippet to create a Streamlit app where users can adjust the parameters of each scenario and see the results.

import streamlit as st
import pandas as pd

# Set up Streamlit app
st.title("Zambia Transport Sector Emissions Prediction")

# Scenario controls
st.sidebar.header("Scenario Adjustments")
ev_percentage = st.sidebar.slider("Electric Vehicle Adoption (%)", 0.0, 1.0, 0.2)
efficiency_increase = st.sidebar.slider("Fuel Efficiency Improvement (%)", 0.0, 0.5, 0.1)
mileage_factor = st.sidebar.slider("Average Mileage Change (%)", 0.8, 1.5, 1.0)

# Apply scenarios
X_ev_scenario = apply_electric_vehicle_scenario(X_test, ev_percentage)
X_efficiency_scenario = apply_fuel_efficiency_improvement(X_test, efficiency_increase)
X_mileage_scenario = apply_mileage_change(X_test, mileage_factor)

# Calculate scenario predictions
y_ev_scenario = rf_model.predict(X_ev_scenario)
y_efficiency_scenario = rf_model.predict(X_efficiency_scenario)
y_mileage_scenario = rf_model.predict(X_mileage_scenario)

# Display results
st.subheader("Scenario Results")
st.write(f"Electric Vehicle Adoption Emissions Reduction (30% EVs): {y_ev_scenario.mean():.2f} kg CO₂")
st.write(f"Fuel Efficiency Improvement Emissions Reduction (10%): {y_efficiency_scenario.mean():.2f} kg CO₂")
st.write(f"Increased Mileage Emissions (20% increase): {y_mileage_scenario.mean():.2f} kg CO₂")

# Plot comparison
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, 'o', label='Original Emissions')
plt.plot(range(len(y_ev_scenario)), y_ev_scenario, 'o', label='EV Scenario')
plt.plot(range(len(y_efficiency_scenario)), y_efficiency_scenario, 'o', label='Efficiency Scenario')
plt.plot(range(len(y_mileage_scenario)), y_mileage_scenario, 'o', label='Mileage Increase Scenario')
plt.xlabel("Sample")
plt.ylabel("Annual Emissions (kg CO₂)")
plt.legend()
st.pyplot(plt)

=========================================================================================================
# Running the Streamlit App
# To run the app locally, use the following command in your terminal (within the directory where this file is saved):

streamlit run your_script_name.py

====================================================================

# Once the app is running, you can adjust the sidebar controls to see how different scenarios impact emissions.
# making it easier to visualize policy outcomes for Zambia’s transport sector.














