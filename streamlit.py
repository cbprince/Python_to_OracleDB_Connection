#To run this code through a webpage interface, you can deploy it as a web app using **Streamlit**, which is perfect for quick, interactive web applications in Python. Here’s a step-by-step guide for creating, running, and deploying this as a webpage.

### Step 1: Setting Up Streamlit for Local Development

# 1. **Install Streamlit** (if you haven't already):
 #  ```bash
   pip install streamlit
   ```

2. **Save the Code**: Copy the code below into a Python file, e.g., `transport_emissions_app.py`.

### Step 2: Writing the Streamlit App Code

#```python

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Sample data setup
data = {
    'Vehicle_Type': ['Car', 'Bus', 'Truck', 'Motorbike', 'Taxi'],
    'Fuel_Type': ['Petrol', 'Diesel', 'Diesel', 'Petrol', 'Petrol'],
    'Average_Mileage_km': [15000, 30000, 40000, 10000, 20000],
    'Fuel_Consumption_L_per_100km': [8, 25, 30, 3, 7],
    'Emission_Factor_kg_CO2_per_L': [2.31, 2.68, 2.68, 2.31, 2.31]
}
df = pd.DataFrame(data)

# Calculate emissions per vehicle type
df['Annual_Emissions_kg_CO2'] = (df['Average_Mileage_km'] / 100) * df['Fuel_Consumption_L_per_100km'] * df['Emission_Factor_kg_CO2_per_L']

# Encode categorical variables
label_encoder = LabelEncoder()
df['Vehicle_Type_Encoded'] = label_encoder.fit_transform(df['Vehicle_Type'])
df['Fuel_Type_Encoded'] = label_encoder.fit_transform(df['Fuel_Type'])

# Select features and target variable
X = df[['Vehicle_Type_Encoded', 'Fuel_Type_Encoded', 'Average_Mileage_km', 'Fuel_Consumption_L_per_100km']]
y = df['Annual_Emissions_kg_CO2']

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Functions for scenario adjustments
def apply_electric_vehicle_scenario(data, ev_percentage=0.2):
    data_scenario = data.copy()
    num_ev = int(ev_percentage * len(data_scenario))
    data_scenario['Fuel_Consumption_L_per_100km'][:num_ev] = 0  # Set fuel consumption to zero for EVs
    return data_scenario

def apply_fuel_efficiency_improvement(data, efficiency_increase=0.15):
    data_scenario = data.copy()
    data_scenario['Fuel_Consumption_L_per_100km'] *= (1 - efficiency_increase)
    return data_scenario

def apply_mileage_change(data, mileage_factor=1.1):
    data_scenario = data.copy()
    data_scenario['Average_Mileage_km'] *= mileage_factor
    return data_scenario

# Streamlit app interface
st.title("Zambia Transport Sector Emissions Prediction")

# Sidebar inputs for scenario adjustments
st.sidebar.header("Scenario Adjustments")
ev_percentage = st.sidebar.slider("Electric Vehicle Adoption (%)", 0.0, 1.0, 0.2)
efficiency_increase = st.sidebar.slider("Fuel Efficiency Improvement (%)", 0.0, 0.5, 0.1)
mileage_factor = st.sidebar.slider("Average Mileage Change (%)", 0.8, 1.5, 1.0)

# Apply scenarios
X_ev_scenario = apply_electric_vehicle_scenario(X, ev_percentage)
X_efficiency_scenario = apply_fuel_efficiency_improvement(X, efficiency_increase)
X_mileage_scenario = apply_mileage_change(X, mileage_factor)

# Predict emissions for each scenario
y_ev_scenario = rf_model.predict(X_ev_scenario)
y_efficiency_scenario = rf_model.predict(X_efficiency_scenario)
y_mileage_scenario = rf_model.predict(X_mileage_scenario)

# Display results
st.subheader("Scenario Results")
st.write(f"Electric Vehicle Adoption Emissions Reduction (30% EVs): {y_ev_scenario.mean():.2f} kg CO₂")
st.write(f"Fuel Efficiency Improvement Emissions Reduction (10%): {y_efficiency_scenario.mean():.2f} kg CO₂")
st.write(f"Increased Mileage Emissions (20% increase): {y_mileage_scenario.mean():.2f} kg CO₂")

# Plot the comparison of emissions
plt.figure(figsize=(10, 6))
plt.plot(range(len(y)), y, 'o', label='Original Emissions')
plt.plot(range(len(y_ev_scenario)), y_ev_scenario, 'o', label='EV Scenario')
plt.plot(range(len(y_efficiency_scenario)), y_efficiency_scenario, 'o', label='Efficiency Scenario')
plt.plot(range(len(y_mileage_scenario)), y_mileage_scenario, 'o', label='Mileage Increase Scenario')
plt.xlabel("Sample")
plt.ylabel("Annual Emissions (kg CO₂)")
plt.legend()
st.pyplot(plt)
```
===============================================================================================================
### Step 3: Run the App Locally
# To test the app locally, open a terminal in the directory where `transport_emissions_app.py` is saved and run:

```bash
streamlit run transport_emissions_app.py
```

# This will open a local server (usually at `http://localhost:8501`), where you can interact with the app.

==================================================================================================================
### Step 4: Deploy the App Online

For public access, you can deploy your app with **Streamlit Cloud**:

1. **Sign up** at [Streamlit Cloud](https://streamlit.io/cloud).
2. Link your GitHub repository containing `transport_emissions_app.py`.
3. Click **New App**, select your repository, and specify the branch and filename (`transport_emissions_app.py`).
4. **Deploy** the app, and Streamlit Cloud will provide you with a URL to access it.

This setup should give you a fully interactive, web-based emissions prediction tool,
 accessible by anyone with the link! Let me know if you need any further customization for this interface.