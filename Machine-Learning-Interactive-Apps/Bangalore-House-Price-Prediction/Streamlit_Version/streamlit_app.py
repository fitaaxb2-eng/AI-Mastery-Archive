# =================================================================================
# HOW TO RUN THIS PROJECT IN TERMINAL:
# ---------------------------------------------------------------------------------
# STEP 1: Navigate to the project directory
# cd C:\Users\cxc\Desktop\BHP\Streamlit_Version
#
# STEP 2: Run the app using the Python module path
# & "C:\Users\cxc\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run streamlit_app.py
# =================================================================================

import streamlit as st
import pickle
import json
import numpy as np
import os

# Set the page configuration for the browser tab
st.set_page_config(page_title="Bangalore House Price Predictor")


# Function to load the machine learning model and the location columns
def load_artifacts():
    # Verify if the necessary files exist in the current directory
    if not os.path.exists("columns.json") or not os.path.exists("banglore_home_prices_model.pickle"):
        return None, None, None

    # Load the location names from the JSON file
    with open("columns.json", "r") as f:
        data_columns = json.load(f)['data_columns']

    # Load the trained Pickle model file
    with open("banglore_home_prices_model.pickle", "rb") as f:
        model = pickle.load(f)

    # Extract the location names (skipping the first 3 columns: sqft, bath, bhk)
    locations = data_columns[3:]
    return data_columns, model, locations


# Initialize and load the data
data_columns, model, locations = load_artifacts()

# --- Main User Interface ---
st.title("Bangalore House Price Predictor")
st.write("Enter the property details below to estimate the price.")

# Check if artifacts were loaded correctly before showing the UI
if locations is None:
    st.error("Error: 'columns.json' or 'model.pickle' files are missing. Please check your folder.")
else:
    # Use columns to create a balanced layout
    col1, col2 = st.columns(2)

    with col1:
        # User selects the location (labels are capitalized for better look)
        selected_location = st.selectbox("Select Location", [loc.title() for loc in locations])
        # User inputs the total area in square feet
        total_sqft = st.number_input("Total Area (Square Feet)", min_value=300, max_value=10000, value=1000)

    with col2:
        # User selects the number of bedrooms (BHK)
        bhk = st.selectbox("Number of Bedrooms (BHK)", [1, 2, 3, 4, 5], index=1)
        # User selects the number of bathrooms
        bath = st.selectbox("Number of Bathrooms", [1, 2, 3, 4, 5], index=1)

    # Trigger prediction when the button is clicked
    if st.button("Predict Price"):
        try:
            # Map the selected location back to the model's column index
            loc_index = data_columns.index(selected_location.lower())

            # Initialize the input vector with zeros
            x = np.zeros(len(data_columns))
            x[0] = total_sqft
            x[1] = bath
            x[2] = bhk

            # Apply One Hot Encoding: set the index of the selected location to 1
            if loc_index >= 0:
                x[loc_index] = 1

            # Predict the price using the loaded model
            prediction = model.predict([x])[0]

            # Show the result formatted to 2 decimal places
            st.success(f"Estimated Price: {round(prediction, 2)} Lakhs")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# --- Footer ---
st.markdown("---")
st.caption("Bangalore Real Estate Price Prediction Project | Developed for Educational Purposes")