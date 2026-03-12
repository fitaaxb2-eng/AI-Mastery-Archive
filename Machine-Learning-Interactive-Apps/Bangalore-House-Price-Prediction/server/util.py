import json
import pickle
import numpy as np

# Global variables to store the model and column data
__locations = None
__data_columns = None
__model = None

# --- 1. Prediction Function ---
def get_estimated_price(location, sqft, bhk, bath):
    """
    Takes input features and returns the predicted price using the loaded model.
    """
    try:
        # Find the index of the location in the data columns
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        # If location is not found, set index to -1
        loc_index = -1

    # Create a numpy array of zeros with the same length as columns
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    # If the location index exists, set its value to 1 (One-Hot Encoding)
    if loc_index >= 0:
        x[loc_index] = 1

    # Return the predicted price rounded to 2 decimal places
    return round(__model.predict([x])[0], 2)


# --- 2. Helper Function to Get Location List ---
def get_location_names():
    return __locations


# --- 3. Loading Model & Metadata ---
def load_saved_artifacts():
    """
    Loads the saved JSON columns and Pickle model into memory.
    """
    print("Loading saved artifacts...start")
    global __data_columns
    global __locations
    global __model

    # Load data columns from JSON
    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        # The first 3 columns are sqft, bath, and bhk; the rest are locations
        __locations = __data_columns[3:]

    # Load the trained model from Pickle file
    if __model is None:
        with open("./artifacts/banglore_home_prices_model.pickle", 'rb') as f:
            __model = pickle.load(f)

    print("Loading saved artifacts...done")


# --- 4. Testing Block ---
if __name__ == '__main__':
    load_saved_artifacts()
    # Test cases to verify the functionality
    print("Location Names:", get_location_names()[:5]) # Show first 5 locations
    print("Prediction (1st Phase JP Nagar):", get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    print("Prediction (Unknown Location):", get_estimated_price('Ejipura', 1000, 2, 2))