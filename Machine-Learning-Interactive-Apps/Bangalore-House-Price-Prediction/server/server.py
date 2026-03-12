from flask import Flask, request, jsonify
import util

app = Flask(__name__)


# --- API 1: Fetching Location Names ---
# This endpoint returns a list of all unique locations for the UI dropdown.
@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': util.get_location_names()
    })
    # Enable CORS to allow requests from the frontend (HTML/JS)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


# --- API 2: Predicting Home Price ---
# This endpoint receives input from the user and returns the estimated price.
@app.route('/predict_home_price', methods=['GET', 'POST'])
def predict_home_price():
    # Read input values from the request form
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    # Call the utility function to get the prediction from the model
    estimated_price = util.get_estimated_price(location, total_sqft, bhk, bath)

    # Return the result as a JSON response
    response = jsonify({
        'estimated_price': estimated_price
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


# --- Main: Starting the Server ---
if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")

    # Pre_load the saved model and column data (Artifacts)
    util.load_saved_artifacts()

    # Run the Flask app
    app.run()


