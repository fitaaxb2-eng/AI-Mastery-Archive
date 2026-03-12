// --- Function to get the number of bathrooms selected ---
function getBathValue() {
    var uiBathrooms = document.getElementsByName("uiBathrooms");
    for(var i in uiBathrooms) {
      if(uiBathrooms[i].checked) {
          // Returning the index + 1 as the number of bathrooms
          return parseInt(i) + 1;
      }
    }
    return -1; // Invalid Value
}

// --- Function to get the BHK (Bedrooms) count selected ---
function getBHKValue() {
    var uiBHK = document.getElementsByName("uiBHK");
    for(var i in uiBHK) {
      if(uiBHK[i].checked) {
          // Returning the index + 1 as the BHK value
          return parseInt(i) + 1;
      }
    }
    return -1; // Invalid Value
}

// --- Function triggered when "Estimate Price" button is clicked ---
function onClickedEstimatePrice() {
    console.log("Estimate price button clicked");
    var sqft = document.getElementById("uiSqft");
    var bhk = getBHKValue();
    var bathrooms = getBathValue();
    var location = document.getElementById("uiLocations");
    var estPrice = document.getElementById("price-value");

    // The endpoint URL of our Flask Server
    var url = "http://127.0.0.1:5000/predict_home_price";

    // Making a POST request using jQuery AJAX
    $.post(url, {
        total_sqft: parseFloat(sqft.value),
        bhk: bhk,
        bath: bathrooms,
        location: location.value
    }, function(data, status) {
        console.log("Estimated Price:", data.estimated_price);
        // Updating the UI with the predicted price
        estPrice.innerHTML = "<h2>" + data.estimated_price.toString() + " Lakh</h2>";
    });
}

// --- Function to populate location dropdown on page load ---
function onPageLoad() {
    console.log("document loaded");
    var url = "http://127.0.0.1:5000/get_location_names";

    // Fetching location names from the Flask server
    $.get(url, function(data, status) {
        console.log("got response for get_location_names request");
        if(data) {
            var locations = data.locations;
            var uiLocations = document.getElementById("uiLocations");
            $('#uiLocations').empty(); // Clear existing options
            for(var i in locations) {
                var opt = new Option(locations[i]);
                $('#uiLocations').append(opt); // Add each location to the dropdown
            }
        }
    });
}

// Initialize the page load function
window.onload = onPageLoad;