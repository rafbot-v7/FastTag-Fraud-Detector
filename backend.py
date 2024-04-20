from flask import Flask, render_template, request, jsonify
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import geodesic
import joblib


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_data', methods=['POST'])
def process_data():
    data = request.json
    print(data)
    if isinstance(data, dict):
        data = [data]  # Convert single dictionary to a list containing one dictionary
    
    # Create DataFrame
     
    # Here, you can process the data as needed
    # For example, you can print it and return a response
    
    df = pd.DataFrame(data)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    # Extracting useful features from timestamp
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month
# Extract relevant features
    selected_features = ['Transaction_Amount', 'Amount_paid', 'Geographical_Location', 'Month']

# Create a new DataFrame with selected features
    new_data = df[selected_features]


    # Load the trained model
    model = joblib.load('model_FasttagFraudDetection.pkl')

    # Load the scaler object
    scaler = joblib.load('scaler.pkl')  # Make sure you've saved the scaler during training

    # Load your new data (assuming it's in a DataFrame called 'new_data')

    # Feature Extraction Using Haversine Distance
    reference_point = (13.059816123454882, 77.77068662374292)
    new_data['distance_from_city_center'] = new_data['Geographical_Location'].apply(
        lambda x: geodesic(reference_point, tuple(map(float, x.split(',')))).kilometers
    )

    # Scaling
    new_data[['Transaction_Amount', 'Amount_paid']] = scaler.transform(
        new_data[['Transaction_Amount', 'Amount_paid']]
    )

    # Now, select only the relevant features used for training
    new_data = new_data[['Transaction_Amount', 'Amount_paid', 'Month', 'distance_from_city_center']]

    # Handling NaN values (if any)
    new_data.fillna(new_data.mean(), inplace=True)

    # Make predictions
    predictionbinary = model.predict(new_data)

    if predictionbinary == 0:
            prediction = "fraud"
    else:
            prediction = " not fraud"
    print(prediction)
    
    # Return a response (optional)
    return jsonify({'prediction': prediction}), 200

if __name__ == '__main__':
    app.run(debug=True)
