import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import Label, Entry, Button

# Loading the saved model
model = joblib.load('model_FasttagFraudDetection.pkl')

# Function to get prediction
def get_prediction():
    user_data = {
        'Month': int(month_entry.get()),
        'distance_from_city_center': float(distance_entry.get()),
        'Transaction_Amount': float(amount_entry.get()),
        'Amount_paid': float(amount_paid_entry.get()),
    }

    feature_names = ['Transaction_Amount', 'Amount_paid', 'Month', 'distance_from_city_center']

    user_data_df = pd.DataFrame([user_data], columns=feature_names)

    # Making predictions using the loaded model
    predictionbinary = model.predict(user_data_df)
    if predictionbinary == 0:
        prediction = "fraud"
    else:
        prediction = " not fraud"

    # Updating the result label
    result_label.config(text="Predicted Fraud Indicator: {}".format(prediction))

# Creating the main window
root = tk.Tk()
root.title("Fasttag Fraud Detection")

# Creating labels and entry widgets for user input
Label(root, text="Month (1-12):").grid(row=0, column=0)
month_entry = Entry(root)
month_entry.grid(row=0, column=1)

Label(root, text="Distance from City Center (km):").grid(row=1, column=0)
distance_entry = Entry(root)
distance_entry.grid(row=1, column=1)

Label(root, text="Transaction Amount:").grid(row=2, column=0)
amount_entry = Entry(root)
amount_entry.grid(row=2, column=1)

Label(root, text="Amount Paid:").grid(row=3, column=0)
amount_paid_entry = Entry(root)
amount_paid_entry.grid(row=3, column=1)

# Creating a button to trigger the prediction
predict_button = Button(root, text="Predict", command=get_prediction)
predict_button.grid(row=4, column=0, columnspan=2)

# Creating a label to display the prediction result
result_label = Label(root, text="Predicted Fraud Indicator: ")
result_label.grid(row=5, column=0, columnspan=2)

# Running the GUI
root.mainloop()
