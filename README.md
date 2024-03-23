## Fastag Fraud Detection Model

### Overview
This repository contains code for building a machine learning model to detect fraud in Fastag transactions. Fastag is an electronic toll collection system in India, and fraud detection in such transactions is crucial for ensuring security and reliability.

### Dataset
The dataset used for training and testing the model is stored in the file `FastagFraudDetection.csv`. It contains information about various features related to Fastag transactions, including timestamps, geographical locations, transaction amounts, vehicle types, lane types, and more.

### Features Engineering
- Extracted useful features from timestamps such as hour, day of the week, and month.
- Applied one-hot encoding to categorical features like vehicle type and lane type.
- Used label encoding to convert the target variable (fraud indicator) into numerical format.
- Utilized Haversine distance to compute the distance of transaction locations from the city center.
- Scaled numerical features using MinMaxScaler.
- Selected relevant features based on correlation with the target variable.

### Model Training and Evaluation
- Split the dataset into training and testing sets (80% training, 20% testing).
- Handled class imbalance using RandomOverSampler to oversample the minority class.
- Trained the model using BalancedRandomForestClassifier.
- Made predictions on the test set and evaluated the model's performance using accuracy, classification report, and confusion matrix.

### Model Deployment
- The trained model (`model_FasttagFraudDetection.pkl`) is saved using joblib for future use.

### Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- geopy
- imbalanced-learn
- seaborn
- matplotlib

### Usage
1. Clone the repository.
2. Install the required dependencies 
3. Run the script to train the model and make predictions.
4. Use the saved model (`model_FasttagFraudDetection.pkl`) for inference or deployment.

 
