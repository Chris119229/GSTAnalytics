import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained models and scalers
gb_clf = joblib.load('gb_clf.pkl')
svm_clf = joblib.load('svm_clf.pkl')
scaler = joblib.load('scaler.pkl')

# Load label encoders (if applicable)
label_encoders = joblib.load('label_encoders.pkl')

def preprocess_input(input_data, label_encoders):
    """
    Preprocess the input data by applying label encoding.
    
    Args:
        input_data (pd.DataFrame): Raw input data for prediction.
        label_encoders (dict): Pre-fitted label encoders for each column.
    
    Returns:
        np.array: Processed and encoded input data.
    """
    for column in input_data.columns:
        input_data[column] = label_encoders[column].transform(input_data[column])
    return input_data

def predict(input_data):
    """
    Predict using the hybrid model (Gradient Boosting + SVM).
    
    Args:
        input_data (pd.DataFrame): Input data for prediction, in the same feature format used in training.
    
    Returns:
        Prediction result (e.g., class label or regression value).
    """
    # Ensure the input data is a DataFrame
    input_data = pd.DataFrame(input_data)

    # Apply label encoding to input data
    input_data = preprocess_input(input_data, label_encoders)

    # Predict with the Gradient Boosting model
    gb_preds = gb_clf.predict_proba(input_data)[:, 1]

    # Combine Gradient Boosting output with original features
    input_data_hybrid = np.hstack((input_data, gb_preds.reshape(-1, 1)))

    # Scale the input features
    input_data_scaled = scaler.transform(input_data_hybrid)

    # Perform prediction with SVM
    prediction = svm_clf.predict(input_data_scaled)

    return prediction

# Example usage:
if __name__ == "__main__":
    # Replace with actual input data (adjust columns accordingly)
    sample_input = {'feature1': [1], 'feature2': [0], 'feature3': [1], 'feature4': [0]}
    result = predict(sample_input)
    print(f"Prediction: {result}")
