import pandas as pd
from sklearn.preprocessing import LabelEncoder
from joblib import load

def preprocess(data):
    # Selecting the same columns as used for training (columns 2 to 18)
    data_processed = data.iloc[:, 1:18]  # Columns H to X in the original data
    return data_processed

# Load new data
new_data_path = 'data2.xlsx'  # Replace with your actual file path
new_data = pd.read_excel(new_data_path)

# Preprocess the new data
processed_new_data = preprocess(new_data)

# Load the XGBoost model
xgb_model = load('xgboost_model.joblib')

# Load the LabelEncoder
label_encoder = load('label_encoder.joblib')  # Assuming you have saved the LabelEncoder as 'label_encoder.joblib'

# Making predictions
predictions = xgb_model.predict(processed_new_data)

# Decoding the predictions into original labels
predicted_segments = label_encoder.inverse_transform(predictions)

# Add the predictions to the DataFrame
new_data['Predicted Segment'] = predicted_segments

# Save the DataFrame with predictions to a new Excel file
output_path = 'classified_data.xlsx'
new_data.to_excel(output_path, index=False)

print(f"Classified data saved to {output_path}")
