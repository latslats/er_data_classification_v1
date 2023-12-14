# Import necessary libraries
import streamlit as st
import pandas as pd
from joblib import load
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess(data):
    # Selecting the same columns as used for training (columns 2 to 18)
    data_processed = data.iloc[:, 1:18]  # Columns H to X in the original data
    return data_processed

def main():
    st.title("Data Classification App")

    # File uploader for the user to upload their data
    uploaded_file = st.file_uploader("Upload your data file", type=["xlsx"])

    if uploaded_file is not None:
        # Read the uploaded data
        data = pd.read_excel(uploaded_file)

        # Preprocess the data
        processed_data = preprocess(data)

        # Load the model and label encoder
        xgb_model = load('xgboost_model.joblib')
        label_encoder = load('label_encoder.joblib')

        # Make predictions
        predictions = xgb_model.predict(processed_data)
        predicted_segments = label_encoder.inverse_transform(predictions)

        # Show the predictions
        data['Predicted Segment'] = predicted_segments
        st.write(data)

        # Option to download the classified data
        st.download_button(label="Download Classified Data",
                           data=data.to_csv(index=False),
                           file_name="classified_data.csv",
                           mime="text/csv")

if __name__ == "__main__":
    main()
