import streamlit as st
from joblib import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load your Random Forest model, LabelEncoders, and StandardScaler
model = load('random_forest_model.joblib')
label_encoder_gender = load('label_encoder_gender.joblib')
label_encoder_smoking = load('label_encoder_smoking.joblib')
scaler = load('scaler.joblib')

# Define the mapping for the prediction labels
PREDICTION_LABELS = {0: "You may not have diabetes", 1: "You may have diabetes"}

# Function to handle encoding
def encode_feature(label_encoder, feature_value):
    feature_value = feature_value.lower()  # Convert input to lowercase
    try:
        return label_encoder.transform([feature_value])[0]
    except ValueError:
        raise ValueError(f"Feature value '{feature_value}' not found in the LabelEncoder categories.")

# Function to encode and standardize data
def preprocess_data(data):
    # Encode categorical features
    data['gender'] = data['gender'].apply(lambda x: encode_feature(label_encoder_gender, x))
    data['smoking_history'] = data['smoking_history'].apply(lambda x: encode_feature(label_encoder_smoking, x))
    
    # Standardize the data
    data_scaled = scaler.transform(data)
    
    return pd.DataFrame(data_scaled, columns=data.columns)

# Function to make predictions
def make_prediction(data):
    # Make prediction
    prediction = model.predict(data)
    
    # Map predictions to human-readable labels
    prediction_labels = [PREDICTION_LABELS[p] for p in prediction]
    return prediction_labels

# Function to handle and display results
def predict_and_display(data):
    # Preprocess the data (encode and standardize)
    preprocessed_data = preprocess_data(data)
    
    # Make predictions
    predictions = make_prediction(preprocessed_data)

    # Combine the input data and predictions into a DataFrame
    results_df = data.copy()
    results_df['Prediction'] = predictions

    # Display the final prediction result
    st.write("Prediction Result:")
    st.table(results_df[['Prediction']])

    # Display histogram of predictions
    st.write("Histogram of Predictions:")
    fig, ax = plt.subplots()
    prediction_counts = pd.Series(predictions).value_counts().sort_index()
    prediction_counts.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
    ax.set_title("Distribution of Diabetes Predictions")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Ensure y-axis has integer ticks
    st.pyplot(fig)

# Streamlit application starts here
def main():
    # Title of your web app
    st.title("Diabetes Prediction")

    # Sidebar for navigation
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose how to input data", ["Enter text", "Upload file"])

    if option == "Enter text":
        # Text boxes for user input
        gender_input = st.selectbox("Select Gender", ["female", "male"])
        age_input = st.number_input("Enter Age", min_value=1, max_value=100)
        hypertension_input = st.radio("Hypertension (No=0, Yes=1)", (0, 1))
        heart_disease_input = st.radio("Heart Disease (No=0, Yes=1)", (0, 1))
        smoking_history_input = st.selectbox("Select Smoking History", ["current", "ever", "former", "never", "not current"])
        bmi_input = st.number_input("Enter BMI (kg/m²)", format="%.2f")
        HbA1c_level_input = st.number_input("Enter HbA1c Level (%)", format="%.2f")
        blood_glucose_level_input = st.number_input("Enter Blood Glucose Level (mg/dL)", format="%.2f")

        # Predict button
        if st.button('Predict'):
            # Validate input values
            if age_input <= 0 or bmi_input <= 0.00 or HbA1c_level_input <= 0.00 or blood_glucose_level_input <= 0.00:
                st.error("Age, BMI, HbA1c Level, and Blood Glucose Level must be greater than 0.00.")
                return

            # Create a DataFrame with the input data
            input_data = pd.DataFrame({
                'gender': [gender_input],
                'age': [age_input],
                'hypertension': [hypertension_input],
                'heart_disease': [heart_disease_input],
                'smoking_history': [smoking_history_input],
                'bmi': [bmi_input],
                'HbA1c_level': [HbA1c_level_input],
                'blood_glucose_level': [blood_glucose_level_input]
            })

            # Make prediction and display results
            predict_and_display(input_data)

    elif option == "Upload file":
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'txt'])
        if uploaded_file is not None:
            # Check if file is empty
            if uploaded_file.size == 0:
                st.error("Uploaded file is empty. Please upload a valid file.")
                return

            # Read the file into a DataFrame
            try:
                if uploaded_file.type == "text/csv" or uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                elif uploaded_file.type == "text/plain" or uploaded_file.name.endswith('.txt'):
                    # Assuming the text file is comma-separated
                    data = pd.read_csv(uploaded_file, delimiter=',')
                else:
                    st.error("Unsupported file type. Please upload a CSV or TXT file.")
                    return
                
                # Normalize column names to lowercase
                data.columns = [col.lower() for col in data.columns]

                # Show a preview of the data
                st.write("Preview of uploaded data:")
                st.dataframe(data.head())

                # Define required columns in lowercase
                required_columns = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'hba1c_level', 'blood_glucose_level']

                # Check if the DataFrame has the required columns
                if all(col in data.columns for col in required_columns):
                    # Check for missing values
                    if data[required_columns].isnull().any().any():
                        st.error("Uploaded file contains missing values. Please ensure all required fields are filled.")
                    else:
                        # Ensure categorical columns are strings
                        data['gender'] = data['gender'].astype(str)
                        data['smoking_history'] = data['smoking_history'].astype(str)

                        # Make prediction and display results
                        predict_and_display(data[required_columns])
                else:
                    missing_cols = [col for col in required_columns if col not in data.columns]
                    st.error(f"Uploaded file is missing required columns: {', '.join(missing_cols)}")
                
            except Exception as e:
                st.error(f"Error reading the file: {e}")
if __name__ == '__main__':
    main()
