import streamlit as st
from joblib import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load your Random Forest model, LabelEncoders, and StandardScaler
try:
    model = load('random_forest_model.joblib')
    label_encoder_gender = load('label_encoder_gender.joblib')
    label_encoder_smoking = load('label_encoder_smoking.joblib')
    scaler = load('scaler.joblib')
except Exception as e:
    st.error(f"Error loading model files: {e}")
    model = None
    label_encoder_gender = None
    label_encoder_smoking = None
    scaler = None

# Function to handle encoding
def encode_feature(label_encoder, feature_value):
    feature_value = feature_value.lower()  # Convert input to lowercase
    try:
        return label_encoder.transform([feature_value])[0]
    except ValueError:
        raise ValueError(f"Feature value '{feature_value}' not found in the LabelEncoder categories.")

# Function to make predictions
def make_prediction(data):
    if model and scaler:
        # Standardize the input data
        data_scaled = scaler.transform(data)
        # Make prediction
        prediction = model.predict(data_scaled)
        return prediction
    else:
        st.error("Model or scaler not loaded.")
        return []

# Function to handle and display results
def predict_and_display(data):
    # Make predictions
    predictions = make_prediction(data)

    # Combine the input data and predictions into a DataFrame
    data['Prediction'] = predictions
    results_df = data

    # Tabulate and display the results
    with st.expander("Show/Hide Prediction Table"):
        st.table(results_df)

    # Display histogram of predictions
    st.write("Histogram of Predictions:")
    fig, ax = plt.subplots()
    prediction_counts = pd.Series(predictions).value_counts().sort_index()
    prediction_counts.plot(kind='bar', ax=ax)
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
    option = st.sidebar.selectbox("Choose how to input data", ["Enter data manually", "Upload file"])

    if option == "Enter data manually":
        # Text boxes for user input
        gender_input = st.selectbox("Select gender", ["female", "male", "other"])
        age_input = st.number_input("Enter age", min_value=0)
        hypertension_input = st.radio("Hypertension (0=No, 1=Yes)", (0, 1))
        heart_disease_input = st.radio("Heart disease (0=No, 1=Yes)", (0, 1))
        smoking_history_input = st.selectbox("Select smoking history", ["current", "ever", "former", "never"])
        bmi_input = st.number_input("Enter BMI", format="%.2f")
        HbA1c_level_input = st.number_input("Enter HbA1c level", format="%.2f")
        blood_glucose_level_input = st.number_input("Enter blood glucose level", format="%.2f")

        # Predict button
        if st.button('Predict'):
            if label_encoder_gender and label_encoder_smoking:
                # Encode categorical features
                try:
                    gender_encoded = encode_feature(label_encoder_gender, gender_input)
                    smoking_history_encoded = encode_feature(label_encoder_smoking, smoking_history_input)
                    
                    # Create a DataFrame with the input data
                    input_data = pd.DataFrame({
                        'gender': [gender_encoded],
                        'age': [age_input],
                        'hypertension': [hypertension_input],
                        'heart_disease': [heart_disease_input],
                        'smoking_history': [smoking_history_encoded],
                        'bmi': [bmi_input],
                        'HbA1c_level': [HbA1c_level_input],
                        'blood_glucose_level': [blood_glucose_level_input]
                    })
                    
                    # Make prediction and display results
                    predict_and_display(input_data)
                except ValueError as e:
                    st.error(e)
            else:
                st.error("Label encoders not loaded.")

    elif option == "Upload file":
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'txt'])
        if uploaded_file is not None:
            if uploaded_file.type == "text/csv" or uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:  # Assume text file
                data = pd.read_csv(uploaded_file, sep="\t", header=None, names=['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'])

            # Check if the file has the right columns
            required_columns = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
            if all(col in data.columns for col in required_columns):
                if label_encoder_gender and label_encoder_smoking:
                    # Encode categorical features
                    data['gender'] = data['gender'].apply(lambda x: encode_feature(label_encoder_gender, x))
                    data['smoking_history'] = data['smoking_history'].apply(lambda x: encode_feature(label_encoder_smoking, x))
                    
                    # Create DataFrame with only the required columns
                    input_data = data[required_columns]
                    
                    # Standardize the data
                    input_data_scaled = scaler.transform(input_data)
                    
                    # Make predictions and display results
                    predict_and_display(pd.DataFrame(input_data_scaled, columns=required_columns))
                else:
                    st.error("Label encoders not loaded.")
            else:
                st.error("Uploaded file does not have the required columns.")

if __name__ == '__main__':
    main()
