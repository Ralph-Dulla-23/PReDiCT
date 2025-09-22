import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder
import os

# --- INITIALIZE THE FLASK APP ---
app = Flask(__name__)

# --- LOAD THE TRAINED SVM MODEL AND ARTIFACTS ---
print("Loading the trained SVM model and scaler...")
try:
    svm_model = joblib.load('svm_multiclass_risk_model.joblib')
    scaler = joblib.load('svm_multiclass_scaler.joblib')
    # Get the exact list of feature names the model was trained on from the scaler
    TRAINING_COLUMNS = scaler.get_feature_names_out()
    print("SVM Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler files: {e}")
    svm_model = None
    scaler = None

# Manually create the label encoder to decode predictions
label_encoder = LabelEncoder()
# IMPORTANT: This order must match the order from your training notebook's label_encoder.classes_
RISK_CLASSES = ['High Risk', 'Low Risk', 'Moderate Risk'] 
label_encoder.fit(RISK_CLASSES)


# --- PREPROCESSING FUNCTION for the Web App ---
def preprocess_web_input(data):
    """
    Takes raw form data (as a list of visit dictionaries), aggregates it into a 
    single patient profile, and prepares it for the SVM model.
    """
    df = pd.DataFrame(data)

    # --- Replicate the exact aggregation logic from your training script ---
    # Create helper columns
    df['Has_STI_History'] = df['diagnosed_sti'].apply(lambda x: 1 if x != 'None' else 0)
    df['transactional_sex'] = df['transactional_sex'].apply(lambda x: 1 if x == 'True' else 0)
    df['substance_use'] = df['substance_use'].apply(lambda x: 1 if x == 'True' else 0)
    df['hiv_screen_positive_count'] = df['hiv_screen'].apply(lambda x: 1 if x == 'Positive' else 0)
    df['num_partners'] = pd.to_numeric(df['num_partners'], errors='coerce').fillna(0)
    df['age_group'] = pd.to_numeric(df['age_group'], errors='coerce').fillna(0)
    
    # Create the single aggregated profile row
    profile = {
        'age': [df['age_group'].iloc[-1]],
        'gender': [df['gender'].iloc[-1]],
        'sexual_orientation': [df['sexual_orientation'].iloc[-1]],
        'education': [df['education'].iloc[-1]],
        'total_visits': [df['Visit_Num'].max()],
        'avg_partners': [df['num_partners'].mean()],
        'max_partners_in_visit': [df['num_partners'].max()],
        'total_partners_reported': [df['num_partners'].sum()],
        'ever_had_transactional_sex': [df['transactional_sex'].max()],
        'ever_used_substances': [df['substance_use'].max()],
        'last_visit_reason': [df['reason_for_visit'].iloc[-1]],
        'last_prep_procedure': [df['prep_procedure'].iloc[-1]],
        'has_sti_history': [df['Has_STI_History'].max()],
        'hiv_screen_<lambda>': [df['hiv_screen_positive_count'].sum()]
    }
    df_profile = pd.DataFrame.from_dict(profile)
    
    # One-hot encode and reindex to match the training data structure
    df_encoded = pd.get_dummies(df_profile)
    df_reindexed = df_encoded.reindex(columns=TRAINING_COLUMNS, fill_value=0)
    
    return df_reindexed

# --- DEFINE THE WEBSITE ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if svm_model is None or scaler is None:
        return "Model or Scaler not loaded. Please check server logs."

    # For this demo, we treat the single form submission as a patient's entire history.
    # In a real app, you would fetch all past visits from a database for this patient.
    form_data = [request.form.to_dict()]
    print("Received form data:", form_data)
    
    # Process the input data using the robust function
    processed_data = preprocess_web_input(form_data)
    
    # Scale the processed data using the loaded scaler
    scaled_data = scaler.transform(processed_data)
    
    # Make a prediction and get probabilities
    predicted_index = svm_model.predict(scaled_data)[0]
    prediction_proba = svm_model.predict_proba(scaled_data)
    
    # Decode the prediction and get confidence
    predicted_risk = label_encoder.inverse_transform([predicted_index])[0]
    confidence = np.max(prediction_proba) * 100
    
    print(f"Prediction: {predicted_risk}, Confidence: {confidence:.2f}%")
    
    return render_template('result.html', prediction=predicted_risk, confidence=f"{confidence:.2f}%")

# --- RUN THE APP ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)