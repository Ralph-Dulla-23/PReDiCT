from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import os

# --- INITIALIZE THE FLASK APP ---
app = Flask(__name__)

# --- LOAD THE TRAINED XGBOOST MODEL ---
print("Loading the trained XGBoost model...")
try:
    xgb_model = joblib.load('xgb_multiclass_risk_model.joblib')
    MODEL_EXPECTED_FEATURES = xgb_model.get_booster().feature_names
    print("XGBoost Model loaded successfully.")
    print(f"Model expects these features: {MODEL_EXPECTED_FEATURES}")
except Exception as e:
    print(f"Error loading model file: {e}")
    xgb_model = None

# Manually create the label encoder
label_encoder = LabelEncoder()
# IMPORTANT: Ensure this order matches the order from your training notebook's label_encoder.classes_
RISK_CLASSES = ['High Risk', 'Low Risk', 'Moderate Risk'] 
label_encoder.fit(RISK_CLASSES)

# --- PREPROCESSING FUNCTION for XGBoost (SIMPLIFIED & CORRECTED) ---
def preprocess_input_for_xgb(form_data):
    """
    Takes raw form data from the website, creates the required features,
    and builds a DataFrame that perfectly matches the model's expectations.
    """
    # 1. Create a single-row DataFrame from the form data
    df = pd.DataFrame([form_data])
    
    # 2. Convert data types safely
    # For a single visit, mean and max are the same as the value itself.
    df['age_group_last'] = pd.to_numeric(df['age_group'], errors='coerce').fillna(0)
    df['num_partners_mean'] = pd.to_numeric(df['num_partners'], errors='coerce').fillna(0)
    df['num_partners_max'] = pd.to_numeric(df['num_partners'], errors='coerce').fillna(0)
    df['transactional_sex_max'] = df['transactional_sex'].apply(lambda x: 1 if x == 'True' else 0)
    df['substance_use_max'] = df['substance_use'].apply(lambda x: 1 if x == 'True' else 0)
    # The tricky lambda column name from training
    df['diagnosed_sti_<lambda>'] = df['diagnosed_sti'].apply(lambda x: 1 if x != 'None' else 0) 
    
    # Handle the categorical feature
    df['sexual_orientation_last'] = df['sexual_orientation'].astype('category')
    
    # 3. Create a final DataFrame with the exact columns the model expects
    final_df = pd.DataFrame(0, index=[0], columns=MODEL_EXPECTED_FEATURES)
    
    # 4. Fill the final DataFrame with our processed values
    for col in final_df.columns:
        if col in df.columns:
            final_df[col] = df[col]
            
    # Ensure categorical dtypes are set correctly
    for col in final_df.select_dtypes(include=['object', 'category']).columns:
        final_df[col] = final_df[col].astype('category')
            
    return final_df


# --- DEFINE THE WEBSITE ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if xgb_model is None:
        return "XGBoost model not loaded. Please check the server logs."

    form_data = request.form.to_dict()
    print("Received form data:", form_data)
    
    processed_data = preprocess_input_for_xgb(form_data)
    
    # Make a prediction
    predicted_class_index = xgb_model.predict(processed_data)[0]
    predicted_risk = label_encoder.inverse_transform([predicted_class_index])[0]
    prediction_proba = xgb_model.predict_proba(processed_data)
    confidence = np.max(prediction_proba) * 100
    
    print(f"Prediction: {predicted_risk}, Confidence: {confidence:.2f}%")
    
    return render_template('result.html', prediction=predicted_risk, confidence=f"{confidence:.2f}%")

# --- RUN THE APP ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 9000))
    app.run(host='0.0.0.0', port=port, debug=True)