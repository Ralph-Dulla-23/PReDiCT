from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# --- INITIALIZE THE FLASK APP ---
app = Flask(__name__)

# --- LOAD THE TRAINED MODEL AND ARTIFACTS ---
print("Loading the trained RNN model...")
try:
    model = load_model('rnn_multiclass_risk_model.keras')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- CRITICAL: These must match your training script EXACTLY ---
# This is the updated list you provided.
EXPECTED_COLUMNS = [
    'Visit_Num', 'age_group', 'num_partners', 'amount', 'gender_Female', 'gender_Male', 
    'gender_Non-binary', 'sexual_orientation_Bisexual', 'sexual_orientation_Heterosexual', 
    'sexual_orientation_MSM', 'sexual_orientation_WSW', 'education_No Formal Education', 
    'education_Primary', 'education_Secondary', 'education_Tertiary', 'education_Vocational', 
    'transactional_sex_False', 'transactional_sex_True', 'substance_use_False', 
    'substance_use_True', 'last_test_result_Negative', 'diagnosed_sti_Chlamydia', 
    'diagnosed_sti_Gonorrhea', 'diagnosed_sti_Syphilis', 'reason_for_visit_Follow-up testing', 
    'reason_for_visit_Partner tested positive', 'reason_for_visit_Routine screening', 
    'reason_for_visit_Symptoms present', 'prep_procedure_Daily', 'prep_procedure_Event-driven', 
    'hiv_screen_Negative', 'hiv_screen_Positive'
]
MAX_SEQ_LENGTH = 3 # This was the max_seq_length from your training script
# The order of the labels the model was trained on (from LabelEncoder)
RISK_CLASSES = ['High Risk', 'Low Risk', 'Moderate Risk']


# --- PREPROCESSING FUNCTION ---
def preprocess_input(data):
    """
    Takes raw input data from the form and prepares it for the model.
    """
    # Create a DataFrame from the single visit data
    df = pd.DataFrame([data])

    # Convert numeric fields from string to number
    for col in ['Visit_Num', 'age_group', 'num_partners', 'amount']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Convert boolean-like strings to actual booleans
    for col in ['transactional_sex', 'substance_use']:
        df[col] = df[col].apply(lambda x: True if x == 'True' else False)

    # One-Hot Encode categorical features
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    # Reindex to ensure the dataframe has the exact same columns as the training data
    df_reindexed = df_encoded.reindex(columns=EXPECTED_COLUMNS, fill_value=0)
    
    # We are only predicting for one visit, but the model expects a sequence.
    # We will treat this single visit as a sequence of length 1.
    sequence = [df_reindexed.values]
    
    # Pad the sequence to the required length
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQ_LENGTH, padding='post', dtype='float32')
    
    return padded_sequence

# --- DEFINE THE WEBSITE ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not loaded. Please check the server logs."

    # Get the data from the form
    form_data = request.form.to_dict()
    print("Received form data:", form_data)
    
    # Process the input data
    processed_data = preprocess_input(form_data)
    
    # Make a prediction
    prediction_proba = model.predict(processed_data)
    predicted_class_index = np.argmax(prediction_proba, axis=1)[0]
    predicted_risk = RISK_CLASSES[predicted_class_index]
    confidence = prediction_proba[0][predicted_class_index] * 100
    
    print(f"Prediction: {predicted_risk}, Confidence: {confidence:.2f}%")
    
    # Render the result page with the prediction
    return render_template('result.html', prediction=predicted_risk, confidence=f"{confidence:.2f}%")

# --- RUN THE APP ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)