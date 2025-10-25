#!/usr/bin/env python3
"""
Blood Drug Concentration Prediction Web Application
Binary classification prediction system based on logistic regression model
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, session, redirect, url_for

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configure paths
app = Flask(__name__)
app.secret_key = 'blood_concentration_predictor_2024'

# Handle Render environment path issues - fix path detection logic
if os.path.exists('/opt/render/project/src'):
    # Render environment, working directory is under src, but model files are in parent directory
    MODEL_DIR = '/opt/render/project/src/web_models'
else:
    # Local environment or other environments
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web_models')

# Global variables to store models
model = None
scaler = None
metadata = None

# Language configuration
LANGUAGES = {
    'zh': {
        'name': 'Chinese',
        'title': 'Blood Drug Concentration Prediction System',
        'subtitle': 'Intelligent blood drug concentration abnormal risk prediction based on machine learning',
        'model_accuracy': '模型准确率: 86.7%',
        'patient_info': 'Patient Information Input',
        'usage_instructions': 'Usage Instructions',
        'predict_button': 'Start Prediction',
        'required_params': 'Missing required parameters',
        'high_risk': 'High Risk',
        'low_risk': 'Low Risk',
        'prediction_result': 'Prediction Result',
        'risk_level': 'Risk Level',
        'confidence': 'Confidence',
        'recommendations': 'Recommendations',
        'high_risk_rec': 'Recommend close monitoring of blood drug concentration, adjust dosage if necessary',
        'low_risk_rec': 'Current medication is relatively safe, recommend regular follow-up',
        'daily_dose': 'Daily Dose (g)',
        'daily_dose_info': 'Total daily medication dose for patient, typical range: 0.1-5.0g',
        'clcr': 'Creatinine Clearance (mL/min)',
        'clcr_info': 'Kidney function indicator, normal range: 80-120 mL/min',
        'ggt': 'GGT (U/L)',
        'ggt_info': 'Liver function indicator, normal range: Male ≤50 U/L, Female ≤32 U/L',
        'sodium': 'Sodium (mmol/L)',
        'sodium_info': 'Electrolyte balance indicator, normal range: 136-145 mmol/L',
        'hdl': 'HDL-C (mmol/L)',
        'hdl_info': '"Good cholesterol", normal range: Male >1.0, Female >1.3 mmol/L',
        'albumin': 'Albumin (g/L)',
        'albumin_info': 'Nutritional status indicator, normal range: 40-55 g/L',
        'usage_title': 'Usage Instructions',
        'usage_step1': 'Fill in patient\'s basic biochemical indicators',
        'usage_step2': 'Ensure all values are within normal clinical ranges',
        'usage_step3': 'Click predict button to get risk assessment results',
        'usage_step4': 'Adjust treatment plan based on prediction results',
        'start_prediction': 'Start Prediction',
        'predicting': 'Predicting...',
        'prediction_target': 'Prediction Target',
        'prediction_target_desc': 'Assess whether patient has abnormal blood drug concentration risk',
        'important_reminder': 'Important Reminder',
        'reminder_1': 'This system is for clinical reference only and cannot replace doctor\'s judgment',
        'reminder_2': 'Please ensure accuracy of input data',
        'reminder_3': 'Abnormal results need to be analyzed in combination with clinical situation',
        'model_performance': 'Model Performance',
        'accuracy': 'Accuracy: 86.7%',
        'auc_value': 'AUC Value: 0.888',
        'features': 'Based on 6 key clinical features',
        'clinical_guidance': 'Clinical Guidance',
        'kidney_function': 'Kidney Function Assessment',
        'liver_function': 'Liver Function Assessment',
        'electrolyte_balance': 'Electrolyte Balance',
        'lipid_metabolism': 'Lipid Metabolism',
        'abnormal_risk': 'Abnormal Blood Drug Concentration Risk',
        'normal_concentration': 'Normal Blood Drug Concentration',
        'monitor_closely': 'Recommend close monitoring of blood drug concentration',
        'normal_range': 'Current indicators show normal range',
        'abnormal_probability': 'Abnormal Probability',
        'input_data': 'Input Data',
        'new_prediction': 'New Prediction',
        'print_result': 'Print Result',
        'clinical_advice': 'Clinical Advice',
        'high_risk_management': 'High Risk Patient Management',
        'monitor_concentration': 'Recommend blood drug concentration monitoring',
        'adjust_dosage': 'Consider adjusting dosage or frequency',
        'watch_adverse': 'Monitor closely for adverse reactions',
        'assess_function': 'Assess kidney and liver function status',
        'monitoring_frequency': 'Monitoring Frequency Recommendations',
        'initial_treatment': 'Initial treatment: Weekly monitoring',
        'after_adjustment': 'After dose adjustment: Recheck in 3-5 days',
        'stable_period': 'Stable period: Monthly monitoring',
        'normal_management': 'Normal Range Management',
        'continue_treatment': 'Continue current treatment plan',
        'regular_followup': 'Regular follow-up monitoring',
        'observe_symptoms': 'Pay attention to clinical symptoms',
        'maintain_compliance': 'Maintain good compliance',
        'followup_advice': 'Follow-up Advice',
        'stable_monitoring': 'Stable period: Monitor every 3 months',
        'symptom_changes': 'Seek medical attention promptly if symptoms change',
        'regular_assessment': 'Regular assessment of kidney and liver function',
        'important_notes': 'Important Notes',
        'reference_only': 'This prediction result is for reference only',
        'clinical_judgment': 'Need to combine with clinical symptoms for comprehensive judgment',
        'consult_specialist': 'Consult specialist for important decisions',
        'patient_communication': 'Maintain good communication with patients',
        'indicator_interpretation': 'Indicator Interpretation',
        'kidney_status': 'Kidney Function Status',
        'liver_status': 'Liver Function Status',
        'electrolyte_status': 'Electrolyte Balance',
        'nutrition_status': 'Nutritional Status',
        'normal': 'Normal',
        'mild_decline': 'Mild Decline',
        'moderate_decline': 'Moderate Decline',
        'severe_decline': 'Severe Decline',
        'mild_elevation': 'Mild Elevation',
        'significant_elevation': 'Significant Elevation',
        'hyponatremia': 'Hyponatremia',
        'hypernatremia': 'Hypernatremia',
        'mild_deficiency': 'Mild Deficiency',
        'significant_deficiency': 'Significant Deficiency'
    },
    'en': {
        'name': 'English',
        'title': 'Blood Drug Concentration Prediction System',
        'subtitle': 'Intelligent blood drug concentration abnormal risk prediction based on machine learning',
        'model_accuracy': 'Model Accuracy: 86.7%',
        'patient_info': 'Patient Information Input',
        'usage_instructions': 'Usage Instructions',
        'predict_button': 'Start Prediction',
        'required_params': 'Missing required parameters',
        'high_risk': 'High Risk',
        'low_risk': 'Low Risk',
        'prediction_result': 'Prediction Result',
        'risk_level': 'Risk Level',
        'confidence': 'Confidence',
        'recommendations': 'Recommendations',
        'high_risk_rec': 'Recommend close monitoring of blood drug concentration, adjust dosage if necessary',
        'low_risk_rec': 'Current medication is relatively safe, recommend regular follow-up',
        'daily_dose': 'Daily Dose (g)',
        'daily_dose_info': 'Total daily medication dose for patient, typical range: 0.1-5.0g',
        'clcr': 'Creatinine Clearance (mL/min)',
        'clcr_info': 'Kidney function indicator, normal range: 80-120 mL/min',
        'ggt': 'GGT (U/L)',
        'ggt_info': 'Liver function indicator, normal range: Male ≤50 U/L, Female ≤32 U/L',
        'sodium': 'Sodium (mmol/L)',
        'sodium_info': 'Electrolyte balance indicator, normal range: 136-145 mmol/L',
        'hdl': 'HDL-C (mmol/L)',
        'hdl_info': '"Good cholesterol", normal range: Male >1.0, Female >1.3 mmol/L',
        'albumin': 'Albumin (g/L)',
        'albumin_info': 'Nutritional status indicator, normal range: 40-55 g/L',
        'usage_title': 'Usage Instructions',
        'usage_step1': 'Fill in patient\'s basic biochemical indicators',
        'usage_step2': 'Ensure all values are within normal clinical ranges',
        'usage_step3': 'Click predict button to get risk assessment results',
        'usage_step4': 'Adjust treatment plan based on prediction results',
        'start_prediction': 'Start Prediction',
        'predicting': 'Predicting...',
        'prediction_target': 'Prediction Target',
        'prediction_target_desc': 'Assess whether patient has abnormal blood drug concentration risk',
        'important_reminder': 'Important Reminder',
        'reminder_1': 'This system is for clinical reference only and cannot replace doctor\'s judgment',
        'reminder_2': 'Please ensure accuracy of input data',
        'reminder_3': 'Abnormal results need to be analyzed in combination with clinical situation',
        'model_performance': 'Model Performance',
        'accuracy': 'Accuracy: 86.7%',
        'auc_value': 'AUC Value: 0.888',
        'features': 'Based on 6 key clinical features',
        'clinical_guidance': 'Clinical Guidance',
        'kidney_function': 'Kidney Function Assessment',
        'liver_function': 'Liver Function Assessment',
        'electrolyte_balance': 'Electrolyte Balance',
        'lipid_metabolism': 'Lipid Metabolism',
        'abnormal_risk': 'Abnormal Blood Drug Concentration Risk',
        'normal_concentration': 'Normal Blood Drug Concentration',
        'monitor_closely': 'Recommend close monitoring of blood drug concentration',
        'normal_range': 'Current indicators show normal range',
        'abnormal_probability': 'Abnormal Probability',
        'input_data': 'Input Data',
        'new_prediction': 'New Prediction',
        'print_result': 'Print Result',
        'clinical_advice': 'Clinical Advice',
        'high_risk_management': 'High Risk Patient Management',
        'monitor_concentration': 'Recommend blood drug concentration monitoring',
        'adjust_dosage': 'Consider adjusting dosage or frequency',
        'watch_adverse': 'Monitor closely for adverse reactions',
        'assess_function': 'Assess kidney and liver function status',
        'monitoring_frequency': 'Monitoring Frequency Recommendations',
        'initial_treatment': 'Initial treatment: Weekly monitoring',
        'after_adjustment': 'After dose adjustment: Recheck in 3-5 days',
        'stable_period': 'Stable period: Monthly monitoring',
        'normal_management': 'Normal Range Management',
        'continue_treatment': 'Continue current treatment plan',
        'regular_followup': 'Regular follow-up monitoring',
        'observe_symptoms': 'Pay attention to clinical symptoms',
        'maintain_compliance': 'Maintain good compliance',
        'followup_advice': 'Follow-up Advice',
        'stable_monitoring': 'Stable period: Monitor every 3 months',
        'symptom_changes': 'Seek medical attention promptly if symptoms change',
        'regular_assessment': 'Regular assessment of kidney and liver function',
        'important_notes': 'Important Notes',
        'reference_only': 'This prediction result is for reference only',
        'clinical_judgment': 'Need to combine with clinical symptoms for comprehensive judgment',
        'consult_specialist': 'Consult specialist for important decisions',
        'patient_communication': 'Maintain good communication with patients',
        'indicator_interpretation': 'Indicator Interpretation',
        'kidney_status': 'Kidney Function Status',
        'liver_status': 'Liver Function Status',
        'electrolyte_status': 'Electrolyte Balance',
        'nutrition_status': 'Nutritional Status',
        'normal': 'Normal',
        'mild_decline': 'Mild Decline',
        'moderate_decline': 'Moderate Decline',
        'severe_decline': 'Severe Decline',
        'mild_elevation': 'Mild Elevation',
        'significant_elevation': 'Significant Elevation',
        'hyponatremia': 'Hyponatremia',
        'hypernatremia': 'Hypernatremia',
        'mild_deficiency': 'Mild Deficiency',
        'significant_deficiency': 'Significant Deficiency'
    }
}

def get_language():
    """Get current language setting, prioritize URL parameters"""
    # Priority from URL parameters
    lang = request.args.get('lang')
    if lang and lang in LANGUAGES:
        session['language'] = lang
        return lang
    
    # Default to English if no language is set
    return session.get('language', 'en')

def get_texts():
    """Get text for current language"""
    current_lang = get_language()
    return LANGUAGES[current_lang]

def load_model():
    """Load model components"""
    global model, scaler, metadata
    try:
        # Print debug information
        logger.info(f"Current working directory: {os.getcwd()}")
        project_root = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Project root directory: {project_root}")
        logger.info(f"Model directory exists: {os.path.exists(MODEL_DIR)}")
        
        # List project root directory contents
        try:
            if os.path.exists(project_root):
                logger.info(f"Project root directory {project_root} contents:")
                for item in os.listdir(project_root):
                    item_path = os.path.join(project_root, item)
                    if os.path.isdir(item_path):
                        logger.info(f"  Directory: {item}")
                    else:
                        logger.info(f"  File: {item}")
        except Exception as e:
            logger.error(f"Unable to list directory contents: {e}")
        
        if os.path.exists(MODEL_DIR):
            logger.info(f"Model directory contents: {os.listdir(MODEL_DIR)}")
        
        # Try multiple possible paths
        possible_paths = [
            os.path.join(MODEL_DIR, 'logistic_regression_model.pkl'),
            os.path.join(project_root, 'web_models', 'logistic_regression_model.pkl'),
            os.path.join(os.getcwd(), 'web_models', 'logistic_regression_model.pkl'),
            'web_models/logistic_regression_model.pkl',
            './web_models/logistic_regression_model.pkl'
        ]
        
        model_path = None
        for path in possible_paths:
            logger.info(f"Trying path: {path}")
            if os.path.exists(path):
                model_path = path
                logger.info(f"Found model file path: {model_path}")
                break
            else:
                logger.info(f"Path does not exist: {path}")
        
        if not model_path:
            raise FileNotFoundError("Unable to find model file")
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        scaler_path = model_path.replace('logistic_regression_model.pkl', 'feature_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load metadata
        metadata_path = model_path.replace('logistic_regression_model.pkl', 'model_metadata.json')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        logger.info("Model components loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return False

def validate_input(data):
    """Validate input data"""
    errors = []
    
    if not model or not scaler:
        return ["Model not loaded correctly"]
    
    # Required features
    required_features = ['Daily dose（g）', 'CLCR', 'GGT(U/L)', 'Na(mmol/L)', 'HDL-C(mmol/L)', 'ALB(g/L)']
    
    for feature in required_features:
        if feature not in data or data[feature] is None or data[feature] == '':
            errors.append(f"Missing required parameter: {feature}")
    
    if errors:
        return errors
    
    # Check clinical ranges
    ranges = {
        'Daily dose（g）': (0.1, 10.0),
        'CLCR': (10, 200),
        'GGT(U/L)': (5, 500),
        'Na(mmol/L)': (120, 160),
        'HDL-C(mmol/L)': (0.5, 5.0),
        'ALB(g/L)': (20, 70)
    }
    
    for feature, (min_val, max_val) in ranges.items():
        try:
            value = float(data[feature])
            if value < min_val * 0.1 or value > max_val * 3:  # Allow certain range of abnormal values
                errors.append(f"{feature} value abnormal: {value} (recommended range: {min_val}-{max_val})")
        except (ValueError, TypeError):
            errors.append(f"{feature} must be numeric")
    
    return errors

@app.route('/')
def index():
    """Homepage"""
    if not model:
        return render_template('error.html', error="Model not loaded correctly, please contact administrator")
    
    texts = get_texts()
    current_lang = get_language()
    
    return render_template('index.html', 
                         texts=texts,
                         current_lang=current_lang,
                         languages=LANGUAGES)

@app.route('/set_language/<lang>')
def set_language(lang):
    """Set language"""
    if lang in LANGUAGES:
        session['language'] = lang
    
    # Get current URL parameters
    referrer = request.referrer or url_for('index')
    
    # If referrer contains lang parameter, replace it; otherwise add lang parameter
    if '?' in referrer:
        # Parse existing parameters
        base_url, params = referrer.split('?', 1)
        param_pairs = params.split('&')
        new_params = []
        lang_found = False
        
        for param in param_pairs:
            if param.startswith('lang='):
                new_params.append(f'lang={lang}')
                lang_found = True
            else:
                new_params.append(param)
        
        if not lang_found:
            new_params.append(f'lang={lang}')
        
        return redirect(f"{base_url}?{'&'.join(new_params)}")
    else:
        return redirect(f"{referrer}?lang={lang}")

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction interface"""
    try:
        # Get input data
        data = {
            'Daily dose（g）': request.form.get('daily_dose'),
            'CLCR': request.form.get('clcr'),
            'GGT(U/L)': request.form.get('ggt'),
            'Na(mmol/L)': request.form.get('sodium'),
            'HDL-C(mmol/L)': request.form.get('hdl'),
            'ALB(g/L)': request.form.get('albumin')
        }
        
        # Validate input
        errors = validate_input(data)
        if errors:
            texts = get_texts()
            return render_template('index.html', 
                                 texts=texts,
                                 current_lang=get_language(),
                                 errors=errors,
                                 form_data=data)
        
        # Prepare prediction data
        input_data = pd.DataFrame([{
            feature: float(data[feature]) for feature in data.keys()
        }])
        
        # Standardize
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        # Interpret results
        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'prediction_text': 'High' if prediction == 1 else 'Normal',
            'input_data': data
        }
        
        # Add clinical recommendations
        if prediction == 1:  # High risk
            if probability > 0.8:
                result['recommendation'] = "Very high risk of elevated blood drug concentration, recommend immediate dosage adjustment and close monitoring"
            elif probability > 0.6:
                result['recommendation'] = "High risk of elevated blood drug concentration, recommend considering dosage adjustment"
            else:
                result['recommendation'] = "Moderate risk of elevated blood drug concentration, recommend enhanced monitoring"
        else:  # Low risk
            if probability < 0.2:
                result['recommendation'] = "Normal blood drug concentration, current medication regimen is appropriate"
            elif probability < 0.4:
                result['recommendation'] = "Blood drug concentration is basically normal, can continue current medication regimen"
            else:
                result['recommendation'] = "Blood drug concentration is close to threshold, recommend regular monitoring"
        
        logger.info(f"Prediction completed: {result}")
        
        texts = get_texts()
        current_lang = get_language()
        
        return render_template('result.html',
                             texts=texts,
                             current_lang=current_lang,
                             result=result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        error_msg = "Error occurred during prediction, please check input data"
        texts = get_texts()
        return render_template('index.html', 
                             texts=texts,
                             current_lang=get_language(),
                             errors=[error_msg],
                             form_data=data if 'data' in locals() else {})

@app.route('/api/model_info')
def model_info():
    """Model information API"""
    if not metadata:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': metadata.get('model_type', 'Logistic Regression'),
        'features': metadata.get('features', []),
        'performance': metadata.get('performance', {}),
        'training_date': metadata.get('training_date', 'Unknown'),
        'version': metadata.get('version', '1.0')
    })

@app.route('/health')
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'metadata_loaded': metadata is not None
    })

@app.errorhandler(404)
def not_found_error(error):
    current_lang = get_language()
    return render_template('error.html', 
                         error="Page Not Found" if current_lang == 'en' else "Page Not Found",
                         texts=get_texts()), 404

@app.errorhandler(500)
def internal_error(error):
    current_lang = get_language()
    return render_template('error.html', 
                         error="Internal Server Error" if current_lang == 'en' else "Internal Server Error",
                         texts=get_texts()), 500

if __name__ == '__main__':
    # Load model at startup
    if load_model():
        print("🚀 Blood Drug Concentration Prediction Web Application Started Successfully!")
        print(f"📊 Model Performance: Accuracy {metadata['performance']['accuracy']:.3f}, AUC {metadata['performance']['auc']:.3f}")
        print(f"🔧 Features Used: {', '.join(metadata['features'])}")
        print("🌐 Access URL: http://localhost:5000")
        
        # Start Flask application
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
    else:
        print("❌ Model loading failed, unable to start application")
        print("Please ensure model files exist in the web_models directory")
        sys.exit(1)
