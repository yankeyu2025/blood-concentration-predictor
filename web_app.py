#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from datetime import datetime

class InputNormalizer:
    def __init__(self, original_ranges, normalized_ranges):
        self.original_ranges = original_ranges
        self.normalized_ranges = normalized_ranges
    
    def normalize_input(self, raw_input):
        """
        将原始医学指标值线性映射到训练数据使用的归一化范围
        """
        normalized = {}
        
        for feature, raw_value in raw_input.items():
            if feature in self.original_ranges and feature in self.normalized_ranges:
                # 获取原始范围和归一化范围
                orig_min, orig_max = self.original_ranges[feature]
                norm_min, norm_max = self.normalized_ranges[feature]
                
                # 线性映射：从原始范围映射到归一化范围
                normalized_value = norm_min + (raw_value - orig_min) * (norm_max - norm_min) / (orig_max - orig_min)
                
                # 确保在归一化范围内
                normalized_value = max(norm_min, min(norm_max, normalized_value))
                
                normalized[feature] = normalized_value
            else:
                print(f"Warning: Feature {feature} not found in ranges")
                normalized[feature] = raw_value
        
        return normalized

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
input_normalizer = None
metadata = None

# Language configuration
LANGUAGES = {
    'zh': {
        'name': '中文',
        'title': '血药浓度预测系统',
        'subtitle': '基于机器学习的智能血药浓度异常风险预测',
        'model_accuracy': '模型准确率',
        'patient_info': '患者信息输入',
        'usage_instructions': '使用说明',
        'predict_button': '开始预测',
        'required_params': '请填写所有必填字段！',
        'high_risk': '高风险',
        'low_risk': '低风险',
        'prediction_result': '预测结果',
        'risk_level': '风险等级',
        'confidence': '置信度',
        'recommendations': '建议',
        'high_risk_rec': '建议密切监测血药浓度，必要时调整剂量',
        'low_risk_rec': '当前用药相对安全，建议定期随访',
        'daily_dose': '日剂量 (g)',
        'daily_dose_info': '患者每日用药总剂量，典型范围：0.1-5.0g',
        'clcr': '肌酐清除率 (mL/min)',
        'clcr_info': '肾功能指标，正常范围：80-120 mL/min',
        'ggt': 'GGT (U/L)',
        'ggt_info': '肝功能指标，正常范围：男性≤50 U/L，女性≤32 U/L',
        'sodium': '钠离子 (mmol/L)',
        'sodium_info': '电解质平衡指标，正常范围：136-145 mmol/L',
        'alp': 'ALP (U/L)',
        'alp_info': '碱性磷酸酶，肝功能指标，正常范围：40-150 U/L',
        'tba': 'TBA (umol/L)',
        'tba_info': '总胆汁酸，肝功能指标，正常范围：0-10 umol/L',
        'usage_title': '使用说明',
        'usage_step1': '填写患者基本生化指标',
        'usage_step2': '确保所有数值在正常临床范围内',
        'usage_step3': '点击预测按钮获得风险评估结果',
        'usage_step4': '根据预测结果调整治疗方案',
        'start_prediction': '开始预测',
        'predicting': '预测中...',
        'prediction_target': '预测目标',
        'prediction_target_desc': '评估患者是否存在血药浓度异常风险',
        'important_reminder': '重要提醒',
        'reminder_1': '本系统仅供临床参考，不能替代医生判断',
        'reminder_2': '请确保输入数据的准确性',
        'reminder_3': '异常结果需结合临床情况综合分析',
        'model_performance': '模型性能',
        'accuracy': '准确率',
        'auc_value': 'AUC值',
        'features': '基于6个关键临床特征',
        'clinical_guidance': '临床指导',
        'kidney_function': '肾功能评估',
        'liver_function': '肝功能评估',
        'electrolyte_balance': '电解质平衡',
        'lipid_metabolism': '脂质代谢',
        'abnormal_risk': '血药浓度不正常 (<0.5)',
        'normal_concentration': '血药浓度正常 (>=0.5)',
        'monitor_closely': '风险提示：血药浓度可能偏低，建议密切监测，并结合临床情况评估是否需要调整治疗方案。',
        'normal_range': '当前预测结果在正常范围内，请继续按医嘱用药并定期随访。',
        'abnormal_probability': '异常概率',
        'input_data': '输入数据',
        'new_prediction': '新预测',
        'print_result': '打印结果',
        'clinical_advice': '临床建议',
        'high_risk_management': '高风险患者管理',
        'monitor_concentration': '建议进行血药浓度监测',
        'adjust_dosage': '考虑调整剂量或给药频次',
        'watch_adverse': '密切观察不良反应',
        'assess_function': '评估肾脏和肝脏功能状态',
        'monitoring_frequency': '监测频率建议',
        'initial_treatment': '初始治疗：每周监测',
        'after_adjustment': '剂量调整后：3-5天复查',
        'stable_period': '稳定期：每月监测',
        'normal_management': '正常范围管理',
        'continue_treatment': '继续当前治疗方案',
        'regular_followup': '定期随访监测',
        'observe_symptoms': '注意观察临床症状',
        'maintain_compliance': '保持良好依从性',
        'followup_advice': '随访建议',
        'stable_monitoring': '稳定期：每3个月监测',
        'symptom_changes': '症状变化时及时就医',
        'regular_assessment': '定期评估肾脏和肝脏功能',
        'important_notes': '重要说明',
        'reference_only': '此预测结果仅供参考',
        'clinical_judgment': '需结合临床症状综合判断',
        'consult_specialist': '重要决策请咨询专科医生',
        'patient_communication': '与患者保持良好沟通',
        'indicator_interpretation': '指标解读',
        'kidney_status': '肾功能状态',
        'liver_status': '肝功能状态',
        'electrolyte_status': '电解质平衡',
        'nutrition_status': '营养状态',
        'normal': '正常',
        'mild_decline': '轻度下降',
        'moderate_decline': '中度下降',
        'severe_decline': '重度下降',
        'mild_decline': '轻度下降',
    'significant_decline': '显著下降',
    'liver_function': '肝功能状态',
        'hyponatremia': '低钠血症',
        'hypernatremia': '高钠血症',
        'mild_deficiency': '轻度不足',
        'significant_deficiency': '显著不足',
        # result.html 模板中使用的变量
        'monitor_drug_concentration': '建议进行血药浓度监测',
        'observe_adverse_reactions': '密切观察不良反应',
        'assess_organ_function': '评估肾脏和肝脏功能状态',
        'after_dose_adjustment': '剂量调整后：3-5天复查',
        'continue_current_treatment': '继续当前治疗方案',
        'regular_follow_up': '定期随访监测',
        'observe_clinical_symptoms': '注意观察临床症状',
        'follow_up_advice': '随访建议',
        'regular_organ_assessment': '定期评估肾脏和肝脏功能',
        'nutritional_status': '营养状态',
        # 新增：将模板中的中文硬编码改为可翻译文本
        'kidney_guide_content': '<strong>肌酐清除率 (CLCR)</strong><br>• 正常：80-120 mL/min<br>• 轻度损害：60-80 mL/min<br>• 中度损害：30-60 mL/min<br>• 重度损害：<30 mL/min',
        'liver_ggt_guide_content': '<strong>GGT参考范围</strong><br>• 男性：≤50 U/L<br>• 女性：≤32 U/L<br>• 升高提示肝胆疾病或药物性肝损伤',
        'sodium_guide_content': '<strong>钠离子正常范围</strong><br>• 136-145 mmol/L<br>• 低钠血症：<136 mmol/L<br>• 高钠血症：>145 mmol/L'
    },
    'en': {
        'name': 'English',
        'title': 'Blood Drug Concentration Prediction System',
        'subtitle': 'Intelligent blood drug concentration abnormal risk prediction based on machine learning',
        'model_accuracy': 'Model Accuracy',
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
        'alp': 'ALP (U/L)',
        'alp_info': 'Alkaline phosphatase, liver function indicator, normal range: 40-150 U/L',
        'tba': 'TBA (umol/L)',
        'tba_info': 'Total bile acid, liver function indicator, normal range: 0-10 umol/L',
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
        'accuracy': 'Accuracy',
        'auc_value': 'AUC Value',
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
        'mild_decline': 'Mild Decline',
    'significant_decline': 'Significant Decline',
    'liver_function': 'Liver Function',
        'hyponatremia': 'Hyponatremia',
        'hypernatremia': 'Hypernatremia',
        'mild_deficiency': 'Mild Deficiency',
        'significant_deficiency': 'Significant Deficiency',
        # result.html 模板中使用的变量
        'monitor_drug_concentration': 'Recommend blood drug concentration monitoring',
        'observe_adverse_reactions': 'Monitor closely for adverse reactions',
        'assess_organ_function': 'Assess kidney and liver function status',
        'after_dose_adjustment': 'After dose adjustment: Recheck in 3-5 days',
        'continue_current_treatment': 'Continue current treatment plan',
        'regular_follow_up': 'Regular follow-up monitoring',
        'observe_clinical_symptoms': 'Pay attention to clinical symptoms',
        'follow_up_advice': 'Follow-up Advice',
        'regular_organ_assessment': 'Regular assessment of kidney and liver function',
        'nutritional_status': 'Nutritional Status',
        # 新增：可翻译文本用于替换模板中文
        'kidney_guide_content': '<strong>Creatinine Clearance (CLCR)</strong><br>• Normal: 80-120 mL/min<br>• Mild impairment: 60-80 mL/min<br>• Moderate impairment: 30-60 mL/min<br>• Severe impairment: <30 mL/min',
        'liver_ggt_guide_content': '<strong>GGT reference range</strong><br>• Male: ≤50 U/L<br>• Female: ≤32 U/L<br>• Elevation suggests hepatobiliary disease or drug-induced liver injury',
        'sodium_guide_content': '<strong>Sodium normal range</strong><br>• 136-145 mmol/L<br>• Hyponatremia: <136 mmol/L<br>• Hypernatremia: >145 mmol/L'
    }
}

def get_language():
    """Get current language setting"""
    # Check URL parameter first
    lang = request.args.get('lang')
    if lang and lang in LANGUAGES:
        session['language'] = lang
        return lang
    
    # Check session
    if 'language' in session and session['language'] in LANGUAGES:
        return session['language']
    
    # Default to Chinese
    return 'zh'

def get_texts():
    """Get text for current language"""
    current_lang = get_language()
    return LANGUAGES[current_lang]

def load_model():
    """Load model components"""
    global model, scaler, input_normalizer, metadata
    try:
        logger.info(f"Current working directory: {os.getcwd()}")
        project_root = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Project root directory: {project_root}")
        logger.info(f"Model directory exists: {os.path.exists(MODEL_DIR)}")

        if os.path.exists(MODEL_DIR):
            logger.info(f"Model directory contents: {os.listdir(MODEL_DIR)}")

        model_path = os.path.join(MODEL_DIR, 'svm_model.pkl')
        scaler_path = os.path.join(MODEL_DIR, 'svm_scaler.pkl')
        input_normalizer_path = os.path.join(MODEL_DIR, 'input_normalizer.pkl')
        metadata_path = os.path.join(MODEL_DIR, 'model_metadata.json')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SVM model not found at {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"SVM scaler not found at {scaler_path}")
        if not os.path.exists(input_normalizer_path):
            raise FileNotFoundError(f"Input normalizer not found at {input_normalizer_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        with open(input_normalizer_path, 'rb') as f:
            input_normalizer = pickle.load(f)
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        logger.info("SVM Model components loaded successfully")
        logger.info(f"Input normalizer loaded with normalize_input function: {hasattr(input_normalizer, 'normalize_input')}")
        return True
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return False

@app.route('/')
def index():
    """Main page route"""
    return render_template('index.html', 
                         texts=get_texts(), 
                         metadata=metadata,
                         languages=LANGUAGES,
                         current_lang=get_language())

@app.route('/lang/<language>')
def set_language(language):
    """Set language"""
    if language in LANGUAGES:
        session['language'] = language
    
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
                new_params.append(f'lang={language}')
                lang_found = True
            else:
                new_params.append(param)
        
        if not lang_found:
            new_params.append(f'lang={language}')
        
        return redirect(f"{base_url}?{'&'.join(new_params)}")
    else:
        return redirect(f"{referrer}?lang={language}")

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction interface"""
    try:
        logger.info("Starting prediction process...")
        
        # Get input data
        data = {
            'Daily dose（g）': request.form.get('Daily dose（g）'),
            'CLCR': request.form.get('CLCR'),
            'GGT(U/L)': request.form.get('GGT(U/L)'),
            'Na(mmol/L)': request.form.get('Na(mmol/L)'),
            'ALP(U/L)': request.form.get('ALP(U/L)'),
            'TBA(umol/L)': request.form.get('TBA(umol/L)')
        }
        
        logger.info(f"Received input data: {data}")
        
        # Validate input
        errors = []
        for key, value in data.items():
            if not value or value.strip() == '':
                errors.append(f"错误：字段 '{key}' 不能为空。")
                logger.warning(f"字段值缺失: {key}")
                continue
            try:
                float(value)
            except (ValueError, TypeError):
                errors.append(f"错误：字段 '{key}' 的值 '{value}' 不是有效的数字。")
                logger.warning(f"字段值无效 {key}: {value}")
        
        if errors:
            logger.warning(f"Input validation failed: {errors}")
            texts = get_texts()
            return render_template('index.html', 
                                 texts=texts,
                                 current_lang=get_language(),
                                 languages=LANGUAGES,
                                 performance=metadata.get('performance', {}) if metadata else {},
                                 errors=errors,
                                 form_data=data)
        
        logger.info("Input validation passed, preparing prediction data...")
        
        # Use input normalizer to convert raw medical values to normalized values
        # that match the training data range
        raw_input = {
            'CLCR': float(data['CLCR']),
            'Daily dose(g)': float(data['Daily dose（g）']),
            'ALP(U/L)': float(data['ALP(U/L)']),
            'GGT(U/L)': float(data['GGT(U/L)']),
            'Na(mmol/L)': float(data['Na(mmol/L)']),
            'TBA(umol/L)': float(data['TBA(umol/L)'])
        }
        
        logger.info(f"Raw input values: {raw_input}")
        
        # Normalize input using our custom normalizer
        normalized_input = input_normalizer.normalize_input(raw_input)
        logger.info(f"Normalized input values: {normalized_input}")
        
        # Create DataFrame for the scaler (which expects normalized values)
        # The scaler will then standardize these normalized values
        all_features = scaler.feature_names_in_
        full_input_data = pd.DataFrame(0, index=[0], columns=all_features)
        
        # Fill in the normalized features
        for feature_name, normalized_value in normalized_input.items():
            if feature_name in all_features:
                full_input_data[feature_name] = normalized_value
            else:
                logger.warning(f"Feature {feature_name} not found in scaler features")
        
        logger.info(f"Created full DataFrame with shape: {full_input_data.shape}")
        logger.info(f"Non-zero features: {[(col, full_input_data[col].iloc[0]) for col in full_input_data.columns if full_input_data[col].iloc[0] != 0]}")
        
        # Standardize the normalized values
        input_scaled = scaler.transform(full_input_data)
        logger.info(f"Data standardized, shape: {input_scaled.shape}")
        
        # Select only the 6 features that the SVM model expects
        # Based on the actual model training features
        # 选择的特征（与Boruta训练时的顺序一致）
        selected_features = ['CLCR', 'Daily dose(g)', 'ALP(U/L)', 'GGT(U/L)', 'Na(mmol/L)', 'TBA(umol/L)']
        feature_indices = [list(all_features).index(feat) for feat in selected_features if feat in all_features]
        
        if len(feature_indices) != 6:
            logger.error(f"Expected 6 features, but found {len(feature_indices)}: {[all_features[i] for i in feature_indices]}")
            raise ValueError("Feature selection mismatch")
        
        input_scaled_selected = input_scaled[:, feature_indices]
        logger.info(f"Selected features for SVM model, shape: {input_scaled_selected.shape}")
        
        # Predict
        prediction = model.predict(input_scaled_selected)[0]
        probability = model.predict_proba(input_scaled_selected)[0][1]
        
        logger.info(f"Raw prediction: {prediction}, probability: {probability}")
        
        # Interpret results based on probability threshold
        # probability represents the likelihood of class 1 (≥ 0.5)
        # So if probability ≥ 0.5, it's normal; if < 0.5, it's abnormal
        is_normal = probability >= 0.5
        
        result = {
            'prediction': 1 if is_normal else 0,
            'probability': float(probability),
            'prediction_text': 'Normal Blood Drug Concentration (≥ 0.5)' if is_normal else 'Abnormal Blood Drug Concentration (< 0.5)',
            'input_data': data
        }
        
        logger.info(f"Interpreted result: {result}")
        
        # Add clinical recommendations
        if is_normal:  # Normal blood drug concentration (≥ 0.5)
            if probability > 0.8:
                result['recommendation'] = "Very high confidence of normal blood drug concentration (≥ 0.5), current dosage is appropriate"
            elif probability > 0.6:
                result['recommendation'] = "High confidence of normal blood drug concentration (≥ 0.5), maintain current treatment"
            else:
                result['recommendation'] = "Moderate confidence of normal blood drug concentration (≥ 0.5), recommend regular monitoring"
        else:  # Abnormal blood drug concentration (< 0.5)
            abnormal_probability = 1 - probability
            if abnormal_probability > 0.8:
                result['recommendation'] = "Very high risk of abnormal blood drug concentration (< 0.5), recommend immediate dosage adjustment and close monitoring"
            elif abnormal_probability > 0.6:
                result['recommendation'] = "High risk of abnormal blood drug concentration (< 0.5), recommend considering dosage adjustment"
            else:
                result['recommendation'] = "Moderate risk of abnormal blood drug concentration (< 0.5), recommend enhanced monitoring"
        
        logger.info(f"Prediction completed: {result}")
        
        texts = get_texts()
        current_lang = get_language()
        
        logger.info(f"Rendering result.html with texts keys: {list(texts.keys())[:10]}...")
        
        return render_template('result.html',
                             texts=texts,
                             current_lang=current_lang,
                             languages=LANGUAGES,
                             result=result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        error_msg = "Error occurred during prediction, please check input data"
        texts = get_texts()
        current_lang = get_language()
        return render_template('index.html', 
                             texts=texts,
                             current_lang=current_lang,
                             languages=LANGUAGES,
                             performance=metadata.get('performance', {}) if metadata else {},
                             errors=[error_msg],
                             form_data=data if 'data' in locals() else {})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API prediction endpoint that returns JSON"""
    try:
        logger.info("Starting API prediction process...")
        
        # Get JSON data
        if request.is_json:
            data = request.get_json()
        else:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        logger.info(f"Received JSON input data: {data}")
        
        # Validate input
        required_fields = ['Daily dose（g）', 'CLCR', 'GGT(U/L)', 'Na(mmol/L)', 'ALP(U/L)', 'TBA(umol/L)']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
            try:
                float(data[field])
            except (ValueError, TypeError):
                return jsonify({'error': f'Invalid value for field {field}: {data[field]}'}), 400
        
        # Perform prediction using the same logic as the web interface
        logger.info("Input validation passed, preparing prediction data...")
        
        # Step 1: Normalize raw input using input_normalizer
        raw_input = {
            'CLCR': float(data['CLCR']),
            'Daily dose（g）': float(data['Daily dose（g）']),
            'ALP(U/L)': float(data['ALP(U/L)']),
            'GGT(U/L)': float(data['GGT(U/L)']),
            'Na(mmol/L)': float(data['Na(mmol/L)']),
            'TBA(umol/L)': float(data['TBA(umol/L)'])
        }
        
        logger.info(f"Raw input: {raw_input}")
        
        # Use input normalizer to convert raw medical values to normalized range
        normalized_input = input_normalizer.normalize_input(raw_input)
        logger.info(f"Normalized input: {normalized_input}")
        
        # Step 2: Create a DataFrame with all features expected by the scaler
        # Initialize all features with 0 (will be handled by standardization)
        all_features = scaler.feature_names_in_
        full_input_data = pd.DataFrame(0, index=[0], columns=all_features)
        
        # Map our normalized features to the expected feature names
        feature_mapping = {
            'Daily dose（g）': 'Daily dose(g)',  # Note: scaler uses English parentheses
            'CLCR': 'CLCR',
            'GGT(U/L)': 'GGT(U/L)',
            'Na(mmol/L)': 'Na(mmol/L)',
            'ALP(U/L)': 'ALP(U/L)',
            'TBA(umol/L)': 'TBA(umol/L)'
        }
        
        # Fill in the normalized features
        for feature_name, normalized_value in normalized_input.items():
            if feature_name in feature_mapping:
                scaler_feature = feature_mapping[feature_name]
                if scaler_feature in all_features:
                    full_input_data[scaler_feature] = normalized_value
                else:
                    logger.warning(f"Feature {scaler_feature} not found in scaler features")
        
        logger.info(f"Created full DataFrame with shape: {full_input_data.shape}")
        logger.info(f"Non-zero features: {[(col, full_input_data[col].iloc[0]) for col in full_input_data.columns if full_input_data[col].iloc[0] != 0]}")
        
        # Standardize
        input_scaled = scaler.transform(full_input_data)
        logger.info(f"Data standardized, shape: {input_scaled.shape}")
        
        # Select only the 6 features that the SVM model expects
        selected_features = ['Daily dose(g)', 'CLCR', 'GGT(U/L)', 'Na(mmol/L)', 'ALP(U/L)', 'TBA(umol/L)']
        feature_indices = [list(all_features).index(feat) for feat in selected_features if feat in all_features]
        
        if len(feature_indices) != 6:
            logger.error(f"Expected 6 features, but found {len(feature_indices)}: {[all_features[i] for i in feature_indices]}")
            return jsonify({'error': f'Feature selection mismatch: expected 6, got {len(feature_indices)}'}), 500
        
        input_scaled_selected = input_scaled[:, feature_indices]
        logger.info(f"Selected features for SVM model, shape: {input_scaled_selected.shape}")
        
        # Predict
        prediction = model.predict(input_scaled_selected)[0]
        probability = model.predict_proba(input_scaled_selected)[0][1]
        
        logger.info(f"Raw prediction: {prediction}, probability: {probability}")
        
        # Interpret results
        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'prediction_text': 'Normal Blood Drug Concentration (≥ 0.5)' if prediction == 1 else 'Low Blood Drug Concentration (< 0.5)',
            'input_data': data
        }
        
        logger.info(f"Interpreted result: {result}")
        
        # Add clinical recommendations
        if prediction == 1:  # Normal blood drug concentration (≥ 0.5)
            if probability > 0.8:
                result['recommendation'] = "Very high confidence of normal blood drug concentration (≥ 0.5), current dosage is appropriate"
            elif probability > 0.6:
                result['recommendation'] = "High confidence of normal blood drug concentration (≥ 0.5), maintain current treatment"
            else:
                result['recommendation'] = "Moderate confidence of normal blood drug concentration (≥ 0.5), recommend regular monitoring"
        else:  # Low blood drug concentration (< 0.5)
            if probability > 0.8:
                result['recommendation'] = "Very high risk of low blood drug concentration (< 0.5), recommend immediate dosage increase and close monitoring"
            elif probability > 0.6:
                result['recommendation'] = "High risk of low blood drug concentration (< 0.5), recommend considering dosage increase"
            else:
                result['recommendation'] = "Moderate risk of low blood drug concentration (< 0.5), recommend enhanced monitoring"
        
        logger.info(f"API Prediction completed: {result}")
        
        # Return JSON response
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return jsonify({'error': 'Internal server error during prediction'}), 500

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
                         texts=get_texts(),
                         current_lang=current_lang,
                         languages=LANGUAGES), 404

@app.errorhandler(500)
def internal_error(error):
    current_lang = get_language()
    return render_template('error.html', 
                         error="Internal Server Error" if current_lang == 'en' else "Internal Server Error",
                         texts=get_texts(),
                         current_lang=current_lang,
                         languages=LANGUAGES), 500

if __name__ == '__main__':
    # Load model at startup
    if load_model():
        print("🚀 Blood Drug Concentration Prediction Web Application Started Successfully!")
        print(f"📊 Model Performance: Accuracy {metadata['performance']['accuracy']:.3f}, AUC {metadata['performance']['auc']:.3f}")
        print(f"🔧 Features Used: {', '.join(metadata['features'])}")
        
        # Get port from environment variable (for Render deployment)
        port = int(os.environ.get('PORT', 5000))
        print(f"🌐 Access URL: http://localhost:{port}")
        
        # Run with debug mode for local testing
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        print("❌ Model loading failed, unable to start application")
        print("Please ensure model files exist in the web_models directory")
        sys.exit(1)