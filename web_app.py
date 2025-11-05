#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blood Drug Concentration Prediction Web Application
Binary classification prediction system based on an ensemble model.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import logging
from waitress import serve

# --- 全局变量和配置 ---
# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 初始化 Flask 应用
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['BABEL_DEFAULT_LOCALE'] = 'en'

# --- 轻量级语言与文本配置（用于模板渲染，避免 500 错误） ---
LANGUAGES = {
    'zh': {'name': '中文'},
    'en': {'name': 'English'}
}

# 双语文本（覆盖 index.html 与 result.html 所需键）
TEXTS = {
    'zh': {
        'title': '血药浓度预测系统',
        'subtitle': '基于集成模型的预测',
        'model_accuracy': '模型准确率',
        'model_auc': 'AUC',
        'patient_info': '患者信息输入',
        'usage_instructions': '使用说明',
        'start_prediction': '开始预测',
        'predicting': '正在预测...',
        'required_params': '请填写所有必填字段！',
        'prediction_result': '预测结果',
        'normal_concentration': '血药浓度 ≥ 0.5 mmol/L（正常）',
        'abnormal_risk': '血药浓度 < 0.5 mmol/L（偏低）',
        'normal_range': '已达标，建议常规随访',
        'monitor_closely': '血药浓度偏低，建议密切监测',
        'abnormal_probability': '血药浓度 ≥ 0.5 mmol/L 概率',
        'input_data': '输入数据',
        'daily_dose': '每日剂量',
        'clcr': '肌酐清除率',
        'ggt': 'γ-谷氨酰转移酶',
        'alp': '碱性磷酸酶',
        'alt': '谷丙转氨酶',
        'height': '身高',
        'alb': '白蛋白',
        'new_prediction': '再次预测',
        'print_result': '打印结果',
        'clinical_advice': '临床建议',
        'high_risk_management': '高风险管理建议',
        'monitor_drug_concentration': '监测血药浓度',
        'adjust_dosage': '评估并调整剂量',
        'observe_adverse_reactions': '观察不良反应',
        'assess_organ_function': '评估肝肾功能',
        'monitoring_frequency': '监测频率建议',
        'initial_treatment': '初始治疗阶段应密切监测',
        'after_dose_adjustment': '剂量调整后加强监测',
        'stable_period': '稳定期定期复查',
        'normal_management': '正常管理建议',
        'continue_current_treatment': '可继续当前治疗方案',
        'regular_follow_up': '定期随访',
        'observe_clinical_symptoms': '观察临床症状',
        'maintain_compliance': '保持良好依从性',
        'follow_up_advice': '随访建议',
        'stable_monitoring': '稳定期每3-6个月监测',
        'symptom_changes': '症状变化需及时复诊',
        'regular_organ_assessment': '定期评估肝肾功能',
        'important_notes': '重要提示',
        'reference_only': '本结果仅供参考',
        'clinical_judgment': '实际用药需结合临床判断',
        'consult_specialist': '必要时请咨询专业医师',
        'patient_communication': '与患者充分沟通',
        'indicator_interpretation': '相关指标解读',
        'kidney_function': '肾功能（CLCR）',
        'normal': '正常',
        'mild_decline': '轻度下降',
        'moderate_decline': '中度下降',
        'severe_decline': '重度下降',
        'liver_function': '肝功能（GGT/ALP/ALT）',
        'mild_elevation': '轻度升高',
        'significant_elevation': '明显升高',
        
        # 首页右侧说明与指南
        'prediction_target': '预测目标',
        'prediction_target_desc': '评估患者是否低于 0.5 mmol/L 阈值',
        'important_reminder': '重要提醒',
        'reminder_1': '本系统仅供临床参考，不能替代医生判断',
        'reminder_2': '请确保输入数据的准确性',
        'reminder_3': '异常结果需结合临床情况综合分析',
        'model_performance': '模型性能',
        'accuracy': '准确率',
        'auc_value': 'AUC值',
        'features': '基于6个关键临床特征',
        'clinical_guidance': '临床指导',
        'kidney_guide_content': '<strong>肌酐清除率 (CLCR)</strong><br>• 正常：80-120 mL/min<br>• 轻度损害：60-80 mL/min<br>• 中度损害：30-60 mL/min<br>• 重度损害：<30 mL/min',
        'liver_ggt_guide_content': '<strong>GGT参考范围</strong><br>• 男性：≤50 U/L<br>• 女性：≤32 U/L<br>• 升高提示肝胆疾病或药物性肝损伤',
        'electrolyte_balance': '电解质平衡',
        'sodium_guide_content': '<strong>钠离子正常范围</strong><br>• 136-145 mmol/L<br>• 低钠血症：<136 mmol/L<br>• 高钠血症：>145 mmol/L',
        
        # 表单字段说明
        'clcr_info': '肌酐清除率',
        'height_info': '身高 (cm)',
        'daily_dose_info': '每日剂量 (g)',
        'alb_info': '白蛋白 (g/L)',
        'alp_info': '碱性磷酸酶 (U/L)',
        'alt_info': '谷丙转氨酶 (U/L)',
        'ggt_info': 'γ-谷氨酰转移酶 (U/L)'
    },
    'en': {
        'title': 'Blood Drug Concentration Prediction System',
        'subtitle': 'Prediction based on ensemble model',
        'model_accuracy': 'Model Accuracy',
        'model_auc': 'AUC',
        'patient_info': 'Patient Information',
        'usage_instructions': 'Usage Instructions',
        'start_prediction': 'Start Prediction',
        'predicting': 'Predicting...',
        'required_params': 'Please fill in all required fields!',
        'prediction_result': 'Prediction Result',
        'normal_concentration': 'Blood concentration ≥ 0.5 mmol/L (Normal)',
        'abnormal_risk': 'Blood concentration < 0.5 mmol/L (Low)',
        'normal_range': 'On target; routine follow-up recommended',
        'monitor_closely': 'Concentration is low; recommend close monitoring',
        'abnormal_probability': 'Probability of ≥ 0.5 mmol/L concentration',
        'input_data': 'Input Data',
        'daily_dose': 'Daily Dose',
        'clcr': 'Creatinine Clearance',
        'ggt': 'Gamma-glutamyl transferase',
        'alp': 'Alkaline phosphatase',
        'alt': 'Alanine aminotransferase',
        'height': 'Height',
        'alb': 'Albumin',
        'new_prediction': 'New Prediction',
        'print_result': 'Print Result',
        'clinical_advice': 'Clinical Advice',
        'high_risk_management': 'High-Risk Management Advice',
        'monitor_drug_concentration': 'Monitor blood drug concentration',
        'adjust_dosage': 'Evaluate and adjust dosage',
        'observe_adverse_reactions': 'Observe adverse reactions',
        'assess_organ_function': 'Assess liver and kidney function',
        'monitoring_frequency': 'Monitoring Frequency Advice',
        'initial_treatment': 'Initial treatment: monitor closely',
        'after_dose_adjustment': 'After dose adjustment: strengthen monitoring',
        'stable_period': 'Stable period: regular review',
        'normal_management': 'Normal Management Advice',
        'continue_current_treatment': 'Continue current regimen',
        'regular_follow_up': 'Regular follow-up',
        'observe_clinical_symptoms': 'Observe clinical symptoms',
        'maintain_compliance': 'Maintain good adherence',
        'follow_up_advice': 'Follow-up Advice',
        'stable_monitoring': 'Stable period: monitor every 3–6 months',
        'symptom_changes': 'Seek medical attention if symptoms change',
        'regular_organ_assessment': 'Regular liver and kidney function assessment',
        'important_notes': 'Important Notes',
        'reference_only': 'This result is for reference only',
        'clinical_judgment': 'Decisions should be combined with clinical judgment',
        'consult_specialist': 'Consult specialist when necessary',
        'patient_communication': 'Communicate fully with the patient',
        'indicator_interpretation': 'Indicator Interpretation',
        'kidney_function': 'Kidney Function (CLCR)',
        'normal': 'Normal',
        'mild_decline': 'Mild Decline',
        'moderate_decline': 'Moderate Decline',
        'severe_decline': 'Severe Decline',
        'liver_function': 'Liver Function (GGT/ALP/ALT)',
        'mild_elevation': 'Mild Elevation',
        'significant_elevation': 'Significant Elevation',
        
        # 首页右侧说明与指南
        'prediction_target': 'Prediction Target',
        'prediction_target_desc': 'Assess whether concentration is below 0.5 mmol/L threshold',
        'important_reminder': 'Important Reminder',
        'reminder_1': 'This system is for clinical reference only and cannot replace doctor’s judgment',
        'reminder_2': 'Please ensure accuracy of input data',
        'reminder_3': 'Interpret abnormal results with clinical context',
        'model_performance': 'Model Performance',
        'accuracy': 'Accuracy',
        'auc_value': 'AUC',
        'features': 'Based on 6 key clinical features',
        'clinical_guidance': 'Clinical Guidance',
        'kidney_guide_content': '<strong>Creatinine Clearance (CLCR)</strong><br>• Normal: 80-120 mL/min<br>• Mild impairment: 60-80 mL/min<br>• Moderate impairment: 30-60 mL/min<br>• Severe impairment: <30 mL/min',
        'liver_ggt_guide_content': '<strong>GGT reference range</strong><br>• Male: ≤50 U/L<br>• Female: ≤32 U/L<br>• Elevation suggests hepatobiliary disease or drug-induced liver injury',
        'electrolyte_balance': 'Electrolyte Balance',
        'sodium_guide_content': '<strong>Sodium normal range</strong><br>• 136-145 mmol/L<br>• Hyponatremia: <136 mmol/L<br>• Hypernatremia: >145 mmol/L',
        
        # 表单字段说明
        'clcr_info': 'Creatinine clearance',
        'height_info': 'Height (cm)',
        'daily_dose_info': 'Daily Dose (g)',
        'alb_info': 'Albumin (g/L)',
        'alp_info': 'Alkaline phosphatase (U/L)',
        'alt_info': 'ALT (U/L)',
        'ggt_info': 'GGT (U/L)'
    }
}

def get_language():
    """从查询参数读取语言，默认英文；不合法值回退英文"""
    try:
        lang = request.args.get('lang', 'en')
        return lang if lang in LANGUAGES else 'en'
    except Exception:
        return 'en'

def get_texts():
    """根据当前语言返回文本字典（默认英文）"""
    current = get_language()
    return TEXTS.get(current, TEXTS['en'])

# --- 模型加载 ---
# 定义模型和相关文件的目录
# 兼容 Render 部署环境和本地环境
if os.path.exists('/opt/render/project/src'):
    MODEL_DIR = '/opt/render/project/src/集成模型'
else:
    MODEL_DIR = os.path.join(os.path.dirname(__file__), '集成模型')

logging.info(f"Attempting to load ensemble model from: {MODEL_DIR}")

ensemble_model = {}

def load_model():
    """加载所有模型组件"""
    try:
        model_data = {}
        model_data['catboost'] = joblib.load(os.path.join(MODEL_DIR, 'CatBoost_最终训练模型.pkl'))
        logging.info("CatBoost model loaded successfully.")
        
        model_data['xgboost'] = joblib.load(os.path.join(MODEL_DIR, 'XGBoost_最终训练模型.pkl'))
        logging.info("XGBoost model loaded successfully.")
        
        model_data['random_forest'] = joblib.load(os.path.join(MODEL_DIR, '随机森林_最终训练模型.pkl'))
        logging.info("Random Forest model loaded successfully.")

        # 加载标准化器
        model_data['scaler'] = joblib.load(os.path.join(MODEL_DIR, '数据标准化scaler.pkl'))
        logging.info("Scaler loaded successfully.")

        # 加载特征名称
        with open(os.path.join(MODEL_DIR, '最终特征集.json'), 'r', encoding='utf-8') as f:
            model_data['feature_names'] = json.load(f)
        logging.info(f"Feature names loaded successfully: {model_data['feature_names']}")

        # 加载集成权重
        with open(os.path.join(MODEL_DIR, '集成模型最佳权重.json'), 'r', encoding='utf-8') as f:
            model_data['weights'] = json.load(f)
        logging.info(f"Ensemble weights loaded successfully: {model_data['weights']}")
        
        logging.info("All ensemble model components loaded successfully.")
        return model_data

    except FileNotFoundError as e:
        logging.error(f"Error loading model components: {e}. Please check file paths.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during model loading: {e}")
        return None

# --- 应用启动时加载模型 ---
ensemble_model = load_model()
if not ensemble_model:
    logging.critical("Application startup failed: Model could not be loaded.")

# --- 请求处理前确保模型加载 ---
@app.before_request
def ensure_model_is_loaded():
    """在每个请求处理前，检查并确保模型已经加载。"""
    global ensemble_model
    if not ensemble_model:
        logging.warning("模型在请求期间未加载，正在尝试重新加载...")
        ensemble_model = load_model()
        if not ensemble_model:
            logging.error("在 before_request 中重新加载模型失败。")

# --- Flask 路由 ---
@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html',
                           texts=get_texts(),
                           current_lang=get_language(),
                           languages=LANGUAGES,
                           performance=None)

@app.route('/predict', methods=['POST'])
def predict():
    """处理预测请求"""
    try:
        # 经过 before_request 钩子后，我们期望模型已经加载
        if not ensemble_model:
            raise RuntimeError('模型未能成功加载，请检查启动日志。')

        # 进一步验证关键组件是否就绪
        required_keys = ['catboost', 'xgboost', 'random_forest', 'scaler', 'feature_names']
        for k in required_keys:
            if ensemble_model.get(k) is None:
                raise RuntimeError(f'模型组件未就绪：{k}')

        # 校验特征名称类型
        feature_names = ensemble_model.get('feature_names')
        if not isinstance(feature_names, (list, tuple)) or len(feature_names) == 0:
            raise RuntimeError('模型特征集未正确加载')

        # 1. 从表单获取输入数据
        form_data = request.form.to_dict()
        logging.info(f"Received form data: {form_data}")

        # 确保所有必需的特征都存在
        if not all(feature in form_data for feature in feature_names):
            missing = [f for f in feature_names if f not in form_data]
            return jsonify({'error': f'Missing required features: {", ".join(missing)}'}), 400

        # 2. 将输入数据转换为浮点数，并按正确的顺序排列
        try:
            input_features = [float(form_data[feature]) for feature in feature_names]
        except (ValueError, TypeError) as e:
            logging.error(f"Data conversion error: {e}. Input data: {form_data}")
            return jsonify({'error': 'Invalid input data. Please ensure all inputs are numbers.'}), 400

        # 3. 创建 DataFrame 并进行标准化（兼容 scaler 为63特征的情况）
        input_df = pd.DataFrame([input_features], columns=feature_names)
        scaler = ensemble_model['scaler']
        try:
            scaler_feature_count = getattr(scaler, 'n_features_in_', None)
            target_feature_count = len(ensemble_model['feature_names'])

            if scaler_feature_count == target_feature_count:
                # 直接使用scaler标准化
                scaled_features = scaler.transform(input_df)
            else:
                # 手动按7个特征进行标准化：使用scaler中对应特征的mean_和scale_
                full_names = list(getattr(scaler, 'feature_names_in_', []))
                if not full_names:
                    # 无法获取名称时，退化为不标准化
                    logging.warning("Scaler feature names unavailable; skipping scaling.")
                    scaled_features = input_df.values
                else:
                    indices = []
                    for fname in feature_names:
                        try:
                            indices.append(full_names.index(fname))
                        except ValueError:
                            raise ValueError(f"Scaler does not contain feature '{fname}'")

                    means = scaler.mean_
                    scales = scaler.scale_

                    # 逐特征标准化
                    row = []
                    for j, fname in enumerate(feature_names):
                        idx = indices[j]
                        val = float(input_df.iloc[0][fname])
                        m = float(means[idx])
                        s = float(scales[idx])
                        if s == 0:
                            # 避免除零，退化为差值
                            row.append(val - m)
                        else:
                            row.append((val - m) / s)
                    scaled_features = np.array([row], dtype=float)

            logging.info(f"Scaled features (shape {np.array(scaled_features).shape}): {scaled_features}")
        except Exception as e:
            logging.error(f"Scaling error: {e}", exc_info=True)
            # 若标准化失败，继续使用原始值，确保预测可用
            scaled_features = input_df.values.astype(float)

        # 4. 使用各个模型进行预测
        pred_catboost = ensemble_model['catboost'].predict_proba(scaled_features)[:, 1]
        pred_xgboost = ensemble_model['xgboost'].predict_proba(scaled_features)[:, 1]
        pred_rf = ensemble_model['random_forest'].predict_proba(scaled_features)[:, 1]

        # 5. 应用集成权重
        # 注意：权重文件结构为 {"models": [...], "weights": [...]}，可能不包含 RandomForest
        # 我们对可用模型进行加权；若权重缺失则回退为等权
        predictions = {
            'CatBoost': pred_catboost,
            'XGBoost': pred_xgboost,
            'RandomForest': pred_rf
        }

        weights_data = ensemble_model.get('weights', {})
        if isinstance(weights_data, dict) and 'models' in weights_data and 'weights' in weights_data:
            try:
                weights_map = {m: float(w) for m, w in zip(weights_data['models'], weights_data['weights'])}
            except Exception:
                weights_map = {}
        else:
            weights_map = {}

        active_models = list(predictions.keys())
        # 如果权重映射中没有任何已加载模型的权重，则使用等权
        if not any(name in weights_map for name in active_models):
            model_weights = {name: 1.0 for name in active_models}
            logging.warning("No matching model weights found; falling back to equal weighting.")
        else:
            model_weights = {name: float(weights_map.get(name, 1.0)) for name in active_models}

        weighted_sum = sum(predictions[name] * model_weights[name] for name in active_models)
        total_weight = sum(model_weights[name] for name in active_models)

        if total_weight == 0:
            logging.error("Total weight is zero, cannot perform weighted average.")
            return jsonify({'error': 'Model weights are not configured correctly.'}), 500
            
        final_prediction_prob = weighted_sum / total_weight

        # 6. 确定最终预测结果（使用0.5作为阈值）
        final_prob = float(final_prediction_prob[0])
        final_prediction = 1 if final_prob >= 0.5 else 0
        prediction_text = "高血药浓度" if final_prediction == 1 else "低血药浓度"

        logging.info(f"Final prediction probability: {final_prob:.4f}, Result: {prediction_text}")

        # 7. 返回结果（与模板结构对齐）
        result = {
            'prediction': int(final_prediction),
            'probability': float(final_prob),
            'input_data': {
                'CLCR': float(form_data['CLCR']),
                'height': float(form_data['height']),
                'Daily dose（g）': float(form_data['Daily dose(g)']) if 'Daily dose（g）' in form_data else float(form_data['Daily dose(g)']),
                'ALB(g/L)': float(form_data['ALB(g/L)']),
                'ALP(U/L)': float(form_data['ALP(U/L)']),
                'ALT(U/L)': float(form_data['ALT(U/L)']),
                'GGT(U/L)': float(form_data['GGT(U/L)'])
            }
        }

        return render_template('result.html',
                               texts=get_texts(),
                               current_lang=get_language(),
                               languages=LANGUAGES,
                               result=result)

    except Exception as e:
        logging.error(f"An error occurred in the predict function: {e}", exc_info=True)
        try:
            return render_template('error.html',
                                   error=str(e),
                                   texts=get_texts(),
                                   current_lang=get_language(),
                                   languages=LANGUAGES)
        except Exception as te:
            logging.error(f"Error rendering error.html: {te}", exc_info=True)
            # 最终兜底：返回纯文本错误，避免500白页
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 使用 Flask 自带的开发服务器运行，方便调试
    logging.info("Starting Flask development server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
