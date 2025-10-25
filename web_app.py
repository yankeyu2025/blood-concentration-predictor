# -*- coding: utf-8 -*-
"""
血药浓度预测Web应用
基于逻辑回归模型的二分类预测系统
"""
import os, json, pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
from werkzeug.exceptions import BadRequest
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'blood_concentration_predictor_2024'

# 配置路径
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# 处理Render环境的路径问题 - 修正路径检测逻辑
if '/opt/render/project/src' in BASE_DIR:
    # Render环境，工作目录在src下，但模型文件在上级目录
    MODEL_DIR = '/opt/render/project/web_models'
else:
    # 本地环境或其他环境
    MODEL_DIR = os.path.join(BASE_DIR, 'web_models')

# 全局变量存储模型
model = None
scaler = None
metadata = None

# 语言配置
LANGUAGES = {
    'zh': {
        'name': '中文',
        'title': '血药浓度预测系统',
        'subtitle': '基于机器学习的智能血药浓度异常风险预测',
        'model_accuracy': '模型准确率：86.7%',
        'patient_info': '患者信息输入',
        'usage_instructions': '使用说明',
        'predict_button': '开始预测',
        'required_params': '缺少必需参数',
        'high_risk': '高风险',
        'low_risk': '低风险',
        'prediction_result': '预测结果',
        'risk_level': '风险等级',
        'confidence': '置信度',
        'recommendations': '建议',
        'high_risk_rec': '建议密切监测血药浓度，必要时调整用药剂量',
        'low_risk_rec': '当前用药相对安全，建议定期复查',
        'daily_dose': '日剂量 (g)',
        'daily_dose_info': '患者每日服用的药物总剂量，通常范围：0.1-5.0g',
        'clcr': '肌酐清除率 (mL/min)',
        'clcr_info': '肾功能指标，正常范围：80-120 mL/min',
        'ggt': 'GGT (U/L)',
        'ggt_info': '肝功能指标，正常范围：男性 ≤50 U/L，女性 ≤32 U/L',
        'sodium': '血钠 (mmol/L)',
        'sodium_info': '电解质平衡指标，正常范围：136-145 mmol/L',
        'hdl': 'HDL-C (mmol/L)',
        'hdl_info': '"好胆固醇"，正常范围：男性 >1.0，女性 >1.3 mmol/L',
        'albumin': '白蛋白 (g/L)',
        'albumin_info': '营养状态指标，正常范围：40-55 g/L',
        'usage_title': '使用说明',
        'usage_step1': '填写患者的基本生化指标',
        'usage_step2': '确保所有数值在正常临床范围内',
        'usage_step3': '点击预测按钮获取风险评估结果',
        'usage_step4': '根据预测结果调整治疗方案',
        'start_prediction': '开始预测',
        'predicting': '预测中...',
        'prediction_target': '预测目标',
        'prediction_target_desc': '评估患者血药浓度是否存在异常风险',
        'important_reminder': '重要提醒',
        'reminder_1': '本系统仅供临床参考，不能替代医生判断',
        'reminder_2': '请确保输入数据的准确性',
        'reminder_3': '异常结果需结合临床实际情况分析',
        'model_performance': '模型性能',
        'accuracy': '准确率：86.7%',
        'auc_value': 'AUC值：0.888',
        'features': '基于6个关键临床特征',
        'clinical_guidance': '临床指导',
        'kidney_function': '肾功能评估',
        'liver_function': '肝功能评估',
        'electrolyte_balance': '电解质平衡',
        # Result page texts
        'abnormal_risk': '血药浓度异常风险',
        'normal_concentration': '血药浓度正常',
        'monitor_closely': '建议密切监测血药浓度',
        'normal_range': '当前指标显示正常范围',
        'abnormal_probability': '异常概率',
        'input_data': '输入数据',
        'new_prediction': '新的预测',
        'print_result': '打印结果',
        'clinical_advice': '临床建议',
        'high_risk_management': '高风险患者管理',
        'monitor_concentration': '建议进行血药浓度监测',
        'adjust_dosage': '考虑调整给药剂量或频次',
        'watch_adverse': '密切观察不良反应',
        'assess_function': '评估肾肝功能状态',
        'monitoring_frequency': '监测频率建议',
        'initial_treatment': '初始治疗：每周监测',
        'after_adjustment': '剂量调整后：3-5天后复查',
        'stable_period': '稳定期：每月监测',
        'normal_management': '正常范围管理',
        'continue_treatment': '继续当前治疗方案',
        'regular_followup': '定期随访监测',
        'observe_symptoms': '注意观察临床症状',
        'maintain_compliance': '维持良好的依从性',
        'followup_advice': '随访建议',
        'stable_monitoring': '稳定期：每3个月监测',
        'symptom_changes': '如有症状变化及时就诊',
        'regular_assessment': '定期评估肾肝功能',
        'important_notes': '注意事项',
        'reference_only': '本预测结果仅供参考',
        'clinical_judgment': '需结合临床症状综合判断',
        'consult_specialist': '重要决策请咨询专科医生',
        'patient_communication': '保持与患者的良好沟通',
        'indicator_interpretation': '指标解读',
        'kidney_status': '肾功能状态',
        'liver_status': '肝功能状态',
        'electrolyte_status': '电解质平衡',
        'nutrition_status': '营养状态',
        'normal': '正常',
        'mild_decline': '轻度下降',
        'moderate_decline': '中度下降',
        'severe_decline': '重度下降',
        'mild_elevation': '轻度升高',
        'significant_elevation': '明显升高',
        'hyponatremia': '低钠血症',
        'hypernatremia': '高钠血症',
        'mild_deficiency': '轻度不足',
        'significant_deficiency': '明显不足'
    },
    'en': {
        'name': 'English',
        'title': 'Blood Drug Concentration Prediction System',
        'subtitle': 'AI-powered Blood Drug Concentration Abnormality Risk Prediction',
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
        'high_risk_rec': 'Close monitoring of blood drug concentration is recommended, adjust dosage if necessary',
        'low_risk_rec': 'Current medication is relatively safe, regular follow-up is recommended',
        'daily_dose': 'Daily Dose (g)',
        'daily_dose_info': 'Total daily medication dose, typical range: 0.1-5.0g',
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
        'usage_step1': 'Fill in the patient\'s basic biochemical indicators',
        'usage_step2': 'Ensure all values are within normal clinical ranges',
        'usage_step3': 'Click the predict button to get risk assessment results',
        'usage_step4': 'Adjust treatment plan based on prediction results',
        'start_prediction': 'Start Prediction',
        'predicting': 'Predicting...',
        'prediction_target': 'Prediction Target',
        'prediction_target_desc': 'Assess whether patients have abnormal blood drug concentration risk',
        'important_reminder': 'Important Reminder',
        'reminder_1': 'This system is for clinical reference only and cannot replace medical judgment',
        'reminder_2': 'Please ensure the accuracy of input data',
        'reminder_3': 'Abnormal results need to be analyzed in combination with clinical conditions',
        'model_performance': 'Model Performance',
        'accuracy': 'Accuracy: 86.7%',
        'auc_value': 'AUC Value: 0.888',
        'features': 'Based on 6 key clinical features',
        'clinical_guidance': 'Clinical Guidance',
        'kidney_function': 'Kidney Function Assessment',
        'liver_function': 'Liver Function Assessment',
        'electrolyte_balance': 'Electrolyte Balance',
        # Result page texts
        'abnormal_risk': 'Abnormal Blood Drug Concentration Risk',
        'normal_concentration': 'Normal Blood Drug Concentration',
        'monitor_closely': 'Close monitoring of blood drug concentration recommended',
        'normal_range': 'Current indicators show normal range',
        'abnormal_probability': 'Abnormal Probability',
        'input_data': 'Input Data',
        'new_prediction': 'New Prediction',
        'print_result': 'Print Result',
        'clinical_advice': 'Clinical Advice',
        'high_risk_management': 'High-Risk Patient Management',
        'monitor_concentration': 'Blood drug concentration monitoring recommended',
        'adjust_dosage': 'Consider adjusting dosage or frequency',
        'watch_adverse': 'Monitor closely for adverse reactions',
        'assess_function': 'Assess kidney and liver function',
        'monitoring_frequency': 'Monitoring Frequency Recommendations',
        'initial_treatment': 'Initial treatment: Weekly monitoring',
        'after_adjustment': 'After dose adjustment: Recheck in 3-5 days',
        'stable_period': 'Stable period: Monthly monitoring',
        'normal_management': 'Normal Range Management',
        'continue_treatment': 'Continue current treatment plan',
        'regular_followup': 'Regular follow-up monitoring',
        'observe_symptoms': 'Monitor clinical symptoms',
        'maintain_compliance': 'Maintain good compliance',
        'followup_advice': 'Follow-up Recommendations',
        'stable_monitoring': 'Stable period: Monitor every 3 months',
        'symptom_changes': 'Seek medical attention if symptoms change',
        'regular_assessment': 'Regular assessment of kidney and liver function',
        'important_notes': 'Important Notes',
        'reference_only': 'This prediction result is for reference only',
        'clinical_judgment': 'Should be combined with clinical symptoms for comprehensive judgment',
        'consult_specialist': 'Consult specialists for important decisions',
        'patient_communication': 'Maintain good communication with patients',
        'indicator_interpretation': 'Indicator Interpretation',
        'kidney_status': 'Kidney Function Status',
        'liver_status': 'Liver Function Status',
        'electrolyte_status': 'Electrolyte Balance Status',
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
    """获取当前语言设置，优先从URL参数获取"""
    # 优先从URL参数获取语言
    lang = request.args.get('lang')
    if lang and lang in LANGUAGES:
        session['language'] = lang
        return lang
    
    # 清除可能存在的中文session，默认为英文
    if 'language' in session and session['language'] == 'zh':
        session.pop('language', None)
    return session.get('language', 'en')

def get_text(key):
    """获取当前语言的文本"""
    lang = get_language()
    return LANGUAGES.get(lang, LANGUAGES['en']).get(key, key)

def load_model_components():
    """加载模型组件"""
    global model, scaler, metadata
    
    try:
        # 打印调试信息
        logger.info(f"当前工作目录: {os.getcwd()}")
        logger.info(f"BASE_DIR: {BASE_DIR}")
        logger.info(f"MODEL_DIR: {MODEL_DIR}")
        logger.info(f"模型目录是否存在: {os.path.exists(MODEL_DIR)}")
        
        # 列出项目根目录的内容
        project_root = '/opt/render/project'
        if os.path.exists(project_root):
            logger.info(f"项目根目录 {project_root} 内容:")
            try:
                for item in os.listdir(project_root):
                    item_path = os.path.join(project_root, item)
                    if os.path.isdir(item_path):
                        logger.info(f"  📁 {item}/")
                    else:
                        logger.info(f"  📄 {item}")
            except Exception as e:
                logger.error(f"无法列出目录内容: {e}")
        
        if os.path.exists(MODEL_DIR):
            logger.info(f"模型目录内容: {os.listdir(MODEL_DIR)}")
        
        # 尝试多个可能的路径
        possible_paths = [
            MODEL_DIR,
            os.path.join(BASE_DIR, 'web_models'),
            '/opt/render/project/web_models',
            '/opt/render/project/src/web_models'
        ]
        
        model_path = None
        for path in possible_paths:
            logger.info(f"尝试路径: {path}")
            model_file = os.path.join(path, 'logistic_regression_model.pkl')
            if os.path.exists(model_file):
                model_path = path
                logger.info(f"找到模型文件路径: {model_path}")
                break
            else:
                logger.info(f"路径不存在: {path}")
        
        if not model_path:
            raise FileNotFoundError("无法找到模型文件")
        
        # 加载模型
        with open(os.path.join(model_path, 'logistic_regression_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        
        # 加载标准化器
        with open(os.path.join(model_path, 'feature_scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        
        # 加载元数据
        with open(os.path.join(model_path, 'model_metadata.json'), 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        logger.info("模型组件加载成功")
        return True
    
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return False

def validate_input(data):
    """验证输入数据"""
    errors = []
    
    if not metadata:
        return ["模型未正确加载"]
    
    features = metadata['features']
    ranges = metadata['clinical_ranges']
    
    for feature in features:
        if feature not in data:
            errors.append(f"缺少必需参数: {feature}")
            continue
        
        try:
            value = float(data[feature])
            
            # 检查临床范围
            if feature in ranges:
                min_val = ranges[feature]['min']
                max_val = ranges[feature]['max']
                
                if value < min_val * 0.1 or value > max_val * 3:  # 允许一定范围的异常值
                    errors.append(f"{feature} 值异常: {value} (建议范围: {min_val}-{max_val})")
        
        except (ValueError, TypeError):
            errors.append(f"{feature} 必须是数值")
    
    return errors

@app.route('/')
def index():
    """主页"""
    if not metadata:
        return render_template('error.html', error="模型未正确加载，请联系管理员")
    
    return render_template('index.html', 
                         features=metadata['features'],
                         descriptions=metadata['feature_descriptions'],
                         ranges=metadata['clinical_ranges'],
                         performance=metadata['performance'],
                         texts=LANGUAGES[get_language()],
                         current_lang=get_language(),
                         languages=LANGUAGES)

@app.route('/set_language/<lang>')
def set_language(lang):
    """设置语言"""
    if lang in LANGUAGES:
        session['language'] = lang
    
    # 获取当前URL参数
    referrer = request.referrer or url_for('index')
    
    # 如果referrer包含lang参数，替换它；否则添加lang参数
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
    parsed_url = urlparse(referrer)
    query_params = parse_qs(parsed_url.query)
    query_params['lang'] = [lang]
    
    new_query = urlencode(query_params, doseq=True)
    new_url = urlunparse((
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        new_query,
        parsed_url.fragment
    ))
    
    return redirect(new_url)

@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        # 获取输入数据
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        # 验证输入
        errors = validate_input(data)
        if errors:
            if request.is_json:
                return jsonify({'success': False, 'errors': errors}), 400
            else:
                for error in errors:
                    flash(error, 'error')
                return redirect(url_for('index'))
        
        # 准备预测数据
        features = metadata['features']
        X = pd.DataFrame([{feature: float(data[feature]) for feature in features}])
        
        # 标准化
        X_scaled = scaler.transform(X)
        
        # 预测
        probability = model.predict_proba(X_scaled)[0, 1]
        prediction = model.predict(X_scaled)[0]
        
        # 解释结果
        result = {
            'probability': float(probability),
            'prediction': int(prediction),
            'prediction_text': '偏高' if prediction == 1 else '正常',
            'confidence': 'high' if abs(probability - 0.5) > 0.3 else 'medium' if abs(probability - 0.5) > 0.15 else 'low',
            'input_data': {feature: float(data[feature]) for feature in features}
        }
        
        # 添加临床建议
        if prediction == 1:
            if probability > 0.8:
                result['recommendation'] = "血药浓度偏高风险很高，建议立即调整用药剂量并密切监测"
            elif probability > 0.6:
                result['recommendation'] = "血药浓度偏高风险较高，建议考虑调整用药剂量"
            else:
                result['recommendation'] = "血药浓度偏高风险中等，建议加强监测"
        else:
            if probability < 0.2:
                result['recommendation'] = "血药浓度正常，当前用药方案合适"
            elif probability < 0.4:
                result['recommendation'] = "血药浓度基本正常，可继续当前用药方案"
            else:
                result['recommendation'] = "血药浓度接近临界值，建议定期监测"
        
        logger.info(f"预测完成: {result}")
        
        if request.is_json:
            return jsonify({'success': True, 'result': result})
        else:
            return render_template('result.html', result=result, 
                                 features=metadata['features'],
                                 descriptions=metadata['feature_descriptions'],
                                 texts=LANGUAGES[get_language()],
                                 current_lang=get_language(),
                                 languages=LANGUAGES)
    
    except Exception as e:
        logger.error(f"预测错误: {e}")
        error_msg = "预测过程中发生错误，请检查输入数据"
        
        if request.is_json:
            return jsonify({'success': False, 'error': error_msg}), 500
        else:
            flash(error_msg, 'error')
            return redirect(url_for('index'))

@app.route('/api/model_info')
def model_info():
    """模型信息API"""
    if not metadata:
        return jsonify({'error': '模型未加载'}), 500
    
    return jsonify({
        'model_type': metadata['model_type'],
        'features': metadata['features'],
        'feature_count': metadata['feature_count'],
        'performance': metadata['performance'],
        'feature_descriptions': metadata['feature_descriptions'],
        'clinical_ranges': metadata['clinical_ranges']
    })

@app.route('/health')
def health_check():
    """健康检查"""
    status = {
        'status': 'healthy' if model and scaler and metadata else 'unhealthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'metadata_loaded': metadata is not None
    }
    
    return jsonify(status)

@app.errorhandler(404)
def not_found(error):
    current_lang = get_language()
    return render_template('error.html', 
                         error="页面未找到" if current_lang == 'zh' else "Page Not Found",
                         current_lang=current_lang,
                         languages=LANGUAGES,
                         texts=LANGUAGES[current_lang]), 404

@app.errorhandler(500)
def internal_error(error):
    current_lang = get_language()
    return render_template('error.html', 
                         error="服务器内部错误" if current_lang == 'zh' else "Internal Server Error",
                         current_lang=current_lang,
                         languages=LANGUAGES,
                         texts=LANGUAGES[current_lang]), 500

if __name__ == '__main__':
    # 启动时加载模型
    if load_model_components():
        print("🚀 血药浓度预测Web应用启动成功!")
        print(f"📊 模型性能: 准确率 {metadata['performance']['accuracy']:.3f}, AUC {metadata['performance']['auc']:.3f}")
        print(f"🔧 使用特征: {', '.join(metadata['features'])}")
        print("🌐 访问地址: http://localhost:5000")
        
        import os
        port = int(os.environ.get('PORT', 5000))
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        print("❌ 模型加载失败，无法启动应用")
        print("请确保模型文件存在于 web_models 目录中")
