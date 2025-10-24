# -*- coding: utf-8 -*-
"""
血药浓度预测Web应用
基于逻辑回归模型的二分类预测系统
"""
import os, json, pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.exceptions import BadRequest
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'blood_concentration_predictor_2024'

# 配置路径
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# 处理Render环境的路径问题
if os.path.exists('/opt/render/project/src'):
    # Render环境，模型文件在src目录下
    MODEL_DIR = '/opt/render/project/src/web_models'
else:
    # 本地环境或其他环境
    MODEL_DIR = os.path.join(BASE_DIR, 'web_models')

# 全局变量存储模型
model = None
scaler = None
metadata = None

def load_model_components():
    """加载模型组件"""
    global model, scaler, metadata
    
    try:
        # 打印调试信息
        logger.info(f"当前工作目录: {os.getcwd()}")
        logger.info(f"BASE_DIR: {BASE_DIR}")
        logger.info(f"MODEL_DIR: {MODEL_DIR}")
        logger.info(f"模型目录是否存在: {os.path.exists(MODEL_DIR)}")
        
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
            model_file = os.path.join(path, 'logistic_regression_model.pkl')
            if os.path.exists(model_file):
                model_path = path
                logger.info(f"找到模型文件路径: {model_path}")
                break
        
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
                         performance=metadata['performance'])

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
                                 descriptions=metadata['feature_descriptions'])
    
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
    return render_template('error.html', error="页面未找到"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error="服务器内部错误"), 500

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