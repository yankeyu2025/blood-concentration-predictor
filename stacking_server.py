#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stacking Web Server (dynamic features)
Serves an interactive form based on '最终特征集.json' and performs stacking inference.
"""

import os
import json
import logging
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__, static_folder='web_ui', template_folder='web_ui')
app.config['SECRET_KEY'] = os.urandom(24)

# 文本与语言（增强版）
LANGUAGES = {
    'zh': {'name': '中文'},
    'en': {'name': 'English'},
}

def get_language():
    lang = request.args.get('lang')
    if lang == 'zh':
        return 'zh'
    return 'en'

def get_texts():
    if get_language() == 'en':
        return {
            'title': 'Lithium Carbonate Blood Drug Concentration Risk',
            'subtitle': '',
            'tips': 'Fill in patient features and start prediction',
            'threshold_label': 'Decision threshold',
            'fill_example_btn': 'Fill example',
            'start_btn': 'Predict',
            'risk_insufficient': 'Low drug exposure risk',
            'risk_adequate': 'Adequate drug exposure or high',
            'error': 'Error',
            'input_placeholder_prefix': 'Please input ',
            'patient_info': 'Patient Information Input',
            'prediction_result': 'Prediction Result',
            'prob_label': 'Positive probability',
            'risk_label': 'Risk level',
            'footer': 'CLCR Prediction · Flask + Stacking',
            'usage_title': 'Usage Instructions',
            'usage_target_title': 'Prediction Target',
            'usage_target_text': 'Assess whether patients have abnormal blood drug concentration risk',
            'usage_reminder_title': 'Important Reminder',
            'usage_reminder_items': [
                'This system is for clinical reference only and cannot replace medical judgment',
                'Ensure input units match required formats'
            ],
            'feature_units': {
                'CLCR': 'CLCR (mL/min)',
                'height': 'Height (cm)',
                'Daily dose(g)': 'Daily dose (g)',
                'ALB(g/L)': 'ALB (g/L)',
                'ALP(U/L)': 'ALP (U/L)',
                'GGT(U/L)': 'GGT (U/L)',
                'Na(mmol/L)': 'Na (mmol/L)'
            },
            'clcr_note_title': 'About CLCR',
            'clcr_note_text': 'CLCR is computed via the Cockcroft–Gault formula and expressed in mL/min.',
        }
    else:
        return {
            'title': '碳酸锂血药浓度预测系统',
            'subtitle': '',
            'tips': '请填写患者特征后进行预测',
            'threshold_label': '判定阈值',
            'fill_example_btn': '填充示例',
            'start_btn': '开始预测',
            'risk_insufficient': '低血药浓度风险',
            'risk_adequate': '血药浓度达标或偏高',
            'error': '错误',
            'input_placeholder_prefix': '请输入 ',
            'patient_info': '患者信息输入',
            'prediction_result': '预测结果',
            'prob_label': '阳性概率',
            'risk_label': '风险等级',
            'footer': 'CLCR 预测系统 · Flask + Stacking',
            'usage_title': '使用说明',
            'usage_target_title': '预测目标',
            'usage_target_text': '判断患者是否存在血药浓度异常风险',
            'usage_reminder_title': '重要提醒',
            'usage_reminder_items': [
                '本系统仅供临床参考，不能替代医疗判断',
                '请确保输入单位与要求一致'
            ],
            'feature_units': {
                'CLCR': 'CLCR（mL/min）',
                'height': '身高（cm）',
                'Daily dose(g)': '每日剂量（g）',
                'ALB(g/L)': '白蛋白（g/L）',
                'ALP(U/L)': '碱性磷酸酶（U/L）',
                'GGT(U/L)': 'γ-谷氨酰转移酶（U/L）',
                'Na(mmol/L)': '钠（mmol/L）'
            },
            'clcr_note_title': '关于 CLCR',
            'clcr_note_text': 'CLCR 为通过 Cockcroft–Gault 公式计算得到的肾清除率，单位 mL/min。',
        }

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.getenv('ENSEMBLE_MODEL_DIR', '')
if not MODEL_DIR:
    candidates = [
        os.path.join(BASE_DIR, '2_集成权重优化'),
        os.path.join(BASE_DIR, '3_最终评估与报告'),
        os.path.join(BASE_DIR, '集成模型'),
        os.path.join(os.getcwd(), '集成模型'),
        '/opt/render/project/src/皮尔逊假borutaCLCR/集成模型',
        '/opt/render/project/src/集成模型',
    ]
    for d in candidates:
        if os.path.exists(d):
            MODEL_DIR = d
            break
    if not MODEL_DIR:
        MODEL_DIR = os.path.join(BASE_DIR, '2_集成权重优化')
logging.info(f"MODEL_DIR -> {MODEL_DIR}")

ensemble_model = {}

def load_model():
    try:
        model = {}
        # 基模型
        model['knn'] = joblib.load(os.path.join(MODEL_DIR, 'K近邻_基模型.pkl'))
        model['lr'] = joblib.load(os.path.join(MODEL_DIR, '逻辑回归_基模型.pkl'))
        model['gnb'] = joblib.load(os.path.join(MODEL_DIR, '高斯朴素贝叶斯_基模型.pkl'))
        logging.info('Base models loaded.')

        # 元学习器
        model['meta'] = joblib.load(os.path.join(MODEL_DIR, 'Stacking_元学习器.pkl'))
        logging.info('Meta learner loaded.')

        # 标准化器（多路径兜底）
        scaler_candidates = [
            os.path.join(MODEL_DIR, '数据标准化scaler.pkl'),
            os.path.join(MODEL_DIR, '数据标准化器.pkl'),
            os.path.join(BASE_DIR, '1_数据划分与模型初选', '数据标准化器.pkl'),
        ]
        scaler_path = next((p for p in scaler_candidates if os.path.exists(p)), None)
        if not scaler_path:
            raise FileNotFoundError('缺少标准化器：数据标准化scaler.pkl 或 数据标准化器.pkl')
        model['scaler'] = joblib.load(scaler_path)
        logging.info(f"Scaler loaded from {os.path.basename(scaler_path)}")

        # 特征集（多路径兜底）
        feat_path = None
        for cand in [
            os.path.join(MODEL_DIR, '最终特征集.json'),
            os.path.join(BASE_DIR, '1_数据划分与模型初选', '最终特征集.json'),
        ]:
            if os.path.exists(cand):
                feat_path = cand
                break
        if not feat_path:
            raise FileNotFoundError('缺少最终特征集.json')
        with open(feat_path, 'r', encoding='utf-8') as f:
            model['feature_names'] = json.load(f)
        logging.info(f"Feature names -> {model['feature_names']}")

        # 标准化器特征顺序（可选）
        order_path = os.path.join(BASE_DIR, '1_数据划分与模型初选', '标准化器特征列.json')
        if os.path.exists(order_path):
            with open(order_path, 'r', encoding='utf-8') as f:
                model['scaler_feature_order'] = json.load(f)
        else:
            model['scaler_feature_order'] = model['feature_names']

        # 缺省填充值（可选）
        defaults_path = os.path.join(BASE_DIR, '1_数据划分与模型初选', '缺失值填充值.json')
        if os.path.exists(defaults_path):
            with open(defaults_path, 'r', encoding='utf-8') as f:
                model['fill_defaults'] = json.load(f)
        else:
            model['fill_defaults'] = {}

        # stacking 配置（可选）
        cfg_path = os.path.join(MODEL_DIR, 'Stacking_配置.json')
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r', encoding='utf-8') as f:
                model['stacking_cfg'] = json.load(f)
            logging.info('Stacking config loaded.')
        else:
            model['stacking_cfg'] = None

        return model
    except Exception as e:
        logging.error(f"load_model error: {e}", exc_info=True)
        return None

ensemble_model = load_model()

@app.before_request
def ensure_model():
    global ensemble_model
    if not ensemble_model:
        ensemble_model = load_model()

@app.get('/')
def index():
    return render_template(
        'index.html',
        texts=get_texts(),
        current_lang=get_language(),
        languages=LANGUAGES,
        feature_names=ensemble_model.get('feature_names', []),
        feature_units=get_texts().get('feature_units', {}),
    )

@app.get('/version')
def version():
    return jsonify({'python': os.sys.version})

@app.get('/form')
def form():
    names = ensemble_model.get('feature_names', [])
    if not names:
        return jsonify({'error': 'Feature names not loaded.'}), 500
    inputs_html = "".join([
        f'<div style="margin-bottom:8px;"><label style="display:block;margin-bottom:4px;">{name}</label>'
        f'<input type="text" name="{name}" placeholder="输入 {name}" style="width:280px;padding:6px;" required /></div>'
        for name in names
    ])
    html = (
        "<html><head><meta charset=\"utf-8\"><title>动态表单</title></head><body>"
        "<h2>动态特征输入（基于最终特征集）</h2>"
        "<form method=\"post\" action=\"/predict\" style=\"max-width:520px;\">"
        f"{inputs_html}"
        "<button type=\"submit\" style=\"padding:8px 16px;\">提交预测</button>"
        "</form>"
        "</body></html>"
    )
    return html

@app.post('/predict')
def predict():
    try:
        feature_names = ensemble_model['feature_names']
        logging.info(f"/predict called. features={feature_names}")

        req_json = request.get_json(silent=True)
        instances = None
        if isinstance(req_json, dict):
            instances = req_json.get('instances') or req_json

        form_data = None
        if not instances:
            form_data = request.form.to_dict()

        def extract_values(src: dict):
            filled = []
            vals_map = {}
            for fname in feature_names:
                v = src.get(fname)
                if v is None and fname == 'Daily dose（g）':
                    v = src.get('Daily dose(g)')
                if v is None or str(v).strip() == '':
                    d = ensemble_model.get('fill_defaults', {}).get(fname)
                    if d is None and fname == 'Daily dose（g）':
                        d = ensemble_model.get('fill_defaults', {}).get('Daily dose(g)')
                    if d is not None:
                        v = d
                        filled.append(fname)
                    else:
                        v = 0.0
                        filled.append(fname)
                if v is None or str(v).strip() == '':
                    vals_map[fname] = None
                    continue
                vals_map[fname] = float(v)
            full_order = ensemble_model.get('scaler_feature_order', feature_names)
            order_used = [f for f in full_order if f in feature_names]
            ensemble_model['input_feature_order'] = order_used
            missing = [f for f in feature_names if vals_map.get(f) is None]
            values = [vals_map[f] for f in order_used]
            return values, missing, filled

        if instances:
            inst = instances[0] if isinstance(instances, list) else instances
            values, missing, filled = extract_values(inst)
        else:
            values, missing, filled = extract_values(form_data or {})
        logging.info(f"filled={filled}, missing={missing}, values_len={len(values)}")

        if missing:
            return jsonify({'ok': False, 'error': 'Missing required features', 'missing': missing, 'filled': filled}), 200

        X = np.array([values], dtype=float)

        scaler = ensemble_model['scaler']
        try:
            Xs = scaler.transform(X)
        except Exception:
            means = getattr(scaler, 'mean_', None)
            scales = getattr(scaler, 'scale_', None)
            order = ensemble_model.get('scaler_feature_order')
            if means is None or scales is None or order is None:
                return jsonify({'ok': False, 'error': 'Scaler does not support manual scaling.'}), 500
            idx_map = {f: i for i, f in enumerate(order)}
            mean_vec = []
            scale_vec = []
            use_order = ensemble_model.get('input_feature_order', feature_names)
            for f in use_order:
                j = idx_map.get(f)
                if j is None:
                    mean_vec.append(0.0)
                    scale_vec.append(1.0)
                else:
                    mean_vec.append(float(means[j]))
                    scale_vec.append(float(scales[j]) if float(scales[j]) != 0 else 1.0)
            mean_vec = np.array(mean_vec, dtype=float)
            scale_vec = np.array(scale_vec, dtype=float)
            Xs = (X - mean_vec) / scale_vec
        logging.info(f"Xs_shape={Xs.shape}")

        p_knn = float(ensemble_model['knn'].predict_proba(Xs)[:, 1][0])
        p_lr  = float(ensemble_model['lr'].predict_proba(Xs)[:, 1][0])
        p_gnb = float(ensemble_model['gnb'].predict_proba(Xs)[:, 1][0])

        cfg = ensemble_model.get('stacking_cfg')
        probs_map = {'K近邻': p_knn, '逻辑回归': p_lr, '高斯朴素贝叶斯': p_gnb}
        if cfg and isinstance(cfg, dict):
            order = cfg.get('base_model_order') or cfg.get('base_models')
            if order and isinstance(order, list) and all(n in probs_map for n in order):
                meta_X = np.array([[probs_map[n] for n in order]], dtype=float)
            else:
                meta_X = np.array([[p_knn, p_lr, p_gnb]], dtype=float)
        else:
            meta_X = np.array([[p_knn, p_lr, p_gnb]], dtype=float)

        final_prob = float(ensemble_model['meta'].predict_proba(meta_X)[:, 1][0])
        threshold = 0.5
        final_label = 1 if final_prob >= threshold else 0

        if instances:
            return jsonify({
                'ok': True,
                'pos_proba': [final_prob],
                'pred': [final_label],
                'threshold': threshold,
                'filled': filled,
            })
        else:
            try:
                return render_template(
                    'result.html',
                    texts=get_texts(),
                    current_lang=get_language(),
                    languages=LANGUAGES,
                    result={
                        'prediction': final_label,
                        'probability': final_prob,
                        'threshold': threshold,
                        'filled': filled,
                    },
                )
            except Exception:
                return jsonify({
                    'ok': True,
                    'pos_proba': [final_prob],
                    'pred': [final_label],
                    'threshold': threshold,
                    'filled': filled,
                })
    except Exception as e:
        logging.error(f"predict error: {e}", exc_info=True)
        return jsonify({'ok': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5000'))
    use_waitress = os.environ.get('USE_WAITRESS', '1') == '1'
    if use_waitress:
        try:
            from waitress import serve
            serve(app, host='0.0.0.0', port=port)
        except Exception:
            app.run(host='0.0.0.0', port=port)
    else:
        app.run(host='0.0.0.0', port=port)
