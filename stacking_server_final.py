import os
import json
import logging
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__, static_folder='web_ui', template_folder='web_ui')
app.config['SECRET_KEY'] = os.urandom(24)

LANGUAGES = {
    'zh': {'name': '中文'},
    'en': {'name': 'English'},
}

def get_language():
    lang = request.args.get('lang')
    return 'zh' if lang == 'zh' else 'en'

def get_texts():
    if get_language() == 'en':
        return {
            'title': 'Lithium Blood Concentration Risk',
            'subtitle': 'Stacking Ensemble (RF + SVM + XGBoost)',
            'tips': 'Fill patient features to predict',
            'threshold_label': 'Decision threshold',
            'fill_example_btn': 'Fill example',
            'start_btn': 'Predict',
            'risk_insufficient': 'Low exposure risk',
            'risk_adequate': 'Adequate or high exposure',
            'error': 'Error',
            'input_placeholder_prefix': 'Please input ',
            'patient_info': 'Patient Information',
            'prediction_result': 'Prediction Result',
            'prob_label': 'Positive probability',
            'risk_label': 'Risk level',
            'footer': 'CLCR Prediction · Stacking',
            'feature_units': {
                'CLCR': 'CLCR (mL/min)',
                'height': 'Height (cm)',
                'Daily dose(g)': 'Daily dose (g)',
                'ALB(g/L)': 'ALB (g/L)',
                'ALP(U/L)': 'ALP (U/L)',
                'GGT(U/L)': 'GGT (U/L)'
            },
            'usage_title': 'Usage Instructions',
            'clcr_note_title': 'About CLCR',
            'clcr_note_text': 'CLCR reflects renal function and impacts lithium clearance.',
            'usage_target_title': 'Intended Use',
            'usage_target_text': 'For clinical decision support, not a diagnostic final.',
            'usage_reminder_title': 'Reminders',
            'usage_reminder_items': [
                'This tool assists clinicians and does not replace medical judgment.',
                'Ensure lab values and daily dose are accurate.',
                'Threshold is fixed at 0.50 per clinical rule.'
            ]
        }
    return {
        'title': '碳酸锂血药浓度预测系统',
        'subtitle': 'Stacking集成（随机森林+支持向量机+XGBoost）',
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
        'footer': 'CLCR 预测系统 · Stacking',
        'feature_units': {
            'CLCR': 'CLCR（mL/min）',
            'height': '身高（cm）',
            'Daily dose(g)': '每日剂量（g）',
            'ALB(g/L)': '白蛋白（g/L）',
            'ALP(U/L)': '碱性磷酸酶（U/L）',
            'GGT(U/L)': 'γ-谷氨酰转移酶（U/L）'
        },
        'usage_title': '使用说明',
        'clcr_note_title': '关于 CLCR',
        'clcr_note_text': 'CLCR 反映肾功能，会影响锂的清除。',
        'usage_target_title': '适用范围',
        'usage_target_text': '用于临床决策支持，不作为最终诊断依据。',
        'usage_reminder_title': '注意事项',
        'usage_reminder_items': [
            '本工具用于辅助，不替代医生专业判断。',
            '请确保化验数值与每日剂量填写准确。',
            '阈值固定 0.50，遵循临床规定。'
        ]
    }

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.getenv('ENSEMBLE_MODEL_DIR', '')
if not MODEL_DIR:
    candidates = [
        os.path.join(BASE_DIR, 'web_models'),
        os.path.join(BASE_DIR, '3_最终评估与报告'),
        os.path.join(BASE_DIR, '2_集成权重优化')
    ]
    for d in candidates:
        if os.path.exists(d):
            MODEL_DIR = d
            break
logging.info(f"MODEL_DIR -> {MODEL_DIR}")

ensemble_model = {}

def load_model():
    try:
        model = {}
        base_model_file_map = {
            '随机森林': '随机森林_最终训练模型.pkl',
            '支持向量机': '支持向量机_最终训练模型.pkl',
            'XGBoost': 'XGBoost_最终训练模型.pkl',
            'K近邻': 'K近邻_基模型.pkl',
            '逻辑回归': '逻辑回归_基模型.pkl',
            '高斯朴素贝叶斯': '高斯朴素贝叶斯_基模型.pkl'
        }
        model['base'] = {}
        base_dir_candidates = []
        for d in [
            MODEL_DIR,
            os.path.join(BASE_DIR, 'web_models'),
            os.path.join(BASE_DIR, '3_最终评估与报告'),
            os.path.join(BASE_DIR, '2_集成权重优化'),
        ]:
            if d and os.path.exists(d) and d not in base_dir_candidates:
                base_dir_candidates.append(d)
        for cname, fname in base_model_file_map.items():
            loaded = False
            for d in base_dir_candidates:
                fpath = os.path.join(d, fname)
                if os.path.exists(fpath):
                    try:
                        model['base'][cname] = joblib.load(fpath)
                        logging.info(f"Base model loaded: {cname} from {d}")
                        loaded = True
                        break
                    except Exception as e:
                        logging.warning(f"Failed to load {cname} from {d}: {e}")
            if not loaded:
                logging.info(f"Base model not found: {cname}")
        if not model['base']:
            raise FileNotFoundError('未找到基模型文件')
        new_set = ['随机森林', '支持向量机', 'XGBoost']
        if all(k in model['base'] for k in new_set):
            model['base'] = {k: model['base'][k] for k in new_set}
            logging.info('Using latest stacking base models: RF + SVM + XGBoost')

        meta_candidates = [
            os.path.join(MODEL_DIR, 'Stacking_元学习器.pkl'),
            os.path.join(BASE_DIR, '2_集成权重优化', 'Stacking_元学习器.pkl')
        ]
        meta_path = next((p for p in meta_candidates if os.path.exists(p)), None)
        if not meta_path:
            raise FileNotFoundError('缺少Stacking_元学习器.pkl')
        model['meta'] = joblib.load(meta_path)
        logging.info('Meta learner loaded')

        scaler_candidates = [
            os.path.join(MODEL_DIR, '数据标准化器.pkl'),
            os.path.join(MODEL_DIR, 'feature_scaler.pkl'),
            os.path.join(BASE_DIR, '1_数据划分与模型初选', '数据标准化器.pkl'),
            os.path.join(BASE_DIR, 'feature_scaler.pkl')
        ]
        scaler_path = next((p for p in scaler_candidates if os.path.exists(p)), None)
        if not scaler_path:
            raise FileNotFoundError('缺少标准化器')
        model['scaler'] = joblib.load(scaler_path)
        logging.info('Scaler loaded')

        feat_candidates = [
            os.path.join(MODEL_DIR, '最终特征集.json'),
            os.path.join(BASE_DIR, '1_数据划分与模型初选', '最终特征集.json')
        ]
        feat_path = next((p for p in feat_candidates if os.path.exists(p)), None)
        if feat_path:
            with open(feat_path, 'r', encoding='utf-8') as f:
                model['feature_names'] = json.load(f)
        else:
            model['feature_names'] = ['CLCR', 'height', 'Daily dose(g)', 'ALB(g/L)', 'ALP(U/L)', 'GGT(U/L)']

        order_path = os.path.join(BASE_DIR, '1_数据划分与模型初选', '标准化器特征列.json')
        if os.path.exists(order_path):
            with open(order_path, 'r', encoding='utf-8') as f:
                model['scaler_feature_order'] = json.load(f)
        else:
            model['scaler_feature_order'] = model['feature_names']

        defaults_path = os.path.join(BASE_DIR, '1_数据划分与模型初选', '缺失值填充值.json')
        if os.path.exists(defaults_path):
            with open(defaults_path, 'r', encoding='utf-8') as f:
                model['fill_defaults'] = json.load(f)
        else:
            model['fill_defaults'] = {}

        cfg_path = os.path.join(MODEL_DIR, 'Stacking_配置.json')
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r', encoding='utf-8') as f:
                model['stacking_cfg'] = json.load(f)
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

@app.post('/predict')
def predict():
    try:
        feature_names = ensemble_model['feature_names']
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

        probs_map = {}
        for name, clf in ensemble_model['base'].items():
            if hasattr(clf, 'predict_proba'):
                probs_map[name] = float(clf.predict_proba(Xs)[:, 1][0])
            elif hasattr(clf, 'decision_function'):
                score = float(clf.decision_function(Xs)[0])
                probs_map[name] = float(1.0 / (1.0 + np.exp(-score)))
            else:
                return jsonify({'ok': False, 'error': f'{name} 不支持概率预测'}), 500

        cfg = ensemble_model.get('stacking_cfg')
        order = None
        if cfg and isinstance(cfg, dict):
            order = cfg.get('base_model_order') or cfg.get('base_models')
        if not order:
            order = list(probs_map.keys())
        missing_models = [n for n in order if n not in probs_map]
        if missing_models:
            return jsonify({'ok': False, 'error': '缺少必要的基模型', 'missing_models': missing_models}), 200
        meta_X = np.array([[probs_map[n] for n in order]], dtype=float)

        final_prob = float(ensemble_model['meta'].predict_proba(meta_X)[:, 1][0])
        threshold = 0.5
        final_label = 1 if final_prob >= threshold else 0

        return jsonify({'ok': True, 'pos_proba': [final_prob], 'pred': [final_label], 'threshold': threshold})
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
