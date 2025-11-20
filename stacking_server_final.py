import os
import json
import logging
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.metrics import roc_curve

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
last_result = None

def canonicalize_name(name: str):
    if not isinstance(name, str):
        return name
    return name.replace('（', '(').replace('）', ')').strip()

def load_model():
    try:
        model = {}
        model['base_paths'] = {}
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
                        model['base_paths'][cname] = fpath
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
        model['meta_path'] = meta_path
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
        model['scaler_path'] = scaler_path
        logging.info('Scaler loaded')

        feat_candidates = [
            os.path.join(MODEL_DIR, '最终特征集.json'),
            os.path.join(BASE_DIR, '1_数据划分与模型初选', '最终特征集.json')
        ]
        feat_path = next((p for p in feat_candidates if os.path.exists(p)), None)
        if feat_path:
            with open(feat_path, 'r', encoding='utf-8') as f:
                raw_feats = json.load(f)
                model['feature_names'] = [canonicalize_name(x) for x in raw_feats]
        else:
            model['feature_names'] = ['CLCR', 'height', 'Daily dose(g)', 'ALB(g/L)', 'ALP(U/L)', 'GGT(U/L)']

        order_candidates = [
            os.path.join(MODEL_DIR, '标准化器特征列.json'),
            os.path.join(BASE_DIR, '1_数据划分与模型初选', '标准化器特征列.json')
        ]
        order_path = next((p for p in order_candidates if os.path.exists(p)), None)
        if order_path:
            with open(order_path, 'r', encoding='utf-8') as f:
                raw_order = json.load(f)
                model['scaler_feature_order'] = [canonicalize_name(x) for x in raw_order]
            model['order_path'] = order_path
        else:
            model['scaler_feature_order'] = model['feature_names']

        defaults_path = os.path.join(BASE_DIR, '1_数据划分与模型初选', '缺失值填充值.json')
        if os.path.exists(defaults_path):
            with open(defaults_path, 'r', encoding='utf-8') as f:
                model['fill_defaults'] = json.load(f)
        else:
            model['fill_defaults'] = {}

        cfg_candidates = [
            os.path.join(MODEL_DIR, 'Stacking_配置.json'),
            os.path.join(BASE_DIR, '2_集成权重优化', 'Stacking_配置.json'),
        ]
        cfg_path = next((p for p in cfg_candidates if os.path.exists(p)), None)
        if cfg_path:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                model['stacking_cfg'] = json.load(f)
            model['cfg_path'] = cfg_path
        else:
            model['stacking_cfg'] = None

        decision_threshold = None
        if isinstance(model['stacking_cfg'], dict):
            decision_threshold = model['stacking_cfg'].get('decision_threshold')

        if decision_threshold is None:
            val_candidates = [
                os.path.join(BASE_DIR, '2_集成权重优化', 'Stacking_验证集预测.csv'),
                os.path.join(BASE_DIR, '3_最终评估与报告', '集成模型测试集预测结果.csv'),
            ]
            val_path = next((p for p in val_candidates if os.path.exists(p)), None)
            if val_path:
                try:
                    import csv
                    ys = []
                    ps = []
                    with open(val_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        cols = reader.fieldnames or []
                        y_cols = [c for c in cols if c.lower() in ['y_true','y','label','target','gt']]
                        p_cols = [c for c in cols if c.lower() in ['pos_proba','proba','prob','pred_proba','score']]
                        y_col = y_cols[0] if y_cols else None
                        p_col = p_cols[0] if p_cols else None
                        for row in reader:
                            if y_col and p_col:
                                ys.append(float(row[y_col]))
                                ps.append(float(row[p_col]))
                    if len(ys) >= 3 and len(ps) == len(ys):
                        fpr, tpr, thr = roc_curve(ys, ps)
                        idx = int(np.nanargmax(tpr - fpr)) if len(thr) > 0 else 0
                        if idx < len(thr):
                            decision_threshold = float(thr[idx])
                except Exception:
                    decision_threshold = None
        if decision_threshold is None:
            decision_threshold = 0.5
        model['decision_threshold'] = float(decision_threshold)
        model['model_dir'] = MODEL_DIR
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

@app.get('/info')
def info():
    cfg = ensemble_model.get('stacking_cfg') or {}
    scaled_from_cfg = []
    try:
        names = cfg.get('scaled_models') or []
        scaled_from_cfg = [canonicalize_name(n) for n in names]
    except Exception:
        scaled_from_cfg = names if isinstance(names, list) else []
    return jsonify({
        'model_dir': ensemble_model.get('model_dir'),
        'feature_names': ensemble_model.get('feature_names'),
        'input_feature_order': ensemble_model.get('input_feature_order'),
        'base_models': list(ensemble_model.get('base', {}).keys()),
        'base_paths': ensemble_model.get('base_paths'),
        'meta_path': ensemble_model.get('meta_path'),
        'scaler_path': ensemble_model.get('scaler_path'),
        'order_path': ensemble_model.get('order_path'),
        'cfg_path': ensemble_model.get('cfg_path'),
        'scaled_models': scaled_from_cfg,
        'decision_threshold': ensemble_model.get('decision_threshold')
    })

@app.get('/scaler')
def scaler_info():
    feats = ensemble_model.get('feature_names', [])
    order = ensemble_model.get('scaler_feature_order', feats)
    use_order = [f for f in order if f in feats]
    scaler = ensemble_model.get('scaler')
    means = getattr(scaler, 'mean_', None)
    scales = getattr(scaler, 'scale_', None)
    idx_map = {f: i for i, f in enumerate(order)}
    m = {f: (float(means[idx_map[f]]) if means is not None and f in idx_map else None) for f in use_order}
    s = {f: (float(scales[idx_map[f]]) if scales is not None and f in idx_map else None) for f in use_order}
    return jsonify({'order_used': use_order, 'means': m, 'scales': s, 'scaler_path': ensemble_model.get('scaler_path'), 'order_path': ensemble_model.get('order_path')})

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

        def extract_values(src: dict, strict: bool = False):
            filled = []
            vals_map = {}
            # 接受等价键名（ASCII 与全角）
            aliases = {
                'Daily dose（g）': 'Daily dose(g)'
            }
            name_set = {f: f for f in feature_names}
            for alt, base in aliases.items():
                name_set[alt] = base
            for fname in feature_names:
                v = src.get(fname)
                if v is None:
                    # 尝试用别名匹配
                    for alt, base in aliases.items():
                        if base == fname:
                            v = src.get(alt)
                            if v is not None:
                                break
                if strict:
                    if v is None or (isinstance(v, str) and v.strip() == ''):
                        vals_map[fname] = None
                        continue
                    vals_map[fname] = float(v)
                else:
                    if v is None or (isinstance(v, str) and v.strip() == ''):
                        d = ensemble_model.get('fill_defaults', {}).get(fname)
                        if d is None:
                            for alt, base in aliases.items():
                                if base == fname:
                                    d = ensemble_model.get('fill_defaults', {}).get(alt)
                                    break
                        if d is not None:
                            v = d
                            filled.append(fname)
                        else:
                            v = 0.0
                            filled.append(fname)
                    vals_map[fname] = float(v)
            order_used = feature_names
            ensemble_model['input_feature_order'] = order_used
            missing = [f for f in feature_names if vals_map.get(f) is None]
            values = [vals_map[f] for f in order_used]
            return values, missing, filled

        if instances:
            inst = instances[0] if isinstance(instances, list) else instances
            logging.info(f"inst_keys={list(inst.keys())}")
            logging.info(f"feature_names={feature_names}")
            values, missing, filled = extract_values(inst, strict=True)
        else:
            logging.info(f"form_keys={list((form_data or {}).keys())}")
            logging.info(f"feature_names={feature_names}")
            values, missing, filled = extract_values(form_data or {}, strict=True)
        if missing:
            return jsonify({'ok': False, 'error': 'Missing required features', 'missing': missing, 'filled': filled}), 200

        X = np.array([values], dtype=float)
        scaler = ensemble_model['scaler']
        means = getattr(scaler, 'mean_', None)
        scales = getattr(scaler, 'scale_', None)
        order = ensemble_model.get('scaler_feature_order')
        if means is not None and scales is not None and order is not None:
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
                    mean_val = float(means[j])
                    scale_val = float(scales[j])
                    mean_vec.append(mean_val)
                    scale_vec.append(scale_val if scale_val != 0 else 1.0)
            mean_vec = np.array(mean_vec, dtype=float)
            scale_vec = np.array(scale_vec, dtype=float)
            Xs = (X - mean_vec) / scale_vec
        else:
            Xs = scaler.transform(X)

        probs_map = {}
        cfg = ensemble_model.get('stacking_cfg')
        scaled_from_cfg = set()
        if isinstance(cfg, dict):
            names = cfg.get('scaled_models') or []
            try:
                scaled_from_cfg = {canonicalize_name(n) for n in names}
            except Exception:
                scaled_from_cfg = set(names)
        scaled_models = scaled_from_cfg if scaled_from_cfg else {'支持向量机', '逻辑回归'}
        use_order = ensemble_model.get('input_feature_order', feature_names)
        cur_idx = {f: i for i, f in enumerate(use_order)}
        for name, clf in ensemble_model['base'].items():
            X_src = Xs if name in scaled_models else X
            base_order = None
            try:
                fo = getattr(clf, 'feature_names_in_', None)
                if fo is not None:
                    base_order = [canonicalize_name(f) for f in list(fo)]
            except Exception:
                base_order = None
            if base_order:
                cols = [cur_idx[f] for f in base_order if f in cur_idx]
                if len(cols) == len(base_order):
                    X_eval = X_src[:, cols]
                else:
                    X_eval = X_src
            else:
                X_eval = X_src
            if hasattr(clf, 'predict_proba'):
                probs_map[name] = float(clf.predict_proba(X_eval)[:, 1][0])
            elif hasattr(clf, 'decision_function'):
                score = float(clf.decision_function(X_eval)[0])
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
        threshold = float(ensemble_model.get('decision_threshold', 0.5))
        final_label = 1 if final_prob >= threshold else 0

        use_order = ensemble_model.get('input_feature_order', feature_names)
        logging.info({'input_order': use_order, 'input_values': values, 'base_probs': probs_map, 'final_prob': final_prob, 'final_label': final_label, 'threshold': threshold})

        inputs_map = {f: float(v) for f, v in zip(use_order, values)}
        resp = {'ok': True, 'pos_proba': [final_prob], 'pred': [final_label], 'threshold': threshold, 'inputs': inputs_map, 'base_probs': probs_map}
        global last_result
        last_result = resp
        return jsonify(resp)
    except Exception as e:
        logging.error(f"predict error: {e}", exc_info=True)
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.get('/last')
def last():
    if last_result:
        return jsonify(last_result)
    return jsonify({'ok': False, 'error': 'no recent prediction'})

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
