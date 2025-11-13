import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
import sys


# 兼容不同部署根目录：优先使用就近的 templates，其次使用上一级目录的 templates
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_TEMPLATE_DIRS = [
    os.path.join(_CUR_DIR, 'templates'),
    os.path.join(os.path.dirname(_CUR_DIR), 'templates'),
]
_TEMPLATE_DIR = next((d for d in _CANDIDATE_TEMPLATE_DIRS if os.path.isdir(d)), None)
app = Flask(__name__, template_folder=_TEMPLATE_DIR) if _TEMPLATE_DIR else Flask(__name__)


def standardize_brackets_in_columns(df):
    df.columns = [col.replace('（', '(').replace('）', ')') for col in df.columns]
    return df


def robust_to_numeric(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            series_no_comma = df[col].astype(str).str.replace(',', '', regex=False)
            numeric_series = pd.to_numeric(series_no_comma, errors='coerce')
            if numeric_series.isnull().any():
                median_val = numeric_series.median()
                df[col] = numeric_series.fillna(median_val)
            else:
                df[col] = numeric_series
    return df


def load_assets_from_web_models():
    # 从仓库根目录下的 web_models 读取（与 GitHub 结构一致）
    root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(root, 'web_models')
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f'未找到目录: {models_dir}，请在仓库根目录创建 web_models 并上传模型文件')

    # 必需文件
    path_scaler = os.path.join(models_dir, '数据标准化器.pkl')
    path_final_features = os.path.join(models_dir, '最终特征集.json')
    path_scaler_features = os.path.join(models_dir, '标准化器特征列.json')
    path_fill_values = os.path.join(models_dir, '缺失值填充值.json')
    path_meta = os.path.join(models_dir, 'Stacking_元学习器.pkl')

    # 基模型文件名（按你的仓库命名习惯）
    base_model_files = {
        '逻辑回归': '逻辑回归_最终训练模型.pkl',
        '高斯朴素贝叶斯': '高斯朴素贝叶斯_最终训练模型.pkl',
        'K近邻': 'K近邻_最终训练模型.pkl',
    }

    # 读取
    scaler = joblib.load(path_scaler)
    with open(path_final_features, 'r', encoding='utf-8') as f:
        final_features = json.load(f)
    with open(path_scaler_features, 'r', encoding='utf-8') as f:
        scaler_features = json.load(f)
    fill_values = {}
    if os.path.exists(path_fill_values):
        with open(path_fill_values, 'r', encoding='utf-8') as f:
            fill_values = json.load(f)
    stacking_meta = joblib.load(path_meta)

    base_models = {}
    for name, filename in base_model_files.items():
        path_model = os.path.join(models_dir, filename)
        if not os.path.exists(path_model):
            raise FileNotFoundError(f'缺少基模型文件: {filename} (web_models/)')
        base_models[name] = joblib.load(path_model)

    # 默认只有逻辑回归使用缩放后的特征
    use_scaled_names = {'逻辑回归'}

    return {
        'scaler': scaler,
        'scaler_features': scaler_features,
        'final_features': final_features,
        'fill_values': fill_values,
        'base_models': base_models,
        'stacking_meta': stacking_meta,
        'use_scaled_names': list(use_scaled_names),
    }


# 兼容补丁：若模型是在 NumPy 2.x（引用 numpy._core）环境下保存，而运行时环境缺少该模块，尝试映射到 numpy.core
try:
    import numpy._core  # type: ignore
except Exception:
    try:
        import numpy.core as _np_core  # type: ignore
        sys.modules['numpy._core'] = _np_core
    except Exception:
        pass

PAYLOAD = None
SCALER = None
SCALER_FEATURES = None
FINAL_FEATURES = None
FILL_VALUES = {}
BASE_MODELS = None
STACKING_META = None
USE_SCALED_NAMES = set()


def ensure_assets_loaded():
    global PAYLOAD, SCALER, SCALER_FEATURES, FINAL_FEATURES, FILL_VALUES, BASE_MODELS, STACKING_META, USE_SCALED_NAMES
    if PAYLOAD is not None:
        return
    PAYLOAD = load_assets_from_web_models()
    SCALER = PAYLOAD['scaler']
    SCALER_FEATURES = PAYLOAD['scaler_features']
    FINAL_FEATURES = PAYLOAD['final_features']
    FILL_VALUES = PAYLOAD.get('fill_values', {})
    BASE_MODELS = PAYLOAD['base_models']
    STACKING_META = PAYLOAD['stacking_meta']
    USE_SCALED_NAMES = set(PAYLOAD.get('use_scaled_names', []))


def example_sample():
    # 示例请求体中的样本，便于用户直接复制调用
    return {
        "Daily dose(g)": 1.0,
        "CLCR": 80,
        "ALB(g/L)": 42,
        "height": 170,
        "Na(mmol/L)": 140,
        "GGT(U/L)": 30,
        "ALP(U/L)": 80,
    }


def scale_final_features(df_final):
    ensure_assets_loaded()
    # 使用标准化器的均值与尺度对最终特征进行手动缩放
    names = list(getattr(SCALER, 'feature_names_in_', SCALER_FEATURES))
    idx = [names.index(f) for f in FINAL_FEATURES]
    means = np.array(SCALER.mean_)[idx]
    scales = np.array(SCALER.scale_)[idx]
    scaled = (df_final[FINAL_FEATURES].astype(float) - means) / scales
    scaled.columns = FINAL_FEATURES
    return scaled


def stacking_pipeline_proba(df_input):
    ensure_assets_loaded()
    # 预处理
    df = df_input.copy()
    standardize_brackets_in_columns(df)
    robust_to_numeric(df)
    # 缺失处理（按保存的填充值或用中位数回退）
    df = df.fillna(FILL_VALUES) if FILL_VALUES else df.fillna(df.median(numeric_only=True))
    # 仅保留最终特征，并按顺序排列
    df_final = df.reindex(columns=FINAL_FEATURES)
    df_final = df_final.fillna(df_final.median(numeric_only=True))

    # 缩放后的最终特征
    df_final_scaled = scale_final_features(df_final)

    base_probas = []
    for name in [*BASE_MODELS.keys()]:
        model = BASE_MODELS[name]
        X_eval = df_final_scaled if name in USE_SCALED_NAMES else df_final
        base_probas.append(model.predict_proba(X_eval)[:, 1])
    stacking_features = np.column_stack(base_probas)
    return STACKING_META.predict_proba(stacking_features)[:, 1]


@app.get('/health')
def health():
    ensure_assets_loaded()
    return jsonify({
        'status': 'ok',
        'base_models': list(BASE_MODELS.keys()),
        'meta': 'Stacking',
        'features_order': FINAL_FEATURES,
    })


@app.get('/')
def index():
    # 优先渲染模板文件，若不存在则回退到简单的内联页面
    try:
        features = FINAL_FEATURES if FINAL_FEATURES is not None else []
        return render_template('index.html', features=features, sample=example_sample())
    except Exception as e:
        print(f"[index] 模板渲染失败，回退到内联页面：{e}", file=sys.stderr)
        html = (
            "<html><head><meta charset='utf-8'><title>Stacking 服务</title></head>"
            "<body style='font-family:system-ui,Segoe UI,Arial;max-width:960px;margin:40px auto;line-height:1.6'>"
            "<h1>Stacking 二分类服务已启动</h1>"
            "<p>健康检查：<a href='/health'>/health</a></p>"
            "<p>预测接口：POST <code>/predict</code>，示例 JSON：</p>"
            "<pre>{\n  \"instances\": [{\n    \"Daily dose(g)\": 1.0,\n    \"CLCR\": 80,\n    \"ALB(g/L)\": 42,\n    \"height\": 170,\n    \"Na(mmol/L)\": 140,\n    \"GGT(U/L)\": 30,\n    \"ALP(U/L)\": 80\n  }]\n}</pre>"
            "</body></html>"
        )
        return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.get('/index.html')
def index_html():
    return index()


# 一些预览器会自动请求 /@vite/client，这里返回一个空 JS 以避免 404
@app.get('/@vite/client')
def vite_client():
    return "// stub for previewer", 200, {"Content-Type": "application/javascript"}


@app.get('/favicon.ico')
def favicon():
    return "", 204, {"Content-Type": "image/x-icon"}


@app.post('/predict')
def predict():
    try:
        payload = request.get_json(force=True)
        instances = payload.get('instances')
        if not isinstance(instances, list) or len(instances) == 0:
            return jsonify({'error': 'instances 需为非空数组'}), 400
        df = pd.DataFrame(instances)
        # 统一括号与列顺序
        standardize_brackets_in_columns(df)
        # 对缺失的最终特征列补齐
        for col in FINAL_FEATURES:
            if col not in df.columns:
                df[col] = np.nan
        df = df[FINAL_FEATURES]

        proba = stacking_pipeline_proba(df)
        label = (proba >= 0.5).astype(int).tolist()
        return jsonify({
            'proba': proba.tolist(),
            'label': label,
            'features_order': FINAL_FEATURES,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.get('/version')
def version():
    # 便于部署环境检查实际安装的版本
    return jsonify({
        'numpy': getattr(np, '__version__', None),
        'pandas': getattr(pd, '__version__', None),
        'scikit_learn': getattr(__import__('sklearn'), '__version__', None),
        'joblib': getattr(joblib, '__version__', None),
        'python': sys.version,
    })


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
