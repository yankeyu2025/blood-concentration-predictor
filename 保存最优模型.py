# -*- coding: utf-8 -*-
"""
保存最优模型和特征处理器
用于Web应用部署
"""
import os, json, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, '流程配置.json')
DATA_FILES = {
    'train': os.path.join(os.path.dirname(BASE_DIR), 'train_set_concentration.csv'),
    'test': os.path.join(os.path.dirname(BASE_DIR), 'test_set_concentration.csv')
}
TARGET_NAME = 'concentration(ng/ml)'
RESULT_ROOT = os.path.join(BASE_DIR, '结果')
MODEL_DIR = os.path.join(BASE_DIR, 'web_models')

ENCODINGS = ['utf-8','utf-8-sig','gbk','gb18030','cp936']

def read_csv_any(path):
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def normalize_target(df):
    t = None
    for c in df.columns:
        if 'concentration' in c.lower(): 
            t = c
            break
    if t and t != TARGET_NAME: 
        df = df.rename(columns={t: TARGET_NAME})
    return df

def load_config():
    cfg = {
        'include_clcr': True,
        'drop_clcr_sources': True,
        'clcr_source_columns': ['UREA','weight','age'],
        'thresholds': {'binary':0.5, 'three_class_low':0.5, 'three_class_high':1.2},
        'use_standard_scaler': True,
        'feature_selection': {'pearson_min_abs_corr':0.1, 'use_boruta':True}
    }
    if os.path.isfile(CONFIG_PATH):
        try:
            user = json.load(open(CONFIG_PATH,'r',encoding='utf-8'))
            cfg.update(user)
        except Exception:
            pass
    return cfg

def build_binary_labels(df, cfg):
    """构建二分类标签"""
    y = df[TARGET_NAME].astype(float)
    th = cfg['thresholds']['binary']
    return (y >= th).astype(int).values

def match_any(col, keys):
    lc = col.lower()
    for k in keys:
        lk = k.lower()
        if lk in lc:
            return True
    cn_map = {'urea':'尿素','weight':'体重','age':'年龄'}
    for en, cn in cn_map.items():
        if en in keys and cn in lc:
            return True
    return False

def load_final_features():
    """加载最终选择的特征"""
    feature_file = os.path.join(RESULT_ROOT, 'binary_未使用SMOTE', '最终特征_binary_未使用SMOTE.json')
    with open(feature_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['最终特征']

def preprocess_data(cfg):
    """数据预处理"""
    print("📊 加载和预处理数据...")
    
    # 加载数据
    train_df = normalize_target(read_csv_any(DATA_FILES['train']))
    test_df = normalize_target(read_csv_any(DATA_FILES['test']))
    
    # 删除CLCR相关列
    if cfg['drop_clcr_sources']:
        drop_cols = [c for c in train_df.columns if match_any(c, cfg['clcr_source_columns'])]
        train_df = train_df.drop(columns=drop_cols, errors='ignore')
        test_df = test_df.drop(columns=drop_cols, errors='ignore')
    
    # 加载最终特征
    final_features = load_final_features()
    print(f"📋 使用特征: {final_features}")
    
    # 分离特征和标签
    X_train = train_df[final_features].copy()
    X_test = test_df[final_features].copy()
    
    # 确保所有特征都是数值型
    for col in final_features:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    
    # 填充缺失值
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())
    
    # 检查数据分布并处理异常值
    print("🔍 检查数据分布...")
    for col in final_features:
        q1 = X_train[col].quantile(0.25)
        q3 = X_train[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # 处理异常值（使用边界值替换）
        outliers_count = ((X_train[col] < lower_bound) | (X_train[col] > upper_bound)).sum()
        if outliers_count > 0:
            print(f"   {col}: 发现 {outliers_count} 个异常值，进行处理")
            X_train[col] = X_train[col].clip(lower_bound, upper_bound)
            X_test[col] = X_test[col].clip(lower_bound, upper_bound)
    
    y_train = build_binary_labels(train_df, cfg)
    y_test = build_binary_labels(test_df, cfg)
    
    print(f"✅ 训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print(f"📈 类别分布 - 训练集: {np.bincount(y_train)}, 测试集: {np.bincount(y_test)}")
    
    # 打印特征统计信息
    print("\n📊 特征统计信息:")
    for col in final_features:
        print(f"   {col}: mean={X_train[col].mean():.3f}, std={X_train[col].std():.3f}")
    
    return X_train, X_test, y_train, y_test, final_features

def train_final_model(X_train, y_train, final_features):
    """训练最终模型"""
    print("🔧 训练最终的逻辑回归模型...")
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # 使用SMOTE处理类别不平衡
    print("⚖️ 使用SMOTE处理类别不平衡...")
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_scaled, y_train)
    
    print(f"📈 SMOTE后类别分布: {np.bincount(y_balanced)}")
    
    # 训练模型（使用优化后的参数）
    model = LogisticRegression(
        C=1,  # 使用超参数优化得到的最佳参数
        max_iter=1000,
        solver='liblinear',
        random_state=42,
        class_weight='balanced'  # 添加类别权重平衡
    )
    model.fit(X_balanced, y_balanced)
    
    print("✅ 模型训练完成")
    return model, scaler

def evaluate_final_model(model, scaler, X_test, y_test):
    """评估最终模型"""
    print("📊 评估最终模型...")
    
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"🎯 最终模型性能:")
    print(f"   准确率: {accuracy:.4f}")
    print(f"   AUC: {auc:.4f}")
    
    return {'accuracy': accuracy, 'auc': auc}

def save_model_for_web(model, scaler, final_features, performance, cfg):
    """保存模型用于Web应用"""
    print("💾 保存模型用于Web应用...")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 保存模型
    with open(os.path.join(MODEL_DIR, 'logistic_regression_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    # 保存标准化器
    with open(os.path.join(MODEL_DIR, 'feature_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # 保存模型元数据
    metadata = {
        'model_type': 'LogisticRegression',
        'features': final_features,
        'feature_count': len(final_features),
        'performance': performance,
        'thresholds': cfg['thresholds'],
        'model_params': model.get_params(),
        'feature_descriptions': {
            'Daily dose（g）': '日剂量（克）',
            'CLCR': '肌酐清除率',
            'GGT(U/L)': 'γ-谷氨酰转移酶',
            'Na(mmol/L)': '钠离子浓度',
            'HDL-C(mmol/L)': '高密度脂蛋白胆固醇',
            'ALB(g/L)': '白蛋白'
        },
        'clinical_ranges': {
            'Daily dose（g）': {'min': 0.1, 'max': 2.0, 'unit': 'g'},
            'CLCR': {'min': 30, 'max': 150, 'unit': 'mL/min'},
            'GGT(U/L)': {'min': 5, 'max': 200, 'unit': 'U/L'},
            'Na(mmol/L)': {'min': 135, 'max': 145, 'unit': 'mmol/L'},
            'HDL-C(mmol/L)': {'min': 0.8, 'max': 2.5, 'unit': 'mmol/L'},
            'ALB(g/L)': {'min': 35, 'max': 55, 'unit': 'g/L'}
        }
    }
    
    with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # 创建预测函数示例
    prediction_example = '''
# 使用保存的模型进行预测的示例代码
import pickle
import numpy as np
import pandas as pd

# 加载模型和标准化器
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 示例输入数据
sample_data = {
    'Daily dose（g）': 0.8,
    'CLCR': 80.0,
    'GGT(U/L)': 25.0,
    'Na(mmol/L)': 140.0,
    'HDL-C(mmol/L)': 1.2,
    'ALB(g/L)': 42.0
}

# 转换为DataFrame并标准化
X = pd.DataFrame([sample_data])
X_scaled = scaler.transform(X)

# 预测
probability = model.predict_proba(X_scaled)[0, 1]
prediction = model.predict(X_scaled)[0]

print(f"血药浓度偏高概率: {probability:.3f}")
print(f"预测结果: {'偏高' if prediction == 1 else '正常'}")
'''
    
    with open(os.path.join(MODEL_DIR, 'prediction_example.py'), 'w', encoding='utf-8') as f:
        f.write(prediction_example)
    
    print(f"✅ 模型已保存到: {MODEL_DIR}")
    print(f"📁 包含文件:")
    print(f"   - logistic_regression_model.pkl (模型文件)")
    print(f"   - feature_scaler.pkl (标准化器)")
    print(f"   - model_metadata.json (模型元数据)")
    print(f"   - prediction_example.py (使用示例)")

def main():
    """主函数"""
    print("🚀 开始保存最优模型...")
    
    # 加载配置
    cfg = load_config()
    
    # 数据预处理
    X_train, X_test, y_train, y_test, final_features = preprocess_data(cfg)
    
    # 训练最终模型
    model, scaler = train_final_model(X_train, y_train, final_features)
    
    # 评估模型
    performance = evaluate_final_model(model, scaler, X_test, y_test)
    
    # 保存模型
    save_model_for_web(model, scaler, final_features, performance, cfg)
    
    print("🎉 模型保存完成！可以开始创建Web应用了。")
    
    return model, scaler, final_features, performance

if __name__ == '__main__':
    main()