#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def retrain_model_for_low_concentration():
    """重新训练模型，专门识别血药浓度是否小于0.5"""
    
    print("🔄 开始重新训练模型，识别血药浓度 < 0.5...")
    
    # 1. 加载训练数据
    try:
        # 根据配置文件中的路径查找训练数据
        possible_paths = [
            'train_set_concentration.csv',
            '../train_set_concentration.csv',
            os.path.join('..', 'train_set_concentration.csv'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'train_set_concentration.csv')
        ]
        
        # 尝试不同的编码方式
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb18030', 'cp936', 'latin-1']
        train_data = None
        
        for path in possible_paths:
            if os.path.exists(path):
                for encoding in encodings:
                    try:
                        train_data = pd.read_csv(path, encoding=encoding)
                        print(f"✅ 训练数据加载成功 (路径: {path}, 编码: {encoding})，共 {len(train_data)} 条记录")
                        break
                    except UnicodeDecodeError:
                        continue
                if train_data is not None:
                    break
        
        if train_data is None:
            print("❌ 无法找到或读取训练数据文件")
            print("尝试的路径:")
            for path in possible_paths:
                print(f"  - {path} (存在: {os.path.exists(path)})")
            return
            
    except Exception as e:
        print(f"❌ 训练数据加载失败: {e}")
        return
    
    # 2. 数据预处理
    # 选择特征（与当前Web应用一致）
    feature_columns = [
        'Daily dose（g）',
        'CLCR', 
        'GGT(U/L)',
        'Na(mmol/L)',
        'HDL-C(mmol/L)',
        'ALB(g/L)'
    ]
    
    # 目标列
    target_column = 'concentration（ng/ml）'
    
    # 检查数据
    print(f"\n📊 数据概览:")
    print(f"特征列: {feature_columns}")
    print(f"目标列: {target_column}")
    
    # 检查缺失值
    missing_features = []
    for col in feature_columns:
        if col not in train_data.columns:
            missing_features.append(col)
    
    if missing_features:
        print(f"❌ 缺失特征列: {missing_features}")
        print("可用列:", list(train_data.columns))
        return
    
    if target_column not in train_data.columns:
        print(f"❌ 缺失目标列: {target_column}")
        print("可用列:", list(train_data.columns))
        return
    
    # 提取特征和目标
    X = train_data[feature_columns].copy()
    y_continuous = train_data[target_column].copy()
    
    print(f"\n📈 血药浓度分布统计:")
    print(f"最小值: {y_continuous.min():.3f}")
    print(f"最大值: {y_continuous.max():.3f}")
    print(f"平均值: {y_continuous.mean():.3f}")
    print(f"中位数: {y_continuous.median():.3f}")
    
    # 3. 重新定义标签
    # 新的标签定义：0(正常) = 血药浓度 >= 0.5, 1(异常) = 血药浓度 < 0.5
    y_binary = (y_continuous < 0.5).astype(int)
    
    print(f"\n🏷️  新标签分布:")
    print(f"异常 (< 0.5): {sum(y_binary == 1)} 条 ({sum(y_binary == 1)/len(y_binary)*100:.1f}%)")
    print(f"正常 (≥ 0.5): {sum(y_binary == 0)} 条 ({sum(y_binary == 0)/len(y_binary)*100:.1f}%)")
    
    # 检查数据完整性
    print(f"\n🔍 数据完整性检查:")
    print(f"特征缺失值: {X.isnull().sum().sum()}")
    print(f"目标缺失值: {y_binary.isnull().sum()}")
    
    # 删除缺失值
    mask = ~(X.isnull().any(axis=1) | y_binary.isnull())
    X_clean = X[mask]
    y_clean = y_binary[mask]
    
    print(f"清理后数据: {len(X_clean)} 条记录")
    
    # 4. 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )
    
    print(f"\n📊 数据分割:")
    print(f"训练集: {len(X_train)} 条")
    print(f"测试集: {len(X_test)} 条")
    
    # 5. 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ 特征标准化完成")
    
    # 6. 训练模型
    print(f"\n🤖 开始训练逻辑回归模型...")
    
    model = LogisticRegression(
        max_iter=1000,
        solver='liblinear',
        random_state=42,
        class_weight='balanced'  # 处理类别不平衡
    )
    
    model.fit(X_train_scaled, y_train)
    print("✅ 模型训练完成")
    
    # 7. 模型评估
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n📊 模型性能评估:")
    print(f"准确率: {accuracy:.3f}")
    print(f"精确率: {precision:.3f}")
    print(f"召回率: {recall:.3f}")
    print(f"F1分数: {f1:.3f}")
    print(f"AUC值: {auc:.3f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n混淆矩阵:")
    print(f"真实正常/预测正常: {cm[0,0]}")
    print(f"真实正常/预测异常: {cm[0,1]}")
    print(f"真实异常/预测正常: {cm[1,0]}")
    print(f"真实异常/预测异常: {cm[1,1]}")
    
    # 8. 保存模型
    print(f"\n💾 保存新模型...")
    
    # 保存模型
    with open('web_models/logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # 保存标准化器
    with open('web_models/feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # 更新元数据
    metadata = {
        "model_type": "LogisticRegression",
        "features": feature_columns,
        "feature_count": len(feature_columns),
        "thresholds": {
            "binary": 0.5
        },
        "label_definition": {
            "0": "正常 (血药浓度 ≥ 0.5)",
            "1": "异常 (血药浓度 < 0.5)"
        },
        "feature_descriptions": {
            "Daily dose（g）": "日剂量（克）",
            "CLCR": "肌酐清除率",
            "GGT(U/L)": "γ-谷氨酰转移酶",
            "Na(mmol/L)": "钠离子浓度",
            "HDL-C(mmol/L)": "高密度脂蛋白胆固醇",
            "ALB(g/L)": "白蛋白"
        },
        "clinical_ranges": {
            "Daily dose（g）": {"min": 0.1, "max": 2.0, "unit": "g"},
            "CLCR": {"min": 30, "max": 150, "unit": "mL/min"},
            "GGT(U/L)": {"min": 5, "max": 200, "unit": "U/L"},
            "Na(mmol/L)": {"min": 135, "max": 145, "unit": "mmol/L"},
            "HDL-C(mmol/L)": {"min": 0.8, "max": 2.5, "unit": "mmol/L"},
            "ALB(g/L)": {"min": 35, "max": 55, "unit": "g/L"}
        },
        "performance": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc": float(auc)
        },
        "training_info": {
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "class_distribution": {
                "normal": int(sum(y_clean == 0)),
                "abnormal": int(sum(y_clean == 1))
            }
        }
    }
    
    with open('web_models/model_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print("✅ 模型和元数据保存完成")
    
    # 9. 测试新模型
    print(f"\n🧪 测试新模型...")
    
    test_cases = [
        {
            'name': '极低血药浓度测试',
            'data': [0.1, 120.0, 15.0, 142.0, 2.0, 50.0],
            'expected': '异常 (< 0.5)'
        },
        {
            'name': '正常血药浓度测试',
            'data': [0.8, 80.0, 25.0, 140.0, 1.2, 42.0],
            'expected': '正常 (≥ 0.5)'
        },
        {
            'name': '高剂量测试',
            'data': [2.0, 40.0, 60.0, 135.0, 0.8, 35.0],
            'expected': '正常 (≥ 0.5)'
        }
    ]
    
    for test_case in test_cases:
        test_input = np.array([test_case['data']])
        test_scaled = scaler.transform(test_input)
        
        prediction = model.predict(test_scaled)[0]
        probability = model.predict_proba(test_scaled)[0]
        
        result = "异常 (< 0.5)" if prediction == 1 else "正常 (≥ 0.5)"
        
        print(f"\n{test_case['name']}:")
        print(f"  输入: {test_case['data']}")
        print(f"  预测: {result}")
        print(f"  期望: {test_case['expected']}")
        print(f"  概率: [正常: {probability[0]:.3f}, 异常: {probability[1]:.3f}]")
        print(f"  ✅ 正确" if result == test_case['expected'] else f"  ❌ 错误")
    
    print(f"\n🎉 模型重新训练完成！")
    print(f"新模型专门识别血药浓度 < 0.5 为异常")
    print(f"模型性能: 准确率 {accuracy:.3f}, AUC {auc:.3f}")

if __name__ == "__main__":
    retrain_model_for_low_concentration()