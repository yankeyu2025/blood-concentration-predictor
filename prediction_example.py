
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
