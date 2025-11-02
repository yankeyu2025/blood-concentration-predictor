import pickle
import numpy as np

# 定义selected_features的医学常识原始值范围
original_ranges = {
    'CLCR': (10, 160),           # 肌酐清除率 mL/min，从严重肾功能不全到正常上限
    'Daily dose(g)': (0.1, 5.0), # 日剂量，常见范围
    'ALP(U/L)': (20, 200),       # 碱性磷酸酶，从低到高
    'GGT(U/L)': (5, 300),        # γ-谷氨酰转移酶，从正常到严重升高
    'Na(mmol/L)': (120, 160),    # 钠离子，从低钠到高钠
    'TBA(umol/L)': (0, 50)       # 总胆汁酸，从正常到严重升高
}

# 从训练数据中获取的归一化范围（基于analyze_original_data.py的结果）
normalized_ranges = {
    'CLCR': (0.0611, 0.9389),
    'Daily dose(g)': (0.1, 1.0),
    'ALP(U/L)': (0.0515, 0.9485),
    'GGT(U/L)': (0.0263, 0.9737),
    'Na(mmol/L)': (0.0, 1.0),
    'TBA(umol/L)': (0.0645, 0.9355)
}

class InputNormalizer:
    def __init__(self, original_ranges, normalized_ranges):
        self.original_ranges = original_ranges
        self.normalized_ranges = normalized_ranges
    
    def normalize_input(self, raw_input):
        """
        将原始医学指标值线性映射到训练数据使用的归一化范围
        """
        normalized = {}
        
        for feature, raw_value in raw_input.items():
            if feature in self.original_ranges and feature in self.normalized_ranges:
                # 获取原始范围和归一化范围
                orig_min, orig_max = self.original_ranges[feature]
                norm_min, norm_max = self.normalized_ranges[feature]
                
                # 线性映射：从原始范围映射到归一化范围
                # normalized_value = norm_min + (raw_value - orig_min) * (norm_max - norm_min) / (orig_max - orig_min)
                normalized_value = norm_min + (raw_value - orig_min) * (norm_max - norm_min) / (orig_max - orig_min)
                
                # 确保在归一化范围内
                normalized_value = max(norm_min, min(norm_max, normalized_value))
                
                normalized[feature] = normalized_value
            else:
                print(f"Warning: Feature {feature} not found in ranges")
                normalized[feature] = raw_value
        
        return normalized

# 创建输入标准化器实例
input_normalizer = InputNormalizer(original_ranges, normalized_ranges)

print("创建输入归一化映射:")
print("=" * 50)

# 测试输入归一化
test_input = {
    'CLCR': 120,
    'Daily dose(g)': 1.0,
    'ALP(U/L)': 70,
    'GGT(U/L)': 25,
    'Na(mmol/L)': 140,
    'TBA(umol/L)': 3
}

print("测试输入归一化:")
normalized_test = input_normalizer.normalize_input(test_input)
for feature, value in normalized_test.items():
    print(f"{feature}: {test_input[feature]} -> {value:.6f}")

# 保存输入归一化器
with open('web_models/input_normalizer.pkl', 'wb') as f:
    pickle.dump(input_normalizer, f)

print(f"\n输入归一化器已保存到 web_models/input_normalizer.pkl")