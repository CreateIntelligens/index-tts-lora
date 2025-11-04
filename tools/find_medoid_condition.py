#!/usr/bin/env python3
"""
找出 condition latent 分佈中心的樣本（medoid）
即距離所有其他樣本距離之和最小的樣本

用法示例：
    python tools/find_medoid_condition.py \
        --input M00100_condition/output.jsonl \
        --output medoid_info.json
"""
import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

# --- 解析命令列 ---
parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help='輸入的 jsonl 檔案，包含 condition 欄位')
parser.add_argument('--output', default='medoid_info.json', help='輸出 medoid 資訊檔案')
parser.add_argument('--distance_metric', default='euclidean', choices=['euclidean', 'cosine'], help='距離度量方式')
args = parser.parse_args()

# 轉換為絕對路徑
args.input = os.path.abspath(args.input)
args.output = os.path.abspath(args.output)

print(f"讀取輸入檔案: {args.input}")
print(f"距離度量: {args.distance_metric}")

# --- 讀取所有 condition latent ---
conditions = []
condition_infos = []  # 儲存每個樣本的資訊
count = 0

with open(args.input, 'r', encoding='utf-8') as f:
    for line_idx, line in enumerate(f):
        item = json.loads(line)
        if 'condition' not in item:
            print(f"警告: 第 {line_idx+1} 行缺少 condition 欄位，跳過")
            continue
        
        condition_path = item['condition']
        if not os.path.exists(condition_path):
            print(f"警告: condition 檔案不存在: {condition_path}，跳過")
            continue
        
        # 載入 condition latent
        cond = np.load(condition_path)  # (32, dim)
        conditions.append(cond)
        condition_infos.append({
            'index': count,
            'line_number': line_idx + 1,
            'condition_path': condition_path,
            'original_item': item
        })
        count += 1
        
        if count % 100 == 0:
            print(f"已讀取 {count} 個樣本")

if len(conditions) == 0:
    raise RuntimeError('沒有找到有效的 condition 檔案')

print(f"總共讀取 {len(conditions)} 個 condition latent")

# 將所有 condition 堆疊成一個數組
conditions_array = np.stack(conditions, axis=0)  # (N, 32, dim)
print(f"Conditions 陣列形狀: {conditions_array.shape}")

# 將資料展平以便計算距離
N, H, W = conditions_array.shape
data_flat = conditions_array.reshape(N, H * W)  # (N, 32*dim)

# --- 計算距離矩陣 ---
print("計算距離矩陣...")

def compute_distance_matrix(data, metric='euclidean'):
    """計算距離矩陣"""
    n = data.shape[0]
    dist_matrix = np.zeros((n, n))
    
    for i in tqdm(range(n), desc="計算距離"):
        for j in range(i+1, n):
            if metric == 'euclidean':
                dist = np.linalg.norm(data[i] - data[j])
            elif metric == 'cosine':
                # 餘弦距離 = 1 - 餘弦相似度
                cos_sim = np.dot(data[i], data[j]) / (np.linalg.norm(data[i]) * np.linalg.norm(data[j]))
                dist = 1 - cos_sim
            else:
                raise ValueError(f"不支援的距離度量: {metric}")
            
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # 對稱矩陣
    
    return dist_matrix

dist_matrix = compute_distance_matrix(data_flat, args.distance_metric)

# --- 找出 medoid ---
print("尋找 medoid...")

# 計算每個樣本到所有其他樣本的距離之和
dist_sums = np.sum(dist_matrix, axis=1)

# 找出距離之和最小的樣本索引
medoid_idx = np.argmin(dist_sums)
medoid_info = condition_infos[medoid_idx]
medoid_condition = conditions[medoid_idx]

print(f"找到 medoid: 索引 {medoid_idx}")
print(f"Medoid 資訊: {medoid_info}")
print(f"Medoid 到所有樣本的距離之和: {dist_sums[medoid_idx]:.6f}")

# --- 計算一些統計資訊 ---
stats = {
    'total_samples': len(conditions),
    'medoid_index': int(medoid_idx),
    'medoid_distance_sum': float(dist_sums[medoid_idx]),
    'distance_metric': args.distance_metric,
    'distance_stats': {
        'mean': float(np.mean(dist_sums)),
        'std': float(np.std(dist_sums)),
        'min': float(np.min(dist_sums)),
        'max': float(np.max(dist_sums))
    }
}

# --- 儲存結果 ---
result = {
    'medoid_info': medoid_info,
    'statistics': stats,
    'medoid_condition_shape': medoid_condition.shape
}

# 儲存 medoid 資訊
with open(args.output, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# 儲存 medoid condition
medoid_condition_path = os.path.splitext(args.output)[0] + '_medoid_condition.npy'
np.save(medoid_condition_path, medoid_condition)

# 儲存距離矩陣（可選，用於進一步分析）
dist_matrix_path = os.path.splitext(args.output)[0] + '_distance_matrix.npy'
np.save(dist_matrix_path, dist_matrix)

print(f"\n結果儲存完成:")
print(f"Medoid 資訊: {args.output}")
print(f"Medoid condition: {medoid_condition_path}")
print(f"距離矩陣: {dist_matrix_path}")
print(f"\nMedoid 樣本詳情:")
print(f"  - 原始行號: {medoid_info['line_number']}")
print(f"  - Condition 檔案: {medoid_info['condition_path']}")
if 'audio' in medoid_info['original_item']:
    print(f"  - 音訊檔案: {medoid_info['original_item']['audio']}")
if 'text' in medoid_info['original_item']:
    print(f"  - 文字: {medoid_info['original_item']['text']}")