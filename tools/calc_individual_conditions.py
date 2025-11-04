#!/usr/bin/env python3
"""
讀取 jsonl 清單中的 mel 譜檔案，透過 UnifiedVoice 的 `get_conditioning` 計算每條樣本的
condition embedding，並將每個樣本的 condition 儲存為單獨的 .npy 檔案，
同時更新 jsonl 檔案新增 condition 欄位。

用法示例：
    python tools/calc_individual_conditions.py \
        --manifest finetune_data/M00100_train.jsonl \
        --config finetune_models/config.yaml \
        --output_dir finetune_data/conditions \
        --output_jsonl finetune_data/M00100_train_with_conditions.jsonl
"""
import argparse
import json
import os
import sys
from typing import List

import numpy as np
import torch
from omegaconf import OmegaConf

# --- 解析命令列 ---
parser = argparse.ArgumentParser()
parser.add_argument('--manifest', "-m", required=True, help='jsonl 資料清單')
parser.add_argument('--config', "-c", required=True, help='finetune_models/config.yaml')
parser.add_argument('--output_dir', "-o", default='conditions', help='condition 檔案輸出目錄')
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

# 建立輸出目錄
os.makedirs(args.output_dir, exist_ok=True)

# --- 載入配置並構建模型 ---
cfg = OmegaConf.load(args.config)
ckpt_path = "checkpoints/gpt.pth.open_source"

sys.path.append('.')  # 方便指令碼在專案根目錄外執行
from indextts.gpt.model import UnifiedVoice  # noqa: E402

uv = UnifiedVoice(**cfg.gpt)

print(f"loading checkpoint from {ckpt_path}")
state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
uv.load_state_dict(state['model'] if 'model' in state else state, strict=True)
uv.eval().to(args.device)

# --- 遍歷樣本，計算並儲存每個 condition embedding ---
count = 0
updated_items = []

with open(args.manifest, 'r', encoding='utf-8') as f:
    for line_idx, line in enumerate(f):
        item = json.loads(line)
        
        # 讀取 mel 檔案
        mel_np: np.ndarray = np.load(item['mels'])  # (1, 100, T)
        if mel_np.ndim == 3 and mel_np.shape[0] == 1:  # (1, 100, T)
            mel_np = mel_np.squeeze(0)
        else:
            raise ValueError(f"未知的 mel 形狀: {mel_np.shape}")

        mel = torch.from_numpy(mel_np).unsqueeze(0).to(args.device)  # (1, 100, T)
        length = torch.tensor([mel_np.shape[0]], device=args.device)
        
        # 計算 condition embedding
        with torch.no_grad():
            cond = uv.get_conditioning(mel, length)  # (1, 32, dim)
            cond = cond.squeeze(0).cpu().float().numpy()  # (32, dim)
        
        # 生成 condition 檔名
        # 使用原始音訊檔名作為基礎，或者使用行號
        if 'audio' in item:
            base_name = os.path.splitext(os.path.basename(item['audio']))[0]
        else:
            base_name = f"sample_{line_idx:06d}"
        
        condition_filename = f"{base_name}_condition.npy"
        condition_path = os.path.join(args.output_dir, condition_filename)
        
        # 儲存 condition embedding
        np.save(condition_path, cond)
        
        # 更新 item，新增 condition 欄位
        updated_item = item.copy()
        updated_item['condition'] = os.path.abspath(condition_path)
        updated_items.append(updated_item)
        
        count += 1
        if count % 100 == 0:
            print(f"processed {count} samples", file=sys.stderr)

if count == 0:
    raise RuntimeError('manifest 為空或讀取失敗')

# 儲存更新後的 jsonl 檔案
with open(os.path.join(args.output_dir, "output.jsonl"), 'w', encoding='utf-8') as f:
    for item in updated_items:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"處理完成！共處理 {count} 個樣本")
print(f"condition 檔案儲存到: {args.output_dir}")