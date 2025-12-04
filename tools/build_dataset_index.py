#!/usr/bin/env python3
"""
快速建立 dataset 索引檔案，避免每次訓練都重新掃描所有 metadata。

使用方式：
    python tools/build_dataset_index.py --data_path finetune_data/processed_data

會生成：
    finetune_data/processed_data/train_index.pkl
    finetune_data/processed_data/valid_index.pkl
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def scan_manifest_batch(tasks):
    """批次掃描多個 manifest 檔案（減少進程啟動開銷）。

    Args:
        tasks: [(manifest_file, speaker_id), ...]

    Returns:
        list: [(manifest_file, offset, speaker_id), ...]
    """
    all_items = []

    for manifest_file, speaker_id in tasks:
        try:
            # 快速掃描：只記錄行號，不用 tell() (更快)
            with open(manifest_file, 'rb') as f:  # 用 binary mode 更快
                offset = 0
                for line in f:
                    if line.strip():
                        all_items.append((manifest_file, offset, speaker_id))
                    offset = f.tell()
        except Exception as e:
            # 靜默處理錯誤，避免打斷進度條
            pass

    return all_items


def build_index(data_path, manifest_type='train'):
    """建立索引檔案。

    Args:
        data_path: processed_data 目錄
        manifest_type: 'train' 或 'valid'
    """
    speaker_info_path = os.path.join(data_path, 'speaker_info.json')

    print(f"載入 speaker_info.json...")
    with open(speaker_info_path, 'r', encoding='utf-8') as f:
        speaker_info_list = json.load(f)

    print(f"開始掃描 {len(speaker_info_list)} 個 speaker 的 {manifest_type} metadata...")

    # 收集所有任務
    tasks = []
    for speaker_info in speaker_info_list:
        speaker_id = speaker_info['speaker']
        manifest_file = speaker_info[f'{manifest_type}_jsonl']

        if not os.path.exists(manifest_file):
            print(f"Warning: {manifest_file} not found, skipping")
            continue

        tasks.append((manifest_file, speaker_id))

    # 批次多線程掃描（每個線程處理多個檔案）
    all_items = []
    max_workers = min(32, len(tasks) // 100 + 1)  # 減少線程數
    batch_size = len(tasks) // max_workers + 1  # 每個線程處理的檔案數

    # 分批
    batches = [tasks[i:i+batch_size] for i in range(0, len(tasks), batch_size)]

    print(f"使用 {len(batches)} 個線程，每個處理約 {batch_size} 個檔案...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(scan_manifest_batch, batch) for batch in batches]

        for future in tqdm(as_completed(futures), total=len(futures),
                          desc=f"掃描 {manifest_type} metadata",
                          unit="batch", ncols=100):
            items = future.result()
            all_items.extend(items)

    print(f"✓ 掃描完成，共 {len(all_items):,} 個樣本")

    # 儲存索引
    index_file = os.path.join(data_path, f'{manifest_type}_index.pkl')
    with open(index_file, 'wb') as f:
        pickle.dump(all_items, f)

    print(f"✓ 索引已儲存至: {index_file}")
    print(f"   大小: {os.path.getsize(index_file) / 1024 / 1024:.2f} MB")

    return index_file


def main():
    parser = argparse.ArgumentParser(description='建立 dataset 索引檔案')
    parser.add_argument('--data_path', type=str, default='finetune_data/processed_data',
                       help='processed_data 目錄路徑')
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: {args.data_path} 不存在")
        return

    # 建立 train 和 valid 索引
    build_index(args.data_path, 'train')
    build_index(args.data_path, 'valid')

    print("\n✓ 全部完成！現在可以開始訓練了。")


if __name__ == '__main__':
    main()
