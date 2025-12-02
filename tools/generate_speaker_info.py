#!/usr/bin/env python3
"""
從所有處理完的 metadata 生成 speaker_info.json
在所有 part 的 extract_codec.py 處理完後執行此腳本
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import glob
from loguru import logger

def generate_speaker_info(processed_data_dir: str):
    """從所有 metadata 生成 speaker_info.json"""

    # 找出所有 processed_data 目錄
    processed_dirs = glob.glob(os.path.join(processed_data_dir, 'audio_list_part_*'))
    logger.info(f"找到 {len(processed_dirs)} 個 part 目錄")

    if not processed_dirs:
        logger.error(f"在 {processed_data_dir} 找不到任何 audio_list_part_* 目錄")
        return False

    # 收集所有 speaker 資訊
    speaker_data = defaultdict(list)
    total_samples = 0

    for proc_dir in sorted(processed_dirs):
        metadata_file = os.path.join(proc_dir, 'metadata.jsonl')
        if not os.path.exists(metadata_file):
            logger.warning(f"{proc_dir} 中找不到 metadata.jsonl，跳過")
            continue

        part_name = os.path.basename(proc_dir)
        logger.info(f"處理 {part_name}...")

        # 讀取 metadata 並按 speaker 分組
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                audio_path = data["audio"]
                path_parts = Path(audio_path).parts

                # 提取 speaker ID: drama_name + character_id
                # 路徑格式: /workspace/index-tts-lora/data/drama1/076/008/xxx.wav
                data_idx = path_parts.index("data") if "data" in path_parts else -1
                if data_idx >= 0 and len(path_parts) > data_idx + 2:
                    drama_name = path_parts[data_idx + 1]
                    character_id = path_parts[data_idx + 2]
                    speaker_id = f"{drama_name}_{character_id}"
                else:
                    speaker_id = part_name

                speaker_data[speaker_id].append(data)
                total_samples += 1

    logger.info(f"總共讀取 {total_samples} 個樣本")
    logger.info(f"識別出 {len(speaker_data)} 個不同的 speaker")

    # 生成 speaker_info
    speaker_info_list = []
    for speaker_id in sorted(speaker_data.keys()):
        lines = speaker_data[speaker_id]
        total_duration = sum(line["duration"] for line in lines)

        # 使用該 speaker 第一筆資料的目錄作為參考
        ref_dir = os.path.dirname(lines[0]["codes"])

        speaker_info = {
            "speaker": speaker_id,
            "avg_duration": total_duration / len(lines),
            "sample_num": len(lines),
            "total_duration": total_duration,
            "train_jsonl": os.path.abspath(os.path.join(ref_dir, "metadata_train.jsonl")),
            "valid_jsonl": os.path.abspath(os.path.join(ref_dir, "metadata_valid.jsonl")),
            "medoid_condition": os.path.abspath(os.path.join(ref_dir, "medoid_condition.npy")),
        }
        speaker_info_list.append(speaker_info)

    # 儲存到 processed_data 目錄下
    output_file = os.path.join(processed_data_dir, 'speaker_info.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(speaker_info_list, f, ensure_ascii=False, indent=4)

    logger.info(f"✅ speaker_info.json 生成完成: {output_file}")
    logger.info(f"   - 總 speaker 數: {len(speaker_info_list)}")
    logger.info(f"   - 總樣本數: {sum(s['sample_num'] for s in speaker_info_list)}")

    # 統計每個 drama 的 speaker 數量
    drama_stats = defaultdict(int)
    for speaker_id in speaker_data.keys():
        drama_name = speaker_id.split('_')[0]
        drama_stats[drama_name] += 1

    for drama_name, count in sorted(drama_stats.items()):
        logger.info(f"   - {drama_name}: {count} 個 speaker")

    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        processed_data_dir = sys.argv[1]
    else:
        processed_data_dir = "finetune_data/processed_data"

    logger.info(f"從 {processed_data_dir} 生成 speaker_info.json")
    success = generate_speaker_info(processed_data_dir)
    sys.exit(0 if success else 1)
