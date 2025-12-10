#!/usr/bin/env python3
"""
多說話人音訊列表準備工具。

支援模式：
1. 單說話人模式
2. 自動掃描模式
3. 手動指定多目錄模式
"""

import argparse
import os
from pathlib import Path
import sys
from typing import List, Tuple


def get_host_uid_gid() -> Tuple[int, int]:
    """
    偵測宿主機的 UID 與 GID。

    Returns:
        Tuple[int, int]: (uid, gid)
    """
    check_files = [
        'docker-compose.yml',
        'Dockerfile',
        'run.sh',
        'webui.py',
        'train.py',
    ]

    for filename in check_files:
        filepath = Path(filename)
        if filepath.exists():
            stat_info = filepath.stat()
            uid = stat_info.st_uid
            gid = stat_info.st_gid
            if uid != 0:
                return uid, gid

    return 1000, 1000


def fix_file_permissions(filepath: Path):
    """
    修復檔案權限以匹配宿主機使用者。
    """
    try:
        uid, gid = get_host_uid_gid()
        os.chown(filepath, uid, gid)
    except Exception as e:
        print(f"[警告] 無法修復檔案權限: {e}")


def has_audio_files(directory: Path) -> bool:
    """
    檢查目錄是否包含 .wav 檔案。
    """
    return len(list(directory.rglob("*.wav"))) > 0


def scan_speaker_dirs(base_dir: Path) -> List[Path]:
    """
    自動掃描資料目錄下的說話人子目錄。
    
    支援結構範例:
      1. data/speaker_001/*.wav
      2. data/drama1/speaker_001/*.wav
      3. data/drama1/character_id/episode/*.wav
    
    Args:
        base_dir (Path): 基礎資料目錄。
    
    Returns:
        List[Path]: 說話人目錄列表。
    """
    speaker_dirs = []
    
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        
        direct_audio = len(list(subdir.glob("*.wav"))) > 0
        
        if direct_audio:
            speaker_dirs.append(subdir)
        else:
            for character_dir in sorted(subdir.iterdir()):
                if not character_dir.is_dir():
                    continue
                
                has_direct_audio = len(list(character_dir.glob("*.wav"))) > 0
                
                if has_direct_audio:
                    speaker_dirs.append(character_dir)
                elif has_audio_files(character_dir):
                    speaker_dirs.append(character_dir)
    
    return speaker_dirs


def print_summary(results: List[dict], total_count: int, merged_file: str | None):
    """
    列印處理結果摘要。
    """
    if not results:
        return

    print("\n📊 統計資訊:")

    speaker_header = "說話人"
    count_header = "數量"
    file_header = "輸出檔案"

    max_name_len = max(len(result['speaker_id']) for result in results + [{'speaker_id': speaker_header}])
    max_count_len = max(len(str(result['success'])) for result in results) if results else len(count_header)
    max_count_len = max(max_count_len, len(str(total_count)), len(count_header))
    max_file_len = max(len(result.get('output_file') or "(未生成)") for result in results) if results else len(file_header)
    if merged_file:
        max_file_len = max(max_file_len, len(merged_file))
    max_file_len = max(max_file_len, len(file_header))

    header = f"   {speaker_header:<{max_name_len}}  {count_header:>{max_count_len}}  {file_header:<{max_file_len}}"
    separator = f"   {'-' * max_name_len}  {'-' * max_count_len}  {'-' * max_file_len}"
    print(header)
    print(separator)

    for result in results:
        output_path = result.get('output_file') or "(未生成)"
        print(f"   {result['speaker_id']:<{max_name_len}}  {result['success']:>{max_count_len}}  {output_path:<{max_file_len}}")

    total_label = "總計"
    total_file = merged_file if merged_file else "(無合併檔案)"
    print(f"   {total_label:<{max_name_len}}  {total_count:>{max_count_len}}  {total_file:<{max_file_len}}")


def prepare_single_speaker(
    data_dir: Path,
    text_suffix: str
) -> Tuple[List[str], int, int]:
    """
    處理單一說話人的資料。

    Returns:
        Tuple: (條目列表, 成功數量, 缺失數量)
    """
    audio_files = sorted(data_dir.rglob("*.wav"))

    entries = []
    missing_text = []

    for audio_file in audio_files:
        text_file = audio_file.with_suffix('').with_suffix(text_suffix)

        if not text_file.exists():
            missing_text.append(str(audio_file))
            continue

        try:
            text = text_file.read_text(encoding='utf-8').strip()
        except Exception as e:
            print(f"[警告] 讀取失敗: {text_file} - {e}")
            continue

        if not text:
            continue

        container_audio_path = str(audio_file.resolve())
        entries.append(f"{container_audio_path}\t{text}")

    return entries, len(entries), len(missing_text)


def main():
    parser = argparse.ArgumentParser(
        description="準備訓練用的 audio_list.txt 檔案",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:

1. 單說話人模式:
   python scripts/prepare_audio_list.py data/080

2. 自動掃描模式:
   python scripts/prepare_audio_list.py data/ --auto-scan

3. 多說話人模式:
   python scripts/prepare_audio_list.py data/080 data/081
        """
    )

    parser.add_argument('data_dirs', nargs='+', help='資料目錄路徑')
    parser.add_argument('-o', '--output', help='指定輸出檔名 (僅單說話人有效)')
    parser.add_argument('--auto-scan', action='store_true', help='自動掃描子目錄')
    parser.add_argument('--text-suffix', default='.normalized.txt', help='文字檔案後綴')
    parser.add_argument('--no-individual', action='store_true', help='不生成個別列表')
    parser.add_argument('--merge-all', action='store_true', help='生成合併列表')
    parser.add_argument('--speaker-id', help='手動指定說話人 ID')
    parser.add_argument('--split-size', type=int, default=0, help='分割大小 (0 表示不分割)')
    parser.add_argument('--output-dir', help='輸出目錄')

    args = parser.parse_args()

    speaker_dirs = []

    if len(args.data_dirs) == 1 and not args.auto_scan:
        single_dir = Path(args.data_dirs[0])

        if not single_dir.exists():
            print(f"[錯誤] 目錄不存在: {single_dir}")
            sys.exit(1)

        direct_audio = has_audio_files(single_dir)
        scanned_speakers = scan_speaker_dirs(single_dir)

        if direct_audio and not scanned_speakers:
            speaker_dirs = [single_dir]
        else:
            speaker_dirs = scanned_speakers
            if not speaker_dirs:
                print(f"[錯誤] 未找到音訊目錄")
                sys.exit(1)

            print(f"🔍 自動掃描發現 {len(speaker_dirs)} 個說話人")

    elif args.auto_scan:
        base_dir = Path(args.data_dirs[0])
        if not base_dir.exists():
            print(f"[錯誤] 目錄不存在: {base_dir}")
            sys.exit(1)

        speaker_dirs = scan_speaker_dirs(base_dir)
        if not speaker_dirs:
            print(f"[錯誤] 未找到音訊目錄")
            sys.exit(1)

        print(f"🔍 自動掃描發現 {len(speaker_dirs)} 個說話人")

    else:
        speaker_dirs = [Path(d) for d in args.data_dirs]
        for d in speaker_dirs:
            if not d.exists():
                print(f"[錯誤] 目錄不存在: {d}")
                sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else Path("finetune_data/audio_list")
    output_dir.mkdir(parents=True, exist_ok=True)
    fix_file_permissions(output_dir)

    results = []
    total_count = 0
    all_entries = []

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name
        parent_dir = speaker_dir.parent.name
        generic_names = ['data', 'audio', 'dataset', 'train', 'finetune', 'wav', 'wavs']

        if parent_dir.lower() not in generic_names:
            speaker_id = f"{parent_dir}_{speaker_id}"

        print(f"📦 處理說話人: {speaker_id}")

        entries, success_count, missing_count = prepare_single_speaker(
            speaker_dir,
            args.text_suffix
        )

        if not entries:
            print(f"   [警告] 無有效音訊檔案\n")
            continue

        results.append({
            'speaker_id': speaker_id,
            'entries': entries,
            'success': success_count,
            'missing': missing_count,
            'output_file': None
        })

        all_entries.extend(entries)
        total_count += success_count

        print(f"   ✅ 成功: {success_count} 個")
        if missing_count > 0:
            print(f"   [警告] 缺失文字: {missing_count} 個")
        print()

    if not results:
        print("[錯誤] 無有效資料")
        sys.exit(1)

    merged_file_path: str | None = None

    if len(results) == 1:
        speaker_id = args.speaker_id if args.speaker_id else results[0]['speaker_id']

        if args.output:
            output_file = Path(args.output)
        else:
            output_file = output_dir / f"{speaker_id}.txt"

        results[0]['output_file'] = str(output_file)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(results[0]['entries']))

        fix_file_permissions(output_file)
        print(f"📄 輸出檔案: {output_file}")

    else:
        if not args.no_individual:
            for result in results:
                speaker_id = result['speaker_id']
                speaker_file = output_dir / f"{speaker_id}.txt"

                with open(speaker_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(result['entries']))

                fix_file_permissions(speaker_file)
                result['output_file'] = str(speaker_file)
                print(f"📄 {speaker_id}: {speaker_file} ({result['success']} 個)")
        else:
            print("ℹ 已跳過個別說話人列表")

        if args.merge_all:
            merged_file = output_dir / "audio_list_all.txt"

            if args.split_size > 0 and len(all_entries) > args.split_size:
                num_parts = (len(all_entries) + args.split_size - 1) // args.split_size
                print(f"\n📦 自動分割: {len(all_entries)} 條 → {num_parts} 個部分")

                for i in range(num_parts):
                    start_idx = i * args.split_size
                    end_idx = min((i + 1) * args.split_size, len(all_entries))
                    part_entries = all_entries[start_idx:end_idx]

                    part_file = output_dir / f"audio_list_part_{i}.txt"
                    with open(part_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(part_entries))

                    fix_file_permissions(part_file)
                    print(f"   📄 Part {i}: {part_file}")

                print(f"\n✅ 分割完成")
            else:
                with open(merged_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(all_entries))

                fix_file_permissions(merged_file)
                merged_file_path = str(merged_file)
                print(f"\n📄 合併檔案: {merged_file}")

    print_summary(results, total_count, merged_file_path)


if __name__ == '__main__':
    main()
