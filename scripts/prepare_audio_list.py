#!/usr/bin/env python3
"""
多說話人音頻列表準備工具
支援單說話人、多說話人和自動掃描模式
"""

import argparse
import os
from pathlib import Path
import sys
from typing import List, Tuple


def get_host_uid_gid() -> Tuple[int, int]:
    """
    從掛載的文件檢測宿主機的 UID 和 GID

    Returns:
        (uid, gid) 元組
    """
    # 嘗試從常見的掛載文件獲取 UID/GID
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
            # 如果不是 root，使用這個
            if uid != 0:
                return uid, gid

    # 默認使用 1000:1000
    return 1000, 1000


def fix_file_permissions(filepath: Path):
    """修復文件權限為宿主機用戶"""
    try:
        uid, gid = get_host_uid_gid()
        os.chown(filepath, uid, gid)
    except Exception as e:
        # 權限修復失敗不影響功能，只是警告
        print(f"⚠️  無法修復文件權限: {e}")


def has_audio_files(directory: Path) -> bool:
    """檢查目錄是否直接包含音頻文件"""
    return len(list(directory.rglob("*.wav"))) > 0


def scan_speaker_dirs(base_dir: Path) -> List[Path]:
    """
    自動掃描數據目錄下的所有說話人子目錄

    Args:
        base_dir: 基礎數據目錄 (如 data/)

    Returns:
        說話人目錄列表
    """
    # 只掃描直接子目錄
    speaker_dirs = []
    for subdir in sorted(base_dir.iterdir()):
        if subdir.is_dir() and has_audio_files(subdir):
            speaker_dirs.append(subdir)

    return speaker_dirs


def print_summary(results: List[dict], total_count: int, merged_file: str | None):
    """列印最終統計摘要"""
    if not results:
        return

    print("\n📊 統計信息:")

    speaker_header = "說話人"
    count_header = "數量"
    file_header = "輸出文件"

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
    total_file = merged_file if merged_file else "(無合併文件)"
    print(f"   {total_label:<{max_name_len}}  {total_count:>{max_count_len}}  {total_file:<{max_file_len}}")


def prepare_single_speaker(
    data_dir: Path,
    text_suffix: str
) -> Tuple[List[str], int, int]:
    """
    處理單個說話人的音頻列表

    Returns:
        (entries, success_count, missing_count)
    """
    # 查找所有音頻文件
    audio_files = sorted(data_dir.rglob("*.wav"))

    entries = []
    missing_text = []

    for audio_file in audio_files:
        # 查找對應的文本文件
        text_file = audio_file.with_suffix('').with_suffix(text_suffix)

        if not text_file.exists():
            missing_text.append(str(audio_file))
            continue

        # 讀取文本內容
        try:
            text = text_file.read_text(encoding='utf-8').strip()
        except Exception as e:
            print(f"⚠️  讀取失敗: {text_file} - {e}")
            continue

        if not text:
            continue

        # 直接使用容器內的絕對路徑
        container_audio_path = str(audio_file.resolve())

        # 格式: 路徑<TAB>文本
        entries.append(f"{container_audio_path}\t{text}")

    return entries, len(entries), len(missing_text)


def main():
    parser = argparse.ArgumentParser(
        description="準備訓練用的 audio_list.txt 文件 (支援多說話人)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:

1. 單說話人模式:
   python scripts/prepare_audio_list.py data/080
   # 生成: finetune_data/audio_list_080.txt

2. 自動掃描模式 (智能判斷):
   python scripts/prepare_audio_list.py data/
   # 如果 data/ 下有子目錄包含音頻，自動掃描所有說話人
   # 生成: audio_list_080.txt, audio_list_081.txt, ..., audio_list_all.txt

3. 多說話人模式 (手動指定):
   python scripts/prepare_audio_list.py data/080 data/081 data/082
   # 生成: finetune_data/audio_list_080.txt (單獨)
   #       finetune_data/audio_list_081.txt (單獨)
   #       finetune_data/audio_list_082.txt (單獨)
   #       finetune_data/audio_list_all.txt (合併)

4. 自定義輸出 (僅單說話人):
   python scripts/prepare_audio_list.py data/080 -o custom_list.txt
        """
    )

    parser.add_argument(
        'data_dirs',
        nargs='+',
        help='一個或多個數據目錄路徑 (例如: data/080 data/081)'
    )

    parser.add_argument(
        '-o', '--output',
        help='指定輸出文件名 (僅單說話人時有效，多說話人會自動命名)'
    )

    parser.add_argument(
        '--auto-scan',
        action='store_true',
        help='自動掃描指定目錄下的所有說話人子目錄'
    )

    parser.add_argument(
        '--text-suffix',
        default='.normalized.txt',
        help='文本文件後綴 (默認: .normalized.txt)'
    )

    parser.add_argument(
        '--no-individual',
        action='store_true',
        help='多說話人模式下不生成單獨的 audio_list 文件'
    )

    parser.add_argument(
        '--merge-all',
        action='store_true',
        help='多說話人模式下生成合併的 audio_list_all.txt'
    )

    parser.add_argument(
        '--speaker-id',
        help='手動指定說話人 ID (僅單說話人時有效，覆蓋自動推斷)'
    )

    args = parser.parse_args()

    # 確定要處理的說話人目錄
    speaker_dirs = []

    if len(args.data_dirs) == 1 and not args.auto_scan:
        # 單個目錄：自動判斷是單說話人還是需要掃描
        single_dir = Path(args.data_dirs[0])

        if not single_dir.exists():
            print(f"❌ 錯誤: 目錄不存在: {single_dir}")
            sys.exit(1)

        direct_audio = has_audio_files(single_dir)
        scanned_speakers = scan_speaker_dirs(single_dir)

        if direct_audio and not scanned_speakers:
            # 單說話人模式（目錄本身直接包含音頻）
            speaker_dirs = [single_dir]
        else:
            # 自動掃描子目錄
            speaker_dirs = scanned_speakers
            if not speaker_dirs:
                print(f"❌ 錯誤: 在 {single_dir} 下沒有找到包含音頻的（子）目錄")
                sys.exit(1)

            print(f"🔍 自動掃描發現 {len(speaker_dirs)} 個說話人:")
            for d in speaker_dirs:
                print(f"   - {d.name}")
            print()

    elif args.auto_scan:
        # 顯式自動掃描模式（向後兼容）
        base_dir = Path(args.data_dirs[0])
        if not base_dir.exists():
            print(f"❌ 錯誤: 目錄不存在: {base_dir}")
            sys.exit(1)

        speaker_dirs = scan_speaker_dirs(base_dir)
        if not speaker_dirs:
            print(f"❌ 錯誤: 在 {base_dir} 下沒有找到包含音頻的子目錄")
            sys.exit(1)

        print(f"🔍 自動掃描發現 {len(speaker_dirs)} 個說話人:")
        for d in speaker_dirs:
            print(f"   - {d.name}")
        print()

    else:
        # 多個目錄：多說話人模式
        speaker_dirs = [Path(d) for d in args.data_dirs]
        for d in speaker_dirs:
            if not d.exists():
                print(f"❌ 錯誤: 目錄不存在: {d}")
                sys.exit(1)

    # 創建輸出目錄
    output_dir = Path("finetune_data/audio_list")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 修復目錄權限
    fix_file_permissions(output_dir)

    # 處理每個說話人
    results = []
    total_count = 0
    all_entries = []  # 用於多說話人模式的合併

    for speaker_dir in speaker_dirs:
        # 智能提取 speaker_id
        speaker_id = speaker_dir.name

        # 如果是通用目錄名（如 data），嘗試智能推斷
        generic_names = ['data', 'audio', 'dataset', 'train', 'finetune', 'wav', 'wavs']
        if speaker_id.lower() in generic_names:
            # 方法 1: 檢查是否有數字命名的子目錄（常見模式）
            subdirs = [d for d in speaker_dir.iterdir() if d.is_dir()]
            numeric_subdirs = [d for d in subdirs if d.name.isdigit()]

            if numeric_subdirs:
                # 使用第一個數字子目錄作為 speaker_id
                speaker_id = f"speaker_{numeric_subdirs[0].name}"
            else:
                # 方法 2: 從絕對路徑中提取
                abs_parts = speaker_dir.resolve().parts
                for part in reversed(abs_parts):
                    if part.lower() not in generic_names and part != '/':
                        speaker_id = part
                        break

        print(f"📦 處理說話人: {speaker_id}")

        entries, success_count, missing_count = prepare_single_speaker(
            speaker_dir,
            args.text_suffix
        )

        if not entries:
            print(f"   ⚠️  沒有找到有效的音頻文件\n")
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
            print(f"   ⚠️  缺失文本: {missing_count} 個")
        print()

    if not results:
        print("❌ 錯誤: 沒有找到任何有效數據")
        sys.exit(1)

    merged_file_path: str | None = None

    if len(results) == 1:
        # 單說話人模式
        speaker_id = args.speaker_id if args.speaker_id else results[0]['speaker_id']

        if args.output:
            output_file = Path(args.output)
        else:
            output_file = output_dir / f"{speaker_id}.txt"

        results[0]['output_file'] = str(output_file)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(results[0]['entries']))

        # 修復文件權限
        fix_file_permissions(output_file)

        print(f"📄 輸出文件: {output_file}")

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
            print("ℹ 已跳過個別說話人的列表 (--no-individual)")

        if args.merge_all:
            merged_file = output_dir / "audio_list_all.txt"
            with open(merged_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(all_entries))

            fix_file_permissions(merged_file)

            merged_file_path = str(merged_file)

            print(f"\n📄 合併文件: {merged_file}")
            print(f"   來源說話人數: {len(results)} 位")

    print_summary(results, total_count, merged_file_path)


if __name__ == '__main__':
    main()
