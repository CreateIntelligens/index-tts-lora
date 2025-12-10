import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

# 匯入與 infer.py 相同的文字處理類別
from indextts.utils.front import TextNormalizer, TextTokenizer


def _load_manifest_batch(batch):
    """
    批次載入多個 manifest 檔案 (減少進程啟動開銷)。
    """
    all_items = []
    total_filtered = 0
    for manifest_file, speaker_id in batch:
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        duration = item.get("duration", 0)
                        if duration > 20 or duration < 1:
                            total_filtered += 1
                            continue
                        item['speaker_id'] = speaker_id
                        all_items.append(item)
        except Exception as e:
            logger.warning(f"[警告] 載入 {manifest_file} 時發生錯誤: {e}")
    return all_items, total_filtered


class FinetuneDataset(Dataset):
    """
    用於 UnifiedVoice 微調的自定義資料集，支援多說話人資料。
    """
    def __init__(self, manifest_files: List[str], bpe_path: str, speaker_ids: List[str], config: DictConfig):
        """
        Args:
            manifest_files (List[str]): 不同說話人的 manifest 檔案路徑列表。
            bpe_path (str): BPE 模型檔案路徑。
            speaker_ids (List[str]): 對應於 manifest 檔案的說話人 ID 列表。
            config (DictConfig): 額外的預處理配置。
        """
        super().__init__()
        self.config = config
        self.bpe_path = bpe_path
        self.data = []
        self.index = []
        self.manifest_offsets = {}  # manifest_idx -> offsets list (用於條件採樣)
        self.manifest_files = manifest_files
        self.manifest_speakers = speaker_ids if speaker_ids else ["unknown"] * len(manifest_files)
        self.use_lazy_metadata = True
        if hasattr(config, "train") and isinstance(config.train, DictConfig) and "lazy_load_metadata" in config.train:
            self.use_lazy_metadata = bool(config.train.lazy_load_metadata)

        # Cross-Speaker Conditioning 比例 (0.0 = 停用, 0.2 = 20% 機率使用其他說話人)
        self.cross_speaker_ratio = 0.0
        if hasattr(config, "train") and isinstance(config.train, DictConfig):
            self.cross_speaker_ratio = float(config.train.get("cross_speaker_ratio", 0.0))

        # 推斷 data_path (用於尋找 index 檔案)
        if manifest_files:
            import os
            first_manifest = manifest_files[0]
            self.data_path = os.path.dirname(os.path.dirname(first_manifest))
        else:
            self.data_path = config.train.data_path if hasattr(config.train, "data_path") else "."

        # 載入所有說話人的資料 (使用多執行緒加速 IO)
        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 獲取當前進程的 rank (用於多 GPU 訓練)
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        dataset_type = "train" if "train" in str(self.manifest_files[0]) else "valid"

        if self.use_lazy_metadata:
            # 僅在 Rank 0 掃描並建立 offset 索引，然後廣播給其他 rank
            if rank == 0:
                print(f">> [Rank 0] 懶載入模式: 掃描 {len(self.manifest_files)} 個 manifest 的 {dataset_type} Metadata (不常駐記憶體)...")
            filtered_count, total_samples = self._build_lazy_index(dataset_type, rank)
        else:
            # 僅在 Rank 0 載入完整 Metadata，然後廣播給其他 rank
            if rank == 0:
                print(f">> [Rank 0] 開始載入 {len(self.manifest_files)} 個 manifest 的 {dataset_type} Metadata...")
            else:
                print(f">> [Rank {rank}] 等待 Rank 0 載入 {dataset_type} Metadata...")
            filtered_count, total_samples = self._load_all_metadata(dataset_type, rank)

        logger.info(
            f"✓ 準備 {total_samples} 個樣本 (來自 {len(self.manifest_files)} 個說話人，"
            f"模式: {'Lazy' if self.use_lazy_metadata else 'In-Memory'})" +
            (f"，已過濾掉 {filtered_count} 個不符合時長的樣本" if filtered_count > 0 else "")
        )

        # Metadata 載入完成後，才初始化 Tokenizer (僅載入一次)
        print(">> [系統] 初始化 TextNormalizer 與 BPE Tokenizer...")
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        self.tokenizer = TextTokenizer(bpe_path, self.normalizer)
        print(">> [系統] TextNormalizer 與 BPE 模型載入完成")

    def __len__(self):
        return len(self.index) if self.use_lazy_metadata else len(self.data)

    def __getitem__(self, index):
        if self.use_lazy_metadata:
            # 使用預先建立的索引
            manifest_idx, offset = self.index[index]
            manifest_file = self.manifest_files[manifest_idx]
            try:
                item = self._read_item(manifest_file, offset)
            except (json.JSONDecodeError, ValueError) as e:
                # 若樣本有問題，嘗試回退至索引中的下一個
                if index + 1 < len(self.index):
                    return self.__getitem__(index + 1)
                else:
                    raise RuntimeError(f"[錯誤] 無法從 {manifest_file} 的 offset {offset} 載入樣本 {index}: {e}")
        else:
            item = self.data[index]
            manifest_idx = None  # 僅用於 fallback
            offset = None

        # 由 audio 路徑推回 speaker_id (drama_name + character_id)
        if "speaker_id" not in item:
            item["speaker_id"] = self._infer_speaker_id(item.get("audio", ""))
        
        text = item["text"]
        codes_path = item.get("codes", None)
        mels_path = item.get("mels", None)
        speaker_id = item["speaker_id"]

        # 使用與 infer.py 完全相同的文字處理流程
        text_tokens_list = self.tokenizer.tokenize(text)
        text_ids = self.tokenizer.convert_tokens_to_ids(text_tokens_list)
        text_ids = torch.LongTensor(text_ids).unsqueeze(0)

        # 載入音訊特徵
        codes_npy = np.load(codes_path)
        mels_npy = np.load(mels_path)

        # 取樣 Conditioning 音檔 (Zero-shot 訓練策略)
        # Cross-Speaker Conditioning: 以一定機率使用其他說話人的音檔，
        # 強迫模型學習依賴 conditioning embedding，避免忽略參考音檔。
        cross_speaker_ratio = getattr(self, 'cross_speaker_ratio', 0.0)
        use_cross_speaker = cross_speaker_ratio > 0 and random.random() < cross_speaker_ratio

        if use_cross_speaker:
            cond_spec = self._sample_cross_speaker_condition(manifest_idx)
        elif self.use_lazy_metadata:
            cond_spec = self._sample_condition_lazy(manifest_idx, offset)
        else:
            cond_spec = self._sample_condition_in_memory(speaker_id, index)

        mel_spec = torch.FloatTensor(mels_npy)  # [B, D, T]
        mel_codes = torch.LongTensor(codes_npy)  # [B, T]
        condition = torch.FloatTensor(cond_spec)

        return (mel_spec, mel_codes, text_ids, condition, speaker_id)

    def _read_item(self, manifest_file: str, offset: int):
        """
        從 manifest 讀取單條樣本，並檢查時長。
        """
        with open(manifest_file, "r", encoding="utf-8") as f:
            f.seek(offset)
            line = f.readline()
        if not line or not line.strip():
            raise ValueError(f"[錯誤] {manifest_file} 在 offset {offset} 處為空行")
        item = json.loads(line.strip())

        duration = item.get("duration", 0)
        if duration > 20 or duration < 1:
            raise ValueError(f"[錯誤] 時長 {duration} 超出範圍 [1, 20]")
        return item

    def _sample_condition_lazy(self, manifest_idx: int, target_offset: int):
        """
        在 Lazy 模式下，隨機抽取同 Manifest (同 Speaker) 的另一條樣本當作 Conditioning。
        """
        offsets = self.manifest_offsets.get(manifest_idx, [])
        cond_offset = target_offset
        if len(offsets) > 1:
            # 直到抽到與 target 不同的 offset
            for _ in range(3):
                candidate = random.choice(offsets)
                if candidate != target_offset:
                    cond_offset = candidate
                    break
        manifest_file = self.manifest_files[manifest_idx]
        cond_item = self._read_item(manifest_file, cond_offset)
        cond_mels_path = cond_item.get("mels", None)
        if cond_mels_path and os.path.exists(cond_mels_path):
            return np.load(cond_mels_path)
        # Fallback: 使用 target 的 mel (極少數單樣本 speaker)
        target_item = self._read_item(manifest_file, target_offset)
        return np.load(target_item.get("mels", None))

    def _sample_condition_in_memory(self, speaker_id: str, target_index: int):
        """
        在 In-Memory 模式下，隨機抽取同 Speaker 的另一條樣本。
        """
        # 懶建立 Bucket
        if not hasattr(self, "_speaker_buckets"):
            buckets = {}
            for idx, it in enumerate(self.data):
                sid = it.get("speaker_id") or self._infer_speaker_id(it.get("audio", ""))
                buckets.setdefault(sid, []).append(idx)
            self._speaker_buckets = buckets

        candidates = self._speaker_buckets.get(speaker_id, [])
        cond_idx = target_index
        if len(candidates) > 1:
            for _ in range(3):
                c = random.choice(candidates)
                if c != target_index:
                    cond_idx = c
                    break
        cond_item = self.data[cond_idx]
        cond_mels_path = cond_item.get("mels", None)
        if cond_mels_path and os.path.exists(cond_mels_path):
            return np.load(cond_mels_path)
        return np.load(self.data[target_index].get("mels", None))

    def _sample_cross_speaker_condition(self, exclude_manifest_idx: int = None):
        """
        隨機抽取其他說話人的音檔作為 Conditioning。

        此策略強迫模型學習依賴 conditioning embedding 來區分說話人，
        避免 LoRA 層學會忽略 conditioning。

        Args:
            exclude_manifest_idx: 要排除的 manifest 索引（當前說話人）

        Returns:
            np.ndarray: 隨機說話人的 mel spectrogram
        """
        available_manifests = list(self.manifest_offsets.keys())

        # 排除當前說話人
        if exclude_manifest_idx is not None and len(available_manifests) > 1:
            available_manifests = [m for m in available_manifests if m != exclude_manifest_idx]

        if not available_manifests:
            # 若無其他說話人可用，回退至同說話人
            if exclude_manifest_idx is not None:
                offsets = self.manifest_offsets.get(exclude_manifest_idx, [])
                if offsets:
                    offset = random.choice(offsets)
                    manifest_file = self.manifest_files[exclude_manifest_idx]
                    item = self._read_item(manifest_file, offset)
                    return np.load(item.get("mels"))
            raise RuntimeError("無法取樣 cross-speaker condition：無可用說話人")

        # 隨機選擇其他說話人
        random_manifest_idx = random.choice(available_manifests)
        offsets = self.manifest_offsets.get(random_manifest_idx, [])

        if not offsets:
            # 回退至任意可用 manifest
            for m_idx in available_manifests:
                offsets = self.manifest_offsets.get(m_idx, [])
                if offsets:
                    random_manifest_idx = m_idx
                    break

        if not offsets:
            raise RuntimeError("無法取樣 cross-speaker condition：所有 manifest 均無有效樣本")

        offset = random.choice(offsets)
        manifest_file = self.manifest_files[random_manifest_idx]
        item = self._read_item(manifest_file, offset)
        mels_path = item.get("mels")

        if mels_path and os.path.exists(mels_path):
            return np.load(mels_path)

        raise RuntimeError(f"Cross-speaker condition 的 mel 檔案不存在: {mels_path}")

    def _infer_speaker_id(self, audio_path: str) -> str:
        """
        從 Audio 路徑推回 speaker_id，用於共用 manifest 的情況。
        """
        try:
            parts = Path(audio_path).parts
            data_idx = parts.index("data") if "data" in parts else -1
            if data_idx >= 0 and len(parts) > data_idx + 2:
                drama_name = parts[data_idx + 1]
                character_id = parts[data_idx + 2]
                return f"{drama_name}_{character_id}"
        except Exception:
            pass
        return "unknown"

    def _build_lazy_index(self, dataset_type: str, rank: int):
        """
        僅建立 Offset 索引，不將 Metadata 常駐記憶體。
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        import multiprocessing
        import torch.distributed as dist
        import pickle

        # 檢查是否有預先建立的 Index 檔案
        index_file = os.path.join(self.data_path, f'{dataset_type}_index.pkl')

        if rank == 0 and os.path.exists(index_file):
            print(f">> [Rank 0] 載入預先建立的索引: {index_file}")
            with open(index_file, 'rb') as f:
                self.index = pickle.load(f)
            total_samples = len(self.index)
            # 重建 Per-manifest Offset Bucket
            buckets = {}
            for m_idx, off in self.index:
                buckets.setdefault(m_idx, []).append(off)
            self.manifest_offsets = buckets
            filtered_count = 0
            print(f">> [Rank 0] 載入完成，共 {total_samples:,} 個樣本")

            # 廣播索引給其他 Rank
            if dist.is_initialized():
                index_bytes = pickle.dumps(self.index)
                index_size = len(index_bytes)
                size_tensor = torch.tensor([index_size, filtered_count, total_samples], dtype=torch.long, device='cuda')
                dist.broadcast(size_tensor, src=0)
                index_tensor = torch.ByteTensor(list(index_bytes)).cuda()
                dist.broadcast(index_tensor, src=0)
                print(f">> [Rank 0] 已廣播 {index_size/1024/1024:.1f} MB 索引")

            return filtered_count, total_samples

        def scan_manifest(manifest_idx: int, manifest_file: str):
            """回傳該 Manifest 的所有行的 Offset (不解析 JSON，快速掃描)。"""
            offsets = []
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    while True:
                        offset = f.tell()
                        line = f.readline()
                        if not line:
                            break
                        if line.strip():
                            offsets.append(offset)
            except Exception as e:
                return manifest_idx, offsets, 0, f"Error scanning {manifest_file}: {e}"
            return manifest_idx, offsets, 0, None

        # 僅在 Rank 0 掃描索引
        if rank == 0:
            max_workers = min(32, (multiprocessing.cpu_count() or 1) * 2)
            entries_per_manifest = [None] * len(self.manifest_files)
            filtered_count = 0
            total_samples = 0
            errors = []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(scan_manifest, idx, mf)
                    for idx, mf in enumerate(self.manifest_files)
                ]
                pbar = tqdm(total=len(futures),
                            desc=f"[Rank 0] 掃描 {dataset_type} Metadata (Lazy)",
                            unit="spk", ncols=100, postfix={"samples": 0})
                for future in as_completed(futures):
                    manifest_idx, offsets, filtered, err = future.result()
                    entries_per_manifest[manifest_idx] = offsets
                    filtered_count += filtered
                    total_samples += len(offsets)
                    if err:
                        errors.append(err)
                    pbar.update(1)
                    pbar.set_postfix({"samples": total_samples})
                pbar.close()

            # 維持 Manifest 順序展平成全域索引
            for m_idx, offsets in enumerate(entries_per_manifest):
                if offsets:
                    self.index.extend([(m_idx, off) for off in offsets])
            
            self.manifest_offsets = {
                idx: offs for idx, offs in enumerate(entries_per_manifest) if offs
            }

            for error in errors:
                logger.warning(error)

            print(f">> [Rank 0] 廣播 {len(self.index)} 個索引給其他 Rank...")
        else:
            filtered_count = 0
            total_samples = 0

        # 廣播索引給其他 Rank
        if dist.is_initialized():
            import pickle
            if rank == 0:
                index_bytes = pickle.dumps(self.index)
                index_size = len(index_bytes)
                size_tensor = torch.tensor([index_size, filtered_count, total_samples], dtype=torch.long, device='cuda')
                dist.broadcast(size_tensor, src=0)
                index_tensor = torch.ByteTensor(list(index_bytes)).cuda()
                dist.broadcast(index_tensor, src=0)
                print(f">> [Rank 0] 已廣播 {index_size/1024/1024:.1f} MB 索引")
            else:
                size_tensor = torch.tensor([0, 0, 0], dtype=torch.long, device='cuda')
                dist.broadcast(size_tensor, src=0)
                index_size, filtered_count, total_samples = size_tensor.cpu().tolist()
                index_tensor = torch.ByteTensor(index_size).cuda()
                dist.broadcast(index_tensor, src=0)
                index_bytes = bytes(index_tensor.cpu().tolist())
                self.index = pickle.loads(index_bytes)
                
                buckets = {}
                for m_idx, off in self.index:
                    if isinstance(off, (list, tuple)):
                        off_val = off[1]
                    else:
                        off_val = off
                    buckets.setdefault(m_idx, []).append(off_val)
                self.manifest_offsets = buckets
                print(f">> [Rank {rank}] 已接收 {len(self.index)} 個索引")

        return filtered_count, total_samples

    def _load_all_metadata(self, dataset_type: str, rank: int):
        """
        沿用原本的完整載入流程 (佔用記憶體)。
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        import torch.distributed as dist

        def load_single_manifest(manifest_file, speaker_id):
            items = []
            filtered = 0
            error_msg = None
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line.strip())
                            duration = item.get("duration", 0)
                            if duration > 20 or duration < 1:
                                filtered += 1
                                continue
                            item['speaker_id'] = self._infer_speaker_id(item.get("audio", ""))
                            items.append(item)
            except Exception as e:
                error_msg = f"Error loading {manifest_file}: {e}"
            return items, filtered, error_msg

        if rank == 0:
            filtered_count = 0
            max_workers = min(64, len(self.manifest_files))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(load_single_manifest, mf, sid): (mf, sid)
                    for mf, sid in zip(self.manifest_files, self.manifest_speakers)
                }

                errors = []
                all_items = []
                pbar = tqdm(total=len(futures), desc=f"[Rank 0] 載入 {dataset_type} Metadata",
                           unit="spk", ncols=100, postfix={"samples": 0})
                for future in as_completed(futures):
                    items, filtered, error_msg = future.result()
                    if error_msg:
                        errors.append(error_msg)
                    all_items.append(items)
                    filtered_count += filtered
                    pbar.update(1)
                    pbar.set_postfix({"samples": sum(len(x) for x in all_items)})
                pbar.close()

                total_samples = sum(len(x) for x in all_items)
                print(f">> [Rank 0] 合併 {total_samples:,} 個 {dataset_type} 樣本...")

                self.data = []
                for items in all_items:
                    self.data.extend(items)

                del all_items
                import gc
                gc.collect()

                for error in errors:
                    logger.warning(error)

            print(f">> [Rank 0] 廣播 {len(self.data)} 個樣本給其他 Rank...")
        else:
            self.data = None
            filtered_count = 0

        # 廣播給其他 Rank
        if dist.is_initialized():
            import pickle
            if rank == 0:
                data_bytes = pickle.dumps(self.data)
                data_size = len(data_bytes)
                size_tensor = torch.tensor([data_size], dtype=torch.long, device='cuda')
                dist.broadcast(size_tensor, src=0)
                data_tensor = torch.ByteTensor(list(data_bytes)).cuda()
                dist.broadcast(data_tensor, src=0)
                print(f">> [Rank 0] 已廣播 {data_size/1024/1024:.1f} MB 資料")
            else:
                size_tensor = torch.tensor([0], dtype=torch.long, device='cuda')
                dist.broadcast(size_tensor, src=0)
                data_size = size_tensor.cpu().item()
                data_tensor = torch.ByteTensor(data_size).cuda()
                dist.broadcast(data_tensor, src=0)
                data_bytes = bytes(data_tensor.cpu().tolist())
                self.data = pickle.loads(data_bytes)
                print(f">> [Rank {rank}] 已接收 {len(self.data)} 個樣本")

        total_samples = len(self.data)
        return filtered_count, total_samples


# ------------------------------------------------------------------------------------------------------------------
# Collate function
# ------------------------------------------------------------------------------------------------------------------


def _pad_sequence(seqs, pad_value=0, dim=-1):
    """
    在指定維度上填充張量列表至最大長度。

    Args:
        seqs (List[torch.Tensor]): 張量列表。
        pad_value (int|float): 填充值。
        dim (int): 填充維度 (預設: 最後一維)。

    Returns:
        Tuple[torch.Tensor, torch.LongTensor]: (填充後的張量, 長度張量)
    """

    assert dim == -1, "填充維度必須是最後一維"

    lengths = torch.tensor([s.shape[dim] for s in seqs], dtype=torch.long)
    max_len = lengths.max().item()

    out_shape = list(seqs[0].shape)
    out_shape[dim] = max_len
    out_shape = [len(seqs)] + out_shape

    padded = seqs[0].new_full(out_shape, pad_value)

    for i, s in enumerate(seqs):
        if s.dim() == 1:
            padded[i, :s.shape[0]] = s
        elif s.dim() == 2:
            padded[i, :, :s.shape[1]] = s
        elif s.dim() == 3:
            padded[i, :, :, :s.shape[2]] = s
        else:
            padded[i, ..., :s.shape[-1]] = s

    return padded, lengths


def collate_finetune_fn(batch):
    """
    FinetuneDataset 的 Collate 函數，支援多說話人資料。

    執行步驟：
    1. 對 mel_spec, mel_codes, text_ids, condition 進行右側填充。
    2. 返回填充後的張量與原始長度。

    Returns:
        Tuple: (mel_specs, mel_codes, text_ids, condition_mels, speaker_ids,
                mel_lengths, codes_lengths, text_lengths, cond_lengths)
    """

    mel_specs, mel_codes, text_ids, conditions, speaker_ids = zip(*batch)

    # 移除額外的 Batch 維度
    mel_specs = [spec.squeeze(0) if spec.dim() >= 3 and spec.size(0) == 1 else spec for spec in mel_specs]
    mel_codes = [codes.squeeze(0) if codes.dim() >= 2 and codes.size(0) == 1 else codes for codes in mel_codes]
    text_ids = [ids.squeeze(0) if ids.dim() == 2 and ids.size(0) == 1 else ids for ids in text_ids]
    
    assert all(cond is not None for cond in conditions), "Conditioning Mel 不能為 None"
    conditions = [cond.squeeze(0) if cond.dim() >= 3 and cond.size(0) == 1 else cond for cond in conditions]

    # Padding
    mel_specs_padded, mel_lengths = _pad_sequence(list(mel_specs), pad_value=0.0, dim=-1)
    mel_codes_padded, codes_lengths = _pad_sequence(list(mel_codes), pad_value=0, dim=-1)
    text_ids_padded, text_lengths = _pad_sequence(list(text_ids), pad_value=0, dim=-1)
    condition_padded, cond_lengths = _pad_sequence(list(conditions), pad_value=0.0, dim=-1)

    return (
        mel_specs_padded, # [B, 100, T_max]
        mel_codes_padded, # [B, T_max_codes]
        text_ids_padded, # [B, T_max]
        condition_padded, # [B, 100, T_cond_max]
        list(speaker_ids), # [B]
        mel_lengths,
        codes_lengths,
        text_lengths,
        cond_lengths,
    )


def load_finetune_datasets(config: DictConfig, bpe_path: str) -> Tuple[Dataset, Dataset]:
    """
    載入多說話人微調的訓練與驗證資料集。

    Args:
        config (DictConfig): 全域配置。
        bpe_path (str): BPE 模型路徑。

    Returns:
        Tuple[Dataset, Dataset]: (train_dataset, validation_dataset)
    """
    speaker_info_path = os.path.join(config.train.data_path, "speaker_info.json")
    with open(speaker_info_path, 'r', encoding='utf-8') as f:
        speaker_info_list = json.load(f)
    
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import multiprocessing

    logger.info(f"載入 {len(speaker_info_list)} 個說話人的資料...")

    def check_speaker_files(speaker_info):
        """檢查單個說話人的檔案是否存在"""
        speaker_id = speaker_info['speaker']
        train_file = speaker_info['train_jsonl']
        valid_file = speaker_info['valid_jsonl']

        try:
            if os.access(train_file, os.R_OK) and os.access(valid_file, os.R_OK):
                return (train_file, valid_file, speaker_id, True)
            else:
                return (None, None, speaker_id, False)
        except:
            return (None, None, speaker_id, False)

    missing_count = 0
    unique_train_files = set()
    unique_valid_files = set()
    valid_speaker_ids = []

    num_workers = min(32, multiprocessing.cpu_count() * 2)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(check_speaker_files, info) for info in speaker_info_list]

        for future in tqdm(as_completed(futures), total=len(speaker_info_list),
                          desc="檢查資料完整性", unit="speaker", ncols=80):
            train_file, valid_file, speaker_id, success = future.result()
            if success:
                unique_train_files.add(train_file)
                unique_valid_files.add(valid_file)
                valid_speaker_ids.append(speaker_id)
            else:
                missing_count += 1

    train_manifest_files = sorted(list(unique_train_files))
    valid_manifest_files = sorted(list(unique_valid_files))

    logger.info(f"✓ 成功載入 {len(valid_speaker_ids)} 個說話人，"
                f"共 {len(train_manifest_files)} 個訓練檔案，"
                f"{len(valid_manifest_files)} 個驗證檔案" +
                (f" (跳過 {missing_count} 個缺失檔案)" if missing_count > 0 else ""))
    
    train_dataset = FinetuneDataset(train_manifest_files, bpe_path, None, config)
    valid_dataset = FinetuneDataset(valid_manifest_files, bpe_path, None, config)
    
    return train_dataset, valid_dataset


def load_speaker_conditions(config: DictConfig) -> dict:
    """
    載入所有說話人的平均條件 (mean_condition)。
    """
    speaker_info_path = os.path.join(config.train.data_path, "speaker_info.json")
    with open(speaker_info_path, 'r', encoding='utf-8') as f:
        speaker_info_list = json.load(f)
    
    speaker_conditions = {}
    from tqdm import tqdm

    for speaker_info in tqdm(speaker_info_list, desc="載入說話人條件", unit="spk", ncols=80):
        speaker_id = speaker_info['speaker']
        medoid_path = speaker_info['medoid_condition']

        if os.path.exists(medoid_path):
            condition = np.load(medoid_path)
            speaker_conditions[speaker_id] = torch.from_numpy(condition).float()
        else:
            raise ValueError(f"說話人 {speaker_id} 缺少 medoid_condition.npy")

    logger.info(f"✓ 載入 {len(speaker_conditions)} 個說話人的 Mean Conditions")
    return speaker_conditions


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="載入微調資料集測試")
    parser.add_argument("--config", type=str, default="finetune_models/config.yaml", help="配置檔案路徑")
    parser.add_argument("--bpe_model", type=str, default="finetune_models/bpe.model", help="SentencePiece 模型路徑")

    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    train_dataset, valid_dataset = load_finetune_datasets(config, args.bpe_model)

    logger.info(f"Train Dataset 大小: {len(train_dataset)}")
    logger.info(f"Validation Dataset 大小: {len(valid_dataset)}")

    loader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_finetune_fn)
    sample_batch = next(iter(loader))

    mel_specs, mel_codes, text_ids, cond_mels, speaker_ids, mel_lens, code_lens, text_lens, cond_lens = sample_batch
    logger.info(f"Sample Batch 形狀 -- mel_specs: {tuple(mel_specs.shape)}, mel_codes: {tuple(mel_codes.shape)}, text_ids: {tuple(text_ids.shape)}, cond_mels: {tuple(cond_mels.shape)}")
    logger.info(f"Speaker IDs: {speaker_ids}")
    logger.info(f"Lengths -- mel: {mel_lens.tolist()}, codes: {code_lens.tolist()}, text: {text_lens.tolist()}, cond: {cond_lens.tolist()}")
