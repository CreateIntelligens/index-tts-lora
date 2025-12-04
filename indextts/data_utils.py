import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

# 匯入與infer.py相同的文字處理類
from indextts.utils.front import TextNormalizer, TextTokenizer


def _load_manifest_batch(batch):
    """批次載入多個 manifest 檔案（減少進程啟動開銷）"""
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
            logger.warning(f"Error loading {manifest_file}: {e}")
    return all_items, total_filtered


class FinetuneDataset(Dataset):
    """
    Custom dataset used for UnifiedVoice fine-tuning, supporting multi-speaker data.
    """
    def __init__(self, manifest_files: List[str], bpe_path: str, speaker_ids: List[str], config: DictConfig):
        """
        Args:
            manifest_files (List[str]): List of paths to manifest files for different speakers.
            bpe_path (str): Path to the BPE model file.
            speaker_ids (List[str]): List of speaker IDs corresponding to manifest files.
            config (DictConfig): Extra preprocessing configuration.
        """
        super().__init__()
        self.config = config
        self.bpe_path = bpe_path
        self.data = []
        self.index = []
        self.manifest_files = manifest_files
        self.manifest_speakers = speaker_ids if speaker_ids else ["unknown"] * len(manifest_files)
        self.use_lazy_metadata = True
        if hasattr(config, "train") and isinstance(config.train, DictConfig) and "lazy_load_metadata" in config.train:
            self.use_lazy_metadata = bool(config.train.lazy_load_metadata)

        # 推斷 data_path（用於找 index 檔案）
        if manifest_files:
            # 從第一個 manifest 檔案的路徑推斷 data_path
            # 例如: finetune_data/processed_data/xxx/train.jsonl -> finetune_data/processed_data
            import os
            first_manifest = manifest_files[0]
            self.data_path = os.path.dirname(os.path.dirname(first_manifest))
        else:
            self.data_path = config.train.data_path if hasattr(config.train, "data_path") else "."

        # 載入所有說話人的資料（使用多線程加速 IO）
        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 取得當前進程的 rank（用於多 GPU 訓練）
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        dataset_type = "train" if "train" in str(self.manifest_files[0]) else "valid"

        if self.use_lazy_metadata:
            # 只在 rank0 掃描並建立 offset 索引，然後廣播給其他 rank
            if rank == 0:
                print(f">> [Rank 0] 懶載入模式：掃描 {len(self.manifest_files)} 個 manifest 的 {dataset_type} metadata（不常駐記憶體）...")
            filtered_count, total_samples = self._build_lazy_index(dataset_type, rank)
        else:
            # 只在 rank 0 載入完整 metadata，然後廣播給其他 rank
            if rank == 0:
                print(f">> [Rank 0] 開始載入 {len(self.manifest_files)} 個 manifest 的 {dataset_type} metadata...")
            else:
                print(f">> [Rank {rank}] 等待 Rank 0 載入 {dataset_type} metadata...")
            filtered_count, total_samples = self._load_all_metadata(dataset_type, rank)

        logger.info(
            f"✓ 準備 {total_samples} 個樣本 (來自 {len(self.manifest_files)} 個 speaker，"
            f"模式: {'lazy' if self.use_lazy_metadata else 'in-memory'})" +
            (f"，過濾掉 {filtered_count} 個不符合時長的樣本" if filtered_count > 0 else "")
        )

        # metadata 載入完成後，才初始化 tokenizer（只載入一次）
        print(">> 初始化 TextNormalizer 和 BPE tokenizer...")
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        self.tokenizer = TextTokenizer(bpe_path, self.normalizer)
        print(">> TextNormalizer 和 BPE model 載入完成")

    def __len__(self):
        return len(self.index) if self.use_lazy_metadata else len(self.data)

    def __getitem__(self, index):
        if self.use_lazy_metadata:
            # 有預先建立的索引
            manifest_idx, offset = self.index[index]
            manifest_file = self.manifest_files[manifest_idx]
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    f.seek(offset)
                    line = f.readline()
                if not line or not line.strip():
                    raise ValueError(f"Empty line at offset {offset} in {manifest_file}")
                item = json.loads(line.strip())

                # 過濾時長（在載入時才檢查）
                duration = item.get("duration", 0)
                if duration > 20 or duration < 1:
                    # 時長不符，跳到下一個樣本（DataLoader 會重試）
                    raise ValueError(f"Duration {duration} out of range [1, 20]")

            except (json.JSONDecodeError, ValueError) as e:
                # 如果這個樣本有問題，嘗試回退到索引中的下一個
                if index + 1 < len(self.index):
                    return self.__getitem__(index + 1)
                else:
                    raise RuntimeError(f"Failed to load sample {index} from {manifest_file} at offset {offset}: {e}")
        else:
            item = self.data[index]

        # 由 audio 路徑推回 speaker_id（drama_name + character_id）
        if "speaker_id" not in item:
            item["speaker_id"] = self._infer_speaker_id(item.get("audio", ""))
        
        text = item["text"]
        codes_path = item.get("codes", None)
        mels_path = item.get("mels", None)
        condition_path = item.get("condition", None)
        speaker_id = item["speaker_id"]

        # 使用與infer.py完全相同的文字處理流程
        # 這確保訓練和推理時的文字處理完全一致
        text_tokens_list = self.tokenizer.tokenize(text)  # 這裡會呼叫normalize + tokenize_by_CJK_char
        text_ids = self.tokenizer.convert_tokens_to_ids(text_tokens_list)
        text_ids = torch.LongTensor(text_ids).unsqueeze(0)

        # 載入音訊特徵
        codes_npy = np.load(codes_path)
        mels_npy = np.load(mels_path)
        condition_npy = np.load(condition_path) if condition_path else None

        mel_spec = torch.FloatTensor(mels_npy)  # [B, D, T]
        mel_codes = torch.LongTensor(codes_npy)  # [B, T]
        condition = torch.FloatTensor(condition_npy) if condition_npy is not None else None

        return (mel_spec, mel_codes, text_ids, condition, speaker_id)

    def _infer_speaker_id(self, audio_path: str) -> str:
        """從 audio 路徑推回 speaker_id，用於共用 manifest 的情況。"""
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
        """只建立 offset 索引，不把 metadata 常駐記憶體。"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        import multiprocessing
        import torch.distributed as dist
        import pickle

        # 檢查是否有預先建立的 index 檔案
        index_file = os.path.join(self.data_path, f'{dataset_type}_index.pkl')

        if rank == 0 and os.path.exists(index_file):
            print(f">> [Rank 0] 載入預先建立的索引: {index_file}")
            with open(index_file, 'rb') as f:
                self.index = pickle.load(f)
            total_samples = len(self.index)
            filtered_count = 0
            print(f">> [Rank 0] 載入完成，共 {total_samples:,} 個樣本")

            # 廣播索引給其他 rank
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
            """回傳該 manifest 的所有行的 offset（不解析 JSON，快速掃描）。"""
            offsets = []
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    while True:
                        offset = f.tell()
                        line = f.readline()
                        if not line:
                            break
                        if line.strip():  # 只跳過空行
                            offsets.append((manifest_idx, offset))
            except Exception as e:
                return manifest_idx, offsets, 0, f"Error scanning {manifest_file}: {e}"
            return manifest_idx, offsets, 0, None

        # 只在 rank 0 掃描索引
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
                            desc=f"[Rank 0] 掃描 {dataset_type} metadata (lazy)",
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

            # 維持 manifest 順序展平成全域索引
            for offsets in entries_per_manifest:
                if offsets:
                    self.index.extend(offsets)

            for error in errors:
                logger.warning(error)

            print(f">> [Rank 0] 廣播 {len(self.index)} 個索引給其他 rank...")
        else:
            filtered_count = 0
            total_samples = 0

        # 廣播索引給其他 rank
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
                print(f">> [Rank {rank}] 已接收 {len(self.index)} 個索引")

        return filtered_count, total_samples

    def _load_all_metadata(self, dataset_type: str, rank: int):
        """沿用原本的完整載入流程（占記憶體）。"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        import torch.distributed as dist

        def load_single_manifest(manifest_file, speaker_id):
            """載入單個 manifest 檔案"""
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
                pbar = tqdm(total=len(futures), desc=f"[Rank 0] 載入 {dataset_type} metadata",
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

            print(f">> [Rank 0] 廣播 {len(self.data)} 個樣本給其他 rank...")
        else:
            self.data = None
            filtered_count = 0

        # 廣播給其他 rank（使用 pickle 序列化）
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
    """Pad a list of tensors on the specified dimension to the max length.

    Args:
        seqs (List[torch.Tensor]): list of tensors with shape (..., L_i)
        pad_value (int|float): value to use for padding
        dim (int): dimension to pad on (default: last dimension)

    Returns:
        Tuple[torch.Tensor, torch.LongTensor]:
            - padded tensor of shape (*batch, max_len)
            - lengths tensor (B,)
    """

    assert dim == -1, "Padding dimension must be the last dimension"

    lengths = torch.tensor([s.shape[dim] for s in seqs], dtype=torch.long)
    max_len = lengths.max().item()

    # Determine output shape
    out_shape = list(seqs[0].shape)
    out_shape[dim] = max_len
    out_shape = [len(seqs)] + out_shape  # prepend batch dim

    padded = seqs[0].new_full(out_shape, pad_value)

    for i, s in enumerate(seqs):
        # 在最後一維進行padding，複製原始資料到對應位置
        if s.dim() == 1:  # 1D tensor: [L] -> [B, L]
            padded[i, :s.shape[0]] = s
        elif s.dim() == 2:  # 2D tensor: [D, L] -> [B, D, L]
            padded[i, :, :s.shape[1]] = s
        elif s.dim() == 3:  # 3D tensor: [C, D, L] -> [B, C, D, L]
            padded[i, :, :, :s.shape[2]] = s
        else:  # 4D及以上維度
            padded[i, ..., :s.shape[-1]] = s

    return padded, lengths


def collate_finetune_fn(batch):
    """Collate function for :class:`FinetuneDataset`, supporting multi-speaker data.

    Steps performed:
    1. Right-pad ``mel_spec``, ``mel_codes``, ``text_ids``, and ``condition`` along the time dimension so they share a common length
       within the batch.
    2. Return the padded tensors **and** the original sequence lengths so the model or loss function can apply masking.

    Args:
        batch (List[Tuple[Tensor, Tensor, Tensor, Tensor, str]]): A list of samples yielded by
            :meth:`FinetuneDataset.__getitem__`.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, List[str], LongTensor, LongTensor, LongTensor]:
            ``mel_specs`` (B, D, T_max), ``mel_codes`` (B, T_max_codes), ``text_ids`` (B, T_max_text),
            ``conditions`` (B, C, T_max_condition), ``speaker_ids`` (List[str]),
            followed by their respective length tensors ``mel_lengths`` / ``codes_lengths`` / ``text_lengths``.
    """

    mel_specs, mel_codes, text_ids, conditions, speaker_ids = zip(*batch)

    # Remove the extra batch dimension left by data preprocessing for mel_specs and mel_codes (1, ...) -> (...)
    mel_specs = [spec.squeeze(0) if spec.dim() >= 3 and spec.size(0) == 1 else spec for spec in mel_specs]
    mel_codes = [codes.squeeze(0) if codes.dim() >= 2 and codes.size(0) == 1 else codes for codes in mel_codes]

    # Remove the extra batch dimension added to text_ids inside the dataset (1, L) -> (L)
    text_ids = [ids.squeeze(0) if ids.dim() == 2 and ids.size(0) == 1 else ids for ids in text_ids]
    
    # conditions mast not None
    assert all(cond is not None for cond in conditions), "conditions must not be None"
    conditions = [cond.squeeze(0) if cond.dim() >= 3 and cond.size(0) == 1 else cond for cond in conditions]


    # Pad
    mel_specs_padded, mel_lengths = _pad_sequence(list(mel_specs), pad_value=0.0, dim=-1)
    mel_codes_padded, codes_lengths = _pad_sequence(list(mel_codes), pad_value=0, dim=-1)
    text_ids_padded, text_lengths = _pad_sequence(list(text_ids), pad_value=0, dim=-1)
    
    # Stack conditions directly since they all have the same shape [32, 1280]
    conditions_padded = torch.stack(conditions, dim=0)  # [B, 32, 1280]

    return (
        mel_specs_padded, # [B, 100, T_max]
        mel_codes_padded, # [B, T_max_codes]
        text_ids_padded, # [B, T_max]
        conditions_padded, # [B, 32, 1280]
        list(speaker_ids), # [B]
        mel_lengths,
        codes_lengths,
        text_lengths,
    )


def load_finetune_datasets(config: DictConfig, bpe_path: str) -> Tuple[Dataset, Dataset]:
    """Utility helper to load the train/validation datasets for multi-speaker training.

    Args:
        config (DictConfig): Global configuration.
        bpe_path (str): Path to the BPE model file.

    Returns:
        Tuple[Dataset, Dataset]: ``(train_dataset, validation_dataset)``.
    """
    # 讀取說話人資訊
    speaker_info_path = os.path.join(config.train.data_path, "speaker_info.json")
    with open(speaker_info_path, 'r', encoding='utf-8') as f:
        speaker_info_list = json.load(f)
    
    train_manifest_files = []
    valid_manifest_files = []
    speaker_ids = []

    # 收集所有說話人的訓練和驗證資料檔案
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import multiprocessing

    logger.info(f"載入 {len(speaker_info_list)} 個 speaker 的資料...")

    def check_speaker_files(speaker_info):
        """檢查單個 speaker 的檔案是否存在"""
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
    # 使用 set 來去重，避免多個 speaker 指向同一個檔案時重複讀取
    unique_train_files = set()
    unique_valid_files = set()
    valid_speaker_ids = []

    # 使用多線程並行檢查（IO bound，用線程比進程快）
    num_workers = min(32, multiprocessing.cpu_count() * 2)  # 最多 32 個 worker

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(check_speaker_files, info) for info in speaker_info_list]

        for future in tqdm(as_completed(futures), total=len(speaker_info_list),
                          desc="載入 speaker 資料", unit="speaker", ncols=80):
            train_file, valid_file, speaker_id, success = future.result()
            if success:
                unique_train_files.add(train_file)
                unique_valid_files.add(valid_file)
                valid_speaker_ids.append(speaker_id)
            else:
                missing_count += 1

    # 轉換回列表並排序（確保順序一致）
    train_manifest_files = sorted(list(unique_train_files))
    valid_manifest_files = sorted(list(unique_valid_files))

    logger.info(f"✓ 成功載入 {len(valid_speaker_ids)} 個 speaker，"
                f"共 {len(train_manifest_files)} 個唯一的訓練檔案，"
                f"{len(valid_manifest_files)} 個唯一的驗證檔案" +
                (f" (跳過 {missing_count} 個缺失檔案)" if missing_count > 0 else ""))
    
    # 建立資料集
    # 注意：由於檔案已去重，speaker_ids 對應關係不再是 1-to-1，
    # 但 FinetuneDataset 在 lazy 模式下並不依賴這個列表來進行 ID 對應，
    # 而是從 audio 路徑推斷，所以這裡傳 None 是安全的。
    train_dataset = FinetuneDataset(train_manifest_files, bpe_path, None, config)
    valid_dataset = FinetuneDataset(valid_manifest_files, bpe_path, None, config)
    
    return train_dataset, valid_dataset


def load_speaker_conditions(config: DictConfig) -> dict:
    """載入所有說話人的mean_condition。
    
    Args:
        config (DictConfig): Global configuration.
        
    Returns:
        dict: Dictionary mapping speaker_id to mean_condition tensor.
    """
    speaker_info_path = os.path.join(config.train.data_path, "speaker_info.json")
    with open(speaker_info_path, 'r', encoding='utf-8') as f:
        speaker_info_list = json.load(f)
    
    speaker_conditions = {}
    from tqdm import tqdm

    for speaker_info in tqdm(speaker_info_list, desc="載入 speaker conditions", unit="spk", ncols=80):
        speaker_id = speaker_info['speaker']
        medoid_path = speaker_info['medoid_condition']

        if os.path.exists(medoid_path):
            condition = np.load(medoid_path)
            speaker_conditions[speaker_id] = torch.from_numpy(condition).float()
        else:
            raise ValueError(f"Missing medoid_condition.npy for speaker {speaker_id}")

    logger.info(f"✓ 載入 {len(speaker_conditions)} 個 speaker 的 mean conditions")
    return speaker_conditions


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Load finetune datasets.")
    parser.add_argument("--config", type=str, default="finetune_models/config.yaml", help="Path to the configuration file.")
    parser.add_argument("--bpe_model", type=str, default="finetune_models/bpe.model", help="Path to the SentencePiece model.")

    args = parser.parse_args()

    # Load config file
    config = OmegaConf.load(args.config)

    # Load datasets (現在直接傳遞BPE路徑)
    train_dataset, valid_dataset = load_finetune_datasets(config, args.bpe_model)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(valid_dataset)}")

    loader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_finetune_fn)
    sample_batch = next(iter(loader))

    mel_specs, mel_codes, text_ids, conditions, speaker_ids, mel_lens, code_lens, text_lens = sample_batch
    logger.info(f"Sample batch shapes -- mel_specs: {tuple(mel_specs.shape)}, mel_codes: {tuple(mel_codes.shape)}, text_ids: {tuple(text_ids.shape)}")
    if conditions is not None:
        logger.info(f"Conditions shape: {tuple(conditions.shape)}")
    logger.info(f"Speaker IDs: {speaker_ids}")
    logger.info(f"Lengths -- mel: {mel_lens.tolist()}, codes: {code_lens.tolist()}, text: {text_lens.tolist()}")
