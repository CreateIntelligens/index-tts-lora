#!/usr/bin/env python3
"""
音訊特徵提取工具

該指令碼用於處理音訊資料，提取梅爾頻譜特徵、離散程式碼本索引和condition latent，
並計算medoid樣本用於語音合成模型的訓練。

主要功能：
1. 音訊資料預處理和特徵提取
2. 離散變分自編碼器(DVAE)編碼
3. Condition latent提取
4. Medoid樣本計算
5. 訓練/驗證集分割
"""

import argparse
import gc
import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

# 確保專案根目錄在 Python 路徑中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from indextts.utils.feature_extractors import MelSpectrogramFeatures
from indextts.vqvae.xtts_dvae import DiscreteVAE

# 常量定義
DEFAULT_OUTPUT_DIR = "finetune_data/processed_data/"
DEFAULT_MODEL_PATH = "checkpoints/gpt.pth.open_source"
FINETUNE_MODEL_DIR = "finetune_models"
CONFIG_FILENAME = "config.yaml"
METADATA_FILENAME = "metadata.jsonl"
TRAIN_SPLIT_RATIO = 0.9
CONDITION_LATENT_DIM = 32


class AudioDataset(torch.utils.data.Dataset):
    """音頻數據集，用於批次加載和處理"""

    def __init__(self, audio_list_path: str):
        """
        Args:
            audio_list_path: audio_list.txt 文件路徑
        """
        self.samples = []

        # 解析 audio_list.txt
        with open(audio_list_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        wav_path, text = parts
                        if os.path.exists(wav_path):
                            self.samples.append({
                                'wav_path': wav_path,
                                'text': text
                            })
                except Exception as e:
                    logger.warning(f"解析失敗: {line}, {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            dict: {
                'wav_path': str,
                'text': str,
                'audio': torch.Tensor or None,
                'sr': int or None,
                'success': bool
            }
        """
        sample = self.samples[idx]

        try:
            audio, sr = sf.read(sample["wav_path"])
            audio = torch.from_numpy(audio.T if audio.ndim > 1 else audio.reshape(1, -1)).float()
            return {
                'wav_path': sample['wav_path'],
                'text': sample['text'],
                'audio': audio,
                'sr': sr,
                'success': True
            }
        except Exception as e:
            logger.warning(f"加載失敗: {sample['wav_path']}, {e}")
            return {
                'wav_path': sample['wav_path'],
                'text': sample['text'],
                'audio': None,
                'sr': None,
                'success': False
            }


def collate_batch(batch):
    """
    整理批次數據

    Args:
        batch: List[dict] from Dataset.__getitem__

    Returns:
        dict: {
            'wav_paths': List[str],
            'texts': List[str],
            'audios': List[torch.Tensor],
            'srs': List[int],
            'valid_indices': List[int]
        }
    """
    wav_paths = []
    texts = []
    audios = []
    srs = []
    valid_indices = []

    for i, item in enumerate(batch):
        if item['success']:
            wav_paths.append(item['wav_path'])
            texts.append(item['text'])
            audios.append(item['audio'])
            srs.append(item['sr'])
            valid_indices.append(i)

    return {
        'wav_paths': wav_paths,
        'texts': texts,
        'audios': audios,
        'srs': srs,
        'valid_indices': valid_indices
    }


def get_host_uid_gid() -> Tuple[int, int]:
    """嘗試推斷宿主機的 UID/GID 以修復檔案權限"""
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


def fix_permissions(path: Union[str, Path]) -> None:
    """將檔案或目錄權限調整為宿主使用者"""
    target = Path(path)
    try:
        uid, gid = get_host_uid_gid()
        os.chown(target, uid, gid)
    except PermissionError:
        logger.warning(f"無法調整權限 (PermissionError): {target}")
    except FileNotFoundError:
        logger.warning(f"無法調整權限 (遺失): {target}")
    except Exception as exc:
        logger.warning(f"無法調整權限: {target} -> {exc}")


class AudioProcessor:
    """音訊處理器類，封裝音訊特徵提取相關功能"""
    
    def __init__(self, dvae: DiscreteVAE, mel_config: Dict, device: str = 'cuda'):
        self.dvae = dvae
        self.mel_config = mel_config
        self.device = device
        self.mel_feature = MelSpectrogramFeatures(**mel_config).to(self.device)
        self.sample_rate = self.mel_config['sample_rate']
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

    def _get_resampler(self, source_sr: int) -> torchaudio.transforms.Resample:
        """取得對應輸入取樣率的重取樣器並搬到正確裝置"""
        if source_sr not in self._resamplers:
            resampler = torchaudio.transforms.Resample(
                orig_freq=source_sr,
                new_freq=self.sample_rate
            ).to(self.device)
            self._resamplers[source_sr] = resampler
        return self._resamplers[source_sr]
    
    @torch.no_grad()
    def process_audio_data(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        sr: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        處理音訊資料，包括提取梅爾頻譜特徵、獲取離散程式碼本索引等。

        Args:
            audio: 輸入的音訊資料
            sr: 音訊的取樣率

        Returns:
            處理後的音訊資料、梅爾頻譜特徵和離散程式碼本索引
        """
        # 資料型別轉換
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)

        audio = audio.to(self.device)
        
        # 重取樣
        if sr != self.sample_rate:
            resampler = self._get_resampler(sr)
            audio = resampler(audio)
        
        # 提取梅爾頻譜特徵
        mel = self.mel_feature(audio)
        
        # 獲取離散程式碼本索引
        codes = self.dvae.get_codebook_indices(mel)
        
        # 處理音訊維度
        if audio.ndim > 1 and audio.shape[0] == 1:
            audio = audio.squeeze(0)
        
        return audio, mel, codes


class ConditionExtractor:
    """Condition latent提取器類"""
    
    def __init__(self, model, device: str = 'cuda'):
        self.model = model
        self.device = device
    
    def extract_condition_latent(
        self, 
        mel: torch.Tensor
    ) -> np.ndarray:
        """
        從梅爾頻譜中提取condition latent
        
        Args:
            mel: 梅爾頻譜特徵 (1, mel_dim, T) 或 (mel_dim, T)
        
        Returns:
            condition latent (32, dim)
        """
        if self.model is None:
            raise ValueError("模型不能為None，需要傳入已載入的UnifiedVoice模型")
        
        # 確保mel資料格式正確
        if isinstance(mel, np.ndarray):
            mel = torch.from_numpy(mel)
        
        # 確保mel是3維的 (1, mel_dim, T)
        if mel.ndim == 2:
            mel = mel.unsqueeze(0)
        elif mel.ndim == 3 and mel.shape[0] == 1:
            pass
        else:
            raise ValueError(f"不支援的mel形狀: {mel.shape}")
        
        mel = mel.to(self.device)
        mel_length = torch.tensor([mel.shape[-1]], device=self.device)
        
        # 使用UnifiedVoice模型提取condition latent
        with torch.no_grad():
            condition = self.model.get_conditioning(mel, mel_length)
            condition = condition.squeeze(0).cpu().float().numpy()

        return condition

    def extract_condition_latent_batch(self, mels: List[torch.Tensor]) -> List[np.ndarray]:
        """
        批次提取 condition latent

        Args:
            mels: List of mel spectrograms, each [1, mel_bins, T] or [mel_bins, T]

        Returns:
            List of condition latents, each (32, dim)
        """
        if self.model is None:
            raise ValueError("模型不能為None，需要傳入已載入的UnifiedVoice模型")

        # 標準化輸入格式
        normalized_mels = []
        for mel in mels:
            if isinstance(mel, np.ndarray):
                mel = torch.from_numpy(mel)
            # 確保是3維
            if mel.ndim == 2:
                mel = mel.unsqueeze(0)
            elif mel.ndim == 3 and mel.shape[0] == 1:
                pass
            else:
                raise ValueError(f"不支援的mel形狀: {mel.shape}")
            normalized_mels.append(mel.squeeze(0))  # [mel_bins, T]

        # 找到最大長度並 padding
        max_len = max(mel.shape[-1] for mel in normalized_mels)
        padded_mels = []
        mel_lengths = []

        for mel in normalized_mels:
            current_len = mel.shape[-1]
            mel_lengths.append(current_len)

            if current_len < max_len:
                pad_len = max_len - current_len
                mel = torch.nn.functional.pad(mel, (0, pad_len))

            padded_mels.append(mel)

        # Stack 成 batch
        batch_mels = torch.stack(padded_mels).to(self.device)  # [B, mel_bins, T]
        mel_lengths = torch.tensor(mel_lengths, device=self.device)

        # 批次推理
        with torch.no_grad():
            batch_conditions = self.model.get_conditioning(batch_mels, mel_lengths)
            # 轉回 list
            conditions = [
                cond.cpu().float().numpy()
                for cond in batch_conditions
            ]

        return conditions


class MedoidCalculator:
    """Medoid計算器類"""
    
    @staticmethod
    def _prepare_medoid_devices(
        default_device: str,
        medoid_devices: Optional[Union[str, List[Union[str, int]]]]
    ) -> List[torch.device]:
        """解析使用者傳入的裝置列表"""
        if medoid_devices is None:
            candidates: List[str] = [default_device]
        elif isinstance(medoid_devices, str):
            candidates = [dev.strip() for dev in medoid_devices.split(',') if dev.strip()]
        else:
            candidates = []
            for dev in medoid_devices:
                if isinstance(dev, int):
                    candidates.append(f"cuda:{dev}")
                else:
                    candidates.append(str(dev))

        parsed_devices: List[torch.device] = []
        seen: set = set()
        for dev in candidates:
            if not dev:
                continue
            if dev.isdigit():
                dev = f"cuda:{dev}"
            try:
                torch_device = torch.device(dev)
            except Exception:
                logger.warning(f"無法解析裝置 {dev}，已忽略")
                continue

            if torch_device.type == 'cuda' and not torch.cuda.is_available():
                logger.warning(f"裝置 {dev} 不可用，改用 CPU")
                continue

            key = str(torch_device)
            if key in seen:
                continue
            seen.add(key)
            parsed_devices.append(torch_device)

        if not parsed_devices:
            logger.warning("未提供有效的 GPU 裝置，將使用 CPU 計算 medoid")
            parsed_devices = [torch.device('cpu')]

        return parsed_devices

    @staticmethod
    def compute_distance_matrix(
        data: np.ndarray,
        metric: str = 'euclidean',
        device: str = 'cpu',
        block_size: int = 512,
        medoid_devices: Optional[Union[str, List[Union[str, int]]]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        以區塊方式計算距離矩陣，支援多 GPU 平行運算

        Args:
            data: (N, features) 數組
            metric: 距離度量方式
            device: 預設裝置
            block_size: 每個計算區塊的樣本數
            medoid_devices: 自訂裝置列表

        Returns:
            (距離矩陣, 每個樣本距離和)
        """
        n = data.shape[0]
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.float64)

        block_size = max(1, int(block_size))
        block_size = min(block_size, n)
        row_chunks = [(start, min(start + block_size, n)) for start in range(0, n, block_size)]
        num_chunks = len(row_chunks)

        devices = MedoidCalculator._prepare_medoid_devices(device, medoid_devices)
        data_tensor = torch.from_numpy(data).to(torch.float32)

        dist_matrix = np.zeros((n, n), dtype=np.float32)
        dist_sums = np.zeros(n, dtype=np.float64)
        chunk_locks = [threading.Lock() for _ in row_chunks]
        task_queue: Queue = Queue()

        for idx in range(num_chunks):
            task_queue.put(idx)

        def worker(device_obj: torch.device) -> None:
            nonlocal metric
            while True:
                try:
                    row_idx = task_queue.get_nowait()
                except Empty:
                    break

                start_i, end_i = row_chunks[row_idx]
                chunk_i = data_tensor[start_i:end_i].to(device_obj, non_blocking=True)

                try:
                    for col_idx in range(row_idx, num_chunks):
                        start_j, end_j = row_chunks[col_idx]
                        chunk_j = data_tensor[start_j:end_j].to(device_obj, non_blocking=True)

                        if metric == 'euclidean':
                            block = torch.cdist(chunk_i, chunk_j, p=2)
                        elif metric == 'cosine':
                            norm_i = F.normalize(chunk_i, dim=1)
                            norm_j = F.normalize(chunk_j, dim=1)
                            block = 1 - torch.matmul(norm_i, norm_j.transpose(0, 1))
                        else:
                            raise ValueError(f"不支援的距離度量: {metric}")

                        block_np = block.cpu().numpy().astype(np.float32)
                        dist_matrix[start_i:end_i, start_j:end_j] = block_np
                        dist_sums[start_i:end_i] += block_np.sum(axis=1, dtype=np.float64)

                        if col_idx != row_idx:
                            dist_matrix[start_j:end_j, start_i:end_i] = block_np.T
                            col_contrib = block_np.sum(axis=0, dtype=np.float64)
                            with chunk_locks[col_idx]:
                                dist_sums[start_j:end_j] += col_contrib

                        del chunk_j, block, block_np
                finally:
                    del chunk_i
                    task_queue.task_done()

        threads = []
        for dev in devices:
            thread = threading.Thread(target=worker, args=(dev,), daemon=True)
            thread.start()
            threads.append(thread)

        task_queue.join()
        for thread in threads:
            thread.join()

        return dist_matrix, dist_sums
    
    @classmethod
    def find_medoid_condition(
        cls,
        condition_files: List[str],
        distance_metric: str = 'euclidean',
        device: str = 'cpu',
        batch_size: int = 512,
        max_samples: int = 5000,
        gpu_ids: Optional[List[int]] = None,
        medoid_devices: Optional[List[str]] = None
    ) -> Dict:
        """
        找出condition latent分佈中心的樣本（medoid）

        Args:
            condition_files: condition檔案路徑列表
            distance_metric: 距離度量方式
            device: 計算裝置 ('cpu' 或 'cuda')
            batch_size: 距離矩陣計算的批次大小（行）
            max_samples: 列分塊大小（用於記憶體優化）
            gpu_ids: GPU ID 列表，用於多 GPU 並行（例如 [0,1,2,3]）

        Returns:
            medoid資訊字典
        """
        total_files = len(condition_files)
        device_list = medoid_devices
        if device_list is None and gpu_ids:
            device_list = [f"cuda:{gid}" for gid in gpu_ids]
        use_multi_gpu = device_list is not None and len(device_list) > 1

        if use_multi_gpu:
            logger.info(f"開始計算medoid，共 {total_files} 個樣本（多 GPU 區塊計算）")
            logger.info(f"使用裝置: {device_list}")
        else:
            logger.info(f"開始計算medoid，共 {total_files} 個樣本（單裝置區塊計算）")
        
        # 讀取所有condition latent
        conditions = []
        condition_infos = []
        
        for condition_path in condition_files:
            if not os.path.exists(condition_path):
                logger.warning(f"Condition檔案不存在: {condition_path}，跳過")
                continue
            
            try:
                cond = np.load(condition_path)
                conditions.append(cond)
                condition_infos.append({
                    'index': len(conditions) - 1,
                    'condition_path': condition_path
                })
            except Exception as e:
                logger.error(f"載入condition檔案失敗: {condition_path}, 錯誤: {e}")
                continue
        
        if len(conditions) == 0:
            raise RuntimeError('沒有找到有效的condition檔案')
        
        logger.info(f"成功讀取{len(conditions)}個condition latent")
        
        # 將所有condition堆疊成一個數組
        conditions_array = np.stack(conditions, axis=0)
        logger.info(f"Conditions陣列形狀: {conditions_array.shape}")
        
        # 將資料展平以便計算距離
        N, H, W = conditions_array.shape
        data_flat = conditions_array.reshape(N, H * W)
        
        # 計算距離矩陣（使用分批避免 OOM）
        logger.info("計算距離矩陣...")
        block_size = max_samples if max_samples > 0 else batch_size
        block_size = max(1, min(batch_size, block_size))
        if block_size != batch_size:
            logger.info(f"採用區塊大小 {block_size} (由 batch_size={batch_size}, chunk_size={max_samples} 計算)")
        else:
            logger.info(f"採用區塊大小 {block_size}")

        dist_matrix, dist_sums = cls.compute_distance_matrix(
            data_flat,
            distance_metric,
            device,
            block_size=block_size,
            medoid_devices=device_list
        )

        # 找出medoid
        logger.info("尋找medoid...")
        medoid_idx = np.argmin(dist_sums)
        medoid_info = condition_infos[medoid_idx]
        medoid_condition = conditions[medoid_idx]
        
        logger.info(f"找到medoid: 索引{medoid_idx}")
        logger.info(f"Medoid到所有樣本的距離之和: {dist_sums[medoid_idx]:.6f}")
        
        # 計算統計資訊
        stats = {
            'total_samples': len(conditions),
            'medoid_index': int(medoid_idx),
            'medoid_distance_sum': float(dist_sums[medoid_idx]),
            'distance_metric': distance_metric,
            'distance_stats': {
                'mean': float(np.mean(dist_sums)),
                'std': float(np.std(dist_sums)),
                'min': float(np.min(dist_sums)),
                'max': float(np.max(dist_sums))
            }
        }
        
        result = {
            'medoid_info': medoid_info,
            'medoid_condition': medoid_condition,
            'statistics': stats,
            'distance_matrix': dist_matrix
        }

        # 清理臨時變數和 GPU 緩存
        del conditions_array, data_flat, dist_matrix, dist_sums
        gc.collect()

        return result


def load_unified_voice_model(
    config_path: str, 
    model_path: str, 
    device: str = 'cuda'
):
    """
    載入UnifiedVoice模型
    
    Args:
        config_path: 配置檔案路徑
        model_path: 模型檢查點路徑
        device: 計算裝置
    
    Returns:
        載入好的UnifiedVoice模型
    """
    sys.path.append('.')
    from indextts.gpt.model import UnifiedVoice
    
    # 載入配置
    cfg = OmegaConf.load(config_path)
    
    # 建立模型
    model = UnifiedVoice(**cfg.gpt)
    
    # 載入預訓練權重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型檢查點檔案未找到: {model_path}")
    
    logger.info(f"正在載入UnifiedVoice模型檢查點: {model_path}")
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(
        state['model'] if 'model' in state else state, 
        strict=True
    )
    model.eval().to(device)
    
    logger.info("UnifiedVoice模型載入完成")
    return model


def parse_audio_line(line: str) -> Tuple[str, str]:
    """解析音訊列表檔案中的一行"""
    line = line.strip()
    if not line:
        raise ValueError("空行")
    
    if "\t" in line:
        return line.split("\t", 1)
    elif "|" in line:
        return line.split("|", 1)
    else:
        raise ValueError(f"不支援的格式: {line}")


def save_medoid_results(
    medoid_result: Dict, 
    output_dir: str
) -> None:
    """儲存medoid計算結果"""
    # 儲存medoid資訊
    medoid_info_path = os.path.join(output_dir, "medoid_info.json")
    with open(medoid_info_path, 'w', encoding='utf-8') as f:
        json.dump({
            'medoid_info': medoid_result['medoid_info'],
            'statistics': medoid_result['statistics'],
            'medoid_condition_shape': list(medoid_result['medoid_condition'].shape)
        }, f, ensure_ascii=False, indent=2)
    fix_permissions(medoid_info_path)
    
    # 儲存medoid condition
    medoid_condition_path = os.path.join(output_dir, "medoid_condition.npy")
    np.save(medoid_condition_path, medoid_result['medoid_condition'])
    fix_permissions(medoid_condition_path)
    
    # 儲存距離矩陣
    distance_matrix_path = os.path.join(output_dir, "distance_matrix.npy")
    np.save(distance_matrix_path, medoid_result['distance_matrix'])
    fix_permissions(distance_matrix_path)
    
    logger.info("Medoid計算完成:")
    logger.info(f"  - Medoid資訊: {medoid_info_path}")
    logger.info(f"  - Medoid condition: {medoid_condition_path}")
    logger.info(f"  - 距離矩陣: {distance_matrix_path}")
    logger.info(f"  - Medoid樣本: {medoid_result['medoid_info']['condition_path']}")


def split_dataset(
    metadata_file: str, 
    output_dir: str
) -> Tuple[str, str]:
    """分割資料集為訓練集和驗證集"""
    with open(metadata_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 隨機打亂資料
    np.random.shuffle(lines)

    # 計算分割點
    valid_size = int(len(lines) * (1 - TRAIN_SPLIT_RATIO))

    
    # 分割資料
    valid_lines = lines[-valid_size:]
    train_lines = lines[:-valid_size]
    
    # 儲存訓練集
    train_file = os.path.join(output_dir, 'metadata_train.jsonl')
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    fix_permissions(train_file)
    
    # 儲存驗證集
    valid_file = os.path.join(output_dir, 'metadata_valid.jsonl')
    with open(valid_file, 'w', encoding='utf-8') as f:
        f.writelines(valid_lines)
    fix_permissions(valid_file)
    
    logger.info("資料集分割完成:")
    logger.info(f"  - 訓練集: {train_file} ({len(train_lines)}條資料)")
    logger.info(f"  - 驗證集: {valid_file} ({len(valid_lines)}條資料)")
    
    return train_file, valid_file


def check_and_generate_speaker_info(output_dir: str) -> None:
    """檢查所有 part 是否處理完成，如果是則生成 speaker_info.json"""
    import glob
    from collections import defaultdict

    processed_data_dir = os.path.dirname(output_dir)

    # 檢查有多少個 audio_list part 檔案
    audio_list_dir = os.path.join(processed_data_dir, "..", "audio_list")
    audio_list_files = glob.glob(os.path.join(audio_list_dir, "audio_list_part_*.txt"))
    total_parts = len(audio_list_files)

    # 檢查有多少個已處理的 part 目錄
    processed_dirs = glob.glob(os.path.join(processed_data_dir, 'audio_list_part_*'))
    completed_parts = len([d for d in processed_dirs if os.path.exists(os.path.join(d, 'metadata.jsonl'))])

    logger.info(f"進度: {completed_parts}/{total_parts} 個 part 已處理完成")

    if completed_parts < total_parts:
        logger.info("等待其他 part 處理完成...")
        return

    # 所有 part 都處理完了，生成 speaker_info.json
    logger.info("🎉 所有 part 已處理完成，開始生成 speaker_info.json...")

    speaker_info_file = os.path.join(processed_data_dir, 'speaker_info.json')
    lock_file = speaker_info_file + '.lock'

    # 使用檔案鎖確保只有一個進程生成
    import fcntl
    try:
        with open(lock_file, 'w') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            # 檢查是否已經生成過了
            if os.path.exists(speaker_info_file):
                logger.info("speaker_info.json 已存在，跳過生成")
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                return

            # 收集所有 speaker 資訊
            speaker_data = defaultdict(list)
            total_samples = 0

            for proc_dir in sorted(processed_dirs):
                metadata_file = os.path.join(proc_dir, 'metadata.jsonl')
                if not os.path.exists(metadata_file):
                    continue

                with open(metadata_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        audio_path = data["audio"]
                        path_parts = Path(audio_path).parts

                        # 提取 speaker ID
                        data_idx = path_parts.index("data") if "data" in path_parts else -1
                        if data_idx >= 0 and len(path_parts) > data_idx + 2:
                            drama_name = path_parts[data_idx + 1]
                            character_id = path_parts[data_idx + 2]
                            speaker_id = f"{drama_name}_{character_id}"
                        else:
                            speaker_id = os.path.basename(proc_dir)

                        speaker_data[speaker_id].append(data)
                        total_samples += 1

            # 生成 speaker_info
            speaker_info_list = []
            for speaker_id in sorted(speaker_data.keys()):
                lines = speaker_data[speaker_id]
                total_duration = sum(line["duration"] for line in lines)
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

            # 儲存
            with open(speaker_info_file, 'w', encoding='utf-8') as f:
                json.dump(speaker_info_list, f, ensure_ascii=False, indent=4)
            fix_permissions(speaker_info_file)

            logger.info(f"✅ speaker_info.json 生成完成")
            logger.info(f"   - 總 speaker 數: {len(speaker_info_list)}")
            logger.info(f"   - 總樣本數: {sum(s['sample_num'] for s in speaker_info_list)}")

            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    except BlockingIOError:
        # 其他進程正在生成，跳過
        logger.info("其他進程正在生成 speaker_info.json，跳過")
        return


def save_speaker_info(
    audio_list_path: str,
    output_dir: str,
    lines: List[str]
) -> None:
    """儲存說話人資訊"""
    # 從 WAV 路徑提取 speaker ID (人物目錄)
    first_line = json.loads(lines[0])
    wav_path = first_line["audio"]  # metadata 中的欄位名稱是 "audio"
    path_parts = Path(wav_path).parts
    data_idx = path_parts.index("data") if "data" in path_parts else -1
    if data_idx >= 0 and len(path_parts) > data_idx + 2:
        drama_name = path_parts[data_idx + 1]
        character_id = path_parts[data_idx + 2]
        speaker_id = f"{drama_name}_{character_id}"
    else:
        speaker_id = Path(audio_list_path).stem

    total_duration = sum(json.loads(line)["duration"] for line in lines)
    speaker_info = {
        "speaker": speaker_id,
        "avg_duration": total_duration / len(lines),
        "sample_num": len(lines),
        "total_duration": total_duration,
        "train_jsonl": os.path.abspath(os.path.join(output_dir, "metadata_train.jsonl")),
        "valid_jsonl": os.path.abspath(os.path.join(output_dir, "metadata_valid.jsonl")),
        "medoid_condition": os.path.abspath(os.path.join(output_dir, "medoid_condition.npy")),
    }
    
    speaker_info_file = os.path.join(output_dir, "..", 'speaker_info.json')
    lock_file = speaker_info_file + '.lock'

    # 使用檔案鎖避免多 GPU 同時寫入
    import fcntl
    import time

    max_retries = 10
    for attempt in range(max_retries):
        try:
            # 取得鎖
            with open(lock_file, 'w') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

                # 讀取現有資訊或建立新列表
                if os.path.exists(speaker_info_file):
                    try:
                        with open(speaker_info_file, 'r', encoding='utf-8') as f:
                            speaker_info_list = json.load(f)
                        logger.debug(f"讀取到 {len(speaker_info_list)} 個現有 speaker")
                    except (json.JSONDecodeError, IOError) as e:
                        # 如果檔案損壞，重新建立
                        logger.warning(f"speaker_info.json 損壞，重新建立: {e}")
                        speaker_info_list = []
                else:
                    logger.debug("speaker_info.json 不存在，建立新檔案")
                    speaker_info_list = []

                # 移除舊的同名 speaker（如果存在）
                current_speaker_id = speaker_info["speaker"]
                speaker_info_list = [s for s in speaker_info_list if s["speaker"] != current_speaker_id]

                # 加入新的 speaker 資訊
                speaker_info_list.append(speaker_info)

                # 儲存更新後的資訊
                with open(speaker_info_file, 'w', encoding='utf-8') as f:
                    json.dump(speaker_info_list, f, ensure_ascii=False, indent=4)
                fix_permissions(speaker_info_file)

                # 釋放鎖
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                break
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"寫入 speaker_info.json 失敗 (嘗試 {attempt+1}/{max_retries}): {e}")
                time.sleep(0.5)
            else:
                raise


def setup_models(
    config: OmegaConf, 
    args: argparse.Namespace
) -> Tuple[DiscreteVAE, Optional[object]]:
    """設定和載入模型"""
    # 載入DiscreteVAE模型
    logger.info("正在載入 DiscreteVAE 模型...")
    dvae = DiscreteVAE(**config.vqvae)
    
    dvae_checkpoint_path = os.path.join(FINETUNE_MODEL_DIR, config.dvae_checkpoint)
    pre_trained_dvae = torch.load(
        dvae_checkpoint_path, 
        map_location=args.device, 
        weights_only=True
    )
    dvae.load_state_dict(
        pre_trained_dvae["model"] if "model" in pre_trained_dvae else pre_trained_dvae,
        strict=True
    )
    dvae.eval()
    dvae.to(args.device)
    del pre_trained_dvae
    
    # 載入UnifiedVoice模型（如果需要）
    unified_voice_model = None
    if args.extract_condition:
        config_path = os.path.join(FINETUNE_MODEL_DIR, CONFIG_FILENAME)
        unified_voice_model = load_unified_voice_model(
            config_path, args.model_path, args.device
        )
    
    return dvae, unified_voice_model


def process_audio_files(
    args: argparse.Namespace,
    config: OmegaConf,
    audio_processor: AudioProcessor,
    condition_extractor: Optional[ConditionExtractor],
    output_dir: str,
    batch_size: int = 16,
    num_workers: int = 8
) -> Tuple[str, List[str]]:
    """
    批次處理音訊檔案列表

    Args:
        batch_size: 批次大小（建議 8-16）
        num_workers: DataLoader worker 數量（建議 4-8）
    """
    metadata_file = os.path.join(output_dir, METADATA_FILENAME)
    metadata_tmp_file = metadata_file + ".tmp"
    condition_files = []
    metadata_path = Path(metadata_file)
    metadata_tmp_path = Path(metadata_tmp_file)

    # 如果存在舊的 tmp 文件，先刪除
    if metadata_tmp_path.exists():
        metadata_tmp_path.unlink()

    # 創建 Dataset 和 DataLoader
    dataset = AudioDataset(args.audio_list)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    try:
        # 批次處理
        with tqdm(total=len(dataset), desc="處理音訊檔案") as pbar:
            for batch in dataloader:
                # 跳過空批次
                if len(batch['valid_indices']) == 0:
                    pbar.update(batch_size)
                    continue

                # 批次處理音頻
                processed_batch = []
                for audio, sr, wav_path, text in zip(
                    batch['audios'],
                    batch['srs'],
                    batch['wav_paths'],
                    batch['texts']
                ):
                    try:
                        # 處理音訊資料
                        processed_audio, mel, codes = audio_processor.process_audio_data(audio, sr)

                        processed_batch.append({
                            'wav_path': wav_path,
                            'text': text,
                            'processed_audio': processed_audio,
                            'mel': mel,
                            'codes': codes,
                            'success': True
                        })
                    except Exception as e:
                        logger.error(f"處理音訊失敗: {wav_path}, {e}")
                        processed_batch.append({
                            'wav_path': wav_path,
                            'success': False
                        })

                # 批次提取 condition（如果需要）
                if condition_extractor is not None:
                    # 收集所有成功的 mel
                    mels_batch = [item['mel'] for item in processed_batch if item['success']]

                    if len(mels_batch) > 0:
                        try:
                            # 批次推理 condition
                            conditions = condition_extractor.extract_condition_latent_batch(mels_batch)

                            # 分配回各個樣本
                            cond_idx = 0
                            for item in processed_batch:
                                if item['success']:
                                    item['condition'] = conditions[cond_idx]
                                    cond_idx += 1
                        except Exception as e:
                            logger.error(f"批次提取 condition 失敗: {e}")
                            # 降級到逐個處理
                            for item in processed_batch:
                                if item['success']:
                                    try:
                                        item['condition'] = condition_extractor.extract_condition_latent(item['mel'])
                                    except Exception as e:
                                        logger.error(f"提取 condition 失敗: {item['wav_path']}, {e}")
                                        item['success'] = False

                # 批次保存結果
                for item in processed_batch:
                    if not item['success']:
                        continue

                    try:
                        # 計算音訊時長
                        duration = item['processed_audio'].shape[-1] / config.dataset.mel.sample_rate

                        # 保存特徵文件
                        base_name = os.path.basename(item['wav_path'])
                        out_codebook = os.path.join(output_dir, f"{base_name}_codes.npy")
                        out_mel = os.path.join(output_dir, f"{base_name}_mel.npy")

                        np.save(out_codebook, item['codes'].detach().cpu().numpy())
                        np.save(out_mel, item['mel'].detach().cpu().numpy())
                        fix_permissions(out_codebook)
                        fix_permissions(out_mel)

                        # 保存 condition
                        condition_path = None
                        if 'condition' in item:
                            condition_path = os.path.join(output_dir, f"{base_name}_condition.npy")
                            np.save(condition_path, item['condition'])
                            fix_permissions(condition_path)
                            condition_files.append(condition_path)

                        # 寫入元資料到 tmp 文件
                        data_entry = {
                            "audio": item['wav_path'],
                            "text": item['text'],
                            "codes": os.path.abspath(out_codebook),
                            "mels": os.path.abspath(out_mel),
                            "duration": round(duration, 4)
                        }

                        if condition_path:
                            data_entry["condition"] = os.path.abspath(condition_path)

                        with open(metadata_tmp_file, "a", encoding="utf-8") as out_f:
                            out_f.write(json.dumps(data_entry, ensure_ascii=False) + "\n")

                    except Exception as e:
                        logger.error(f"保存失敗: {item['wav_path']}, {e}")

                # 更新進度
                pbar.update(len(batch['audios']))
        
        # 成功完成後，將 tmp 文件重命名為正式文件
        if metadata_tmp_path.exists():
            metadata_tmp_path.rename(metadata_path)
            fix_permissions(metadata_path)
            logger.info(f"元資料寫入完成: {metadata_file}")

    except Exception as e:
        logger.error(f"處理過程中發生錯誤: {e}")
        # 發生錯誤時保留 tmp 文件以便除錯，但不生成正式文件
        raise
    finally:
        # 清理 DataLoader 和 worker 進程
        del dataloader
        del dataset
        gc.collect()

    return metadata_file, condition_files


def main():
    """主函式"""
    parser = argparse.ArgumentParser(
        description="Process audio data using DiscreteVAE and find medoid condition."
    )
    parser.add_argument(
        "--audio_list", 
        type=str, 
        required=True, 
        help="Path to the input audio list file."
    )
    parser.add_argument(
        "--extract_condition", 
        action="store_true", 
        help="Extract condition latents and find medoid."
    )
    parser.add_argument(
        "--distance_metric", 
        default='euclidean', 
        choices=['euclidean', 'cosine'], 
        help="Distance metric for medoid calculation."
    )
    parser.add_argument(
        "--output_dir", 
        default=DEFAULT_OUTPUT_DIR, 
        help="Output directory for processed data."
    )
    parser.add_argument(
        "--model_path", 
        default=DEFAULT_MODEL_PATH, 
        help="Path to the UnifiedVoice model checkpoint."
    )
    parser.add_argument(
        "--device",
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device to use for computation."
    )
    parser.add_argument(
        "--medoid_devices",
        type=str,
        default=None,
        help="Comma-separated list of devices for medoid calculation (e.g., 'cuda:0,cuda:1')."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="批次大小（建議 8-16，根據 GPU 記憶體調整）"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader worker 數量（建議 4-8）"
    )

    args = parser.parse_args()
    
    # 檢查配置檔案
    config_path = os.path.join(FINETUNE_MODEL_DIR, CONFIG_FILENAME)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置檔案未找到: {config_path}")
    
    # 載入配置
    config = OmegaConf.load(config_path)
    
    # 設定輸出目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_list_name = Path(args.audio_list).stem
    output_dir = os.path.join(args.output_dir, f"{audio_list_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    fix_permissions(Path(output_dir).parent)
    fix_permissions(output_dir)
    
    # 設定模型
    dvae, unified_voice_model = setup_models(config, args)
    
    # 建立處理器
    audio_processor = AudioProcessor(dvae, config.dataset.mel, args.device)
    condition_extractor = None
    if unified_voice_model is not None:
        condition_extractor = ConditionExtractor(unified_voice_model, args.device)
    
    # 處理音訊檔案
    metadata_file, condition_files = process_audio_files(
        args, config, audio_processor, condition_extractor, output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    logger.info(f"特徵提取完成，結果儲存在: {output_dir}")
    
    # 計算medoid（如果需要）
    if args.extract_condition and condition_files:
        logger.info("開始計算medoid condition...")
        total_samples = len(condition_files)
        logger.info(f"共 {total_samples} 個樣本，使用分批計算（無採樣，完整精確）")

        # 從配置文件讀取 medoid 計算參數
        config_file_path = "scripts/config.yaml"
        batch_size = 1000  # 默認值
        chunk_size = 2000  # 默認值

        if os.path.exists(config_file_path):
            try:
                medoid_config = OmegaConf.load(config_file_path)
                if hasattr(medoid_config, 'medoid_batch_size'):
                    batch_size = medoid_config.medoid_batch_size
                if hasattr(medoid_config, 'medoid_chunk_size'):
                    chunk_size = medoid_config.medoid_chunk_size
                logger.info(f"從配置文件讀取: medoid_batch_size={batch_size}, medoid_chunk_size={chunk_size}")
            except Exception as e:
                logger.warning(f"讀取配置文件失敗，使用默認值: {e}")
        else:
            logger.info(f"使用默認值: batch_size={batch_size}, chunk_size={chunk_size}")

        # 檢測可用的 GPU / 裝置
        gpu_ids = None
        medoid_device_list: Optional[List[str]] = None
        if args.medoid_devices:
            medoid_device_list = [dev.strip() for dev in args.medoid_devices.split(',') if dev.strip()]
            if medoid_device_list:
                logger.info(f"使用使用者指定裝置計算 medoid: {medoid_device_list}")

        if medoid_device_list is None and torch.cuda.is_available() and args.device.startswith('cuda'):
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                gpu_ids = list(range(num_gpus))
                medoid_device_list = [f"cuda:{gid}" for gid in gpu_ids]
                logger.info(f"檢測到 {num_gpus} 張 GPU，使用多 GPU 並行計算 medoid")
            else:
                logger.info("只有 1 張 GPU，使用單 GPU 計算 medoid")

        try:
            medoid_result = MedoidCalculator.find_medoid_condition(
                condition_files, args.distance_metric, args.device,
                batch_size=batch_size, max_samples=chunk_size,
                gpu_ids=gpu_ids,
                medoid_devices=medoid_device_list
            )
            save_medoid_results(medoid_result, output_dir)

            # 清理 GPU 緩存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"GPU 記憶體不足: {e}")
                logger.info("降級使用 CPU 計算 medoid...")
                # 降級到 CPU
                try:
                    torch.cuda.empty_cache()
                    medoid_result = MedoidCalculator.find_medoid_condition(
                        condition_files, args.distance_metric, 'cpu',
                        batch_size=batch_size, max_samples=chunk_size,
                        gpu_ids=None,
                        medoid_devices=['cpu']
                    )
                    save_medoid_results(medoid_result, output_dir)
                except Exception as e2:
                    logger.error(f"計算medoid時出錯: {e2}")
            else:
                raise
        except Exception as e:
            logger.error(f"計算medoid時出錯: {e}")
    
    # 分割資料集
    split_dataset(metadata_file, output_dir)

    # 檢查是否所有 part 都處理完了，如果是則自動生成 speaker_info.json
    check_and_generate_speaker_info(output_dir)

    # 清理 GPU 記憶體
    logger.info("清理 GPU 記憶體...")

    # 同步所有 GPU 操作
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 刪除模型和處理器
    del dvae
    del unified_voice_model
    del audio_processor
    del condition_extractor

    # 強制垃圾回收
    gc.collect()

    # 清空 GPU 緩存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 再次同步確保清理完成
        torch.cuda.synchronize()

    logger.info("記憶體清理完成")


if __name__ == "__main__":
    main()
