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
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

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
        self.mel_feature = MelSpectrogramFeatures(**mel_config)
    
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
        
        # 重取樣
        if sr != self.mel_config['sample_rate']:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, 
                new_freq=self.mel_config['sample_rate']
            )
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


class MedoidCalculator:
    """Medoid計算器類"""
    
    @staticmethod
    def compute_distance_matrix(
        data: np.ndarray, 
        metric: str = 'euclidean'
    ) -> np.ndarray:
        """
        計算距離矩陣
        
        Args:
            data: 資料矩陣 (N, features)
            metric: 距離度量方式 ('euclidean' 或 'cosine')
        
        Returns:
            距離矩陣 (N, N)
        """
        n = data.shape[0]
        dist_matrix = np.zeros((n, n))
        
        for i in tqdm(range(n), desc="計算距離矩陣"):
            for j in range(i + 1, n):
                if metric == 'euclidean':
                    dist = np.linalg.norm(data[i] - data[j])
                elif metric == 'cosine':
                    dist = 1 - np.dot(data[i], data[j]) / (
                        np.linalg.norm(data[i]) * np.linalg.norm(data[j])
                    )
                else:
                    raise ValueError(f"不支援的距離度量: {metric}")
                
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        return dist_matrix
    
    @classmethod
    def find_medoid_condition(
        cls, 
        condition_files: List[str], 
        distance_metric: str = 'euclidean'
    ) -> Dict:
        """
        找出condition latent分佈中心的樣本（medoid）
        
        Args:
            condition_files: condition檔案路徑列表
            distance_metric: 距離度量方式
        
        Returns:
            medoid資訊字典
        """
        logger.info(f"開始計算medoid，共{len(condition_files)}個樣本")
        
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
        
        # 計算距離矩陣
        logger.info("計算距離矩陣...")
        dist_matrix = cls.compute_distance_matrix(data_flat, distance_metric)
        
        # 找出medoid
        logger.info("尋找medoid...")
        dist_sums = np.sum(dist_matrix, axis=1)
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
        
        return {
            'medoid_info': medoid_info,
            'medoid_condition': medoid_condition,
            'statistics': stats,
            'distance_matrix': dist_matrix
        }


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


def save_speaker_info(
    audio_list_path: str,
    output_dir: str,
    lines: List[str]
) -> None:
    """儲存說話人資訊"""
    total_duration = sum(json.loads(line)["duration"] for line in lines)
    speaker_info = {
        "speaker": Path(audio_list_path).stem,
        "avg_duration": total_duration / len(lines),
        "sample_num": len(lines),
        "total_duration_in_seconds": total_duration,
        "total_duration_in_minutes": total_duration / 60,
        "total_duration_in_hours": total_duration / 3600,
        "train_jsonl": os.path.abspath(os.path.join(output_dir, "metadata_train.jsonl")),
        "valid_jsonl": os.path.abspath(os.path.join(output_dir, "metadata_valid.jsonl")),
        "medoid_condition": os.path.abspath(os.path.join(output_dir, "medoid_condition.npy")),
    }
    
    speaker_info_file = os.path.join(output_dir, "..", 'speaker_info.json')

    # 讀取現有資訊或建立新列表
    if os.path.exists(speaker_info_file):
        with open(speaker_info_file, 'r', encoding='utf-8') as f:
            speaker_info_list = json.load(f)
    else:
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
    output_dir: str
) -> Tuple[str, List[str]]:
    """處理音訊檔案列表"""
    metadata_file = os.path.join(output_dir, METADATA_FILENAME)
    condition_files = []
    metadata_path = Path(metadata_file)
    
    with open(args.audio_list, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="處理音訊檔案"):
            try:
                wav_path, txt = parse_audio_line(line)
            except ValueError as e:
                logger.warning(f"跳過無效行: {line.strip()}, 錯誤: {e}")
                continue
            
            if not os.path.exists(wav_path):
                logger.warning(f"音訊檔案未找到: {wav_path}")
                continue
            
            # 讀取音訊檔案
            try:
                audio, sr = torchaudio.load(wav_path)
            except Exception as e:
                logger.error(f"讀取音訊檔案時出錯: {wav_path}, 錯誤資訊: {e}")
                continue
            
            # 處理音訊資料
            try:
                processed_audio, mel, codes = audio_processor.process_audio_data(audio, sr)
            except Exception as e:
                logger.error(f"處理音訊資料時出錯: {wav_path}, 錯誤資訊: {e}")
                continue
            
            # 計算音訊時長
            duration = processed_audio.shape[-1] / config.dataset.mel.sample_rate
            
            # 儲存特徵檔案
            base_name = os.path.basename(wav_path)
            out_codebook = os.path.join(output_dir, f"{base_name}_codes.npy")
            out_mel = os.path.join(output_dir, f"{base_name}_mel.npy")
            
            try:
                np.save(out_codebook, codes.cpu().numpy())
                np.save(out_mel, mel.cpu().numpy())
                fix_permissions(out_codebook)
                fix_permissions(out_mel)
            except Exception as e:
                logger.error(f"儲存特徵檔案時出錯: {out_codebook}, 錯誤資訊: {e}")
                continue
            
            # 提取condition latent（如果需要）
            condition_path = None
            if condition_extractor is not None:
                try:
                    condition = condition_extractor.extract_condition_latent(mel)
                    condition_path = os.path.join(output_dir, f"{base_name}_condition.npy")
                    np.save(condition_path, condition)
                    fix_permissions(condition_path)
                    condition_files.append(condition_path)
                except Exception as e:
                    logger.error(f"提取condition時出錯: {wav_path}, 錯誤資訊: {e}")
                    continue
            
            # 寫入元資料
            try:
                data_entry = {
                    "audio": wav_path,
                    "text": txt,
                    "codes": os.path.abspath(out_codebook),
                    "mels": os.path.abspath(out_mel),
                    "duration": round(duration, 4)
                }
                
                if condition_path:
                    data_entry["condition"] = os.path.abspath(condition_path)
                
                with open(metadata_file, "a", encoding="utf-8") as out_f:
                    out_f.write(json.dumps(data_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"寫入元資料時出錯: {metadata_file}, 錯誤資訊: {e}")
                continue
            finally:
                if metadata_path.exists():
                    fix_permissions(metadata_path)
    
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
        args, config, audio_processor, condition_extractor, output_dir
    )
    
    logger.info(f"特徵提取完成，結果儲存在: {output_dir}")
    
    # 計算medoid（如果需要）
    if args.extract_condition and condition_files:
        logger.info("開始計算medoid condition...")
        try:
            medoid_result = MedoidCalculator.find_medoid_condition(
                condition_files, args.distance_metric
            )
            save_medoid_results(medoid_result, output_dir)
        except Exception as e:
            logger.error(f"計算medoid時出錯: {e}")
    
    # 分割資料集
    split_dataset(metadata_file, output_dir)
    
    # 儲存說話人資訊
    with open(metadata_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    save_speaker_info(args.audio_list, output_dir, lines)


if __name__ == "__main__":
    main()
