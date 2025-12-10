import copy
import gc
import os
import random
from datetime import datetime
from typing import List, Optional, Tuple, Dict

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from peft.optimizers import create_loraplus_optimizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from indextts.BigVGAN.models import BigVGAN
from indextts.data_utils import (
    collate_finetune_fn,
    load_finetune_datasets,
)
from indextts.gpt.model import UnifiedVoice

# 嘗試載入 GPU 管理器
try:
    from indextts.gpu_manager import GPUManager, get_global_gpu_manager
    GPU_MANAGER_AVAILABLE = True
except ImportError:
    GPU_MANAGER_AVAILABLE = False
    logger.warning("⚠️  GPU Manager 未安裝，將停用多 GPU 支援")


def normalize_state_dict_keys(state_dict: dict) -> dict:
    """
    標準化狀態字典的鍵值名稱，移除 DataParallel/DDP 產生的 `module.` 前綴。

    Args:
        state_dict (dict): 原始的狀態字典。

    Returns:
        dict: 處理後的狀態字典。
    """
    if not any(key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_UnifiedVoice(gpt_config: DictConfig, gpt_checkpoint_path: str, device: torch.device) -> UnifiedVoice:
    """
    載入並初始化 UnifiedVoice 模型。

    Args:
        gpt_config (DictConfig): GPT 模型配置參數。
        gpt_checkpoint_path (str): 模型權重檔案路徑。
        device (torch.device): 目標運算裝置。

    Returns:
        UnifiedVoice: 初始化完成的模型實例。
    """
    state_dict = torch.load(gpt_checkpoint_path, map_location=device, weights_only=True)
    state_dict = state_dict["model"] if "model" in state_dict else state_dict
    state_dict = normalize_state_dict_keys(state_dict)
    
    model = UnifiedVoice(**gpt_config)
    model.load_state_dict(state_dict, strict=True)
    model.post_init_gpt2_config()
    del state_dict
    return model.to(device)

def clear_torch_cache():
    """清理 PyTorch 的 CUDA 快取記憶體。"""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

def forward_gpt2(
    model: UnifiedVoice,
    inputs_embeds: torch.FloatTensor,
    text_lengths: torch.LongTensor,
    codes_lengths: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_latent: bool = False,
    output_logits: bool = True,
):
    """
    執行 UnifiedVoice GPT2 部分的前向傳播。
    
    此函數處理輸入嵌入，通過 GPT 模型，並計算文字和 Mel 的 logits。
    
    Args:
        model (UnifiedVoice): 模型實例。
        inputs_embeds (torch.FloatTensor): 輸入嵌入張量。
        text_lengths (torch.LongTensor): 文字序列長度。
        codes_lengths (torch.LongTensor): Mel 代碼序列長度。
        attention_mask (Optional[torch.Tensor]): 注意力遮罩。
        output_latent (bool): 是否輸出隱向量。
        output_logits (bool): 是否輸出 logits。

    Returns:
        dict: 包含 logits 和/或 latent 的字典。
    """
    assert attention_mask is not None, "UnifiedVoice 前向傳播必須提供 attention_mask。"

    # 處理 DataParallel 封裝
    actual_model = model.module if isinstance(model, nn.DataParallel) else model

    b = inputs_embeds.shape[0]
    gpt_out = actual_model.gpt(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)
    hidden_state = gpt_out.last_hidden_state

    # 向量化實現以替代迴圈
    conditioning_len = 32

    # 移除條件部分
    h_no_cond = hidden_state[:, conditioning_len:]  # [b, seq_len, hidden_dim]
    attention_no_cond = attention_mask[:, conditioning_len:]  # [b, seq_len]

    # 批次應用 final_norm
    latent = actual_model.final_norm(h_no_cond)  # [b, seq_len, hidden_dim]
    
    max_text_len = text_lengths.max().item()
    max_mel_len = codes_lengths.max().item()
    
    # 建立批次張量
    batch_text_latents = torch.zeros(b, max_text_len, latent.shape[-1], device=latent.device, dtype=latent.dtype)
    batch_mel_latents = torch.zeros(b, max_mel_len, latent.shape[-1], device=latent.device, dtype=latent.dtype)
    
    # 填充批次張量
    for i in range(b):
        text_len = text_lengths[i].item()
        mel_len = codes_lengths[i].item()
        
        # 提取有效 latent
        sample_valid_mask = attention_no_cond[i] == 1
        sample_latent = latent[i][sample_valid_mask]  # [valid_len, hidden_dim]
        
        expected_len = text_len + mel_len
        assert sample_latent.shape[0] == expected_len, \
            f"Expected valid_latent shape {expected_len}, got {sample_latent.shape[0]}, " \
            f"text_len: {text_len}, mel_len: {mel_len}"
        
        # 分割並分配
        batch_text_latents[i, :text_len] = sample_latent[:text_len]
        batch_mel_latents[i, :mel_len] = sample_latent[text_len:text_len + mel_len]
    
    # 向量化頭部處理
    batch_text_logits = actual_model.text_head(batch_text_latents)  # [b, max_text_len, vocab_size]
    batch_text_logits = batch_text_logits.permute(0, 2, 1)  # [b, vocab_size, max_text_len]

    batch_mel_logits = actual_model.mel_head(batch_mel_latents)  # [b, max_mel_len, vocab_size]
    batch_mel_logits = batch_mel_logits.permute(0, 2, 1)  # [b, vocab_size, max_mel_len]
    
    output = {}
    if output_logits:
        output["logits"] = (batch_text_logits, batch_mel_logits)
    if output_latent:
        output["latent"] = (batch_text_latents, batch_mel_latents)
    return output

def forward_UnifiedVoice(
    model: UnifiedVoice,
    mel_spec: torch.FloatTensor,
    mel_codes: torch.LongTensor,
    text_ids: torch.LongTensor,
    mel_lengths: torch.LongTensor,
    codes_lengths: torch.LongTensor,
    text_lengths: torch.LongTensor,
    condition_mels: torch.FloatTensor = None,
    condition_lengths: torch.LongTensor = None,
    speaker_ids: List[str] = None,
    add_mel_stop_token: bool = True,
    output_loss: bool = True,
    output_logits: bool = True,
    output_latent: bool = False,
    loss_reduction: str = "mean",
):
    """
    執行 UnifiedVoice 模型的完整前向傳播流程。

    此函數整合了輸入嵌入、位置編碼、條件輸入處理，並調用 GPT2 模型進行計算。
    如果需要，還會計算 Loss。

    Args:
        model (UnifiedVoice): 模型實例。
        mel_spec (torch.FloatTensor): Mel 頻譜圖輸入。
        mel_codes (torch.LongTensor): Mel 代碼輸入。
        text_ids (torch.LongTensor): 文字 Token ID。
        mel_lengths, codes_lengths, text_lengths (torch.LongTensor): 各序列的長度。
        condition_mels (torch.FloatTensor, optional): 條件 Mel 頻譜。
        speaker_ids (List[str], optional): 說話人 ID 列表。
        add_mel_stop_token (bool): 是否添加 Mel 結束 Token。
        output_loss (bool): 是否計算並回傳 Loss。
        output_logits (bool): 是否回傳 logits。
        output_latent (bool): 是否回傳 latent。
        loss_reduction (str): Loss 縮減方式。

    Returns:
        dict: 包含 loss, logits, targets, mel_accuracy 等結果的字典。
    """

    actual_model = model.module if isinstance(model, nn.DataParallel) else model

    # 處理條件輸入來源
    cond_source = condition_mels if condition_mels is not None else mel_spec
    cond_lengths = condition_lengths if condition_lengths is not None else mel_lengths
    conditioning_latent = actual_model.get_conditioning(cond_source, cond_lengths, speaker_ids=speaker_ids)
    
    # 構建文字輸入 (加入 start/stop tokens)
    B, T_pad = text_ids.shape
    max_out_text = T_pad + 2
    text_inputs = text_ids.new_zeros((B, max_out_text))
    for i, L in enumerate(text_lengths):
        L = L.item()
        text_inputs[i, 0] = actual_model.start_text_token
        text_inputs[i, 1 : L + 1] = text_ids[i, :L]
        text_inputs[i, L + 1] = actual_model.stop_text_token
    text_targets = text_inputs[:, 1:].clone().contiguous()

    # 構建 Mel 輸入 (加入 start/stop tokens)
    B, M_pad = mel_codes.shape
    extra_stop = 1 if add_mel_stop_token else 0
    max_out_mel = M_pad + 1 + extra_stop
    mel_inputs = mel_codes.new_zeros((B, max_out_mel))
    for i, L in enumerate(codes_lengths):
        L = L.item()
        mel_inputs[i, 0] = actual_model.start_mel_token
        mel_inputs[i, 1 : L + 1] = mel_codes[i, :L]
        if add_mel_stop_token:
            mel_inputs[i, L + 1] = actual_model.stop_mel_token
    mel_targets = mel_inputs[:, 1:].clone().contiguous()

    # 計算嵌入
    text_emb = actual_model.text_embedding(text_inputs) + actual_model.text_pos_embedding(text_inputs)
    mel_emb = actual_model.mel_embedding(mel_inputs) + actual_model.mel_pos_embedding(mel_inputs)

    mel_codes = mel_inputs
    
    inputs_embeds = torch.cat([conditioning_latent, text_emb, mel_emb], dim=1)
    
    # 建立注意力遮罩
    batch_size, total_seq_len = inputs_embeds.shape[:2]
    attention_mask = torch.zeros(batch_size, total_seq_len, dtype=torch.long, device=inputs_embeds.device)
    
    conditioning_len = conditioning_latent.shape[1]
    actual_text_lengths = text_lengths + 2
    actual_mel_lengths = codes_lengths + 1 + int(add_mel_stop_token)
    
    for i in range(batch_size):
        attention_mask[i, :conditioning_len] = 1
        
        text_start = conditioning_len
        text_end = text_start + actual_text_lengths[i].item()
        attention_mask[i, text_start:text_end] = 1
        
        mel_start = conditioning_len + text_emb.shape[1]
        mel_end = mel_start + actual_mel_lengths[i].item()
        attention_mask[i, mel_start:mel_end] = 1

    gpt2_outputs = forward_gpt2(
        model,
        inputs_embeds,
        text_lengths + 2,
        codes_lengths + 1 + int(add_mel_stop_token),
        attention_mask=attention_mask,
        output_latent=output_latent,
        output_logits=output_logits or output_loss,
    )
    
    outputs = {}
    if output_logits or output_loss:
        text_logits, mel_logits = gpt2_outputs["logits"]
        text_logits = text_logits[:, :, :-1].contiguous()
        mel_logits = mel_logits[:, :, :-1].contiguous()
        if output_loss:
            batch_size = text_targets.size(0)
            
            # 計算文字遮罩
            text_mask = torch.zeros_like(text_targets, dtype=torch.bool)
            for i in range(batch_size):
                actual_text_len = text_lengths[i].item() + 1
                text_mask[i, :actual_text_len] = True
            
            # 計算 Mel 遮罩
            mel_mask = torch.zeros_like(mel_targets, dtype=torch.bool)
            for i in range(batch_size):
                actual_mel_len = codes_lengths[i].item() + int(add_mel_stop_token)
                mel_mask[i, :actual_mel_len] = True
            
            loss_text = F.cross_entropy(text_logits, text_targets.long(), reduction='none')
            loss_mel = F.cross_entropy(mel_logits, mel_targets.long(), reduction='none')
            
            # 應用遮罩並計算平均 Loss
            loss_text = (loss_text * text_mask).sum() / text_mask.sum() if text_mask.sum() > 0 else torch.tensor(0.0, device=text_logits.device)
            loss_mel = (loss_mel * mel_mask).sum() / mel_mask.sum() if mel_mask.sum() > 0 else torch.tensor(0.0, device=mel_logits.device)
            
            outputs["loss"] = (loss_text, loss_mel)
            
            # 計算 Mel 預測準確率
            with torch.no_grad():
                mel_logits_flat = mel_logits.permute(0, 2, 1).reshape(-1, mel_logits.size(1))
                mel_targets_flat = mel_targets.view(-1)
                mel_mask_flat = mel_mask.view(-1)
                
                if mel_mask_flat.sum() > 0:
                    valid_mel_logits = mel_logits_flat[mel_mask_flat]
                    valid_mel_targets = mel_targets_flat[mel_mask_flat]
                    mel_acc_1, mel_acc_10, mel_acc_20 = top_k_accuracy(valid_mel_logits, valid_mel_targets, k=(1, 10, 20))
                    outputs["mel_accuracy"] = {"acc_1": mel_acc_1, "acc_10": mel_acc_10, "acc_20": mel_acc_20}
                else:
                    outputs["mel_accuracy"] = {"acc_1": 0.0, "acc_10": 0.0, "acc_20": 0.0}
                    
        if output_logits:
            outputs["logits"] = (text_logits, mel_logits)
            outputs["targets"] = (text_targets, mel_targets)
        

    if output_latent:
        outputs["latent"] = gpt2_outputs["latent"]
        
    clear_torch_cache()
    return outputs

def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: Tuple[int, ...] = (1, 10, 20)) -> List[float]:
    """
    計算 Top-K 準確率。

    Args:
        logits (torch.Tensor): 預測的 logits。
        targets (torch.Tensor): 真實標籤。
        k (Tuple[int, ...]): 要計算的 k 值列表。

    Returns:
        List[float]: 對應每個 k 值的準確率列表。
    """
    max_k = max(k)
    _, topk_preds = torch.topk(logits, max_k, dim=1)  # (B*L, max_k)
    
    targets_reshaped = targets.view(-1, 1) # (B*L, 1)
    topk_preds_reshaped = topk_preds.view(-1, max_k) # (B*L, max_k)

    res = []
    for ki in k:
        correct_k = (topk_preds_reshaped[:, :ki] == targets_reshaped).any(dim=-1)
        acc = correct_k.float().mean().item() * 100
        res.append(acc)
    return res

class Trainer:
    """
    UnifiedVoice 模型的訓練管理器。

    負責管理訓練流程、驗證、檢查點儲存以及混合精度訓練設定。
    """

    def __init__(self, config: DictConfig, use_multi_gpu: bool = True):
        """
        初始化訓練器。

        Args:
            config (DictConfig): 訓練配置參數 (OmegaConf)。
            use_multi_gpu (bool): 是否啟用多 GPU 訓練支援。
        """
        self.config = config
        self.use_multi_gpu = use_multi_gpu and GPU_MANAGER_AVAILABLE
        self.gpu_manager = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.use_multi_gpu and torch.cuda.is_available():
            try:
                self.gpu_manager = get_global_gpu_manager()
                gpu_count = self.gpu_manager.get_gpu_count()
                if gpu_count > 1:
                    logger.info(f"🎮 多 GPU 訓練模式：偵測到 {gpu_count} 個 GPU")
                    self.gpu_manager.print_summary()
                elif gpu_count == 1:
                    logger.info("🎮 單 GPU 訓練模式")
                    self.use_multi_gpu = False
            except Exception as e:
                logger.warning(f"⚠️  GPU Manager 初始化失敗: {e}")
                self.use_multi_gpu = False

        self._set_seed(self.config.train.seed)
        self.grad_scaler = None
        self.train_dtype, self.use_amp = self._setup_mixed_precision()

        # 設定路徑
        self.finetune_dir = self.config.train.finetune_model_dir
        self.checkpoint_dir = os.path.join(self.finetune_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        env_run_name = os.environ.get("RUN_NAME")
        env_log_dir = os.environ.get("RUN_LOG_DIR")
        self.run_name = env_run_name if env_run_name else f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if env_log_dir:
            self.log_dir = os.path.abspath(env_log_dir)
        else:
            self.log_dir = os.path.abspath(os.path.join("logs", self.run_name))
        os.makedirs(self.log_dir, exist_ok=True)
        
        self._setup_logging(os.path.join(self.log_dir, "train.log"))

        self.writer = SummaryWriter(log_dir=self.log_dir)
        logger.info(f"TensorBoard 記錄目錄: {self.log_dir}")

        self._load_models()
        self._setup_optimizer_and_scheduler()

        self.best_val_loss = (0, float('inf'), float('inf'))  # (epoch, text_loss, mel_loss)
        self.update_steps = 0

    def _set_seed(self, seed: int):
        """設定隨機種子以確保實驗可重複性。"""
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"設定隨機種子為: {seed}")

    def _resolve_dtype(self, precision_str: str):
        """
        將精度設定字串轉換為 torch.dtype。
        
        Args:
            precision_str (str): 精度設定 (如 'fp16', 'bf16', 'auto').

        Returns:
            torch.dtype: 對應的 PyTorch 資料型態。
        """
        def supports_fp8():
            if not torch.cuda.is_available():
                return False
            capability = torch.cuda.get_device_capability()
            compute_capability = capability[0] * 10 + capability[1]
            return compute_capability >= 89

        if precision_str == "no" or precision_str == "fp32":
            return torch.float32
        elif precision_str == "auto":
            if supports_fp8():
                return torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else torch.bfloat16
            elif torch.cuda.is_bf16_supported():
                return torch.bfloat16
            else:
                return torch.float16
        elif precision_str == "fp8":
            if supports_fp8() and hasattr(torch, 'float8_e4m3fn'):
                return torch.float8_e4m3fn
            else:
                logger.warning(f"⚠️  當前硬體不支援 FP8，退回 BF16")
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif precision_str == "bf16":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif precision_str == "fp16":
            return torch.float16
        else:
            logger.warning(f"⚠️  未知精度設定: {precision_str}，將使用 FP32")
            return torch.float32

    def _setup_mixed_precision(self):
        """
        配置混合精度訓練環境。

        根據配置選擇適當的精度 (BF16/FP16/FP8)，並在需要時初始化 GradScaler。

        Returns:
            Tuple[torch.dtype, bool]: (運算精度, 是否啟用 AMP)
        """
        mixed_precision = self.config.train.get("mixed_precision", "auto")

        if not torch.cuda.is_available():
            logger.warning("⚠️  CUDA 不可用，強制使用 FP32 訓練")
            return None, False

        dtype = self._resolve_dtype(mixed_precision)

        logger.info("🚀 混合精度訓練配置")
        logger.info(f"   運算精度: {dtype}")

        use_grad_scaler = (dtype == torch.float16)

        if use_grad_scaler:
            self.grad_scaler = GradScaler()
            logger.info("   📊 啟用 GradScaler (針對 FP16 防止梯度下溢)")
        else:
            self.grad_scaler = None

        logger.info("=" * 50)
        return dtype, True

    def _setup_logging(self, log_path: str):
        """配置 loguru 日誌系統。"""
        logger.add(log_path, level="INFO", encoding="utf-8")
        logger.info("日誌系統已配置。")
        logger.info("完整配置參數:\n" + OmegaConf.to_yaml(self.config))

    def _load_models(self):
        """載入 BPE 模型、BigVGAN 與 UnifiedVoice 模型，並應用 LoRA。"""
        logger.info("正在載入模型...")
        
        # 載入 BPE
        bpe_model_path = os.path.join(self.finetune_dir, self.config.dataset.bpe_model)
        self.bpe_model = spm.SentencePieceProcessor(bpe_model_path)
        logger.info("BPE 模型載入完成。")
        
        # 載入 UnifiedVoice
        gpt_checkpoint_path = os.path.join(self.finetune_dir, self.config.gpt_checkpoint)
        self.model = load_UnifiedVoice(self.config.gpt, gpt_checkpoint_path, self.device)
        logger.info("UnifiedVoice 基礎模型載入完成。")
    
        # 應用 LoRA
        self.model = self._apply_lora(self.model)
        logger.info("LoRA 適配器已應用。")

        # 多 GPU 支援
        if self.use_multi_gpu and self.gpu_manager and self.gpu_manager.get_gpu_count() > 1:
            logger.info("🚀 啟用 DataParallel 多 GPU 訓練")
            device_ids = list(range(torch.cuda.device_count()))
            logger.info(f"  使用 GPU 裝置: {device_ids}")
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
            logger.info(f"  模型已分散至 {len(device_ids)} 個 GPU")

    def _apply_lora(self, model: UnifiedVoice) -> UnifiedVoice:
        """
        為模型配置並應用 LoRA (Low-Rank Adaptation)。
        
        這會凍結大部分參數，僅開放 LoRA 層與特定編碼器進行訓練。
        """
        lora_cfg = self.config.train.lora
        gpt_lora_config = LoraConfig(
            r=lora_cfg.r,
            target_modules=lora_cfg.target_modules,
            task_type=TaskType.CAUSAL_LM,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
            bias="none",
            fan_in_fan_out=True,
        )
        model.requires_grad_(False)
        model.inference_model = get_peft_model(model.inference_model, gpt_lora_config)

        # ⚠️ 重要：凍結 conditioning_encoder 和 perceiver_encoder
        # 原因：預訓練的 encoder 已經學會抽取音色，如果繼續訓練，
        # 它會學習編碼更多資訊（包括內容），導致推論時複製參考音檔的文字內容。
        # 這是 zero-shot TTS 的常見問題。
        if hasattr(model, "conditioning_encoder"):
            model.conditioning_encoder.requires_grad_(False)
            logger.info("✓ conditioning_encoder 已凍結（防止內容洩漏）")
        if hasattr(model, "perceiver_encoder"):
            model.perceiver_encoder.requires_grad_(False)
            logger.info("✓ perceiver_encoder 已凍結（防止內容洩漏）")

        # 只訓練 LoRA 層，讓 GPT 學習如何根據「固定的音色 embedding」生成對應內容
        return model

    def _setup_optimizer_and_scheduler(self, num_training_steps: int = 1000):
        """配置 LoRA+ 最佳化器與 Cosine 學習率排程器。"""
        opt_cfg = self.config.train.optimizer
        optimizer = create_loraplus_optimizer(
            model=self.model,
            optimizer_cls=AdamW,
            lr=opt_cfg.learning_rate,
            loraplus_lr_ratio=opt_cfg.loraplus_lr_ratio,
            loraplus_weight_decay=opt_cfg.weight_decay,
        )
        self.optimizer = optimizer

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_training_steps * opt_cfg.warmup_ratio),
            num_training_steps=num_training_steps,
        )
        self.scheduler = scheduler
        logger.info("Optimizer (LoRA+) 與 Scheduler 已配置完成。")

    def _train_step(self, data_batch: tuple) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        執行單一訓練步：前向傳播、Loss 計算。

        Args:
            data_batch (tuple): 包含所有輸入資料的 tuple。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, dict]: (text_loss, mel_loss, mel_accuracy_dict)
        """
        self.model.train()
        actual_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        actual_model.inference_model.kv_cache = False
    
        mel_spec, mel_codes, text_ids, cond_mels, speaker_ids, mel_lengths, codes_lengths, text_lengths, cond_lengths = data_batch

        # 混合精度上下文
        context = torch.autocast(device_type='cuda', dtype=self.train_dtype) if (self.use_amp and self.train_dtype) else torch.no_grad()
        if not (self.use_amp and self.train_dtype):
             # 如果不是 AMP，則使用 dummy context (no_grad 只是 placeholder，實際上不會用)
             # 但為了避免 indent 混亂，直接使用條件判斷
             pass

        if self.use_amp and self.train_dtype:
             with torch.autocast(device_type='cuda', dtype=self.train_dtype):
                outputs = forward_UnifiedVoice(
                    self.model, mel_spec, mel_codes, text_ids, mel_lengths, codes_lengths, text_lengths,
                    condition_mels=cond_mels, condition_lengths=cond_lengths, speaker_ids=None,
                    output_loss=True, output_logits=True,
                )
        else:
             outputs = forward_UnifiedVoice(
                self.model, mel_spec, mel_codes, text_ids, mel_lengths, codes_lengths, text_lengths,
                condition_mels=cond_mels, condition_lengths=cond_lengths, speaker_ids=None,
                output_loss=True, output_logits=True,
            )

        loss_text, loss_mel = outputs["loss"]
        mel_accuracy = outputs.get("mel_accuracy", {"acc_1": 0.0, "acc_10": 0.0, "acc_20": 0.0})
        return loss_text, loss_mel, mel_accuracy

    @torch.no_grad()
    def _validate_epoch(self, valid_ds: Dataset, epoch: int):
        """
        執行驗證流程。

        Args:
            valid_ds (Dataset): 驗證資料集。
            epoch (int): 當前 Epoch 數。

        Returns:
            Tuple: (avg_text_loss, avg_mel_loss, acc_1, acc_10, acc_20)
        """
        self.model.eval()
        logger.info(f"正在進行第 {epoch + 1} 輪驗證...")
        
        total_text_loss, total_mel_loss = 0.0, 0.0
        total_text_tokens, total_mel_tokens = 0, 0
        all_mel_logits, all_mel_targets = [], []

        for batch in tqdm(valid_ds, desc="驗證中", dynamic_ncols=True):
            data_batch = []
            for item in batch:
                if torch.is_tensor(item):
                    data_batch.append(item.to(self.device))
                else:
                    data_batch.append(item)

            mel_spec, mel_codes, text_ids, cond_mels, speaker_ids, mel_lengths, codes_lengths, text_lengths, cond_lengths = data_batch

            if self.use_amp and self.train_dtype:
                with torch.autocast(device_type='cuda', dtype=self.train_dtype):
                    outputs = forward_UnifiedVoice(
                        self.model, mel_spec, mel_codes, text_ids, mel_lengths, codes_lengths, text_lengths,
                        condition_mels=cond_mels, condition_lengths=cond_lengths, speaker_ids=None,
                        output_loss=True, output_logits=True,
                    )
            else:
                outputs = forward_UnifiedVoice(
                    self.model, mel_spec, mel_codes, text_ids, mel_lengths, codes_lengths, text_lengths,
                    condition_mels=cond_mels, condition_lengths=cond_lengths, speaker_ids=None,
                    output_loss=True, output_logits=True,
                )
            
            loss_text, loss_mel = outputs["loss"]
            batch_text_tokens = text_lengths.sum().item()
            batch_mel_tokens = (codes_lengths + 1).sum().item()

            total_text_loss += loss_text.item() * batch_text_tokens
            total_mel_loss += loss_mel.item() * batch_mel_tokens
            total_text_tokens += batch_text_tokens
            total_mel_tokens += batch_mel_tokens

            # 收集數據計算準確率
            current_mel_logits = outputs["logits"][1]
            current_mel_targets = outputs["targets"][1]
            if current_mel_logits.numel() > 0 and current_mel_targets.numel() > 0:
                batch_size = current_mel_targets.size(0)
                mel_mask = torch.zeros_like(current_mel_targets, dtype=torch.bool)
                for i in range(batch_size):
                    actual_mel_len = codes_lengths[i].item() + 1
                    mel_mask[i, :actual_mel_len] = True
                
                valid_mask = mel_mask.view(-1)
                if valid_mask.sum() > 0:
                    mel_logits_filtered = current_mel_logits.permute(0, 2, 1).reshape(-1, current_mel_logits.size(1))[valid_mask]
                    mel_targets_filtered = current_mel_targets.view(-1)[valid_mask]
                    all_mel_logits.append(mel_logits_filtered)
                    all_mel_targets.append(mel_targets_filtered)
            
            clear_torch_cache()

        avg_text_loss = total_text_loss / total_text_tokens
        avg_mel_loss = total_mel_loss / total_mel_tokens
        
        all_mel_logits = torch.cat(all_mel_logits, dim=0)
        all_mel_targets = torch.cat(all_mel_targets, dim=0)
        acc_1, acc_10, acc_20 = top_k_accuracy(all_mel_logits, all_mel_targets, k=(1, 10, 20))

        logger.info(f"**第 {epoch + 1} 輪驗證結果**")
        logger.info(f"Text Loss: {avg_text_loss:.4f}, Mel Loss: {avg_mel_loss:.4f}")
        logger.info(f"Accuracy@1: {acc_1:.2f}%, Accuracy@10: {acc_10:.2f}%, Accuracy@20: {acc_20:.2f}%")
        
        return avg_text_loss, avg_mel_loss, acc_1, acc_10, acc_20

    def _save_checkpoint(self, file_name: str, merge_lora: bool, unload_after_merge: bool):
        """
        儲存模型檢查點。

        Args:
            file_name (str): 檔案名稱。
            merge_lora (bool): 是否將 LoRA 權重合併進主模型。
            unload_after_merge (bool): 合併後是否卸載 LoRA (若為 True 則不影響訓練中的模型實例)。
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, file_name)
        self.model.eval()

        actual_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model_to_save = actual_model

        if merge_lora:
            logger.info("正在合併 LoRA 權重以進行儲存...")
            if unload_after_merge:
                # 建立深複製以避免影響訓練狀態
                logger.info("正在複製模型以進行安全合併...")
                model_to_save = copy.deepcopy(actual_model)
                fused_inference_model = model_to_save.inference_model.merge_and_unload()
                model_to_save.inference_model = fused_inference_model
                logger.info("合併完成。")
            else:
                actual_model.inference_model.merge_adapter()
    
        state_dict = model_to_save.state_dict()
        checkpoint_data = {'model': state_dict}
        
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"檢查點已儲存至: {checkpoint_path}")
    
        if merge_lora and unload_after_merge:
            del model_to_save
            clear_torch_cache()
            logger.info("暫存模型已清理。")
        
        if merge_lora and not unload_after_merge:
            logger.info("正在解除合併 LoRA 權重以繼續訓練...")
            actual_model.inference_model.unmerge_adapter()

        self.model.train()

    def train(self, train_ds: Dataset, valid_ds: Dataset):
        """
        執行主要訓練迴圈。

        Args:
            train_ds (Dataset): 訓練資料集。
            valid_ds (Dataset): 驗證資料集。
        """
        train_cfg = self.config.train
        total_ds_count = len(train_ds)
        
        samples_per_epoch = total_ds_count
        total_update_steps = samples_per_epoch * train_cfg.epochs
        self._setup_optimizer_and_scheduler(num_training_steps=total_update_steps)
        
        logger.info(f"開始訓練，共 {train_cfg.epochs} 輪 (Epochs)。")
        logger.info(f"每輪樣本數: {samples_per_epoch}")
        logger.info(f"總更新步數: {total_update_steps}")

        text_weight = train_cfg.text_weight

        for epoch in range(train_cfg.epochs):
            logger.info(f"EPOCH {epoch + 1}/{train_cfg.epochs} 開始 " + "=" * 30)

            # 使用 tqdm 包裝訓練資料載入器，實現進度條顯示
            pbar = tqdm(enumerate(train_ds), total=len(train_ds), desc=f"Epoch {epoch + 1}", dynamic_ncols=True)
            
            for batch_idx, batch in pbar:
                data_batch = []
                for item in batch:
                    if torch.is_tensor(item):
                        data_batch.append(item.to(self.device))
                    else:
                        data_batch.append(item)

                loss_text, loss_mel, mel_accuracy = self._train_step(tuple(data_batch))
                acc_1, acc_10, acc_20 = mel_accuracy["acc_1"], mel_accuracy["acc_10"], mel_accuracy["acc_20"]

                weighted_loss = text_weight * loss_text + (1.0 - text_weight) * loss_mel
                
                # 檢查 NaN/Inf Loss
                if torch.isnan(weighted_loss) or torch.isinf(weighted_loss):
                    logger.warning(f"Epoch {epoch}, Batch {batch_idx} 發現 NaN 或 Inf Loss，已跳過。")
                    continue

                # 最佳化步驟
                self.optimizer.zero_grad()
                
                if self.grad_scaler is not None:
                    self.grad_scaler.scale(weighted_loss).backward()
                    self.grad_scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), train_cfg.max_grad_norm)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    weighted_loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), train_cfg.max_grad_norm)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.update_steps += 1

                # 更新進度條顯示資訊
                pbar.set_postfix({
                    "txt_loss": f"{loss_text.item():.3f}",
                    "mel_loss": f"{loss_mel.item():.3f}",
                    "acc@1": f"{acc_1:.1f}%",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                })

                # 記錄訓練指標至 TensorBoard
                self.writer.add_scalar("loss/text", loss_text.item(), self.update_steps)
                self.writer.add_scalar("loss/mel", loss_mel.item(), self.update_steps)
                self.writer.add_scalar("loss/total", weighted_loss.item(), self.update_steps)
                self.writer.add_scalar("accuracy/top1", acc_1, self.update_steps)
                self.writer.add_scalar("accuracy/top10", acc_10, self.update_steps)
                self.writer.add_scalar("accuracy/top20", acc_20, self.update_steps)
                self.writer.add_scalar("train/grad_norm", grad_norm.item(), self.update_steps)
                self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.update_steps)

            # --- Epoch 結束 ---
            val_text_loss, val_mel_loss, val_acc1, val_acc10, val_acc20 = self._validate_epoch(valid_ds, epoch)
            
            self.writer.add_scalar("val/loss_text", val_text_loss, epoch + 1)
            self.writer.add_scalar("val/loss_mel", val_mel_loss, epoch + 1)
            self.writer.add_scalar("val/accuracy_top1", val_acc1, epoch + 1)
            self.writer.add_scalar("val/accuracy_top10", val_acc10, epoch + 1)
            self.writer.add_scalar("val/accuracy_top20", val_acc20, epoch + 1)

            epoch_checkpoint_name = f"gpt_epoch_{epoch + 1}.pth"
            logger.info(f"儲存 Epoch {epoch + 1} 模型: {epoch_checkpoint_name}")
            self._save_checkpoint(epoch_checkpoint_name, merge_lora=True, unload_after_merge=True)
            
            if val_mel_loss < self.best_val_loss[2]:
                logger.info(f"發現最佳驗證 Mel Loss: {val_mel_loss:.4f}。儲存最佳模型。")
                self.best_val_loss = (epoch, val_text_loss, val_mel_loss)
                self._save_checkpoint("gpt_best.pth", merge_lora=True, unload_after_merge=True)

            clear_torch_cache()

        # --- 訓練結束 ---
        logger.info("訓練完成。儲存最終模型。")
        self._save_checkpoint("gpt_finetuned.pth", merge_lora=True, unload_after_merge=True)
        
        final_config_path = os.path.join(self.finetune_dir, "config_finetuned.yaml")
        final_config = self.config.copy()
        final_config.gpt_checkpoint = "checkpoints/gpt_finetuned.pth"
        OmegaConf.save(final_config, final_config_path)
        logger.info(f"最終配置已儲存至 {final_config_path}")
        
        logger.info(f"最佳驗證結果 (Epoch {self.best_val_loss[0] + 1}): "
                    f"text_loss: {self.best_val_loss[1]:.4f}, mel_loss: {self.best_val_loss[2]:.4f}")
        
        self.writer.close()

def main():
    config_path = "finetune_models/config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置檔案: {config_path}")
    
    config = OmegaConf.load(config_path)
    bpe_model_path = os.path.join(config.train.finetune_model_dir, config.dataset.bpe_model)

    train_ds, valid_ds = load_finetune_datasets(config, bpe_model_path) 
    train_ds = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_finetune_fn, num_workers=4)
    valid_ds = DataLoader(valid_ds, batch_size=8, shuffle=False, collate_fn=collate_finetune_fn, num_workers=2)

    trainer = Trainer(config)
    trainer.train(train_ds, valid_ds)
    logger.info("UnifiedVoice 微調流程結束。")


if __name__ == "__main__":
    main()

