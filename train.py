import copy
import gc
import os
import random
from datetime import datetime
from typing import List, Optional, Tuple

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

from indextts.BigVGAN.models import BigVGAN
from indextts.data_utils import (
    collate_finetune_fn,
    load_finetune_datasets,
    load_speaker_conditions,
)
from indextts.gpt.model import UnifiedVoice

# Import GPU Manager
try:
    from indextts.gpu_manager import GPUManager, get_global_gpu_manager
    GPU_MANAGER_AVAILABLE = True
except ImportError:
    GPU_MANAGER_AVAILABLE = False
    logger.warning("⚠️  GPU Manager not available, multi-GPU support disabled")


def normalize_state_dict_keys(state_dict: dict) -> dict:
    """標準化權重名稱，移除 DataParallel/DDP 產生的 module. 前綴。"""
    if not any(key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_UnifiedVoice(gpt_config: DictConfig, gpt_checkpoint_path: str, device: torch.device) -> UnifiedVoice:
    """載入 UnifiedVoice 模型權重。"""
    state_dict = torch.load(gpt_checkpoint_path, map_location=device, weights_only=True)
    state_dict = state_dict["model"] if "model" in state_dict else state_dict
    state_dict = normalize_state_dict_keys(state_dict)
    model = UnifiedVoice(**gpt_config)
    model.load_state_dict(state_dict, strict=True)
    model.post_init_gpt2_config()
    del state_dict
    return model.to(device)

def clear_torch_cache():
    """清理 GPU 快取。"""
    if torch.cuda.is_available():
        # logger.info("Clearing CUDA cache...")
        # logger.info(f"{torch.cuda.memory_reserved() / (1024**2):.2f} MiB reserved")
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # logger.info("CUDA cache cleared.")
        # logger.info(f"{torch.cuda.memory_reserved() / (1024**2):.2f} MiB reserved")

def forward_gpt2(
    model: UnifiedVoice,
    inputs_embeds: torch.FloatTensor,
    text_lengths: torch.LongTensor,
    codes_lengths: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_latent: bool = False,
    output_logits: bool = True,
):
    assert attention_mask is not None, "Attention mask must be provided for UnifiedVoice forward pass."

    """UnifiedVoice GPT2 部分的前向傳播。"""
    # 處理 DataParallel 包裝：取得實際的模型
    actual_model = model.module if isinstance(model, nn.DataParallel) else model

    b = inputs_embeds.shape[0]
    gpt_out = actual_model.gpt(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)
    hidden_state = gpt_out.last_hidden_state

    # Vectorized implementation to replace the for loop
    conditioning_len = 32

    # Remove conditioning part from hidden states and attention mask
    h_no_cond = hidden_state[:, conditioning_len:]  # [b, seq_len, hidden_dim]
    attention_no_cond = attention_mask[:, conditioning_len:]  # [b, seq_len]

    # Apply final_norm to all samples at once
    latent = actual_model.final_norm(h_no_cond)  # [b, seq_len, hidden_dim]
    
    # Get max lengths for efficient processing
    max_text_len = text_lengths.max().item()
    max_mel_len = codes_lengths.max().item()
    
    # Create batched tensors for text and mel latents
    batch_text_latents = torch.zeros(b, max_text_len, latent.shape[-1], device=latent.device, dtype=latent.dtype)
    batch_mel_latents = torch.zeros(b, max_mel_len, latent.shape[-1], device=latent.device, dtype=latent.dtype)
    
    ## Create masks for valid positions in batched tensors
    #text_valid_mask = torch.zeros(b, max_text_len, device=latent.device, dtype=torch.int32)
    #mel_valid_mask = torch.zeros(b, max_mel_len, device=latent.device, dtype=torch.int32)
    
    # Fill the batched tensors
    for i in range(b):
        text_len = text_lengths[i].item()
        mel_len = codes_lengths[i].item()
        
        # Extract valid latent for this sample
        sample_valid_mask = attention_no_cond[i] == 1
        sample_latent = latent[i][sample_valid_mask]  # [valid_len, hidden_dim]
        
        # Verify the expected length
        expected_len = text_len + mel_len
        assert sample_latent.shape[0] == expected_len, \
            f"Expected valid_latent shape {expected_len}, got {sample_latent.shape[0]}, " \
            f"text_len: {text_len}, mel_len: {mel_len}"
        
        # Split and assign to batched tensors
        batch_text_latents[i, :text_len] = sample_latent[:text_len]
        batch_mel_latents[i, :mel_len] = sample_latent[text_len:text_len + mel_len]
        
        ## Set valid masks
        #text_valid_mask[i, :text_len] = 1
        #mel_valid_mask[i, :mel_len] = 1
    
    # Vectorized head processing
    # Process all text latents at once
    batch_text_logits = actual_model.text_head(batch_text_latents)  # [b, max_text_len, vocab_size]
    batch_text_logits = batch_text_logits.permute(0, 2, 1)  # [b, vocab_size, max_text_len]

    # Process all mel latents at once
    batch_mel_logits = actual_model.mel_head(batch_mel_latents)  # [b, max_mel_len, vocab_size]
    batch_mel_logits = batch_mel_logits.permute(0, 2, 1)  # [b, vocab_size, max_mel_len]
    
    # Return the processed batches directly.
    # The tensors are already padded to the max length in the batch.
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
    speaker_ids: List[str] = None,  # 新增引數
    add_mel_stop_token: bool = True,
    output_loss: bool = True,
    output_logits: bool = True,
    output_latent: bool = False,
    loss_reduction: str = "mean",
):
    """UnifiedVoice 模型的完整前向傳播。"""

    # 處理 DataParallel 包裝：取得實際的模型
    actual_model = model.module if isinstance(model, nn.DataParallel) else model

    conditioning_latent = actual_model.get_conditioning(mel_spec, mel_lengths, speaker_ids=speaker_ids)  # 傳遞 speaker_ids
    
    # -------- build text_inputs with start/stop tokens --------
    B, T_pad = text_ids.shape
    max_out_text = T_pad + 2  # +<start> +<stop>
    text_inputs = text_ids.new_zeros((B, max_out_text))
    for i, L in enumerate(text_lengths):
        L = L.item()
        text_inputs[i, 0] = actual_model.start_text_token
        text_inputs[i, 1 : L + 1] = text_ids[i, :L]
        text_inputs[i, L + 1] = actual_model.stop_text_token
    text_targets = text_inputs[:, 1:].clone().contiguous()

    # -------- build mel_inputs with start/stop tokens --------
    B, M_pad = mel_codes.shape
    extra_stop = 1 if add_mel_stop_token else 0
    max_out_mel = M_pad + 1 + extra_stop  # +<start> (+<stop>)
    mel_inputs = mel_codes.new_zeros((B, max_out_mel))
    for i, L in enumerate(codes_lengths):
        L = L.item()
        mel_inputs[i, 0] = actual_model.start_mel_token
        mel_inputs[i, 1 : L + 1] = mel_codes[i, :L]
        if add_mel_stop_token:
            mel_inputs[i, L + 1] = actual_model.stop_mel_token
    mel_targets = mel_inputs[:, 1:].clone().contiguous()

    # Embeddings
    text_emb = actual_model.text_embedding(text_inputs) + actual_model.text_pos_embedding(text_inputs)
    mel_emb = actual_model.mel_embedding(mel_inputs) + actual_model.mel_pos_embedding(mel_inputs)

    # for later use in loss and lengths
    mel_codes = mel_inputs
    
    inputs_embeds = torch.cat([conditioning_latent, text_emb, mel_emb], dim=1)
    
    # Create attention mask for the combined sequence
    batch_size, total_seq_len = inputs_embeds.shape[:2]
    attention_mask = torch.zeros(batch_size, total_seq_len, dtype=torch.long, device=inputs_embeds.device)
    
    # Calculate actual sequence lengths for each sample
    conditioning_len = conditioning_latent.shape[1]
    actual_text_lengths = text_lengths + 2  # +2 for start/stop tokens
    actual_mel_lengths = codes_lengths + 1 + int(add_mel_stop_token)  # +1 for start token + optional stop token
    
    for i in range(batch_size):
        # Set conditioning part (always valid)
        attention_mask[i, :conditioning_len] = 1
        
        # Set text part (considering actual text length without padding)
        text_start = conditioning_len
        text_end = text_start + actual_text_lengths[i].item()
        attention_mask[i, text_start:text_end] = 1
        
        # Set mel part (considering actual mel length without padding)
        # mel_start should be based on the actual text_emb length, not text_end
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
            # Create masks based on actual sequence lengths
            batch_size = text_targets.size(0)
            
            # Text mask: consider actual text length + 1 (stop token)
            text_mask = torch.zeros_like(text_targets, dtype=torch.bool)
            for i in range(batch_size):
                actual_text_len = text_lengths[i].item() + 1  # +1 for stop token
                text_mask[i, :actual_text_len] = True
            
            # Mel mask: consider actual codes length + stop token (if any)
            mel_mask = torch.zeros_like(mel_targets, dtype=torch.bool)
            for i in range(batch_size):
                actual_mel_len = codes_lengths[i].item() + int(add_mel_stop_token)
                mel_mask[i, :actual_mel_len] = True
            
            loss_text = F.cross_entropy(text_logits, text_targets.long(), reduction='none')
            loss_mel = F.cross_entropy(mel_logits, mel_targets.long(), reduction='none')
            
            # Apply masking and reduce - only calculate loss on valid positions
            loss_text = (loss_text * text_mask).sum() / text_mask.sum() if text_mask.sum() > 0 else torch.tensor(0.0, device=text_logits.device)
            loss_mel = (loss_mel * mel_mask).sum() / mel_mask.sum() if mel_mask.sum() > 0 else torch.tensor(0.0, device=mel_logits.device)
            
            outputs["loss"] = (loss_text, loss_mel)
            
            # Calculate mel prediction accuracy
            with torch.no_grad():
                # Apply mask to get valid positions for accuracy calculation
                mel_logits_flat = mel_logits.permute(0, 2, 1).reshape(-1, mel_logits.size(1))
                mel_targets_flat = mel_targets.view(-1)
                mel_mask_flat = mel_mask.view(-1)
                
                # Only calculate accuracy on valid positions
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
    """計算 top-k 準確率。"""
    max_k = max(k)
    _, topk_preds = torch.topk(logits, max_k, dim=1)  # (B*L, max_k)
    
    # Reshape for comparison
    targets_reshaped = targets.view(-1, 1) # (B*L, 1)
    topk_preds_reshaped = topk_preds.view(-1, max_k) # (B*L, max_k)

    # Remove unused variable and optimize calculation
    # correct = (topk_preds_reshaped == targets_reshaped).any(dim=-1)
    
    res = []
    for ki in k:
        # Check if the target is in the top-ki predictions
        correct_k = (topk_preds_reshaped[:, :ki] == targets_reshaped).any(dim=-1)
        acc = correct_k.float().mean().item() * 100
        res.append(acc)
    return res

class Trainer:
    """封裝 UnifiedVoice 微調過程的訓練器。"""
    def __init__(self, config: DictConfig, use_multi_gpu: bool = True):
        """
        初始化訓練器。

        Args:
            config (DictConfig): 從 YAML 檔案載入的 OmegaConf 配置物件。
            use_multi_gpu (bool): 是否啟用多 GPU 訓練（預設 True）
        """
        self.config = config
        self.use_multi_gpu = use_multi_gpu and GPU_MANAGER_AVAILABLE
        self.gpu_manager = None
        # 使用 cuda:0 作為主設備，確保 DataParallel 的主進程在第一張卡
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 初始化 GPU Manager
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

        # 設定隨機種子
        self._set_seed(self.config.train.seed)

        # 初始化 GradScaler（會在 _setup_mixed_precision 中設定）
        self.grad_scaler = None

        # 設定混合精度訓練（細粒度配置）
        self.train_dtype, self.use_amp = self._setup_mixed_precision()

        # 準備目錄和日誌
        self.finetune_dir = self.config.train.finetune_model_dir
        self.checkpoint_dir = os.path.join(self.finetune_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._setup_logging()

        # 載入說話人條件向量
        self.speaker_conditions = load_speaker_conditions(config)
        self.speaker_list = list(self.speaker_conditions.keys())
        logger.info(f"Loaded conditions for {len(self.speaker_list)} speakers: {self.speaker_list}")

        # 載入模型和分詞器
        self._load_models()

        # 設定最佳化器和排程器
        self._setup_optimizer_and_scheduler()

        # 初始化訓練狀態
        self.best_val_loss = (0, float('inf'), float('inf'))  # (epoch, text_loss, mel_loss)
        self.update_steps = 0

    def _set_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Set random seed to {seed}")

    def _resolve_dtype(self, precision_str: str):
        """解析精度字串為 torch.dtype"""
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
                logger.warning(f"⚠️  FP8 不支援，退回 BF16")
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif precision_str == "bf16":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif precision_str == "fp16":
            return torch.float16
        else:
            logger.warning(f"⚠️  未知精度: {precision_str}，使用 FP32")
            return torch.float32

    def _setup_mixed_precision(self):
        """
        設定混合精度訓練。

        混合精度訓練流程說明：
        - 資料前處理：FP32（保持數值穩定性）
        - 前向/反向運算：BF16/FP16/FP8（加速計算、節省顯存）
        - Loss 計算：FP32（避免數值溢出）
        - Optimizer state：FP32（Adam/AdamW 內部自動維持 FP32 避免精度累積誤差）

        Returns:
            Tuple[torch.dtype, bool]: (運算精度, 是否啟用AMP)
        """
        mixed_precision = self.config.train.get("mixed_precision", "auto")

        if not torch.cuda.is_available():
            logger.warning("⚠️  CUDA 不可用，使用 FP32 訓練")
            return None, False

        # 解析精度字串
        dtype = self._resolve_dtype(mixed_precision)

        logger.info("🚀 混合精度訓練")
        logger.info(f"   運算精度: {dtype}")

        # 檢查是否需要 GradScaler（FP16 需要）
        use_grad_scaler = (dtype == torch.float16)

        if use_grad_scaler:
            self.grad_scaler = GradScaler()
            logger.info("   📊 啟用 GradScaler（FP16 防止梯度下溢）")
        else:
            self.grad_scaler = None

        logger.info("=" * 50)
        return dtype, True

    def _setup_logging(self):
        """配置 loguru 日誌記錄器。"""
        log_path = os.path.join(self.checkpoint_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        logger.add(log_path, level="INFO", encoding="utf-8")
        logger.info("Logging configured. Logs will be saved to console and file.")
        logger.info("Full configuration:\n" + OmegaConf.to_yaml(self.config))

    def _load_models(self):
        """載入 BPE、BigVGAN 和 UnifiedVoice 模型。"""
        logger.info("Loading models...")
        # BPE
        bpe_model_path = os.path.join(self.finetune_dir, self.config.dataset.bpe_model)
        self.bpe_model = spm.SentencePieceProcessor(bpe_model_path)
        logger.info("BPE model loaded.")
        
        # BigVGAN (用於可能的推理測試)
        # bigvgan_checkpoint_path = os.path.join(self.finetune_dir, self.config.bigvgan_checkpoint)
        # self.bigvgan = BigVGAN(self.config.bigvgan)
        # bigvgan_state_dict = torch.load(bigvgan_checkpoint_path, map_location="cpu", weights_only=True)["generator"]
        # self.bigvgan.load_state_dict(bigvgan_state_dict, strict=True)
        # self.bigvgan.remove_weight_norm()
        # self.bigvgan.eval().to(self.device)
        # logger.info("BigVGAN model loaded.")
        # del bigvgan_state_dict
        
        # UnifiedVoice
        gpt_checkpoint_path = os.path.join(self.finetune_dir, self.config.gpt_checkpoint)
        self.model = load_UnifiedVoice(self.config.gpt, gpt_checkpoint_path, self.device)
        logger.info("UnifiedVoice base model loaded.")
    
        # 應用 LoRA（這會凍結所有引數）
        self.model = self._apply_lora(self.model)
        logger.info("LoRA applied to the model.")

        # 多 GPU 支援：使用 DataParallel 包裝模型
        if self.use_multi_gpu and self.gpu_manager and self.gpu_manager.get_gpu_count() > 1:
            logger.info("🚀 使用 DataParallel 包裝模型進行多 GPU 訓練")
            # 明確指定使用所有可用的 GPU
            device_ids = list(range(torch.cuda.device_count()))
            logger.info(f"  使用的 GPU 裝置: {device_ids}")
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
            logger.info(f"  模型已分散到 {len(device_ids)} 個 GPU")

        # 註冊多說話人的mean_condition作為可學習引數
        self.speaker_mean_conditions = {}

        # 獲取實際模型（處理 DataParallel 包裝）
        actual_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        from tqdm import tqdm
        pbar = tqdm(self.speaker_conditions.items(), desc="Register speaker conditions", ncols=100)
        for speaker_id, condition in pbar:
            # 驗證和調整形狀
            if condition.ndim == 2:  # (32, dim) -> (1, 32, dim)
                condition = condition.unsqueeze(0)
            elif condition.ndim != 3:
                raise ValueError(f"Expected condition tensor to have 2 or 3 dimensions, got {condition.ndim} for speaker {speaker_id}")

            # 將每個說話人的condition註冊為可學習引數
            param_name = f"mean_condition_{speaker_id}"
            param = torch.nn.Parameter(condition.to(self.device), requires_grad=True)

            # 使用正確的方式註冊引數（註冊到實際模型）
            actual_model.register_parameter(param_name, param)
            self.speaker_mean_conditions[speaker_id] = param

            pbar.set_postfix({"last": speaker_id})
        pbar.close()

        logger.info(f"Loaded and registered {len(self.speaker_mean_conditions)} speaker conditions.")

    def _apply_lora(self, model: UnifiedVoice) -> UnifiedVoice:
        """為模型配置並應用 LoRA。"""
        lora_cfg = self.config.train.lora
        gpt_lora_config = LoraConfig(
            r=lora_cfg.r,
            target_modules=lora_cfg.target_modules,
            task_type=TaskType.CAUSAL_LM,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
            bias="none",
        )
        model.requires_grad_(False)
        model.inference_model = get_peft_model(model.inference_model, gpt_lora_config)
        return model

    def _setup_optimizer_and_scheduler(self, num_training_steps: int = 1000):
        """建立最佳化器和學習率排程器。"""
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
        logger.info("Optimizer (LoRA+) and Scheduler (CosineAnnealingWithWarmup) created.")

    def _train_step(self, data_batch: tuple) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        執行單個訓練步驟（前向和後向傳播）。
        
        混合精度處理說明：
        - autocast 區塊內：前向運算使用低精度 (BF16/FP16)
        - Loss 計算：自動提升為 FP32
        - 梯度計算：與前向精度一致
        - FP16 模式：使用 GradScaler 防止梯度下溢
        
        Args:
            data_batch: 包含 mel_spec, mel_codes, text_ids, conditions, 
                       speaker_ids, mel_lengths, codes_lengths, text_lengths
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, dict]: (text_loss, mel_loss, mel_accuracy)
        """
        self.model.train()
        # 獲取實際模型（處理 DataParallel 包裝）
        actual_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        actual_model.inference_model.kv_cache = False  # 訓練時停用 KV 快取
    
        # Unpack data_batch: mel_spec, mel_codes, text_ids, conditions, speaker_ids, mel_lengths, codes_lengths, text_lengths
        mel_spec, mel_codes, text_ids, conditions, speaker_ids, mel_lengths, codes_lengths, text_lengths = data_batch

        # 使用混合精度訓練
        if self.use_amp and self.train_dtype:
            with torch.autocast(device_type='cuda', dtype=self.train_dtype):
                outputs = forward_UnifiedVoice(
                    self.model,
                    mel_spec,
                    mel_codes,
                    text_ids,
                    mel_lengths,
                    codes_lengths,
                    text_lengths,
                    speaker_ids=speaker_ids,
                    output_loss=True,
                    output_logits=True,
                )
        else:
            # FP32 訓練
            outputs = forward_UnifiedVoice(
                self.model,
                mel_spec,
                mel_codes,
                text_ids,
                mel_lengths,
                codes_lengths,
                text_lengths,
                speaker_ids=speaker_ids,
                output_loss=True,
                output_logits=True,
            )

        # Loss 會自動轉為 FP32（PyTorch autocast 特性）
        loss_text, loss_mel = outputs["loss"]
        mel_accuracy = outputs.get("mel_accuracy", {"acc_1": 0.0, "acc_10": 0.0, "acc_20": 0.0})
        return loss_text, loss_mel, mel_accuracy

    @torch.no_grad()
    def _validate_epoch(self, valid_ds: Dataset, epoch: int):
        """在驗證集上評估模型。"""
        self.model.eval()
        logger.info(f"Validating at epoch {epoch + 1}...")
        
        total_text_loss, total_mel_loss = 0.0, 0.0
        total_text_tokens, total_mel_tokens = 0, 0
        all_mel_logits, all_mel_targets = [], []
        num_batches = 0

        for batch in tqdm(valid_ds, desc="Validation", dynamic_ncols=True):
            # 正確處理新的 data_batch 格式，區分 tensor 和非 tensor 資料
            data_batch = []
            for item in batch:
                if torch.is_tensor(item):
                    data_batch.append(item.to(self.device))
                else:
                    data_batch.append(item)  # speaker_ids 是字串列表，不需要 .to(device)

            # 正確解包新格式：mel_spec, mel_codes, text_ids, conditions, speaker_ids, mel_lengths, codes_lengths, text_lengths
            mel_spec, mel_codes, text_ids, conditions, speaker_ids, mel_lengths, codes_lengths, text_lengths = data_batch

            # 使用混合精度驗證
            if self.use_amp and self.train_dtype:
                with torch.autocast(device_type='cuda', dtype=self.train_dtype):
                    outputs = forward_UnifiedVoice(
                        self.model,
                        mel_spec,
                        mel_codes,
                        text_ids,
                        mel_lengths,
                        codes_lengths,
                        text_lengths,
                        speaker_ids=speaker_ids,
                        output_loss=True,
                        output_logits=True,
                    )
            else:
                outputs = forward_UnifiedVoice(
                    self.model,
                    mel_spec,
                    mel_codes,
                    text_ids,
                    mel_lengths,
                    codes_lengths,
                    text_lengths,
                    speaker_ids=speaker_ids,
                    output_loss=True,
                    output_logits=True,
                )
            
            loss_text, loss_mel = outputs["loss"]
            batch_text_tokens = text_lengths.sum().item()
            batch_mel_tokens = (codes_lengths + 1).sum().item()  # +1 for stop token

            total_text_loss += loss_text.item() * batch_text_tokens
            total_mel_loss += loss_mel.item() * batch_mel_tokens
            total_text_tokens += batch_text_tokens
            total_mel_tokens += batch_mel_tokens
            num_batches += 1

            # Collect logits and targets for accuracy calculation
            # 僅使用 MEL 部分的 logits 和 targets
            current_mel_logits = outputs["logits"][1]  # mel logits [B, V, L]
            current_mel_targets = outputs["targets"][1]  # mel targets [B, L]
            if current_mel_logits.numel() > 0 and current_mel_targets.numel() > 0:
                # Create mask based on actual sequence lengths instead of assuming 0 is padding
                batch_size = current_mel_targets.size(0)
                mel_mask = torch.zeros_like(current_mel_targets, dtype=torch.bool)
                for i in range(batch_size):
                    actual_mel_len = codes_lengths[i].item() + 1  # +1 for stop token if add_mel_stop_token is True
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
        
        # 計算整體準確率
        all_mel_logits = torch.cat(all_mel_logits, dim=0)
        all_mel_targets = torch.cat(all_mel_targets, dim=0)
        acc_1, acc_10, acc_20 = top_k_accuracy(all_mel_logits, all_mel_targets, k=(1, 10, 20))

        logger.info(f"**Validation results at epoch {epoch + 1}**")
        logger.info(f"Text Loss: {avg_text_loss:.4f}, Mel Loss: {avg_mel_loss:.4f}")
        logger.info(f"Accuracy@1: {acc_1:.2f}%, Accuracy@10: {acc_10:.2f}%, Accuracy@20: {acc_20:.2f}%")
        
        return avg_text_loss, avg_mel_loss, acc_1, acc_10, acc_20

    def _save_checkpoint(self, file_name: str, merge_lora: bool, unload_after_merge: bool):
        """儲存模型檢查點，包含說話人資訊。"""
        checkpoint_path = os.path.join(self.checkpoint_dir, file_name)

        self.model.eval()

        # 獲取實際模型（處理 DataParallel 包裝）
        actual_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model_to_save = actual_model

        if merge_lora:
            logger.info("Merging LoRA weights into the model for saving...")
            if unload_after_merge:
                # 為了在不影響繼續訓練的情況下儲存完全融合的模型，我們建立一個深複製
                # 注意：這可能會消耗較多記憶體和時間
                logger.info("Creating a deep copy of the model for a clean merge. This may take a moment...")
                model_to_save = copy.deepcopy(actual_model)

                # 在深複製上進行融合與解除安裝
                fused_inference_model = model_to_save.inference_model.merge_and_unload()
                model_to_save.inference_model = fused_inference_model
                logger.info("LoRA weights merged and unloaded in the copied model.")
            else:
                # 如果只是臨時融合，直接在原模型上操作，後續再unmerge
                actual_model.inference_model.merge_adapter()
    
        # 儲存模型狀態和說話人資訊
        state_dict = model_to_save.state_dict()
        checkpoint_data = {
            'model': state_dict,
            'speakers': self.speaker_list,  # 儲存說話人列表
            'speaker_conditions': {speaker_id: param.detach().cpu().numpy() 
                                 for speaker_id, param in self.speaker_mean_conditions.items()}
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Checkpoint saved to: {checkpoint_path}")
        logger.info(f"Saved conditions for speakers: {self.speaker_list}")
    
        # 如果建立了深複製，清理它
        if merge_lora and unload_after_merge:
            del model_to_save
            clear_torch_cache()
            logger.info("Cleaned up the temporary merged model.")
        
        # 如果是臨時融合，恢復模型狀態以便繼續訓練
        if merge_lora and not unload_after_merge:
            logger.info("Unmerging LoRA weights to continue training...")
            actual_model.inference_model.unmerge_adapter()

        self.model.train()

    def train(self, train_ds: Dataset, valid_ds: Dataset):
        """
        啟動完整的訓練流程。

        Args:
            train_ds (Dataset): 訓練資料集。
            valid_ds (Dataset): 驗證資料集。
        """
        train_cfg = self.config.train
        total_ds_count = len(train_ds)
        
        samples_per_epoch = total_ds_count
        total_update_steps = samples_per_epoch * train_cfg.epochs
        self._setup_optimizer_and_scheduler(num_training_steps=total_update_steps)
        
        logger.info(f"Starting training for {train_cfg.epochs} epochs.")
        logger.info(f"Total samples per epoch: {samples_per_epoch}")
        logger.info(f"Total update steps: {total_update_steps}")

        text_weight = train_cfg.text_weight

        for epoch in range(train_cfg.epochs):
            logger.info(f"EPOCH {epoch + 1}/{train_cfg.epochs} started" + "=" * 30)

            for batch_idx, batch in enumerate(train_ds):
                # 正確處理新的 data_batch 格式，區分 tensor 和非 tensor 資料
                data_batch = []
                for item in batch:
                    if torch.is_tensor(item):
                        data_batch.append(item.to(self.device))
                    else:
                        data_batch.append(item)  # speaker_ids 是字串列表，不需要 .to(device)

                loss_text, loss_mel, mel_accuracy = self._train_step(tuple(data_batch))
                acc_1, acc_10, acc_20 = mel_accuracy["acc_1"], mel_accuracy["acc_10"], mel_accuracy["acc_20"]

                weighted_loss = text_weight * loss_text + (1.0 - text_weight) * loss_mel
                if torch.isnan(weighted_loss) or torch.isinf(weighted_loss):
                    logger.warning(f"NaN or Inf loss at epoch {epoch}, batch {batch_idx}. Skipping.")
                    continue

                # ------------------ Optimisation Step ------------------
                # 混合精度梯度處理：
                # - BF16: 直接 backward，不需要 scaling
                # - FP16: 使用 GradScaler 防止梯度下溢
                self.optimizer.zero_grad()
                
                if self.grad_scaler is not None:
                    # FP16 模式：使用 GradScaler
                    self.grad_scaler.scale(weighted_loss).backward()
                    self.grad_scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), train_cfg.max_grad_norm)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    # BF16/FP32 模式：直接 backward
                    weighted_loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), train_cfg.max_grad_norm)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.update_steps += 1

                # Logging
                logger.info(
                    f"Epoch {epoch + 1}/{train_cfg.epochs} | Batch {batch_idx + 1}/{len(train_ds)} | "
                    f"text_loss={loss_text.item():.4f}, mel_loss={loss_mel.item():.4f}, "
                    f"acc@1={acc_1:.2f}%, acc@10={acc_10:.2f}%, acc@20={acc_20:.2f}%, "
                    f"grad_norm={grad_norm.item():.2f}"
                )

            # --- Epoch End ---
            val_text_loss, val_mel_loss, _, _, _ = self._validate_epoch(valid_ds, epoch)
            
            # 每個epoch結束後都儲存當前模型
            epoch_checkpoint_name = f"gpt_epoch_{epoch + 1}.pth"
            logger.info(f"Saving model for epoch {epoch + 1}: {epoch_checkpoint_name}")
            self._save_checkpoint(epoch_checkpoint_name, merge_lora=True, unload_after_merge=True)
            
            # 檢查早停
            #patience = train_cfg.early_stopping_patience
            #if epoch > 0 and val_mel_loss >= self.best_val_loss[2] and (epoch - self.best_val_loss[0]) >= patience:
            #    logger.info(
            #        f"Early stopping at epoch {epoch + 1} as validation mel_loss has not improved for {patience} epochs."
            #    )
            #    break

            if val_mel_loss < self.best_val_loss[2]:
                logger.info(f"New best validation mel_loss: {val_mel_loss:.4f}. Saving best model.")
                self.best_val_loss = (epoch, val_text_loss, val_mel_loss)
                self._save_checkpoint("gpt_best.pth", merge_lora=True, unload_after_merge=True)

            clear_torch_cache()

        # --- Training End ---
        logger.info("Training finished. Saving final model.")
        self._save_checkpoint("gpt_finetuned.pth", merge_lora=True, unload_after_merge=True)
        
        # 儲存最終配置
        final_config_path = os.path.join(self.finetune_dir, "config_finetuned.yaml")
        final_config = self.config.copy()
        final_config.gpt_checkpoint = "checkpoints/gpt_finetuned.pth"
        OmegaConf.save(final_config, final_config_path)
        logger.info(f"Final config saved to {final_config_path}")
        
        logger.info(f"Best validation loss at epoch {self.best_val_loss[0] + 1}: "
                    f"text_loss: {self.best_val_loss[1]:.4f}, mel_loss: {self.best_val_loss[2]:.4f}")

def main():
    config_path = "finetune_models/config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}.")
    
    config = OmegaConf.load(config_path)
    bpe_model_path = os.path.join(config.train.finetune_model_dir, config.dataset.bpe_model)

    # 使用新的多說話人資料載入函式，傳遞BPE路徑而不是物件
    train_ds, valid_ds = load_finetune_datasets(config, bpe_model_path) 
    train_ds = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_finetune_fn, num_workers=4)
    valid_ds = DataLoader(valid_ds, batch_size=8, shuffle=False, collate_fn=collate_finetune_fn, num_workers=2)

    trainer = Trainer(config)
    trainer.train(train_ds, valid_ds)
    logger.info("Finetuning UnifiedVoice completed.")


if __name__ == "__main__":
    main()
