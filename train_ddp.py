#!/usr/bin/env python3
"""
DistributedDataParallel (DDP) 訓練腳本 - 多 GPU 高效訓練

相比 DataParallel:
- 更高的訓練效率（無主 GPU 瓶頸）
- 更好的記憶體利用
- 可以跨節點擴展

使用方法:
    # 單機多 GPU（推薦）
    python -m torch.distributed.launch --nproc_per_node=8 train_ddp.py

    # 或使用 torchrun（PyTorch 1.10+）
    torchrun --nproc_per_node=8 train_ddp.py

    # 指定特定 GPU
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_ddp.py

Author: TTS ETL Pipeline
Version: 1.0
"""

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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from peft.optimizers import create_loraplus_optimizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from indextts.BigVGAN.models import BigVGAN
from indextts.data_utils import (
    collate_finetune_fn,
    load_finetune_datasets,
    load_speaker_conditions,
)
from indextts.gpt.model import UnifiedVoice

# Import train utilities
from train import (
    load_UnifiedVoice,
    clear_torch_cache,
    forward_gpt2,
    forward_fn,
    top_k_accuracy,
)


def setup_ddp(rank: int, world_size: int):
    """初始化 DDP 環境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 初始化進程組
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """清理 DDP 環境"""
    dist.destroy_process_group()


class DDPTrainer:
    """DDP 訓練器 - 高效多 GPU 訓練"""

    def __init__(self, config: DictConfig, rank: int, world_size: int):
        """
        初始化 DDP 訓練器

        Args:
            config: 配置物件
            rank: 當前進程的 GPU ID (0-7)
            world_size: 總 GPU 數量 (8)
        """
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")
        self.is_main_process = (rank == 0)

        # 只有主進程打印訊息
        if not self.is_main_process:
            logger.remove()  # 移除其他進程的日誌輸出

        # 設定隨機種子
        self._set_seed(self.config.train.seed + rank)  # 每個 rank 不同的種子

        # 準備目錄和日誌
        self.finetune_dir = self.config.train.finetune_model_dir
        self.checkpoint_dir = os.path.join(self.finetune_dir, "checkpoints")
        if self.is_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self._setup_logging()

        # 載入說話人條件向量
        self.speaker_conditions = load_speaker_conditions(config)
        self.speaker_list = list(self.speaker_conditions.keys())
        if self.is_main_process:
            logger.info(f"Loaded conditions for {len(self.speaker_list)} speakers: {self.speaker_list}")

        # 載入模型和分詞器
        self._load_models()

        # 設定最佳化器和排程器
        self._setup_optimizer_and_scheduler()

        # 初始化訓練狀態
        self.best_val_loss = (0, float('inf'), float('inf'))
        self.update_steps = 0

    def _set_seed(self, seed: int):
        """設定隨機種子"""
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if self.is_main_process:
            logger.info(f"Set random seed to {seed} for rank {self.rank}")

    def _setup_logging(self):
        """配置日誌記錄器（只在主進程）"""
        log_path = os.path.join(self.checkpoint_dir, f"train_ddp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        logger.add(log_path, level="INFO", encoding="utf-8")
        logger.info(f"🚀 DDP Training with {self.world_size} GPUs")
        logger.info("Full configuration:\n" + OmegaConf.to_yaml(self.config))

    def _load_models(self):
        """載入模型"""
        if self.is_main_process:
            logger.info("Loading models...")

        # BPE
        bpe_model_path = os.path.join(self.finetune_dir, self.config.dataset.bpe_model)
        self.bpe_model = spm.SentencePieceProcessor(bpe_model_path)

        # UnifiedVoice
        gpt_checkpoint_path = os.path.join(self.finetune_dir, self.config.gpt_checkpoint)
        self.model = load_UnifiedVoice(self.config.gpt, gpt_checkpoint_path, self.device)

        # 應用 LoRA
        self.model = self._apply_lora(self.model)

        # 使用 DDP 包裝模型
        self.model = DDP(
            self.model,
            device_ids=[self.rank],
            output_device=self.rank,
            find_unused_parameters=False  # 設為 True 如果有未使用的參數
        )

        if self.is_main_process:
            logger.info(f"✅ Model wrapped with DDP on {self.world_size} GPUs")

        # 註冊說話人條件
        self.speaker_mean_conditions = {}
        for speaker_id, condition in self.speaker_conditions.items():
            if condition.ndim == 2:
                condition = condition.unsqueeze(0)
            param = torch.nn.Parameter(condition.to(self.device), requires_grad=True)
            param_name = f"mean_condition_{speaker_id}"
            self.model.module.register_parameter(param_name, param)
            self.speaker_mean_conditions[speaker_id] = param

        if self.is_main_process:
            logger.info(f"Loaded {len(self.speaker_mean_conditions)} speaker conditions")

    def _apply_lora(self, model: UnifiedVoice) -> UnifiedVoice:
        """應用 LoRA 配置"""
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
        """設定最佳化器和排程器"""
        opt_cfg = self.config.train.optimizer
        self.optimizer = create_loraplus_optimizer(
            model=self.model,
            optimizer_cls=AdamW,
            lr=opt_cfg.learning_rate,
            loraplus_lr_ratio=opt_cfg.loraplus_lr_ratio,
            loraplus_lr_embedding=opt_cfg.get("loraplus_lr_embedding", 1e-6),
            weight_decay=opt_cfg.weight_decay,
        )

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=opt_cfg.warmup_steps,
            num_training_steps=num_training_steps,
        )

    def train(self, train_ds: Dataset, valid_ds: Dataset):
        """訓練流程"""
        train_cfg = self.config.train

        # 使用 DistributedSampler
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )

        # 計算訓練步數（注意：DDP 下每個進程看到的資料量是總量的 1/world_size）
        samples_per_epoch = len(train_ds) // self.world_size
        total_update_steps = samples_per_epoch * train_cfg.epochs
        self._setup_optimizer_and_scheduler(num_training_steps=total_update_steps)

        if self.is_main_process:
            logger.info(f"Starting DDP training for {train_cfg.epochs} epochs")
            logger.info(f"Samples per epoch (per GPU): {samples_per_epoch}")
            logger.info(f"Total samples: {len(train_ds)}")
            logger.info(f"Total update steps (per GPU): {total_update_steps}")

        text_weight = train_cfg.text_weight

        for epoch in range(train_cfg.epochs):
            # 設定 epoch 以確保 shuffle 正確
            train_sampler.set_epoch(epoch)

            if self.is_main_process:
                logger.info(f"=" * 60)
                logger.info(f"EPOCH {epoch + 1}/{train_cfg.epochs} started")
                logger.info(f"=" * 60)

            self.model.train()

            for batch_idx, batch in enumerate(train_ds):
                # 將資料移到對應的 GPU
                data_batch = []
                for item in batch:
                    if torch.is_tensor(item):
                        data_batch.append(item.to(self.device))
                    else:
                        data_batch.append(item)

                loss_text, loss_mel, mel_accuracy = self._train_step(tuple(data_batch))
                acc_1, acc_10, acc_20 = mel_accuracy["acc_1"], mel_accuracy["acc_10"], mel_accuracy["acc_20"]

                weighted_loss = text_weight * loss_text + (1.0 - text_weight) * loss_mel

                if torch.isnan(weighted_loss) or torch.isinf(weighted_loss):
                    if self.is_main_process:
                        logger.warning(f"NaN/Inf loss at epoch {epoch}, batch {batch_idx}. Skipping.")
                    continue

                # 最佳化步驟
                self.optimizer.zero_grad()
                weighted_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), train_cfg.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.update_steps += 1

                # 只有主進程打印日誌
                if self.is_main_process and batch_idx % 10 == 0:  # 每 10 batch 打印一次
                    logger.info(
                        f"[GPU 0/{self.world_size}] Epoch {epoch + 1}/{train_cfg.epochs} | "
                        f"Batch {batch_idx}/{len(train_ds)} | "
                        f"text_loss={loss_text.item():.4f}, mel_loss={loss_mel.item():.4f}, "
                        f"acc@1={acc_1:.2f}%, acc@10={acc_10:.2f}%, acc@20={acc_20:.2f}%, "
                        f"grad_norm={grad_norm.item():.2f}"
                    )

            # 驗證（只在主進程）
            if self.is_main_process:
                val_text_loss, val_mel_loss, _, _, _ = self._validate_epoch(valid_ds, epoch)
                self._save_checkpoint(epoch, val_text_loss, val_mel_loss)

            # 同步所有進程
            dist.barrier()

    def _train_step(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """單個訓練步驟"""
        # 解包 batch（與原始 Trainer 相同）
        text_ids, text_lengths, mel_spec, mel_lengths, mel_codes, codes_lengths, speaker_ids = batch

        # 使用說話人條件
        batch_speaker_ids = list(speaker_ids)
        speaker_means = [self.speaker_mean_conditions[sid] for sid in batch_speaker_ids]
        speaker_means = torch.cat(speaker_means, dim=0)

        # 前向傳播
        outputs = forward_fn(
            self.model.module,  # DDP 需要使用 .module
            text_ids,
            text_lengths,
            mel_spec,
            mel_lengths,
            mel_codes,
            codes_lengths,
            speaker_ids=batch_speaker_ids,
            output_loss=True,
            output_logits=False,
            add_mel_stop_token=self.config.train.add_mel_stop_token,
        )

        loss_text, loss_mel = outputs["loss"]
        mel_accuracy = outputs["mel_accuracy"]

        return loss_text, loss_mel, mel_accuracy

    def _validate_epoch(self, valid_ds: Dataset, epoch: int):
        """驗證（簡化版，只在主進程執行）"""
        self.model.eval()
        total_text_loss = 0.0
        total_mel_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in valid_ds:
                data_batch = []
                for item in batch:
                    if torch.is_tensor(item):
                        data_batch.append(item.to(self.device))
                    else:
                        data_batch.append(item)

                loss_text, loss_mel, _ = self._train_step(tuple(data_batch))
                total_text_loss += loss_text.item()
                total_mel_loss += loss_mel.item()
                num_batches += 1

        avg_text_loss = total_text_loss / max(num_batches, 1)
        avg_mel_loss = total_mel_loss / max(num_batches, 1)

        logger.info(f"Validation Epoch {epoch + 1}: text_loss={avg_text_loss:.4f}, mel_loss={avg_mel_loss:.4f}")

        return avg_text_loss, avg_mel_loss, 0.0, 0.0, 0.0

    def _save_checkpoint(self, epoch: int, val_text_loss: float, val_mel_loss: float):
        """儲存 checkpoint（只在主進程）"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),  # DDP 需要使用 .module
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_text_loss': val_text_loss,
            'val_mel_loss': val_mel_loss,
        }, checkpoint_path)
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")


def main():
    """主函數"""
    # 獲取 DDP 環境變數
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("❌ 請使用 torch.distributed.launch 或 torchrun 啟動此腳本")
        print("範例: torchrun --nproc_per_node=8 train_ddp.py")
        return

    # 初始化 DDP
    setup_ddp(local_rank, world_size)

    try:
        # 載入配置
        config = OmegaConf.load("config.yaml")

        # 創建訓練器
        trainer = DDPTrainer(config, local_rank, world_size)

        # 載入資料集
        train_ds, valid_ds = load_finetune_datasets(config)

        # 開始訓練
        trainer.train(train_ds, valid_ds)

    finally:
        # 清理
        cleanup_ddp()


if __name__ == "__main__":
    main()
