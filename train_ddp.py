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
import datetime
from datetime import datetime as dt
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
from torch.utils.tensorboard import SummaryWriter

from indextts.BigVGAN.models import BigVGAN
from indextts.data_utils import (
    collate_finetune_fn,
    load_finetune_datasets,
)
from indextts.gpt.model import UnifiedVoice

# Import train utilities
from train import (
    load_UnifiedVoice,
    normalize_state_dict_keys,
    clear_torch_cache,
    forward_gpt2,
    forward_UnifiedVoice,
    top_k_accuracy,
)


def setup_ddp(rank: int, world_size: int):
    """初始化 DDP 環境"""
    # 設定預設值（如果 torchrun 沒有設定）
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'

    # Fix for NCCL connection issues
    # 預設禁用 P2P 和 IB 以防止在某些環境（如 Docker 或消費級 GPU）中出現連線逾時或掛死
    if 'NCCL_P2P_DISABLE' not in os.environ:
        os.environ['NCCL_P2P_DISABLE'] = '1'
    if 'NCCL_IB_DISABLE' not in os.environ:
        os.environ['NCCL_IB_DISABLE'] = '1'
    if 'NCCL_NET' not in os.environ:
        os.environ['NCCL_NET'] = 'Socket'

    # 初始化進程組
    # 設定 30 分鐘的超時，防止主進程保存模型時其他進程超時
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30))
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
        # if not self.is_main_process:
        #    logger.remove()  # 移除其他進程的日誌輸出

        # 設定隨機種子
        self._set_seed(self.config.train.seed + rank)  # 每個 rank 不同的種子

        # 準備目錄和日誌
        self.finetune_dir = self.config.train.finetune_model_dir
        self.checkpoint_dir = os.path.join(self.finetune_dir, "checkpoints")
        # 為文字 log 與 TensorBoard 統一使用同一個 run 名稱/目錄（使用絕對路徑避免 cwd 變動）
        # 若外部（run.sh）已指定 RUN_NAME/RUN_LOG_DIR，則沿用同一個名字與路徑，避免產生多個目錄
        env_run_name = os.environ.get("RUN_NAME")
        env_log_dir = os.environ.get("RUN_LOG_DIR")
        if env_run_name:
            self.run_name = env_run_name
        else:
            self.run_name = f"train_{dt.now().strftime('%Y%m%d_%H%M%S')}"

        if env_log_dir:
            self.log_dir = os.path.abspath(env_log_dir)
        else:
            self.log_dir = os.path.abspath(os.path.join(os.getcwd(), "logs", self.run_name))

        if self.is_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)
            # 文字 log 與 TensorBoard 放在同一目錄，方便對齊時間戳
            log_path = os.path.join(self.log_dir, "train.log")
            self._setup_logging(log_path)
            # Initialize TensorBoard Writer (only on main process)
            self.writer = SummaryWriter(log_dir=self.log_dir)
            logger.info(f"TensorBoard logging to: {self.log_dir}")

        # 載入模型和分詞器
        self._load_models()

        # 初始化訓練狀態（optimizer 會在 train() 裡設定）
        self.optimizer = None
        self.scheduler = None
        self.best_val_loss = (0, float('inf'), float('inf'))
        self.update_steps = 0

    def _set_seed(self, seed: int):
        """設定隨機種子"""
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if self.is_main_process:
            logger.info(f"Set random seed to {seed} for rank {self.rank}")

    def _setup_logging(self, log_path: str):
        """配置日誌記錄器（只在主進程）"""
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
            # conditioning/perceiver 可能在特定分支未被使用，保持 True 避免梯度同步死鎖
            find_unused_parameters=True
        )

        if self.is_main_process:
            logger.info(f"✅ Model wrapped with DDP on {self.world_size} GPUs")

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
            fan_in_fan_out=True,
        )
        model.requires_grad_(False)
        model.inference_model = get_peft_model(model.inference_model, gpt_lora_config)

        # ⚠️ 重要：凍結 conditioning_encoder 和 perceiver_encoder
        #
        # 策略說明：
        # - conditioning_encoder: 凍結（防止編碼內容資訊，保留預訓練的音色提取能力）
        # - perceiver_encoder: 凍結（保留預訓練的 embedding space，防止被覆寫）
        # - GPT LoRA: 訓練（學習如何使用 conditioning）
        #
        # 理由：
        # 1. 預訓練的 encoder 已具備通用音色提取能力
        # 2. 訓練 perceiver 會破壞原本的 embedding space，導致 clone 能力崩潰
        # 3. 僅訓練 LoRA 層，讓模型學習如何使用固定的 conditioning 來生成語音
        if hasattr(model, 'conditioning_encoder'):
            model.conditioning_encoder.requires_grad_(False)
            if self.is_main_process:
                logger.info("✓ conditioning_encoder 已凍結（防止內容洩漏）")
        if hasattr(model, 'perceiver_encoder'):
            model.perceiver_encoder.requires_grad_(False)
            if self.is_main_process:
                logger.info("✓ perceiver_encoder 已凍結（保留 embedding space）")

        return model

    def _setup_optimizer_and_scheduler(self, num_training_steps: int = 0):
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

        warmup_steps = opt_cfg.get("warmup_steps", None)
        if warmup_steps is None:
            warmup_ratio = opt_cfg.get("warmup_ratio", 0.0)
            if warmup_ratio > 0 and num_training_steps > 0:
                warmup_steps = max(1, int(num_training_steps * warmup_ratio))
            else:
                warmup_steps = 0

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(num_training_steps, warmup_steps + 1),
        )

    def train(self, train_ds: Dataset, valid_ds: Dataset, resume_checkpoint: str = None):
        """訓練流程"""
        train_cfg = self.config.train
        start_epoch = 0

        # 嘗試取得加權取樣器（若資料集有實作）
        if hasattr(train_ds, 'get_sampler'):
            train_sampler = train_ds.get_sampler(
                shuffle=True,
                num_replicas=self.world_size,
                rank=self.rank
            )
        else:
            train_sampler = None

        # 若沒有加權取樣器，使用標準 DistributedSampler
        if train_sampler is None:
            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
        elif self.is_main_process:
            logger.info("✓ 使用加權取樣策略 (Weighted Sampling)")

        train_batch_size = train_cfg.get("batch_size", 1)
        train_num_workers = train_cfg.get("num_workers", 2)
        train_loader = DataLoader(
            train_ds,
            batch_size=train_batch_size,
            sampler=train_sampler,
            collate_fn=collate_finetune_fn,
            num_workers=train_num_workers,
            pin_memory=True,
            drop_last=True,  # 避免 DDP 末尾 batch 不齊卡住
        )

        valid_batch_size = train_cfg.get("valid_batch_size", 4)
        valid_num_workers = train_cfg.get("valid_num_workers", 2)
        valid_loader = DataLoader(
            valid_ds,
            batch_size=valid_batch_size,
            shuffle=False,
            collate_fn=collate_finetune_fn,
            num_workers=valid_num_workers,
            pin_memory=True,
            drop_last=True,  # 驗證也對齊 batch 長度
        )

        steps_per_epoch = len(train_loader)
        total_update_steps = steps_per_epoch * train_cfg.epochs

        # 先創建 optimizer 和 scheduler
        self._setup_optimizer_and_scheduler(num_training_steps=total_update_steps)

        # 如果有 checkpoint，在創建 optimizer 後載入狀態
        if resume_checkpoint:
            loaded_epoch = self._load_checkpoint_states(resume_checkpoint)
            if loaded_epoch > 0:
                start_epoch = loaded_epoch
                # 更新 update_steps
                self.update_steps = start_epoch * steps_per_epoch
                
                # 重新計算剩餘步數並更新 scheduler
                remaining_epochs = train_cfg.epochs - start_epoch
                remaining_steps = steps_per_epoch * remaining_epochs
                if self.is_main_process:
                    logger.info(f"Updated global step to {self.update_steps}")
                    logger.info(f"Remaining training steps: {remaining_steps}")

        if self.is_main_process:
            if start_epoch > 0:
                logger.info(f"🔄 Resuming DDP training from epoch {start_epoch + 1}")
            else:
                logger.info(f"Starting DDP training for {train_cfg.epochs} epochs")
            logger.info(f"Steps per epoch (per GPU): {steps_per_epoch}")
            logger.info(f"Total samples: {len(train_ds)}")
            logger.info(f"Total update steps (per GPU): {total_update_steps}")

        text_weight = train_cfg.text_weight

        for epoch in range(start_epoch, train_cfg.epochs):
            # 設定 epoch 以確保 shuffle 正確
            train_sampler.set_epoch(epoch)

            if self.is_main_process:
                logger.info(f"=" * 60)
                logger.info(f"EPOCH {epoch + 1}/{train_cfg.epochs} started")
                logger.info(f"=" * 60)

            self.model.train()

            for batch_idx, batch in enumerate(train_loader):
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
                        logger.warning(f"NaN/Inf loss at epoch {epoch}, batch {batch_idx}. Zeroing loss to maintain DDP sync.")
                    # DDP CRITICAL FIX: 不能直接 continue，否則會導致該 Rank 跳過 backward，造成全體死鎖。
                    # 必須傳送一個 0 的 loss 讓 DDP 完成梯度同步。
                    weighted_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

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
                        f"Batch {batch_idx}/{steps_per_epoch} | "
                        f"text_loss={loss_text.item():.4f}, mel_loss={loss_mel.item():.4f}, "
                        f"acc@1={acc_1:.2f}%, acc@10={acc_10:.2f}%, acc@20={acc_20:.2f}%, "
                        f"grad_norm={grad_norm.item():.2f}"
                    )
                    
                    # TensorBoard Logging
                    self.writer.add_scalar("loss/text", loss_text.item(), self.update_steps)
                    self.writer.add_scalar("loss/mel", loss_mel.item(), self.update_steps)
                    self.writer.add_scalar("loss/total", weighted_loss.item(), self.update_steps)
                    self.writer.add_scalar("accuracy/top1", acc_1, self.update_steps)
                    self.writer.add_scalar("accuracy/top10", acc_10, self.update_steps)
                    self.writer.add_scalar("accuracy/top20", acc_20, self.update_steps)
                    self.writer.add_scalar("train/grad_norm", grad_norm.item(), self.update_steps)
                    self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.update_steps)

            # 驗證（只在主進程）
            if self.is_main_process:
                val_text_loss, val_mel_loss, val_acc1, val_acc10, val_acc20 = self._validate_epoch(valid_loader, epoch)
                self._save_checkpoint(epoch, val_text_loss, val_mel_loss)
                
                # TensorBoard Validation Logging
                self.writer.add_scalar("val/loss_text", val_text_loss, epoch + 1)
                self.writer.add_scalar("val/loss_mel", val_mel_loss, epoch + 1)
                self.writer.add_scalar("val/accuracy_top1", val_acc1, epoch + 1)
                self.writer.add_scalar("val/accuracy_top10", val_acc10, epoch + 1)
                self.writer.add_scalar("val/accuracy_top20", val_acc20, epoch + 1)

            # 同步所有進程
            # print(f"[Rank {self.rank}] Waiting at barrier...", flush=True)
            # dist.barrier(device_ids=[self.rank])
            # print(f"[Rank {self.rank}] Passed barrier!", flush=True)

    def _train_step(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """單個訓練步驟：conditioning 來自同 speaker 的另一段語音"""
        # 解包 batch（新增 cond_mels、cond_lengths）
        mel_spec, mel_codes, text_ids, cond_mels, speaker_ids, mel_lengths, codes_lengths, text_lengths, cond_lengths = batch

        outputs = forward_UnifiedVoice(
            self.model.module,  # DDP 需要使用 .module
            mel_spec,           # target mel（用於 loss）
            mel_codes,
            text_ids,
            mel_lengths,
            codes_lengths,
            text_lengths,
            condition_mels=cond_mels,
            condition_lengths=cond_lengths,
            speaker_ids=None,  # 不用 speaker_id 查表，強制 encoder 學習
            add_mel_stop_token=self.config.train.get('add_mel_stop_token', True),
            output_loss=True,
            output_logits=True,
        )

        loss_text, loss_mel = outputs["loss"]
        mel_accuracy = outputs["mel_accuracy"]

        return loss_text, loss_mel, mel_accuracy

    def _forward_with_precomputed_conditioning(self, *args, **kwargs):
        """Deprecated after dynamic conditioning change."""
        raise NotImplementedError('Use _train_step with dynamic conditioning instead.')

    def _validate_epoch(self, valid_ds: Dataset, epoch: int):
        """驗證（簡化版，只在主進程執行）"""
        self.model.eval()
        total_text_loss = 0.0
        total_mel_loss = 0.0
        num_batches = 0
        total_batches = len(valid_ds)

        logger.info(f"開始驗證 Epoch {epoch + 1}，共 {total_batches} 個 batch")

        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_ds):
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

                # 每 50 個 batch 打印一次進度
                if batch_idx % 50 == 0:
                    logger.info(
                        f"Validation [{batch_idx}/{total_batches}] | "
                        f"val_txt={loss_text.item():.3f}, val_mel={loss_mel.item():.3f}"
                    )

        avg_text_loss = total_text_loss / max(num_batches, 1)
        avg_mel_loss = total_mel_loss / max(num_batches, 1)
        
        # 計算整體準確率 (簡單平均，雖然不完全精確但對於監控足夠)
        # 在 DDP 驗證中，由於我們沒有 gather 所有的 logits，這裡只計算主 GPU 的準確率
        # 若要精確，需要 dist.all_gather
        acc_1 = 0.0
        acc_10 = 0.0
        acc_20 = 0.0
        # 這裡需要補上準確率計算邏輯，但因為原本程式碼沒有收集 logits，暫時回傳 0
        # 若要實作，需要像 train.py 一樣收集 logits
        
        logger.info(f"Validation Epoch {epoch + 1}: text_loss={avg_text_loss:.4f}, mel_loss={avg_mel_loss:.4f}")

        return avg_text_loss, avg_mel_loss, acc_1, acc_10, acc_20

    def _load_checkpoint_states(self, checkpoint_path: str) -> int:
        """
        從 checkpoint 恢復訓練

        Args:
            checkpoint_path: checkpoint 檔案路徑

        Returns:
            start_epoch: 要從哪個 epoch 開始繼續訓練
        """
        if not os.path.exists(checkpoint_path):
            if self.is_main_process:
                logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
            return 0

        if self.is_main_process:
            logger.info(f"📂 Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 載入模型權重
        cleaned_state = normalize_state_dict_keys(checkpoint['model_state_dict'])
        self.model.module.load_state_dict(cleaned_state)
        if self.is_main_process:
            logger.info("✓ Model state loaded")

        # 載入 optimizer 和 scheduler（如果已經初始化）
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.is_main_process:
                logger.info("✓ Optimizer state loaded")

        if hasattr(self, 'scheduler') and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.is_main_process:
                logger.info("✓ Scheduler state loaded")

        start_epoch = checkpoint['epoch'] + 1  # 從下一個 epoch 開始
        if self.is_main_process:
            logger.info(f"✓ Resuming from epoch {start_epoch}")
            logger.info(f"   Last val_text_loss: {checkpoint.get('val_text_loss', 'N/A'):.4f}")
            logger.info(f"   Last val_mel_loss: {checkpoint.get('val_mel_loss', 'N/A'):.4f}")

        return start_epoch

    def _save_checkpoint(self, epoch: int, val_text_loss: float, val_mel_loss: float):
        """儲存 checkpoint（只在主進程）"""
        # 1. 先儲存訓練用的 checkpoint（包含 optimizer 等，用於恢復訓練）
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),  # DDP 需要使用 .module
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_text_loss': val_text_loss,
            'val_mel_loss': val_mel_loss,
        }, checkpoint_path)
        logger.info(f"💾 Training checkpoint saved: {checkpoint_path}")

        # 2. 儲存合併後的模型（用於推理）
        merged_model_path = os.path.join(self.checkpoint_dir, f"gpt_epoch_{epoch + 1}.pth")
        logger.info("🔄 Merging LoRA weights for inference model...")

        # 獲取實際模型（DDP wrapper）
        actual_model = self.model.module

        # 建立深複製以避免影響繼續訓練
        import copy
        logger.info("Creating a deep copy of the model for merge...")
        model_to_save = copy.deepcopy(actual_model)

        # 在深複製上進行 LoRA 融合與解除安裝
        fused_inference_model = model_to_save.inference_model.merge_and_unload()
        model_to_save.inference_model = fused_inference_model
        logger.info("✓ LoRA weights merged and unloaded")

        # 選擇儲存精度（預設 fp16，與底模一致，體積較小）
        save_dtype = self.config.train.get("save_dtype", "fp16")
        if save_dtype == "fp16":
            model_to_save = model_to_save.half()
            logger.info("💾 Saving merged model in FP16")
        elif save_dtype == "bf16":
            model_to_save = model_to_save.bfloat16()
            logger.info("💾 Saving merged model in BF16")
        else:
            logger.info("💾 Saving merged model in FP32")

        # 儲存完整模型（格式與 train.py 一致）
        # 排除 inference_model.* 避免重複儲存 gpt 參數
        state_dict = model_to_save.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('inference_model')}
        checkpoint_data = {'model': filtered_state_dict}

        torch.save(checkpoint_data, merged_model_path)
        logger.info(f"💾 Merged model saved: {merged_model_path}")

        # 清理深複製
        del model_to_save
        torch.cuda.empty_cache()
        logger.info("✓ Cleaned up temporary merged model")


def main():
    """主函數"""
    import argparse
    parser = argparse.ArgumentParser(description='DDP Training with resume support')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from (e.g., finetune_models/checkpoints/checkpoint_epoch_5.pt)')
    args = parser.parse_args()

    # 獲取 DDP 環境變數
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("❌ 請使用 torch.distributed.launch 或 torchrun 啟動此腳本")
        print("範例: torchrun --nproc_per_node=8 train_ddp.py")
        print("恢復訓練: torchrun --nproc_per_node=8 train_ddp.py --resume finetune_models/checkpoints/checkpoint_epoch_5.pt")
        return

    # 初始化 DDP
    setup_ddp(local_rank, world_size)

    try:
        # 載入配置
        config = OmegaConf.load("finetune_models/config.yaml")

        # 創建訓練器
        trainer = DDPTrainer(config, local_rank, world_size)

        # 載入資料集
        bpe_model_path = os.path.join(
            config.train.finetune_model_dir,
            config.dataset.bpe_model
        )
        train_ds, valid_ds = load_finetune_datasets(config, bpe_model_path)

        # 開始訓練（resume_checkpoint 會在 train() 內部處理）
        trainer.train(train_ds, valid_ds, resume_checkpoint=args.resume)

    finally:
        # 清理
        if hasattr(trainer, 'writer'):
            trainer.writer.close()
        cleanup_ddp()


if __name__ == "__main__":
    main()
