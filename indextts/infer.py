import json  # 新增這行
import os
import sys
import time
import warnings
from subprocess import CalledProcessError
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import random

import numpy as np

from indextts.BigVGAN.models import BigVGAN as Generator
from indextts.gpt.model import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.feature_extractors import MelSpectrogramFeatures
from indextts.utils.front import TextNormalizer, TextTokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_parent_module(model: nn.Module, name: str) -> nn.Module:
    """
    獲取模型中指定名稱的父模組。
    
    Args:
        model: 根模型
        name: 完整的模組路徑名稱（如 'gpt.h.0.attn.c_attn'）
    
    Returns:
        父模組
    """
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent


def _quantize_linear_layers_to_int8(model: nn.Module, target_modules: Optional[List[str]] = None, verbose: bool = True) -> int:
    """
    將模型中的 nn.Linear 層替換為 bitsandbytes 的 Linear8bitLt。
    
    Args:
        model: 要量化的模型
        target_modules: 要量化的模組名稱列表。如果為 None，則量化所有 Linear 層。
                       例如: ['gpt', 'text_head', 'mel_head']
        verbose: 是否輸出詳細日誌
    
    Returns:
        替換的層數
    
    Note:
        此函數會就地修改模型。
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "bitsandbytes 未安裝。請使用以下命令安裝：\n"
            "pip install bitsandbytes"
        )
    
    replaced_count = 0
    total_params_before = 0
    total_params_after = 0
    
    # 收集所有需要替換的模組
    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 檢查是否在目標模組列表中
            if target_modules is not None:
                should_replace = any(name.startswith(target) or name == target for target in target_modules)
                if not should_replace:
                    continue
            
            modules_to_replace.append((name, module))
    
    if verbose:
        print(f">> [量化] 找到 {len(modules_to_replace)} 個可量化的 Linear 層")
    
    # 替換模組
    for name, module in modules_to_replace:
        parent = _get_parent_module(model, name)
        child_name = name.split('.')[-1]
        
        # 計算參數量（用於顯存估算）
        param_count = module.in_features * module.out_features
        total_params_before += param_count * 4  # FP32 = 4 bytes
        total_params_after += param_count * 1   # INT8 = 1 byte
        
        # 創建 8bit Linear 層
        has_bias = module.bias is not None
        quantized_linear = bnb.nn.Linear8bitLt(
            module.in_features,
            module.out_features,
            bias=has_bias,
            has_fp16_weights=False,
            threshold=6.0,  # 離群值閾值
        )
        
        # 複製權重（bitsandbytes 會自動量化）
        quantized_linear.weight = bnb.nn.Int8Params(
            module.weight.data.contiguous(),
            requires_grad=False
        )
        if has_bias:
            quantized_linear.bias = nn.Parameter(module.bias.data.clone())
        
        # 替換模組
        setattr(parent, child_name, quantized_linear)
        replaced_count += 1
        
        if verbose and replaced_count <= 5:
            print(f">>   - 量化: {name} ({module.in_features}x{module.out_features})")
    
    if verbose and replaced_count > 5:
        print(f">>   - ... 還有 {replaced_count - 5} 個層")
    
    if verbose:
        mem_before_mb = total_params_before / (1024 * 1024)
        mem_after_mb = total_params_after / (1024 * 1024)
        savings_pct = (1 - total_params_after / total_params_before) * 100 if total_params_before > 0 else 0
        print(f">> [量化] 權重記憶體: {mem_before_mb:.1f}MB → {mem_after_mb:.1f}MB (節省 {savings_pct:.0f}%)")
    
    return replaced_count


def _quantize_linear_layers_to_int4(model: nn.Module, target_modules: Optional[List[str]] = None, verbose: bool = True, compute_dtype: torch.dtype = torch.bfloat16) -> int:
    """
    將模型中的 nn.Linear 層替換為 bitsandbytes 的 Linear4bit (NF4)。

    Args:
        model: 要量化的模型
        target_modules: 要量化的模組名稱列表
        verbose: 是否輸出詳細日誌
        compute_dtype: 量化層的運算精度（預設 BF16）

    Returns:
        替換的層數
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "bitsandbytes 未安裝。請使用以下命令安裝：\n"
            "pip install bitsandbytes"
        )
    
    replaced_count = 0
    total_params_before = 0
    total_params_after = 0
    
    # 收集所有需要替換的模組
    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if target_modules is not None:
                should_replace = any(name.startswith(target) or name == target for target in target_modules)
                if not should_replace:
                    continue
            modules_to_replace.append((name, module))
    
    if verbose:
        print(f">> [量化] 找到 {len(modules_to_replace)} 個可量化的 Linear 層")
    
    # 替換模組
    for name, module in modules_to_replace:
        parent = _get_parent_module(model, name)
        child_name = name.split('.')[-1]
        
        # 計算參數量
        param_count = module.in_features * module.out_features
        total_params_before += param_count * 4   # FP32 = 4 bytes
        total_params_after += param_count * 0.5  # INT4 = 0.5 byte
        
        has_bias = module.bias is not None
        quantized_linear = bnb.nn.Linear4bit(
            module.in_features,
            module.out_features,
            bias=has_bias,
            compute_dtype=compute_dtype,  # 使用配置的運算精度
            quant_type='nf4',  # 使用 NF4 量化
        )
        
        # 複製權重
        quantized_linear.weight = bnb.nn.Params4bit(
            module.weight.data.contiguous(),
            requires_grad=False,
            quant_type='nf4'
        )
        if has_bias:
            quantized_linear.bias = nn.Parameter(module.bias.data.clone())
        
        setattr(parent, child_name, quantized_linear)
        replaced_count += 1
        
        if verbose and replaced_count <= 5:
            print(f">>   - 量化: {name} ({module.in_features}x{module.out_features})")
    
    if verbose and replaced_count > 5:
        print(f">>   - ... 還有 {replaced_count - 5} 個層")
    
    if verbose:
        mem_before_mb = total_params_before / (1024 * 1024)
        mem_after_mb = total_params_after / (1024 * 1024)
        savings_pct = (1 - total_params_after / total_params_before) * 100 if total_params_before > 0 else 0
        print(f">> [量化] 權重記憶體: {mem_before_mb:.1f}MB → {mem_after_mb:.1f}MB (節省 {savings_pct:.0f}%)")
    
    return replaced_count

class IndexTTS:
    def __init__(
        self, cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=True, device=None, use_cuda_kernel=None,
        speaker_info_path=None,  # 新增：說話人資訊檔案路徑
        precision_config=None,  # 新增：細粒度混合精度配置
    ):
        """
        Args:
            cfg_path (str): path to the config file.
            model_dir (str): path to the model directory.
            is_fp16 (bool): whether to use fp16 (deprecated, use precision_config instead).
            device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
            use_cuda_kernel (None | bool): whether to use BigVGan custom fused activation CUDA kernel, only for CUDA device.
            precision_config (dict): 細粒度混合精度配置，例如: {'gpt': 'bf16', 'vocoder': 'bf16'}
        """
        if device is not None:
            self.device = device
            self.is_fp16 = False if device == "cpu" else is_fp16
            self.use_cuda_kernel = use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.is_fp16 = is_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.is_fp16 = False # Use float16 on MPS is overhead than float32
            self.use_cuda_kernel = False
        else:
            self.device = "cpu"
            self.is_fp16 = False
            self.use_cuda_kernel = False
            print(">> Be patient, it may take a while to run in CPU mode.")

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir

        # 處理混合精度配置
        # 優先順序：1. precision_config 參數 -> 2. config_inference.yaml -> 3. config.yaml 的 inference 區塊 -> 4. is_fp16（向後兼容）
        config_source = None
        if precision_config is None:
            # 先嘗試讀取專門的推理配置檔
            inference_config_path = os.path.join(model_dir, "config_inference.yaml")
            if os.path.exists(inference_config_path):
                inference_cfg = OmegaConf.load(inference_config_path)
                if hasattr(inference_cfg, 'inference'):
                    precision_config = inference_cfg.inference
                    config_source = f"config_inference.yaml"
            # 回退到原始 config.yaml 的 inference 區塊
            elif hasattr(self.cfg, 'inference'):
                precision_config = self.cfg.inference
                config_source = "config.yaml [inference]"
        else:
            config_source = "程式碼參數 (precision_config)"

        # 解析精度配置
        def resolve_dtype(precision_str):
            if precision_str in ["bf16", "bfloat16"]:
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            elif precision_str in ["fp16", "float16"]:
                return torch.float16
            elif precision_str in ["fp8"]:
                return torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else torch.bfloat16
            else:  # fp32, no, None
                return torch.float32

        if precision_config and isinstance(precision_config, dict):
            # 讀取推理配置（inference.gpt, inference.vocoder, inference.quantization）
            gpt_precision = precision_config.get('gpt', 'bf16')
            vocoder_precision = precision_config.get('vocoder', 'bf16')

            quant_cfg = precision_config.get('quantization', {})
            quant_enabled = quant_cfg.get('enabled', False)

            # 設定精度和量化
            if quant_enabled:
                # 進階模式：weight_dtype + compute_dtype
                weight_dtype = quant_cfg.get('weight_dtype', 'int8')
                compute_dtype = quant_cfg.get('compute_dtype', 'bf16')

                self.gpt_weight_dtype = weight_dtype
                self.gpt_compute_dtype = resolve_dtype(compute_dtype)
                self.use_quantization = True
                self.load_in_8bit = (weight_dtype == 'int8')
                self.load_in_4bit = (weight_dtype == 'int4')

                print(f">> 使用量化推理 (進階模式) - 配置來源: {config_source}")
                print(f"   - 權重存儲: {weight_dtype.upper()} (省 {'75%' if weight_dtype == 'int8' else '87.5%'} 顯存)")
                print(f"   - 運算精度: {self.gpt_compute_dtype}")
                print(f"   - Vocoder: {vocoder_precision}")

            elif gpt_precision == 'int8':
                # 簡單模式：int8 (權重 INT8 + 運算 BF16)
                self.gpt_weight_dtype = 'int8'
                self.gpt_compute_dtype = torch.bfloat16
                self.use_quantization = True
                self.load_in_8bit = True
                self.load_in_4bit = False
                print(f">> 使用 INT8 量化推理 - 配置來源: {config_source}")
                print(f"   權重=INT8, 運算=BF16, Vocoder={vocoder_precision}")

            elif gpt_precision == 'int4':
                # 簡單模式：int4 (權重 INT4 + 運算 BF16)
                self.gpt_weight_dtype = 'int4'
                self.gpt_compute_dtype = torch.bfloat16
                self.use_quantization = True
                self.load_in_8bit = False
                self.load_in_4bit = True
                print(f">> 使用 INT4 量化推理 - 配置來源: {config_source}")
                print(f"   權重=INT4, 運算=BF16, Vocoder={vocoder_precision}")

            else:
                # 標準模式：直接使用指定精度
                self.gpt_dtype = resolve_dtype(gpt_precision)
                self.use_quantization = False
                self.load_in_8bit = False
                self.load_in_4bit = False
                print(f">> 使用混合精度推理 - 配置來源: {config_source}")
                print(f"   GPT={self.gpt_dtype}, Vocoder={vocoder_precision}")

            self.vocoder_dtype = resolve_dtype(vocoder_precision)
            self.dvae_dtype = self.gpt_dtype if not self.use_quantization and isinstance(self.gpt_dtype, torch.dtype) else torch.bfloat16
        else:
            # 向後兼容：使用 is_fp16（自動選擇 BF16 或 FP16）
            if self.is_fp16:
                # 優先使用 BF16（數值穩定性更好），不支援才用 FP16
                if torch.cuda.is_bf16_supported():
                    self.gpt_dtype = torch.bfloat16
                    self.vocoder_dtype = torch.float32 # BigVGAN 在 BF16 下可能不穩定，預設回退到 FP32
                    self.dvae_dtype = torch.bfloat16
                    print(">> 使用 BF16 推理 (GPT) / FP32 (Vocoder) - 配置來源: is_fp16 參數（向後兼容模式）")
                else:
                    self.gpt_dtype = torch.float16
                    self.vocoder_dtype = torch.float32 # BigVGAN 在 FP16 下可能不穩定，預設回退到 FP32
                    self.dvae_dtype = torch.float16
                    print(">> 使用 FP16 推理 (GPT) / FP32 (Vocoder) - 配置來源: is_fp16 參數（向後兼容模式）")
                print("   建議: 使用 config_inference.yaml 或 config.yaml [inference] 進行精度配置")
            else:
                self.gpt_dtype = torch.float32
                self.vocoder_dtype = torch.float32
                self.dvae_dtype = torch.float32
                print(">> 使用 FP32 推理 - 配置來源: 預設值（向後兼容模式）")
                print("   建議: 使用 config_inference.yaml 或 config.yaml [inference] 進行精度配置")

            # 向後兼容模式不使用量化
            self.use_quantization = False
            self.load_in_8bit = False
            self.load_in_4bit = False

        # 向後兼容
        self.dtype = self.gpt_dtype if self.gpt_dtype != torch.float32 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        # Comment-off to load the VQ-VAE model for debugging tokenizer
        #   https://github.com/index-tts/index-tts/issues/34
        #
        # from indextts.vqvae.xtts_dvae import DiscreteVAE
        # self.dvae = DiscreteVAE(**self.cfg.vqvae)
        # self.dvae_path = os.path.join(self.model_dir, self.cfg.dvae_checkpoint)
        # load_checkpoint(self.dvae, self.dvae_path)
        # self.dvae = self.dvae.to(self.device)
        # if self.is_fp16:
        #     self.dvae.eval().half()
        # else:
        #     self.dvae.eval()
        # print(">> vqvae weights restored from:", self.dvae_path)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)

        # 使用量化載入
        if self.use_quantization:
            try:
                # 載入模型（先以 FP32 載入）
                self.gpt = UnifiedVoice(**self.cfg.gpt)
                load_checkpoint(self.gpt, self.gpt_path)
                self.gpt = self.gpt.to(self.device)
                
                # 定義要量化的模組（GPT 核心部分）
                # gpt 是 HuggingFace GPT2Model，包含主要的 Transformer 層
                target_modules = ['gpt', 'text_head', 'mel_head']
                
                # 執行動態量化
                print("=" * 60)
                print(">> 🔧 開始 GPT 模型量化...")
                print("=" * 60)
                
                if self.load_in_8bit:
                    replaced = _quantize_linear_layers_to_int8(self.gpt, target_modules, verbose=True)
                    quant_type = "INT8"
                elif self.load_in_4bit:
                    replaced = _quantize_linear_layers_to_int4(self.gpt, target_modules, verbose=True, compute_dtype=self.gpt_compute_dtype)
                    quant_type = "INT4 (NF4)"
                else:
                    replaced = 0
                    quant_type = "UNKNOWN"
                
                if replaced > 0:
                    print("=" * 60)
                    print(f">> ✅ 量化完成！")
                    print(f">>    - 量化類型: {quant_type}")
                    print(f">>    - 量化層數: {replaced}")
                    print(f">>    - 模型路徑: {self.gpt_path}")
                    print("=" * 60)
                    self.gpt.eval()
                else:
                    print(">> ⚠️  未找到可量化的層，回退到 BF16")
                    self.use_quantization = False
                    self.gpt.eval().to(torch.bfloat16)
                    print(f">> GPT weights restored from: {self.gpt_path} (dtype: BF16)")

            except ImportError:
                print(">> ⚠️  bitsandbytes 未安裝，回退到 BF16")
                print(">> 安裝: pip install bitsandbytes")
                self.use_quantization = False
                self.gpt = UnifiedVoice(**self.cfg.gpt)
                load_checkpoint(self.gpt, self.gpt_path)
                self.gpt = self.gpt.to(self.device)
                self.gpt.eval().to(torch.bfloat16)
                print(f">> GPT weights restored from: {self.gpt_path} (dtype: BF16)")
            except Exception as e:
                print(f">> ⚠️  量化失敗: {e}")
                print(">> 回退到 BF16 精度")
                self.use_quantization = False
                # 重新載入模型
                self.gpt = UnifiedVoice(**self.cfg.gpt)
                load_checkpoint(self.gpt, self.gpt_path)
                self.gpt = self.gpt.to(self.device)
                self.gpt.eval().to(torch.bfloat16)
                print(f">> GPT weights restored from: {self.gpt_path} (dtype: BF16)")
        else:
            # 標準精度載入
            self.gpt = UnifiedVoice(**self.cfg.gpt)
            load_checkpoint(self.gpt, self.gpt_path)
            self.gpt = self.gpt.to(self.device)

            # 使用細粒度精度
            if self.gpt_dtype == torch.float16:
                self.gpt.eval().half()
            elif self.gpt_dtype == torch.bfloat16:
                self.gpt.eval().to(torch.bfloat16)
            else:
                self.gpt.eval()
            print(f">> GPT weights restored from: {self.gpt_path} (dtype: {self.gpt_dtype})")
        if self.is_fp16:
            try:
                import deepspeed

                use_deepspeed = True
            except (ImportError, OSError, CalledProcessError) as e:
                use_deepspeed = False
                print(f">> DeepSpeed載入失敗，回退到標準推理: {e}")
                print("See more details https://www.deepspeed.ai/tutorials/advanced-install/")

            self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=True)
        else:
            self.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=False)

        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from indextts.BigVGAN.alias_free_activation.cuda import (
                    load as anti_alias_activation_loader,
                )
                anti_alias_activation_cuda = anti_alias_activation_loader.load()
                print(">> Preload custom CUDA kernel for BigVGAN", anti_alias_activation_cuda)
            except Exception as e:
                print(">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.", e, file=sys.stderr)
                print(" Reinstall with `pip install -e . --no-deps --no-build-isolation` to prebuild `anti_alias_activation_cuda` kernel.", file=sys.stderr)
                print(
                    "See more details: https://github.com/index-tts/index-tts/issues/164#issuecomment-2903453206", file=sys.stderr
                )
                self.use_cuda_kernel = False
        self.bigvgan = Generator(self.cfg.bigvgan, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan_path = os.path.join(self.model_dir, self.cfg.bigvgan_checkpoint)
        vocoder_dict = torch.load(self.bigvgan_path, map_location="cpu")
        self.bigvgan.load_state_dict(vocoder_dict["generator"])
        self.bigvgan = self.bigvgan.to(self.device)

        # 使用細粒度精度（保持 BatchNorm 為 FP32）
        if self.vocoder_dtype == torch.float16:
            self.bigvgan.half()
            # BatchNorm 層回退到 FP32
            for module in self.bigvgan.modules():
                if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    module.float()
        elif self.vocoder_dtype == torch.bfloat16:
            self.bigvgan.to(torch.bfloat16)
            # BatchNorm 層回退到 FP32
            for module in self.bigvgan.modules():
                if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    module.float()

        # remove weight norm on eval mode
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        print(f">> bigvgan weights restored from: {self.bigvgan_path} (dtype: {self.vocoder_dtype})")
        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        print(">> TextNormalizer loaded")
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        print(">> bpe model loaded from:", self.bpe_path)
        # 快取參考音訊mel：
        self.cache_audio_prompt = None
        self.cache_cond_mel = None
        # 進度引用顯示（可選）
        self.gr_progress = None
        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None
        
        # 初始化多說話人支援
        self.speaker_list = []
        if speaker_info_path and os.path.exists(speaker_info_path):
            try:
                with open(speaker_info_path, 'r', encoding='utf-8') as f:
                    speaker_info = json.load(f)
                # speaker_info.json 是一個數組，每個元素包含 speaker 欄位
                self.speaker_list = [item['speaker'] for item in speaker_info if 'speaker' in item]
                print(f">> Multi-speaker support enabled with {len(self.speaker_list)} speakers: {self.speaker_list}")
            except Exception as e:
                print(f">> Failed to load speaker_info from {speaker_info_path}: {e}")
                self.speaker_list = []
        else:
            print(">> Single-speaker mode (no speaker_info_path provided)")

        # 驗證模型精度
        self._verify_model_precision()

    def _verify_model_precision(self):
        """
        驗證模型實際載入的精度是否符合預期。
        這有助於及早發現精度配置錯誤。
        """
        print("=" * 60)
        print(">> 🔍 驗證模型精度...")

        # 驗證 GPT 模型精度
        try:
            # 獲取 GPT 模型的第一個參數的精度
            gpt_actual_dtype = next(self.gpt.parameters()).dtype

            if self.use_quantization:
                # 量化模式：檢查運算精度（權重可能是 INT8/INT4）
                expected_dtype = self.gpt_compute_dtype
                print(f">> GPT 模型 (量化模式):")
                print(f"   - 預期運算精度: {expected_dtype}")
                print(f"   - 實際參數精度: {gpt_actual_dtype}")
                # 注意：量化後某些層可能是量化類型，這裡只是檢查非量化參數
                if hasattr(gpt_actual_dtype, '__name__'):
                    dtype_name = gpt_actual_dtype.__name__ if hasattr(gpt_actual_dtype, '__name__') else str(gpt_actual_dtype)
                    if 'int' in dtype_name.lower() or 'Int' in str(type(gpt_actual_dtype)):
                        print(f"   ✅ 量化參數偵測到: {gpt_actual_dtype}")
                    else:
                        print(f"   ✅ 非量化參數精度: {gpt_actual_dtype}")
            else:
                # 標準精度模式
                expected_dtype = self.gpt_dtype
                print(f">> GPT 模型:")
                print(f"   - 預期精度: {expected_dtype}")
                print(f"   - 實際精度: {gpt_actual_dtype}")

                if gpt_actual_dtype != expected_dtype:
                    print(f"   ⚠️  警告：精度不符！請檢查模型載入流程")
                else:
                    print(f"   ✅ 精度驗證通過")
        except Exception as e:
            print(f"   ⚠️  GPT 精度驗證失敗: {e}")

        # 驗證 BigVGAN 模型精度
        try:
            vocoder_actual_dtype = next(self.bigvgan.parameters()).dtype
            expected_vocoder_dtype = self.vocoder_dtype

            print(f">> BigVGAN 聲碼器:")
            print(f"   - 預期精度: {expected_vocoder_dtype}")
            print(f"   - 實際精度: {vocoder_actual_dtype}")

            if vocoder_actual_dtype != expected_vocoder_dtype:
                print(f"   ⚠️  警告：精度不符！請檢查模型載入流程")
            else:
                print(f"   ✅ 精度驗證通過")
        except Exception as e:
            print(f"   ⚠️  Vocoder 精度驗證失敗: {e}")

        print("=" * 60)

    def remove_long_silence(self, codes: torch.Tensor, silent_token=52, max_consecutive=30):
        """
        Shrink special tokens (silent_token and stop_mel_token) in codes
        codes: [B, T]
        """
        code_lens = []
        codes_list = []
        device = codes.device
        dtype = codes.dtype
        isfix = False
        for i in range(0, codes.shape[0]):
            code = codes[i]
            if not torch.any(code == self.stop_mel_token).item():
                len_ = code.size(0)
            else:
                stop_mel_idx = (code == self.stop_mel_token).nonzero(as_tuple=False)
                len_ = stop_mel_idx[0].item() if len(stop_mel_idx) > 0 else code.size(0)

            count = torch.sum(code == silent_token).item()
            if count > max_consecutive:
                # code = code.cpu().tolist()
                ncode_idx = []
                n = 0
                for k in range(len_):
                    assert code[k] != self.stop_mel_token, f"stop_mel_token {self.stop_mel_token} should be shrinked here"
                    if code[k] != silent_token:
                        ncode_idx.append(k)
                        n = 0
                    elif code[k] == silent_token and n < 10:
                        ncode_idx.append(k)
                        n += 1
                    # if (k == 0 and code[k] == 52) or (code[k] == 52 and code[k-1] == 52):
                    #    n += 1
                # new code
                len_ = len(ncode_idx)
                codes_list.append(code[ncode_idx])
                isfix = True
            else:
                # shrink to len_
                codes_list.append(code[:len_])
            code_lens.append(len_)
        if isfix:
            if len(codes_list) > 1:
                codes = pad_sequence(codes_list, batch_first=True, padding_value=self.stop_mel_token)
            else:
                codes = codes_list[0].unsqueeze(0)
        else:
            # unchanged
            pass
        # clip codes to max length
        max_len = max(code_lens)
        if max_len < codes.shape[1]:
            codes = codes[:, :max_len]
        code_lens = torch.tensor(code_lens, dtype=torch.long, device=device)
        return codes, code_lens

    def bucket_sentences(self, sentences, bucket_max_size=4) -> List[List[Dict]]:
        """
        Sentence data bucketing.
        if ``bucket_max_size=1``, return all sentences in one bucket.
        """
        outputs: List[Dict] = []
        for idx, sent in enumerate(sentences):
            outputs.append({"idx": idx, "sent": sent, "len": len(sent)})
       
        if len(outputs) > bucket_max_size:
            # split sentences into buckets by sentence length
            buckets: List[List[Dict]] = []
            factor = 1.5
            last_bucket = None
            last_bucket_sent_len_median = 0

            for sent in sorted(outputs, key=lambda x: x["len"]):
                current_sent_len = sent["len"]
                if current_sent_len == 0:
                    print(">> skip empty sentence")
                    continue
                if last_bucket is None \
                        or current_sent_len >= int(last_bucket_sent_len_median * factor) \
                        or len(last_bucket) >= bucket_max_size:
                    # new bucket
                    buckets.append([sent])
                    last_bucket = buckets[-1]
                    last_bucket_sent_len_median = current_sent_len
                else:
                    # current bucket can hold more sentences
                    last_bucket.append(sent) # sorted
                    mid = len(last_bucket) // 2
                    last_bucket_sent_len_median = last_bucket[mid]["len"]
            last_bucket=None
            # merge all buckets with size 1
            out_buckets: List[List[Dict]] = []
            only_ones: List[Dict] = []
            for b in buckets:
                if len(b) == 1:
                    only_ones.append(b[0])
                else:
                    out_buckets.append(b)
            if len(only_ones) > 0:
                # merge into previous buckets if possible
                # print("only_ones:", [(o["idx"], o["len"]) for o in only_ones])
                for i in range(len(out_buckets)):
                    b = out_buckets[i]
                    if len(b) < bucket_max_size:
                        b.append(only_ones.pop(0))
                        if len(only_ones) == 0:
                            break
                # combined all remaining sized 1 buckets
                if len(only_ones) > 0:
                    out_buckets.extend([only_ones[i:i+bucket_max_size] for i in range(0, len(only_ones), bucket_max_size)])
            return out_buckets
        return [outputs]

    def pad_tokens_cat(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        if self.model_version and self.model_version >= 1.5:
            # 1.5版本以上，使用 stop_text_token 右側填充
            # [1, N] -> [N,]
            tokens = [t.squeeze(0) for t in tokens]
            # 手動實現 right padding（PyTorch pad_sequence 不支援 padding_side）
            max_len = max(t.size(0) for t in tokens)
            outputs = []
            for t in tokens:
                pad_len = max_len - t.size(0)
                if pad_len > 0:
                    # 在右側填充 stop_text_token
                    padded = torch.cat([t, torch.full((pad_len,), self.cfg.gpt.stop_text_token, dtype=t.dtype, device=t.device)])
                else:
                    padded = t
                outputs.append(padded)
            return torch.stack(outputs)  # [batch_size, max_len]
        max_len = max(t.size(1) for t in tokens)
        outputs = []
        for tensor in tokens:
            pad_len = max_len - tensor.size(1)
            if pad_len > 0:
                n = min(8, pad_len)
                tensor = torch.nn.functional.pad(tensor, (0, n), value=self.cfg.gpt.stop_text_token)
                tensor = torch.nn.functional.pad(tensor, (0, pad_len - n), value=self.cfg.gpt.start_text_token)
            tensor = tensor[:, :max_len]
            outputs.append(tensor)
        tokens = torch.cat(outputs, dim=0)
        return tokens

    def torch_empty_cache(self):
        try:
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()
            elif "mps" in str(self.device):
                torch.mps.empty_cache()
        except Exception as e:
            pass

    def _set_gr_progress(self, value, desc):
        if self.gr_progress is not None:
            self.gr_progress(value, desc=desc)

    # 快速推理：對於“多句長文字”，可實現至少 2~10 倍以上的速度提升~ （First modified by sunnyboxs 2025-04-16）
    def infer_fast(self, audio_prompt, text, output_path, verbose=False, max_text_tokens_per_sentence=100, sentences_bucket_max_size=4, **generation_kwargs):
        """
        Args:
            ``max_text_tokens_per_sentence``: 分句的最大token數，預設``100``，可以根據GPU硬體情況調整
                - 越小，batch 越多，推理速度越*快*，佔用記憶體更多，可能影響質量
                - 越大，batch 越少，推理速度越*慢*，佔用記憶體和質量更接近於非快速推理
            ``sentences_bucket_max_size``: 分句分桶的最大容量，預設``4``，可以根據GPU記憶體調整
                - 越大，bucket數量越少，batch越多，推理速度越*快*，佔用記憶體更多，可能影響質量
                - 越小，bucket數量越多，batch越少，推理速度越*慢*，佔用記憶體和質量更接近於非快速推理
        """
        print(">> start fast inference...")
        
        self._set_gr_progress(0, "start fast inference...")
        if verbose:
            print(f"origin text:{text}")
        start_time = time.perf_counter()

        # 如果參考音訊改變了，才需要重新生成 cond_mel, 提升速度
        if self.cache_cond_mel is None or self.cache_audio_prompt != audio_prompt:
            audio, sr = sf.read(audio_prompt)
            audio = torch.from_numpy(audio.T if audio.ndim > 1 else audio.reshape(1, -1)).float()
            audio = torch.mean(audio, dim=0, keepdim=True)
            if audio.shape[0] > 1:
                audio = audio[0].unsqueeze(0)
            audio = torchaudio.transforms.Resample(sr, 24000)(audio)
            cond_mel = MelSpectrogramFeatures()(audio).to(self.device)
            cond_mel_frame = cond_mel.shape[-1]
            if verbose:
                print(f"cond_mel shape: {cond_mel.shape}", "dtype:", cond_mel.dtype)

            self.cache_audio_prompt = audio_prompt
            self.cache_cond_mel = cond_mel
        else:
            cond_mel = self.cache_cond_mel
            cond_mel_frame = cond_mel.shape[-1]
            pass

        auto_conditioning = cond_mel
        cond_mel_lengths = torch.tensor([cond_mel_frame], device=self.device)

        # text_tokens
        text_tokens_list = self.tokenizer.tokenize(text)

        sentences = self.tokenizer.split_sentences(text_tokens_list, max_tokens_per_sentence=max_text_tokens_per_sentence)
        if verbose:
            print(">> text token count:", len(text_tokens_list))
            print("   splited sentences count:", len(sentences))
            print("   max_text_tokens_per_sentence:", max_text_tokens_per_sentence)
            print(*sentences, sep="\n")
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 1.0)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 600)
        sampling_rate = 24000
        # lang = "EN"
        # lang = "ZH"
        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        bigvgan_time = 0

        # text processing
        all_text_tokens: List[List[torch.Tensor]] = []
        self._set_gr_progress(0.1, "text processing...")
        bucket_max_size = sentences_bucket_max_size if self.device != "cpu" else 1
        all_sentences = self.bucket_sentences(sentences, bucket_max_size=bucket_max_size)
        bucket_count = len(all_sentences)
        if verbose:
            print(">> sentences bucket_count:", bucket_count,
                  "bucket sizes:", [(len(s), [t["idx"] for t in s]) for s in all_sentences],
                  "bucket_max_size:", bucket_max_size)
        for sentences in all_sentences:
            temp_tokens: List[torch.Tensor] = []
            all_text_tokens.append(temp_tokens)
            for item in sentences:
                sent = item["sent"]
                text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
                text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
                if verbose:
                    print(text_tokens)
                    print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
                    # debug tokenizer
                    text_token_syms = self.tokenizer.convert_ids_to_tokens(text_tokens[0].tolist())
                    print("text_token_syms is same as sentence tokens", text_token_syms == sent) 
                temp_tokens.append(text_tokens)
        
            
        # Sequential processing of bucketing data
        all_batch_num = sum(len(s) for s in all_sentences)
        all_batch_codes = []
        processed_num = 0
        for item_tokens in all_text_tokens:
            batch_num = len(item_tokens)
            if batch_num > 1:
                batch_text_tokens = self.pad_tokens_cat(item_tokens)
            else:
                batch_text_tokens = item_tokens[0]
            processed_num += batch_num
            # gpt speech
            self._set_gr_progress(0.2 + 0.3 * processed_num/all_batch_num, f"gpt inference speech... {processed_num}/{all_batch_num}")
            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(batch_text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    temp_codes = self.gpt.inference_speech(auto_conditioning, batch_text_tokens,
                                        cond_mel_lengths=cond_mel_lengths,
                                        # text_lengths=text_len,
                                        do_sample=do_sample,
                                        top_p=top_p,
                                        top_k=top_k,
                                        temperature=temperature,
                                        num_return_sequences=autoregressive_batch_size,
                                        length_penalty=length_penalty,
                                        num_beams=num_beams,
                                        repetition_penalty=repetition_penalty,
                                        max_generate_length=max_mel_tokens,
                                        **generation_kwargs)
                    all_batch_codes.append(temp_codes)
            gpt_gen_time += time.perf_counter() - m_start_time

        # gpt latent
        self._set_gr_progress(0.5, "gpt inference latents...")
        all_idxs = []
        all_latents = []
        has_warned = False
        for batch_codes, batch_tokens, batch_sentences in zip(all_batch_codes, all_text_tokens, all_sentences):
            for i in range(batch_codes.shape[0]):
                codes = batch_codes[i]  # [x]
                if not has_warned and codes[-1] != self.stop_mel_token:
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Consider reducing `max_text_tokens_per_sentence`({max_text_tokens_per_sentence}) or increasing `max_mel_tokens`.",
                        category=RuntimeWarning
                    )
                    has_warned = True
                codes = codes.unsqueeze(0)  # [x] -> [1, x]
                if verbose:
                    print("codes:", codes.shape)
                    print(codes)
                codes, code_lens = self.remove_long_silence(codes, silent_token=52, max_consecutive=30)
                if verbose:
                    print("fix codes:", codes.shape)
                    print(codes)
                    print("code_lens:", code_lens)
                text_tokens = batch_tokens[i]
                all_idxs.append(batch_sentences[i]["idx"])
                m_start_time = time.perf_counter()
                with torch.no_grad():
                    with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                        latent = \
                            self.gpt(auto_conditioning, text_tokens,
                                        torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                        code_lens*self.gpt.mel_length_compression,
                                        cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                                        return_latent=True, clip_inputs=False)
                        gpt_forward_time += time.perf_counter() - m_start_time
                        all_latents.append(latent)
        del all_batch_codes, all_text_tokens, all_sentences
        # bigvgan chunk
        chunk_size = 2
        all_latents = [all_latents[all_idxs.index(i)] for i in range(len(all_latents))]
        if verbose:
            print(">> all_latents:", len(all_latents))
            print("  latents length:", [l.shape[1] for l in all_latents])
        chunk_latents = [all_latents[i : i + chunk_size] for i in range(0, len(all_latents), chunk_size)]
        chunk_length = len(chunk_latents)
        latent_length = len(all_latents)

        # bigvgan chunk decode
        self._set_gr_progress(0.7, "bigvgan decode...")
        tqdm_progress = tqdm(total=latent_length, desc="bigvgan")
        for items in chunk_latents:
            tqdm_progress.update(len(items))
            latent = torch.cat(items, dim=1)
            with torch.no_grad():
                # Determine autocast settings for vocoder
                vocoder_autocast_enabled = self.vocoder_dtype != torch.float32
                vocoder_autocast_dtype = self.vocoder_dtype if vocoder_autocast_enabled else None

                # Explicitly cast inputs if vocoder is FP32
                if not vocoder_autocast_enabled:
                    latent = latent.float()
                    cond_input = auto_conditioning.transpose(1, 2).float()
                else:
                    cond_input = auto_conditioning.transpose(1, 2)

                with torch.amp.autocast(latent.device.type, enabled=vocoder_autocast_enabled, dtype=vocoder_autocast_dtype):
                    m_start_time = time.perf_counter()
                    wav, _ = self.bigvgan(latent, cond_input)
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)
                    pass
            wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
            wavs.append(wav.cpu()) # to cpu before saving

        # clear cache
        tqdm_progress.close()  # 確保進度條被關閉
        del all_latents, chunk_latents
        end_time = time.perf_counter()
        self.torch_empty_cache()

        # wav audio output
        self._set_gr_progress(0.9, "save audio...")
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        print(f">> Reference audio length: {cond_mel_frame * 256 / sampling_rate:.2f} seconds")
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        print(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total fast inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> [fast] bigvgan chunk_length: {chunk_length}")
        print(f">> [fast] batch_num: {all_batch_num} bucket_max_size: {bucket_max_size}", f"bucket_count: {bucket_count}" if bucket_max_size > 1 else "")
        print(f">> [fast] RTF: {(end_time - start_time) / wav_length:.4f}")

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            # 直接儲存音訊到指定路徑中
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # 直接使用 soundfile 避免 torchaudio.save 的 torchcodec 依賴問題
            # 先轉成 int16，再轉 numpy（soundfile 會正確處理 int16）
            wav_int16 = wav.squeeze(0).to(torch.float32).numpy().astype('int16')
            sf.write(output_path, wav_int16, sampling_rate, subtype='PCM_16')
            print(">> wav file saved to:", output_path)
            return output_path
        else:
            # 返回以符合Gradio的格式要求
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            return (sampling_rate, wav_data)

    # 原始推理模式
    def infer(self, audio_prompt, text, output_path, verbose=False, max_text_tokens_per_sentence=120, speaker_id=None, **generation_kwargs):
        # 驗證speaker_id
        if speaker_id is not None:
            if not hasattr(self, 'speaker_list') or not self.speaker_list:
                raise ValueError("Multi-speaker support not enabled. Please initialize with speaker_info_path.")
            if speaker_id not in self.speaker_list:
                raise ValueError(f"Invalid speaker_id: {speaker_id}. Available speakers: {self.speaker_list}")
        
        if verbose:
            print(f"origin text:{text}")
            if speaker_id:
                print(f"using speaker: {speaker_id}")
        start_time = time.perf_counter()

        # 如果參考音訊改變了，才需要重新生成 cond_mel, 提升速度
        if self.cache_cond_mel is None or self.cache_audio_prompt != audio_prompt:
            audio, sr = sf.read(audio_prompt)
            audio = torch.from_numpy(audio.T if audio.ndim > 1 else audio.reshape(1, -1)).float()
            audio = torch.mean(audio, dim=0, keepdim=True)
            if audio.shape[0] > 1:
                audio = audio[0].unsqueeze(0)
            audio = torchaudio.transforms.Resample(sr, 24000)(audio)
            cond_mel = MelSpectrogramFeatures()(audio).to(self.device)
            cond_mel_frame = cond_mel.shape[-1]
            if verbose:
                print(f"cond_mel shape: {cond_mel.shape}", "dtype:", cond_mel.dtype)

            self.cache_audio_prompt = audio_prompt
            self.cache_cond_mel = cond_mel
        else:
            cond_mel = self.cache_cond_mel
            cond_mel_frame = cond_mel.shape[-1]
            pass

        self._set_gr_progress(0.1, "text processing...")
        auto_conditioning = cond_mel
        text_tokens_list = self.tokenizer.tokenize(text)
        sentences = self.tokenizer.split_sentences(text_tokens_list, max_text_tokens_per_sentence)
        if verbose:
            print("text token count:", len(text_tokens_list))
            print("sentences count:", len(sentences))
            print("max_text_tokens_per_sentence:", max_text_tokens_per_sentence)
            print(*sentences, sep="\n")
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 1.0)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 600)
        sampling_rate = 24000
        # lang = "EN"
        # lang = "ZH"
        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        bigvgan_time = 0
        progress = 0
        has_warned = False
        for sent in sentences:
            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
            # text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
            # text_tokens = F.pad(text_tokens, (1, 0), value=0)
            # text_tokens = F.pad(text_tokens, (0, 1), value=1)
            if verbose:
                print(text_tokens)
                print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
                # debug tokenizer
                text_token_syms = self.tokenizer.convert_ids_to_tokens(text_tokens[0].tolist())
                print("text_token_syms is same as sentence tokens", text_token_syms == sent)

            # text_len = torch.IntTensor([text_tokens.size(1)], device=text_tokens.device)
            # print(text_len)
            progress += 1
            self._set_gr_progress(0.2 + 0.4 * (progress-1) / len(sentences), f"gpt inference latent... {progress}/{len(sentences)}")
            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    codes = self.gpt.inference_speech(auto_conditioning, text_tokens,
                                                        cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]],
                                                                                      device=text_tokens.device),
                                                        speaker_ids=[speaker_id] if speaker_id else None,  # 新增這行
                                                        do_sample=do_sample,
                                                        top_p=top_p,
                                                        top_k=top_k,
                                                        temperature=temperature,
                                                        num_return_sequences=autoregressive_batch_size,
                                                        length_penalty=length_penalty,
                                                        num_beams=num_beams,
                                                        repetition_penalty=repetition_penalty,
                                                        # 移除 speaker_id=speaker_id 這一行
                                                        )
                gpt_gen_time += time.perf_counter() - m_start_time
                if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Input text tokens: {text_tokens.shape[1]}. "
                        f"Consider reducing `max_text_tokens_per_sentence`({max_text_tokens_per_sentence}) or increasing `max_mel_tokens`.",
                        category=RuntimeWarning
                    )
                    has_warned = True

                code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)
                if verbose:
                    print(codes, type(codes))
                    print(f"codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")

                # remove ultra-long silence if exits
                # temporarily fix the long silence bug.
                codes, code_lens = self.remove_long_silence(codes, silent_token=52, max_consecutive=30)
                if verbose:
                    print(codes, type(codes))
                    print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")
                self._set_gr_progress(0.2 + 0.4 * progress / len(sentences), f"gpt inference speech... {progress}/{len(sentences)}")
                m_start_time = time.perf_counter()
                # latent, text_lens_out, code_lens_out = \
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    latent = self.gpt(
                        speech_conditioning_latent=auto_conditioning,  # 修正：mel_spec -> speech_conditioning_latent
                        text_inputs=text_tokens,                        # 修正：text_ids -> text_inputs
                        text_lengths=torch.tensor([text_tokens.shape[-1]], device=text_tokens.device),
                        mel_codes=codes,
                        wav_lengths=code_lens*self.gpt.mel_length_compression,  # 修正：codes_lengths -> wav_lengths
                        cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),  # 修正：mel_lengths -> cond_mel_lengths
                        speaker_ids=[speaker_id] if speaker_id else None,
                        return_latent=True
                    )
                    gpt_forward_time += time.perf_counter() - m_start_time

                # Determine autocast settings for vocoder
                vocoder_autocast_enabled = self.vocoder_dtype != torch.float32
                vocoder_autocast_dtype = self.vocoder_dtype if vocoder_autocast_enabled else None

                # Explicitly cast inputs if vocoder is FP32
                if not vocoder_autocast_enabled:
                    latent = latent.float()
                    cond_input = auto_conditioning.transpose(1, 2).float()
                else:
                    cond_input = auto_conditioning.transpose(1, 2)

                with torch.amp.autocast(text_tokens.device.type, enabled=vocoder_autocast_enabled, dtype=vocoder_autocast_dtype):
                    m_start_time = time.perf_counter()
                    wav, _ = self.bigvgan(latent, cond_input)
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)

                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                if verbose:
                    print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())
                # wavs.append(wav[:, :-512])
                wavs.append(wav.cpu())  # to cpu before saving
        end_time = time.perf_counter()
        self._set_gr_progress(0.9, "save audio...")
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        print(f">> Reference audio length: {cond_mel_frame * 256 / sampling_rate:.2f} seconds")
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        print(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> RTF: {(end_time - start_time) / wav_length:.4f}")

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            # 直接儲存音訊到指定路徑中
            if os.path.isfile(output_path):
                os.remove(output_path)
                print(">> remove old wav file:", output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # 直接使用 soundfile 避免 torchaudio.save 的 torchcodec 依賴問題
            # 先轉成 int16，再轉 numpy（soundfile 會正確處理 int16）
            wav_int16 = wav.squeeze(0).to(torch.float32).numpy().astype('int16')
            sf.write(output_path, wav_int16, sampling_rate, subtype='PCM_16')
            print(">> wav file saved to:", output_path)
            return output_path
        else:
            # 返回以符合Gradio的格式要求
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            return (sampling_rate, wav_data)


if __name__ == "__main__":
    set_seed(1234)
    
    # 指定說話人資訊檔案
    speaker_info_path = "finetune_data/processed_data/speaker_info.json"

    ifile = sys.argv[1]
    target_txt_list = []
    with open(ifile, 'r') as f:
        for line in f:
            line = line.strip()
            uid, prompt_txt, prompt_wav, target_txt = line.split('|')
            target_txt_list.append((uid, target_txt))
    
    # 初始化TTS，載入多說話人支援
    tts = IndexTTS(
        cfg_path="checkpoints/config.yaml", 
        model_dir="checkpoints", 
        is_fp16=True, 
        use_cuda_kernel=False,
        speaker_info_path=speaker_info_path  # 新增引數
    )

    prompts = [
        ("kaishu_30min", "/path/to/prompt.wav"),
        ]
    
    for speaker_id, prompt_wav in prompts:

        output_dir = f"result/{speaker_id}_{os.path.basename(ifile).rstrip('.lst')}_{time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)

        # 使用不同說話人進行推理
        for i, (uid, target_txt) in enumerate(target_txt_list):

            output_wav_path = f"{output_dir}/{uid}.wav"
            tts.infer(
                audio_prompt=prompt_wav, 
                text=target_txt, 
                output_path=output_wav_path, 
                verbose=True,
                speaker_id=speaker_id  # 新增引數
            )
