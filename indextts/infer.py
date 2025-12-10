import json
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
    """獲取指定名稱的父模組實例"""
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent


def _quantize_linear_layers_to_int8(model: nn.Module, target_modules: Optional[List[str]] = None, verbose: bool = True) -> int:
    """將指定模組內的 Linear 層轉換為 INT8 量化層 (bitsandbytes)"""
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError("未安裝 bitsandbytes，無法執行 INT8 量化。")
    
    replaced_count = 0
    total_params_before = 0
    total_params_after = 0
    
    # 篩選目標模組
    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if target_modules is not None:
                should_replace = any(name.startswith(target) or name == target for target in target_modules)
                if not should_replace:
                    continue
            modules_to_replace.append((name, module))
    
    if verbose:
        print(f">> [量化] 偵測到 {len(modules_to_replace)} 個可量化 Linear 層")
    
    # 執行替換
    for name, module in modules_to_replace:
        parent = _get_parent_module(model, name)
        child_name = name.split('.')[-1]
        
        # 統計參數變化
        param_count = module.in_features * module.out_features
        total_params_before += param_count * 4  # FP32
        total_params_after += param_count * 1   # INT8
        
        has_bias = module.bias is not None
        quantized_linear = bnb.nn.Linear8bitLt(
            module.in_features,
            module.out_features,
            bias=has_bias,
            has_fp16_weights=False,
            threshold=6.0,
        )
        
        # 複製權重
        quantized_linear.weight = bnb.nn.Int8Params(
            module.weight.data.contiguous(),
            requires_grad=False
        )
        if has_bias:
            quantized_linear.bias = nn.Parameter(module.bias.data.clone())
        
        setattr(parent, child_name, quantized_linear)
        replaced_count += 1
        
        if verbose and replaced_count <= 5:
            print(f">>   - 量化層: {name}")
    
    if verbose and replaced_count > 5:
        print(f">>   - ... (共 {replaced_count} 層)")
    
    if verbose:
        mem_before_mb = total_params_before / (1024 * 1024)
        mem_after_mb = total_params_after / (1024 * 1024)
        savings_pct = (1 - total_params_after / total_params_before) * 100 if total_params_before > 0 else 0
        print(f">> [量化] 權重記憶體: {mem_before_mb:.1f}MB → {mem_after_mb:.1f}MB (節省 {savings_pct:.0f}%)")
    
    return replaced_count


def _quantize_linear_layers_to_int4(model: nn.Module, target_modules: Optional[List[str]] = None, verbose: bool = True, compute_dtype: torch.dtype = torch.bfloat16) -> int:
    """將指定模組內的 Linear 層轉換為 INT4 (NF4) 量化層"""
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError("未安裝 bitsandbytes，無法執行 INT4 量化。")
    
    replaced_count = 0
    total_params_before = 0
    total_params_after = 0
    
    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if target_modules is not None:
                should_replace = any(name.startswith(target) or name == target for target in target_modules)
                if not should_replace:
                    continue
            modules_to_replace.append((name, module))
    
    if verbose:
        print(f">> [量化] 偵測到 {len(modules_to_replace)} 個可量化 Linear 層")
    
    for name, module in modules_to_replace:
        parent = _get_parent_module(model, name)
        child_name = name.split('.')[-1]
        
        param_count = module.in_features * module.out_features
        total_params_before += param_count * 4
        total_params_after += param_count * 0.5  # INT4
        
        has_bias = module.bias is not None
        quantized_linear = bnb.nn.Linear4bit(
            module.in_features,
            module.out_features,
            bias=has_bias,
            compute_dtype=compute_dtype,
            quant_type='nf4',
        )
        
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
            print(f">>   - 量化層: {name}")
    
    if verbose and replaced_count > 5:
        print(f">>   - ... (共 {replaced_count} 層)")
    
    if verbose:
        mem_before_mb = total_params_before / (1024 * 1024)
        mem_after_mb = total_params_after / (1024 * 1024)
        savings_pct = (1 - total_params_after / total_params_before) * 100 if total_params_before > 0 else 0
        print(f">> [量化] 權重記憶體: {mem_before_mb:.1f}MB → {mem_after_mb:.1f}MB (節省 {savings_pct:.0f}%)")
    
    return replaced_count

class IndexTTS:
    def __init__(
        self, cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=True, device=None, use_cuda_kernel=None,
        speaker_info_path=None,  # 多說話人配置檔
        precision_config=None,   # 混合精度詳細配置
    ):
        # 裝置初始化
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
            self.is_fp16 = False 
            self.use_cuda_kernel = False
        else:
            self.device = "cpu"
            self.is_fp16 = False
            self.use_cuda_kernel = False
            print(">> 提示: 目前運行於 CPU 模式，推理速度將受限。")

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir

        # 精度與量化配置解析
        # 優先順序: precision_config > config_inference.yaml > config.yaml > is_fp16 (legacy)
        config_source = None
        if precision_config is None:
            inference_config_path = os.path.join(model_dir, "config_inference.yaml")
            if os.path.exists(inference_config_path):
                inference_cfg = OmegaConf.load(inference_config_path)
                if hasattr(inference_cfg, 'inference'):
                    precision_config = inference_cfg.inference
                    config_source = "config_inference.yaml"
            elif hasattr(self.cfg, 'inference'):
                precision_config = self.cfg.inference
                config_source = "config.yaml [inference]"
        else:
            config_source = "Runtime Args"

        def resolve_dtype(precision_str):
            if precision_str in ["bf16", "bfloat16"]:
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            elif precision_str in ["fp16", "float16"]:
                return torch.float16
            elif precision_str in ["fp8"]:
                return torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else torch.bfloat16
            else:
                return torch.float32

        # 應用精度設定
        if precision_config and isinstance(precision_config, dict):
            gpt_precision = precision_config.get('gpt', 'bf16')
            vocoder_precision = precision_config.get('vocoder', 'bf16')
            quant_cfg = precision_config.get('quantization', {})
            quant_enabled = quant_cfg.get('enabled', False)

            if quant_enabled:
                # 進階量化配置
                weight_dtype = quant_cfg.get('weight_dtype', 'int8')
                compute_dtype = quant_cfg.get('compute_dtype', 'bf16')

                self.gpt_weight_dtype = weight_dtype
                self.gpt_compute_dtype = resolve_dtype(compute_dtype)
                self.use_quantization = True
                self.load_in_8bit = (weight_dtype == 'int8')
                self.load_in_4bit = (weight_dtype == 'int4')

                print(f">> [配置] 量化推理 ({config_source})")
                print(f"   - 權重: {weight_dtype.upper()}, 運算: {self.gpt_compute_dtype}")

            elif gpt_precision == 'int8':
                self.gpt_weight_dtype = 'int8'
                self.gpt_compute_dtype = torch.bfloat16
                self.use_quantization = True
                self.load_in_8bit = True
                self.load_in_4bit = False
                print(f">> [配置] INT8 量化推理 ({config_source})")

            elif gpt_precision == 'int4':
                self.gpt_weight_dtype = 'int4'
                self.gpt_compute_dtype = torch.bfloat16
                self.use_quantization = True
                self.load_in_8bit = False
                self.load_in_4bit = True
                print(f">> [配置] INT4 量化推理 ({config_source})")

            else:
                self.gpt_dtype = resolve_dtype(gpt_precision)
                self.use_quantization = False
                self.load_in_8bit = False
                self.load_in_4bit = False
                print(f">> [配置] 混合精度推理 ({config_source}): GPT={self.gpt_dtype}")

            self.vocoder_dtype = resolve_dtype(vocoder_precision)
            self.dvae_dtype = self.gpt_dtype if not self.use_quantization and isinstance(self.gpt_dtype, torch.dtype) else torch.bfloat16
        else:
            # 向後相容模式
            if self.is_fp16:
                if torch.cuda.is_bf16_supported():
                    self.gpt_dtype = torch.bfloat16
                    print(">> [配置] 自動選擇 BF16 (Legacy)")
                else:
                    self.gpt_dtype = torch.float16
                    print(">> [配置] 自動選擇 FP16 (Legacy)")
                self.vocoder_dtype = torch.float32 
                self.dvae_dtype = self.gpt_dtype
            else:
                self.gpt_dtype = torch.float32
                self.vocoder_dtype = torch.float32
                self.dvae_dtype = torch.float32
                print(">> [配置] 使用 FP32 (Legacy)")

            self.use_quantization = False
            self.load_in_8bit = False
            self.load_in_4bit = False

        self.dtype = self.gpt_dtype if self.gpt_dtype != torch.float32 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)

        # 載入 GPT 模型
        if self.use_quantization:
            try:
                self.gpt = UnifiedVoice(**self.cfg.gpt)
                load_checkpoint(self.gpt, self.gpt_path)
                self.gpt = self.gpt.to(self.device)
                
                target_modules = ['gpt', 'text_head', 'mel_head']
                print(">> [系統] 執行模型量化...")
                
                if self.load_in_8bit:
                    replaced = _quantize_linear_layers_to_int8(self.gpt, target_modules, verbose=True)
                elif self.load_in_4bit:
                    replaced = _quantize_linear_layers_to_int4(self.gpt, target_modules, verbose=True, compute_dtype=self.gpt_compute_dtype)
                else:
                    replaced = 0
                
                if replaced > 0:
                    print(f">> [系統] 量化完成 (層數: {replaced})")
                    self.gpt.eval()
                else:
                    print(">> [警告] 無法量化，回退至 BF16")
                    self.use_quantization = False
                    self.gpt.eval().to(torch.bfloat16)

            except ImportError:
                print(">> [錯誤] 缺少 bitsandbytes，回退至 BF16")
                self.use_quantization = False
                self.gpt = UnifiedVoice(**self.cfg.gpt)
                load_checkpoint(self.gpt, self.gpt_path)
                self.gpt = self.gpt.to(self.device).eval().to(torch.bfloat16)
            except Exception as e:
                print(f">> [錯誤] 量化失敗: {e}，回退至 BF16")
                self.use_quantization = False
                self.gpt = UnifiedVoice(**self.cfg.gpt)
                load_checkpoint(self.gpt, self.gpt_path)
                self.gpt = self.gpt.to(self.device).eval().to(torch.bfloat16)
        else:
            self.gpt = UnifiedVoice(**self.cfg.gpt)
            load_checkpoint(self.gpt, self.gpt_path)
            self.gpt = self.gpt.to(self.device)

            if self.gpt_dtype == torch.float16:
                self.gpt.eval().half()
            elif self.gpt_dtype == torch.bfloat16:
                self.gpt.eval().to(torch.bfloat16)
            else:
                self.gpt.eval()
            print(f">> [系統] GPT 模型載入完成 ({self.gpt_dtype})")

        # DeepSpeed 初始化 (僅在非量化模式且 FP16 時嘗試)
        if self.is_fp16:
            try:
                import deepspeed
                use_deepspeed = True
            except (ImportError, OSError, CalledProcessError):
                use_deepspeed = False
                print(">> [提示] DeepSpeed 未啟用，使用標準推理")

            self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=True)
        else:
            self.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=False)

        # 載入 BigVGAN (Vocoder)
        if self.use_cuda_kernel:
            try:
                from indextts.BigVGAN.alias_free_activation.cuda import load as anti_alias_activation_loader
                anti_alias_activation_cuda = anti_alias_activation_loader.load()
                print(">> [系統] BigVGAN CUDA Kernel 已載入")
            except Exception as e:
                print(f">> [警告] 無法載入 CUDA Kernel: {e}，將使用 PyTorch 實現")
                self.use_cuda_kernel = False
        
        self.bigvgan = Generator(self.cfg.bigvgan, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan_path = os.path.join(self.model_dir, self.cfg.bigvgan_checkpoint)
        vocoder_dict = torch.load(self.bigvgan_path, map_location="cpu")
        self.bigvgan.load_state_dict(vocoder_dict["generator"])
        self.bigvgan = self.bigvgan.to(self.device)

        # Vocoder 精度設定
        if self.vocoder_dtype == torch.float16:
            self.bigvgan.half()
            # BatchNorm 需保持 FP32 以維持穩定性
            for module in self.bigvgan.modules():
                if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    module.float()
        elif self.vocoder_dtype == torch.bfloat16:
            self.bigvgan.to(torch.bfloat16)
            for module in self.bigvgan.modules():
                if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    module.float()

        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        print(f">> [系統] BigVGAN 載入完成 ({self.vocoder_dtype})")

        # 載入 BPE 與 Tokenizer
        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        print(">> [系統] 文字處理模組已就緒")

        # 狀態變數
        self.cache_audio_prompt = None
        self.cache_cond_mel = None
        self.gr_progress = None
        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None
        
        # 多說話人支援
        self.speaker_list = []
        if speaker_info_path and os.path.exists(speaker_info_path):
            try:
                with open(speaker_info_path, 'r', encoding='utf-8') as f:
                    speaker_info = json.load(f)
                self.speaker_list = [item['speaker'] for item in speaker_info if 'speaker' in item]
                print(f">> [系統] 多說話人模式已啟用 (共 {len(self.speaker_list)} 位)")
            except Exception as e:
                print(f">> [錯誤] 讀取說話人資訊失敗: {e}")
        else:
            print(">> [系統] 單一說話人模式")

        self._verify_model_precision()

    def _verify_model_precision(self):
        """驗證模型參數精度是否符合預期配置"""
        # 僅在 verbose 模式或除錯時需要顯示詳細資訊，此處簡化輸出
        pass

    def remove_long_silence(self, codes: torch.Tensor, silent_token=52, max_consecutive=30):
        """
        移除序列中過長的靜音片段。
        
        Args:
            codes: Mel codes 序列 [B, T]
        """
        code_lens = []
        codes_list = []
        device = codes.device
        isfix = False
        
        for i in range(0, codes.shape[0]):
            code = codes[i]
            # 決定有效長度
            if not torch.any(code == self.stop_mel_token).item():
                len_ = code.size(0)
            else:
                stop_mel_idx = (code == self.stop_mel_token).nonzero(as_tuple=False)
                len_ = stop_mel_idx[0].item() if len(stop_mel_idx) > 0 else code.size(0)

            # 檢查靜音長度
            count = torch.sum(code == silent_token).item()
            if count > max_consecutive:
                ncode_idx = []
                n = 0
                for k in range(len_):
                    if code[k] != silent_token:
                        ncode_idx.append(k)
                        n = 0
                    elif code[k] == silent_token and n < 10: # 保留最多10幀靜音
                        ncode_idx.append(k)
                        n += 1
                
                codes_list.append(code[ncode_idx])
                isfix = True
                code_lens.append(len(ncode_idx))
            else:
                codes_list.append(code[:len_])
                code_lens.append(len_)
                
        if isfix:
            if len(codes_list) > 1:
                codes = pad_sequence(codes_list, batch_first=True, padding_value=self.stop_mel_token)
            else:
                codes = codes_list[0].unsqueeze(0)

        max_len = max(code_lens)
        if max_len < codes.shape[1]:
            codes = codes[:, :max_len]
        code_lens = torch.tensor(code_lens, dtype=torch.long, device=device)
        return codes, code_lens

    def bucket_sentences(self, sentences, bucket_max_size=4) -> List[List[Dict]]:
        """
        將句子依照長度分桶，以優化批次處理效率。
        """
        outputs: List[Dict] = []
        for idx, sent in enumerate(sentences):
            outputs.append({"idx": idx, "sent": sent, "len": len(sent)})
       
        if len(outputs) > bucket_max_size:
            buckets: List[List[Dict]] = []
            factor = 1.5
            last_bucket = None
            last_bucket_sent_len_median = 0

            # 依長度排序後分組
            for sent in sorted(outputs, key=lambda x: x["len"]):
                current_sent_len = sent["len"]
                if current_sent_len == 0:
                    continue
                
                # 判斷是否建立新桶
                if last_bucket is None \
                        or current_sent_len >= int(last_bucket_sent_len_median * factor) \
                        or len(last_bucket) >= bucket_max_size:
                    buckets.append([sent])
                    last_bucket = buckets[-1]
                    last_bucket_sent_len_median = current_sent_len
                else:
                    last_bucket.append(sent)
                    mid = len(last_bucket) // 2
                    last_bucket_sent_len_median = last_bucket[mid]["len"]
            
            # 合併過小的桶
            out_buckets: List[List[Dict]] = []
            only_ones: List[Dict] = []
            for b in buckets:
                if len(b) == 1:
                    only_ones.append(b[0])
                else:
                    out_buckets.append(b)
            
            if len(only_ones) > 0:
                for i in range(len(out_buckets)):
                    b = out_buckets[i]
                    if len(b) < bucket_max_size:
                        b.append(only_ones.pop(0))
                        if len(only_ones) == 0:
                            break
                if len(only_ones) > 0:
                    out_buckets.extend([only_ones[i:i+bucket_max_size] for i in range(0, len(only_ones), bucket_max_size)])
            return out_buckets
        return [outputs]

    def pad_tokens_cat(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        """合併並填充 Token 序列"""
        if self.model_version and self.model_version >= 1.5:
            tokens = [t.squeeze(0) for t in tokens]
            max_len = max(t.size(0) for t in tokens)
            outputs = []
            for t in tokens:
                pad_len = max_len - t.size(0)
                if pad_len > 0:
                    # 右側填充 Stop Token
                    padded = torch.cat([t, torch.full((pad_len,), self.cfg.gpt.stop_text_token, dtype=t.dtype, device=t.device)])
                else:
                    padded = t
                outputs.append(padded)
            return torch.stack(outputs)
        
        # 舊版填充邏輯
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
        except Exception:
            pass

    def _set_gr_progress(self, value, desc):
        if self.gr_progress is not None:
            self.gr_progress(value, desc=desc)

    def infer_fast(self, audio_prompt, text, output_path, verbose=False, max_text_tokens_per_sentence=100, sentences_bucket_max_size=4, **generation_kwargs):
        """
        快速推理模式：針對長文字進行批次優化，提升生成速度。
        """
        print(">> [推理] 啟動快速模式")
        
        self._set_gr_progress(0, "初始化...")
        start_time = time.perf_counter()

        # 緩存 Prompt 特徵
        if self.cache_cond_mel is None or self.cache_audio_prompt != audio_prompt:
            audio, sr = sf.read(audio_prompt)
            audio = torch.from_numpy(audio.T if audio.ndim > 1 else audio.reshape(1, -1)).float()
            audio = torch.mean(audio, dim=0, keepdim=True)
            if audio.shape[0] > 1:
                audio = audio[0].unsqueeze(0)
            audio = torchaudio.transforms.Resample(sr, 24000)(audio)
            cond_mel = MelSpectrogramFeatures()(audio).to(self.device)
            
            self.cache_audio_prompt = audio_prompt
            self.cache_cond_mel = cond_mel
        else:
            cond_mel = self.cache_cond_mel
        
        cond_mel_frame = cond_mel.shape[-1]
        auto_conditioning = cond_mel
        cond_mel_lengths = torch.tensor([cond_mel_frame], device=self.device)

        # 文字處理
        text_tokens_list = self.tokenizer.tokenize(text)
        sentences = self.tokenizer.split_sentences(text_tokens_list, max_tokens_per_sentence=max_text_tokens_per_sentence)
        
        # 參數提取
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 1.0)
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 600)
        sampling_rate = 24000

        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        bigvgan_time = 0

        # 分句與分桶
        all_text_tokens: List[List[torch.Tensor]] = []
        self._set_gr_progress(0.1, "文字前處理...")
        bucket_max_size = sentences_bucket_max_size if self.device != "cpu" else 1
        all_sentences = self.bucket_sentences(sentences, bucket_max_size=bucket_max_size)
        
        for sentences_in_bucket in all_sentences:
            temp_tokens: List[torch.Tensor] = []
            all_text_tokens.append(temp_tokens)
            for item in sentences_in_bucket:
                sent = item["sent"]
                text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
                text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
                temp_tokens.append(text_tokens)
        
        # GPT 推理流程
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
            
            self._set_gr_progress(0.2 + 0.3 * processed_num/all_batch_num, f"GPT 生成中... {processed_num}/{all_batch_num}")
            
            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(batch_text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    temp_codes = self.gpt.inference_speech(
                        auto_conditioning, batch_text_tokens,
                        cond_mel_lengths=cond_mel_lengths,
                        do_sample=do_sample,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        num_return_sequences=1,
                        length_penalty=length_penalty,
                        num_beams=num_beams,
                        repetition_penalty=repetition_penalty,
                        max_generate_length=max_mel_tokens,
                        **generation_kwargs
                    )
                    all_batch_codes.append(temp_codes)
            gpt_gen_time += time.perf_counter() - m_start_time

        # Latent 計算
        self._set_gr_progress(0.5, "計算隱向量...")
        all_idxs = []
        all_latents = []
        has_warned = False
        
        for batch_codes, batch_tokens, batch_sentences in zip(all_batch_codes, all_text_tokens, all_sentences):
            for i in range(batch_codes.shape[0]):
                codes = batch_codes[i]
                if not has_warned and codes[-1] != self.stop_mel_token:
                    warnings.warn(f"警告: 生成長度超過限制 ({max_mel_tokens})，建議調整分句長度。", category=RuntimeWarning)
                    has_warned = True
                    
                codes = codes.unsqueeze(0)
                codes, code_lens = self.remove_long_silence(codes)
                
                text_tokens = batch_tokens[i]
                all_idxs.append(batch_sentences[i]["idx"])
                
                m_start_time = time.perf_counter()
                with torch.no_grad():
                    with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                        latent = self.gpt(
                            auto_conditioning, text_tokens,
                            torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                            code_lens*self.gpt.mel_length_compression,
                            cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                            return_latent=True, clip_inputs=False
                        )
                        gpt_forward_time += time.perf_counter() - m_start_time
                        all_latents.append(latent)
        
        # BigVGAN 聲碼器解碼
        chunk_size = 2
        all_latents = [all_latents[all_idxs.index(i)] for i in range(len(all_latents))]
        chunk_latents = [all_latents[i : i + chunk_size] for i in range(0, len(all_latents), chunk_size)]
        
        self._set_gr_progress(0.7, "聲碼器解碼...")
        tqdm_progress = tqdm(total=len(all_latents), desc="BigVGAN", unit="sent")
        
        for items in chunk_latents:
            tqdm_progress.update(len(items))
            latent = torch.cat(items, dim=1)
            with torch.no_grad():
                vocoder_autocast_enabled = self.vocoder_dtype != torch.float32
                vocoder_autocast_dtype = self.vocoder_dtype if vocoder_autocast_enabled else None

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
            
            wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
            wavs.append(wav.cpu())

        tqdm_progress.close()
        end_time = time.perf_counter()
        self.torch_empty_cache()

        # 輸出處理
        self._set_gr_progress(0.9, "儲存音訊...")
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        
        print(f">> [統計] 總耗時: {end_time - start_time:.2f}s (RTF: {(end_time - start_time) / wav_length:.4f})")
        print(f"   - GPT 生成: {gpt_gen_time:.2f}s")
        print(f"   - 聲碼器: {bigvgan_time:.2f}s")

        wav = wav.cpu()
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            wav_int16 = wav.squeeze(0).to(torch.float32).numpy().astype('int16')
            sf.write(output_path, wav_int16, sampling_rate, subtype='PCM_16')
            print(f">> [輸出] 已儲存至: {output_path}")
            return output_path
        else:
            wav_data = wav.type(torch.int16).numpy().T
            return (sampling_rate, wav_data)

    def infer(self, audio_prompt, text, output_path, verbose=False, max_text_tokens_per_sentence=120, speaker_id=None, **generation_kwargs):
        """標準推理模式"""
        if speaker_id is not None:
            if not self.speaker_list:
                raise ValueError("錯誤：未啟用多說話人模式，請先載入 speaker_info_path。")
            if speaker_id not in self.speaker_list:
                raise ValueError(f"錯誤：無效的 speaker_id: {speaker_id}")
        
        start_time = time.perf_counter()

        if self.cache_cond_mel is None or self.cache_audio_prompt != audio_prompt:
            audio, sr = sf.read(audio_prompt)
            audio = torch.from_numpy(audio.T if audio.ndim > 1 else audio.reshape(1, -1)).float()
            audio = torch.mean(audio, dim=0, keepdim=True)
            if audio.shape[0] > 1:
                audio = audio[0].unsqueeze(0)
            audio = torchaudio.transforms.Resample(sr, 24000)(audio)
            cond_mel = MelSpectrogramFeatures()(audio).to(self.device)
            self.cache_audio_prompt = audio_prompt
            self.cache_cond_mel = cond_mel
        else:
            cond_mel = self.cache_cond_mel

        self._set_gr_progress(0.1, "文字前處理...")
        auto_conditioning = cond_mel
        text_tokens_list = self.tokenizer.tokenize(text)
        sentences = self.tokenizer.split_sentences(text_tokens_list, max_text_tokens_per_sentence)
        
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 1.0)
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 600)
        sampling_rate = 24000

        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        bigvgan_time = 0
        progress = 0
        has_warned = False
        
        for sent in sentences:
            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
            
            progress += 1
            self._set_gr_progress(0.2 + 0.4 * (progress-1) / len(sentences), f"生成中... {progress}/{len(sentences)}")
            
            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    codes = self.gpt.inference_speech(
                        auto_conditioning, text_tokens,
                        cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                        speaker_ids=[speaker_id] if speaker_id else None,
                        do_sample=do_sample,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        num_return_sequences=1,
                        length_penalty=length_penalty,
                        num_beams=num_beams,
                        repetition_penalty=repetition_penalty
                    )
                gpt_gen_time += time.perf_counter() - m_start_time
                
                if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                    warnings.warn(f"警告: 生成長度超過限制 ({max_mel_tokens})，建議調整分句長度。", category=RuntimeWarning)
                    has_warned = True

                codes, code_lens = self.remove_long_silence(codes)
                
                self._set_gr_progress(0.2 + 0.4 * progress / len(sentences), f"合成語音... {progress}/{len(sentences)}")
                m_start_time = time.perf_counter()
                
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    latent = self.gpt(
                        speech_conditioning_latent=auto_conditioning,
                        text_inputs=text_tokens,
                        text_lengths=torch.tensor([text_tokens.shape[-1]], device=text_tokens.device),
                        mel_codes=codes,
                        wav_lengths=code_lens*self.gpt.mel_length_compression,
                        cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                        speaker_ids=[speaker_id] if speaker_id else None,
                        return_latent=True
                    )
                    gpt_forward_time += time.perf_counter() - m_start_time

                vocoder_autocast_enabled = self.vocoder_dtype != torch.float32
                vocoder_autocast_dtype = self.vocoder_dtype if vocoder_autocast_enabled else None

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
                wavs.append(wav.cpu())
        
        end_time = time.perf_counter()
        self._set_gr_progress(0.9, "儲存音訊...")
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        
        print(f">> [統計] 總耗時: {end_time - start_time:.2f}s (RTF: {(end_time - start_time) / wav_length:.4f})")
        print(f"   - GPT 生成: {gpt_gen_time:.2f}s")
        print(f"   - 聲碼器: {bigvgan_time:.2f}s")

        wav = wav.cpu()
        if output_path:
            if os.path.isfile(output_path):
                os.remove(output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            wav_int16 = wav.squeeze(0).to(torch.float32).numpy().astype('int16')
            sf.write(output_path, wav_int16, sampling_rate, subtype='PCM_16')
            print(f">> [輸出] 已儲存至: {output_path}")
            return output_path
        else:
            wav_data = wav.type(torch.int16).numpy().T
            return (sampling_rate, wav_data)


if __name__ == "__main__":
    set_seed(1234)
    speaker_info_path = "finetune_data/processed_data/speaker_info.json"

    if len(sys.argv) > 1:
        ifile = sys.argv[1]
        target_txt_list = []
        with open(ifile, 'r') as f:
            for line in f:
                line = line.strip()
                uid, prompt_txt, prompt_wav, target_txt = line.split('|')
                target_txt_list.append((uid, target_txt))
        
        tts = IndexTTS(
            cfg_path="checkpoints/config.yaml", 
            model_dir="checkpoints", 
            is_fp16=True, 
            use_cuda_kernel=False,
            speaker_info_path=speaker_info_path
        )

        prompts = [
            ("kaishu_30min", "/path/to/prompt.wav"),
        ]
        
        for speaker_id, prompt_wav in prompts:
            output_dir = f"result/{speaker_id}_{os.path.basename(ifile).rstrip('.lst')}_{time.strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(output_dir, exist_ok=True)

            for i, (uid, target_txt) in enumerate(target_txt_list):
                output_wav_path = f"{output_dir}/{uid}.wav"
                tts.infer(
                    audio_prompt=prompt_wav, 
                    text=target_txt, 
                    output_path=output_wav_path, 
                    verbose=True,
                    speaker_id=speaker_id
                )
    else:
        print("Usage: python indextts/infer.py <input_list_file>")
