import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import (
    GenerationMixin,
    GPT2Config,
    GPT2PreTrainedModel,
    LogitsProcessorList,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.perceiver import PerceiverResampler
from indextts.utils.arch_util import AttentionBlock
from indextts.utils.typical_sampling import TypicalLogitsWarper


def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)


class ResBlock(nn.Module):
    """
    使用 GroupNorm 的基礎殘差卷積區塊。
    """

    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(chan // 8, chan),
            nn.ReLU(),
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(chan // 8, chan)
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)


class GPT2InferenceModel(GPT2PreTrainedModel, GenerationMixin):
    """
    用於推理的 GPT2 模型封裝，支援模型並行與 KV 快取。
    """
    def __init__(self, config, gpt, text_pos_emb, embeddings, norm, linear, kv_cache=False):
        super().__init__(config)
        # 注意: text_pos_emb 實際代表 Mel 位置編碼
        self.transformer = gpt
        self.text_pos_embedding = text_pos_emb
        self.embeddings = embeddings
        self.final_norm = norm
        self.lm_head = nn.Sequential(norm, linear)
        self.kv_cache = kv_cache

        # 模型並行設定
        self.model_parallel = False
        self.device_map = None
        self.cached_mel_emb = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(max(1, torch.cuda.device_count())))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def store_mel_emb(self, mel_emb):
        self.cached_mel_emb = mel_emb

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        if not self.kv_cache:
            past_key_values = None
        
        # 若存在 past_key_values，僅使用最後一個 Token
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # 動態建立 position_ids
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
            
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        assert self.cached_mel_emb is not None
        assert inputs_embeds is None  # 推理模型不支援直接輸入 Embeddings
        assert labels is None  # 推理模型不支援訓練模式
        
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        
        # 建立 Embedding
        mel_len = self.cached_mel_emb.shape[1]
        if input_ids.shape[1] != 1:
            text_inputs = input_ids[:, mel_len:]
            text_emb = self.embeddings(text_inputs)
            text_emb = text_emb + self.text_pos_embedding(text_emb)
            if self.cached_mel_emb.shape[0] != text_emb.shape[0]:
                mel_emb = self.cached_mel_emb.repeat_interleave(
                    text_emb.shape[0] // self.cached_mel_emb.shape[0], 0
                )
            else:
                mel_emb = self.cached_mel_emb
            emb = torch.cat([mel_emb, text_emb], dim=1)
        else:
            emb = self.embeddings(input_ids)
            emb = emb + self.text_pos_embedding.get_fixed_embedding(
                attention_mask.shape[1] - mel_len, attention_mask.device
            )
            
        transformer_outputs = self.transformer(
            inputs_embeds=emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # 設定並行運算裝置
        if self.model_parallel:
            if torch.backends.mps.is_available():
                self.to(self.transformer.first_device)
            else:
                torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + transformer_outputs[1:]

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        重排序 Cache 以支援 Beam Search。
        """
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past
        )


class ConditioningEncoder(nn.Module):
    def __init__(self,
                 spec_dim,
                 embedding_dim,
                 attn_blocks=6,
                 num_attn_heads=4,
                 do_checkpointing=False,
                 mean=False):
        super().__init__()
        attn = []
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim
        self.do_checkpointing = do_checkpointing
        self.mean = mean

    def forward(self, x):
        h = self.init(x)
        h = self.attn(h)
        if self.mean:
            return h.mean(dim=2)
        else:
            return h


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # GPT-2 標準初始化方式
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


def build_hf_gpt_transformer(layers, model_dim, heads, max_mel_seq_len, max_text_seq_len, checkpointing, activation_function):
    """
    建構 HuggingFace GPT-2 Transformer 模型。
    """
    from transformers import GPT2Config, GPT2Model
    gpt_config = GPT2Config(vocab_size=256,  # 未使用
                            n_positions=max_mel_seq_len + max_text_seq_len,
                            n_ctx=max_mel_seq_len + max_text_seq_len,
                            n_embd=model_dim,
                            n_layer=layers,
                            n_head=heads,
                            activation_function=activation_function or "gelu_new",
                            use_cache=not checkpointing)
    gpt = GPT2Model(gpt_config)
    if checkpointing:
        gpt.gradient_checkpointing_enable()
    
    # 覆蓋內建的位置與 Token Embeddings
    del gpt.wpe
    gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)
    del gpt.wte
    
    return gpt, LearnedPositionEmbeddings(max_mel_seq_len, model_dim), LearnedPositionEmbeddings(max_text_seq_len, model_dim), \
        None, None


class MelEncoder(nn.Module):
    def __init__(self, channels, mel_channels=80, resblocks_per_reduction=2):
        super().__init__()
        self.channels = channels
        self.encoder = nn.Sequential(nn.Conv1d(mel_channels, channels // 4, kernel_size=3, padding=1),
                                     nn.Sequential(*[ResBlock(channels // 4) for _ in range(resblocks_per_reduction)]),
                                     nn.Conv1d(channels // 4, channels // 2, kernel_size=3, stride=2, padding=1),
                                     nn.GroupNorm(channels // 16, channels // 2),
                                     nn.ReLU(),
                                     nn.Sequential(*[ResBlock(channels // 2) for _ in range(resblocks_per_reduction)]),
                                     nn.Conv1d(channels // 2, channels, kernel_size=3, stride=2, padding=1),
                                     nn.GroupNorm(channels // 8, channels),
                                     nn.ReLU(),
                                     nn.Sequential(*[ResBlock(channels) for _ in range(resblocks_per_reduction)]),
                                     )
        self.reduction = 4

    def forward(self, x):
        for e in self.encoder:
            x = e(x)
        return x.permute(0, 2, 1)


class UnifiedVoice(nn.Module):
    def __init__(self, layers=8, model_dim=512, heads=8, max_text_tokens=120, max_mel_tokens=250, max_conditioning_inputs=1,
                 mel_length_compression=1024, number_text_tokens=256,
                 start_text_token=0, stop_text_token=1, number_mel_codes=8194, start_mel_token=8192, stop_mel_token=8193,
                 train_solo_embeddings=False, use_mel_codes_as_input=True,
                 checkpointing=True, types=1, activation_function=None,
                 condition_num_latent=32, condition_type="perceiver", condition_module=None):
        """
        統一語音模型初始化。

        Args:
            layers: Transformer 層數
            model_dim: 模型維度
            heads: 注意力頭數
            max_text_tokens: 最大文字 Token 數
            max_mel_tokens: 最大 Mel Token 數
            condition_type: 條件編碼器類型 ('perceiver', 'conformer_perceiver' 等)
        """
        super().__init__()
        self.number_text_tokens = number_text_tokens
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.number_mel_codes = number_mel_codes
        self.start_mel_token = start_mel_token
        self.stop_mel_token = stop_mel_token
        self.layers = layers
        self.heads = heads
        self.max_mel_tokens = max_mel_tokens
        self.max_text_tokens = max_text_tokens
        self.model_dim = model_dim
        self.max_conditioning_inputs = max_conditioning_inputs
        self.mel_length_compression = mel_length_compression
        self.condition_type = condition_type
        self.cond_num = condition_num_latent
        self.cond_mask_pad = nn.ConstantPad1d((self.cond_num, 0), True)
        
        if condition_type == "perceiver":
            self.conditioning_encoder = ConditioningEncoder(100, model_dim, num_attn_heads=heads)
            self.perceiver_encoder = PerceiverResampler(model_dim, dim_context=model_dim, num_latents=self.cond_num)
        elif condition_type in ["conformer_perceiver", "conformer_encoder"]:
            self.conditioning_encoder = ConformerEncoder(input_size=100,
                                                         output_size=condition_module['output_size'],
                                                         linear_units=condition_module['linear_units'],
                                                         attention_heads=condition_module['attention_heads'],
                                                         num_blocks=condition_module['num_blocks'],
                                                         input_layer=condition_module['input_layer'])
            if condition_type == "conformer_perceiver":
                self.perceiver_encoder = PerceiverResampler(model_dim, dim_context=condition_module['output_size'],
                                                            ff_mult=condition_module['perceiver_mult'],
                                                            heads=condition_module['attention_heads'],
                                                            num_latents=self.cond_num)
        else:
            self.conditioning_encoder = ConditioningEncoder(100, model_dim, num_attn_heads=heads, mean=True)

        self.text_embedding = nn.Embedding(self.number_text_tokens * types + 1, model_dim)
        if use_mel_codes_as_input:
            self.mel_embedding = nn.Embedding(self.number_mel_codes, model_dim)
        else:
            self.mel_embedding = MelEncoder(model_dim, resblocks_per_reduction=1)
            
        self.gpt, self.mel_pos_embedding, self.text_pos_embedding, self.mel_layer_pos_embedding, self.text_layer_pos_embedding = \
            build_hf_gpt_transformer(layers, model_dim, heads, self.max_mel_tokens + 2 + self.max_conditioning_inputs,
                                     self.max_text_tokens + 2, checkpointing, activation_function)
        
        if train_solo_embeddings:
            self.mel_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * .02, requires_grad=True)
            self.text_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * .02, requires_grad=True)
        else:
            self.mel_solo_embedding = 0
            self.text_solo_embedding = 0

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.number_text_tokens * types + 1)
        self.mel_head = nn.Linear(model_dim, self.number_mel_codes)

        embeddings = [self.text_embedding]
        if use_mel_codes_as_input:
            embeddings.append(self.mel_embedding)
        for module in embeddings:
            module.weight.data.normal_(mean=0.0, std=.02)
            
        self.mean_condition = None

    def post_init_gpt2_config(self, use_deepspeed=False, kv_cache=False, half=False):
        seq_length = self.max_mel_tokens + self.max_text_tokens + 2
        gpt_config = GPT2Config(
            vocab_size=self.number_mel_codes,
            n_positions=seq_length,
            n_ctx=seq_length,
            n_embd=self.model_dim,
            n_layer=self.layers,
            n_head=self.heads,
            use_cache=True,
        )
        self.inference_model = GPT2InferenceModel(
            gpt_config,
            self.gpt,
            self.mel_pos_embedding,
            self.mel_embedding,
            self.final_norm,
            self.mel_head,
            kv_cache=kv_cache,
        )
        if use_deepspeed and half and torch.cuda.is_available():
            import deepspeed
            self.ds_engine = deepspeed.init_inference(model=self.inference_model,
                                                      mp_size=1,
                                                      replace_with_kernel_inject=False,
                                                      dtype=torch.float16)
            self.inference_model = self.ds_engine.module.eval()
        elif use_deepspeed and torch.cuda.is_available():
            import deepspeed
            self.ds_engine = deepspeed.init_inference(model=self.inference_model,
                                                      mp_size=1,
                                                      replace_with_kernel_inject=False,
                                                      dtype=torch.float32)
            self.inference_model = self.ds_engine.module.eval()
        else:
            self.inference_model = self.inference_model.eval()

        self.gpt.wte = self.mel_embedding

    def build_aligned_inputs_and_targets(self, input, start_token, stop_token):
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar

    def set_mel_padding(self, mel_input_tokens, mel_lengths):
        """
        將 Mel Tokens 中的 padding 替換為 STOP_MEL_TOKEN。
        """
        for b in range(len(mel_lengths)):
            actual_end = mel_lengths[b]
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.stop_mel_token
        return mel_input_tokens

    def set_text_padding(self, text_input_tokens, text_lengths):
        """
        將 Text Tokens 中的 padding 替換為 STOP_TEXT_TOKEN。
        """
        for b in range(len(text_lengths)):
            actual_end = text_lengths[b]
            if actual_end < text_input_tokens.shape[-1]:
                text_input_tokens[b, actual_end:] = self.stop_text_token
        return text_input_tokens

    def get_logits(self, speech_conditioning_inputs, first_inputs, first_head, second_inputs=None, second_head=None, get_attns=False, return_latent=False):
        if second_inputs is not None:
            emb = torch.cat([speech_conditioning_inputs, first_inputs, second_inputs], dim=1)
        else:
            emb = torch.cat([speech_conditioning_inputs, first_inputs], dim=1)

        gpt_out = self.gpt(inputs_embeds=emb, return_dict=True, output_attentions=get_attns)
        if get_attns:
            return gpt_out.attentions

        offset = speech_conditioning_inputs.shape[1]
        enc = gpt_out.last_hidden_state[:, offset:]
        enc = self.final_norm(enc)

        if return_latent:
            return enc[:, :first_inputs.shape[1]], enc[:, -second_inputs.shape[1]:]

        first_logits = enc[:, :first_inputs.shape[1]]
        first_logits = first_head(first_logits)
        first_logits = first_logits.permute(0, 2, 1)
        if second_inputs is not None:
            second_logits = enc[:, -second_inputs.shape[1]:]
            second_logits = second_head(second_logits)
            second_logits = second_logits.permute(0, 2, 1)
            return first_logits, second_logits
        else:
            return first_logits

    def get_conditioning(self, speech_conditioning_input, cond_mel_lengths=None, speaker_ids=None):
        # ⚠️ 重要：speaker_ids 僅在「無參考音檔」時才使用
        # 有參考音檔時，應始終走 encoder 以實現 zero-shot clone
        if speaker_ids is not None and speech_conditioning_input is None:
            batch_size = len(speaker_ids)
            device = next(self.parameters()).device

            conds_list = []
            for speaker_id in speaker_ids:
                param_name = f'mean_condition_{speaker_id}'
                if hasattr(self, param_name):
                    speaker_condition = getattr(self, param_name)
                    if speaker_condition.device != device:
                        speaker_condition = speaker_condition.to(device)
                    if speaker_condition.ndim == 2:
                        speaker_condition = speaker_condition.unsqueeze(0)
                    elif speaker_condition.ndim == 4:
                        speaker_condition = speaker_condition.squeeze(0)
                    conds_list.append(speaker_condition)
                else:
                    raise ValueError(f"找不到說話人 {speaker_id} 的條件向量")

            return torch.cat(conds_list, dim=0)

        if hasattr(self, 'mean_condition') and self.mean_condition is not None and speech_conditioning_input is None:
            batch_size = 1
            device = next(self.parameters()).device
            if self.mean_condition.device != device:
                logger.info(f"警告: mean_condition 裝置不符，移動至 {device}")
                self.mean_condition = self.mean_condition.to(device)
            return self.mean_condition.expand(batch_size, -1, -1)
        
        if self.condition_type == "perceiver":
            if speech_conditioning_input.ndim == 4:
                speech_conditioning_input = speech_conditioning_input.squeeze(1)
            speech_conditioning_input = self.conditioning_encoder(speech_conditioning_input)  # (b, d, s)
            conds = self.perceiver_encoder(speech_conditioning_input.transpose(1, 2))  # (b, 32, d)
        elif self.condition_type == "conformer_perceiver":
            speech_conditioning_input, mask = self.conditioning_encoder(speech_conditioning_input.transpose(1, 2),
                                                                        cond_mel_lengths)  # (b, s, d), (b, 1, s)
            if self.condition_type == "conformer_perceiver":
                conds_mask = self.cond_mask_pad(mask.squeeze(1))
                conds = self.perceiver_encoder(speech_conditioning_input, conds_mask)  # (b, 32, d)
        elif self.condition_type == "gst":
            if speech_conditioning_input.ndim == 4:
                speech_conditioning_input = speech_conditioning_input.squeeze(1)
            conds = self.gst_encoder(speech_conditioning_input.transpose(1, 2))  # (b, 1, d)
        else:
            speech_conditioning_input = (
                speech_conditioning_input.unsqueeze(1)
                if len(speech_conditioning_input.shape) == 3
                else speech_conditioning_input
            )
            conds = []
            for j in range(speech_conditioning_input.shape[1]):
                conds.append(self.conditioning_encoder(speech_conditioning_input[:, j]))
            conds = torch.stack(conds, dim=1)
            conds = conds.mean(dim=1)
            conds = conds.unsqueeze(1)
        return conds

    def forward(self, speech_conditioning_latent, text_inputs, text_lengths, mel_codes, wav_lengths,
                cond_mel_lengths=None, types=None, text_first=True, raw_mels=None, return_attentions=False,
                return_latent=False, clip_inputs=False, speaker_ids=None):
        """
        前向傳播方法，支援文字條件或語音條件模式。

        Args:
            speech_conditioning_latent: Mel 頻譜圖輸入 (b, 1024)
            text_inputs: 文字輸入序列 (b, t)
            mel_codes: Mel 代碼輸入 (b, m)
            clip_inputs: 是否裁剪輸入至最小尺寸
        """

        speech_conditioning_latent = self.get_conditioning(speech_conditioning_latent, cond_mel_lengths, speaker_ids)
        if types is not None:
            text_inputs = text_inputs * (1 + types).unsqueeze(-1)

        if clip_inputs:
            max_text_len = text_lengths.max()
            text_inputs = text_inputs[:, :max_text_len]
            max_mel_len = wav_lengths.max() // self.mel_length_compression
            mel_codes = mel_codes[:, :max_mel_len]
            if raw_mels is not None:
                raw_mels = raw_mels[:, :, :max_mel_len * 4]

        mel_codes_lengths = torch.ceil(wav_lengths / self.mel_length_compression).long() + 1
        mel_codes = self.set_mel_padding(mel_codes, mel_codes_lengths)
        text_inputs = self.set_text_padding(text_inputs, text_lengths)
        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
        mel_codes = F.pad(mel_codes, (0, 1), value=self.stop_mel_token)

        conds = speech_conditioning_latent
        text_inputs, text_targets = self.build_aligned_inputs_and_targets(text_inputs, self.start_text_token, self.stop_text_token)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        mel_codes, mel_targets = self.build_aligned_inputs_and_targets(mel_codes, self.start_mel_token, self.stop_mel_token)
        if raw_mels is not None:
            mel_inp = F.pad(raw_mels, (0, 8))
        else:
            mel_inp = mel_codes
        mel_emb = self.mel_embedding(mel_inp)
        mel_emb = mel_emb + self.mel_pos_embedding(mel_codes)

        if text_first:
            text_logits, mel_logits = self.get_logits(conds, text_emb, self.text_head, mel_emb, self.mel_head, get_attns=return_attentions, return_latent=return_latent)
            if return_latent:
                return mel_logits[:, :-2]
        else:
            mel_logits, text_logits = self.get_logits(conds, mel_emb, self.mel_head, text_emb, self.text_head, get_attns=return_attentions, return_latent=return_latent)
            if return_latent:
                return text_logits[:, :-2]

        if return_attentions:
            return mel_logits

        loss_text = F.cross_entropy(text_logits, text_targets.long())
        loss_mel = F.cross_entropy(mel_logits, mel_targets.long())
        return loss_text.mean(), loss_mel.mean(), mel_logits

    def prepare_gpt_inputs(
        self,
        conditional_latents: torch.Tensor,
        text_inputs: torch.Tensor,
    ):
        
        """
        準備 GPT2InferenceModel 的生成輸入。

        Args:
            conds_latent: 音訊條件嵌入 (b, 32, dim)
            text_inputs: 文字輸入 (b, L)
        
        Returns:
            input_ids, inputs_embeds, attention_mask
        """
        b, L = text_inputs.shape[:2]
        device = text_inputs.device
        single_cond = conditional_latents.ndim == 3 and conditional_latents.shape[0] == 1
        if not single_cond:
            assert conditional_latents.shape[0] == b, f"Batch size 不符: {conditional_latents.shape[0]} vs {b}"
        batched_mel_emb = []
        attention_masks = []
        target_len = conditional_latents.shape[1] + L + 2
        for i in range(b):
            valid_mask = (text_inputs[i] != self.stop_text_token) & (text_inputs[i] != self.start_text_token)
            text_input = text_inputs[i][valid_mask]
            text_input = F.pad(text_input, (1, 0), value=self.start_text_token)
            text_input = F.pad(text_input, (0, 1), value=self.stop_text_token)
            text_input_pos = torch.arange(0, text_input.size(-1), device=device)
            text_emb = self.text_embedding(text_input) + self.text_pos_embedding.emb(text_input_pos)
            
            conds_text_emb = [
                conditional_latents.squeeze(0) if single_cond else conditional_latents[i],
                text_emb,
            ]
            
            attention_mask = torch.ones(target_len+1, dtype=torch.long, device=device)
            padding: int = L + 2 - text_input.size(-1)
            
            if padding > 0:
                pad = torch.zeros((padding, conditional_latents.size(-1)), dtype=text_emb.dtype, device=device)
                conds_text_emb.insert(0, pad)
                attention_mask[:padding] = 0
            mel_emb = torch.cat(conds_text_emb)
            assert mel_emb.shape[0] == target_len, f"mel_emb.shape: {mel_emb.shape}, target_len: {target_len}"
            batched_mel_emb.append(mel_emb)
            attention_masks.append(attention_mask)
        
        batched_mel_emb = torch.stack(batched_mel_emb, dim=0)
        attention_mask = torch.stack(attention_masks, dim=0)
        
        fake_inputs = torch.ones(
            (
                batched_mel_emb.shape[0],
                batched_mel_emb.shape[1] + 1,
            ),
            dtype=torch.long,
            device=device,
        )
        fake_inputs[:, -1] = self.start_mel_token
        return fake_inputs, batched_mel_emb, attention_mask

    def inference_speech(self, speech_conditioning_mel, text_inputs, cond_mel_lengths=None, input_tokens=None, num_return_sequences=1,
                         max_generate_length=None, typical_sampling=False, typical_mass=.9, speaker_ids=None, **hf_generate_kwargs):
        """
        執行語音推論生成。

        Args:
            speech_conditioning_mel: 條件 Mel 頻譜圖
            text_inputs: 文字輸入
            max_generate_length: 生成長度限制
        """
        if speech_conditioning_mel.ndim == 2:
            speech_conditioning_mel = speech_conditioning_mel.unsqueeze(0)
        if cond_mel_lengths is None:
            cond_mel_lengths = torch.tensor([speech_conditioning_mel.shape[-1]], device=speech_conditioning_mel.device)
        conds_latent = self.get_conditioning(speech_conditioning_mel, cond_mel_lengths, speaker_ids=speaker_ids)
        input_ids, inputs_embeds, attention_mask = self.prepare_gpt_inputs(conds_latent, text_inputs)
        self.inference_model.store_mel_emb(inputs_embeds)
        if input_tokens is None:
            inputs = input_ids
        else:
            if input_tokens.ndim == 1:
                input_tokens = input_tokens.unsqueeze(0)
            assert num_return_sequences % input_tokens.shape[0] == 0, \
                    "num_return_sequences 必須能被 input_tokens 的 Batch size 整除"
            assert num_return_sequences % text_inputs.shape[0] == 0, \
                    "num_return_sequences 必須能被 text_inputs 的 Batch size 整除"
            b = num_return_sequences // input_ids.shape[0]
            if b > 1:
                input_ids = input_ids.repeat(b, 1)
                attention_mask = attention_mask.repeat(b, 1)
            input_tokens = input_tokens.repeat(num_return_sequences // input_tokens.shape[0], 1)
            inputs = torch.cat([input_ids, input_tokens], dim=1)
            attention_mask = F.pad(attention_mask, (0, input_tokens.shape[1]), value=1)
        trunc_index = inputs.shape[1]
        logits_processor = LogitsProcessorList()
        if typical_sampling:
            if not (typical_mass > 0.0 and typical_mass < 1.0):
                raise ValueError(f"`typical_mass` 必須在 0 和 1 之間，但得到 {typical_mass}")
            min_tokens_to_keep = 2 if hf_generate_kwargs.get("num_beams", 1) > 1 else 1
            logits_processor.append(TypicalLogitsWarper(mass=typical_mass, min_tokens_to_keep=min_tokens_to_keep))
        max_length = (trunc_index + self.max_mel_tokens - 1) if max_generate_length is None else trunc_index + max_generate_length
        output = self.inference_model.generate(inputs, 
                                            bos_token_id=self.start_mel_token, pad_token_id=self.stop_mel_token,
                                            eos_token_id=self.stop_mel_token, attention_mask=attention_mask,
                                            max_length=max_length, logits_processor=logits_processor,
                                            num_return_sequences=num_return_sequences,
                                            **hf_generate_kwargs)
        if isinstance(output, torch.Tensor):
            return output[:, trunc_index:]
        # GenerateOutput
        output.sequences = output.sequences[:, trunc_index:]
        return output
