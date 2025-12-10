# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import json
import os
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

import indextts.BigVGAN.activations as activations
from indextts.BigVGAN.alias_free_activation.torch.act import \
    Activation1d as TorchActivation1d
from indextts.BigVGAN.ECAPA_TDNN import ECAPA_TDNN
from indextts.BigVGAN.env import AttrDict
from indextts.BigVGAN.utils import get_padding, init_weights


def load_hparams_from_json(path) -> AttrDict:
    with open(path) as f:
        data = f.read()
    return AttrDict(json.loads(data))


class AMPBlock1(torch.nn.Module):
    """
    AMPBlock1 (Anti-aliased Multi-Periodicity Block 1)。

    應用 Snake/SnakeBeta 激活函數，包含可訓練的週期性參數。
    AMPBlock1 額外包含一組固定膨脹率 (dilation=1) 的卷積層 (convs2)。

    Args:
        h (AttrDict): 超參數設定。
        channels (int): 卷積通道數。
        kernel_size (int): 卷積核大小。
        dilation (tuple): 膨脹率列表。
        activation (str): 激活函數類型 ('snake' 或 'snakebeta')。
    """

    def __init__(
        self,
        h: AttrDict,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
    ):
        super().__init__()

        self.h = h

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in range(len(dilation))
            ]
        )
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(
            self.convs2
        )

        if self.h.get("use_cuda_kernel", False):
            from alias_free_activation.cuda.activation1d import \
                Activation1d as CudaActivation1d

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.Snake(
                            channels, alpha_logscale=h.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.SnakeBeta(
                            channels, alpha_logscale=h.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "激活函數指定錯誤。請檢查配置檔中的 'activation' 欄位。"
            )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class AMPBlock2(torch.nn.Module):
    """
    AMPBlock2 (Anti-aliased Multi-Periodicity Block 2)。

    應用 Snake/SnakeBeta 激活函數。與 AMPBlock1 不同，此區塊不包含額外的固定膨脹卷積層。

    Args:
        h (AttrDict): 超參數設定。
        channels (int): 卷積通道數。
        kernel_size (int): 卷積核大小。
        dilation (tuple): 膨脹率列表。
        activation (str): 激活函數類型。
    """

    def __init__(
        self,
        h: AttrDict,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
    ):
        super().__init__()

        self.h = h

        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs.apply(init_weights)

        self.num_layers = len(self.convs)

        if self.h.get("use_cuda_kernel", False):
            from alias_free_activation.cuda.activation1d import \
                Activation1d as CudaActivation1d

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.Snake(
                            channels, alpha_logscale=h.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.SnakeBeta(
                            channels, alpha_logscale=h.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "激活函數指定錯誤。請檢查配置檔中的 'activation' 欄位。"
            )

    def forward(self, x):
        for c, a in zip(self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class BigVGAN(
    torch.nn.Module,
):
    """
    BigVGAN 神經聲碼器模型。

    應用抗鋸齒週期性激活函數於殘差區塊 (ResBlock)。
    BigVGAN-v2 支援使用優化的 CUDA Kernel 進行 AMP 計算 (僅限推理)。

    Args:
        h (AttrDict): 超參數設定。
        use_cuda_kernel (bool): 是否使用 CUDA Kernel (僅推理)。
    """

    def __init__(self, h: AttrDict, use_cuda_kernel: bool = False):
        super().__init__()
        self.h = h
        self.h["use_cuda_kernel"] = use_cuda_kernel

        if self.h.get("use_cuda_kernel", False):
            from alias_free_activation.cuda.activation1d import \
                Activation1d as CudaActivation1d

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        self.feat_upsample = h.feat_upsample
        self.cond_in_each_up_layer = h.cond_d_vector_in_each_upsampling_layer

        self.conv_pre = weight_norm(
            Conv1d(h.gpt_dim, h.upsample_initial_channel, 7, 1, padding=3)
        )

        if h.resblock == "1":
            resblock_class = AMPBlock1
        elif h.resblock == "2":
            resblock_class = AMPBlock2
        else:
            raise ValueError(
                f"錯誤的 resblock 類別指定: {h.resblock}"
            )

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                h.upsample_initial_channel // (2**i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k,
                                u,
                                padding=(k - u) // 2,
                            )
                        )
                    ]
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    resblock_class(h, ch, k, d, activation=h.activation)
                )

        activation_post = (
            activations.Snake(ch, alpha_logscale=h.snake_logscale)
            if h.activation == "snake"
            else (
                activations.SnakeBeta(ch, alpha_logscale=h.snake_logscale)
                if h.activation == "snakebeta"
                else None
            )
        )
        if activation_post is None:
            raise NotImplementedError(
                "激活函數指定錯誤。請檢查配置檔中的 'activation' 欄位。"
            )

        self.activation_post = Activation1d(activation=activation_post)

        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.conv_post = weight_norm(
            Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final)
        )

        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

        self.speaker_encoder = ECAPA_TDNN(h.num_mels, lin_neurons=h.speaker_embedding_dim)
        self.cond_layer = nn.Conv1d(h.speaker_embedding_dim, h.upsample_initial_channel, 1)
        if self.cond_in_each_up_layer:
            self.conds = nn.ModuleList()
            for i in range(len(self.ups)):
                ch = h.upsample_initial_channel // (2 ** (i + 1))
                self.conds.append(nn.Conv1d(h.speaker_embedding_dim, ch, 1))

    def forward(self, x, mel_refer, lens=None):
        speaker_embedding = self.speaker_encoder(mel_refer, lens)
        n_batch = x.size(0)
        contrastive_loss = None
        if n_batch * 2 == speaker_embedding.size(0):
            spe_emb_chunk1, spe_emb_chunk2 = speaker_embedding[:n_batch, :, :], speaker_embedding[n_batch:, :, :]
            contrastive_loss = self.cal_clip_loss(spe_emb_chunk1.squeeze(1), spe_emb_chunk2.squeeze(1),
                                                  self.logit_scale.exp())

            speaker_embedding = speaker_embedding[:n_batch, :, :]
        speaker_embedding = speaker_embedding.transpose(1, 2)

        if self.feat_upsample:
            x = torch.nn.functional.interpolate(
                x.transpose(1, 2),
                scale_factor=[4],
                mode="linear",
            ).squeeze(1)
        else:
            x = x.transpose(1, 2)

        x = self.conv_pre(x)
        x = x + self.cond_layer(speaker_embedding)

        for i in range(self.num_upsamples):
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)

            if self.cond_in_each_up_layer:
                x = x + self.conds[i](speaker_embedding)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = self.activation_post(x)
        x = self.conv_post(x)
        
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)

        return x, contrastive_loss

    def remove_weight_norm(self):
        try:
            print("[資訊] 移除 Weight Norm...")
            for l in self.ups:
                for l_i in l:
                    remove_weight_norm(l_i)
            for l in self.resblocks:
                l.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)
        except ValueError:
            print("[資訊] 模型已移除 Weight Norm，略過。")
            pass

    def _save_pretrained(self, save_directory: Path) -> None:
        """
        儲存模型權重與配置至本地目錄。
        """

        model_path = save_directory / "bigvgan_generator.pt"
        torch.save({"generator": self.state_dict()}, model_path)

        config_path = save_directory / "config.json"
        with open(config_path, "w") as config_file:
            json.dump(self.h, config_file, indent=4)

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str,
        cache_dir: str,
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        use_cuda_kernel: bool = False,
        **model_kwargs,
    ):
        """
        從預訓練權重載入模型。
        """

        if os.path.isdir(model_id):
            print("[資訊] 從本地目錄載入 config.json")
            config_file = os.path.join(model_id, "config.json")
        else:
            config_file = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        h = load_hparams_from_json(config_file)

        if use_cuda_kernel:
            print(
                f"[警告] 指定了 use_cuda_kernel=True。此選項僅支援推理 (不支援訓練)！"
            )
            print(
                f"[警告] 請確保系統已安裝與 PyTorch 版本匹配的 nvcc 和 ninja，否則模型初始化或生成將失敗。"
            )
        model = cls(h, use_cuda_kernel=use_cuda_kernel)

        if os.path.isdir(model_id):
            print("[資訊] 從本地目錄載入權重")
            model_file = os.path.join(model_id, "bigvgan_generator.pt")
        else:
            print(f"[資訊] 從 {model_id} 載入權重")
            model_file = hf_hub_download(
                repo_id=model_id,
                filename="bigvgan_generator.pt",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        checkpoint_dict = torch.load(model_file, map_location=map_location)

        try:
            model.load_state_dict(checkpoint_dict["generator"])
        except RuntimeError:
            print(
                f"[資訊] 預訓練權重不包含 Weight Norm，將在移除 Weight Norm 後載入！"
            )
            model.remove_weight_norm()
            model.load_state_dict(checkpoint_dict["generator"])

        return model
