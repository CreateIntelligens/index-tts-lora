
from typing import Optional, Tuple

import torch
import torch.nn as nn

from indextts.gpt.conformer.attention import (MultiHeadedAttention,
                                              RelPositionMultiHeadedAttention)
from indextts.gpt.conformer.embedding import (NoPositionalEncoding,
                                              PositionalEncoding,
                                              RelPositionalEncoding)
from indextts.gpt.conformer.subsampling import (Conv2dSubsampling2,
                                                Conv2dSubsampling4,
                                                Conv2dSubsampling6,
                                                Conv2dSubsampling8,
                                                LinearNoSubsampling)
from indextts.utils.common import make_pad_mask


class PositionwiseFeedForward(torch.nn.Module):
    """
    逐位置前饋網路 (Positionwise Feed-Forward Network)。

    此模組獨立應用於序列的每個位置，輸入與輸出維度相同。

    Args:
        idim (int): 輸入維度。
        hidden_units (int): 隱藏層單元數量。
        dropout_rate (float): Dropout 比率。
        activation (torch.nn.Module): 激活函數。
    """

    def __init__(self,
                 idim: int,
                 hidden_units: int,
                 dropout_rate: float,
                 activation: torch.nn.Module = torch.nn.ReLU()):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        前向傳播。

        Args:
            xs: 輸入張量 (B, L, D)
        Returns:
            輸出張量 (B, L, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class ConvolutionModule(nn.Module):
    """
    Conformer 模型中的卷積模組。
    """

    def __init__(self,
                 channels: int,
                 kernel_size: int = 15,
                 activation: nn.Module = nn.ReLU(),
                 bias: bool = True):
        """
        建構卷積模組。

        Args:
            channels (int): 卷積層的通道數。
            kernel_size (int): 卷積核大小。
            bias (bool): 是否使用偏置。
        """
        super().__init__()

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        
        # kernel_size 必須為奇數以確保對稱卷積
        assert (kernel_size - 1) % 2 == 0
        padding = (kernel_size - 1) // 2
        self.lorder = 0

        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )

        self.use_layer_norm = True
        self.norm = nn.LayerNorm(channels)

        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation

    def forward(
            self,
            x: torch.Tensor,
            mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
            cache: torch.Tensor = torch.zeros((0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        計算卷積模組。

        Args:
            x (torch.Tensor): 輸入張量 (#batch, time, channels)。
            mask_pad (torch.Tensor): 用於批次填充的遮罩 (#batch, 1, time)。
            cache (torch.Tensor): 左側上下文快取 (僅用於因果卷積)。

        Returns:
            torch.Tensor: 輸出張量 (#batch, time, channels)。
        """
        # 交換時間與特徵維度
        x = x.transpose(1, 2)  # (#batch, channels, time)

        if mask_pad.size(2) > 0:
            x.masked_fill_(~mask_pad, 0.0)

        if self.lorder > 0:
            if cache.size(2) == 0:
                x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)
            else:
                assert cache.size(0) == x.size(0)
                assert cache.size(1) == x.size(1)
                x = torch.cat((cache, x), dim=2)
            assert (x.size(2) > self.lorder)
            new_cache = x[:, :, -self.lorder:]
        else:
            new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)

        # GLU 機制
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)
        
        if mask_pad.size(2) > 0:
            x.masked_fill_(~mask_pad, 0.0)

        return x.transpose(1, 2), new_cache


class ConformerEncoderLayer(nn.Module):
    """
    Conformer 編碼器層模組。

    Args:
        size (int): 輸入維度。
        self_attn (torch.nn.Module): 自注意力模組實例。
        feed_forward (torch.nn.Module): 前饋模組實例。
        feed_forward_macaron (torch.nn.Module): 額外的 Macaron 風格前饋模組實例。
        conv_module (torch.nn.Module): 卷積模組實例。
        dropout_rate (float): Dropout 比率。
        normalize_before (bool): 是否在子區塊之前使用 LayerNorm。
        concat_after (bool): 是否將注意力層的輸入與輸出串聯。
    """

    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: Optional[nn.Module] = None,
        feed_forward_macaron: Optional[nn.Module] = None,
        conv_module: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, eps=1e-5)
        self.norm_mha = nn.LayerNorm(size, eps=1e-5)
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-5)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = nn.LayerNorm(size, eps=1e-5)
            self.norm_final = nn.LayerNorm(size, eps=1e-5)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        else:
            self.concat_linear = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        計算編碼特徵。

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): 輸入遮罩 (#batch, time, time)
            pos_emb (torch.Tensor): 位置編碼
            mask_pad (torch.Tensor): 批次填充遮罩 (#batch, 1, time)
            att_cache (torch.Tensor): 注意力快取
            cnn_cache (torch.Tensor): 卷積快取

        Returns:
            Tuple[torch.Tensor, ...]: 輸出張量, 遮罩, 更新後的 att_cache, 更新後的 cnn_cache
        """

        # Macaron 風格前饋
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # 多頭自注意力模組
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        x_att, new_att_cache = self.self_attn(
            x, x, x, mask, pos_emb, att_cache)
        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # 卷積模組
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                x = self.norm_conv(x)

        # 前饋模組
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache


class BaseEncoder(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        """
        編碼器基礎類別。

        Args:
            input_size (int): 輸入維度。
            output_size (int): 注意力輸出維度。
            attention_heads (int): 多頭注意力的頭數。
            linear_units (int): 前饋層隱藏單元數。
            num_blocks (int): 解碼器區塊數量。
            dropout_rate (float): Dropout 比率。
            input_layer (str): 輸入層類型。
            pos_enc_layer_type (str): 位置編碼層類型。
            normalize_before (bool): 是否在子區塊前進行 LayerNorm。
            concat_after (bool): 是否串聯注意力層輸入與輸出。
        """
        super().__init__()
        self._output_size = output_size

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "no_pos":
            pos_enc_class = NoPositionalEncoding
        else:
            raise ValueError("未知的 pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            subsampling_class = LinearNoSubsampling
        elif input_layer == "conv2d2":
            subsampling_class = Conv2dSubsampling2
        elif input_layer == "conv2d":
            subsampling_class = Conv2dSubsampling4
        elif input_layer == "conv2d6":
            subsampling_class = Conv2dSubsampling6
        elif input_layer == "conv2d8":
            subsampling_class = Conv2dSubsampling8
        else:
            raise ValueError("未知的 input_layer: " + input_layer)

        self.embed = subsampling_class(
            input_size,
            output_size,
            dropout_rate,
            pos_enc_class(output_size, dropout_rate),
        )

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-5)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        嵌入位置並進行前向傳播。

        Args:
            xs: 填充後的輸入張量 (B, T, D)
            xs_lens: 輸入長度 (B)

        Returns:
            xs: 輸出張量 (B, T', D)
            masks: 填充遮罩 (B, 1, T')
        """
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        xs, pos_emb, masks = self.embed(xs, masks)
        chunk_masks = masks
        mask_pad = masks
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)
        
        return xs, masks


class ConformerEncoder(BaseEncoder):
    """
    Conformer 編碼器模組。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        concat_after: bool = False,
        macaron_style: bool = False,
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
    ):
        """
        建構 ConformerEncoder。

        Args:
            macaron_style (bool): 是否使用 Macaron 風格。
            use_cnn_module (bool): 是否使用卷積模組。
            cnn_module_kernel (int): 卷積模組的卷積核大小。
        """

        super().__init__(input_size, output_size, attention_heads,
                         linear_units, num_blocks, dropout_rate,
                         input_layer, pos_enc_layer_type, normalize_before,
                         concat_after)

        activation = torch.nn.SiLU()

        # 定義自注意力模組
        if pos_enc_layer_type != "rel_pos":
            encoder_selfattn_layer = MultiHeadedAttention
        else:
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            dropout_rate,
        )

        # 定義前饋模組
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        
        # 定義卷積模組
        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size,
                                  cnn_module_kernel,
                                  activation,)

        self.encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(
                    *positionwise_layer_args) if macaron_style else None,
                convolution_layer(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ) for _ in range(num_blocks)
        ])
