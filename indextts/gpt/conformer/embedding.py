# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)

"""Positonal Encoding Module."""

import math
from typing import Tuple, Union

import torch
import torch.nn.functional as F


class PositionalEncoding(torch.nn.Module):
    """
    位置編碼 (Positional Encoding)。

    使用正弦和餘弦函數生成位置編碼。

    Args:
        d_model (int): 嵌入維度。
        dropout_rate (float): Dropout 比率。
        max_len (int): 最大輸入長度。
        reverse (bool): 是否反轉 (通常不用於標準 PE)。
    """
    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 max_len: int = 5000,
                 reverse: bool = False):
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2) *
            -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self,
                x: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        添加位置編碼。

        Args:
            x (torch.Tensor): 輸入張量 (#batch, time, ...)。
            offset (Union[int, torch.Tensor]): 位置偏移量。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 編碼後的張量與位置嵌入張量。
        """

        self.pe = self.pe.to(x.device)
        pos_emb = self.position_encoding(offset, x.size(1), False)
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset: Union[int, torch.Tensor], size: int,
                          apply_dropout: bool = True) -> torch.Tensor:
        """
        以串流方式獲取位置編碼。

        Args:
            offset (Union[int, torch.Tensor]): 起始偏移量。
            size (int): 所需的位置編碼長度。
            apply_dropout (bool): 是否應用 Dropout。

        Returns:
            torch.Tensor: 對應的位置編碼。
        """
        if isinstance(offset, int):
            assert offset + size < self.max_len
            pos_emb = self.pe[:, offset:offset + size]
        elif isinstance(offset, torch.Tensor) and offset.dim() == 0:  # scalar
            assert offset + size < self.max_len
            pos_emb = self.pe[:, offset:offset + size]
        else:
            assert torch.max(offset) + size < self.max_len
            index = offset.unsqueeze(1) + \
                torch.arange(0, size).to(offset.device)  # B X T
            flag = index > 0
            index = index * flag
            pos_emb = F.embedding(index, self.pe[0])  # B X T X d_model

        if apply_dropout:
            pos_emb = self.dropout(pos_emb)
        return pos_emb

class RelPositionalEncoding(PositionalEncoding):
    """
    相對位置編碼模組 (Relative Positional Encoding)。
    
    參考: https://arxiv.org/abs/1901.02860 Appendix B

    Args:
        d_model (int): 嵌入維度。
        dropout_rate (float): Dropout 比率。
        max_len (int): 最大輸入長度。
    """
    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(self,
                x: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        計算位置編碼。

        Args:
            x (torch.Tensor): 輸入張量 (batch, time, *)。
            offset: 偏移量。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 編碼後的張量與位置嵌入。
        """
        self.pe = self.pe.to(x.device)
        x = x * self.xscale
        pos_emb = self.position_encoding(offset, x.size(1), False)
        return self.dropout(x), self.dropout(pos_emb)


class NoPositionalEncoding(torch.nn.Module):
    """
    無位置編碼 (No Positional Encoding)。
    """
    def __init__(self, d_model: int, dropout_rate: float):
        super().__init__()
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self,
                x: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        僅返回零向量以保持介面相容性。
        """
        pos_emb = torch.zeros(1, x.size(1), self.d_model).to(x.device)
        return self.dropout(x), pos_emb

    def position_encoding(
            self, offset: Union[int, torch.Tensor], size: int) -> torch.Tensor:
        return torch.zeros(1, size, self.d_model)
