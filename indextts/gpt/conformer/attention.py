# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-Head Attention layer definition."""

import math
from typing import Tuple

import torch
from torch import nn


class MultiHeadedAttention(nn.Module):
    """
    多頭注意力層 (Multi-Head Attention layer)。

    Args:
        n_head (int): 注意力頭數。
        n_feat (int): 特徵維度。
        dropout_rate (float): Dropout 比率。
    """
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
        super().__init__()
        assert n_feat % n_head == 0
        # 假設 d_v 始終等於 d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        轉換 Query, Key 與 Value。

        Args:
            query (torch.Tensor): Query 張量 (#batch, time1, size)。
            key (torch.Tensor): Key 張量 (#batch, time2, size)。
            value (torch.Tensor): Value 張量 (#batch, time2, size)。

        Returns:
            Tuple[torch.Tensor, ...]: 轉換後的 Q, K, V 張量。
                尺寸皆為 (#batch, n_head, time, d_k)。
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(
        self, value: torch.Tensor, scores: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)
    ) -> torch.Tensor:
        """
        計算注意力上下文向量。

        Args:
            value (torch.Tensor): 轉換後的 Value (#batch, n_head, time2, d_k)。
            scores (torch.Tensor): 注意力分數 (#batch, n_head, time1, time2)。
            mask (torch.Tensor): 遮罩張量，支援 (#batch, 1, time2) 或 (#batch, time1, time2)。

        Returns:
            torch.Tensor: 加權後的 Value (#batch, time1, d_model)。
        """
        n_batch = value.size(0)
        
        if mask.size(2) > 0 :  # time2 > 0
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            # 對於最後一個 chunk，time2 可能大於 scores 的最後一維
            mask = mask[:, :, :, :scores.size(-1)]  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0)  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (x.transpose(1, 2).contiguous().view(n_batch, -1,
                                                 self.h * self.d_k)
             )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
                pos_emb: torch.Tensor = torch.empty(0),
                cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        計算縮放點積注意力 (Scaled Dot Product Attention)。

        Args:
            query (torch.Tensor): Query (#batch, time1, size)。
            key (torch.Tensor): Key (#batch, time2, size)。
            value (torch.Tensor): Value (#batch, time2, size)。
            mask (torch.Tensor): 遮罩張量。
            cache (torch.Tensor): 快取張量 (1, head, cache_t, d_k * 2)。

        Returns:
            torch.Tensor: 輸出張量 (#batch, time1, d_model)。
            torch.Tensor: 更新後的快取張量。
        """
        q, k, v = self.forward_qkv(query, key, value)

        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(
                cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        
        new_cache = torch.cat((k, v), dim=-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """
    帶有相對位置編碼的多頭注意力層。
    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): 注意力頭數。
        n_feat (int): 特徵維度。
        dropout_rate (float): Dropout 比率。
    """
    def __init__(self, n_head, n_feat, dropout_rate):
        super().__init__(n_head, n_feat, dropout_rate)
        # 位置編碼的線性變換
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # 兩個可學習的偏置參數 u 和 v
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, zero_triu: bool = False):
        """
        計算相對位置編碼的移位操作。

        Args:
            x (torch.Tensor): 輸入張量 (batch, time, size)。
            zero_triu (bool): 是否將下三角部分置零。

        Returns:
            torch.Tensor: 輸出張量。
        """

        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1),
                               device=x.device,
                               dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0],
                                 x.size()[1],
                                 x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query: torch.Tensor,
                key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
                pos_emb: torch.Tensor = torch.empty(0),
                cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        計算帶有相對位置編碼的縮放點積注意力。

        Args:
            query (torch.Tensor): Query (#batch, time1, size)。
            key (torch.Tensor): Key (#batch, time2, size)。
            value (torch.Tensor): Value (#batch, time2, size)。
            mask (torch.Tensor): 遮罩張量。
            pos_emb (torch.Tensor): 位置嵌入張量 (#batch, time2, size)。
            cache (torch.Tensor): 快取張量。

        Returns:
            torch.Tensor: 輸出張量 (#batch, time1, d_model)。
            torch.Tensor: 更新後的快取張量。
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(
                cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        
        new_cache = torch.cat((k, v), dim=-1)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # 計算注意力分數
        # matrix_ac: term (a) + term (c)
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # matrix_bd: term (b) + term (d)
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        
        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask), new_cache
