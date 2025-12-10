# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
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

import os
import re

import torch
import yaml
from loguru import logger


def load_checkpoint(model: torch.nn.Module, model_pth: str) -> dict:
    """
    載入模型檢查點 (Checkpoint)。

    支援載入單一模型權重以及多說話人的條件嵌入 (Speaker Conditions)。

    Args:
        model (torch.nn.Module): 目標模型實例。
        model_pth (str): 檢查點檔案路徑 (.pth 或 .pt)。

    Returns:
        dict: 載入的設定檔內容 (如果存在對應的 .yaml 檔案)。
    """
    checkpoint = torch.load(model_pth, map_location='cpu', weights_only=False)
    
    # 獲取模型所在的運算裝置
    device = next(model.parameters()).device
    
    # 處理多說話人條件嵌入 (若存在於 checkpoint 頂層)
    if 'speaker_conditions' in checkpoint:
        speaker_conditions = checkpoint['speaker_conditions']
        num_speakers = len(speaker_conditions)
        logger.info(f"正在從檢查點載入 {num_speakers} 個說話人條件...")

        from tqdm import tqdm
        for speaker_id, condition_array in tqdm(speaker_conditions.items(),
                                                  desc="載入說話人條件",
                                                  unit="spk", ncols=80):
            # 轉換為張量並移動至正確裝置
            condition_tensor = torch.from_numpy(condition_array).float().to(device)

            # 確保維度為 (1, 32, dim)
            if condition_tensor.dim() == 2:
                condition_tensor = condition_tensor.unsqueeze(0)

            # 註冊為模型參數，名稱格式: mean_condition_{speaker_id}
            param_name = f"mean_condition_{speaker_id}"
            setattr(model, param_name, torch.nn.Parameter(condition_tensor))

        logger.info(f"✓ 已載入 {num_speakers} 個說話人條件")
    
    # 提取模型狀態字典
    model_state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # 處理單一平均條件 (向後相容：若存在於 model_state_dict 中)
    if 'mean_condition' in model_state_dict and hasattr(model, 'mean_condition'):
        logger.info(f"載入單一 mean_condition: 來源裝置={model_state_dict['mean_condition'].device}, 目標裝置={device}")
        model.mean_condition = model_state_dict['mean_condition'].to(device)
        
        # 從狀態字典中移除，避免 load_state_dict 時發生衝突
        model_state_dict = {k: v for k, v in model_state_dict.items() if k != 'mean_condition'}
    
    # 載入權重 (非嚴格模式，允許部分鍵值不匹配)
    model.load_state_dict(model_state_dict, strict=False)
    
    # 嘗試載入同名的 YAML 設定檔
    info_path = re.sub(r'\.(pth|pt)$', '.yaml', model_pth)
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    
    # 如果檢查點中包含說話人列表，則合併至設定中
    if 'speakers' in checkpoint:
        configs['speakers'] = checkpoint['speakers']
    
    return configs
