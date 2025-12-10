import os
import random
import re

import torch
import torchaudio

MATPLOTLIB_FLAG = False


def load_audio(audiopath, sampling_rate):
    """
    載入音訊檔案並重取樣。

    Args:
        audiopath (str): 音訊檔案路徑。
        sampling_rate (int): 目標取樣率。

    Returns:
        torch.Tensor: 音訊波形張量 (1, T)。若失敗則返回 None。
    """
    audio, sr = torchaudio.load(audiopath)

    if audio.size(0) > 1:  # 混音為單聲道
        audio = audio[0].unsqueeze(0)

    if sr != sampling_rate:
        try:
            audio = torchaudio.functional.resample(audio, sr, sampling_rate)
        except Exception as e:
            print(f"[警告] 重取樣失敗: {audiopath}, 原始格式: {audio.shape}, SR: {sr}")
            return None
            
    # 裁剪無效值
    audio.clip_(-1, 1)
    return audio


def tokenize_by_CJK_char(line: str, do_upper_case=True) -> str:
    """
    對文字進行 CJK 字元切分。

    注意：所有返回的字元將被轉換為大寫 (若 do_upper_case 為 True)。

    範例:
      輸入 = "你好世界是 hello world 的中文"
      輸出 = "你 好 世 界 是 HELLO WORLD 的 中 文"

    Args:
      line: 輸入文字字串。

    Return:
      切分後的字串。
    """
    CJK_RANGE_PATTERN = (
        r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
    )
    chars = re.split(CJK_RANGE_PATTERN, line.strip())
    return " ".join([w.strip().upper() if do_upper_case else w.strip() for w in chars if w.strip()])


def de_tokenized_by_CJK_char(line: str, do_lower_case=False) -> str:
    """
    還原被 CJK 字元切分的文字。

    範例:
      輸入 = "你 好 世 界 是 HELLO WORLD 的 中 文"
      輸出 = "你好世界是 hello world 的中文"
    """
    # 將英文單詞替換為佔位符
    english_word_pattern = re.compile(r"([A-Z]+(?:[\s-][A-Z-]+)*)", re.IGNORECASE)
    english_sents = english_word_pattern.findall(line)
    for i, sent in enumerate(english_sents):
        line = line.replace(sent, f"<sent_{i}>")

    words = line.split()
    # 恢復英文句子
    sent_placeholder_pattern = re.compile(r"^.*?(<sent_(\d+)>)")
    for i in range(len(words)):
        m = sent_placeholder_pattern.match(words[i])
        if m:
            placeholder_index = int(m.group(2))
            words[i] = words[i].replace(m.group(1), english_sents[placeholder_index])
            if do_lower_case:
                words[i] = words[i].lower()
    return "".join(words)


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    建立填充遮罩張量 (Padding Mask)。

    Args:
        lengths (torch.Tensor): 每個序列的長度 (B,)。
        max_len (int): 最大長度，若為 0 則使用 lengths 中的最大值。

    Returns:
        torch.Tensor: 遮罩張量，填充部分為 True。

    範例:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    計算安全對數 (Safe Logarithm)。

    對輸入進行裁剪以避免 log(0) 產生無窮大值。

    Args:
        x (Tensor): 輸入張量。
        clip_val (float, optional): 裁剪的最小值。預設為 1e-7。

    Returns:
        Tensor: 計算對數後的張量。
    """
    return torch.log(torch.clip(x, min=clip_val))
