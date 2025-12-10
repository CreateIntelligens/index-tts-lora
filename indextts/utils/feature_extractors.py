import torch
import torchaudio
from torch import nn
from indextts.utils.common import safe_log


class FeatureExtractor(nn.Module):
    """
    特徵提取器基礎類別 (Base class for feature extractors)。
    """

    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        從輸入音訊中提取特徵。

        Args:
            audio (Tensor): 輸入音訊波形。

        Returns:
            Tensor: 提取的特徵張量，形狀為 (B, C, L)，
                    其中 B 為批次大小，C 為輸出特徵數，L 為序列長度。
        """
        raise NotImplementedError("子類別必須實作 forward 方法。")


class MelSpectrogramFeatures(FeatureExtractor):
    """
    Mel 頻譜圖特徵提取器。

    使用 torchaudio.transforms.MelSpectrogram 進行提取，並應用對數轉換。

    Args:
        sample_rate (int): 取樣率。
        n_fft (int): FFT 視窗大小。
        hop_length (int): 幀移 (Hop length)。
        win_length (int): 視窗長度。
        n_mels (int): Mel 濾波器組數量。
        mel_fmin (float): Mel 頻率下限。
        mel_fmax (float): Mel 頻率上限。
        normalize (bool): 是否標準化。
        padding (str): 填充模式 ('center' 或 'same')。
    """
    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256, win_length=None,
                 n_mels=100, mel_fmin=0, mel_fmax=None, normalize=False, padding="center"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding 模式必須為 'center' 或 'same'。")
        self.padding = padding
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=1,
            normalized=normalize,
            f_min=mel_fmin,
            f_max=mel_fmax,
            n_mels=n_mels,
            center=padding == "center",
        )

    def forward(self, audio, **kwargs):
        if self.padding == "same":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            audio = torch.nn.functional.pad(audio, (pad // 2, pad // 2), mode="reflect")
        mel = self.mel_spec(audio)
        mel = safe_log(mel)
        return mel
