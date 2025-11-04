# index-tts-lora

[中文版本](https://github.com/asr-pub/index-tts-lora/blob/main/README_zh.md) | [English Version](https://github.com/asr-pub/index-tts-lora/blob/main/README.md)

This project is based on Bilibili's [index-tts](https://github.com/index-tts/index-tts), providing **LoRA fine-tuning** solutions for both **single-speaker and multi-speaker** setups. It aims to enhance **prosody and naturalness** in high-quality speaker audio synthesis.

### Training & Inference

#### 1. Audio token and speaker condition extraction

```shell
# Extract tokens and speaker conditions
python tools/extract_codec.py --audio_list ${audio_list} --extract_condition

# audio_list format: audio_path + transcript, separated by \t
/path/to/audio.wav 小朋友們，大家好，我是凱叔，今天我們講一個龜兔賽跑的故事。
```

After extraction, the processed files and `speaker_info.json` will be generated under the `finetune_data/processed_data/` directory. For example:

```json
[
    {
        "speaker": "kaishu_30min",
        "avg_duration": 6.6729,
        "sample_num": 270,
        "total_duration_in_seconds": 1801.696,
        "total_duration_in_minutes": 30.028,
        "total_duration_in_hours": 0.500,
        "train_jsonl": "/path/to/kaishu_30min/metadata_train.jsonl",
        "valid_jsonl": "/path/to/kaishu_30min/metadata_valid.jsonl",
        "medoid_condition": "/path/to/kaishu_30min/medoid_condition.npy"
    }
]
```

#### 2. Training

```shell
python train.py
```

#### 3. Inference

```shell
python indextts/infer.py
```

### Fine-tuning Results

This experiment uses **Chinese audio data** from *Kai Shu Tells Stories*, with a total duration of **\~30 minutes** and **270 audio clips**.
The dataset is split into **244 training samples** and **26 validation samples**.
Note: Transcripts were generated automatically via ASR and punctuation models, without manual correction, so some errors are expected.

Example training sample, `他上了馬車，來到了皇宮之中。`：[kaishu_train_01.wav](https://github.com/user-attachments/files/22354621/kaishu_train_01.wav)


#### 1. Speech Synthesis Examples


| Text                                                         | Audio                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 老宅的鐘錶停在午夜三點，灰塵中浮現一串陌生腳印。偵探蹲下身，發現地板縫隙裡藏著一枚帶血的戒指。 | [kaishu_cn_1.wav](https://github.com/user-attachments/files/22354649/kaishu_cn_1.wav) |
| 月光下，南瓜突然長出笑臉，藤蔓扭動著推開花園柵欄。小女孩踮起腳，聽見蘑菇在哼唱古老的搖籃曲。 | [kaishu_cn_2.wav](https://github.com/user-attachments/files/22354652/kaishu_cn_2.wav) |
| 那麼Java裡面中級還要學，M以及到外部前端的應用系統開發，要學到Java Script的資料庫，要學做動態的網站。 | [kaishu_cn_en_mix_1.wav](https://github.com/user-attachments/files/22354654/kaishu_cn_en_mix_1.wav) |
| 這份 financial report 詳細分析了公司在過去一個季度的 revenue performance 和 expenditure trends。 | [kaishu_cn_en_mix_2.wav](https://github.com/user-attachments/files/22354656/kaishu_cn_en_mix_2.wav) |
| 上山下山上一山，下一山，跑了三里三米三，登了一座大高山，山高海拔三百三。上了山，大聲喊：我比山高三尺三。 | [kaishu_raokouling.wav](https://github.com/user-attachments/files/22354658/kaishu_raokouling.wav) |
| A thin man lies against the side of the street with his shirt and a shoe off and bags nearby. | [kaishu_en_1.wav](https://github.com/user-attachments/files/22354659/kaishu_en_1.wav) |
| As research continued, the protective effect of fluoride against dental decay was demonstrated. | [kaishu_en_2.wav](https://github.com/user-attachments/files/22354661/kaishu_en_2.wav) |

#### 2. Model Evaluation
For details of the evaluation set, see: [2025 Benchmark of Mainstream TTS Models: Who Is the Best Voice Synthesis Solution?](https://mp.weixin.qq.com/s/5z_aRKQG3OIv7fnSdxegqQ)
<img width="1182" height="261" alt="image" src="https://github.com/user-attachments/assets/fb86938d-95d9-4b10-9588-2de1e43b51d1" />

### Acknowledgements

[index-tts](https://github.com/index-tts/index-tts)

[finetune-index-tts](https://github.com/yrom/finetune-index-tts)
