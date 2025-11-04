# index-tts-lora
本專案基於 Bilibili 的 [index-tts](https://github.com/index-tts/index-tts) ，提供 **LoRA 單說話人 / 多說話人** 的微調方案，用於提升精品說話人合成音訊的 **韻律和自然度**。

### 訓練與推理

#### 1. 音訊 token 與 speaker condition 提取

```shell
# 提取 token 和 speaker condition
python tools/extract_codec.py --audio_list ${audio_list} --extract_condition

# audio_list 格式：音訊路徑 + 文字，以 \t 分隔
/path/to/audio.wav 小朋友們，大家好，我是凱叔，今天我們講一個龜兔賽跑的故事。
```

提取完成後，會在 `finetune_data/processed_data/` 目錄下生成對應資料夾及 `speaker_info.json` 檔案。例如：

```shell
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

#### 2. 訓練

```shell
python train.py
```

#### 3. 推理

```
python indextts/infer.py
```
### 微調結果

本次實驗資料來自 **凱叔講故事** 的純中文音訊，總時長約 **30 分鐘**，共 **270 條音訊**。資料劃分為 **訓練集 244 條**、**驗證集 26 條**。需要注意的是，文字是透過 ASR 和標點模型自動生成的，未經過人工校對，因此存在一定錯誤率。

訓練樣音如下，`他上了馬車，來到了皇宮之中。`：[kaishu_train_01.wav](https://github.com/user-attachments/files/22354621/kaishu_train_01.wav)


#### 1. 音訊合成效果


|                        合成文字                              | 合成音訊                                           |
| ------------------------------------------------------------ | -------------------------------------------------- |
| 老宅的鐘錶停在午夜三點，灰塵中浮現一串陌生腳印。偵探蹲下身，發現地板縫隙裡藏著一枚帶血的戒指。 |  [kaishu_cn_1.wav](https://github.com/user-attachments/files/22354649/kaishu_cn_1.wav) |
| 月光下，南瓜突然長出笑臉，藤蔓扭動著推開花園柵欄。小女孩踮起腳，聽見蘑菇在哼唱古老的搖籃曲。 | [kaishu_cn_2.wav](https://github.com/user-attachments/files/22354652/kaishu_cn_2.wav) |
| 那麼Java裡面中級還要學，M以及到外部前端的應用系統開發，要學到Java Script的資料庫，要學做動態的網站。 | [kaishu_cn_en_mix_1.wav](https://github.com/user-attachments/files/22354654/kaishu_cn_en_mix_1.wav) |
| 這份 financial report 詳細分析了公司在過去一個季度的 revenue performance 和 expenditure trends。 | [kaishu_cn_en_mix_2.wav](https://github.com/user-attachments/files/22354656/kaishu_cn_en_mix_2.wav) |
| 上山下山上一山，下一山，跑了三里三米三，登了一座大高山，山高海拔三百三。上了山，大聲喊：我比山高三尺三。 | [kaishu_raokouling.wav](https://github.com/user-attachments/files/22354658/kaishu_raokouling.wav) |
| A thin man lies against the side of the street with his shirt and a shoe off and bags nearby. | [kaishu_en_1.wav](https://github.com/user-attachments/files/22354659/kaishu_en_1.wav)  |
| As research continued, the protective effect of fluoride against dental decay was demonstrated. |  [kaishu_en_2.wav](https://github.com/user-attachments/files/22354661/kaishu_en_2.wav) |

#### 2. 模型精度測試
測試集詳情，可參考: [2025 主流 TTS 模型橫評：誰才是最佳語音合成方案？](https://mp.weixin.qq.com/s/5z_aRKQG3OIv7fnSdxegqQ)
<img width="1182" height="261" alt="image" src="https://github.com/user-attachments/assets/fb86938d-95d9-4b10-9588-2de1e43b51d1" />

### 感謝

[index-tts](https://github.com/index-tts/index-tts)

[finetune-index-tts](https://github.com/yrom/finetune-index-tts)
