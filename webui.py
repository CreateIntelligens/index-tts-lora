import json
import os
import sys
import threading
import time
import pandas as pd
import argparse
import gradio as gr
import torch
import warnings

# 過濾非必要的警告訊息
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 設定專案路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

from indextts.infer import IndexTTS

# 參數解析
parser = argparse.ArgumentParser(description="IndexTTS WebUI - 語音合成網頁介面")
parser.add_argument("--verbose", action="store_true", default=False, help="啟用詳細日誌模式")
parser.add_argument("--port", type=int, default=7860, help="WebUI 執行埠號")
parser.add_argument("--host", type=str, default="127.0.0.1", help="WebUI 監聽位址")
parser.add_argument("--model_dir", type=str, default="checkpoints", help="模型檢查點目錄")
cmd_args = parser.parse_args()

# 驗證模型目錄與必要檔案
if not os.path.exists(cmd_args.model_dir):
    print(f"錯誤：模型目錄 {cmd_args.model_dir} 不存在。請先下載模型。")
    sys.exit(1)

required_files = [
    "bigvgan_generator.pth",
    "bpe.model",
    "gpt.pth",
    "config.yaml",
]

for file in required_files:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"錯誤：缺少必要檔案 {file_path}。請下載該檔案。")
        sys.exit(1)

# I18n 模組相容性處理
try:
    from tools.i18n.i18n import I18nAuto
except ModuleNotFoundError:
    class I18nAuto:  # type: ignore
        """
        I18nAuto 的簡易替換類別，用於在缺少 tools 模組時提供基本功能。
        """
        def __init__(self, language="zh_CN"):
            self.language = language

        def __call__(self, text: str) -> str:
            return text

        def __getattr__(self, name):
            # 攔截所有未定義屬性的存取，防止程式崩潰
            return self

i18n = I18nAuto(language="zh_CN")
MODE = 'local'

# 初始化 TTS 引擎
tts = IndexTTS(model_dir=cmd_args.model_dir, cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"))

os.makedirs("outputs/tasks", exist_ok=True)
os.makedirs("prompts", exist_ok=True)


def get_available_models() -> dict:
    """
    掃描並回傳所有可用的 GPT 模型檢查點。

    Returns:
        dict: 鍵為顯示名稱，值為檔案路徑的字典。
    """
    models = {}

    # 1. 預設模型
    default_model = os.path.join(cmd_args.model_dir, "gpt.pth")
    if os.path.exists(default_model):
        models["預設模型 (gpt.pth)"] = default_model

    # 2. 微調後的模型
    finetune_dir = "finetune_models/checkpoints"
    if os.path.exists(finetune_dir):
        # 支援 .pth 與 .pt 格式
        pth_files = sorted([
            f for f in os.listdir(finetune_dir) 
            if f.endswith('.pth')
        ])
        for pth_file in pth_files:
            display_name = f"訓練模型 - {pth_file}"
            full_path = os.path.join(finetune_dir, pth_file)
            models[display_name] = full_path

    return models


def reload_gpt_model(model_path: str, progress=gr.Progress()) -> str:
    """
    重新載入指定的 GPT 模型權重。

    Args:
        model_path (str): 模型檔案路徑。
        progress (gr.Progress): Gradio 進度條物件。

    Returns:
        str: 操作結果訊息。
    """
    global tts
    try:
        progress(0, desc="正在初始化模型...")

        from indextts.gpt.model import UnifiedVoice
        from indextts.utils.checkpoint import load_checkpoint
        
        # 建立新的模型實例
        new_gpt = UnifiedVoice(**tts.cfg.gpt)

        progress(0.3, desc="載入權重...")
        load_checkpoint(new_gpt, model_path)

        progress(0.6, desc="配置運算裝置與精度...")
        new_gpt = new_gpt.to(tts.device)
        
        # 根據全域設定配置精度與 DeepSpeed
        if tts.is_fp16:
            new_gpt.eval().half()
            try:
                import deepspeed
                use_deepspeed = True
            except ImportError:
                use_deepspeed = False
            new_gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=True)
        else:
            new_gpt.eval()
            new_gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=False)

        progress(0.9, desc="切換模型實例...")
        
        # 釋放舊模型記憶體
        del tts.gpt
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 替換為新模型
        tts.gpt = new_gpt
        tts.gpt_path = model_path

        progress(1.0, desc="完成")
        return f"✅ 成功載入模型: {os.path.basename(model_path)}"

    except Exception as e:
        return f"❌ 模型載入失敗: {str(e)}"


available_models = get_available_models()

# 載入範例測試案例
example_cases = []
try:
    with open("tests/cases.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            example_cases.append([
                os.path.join("tests", example.get("prompt_audio", "sample_prompt.wav")),
                example.get("text"), 
                ["普通推理", "批次推理"][example.get("infer_mode", 0)]
            ])
except FileNotFoundError:
    pass


def gen_single(prompt, text, infer_mode, max_text_tokens_per_sentence=120, sentences_bucket_max_size=4,
                *args, progress=gr.Progress()):
    """
    執行單次語音生成任務。
    """
    output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    
    # 設定 Gradio 進度回調
    tts.gr_progress = progress
    
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
        
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
    }

    if infer_mode == "普通推理":
        output = tts.infer(
            prompt, text, output_path, 
            verbose=cmd_args.verbose,
            max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
            **kwargs
        )
    else:
        # 批次推理模式
        output = tts.infer_fast(
            prompt, text, output_path, 
            verbose=cmd_args.verbose,
            max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
            sentences_bucket_max_size=(sentences_bucket_max_size),
            **kwargs
        )
    return gr.update(value=output, visible=True)


def update_prompt_audio():
    return gr.update(interactive=True)


# 建構 Gradio 介面
with gr.Blocks(title="IndexTTS Demo") as demo:
    mutex = threading.Lock()
    gr.HTML('''
    <h2><center>IndexTTS: 工業級高效零樣本文字轉語音系統</center></h2>
    <p align="center">
        <a href='https://arxiv.org/abs/2502.05512'><img src='https://img.shields.io/badge/ArXiv-2502.05512-red'></a>
    </p>
    ''')

    # 模型選擇區域
    with gr.Accordion("🎯 模型選擇", open=True):
        with gr.Row():
            model_choices = list(available_models.keys())
            default_choice = model_choices[0] if model_choices else None

            model_dropdown = gr.Dropdown(
                choices=model_choices,
                value=default_choice,
                label="選擇 GPT 模型",
                info=f"當前已載入: {os.path.basename(tts.gpt_path)}"
            )

            reload_button = gr.Button("🔄 載入模型", variant="primary")
            refresh_button = gr.Button("🔍 重新掃描", variant="secondary")

        def on_reload_model(selected_model, progress=gr.Progress()):
            if selected_model not in available_models:
                gr.Warning("無效的模型選擇")
                return gr.update()
            model_path = available_models[selected_model]
            result = reload_gpt_model(model_path, progress)
            new_info = f"當前已載入: {os.path.basename(tts.gpt_path)}"
            if "成功" in result:
                gr.Info(result)
            else:
                gr.Warning(result)
            return gr.update(info=new_info)

        def on_refresh_models():
            global available_models
            available_models = get_available_models()
            new_choices = list(available_models.keys())
            gr.Info(f"掃描完成，找到 {len(new_choices)} 個模型")
            return gr.update(choices=new_choices, value=new_choices[0] if new_choices else None)

        reload_button.click(
            on_reload_model,
            inputs=[model_dropdown],
            outputs=[model_dropdown]
        )

        refresh_button.click(
            on_refresh_models,
            inputs=[],
            outputs=[model_dropdown]
        )

    with gr.Tab("音訊生成"):
        with gr.Row():
            os.makedirs("prompts", exist_ok=True)
            prompt_audio = gr.Audio(label="參考音訊", key="prompt_audio",
                                    sources=["upload", "microphone"], type="filepath")
            
            with gr.Column():
                input_text_single = gr.TextArea(
                    label="文字輸入",
                    key="input_text_single", 
                    placeholder="請輸入目標文字", 
                    info=f"當前模型版本: {tts.model_version or '1.0'}"
                )
                infer_mode = gr.Radio(
                    choices=["普通推理", "批次推理"], 
                    label="推理模式",
                    info="批次推理：更適合長句，效能較高",
                    value="普通推理"
                )        
                gen_button = gr.Button("生成語音", key="gen_button", interactive=True)
            output_audio = gr.Audio(label="生成結果", visible=True, key="output_audio")
        
        with gr.Accordion("進階生成參數設定", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**GPT2 取樣設定**\n參數會影響音訊多樣性和生成速度。")
                    with gr.Row():
                        do_sample = gr.Checkbox(label="啟用取樣 (Do Sample)", value=True, info="是否進行隨機取樣")
                        temperature = gr.Slider(label="溫度 (Temperature)", minimum=0.1, maximum=2.0, value=1.0, step=0.1)
                    with gr.Row():
                        top_p = gr.Slider(label="Top-P", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                        top_k = gr.Slider(label="Top-K", minimum=0, maximum=100, value=30, step=1)
                        num_beams = gr.Slider(label="Beam Search 數量", value=3, minimum=1, maximum=10, step=1)
                    with gr.Row():
                        repetition_penalty = gr.Number(label="重複懲罰 (Repetition Penalty)", value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                        length_penalty = gr.Number(label="長度懲罰 (Length Penalty)", value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                    max_mel_tokens = gr.Slider(
                        label="最大 Mel Token 數", 
                        value=600, 
                        minimum=50, 
                        maximum=tts.cfg.gpt.max_mel_tokens, 
                        step=10, 
                        info="生成 Token 最大數量，過小會導致音訊被截斷", 
                        key="max_mel_tokens"
                    )

                with gr.Column(scale=2):
                    gr.Markdown("**分句設定**\n影響音訊品質與生成效率。")
                    with gr.Row():
                        max_text_tokens_per_sentence = gr.Slider(
                            label="分句最大 Token 數", 
                            value=120, minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2, 
                            key="max_text_tokens_per_sentence",
                            info="建議 80~200。值越大分句越長；值越小分句越碎。"
                        )
                        sentences_bucket_max_size = gr.Slider(
                            label="分句分桶容量 (批次推理)", 
                            value=4, minimum=1, maximum=16, step=1, 
                            key="sentences_bucket_max_size",
                            info="建議 2-8。值越大批次處理的分句數越多，但記憶體消耗較大。"
                        )
                    with gr.Accordion("分句結果預覽", open=True):
                        sentences_preview = gr.Dataframe(
                            headers=["序號", "分句內容", "Token數"],
                            key="sentences_preview",
                            wrap=True,
                        )
            
            advanced_params = [
                do_sample, top_p, top_k, temperature,
                length_penalty, num_beams, repetition_penalty, max_mel_tokens,
            ]
        
        if len(example_cases) > 0:
            gr.Examples(
                examples=example_cases,
                inputs=[prompt_audio, input_text_single, infer_mode],
            )

    def on_input_text_change(text, max_tokens_per_sentence):
        if text and len(text) > 0:
            text_tokens_list = tts.tokenizer.tokenize(text)
            sentences = tts.tokenizer.split_sentences(
                text_tokens_list, 
                max_tokens_per_sentence=int(max_tokens_per_sentence)
            )
            data = []
            for i, s in enumerate(sentences):
                sentence_str = ''.join(s)
                tokens_count = len(s)
                data.append([i, sentence_str, tokens_count])
            
            return {
                sentences_preview: gr.update(value=data, visible=True, type="array"),
            }
        else:
            df = pd.DataFrame([], columns=["序號", "分句內容", "Token數"])
            return {
                sentences_preview: gr.update(value=df)
            }

    # 事件綁定
    input_text_single.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_sentence],
        outputs=[sentences_preview]
    )
    max_text_tokens_per_sentence.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_sentence],
        outputs=[sentences_preview]
    )
    prompt_audio.upload(
        update_prompt_audio,
        inputs=[],
        outputs=[gen_button]
    )

    gen_button.click(
        gen_single,
        inputs=[
            prompt_audio, input_text_single, infer_mode,
            max_text_tokens_per_sentence, sentences_bucket_max_size,
            *advanced_params,
        ],
        outputs=[output_audio]
    )

if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port)
