import json
import os
import sys
import threading
import time
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(description="IndexTTS WebUI")
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="checkpoints", help="Model checkpoints directory")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bigvgan_generator.pth",
    "bpe.model",
    "gpt.pth",
    "config.yaml",
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr

from indextts.infer import IndexTTS
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="zh_CN")
MODE = 'local'
tts = IndexTTS(model_dir=cmd_args.model_dir, cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),)


os.makedirs("outputs/tasks",exist_ok=True)
os.makedirs("prompts",exist_ok=True)

# 掃描可用的 GPT 模型
def get_available_models():
    """掃描並返回所有可用的 GPT 模型"""
    models = {}

    # 1. 預設模型（checkpoints/gpt.pth）
    default_model = os.path.join(cmd_args.model_dir, "gpt.pth")
    if os.path.exists(default_model):
        models["預設模型 (gpt.pth)"] = default_model

    # 2. 訓練的模型（finetune_models/checkpoints/*.pth）
    finetune_dir = "finetune_models/checkpoints"
    if os.path.exists(finetune_dir):
        pth_files = sorted([f for f in os.listdir(finetune_dir) if f.endswith('.pth')])
        for pth_file in pth_files:
            display_name = f"訓練模型 - {pth_file}"
            full_path = os.path.join(finetune_dir, pth_file)
            models[display_name] = full_path

    return models

def reload_gpt_model(model_path, progress=gr.Progress()):
    """重新載入 GPT 模型"""
    global tts
    try:
        progress(0, desc="正在載入模型...")

        # 載入新模型
        from indextts.gpt.model import UnifiedVoice
        from indextts.utils.checkpoint import load_checkpoint
        import torch

        # 創建新的 GPT 模型實例
        new_gpt = UnifiedVoice(**tts.cfg.gpt)

        progress(0.3, desc="載入模型權重...")
        load_checkpoint(new_gpt, model_path)

        progress(0.6, desc="配置模型...")
        new_gpt = new_gpt.to(tts.device)
        if tts.is_fp16:
            new_gpt.eval().half()
            try:
                import deepspeed
                use_deepspeed = True
            except:
                use_deepspeed = False
            new_gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=True)
        else:
            new_gpt.eval()
            new_gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=False)

        progress(0.9, desc="替換模型...")
        # 替換舊模型
        del tts.gpt
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tts.gpt = new_gpt
        tts.gpt_path = model_path

        progress(1.0, desc="完成！")
        return f"✅ 成功載入模型: {os.path.basename(model_path)}"

    except Exception as e:
        return f"❌ 載入模型失敗: {str(e)}"

available_models = get_available_models()

with open("tests/cases.jsonl", "r", encoding="utf-8") as f:
    example_cases = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        example = json.loads(line)
        example_cases.append([os.path.join("tests", example.get("prompt_audio", "sample_prompt.wav")),
                              example.get("text"), ["普通推理", "批次推理"][example.get("infer_mode", 0)]])

def gen_single(prompt, text, infer_mode, max_text_tokens_per_sentence=120, sentences_bucket_max_size=4,
                *args, progress=gr.Progress()):
    output_path = None
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    # set gradio progress
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
        # "typical_sampling": bool(typical_sampling),
        # "typical_mass": float(typical_mass),
    }
    if infer_mode == "普通推理":
        output = tts.infer(prompt, text, output_path, verbose=cmd_args.verbose,
                           max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
                           **kwargs)
    else:
        # 批次推理
        output = tts.infer_fast(prompt, text, output_path, verbose=cmd_args.verbose,
            max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
            sentences_bucket_max_size=(sentences_bucket_max_size),
            **kwargs)
    return gr.update(value=output,visible=True)

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button

with gr.Blocks(title="IndexTTS Demo") as demo:
    mutex = threading.Lock()
    gr.HTML('''
    <h2><center>IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System</h2>
    <h2><center>(一款工業級可控且高效的零樣本文字轉語音系統)</h2>
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

        model_status = gr.Textbox(label="狀態", interactive=False, value=f"✅ 當前模型: {os.path.basename(tts.gpt_path)}")

        # 綁定載入按鈕
        def on_reload_model(selected_model, progress=gr.Progress()):
            if selected_model not in available_models:
                return "❌ 無效的模型選擇"
            model_path = available_models[selected_model]
            result = reload_gpt_model(model_path, progress)
            return result

        # 重新掃描模型列表
        def on_refresh_models():
            global available_models
            available_models = get_available_models()
            new_choices = list(available_models.keys())
            return gr.update(choices=new_choices, value=new_choices[0] if new_choices else None), \
                   f"✅ 掃描完成，找到 {len(new_choices)} 個模型"

        reload_button.click(
            on_reload_model,
            inputs=[model_dropdown],
            outputs=[model_status]
        )

        refresh_button.click(
            on_refresh_models,
            inputs=[],
            outputs=[model_dropdown, model_status]
        )

    with gr.Tab("音訊生成"):
        with gr.Row():
            os.makedirs("prompts",exist_ok=True)
            prompt_audio = gr.Audio(label="參考音訊",key="prompt_audio",
                                    sources=["upload","microphone"],type="filepath")
            prompt_list = os.listdir("prompts")
            default = ''
            if prompt_list:
                default = prompt_list[0]
            with gr.Column():
                input_text_single = gr.TextArea(label="文字",key="input_text_single", placeholder="請輸入目標文字", info="當前模型版本{}".format(tts.model_version or "1.0"))
                infer_mode = gr.Radio(choices=["普通推理", "批次推理"], label="推理模式",info="批次推理：更適合長句，效能翻倍",value="普通推理")        
                gen_button = gr.Button("生成語音", key="gen_button",interactive=True)
            output_audio = gr.Audio(label="生成結果", visible=True,key="output_audio")
        with gr.Accordion("高階生成引數設定", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**GPT2 取樣設定** _引數會影響音訊多樣性和生成速度詳見[Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)_")
                    with gr.Row():
                        do_sample = gr.Checkbox(label="do_sample", value=True, info="是否進行取樣")
                        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=1.0, step=0.1)
                    with gr.Row():
                        top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                        top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                        num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                    with gr.Row():
                        repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                        length_penalty = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                    max_mel_tokens = gr.Slider(label="max_mel_tokens", value=600, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10, info="生成Token最大數量，過小導致音訊被截斷", key="max_mel_tokens")
                    # with gr.Row():
                    #     typical_sampling = gr.Checkbox(label="typical_sampling", value=False, info="不建議使用")
                    #     typical_mass = gr.Slider(label="typical_mass", value=0.9, minimum=0.0, maximum=1.0, step=0.1)
                with gr.Column(scale=2):
                    gr.Markdown("**分句設定** _引數會影響音訊質量和生成速度_")
                    with gr.Row():
                        max_text_tokens_per_sentence = gr.Slider(
                            label="分句最大Token數", value=120, minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2, key="max_text_tokens_per_sentence",
                            info="建議80~200之間，值越大，分句越長；值越小，分句越碎；過小過大都可能導致音訊質量不高",
                        )
                        sentences_bucket_max_size = gr.Slider(
                            label="分句分桶的最大容量（批次推理生效）", value=4, minimum=1, maximum=16, step=1, key="sentences_bucket_max_size",
                            info="建議2-8之間，值越大，一批次推理包含的分句數越多，過大可能導致記憶體溢位",
                        )
                    with gr.Accordion("預覽分句結果", open=True) as sentences_settings:
                        sentences_preview = gr.Dataframe(
                            headers=["序號", "分句內容", "Token數"],
                            key="sentences_preview",
                            wrap=True,
                        )
            advanced_params = [
                do_sample, top_p, top_k, temperature,
                length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                # typical_sampling, typical_mass,
            ]
        
        if len(example_cases) > 0:
            gr.Examples(
                examples=example_cases,
                inputs=[prompt_audio, input_text_single, infer_mode],
            )

    def on_input_text_change(text, max_tokens_per_sentence):
        if text and len(text) > 0:
            text_tokens_list = tts.tokenizer.tokenize(text)

            sentences = tts.tokenizer.split_sentences(text_tokens_list, max_tokens_per_sentence=int(max_tokens_per_sentence))
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
    prompt_audio.upload(update_prompt_audio,
                         inputs=[],
                         outputs=[gen_button])

    gen_button.click(gen_single,
                     inputs=[prompt_audio, input_text_single, infer_mode,
                             max_text_tokens_per_sentence, sentences_bucket_max_size,
                             *advanced_params,
                     ],
                     outputs=[output_audio])


if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port)
