import argparse
import io
import json
import os
import sys
import tempfile
import time
from typing import Optional
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# 將專案根目錄加入 path 以便匯入 indextts
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

from indextts.infer import IndexTTS


# 全域變數
tts: Optional[IndexTTS] = None
model_dir: str = "checkpoints"
config_path: str = "checkpoints/config.yaml"
device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
use_fp16: bool = True

class TTSRequest(BaseModel):
    text: str
    prompt_audio_path: str  # 伺服器上的檔案路徑
    infer_mode: str = "fast"  # 'normal' or 'fast'
    max_text_tokens_per_sentence: int = 120
    sentences_bucket_max_size: int = 4
    do_sample: bool = True
    top_p: float = 0.8
    top_k: int = 30
    temperature: float = 1.0
    length_penalty: float = 0.0
    num_beams: int = 3
    repetition_penalty: float = 10.0
    max_mel_tokens: int = 600
    speaker_id: Optional[str] = None

class ModelReloadRequest(BaseModel):
    model_filename: str

def initialize_tts():
    """初始化或重新載入 TTS 模型"""
    global tts
    if not os.path.exists(model_dir):
        raise RuntimeError(f"模型目錄 {model_dir} 不存在")
    
    print(f">> [系統] 正在初始化 TTS 引擎 (Device: {device}, FP16: {use_fp16})...")
    try:
        tts = IndexTTS(
            model_dir=model_dir,
            cfg_path=config_path,
            device=device,
            is_fp16=use_fp16,
            gpt_path=os.path.abspath("finetune_models/gpt_1211_epoch_9.pth")
        )
        print(">> [系統] TTS 引擎初始化完成")
    except Exception as e:
        print(f">> [錯誤] 初始化失敗: {e}")
        raise e

@asynccontextmanager
async def lifespan(app: FastAPI):
    """伺服器啟動與關閉時的生命週期管理"""
    initialize_tts()
    yield

# 初始化 FastAPI
app = FastAPI(
    title="IndexTTS API",
    description="Index-TTS 的高效能語音合成 API 服務",
    version="1.0.0",
    lifespan=lifespan
)

# 掛載靜態檔案
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.get("/models")
def list_models():
    """列出所有可用的模型檢查點"""
    models = []
    
    # 預設模型
    if os.path.exists(os.path.join(model_dir, "gpt.pth")):
        models.append({"name": "Default (gpt.pth)", "filename": "gpt.pth", "type": "base"})
        
    # 微調模型
    finetune_dir = os.path.join("finetune_models", "checkpoints")
    if os.path.exists(finetune_dir):
        for f in sorted(os.listdir(finetune_dir)):
            if f.endswith(".pth"):  # 只列出 .pth 檔案
                models.append({
                    "name": f"Finetuned - {f}",
                    "filename": os.path.join("finetune_models", "checkpoints", f),
                    "type": "finetune"
                })
    return {"models": models, "current_model": os.path.basename(tts.gpt_path) if tts else "None"}

@app.post("/model/reload")
def reload_model(request: ModelReloadRequest):
    """
    動態切換 GPT 模型權重。
    """
    global tts
    model_path = request.model_filename
    
    # 處理相對路徑
    if not os.path.isabs(model_path):
        # 嘗試在預設目錄尋找
        if os.path.exists(os.path.join(model_dir, model_path)):
            model_path = os.path.join(model_dir, model_path)
        elif os.path.exists(model_path):
            pass # 已經是正確的路徑
        else:
             raise HTTPException(status_code=404, detail=f"模型檔案 {model_path} 不存在")

    try:
        print(f">> [系統] 正在重新載入模型: {model_path}")
        
        # 重新載入邏輯 (參考 webui.py 的 reload_gpt_model)
        from indextts.gpt.model import UnifiedVoice
        from indextts.utils.checkpoint import load_checkpoint
        
        # 建立新模型
        new_gpt = UnifiedVoice(**tts.cfg.gpt)
        load_checkpoint(new_gpt, model_path)
        
        new_gpt = new_gpt.to(tts.device)
        
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
            
        # 替換
        del tts.gpt
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        tts.gpt = new_gpt
        tts.gpt_path = model_path
        
        return {"status": "success", "message": f"已切換至模型: {os.path.basename(model_path)}"}
        
    except Exception as e:
        print(f">> [錯誤] 模型切換失敗: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    prompt_audio: UploadFile = File(None),
    prompt_audio_path: str = Form(None),
    infer_mode: str = Form("normal"),
    speaker_id: str = Form(None),
    # 進階參數
    max_text_tokens_per_sentence: int = Form(120),
    sentences_bucket_max_size: int = Form(4),
    do_sample: bool = Form(True),
    top_p: float = Form(0.8),
    top_k: int = Form(30),
    temperature: float = Form(0.3),
    repetition_penalty: float = Form(10.0),
    length_penalty: float = Form(0.0),
    max_mel_tokens: int = Form(600)
):
    """
    執行語音合成。
    
    支援上傳參考音訊 (prompt_audio) 或指定伺服器上的檔案路徑 (prompt_audio_path)。
    """
    if not tts:
        raise HTTPException(status_code=503, detail="TTS 引擎尚未初始化")

    if not prompt_audio and not prompt_audio_path:
        raise HTTPException(status_code=400, detail="必須提供參考音訊 (上傳檔案或指定路徑)")

    temp_audio_file = None
    try:
        # 處理參考音訊
        target_prompt_path = ""
        if prompt_audio:
            # 儲存上傳的檔案到暫存區
            suffix = os.path.splitext(prompt_audio.filename)[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await prompt_audio.read()
                tmp.write(content)
                target_prompt_path = tmp.name
                temp_audio_file = tmp.name
        else:
            if not os.path.exists(prompt_audio_path):
                raise HTTPException(status_code=404, detail=f"指定的參考音訊路徑 {prompt_audio_path} 不存在")
            target_prompt_path = prompt_audio_path

        # 準備輸出路徑
        output_dir = "outputs/api"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"gen_{int(time.time())}_{os.urandom(2).hex()}.wav"
        output_path = os.path.join(output_dir, output_filename)

        # 執行推理
        print(f">> [API] 收到請求: text='{text[:20]}...', prompt='{target_prompt_path}', mode={infer_mode}")
        
        kwargs = {
            "do_sample": do_sample,
            "top_p": top_p,
            "top_k": top_k if top_k > 0 else None,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "length_penalty": length_penalty,
            "num_beams": 3,
            "max_mel_tokens": max_mel_tokens
        }

        if infer_mode == "fast":
            # 批次推理
            tts.infer_fast(
                audio_prompt=target_prompt_path,
                text=text,
                output_path=output_path,
                max_text_tokens_per_sentence=max_text_tokens_per_sentence,
                sentences_bucket_max_size=sentences_bucket_max_size,
                **kwargs
            )
        else:
            # 普通推理
            tts.infer(
                audio_prompt=target_prompt_path,
                text=text,
                output_path=output_path,
                max_text_tokens_per_sentence=max_text_tokens_per_sentence,
                speaker_id=speaker_id,
                **kwargs
            )

        if not os.path.exists(output_path):
            raise RuntimeError("音訊生成失敗，輸出檔案未建立")

        # 回傳音訊串流
        def iterfile():
            with open(output_path, mode="rb") as file_like:
                yield from file_like
            # 傳送後刪除 (可選，這裡保留以便除錯，或使用背景任務刪除)
            # os.remove(output_path)

        return StreamingResponse(iterfile(), media_type="audio/wav", headers={"Content-Disposition": f"attachment; filename={output_filename}"})

    except Exception as e:
        print(f">> [API 錯誤] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理上傳的暫存檔
        if temp_audio_file and os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IndexTTS API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="監聽位址")
    parser.add_argument("--port", type=int, default=7859, help="監聽埠號")
    parser.add_argument("--model_dir", type=str, default="checkpoints", help="模型目錄")
    parser.add_argument("--config", type=str, default="checkpoints/config.yaml", help="設定檔路徑")
    parser.add_argument("--device", type=str, default=None, help="指定裝置 (例如 cuda:0)")
    parser.add_argument("--no-fp16", action="store_true", help="禁用 FP16")
    
    args = parser.parse_args()
    
    model_dir = args.model_dir
    config_path = args.config
    if args.device:
        device = args.device
    use_fp16 = not args.no_fp16

    print(f"啟動 API Server 於 {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
