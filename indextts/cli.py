import os
import sys
import warnings
# Suppress warnings from tensorflow and other libraries
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
def main():
    import argparse
    parser = argparse.ArgumentParser(description="IndexTTS 命令列工具")
    parser.add_argument("text", type=str, help="要合成的文字")
    parser.add_argument("-v", "--voice", type=str, required=True, help="參考音訊檔案路徑 (wav 格式)")
    parser.add_argument("-o", "--output_path", type=str, default="gen.wav", help="輸出音訊檔案路徑")
    parser.add_argument("-c", "--config", type=str, default="checkpoints/config.yaml", help="設定檔路徑。預設為 'checkpoints/config.yaml'")
    parser.add_argument("--model_dir", type=str, default="checkpoints", help="模型目錄路徑。預設為 'checkpoints'")
    parser.add_argument("--fp16", action="store_true", default=True, help="若可用則使用 FP16 推理")
    parser.add_argument("-f", "--force", action="store_true", default=False, help="強制覆蓋已存在的輸出檔案")
    parser.add_argument("-d", "--device", type=str, default=None, help="執行模型的裝置 (cpu, cuda, mps)。" )
    args = parser.parse_args()
    if len(args.text.strip()) == 0:
        print("[錯誤] 文字內容為空。")
        parser.print_help()
        sys.exit(1)
    if not os.path.exists(args.voice):
        print(f"[錯誤] 參考音訊檔案 {args.voice} 不存在。")
        parser.print_help()
        sys.exit(1)
    if not os.path.exists(args.config):
        print(f"[錯誤] 設定檔 {args.config} 不存在。")
        parser.print_help()
        sys.exit(1)

    output_path = args.output_path
    if os.path.exists(output_path):
        if not args.force:
            print(f"[錯誤] 輸出檔案 {output_path} 已存在。請使用 --force 強制覆蓋。")
            parser.print_help()
            sys.exit(1)
        else:
            os.remove(output_path)
    
    try:
        import torch
    except ImportError:
        print("[錯誤] 未安裝 PyTorch。請先安裝。")
        sys.exit(1)

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda:0"
        elif torch.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
            args.fp16 = False # Disable FP16 on CPU
            print("[警告] 使用 CPU 運行可能較慢。")

    from indextts.infer import IndexTTS
    tts = IndexTTS(cfg_path=args.config, model_dir=args.model_dir, is_fp16=args.fp16, device=args.device)
    tts.infer(audio_prompt=args.voice, text=args.text.strip(), output_path=output_path)

if __name__ == "__main__":
    main()