#!/usr/bin/env python3
"""
簡單的 SentencePiece tokenizer 測試指令碼
"""

import argparse
import os

import sentencepiece as smp


def main():
    parser = argparse.ArgumentParser(description="測試 SentencePiece tokenizer")
    parser.add_argument(
        "--bpe-model",
        type=str,
        default="finetune_models/bpe.model",
        help="BPE 模型檔案路徑"
    )
    
    args = parser.parse_args()
    
    # 載入模型
    if not os.path.exists(args.bpe_model):
        print(f"❌ BPE 模型檔案不存在: {args.bpe_model}")
        return
    
    bpe_model = smp.SentencePieceProcessor()
    bpe_model.Load(args.bpe_model)
    
    print(f"✅ 已載入 BPE 模型: {args.bpe_model}")
    print(f"📊 詞彙表大小: {bpe_model.GetPieceSize()}")

    # 輸出詞彙表
    #for i in range(bpe_model.GetPieceSize()):
    #    print(f"{i}: {bpe_model.IdToPiece(i)}")

    text = "HELLO"
    tokens = bpe_model.Encode(text, out_type=int)
    print(f"📝 輸入文字: '{text}'")
    print(f"🔢 Token IDs: {tokens}")
    for i in tokens:
        print(f"{i}: {bpe_model.IdToPiece(i)}")


if __name__ == "__main__":
    main()
