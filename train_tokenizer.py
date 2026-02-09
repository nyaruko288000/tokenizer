#!/usr/bin/env python3
"""
Train a HuggingFace-compatible BPE tokenizer with 256 byte tokens.
"""

import argparse
import os
import json
import time
from pathlib import Path

from tokenizers import Tokenizer, pre_tokenizers, processors, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


def create_byte_level_bpe_tokenizer(
    vocab_size: int = 8192,
    special_tokens: list = None
) -> tuple[Tokenizer, BpeTrainer]:
    """
    创建一个包含256字节码的BPE分词器。
    """
    if special_tokens is None:
        # 4个特殊token
        special_tokens = ["<|pad|>", "<|eos|>", "<|bos|>", "<|unk|>"]
    
    # 初始化 BPE 模型
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    
    # ByteLevel pre-tokenizer：自动处理256个字节码
    # add_prefix_space=False: 不在开头添加空格
    # use_regex=True: 使用GPT-2风格的正则分割
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        ByteLevel(add_prefix_space=False, use_regex=True)
    ])
    
    # ByteLevel decoder
    tokenizer.decoder = decoders.ByteLevel()
    
    # 计算实际需要训练的词表大小
    # 256 bytes + 4 special tokens = 260 已占用
    # 剩余 vocab_size - 260 用于 BPE merges
    num_byte_tokens = 256
    num_special = len(special_tokens)
    
    print(f"配置:")
    print(f"  - 目标词表大小: {vocab_size}")
    print(f"  - 字节码数量: {num_byte_tokens}")
    print(f"  - 特殊token数量: {num_special}")
    print(f"  - BPE合并产生的token: {vocab_size - num_byte_tokens - num_special}")
    
    # 创建训练器
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),  # 256 bytes
        min_frequency=2,  # 至少出现2次才合并
    )
    
    return tokenizer, trainer


def train_tokenizer(
    corpus_path: str,
    output_dir: str,
    vocab_size: int = 8192,
    special_tokens: list = None
):
    """
    训练分词器并保存为HuggingFace格式。
    """
    if special_tokens is None:
        special_tokens = ["<|pad|>", "<|eos|>", "<|bos|>", "<|unk|>"]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"开始训练分词器")
    print(f"{'='*60}")
    print(f"语料文件: {corpus_path}")
    print(f"输出目录: {output_dir}")
    print(f"词表大小: {vocab_size}")
    print(f"特殊tokens: {special_tokens}")
    
    # 获取语料文件大小
    corpus_size = os.path.getsize(corpus_path)
    print(f"语料大小: {corpus_size / 1024 / 1024:.2f} MB")
    
    # 创建分词器和训练器
    tokenizer, trainer = create_byte_level_bpe_tokenizer(
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )
    
    # 训练
    print(f"\n开始训练...")
    start_time = time.time()
    
    tokenizer.train(files=[corpus_path], trainer=trainer)
    
    elapsed = time.time() - start_time
    print(f"训练完成! 耗时: {elapsed:.2f} 秒")
    
    # 添加后处理器 (可选：添加 bos/eos)
    tokenizer.post_processor = TemplateProcessing(
        single="<|bos|> $A <|eos|>",
        pair="<|bos|> $A <|eos|> <|bos|> $B <|eos|>",
        special_tokens=[
            ("<|bos|>", tokenizer.token_to_id("<|bos|>")),
            ("<|eos|>", tokenizer.token_to_id("<|eos|>")),
        ],
    )
    
    # 保存底层 tokenizer.json
    tokenizer_json_path = output_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_json_path))
    print(f"已保存: {tokenizer_json_path}")
    
    # 包装为 HuggingFace PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        clean_up_tokenization_spaces=False,
    )
    
    # 保存为完整的 HuggingFace 格式
    hf_tokenizer.save_pretrained(str(output_path))
    print(f"已保存 HuggingFace 格式到: {output_path}")
    
    # 验证
    print(f"\n{'='*60}")
    print("验证分词器")
    print(f"{'='*60}")
    
    print(f"词表大小: {hf_tokenizer.vocab_size}")
    print(f"特殊tokens: {hf_tokenizer.special_tokens_map}")
    
    # 测试编解码
    test_texts = [
        "Hello, world!",
        "你好，世界！",
        "🎉 Emoji test 🚀",
        "Binary: \x00\x01\x02\xff",
        "Mixed: Hello世界🌍",
    ]
    
    print("\n编解码测试:")
    for text in test_texts:
        encoded = hf_tokenizer.encode(text)
        decoded = hf_tokenizer.decode(encoded)
        print(f"  原文: {repr(text)}")
        print(f"  编码: {encoded[:20]}{'...' if len(encoded) > 20 else ''}")
        print(f"  解码: {repr(decoded)}")
        print(f"  token数: {len(encoded)}")
        print()
    
    # 保存元信息
    meta_info = {
        "vocab_size": hf_tokenizer.vocab_size,
        "special_tokens": special_tokens,
        "byte_tokens": 256,
        "corpus_size_mb": corpus_size / 1024 / 1024,
        "training_time_seconds": elapsed,
        "model_type": "BPE",
        "byte_level": True,
    }
    
    meta_path = output_path / "training_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2, ensure_ascii=False)
    print(f"已保存训练元信息: {meta_path}")
    
    # 列出所有输出文件
    print(f"\n输出文件:")
    for f in output_path.iterdir():
        size = f.stat().st_size
        print(f"  {f.name}: {size:,} bytes")
    
    return hf_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train a HuggingFace-compatible tokenizer")
    parser.add_argument("--corpus", type=str, required=True, help="Path to corpus file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=8192, help="Vocabulary size")
    parser.add_argument("--special-tokens", type=str, nargs="+", 
                        default=["<|pad|>", "<|eos|>", "<|bos|>", "<|unk|>"],
                        help="Special tokens")
    
    args = parser.parse_args()
    
    train_tokenizer(
        corpus_path=args.corpus,
        output_dir=args.output,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
    )


if __name__ == "__main__":
    main()
