#!/usr/bin/env python
"""检查Qwen3模型下载状态"""
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = 'models/models--Qwen--Qwen3-8B'
snapshots_dir = os.path.join(model_path, 'snapshots')

print("="*60)
print("Qwen3 模型检查")
print("="*60)

# 检查文件
if os.path.exists(snapshots_dir):
    snap_dirs = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
    if snap_dirs:
        actual_model_path = os.path.join(snapshots_dir, snap_dirs[0])
        print(f"✓ 找到模型目录: {actual_model_path}")
        
        # 检查文件
        files = os.listdir(actual_model_path)
        model_files = [f for f in files if f.endswith(('.safetensors', '.bin'))]
        config_files = [f for f in files if 'config' in f.lower()]
        
        print(f"  - 模型文件: {len(model_files)}个")
        print(f"  - 配置文件: {len(config_files)}个")
        
        # 尝试加载tokenizer
        try:
            print("\n测试加载tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(actual_model_path, trust_remote_code=True)
            print("✓ Tokenizer加载成功")
            print(f"  - Vocab大小: {len(tokenizer)}")
        except Exception as e:
            print(f"❌ Tokenizer加载失败: {e}")
        
        # 检查模型文件大小
        total_size = 0
        for f in model_files:
            fp = os.path.join(actual_model_path, f)
            size = os.path.getsize(fp) / (1024**3)
            total_size += size
            print(f"  - {f}: {size:.2f} GB")
        
        print(f"\n总大小: {total_size:.2f} GB")
        
        if total_size > 10:
            print("✓ 模型文件完整")
        else:
            print("⚠️ 模型文件可能不完整（小于10GB）")
    else:
        print("❌ 未找到快照目录")
else:
    print("❌ 模型目录不存在")

print("="*60)

