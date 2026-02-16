#!/usr/bin/env python3
"""
使用 Qwen3-Embedding-4B 构建 POI 向量
直接使用 transformers，不依赖 FlagEmbedding
"""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'

import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import time
import pickle

# 添加项目路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from utils.id_mapping import normalize_poi_id


def get_cache_file_path(poi_csv, output_dir):
    """生成缓存文件路径"""
    import hashlib
    csv_hash = hashlib.md5(str(poi_csv).encode()).hexdigest()[:8]
    cache_dir = Path(output_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"poi_cache_qwen3_{csv_hash}.pkl"


def build_poi_texts(df: pd.DataFrame) -> list:
    """构建POI文本描述"""
    texts = []
    for _, row in df.iterrows():
        parts = [str(row['name'])]

        if pd.notna(row.get('province')):
            parts.append(str(row['province']))
        if pd.notna(row.get('city')) and row.get('city') != row.get('province'):
            parts.append(str(row['city']))

        if pd.notna(row.get('description')) and row.get('description'):
            desc = str(row['description']).replace('\n', ' ')[:200]
            parts.append(desc)

        stay_hours = row['stay_min'] / 60
        parts.append(f"建议停留{stay_hours:.1f}小时")

        texts.append(" ".join(parts))
    return texts


def main():
    """主函数"""
    # 配置
    poi_csv = 'data/all/poi_expanded.csv'
    output_dir = 'outputs/emb'
    batch_size = 64
    use_gpu = True

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 输出文件
    output_emb_file = output_dir / "poi_emb.npy"
    output_meta_file = output_dir / "poi_meta.csv"
    cache_file = get_cache_file_path(poi_csv, output_dir)

    print("=" * 60)
    print("POI 向量构建工具 (Qwen3-Embedding-4B)")
    print("=" * 60)
    print()

    # 检查GPU
    if use_gpu and torch.cuda.is_available():
        device = "cuda:0"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        device = "cpu"
        batch_size = 32
        print("⚠️ 使用CPU")

    print()

    # 加载模型
    print("加载 Qwen3-Embedding-4B 模型...")
    from transformers import AutoModel, AutoTokenizer

    model_path = "models/Qwen3-Embedding-4B"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(device)
    model.eval()

    print(f"✓ 模型加载完成 (设备: {device})")
    print()

    # 读取POI数据
    print(f"读取POI数据: {poi_csv}")
    df = pd.read_csv(poi_csv)
    if 'poi_id' in df.columns:
        df['poi_id'] = df['poi_id'].apply(normalize_poi_id)
    total_pois = len(df)
    print(f"✓ 加载 {total_pois} 个 POI")
    print()

    # 检查缓存
    cache_data = None
    if cache_file.exists():
        print(f"✓ 发现断点缓存: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            cached_count = cache_data.get('count', 0)
            print(f"  已编码 {cached_count} 个POI")
        except Exception as e:
            print(f"  ⚠️ 缓存损坏: {e}")
            cache_data = None

    # 构建文本
    print("构建文本描述...")
    texts = build_poi_texts(df)
    print(f"✓ 文本样例:")
    for i in range(min(2, len(texts))):
        print(f"  [{i}] {texts[i][:80]}...")
    print()

    # 获取embedding维度
    with torch.no_grad():
        dummy_input = tokenizer("测试", return_tensors="pt")
        dummy_input = {k: v.to(device) for k, v in dummy_input.items()}
        outputs = model(**dummy_input)
        emb_dim = outputs.last_hidden_state.shape[-1]

    print(f"Embedding维度: {emb_dim}")
    print()

    # 编码函数
    def encode_batch(texts_batch):
        """编码一个批次"""
        inputs = tokenizer(
            texts_batch,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            model_output = model(**inputs)
            # 使用平均池化
            embeddings = model_output.last_hidden_state.mean(dim=1)
            # 转换为float32并归一化
            embeddings = embeddings.float()
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    # 初始化或恢复向量
    if cache_data and 'embeddings' in cache_data:
        embeddings = cache_data['embeddings']
        start_idx = cache_data['count']
        print(f"从第 {start_idx} 个POI继续...")
    else:
        # 编码第一个batch确定维度
        print("初始化向量数组...")
        first_batch = encode_batch(texts[:min(batch_size, total_pois)])
        embeddings = np.zeros((total_pois, emb_dim), dtype=np.float32)
        embeddings[:len(first_batch)] = first_batch
        start_idx = len(first_batch)

    # 批量编码
    remaining = total_pois - start_idx
    num_batches = (remaining + batch_size - 1) // batch_size

    print(f"\n开始编码 {total_pois} 个POI (batch_size={batch_size})...\n")

    start_time = time.time()

    with tqdm(total=remaining, desc="编码进度", unit="POI") as pbar:
        for batch_idx in range(num_batches):
            batch_start = start_idx + batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_pois)
            batch_texts = texts[batch_start:batch_end]

            # 编码
            batch_embeddings = encode_batch(batch_texts)
            embeddings[batch_start:batch_end] = batch_embeddings

            # 更新进度
            pbar.update(len(batch_texts))

            # 定期保存缓存
            if (batch_end) % 500 == 0 or batch_end == total_pois:
                cache_data = {
                    'embeddings': embeddings[:batch_end],
                    'count': batch_end,
                    'model_name': 'qwen3',
                    'poi_csv': str(poi_csv)
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                pbar.set_postfix({'cached': batch_end})

            # 清理GPU缓存
            if (batch_end) % 200 == 0:
                torch.cuda.empty_cache()

    elapsed = time.time() - start_time

    print()
    print(f"✓ 编码完成")
    print(f"  向量维度: {embeddings.shape}")
    print(f"  耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
    print(f"  速度: {total_pois/elapsed:.1f} POI/秒")

    # 保存最终结果
    print()
    print("保存文件...")
    np.save(output_emb_file, embeddings)
    df.to_csv(output_meta_file, index=False)

    print(f"✓ 向量: {output_emb_file}")
    print(f"✓ 元数据: {output_meta_file}")

    # 清理缓存
    if cache_file.exists():
        cache_file.unlink()
        print(f"✓ 已清理缓存")

    print()
    print("=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
