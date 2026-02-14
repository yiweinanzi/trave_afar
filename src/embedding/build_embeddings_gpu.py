"""
GPU加速的POI向量构建
支持 BGE-M3 和 Qwen3-Embedding-4B 模型
支持断点续传和进度显示
"""
import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import time
import hashlib
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding.bge_m3_encoder import BGEM3Encoder
from embedding.qwen3_encoder import Qwen3Embedding
from utils.id_mapping import normalize_poi_id

def get_cache_file_path(poi_csv, output_dir, model_type):
    """生成缓存文件路径"""
    # 使用文件路径的hash作为缓存标识
    csv_hash = hashlib.md5(str(poi_csv).encode()).hexdigest()[:8]
    cache_dir = Path(output_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"poi_cache_{model_type}_{csv_hash}.pkl"


def build_embeddings_with_gpu(
    poi_csv='data/all/poi_expanded.csv',
    output_dir='outputs/emb',
    model_path=None,
    model_type='qwen3',  # 'bge_m3' or 'qwen3'
    batch_size=128,
    use_cache=True,
    use_gpu=True
):
    """
    使用GPU构建POI向量（支持缓存和断点续传）

    Args:
        poi_csv: POI数据文件
        output_dir: 输出目录
        model_path: 模型路径（如果为None，使用默认路径）
        model_type: 模型类型 ('bge_m3' or 'qwen3')
        batch_size: 批处理大小（GPU可用更大值）
        use_cache: 是否使用缓存
        use_gpu: 是否使用GPU

    Returns:
        tuple: (embeddings, metadata)
    """
    os.makedirs(output_dir, exist_ok=True)

    # 输出文件路径
    model_suffix = model_type.replace('_', '')
    output_emb_file = f"{output_dir}/poi_emb_{model_suffix}.npy"
    output_meta_file = f"{output_dir}/poi_meta_{model_suffix}.csv"
    cache_file = get_cache_file_path(poi_csv, output_dir, model_type)

    # 检查是否有完全完成的缓存
    if use_cache and os.path.exists(output_emb_file) and os.path.exists(output_meta_file):
        print(f"✓ 发现已完成的向量文件，直接加载")
        embeddings = np.load(output_emb_file)
        metadata = pd.read_csv(output_meta_file)
        print(f"  向量维度: {embeddings.shape}")
        print(f"  POI数量: {len(metadata)}")
        return embeddings, metadata
    
    # 读取POI数据
    df = pd.read_csv(poi_csv)
    if 'poi_id' in df.columns:
        df['poi_id'] = df['poi_id'].apply(normalize_poi_id)
    total_pois = len(df)
    print(f"✓ 加载 {total_pois} 个 POI")

    # 检查断点续传缓存
    cache_data = None
    if use_cache and os.path.exists(cache_file):
        print(f"✓ 发现断点缓存文件: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            cached_count = cache_data.get('count', 0)
            print(f"  已编码 {cached_count} 个POI，将继续处理")
            if cached_count >= total_pois:
                print("  缓存已完成，但输出文件不存在，正在恢复...")
        except Exception as e:
            print(f"  ⚠️ 缓存文件损坏，将重新生成: {e}")
            cache_data = None
    
    # 检查GPU
    actual_use_gpu = use_gpu and torch.cuda.is_available()
    if actual_use_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n✓ 检测到GPU: {gpu_name}")
        print(f"  显存: {gpu_memory:.1f} GB")
    else:
        print("\n⚠️ 未检测到GPU或未启用，将使用CPU（较慢）")
        batch_size = min(batch_size, 32)  # CPU用小batch
    
    # 构建文本
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
    
    print(f"\n文本样例:")
    for i in range(min(3, len(texts))):
        print(f"  [{i}] {texts[i][:100]}...")

    # 初始化编码器
    print(f"\n初始化 {model_type.upper()} 编码器（GPU={actual_use_gpu}）...")

    # 确定模型路径
    if model_path is None:
        if model_type == 'bge_m3':
            model_env = os.getenv("GOAFAR_BGE_MODEL", "models/Xorbits/bge-m3")
        else:  # qwen3
            model_env = os.getenv("GOAFAR_QWEN3_MODEL", "models/Qwen3-Embedding-4B")
        model_path = Path(model_env)
        if not model_path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            model_path = project_root / model_path

    # 创建编码器
    if model_type == 'bge_m3':
        encoder = BGEM3Encoder(model_path=str(model_path), use_gpu=actual_use_gpu)
    else:  # qwen3
        encoder = Qwen3Embedding(model_path=str(model_path), use_gpu=actual_use_gpu)

    # 生成向量
    print(f"\n生成 {total_pois} 个向量（batch_size={batch_size}）...")

    start_time = time.time()

    # 断点续传逻辑
    if cache_data and 'embeddings' in cache_data and 'count' in cache_data:
        # 从缓存恢复
        dense_vecs = cache_data['embeddings']
        start_idx = cache_data['count']
        print(f"  从第 {start_idx} 个POI继续...")
    else:
        # 全部重新生成
        dense_vecs = None
        start_idx = 0

    # 批量编码
    if start_idx < total_pois:
        # 初始化向量数组（从第一次编码推断维度）
        if start_idx == 0:
            # 先编码一个batch来确定维度
            test_batch = encoder.encode_texts(
                texts[:min(batch_size, total_pois)],
                batch_size=batch_size,
                return_dense=True,
                return_sparse=False,
                return_colbert=False,
                show_progress=False
            )
            emb_dim = test_batch['dense_vecs'].shape[1]
            dense_vecs = np.zeros((total_pois, emb_dim), dtype=np.float32)
            dense_vecs[:len(test_batch['dense_vecs'])] = test_batch['dense_vecs']
            start_idx = len(test_batch['dense_vecs'])

        # 分批处理剩余数据
        remaining = total_pois - start_idx
        num_batches = (remaining + batch_size - 1) // batch_size

        with tqdm(total=remaining, desc="编码进度", unit="POI") as pbar:
            for batch_idx in range(num_batches):
                batch_start = start_idx + batch_idx * batch_size
                batch_end = min(batch_start + batch_size, total_pois)
                batch_texts = texts[batch_start:batch_end]

                # 编码当前批次
                batch_embeddings = encoder.encode_texts(
                    batch_texts,
                    batch_size=batch_size,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert=False,
                    show_progress=False
                )

                # 保存到数组
                dense_vecs[batch_start:batch_end] = batch_embeddings['dense_vecs']

                # 更新进度
                pbar.update(len(batch_texts))

                # 定期保存缓存（每1000个）
                if (batch_end) % 1000 == 0 or batch_end == total_pois:
                    cache_data = {
                        'embeddings': dense_vecs[:batch_end],
                        'count': batch_end,
                        'model_type': model_type,
                        'poi_csv': str(poi_csv)
                    }
                    with open(cache_file, 'wb') as f:
                        pickle.dump(cache_data, f)
                    pbar.set_postfix({'cached': batch_end})

    elapsed = time.time() - start_time

    print(f"\n✓ 向量维度: {dense_vecs.shape}")
    print(f"  耗时: {elapsed:.2f}秒 ({elapsed/60:.1f}分钟)")
    print(f"  速度: {total_pois/elapsed:.1f} POI/秒")
    
    # 保存最终结果
    np.save(output_emb_file, dense_vecs)
    df.to_csv(output_meta_file, index=False)

    print(f"\n✓ 保存完成:")
    print(f"  - 向量: {output_emb_file}")
    print(f"  - 元数据: {output_meta_file}")

    # 清理缓存文件
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"  ✓ 已清理临时缓存: {cache_file}")

    return dense_vecs, df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='GPU加速的POI向量构建')
    parser.add_argument('--poi-csv', type=str, default='data/all/poi_expanded.csv',
                        help='POI数据文件路径')
    parser.add_argument('--output-dir', type=str, default='outputs/emb',
                        help='输出目录')
    parser.add_argument('--model-path', type=str, default=None,
                        help='模型路径（默认使用环境变量或默认路径）')
    parser.add_argument('--model-type', type=str, default='qwen3',
                        choices=['bge_m3', 'qwen3'],
                        help='模型类型')
    parser.add_argument('--batch-size', type=int, default=128, help='批处理大小')
    parser.add_argument('--no-cache', action='store_true', help='不使用缓存')
    parser.add_argument('--no-gpu', action='store_true', help='不使用GPU')
    args = parser.parse_args()

    embeddings, metadata = build_embeddings_with_gpu(
        poi_csv=args.poi_csv,
        output_dir=args.output_dir,
        model_path=args.model_path,
        model_type=args.model_type,
        batch_size=args.batch_size,
        use_cache=not args.no_cache,
        use_gpu=not args.no_gpu
    )

    print("\n✓ 向量构建完成！")
    print(f"\n最终统计:")
    print(f"  POI数量: {len(metadata)}")
    print(f"  向量维度: {embeddings.shape}")
    print(f"  输出文件: {args.output_dir}/poi_emb_{args.model_type.replace('_', '')}.npy")

    print(f"\n下一步: 测试语义检索")
    print(f"  python -c \"from src.embedding.vector_builder import search_similar_pois; search_similar_pois('想去喀纳斯', topk=10, use_gpu={not args.no_gpu})\"")
