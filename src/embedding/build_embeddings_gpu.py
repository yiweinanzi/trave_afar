"""
GPU加速的POI向量构建
使用GPU大幅提升BGE-M3编码速度
"""
import os
import sys
import pandas as pd
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding.bge_m3_encoder import BGEM3Encoder

def build_embeddings_with_gpu(poi_csv='data/poi.csv',
                              output_dir='outputs/emb',
                              batch_size=128,  # GPU可以用更大的batch
                              use_cache=True):
    """
    使用GPU构建POI向量（支持缓存）
    
    Args:
        poi_csv: POI数据文件
        output_dir: 输出目录
        batch_size: 批处理大小（GPU可用更大值）
        use_cache: 是否使用缓存
    
    Returns:
        tuple: (embeddings, metadata)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_emb_file = f"{output_dir}/poi_emb.npy"
    output_meta_file = f"{output_dir}/poi_meta.csv"
    
    # 检查缓存
    if use_cache and os.path.exists(output_emb_file) and os.path.exists(output_meta_file):
        print(f"✓ 发现缓存文件，直接加载")
        embeddings = np.load(output_emb_file)
        metadata = pd.read_csv(output_meta_file)
        print(f"  向量维度: {embeddings.shape}")
        print(f"  POI数量: {len(metadata)}")
        return embeddings, metadata
    
    # 读取POI数据
    df = pd.read_csv(poi_csv)
    print(f"✓ 加载 {len(df)} 个 POI")
    
    # 检查GPU
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n✓ 检测到GPU: {gpu_name}")
        print(f"  显存: {gpu_memory:.1f} GB")
    else:
        print("\n⚠️ 未检测到GPU，将使用CPU（较慢）")
        batch_size = 32  # CPU用小batch
    
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
    print(f"\n初始化 BGE-M3 编码器（GPU={use_gpu}）...")
    encoder = BGEM3Encoder(
        model_path='/root/autodl-tmp/goafar_project/models/Xorbits/bge-m3',
        use_gpu=use_gpu
    )
    
    # 生成向量
    print(f"\n生成 {len(texts)} 个向量（batch_size={batch_size}）...")
    
    import time
    start_time = time.time()
    
    embeddings_dict = encoder.encode_texts(
        texts,
        batch_size=batch_size,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert=False
    )
    
    elapsed = time.time() - start_time
    
    dense_vecs = embeddings_dict['dense_vecs']
    print(f"✓ 向量维度: {dense_vecs.shape}")
    print(f"  耗时: {elapsed:.2f}秒")
    print(f"  速度: {len(texts)/elapsed:.1f} POI/秒")
    
    # 保存
    np.save(output_emb_file, dense_vecs)
    df.to_csv(output_meta_file, index=False)
    
    print(f"\n✓ 保存完成:")
    print(f"  - 向量: {output_emb_file}")
    print(f"  - 元数据: {output_meta_file}")
    
    return dense_vecs, df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='GPU加速的POI向量构建')
    parser.add_argument('--batch-size', type=int, default=128, help='批处理大小')
    parser.add_argument('--no-cache', action='store_true', help='不使用缓存')
    args = parser.parse_args()
    
    embeddings, metadata = build_embeddings_with_gpu(
        batch_size=args.batch_size,
        use_cache=not args.no_cache
    )
    
    print("\n✓ 向量构建完成！")
    print(f"\n下一步: 测试语义检索")
    print(f"  python -c \"from src.embedding.vector_builder import search_similar_pois; search_similar_pois('想去喀纳斯', topk=10, use_gpu=True)\"")

