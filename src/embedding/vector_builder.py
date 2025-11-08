"""
POI向量构建器
基于BGE-M3生成和管理POI向量
"""
import os
import sys
import pandas as pd
import numpy as np

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding.bge_m3_encoder import BGEM3Encoder

def build_poi_embeddings(poi_csv='data/poi.csv', 
                        output_dir='outputs/emb',
                        model_path='/root/autodl-tmp/goafar_project/models/Xorbits/bge-m3',
                        use_gpu=False):
    """
    构建POI嵌入向量
    
    Args:
        poi_csv: POI数据文件路径
        output_dir: 输出目录
        model_path: 模型路径
        use_gpu: 是否使用GPU
    
    Returns:
        tuple: (embeddings_dict, metadata_df)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取POI数据
    df = pd.read_csv(poi_csv)
    print(f"✓ 加载 {len(df)} 个 POI")
    
    # 构建文本表示
    texts = []
    for _, row in df.iterrows():
        parts = [str(row['name'])]
        
        # 添加地理信息
        if pd.notna(row.get('province')):
            parts.append(str(row['province']))
        if pd.notna(row.get('city')) and row.get('city') != row.get('province'):
            parts.append(str(row['city']))
        
        # 添加描述
        if pd.notna(row.get('description')) and row.get('description'):
            desc = str(row['description']).replace('\n', ' ')[:200]
            parts.append(desc)
        
        # 添加停留时长
        stay_hours = row['stay_min'] / 60
        parts.append(f"建议停留{stay_hours:.1f}小时")
        
        texts.append(" ".join(parts))
    
    print(f"\n文本样例:")
    for i in range(min(3, len(texts))):
        print(f"  [{i}] {texts[i][:150]}...")
    
    # 初始化编码器
    print(f"\n初始化 BGE-M3 编码器...")
    encoder = BGEM3Encoder(
        model_path=model_path,
        use_gpu=use_gpu
    )
    
    # 生成嵌入向量
    print(f"\n生成 {len(texts)} 个向量...")
    embeddings_dict = encoder.encode_texts(
        texts,
        batch_size=64,
        max_length=512,
        return_dense=True,
        return_sparse=False,  # 暂时只用dense
        return_colbert=False
    )
    
    # 提取dense向量并保存
    dense_vecs = embeddings_dict['dense_vecs']
    print(f"✓ 向量维度: {dense_vecs.shape}")
    
    # 保存
    np.save(f"{output_dir}/poi_emb.npy", dense_vecs)
    df.to_csv(f"{output_dir}/poi_meta.csv", index=False)
    
    print(f"\n✓ 保存完成:")
    print(f"  - 向量文件: {output_dir}/poi_emb.npy")
    print(f"  - 元数据文件: {output_dir}/poi_meta.csv")
    
    return embeddings_dict, df

def search_similar_pois(query_text, topk=50,
                       emb_file='outputs/emb/poi_emb.npy',
                       meta_file='outputs/emb/poi_meta.csv',
                       model_path='/root/autodl-tmp/goafar_project/models/Xorbits/bge-m3',
                       use_gpu=False):
    """
    搜索与查询相似的POI
    
    Args:
        query_text: 查询文本
        topk: 返回Top-K个结果
        emb_file: 向量文件
        meta_file: 元数据文件
        model_path: 模型路径
        use_gpu: 是否使用GPU
    
    Returns:
        DataFrame: Top-K相似POI及其分数
    """
    # 加载向量和元数据
    embeddings = np.load(emb_file)
    metadata = pd.read_csv(meta_file)
    
    print(f"\n=== 语义检索 ===")
    print(f"查询: '{query_text}'")
    
    # 初始化编码器
    encoder = BGEM3Encoder(model_path=model_path, use_gpu=use_gpu)
    
    # 编码查询
    query_emb = encoder.encode_query(query_text, return_dense=True)
    query_vec = query_emb['dense_vec']
    
    # 计算相似度（余弦相似度，向量已归一化）
    scores = embeddings @ query_vec
    
    # 排序并获取Top-K
    top_indices = np.argsort(-scores)[:topk]
    
    # 构建结果
    results = metadata.iloc[top_indices].copy()
    results['semantic_score'] = scores[top_indices]
    results['rank'] = range(1, len(results) + 1)
    
    print(f"✓ 检索到 {len(results)} 个相似POI")
    print(f"\nTop 10 结果:")
    for i, (_, row) in enumerate(results.head(10).iterrows(), 1):
        print(f"  {i}. {row['name']:<25} | {row['city']:<12} | 分数:{row['semantic_score']:.4f}")
    
    return results

if __name__ == "__main__":
    # 测试
    print("="*60)
    print("测试 POI 向量构建")
    print("="*60)
    
    embeddings_dict, df = build_poi_embeddings(use_gpu=False)
    
    print("\n" + "="*60)
    print("测试语义检索")
    print("="*60)
    
    results = search_similar_pois(
        "想去新疆看雪山和湖泊",
        topk=10,
        use_gpu=False
    )

