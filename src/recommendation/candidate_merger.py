"""
候选池合并
融合语义检索和序列推荐的结果
"""
import pandas as pd
import numpy as np
import os
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding.vector_builder import search_similar_pois

def merge_candidates(query_text, user_id=None,
                    topk_dense=50, topk_seq=30,
                    province_filter=None):
    """
    合并多路召回的候选POI
    
    Args:
        query_text: 用户查询
        user_id: 用户ID（用于序列推荐）
        topk_dense: 语义检索Top-K
        topk_seq: 序列推荐Top-K
        province_filter: 省份过滤
    
    Returns:
        DataFrame: 合并后的候选POI
    """
    print("\n" + "="*60)
    print("候选池合并")
    print("="*60)
    
    # 1. 语义检索召回
    print(f"\n【路径1】语义检索召回")
    dense_results = search_similar_pois(
        query_text,
        topk=topk_dense,
        use_gpu=False
    )
    
    # 2. 序列推荐召回（简化版：基于流行度）
    print(f"\n【路径2】序列推荐召回")
    seq_results = _get_popular_pois(topk=topk_seq)
    
    # 3. 合并候选池
    print(f"\n【合并】候选池去重")
    
    # 标记来源
    dense_results['from_dense'] = True
    seq_results['from_seq'] = True
    
    # 合并（基于poi_id去重）
    merged = pd.merge(
        dense_results,
        seq_results[['poi_id', 'popularity_score', 'from_seq']],
        on='poi_id',
        how='outer'
    )
    
    # 填充缺失值
    merged['from_dense'] = merged['from_dense'].fillna(False)
    merged['from_seq'] = merged['from_seq'].fillna(False)
    merged['semantic_score'] = merged['semantic_score'].fillna(0.0)
    merged['popularity_score'] = merged['popularity_score'].fillna(0.0)
    
    # 计算综合分数
    merged['final_score'] = (
        0.7 * merged['semantic_score'] + 
        0.3 * merged['popularity_score']
    )
    
    # 按省份过滤
    if province_filter:
        merged = merged[merged['province'] == province_filter]
        print(f"  省份过滤: {province_filter} -> {len(merged)} 个候选")
    
    # 排序
    merged = merged.sort_values('final_score', ascending=False).reset_index(drop=True)
    
    print(f"\n合并结果:")
    print(f"  总候选数: {len(merged)}")
    print(f"  仅来自语义: {sum(merged['from_dense'] & ~merged['from_seq'])}")
    print(f"  仅来自序列: {sum(merged['from_seq'] & ~merged['from_dense'])}")
    print(f"  两者交集: {sum(merged['from_dense'] & merged['from_seq'])}")
    
    return merged

def _get_popular_pois(topk=30):
    """获取热门POI（基于用户事件统计）"""
    if not os.path.exists('data/user_events.csv'):
        print("  警告: 未找到用户事件数据，跳过序列推荐")
        return pd.DataFrame()
    
    # 统计POI流行度
    events = pd.read_csv('data/user_events.csv')
    popularity = events.groupby('poi_id').size().reset_index(name='count')
    popularity = popularity.sort_values('count', ascending=False).head(topk)
    
    # 归一化流行度分数
    max_count = popularity['count'].max()
    popularity['popularity_score'] = popularity['count'] / max_count
    
    # 加载POI元数据
    poi_meta = pd.read_csv('outputs/emb/poi_meta.csv')
    results = poi_meta[poi_meta['poi_id'].isin(popularity['poi_id'])].copy()
    results = results.merge(popularity[['poi_id', 'popularity_score']], on='poi_id')
    
    print(f"  ✓ 流行度召回: {len(results)} 个POI")
    
    return results

if __name__ == "__main__":
    # 测试
    candidates = merge_candidates(
        query_text="想去新疆看雪山和草原",
        topk_dense=30,
        topk_seq=20,
        province_filter="新疆"
    )
    
    print(f"\nTop 10 候选:")
    print(candidates.head(10)[['name', 'city', 'final_score', 'from_dense', 'from_seq']])

