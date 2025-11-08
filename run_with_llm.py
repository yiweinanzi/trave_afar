"""
GoAfar - LLM增强版运行脚本
使用Qwen3-8B进行意图理解、重排序和文案生成
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llm4rec.qwen_recommender import QwenRecommender
from recommendation.candidate_merger import merge_candidates
from routing.time_matrix_builder import build_time_matrix
from routing.vrptw_solver import VRPTWSolver
import pandas as pd

def recommend_with_llm(query_text, use_gpu=False, max_hours=10, topk_candidates=30):
    """
    使用LLM增强的推荐流程
    
    Args:
        query_text: 用户查询
        use_gpu: 是否使用GPU
        max_hours: 最大行程时间
        topk_candidates: 候选数量
    
    Returns:
        dict: 推荐结果
    """
    print("\n" + "="*80)
    print("GoAfar 智能路线推荐 - LLM增强版")
    print("="*80)
    
    # 初始化Qwen推荐器
    print(f"\n【初始化】加载 Qwen3 模型")
    print("-"*80)
    recommender = QwenRecommender(
        model_name_or_path='Qwen/Qwen3-8B',
        use_gpu=use_gpu
    )
    
    # Step 1: LLM意图理解
    print(f"\n【步骤 1/5】LLM 意图理解")
    print("-"*80)
    print(f"原始查询: {query_text}")
    
    user_intent = recommender.understand_intent(query_text)
    
    print(f"\n意图分析结果:")
    print(f"  省份: {user_intent.get('province', '未识别')}")
    print(f"  兴趣: {', '.join(user_intent.get('interests', []))}")
    print(f"  活动: {', '.join(user_intent.get('activities', []))}")
    print(f"  风格: {user_intent.get('style', '观光游')}")
    if user_intent.get('duration_days'):
        print(f"  天数: {user_intent['duration_days']}天")
    
    province = user_intent.get('province')
    
    # Step 2: 多路召回
    print(f"\n【步骤 2/5】候选召回")
    print("-"*80)
    
    # 使用扩展后的查询
    search_query = user_intent.get('expanded_query', query_text)
    if 'keywords' in user_intent and user_intent['keywords']:
        search_query = ' '.join(user_intent['keywords'])
    
    print(f"扩展查询: {search_query}")
    
    candidates = merge_candidates(
        query_text=search_query,
        topk_dense=50,
        topk_seq=30,
        province_filter=province
    )
    
    if len(candidates) == 0:
        return {"error": "未找到匹配的景点"}
    
    # Step 3: LLM重排序
    print(f"\n【步骤 3/5】LLM 重排序")
    print("-"*80)
    
    # 转换为字典列表
    candidates_list = candidates.head(30).to_dict('records')
    
    if recommender.model is not None:
        print("使用 Qwen LLM 进行重排序...")
        ranked_poi_ids = recommender.rerank_pois(candidates_list, user_intent, topk=topk_candidates)
        
        # 按LLM排序结果重排
        candidates_reranked = candidates[candidates['poi_id'].isin(ranked_poi_ids)].copy()
        # 保持LLM的顺序
        candidates_reranked['llm_rank'] = candidates_reranked['poi_id'].map(
            {poi_id: i for i, poi_id in enumerate(ranked_poi_ids)}
        )
        candidates_reranked = candidates_reranked.sort_values('llm_rank')
    else:
        print("模型未加载，使用规则重排序...")
        candidates_reranked = candidates.head(topk_candidates)
    
    print(f"✓ 重排序完成，保留 {len(candidates_reranked)} 个候选")
    
    # Step 4: 路线规划
    print(f"\n【步骤 4/5】VRPTW 路线规划")
    print("-"*80)
    
    time_matrix, poi_df_filtered = build_time_matrix(
        poi_ids=candidates_reranked['poi_id'].tolist()
    )
    
    solver = VRPTWSolver(poi_df_filtered, time_matrix)
    solution = solver.solve(
        depot_index=0,
        max_duration_hours=max_hours,
        time_limit_seconds=30
    )
    
    if not solution:
        return {"error": "未找到可行路线"}
    
    # Step 5: LLM文案生成
    print(f"\n【步骤 5/5】LLM 文案生成")
    print("-"*80)
    
    route_pois = solution['routes'][0]
    
    content = recommender.generate_content(
        route_pois,
        province or candidates_reranked.iloc[0]['province'],
        solution['total_time_hours'],
        query_text
    )
    
    print(f"\n✨ 标题: {content['title']}")
    print(f"📝 描述: {content['description']}")
    
    # 组装结果
    result = {
        'title': content['title'],
        'description': content['description'],
        'route': route_pois,
        'total_hours': solution['total_time_hours'],
        'num_pois': solution['visited_pois'],
        'query': query_text,
        'user_intent': user_intent,
        'province': province or candidates_reranked.iloc[0]['province']
    }
    
    return result

def main():
    """主函数"""
    
    # 测试场景
    scenarios = [
        {
            'query': '想去新疆喀纳斯看3天秋天的景色，拍照',
            'max_hours': 10
        },
        {
            'query': '计划西藏拉萨5日游，朝拜布达拉宫，体验藏族文化',
            'max_hours': 12
        },
        {
            'query': '云南大理洱海2天骑行，轻松休闲',
            'max_hours': 8
        }
    ]
    
    results = []
    
    for idx, scenario in enumerate(scenarios, 1):
        print("\n\n" + "="*80)
        print(f"场景 {idx}/{len(scenarios)}")
        print("="*80)
        
        try:
            result = recommend_with_llm(**scenario, use_gpu=False)
            
            if 'error' not in result:
                results.append(result)
                
                # 保存结果
                output_file = f"outputs/results/llm_scenario_{idx}.json"
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                print(f"\n✓ 结果已保存: {output_file}")
            else:
                print(f"\n✗ {result['error']}")
                
        except Exception as e:
            print(f"\n✗ 场景执行失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n\n" + "="*80)
    print(f"✓ 完成 {len(results)}/{len(scenarios)} 个场景")
    print("="*80)
    print("\n查看结果: outputs/results/")

if __name__ == "__main__":
    main()

