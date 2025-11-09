"""
GoAfar 智能旅行路线推荐系统 - 主入口
"""
import sys
import os

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from embedding.vector_builder import build_poi_embeddings, search_similar_pois
from recommendation.candidate_merger import merge_candidates
from routing.time_matrix_builder import build_time_matrix
from routing.vrptw_solver import VRPTWSolver
from llm_integration import GoAfarLLM
from llm4rec.intent_understanding import IntentUnderstandingModule
from llm4rec.llm_reranker import LLMReranker
import pandas as pd
import json

# 全局LLM实例（避免重复加载）
_llm_instance = None

def get_llm(mode='template'):
    """获取LLM实例（单例模式）"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = GoAfarLLM(mode=mode)
    return _llm_instance

def recommend_route(query_text, province=None, max_hours=10, topk_candidates=20, user_id=None, use_llm=True):
    """
    端到端路线推荐
    
    Args:
        query_text: 用户查询
        province: 目标省份（如果为None，自动识别）
        max_hours: 最大行程时间（小时）
        topk_candidates: 候选POI数量
        user_id: 用户ID（可选）
        use_llm: 是否使用LLM4Rec增强（意图理解和重排序）
    
    Returns:
        dict: 推荐结果
    """
    print("\n" + "="*80)
    print("GoAfar 智能路线推荐")
    print("="*80)
    print(f"查询: {query_text}")
    print(f"省份: {province or '自动识别'}")
    print(f"最大行程: {max_hours} 小时")
    print(f"LLM增强: {'是' if use_llm else '否'}")
    
    # Step 1: LLM意图理解（如果启用）
    if use_llm:
        print(f"\n【步骤 1/5】LLM 意图理解")
        print("-"*80)
        intent_module = IntentUnderstandingModule(use_template=True)
        intent = intent_module.understand(query_text)
        
        # 如果自动识别到省份，使用它
        if province is None and intent.get('province'):
            province = intent['province']
            print(f"✓ 自动识别省份: {province}")
        
        # 使用意图中的关键词进行检索
        if intent.get('keywords'):
            search_query = ' '.join(intent['keywords'])
            print(f"✓ 扩展查询: {search_query}")
        else:
            search_query = query_text
    else:
        intent = {}
        search_query = query_text
    
    # Step 2: 候选召回
    print(f"\n【步骤 2/5】候选池召回")
    print("-"*80)
    
    candidates = merge_candidates(
        query_text=search_query,
        user_id=user_id,
        topk_dense=50,
        topk_seq=30,
        province_filter=province
    )
    
    if len(candidates) == 0:
        return {"error": "未找到匹配的景点"}
    
    # Step 3: LLM重排序（如果启用）
    if use_llm and len(candidates) > 0:
        print(f"\n【步骤 3/5】LLM 重排序")
        print("-"*80)
        reranker = LLMReranker(use_template=True)
        candidates_reranked = reranker.rerank(candidates, intent, topk=topk_candidates)
        print(f"✓ 重排序完成，保留 {len(candidates_reranked)} 个候选")
    else:
        candidates_reranked = candidates.head(topk_candidates)
        print(f"\n选择 Top {len(candidates_reranked)} 候选进行路线规划")
    
    # Step 4: 构建时间矩阵
    step_num = "4/5" if use_llm else "3/4"
    print(f"\n【步骤 {step_num}】构建时间矩阵")
    print("-"*80)
    
    time_matrix, poi_df_filtered = build_time_matrix(
        poi_ids=candidates_reranked['poi_id'].tolist()
    )
    
    # Step 5: VRPTW路线规划
    step_num = "5/5" if use_llm else "4/4"
    print(f"\n【步骤 {step_num}】VRPTW 路线规划")
    print("-"*80)
    
    solver = VRPTWSolver(poi_df_filtered, time_matrix)
    solution = solver.solve(
        depot_index=0,
        max_duration_hours=max_hours,
        time_limit_seconds=30
    )
    
    if not solution:
        return {"error": "未找到可行路线"}
    
    # Step 6: 生成文案
    step_num = "6/6" if use_llm else "5/5"
    print(f"\n【步骤 {step_num}】生成推荐文案")
    print("-"*80)
    
    route_pois = solution['routes'][0]  # 取第一条路线
    province_name = province or candidates_reranked.iloc[0]['province']
    
    # 使用LLM生成（或模板）
    llm = get_llm(mode='template')  # 可改为 'local' 或 'api'
    title = llm.generate_route_title(route_pois, province_name, query_text)
    description = llm.generate_route_description(
        route_pois, 
        province_name, 
        solution['total_time_hours'],
        query_text
    )
    
    print(f"\n✨ 标题: {title}")
    print(f"📝 描述: {description}")
    
    # 组装最终结果
    result = {
        'title': title,
        'description': description,
        'route': route_pois,
        'total_hours': solution['total_time_hours'],
        'num_pois': solution['visited_pois'],
        'query': query_text,
        'province': province_name,
        'user_intent': intent if use_llm else None
    }
    
    return result

def main():
    """主函数 - 演示多个场景"""
    
    scenarios = [
        {
            'query': '想去新疆喀纳斯看秋天的景色，拍照',
            'province': '新疆',
            'max_hours': 10
        },
        {
            'query': '去西藏朝拜布达拉宫，体验藏族文化',
            'province': '西藏',
            'max_hours': 8
        },
        {
            'query': '云南大理洱海骑行，逛古镇',
            'province': '云南',
            'max_hours': 6
        }
    ]
    
    results_all = []
    
    for idx, scenario in enumerate(scenarios, 1):
        print("\n\n" + "="*80)
        print(f"场景 {idx}/{len(scenarios)}")
        print("="*80)
        
        try:
            result = recommend_route(**scenario)
            results_all.append(result)
            
            # 保存结果
            output_file = f"outputs/results/scenario_{idx}.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"\n✓ 结果已保存: {output_file}")
            
        except Exception as e:
            print(f"\n✗ 场景执行失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n\n" + "="*80)
    print(f"✓ 完成 {len(results_all)}/{len(scenarios)} 个场景")
    print("="*80)

if __name__ == "__main__":
    main()

