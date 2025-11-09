#!/usr/bin/env python
"""
全链路测试脚本
检查GoAfar系统的每个模块是否正常工作
"""
import os
import sys
import traceback
import pandas as pd
import numpy as np

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_data_preparation():
    """测试数据准备"""
    print("\n" + "="*60)
    print("1. 数据准备测试")
    print("="*60)
    
    checks = []
    
    # 检查POI数据
    if os.path.exists('data/poi.csv'):
        df = pd.read_csv('data/poi.csv')
        checks.append(("POI数据文件存在", True))
        checks.append(("POI数量", len(df) > 0))
        checks.append(("必需列", all(col in df.columns for col in ['poi_id', 'name', 'lat', 'lon', 'province', 'city'])))
        print(f"  ✓ POI数据: {len(df)}个景点")
    else:
        checks.append(("POI数据文件存在", False))
        print("  ❌ POI数据文件不存在")
    
    # 检查用户数据
    if os.path.exists('data/user_events.csv'):
        df = pd.read_csv('data/user_events.csv')
        checks.append(("用户事件文件存在", True))
        checks.append(("用户事件数量", len(df) > 0))
        print(f"  ✓ 用户事件: {len(df)}条")
    else:
        checks.append(("用户事件文件存在", False))
        print("  ⚠️ 用户事件文件不存在（可选）")
    
    return all(c[1] for c in checks), checks

def test_embedding():
    """测试嵌入向量"""
    print("\n" + "="*60)
    print("2. 嵌入向量测试")
    print("="*60)
    
    checks = []
    
    # 检查向量文件
    if os.path.exists('outputs/emb/poi_emb.npy'):
        emb = np.load('outputs/emb/poi_emb.npy')
        checks.append(("向量文件存在", True))
        checks.append(("向量维度正确", emb.shape[1] == 1024))  # BGE-M3 dense维度
        print(f"  ✓ 向量文件: {emb.shape[0]}个POI, {emb.shape[1]}维")
    else:
        checks.append(("向量文件存在", False))
        print("  ❌ 向量文件不存在")
    
    # 检查元数据
    if os.path.exists('outputs/emb/poi_meta.csv'):
        meta = pd.read_csv('outputs/emb/poi_meta.csv')
        checks.append(("元数据文件存在", True))
        checks.append(("元数据数量匹配", len(meta) == emb.shape[0]))
        print(f"  ✓ 元数据: {len(meta)}条")
    else:
        checks.append(("元数据文件存在", False))
        print("  ❌ 元数据文件不存在")
    
    # 测试语义检索
    try:
        from embedding.vector_builder import search_similar_pois
        results = search_similar_pois("雪山", topk=5, use_gpu=False)
        checks.append(("语义检索功能", len(results) > 0))
        print(f"  ✓ 语义检索: 成功检索到{len(results)}个结果")
    except Exception as e:
        checks.append(("语义检索功能", False))
        print(f"  ❌ 语义检索失败: {e}")
    
    return all(c[1] for c in checks), checks

def test_intent_understanding():
    """测试意图理解"""
    print("\n" + "="*60)
    print("3. 意图理解测试")
    print("="*60)
    
    checks = []
    
    try:
        from llm4rec.intent_understanding import IntentUnderstandingModule
        module = IntentUnderstandingModule(use_template=True)
        
        test_query = "想去新疆看雪山和湖泊，拍照"
        intent = module.understand(test_query)
        
        checks.append(("模块导入", True))
        checks.append(("意图理解", 'province' in intent or 'interests' in intent))
        checks.append(("返回字典", isinstance(intent, dict)))
        
        print(f"  ✓ 意图理解成功")
        print(f"    省份: {intent.get('province', '未识别')}")
        print(f"    兴趣: {intent.get('interests', [])}")
    except Exception as e:
        checks.append(("模块导入", False))
        print(f"  ❌ 意图理解失败: {e}")
        traceback.print_exc()
    
    return all(c[1] for c in checks), checks

def test_reranking():
    """测试重排序"""
    print("\n" + "="*60)
    print("4. 重排序测试")
    print("="*60)
    
    checks = []
    
    try:
        from llm4rec.llm_reranker import LLMReranker
        reranker = LLMReranker(use_template=True)
        
        # 构造测试数据
        test_candidates = pd.DataFrame({
            'poi_id': [f'POI_{i:04d}' for i in range(10)],
            'name': ['喀纳斯湖', '天山天池', '赛里木湖', '那拉提草原', '禾木村'] * 2,
            'city': ['阿勒泰', '乌鲁木齐', '伊犁', '伊犁', '阿勒泰'] * 2,
            'province': ['新疆'] * 10,
            'description': ['湖泊', '天池', '湖泊', '草原', '村庄'] * 2,
            'semantic_score': [0.9 - i*0.05 for i in range(10)]
        })
        
        test_intent = {
            'province': '新疆',
            'interests': ['湖泊', '草原'],
            'activities': ['拍照']
        }
        
        reranked = reranker.rerank(test_candidates, test_intent, topk=5)
        
        checks.append(("模块导入", True))
        checks.append(("重排序功能", len(reranked) > 0))
        checks.append(("返回DataFrame", isinstance(reranked, pd.DataFrame)))
        checks.append(("rerank_score列", 'rerank_score' in reranked.columns))
        
        print(f"  ✓ 重排序成功: {len(reranked)}个结果")
    except Exception as e:
        checks.append(("模块导入", False))
        print(f"  ❌ 重排序失败: {e}")
        traceback.print_exc()
    
    return all(c[1] for c in checks), checks

def test_routing():
    """测试路线规划"""
    print("\n" + "="*60)
    print("5. 路线规划测试")
    print("="*60)
    
    checks = []
    
    try:
        from routing.time_matrix_builder import build_time_matrix
        from routing.vrptw_solver import VRPTWSolver
        
        # 测试时间矩阵 - 使用实际存在的POI ID
        # 先读取POI数据获取实际ID
        poi_data = pd.read_csv('data/poi.csv')
        test_poi_ids = poi_data['poi_id'].head(5).tolist()
        print(f"  使用POI ID: {test_poi_ids}")
        
        time_matrix, poi_df = build_time_matrix(poi_ids=test_poi_ids)
        
        checks.append(("时间矩阵构建", time_matrix is not None))
        checks.append(("时间矩阵形状", time_matrix.shape[0] == len(test_poi_ids)))
        checks.append(("POI数据", len(poi_df) == len(test_poi_ids)))
        
        print(f"  ✓ 时间矩阵: {time_matrix.shape}")
        print(f"  ✓ POI数据: {len(poi_df)}个")
        
        # 测试VRPTW求解
        solver = VRPTWSolver(poi_df, time_matrix, start_time_min=480)
        solution = solver.solve(
            depot_index=0,
            max_duration_hours=8,
            time_limit_seconds=10
        )
        
        if solution:
            checks.append(("VRPTW求解", True))
            checks.append(("路线存在", 'routes' in solution))
            checks.append(("总时长", 'total_hours' in solution))
            print(f"  ✓ VRPTW求解成功: {solution['visited_pois']}个POI")
        else:
            checks.append(("VRPTW求解", False))
            print("  ⚠️ VRPTW未找到可行解（可能正常）")
    except Exception as e:
        checks.append(("路线规划", False))
        print(f"  ❌ 路线规划失败: {e}")
        traceback.print_exc()
    
    return all(c[1] for c in checks), checks

def test_content_generation():
    """测试内容生成"""
    print("\n" + "="*60)
    print("6. 内容生成测试")
    print("="*60)
    
    checks = []
    
    try:
        from content_generation.title_generator import generate_title, generate_description
        
        # 模拟路线数据
        test_route = [
            {'poi_name': '喀纳斯湖', 'poi_city': '阿勒泰'},
            {'poi_name': '天山天池', 'poi_city': '乌鲁木齐'},
            {'poi_name': '赛里木湖', 'poi_city': '伊犁'}
        ]
        
        title = generate_title(test_route, '新疆', '想去新疆看湖泊')
        description = generate_description(test_route, '新疆', 8.5, '想去新疆看湖泊')
        
        checks.append(("标题生成", len(title) > 0))
        checks.append(("描述生成", len(description) > 0))
        
        print(f"  ✓ 标题生成: {title[:50]}...")
        print(f"  ✓ 描述生成: {len(description)}字符")
    except Exception as e:
        checks.append(("内容生成", False))
        print(f"  ❌ 内容生成失败: {e}")
        traceback.print_exc()
    
    return all(c[1] for c in checks), checks

def test_end_to_end():
    """测试端到端流程"""
    print("\n" + "="*60)
    print("7. 端到端测试")
    print("="*60)
    
    checks = []
    
    try:
        # 模拟完整推荐流程
        from embedding.vector_builder import search_similar_pois
        from llm4rec.intent_understanding import IntentUnderstandingModule
        from llm4rec.llm_reranker import LLMReranker
        from routing.time_matrix_builder import build_time_matrix
        from routing.vrptw_solver import VRPTWSolver
        from content_generation.title_generator import generate_title, generate_description
        
        # 1. 意图理解
        intent_module = IntentUnderstandingModule(use_template=True)
        query = "想去新疆看雪山"
        intent = intent_module.understand(query)
        checks.append(("意图理解", True))
        
        # 2. 语义检索
        candidates = search_similar_pois(query, topk=20, use_gpu=False)
        if 'province' in intent and intent['province']:
            candidates = candidates[candidates['province'] == intent['province']]
        checks.append(("语义检索", len(candidates) > 0))
        
        # 3. 重排序
        reranker = LLMReranker(use_template=True)
        candidates_reranked = reranker.rerank(candidates.head(10), intent, topk=5)
        checks.append(("重排序", len(candidates_reranked) > 0))
        
        # 4. 路线规划（如果有足够的候选）
        if len(candidates_reranked) >= 3 and 'poi_id' in candidates_reranked.columns:
            poi_ids = candidates_reranked['poi_id'].head(5).tolist()
            time_matrix, poi_df = build_time_matrix(poi_ids=poi_ids)
            
            solver = VRPTWSolver(poi_df, time_matrix, start_time_min=480)
            solution = solver.solve(depot_index=0, max_duration_hours=8, time_limit_seconds=10)
            
            if solution:
                checks.append(("路线规划", True))
                print(f"  ✓ 端到端成功: 生成{len(solution['routes'][0])}个站点的路线")
            else:
                checks.append(("路线规划", False))
                print("  ⚠️ 路线规划未找到可行解")
        else:
            checks.append(("路线规划", False))
            print("  ⚠️ 候选POI不足，跳过路线规划")
        
    except Exception as e:
        checks.append(("端到端", False))
        print(f"  ❌ 端到端测试失败: {e}")
        traceback.print_exc()
    
    return all(c[1] for c in checks), checks

def main():
    """主函数"""
    print("\n" + "="*60)
    print("GoAfar 全链路测试")
    print("="*60)
    
    results = {}
    
    # 运行所有测试
    results['数据准备'] = test_data_preparation()
    results['嵌入向量'] = test_embedding()
    results['意图理解'] = test_intent_understanding()
    results['重排序'] = test_reranking()
    results['路线规划'] = test_routing()
    results['内容生成'] = test_content_generation()
    results['端到端'] = test_end_to_end()
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    all_passed = True
    for module, (passed, checks) in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{module:15s}: {status}")
        if not passed:
            all_passed = False
            # 显示失败的检查项
            for check_name, check_result in checks:
                if not check_result:
                    print(f"  - {check_name}: ❌")
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 所有测试通过！系统运行正常！")
    else:
        print("⚠️ 部分测试失败，请检查上述错误")
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

