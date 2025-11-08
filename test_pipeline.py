"""
GoAfar 完整流程测试脚本
测试从数据准备到最终推荐的完整pipeline
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_step1_data():
    """测试步骤1: 数据准备"""
    print("\n" + "="*80)
    print("步骤 1: 数据准备")
    print("="*80)
    
    from data_processing.sql_extractor import parse_go_address_sql
    
    if not os.path.exists('data/poi.csv'):
        print("提取SQL数据...")
        parse_go_address_sql()
    else:
        print("✓ 数据已存在: data/poi.csv")
    
    import pandas as pd
    df = pd.read_csv('data/poi.csv')
    print(f"\n数据统计:")
    print(f"  景点总数: {len(df)}")
    print(f"  省份数: {df['province'].nunique()}")
    print(f"  省份分布:\n{df['province'].value_counts().to_string()}")
    
    return True

def test_step2_embedding():
    """测试步骤2: BGE-M3向量构建"""
    print("\n" + "="*80)
    print("步骤 2: BGE-M3向量构建（测试少量数据）")
    print("="*80)
    
    from embedding.bge_m3_encoder import BGEM3Encoder
    import pandas as pd
    import numpy as np
    
    # 只测试前10个POI
    df = pd.read_csv('data/poi.csv').head(10)
    
    print(f"测试 {len(df)} 个 POI...")
    
    encoder = BGEM3Encoder(
        model_path='/root/autodl-tmp/goafar_project/models/Xorbits/bge-m3',
        use_gpu=False
    )
    
    # 构建文本
    texts = [f"{row['name']} {row['province']} {row['city']}" for _, row in df.iterrows()]
    
    # 编码
    result = encoder.encode_texts(texts, batch_size=5)
    
    print(f"✓ 向量维度: {result['dense_vecs'].shape}")
    
    # 保存测试向量
    os.makedirs('outputs/emb', exist_ok=True)
    np.save('outputs/emb/test_poi_emb.npy', result['dense_vecs'])
    df.to_csv('outputs/emb/test_poi_meta.csv', index=False)
    
    print(f"✓ 测试向量已保存")
    
    return True

def test_step3_intent():
    """测试步骤3: 意图理解"""
    print("\n" + "="*80)
    print("步骤 3: 意图理解（模板模式）")
    print("="*80)
    
    from llm4rec.intent_understanding import IntentUnderstandingModule
    
    module = IntentUnderstandingModule(use_template=True)
    
    test_queries = [
        "想去新疆喀纳斯看3天秋天的景色，拍照",
        "西藏拉萨5日深度游，布达拉宫和纳木错",
        "云南大理洱海2天骑行，轻松休闲"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        intent = module.understand(query)
        print(f"  → 省份: {intent.get('province')}")
        print(f"  → 兴趣: {', '.join(intent.get('interests', []))}")
        print(f"  → 活动: {', '.join(intent.get('activities', []))}")
    
    return True

def test_step4_reranking():
    """测试步骤4: LLM重排序"""
    print("\n" + "="*80)
    print("步骤 4: LLM重排序（规则模式）")
    print("="*80)
    
    from llm4rec.llm_reranker import LLMReranker
    import pandas as pd
    
    reranker = LLMReranker(use_template=True)
    
    # 构造测试候选
    test_candidates = pd.DataFrame({
        'poi_id': [f'POI_{i:04d}' for i in range(10)],
        'name': ['喀纳斯湖', '禾木村', '天山天池', '赛里木湖', '那拉提草原',
                 '火焰山', '葡萄沟', '天山大峡谷', '博斯腾湖', '巴音布鲁克'],
        'city': ['阿勒泰', '阿勒泰', '乌鲁木齐', '伊犁', '伊犁',
                 '吐鲁番', '吐鲁番', '乌鲁木齐', '巴音郭楞', '巴音郭楞'],
        'province': ['新疆'] * 10,
        'description': ['湖泊美景'] * 10,
        'semantic_score': [0.9 - i*0.05 for i in range(10)],
        'final_score': [0.9 - i*0.05 for i in range(10)]
    })
    
    test_intent = {
        'original_query': '想去新疆看雪山和草原，拍秋天的景色',
        'province': '新疆',
        'interests': ['雪山', '草原'],
        'activities': ['拍照'],
        'season_preference': '秋',
        'travel_style': '摄影游'
    }
    
    reranked = reranker.rerank(test_candidates, test_intent, topk=5)
    
    print(f"\n重排序结果 Top 5:")
    for i, (_, row) in enumerate(reranked.iterrows(), 1):
        print(f"  {i}. {row['name']:<15} {row['city']:<10} 分数:{row['rerank_score']:.4f}")
    
    return True

def test_step5_routing():
    """测试步骤5: 路线规划"""
    print("\n" + "="*80)
    print("步骤 5: VRPTW路线规划（小规模测试）")
    print("="*80)
    
    from routing.time_matrix_builder import build_time_matrix
    from routing.vrptw_solver import VRPTWSolver
    import pandas as pd
    
    # 使用前10个POI测试
    df = pd.read_csv('data/poi.csv')
    test_pois = df[df['province'] == '新疆'].head(10)
    
    print(f"测试 {len(test_pois)} 个新疆景点...")
    
    # 构建时间矩阵
    time_matrix, poi_df = build_time_matrix(
        poi_ids=test_pois['poi_id'].tolist()
    )
    
    # 求解
    solver = VRPTWSolver(poi_df, time_matrix)
    solution = solver.solve(max_duration_hours=8, time_limit_seconds=10)
    
    if solution:
        print(f"\n✓ 找到可行路线")
        print(f"  访问景点: {solution['visited_pois']}")
        print(f"  总时长: {solution['total_hours']:.2f}小时")
        return True
    else:
        print(f"\n✗ 未找到可行路线（可能需要调整参数）")
        return False

def test_all():
    """运行所有测试"""
    print("="*80)
    print("GoAfar 完整流程测试")
    print("="*80)
    
    steps = [
        ("数据准备", test_step1_data),
        ("BGE-M3向量", test_step2_embedding),
        ("意图理解", test_step3_intent),
        ("LLM重排序", test_step4_reranking),
        ("路线规划", test_step5_routing)
    ]
    
    results = {}
    
    for step_name, test_func in steps:
        try:
            success = test_func()
            results[step_name] = "✓" if success else "✗"
        except Exception as e:
            print(f"\n✗ {step_name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results[step_name] = "✗"
    
    # 总结
    print("\n\n" + "="*80)
    print("测试总结")
    print("="*80)
    
    for step_name, status in results.items():
        print(f"  {status} {step_name}")
    
    all_passed = all(v == "✓" for v in results.values())
    
    if all_passed:
        print("\n✓ 所有测试通过！")
        print("\n下一步: 运行完整推荐")
        print("  - 基础版: python main.py")
        print("  - LLM版: python run_with_llm.py")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息")
    
    return all_passed

if __name__ == "__main__":
    test_all()
