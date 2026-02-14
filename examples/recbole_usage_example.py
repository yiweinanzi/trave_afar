#!/usr/bin/env python3
"""
RecBole 在线召回使用示例

演示如何在推荐系统中使用 RecBole 行为召回
"""
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def example_1_popularity_recall():
    """示例 1: 使用原有流行度召回（默认）"""
    print("\n" + "=" * 60)
    print("示例 1: 流行度召回")
    print("=" * 60)

    from recommendation.candidate_merger import merge_candidates

    candidates = merge_candidates(
        query_text="想去新疆看雪山和草原",
        user_id="user_001",
        topk_dense=30,
        topk_seq=20,
        topk_geo=20,
        province_filter="新疆",
        use_recbole=False,  # 不使用 RecBole
        adaptive_fusion_enabled=False
    )

    print(f"\n返回候选数: {len(candidates)}")
    if len(candidates) > 0:
        print("\nTop 5 候选:")
        print(candidates.head(5)[["name", "city", "final_score", "from_behavior"]])


def example_2_recbole_recall():
    """示例 2: 使用 RecBole 模型召回"""
    print("\n" + "=" * 60)
    print("示例 2: RecBole 模型召回")
    print("=" * 60)

    from recommendation.candidate_merger import merge_candidates

    candidates = merge_candidates(
        query_text="想去新疆看雪山和草原",
        user_id="user_001",
        topk_dense=30,
        topk_seq=20,
        topk_geo=20,
        province_filter="新疆",
        use_recbole=True,  # 使用 RecBole
        recbole_model_path="outputs/recbole/saved",
        recbole_config="configs/recbole.yaml",
        recbole_use_gpu=True,
        adaptive_fusion_enabled=False
    )

    print(f"\n返回候选数: {len(candidates)}")
    if len(candidates) > 0:
        print("\nTop 5 候选:")
        print(candidates.head(5)[["name", "city", "final_score", "from_behavior"]])


def example_3_adaptive_fusion():
    """示例 3: 启用动态权重融合"""
    print("\n" + "=" * 60)
    print("示例 3: 动态权重融合")
    print("=" * 60)

    from recommendation.candidate_merger import merge_candidates

    # 测试不同历史长度的用户
    test_users = [
        ("user_new", "新用户（无历史）"),
        ("user_002", "普通用户"),
        ("user_003", "活跃用户"),
    ]

    for user_id, desc in test_users:
        print(f"\n{desc}: {user_id}")

        candidates = merge_candidates(
            query_text="想去新疆看雪山和草原",
            user_id=user_id,
            topk_dense=30,
            topk_seq=20,
            topk_geo=20,
            province_filter="新疆",
            use_recbole=True,
            adaptive_fusion_enabled=True  # 启用动态权重
        )

        if hasattr(candidates, 'attrs'):
            weights = candidates.attrs.get('fusion_weights', {})
            print(f"  权重: dense={weights['dense']:.3f}, behavior={weights['behavior']:.3f}, geo={weights['geo']:.3f}")
            print(f"  历史交互: {weights['user_history_length']} 次")
            print(f"  返回候选: {len(candidates)} 个")


def example_4_provider_directly():
    """示例 4: 直接使用 RecBoleProvider"""
    print("\n" + "=" * 60)
    print("示例 4: 直接使用 RecBoleProvider")
    print("=" * 60)

    from recommendation.recbole_trainer import RecBoleProvider
    import pandas as pd

    # 创建 Provider
    provider = RecBoleProvider(
        model_path="outputs/recbole/saved",
        config_file="configs/recbole.yaml",
        use_gpu=True,
        fallback_to_popular=True
    )

    print(f"\n模型可用: {provider.available}")

    # 创建模拟 POI 数据
    poi_df = pd.DataFrame({
        'poi_id': ['poi_1', 'poi_2', 'poi_3'],
        'name': ['POI 1', 'POI 2', 'POI 3'],
        'city': ['City 1', 'City 2', 'City 3'],
        'province': ['Province 1', 'Province 2', 'Province 3']
    })

    # 预测
    rec_df, metadata = provider.predict(
        user_id="test_user",
        topk=10,
        poi_df=poi_df
    )

    print(f"\n预测方法: {metadata['method']}")
    print(f"返回结果: {len(rec_df)} 个")

    if len(rec_df) > 0:
        print("\n推荐结果:")
        print(rec_df[['poi_id', 'recbole_score']])


def example_5_adaptive_fusion_weights():
    """示例 5: 查看动态权重"""
    print("\n" + "=" * 60)
    print("示例 5: 动态权重计算")
    print("=" * 60)

    from recommendation.candidate_merger import adaptive_fusion

    test_cases = [
        (0, "无历史行为"),
        (2, "历史行为较少"),
        (5, "历史行为中等"),
        (10, "历史行为较多"),
        (20, "历史行为充足"),
    ]

    print("\n用户历史长度 -> 权重分布")
    print("-" * 60)

    for history_len, desc in test_cases:
        dense_w, behavior_w, geo_w = adaptive_fusion(
            user_history_length=history_len,
            base_dense_weight=0.55,
            base_behavior_weight=0.30,
            base_geo_weight=0.15,
        )

        print(f"{desc:20} ({history_len:2} 次) -> "
              f"dense={dense_w:.3f}, behavior={behavior_w:.3f}, geo={geo_w:.3f}")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("RecBole 在线召回使用示例")
    print("=" * 60)

    # 示例 1: 流行度召回
    # example_1_popularity_recall()

    # 示例 2: RecBole 召回
    # example_2_recbole_recall()

    # 示例 3: 动态权重融合
    # example_3_adaptive_fusion()

    # 示例 4: 直接使用 Provider
    # example_4_provider_directly()

    # 示例 5: 动态权重计算
    example_5_adaptive_fusion_weights()

    print("\n" + "=" * 60)
    print("提示：取消注释其他示例以运行")
    print("=" * 60)


if __name__ == "__main__":
    main()
