#!/usr/bin/env python3
"""
RecBole ��线召回集成测试脚本

测试场景：
1. RecBole 模型未训练时，降级到流行度召回
2. RecBole 模型存在时，使用真正的模型预测
3. 动态权重融合功能
"""
import sys
import os
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from recommendation.candidate_merger import merge_candidates, adaptive_fusion


def test_adaptive_fusion():
    """测试动态权重融合"""
    print("\n" + "=" * 60)
    print("测试 1: 动态权重融合")
    print("=" * 60)

    test_cases = [
        (0, "无历史行为"),
        (2, "历史行为较少"),
        (5, "历史行为中等"),
        (20, "历史行为充足"),
    ]

    for history_len, desc in test_cases:
        dense_w, behavior_w, geo_w = adaptive_fusion(
            user_history_length=history_len,
            base_dense_weight=0.55,
            base_behavior_weight=0.30,
            base_geo_weight=0.15,
        )
        print(f"\n{desc} ({history_len} 次交互):")
        print(f"  权重: dense={dense_w:.3f}, behavior={behavior_w:.3f}, geo={geo_w:.3f}")


def test_candidate_merger_with_recbole():
    """测试候选合并器（不使用 RecBole，使用流行度）"""
    print("\n" + "=" * 60)
    print("测试 2: 候选合并器（流行度召回）")
    print("=" * 60)

    try:
        candidates = merge_candidates(
            query_text="想去新疆看雪山和草原",
            user_id="test_user_001",
            topk_dense=30,
            topk_seq=20,
            topk_geo=20,
            province_filter="新疆",
            fusion="rrf",
            use_recbole=False,  # 不使用 RecBole
            adaptive_fusion_enabled=True,
        )

        if len(candidates) > 0:
            print(f"\n✓ 成功获取 {len(candidates)} 个候选")
            print("\nTop 10 候选:")
            print(candidates.head(10)[["name", "city", "final_score", "from_dense", "from_behavior", "from_geo"]])

            # 检查融合权重
            if hasattr(candidates, 'attrs'):
                weights = candidates.attrs.get('fusion_weights', {})
                if weights:
                    print(f"\n融合权重:")
                    for key, value in weights.items():
                        print(f"  {key}: {value}")
        else:
            print("\n⚠️ 未获取到候选结果")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_candidate_merger_with_recbole_model():
    """测试候选合并器（尝试使用 RecBole 模型）"""
    print("\n" + "=" * 60)
    print("测试 3: 候选合并器（RecBole 召回）")
    print("=" * 60)

    # 检查模型是否存在
    model_path = Path("outputs/recbole/saved")
    if not model_path.exists():
        print(f"\n⚠️ RecBole 模型不存在: {model_path}")
        print("  将测试降级逻辑...")

    try:
        candidates = merge_candidates(
            query_text="想去新疆看雪山和草原",
            user_id="test_user_001",
            topk_dense=30,
            topk_seq=20,
            topk_geo=20,
            province_filter="新疆",
            fusion="rrf",
            use_recbole=True,  # 尝试使用 RecBole
            recbole_model_path="outputs/recbole/saved",
            recbole_config="configs/recbole.yaml",
            recbole_use_gpu=True,
            adaptive_fusion_enabled=True,
        )

        if len(candidates) > 0:
            print(f"\n✓ 成功获取 {len(candidates)} 个候选")
            print("\nTop 10 候选:")
            print(candidates.head(10)[["name", "city", "final_score", "from_dense", "from_behavior", "from_geo"]])
        else:
            print("\n⚠️ 未获取到候选结果")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("RecBole 在线召回集成测试")
    print("=" * 60)

    # 测试 1: 动态权重融合
    test_adaptive_fusion()

    # 测试 2: 流行度召回
    test_candidate_merger_with_recbole()

    # 测试 3: RecBole 模型召回
    test_candidate_merger_with_recbole_model()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
