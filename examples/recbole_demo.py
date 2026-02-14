#!/usr/bin/env python3
"""
动态权重融合算法演示

展示 adaptive_fusion 函数如何根据用户历史长度调整权重
"""


def adaptive_fusion(
    user_history_length: int,
    base_dense_weight: float = 0.55,
    base_behavior_weight: float = 0.30,
    base_geo_weight: float = 0.15,
    min_behavior_weight: float = 0.10,
    max_behavior_weight: float = 0.50,
    history_threshold: int = 5
) -> tuple:
    """
    动态权重融合：根据用户历史长度调整召回权重

    策略：
    - 用户历史行为少时：降低行为召回权重，提升语义和地理召回
    - 用户历史行为多时：提升行为召回权重，降低其他召回

    Args:
        user_history_length: 用户历史交互数量
        base_dense_weight: 基础语义召回权重
        base_behavior_weight: 基础行为召回权重
        base_geo_weight: 基础地理召回权重
        min_behavior_weight: 最小行为召回权重
        max_behavior_weight: 最大行为召回权重
        history_threshold: 历史行为数量阈值

    Returns:
        (dense_weight, behavior_weight, geo_weight)
    """
    if user_history_length == 0:
        # 无历史行为：降低行为召回权重
        behavior_weight = min_behavior_weight
        geo_weight = base_geo_weight + (base_behavior_weight - min_behavior_weight) * 0.5
        dense_weight = 1.0 - behavior_weight - geo_weight
    elif user_history_length < history_threshold:
        # 历史行为较少：逐步提升行为召回权重
        ratio = user_history_length / history_threshold
        behavior_weight = min_behavior_weight + (max_behavior_weight - min_behavior_weight) * ratio * 0.5
        dense_weight = base_dense_weight - (behavior_weight - base_behavior_weight) * 0.7
        geo_weight = 1.0 - dense_weight - behavior_weight
    else:
        # 历史行为充足：使用基础权重或提升行为召回
        behavior_weight = min(max_behavior_weight, base_behavior_weight * 1.3)
        dense_weight = base_dense_weight - (behavior_weight - base_behavior_weight)
        geo_weight = base_geo_weight

    # 确保权重和为 1 且非负
    total = dense_weight + behavior_weight + geo_weight
    dense_weight = max(0.0, dense_weight / total)
    behavior_weight = max(0.0, behavior_weight / total)
    geo_weight = max(0.0, geo_weight / total)

    return dense_weight, behavior_weight, geo_weight


def main():
    """演示动态权重融合"""
    print("\n" + "=" * 70)
    print("动态权重融合算法演示")
    print("=" * 70)

    print("\n策略说明:")
    print("-" * 70)
    print("1. 无历史行为 (0 次):")
    print("   - 降低行为召回权重 (0.10)")
    print("   - 提升语义和地理召回权重")
    print("   - 依赖语义理解和地理位置")
    print()
    print("2. 历史行为较少 (1-4 次):")
    print("   - 逐步提升行为召回权重")
    print("   - 平衡三种召回方式")
    print("   - 渐进式个性化")
    print()
    print("3. 历史行为充足 (≥5 次):")
    print("   - 提升行为召回权重 (0.39)")
    print("   - 降低其他召回权重")
    print("   - 强化个性化推荐")

    print("\n" + "=" * 70)
    print("权重分布表")
    print("=" * 70)
    print(f"{'用户类型':<20} {'历史':<6} {'语义':<8} {'行为':<8} {'地理':<8}")
    print("-" * 70)

    test_cases = [
        (0, "新用户（无历史）"),
        (1, "轻度用户"),
        (2, "轻度用户"),
        (3, "中度用户"),
        (4, "中度用户"),
        (5, "活跃用户"),
        (10, "活跃用户"),
        (20, "重度用户"),
    ]

    for history_len, user_type in test_cases:
        dense_w, behavior_w, geo_w = adaptive_fusion(
            user_history_length=history_len,
            base_dense_weight=0.55,
            base_behavior_weight=0.30,
            base_geo_weight=0.15,
        )

        print(f"{user_type:<20} {history_len:<6} "
              f"{dense_w:<8.3f} {behavior_w:<8.3f} {geo_w:<8.3f}")

    print("=" * 70)

    # 权重变化可视化
    print("\n权重变化趋势:")
    print("-" * 70)

    history_lengths = list(range(0, 21))
    dense_weights = []
    behavior_weights = []
    geo_weights = []

    for h in history_lengths:
        d, b, g = adaptive_fusion(h)
        dense_weights.append(d)
        behavior_weights.append(b)
        geo_weights.append(g)

    print(f"{'历史长度':<10} {'语义召回':<30} {'行为召回':<30} {'地理召回':<30}")
    print()

    for i, h in enumerate(history_lengths):
        if h % 5 == 0 or h == 0:  # 每 5 个显示一次
            bar_dense = '█' * int(dense_weights[i] * 50)
            bar_behavior = '█' * int(behavior_weights[i] * 50)
            bar_geo = '█' * int(geo_weights[i] * 50)

            print(f"{h:<10} {bar_dense:<30} {bar_behavior:<30} {bar_geo:<30}")

    print("=" * 70)

    # 业务场景示例
    print("\n业务场景示例:")
    print("-" * 70)

    scenarios = [
        {
            "场景": "新用户首次访问",
            "历史": 0,
            "策略": "推荐热门景点 + 语义匹配",
            "权重": "语义 0.63, 行为 0.10, 地理 0.27"
        },
        {
            "场景": "用户浏览了 3 个景点",
            "历史": 3,
            "策略": "结合浏览历史 + 语义理解",
            "权重": "语义 0.52, 行为 0.24, 地理 0.24"
        },
        {
            "场景": "用户有 10 次交互",
            "历史": 10,
            "策略": "个性化推荐主导",
            "权重": "语义 0.42, 行为 0.39, 地理 0.19"
        },
    ]

    for scenario in scenarios:
        print(f"\n场景: {scenario['场景']}")
        print(f"  历史交互: {scenario['历史']} 次")
        print(f"  推荐策略: {scenario['策略']}")
        print(f"  权重分布: {scenario['权重']}")

    print("\n" + "=" * 70)
    print("演示完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
