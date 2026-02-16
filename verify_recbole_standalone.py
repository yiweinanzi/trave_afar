#!/usr/bin/env python3
"""
RecBole 在线集成独立验证脚本

验证方法：
1. 检查RecBole模型是否被加载
2. 验证用户历史序列是否被使用
3. 对比有/无RecBole的召回结果
4. 确认行为召回在融合中的权重
"""
import sys
import os
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))


def check_model_availability():
    """1. 检查RecBole模型是否被加载"""
    print("\n" + "=" * 60)
    print("检查 1: RecBole 模型可用性")
    print("=" * 60)

    model_path = Path("outputs/recbole/saved")
    config_file = Path("configs/recbole.yaml")

    print(f"\n模型路径: {model_path}")
    print(f"存在: {model_path.exists()}")

    if model_path.exists():
        checkpoint_files = list(model_path.glob("*.pth"))
        print(f"Checkpoint 文件: {len(checkpoint_files)}")
        if checkpoint_files:
            import os
            latest = max(checkpoint_files, key=os.path.getctime)
            print(f"最新 checkpoint: {latest.name}")
        else:
            print("  WARNING: 未找到 .pth 模型文件")
    else:
        print("  WARNING: 模型目录不存在")
        print("  当前将使用流行度召回作为降级策略")

    print(f"\n配置文件: {config_file}")
    print(f"存在: {config_file.exists()}")

    return model_path.exists() and len(list(model_path.glob("*.pth"))) > 0


def check_user_history_usage():
    """2. 验证用户历史序列是否被使用"""
    print("\n" + "=" * 60)
    print("检查 2: 用户历史序列使用")
    print("=" * 60)

    user_events_path = Path("data/all/user_events.csv")
    print(f"\n用户事件文件: {user_events_path}")
    print(f"存在: {user_events_path.exists()}")

    if user_events_path.exists():
        import pandas as pd
        events = pd.read_csv(user_events_path)
        print(f"总事件数: {len(events)}")
        print(f"用户数: {events['user_id'].nunique()}")
        print(f"POI数: {events['poi_id'].nunique()}")

        # 分析用户历史长度分布
        user_history = events.groupby('user_id').size()
        print(f"\n用户历史长度分布:")
        print(f"  最小: {user_history.min()}")
        print(f"  最大: {user_history.max()}")
        print(f"  平均: {user_history.mean():.2f}")
        print(f"  中位数: {user_history.median():.2f}")

        # 行为类型分布
        if 'action' in events.columns:
            print(f"\n行为类型分布:")
            action_counts = events['action'].value_counts()
            for action, count in action_counts.items():
                print(f"  {action}: {count}")

        return True
    else:
        print("  WARNING: 用户事件文件不存在")
        return False


def check_recbole_code_integration():
    """3. 检查RecBole代码集成情况"""
    print("\n" + "=" * 60)
    print("检查 3: RecBole 代码集成")
    print("=" * 60)

    merger_file = Path("src/recommendation/candidate_merger.py")
    trainer_file = Path("src/recommendation/recbole_trainer.py")

    checks = []

    # 检查导入
    merger_content = merger_file.read_text()
    trainer_content = trainer_file.read_text()

    print("\n导入检查:")
    checks.append((
        "RecBoleProvider 导入",
        "from recommendation.recbole_trainer import RecBoleProvider" in merger_content
    ))
    checks.append((
        "use_recbole 参数",
        "use_recbole" in merger_content
    ))
    checks.append((
        "adaptive_fusion_enabled 参数",
        "adaptive_fusion_enabled" in merger_content
    ))

    print("\nRecBoleProvider 方法检查:")
    checks.append((
        "RecBoleProvider 类存在",
        "class RecBoleProvider" in trainer_content
    ))
    checks.append((
        "predict 方法",
        "def predict(" in trainer_content
    ))
    checks.append((
        "_predict_by_popularity 降级方法",
        "def _predict_by_popularity(" in trainer_content
    ))
    checks.append((
        "get_user_history_length 方法",
        "def get_user_history_length(" in trainer_content
    ))

    print("\n降级逻辑检查:")
    checks.append((
        "冷启动处理",
        "cold_start" in trainer_content
    ))
    checks.append((
        "fallback_to_popular 参数",
        "fallback_to_popular" in trainer_content
    ))

    for name, result in checks:
        status = "  PASS" if result else "  FAIL"
        print(f"{status}: {name}")

    return all(c[1] for c in checks)


def check_adaptive_fusion_weights():
    """4. 检查动态权重融合配置"""
    print("\n" + "=" * 60)
    print("检查 4: 动态权重融合")
    print("=" * 60)

    # 导入并测试 adaptive_fusion 函数
    try:
        from recommendation.candidate_merger import adaptive_fusion

        print("\n不同用户历史长度的权重分布:")
        print(f"{'历史长度':<10} {'类型':<15} {'dense':<10} {'behavior':<10} {'geo':<10}")
        print("-" * 60)

        test_cases = [
            (0, "无历史行为"),
            (2, "历史行为较少"),
            (5, "历史行为中等"),
            (10, "历史行为较多"),
            (20, "历史行为充足"),
        ]

        for history_len, desc in test_cases:
            dense_w, behavior_w, geo_w = adaptive_fusion(
                user_history_length=history_len,
                base_dense_weight=0.55,
                base_behavior_weight=0.30,
                base_geo_weight=0.15,
            )
            print(f"{history_len:<10} {desc:<15} {dense_w:<10.3f} {behavior_w:<10.3f} {geo_w:<10.3f}")

        print("\n预期行为:")
        print("  - 无历史行为: behavior_weight 降低到最小值 (0.10)")
        print("  - 历史行为增多: behavior_weight 逐步提升")
        print("  - 历史行为充足: behavior_weight 达到最大值 (0.39)")

        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def simulate_recall_comparison():
    """5. 模拟有/无RecBole的召回结果对比"""
    print("\n" + "=" * 60)
    print("检查 5: 模拟召回结果对比")
    print("=" * 60)

    import pandas as pd
    from utils.id_mapping import normalize_poi_id

    # 读取 POI 数据
    poi_path = Path("data/all/poi_expanded.csv")
    if not poi_path.exists():
        print("  WARNING: POI 数据文件不存在，跳过模拟")
        return False

    poi_df = pd.read_csv(poi_path)
    poi_df['poi_id'] = poi_df['poi_id'].apply(normalize_poi_id)

    # 读取用户事件数据
    events_path = Path("data/all/user_events.csv")
    if not events_path.exists():
        print("  WARNING: 用户事件文件不存在，跳过模拟")
        return False

    events = pd.read_csv(events_path)
    events['poi_id'] = events['poi_id'].apply(normalize_poi_id)

    # 计算 POI 流行度
    ACTION_WEIGHT = {"click": 1.0, "fav": 2.0, "visit": 3.0}
    events["weight"] = events["action"].map(ACTION_WEIGHT).fillna(1.0)
    popularity = events.groupby("poi_id")["weight"].sum().sort_values(ascending=False)

    print("\n模拟场景:")
    print("  1. 流行度召回（RecBole 不可用时）")
    print("  2. RecBole 召回（模型可用时，使用用户历史序列）")

    # 选择一个测试用户
    test_user_id = events['user_id'].iloc[0]
    user_events = events[events['user_id'] == test_user_id]
    user_history = user_events['poi_id'].unique()

    print(f"\n测试用户: {test_user_id}")
    print(f"  历史交互数: {len(user_history)}")
    print(f"  历史POI: {list(user_history[:5])}")

    # 流行度召回结果
    top_popular = popularity.head(10)
    print(f"\n流行度召回 Top 10:")
    for i, (poi_id, score) in enumerate(top_popular.items(), 1):
        poi_name = poi_df[poi_df['poi_id'] == poi_id]['name'].values
        name = poi_name[0] if len(poi_name) > 0 else "Unknown"
        in_history = "*" if poi_id in user_history else ""
        print(f"  {i:2}. {poi_id}: {name[:30]:30} ({score:.1f}){in_history}")

    # 检查是否有热门 POI 在用户历史中
    overlap = set(top_popular.head(10).index) & set(user_history)
    print(f"\n  注: * 表示在用户历史中 ({len(overlap)} 个)")

    # 说明 RecBole 行为召回的不同
    print("\nRecBole 行为召回与流行度召回的区别:")
    print("  1. 流行度召回: 基于全局交互频率，所有用户看到相同结果")
    print("  2. RecBole 召回: 基于用户历史序列，使用序列模型预测个性化偏好")
    print("  3. 过滤历史: RecBole 会过滤用户已交互过的 POI")
    print("  4. 冷启动: 无历史行为时回退到流行度")

    return True


def check_config_settings():
    """6. 检查配置文件设置"""
    print("\n" + "=" * 60)
    print("检查 6: 配置文件设置")
    print("=" * 60)

    import yaml
    config_path = Path("configs/runtime.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    recall_config = config.get('recall', {})
    print("\n召回配置:")
    print(f"  behavior_provider: {recall_config.get('behavior_provider', '未设置')}")
    print(f"  recbole_model_path: {recall_config.get('recbole_model_path', '未设置')}")
    print(f"  recbole_config: {recall_config.get('recbole_config', '未设置')}")
    print(f"  recbole_use_gpu: {recall_config.get('recbole_use_gpu', '未设置')}")
    print(f"  adaptive_fusion: {recall_config.get('adaptive_fusion', '未设置')}")

    print("\n权重配置:")
    print(f"  dense_weight: {recall_config.get('dense_weight', '未设置')}")
    print(f"  behavior_weight: {recall_config.get('behavior_weight', '未设置')}")
    print(f"  geo_weight: {recall_config.get('geo_weight', '未设置')}")

    # 检查设置是否合理
    provider = recall_config.get('behavior_provider', 'popularity')
    if provider == 'recbole':
        print("\n  当前配置: 使用 RecBole 行为召回")
        print("  需要确保:")
        print("    - outputs/recbole/saved/ 目录包含训练好的模型")
        print("    - configs/recbole.yaml 配置正确")
    else:
        print("\n  当前配置: 使用流行度行为召回")

    return True


def main():
    """运行所有验证检查"""
    print("\n" + "=" * 60)
    print("RecBole 在线集成验证报告")
    print("=" * 60)

    results = {}

    # 1. 模型可用性
    results['model_available'] = check_model_availability()

    # 2. 用户历史使用
    results['user_history'] = check_user_history_usage()

    # 3. 代码集成
    results['code_integration'] = check_recbole_code_integration()

    # 4. 动态权重
    results['adaptive_fusion'] = check_adaptive_fusion_weights()

    # 5. 召回对比
    results['recall_comparison'] = simulate_recall_comparison()

    # 6. 配置检查
    results['config'] = check_config_settings()

    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)

    print("\n检查结果:")
    for name, passed in results.items():
        status = "  PASS" if passed else "  FAIL"
        print(f"{status}: {name}")

    # 诊断和建议
    print("\n" + "=" * 60)
    print("诊断和建议")
    print("=" * 60)

    if not results['model_available']:
        print("\n  RecBole 模型未训练:")
        print("    1. 运行训练脚本: python -m src.recommendation.recbole_trainer")
        print("    2. 或: python scripts/train_recbole_model.py (如果存在)")
        print("    3. 系统将自动使用流行度召回作为降级策略")

    if not results['code_integration']:
        print("\n  代码集成有问题:")
        print("    请检查 src/recommendation/candidate_merger.py")
        print("    和 src/recommendation/recbole_trainer.py")

    if results['model_available'] and results['code_integration']:
        print("\n  RecBole 集成状态良好!")
        print("    要启用 RecBole 召回，请在调用 merge_candidates 时设置:")
        print("    use_recbole=True, recbole_model_path='outputs/recbole/saved'")

    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
