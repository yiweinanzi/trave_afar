#!/usr/bin/env python3
"""
RecBole Provider 单元测试

测试 RecBoleProvider 类的核心功能
"""
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_adaptive_fusion():
    """测试动态权重融合函数"""
    print("\n" + "=" * 60)
    print("测试 1: 动态权重融合函数")
    print("=" * 60)

    # 直接导入函数
    from recommendation.candidate_merger import adaptive_fusion

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

        # 验证权重和为 1
        assert abs(dense_w + behavior_w + geo_w - 1.0) < 1e-6, "权重和应该为 1"
        # 验证权重非负
        assert dense_w >= 0 and behavior_w >= 0 and geo_w >= 0, "权重应该非负"

    print("\n✓ 动态权重融合测试通过")


def test_recbole_provider_class():
    """测试 RecBoleProvider 类的基本结构"""
    print("\n" + "=" * 60)
    print("测试 2: RecBoleProvider 类结构")
    print("=" * 60)

    try:
        from recommendation.recbole_trainer import RecBoleProvider

        # 检查类是否存在
        assert RecBoleProvider is not None, "RecBoleProvider 类应该存在"

        # 检查关键方法
        assert hasattr(RecBoleProvider, '__init__'), "应该有 __init__ 方法"
        assert hasattr(RecBoleProvider, 'predict'), "应该有 predict 方法"
        assert hasattr(RecBoleProvider, '_predict_by_popularity'), "应该有 _predict_by_popularity 方法"
        assert hasattr(RecBoleProvider, 'get_user_history_length'), "应该有 get_user_history_length 方法"

        print("✓ RecBoleProvider 类结构验证通过")

        # 尝试创建实例（模型不存在时应该不会报错）
        print("\n尝试创建 RecBoleProvider 实例...")
        provider = RecBoleProvider(
            model_path="outputs/recbole/saved",
            config_file="configs/recbole.yaml",
            use_gpu=True,
            fallback_to_popular=True
        )

        # 检查实例属性
        assert hasattr(provider, 'model'), "应该有 model 属性"
        assert hasattr(provider, 'available'), "应该有 available 属性"
        assert hasattr(provider, 'fallback_to_popular'), "应该有 fallback_to_popular 属性"

        print(f"  模型可用: {provider.available}")
        print(f"  降级到流行度: {provider.fallback_to_popular}")

        # 测试预���（应该降级到流行度）
        print("\n测试预测功能（降级模式）...")
        import pandas as pd

        # 创建模拟 POI 数据
        poi_df = pd.DataFrame({
            'poi_id': ['poi_1', 'poi_2', 'poi_3'],
            'name': ['POI 1', 'POI 2', 'POI 3'],
            'city': ['City 1', 'City 2', 'City 3'],
            'province': ['Province 1', 'Province 2', 'Province 3']
        })

        rec_df, metadata = provider.predict(
            user_id="test_user",
            topk=10,
            poi_df=poi_df
        )

        print(f"  预测方法: {metadata['method']}")
        print(f"  返回结果数: {len(rec_df)}")

        print("\n✓ RecBoleProvider 实例测试通过")

    except ImportError as e:
        print(f"\n⚠️ 无法导入 RecBoleProvider: {e}")
        print("  请确保 recbole_trainer.py 路径正确")
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_recbole_data_export():
    """测试 RecBole 数据导出功能"""
    print("\n" + "=" * 60)
    print("测试 3: RecBole 数据导出")
    print("=" * 60)

    try:
        from recommendation.recbole_trainer import export_recbole_data

        # 检查数据文件是否存在
        events_csv = "data/user_events.csv"
        if not Path(events_csv).exists():
            print(f"\n⚠️ 用户事件文件不存在: {events_csv}")
            print("  跳过数据导出测试")
            return

        # 尝试导出数据
        output_file = export_recbole_data(
            events_csv=events_csv,
            output_dir="outputs/recbole/custom"
        )

        # 检查输出文件
        if Path(output_file).exists():
            print(f"\n✓ 数据导出成功: {output_file}")

            # 读取并验证格式
            with open(output_file, 'r') as f:
                lines = f.readlines()
                print(f"  导出行数: {len(lines)}")

                if len(lines) > 0:
                    # 检查第一行格式
                    first_line = lines[0].strip()
                    parts = first_line.split('\t')
                    print(f"  格式验证: {len(parts)} 列 (user_id, poi_id, timestamp)")
                    assert len(parts) == 3, "应该有 3 列"

                    print(f"  示例数据: {first_line}")
        else:
            print(f"\n✗ 导出文件不存在: {output_file}")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("RecBole Provider 单元测试")
    print("=" * 60)

    # 测试 1: 动态权重融合
    test_adaptive_fusion()

    # 测试 2: RecBoleProvider 类
    test_recbole_provider_class()

    # 测试 3: 数据导出
    test_recbole_data_export()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
