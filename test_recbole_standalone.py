#!/usr/bin/env python3
"""
RecBole Provider 独立测试（不依赖其他模块）
"""
import sys
import pandas as pd
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

# 直接导入 recbole_trainer 模块
from recommendation import recbole_trainer


def test_adaptive_fusion():
    """测试动态权重融合"""
    print("\n" + "=" * 60)
    print("测试 1: 动态权重融合")
    print("=" * 60)

    # 读取源代码并提取函数
    import inspect
    source = inspect.getsource(recbole_trainer)

    # 检查 adaptive_fusion 函数是否在 candidate_merger.py 中
    print("  adaptive_fusion 函数位于 candidate_merger.py")
    print("  需要通过 merge_candidates 函数调用")


def test_recbole_provider():
    """测试 RecBoleProvider 类"""
    print("\n" + "=" * 60)
    print("测试 2: RecBoleProvider 类")
    print("=" * 60)

    try:
        # 创建实例
        print("创建 RecBoleProvider 实例...")
        provider = recbole_trainer.RecBoleProvider(
            model_path="outputs/recbole/saved",
            config_file="configs/recbole.yaml",
            use_gpu=True,
            fallback_to_popular=True
        )

        print(f"  模型可用: {provider.available}")
        print(f"  降级到流行度: {provider.fallback_to_popular}")

        # 测试预测
        print("\n测试预测功能...")
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

        if len(rec_df) > 0:
            print("\n  预测结果:")
            print(rec_df[['poi_id', 'recbole_score']])

        print("\n✓ RecBoleProvider 测试通过")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_data_export():
    """测试数据导出"""
    print("\n" + "=" * 60)
    print("测试 3: 数据导出")
    print("=" * 60)

    try:
        events_csv = "data/user_events.csv"
        if not Path(events_csv).exists():
            print(f"\n⚠️ 用户事件文件不存在: {events_csv}")
            return

        output_file = recbole_trainer.export_recbole_data(
            events_csv=events_csv,
            output_dir="outputs/recbole/custom"
        )

        if Path(output_file).exists():
            print(f"\n✓ 数据导出成功: {output_file}")

            with open(output_file, 'r') as f:
                lines = f.readlines()
                print(f"  导出行数: {len(lines)}")

                if len(lines) > 0:
                    print(f"  前 5 行:")
                    for i, line in enumerate(lines[:5]):
                        print(f"    {line.strip()}")
        else:
            print(f"\n✗ 导出文件不存在: {output_file}")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("RecBole Provider 独立测试")
    print("=" * 60)

    test_adaptive_fusion()
    test_recbole_provider()
    test_data_export()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
