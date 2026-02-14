"""
验证生成的POI向量索引
检查向量数量、维度和质量
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def validate_embeddings(
    emb_file: str,
    meta_file: str,
    expected_count: int = None
):
    """
    验证向量文件

    Args:
        emb_file: 向量文件路径
        meta_file: 元数据文件路径
        expected_count: 期望的POI数量
    """
    print("=" * 60)
    print("POI 向量验证工具")
    print("=" * 60)
    print()

    # 检查文件存在
    if not os.path.exists(emb_file):
        print(f"❌ 向量文件不存在: {emb_file}")
        return False

    if not os.path.exists(meta_file):
        print(f"❌ 元数据文件不存在: {meta_file}")
        return False

    # 加载数据
    print(f"加载向量文件: {emb_file}")
    embeddings = np.load(emb_file)
    print(f"  ✓ 加载成功")

    print(f"加载元数据文件: {meta_file}")
    metadata = pd.read_csv(meta_file)
    print(f"  ✓ 加载成功")
    print()

    # 基本信息
    print("基本信息:")
    print(f"  向量形状: {embeddings.shape}")
    print(f"  向量维度: {embeddings.shape[1]}")
    print(f"  向量数量: {embeddings.shape[0]}")
    print(f"  元数据行数: {len(metadata)}")
    print()

    # 检查数量一致性
    if embeddings.shape[0] != len(metadata):
        print(f"❌ 错误: 向量数量 ({embeddings.shape[0]}) 与元数据行数 ({len(metadata)}) 不一致")
        return False

    # 检查期望数量
    if expected_count is not None:
        if embeddings.shape[0] == expected_count:
            print(f"✓ 向量数量符合预期: {expected_count}")
        else:
            print(f"⚠️  警告: 向量数量 ({embeddings.shape[0]}) 与预期 ({expected_count}) 不一致")

    print()

    # 统计信息
    print("向量统计:")
    print(f"  数据类型: {embeddings.dtype}")
    print(f"  最小值: {embeddings.min():.6f}")
    print(f"  最大值: {embeddings.max():.6f}")
    print(f"  均值: {embeddings.mean():.6f}")
    print(f"  标准差: {embeddings.std():.6f}")
    print(f"  范数（每行）:")
    row_norms = np.linalg.norm(embeddings, axis=1)
    print(f"    最小: {row_norms.min():.6f}")
    print(f"    最大: {row_norms.max():.6f}")
    print(f"    平均: {row_norms.mean():.6f}")
    print()

    # 检查是否归一化
    if row_norms.mean() > 0.99 and row_norms.mean() < 1.01:
        print("✓ 向量已归一化（适合内积检索）")
    else:
        print("⚠️  向量可能未归一化")

    print()

    # 检查NaN和Inf
    nan_count = np.isnan(embeddings).sum()
    inf_count = np.isinf(embeddings).sum()
    if nan_count > 0:
        print(f"❌ 发现 {nan_count} 个NaN值")
        return False
    if inf_count > 0:
        print(f"❌ 发现 {inf_count} 个Inf值")
        return False
    print("✓ 无NaN或Inf值")

    print()

    # 元数据检查
    print("元数据检查:")
    required_columns = ['poi_id', 'name']
    missing_columns = [col for col in required_columns if col not in metadata.columns]
    if missing_columns:
        print(f"⚠️  缺少列: {missing_columns}")
    else:
        print(f"✓ 包含必需列: {required_columns}")

    # 显示列名
    print(f"  所有列: {list(metadata.columns)}")
    print()

    # 示例数据
    print("示例数据 (前3条):")
    for i in range(min(3, len(metadata))):
        row = metadata.iloc[i]
        print(f"  [{i}] POI_ID: {row.get('poi_id', 'N/A')}, 名称: {row.get('name', 'N/A')}")
        if 'city' in row:
            print(f"      城市: {row['city']}")
        if 'province' in row:
            print(f"      省份: {row['province']}")

    print()
    print("=" * 60)
    print("✓ 验证通过!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='验证POI向量索引')
    parser.add_argument('--emb-file', type=str,
                        default='outputs/emb/poi_emb_qwen3.npy',
                        help='向量文件路径')
    parser.add_argument('--meta-file', type=str,
                        default='outputs/emb/poi_meta_qwen3.csv',
                        help='元数据文件路径')
    parser.add_argument('--expected-count', type=int,
                        help='期望的POI数量')
    args = parser.parse_args()

    validate_embeddings(
        emb_file=args.emb_file,
        meta_file=args.meta_file,
        expected_count=args.expected_count
    )
