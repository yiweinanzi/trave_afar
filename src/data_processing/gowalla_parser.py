"""
Gowalla签到数据解析器
解析Gowalla数据集，提取用户签到行为用于增强行为召回

数据格式：
loc-gowalla_edges.txt.gz - 社交关系边
loc-gowalla_totalCheckins.txt.gz - 签到记录

签到记录格式: 用户ID, 签到时间, 纬度, 经度, 地点ID

用途：
1. 增强行为召回数据
2. 用户-地点交互矩阵
3. 序列推荐训练
"""
import gzip
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from datetime import datetime

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOWALLA_CHECKINS = PROJECT_ROOT / "data" / "external" / "loc-gowalla_totalCheckins.txt.gz"
GOWALLA_EDGES = PROJECT_ROOT / "data" / "external" / "loc-gowalla_edges.txt.gz"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class GowallaParser:
    """Gowalla签到数据解析器"""

    def __init__(self, checkins_file: Path = None, edges_file: Path = None):
        """
        初始化解析器

        Args:
            checkins_file: 签到记录文件
            edges_file: 社交关系边文件
        """
        self.checkins_file = Path(checkins_file) if checkins_file else GOWALLA_CHECKINS
        self.edges_file = Path(edges_file) if edges_file else GOWALLA_EDGES

    def parse_checkins(self, max_lines: int = None) -> pd.DataFrame:
        """
        解析签到记录

        Args:
            max_lines: 最大读取行数（用于测试）

        Returns:
            签到记录DataFrame (user_id, timestamp, lat, lon, poi_id)
        """
        if not self.checkins_file.exists():
            logger.error(f"签到文件不存在: {self.checkins_file}")
            return pd.DataFrame()

        logger.info(f"正在解析签到记录...")
        records = []

        with gzip.open(self.checkins_file, 'rt') as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break

                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    user_id = parts[0]
                    iso_time = parts[1]  # ISO 8601 format
                    lat = float(parts[2])
                    lon = float(parts[3])
                    poi_id = parts[4]

                    # 转换ISO时间戳
                    try:
                        from dateutil import parser as date_parser
                        dt = date_parser.parse(iso_time)
                        timestamp = int(dt.timestamp())
                    except:
                        continue

                    records.append({
                        'user_id': f"G{user_id}",  # 添加前缀区分
                        'poi_id': f"G{poi_id}",
                        'timestamp': timestamp,
                        'datetime': dt,
                        'lat': lat,
                        'lon': lon,
                        'action': 'visit'
                    })

                if (i + 1) % 100000 == 0:
                    logger.info(f"已处理 {i + 1} 行")

        df = pd.DataFrame(records)
        logger.info(f"解析完成: {len(df)} 条签到记录")

        return df

    def parse_edges(self, max_lines: int = None) -> pd.DataFrame:
        """
        解析社交关系边

        Args:
            max_lines: 最大读取行数

        Returns:
            社交关系DataFrame (user_id, friend_id)
        """
        if not self.edges_file.exists():
            logger.warning(f"社交关系文件不存在: {self.edges_file}")
            return pd.DataFrame()

        logger.info(f"正在解析社交关系...")
        records = []

        with gzip.open(self.edges_file, 'rt') as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break

                parts = line.strip().split()
                if len(parts) >= 2:
                    user_id = parts[0]
                    friend_id = parts[1]

                    records.append({
                        'user_id': f"G{user_id}",
                        'friend_id': f"G{friend_id}"
                    })

        df = pd.DataFrame(records)
        logger.info(f"解析完成: {len(df)} 条社交关系")

        return df

    def extract_user_sequences(self, checkins_df: pd.DataFrame,
                               min_visits: int = 5) -> pd.DataFrame:
        """
        提取用户访问序列

        Args:
            checkins_df: 签到记录DataFrame
            min_visits: 最小访问次数

        Returns:
            用户序列DataFrame
        """
        # 统计每个用户的访问次数
        user_counts = checkins_df['user_id'].value_counts()
        active_users = user_counts[user_counts >= min_visits].index

        # 过滤活跃用户
        active_df = checkins_df[checkins_df['user_id'].isin(active_users)]

        # 按用户和时间排序
        active_df = active_df.sort_values(['user_id', 'timestamp'])

        logger.info(f"活跃用户数: {len(active_users)}")
        logger.info(f"活跃用户签到数: {len(active_df)}")

        return active_df

    def create_poi_profile(self, checkins_df: pd.DataFrame) -> pd.DataFrame:
        """
        创建POI画像

        Args:
            checkins_df: 签到记录DataFrame

        Returns:
            POI画像DataFrame
        """
        poi_stats = checkins_df.groupby('poi_id').agg({
            'user_id': 'nunique',  # 独立用户数
            'timestamp': 'count',  # 总签到数
            'lat': 'first',
            'lon': 'first'
        }).reset_index()

        poi_stats.columns = ['poi_id', 'unique_users', 'total_checkins', 'lat', 'lon']

        # 计算热度
        poi_stats['popularity'] = poi_stats['total_checkins'] / poi_stats['unique_users']

        logger.info(f"POI数量: {len(poi_stats)}")

        return poi_stats

    def export_to_csv(self, output_dir: Path = None, max_lines: int = None):
        """
        导出数据到CSV

        Args:
            output_dir: 输出目录
            max_lines: 最大读取行数
        """
        if output_dir is None:
            output_dir = PROJECT_ROOT / "data" / "gowalla"

        output_dir.mkdir(parents=True, exist_ok=True)

        # 解析签到记录
        checkins_df = self.parse_checkins(max_lines)

        if not checkins_df.empty:
            # 导出原始签到记录
            checkins_file = output_dir / "gowalla_checkins.csv"
            checkins_df.to_csv(checkins_file, index=False)
            logger.info(f"导出签到记录: {checkins_file}")

            # 导出用户序列
            sequences_df = self.extract_user_sequences(checkins_df)
            sequences_file = output_dir / "gowalla_sequences.csv"
            sequences_df.to_csv(sequences_file, index=False)
            logger.info(f"导出用户序列: {sequences_file}")

            # 导出POI画像
            poi_df = self.create_poi_profile(checkins_df)
            poi_file = output_dir / "gowalla_pois.csv"
            poi_df.to_csv(poi_file, index=False)
            logger.info(f"导出POI画像: {poi_file}")

        # 解析社交关系
        edges_df = self.parse_edges(max_lines)
        if not edges_df.empty:
            edges_file = output_dir / "gowalla_edges.csv"
            edges_df.to_csv(edges_file, index=False)
            logger.info(f"导出社交关系: {edges_file}")

    def merge_with_existing_events(self, checkins_df: pd.DataFrame,
                                   existing_events_csv: Path = None) -> pd.DataFrame:
        """
        合并Gowalla签到数据到现有user_events.csv

        Args:
            checkins_df: Gowalla签到记录
            existing_events_csv: 现有事件文件

        Returns:
            合并后的DataFrame
        """
        if existing_events_csv is None:
            existing_events_csv = PROJECT_ROOT / "data" / "user_events.csv"

        if not existing_events_csv.exists():
            logger.warning("现有事件文件不存在，只使用Gowalla数据")
            return checkins_df[['user_id', 'poi_id', 'timestamp', 'action']]

        # 读取现有数据
        existing_df = pd.read_csv(existing_events_csv)

        # 合并
        merged_df = pd.concat([
            existing_df,
            checkins_df[['user_id', 'poi_id', 'timestamp', 'action']]
        ], ignore_index=True)

        logger.info(f"合并后事件数: {len(merged_df)}")

        return merged_df


def analyze_gowalla_data(max_lines: int = 100000):
    """分析Gowalla数据集"""
    parser = GowallaParser()

    print("\n正在解析Gowalla签到数据...")
    checkins_df = parser.parse_checkins(max_lines=max_lines)

    if checkins_df.empty:
        print("没有签到数据")
        return

    # 统计信息
    stats = {
        'total_checkins': len(checkins_df),
        'unique_users': checkins_df['user_id'].nunique(),
        'unique_pois': checkins_df['poi_id'].nunique(),
        'avg_checkins_per_user': checkins_df.groupby('user_id').size().mean(),
        'time_range': (
            checkins_df['datetime'].min(),
            checkins_df['datetime'].max()
        )
    }

    print(f"\nGowalla数据分析:")
    print(f"  总签到数: {stats['total_checkins']:,}")
    print(f"  独立用户数: {stats['unique_users']:,}")
    print(f"  独立地点数: {stats['unique_pois']:,}")
    print(f"  平均每用户签到: {stats['avg_checkins_per_user']:.1f} 次")
    print(f"  时间范围: {stats['time_range'][0]} ~ {stats['time_range'][1]}")

    # 用户活跃度分布
    user_activity = checkins_df.groupby('user_id').size()
    print(f"\n用户活跃度分布:")
    print(f"  中位数: {user_activity.median():.1f}")
    print(f"  75分位: {user_activity.quantile(0.75):.1f}")
    print(f"  90分位: {user_activity.quantile(0.90):.1f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Gowalla签到数据解析器')
    parser.add_argument('--action', choices=['analyze', 'export', 'merge'],
                        default='analyze', help='操作类型')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录')
    parser.add_argument('--max-lines', type=int, default=1000000,
                        help='最大读取行数（用于测试）')

    args = parser.parse_args()

    if args.action == 'analyze':
        analyze_gowalla_data(max_lines=args.max_lines)

    elif args.action == 'export':
        gowalla = GowallaParser()
        output_dir = Path(args.output_dir) if args.output_dir else None
        gowalla.export_to_csv(output_dir, max_lines=args.max_lines)

    elif args.action == 'merge':
        gowalla = GowallaParser()
        checkins_df = gowalla.parse_checkins(max_lines=args.max_lines)
        merged_df = gowalla.merge_with_existing_events(checkins_df)

        # 保存合并数据
        output = Path(args.output_dir) / "merged_user_events.csv" if args.output_dir else PROJECT_ROOT / "data" / "merged_user_events.csv"
        merged_df.to_csv(output, index=False)
        print(f"导出合并数据: {output}")
