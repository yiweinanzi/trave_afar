"""
Geolife GPS轨迹数据解析器
解析微软Geolife数据集，提取真实GPS轨迹用于训练/验证规划模型

Geolife数据集结构：
Geolife Trajectories 1.3/
├── Data/
│   ├── 000/
│   │   ├── Trajectory/
│   │   │   ├── 2008-10-02.COL (记录格式: 开始时间, 结束时间, 用户标签)
│   │   │   └── 2008-10-02.plt (轨迹点: 纬度, 经度, 0, 1, 2, 3, 日期, 时间)
│   │   └── labels.txt (用户交通方式标签)
│   └── ...

用途：
1. 学习真实��动模式
2. 训练/验证规划模型
3. 合成训练数据
"""
import os
import gzip
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEOLIFE_ZIP = PROJECT_ROOT / "data" / "external" / "Geolife Trajectories 1.3.zip"
# 实际解压后的目录名包含空格
GEOLIFE_DIR = PROJECT_ROOT / "data" / "external" / "Geolife Trajectories 1.3"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class GeolifeParser:
    """Geolife GPS轨迹解析器"""

    def __init__(self, data_dir: Path = None):
        """
        初始化解析器

        Args:
            data_dir: Geolife数据目录，如果未提供则尝试解压ZIP
        """
        self.data_dir = Path(data_dir) if data_dir else GEOLIFE_DIR
        self.users = []

        # 如果目录不存在，尝试解压ZIP
        if not self.data_dir.exists():
            self._extract_zip()

        # 扫描用户目录
        self._scan_users()

    def _extract_zip(self):
        """解压Geolife ZIP文件"""
        if not GEOLIFE_ZIP.exists():
            logger.error(f"Geolife ZIP文件不存在: {GEOLIFE_ZIP}")
            return

        logger.info(f"正在解压 {GEOLIFE_ZIP.name}...")
        with zipfile.ZipFile(GEOLIFE_ZIP, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir.parent)
        logger.info(f"解压完成: {self.data_dir}")

    def _scan_users(self):
        """扫描所有用户目录"""
        if not self.data_dir.exists():
            return

        data_path = self.data_dir / "Data"
        if data_path.exists():
            self.users = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
            logger.info(f"找到 {len(self.users)} 个用户")

    def list_user_trajectories(self, user_id: str) -> List[Path]:
        """
        列出指定用户的所有轨迹文件

        Args:
            user_id: 用户ID

        Returns:
            plt文件路径列表
        """
        traj_dir = self.data_dir / "Data" / user_id / "Trajectory"
        if not traj_dir.exists():
            return []

        return sorted(traj_dir.glob("*.plt"))

    def parse_plt_file(self, plt_file: Path) -> pd.DataFrame:
        """
        解析单个plt轨迹文件

        plt文件格式（跳过前6行）:
        纬度, 经度, 高度, 天数, 时间戳, 日期, 时间

        Args:
            plt_file: plt文件路径

        Returns:
            轨迹点DataFrame
        """
        try:
            # 跳过前6行，列名按实际格式
            df = pd.read_csv(plt_file, skiprows=6, header=None,
                           names=['lat', 'lon', 'alt', 'days', 'ignored', 'date', 'time'])

            # 转换坐标
            df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            df['lon'] = pd.to_numeric(df['lon'], errors='coerce')

            # 合并日期和时间
            df['date'] = df['date'].astype(str)
            df['time'] = df['time'].astype(str)
            df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')

            # 过滤无效数据
            df = df.dropna(subset=['lat', 'lon', 'timestamp'])

            # 计算时间差（秒）
            df = df.sort_values('timestamp')
            df['time_delta'] = df['timestamp'].diff().dt.total_seconds()

            # 计算距离（米，使用Haversine公式近似）
            import numpy as np
            df['distance'] = self._haversine_distance(
                df['lat'].shift(1), df['lon'].shift(1),
                df['lat'], df['lon']
            )

            # 计算速度（米/秒）
            df['speed'] = np.where(df['time_delta'] > 0, df['distance'] / df['time_delta'], 0)

            # 移除第一行（NaN）
            df = df.iloc[1:]

            return df

        except Exception as e:
            logger.error(f"解析文件失败 {plt_file.name}: {e}")
            return pd.DataFrame()

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """计算两点之间的Haversine距离（米）"""
        import numpy as np

        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        r = 6371000  # 地球半径（米）
        return c * r

    def parse_user_labels(self, user_id: str) -> pd.DataFrame:
        """
        解析用户交通方式标签

        Args:
            user_id: 用户ID

        Returns:
            标签DataFrame (start_time, end_time, transport_mode)
        """
        label_file = self.data_dir / "Data" / user_id / "labels.txt"
        if not label_file.exists():
            return pd.DataFrame()

        try:
            df = pd.read_csv(label_file, sep='\t', header=None,
                           names=['start_time', 'end_time', 'transport_mode'])

            df['start_time'] = pd.to_datetime(df['start_time'])
            df['end_time'] = pd.to_datetime(df['end_time'])

            return df

        except Exception as e:
            logger.error(f"解析标签失败 {user_id}: {e}")
            return pd.DataFrame()

    def extract_trip_segments(self, plt_df: pd.DataFrame,
                             min_distance: float = 500,
                             max_speed: float = 50) -> List[Dict]:
        """
        从轨迹中提取出行段

        Args:
            plt_df: 轨迹DataFrame
            min_distance: 最小距离（米）
            max_speed: 最大速度（米/秒，约180km/h）

        Returns:
            出行段列表
        """
        if plt_df.empty:
            return []

        segments = []
        current_segment = []

        for _, row in plt_df.iterrows():
            # 过滤异常数据
            if pd.isna(row['speed']) or row['speed'] > max_speed:
                if current_segment:
                    # 结束当前段
                    if len(current_segment) >= 2:
                        distance = sum(p['distance'] for p in current_segment)
                        if distance >= min_distance:
                            segments.append({
                                'start_time': current_segment[0]['timestamp'],
                                'end_time': current_segment[-1]['timestamp'],
                                'duration': (current_segment[-1]['timestamp'] -
                                          current_segment[0]['timestamp']).total_seconds(),
                                'distance': distance,
                                'points': [(p['lat'], p['lon']) for p in current_segment]
                            })
                    current_segment = []
                continue

            current_segment.append({
                'lat': row['lat'],
                'lon': row['lon'],
                'timestamp': row['timestamp'],
                'distance': row['distance']
            })

        # 处理最后一段
        if current_segment and len(current_segment) >= 2:
            distance = sum(p['distance'] for p in current_segment)
            if distance >= min_distance:
                segments.append({
                    'start_time': current_segment[0]['timestamp'],
                    'end_time': current_segment[-1]['timestamp'],
                    'duration': (current_segment[-1]['timestamp'] -
                              current_segment[0]['timestamp']).total_seconds(),
                    'distance': distance,
                    'points': [(p['lat'], p['lon']) for p in current_segment]
                })

        return segments

    def generate_training_samples(self, max_users: int = 100,
                                  max_trajectories_per_user: int = 10) -> List[Dict]:
        """
        生成训练样本（轨迹序列）

        Args:
            max_users: 最大用户数
            max_trajectories_per_user: 每个用户最大轨迹数

        Returns:
            训练样本列表
        """
        samples = []

        for user_id in self.users[:max_users]:
            plt_files = self.list_user_trajectories(user_id)

            for plt_file in plt_files[:max_trajectories_per_user]:
                df = self.parse_plt_file(plt_file)
                if df.empty:
                    continue

                segments = self.extract_trip_segments(df)

                for segment in segments:
                    samples.append({
                        'user_id': user_id,
                        'start_time': segment['start_time'],
                        'end_time': segment['end_time'],
                        'duration': segment['duration'],
                        'distance': segment['distance'],
                        'points': segment['points'][:100]  # 限制点数
                    })

        logger.info(f"生成 {len(samples)} 个训练样本")
        return samples

    def export_to_csv(self, output_file: Path = None):
        """
        导出轨迹数据到CSV

        Args:
            output_file: 输出文件路径
        """
        if output_file is None:
            output_file = PROJECT_ROOT / "data" / "geolife_trajectories.csv"

        samples = self.generate_training_samples()

        # 展开数据
        rows = []
        for sample in samples:
            for i, (lat, lon) in enumerate(sample['points']):
                rows.append({
                    'user_id': sample['user_id'],
                    'trip_id': f"{sample['user_id']}_{sample['start_time'].strftime('%Y%m%d%H%M%S')}",
                    'point_index': i,
                    'lat': lat,
                    'lon': lon
                })

        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        logger.info(f"导出 {len(df)} 个轨迹点到 {output_file}")


def analyze_geolife_data():
    """分析Geolife数据集"""
    parser = GeolifeParser()

    stats = {
        'total_users': len(parser.users),
        'sample_users': min(10, len(parser.users)),
        'trajectories_per_user': [],
        'total_points': 0
    }

    for user_id in parser.users[:10]:
        plt_files = parser.list_user_trajectories(user_id)
        stats['trajectories_per_user'].append(len(plt_files))

        for plt_file in plt_files[:3]:  # 只解析前3个
            df = parser.parse_plt_file(plt_file)
            stats['total_points'] += len(df)

    avg_traj = sum(stats['trajectories_per_user']) / len(stats['trajectories_per_user']) if stats['trajectories_per_user'] else 0

    print(f"\nGeolife数据集分析:")
    print(f"  总用户数: {stats['total_users']}")
    print(f"  采样用户数: {stats['sample_users']}")
    print(f"  平均每个用户轨迹数: {avg_traj:.1f}")
    print(f"  采样轨迹点总数: {stats['total_points']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Geolife GPS轨迹解析器')
    parser.add_argument('--action', choices=['analyze', 'export', 'sample'],
                        default='analyze', help='操作类型')
    parser.add_argument('--output', type=str, default=None, help='输出文件')
    parser.add_argument('--max-users', type=int, default=100, help='最大用户数')

    args = parser.parse_args()

    if args.action == 'analyze':
        analyze_geolife_data()

    elif args.action == 'export':
        geolife = GeolifeParser()
        output = Path(args.output) if args.output else None
        geolife.export_to_csv(output)

    elif args.action == 'sample':
        geolife = GeolifeParser()
        samples = geolife.generate_training_samples(max_users=args.max_users)

        output = Path(args.output) if args.output else PROJECT_ROOT / "data" / "geolife_samples.json"
        import json
        with open(output, 'w') as f:
            json.dump(samples, f, default=str, indent=2)

        print(f"导出 {len(samples)} 个样本到 {output}")
