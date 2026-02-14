"""
Solomon VRPTW基准测试数据解析器
解析Solomon基准测试集，用于验证规划算法性能

Solomon数据集格式：
- 100个客户问题 (C1, C2, R1, R2, RC1, RC2)
- 每个文件包含：客户坐标、需求、时间窗、服务时间

用途：
1. VRPTW算法基准测试
2. 消融实验对比
3. 论文/简历可展示的基准测试

数据格式:
VEHICLE
NUMBER CAPACITY
CUSTOMER
CUST NO. XCOORD. YCOORD. DEMAND READY TIME DUE DATE SERVICE TIME
"""
import re
from pathlib import Path
from typing import Dict, List, Tuple
import logging

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOLOMON_ZIP = PROJECT_ROOT / "data" / "external" / "homberger_1000_customer_instances.zip"
SOLOMON_DIR = PROJECT_ROOT / "data" / "external" / "solomon"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SolomonParser:
    """Solomon VRPTW基准数据解析器"""

    def __init__(self, data_dir: Path = None):
        """
        初始化解析器

        Args:
            data_dir: Solomon数据目录
        """
        self.data_dir = Path(data_dir) if data_dir else SOLOMON_DIR

        # 如果目录不存在，尝试解压ZIP
        if not self.data_dir.exists():
            self._extract_zip()

        # 扫描数据文件
        self.data_files = list(self.data_dir.glob("*.txt")) + list(self.data_dir.glob("*.TXT")) + list(self.data_dir.glob("*.vrp"))
        logger.info(f"找到 {len(self.data_files)} 个数据文件")

    def _extract_zip(self):
        """解压Solomon ZIP文件"""
        import zipfile

        if not SOLOMON_ZIP.exists():
            logger.warning(f"Solomon ZIP文件不存在: {SOLOMON_ZIP}")
            return

        logger.info(f"正在解压 {SOLOMON_ZIP.name}...")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(SOLOMON_ZIP, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            logger.info(f"解压完成: {self.data_dir}")

        except Exception as e:
            logger.error(f"解压失败: {e}")

    def parse_solomon_file(self, file_path: Path) -> Dict:
        """
        解析单个Solomon数据文件

        Args:
            file_path: 文件路径

        Returns:
            问题实例字典
        """
        customers = []
        vehicle_info = {}
        depot = None

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            section = None
            for line in lines:
                line = line.strip()

                if not line or line.startswith(':'):
                    # 检测节
                    if 'VEHICLE' in line.upper():
                        section = 'vehicle'
                    elif 'CUSTOMER' in line.upper():
                        section = 'customer'
                    continue

                parts = line.split()

                if section == 'vehicle' and len(parts) >= 3:
                    vehicle_info = {
                        'number': int(parts[0]),
                        'capacity': int(parts[1]) if len(parts) > 1 else 200
                    }

                elif section == 'customer' or section is None:
                    # 客户数据: ID X Y DEMAND READY_TIME DUE_DATE SERVICE_TIME
                    if len(parts) >= 7:
                        try:
                            cust_id = int(parts[0])
                            x = float(parts[1])
                            y = float(parts[2])
                            demand = float(parts[3])
                            ready_time = float(parts[4])
                            due_date = float(parts[5])
                            service_time = float(parts[6])

                            customer = {
                                'id': cust_id,
                                'x': x,
                                'y': y,
                                'demand': demand,
                                'ready_time': ready_time,
                                'due_date': due_date,
                                'service_time': service_time
                            }

                            if cust_id == 0:
                                depot = customer
                            else:
                                customers.append(customer)

                        except ValueError:
                            continue

            return {
                'file_name': file_path.stem,
                'vehicle': vehicle_info,
                'depot': depot,
                'customers': customers,
                'num_customers': len(customers)
            }

        except Exception as e:
            logger.error(f"解析文件失败 {file_path.name}: {e}")
            return {}

    def get_benchmark_instances(self) -> List[Dict]:
        """
        获取所有基准测试实例

        Returns:
            问题实例列表
        """
        instances = []

        for file_path in self.data_files:
            instance = self.parse_solomon_file(file_path)
            if instance and instance['num_customers'] > 0:
                instances.append(instance)

        logger.info(f"解析完成: {len(instances)} 个实例")
        return instances

    def export_to_dataframe(self, instances: List[Dict] = None) -> pd.DataFrame:
        """
        将实例导出为DataFrame

        Args:
            instances: 实例列表，如果为None则自动加载

        Returns:
            DataFrame with columns:
            instance_name, customer_id, x, y, demand, ready_time, due_date, service_time
        """
        if instances is None:
            instances = self.get_benchmark_instances()

        rows = []
        for instance in instances:
            for cust in instance['customers']:
                rows.append({
                    'instance_name': instance['file_name'],
                    'num_customers': instance['num_customers'],
                    'vehicle_capacity': instance['vehicle'].get('capacity', 200),
                    'customer_id': cust['id'],
                    'x': cust['x'],
                    'y': cust['y'],
                    'demand': cust['demand'],
                    'ready_time': cust['ready_time'],
                    'due_date': cust['due_date'],
                    'service_time': cust['service_time']
                })

        return pd.DataFrame(rows)

    def calculate_distance_matrix(self, instance: Dict) -> List[List[float]]:
        """
        计算距离矩阵（欧几里得距离）

        Args:
            instance: 问题实例

        Returns:
            距离矩阵
        """
        # 包含depot的所有点
        all_points = [instance['depot']] + instance['customers']
        n = len(all_points)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = all_points[i]['x'] - all_points[j]['x']
                    dy = all_points[i]['y'] - all_points[j]['y']
                    matrix[i][j] = (dx ** 2 + dy ** 2) ** 0.5

        return matrix


def analyze_solomon_instances():
    """分析Solomon基准实例"""
    parser = SolomonParser()
    instances = parser.get_benchmark_instances()

    if not instances:
        print("没有找到Solomon实例")
        return

    # 统计信息
    stats = {
        'total_instances': len(instances),
        'customer_counts': [inst['num_customers'] for inst in instances],
        'capacity_values': [inst['vehicle'].get('capacity', 200) for inst in instances]
    }

    print(f"\nSolomon基准实例分析:")
    print(f"  总实例数: {stats['total_instances']}")
    print(f"  客户数范围: {min(stats['customer_counts'])} - {max(stats['customer_counts'])}")
    print(f"  平均客户数: {sum(stats['customer_counts']) / len(stats['customer_counts']):.1f}")
    print(f"  车辆容量: {set(stats['capacity_values'])}")

    # 按客户数分组
    from collections import Counter
    size_dist = Counter(inst['num_customers'] for inst in instances)
    print(f"\n实例规模分布:")
    for size, count in sorted(size_dist.items()):
        print(f"  {size} 客户: {count} 个实例")

    # 显示示例
    print(f"\n示例实例:")
    for inst in instances[:3]:
        print(f"  {inst['file_name']}: {inst['num_customers']} 客户, 容量 {inst['vehicle'].get('capacity', 200)}")


def export_solomon_benchmark(output_file: Path = None):
    """
    导出Solomon基准数据为CSV

    Args:
        output_file: 输出文件路径
    """
    if output_file is None:
        output_file = PROJECT_ROOT / "data" / "solomon_benchmark.csv"

    parser = SolomonParser()
    df = parser.export_to_dataframe()

    if not df.empty:
        df.to_csv(output_file, index=False)
        logger.info(f"导出基准数据: {output_file}")

    return df


def create_test_instances_from_poi(poi_df: pd.DataFrame,
                                   province: str = "新疆",
                                   num_customers: int = 50) -> Dict:
    """
    从现有POI创建VRPTW测试实例

    Args:
        poi_df: POI DataFrame
        province: 省份筛选
        num_customers: 客户数量

    Returns:
        VRPTW实例字典
    """
    # 筛选省份POI
    province_pois = poi_df[poi_df['province'] == province].head(num_customers + 1)

    if len(province_pois) < 2:
        logger.warning(f"省份 {province} POI数量不足")
        return {}

    # 使用第一个POI作为depot
    depot = province_pois.iloc[0]
    customers = []

    for _, poi in province_pois.iloc[1:].iterrows():
        customers.append({
            'id': poi['poi_id'],
            'x': poi['lon'],
            'y': poi['lat'],
            'demand': 1,  # 默认需求
            'ready_time': poi.get('open_min', 480),
            'due_date': poi.get('close_min', 1140),
            'service_time': poi.get('stay_min', 90)
        })

    instance = {
        'file_name': f"{province}_poi_{num_customers}",
        'vehicle': {
            'number': 5,
            'capacity': 20
        },
        'depot': {
            'id': depot['poi_id'],
            'x': depot['lon'],
            'y': depot['lat']
        },
        'customers': customers,
        'num_customers': len(customers)
    }

    return instance


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Solomon VRPTW基准解析器')
    parser.add_argument('--action', choices=['analyze', 'export', 'create-poi'],
                        default='analyze', help='操作类型')
    parser.add_argument('--output', type=str, default=None, help='输出文件')
    parser.add_argument('--province', type=str, default='新疆', help='省份')

    args = parser.parse_args()

    if args.action == 'analyze':
        analyze_solomon_instances()

    elif args.action == 'export':
        output = Path(args.output) if args.output else None
        export_solomon_benchmark(output)

    elif args.action == 'create-poi':
        poi_df = pd.read_csv(PROJECT_ROOT / "data" / "poi.csv")
        instance = create_test_instances_from_poi(poi_df, args.province)

        print(f"\nPOI VRPTW实例:")
        print(f"  文件名: {instance['file_name']}")
        print(f"  客户数: {instance['num_customers']}")
        print(f"  车辆数: {instance['vehicle']['number']}")
        print(f"  车辆容量: {instance['vehicle']['capacity']}")
