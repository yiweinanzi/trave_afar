"""
路线模板解析器
从default_route.sql中提取预设路线模板，用于冷启动和快速推荐

用途：
1. 解析SQL路线数据为系统可用格式
2. 将路线模板转换为JSON
3. 用于冷启动推荐和路线规划基线
"""
import re
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROUTE_SQL = PROJECT_ROOT / "data" / "default_route.sql"


def parse_route_sql(sql_file: Path = ROUTE_SQL) -> List[Dict]:
    """
    解析default_route.sql文件，提取路线模板

    Returns:
        路线模板列表
    """
    with open(sql_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取所有 INSERT 语句
    pattern = r"INSERT INTO `default_route` VALUES \((\d+), (\d+), '([^']+)', (\d+)\);"
    matches = re.findall(pattern, content)

    print(f"找到 {len(matches)} 条路线模板")

    routes = []
    for idx, match in enumerate(matches):
        route_id, area, route_str, days = match
        route_id = int(route_id)
        area = int(area)
        days = int(days)

        # 按天分割路线
        day_routes = route_str.split(',')
        daily_routes = []
        for day_route in day_routes:
            # 移除多余空格
            day_route = day_route.strip()
            # 按破折号分割景点
            pois = [p.strip() for p in day_route.split('-') if p.strip()]
            daily_routes.append(pois)

        # 确定省份
        province = AREA_TO_PROVINCE.get(area, "未知")

        route = {
            'route_id': route_id,
            'area': area,
            'province': province,
            'days': days,
            'daily_routes': daily_routes,
            'raw_route': route_str,
            'poi_count': sum(len(day) for day in daily_routes)
        }
        routes.append(route)

    return routes


# 区域ID到省份的映射
AREA_TO_PROVINCE = {
    1: "新疆",
    2: "西藏",
    3: "云南",
    4: "四川",
    5: "甘肃",
    6: "宁夏",
    7: "内蒙古",
    8: "青海",
}


def route_to_poi_sequence(route: Dict, poi_df: pd.DataFrame) -> List[str]:
    """
    将路线中的景点名称匹配到poi_id

    Args:
        route: 路线模板
        poi_df: POI数据

    Returns:
        poi_id列表
    """
    poi_ids = []

    # 创建名称到poi_id的映射
    name_to_id = dict(zip(poi_df['name'], poi_df['poi_id']))

    for day_route in route['daily_routes']:
        for poi_name in day_route:
            # 精确匹配
            if poi_name in name_to_id:
                poi_ids.append(name_to_id[poi_name])
            else:
                # 模糊匹配
                for name, pid in name_to_id.items():
                    if poi_name in name or name in poi_name:
                        poi_ids.append(pid)
                        break
                else:
                    # 未找到，使用None占位
                    poi_ids.append(None)

    return poi_ids


def save_route_templates(routes: List[Dict], output_file: Path = None):
    """
    保存路线模板为JSON文件

    Args:
        routes: 路线模板列表
        output_file: 输出文件路径
    """
    if output_file is None:
        output_file = PROJECT_ROOT / "data" / "route_templates.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(routes, f, ensure_ascii=False, indent=2)

    print(f"✓ 路线模板已保存到 {output_file}")


def load_route_templates(input_file: Path = None) -> List[Dict]:
    """
    加载路线模板JSON文件

    Args:
        input_file: 输入文件路径

    Returns:
        路线模板列表
    """
    if input_file is None:
        input_file = PROJECT_ROOT / "data" / "route_templates.json"

    if not input_file.exists():
        return []

    with open(input_file, 'r', encoding='utf-8') as f:
        routes = json.load(f)

    return routes


def get_route_by_province(routes: List[Dict], province: str, days: int = None) -> List[Dict]:
    """
    根据省份筛选路线

    Args:
        routes: 路线模板列表
        province: 省份名称
        days: 天数筛选（可选）

    Returns:
        匹配的路线列表
    """
    filtered = [r for r in routes if r['province'] == province]

    if days is not None:
        filtered = [r for r in filtered if r['days'] == days]

    return filtered


def analyze_route_templates(routes: List[Dict]) -> Dict:
    """
    分析路线模板统计信息

    Args:
        routes: 路线模板列表

    Returns:
        统计信息字典
    """
    stats = {
        'total_routes': len(routes),
        'by_province': {},
        'by_days': {},
        'avg_poi_per_day': []
    }

    for route in routes:
        # 按省份统计
        province = route['province']
        stats['by_province'][province] = stats['by_province'].get(province, 0) + 1

        # 按天数统计
        days = route['days']
        stats['by_days'][days] = stats['by_days'].get(days, 0) + 1

        # 平均每天POI数量
        if days > 0:
            avg_poi = route['poi_count'] / days
            stats['avg_poi_per_day'].append(avg_poi)

    # 计算平均
    if stats['avg_poi_per_day']:
        stats['avg_poi_per_day'] = sum(stats['avg_poi_per_day']) / len(stats['avg_poi_per_day'])

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='路线模板解析器')
    parser.add_argument('--action', choices=['parse', 'analyze', 'match'],
                        default='parse', help='操作类型')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径')
    parser.add_argument('--province', type=str, default=None,
                        help='筛选省份')
    parser.add_argument('--days', type=int, default=None,
                        help='筛选天数')

    args = parser.parse_args()

    if args.action == 'parse':
        routes = parse_route_sql()
        save_route_templates(routes, Path(args.output) if args.output else None)

        # 打印统计
        stats = analyze_route_templates(routes)
        print(f"\n路线模板统计:")
        print(f"  总路线数: {stats['total_routes']}")
        print(f"  按省份分布:")
        for prov, count in sorted(stats['by_province'].items()):
            print(f"    {prov}: {count} 条")
        print(f"  按天数分布:")
        for days, count in sorted(stats['by_days'].items()):
            print(f"    {days}天: {count} 条")
        print(f"  平均每天POI数: {stats['avg_poi_per_day']:.2f}")

    elif args.action == 'analyze':
        routes = load_route_templates()
        stats = analyze_route_templates(routes)

        print(f"\n路线模板分析:")
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    elif args.action == 'match':
        # 匹配路线到POI
        routes = load_route_templates()
        poi_df = pd.read_csv(PROJECT_ROOT / "data" / "poi.csv")

        if args.province:
            routes = get_route_by_province(routes, args.province, args.days)

        print(f"\n匹配 {len(routes)} 条路线到POI:")
        for route in routes[:5]:  # 显示前5条
            poi_ids = route_to_poi_sequence(route, poi_df)
            matched = sum(1 for pid in poi_ids if pid is not None)
            print(f"  [{route['province']}] {route['days']}天: {matched}/{route['poi_count']} 匹配")
