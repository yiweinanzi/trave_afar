#!/usr/bin/env python
"""
数据处理统一入口脚本
一键处理所有外部数据源

使用方法:
    python scripts/process_all_data.py --all
    python scripts/process_all_data.py --route --geolife
    python scripts/process_all_data.py --list
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
from datetime import datetime

from data_processing.route_template import (
    parse_route_sql, save_route_templates, analyze_route_templates
)
from data_processing.geolife_parser import GeolifeParser, analyze_geolife_data
from data_processing.gowalla_parser import GowallaParser, analyze_gowalla_data
from data_processing.yelp_parser import YelpParser, analyze_yelp_data
from data_processing.solomon_parser import SolomonParser, analyze_solomon_instances


def process_route_templates():
    """处理路线模板"""
    print("\n" + "="*50)
    print("处理路线模板数据...")
    print("="*50)

    routes = parse_route_sql()
    save_route_templates(routes)

    stats = analyze_route_templates(routes)
    print(f"\n✓ 路线模板处理完成:")
    print(f"  总路线数: {stats['total_routes']}")
    print(f"  省份数: {len(stats['by_province'])}")

    return routes


def process_geolife(max_users=50):
    """处理Geolife轨迹数据"""
    print("\n" + "="*50)
    print("处理Geolife GPS轨迹数据...")
    print("="*50)

    parser = GeolifeParser()
    samples = parser.generate_training_samples(max_users=max_users)

    # 导出
    output_file = PROJECT_ROOT / "data" / "geolife_samples.json"
    import json
    with open(output_file, 'w') as f:
        json.dump(samples, f, default=str, indent=2)

    print(f"\n✓ Geolife数据处理完成:")
    print(f"  样本数: {len(samples)}")
    print(f"  输出: {output_file}")

    return samples


def process_gowalla(max_lines=1000000):
    """处理Gowalla签到数据"""
    print("\n" + "="*50)
    print("处理Gowalla签到数据...")
    print("="*50)

    parser = GowallaParser()
    output_dir = PROJECT_ROOT / "data" / "gowalla"
    parser.export_to_csv(output_dir, max_lines=max_lines)

    print(f"\n✓ Gowalla数据处理完成:")
    print(f"  输出目录: {output_dir}")


def process_yelp(max_lines=50000):
    """处理Yelp数据"""
    print("\n" + "="*50)
    print("处理Yelp数据...")
    print("="*50)

    parser = YelpParser()
    output_dir = PROJECT_ROOT / "data" / "yelp"
    parser.export_to_csv(output_dir, max_businesses=max_lines)

    print(f"\n✓ Yelp数据处理完成:")
    print(f"  输出目录: {output_dir}")


def process_solomon():
    """处理Solomon VRPTW基准数据"""
    print("\n" + "="*50)
    print("处理Solomon VRPTW基准数据...")
    print("="*50)

    parser = SolomonParser()
    instances = parser.get_benchmark_instances()

    # 导出
    output_file = PROJECT_ROOT / "data" / "solomon_benchmark.csv"
    df = parser.export_to_dataframe(instances)
    df.to_csv(output_file, index=False)

    print(f"\n✓ Solomon数据处理完成:")
    print(f"  实例数: {len(instances)}")
    print(f"  输出: {output_file}")


def list_data_sources():
    """列出所有数据源状态"""
    print("\n" + "="*50)
    print("数据源状态检查")
    print("="*50)

    data_dir = PROJECT_ROOT / "data"

    sources = {
        "POI数据": data_dir / "poi.csv",
        "用户事件": data_dir / "user_events.csv",
        "路线模板SQL": data_dir / "default_route.sql",
        "地址SQL": data_dir / "go_address.sql",
        "SHP省份": data_dir / "external" / "shengfen",
        "Geolife轨迹": data_dir / "external" / "Geolife Trajectories 1.3.zip",
        "Yelp数据": data_dir / "external" / "Yelp-JSON.zip",
        "Gowalla签到": data_dir / "external" / "loc-gowalla_totalCheckins.txt.gz",
        "Solomon基准": data_dir / "external" / "homberger_1000_customer_instances.zip",
    }

    for name, path in sources.items():
        if path.exists():
            if path.is_file():
                size = path.stat().st_size / 1024 / 1024  # MB
                print(f"  ✓ {name}: {path.name} ({size:.1f} MB)")
            else:
                files = list(path.glob("*"))
                print(f"  ✓ {name}: {len(files)} 个文件")
        else:
            print(f"  ✗ {name}: 不存在")


def main():
    parser = argparse.ArgumentParser(
        description='GoAfar数据处理统一入口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/process_all_data.py --list          # 列出数据源状态
  python scripts/process_all_data.py --all           # 处理所有数据
  python scripts/process_all_data.py --route         # 只处理路线模板
  python scripts/process_all_data.py --geolife       # 只处理Geolife数据
  python scripts/process_all_data.py --gowalla       # 只处理Gowalla数据
        """
    )

    parser.add_argument('--all', action='store_true', help='处理所有数据')
    parser.add_argument('--list', action='store_true', help='列出数据源状态')
    parser.add_argument('--route', action='store_true', help='处理路线模板')
    parser.add_argument('--geolife', action='store_true', help='处理Geolife数据')
    parser.add_argument('--gowalla', action='store_true', help='处理Gowalla数据')
    parser.add_argument('--yelp', action='store_true', help='处理Yelp数据')
    parser.add_argument('--solomon', action='store_true', help='处理Solomon基准数据')
    parser.add_argument('--max-users', type=int, default=50, help='Geolife最大用户数')
    parser.add_argument('--max-lines', type=int, default=1000000, help='Gowalla/Yelp最大行数')

    args = parser.parse_args()

    # 列出数据源
    if args.list:
        list_data_sources()
        return

    print(f"\nGoAfar 数据处理开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 处理路线模板（优先，数据量小）
    if args.all or args.route:
        process_route_templates()

    # 处理Geolife数据
    if args.all or args.geolife:
        process_geolife(max_users=args.max_users)

    # 处理Gowalla数据
    if args.all or args.gowalla:
        process_gowalla(max_lines=args.max_lines)

    # 处理Yelp数据
    if args.all or args.yelp:
        process_yelp(max_lines=args.max_lines)

    # 处理Solomon数据
    if args.all or args.solomon:
        process_solomon()

    # 如果没有指定任何操作
    if not any([args.all, args.route, args.geolife, args.gowalla, args.yelp, args.solomon, args.list]):
        parser.print_help()

    print(f"\n数据处理完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
