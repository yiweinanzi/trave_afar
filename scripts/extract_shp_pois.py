#!/usr/bin/env python
"""
SHP POI数据批量提取器
使用ogr2ogr从SHP文件中提取POI数据并整合到系统
"""
import sys
import subprocess
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHP_DIR = PROJECT_ROOT / "data" / "external" / "shengfen"
OUTPUT_DIR = PROJECT_ROOT / "data" / "shengfen_pois"

# 省份名称映射
PROVINCE_NAMES = {
    "anhui": "安徽", "beijing": "北京", "chongqing": "重庆",
    "fujian": "福建", "gansu": "甘肃", "guangdong": "广东",
    "guangxi": "广西", "hainan": "海南", "hebei": "河北",
    "heilongjiang": "黑龙江", "henan": "河南", "hubei": "湖北",
    "hunan": "湖南", "inner-mongolia": "内蒙古", "jiangsu": "江苏",
    "jilin": "吉林", "liaoning": "辽宁", "macau": "澳门",
    "ningxia": "宁夏", "qinghai": "青海"
}

def extract_province_pois(province_dir: Path, province: str, output_dir: Path):
    """提取单个省份数据"""
    # 查找POI文件
    poi_shp = province_dir / "gis_osm_pois_a_free_1.shp"
    places_shp = province_dir / "gis_osm_places_a_free_1.shp"

    all_pois = []

    # 提取POI
    if poi_shp.exists():
        csv_path = output_dir / f"{province}_pois.csv"
        result = subprocess.run([
            "ogr2ogr", "-f", "CSV", str(csv_path), str(poi_shp),
            "-lco", "GEOMETRY=AS_XY"
        ], capture_output=True, text=True)

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"  {province} POI: {len(df)} 条")
            all_pois.append(df)

    # 提取Places
    if places_shp.exists():
        csv_path = output_dir / f"{province}_places.csv"
        result = subprocess.run([
            "ogr2ogr", "-f", "CSV", str(csv_path), str(places_shp),
            "-lco", "GEOMETRY=AS_XY"
        ], capture_output=True, text=True)

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"  {province} Places: {len(df)} 条")
            all_pois.append(df)

    if all_pois:
        combined = pd.concat(all_pois, ignore_index=True)
        # 添加省份列
        combined['province'] = province
        return combined
    return None

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='SHP POI提取器')
    parser.add_argument('--provinces', nargs='+', help='指定省份')
    parser.add_argument('--limit', type=int, help='限制提取省份数量')
    parser.add_argument('--merge', action='store_true', help='合并到现有poi.csv')

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 收集所有省份目录
    province_dirs = sorted([d for d in SHP_DIR.iterdir() if d.is_dir()])

    results = []

    for province_dir in province_dirs[:args.limit] if args.limit else province_dirs:
        if 'free' not in province_dir.name.lower():
            continue

        province_key = province_dir.name.replace('-260213-free.shp', '').replace('-260212-free.shp', '')
        province = PROVINCE_NAMES.get(province_key, province_key)

        # 过滤指定省份
        if args.provinces and province not in args.provinces:
            continue

        print(f"处理 {province}...")
        df = extract_province_pois(province_dir, province, OUTPUT_DIR)
        if df is not None:
            results.append(df)

    # 合并所有省份
    if results:
        combined = pd.concat(results, ignore_index=True)
        output_path = OUTPUT_DIR / "all_provinces_pois.csv"
        combined.to_csv(output_path, index=False)
        print(f"\n保存合并数据: {output_path} ({len(combined)} 条POI)")

        # 统计
        print(f"\n省份POI分布:")
        for prov, count in combined['province'].value_counts().items():
            print(f"  {prov}: {count}")

        # 合并到现有poi.csv
        if args.merge:
            existing_poi = pd.read_csv(PROJECT_ROOT / "data" / "poi.csv")

            # 转换列名匹配
            combined.rename(columns={'name': 'name'}, inplace=True)

            # 添加缺失列
            for col in ['poi_id', 'lat', 'lon', 'open_min', 'close_min', 'stay_min', 'city', 'description', 'time_str']:
                if col not in combined.columns:
                    combined[col] = None

            # 生成新poi_id
            if 'poi_id' not in combined.columns or combined['poi_id'].isna().all():
                combined['poi_id'] = [f"S{i:06d}" for i in range(len(combined))]

            # 只保留有名称的POI
            combined = combined[combined['name'].notna()]

            # 去重（按名称）
            existing_names = set(existing_poi['name'].dropna().astype(str).tolist())
            new_pois = combined[~combined['name'].isin(existing_names)]

            print(f"\n合并到现有POI:")
            print(f"  现有POI: {len(existing_poi)}")
            print(f"  新增POI: {len(new_pois)}")

            if len(new_pois) > 0:
                merged = pd.concat([existing_poi, new_pois], ignore_index=True)
                merged.to_csv(PROJECT_ROOT / "data" / "poi_expanded.csv", index=False)
                print(f"  保存到: data/poi_expanded.csv")

if __name__ == "__main__":
    main()
