#!/usr/bin/env python
"""
GoAfar 数据整合主脚本
一键整合所有数据源

功能：
1. 处理外部数据（Geolife、Gowalla、Yelp、Solomon）
2. 从shengfen省份数据提取POI
3. 使用Wikidata补充POI数据
4. 合并所有数据到系统

使用方法:
    python scripts/integrate_all_data.py
    python scripts/integrate_all_data.py --skip-wikidata  # 跳过Wikidata（慢）
    python scripts/integrate_all_data.py --provinces 新疆 四川 浙江
"""
import sys
from pathlib import Path
import json
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_existing_poi():
    """加载现有POI数据"""
    poi_path = PROJECT_ROOT / "data" / "poi.csv"
    if poi_path.exists():
        df = pd.read_csv(poi_path)
        logger.info(f"现有POI: {len(df)} 条")
        return df
    return pd.DataFrame(columns=[
        "poi_id", "name", "lat", "lon", "open_min", "close_min",
        "stay_min", "province", "city", "description", "time_str"
    ])


def save_poi(df, suffix=""):
    """保存POI数据"""
    output_path = PROJECT_ROOT / f"data/poi{suffix}.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"保存POI: {output_path} ({len(df)} 条)")
    return output_path


# -----------------------------
# 1. 处理路线模板
# -----------------------------
def process_route_templates():
    """处理路线模板"""
    logger.info("=" * 50)
    logger.info("处理路线模板...")
    logger.info("=" * 50)

    try:
        from data_processing.route_template import parse_route_sql, save_route_templates

        routes = parse_route_sql()
        save_route_templates(routes)

        stats = {}
        for route in routes:
            prov = route['province']
            stats[prov] = stats.get(prov, 0) + 1

        logger.info(f"路线模板: {len(routes)} 条")
        for prov, count in sorted(stats.items()):
            logger.info(f"  {prov}: {count} 条")

        return routes

    except Exception as e:
        logger.error(f"处理路线模板失败: {e}")
        return []


# -----------------------------
# 2. 从路线模板提取缺失POI
# -----------------------------
def extract_missing_route_pois(routes, existing_df):
    """从路线模板提取缺失POI"""
    logger.info("\n" + "=" * 50)
    logger.info("提取路线中缺失的POI...")
    logger.info("=" * 50)

    existing_names = set(existing_df["name"].dropna().astype(str).tolist())
    missing_pois = {}

    for route in routes:
        province = route['province']
        if province not in missing_pois:
            missing_pois[province] = set()

        for day_route in route.get("daily_routes", []):
            for poi_name in day_route:
                if poi_name not in existing_names:
                    missing_pois[province].add(poi_name)

    total_missing = sum(len(s) for s in missing_pois.values())
    logger.info(f"缺失POI总数: {total_missing}")

    for prov, names in sorted(missing_pois.items()):
        if names:
            logger.info(f"  {prov}: {len(names)} 个")

    return missing_pois


# -----------------------------
# 3. 使用Wikidata搜索缺失POI
# -----------------------------
def search_wikidata_for_pois(missing_pois, max_per_province=100):
    """使用Wikidata搜索缺失POI"""
    logger.info("\n" + "=" * 50)
    logger.info("使用Wikidata搜索POI...")
    logger.info("=" * 50)

    try:
        import requests
        from tqdm import tqdm
        import time

        WIKIDATA_API = "https://www.wikidata.org/w/api.php"
        WDQS_URL = "https://query.wikidata.org/sparql"

        found_pois = []

        for province, names in missing_pois.items():
            if not names:
                continue

            logger.info(f"搜索 {province} 的 {len(names)} 个POI...")

            for name in tqdm(list(names)[:max_per_province], desc=f"{province}"):
                # 搜索实体
                params = {
                    "action": "wbsearchentities",
                    "search": name,
                    "language": "zh",
                    "limit": 5,
                    "format": "json"
                }

                try:
                    r = requests.get(WIKIDATA_API, params=params, timeout=10)
                    r.raise_for_status()
                    hits = r.json().get("search", [])

                    if hits:
                        hit = hits[0]
                        qid = hit.get("id")

                        # 查询坐标
                        query = f"""
SELECT ?lat ?lon ?image WHERE {{
  wd:{qid} wdt:P625 ?coord .
  OPTIONAL {{ wd:{qid} wdt:P18 ?image . }}
  BIND(geof:latitude(?coord) AS ?lat)
  BIND(geof:longitude(?coord) AS ?lon)
}} LIMIT 1
"""

                        r2 = requests.get(WDQS_URL, params={"query": query, "format": "json"}, timeout=10)
                        r2.raise_for_status()
                        coords = r2.json().get("results", {}).get("bindings", [])

                        if coords:
                            found_pois.append({
                                "poi_id": f"WD{qid}",
                                "name": name,
                                "lat": float(coords[0]["lat"]["value"]),
                                "lon": float(coords[0]["lon"]["value"]),
                                "province": province,
                                "city": "",
                                "description": hit.get("description", "") or "",
                                "open_min": 480,
                                "close_min": 1140,
                                "stay_min": 120,
                                "time_str": "08:00-19:00",
                                "airport": "",
                                "wd_qid": qid,
                                "source": "wikidata"
                            })

                except Exception as e:
                    logger.debug(f"搜索 {name} 失败: {e}")

                time.sleep(0.2)

        logger.info(f"成功找到: {len(found_pois)} 个POI")
        return found_pois

    except ImportError:
        logger.warning("缺少依赖，跳过Wikidata搜索")
        return []


# -----------------------------
# 4. 从Wikidata扩展省份POI
# -----------------------------
def expand_province_pois(provinces, existing_df, topk=200):
    """从Wikidata扩展省份POI"""
    logger.info("\n" + "=" * 50)
    logger.info(f"扩展省份POI: {provinces}")
    logger.info("=" * 50)

    try:
        import requests
        from tqdm import tqdm
        import time

        WDQS_URL = "https://query.wikidata.org/sparql"

        # 省份QID
        province_qids = {
            "新疆": "Q34800", "西藏": "Q17241", "云南": "Q43194",
            "四川": "Q31304", "甘肃": "Q42621", "宁夏": "Q26598",
            "内蒙古": "Q5033", "青海": "Q46592", "浙江": "Q46862",
            "江苏": "Q16672", "广东": "Q16748", "福建": "Q42385",
        }

        existing_names = set(existing_df["name"].dropna().astype(str).tolist())
        new_pois = []

        for province in provinces:
            qid = province_qids.get(province)
            if not qid:
                logger.warning(f"未找到省份QID: {province}")
                continue

            logger.info(f"处理 {province} (QID: {qid})...")

            query = f"""
SELECT ?item ?itemLabel ?itemDescription ?lat ?lon ?image ?sitelinks WHERE {{
  ?item wdt:P131* wd:{qid} ;
        wdt:P625 ?coord .
  OPTIONAL {{ ?item wdt:P31/wdt:P279* wd:Q570116 . }}
  OPTIONAL {{ ?item wdt:P18 ?image . }}
  ?item wikibase:sitelinks ?sitelinks .
  FILTER(?sitelinks >= 3)
  BIND(geof:latitude(?coord) AS ?lat)
  BIND(geof:longitude(?coord) AS ?lon)
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "zh,en". }}
}}
ORDER BY DESC(?sitelinks)
LIMIT {topk}
"""

            try:
                r = requests.get(WDQS_URL, params={"query": query, "format": "json"}, timeout=30)
                r.raise_for_status()
                results = r.json().get("results", {}).get("bindings", [])

                for item in tqdm(results, desc=f"{province}"):
                    name = item.get("itemLabel", {}).get("value", "")
                    if name in existing_names:
                        continue

                    qid = item["item"]["value"].split("/")[-1]
                    lat = float(item["lat"]["value"])
                    lon = float(item["lon"]["value"])

                    new_pois.append({
                        "poi_id": f"WD{qid}",
                        "name": name,
                        "lat": lat,
                        "lon": lon,
                        "province": province,
                        "city": "",
                        "description": item.get("itemDescription", {}).get("value", "") or "",
                        "open_min": 480,
                        "close_min": 1140,
                        "stay_min": 120,
                        "time_str": "08:00-19:00",
                        "airport": "",
                        "wd_qid": qid,
                        "wd_sitelinks": int(item.get("sitelinks", {}).get("value", 0)),
                        "source": "wikidata"
                    })

                logger.info(f"  {province}: 新增 {len([p for p in new_pois if p['province'] == province])} 个POI")

            except Exception as e:
                logger.error(f"查询 {province} 失败: {e}")

            time.sleep(1)

        logger.info(f"总共新增: {len(new_pois)} 个POI")
        return new_pois

    except ImportError:
        logger.warning("缺少依赖，跳过省份扩展")
        return []


# -----------------------------
# 5. 从shengfen SHP提取省份信息
# -----------------------------
def process_shp_provinces():
    """处理SHP省份数据"""
    logger.info("\n" + "=" * 50)
    logger.info("处理SHP省份数据...")
    logger.info("=" * 50)

    try:
        from data_processing.shp_parser import SHPParser, parse_all_shp_files

        shp_dir = PROJECT_ROOT / "data" / "external" / "shengfen"
        parser = SHPParser(shp_dir)

        files = parser.list_shp_files()
        logger.info(f"SHP文件: {len(files)} 个")

        province_info = {}
        for f in files:
            province = parser.extract_province_name(f)
            gdf = parser.parse_shp_file(f)
            if gdf is not None:
                bbox = parser.get_bbox(gdf)
                province_info[province] = {
                    "file": f.name,
                    "rows": len(gdf),
                    "bbox": bbox
                }
                logger.info(f"  {province}: {len(gdf)} 行")

        # 保存省份信息
        info_path = PROJECT_ROOT / "data" / "province_info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(province_info, f, ensure_ascii=False, indent=2, default=float)

        logger.info(f"保存省份信息: {info_path}")
        return province_info

    except Exception as e:
        logger.error(f"处理SHP数据失败: {e}")
        return {}


# -----------------------------
# 6. 处理Geolife数据
# -----------------------------
def process_geolife(max_users=20):
    """处理Geolife数据"""
    logger.info("\n" + "=" * 50)
    logger.info("处理Geolife数据...")
    logger.info("=" * 50)

    try:
        from data_processing.geolife_parser import GeolifeParser

        parser = GeolifeParser()
        samples = parser.generate_training_samples(max_users=max_users)

        output_file = PROJECT_ROOT / "data" / "geolife_samples.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Geolife样本: {len(samples)} 条")
        logger.info(f"保存到: {output_file}")

        return samples

    except Exception as e:
        logger.error(f"处理Geolife数据失败: {e}")
        return []


# -----------------------------
# 7. 处理Gowalla数据
# -----------------------------
def process_gowalla(max_lines=500000):
    """处理Gowalla数据"""
    logger.info("\n" + "=" * 50)
    logger.info("处理Gowalla数据...")
    logger.info("=" * 50)

    try:
        from data_processing.gowalla_parser import GowallaParser

        parser = GowallaParser()
        output_dir = PROJECT_ROOT / "data" / "gowalla"
        parser.export_to_csv(output_dir, max_lines=max_lines)

        logger.info(f"Gowalla数据已导出到: {output_dir}")
        return True

    except Exception as e:
        logger.error(f"处理Gowalla数据失败: {e}")
        return False


# -----------------------------
# 主流程
# -----------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="GoAfar数据整合")
    parser.add_argument("--skip-wikidata", action="store_true",
                       help="跳过Wikidata数据获取（较慢）")
    parser.add_argument("--skip-external", action="store_true",
                       help="跳过外部数据处理（Geolife、Gowalla等）")
    parser.add_argument("--provinces", nargs="+",
                       default=["新疆", "四川", "云南", "西藏"],
                       help="要扩展的省份")
    parser.add_argument("--topk", type=int, default=200,
                       help="每个省份从Wikidata获取的POI数量")

    args = parser.parse_args()

    logger.info("GoAfar 数据整合开始")
    logger.info("=" * 50)

    # 加载现有POI
    poi_df = load_existing_poi()
    initial_count = len(poi_df)

    # 1. 处理路线模板
    routes = process_route_templates()

    # 2. 提取缺失POI
    if routes:
        missing_pois = extract_missing_route_pois(routes, poi_df)

        # 3. 搜索Wikidata补充
        if not args.skip_wikidata and missing_pois:
            found_pois = search_wikidata_for_pois(missing_pois)
            if found_pois:
                new_df = pd.DataFrame(found_pois)
                poi_df = pd.concat([poi_df, new_df], ignore_index=True)
                logger.info(f"POI更新: {initial_count} -> {len(poi_df)}")

    # 4. 扩展省份POI
    if not args.skip_wikidata:
        expanded_pois = expand_province_pois(args.provinces, poi_df, topk=args.topk)
        if expanded_pois:
            new_df = pd.DataFrame(expanded_pois)
            poi_df = pd.concat([poi_df, new_df], ignore_index=True)
            logger.info(f"POI更新: {len(poi_df)} 条")

    # 5. 处理SHP数据
    if not args.skip_external:
        process_shp_provinces()

    # 6. 处理外部数据
    if not args.skip_external:
        process_geolife(max_users=20)
        process_gowalla(max_lines=500000)

    # 保存最终POI数据
    logger.info("\n" + "=" * 50)
    logger.info("保存最终数据...")
    logger.info("=" * 50)

    # 按省份统计
    logger.info("\n省份POI分布:")
    for prov in sorted(poi_df["province"].unique()):
        count = len(poi_df[poi_df["province"] == prov])
        logger.info(f"  {prov}: {count} 个")

    save_poi(poi_df)

    logger.info(f"\n数据整合完成!")
    logger.info(f"  初始POI: {initial_count}")
    logger.info(f"  最终POI: {len(poi_df)}")
    logger.info(f"  新增POI: {len(poi_df) - initial_count}")


if __name__ == "__main__":
    main()
