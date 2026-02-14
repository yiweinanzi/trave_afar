#!/usr/bin/env python
"""
快速POI补充脚本
从Wikidata批量获取旅游景点POI数据

使用方法:
    python scripts/fetch_wikidata_pois.py --provinces 新疆 四川 浙江
    python scripts/fetch_wikidata_pois.py --all  # 获取所有配置的省份
"""
import sys
from pathlib import Path
import json
import time
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 省份QID映射（Wikidata实体ID）
PROVINCE_QIDS = {
    "新疆": "Q34800",
    "新疆维吾尔自治区": "Q34800",
    "西藏": "Q17241",
    "西藏自治区": "Q17241",
    "云南": "Q43194",
    "四川": "Q31304",
    "甘肃省": "Q42621",
    "甘肃": "Q42621",
    "宁夏": "Q26598",
    "内蒙古": "Q5033",
    "青海": "Q46592",
    "浙江": "Q46862",
    "江苏": "Q16672",
    "广东": "Q16748",
    "福建": "Q42385",
    "安徽": "Q46862",
    "北京": "Q956",
    "上海": "Q8686",
    "重庆": "Q17917",
    "陕西": "Q43600",
    "山东": "Q43384",
    "河南": "Q43384",
    "湖北": "Q47092",
    "湖南": "Q47092",
    "江西": "Q43384",
    "广西": "Q43600",
    "海南": "Q43600",
    "辽宁": "Q43600",
    "吉林": "Q43600",
    "黑龙江": "Q43600",
    "天津": "Q43942",
}


def query_wikidata(sparql, timeout=30):
    """执行SPARQL查询"""
    import requests
    url = "https://query.wikidata.org/sparql"
    params = {"query": sparql, "format": "json"}
    headers = {"User-Agent": "goafar-tourism/1.0"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json().get("results", {}).get("bindings", [])
    except Exception as e:
        logger.error(f"查询失败: {e}")
        return []


def get_province_pois(province, topk=500, min_sitelinks=3):
    """获取省份旅游景点"""
    qid = PROVINCE_QIDS.get(province)
    if not qid:
        logger.warning(f"未找到省份QID: {province}")
        return []

    logger.info(f"获取 {province} 的旅游景点 (QID: {qid})...")

    # SPARQL查询：获取省份内的旅游景点
    query = f"""
SELECT ?item ?itemLabel ?itemDescription ?lat ?lon ?image ?sitelinks ?zhTitle WHERE {{
  ?item wdt:P131* wd:{qid} ;
        wdt:P625 ?coord .
  OPTIONAL {{ ?item wdt:P31/wdt:P279* wd:Q570116 . }}
  OPTIONAL {{ ?item wdt:P18 ?image . }}
  OPTIONAL {{
    ?zhwiki schema:about ?item ;
            schema:isPartOf <https://zh.wikipedia.org/> ;
            schema:name ?zhTitle .
  }}
  ?item wikibase:sitelinks ?sitelinks .
  FILTER(?sitelinks >= {min_sitelinks})
  BIND(geof:latitude(?coord) AS ?lat)
  BIND(geof:longitude(?coord) AS ?lon)
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "zh,en". }}
}}
ORDER BY DESC(?sitelinks)
LIMIT {topk}
"""

    results = query_wikidata(query)
    logger.info(f"  获取到 {len(results)} 个景点")
    return results


def parse_wikidata_poi(wd_result, province):
    """解析Wikidata结果为POI格式"""
    qid = wd_result["item"]["value"].split("/")[-1]
    name = wd_result.get("itemLabel", {}).get("value", "")
    desc = wd_result.get("itemDescription", {}).get("value", "")
    lat = float(wd_result["lat"]["value"])
    lon = float(wd_result["lon"]["value"])
    sitelinks = int(wd_result.get("sitelinks", {}).get("value", 0))
    img = wd_result.get("image", {}).get("value")
    zh_title = wd_result.get("zhTitle", {}).get("value")

    return {
        "poi_id": f"WD{qid}",
        "name": name,
        "lat": lat,
        "lon": lon,
        "province": province,
        "city": "",
        "description": desc or "",
        "open_min": 480,
        "close_min": 1140,
        "stay_min": 120,
        "time_str": "08:00-19:00",
        "airport": "",
        "wd_qid": qid,
        "wd_sitelinks": sitelinks,
        "zhwiki_title": zh_title,
        "wd_image": img,
        "source": "wikidata"
    }


def deduplicate_pois(new_pois, existing_df, dedupe_km=0.5):
    """去重POI"""
    import math

    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
        return 2 * R * math.asin(math.sqrt(a))

    existing_names = set(existing_df["name"].dropna().astype(str).tolist())
    existing_points = existing_df.dropna(subset=["lat", "lon"])[["name", "lat", "lon"]].to_dict("records")

    keep = []
    for poi in new_pois:
        # 检查名称
        if poi["name"] in existing_names:
            continue

        # 检查距离
        is_dup = False
        for existing in existing_points:
            d = haversine_km(poi["lat"], poi["lon"], existing["lat"], existing["lon"])
            if d <= dedupe_km:
                is_dup = True
                break

        if not is_dup:
            keep.append(poi)

    logger.info(f"  去重后: {len(keep)} 个 (原始 {len(new_pois)} 个)")
    return keep


def main():
    import argparse

    parser = argparse.ArgumentParser(description="从Wikidata获取POI数据")
    parser.add_argument("--provinces", nargs="+", help="省份列表")
    parser.add_argument("--all", action="store_true", help="获取所有配置的省份")
    parser.add_argument("--topk", type=int, default=500, help="每个省份获取的数量")
    parser.add_argument("--min-sitelinks", type=int, default=3, help="最小维基链接数")
    parser.add_argument("--output", default=None, help="输出文件路径")
    parser.add_argument("--append", action="store_true", help="追加到现有poi.csv")

    args = parser.parse_args()

    # 确定省份列表
    if args.all:
        provinces = list(PROVINCE_QIDS.keys())
        # 去重（有些省份有多个名称）
        provinces = list(set(provinces))
        # 优先处理主要名称
        provinces = [p for p in provinces if not "自治区" in p and not "省" in p]
    elif args.provinces:
        provinces = args.provinces
    else:
        # 默认处理现有省份
        provinces = ["新疆", "四川", "云南", "西藏", "甘肃", "青海", "宁夏", "内蒙古"]

    logger.info(f"将处理 {len(provinces)} 个省份: {provinces}")

    # 加载现有POI（用于去重）
    existing_df = None
    if args.append:
        import pandas as pd
        poi_path = PROJECT_ROOT / "data" / "poi.csv"
        if poi_path.exists():
            existing_df = pd.read_csv(poi_path)
            logger.info(f"现有POI: {len(existing_df)} 条")

    # 获取所有POI
    all_pois = []
    for province in provinces:
        wd_results = get_province_pois(province, topk=args.topk, min_sitelinks=args.min_sitelinks)

        pois = []
        for wd in wd_results:
            try:
                poi = parse_wikidata_poi(wd, province)
                pois.append(poi)
            except Exception as e:
                logger.debug(f"解析失败: {e}")
                continue

        # 去重
        if existing_df is not None:
            pois = deduplicate_pois(pois, existing_df)

        all_pois.extend(pois)
        time.sleep(1)  # 避免请求过快

    logger.info(f"\n总共获取到 {len(all_pois)} 个新POI")

    # 保存
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = PROJECT_ROOT / "data" / "wikidata_pois.csv"

    import pandas as pd
    df = pd.DataFrame(all_pois)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"保存到: {output_path}")

    # 如果需要追加
    if args.append and existing_df is not None:
        combined = pd.concat([existing_df, df], ignore_index=True)
        combined_path = PROJECT_ROOT / "data" / "poi_expanded.csv"
        combined.to_csv(combined_path, index=False, encoding="utf-8")
        logger.info(f"合并后保存: {combined_path} ({len(combined)} 条)")


if __name__ == "__main__":
    main()
