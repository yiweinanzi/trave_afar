#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wikidata数据增强器 - 完整版
基于search_data.md中的方法，实现完整的POI数据获取和增强

功能：
1. enrich - 用Wikidata补齐现有POI（QID/图片/简介等）
2. expand - 扩展省份POI
3. route-fill - 补全路线中缺失的POI
4. search - 搜索指定POI的Wikidata信息

使用方法:
    # 增强现有POI
    python src/data_processing/wikidata_enricher_full.py enrich --input data/poi.csv --output data/poi_enriched.csv

    # 扩展省份
    python src/data_processing/wikidata_enricher_full.py expand --provinces 新疆 四川 浙江 --topk 500

    # 补全路线POI
    python src/data_processing/wikidata_enricher_full.py route-fill

    # 搜索POI
    python src/data_processing/wikidata_enricher_full.py search "天山天池"
"""
import argparse
import ast
import json
import math
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from rapidfuzz import fuzz
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Wikidata API端点
WDQS_URL = "https://query.wikidata.org/sparql"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

USER_AGENT = "goafar-wikidata-enricher/1.0"

# 省份QID映射
PROVINCE_QIDS = {
    "新疆": "Q34800", "新疆维吾尔自治区": "Q34800",
    "西藏": "Q17241", "西藏自治区": "Q17241",
    "云南": "Q43194",
    "四川": "Q31304",
    "甘肃": "Q42621", "甘肃省": "Q42621",
    "宁夏": "Q26598", "宁夏回族自治区": "Q26598",
    "内蒙古": "Q5033", "内蒙古自治区": "Q5033",
    "青海": "Q46592", "青海省": "Q46592",
    "浙江": "Q46862", "浙江省": "Q46862",
    "江苏": "Q16672", "江苏省": "Q16672",
    "广东": "Q16748", "广东省": "Q16748",
    "福建": "Q42385", "福建省": "Q42385",
    "安徽": "Q46862", "安徽省": "Q46862",
    "北京": "Q956", "北京市": "Q956",
    "上海": "Q8686", "上海市": "Q8686",
    "重庆": "Q17917", "重庆市": "Q17917",
    "陕西": "Q43600", "陕西省": "Q43600",
    "山东": "Q43384", "山东省": "Q43384",
    "河南": "Q43384", "河南省": "Q43384",
    "湖北": "Q47092", "湖北省": "Q47092",
    "湖南": "Q47092", "湖南省": "Q47092",
    "江西": "Q43384", "江西省": "Q43384",
    "广西": "Q43600", "广西壮族自治区": "Q43600",
    "海南": "Q43600", "海南省": "Q43600",
    "辽宁": "Q43600", "辽宁省": "Q43600",
    "吉林": "Q43600", "吉林省": "Q43600",
    "黑龙江": "Q43600", "黑龙江省": "Q43600",
    "天津": "Q43942", "天津市": "Q43942",
    "河北": "Q43384", "河北省": "Q43384",
    "山西": "Q43384", "山西省": "Q43384",
    "台湾": "Q86574", "台湾省": "Q86574",
    "香港": "Q8646", "香港特别行政区": "Q8646",
    "澳门": "Q14773", "澳门特别行政区": "Q14773",
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算Haversine距离"""
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))


# ==================== Wikidata API 封装 ====================

def wdqs_query(query: str, timeout: int = 60) -> List[Dict]:
    """执行SPARQL查询"""
    headers = {"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT}
    params = {"query": query, "format": "json"}
    r = requests.get(WDQS_URL, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json().get("results", {}).get("bindings", [])


def wb_search_entities(search: str, language: str = "zh", limit: int = 5) -> List[Dict]:
    """搜索Wikidata实体"""
    params = {
        "action": "wbsearchentities",
        "search": search,
        "language": language,
        "format": "json",
        "limit": limit,
        "type": "item",
    }
    r = requests.get(WIKIDATA_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    return r.json().get("search", [])


def get_entity_details(qid: str) -> Dict:
    """获取实体详细信息"""
    params = {
        "action": "wbgetentities",
        "ids": qid,
        "format": "json",
        "props": "labels|descriptions|claims|sitelinks"
    }
    r = requests.get(WIKIDATA_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    return r.json().get("entities", {}).get(qid, {})


# ==================== SPARQL 查询构建 ====================

def build_nearby_query(lat: float, lon: float, radius_km: float = 3.0, limit: int = 20) -> str:
    """构建附近地点查询"""
    return f"""
SELECT ?item ?itemLabel ?itemDescription ?lat ?lon ?image ?sitelinks ?dist ?zhTitle WHERE {{
  SERVICE wikibase:around {{
    ?item wdt:P625 ?coord .
    bd:serviceParam wikibase:center "Point({lon} {lat})"^^geo:wktLiteral ;
                    wikibase:radius "{radius_km}" .
    bd:serviceParam wikibase:distance ?dist .
  }}
  FILTER NOT EXISTS {{ ?item wdt:P31 wd:Q5 }}
  OPTIONAL {{ ?item wdt:P18 ?image . }}
  OPTIONAL {{
    ?zhwiki schema:about ?item ;
            schema:isPartOf <https://zh.wikipedia.org/> ;
            schema:name ?zhTitle .
  }}
  ?item wikibase:sitelinks ?sitelinks .
  BIND(geof:latitude(?coord) AS ?lat)
  BIND(geof:longitude(?coord) AS ?lon)
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "zh,en". }}
}}
ORDER BY ?dist DESC(?sitelinks)
LIMIT {limit}
""".strip()


def build_province_query(province_qid: str, topk: int = 500, min_sitelinks: int = 3) -> str:
    """构建省份景点查询"""
    return f"""
SELECT ?item ?itemLabel ?itemDescription ?lat ?lon ?image ?sitelinks ?zhTitle WHERE {{
  ?item wdt:P131* wd:{province_qid} ;
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
""".strip()


def build_name_query(name: str) -> str:
    """构建名称查询"""
    return f"""
SELECT ?item ?itemLabel ?itemDescription ?lat ?lon ?image ?sitelinks WHERE {{
  ?item rdfs:label "{name}"@zh .
  OPTIONAL {{ ?item wdt:P625 ?coord . }}
  OPTIONAL {{ ?item wdt:P18 ?image . }}
  OPTIONAL {{ ?item wdt:P31/wdt:P279* wd:Q570116 . }}
  ?item wikibase:sitelinks ?sitelinks .
  BIND(geof:latitude(?coord) AS ?lat)
  BIND(geof:longitude(?coord) AS ?lon)
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "zh,en". }}
}}
LIMIT 10
""".strip()


# ==================== 数据处理 ====================

def enrich_existing_poi(poi_data: Dict, radius_km: float = 3.0) -> Dict:
    """增强单个POI数据"""
    result = poi_data.copy()
    result["wd_qid"] = None
    result["wd_image"] = None
    result["wd_sitelinks"] = None
    result["zhwiki_title"] = None

    lat = poi_data.get("lat")
    lon = poi_data.get("lon")
    name = poi_data.get("name", "")

    if pd.isna(lat) or pd.isna(lon) or not name:
        return result

    # 执行附近查询
    query = build_nearby_query(lat, lon, radius_km=radius_km)
    try:
        candidates = wdqs_query(query)
    except:
        return result

    # 选择最佳匹配
    best = None
    best_score = -1

    for c in candidates:
        label = c.get("itemLabel", {}).get("value", "")
        dist = float(c.get("dist", {}).get("value", "999"))
        sim = fuzz.token_set_ratio(name, label)
        score = sim - dist * 8

        if score > best_score and score >= 55:
            best_score = score
            best = c

    if best:
        qid = best["item"]["value"].split("/")[-1]
        result["wd_qid"] = qid
        result["wd_image"] = best.get("image", {}).get("value")
        result["wd_sitelinks"] = int(best.get("sitelinks", {}).get("value", 0))
        result["zhwiki_title"] = best.get("zhTitle", {}).get("value")

    return result


def fetch_province_pois(province: str, topk: int = 500, min_sitelinks: int = 3) -> List[Dict]:
    """获取省份旅游景点"""
    qid = PROVINCE_QIDS.get(province)
    if not qid:
        print(f"警告: 未找到省份QID: {province}")
        return []

    print(f"获取 {province} 的景点...")

    query = build_province_query(qid, topk=topk, min_sitelinks=min_sitelinks)
    results = wdqs_query(query)

    pois = []
    for r in results:
        try:
            qid = r["item"]["value"].split("/")[-1]
            name = r.get("itemLabel", {}).get("value", "")
            if not name:
                continue

            pois.append({
                "poi_id": f"WD{qid}",
                "name": name,
                "lat": float(r["lat"]["value"]),
                "lon": float(r["lon"]["value"]),
                "province": province,
                "city": "",
                "description": r.get("itemDescription", {}).get("value", "") or "",
                "open_min": 480,
                "close_min": 1140,
                "stay_min": 120,
                "time_str": "08:00-19:00",
                "airport": "",
                "wd_qid": qid,
                "wd_sitelinks": int(r.get("sitelinks", {}).get("value", 0)),
                "zhwiki_title": r.get("zhTitle", {}).get("value"),
                "wd_image": r.get("image", {}).get("value"),
                "source": "wikidata"
            })
        except (KeyError, ValueError) as e:
            continue

    print(f"  获取到 {len(pois)} 个景点")
    time.sleep(1)
    return pois


def search_poi_by_name(name: str) -> Optional[Dict]:
    """通过名称搜索POI"""
    # 先搜索实体
    hits = wb_search_entities(name, language="zh", limit=5)
    if not hits:
        return None

    # 获取第一个有坐标的实体
    for hit in hits:
        qid = hit["id"]
        query = f"""
SELECT ?lat ?lon ?image WHERE {{
  wd:{qid} wdt:P625 ?coord .
  OPTIONAL {{ wd:{qid} wdt:P18 ?image . }}
  BIND(geof:latitude(?coord) AS ?lat)
  BIND(geof:longitude(?coord) AS ?lon)
}} LIMIT 1
"""
        try:
            results = wdqs_query(query)
            if results:
                return {
                    "poi_id": f"WD{qid}",
                    "name": name,
                    "lat": float(results[0]["lat"]["value"]),
                    "lon": float(results[0]["lon"]["value"]),
                    "description": hit.get("description", ""),
                    "wd_qid": qid,
                    "wd_image": results[0].get("image", {}).get("value")
                }
        except:
            continue

    return None


# ==================== 文件I/O ====================

def load_poi_csv(path: Path = None) -> pd.DataFrame:
    """加载POI CSV"""
    if path is None:
        path = PROJECT_ROOT / "data" / "poi.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_route_templates(path: Path = None) -> List[Dict]:
    """加载路线模板"""
    if path is None:
        path = PROJECT_ROOT / "data" / "route_templates.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_poi_csv(df: pd.DataFrame, path: Path = None):
    """保存POI CSV"""
    if path is None:
        path = PROJECT_ROOT / "data" / "poi_enriched.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"保存到: {path} ({len(df)} 条)")


# ==================== 命令 ====================

def cmd_enrich(args):
    """增强现有POI"""
    df = load_poi_csv(Path(args.input) if args.input else None)
    if df.empty:
        print("错误: POI文件为空")
        return

    print(f"现有POI: {len(df)} 条")

    enriched = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="增强POI"):
        poi = row.to_dict()
        enriched.append(enrich_existing_poi(poi))

    result_df = pd.DataFrame(enriched)
    save_poi_csv(result_df, Path(args.output) if args.output else None)

    # 统计
    has_qid = result_df["wd_qid"].notna().sum()
    has_image = result_df["wd_image"].notna().sum()
    print(f"\n增强结果:")
    print(f"  匹配到QID: {has_qid} ({has_qid/len(result_df)*100:.1f}%)")
    print(f"  有图片: {has_image} ({has_image/len(result_df)*100:.1f}%)")


def cmd_expand(args):
    """扩展省份POI"""
    existing_df = load_poi_csv(Path(args.input) if args.input else None)
    print(f"现有POI: {len(existing_df)} 条")

    all_new = []
    for province in args.provinces:
        pois = fetch_province_pois(province, topk=args.topk, min_sitelinks=args.min_sitelinks)

        # 去重
        existing_names = set(existing_df["name"].dropna().astype(str).tolist())
        new_pois = [p for p in pois if p["name"] not in existing_names]

        print(f"    新增: {len(new_pois)} 个")
        all_new.extend(new_pois)

    if all_new:
        new_df = pd.DataFrame(all_new)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        save_poi_csv(combined, Path(args.output) if args.output else None)
        print(f"\n总共新增: {len(all_new)} 个POI")


def cmd_route_fill(args):
    """补全路线POI"""
    df = load_poi_csv(Path(args.input) if args.input else None)
    routes = load_route_templates()

    existing_names = set(df["name"].dropna().astype(str).tolist())

    # 提取缺失POI
    missing = set()
    for route in routes:
        for day_route in route.get("daily_routes", []):
            for name in day_route:
                if name not in existing_names:
                    missing.add(name)

    print(f"路线中缺失POI: {len(missing)} 个")

    # 搜索
    found = []
    for name in tqdm(list(missing), desc="搜索Wikidata"):
        poi = search_poi_by_name(name)
        if poi:
            poi["province"] = ""  # 需要推断
            poi["city"] = ""
            poi["time_str"] = "08:00-19:00"
            poi["source"] = "wikidata_route"
            found.append(poi)
        time.sleep(0.3)

    print(f"成功匹配: {len(found)} 个")

    if found:
        new_df = pd.DataFrame(found)
        combined = pd.concat([df, new_df], ignore_index=True)
        save_poi_csv(combined, Path(args.output) if args.output else None)


def cmd_search(args):
    """搜索POI"""
    result = search_poi_by_name(args.name)
    if result:
        print(f"\n找到: {result['name']}")
        print(f"  QID: {result.get('wd_qid')}")
        print(f"  坐标: {result['lat']}, {result['lon']}")
        print(f"  描述: {result.get('description', '')}")
        if result.get('wd_image'):
            print(f"  图片: {result['wd_image']}")
    else:
        print(f"未找到: {args.name}")


def main():
    parser = argparse.ArgumentParser(description="Wikidata数据增强器")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # enrich命令
    ap_enrich = sub.add_parser("enrich", help="增强现有POI")
    ap_enrich.add_argument("--input", help="输入POI CSV")
    ap_enrich.add_argument("--output", help="输出POI CSV")

    # expand命令
    ap_expand = sub.add_parser("expand", help="扩展省份POI")
    ap_expand.add_argument("--provinces", nargs="+", required=True, help="省份列表")
    ap_expand.add_argument("--topk", type=int, default=500, help="每个省份数量")
    ap_expand.add_argument("--min-sitelinks", type=int, default=3, help="最小维基链接数")
    ap_expand.add_argument("--input", help="现有POI CSV")
    ap_expand.add_argument("--output", help="输出POI CSV")

    # route-fill命令
    ap_fill = sub.add_parser("route-fill", help="补全路线POI")
    ap_fill.add_argument("--input", help="输入POI CSV")
    ap_fill.add_argument("--output", help="输出POI CSV")

    # search命令
    ap_search = sub.add_parser("search", help="搜索POI")
    ap_search.add_argument("name", help="POI名称")

    args = parser.parse_args()

    if args.cmd == "enrich":
        cmd_enrich(args)
    elif args.cmd == "expand":
        cmd_expand(args)
    elif args.cmd == "route-fill":
        cmd_route_fill(args)
    elif args.cmd == "search":
        cmd_search(args)


if __name__ == "__main__":
    main()
