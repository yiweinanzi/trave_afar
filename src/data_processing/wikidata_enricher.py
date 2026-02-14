#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wikidata数据增强器
基于search_data.md中的方法，从Wikidata获取POI补充数据

功能：
1. 现有POI数据增强 - 补充图片、简介、QID
2. 扩展省份POI - 获取省份内所有旅游景点
3. 路线POI补全 - 从路线模板中提取缺失的POI

依赖:
    pip install pandas requests tqdm rapidfuzz shapely
"""
import argparse
import ast
import json
import math
import time
import urllib.parse
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import requests
from rapidfuzz import fuzz, process
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

WDQS_URL = "https://query.wikidata.org/sparql"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
COMMONS_RAW = "https://commons.wikimedia.org/w/index.php"

USER_AGENT = "goafar-wikidata-enricher/0.1 (tourism-research)"

# 省份QID映射
PROVINCE_QIDS = {
    "新疆": "Q34800",
    "新疆维吾尔自治区": "Q34800",
    "西藏": "Q17241",
    "西藏自治区": "Q17241",
    "云南": "Q43194",
    "四川省": "Q31304",
    "四川": "Q31304",
    "甘肃": "Q42621",
    "甘肃省": "Q42621",
    "宁夏": "Q26598",
    "宁夏回族自治区": "Q26598",
    "内蒙古": "Q5033",
    "内蒙古自治区": "Q5033",
    "青海": "Q46592",
    "青海省": "Q46592",
    # 新增省份
    "浙江": "Q46862",
    "浙江省": "Q46862",
    "江苏": "Q16672",
    "江苏省": "Q16672",
    "广东": "Q16748",
    "广东省": "Q16748",
    "福建": "Q42385",
    "福建省": "Q42385",
    "安徽": "Q16672",
    "安徽省": "Q16672",
    "北京": "Q956",
    "北京市": "Q956",
    "上海": "Q8686",
    "上海市": "Q8686",
    "重庆": "Q17917",
    "重庆市": "Q17917",
    "天津": "Q956",
    "天津市": "Q956",
    "陕西": "Q43600",
    "陕西省": "Q43600",
    "山东": "Q43600",
    "山东省": "Q43600",
    "河南": "Q43600",
    "河南省": "Q43600",
    "湖北": "Q47092",
    "湖北省": "Q47092",
    "湖南": "Q47092",
    "湖南省": "Q47092",
    "江西": "Q43600",
    "江西省": "Q43600",
    "广西": "Q43600",
    "广西壮族自治区": "Q43600",
    "海南": "Q43600",
    "海南省": "Q43600",
    "辽宁": "Q43600",
    "辽宁省": "Q43600",
    "吉林": "Q43600",
    "吉林省": "Q43600",
    "黑龙江": "Q43600",
    "黑龙江省": "Q43600",
}


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """计算两点之间的Haversine距离（公里）"""
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# -----------------------------
# Wikidata / WDQS 基础封装
# -----------------------------
def wdqs_query(query: str, timeout: int = 60) -> List[Dict]:
    """执行SPARQL查询"""
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": USER_AGENT,
    }
    params = {"query": query, "format": "json"}
    r = requests.get(WDQS_URL, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["results"]["bindings"]


def wbsearchentities(search: str, language: str = "zh", limit: int = 5) -> List[Dict]:
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


def commons_raw_json(title: str) -> Dict:
    """获取Commons原始数据"""
    params = {"title": title, "action": "raw"}
    r = requests.get(COMMONS_RAW, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    return r.json()


# -----------------------------
# 现有POI数据增强
# -----------------------------
def build_nearby_match_query(lat: float, lon: float, radius_km: float = 3.0, limit: int = 20) -> str:
    """构建附近地点查询"""
    return f"""
SELECT ?item ?itemLabel ?itemDescription ?lat ?lon ?image ?sitelinks ?dist ?zhTitle WHERE {{
  SERVICE wikibase:around {{
    ?item wdt:P625 ?coord .
    bd:serviceParam wikibase:center "Point({lon} {lat})"^^geo:wktLiteral ;
                    wikibase:radius "{radius_km}" ;
                    wikibase:distance ?dist .
  }}
  FILTER NOT EXISTS {{ ?item wdt:P31 wd:Q5 }}
  OPTIONAL {{ ?item wdt:P18 ?image . }}
  OPTIONAL {{
    ?zhwiki schema:about ?item ;
            schema:isPartOf <https://zh.wikipedia.org/> ;
            schema:name ?zhTitle .
  }}
  ?item wikibase:sitelinks ?sitelinks .
  BIND(geof:latitude(?coord)  AS ?lat)
  BIND(geof:longitude(?coord) AS ?lon)
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "zh,en". }}
}}
ORDER BY ?dist DESC(?sitelinks)
LIMIT {limit}
""".strip()


def pick_best_candidate(name_zh: str, candidates: List[Dict]) -> Optional[Dict]:
    """选择最佳候选"""
    if not candidates:
        return None
    best = None
    best_score = -1
    for c in candidates:
        label = c.get("itemLabel", {}).get("value", "")
        dist = float(c.get("dist", {}).get("value", "999"))
        sim = fuzz.token_set_ratio(name_zh, label)
        score = sim - dist * 8
        if score > best_score:
            best_score = score
            best = c
    if best_score < 55:
        return None
    return best


def enrich_poi_row(row: Dict, sleep_s: float = 0.1) -> Dict:
    """增强单个POI"""
    enriched = row.copy()
    enriched["wd_qid"] = None
    enriched["wd_image"] = None
    enriched["wd_sitelinks"] = None
    enriched["zhwiki_title"] = None

    if pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
        return enriched

    q = build_nearby_match_query(row["lat"], row["lon"], radius_km=3.0, limit=20)
    try:
        res = wdqs_query(q)
    except Exception as e:
        time.sleep(sleep_s)
        return enriched

    best = pick_best_candidate(row.get("name", ""), res)
    if best:
        qid = best["item"]["value"].split("/")[-1]
        enriched["wd_qid"] = qid
        enriched["wd_image"] = best.get("image", {}).get("value")
        enriched["wd_sitelinks"] = int(best.get("sitelinks", {}).get("value", 0))
        enriched["zhwiki_title"] = best.get("zhTitle", {}).get("value")

    time.sleep(sleep_s)
    return enriched


# -----------------------------
# 省份扩展
# -----------------------------
def get_province_tourist_pois(province_name: str, topk: int = 300, min_sitelinks: int = 5) -> List[Dict]:
    """获取省份旅游景点"""
    # 查找省份QID
    qid = PROVINCE_QIDS.get(province_name)
    if not qid:
        # 尝试搜索
        hits = wbsearchentities(province_name, language="zh", limit=1)
        if hits:
            qid = hits[0]["id"]
        else:
            return []

    print(f"  省份QID: {qid}")

    # SPARQL查询
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
  BIND(geof:latitude(?coord)  AS ?lat)
  BIND(geof:longitude(?coord) AS ?lon)
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "zh,en". }}
}}
ORDER BY DESC(?sitelinks)
LIMIT {topk}
""".strip()

    try:
        results = wdqs_query(query)
        print(f"  获取到 {len(results)} 个景点")
        return results
    except Exception as e:
        print(f"  查询失败: {e}")
        return []


def normalize_wikidata_poi(wd_data: Dict, province: str) -> Dict:
    """标准化Wikidata POI"""
    qid = wd_data["item"]["value"].split("/")[-1]
    label = wd_data.get("itemLabel", {}).get("value", "")
    desc = wd_data.get("itemDescription", {}).get("value", "")
    lat = float(wd_data["lat"]["value"])
    lon = float(wd_data["lon"]["value"])
    sitelinks = int(wd_data.get("sitelinks", {}).get("value", 0))
    img = wd_data.get("image", {}).get("value")
    zh_title = wd_data.get("zhTitle", {}).get("value")

    # 构建图片URL
    photo_url = None
    if img:
        filename = img.split("/")[-1]
        photo_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{urllib.parse.quote(filename)}?width=1024"

    return {
        "poi_id": f"WD{qid}",
        "name": label,
        "lat": lat,
        "lon": lon,
        "province": province,
        "city": "",  # 需���后续填充
        "description": desc or "",
        "open_min": 480,
        "close_min": 1140,
        "stay_min": 120,
        "time_str": "08:00-19:00",
        "airport": "",
        "wd_qid": qid,
        "wd_image": photo_url,
        "wd_sitelinks": sitelinks,
        "zhwiki_title": zh_title,
        "source": "wikidata"
    }


# -----------------------------
# 路线POI补全
# -----------------------------
def extract_missing_pois_from_routes(route_templates: List[Dict], existing_pois: set) -> List[str]:
    """从路线模板中提取缺失的POI名称"""
    missing_names = set()

    for route in route_templates:
        for day_route in route.get("daily_routes", []):
            for poi_name in day_route:
                if poi_name not in existing_pois:
                    missing_names.add(poi_name)

    return list(missing_names)


def search_wikidata_by_name(name: str) -> Optional[Dict]:
    """通过名称搜索Wikidata"""
    hits = wbsearchentities(name, language="zh", limit=5)
    if not hits:
        return None

    # 优先选择有坐标的
    for hit in hits:
        qid = hit["id"]
        # 查询坐标
        query = f"""
SELECT ?lat ?lon WHERE {{
  wd:{qid} wdt:P625 ?coord .
  BIND(geof:latitude(?coord) AS ?lat)
  BIND(geof:longitude(?coord) AS ?lon)
}} LIMIT 1
""".strip()
        try:
            res = wdqs_query(query)
            if res:
                return {
                    "qid": qid,
                    "label": hit.get("label", ""),
                    "description": hit.get("description", ""),
                    "lat": float(res[0]["lat"]["value"]),
                    "lon": float(res[0]["lon"]["value"])
                }
        except:
            continue

    return None


# -----------------------------
# CSV数据加载
# -----------------------------
def load_poi_csv(csv_path: Path = None) -> pd.DataFrame:
    """加载POI CSV"""
    if csv_path is None:
        csv_path = PROJECT_ROOT / "data" / "poi.csv"

    if not csv_path.exists():
        return pd.DataFrame(columns=[
            "poi_id", "name", "lat", "lon", "open_min", "close_min",
            "stay_min", "province", "city", "description", "time_str"
        ])

    return pd.read_csv(csv_path)


def load_route_templates(json_path: Path = None) -> List[Dict]:
    """加载路线模板"""
    if json_path is None:
        json_path = PROJECT_ROOT / "data" / "route_templates.json"

    if not json_path.exists():
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_poi_csv(df: pd.DataFrame, output_path: Path = None):
    """保存POI CSV"""
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "poi.csv"

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"保存POI数据: {output_path} ({len(df)} 条)")


# -----------------------------
# 主流程
# -----------------------------
def expand_provinces(provinces: List[str], existing_df: pd.DataFrame,
                     topk: int = 300, dedupe_km: float = 0.5) -> pd.DataFrame:
    """扩展多个省份的POI"""
    all_new = []

    # 现有POI名称集合
    existing_names = set(existing_df["name"].dropna().tolist())
    existing_points = existing_df.dropna(subset=["lat", "lon"])[["name", "lat", "lon"]].to_dict("records")

    for province in provinces:
        print(f"\n处理省份: {province}")

        # 获取Wikidata数据
        wd_results = get_province_tourist_pois(province, topk=topk)
        if not wd_results:
            continue

        # 标准化
        pois = []
        for wd in wd_results:
            poi = normalize_wikidata_poi(wd, province)

            # 去重检查
            is_dup = False
            for existing in existing_points:
                if poi["name"] == existing["name"]:
                    is_dup = True
                    break
                d = haversine_km(poi["lat"], poi["lon"], existing["lat"], existing["lon"])
                if d <= dedupe_km:
                    is_dup = True
                    break

            if not is_dup:
                pois.append(poi)

        print(f"  新增POI: {len(pois)}")
        all_new.extend(pois)

    if all_new:
        new_df = pd.DataFrame(all_new)
        return pd.concat([existing_df, new_df], ignore_index=True)
    return existing_df


def fill_missing_route_pois(existing_df: pd.DataFrame, route_templates: List[Dict]) -> pd.DataFrame:
    """补全路线中缺失的POI"""
    existing_names = set(existing_df["name"].dropna().tolist())

    # 提取缺失名称
    missing_names = extract_missing_pois_from_routes(route_templates, existing_names)
    print(f"\n路线中缺失的POI数量: {len(missing_names)}")

    if not missing_names:
        return existing_df

    # 搜索Wikidata
    new_pois = []
    for name in tqdm(missing_names, desc="搜索Wikidata"):
        wd_data = search_wikidata_by_name(name)
        if wd_data:
            new_pois.append({
                "poi_id": f"WD{wd_data['qid']}",
                "name": name,  # 使用原始名称
                "lat": wd_data["lat"],
                "lon": wd_data["lon"],
                "province": "",  # 需要后续推断
                "city": "",
                "description": wd_data.get("description", ""),
                "open_min": 480,
                "close_min": 1140,
                "stay_min": 120,
                "time_str": "08:00-19:00",
                "airport": "",
                "source": "wikidata_route"
            })
        time.sleep(0.2)

    if new_pois:
        print(f"  成功补全: {len(new_pois)} 个POI")
        new_df = pd.DataFrame(new_pois)
        return pd.concat([existing_df, new_df], ignore_index=True)

    return existing_df


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Wikidata数据增强器")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # 现有POI增强
    ap_enrich = sub.add_parser("enrich", help="增强现有POI数据")
    ap_enrich.add_argument("--input", default=None, help="POI CSV文件路径")
    ap_enrich.add_argument("--output", default=None, help="输出文件路径")

    # 扩展省份
    ap_expand = sub.add_parser("expand", help="扩展省份POI")
    ap_expand.add_argument("--provinces", nargs="+", required=True,
                          help="省份列表，如: 新疆 四川 浙江")
    ap_expand.add_argument("--topk", type=int, default=300,
                          help="每个省份获取的POI数量")
    ap_expand.add_argument("--input", default=None, help="现有POI CSV")
    ap_expand.add_argument("--output", default=None, help="输出文件路径")

    # 补全路线POI
    ap_fill = sub.add_parser("fill-route", help="补全路线中缺失的POI")
    ap_fill.add_argument("--input", default=None, help="现有POI CSV")
    ap_fill.add_argument("--routes", default=None, help="路线模板JSON")
    ap_fill.add_argument("--output", default=None, help="输出文件路径")

    # 组合命令
    ap_all = sub.add_parser("all", help="执行所有操作")
    ap_all.add_argument("--provinces", nargs="+",
                       default=["新疆", "四川", "云南", "西藏", "浙江", "江苏", "广东"],
                       help="要扩展的省份")

    args = ap.parse_args()

    # 路径处理
    input_path = Path(args.input) if args.input else PROJECT_ROOT / "data" / "poi.csv"
    output_path = Path(args.output) if args.output else PROJECT_ROOT / "data" / "poi_expanded.csv"
    routes_path = Path(args.routes) if args.routes else PROJECT_ROOT / "data" / "route_templates.json"

    if args.cmd == "enrich":
        df = load_poi_csv(input_path)
        print(f"加载POI: {len(df)} 条")
        # TODO: 实现增强逻辑
        print("现有POI增强功能开发中...")

    elif args.cmd == "expand":
        df = load_poi_csv(input_path)
        print(f"现有POI: {len(df)} 条")

        expanded = expand_provinces(args.provinces, df, topk=args.topk)
        save_poi_csv(expanded, output_path)

    elif args.cmd == "fill-route":
        df = load_poi_csv(input_path)
        routes = load_route_templates(routes_path)

        filled = fill_missing_route_pois(df, routes)
        save_poi_csv(filled, output_path)

    elif args.cmd == "all":
        df = load_poi_csv(input_path)
        routes = load_route_templates()

        print(f"现有POI: {len(df)} 条")
        print(f"路线模板: {len(routes)} 条")

        # 1. 补全路线POI
        print("\n[1/3] 补全路线POI...")
        df = fill_missing_route_pois(df, routes)

        # 2. 扩展省份
        print(f"\n[2/3] 扩展省份POI: {args.provinces}")
        df = expand_provinces(args.provinces, df, topk=args.topk)

        # 保存
        save_poi_csv(df, output_path)

        print(f"\n最终POI数量: {len(df)}")
        print(f"新增POI: {len(df) - len(load_poi_csv(input_path))}")


if __name__ == "__main__":
    main()
