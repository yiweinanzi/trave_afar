#!/usr/bin/env python3
"""
Filter POI CSV to rows with valid coordinates.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="仅保留有有效经纬度的 POI 行")
    parser.add_argument(
        "--input",
        default="data/all/poi_expanded.csv",
        help="输入 POI CSV",
    )
    parser.add_argument(
        "--output",
        default="data/all/poi_with_coords.csv",
        help="输出 POI CSV",
    )
    parser.add_argument(
        "--china-bbox",
        action="store_true",
        help="可选：限制在中国常用经纬度范围内 (lon 73~136, lat 3~54)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {in_path}")

    df = pd.read_csv(in_path, low_memory=False)
    required_cols = {"lat", "lon"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"输入缺少坐标列: {sorted(missing_cols)}")

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    valid = df["lat"].notna() & df["lon"].notna()
    if args.china_bbox:
        valid = valid & df["lon"].between(73.0, 136.0) & df["lat"].between(3.0, 54.0)

    filtered = df.loc[valid].copy()
    filtered.to_csv(out_path, index=False)

    total = len(df)
    kept = len(filtered)
    dropped = total - kept
    ratio = (kept / total * 100) if total else 0.0
    print(f"输入: {in_path} ({total} rows)")
    print(f"输出: {out_path} ({kept} rows)")
    print(f"过滤: {dropped} rows")
    print(f"保留率: {ratio:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
