"""
Time matrix providers: OSRM with Haversine fallback.
"""
from __future__ import annotations

import hashlib
import json
import os
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

try:
    from src.utils.cache_manager import CacheManager
    from src.utils.id_mapping import normalize_poi_id
except ImportError:
    from utils.cache_manager import CacheManager
    from utils.id_mapping import normalize_poi_id


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算两点间的 Haversine 距离（公里）"""
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return radius * c


class TimeMatrixProvider:
    name = "base"

    def build(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class HaversineTimeMatrixProvider(TimeMatrixProvider):
    name = "haversine"

    def __init__(self, avg_speed_kmh: float = 60):
        self.avg_speed_kmh = avg_speed_kmh

    def build(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        matrix = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            lat1, lon1 = float(df.iloc[i]["lat"]), float(df.iloc[i]["lon"])
            for j in range(n):
                if i == j:
                    continue
                lat2, lon2 = float(df.iloc[j]["lat"]), float(df.iloc[j]["lon"])
                dist_km = haversine_distance(lat1, lon1, lat2, lon2)
                matrix[i, j] = int((dist_km / self.avg_speed_kmh) * 3600)
        return matrix


class OSRMTimeMatrixProvider(TimeMatrixProvider):
    name = "osrm"

    def __init__(self, osrm_url: str = "http://router.project-osrm.org", timeout_sec: int = 10, chunk_size: int = 100):
        self.osrm_url = osrm_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.chunk_size = chunk_size

    def _request_table(self, coords: Sequence[str], sources: Optional[Sequence[int]] = None) -> dict:
        coord_str = ";".join(coords)
        params = {"annotations": "duration"}
        if sources is not None:
            params["sources"] = ";".join(str(i) for i in sources)
        endpoint = f"{self.osrm_url}/table/v1/driving/{coord_str}?{urlencode(params)}"
        with urlopen(endpoint, timeout=self.timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if payload.get("code") != "Ok":
            raise RuntimeError(f"OSRM table error: {payload.get('code')}")
        return payload

    def build(self, df: pd.DataFrame) -> np.ndarray:
        coords = [f"{float(row['lon'])},{float(row['lat'])}" for _, row in df.iterrows()]
        n = len(coords)
        if n == 0:
            return np.zeros((0, 0), dtype=np.int32)

        if n <= self.chunk_size:
            payload = self._request_table(coords)
            durations = payload.get("durations") or []
            return _durations_to_matrix(durations, n)

        # Source-chunking for larger candidate sets.
        matrix = np.zeros((n, n), dtype=np.int32)
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            source_idx = list(range(start, end))
            payload = self._request_table(coords, sources=source_idx)
            rows = payload.get("durations") or []
            if len(rows) != len(source_idx):
                raise RuntimeError("OSRM table returned unexpected row count")
            for i, row in enumerate(rows):
                matrix[source_idx[i], :] = _duration_row_to_seconds(row, n)
        return matrix


def _duration_row_to_seconds(row: Sequence[float | None], n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.int32)
    for i, value in enumerate(row):
        if value is None:
            out[i] = 0
        else:
            out[i] = int(float(value))
    return out


def _durations_to_matrix(durations: Sequence[Sequence[float | None]], n: int) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=np.int32)
    for i, row in enumerate(durations):
        matrix[i, :] = _duration_row_to_seconds(row, n)
    return matrix


def _provider_factory(provider: str, avg_speed_kmh: float, osrm_url: str) -> TimeMatrixProvider:
    provider = provider.lower().strip()
    if provider == "haversine":
        return HaversineTimeMatrixProvider(avg_speed_kmh=avg_speed_kmh)
    if provider == "osrm":
        return OSRMTimeMatrixProvider(osrm_url=osrm_url)
    # "auto"
    return OSRMTimeMatrixProvider(osrm_url=osrm_url)


def _filter_pois(df: pd.DataFrame, poi_ids: Optional[Sequence[str]]) -> pd.DataFrame:
    if poi_ids is None:
        return df.reset_index(drop=True)

    ids = [normalize_poi_id(v) for v in poi_ids]
    scoped = df[df["poi_id"].isin(ids)].copy()
    if len(scoped) == 0:
        raise ValueError(f"未找到指定的POI ID: {ids[:5]}... (共{len(ids)}个)")

    order_map = {poi_id: idx for idx, poi_id in enumerate(ids)}
    scoped["__order"] = scoped["poi_id"].map(order_map)
    scoped = scoped.sort_values("__order").drop(columns=["__order"]).reset_index(drop=True)
    missing = set(ids) - set(scoped["poi_id"].tolist())
    if missing:
        print(f"⚠️ 警告: 以下POI ID不存在: {list(missing)[:5]}...")
    return scoped


def _cache_key(provider: str, osrm_url: str, avg_speed_kmh: float, poi_ids: Sequence[str]) -> str:
    payload = {
        "provider": provider,
        "osrm_url": osrm_url,
        "avg_speed_kmh": avg_speed_kmh,
        "poi_ids": list(poi_ids),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def build_time_matrix(
    poi_csv: str = "data/all/poi_with_coords.csv",
    output_path: str = "outputs/routing/time_matrix.npy",
    avg_speed_kmh: float = 60,
    poi_ids: Optional[Sequence[str]] = None,
    provider: str = "haversine",
    osrm_url: str = "http://router.project-osrm.org",
    use_cache: bool = True,
    cache_dir: str = "outputs/cache",
    cache_ttl_hours: int = 24,
    return_provider: bool = False,
):
    """
    构建时间矩阵（秒）.

    provider:
    - haversine: 纯地理近似
    - osrm: 强制 OSRM table service
    - auto: 先 OSRM，失败后回退 haversine
    """
    output_file = _resolve_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    poi_df = pd.read_csv(_resolve_path(poi_csv), low_memory=False)
    poi_df["poi_id"] = poi_df["poi_id"].apply(normalize_poi_id)
    poi_df = _filter_pois(poi_df, poi_ids=poi_ids)

    # 坐标清洗：避免 NaN/非法值在距离计算阶段触发 int(NaN) 异常
    for col in ("lat", "lon"):
        poi_df[col] = pd.to_numeric(poi_df[col], errors="coerce")
    invalid_coord = poi_df["lat"].isna() | poi_df["lon"].isna()
    if invalid_coord.any():
        dropped = poi_df.loc[invalid_coord, "poi_id"].astype(str).tolist()
        preview = dropped[:5]
        suffix = "..." if len(dropped) > 5 else ""
        print(f"⚠️ 跳过 {len(dropped)} 个坐标缺失POI: {preview}{suffix}")
        poi_df = poi_df.loc[~invalid_coord].reset_index(drop=True)

    n = len(poi_df)
    if n == 0:
        raise ValueError("POI数据为空或坐标无效，无法构建时间矩阵")

    print(f"构建时间矩阵: {n}x{n}")
    print(f"provider: {provider}")

    matrix_provider = _provider_factory(provider, avg_speed_kmh=avg_speed_kmh, osrm_url=osrm_url)
    cache = CacheManager(cache_dir=str(_resolve_path(cache_dir)), cache_ttl_hours=cache_ttl_hours)
    cache_payload = None

    if use_cache:
        key = _cache_key(
            provider=provider,
            osrm_url=osrm_url,
            avg_speed_kmh=avg_speed_kmh,
            poi_ids=poi_df["poi_id"].tolist(),
        )
        cache_payload = cache.get("time_matrix", {"key": key})
        if cache_payload is not None:
            matrix = cache_payload["matrix"].astype(np.int32)
            provider_used = cache_payload.get("provider_used", provider)
            np.save(output_file, matrix)
            if return_provider:
                return matrix, poi_df, provider_used
            return matrix, poi_df

    try:
        matrix = matrix_provider.build(poi_df)
        provider_used = matrix_provider.name
    except Exception as exc:
        if provider not in {"auto", "osrm"}:
            raise
        print(f"⚠️ OSRM 构建失败，回退 Haversine: {exc}")
        fallback = HaversineTimeMatrixProvider(avg_speed_kmh=avg_speed_kmh)
        matrix = fallback.build(poi_df)
        provider_used = fallback.name

    np.save(output_file, matrix)
    if use_cache:
        key = _cache_key(
            provider=provider_used,
            osrm_url=osrm_url,
            avg_speed_kmh=avg_speed_kmh,
            poi_ids=poi_df["poi_id"].tolist(),
        )
        cache.set("time_matrix", {"key": key}, {"matrix": matrix, "provider_used": provider_used})

    print(f"✓ 时间矩阵保存到: {output_file}")
    print(f"✓ 实际 provider: {provider_used}")
    if return_provider:
        return matrix, poi_df, provider_used
    return matrix, poi_df


if __name__ == "__main__":
    matrix, df = build_time_matrix(provider="auto")
    print("\n示例行程时间（前5个POI）:")
    for i in range(min(5, len(df))):
        for j in range(min(5, len(df))):
            if i != j:
                print(f"  {df.iloc[i]['name'][:15]} -> {df.iloc[j]['name'][:15]}: {matrix[i, j] / 60:.1f}分钟")
