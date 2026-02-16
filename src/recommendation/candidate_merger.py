"""
Multi-recall candidate merger with RRF fusion.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.embedding.vector_builder import search_similar_pois
    from src.utils.id_mapping import normalize_poi_id
except ImportError:
    from embedding.vector_builder import search_similar_pois
    from utils.id_mapping import normalize_poi_id

# 导入 RecBole Provider
try:
    try:
        from src.recommendation.recbole_trainer import RecBoleProvider
    except ImportError:
        from recommendation.recbole_trainer import RecBoleProvider
    RECOBOLE_AVAILABLE = True
except ImportError:
    RECOBOLE_AVAILABLE = False
    print("⚠️ RecBoleProvider 不可用，将使用流行度召回")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTION_WEIGHT = {"click": 1.0, "fav": 2.0, "visit": 3.0}

# 全局 RecBole Provider 实例（延迟加载）
_recbole_provider: Optional[RecBoleProvider] = None


def _get_recbole_provider(
    model_path: Optional[str] = None,
    config_file: str = "configs/recbole.yaml",
    use_gpu: bool = True
) -> Optional[RecBoleProvider]:
    """
    获取或创建 RecBole Provider 实例（单例模式）

    Args:
        model_path: RecBole 模型路径
        config_file: RecBole 配置文件
        use_gpu: 是否使用 GPU

    Returns:
        RecBoleProvider 实例或 None
    """
    global _recbole_provider

    if not RECOBOLE_AVAILABLE:
        return None

    if _recbole_provider is None and model_path:
        _recbole_provider = RecBoleProvider(
            model_path=model_path,
            config_file=config_file,
            use_gpu=use_gpu,
            fallback_to_popular=True
        )

    return _recbole_provider


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _calibrate_scores(series: pd.Series, method: str = "minmax") -> pd.Series:
    if len(series) == 0:
        return series
    if method != "minmax":
        return series.fillna(0.0)

    values = series.fillna(0.0).astype(float)
    min_v = values.min()
    max_v = values.max()
    if abs(max_v - min_v) < 1e-8:
        return values * 0 + 1.0
    return (values - min_v) / (max_v - min_v)


def _attach_rank(df: pd.DataFrame, source_col: str) -> pd.DataFrame:
    if len(df) == 0:
        return df
    out = df.copy()
    out[f"{source_col}_rank"] = range(1, len(out) + 1)
    out[source_col] = out[source_col].fillna(0.0)
    return out


def adaptive_fusion(
    user_history_length: int,
    base_dense_weight: float = 0.55,
    base_behavior_weight: float = 0.30,
    base_geo_weight: float = 0.15,
    min_behavior_weight: float = 0.10,
    max_behavior_weight: float = 0.50,
    history_threshold: int = 5
) -> tuple:
    """
    动态权重融合：根据用户历史长度调整召回权重

    策略：
    - 用户历史行为少时：降低行为召回权重，提升语义和地理召回
    - 用户历史行为多时：提升行为召回权重，降低其他召回

    Args:
        user_history_length: 用户历史交互数量
        base_dense_weight: 基础语义召回权重
        base_behavior_weight: 基础行为召回权重
        base_geo_weight: 基础地理召回权重
        min_behavior_weight: 最小行为召回权重
        max_behavior_weight: 最大行为召回权重
        history_threshold: 历史行为数量阈值

    Returns:
        (dense_weight, behavior_weight, geo_weight)
    """
    if user_history_length == 0:
        # 无历史行为：降低行为召回权重
        behavior_weight = min_behavior_weight
        geo_weight = base_geo_weight + (base_behavior_weight - min_behavior_weight) * 0.5
        dense_weight = 1.0 - behavior_weight - geo_weight
    elif user_history_length < history_threshold:
        # 历史行为较少：逐步提升行为召回权重
        ratio = user_history_length / history_threshold
        behavior_weight = min_behavior_weight + (max_behavior_weight - min_behavior_weight) * ratio * 0.5
        dense_weight = base_dense_weight - (behavior_weight - base_behavior_weight) * 0.7
        geo_weight = 1.0 - dense_weight - behavior_weight
    else:
        # 历史行为充足：使用基础权重或提升行为召回
        behavior_weight = min(max_behavior_weight, base_behavior_weight * 1.3)
        dense_weight = base_dense_weight - (behavior_weight - base_behavior_weight)
        geo_weight = base_geo_weight

    # 确保权重和为 1 且非负
    total = dense_weight + behavior_weight + geo_weight
    dense_weight = max(0.0, dense_weight / total)
    behavior_weight = max(0.0, behavior_weight / total)
    geo_weight = max(0.0, geo_weight / total)

    return dense_weight, behavior_weight, geo_weight


def _prepare_poi_df(poi_csv: str) -> pd.DataFrame:
    poi = pd.read_csv(_resolve_path(poi_csv), low_memory=False)
    poi["poi_id"] = poi["poi_id"].apply(normalize_poi_id)
    if {"lat", "lon"}.issubset(poi.columns):
        poi["lat"] = pd.to_numeric(poi["lat"], errors="coerce")
        poi["lon"] = pd.to_numeric(poi["lon"], errors="coerce")
        valid_coord = poi["lat"].notna() & poi["lon"].notna()
        dropped = int((~valid_coord).sum())
        if dropped > 0:
            print(f"⚠️ 过滤 {dropped} 条无坐标 POI，仅保留可规划候选")
        poi = poi.loc[valid_coord].reset_index(drop=True)
    return poi


def _behavior_recall(
    poi_df: pd.DataFrame,
    user_events_csv: str = "data/all/user_events.csv",
    user_id: Optional[str] = None,
    topk: int = 30,
    use_recbole: bool = False,
    recbole_model_path: Optional[str] = None,
    recbole_config: str = "configs/recbole.yaml",
    recbole_use_gpu: bool = True,
) -> pd.DataFrame:
    """
    行为召回：支持 RecBole 模型或传统的流行度方法

    Args:
        poi_df: POI 数据
        user_events_csv: 用户事件文件
        user_id: 用户 ID
        topk: 返回数量
        use_recbole: 是否使用 RecBole 模型
        recbole_model_path: RecBole 模型路径
        recbole_config: RecBole 配置文件
        recbole_use_gpu: 是否使用 GPU

    Returns:
        行为召回结果 DataFrame
    """
    # 尝试使用 RecBole 模型
    if use_recbole and user_id:
        provider = _get_recbole_provider(
            model_path=recbole_model_path,
            config_file=recbole_config,
            use_gpu=recbole_use_gpu
        )

        if provider is not None:
            rec_df, metadata = provider.predict(
                user_id=str(user_id),
                topk=topk,
                poi_df=poi_df,
                filter_history=True
            )

            if len(rec_df) > 0 and metadata["method"] in ["recbole", "popularity"]:
                # 合并 POI 信息
                rec_df = rec_df.merge(
                    poi_df,
                    on="poi_id",
                    how="left"
                )

                # 将 recbole_score 重命名为 behavior_score
                rec_df = rec_df.rename(columns={"recbole_score": "behavior_score"})

                print(f"  RecBole 召回: {metadata['method']} ({metadata.get('num_interactions', 0)} 次交互)")
                return rec_df.sort_values("behavior_score", ascending=False).reset_index(drop=True)

    # 降级到传统的流行度方法
    path = _resolve_path(user_events_csv)
    if not path.exists():
        return pd.DataFrame(columns=list(poi_df.columns) + ["behavior_score"])

    events = pd.read_csv(path, low_memory=False)
    if len(events) == 0:
        return pd.DataFrame(columns=list(poi_df.columns) + ["behavior_score"])

    events["poi_id"] = events["poi_id"].apply(normalize_poi_id)
    events["weight"] = events["action"].map(ACTION_WEIGHT).fillna(1.0)

    # If user-specific history exists, prioritize it; otherwise fallback to global popularity.
    if user_id and user_id in set(events["user_id"].astype(str)):
        subset = events[events["user_id"].astype(str) == str(user_id)].copy()
        if len(subset) >= 3:
            subset = subset.sort_values("timestamp")
            subset["recency_rank"] = range(1, len(subset) + 1)
            subset["recency_weight"] = subset["recency_rank"] / max(1, len(subset))
            subset["score"] = subset["weight"] * (0.5 + subset["recency_weight"])
        else:
            subset["score"] = subset["weight"]
    else:
        subset = events.copy()
        subset["score"] = subset["weight"]

    scores = subset.groupby("poi_id", as_index=False)["score"].sum()
    if len(scores) == 0:
        return pd.DataFrame(columns=list(poi_df.columns) + ["behavior_score"])

    max_score = max(scores["score"].max(), 1e-8)
    scores["behavior_score"] = scores["score"] / max_score
    scores = scores.sort_values("behavior_score", ascending=False).head(topk)

    merged = poi_df.merge(scores[["poi_id", "behavior_score"]], on="poi_id", how="inner")
    return merged.sort_values("behavior_score", ascending=False).reset_index(drop=True)


def _geo_recall(
    poi_df: pd.DataFrame,
    province_filter: Optional[str],
    topk: int = 40,
    user_events_csv: str = "data/all/user_events.csv",
) -> pd.DataFrame:
    scoped = poi_df
    if province_filter:
        scoped = scoped[scoped["province"] == province_filter]
    if len(scoped) == 0:
        return pd.DataFrame(columns=list(poi_df.columns) + ["geo_score"])

    behavior = _behavior_recall(scoped, user_events_csv=user_events_csv, user_id=None, topk=topk * 2)
    if len(behavior) > 0:
        behavior["geo_score"] = behavior["behavior_score"]
        cols = [c for c in behavior.columns if c != "behavior_score"]
        return behavior[cols].sort_values("geo_score", ascending=False).head(topk).reset_index(drop=True)

    scoped = scoped.head(topk).copy()
    scoped["geo_score"] = np.linspace(1.0, 0.6, len(scoped))
    return scoped


def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "poi_id",
            "name",
            "city",
            "province",
            "semantic_score",
            "behavior_score",
            "geo_score",
            "final_score",
            "from_dense",
            "from_behavior",
            "from_geo",
        ]
    )


def merge_candidates(
    query_text: str,
    user_id: Optional[str] = None,
    topk_dense: int = 50,
    topk_seq: int = 30,
    topk_geo: int = 30,
    province_filter: Optional[str] = None,
    poi_csv: str = "data/all/poi_with_coords.csv",
    user_events_csv: str = "data/all/user_events.csv",
    emb_file: str = "outputs/emb/poi_emb.npy",
    meta_file: str = "outputs/emb/poi_meta.csv",
    model_path: Optional[str] = None,
    use_gpu: bool = True,
    backend: str = "auto",
    allow_without_embeddings: bool = True,
    fusion: str = "rrf",
    rrf_k: int = 60,
    dense_weight: float = 0.55,
    behavior_weight: float = 0.30,
    geo_weight: float = 0.15,
    calibrate: str = "minmax",
    final_topk: Optional[int] = None,
    use_recbole: bool = False,
    recbole_model_path: Optional[str] = None,
    recbole_config: str = "configs/recbole.yaml",
    recbole_use_gpu: bool = True,
    adaptive_fusion_enabled: bool = False,
) -> pd.DataFrame:
    """
    Merge semantic / behavior / geo recall candidates.

    Args:
        query_text: 查询文本
        user_id: 用户 ID
        topk_dense: 语义召回数量
        topk_seq: 行为召回数量
        topk_geo: 地理召回数量
        province_filter: 省份过滤
        poi_csv: POI 数据文件
        user_events_csv: 用户事件文件
        emb_file: 向量文件
        meta_file: 元数据文件
        model_path: 模型路径
        use_gpu: 是否使用 GPU
        backend: 后端类型
        allow_without_embeddings: 是否允许无向量召回
        fusion: 融合方法
        rrf_k: RRF 参数
        dense_weight: 语义召回权重
        behavior_weight: 行为召回权重
        geo_weight: 地理召回权重
        calibrate: 分数校准方法
        final_topk: 最终返回数量
        use_recbole: 是否使用 RecBole 模型
        recbole_model_path: RecBole 模型路径
        recbole_config: RecBole 配置文件
        recbole_use_gpu: RecBole 是否使用 GPU
        adaptive_fusion_enabled: 是否启用动态权重融合

    Returns:
        融合后的候选 DataFrame
    """
    print("\n" + "=" * 60)
    print("候选池合并（Multi-Recall）")
    print("=" * 60)

    poi_df = _prepare_poi_df(poi_csv)

    # 获取用户历史长度（用于动态权重融合）
    user_history_length = 0
    if adaptive_fusion_enabled and user_id:
        provider = _get_recbole_provider(
            model_path=recbole_model_path,
            config_file=recbole_config,
            use_gpu=recbole_use_gpu
        )
        if provider is not None:
            user_history_length = provider.get_user_history_length(str(user_id))

    semantic = pd.DataFrame()
    try:
        semantic = search_similar_pois(
            query_text=query_text,
            topk=topk_dense,
            emb_file=emb_file,
            meta_file=meta_file,
            model_path=model_path,
            use_gpu=use_gpu,
            backend=backend,
            auto_build=False,
            poi_csv=poi_csv,
        )
        semantic["poi_id"] = semantic["poi_id"].apply(normalize_poi_id)
        semantic = semantic.merge(
            poi_df[["poi_id", "name", "city", "province", "lat", "lon", "open_min", "close_min", "stay_min", "description"]],
            on="poi_id",
            how="left",
            suffixes=("", "_poi"),
        )
    except Exception as exc:
        if not allow_without_embeddings:
            raise
        print(f"⚠️ 语义召回不可用，降级到行为/地理召回: {exc}")
        semantic = pd.DataFrame(columns=list(poi_df.columns) + ["semantic_score"])

    behavior = _behavior_recall(
        poi_df,
        user_events_csv=user_events_csv,
        user_id=user_id,
        topk=topk_seq,
        use_recbole=use_recbole,
        recbole_model_path=recbole_model_path,
        recbole_config=recbole_config,
        recbole_use_gpu=recbole_use_gpu
    )
    geo = _geo_recall(poi_df, province_filter=province_filter, topk=topk_geo, user_events_csv=user_events_csv)

    if province_filter:
        semantic = semantic[semantic["province"] == province_filter] if len(semantic) > 0 else semantic
        behavior = behavior[behavior["province"] == province_filter] if len(behavior) > 0 else behavior
        geo = geo[geo["province"] == province_filter] if len(geo) > 0 else geo

    semantic = _attach_rank(semantic.sort_values("semantic_score", ascending=False), "semantic_score")
    behavior = _attach_rank(behavior.sort_values("behavior_score", ascending=False), "behavior_score")
    geo = _attach_rank(geo.sort_values("geo_score", ascending=False), "geo_score")

    candidate_ids = set(semantic.get("poi_id", [])) | set(behavior.get("poi_id", [])) | set(geo.get("poi_id", []))
    if not candidate_ids:
        return _empty_candidates()

    merged = poi_df[poi_df["poi_id"].isin(candidate_ids)].copy()
    merged = merged.merge(
        semantic[["poi_id", "semantic_score", "semantic_score_rank"]] if len(semantic) > 0 else pd.DataFrame(columns=["poi_id", "semantic_score", "semantic_score_rank"]),
        on="poi_id",
        how="left",
    )
    merged = merged.merge(
        behavior[["poi_id", "behavior_score", "behavior_score_rank"]] if len(behavior) > 0 else pd.DataFrame(columns=["poi_id", "behavior_score", "behavior_score_rank"]),
        on="poi_id",
        how="left",
    )
    merged = merged.merge(
        geo[["poi_id", "geo_score", "geo_score_rank"]] if len(geo) > 0 else pd.DataFrame(columns=["poi_id", "geo_score", "geo_score_rank"]),
        on="poi_id",
        how="left",
    )

    for score_col in ("semantic_score", "behavior_score", "geo_score"):
        merged[score_col] = (
            pd.to_numeric(merged[score_col], errors="coerce")
            .fillna(0.0)
            .astype(float)
        )

    merged["from_dense"] = merged["semantic_score"] > 0
    merged["from_behavior"] = merged["behavior_score"] > 0
    merged["from_geo"] = merged["geo_score"] > 0

    # 动态权重融合
    if adaptive_fusion_enabled:
        actual_dense_weight, actual_behavior_weight, actual_geo_weight = adaptive_fusion(
            user_history_length=user_history_length,
            base_dense_weight=dense_weight,
            base_behavior_weight=behavior_weight,
            base_geo_weight=geo_weight
        )
        print(f"  动态权重: dense={actual_dense_weight:.3f}, behavior={actual_behavior_weight:.3f}, geo={actual_geo_weight:.3f}")
        print(f"  (用户历史: {user_history_length} 次交互)")
    else:
        actual_dense_weight, actual_behavior_weight, actual_geo_weight = dense_weight, behavior_weight, geo_weight

    if fusion == "rrf":
        merged["final_score"] = 0.0
        for rank_col in ["semantic_score_rank", "behavior_score_rank", "geo_score_rank"]:
            merged["final_score"] += merged[rank_col].apply(
                lambda x: 0.0 if pd.isna(x) else 1.0 / (rrf_k + float(x))
            )
    else:
        merged["semantic_cal"] = _calibrate_scores(merged["semantic_score"], method=calibrate)
        merged["behavior_cal"] = _calibrate_scores(merged["behavior_score"], method=calibrate)
        merged["geo_cal"] = _calibrate_scores(merged["geo_score"], method=calibrate)
        merged["final_score"] = (
            actual_dense_weight * merged["semantic_cal"]
            + actual_behavior_weight * merged["behavior_cal"]
            + actual_geo_weight * merged["geo_cal"]
        )

    merged = merged.sort_values("final_score", ascending=False).reset_index(drop=True)
    if final_topk:
        merged = merged.head(final_topk).reset_index(drop=True)

    print(f"  语义召回: {len(semantic)}")
    print(f"  行为召回: {len(behavior)}")
    print(f"  地理召回: {len(geo)}")
    print(f"  融合候选: {len(merged)}")

    # 添加融合权重信息到返回结果（用于调试）
    merged.attrs['fusion_weights'] = {
        'dense': actual_dense_weight,
        'behavior': actual_behavior_weight,
        'geo': actual_geo_weight,
        'adaptive': adaptive_fusion_enabled,
        'user_history_length': user_history_length
    }

    return merged


if __name__ == "__main__":
    candidates = merge_candidates(
        query_text="想去新疆看雪山和草原",
        topk_dense=30,
        topk_seq=20,
        topk_geo=20,
        province_filter="新疆",
        fusion="rrf",
    )
    print(candidates.head(10)[["name", "city", "final_score", "from_dense", "from_behavior", "from_geo"]])
