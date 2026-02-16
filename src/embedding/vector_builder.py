"""
POI embedding build/search with optional FAISS acceleration.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Optional import for BGE-M3 (may not be available due to version conflicts)
try:
    try:
        from src.embedding.bge_m3_encoder import BGEM3Encoder
    except ImportError:
        from embedding.bge_m3_encoder import BGEM3Encoder
except (ImportError, Exception):
    BGEM3Encoder = None

# Optional import for Qwen3 embedding
try:
    try:
        from src.embedding.qwen3_encoder import Qwen3Embedding
    except ImportError:
        from embedding.qwen3_encoder import Qwen3Embedding
except (ImportError, Exception):
    Qwen3Embedding = None

try:
    from src.utils.id_mapping import normalize_poi_id
except ImportError:
    from utils.id_mapping import normalize_poi_id

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BGE_MODEL = os.getenv("GOAFAR_BGE_MODEL", "models/Xorbits/bge-m3")
DEFAULT_QWEN_MODEL = os.getenv("GOAFAR_QWEN3_EMBEDDING_MODEL", "models/Qwen3-Embedding-4B")


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _resolve_model_path(model_path: str | None) -> str:
    if model_path:
        candidate = _resolve_path(model_path)
        if candidate.exists():
            return str(candidate)
        return model_path

    candidate = _resolve_path(DEFAULT_BGE_MODEL)
    if candidate.exists():
        return str(candidate)
    return DEFAULT_BGE_MODEL


def _resolve_qwen_model_path(model_path: str | None) -> str:
    if model_path:
        candidate = _resolve_path(model_path)
        if candidate.exists() and "qwen" in candidate.name.lower():
            return str(candidate)
        if "qwen" in str(model_path).lower():
            return model_path

    candidate = _resolve_path(DEFAULT_QWEN_MODEL)
    if candidate.exists():
        return str(candidate)
    return DEFAULT_QWEN_MODEL


def _looks_like_qwen_model(model_path: str | None) -> bool:
    if not model_path:
        return False
    return "qwen" in str(model_path).lower()


def _create_encoder(model_path: str | None, use_gpu: bool) -> tuple[Any, str, str]:
    """
    创建可用的 embedding encoder，优先遵循 model_path 指向。
    返回: (encoder, resolved_model_path, backend_name)
    """
    errors: list[str] = []
    prefer_qwen = _looks_like_qwen_model(model_path)

    # 1) 优先尝试 Qwen（当显式配置了 qwen 路径时）
    if prefer_qwen and Qwen3Embedding is not None:
        qwen_model = _resolve_qwen_model_path(model_path)
        try:
            encoder = Qwen3Embedding(model_path=qwen_model, use_gpu=use_gpu)
            if hasattr(encoder, "is_available") and not encoder.is_available():
                raise RuntimeError(f"Qwen3Embedding 不可用: {qwen_model}")
            return encoder, qwen_model, "qwen3"
        except Exception as exc:
            errors.append(f"Qwen3 初始化失败: {exc}")

    # 2) 尝试 BGE-M3
    if BGEM3Encoder is not None:
        bge_model = _resolve_model_path(model_path if not prefer_qwen else None)
        try:
            return BGEM3Encoder(model_path=bge_model, use_gpu=use_gpu), bge_model, "bge-m3"
        except Exception as exc:
            errors.append(f"BGE-M3 初始化失败: {exc}")
    else:
        errors.append("BGE-M3 不可用（FlagEmbedding 导入失败）")

    # 3) 最后回退 Qwen（即使 model_path 没显式写 qwen，也可作为后备）
    if Qwen3Embedding is not None:
        qwen_model = _resolve_qwen_model_path(model_path)
        try:
            encoder = Qwen3Embedding(model_path=qwen_model, use_gpu=use_gpu)
            if hasattr(encoder, "is_available") and not encoder.is_available():
                raise RuntimeError(f"Qwen3Embedding 不可用: {qwen_model}")
            return encoder, qwen_model, "qwen3"
        except Exception as exc:
            errors.append(f"Qwen3 回退失败: {exc}")
    else:
        errors.append("Qwen3Embedding 不可用")

    raise RuntimeError("无法初始化任何 embedding 模型；" + "；".join(errors))


def _build_poi_texts(df: pd.DataFrame) -> list[str]:
    texts: list[str] = []
    for _, row in df.iterrows():
        parts = [str(row.get("name", ""))]

        if pd.notna(row.get("province")):
            parts.append(str(row["province"]))
        if pd.notna(row.get("city")) and row.get("city") != row.get("province"):
            parts.append(str(row["city"]))

        if pd.notna(row.get("description")) and row.get("description"):
            desc = str(row["description"]).replace("\n", " ")[:220]
            parts.append(desc)

        stay_min = row.get("stay_min", 60)
        if pd.notna(stay_min):
            parts.append(f"建议停留{float(stay_min) / 60:.1f}小时")

        texts.append(" ".join(parts))
    return texts


def build_faiss_index(
    embeddings: np.ndarray,
    index_file: str = "outputs/emb/poi_faiss.index",
    index_type: str = "flat",
    nlist: int = 100
) -> bool:
    """
    Build a FAISS index for fast similarity search.

    Args:
        embeddings: 向量矩阵 (n, d)
        index_file: 索引文件保存路径
        index_type: 索引类型
            - "flat": 精确搜索（IndexFlatIP），适合小规模数据
            - "ivf": 倒排索引（IndexIVFFlat），适合大规模数据
        nlist: IVF索引的聚类中心数量（仅当index_type="ivf"时有效）

    Returns:
        bool: 是否成功构建索引
    """
    if faiss is None:
        print("⚠️ faiss 未安装，跳过索引构建")
        return False

    index_path = _resolve_path(index_file)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    vecs = embeddings.astype("float32")
    n_vecs, dim = vecs.shape

    print(f"构建 FAISS 索引: {n_vecs} 个向量, 维度 {dim}")

    if index_type == "flat":
        # 精确搜索（适合小规模数据）
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
    elif index_type == "ivf":
        # 倒排索引（适合大规模数据）
        # 根据数据量自动调整nlist
        if n_vecs < 10000:
            nlist = max(100, n_vecs // 50)
        elif n_vecs < 100000:
            nlist = max(500, n_vecs // 100)
        else:
            nlist = max(1000, n_vecs // 200)

        print(f"  使用 IVF 索引，聚类中心数: {nlist}")

        # 需要先训练索引
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

        # 确保有足够的向量进行训练
        if n_vecs < nlist:
            print(f"  ⚠️ 向量数量 ({n_vecs}) 小于聚类中心数 ({nlist})，改用Flat索引")
            index = faiss.IndexFlatIP(dim)
            index.add(vecs)
        else:
            # 训练索引
            print("  正在训练索引...")
            training_size = min(n_vecs, nlist * 10)  # 使用足够的向量进行训练
            index.train(vecs[:training_size].copy())
            # 添加向量
            print("  正在添加向量...")
            index.add(vecs)
            print(f"  ✓ 索引训练完成，已添加 {index.ntotal} 个向量")
    else:
        raise ValueError(f"不支持的索引类型: {index_type}")

    # 保存索引
    faiss.write_index(index, str(index_path))
    print(f"✓ FAISS 索引已写入: {index_path}")
    print(f"  索引类型: {index_type}")
    print(f"  向量数量: {index.ntotal}")

    return True


def build_poi_embeddings(
    poi_csv: str = "data/all/poi_with_coords.csv",
    output_dir: str = "outputs/emb",
    model_path: str | None = None,
    use_gpu: bool = True,
    build_faiss: bool = True,
    faiss_index_file: str = "outputs/emb/poi_faiss.index",
):
    """
    Build dense POI embeddings and metadata.
    """
    output_path = _resolve_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    poi_path = _resolve_path(poi_csv)
    df = pd.read_csv(poi_path, low_memory=False)
    if "poi_id" in df.columns:
        df["poi_id"] = df["poi_id"].apply(normalize_poi_id)

    print(f"✓ 加载 {len(df)} 个 POI")
    texts = _build_poi_texts(df)

    encoder, resolved_model, backend_name = _create_encoder(model_path=model_path, use_gpu=use_gpu)
    print(f"初始化语义编码器[{backend_name}]: {resolved_model}")

    embeddings_dict = encoder.encode_texts(
        texts,
        batch_size=64,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert=False,
    )

    dense_vecs = embeddings_dict["dense_vecs"].astype("float32")
    emb_file = output_path / "poi_emb.npy"
    meta_file = output_path / "poi_meta.csv"

    np.save(emb_file, dense_vecs)
    df.to_csv(meta_file, index=False)

    print(f"✓ 保存向量: {emb_file}")
    print(f"✓ 保存元数据: {meta_file}")

    if build_faiss:
        try:
            build_faiss_index(dense_vecs, index_file=faiss_index_file)
        except Exception as exc:
            print(f"⚠️ 构建 FAISS 索引失败，已降级到 numpy 检索: {exc}")

    return embeddings_dict, df


def ensure_embedding_artifacts(
    emb_file: str = "outputs/emb/poi_emb.npy",
    meta_file: str = "outputs/emb/poi_meta.csv",
    poi_csv: str = "data/all/poi_with_coords.csv",
    output_dir: str = "outputs/emb",
    model_path: str | None = None,
    use_gpu: bool = True,
    auto_build: bool = True,
    build_faiss: bool = True,
    faiss_index_file: str = "outputs/emb/poi_faiss.index",
) -> bool:
    """
    Check whether embedding artifacts exist; auto-build when allowed.
    """
    emb_path = _resolve_path(emb_file)
    meta_path = _resolve_path(meta_file)
    poi_path = _resolve_path(poi_csv)

    def _artifacts_match_source() -> tuple[bool, str]:
        if not emb_path.exists() or not meta_path.exists():
            return False, "artifact_missing"
        if not poi_path.exists():
            # 无法校验源数据时，仅要求产物存在
            return True, "source_missing_skip_check"

        try:
            meta_df = pd.read_csv(meta_path, usecols=["poi_id"], low_memory=False)
            poi_df = pd.read_csv(poi_path, usecols=["poi_id"], low_memory=False)

            meta_ids = set(meta_df["poi_id"].astype(str).map(normalize_poi_id))
            poi_ids = set(poi_df["poi_id"].astype(str).map(normalize_poi_id))

            if meta_ids != poi_ids:
                missing = len(poi_ids - meta_ids)
                extra = len(meta_ids - poi_ids)
                return False, f"id_mismatch(missing={missing},extra={extra})"

            emb_rows = int(np.load(emb_path, mmap_mode="r").shape[0])
            if emb_rows != len(meta_df):
                return False, f"row_mismatch(emb={emb_rows},meta={len(meta_df)})"

            return True, "ok"
        except Exception as exc:
            return False, f"validation_error:{exc}"

    matched, reason = _artifacts_match_source()
    if matched:
        return True

    if not auto_build:
        print(f"⚠️ 向量产物不可用或不一致({reason})，且 auto_build=False")
        return False

    print(f"⚠️ 检测到向量产物缺失/不一致({reason})，正在自动构建...")
    build_poi_embeddings(
        poi_csv=poi_csv,
        output_dir=output_dir,
        model_path=model_path,
        use_gpu=use_gpu,
        build_faiss=build_faiss,
        faiss_index_file=faiss_index_file,
    )
    matched, _ = _artifacts_match_source()
    return matched


def _search_numpy(embeddings: np.ndarray, query_vec: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    scores = embeddings @ query_vec
    top_indices = np.argsort(-scores)[:topk]
    return top_indices, scores[top_indices]


def _search_faiss(
    embeddings: np.ndarray,
    query_vec: np.ndarray,
    topk: int,
    faiss_index_file: str = "outputs/emb/poi_faiss.index",
) -> Tuple[np.ndarray, np.ndarray]:
    if faiss is None:
        raise RuntimeError("faiss 未安装")

    index_path = _resolve_path(faiss_index_file)
    if not index_path.exists():
        build_faiss_index(embeddings, index_file=faiss_index_file)

    index = faiss.read_index(str(index_path))
    scores, indices = index.search(query_vec.reshape(1, -1).astype("float32"), topk)
    return indices[0], scores[0]


def search_similar_pois(
    query_text: str,
    topk: int = 50,
    emb_file: str = "outputs/emb/poi_emb.npy",
    meta_file: str = "outputs/emb/poi_meta.csv",
    model_path: str | None = None,
    use_gpu: bool = True,
    backend: str = "auto",
    faiss_index_file: str = "outputs/emb/poi_faiss.index",
    auto_build: bool = False,
    poi_csv: str = "data/all/poi_with_coords.csv",
):
    """
    Search similar POIs by dense embedding.

    backend:
    - auto: prefer FAISS when available
    - faiss: force FAISS
    - numpy: force brute-force matrix multiply
    """
    if not ensure_embedding_artifacts(
        emb_file=emb_file,
        meta_file=meta_file,
        poi_csv=poi_csv,
        output_dir=str(Path(meta_file).parent),
        model_path=model_path,
        use_gpu=use_gpu,
        auto_build=auto_build,
        build_faiss=(backend in {"auto", "faiss"}),
        faiss_index_file=faiss_index_file,
    ):
        raise FileNotFoundError(
            "缺少向量产物，请先构建: python src/embedding/build_embeddings_gpu.py"
        )

    emb_path = _resolve_path(emb_file)
    meta_path = _resolve_path(meta_file)
    embeddings = np.load(emb_path).astype("float32")
    metadata = pd.read_csv(meta_path, low_memory=False)
    if "poi_id" in metadata.columns:
        metadata["poi_id"] = metadata["poi_id"].apply(normalize_poi_id)

    topk = max(1, min(topk, len(metadata)))

    encoder, resolved_model, backend_name = _create_encoder(model_path=model_path, use_gpu=use_gpu)
    print(f"语义检索使用编码器[{backend_name}]: {resolved_model}")
    query_emb = encoder.encode_query(query_text, return_dense=True)
    query_vec = query_emb["dense_vec"].astype("float32")

    retrieval_backend = "numpy"
    if backend == "faiss" or (backend == "auto" and faiss is not None):
        try:
            top_indices, top_scores = _search_faiss(
                embeddings,
                query_vec,
                topk=topk,
                faiss_index_file=faiss_index_file,
            )
            retrieval_backend = "faiss"
        except Exception as exc:
            if backend == "faiss":
                raise
            print(f"⚠️ FAISS 检索失败，降级 numpy: {exc}")
            top_indices, top_scores = _search_numpy(embeddings, query_vec, topk=topk)
    else:
        top_indices, top_scores = _search_numpy(embeddings, query_vec, topk=topk)

    results = metadata.iloc[top_indices].copy()
    results["semantic_score"] = top_scores
    results["rank"] = range(1, len(results) + 1)
    results["retrieval_backend"] = retrieval_backend
    return results.reset_index(drop=True)


if __name__ == "__main__":
    print("=" * 60)
    print("测试 POI 向量构建")
    print("=" * 60)

    build_poi_embeddings(use_gpu=False)
    results = search_similar_pois("想去新疆看雪山和湖泊", topk=10, use_gpu=False, backend="auto")
    print(results.head(10)[["name", "city", "semantic_score", "retrieval_backend"]])
