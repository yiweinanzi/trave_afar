"""
Qwen3-Embedding-4B 语义编码器
使用 sentence-transformers 加载 Qwen3-Embedding-4B 模型
"""
import os
import numpy as np
from typing import List, Dict, Union

class Qwen3Embedding:
    """Qwen3-Embedding-4B 编码器封装类"""

    def __init__(self, model_path=None, use_gpu=True, cache_dir=None):
        """
        初始化编码器

        Args:
            model_path: 本地模型路径（如果为None，使用默认路径）
            use_gpu: 是否使用GPU
            cache_dir: 模型缓存目录
        """
        self.model_path = model_path or "models/Qwen3-Embedding-4B"
        self.use_gpu = use_gpu and self._check_gpu()
        self.cache_dir = cache_dir

        # 确定设备
        if self.use_gpu:
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        print(f"初始化 Qwen3-Embedding-4B 编码器...")
        print(f"  模型: {self.model_path}")
        print(f"  设备: {self.device}")

        # 动态导入 sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "请安装 sentence-transformers: pip install sentence-transformers"
            )

        # 加载模型
        self.model = SentenceTransformer(
            self.model_path,
            device=self.device,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print("✓ 模型加载完成")

        # 获取向量维度
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"  向量维度: {self.embedding_dim}")

    def _check_gpu(self):
        """检查GPU是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 64,
        max_length: int = 512,
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert: bool = False,
        show_progress: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        编码文本列表

        Args:
            texts: 文本列表
            batch_size: 批处理大小
            max_length: 最大文本长度（注意：sentence-transformers可能不直接支持）
            return_dense: 是否返回dense向量
            return_sparse: 是否返回sparse向量（暂不支持）
            return_colbert: 是否返回colbert向量（暂不支持）
            show_progress: 是否显示进度条

        Returns:
            字典，包含 dense_vecs
        """
        if return_sparse:
            print("⚠️ Qwen3-Embedding 不支持稀疏向量，忽略 return_sparse")
        if return_colbert:
            print("⚠️ Qwen3-Embedding 不支持 ColBERT 向量，忽略 return_colbert")

        if not return_dense:
            raise ValueError("至少需要返回 dense 向量")

        # 使用 sentence-transformers 编码
        dense_vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # 归一化以便使用内积计算相似度
        )

        return {
            'dense_vecs': dense_vecs
        }

    def encode_query(
        self,
        query: str,
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        编码单个查询

        Args:
            query: 查询文本
            return_dense: 是否返回dense向量
            return_sparse: 是否返回sparse向量（暂不支持）
            return_colbert: 是否返回colbert向量（暂不支持）

        Returns:
            字典，包含 dense_vec
        """
        if return_sparse:
            print("⚠️ Qwen3-Embedding 不支持稀疏向量，忽略 return_sparse")
        if return_colbert:
            print("⚠️ Qwen3-Embedding 不支持 ColBERT 向量，忽略 return_colbert")

        if not return_dense:
            raise ValueError("至少需要返回 dense 向量")

        # 编码查询
        dense_vec = self.model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        return {
            'dense_vec': dense_vec
        }

    def compute_similarity(
        self,
        query_embedding: Dict[str, np.ndarray],
        corpus_embeddings: Dict[str, np.ndarray],
        method: str = 'dense'
    ) -> np.ndarray:
        """
        计算相似度

        Args:
            query_embedding: 查询向量字典
            corpus_embeddings: 语料库向量字典
            method: 只支持 'dense'

        Returns:
            相似度分数数组
        """
        if method != 'dense':
            raise ValueError(f"Qwen3-Embedding 只支持 dense 方法，不支持: {method}")

        # Dense向量：内积相似度（向量已归一化）
        query_vec = query_embedding['dense_vec']
        corpus_vecs = corpus_embeddings['dense_vecs']
        scores = corpus_vecs @ query_vec
        return scores


if __name__ == "__main__":
    # 测试
    from pathlib import Path

    # 测试模型路径
    model_env = os.getenv("GOAFAR_QWEN3_MODEL", "models/Qwen3-Embedding-4B")
    model_path = Path(model_env)
    if not model_path.is_absolute():
        model_path = Path(__file__).resolve().parents[2] / model_path

    encoder = Qwen3Embedding(
        model_path=str(model_path),
        use_gpu=False  # 测试时使用CPU
    )

    # 测试编码
    texts = [
        "喀纳斯湖位于新疆北部，是一个美丽的高山湖泊",
        "赛里木湖被誉为大西洋的最后一滴眼泪",
        "天池是新疆著名的旅游景点"
    ]
    print("\n测试编码...")
    embeddings = encoder.encode_texts(texts, show_progress=True)
    print(f"✓ Dense向量维度: {embeddings['dense_vecs'].shape}")

    # 测试查询
    print("\n测试查询编码...")
    query_emb = encoder.encode_query("想去新疆看湖泊")
    print(f"✓ 查询向量维度: {query_emb['dense_vec'].shape}")

    # 测试相似度计算
    print("\n测试相似度计算...")
    scores = encoder.compute_similarity(query_emb, embeddings)
    for i, (text, score) in enumerate(zip(texts, scores)):
        print(f"  [{i+1}] {text[:40]}... -> {score:.4f}")
