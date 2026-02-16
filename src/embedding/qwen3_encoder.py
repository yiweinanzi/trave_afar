"""
Qwen3-Embedding-4B 语义编码器
支持两种加载方式：
1. sentence_transformers（优先）
2. transformers原生接口（兼容）

自动检测可用方式，统一输出格式
"""
import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

import numpy as np

# 设置日志
logger = logging.getLogger(__name__)


def retry_on_error(max_retries: int = 3, delay: float = 1.0, exceptions: Tuple = (Exception,)):
    """
    装饰器：在模型加载失败时自动重试

    Args:
        max_retries: 最大重试次数
        delay: 重试延迟（秒）
        exceptions: 需要重试的异常类型
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"{func.__name__} 失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                        logger.info(f"等待 {delay} 秒后重试...")
                        time.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} 在 {max_retries} 次尝试后仍然失败")
            raise last_exception
        return wrapper
    return decorator


class Qwen3Embedding:
    """Qwen3-Embedding-4B 编码器封装类

    支持两种加载方式：
    - sentence_transformers: 使用SentenceTransformer类（优先）
    - transformers: 使用AutoModel类（后备）

    自动检测可用方式，统一输出格式
    """

    # 类变量：记录加载方式
    _loading_method: Optional[str] = None

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_gpu: bool = True,
        cache_dir: Optional[str] = None,
        prefer_sentence_transformers: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        初始化编码器

        Args:
            model_path: 本地模型路径（如果为None，使用默认路径）
            use_gpu: 是否使用GPU
            cache_dir: 模型缓存目录
            prefer_sentence_transformers: 优先使用sentence_transformers
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
        """
        self.model_path = self._resolve_model_path(model_path)
        self.use_gpu = use_gpu and self._check_gpu()
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.prefer_sentence_transformers = prefer_sentence_transformers

        # 确定设备
        if self.use_gpu:
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        logger.info(f"初始化 Qwen3-Embedding-4B 编码器...")
        logger.info(f"  模型: {self.model_path}")
        logger.info(f"  设备: {self.device}")

        # 模型组件
        self.model = None
        self.tokenizer = None
        self.backend = None  # 'sentence_transformers' 或 'transformers'
        self.embedding_dim = None

        # 加载模型（带重试）
        self._load_model()

    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        """解析模型路径"""
        if model_path is None:
            # 检查环境变量
            env_path = os.getenv("GOAFAR_QWEN3_EMBEDDING_MODEL")
            if env_path and Path(env_path).exists():
                return env_path

            # 默认路径
            project_root = Path(__file__).resolve().parents[2]
            default_paths = [
                project_root / "models" / "Qwen3-Embedding-4B",
                project_root / "models" / "Qwen3-Embedding-4B",
                Path("models/Qwen3-Embedding-4B"),
            ]

            for path in default_paths:
                if path.exists():
                    return str(path)

            # 如果都不存在，返回默认路径
            return str(project_root / "models" / "Qwen3-Embedding-4B")

        return model_path

    def _check_gpu(self) -> bool:
        """检查GPU是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _check_model_path(self) -> bool:
        """检查模型路径是否存在"""
        model_path = Path(self.model_path)
        return model_path.exists() and model_path.is_dir()

    @retry_on_error(max_retries=3, delay=1.0)
    def _try_load_sentence_transformers(self) -> bool:
        """尝试使用sentence_transformers加载"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.info("sentence_transformers 未安装")
            return False

        try:
            logger.info("尝试使用 sentence_transformers 加载...")
            self.model = SentenceTransformer(
                self.model_path,
                device=self.device,
                cache_folder=self.cache_dir,
                trust_remote_code=True
            )
            self.backend = 'sentence_transformers'
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info("✓ 使用 sentence_transformers 加载成功")
            Qwen3Embedding._loading_method = 'sentence_transformers'
            return True
        except Exception as e:
            logger.warning(f"sentence_transformers 加载失败: {e}")
            return False

    @retry_on_error(max_retries=2, delay=1.0)
    def _try_load_transformers(self) -> bool:
        """尝试使用transformers原生接口加载"""
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
        except ImportError:
            logger.info("transformers 未安装")
            return False

        try:
            logger.info("尝试使用 transformers 原生接口加载...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )

            if self.use_gpu:
                self.model = self.model.to(self.device)

            self.backend = 'transformers'

            # 获取embedding维度（尝试编码一次）
            with torch.no_grad():
                dummy_input = self.tokenizer("测试", return_tensors="pt")
                if self.use_gpu:
                    dummy_input = {k: v.to(self.device) for k, v in dummy_input.items()}
                outputs = self.model(**dummy_input)
                self.embedding_dim = outputs.last_hidden_state.shape[-1]

            logger.info("✓ 使用 transformers 加载成功")
            Qwen3Embedding._loading_method = 'transformers'
            return True
        except Exception as e:
            logger.warning(f"transformers 加载失败: {e}")
            return False

    def _load_model(self):
        """加载模型（自动检测可用方式）"""
        if not self._check_model_path():
            logger.warning(f"模型路径不存在: {self.model_path}")
            logger.warning("将使用空的Embedding实例")

        # 检查路径是否存在，如果不存在且强制加载，则尝试
        if self.prefer_sentence_transformers:
            # 优先使用 sentence_transformers
            if self._try_load_sentence_transformers():
                return
            # 后备使用 transformers
            if self._try_load_transformers():
                return
        else:
            # 优先使用 transformers
            if self._try_load_transformers():
                return
            # 后备使用 sentence_transformers
            if self._try_load_sentence_transformers():
                return

        # 两种方式都失败
        if self._check_model_path():
            logger.error("两种加载方式都失败，请检查模型文件完整性")
        else:
            logger.warning(f"模型路径不存在，将使用fallback模式")

        self.model = None
        self.tokenizer = None
        self.backend = None
        self.embedding_dim = None

    def is_available(self) -> bool:
        """检查模型是否可用"""
        return self.model is not None

    def get_loading_method(self) -> Optional[str]:
        """获取实际使用的加载方式"""
        return self.backend

    @classmethod
    def get_global_loading_method(cls) -> Optional[str]:
        """获取全局记录的加载方式"""
        return cls._loading_method

    def _encode_with_sentence_transformers(
        self,
        texts: List[str],
        batch_size: int,
        show_progress: bool
    ) -> np.ndarray:
        """使用sentence_transformers编码"""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def _encode_with_transformers(
        self,
        texts: List[str],
        batch_size: int,
        show_progress: bool
    ) -> np.ndarray:
        """使用transformers原生接口编码"""
        import torch
        from tqdm import tqdm

        embeddings = []

        # 批量处理
        for i in tqdm(range(0, len(texts), batch_size), desc="编码", disable=not show_progress):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            if self.use_gpu:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 编码
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用平均池化
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                # 归一化
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)

            embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)

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
        编码文本列表（统一接口）

        Args:
            texts: 文本列表
            batch_size: 批处理大小
            max_length: 最大文本长度
            return_dense: 是否返回dense向量
            return_sparse: 是否返回sparse向量（暂不支持）
            return_colbert: 是否返回colbert向量（暂不支持）
            show_progress: 是否显示进度条

        Returns:
            字典，包含 dense_vecs（统一输出格式）
        """
        if return_sparse:
            logger.warning("Qwen3-Embedding 不支持稀疏向量，忽略 return_sparse")
        if return_colbert:
            logger.warning("Qwen3-Embedding 不支持 ColBERT 向量，忽略 return_colbert")

        if not return_dense:
            raise ValueError("至少需要返回 dense 向量")

        # 模型未加载时的fallback
        if self.model is None:
            logger.warning("模型未加载，使用零向量")
            return {'dense_vecs': np.zeros((len(texts), self.embedding_dim or 768))}

        try:
            # 根据backend选择编码方式
            if self.backend == 'sentence_transformers':
                dense_vecs = self._encode_with_sentence_transformers(texts, batch_size, show_progress)
            else:  # 'transformers'
                dense_vecs = self._encode_with_transformers(texts, batch_size, show_progress)

            return {'dense_vecs': dense_vecs}
        except Exception as e:
            logger.error(f"编码失败: {e}")
            return {'dense_vecs': np.zeros((len(texts), self.embedding_dim or 768))}

    def encode_query(
        self,
        query: str,
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        编码单个查询（统一接口）

        Args:
            query: 查询文本
            return_dense: 是否返回dense向量
            return_sparse: 是否返回sparse向量（暂不支持）
            return_colbert: 是否返回colbert向量（暂不支持）

        Returns:
            字典，包含 dense_vec（统一输出格式）
        """
        if return_sparse:
            logger.warning("Qwen3-Embedding 不支持稀疏向量，忽略 return_sparse")
        if return_colbert:
            logger.warning("Qwen3-Embedding 不支持 ColBERT 向量，忽略 return_colbert")

        if not return_dense:
            raise ValueError("至少需要返回 dense 向量")

        # 模型未加载时的fallback
        if self.model is None:
            logger.warning("模型未加载，使用零向量")
            return {'dense_vec': np.zeros(self.embedding_dim or 768)}

        try:
            # 编码单个查询
            result = self.encode_texts(
                [query],
                batch_size=1,
                return_dense=return_dense,
                show_progress=False
            )
            return {'dense_vec': result['dense_vecs'][0]}
        except Exception as e:
            logger.error(f"查询编码失败: {e}")
            return {'dense_vec': np.zeros(self.embedding_dim or 768)}

    def compute_similarity(
        self,
        query_embedding: Dict[str, np.ndarray],
        corpus_embeddings: Dict[str, np.ndarray],
        method: str = 'dense'
    ) -> np.ndarray:
        """
        计算相似度（统一接口）

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

    def get_embedding_dim(self) -> Optional[int]:
        """获取embedding维度"""
        return self.embedding_dim

    def cleanup(self):
        """清理模型资源"""
        if self.model is not None:
            if self.use_gpu:
                import torch
                if hasattr(self.model, 'to'):
                    self.model.to('cpu')
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        import gc
        gc.collect()

        if self.use_gpu:
            import torch
            torch.cuda.empty_cache()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，自动清理"""
        self.cleanup()
        return False


if __name__ == "__main__":
    import logging
    from pathlib import Path

    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 测试模型路径
    model_env = os.getenv("GOAFAR_QWEN3_MODEL", "models/Qwen3-Embedding-4B")
    model_path = Path(model_env)
    if not model_path.is_absolute():
        model_path = Path(__file__).resolve().parents[2] / model_path

    print("=" * 60)
    print("Qwen3 Embedding 统一接口测试")
    print("=" * 60)

    # 测试1: 使用 sentence_transformers
    print("\n[测试1] 尝试优先使用 sentence_transformers 加载:")
    encoder = Qwen3Embedding(
        model_path=str(model_path),
        use_gpu=False,
        prefer_sentence_transformers=True
    )
    print(f"加载方式: {encoder.get_loading_method()}")
    print(f"模型可用: {encoder.is_available()}")
    print(f"Embedding维度: {encoder.get_embedding_dim()}")

    if encoder.is_available():
        # 测试编码
        texts = [
            "喀纳斯湖位于新疆北部，是一个美丽的高山湖泊",
            "赛里木湖被誉为大西洋的最后一滴眼泪",
            "天池是新疆著名的旅游景点"
        ]
        print("\n测试编码...")
        embeddings = encoder.encode_texts(texts, show_progress=True)
        print(f"Dense向量维度: {embeddings['dense_vecs'].shape}")

        # 测试查询
        print("\n测试查询编码...")
        query_emb = encoder.encode_query("想去新疆看湖泊")
        print(f"查询向量维度: {query_emb['dense_vec'].shape}")

        # 测试相似度计算
        print("\n测试相似度计算...")
        scores = encoder.compute_similarity(query_emb, embeddings)
        for i, (text, score) in enumerate(zip(texts, scores)):
            print(f"  [{i+1}] {text[:40]}... -> {score:.4f}")

        # 测试上下文管理器
        print("\n[测试2] 测试上下文管理器（自动清理）:")
        with Qwen3Embedding(model_path=str(model_path), use_gpu=False) as enc:
            print(f"内部加载方式: {enc.get_loading_method()}")
            test_emb = enc.encode_query("测试")
            print(f"测试查询向量维度: {test_emb['dense_vec'].shape}")
        print("上下文退出，资源已释放")

        # 测试3: 使用 transformers
        print("\n[测试3] 尝试优先使用 transformers 加载:")
        encoder2 = Qwen3Embedding(
            model_path=str(model_path),
            use_gpu=False,
            prefer_sentence_transformers=False
        )
        print(f"加载方式: {encoder2.get_loading_method()}")
        print(f"模型可用: {encoder2.is_available()}")

        if encoder2.is_available():
            test_emb2 = encoder2.encode_query("测试 transformers")
            print(f"测试查询向量维度: {test_emb2['dense_vec'].shape}")
    else:
        print("模型不可用，请检查模型路径")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
