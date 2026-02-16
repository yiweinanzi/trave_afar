"""
Qwen3 Reranker
使用Qwen3-Reranker-4B对Top-K候选进行重排序

参考: models/download.md
from transformers import AutoModelForSequenceClassification, AutoTokenizer
pipe = pipeline("text-generation", model="Qwen/Qwen3-Reranker-4B")

包含自动重试机制和详细错误日志
"""
import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable, Any

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import pipeline as PipelineFactory
except ImportError:
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 设置日志
logger = logging.getLogger(__name__)


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """
    装饰器：在函数执行失败时自动重试，支持指数退避
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning("{} 失败 (尝试 {}/{}): {}".format(
                            func.__name__, attempt + 1, max_retries, e
                        ))
                        if on_retry:
                            on_retry(attempt + 1, e)
                        logger.info("等待 {:.1f} 秒后重试...".format(current_delay))
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error("{} 在 {} 次尝试后仍然失败".format(
                            func.__name__, max_retries
                        ))
                        logger.error("最后错误: {}".format(e))
                        logger.debug("错误详情:", exc_info=True)
            raise last_exception
        return wrapper
    return decorator


class QwenReranker:
    """
    Qwen3 Reranker
    使用Qwen3-Reranker-4B对候选POI进行精排
    包含自动重试机制和详细错误日志
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_gpu: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff

        if torch is None:
            logger.warning("torch未安装，Reranker将使用规则回退")
            self.model = None
            self.tokenizer = None
            self.use_gpu = False
            self.device = "cpu"
            return

        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"

        if AutoModelForSequenceClassification is None:
            logger.warning("transformers未安装，Reranker将使用规则回退")
            self.model = None
            self.tokenizer = None
            return

        # 解析模型路径
        if model_path is None:
            model_path = os.getenv(
                "GOAFAR_RERANKER_MODEL",
                str(PROJECT_ROOT / "models/Qwen3-Reranker-4B")
            )

        if not Path(model_path).exists():
            logger.warning("模型路径不存在: {}".format(model_path))
            logger.warning("Reranker将使用规则回退")
            self.model = None
            self.tokenizer = None
            return

        # 使用重试机制加载模型
        self._load_model_with_retry(model_path)

    def _load_model_with_retry(self, model_path: str):
        """带重试机制的模型加载"""
        last_exception = None
        current_delay = self.retry_delay

        for attempt in range(self.max_retries):
            try:
                logger.info("加载Qwen3-Reranker-4B: {}".format(model_path))
                logger.info("设备: {}".format(self.device))

                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )

                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.use_gpu else torch.float32
                )

                if self.use_gpu:
                    self.model = self.model.to(self.device)

                self.model.eval()
                logger.info("Reranker模型加载完成")
                return

            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    logger.warning("模型加载失败 (尝试 {}/{}): {}".format(
                        attempt + 1, self.max_retries, e
                    ))
                    logger.info("等待 {:.1f} 秒后重试...".format(current_delay))
                    time.sleep(current_delay)
                    current_delay *= self.retry_backoff
                else:
                    logger.error("模型加载在 {} 次尝试后仍然失败".format(
                        self.max_retries
                    ))
                    logger.error("最后错误: {}".format(e))
                    logger.debug("错误详情:", exc_info=True)

        # 所有重试都失败
        logger.warning("将使用规则回退")
        self.model = None
        self.tokenizer = None

    def is_available(self) -> bool:
        """检查模型是否可用"""
        return self.model is not None

    def get_device(self) -> str:
        """获取当前使用的设备"""
        return self.device

    @staticmethod
    def _safe_text(value: Any) -> str:
        """将候选字段安全转换为字符串，屏蔽 None/NaN。"""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, float) and value != value:  # NaN
            return ""
        return str(value)

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        topk: int = 20,
        score_key: str = "score"
    ) -> List[Dict]:
        """对候选POI进行重排序（带重试）"""
        if self.model is None or len(candidates) <= topk:
            return self._rule_based_rerank(query, candidates, topk)

        try:
            scores = self._compute_scores_with_retry(query, candidates)

            # 更新分数并排序
            for i, candidate in enumerate(candidates):
                if i < len(scores):
                    candidate["reranker_score"] = float(scores[i])

            ranked = sorted(candidates, key=lambda x: x.get("reranker_score", 0), reverse=True)
            return ranked[:topk]

        except Exception as e:
            logger.error("Reranker计算失败: {}".format(e))
            logger.debug("使用规则回退", exc_info=True)
            return self._rule_based_rerank(query, candidates, topk)

    def _compute_scores_with_retry(self, query: str, candidates: List[Dict]) -> List[float]:
        """带重试机制的分数计算"""
        last_exception = None
        current_delay = self.retry_delay

        for attempt in range(self.max_retries):
            try:
                return self._compute_scores(query, candidates)
            except (RuntimeError, OSError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    logger.warning("分数计算失败 (尝试 {}/{}): {}".format(
                        attempt + 1, self.max_retries, e
                    ))
                    logger.info("等待 {:.1f} 秒后重试...".format(current_delay))
                    time.sleep(current_delay)
                    current_delay *= self.retry_backoff
                else:
                    logger.error("分数计算在 {} 次尝试后仍然失败".format(
                        self.max_retries
                    ))

        # 如果所有重试都失败，返回默认分数
        logger.warning("所有重试失败，返回默认分数")
        return [0.0] * len(candidates)

    def _compute_scores(self, query: str, candidates: List[Dict], batch_size: int = 8) -> List[float]:
        """计算query-candidate对的分数（支持批量推理）"""
        if self.model is None:
            return [0.0] * len(candidates)

        scores = []
        num_batches = (len(candidates) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(candidates))
            batch_candidates = candidates[start_idx:end_idx]

            # 构建批量输入
            prompts = []
            for candidate in batch_candidates:
                name = self._safe_text(candidate.get("name", ""))
                desc = self._safe_text(candidate.get("description", ""))[:200]
                city = self._safe_text(candidate.get("city", ""))
                province = self._safe_text(candidate.get("province", ""))
                prompt = "查询：{}\n景点：{}（{}，{}）\n描述：{}".format(
                    query, name, city, province, desc
                )
                prompts.append(prompt)

            # Tokenize（批量）
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            # 计算分数
            with torch.no_grad():
                outputs = self.model(**inputs)
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                    if logits.dim() > 1:
                        batch_scores = logits[:, 0].cpu().tolist()
                    else:
                        batch_scores = [logits[0].item()]
                else:
                    batch_scores = [0.0] * len(batch_candidates)

            scores.extend(batch_scores)

        return scores

    def _rule_based_rerank(self, query: str, candidates: List[Dict], topk: int) -> List[Dict]:
        """规则回退方案"""
        query_lower = self._safe_text(query).lower()

        for candidate in candidates:
            score = 0.0
            name = self._safe_text(candidate.get("name", "")).lower()
            desc = self._safe_text(candidate.get("description", "")).lower()
            city = self._safe_text(candidate.get("city", "")).lower()
            province = self._safe_text(candidate.get("province", "")).lower()

            # 关键词匹配
            for keyword in query_lower.split():
                if keyword in name:
                    score += 3.0
                if keyword in desc:
                    score += 1.0
                if keyword in city or keyword in province:
                    score += 2.0

            # 保留原有分数
            original_score = candidate.get("reranker_score", candidate.get("score", 0))
            candidate["reranker_score"] = score + original_score * 0.5

        return sorted(candidates, key=lambda x: x.get("reranker_score", 0), reverse=True)[:topk]

    def compute_pairwise_score(self, query: str, doc: str) -> float:
        """计算query-doc对的分数（单个，带重试）"""
        if self.model is None:
            query_words = set(self._safe_text(query).lower().split())
            doc_words = set(self._safe_text(doc).lower().split())
            overlap = len(query_words & doc_words)
            return float(overlap) / max(len(query_words), 1)

        last_exception = None
        current_delay = self.retry_delay

        for attempt in range(self.max_retries):
            try:
                prompt = "查询：{}\n文档：{}".format(query, doc)

                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits

                    if logits.dim() > 1:
                        return logits[0, 0].item()
                    return logits[0].item()

            except (RuntimeError, OSError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    logger.warning("成对分数计算失败 (尝试 {}/{}): {}".format(
                        attempt + 1, self.max_retries, e
                    ))
                    time.sleep(current_delay)
                    current_delay *= self.retry_backoff
                else:
                    logger.error("成对分数计算在 {} 次尝试后仍然失败".format(
                        self.max_retries
                    ))
                    break

        # Fallback
        logger.warning("使用关键词匹配作为fallback")
        query_words = set(self._safe_text(query).lower().split())
        doc_words = set(self._safe_text(doc).lower().split())
        overlap = len(query_words & doc_words)
        return float(overlap) / max(len(query_words), 1)

    def compute_batch_pairwise_scores(self, query: str, docs: List[str], batch_size: int = 8) -> List[float]:
        """批量计算query-doc对的分数（带重试）"""
        if self.model is None:
            return [self.compute_pairwise_score(query, doc) for doc in docs]

        scores = []
        num_batches = (len(docs) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(docs))
            batch_docs = docs[start_idx:end_idx]

            # 构建批量输入
            prompts = ["查询：{}\n文档：{}".format(query, doc) for doc in batch_docs]

            try:
                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits

                    if logits.dim() > 1:
                        batch_scores = logits[:, 0].cpu().tolist()
                    else:
                        batch_scores = [logits[0].item()]

                    scores.extend(batch_scores)

            except Exception as e:
                logger.warning("批量计算失败，使用fallback: {}".format(e))
                for doc in batch_docs:
                    scores.append(self.compute_pairwise_score(query, doc))

        return scores

    def cleanup(self):
        """清理模型资源"""
        if self.model is not None:
            if self.use_gpu and hasattr(self.model, 'to'):
                try:
                    self.model.to('cpu')
                except Exception:
                    pass
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # 清理GPU缓存
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        import gc
        gc.collect()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("=" * 60)
    print("测试 Qwen3 Reranker")
    print("=" * 60)

    # 测试数据
    query = "想去新疆看雪山和草原"

    candidates = [
        {"name": "喀纳斯湖", "city": "阿勒泰", "province": "新疆", "description": "新疆著名的高山湖泊，雪山环绕"},
        {"name": "那拉提草原", "city": "伊犁", "province": "新疆", "description": "空中草原，风景优美"},
        {"name": "布达拉宫", "city": "拉萨", "province": "西藏", "description": "西藏标志性建筑"},
        {"name": "禾木村", "city": "阿勒泰", "province": "新疆", "description": "图瓦人村落，秋季景色迷人"},
    ]

    reranker = QwenReranker(use_gpu=False)
    print("Reranker可用: {}".format(reranker.is_available()))

    ranked = reranker.rerank(query, candidates, topk=4)

    print("\n重排序结果:")
    for i, item in enumerate(ranked, 1):
        score = item.get("reranker_score", 0)
        print("{}. {} - {} - 分数: {:.2f}".format(i, item['name'], item['province'], score))
