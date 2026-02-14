"""
Qwen3 Reranker
使用Qwen3-Reranker-4B对Top-K候选进行重排序

参考: models/download.md
from transformers import AutoModelForSequenceClassification, AutoTokenizer
pipe = pipeline("text-generation", model="Qwen/Qwen3-Reranker-4B")
"""
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

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


class QwenReranker:
    """
    Qwen3 Reranker
    使用Qwen3-Reranker-4B对候选POI进行精排

    用法:
        reranker = QwenReranker(model_path="models/Qwen3-Reranker-4B")
        ranked_pois = reranker.rerank(query, candidates, topk=20)
    """

    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = True):
        """
        初始化Reranker

        Args:
            model_path: 模型路径，默认为models/Qwen3-Reranker-4B
            use_gpu: 是否使用GPU
        """
        if torch is None:
            print("torch未安装，Reranker将使用规则回退")
            self.model = None
            self.tokenizer = None
            self.use_gpu = False
            self.device = "cpu"
            return

        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"

        if AutoModelForSequenceClassification is None:
            print("transformers未安装，Reranker将使用规则回退")
            self.model = None
            self.tokenizer = None
            return

        # 解析模型路径
        if model_path is None:
            model_path = os.getenv("GOAFAR_RERANKER_MODEL", str(PROJECT_ROOT / "models/Qwen3-Reranker-4B"))

        if not Path(model_path).exists():
            print(f"模型路径不存在: {model_path}")
            print("Reranker将使用规则回退")
            self.model = None
            self.tokenizer = None
            return

        try:
            print(f"加载Qwen3-Reranker-4B: {model_path}")
            print(f"设备: {self.device}")

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
            print("✓ Reranker模型加载完成")

        except Exception as e:
            print(f"Reranker模型加载失败: {e}")
            print("将使用规则回退")
            self.model = None
            self.tokenizer = None

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        topk: int = 20,
        score_key: str = "score"
    ) -> List[Dict]:
        """
        对候选POI进行重排序

        Args:
            query: 用户查询
            candidates: 候选POI列表，每个元素为字典，包含name, description等字段
            topk: 返回Top-K
            score_key: 原始分数字段名

        Returns:
            重排序后的候选列表
        """
        if self.model is None or len(candidates) <= topk:
            return self._rule_based_rerank(query, candidates, topk)

        # 批量计算分数
        try:
            scores = self._compute_scores(query, candidates)

            # 更新分数并排序
            for i, candidate in enumerate(candidates):
                if i < len(scores):
                    candidate["reranker_score"] = float(scores[i])

            ranked = sorted(candidates, key=lambda x: x.get("reranker_score", 0), reverse=True)
            return ranked[:topk]

        except Exception as e:
            print(f"Reranker计算失败: {e}")
            return self._rule_based_rerank(query, candidates, topk)

    def _compute_scores(self, query: str, candidates: List[Dict], batch_size: int = 8) -> List[float]:
        """
        计算query-candidate对的分数（支持批量推理）

        Args:
            query: 用户查询
            candidates: 候选POI列表
            batch_size: 批量大小

        Returns:
            相关性分数列表
        """
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
                name = candidate.get("name", "")
                desc = candidate.get("description", "")[:200]
                city = candidate.get("city", "")
                province = candidate.get("province", "")
                prompt = f"查询：{query}\n景点：{name}（{city}，{province}）\n描述：{desc}"
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
                        # 取第一个logit作为相关性分数
                        batch_scores = logits[:, 0].cpu().tolist()
                    else:
                        batch_scores = [logits[0].item()]
                else:
                    batch_scores = [0.0] * len(batch_candidates)

            scores.extend(batch_scores)

        return scores

    def _rule_based_rerank(self, query: str, candidates: List[Dict], topk: int) -> List[Dict]:
        """规则回退方案"""
        # 简单关键词匹配
        query_lower = query.lower()

        for candidate in candidates:
            score = 0.0
            name = candidate.get("name", "").lower()
            desc = candidate.get("description", "").lower()
            city = candidate.get("city", "").lower()
            province = candidate.get("province", "").lower()

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
        """
        计算query-doc对的分数（单个）

        Args:
            query: 查询文本
            doc: 文档文本

        Returns:
            相关性分数
        """
        if self.model is None:
            # 简单关键词匹配
            query_words = set(query.lower().split())
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            return float(overlap) / max(len(query_words), 1)

        try:
            prompt = f"查询：{query}\n文��：{doc}"

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

        except Exception as e:
            print(f"分数计算失败: {e}")
            return 0.0

    def compute_batch_pairwise_scores(self, query: str, docs: List[str], batch_size: int = 8) -> List[float]:
        """
        批量计算query-doc对的分数

        Args:
            query: 查询文本
            docs: 文档文本列表
            batch_size: 批量大小

        Returns:
            相关性分数列表
        """
        if self.model is None:
            return [self.compute_pairwise_score(query, doc) for doc in docs]

        scores = []
        num_batches = (len(docs) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(docs))
            batch_docs = docs[start_idx:end_idx]

            # 构建批量输入
            prompts = [f"查询：{query}\n文档：{doc}" for doc in batch_docs]

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

        return scores


if __name__ == "__main__":
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
    ranked = reranker.rerank(query, candidates, topk=4)

    print("\n重排序结果:")
    for i, item in enumerate(ranked, 1):
        score = item.get("reranker_score", 0)
        print(f"{i}. {item['name']} - {item['province']} - 分数: {score:.2f}")
