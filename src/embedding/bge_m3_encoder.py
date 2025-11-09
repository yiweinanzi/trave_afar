"""
BGE-M3 语义编码器
参考: FlagEmbedding/examples/inference/embedder/encoder_only/auto_m3_single_device.py
"""
import os
import numpy as np
from FlagEmbedding import FlagAutoModel

class BGEM3Encoder:
    """BGE-M3 编码器封装类"""
    
    def __init__(self, model_path=None, use_gpu=True, cache_dir=None):
        """
        初始化编码器
        
        Args:
            model_path: 本地模型路径（如果为None，从HuggingFace下载）
            use_gpu: 是否使用GPU
            cache_dir: 模型缓存目录
        """
        self.model_path = model_path or "BAAI/bge-m3"
        self.device = "cuda:0" if use_gpu and self._check_gpu() else "cpu"
        self.cache_dir = cache_dir
        
        print(f"初始化 BGE-M3 编码器...")
        print(f"  模型: {self.model_path}")
        print(f"  设备: {self.device}")
        
        self.model = FlagAutoModel.from_finetuned(
            self.model_path,
            devices=self.device,
            cache_dir=self.cache_dir
        )
        print("✓ 模型加载完成")
    
    def _check_gpu(self):
        """检查GPU是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def encode_texts(self, texts, batch_size=64, max_length=512, 
                     return_dense=True, return_sparse=False, return_colbert=False):
        """
        编码文本列表
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            max_length: 最大文本长度
            return_dense: 是否返回dense向量
            return_sparse: 是否返回sparse向量（BM25风格）
            return_colbert: 是否返回colbert向量
        
        Returns:
            字典，包含 dense_vecs, sparse_weights, colbert_vecs
        """
        print(f"编码 {len(texts)} 个文本...")
        
        embeddings = self.model.encode_corpus(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert
        )
        
        print("✓ 编码完成")
        return embeddings
    
    def encode_query(self, query, return_dense=True, return_sparse=False, return_colbert=False):
        """
        编码单个查询
        
        Args:
            query: 查询文本
            return_dense: 是否返回dense向量
            return_sparse: 是否返回sparse向量
            return_colbert: 是否返回colbert向量
        
        Returns:
            字典，包含 dense_vecs, sparse_weights, colbert_vecs
        """
        embeddings = self.model.encode_queries(
            [query],
            batch_size=1,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert
        )
        
        # 提取第一个结果
        result = {}
        if return_dense and 'dense_vecs' in embeddings:
            result['dense_vec'] = embeddings['dense_vecs'][0]
        if return_sparse and 'lexical_weights' in embeddings:
            result['lexical_weights'] = embeddings['lexical_weights'][0]
        if return_colbert and 'colbert_vecs' in embeddings:
            result['colbert_vecs'] = embeddings['colbert_vecs'][0]
        
        return result
    
    def compute_similarity(self, query_embedding, corpus_embeddings, method='dense'):
        """
        计算相似度
        
        Args:
            query_embedding: 查询向量
            corpus_embeddings: 语料库向量
            method: 'dense', 'sparse', 或 'colbert'
        
        Returns:
            相似度分数数组
        """
        if method == 'dense':
            # Dense向量：余弦相似度（向量已归一化）
            query_vec = query_embedding['dense_vec']
            corpus_vecs = corpus_embeddings['dense_vecs']
            scores = corpus_vecs @ query_vec
            return scores
        
        elif method == 'sparse':
            # Sparse向量：词法匹配分数
            query_weights = query_embedding['lexical_weights']
            corpus_weights = corpus_embeddings['lexical_weights']
            scores = self.model.compute_lexical_matching_score(
                [query_weights],
                corpus_weights
            )
            return scores[0]
        
        elif method == 'colbert':
            # ColBERT: 多向量交互
            # query_vecs: (seq_len, dim)
            # corpus_vecs: (n_docs, seq_len, dim)
            query_vecs = query_embedding['colbert_vecs']
            corpus_vecs = corpus_embeddings['colbert_vecs']
            
            # ColBERT相似度：max pooling over query tokens
            # score = max_i sum_j max(query_vecs[i] @ corpus_vecs[j])
            scores = []
            for doc_vecs in corpus_vecs:
                # 计算每个query token与doc的最大相似度
                max_sims = []
                for q_vec in query_vecs:
                    # q_vec: (dim,), doc_vecs: (seq_len, dim)
                    sims = q_vec @ doc_vecs.T  # (seq_len,)
                    max_sims.append(sims.max().item())
                # 对所有query token的最大相似度求和
                scores.append(sum(max_sims))
            
            return np.array(scores)
        
        else:
            raise ValueError(f"不支持的方法: {method}")

if __name__ == "__main__":
    # 测试
    encoder = BGEM3Encoder(
        model_path="/root/autodl-tmp/goafar_project/models/Xorbits/bge-m3",
        use_gpu=False
    )
    
    # 测试编码
    texts = ["测试文本1", "测试文本2"]
    embeddings = encoder.encode_texts(texts)
    print(f"Dense向量维度: {embeddings['dense_vecs'].shape}")

