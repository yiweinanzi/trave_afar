# Qwen3-Reranker-4B 集成文档

## 概述

本文档描述了 Qwen3-Reranker-4B 模型在 GoAfar 推荐系统中的集成。该模型用于在 LLM 初步重排后对候选 POI 进行精排，提高推荐质量。

## 架构设计

### 重排序流程

```
召回阶段 (Recall)
    ↓
候选融合 (Merge)
    ↓
LLM 初步重排 (Qwen3-8B)
    ↓
Qwen3-Reranker-4B 精排 ← 新增
    ↓
路线规划 (Routing)
```

### 两阶段重排策略

1. **LLM 初步重排** (Qwen3-8B)
   - 对召回的 Top-K 候选进行初步筛选
   - 考虑用户意图、兴趣、活动等因素
   - 将候选数量从 80+ 降至 30

2. **Reranker 精排** (Qwen3-Reranker-4B)
   - 使用 cross-encoder 模式计算 query-POI 相关性
   - 更精确的语义匹配
   - 从 30 个候选中精选出 Top 20

## 配置说明

### runtime.yaml 配置

```yaml
rerank:
  enabled: true                      # 是否启用重排序
  use_template: false                # 是否使用规则模板（LLM不可用时）
  use_reranker_model: true           # 是否使用 Qwen3-Reranker-4B
  qwen_reranker_path: models/Qwen3-Reranker-4B  # 模型路径
  rerank_topk: 20                    # Reranker 精排后的 Top-K 数量
  topk: 30                           # LLM 初步重排的 Top-K 数量
```

### 配置参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | `true` | 是否启用重排序功能 |
| `use_template` | bool | `false` | LLM 不可用时是否使用规则模板 |
| `use_reranker_model` | bool | `true` | 是否使用 Qwen3-Reranker-4B 模型 |
| `qwen_reranker_path` | str | `models/Qwen3-Reranker-4B` | Reranker 模型路径 |
| `rerank_topk` | int | `20` | Reranker 精排后返回的候选数量 |
| `topk` | int | `30` | LLM 初步重排后返回的候选数量 |

## 代码实现

### 1. QwenReranker 类

位置: `/root/autodl-tmp/goafar_project_broken/src/reranking/qwen_reranker.py`

主要方法:

```python
class QwenReranker:
    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = True)
    def rerank(self, query: str, candidates: List[Dict], topk: int = 20) -> List[Dict]
    def _compute_scores(self, query: str, candidates: List[Dict], batch_size: int = 8) -> List[float]
    def compute_batch_pairwise_scores(self, query: str, docs: List[str], batch_size: int = 8) -> List[float]
```

#### 特性

- **批量推理**: 支持批量计算 query-candidate 对的分数，提高效率
- **自动降级**: 模型不可用时自动回退到规则匹配
- **GPU 支持**: 自动检测并使用 GPU（如果可用）
- **容错处理**: 完善的异常处理和日志记录

### 2. Pipeline 集成

位置: `/root/autodl-tmp/goafar_project_broken/src/service/pipeline.py`

集成流程:

```python
# Step 3: rerank
if self.config.rerank.enabled:
    # 3.1: LLM 初步重排
    if request.use_llm and self.config.llm.enabled:
        # ... LLM 重排逻辑 ...

    # 3.2: Qwen3-Reranker-4B 精排
    if self.config.rerank.use_reranker_model and len(candidates) > 0:
        try:
            qwen_reranker = self._maybe_get_qwen_reranker()
            if qwen_reranker is not None and qwen_reranker.model is not None:
                # 使用 Reranker 模型进行精排
                reranked = qwen_reranker.rerank(
                    query=query,
                    candidates=candidate_list,
                    topk=rerank_topk
                )
                # ... 结果处理 ...
        except Exception as exc:
            # 失败时回退到 LLM 重排结果
            debug.fallback_events.append(f"qwen_reranker_failed:{exc}")
```

### 3. 配置类更新

位置: `/root/autodl-tmp/goafar_project_broken/src/service/config.py`

```python
@dataclass
class RerankConfig:
    enabled: bool = True
    use_template: bool = True
    use_reranker_model: bool = True          # 新增
    qwen_reranker_path: str = "models/Qwen3-Reranker-4B"  # 新增
    rerank_topk: int = 20                    # 新增
    topk: int = 30
```

## 使用方法

### 基本使用

```python
from reranking.qwen_reranker import QwenReranker

# 初始化 Reranker
reranker = QwenReranker(
    model_path="models/Qwen3-Reranker-4B",
    use_gpu=True
)

# 准备候选数据
query = "想去新疆看雪山和草原"
candidates = [
    {
        "poi_id": "POI_0001",
        "name": "喀纳斯湖",
        "city": "阿勒泰",
        "province": "新疆",
        "description": "新疆著名的高山湖泊，雪山环绕"
    },
    # ... 更多候选 ...
]

# 执行重排序
ranked = reranker.rerank(query, candidates, topk=20)

# 处理结果
for item in ranked:
    print(f"{item['name']}: {item.get('reranker_score', 0):.4f}")
```

### 批量分数计算

```python
# 计算多个文档的相关性分数
query = "新疆旅游"
docs = [
    "喀纳斯湖是新疆著名的高山湖泊",
    "那拉提草原是新疆最美的草原",
    "布达拉宫位于西藏拉萨",
]

scores = reranker.compute_batch_pairwise_scores(query, docs, batch_size=8)
```

## 降级机制

系统实现了多层降级机制，确保在模型不可用时仍能正常工作:

1. **模型加载失败**
   - 自动回退到规则匹配模式
   - 记录降级事件到 debug 信息

2. **Reranker 推理失败**
   - 保留 LLM 初步重排结果
   - 记录失败原因到 debug 信息

3. **LLM 不可用**
   - 使用规则模板进行重排
   - 不影响 Reranker 的后续精排

4. **完全降级**
   - 所有模型都不可用时，使用原始召回分数排序

## 性能优化

### 批量推理

QwenReranker 支持批量推理，默认批次大小为 8:

```python
def _compute_scores(self, query: str, candidates: List[Dict], batch_size: int = 8)
```

- 减少模型调用次数
- 提高 GPU 利用率
- 可通过 `batch_size` 参数调整

### 懒加载

Reranker 模型采用懒加载策略:

- 仅在首次使用时加载模型
- 通过 `_maybe_get_qwen_reranker()` 方法获取实例
- 避免不必要的资源占用

## 测试验证

运行集成测试:

```bash
python test_qwen_reranker_integration.py
```

测试覆盖:

1. ✓ 配置加载测试
2. ✓ QwenReranker 初始化测试
3. ✓ 重排序功能测试
4. ✓ 批量分数计算测试

## 模型下载

确保已下载 Qwen3-Reranker-4B 模型到指定路径:

```bash
# 使用 HF Mirror 下载
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-Reranker-4B --local-dir models/Qwen3-Reranker-4B
```

参考: `/root/autodl-tmp/goafar_project_broken/models/download.md`

## 依赖要求

```bash
pip install torch transformers
```

- `torch`: GPU 加速支持
- `transformers`: 模型加载和推理

## 故障排查

### 问题 1: 模型加载失败

**现象**: 日志显示 "Reranker模型加载失败"

**解决方案**:
1. 检查模型路径是否正确
2. 确认模型文件完整性
3. 查看错误日志了解具体原因

### 问题 2: CUDA OOM

**现象**: GPU 内存不足错误

**解决方案**:
1. 减小 `batch_size` 参数
2. 使用 CPU 模式: `use_gpu=False`
3. 减少候选数量

### 问题 3: 推理速度慢

**现象**: 重排序耗时过长

**解决方案**:
1. 启用 GPU 加速
2. 增大 `batch_size` 提高吞吐量
3. 减少候选数量

## 后续优化方向

1. **模型量化**: 使用 INT8/FP16 量化减少内存占用
2. **结果缓存**: 缓存常见 query 的重排结果
3. **异步推理**: 使用异步 API 提高并发能力
4. **模型蒸馏**: 训练更小更快的专用 reranker

## 相关文件

- `/root/autodl-tmp/goafar_project_broken/src/reranking/qwen_reranker.py` - Reranker 实现
- `/root/autodl-tmp/goafar_project_broken/src/service/pipeline.py` - Pipeline 集成
- `/root/autodl-tmp/goafar_project_broken/src/service/config.py` - 配置类
- `/root/autodl-tmp/goafar_project_broken/configs/runtime.yaml` - 运行时配置
- `/root/autodl-tmp/goafar_project_broken/test_qwen_reranker_integration.py` - 集成测试

## 版本历史

- **v1.0** (2025-02-15): 初始版本，集成 Qwen3-Reranker-4B
  - 实现批量推理
  - 添加降级机制
  - 完善错误处理
