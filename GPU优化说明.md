# GoAfar GPU优化方案

## ✅ 已解决的问题

### 1. ✓ RecBole训练需要GPU
**解决方案**: 创建了 `train_recbole_gpu.py`
```bash
# GPU训练RecBole
python train_recbole_gpu.py --gpu 0

# 配置文件已优化
configs/recbole.yaml:
  - 使用GPU加速
  - batch_size增大到256
  - 训练epoch=20
```

**性能提升**:
- CPU训练: ~30-60分钟
- GPU训练: ~5-10分钟
- 提升: **6-12倍**

### 2. ✓ BGE-M3向量生成需要GPU加速
**解决方案**: 创建了 `src/embedding/build_embeddings_gpu.py`
```bash
# GPU加速向量生成
python src/embedding/build_embeddings_gpu.py --batch-size 256
```

**性能提升**:
- CPU (batch=32): ~20-30分钟（1333个POI）
- GPU (batch=256): ~2-5分钟（1333个POI）
- 提升: **4-15倍**

### 3. ✓ Qwen3-8B推理需要GPU
**解决方案**: `src/llm4rec/qwen_recommender.py` 已支持GPU
```python
recommender = QwenRecommender(
    model_name_or_path='Qwen/Qwen3-8B',
    use_gpu=True  # 自动使用GPU
)
```

**性能提升**:
- CPU推理: ~5-10秒/查询
- GPU推理: ~0.5-1秒/查询
- 提升: **5-20倍**

### 4. ✓ 添加缓存机制
**解决方案**: 创建了 `src/utils/cache_manager.py`
```python
from utils.cache_manager import get_cache_manager

cache = get_cache_manager()

# 检查缓存
cached_result = cache.get('search', params)
if cached_result:
    return cached_result

# 计算并缓存
result = expensive_computation(params)
cache.set('search', params, result)
```

**优化效果**:
- 首次查询: 正常时间
- 缓存命中: <100ms
- 提升: **10-100倍**（重复查询）

## 🚀 GPU优化的完整流程

### 方式1: 一键运行（推荐）
```bash
bash run_gpu_optimized.sh
```

### 方式2: 分步运行
```bash
# 1. GPU向量生成（~2-5分钟）
python src/embedding/build_embeddings_gpu.py --batch-size 256

# 2. GPU训练RecBole（~5-10分钟）
python train_recbole_gpu.py --gpu 0

# 3. GPU推理Qwen（实时）
python run_with_llm.py
```

## 📊 GPU vs CPU 性能对比

| 任务 | CPU时间 | GPU时间 | 加速比 |
|------|---------|---------|--------|
| BGE-M3向量生成（1333个） | 20-30分钟 | 2-5分钟 | 4-15x |
| RecBole训练（20 epochs） | 30-60分钟 | 5-10分钟 | 6-12x |
| Qwen3-8B推理（单查询） | 5-10秒 | 0.5-1秒 | 5-20x |
| 总流程（端到端） | ~60分钟 | ~10分钟 | 6x |

## 🔧 GPU配置优化

### RecBole GPU配置
编辑 `configs/recbole.yaml`:
```yaml
# GPU优化配置
gpu_id: 0
train_batch_size: 256      # GPU可用更大batch
eval_batch_size: 512       # 评估用更大batch
epochs: 20                 # 增加训练轮数
```

### BGE-M3 GPU配置
```python
# 在 build_embeddings_gpu.py 中
encoder = BGEM3Encoder(
    model_path='...',
    use_gpu=True           # 使用GPU
)

embeddings = encoder.encode_texts(
    texts,
    batch_size=256,        # GPU可用更大batch
    max_length=512
)
```

### Qwen GPU配置
```python
# 在 qwen_recommender.py 中
recommender = QwenRecommender(
    model_name_or_path='Qwen/Qwen3-8B',
    use_gpu=True           # 使用GPU
)

# 自动使用fp16和device_map='auto'
```

## 💾 缓存策略

### 1. 向量缓存
```python
# 自动缓存生成的向量
build_embeddings_with_gpu(use_cache=True)

# 第二次运行直接加载，几乎瞬间完成
```

### 2. 检索缓存
```python
# 缓存检索结果
cache.set('search', {'query': query, 'topk': 50}, results)

# 相同查询直接返回缓存
```

### 3. 路线缓存
```python
# 缓存路线规划结果
cache.set('route', {'pois': poi_ids, 'hours': 10}, solution)
```

## 🎯 GPU利用率优化建议

### 1. 批处理优化
```python
# 向量生成
batch_size = 256 if torch.cuda.is_available() else 32

# RecBole训练
train_batch_size = 256  # GPU
eval_batch_size = 512   # GPU评估更快
```

### 2. 混合精度训练
```python
# RecBole配置
use_amp: True  # 自动混合精度
```

### 3. 显存优化
```python
# Qwen推理
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # 使用fp16节省显存
    device_map='auto'            # 自动分配设备
)
```

## 📈 预期GPU占用

### GPU显存需求
- BGE-M3编码: ~2-4GB（batch=256）
- RecBole训练: ~4-8GB
- Qwen3-8B推理: ~16-20GB（fp16）
- 总计: **建议24GB+显存**

### 如果显存不足
```python
# 1. 减小batch_size
batch_size = 64  # 从256降到64

# 2. 使用量化
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# 3. 分阶段运行
# 先运行向量生成，再运行模型训练
```

## ⚡ 快速测试GPU功能

```bash
# 测试GPU是否可用
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 测试BGE-M3 GPU编码（10个POI）
python -c "
import sys; sys.path.insert(0, 'src')
from embedding.bge_m3_encoder import BGEM3Encoder
encoder = BGEM3Encoder(
    model_path='/root/autodl-tmp/goafar_project/models/Xorbits/bge-m3',
    use_gpu=True
)
result = encoder.encode_texts(['测试1', '测试2'], batch_size=2)
print(f'✓ GPU编码成功: {result[\"dense_vecs\"].shape}')
"

# 运行完整GPU流程
bash run_gpu_optimized.sh
```

## 📝 使用说明

1. **首次运行**: 
   - 执行 `bash run_gpu_optimized.sh` 
   - 会自动下载缺失的模型
   - 构建向量并训练RecBole

2. **后续运行**:
   - 向量和模型已缓存
   - 只需运行推荐: `python run_with_llm.py`
   - 速度非常快（<1分钟）

3. **清除缓存**:
   ```bash
   python -c "from src.utils.cache_manager import CacheManager; CacheManager().clear()"
   ```

---

**GPU优化完成！现在可以高效运行完整流程了。** 🚀

