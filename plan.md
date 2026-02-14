# GoAfar 项目改造计划

## 0. 环境准备（优先执行）

### 0.1 硬件信息
- **GPU**: RTX 5090 (24GB VRAM)
- **CUDA要求**: 12.8+
- **PyTorch要求**: 2.9+

### 0.2 环境清理与重装
```bash
# 1. 备份当前环境（可选）
conda pack -n goafar -o goafar_backup.tar.gz

# 2. 删除并重建环境
conda deactivate
conda env remove -n goafar -y
conda create -n goafar python=3.12 -y
conda activate goafar

# 3. 安装PyTorch 2.9 + CUDA 12.8
pip install torch==2.9.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. 安装核心依赖
pip install transformers accelerate bitsandbytes peft
pip install trl datasets
pip install fastapi uvicorn pydantic
pip install pandas numpy geopandas fiona
pip install FlagEmbedding faiss-cpu
pip install recbole
pip install ortools

# 5. 安装项目依赖
pip install -r requirements.txt
```

### 0.3 后台模型下载
创建`scripts/download_models_bg.sh`：
```bash
#!/bin/bash
# 后台下载Qwen3系列模型
export HF_ENDPOINT=https://hf-mirror.com
MODELS_DIR="./models"
mkdir -p $MODELS_DIR

# 使用nohup后台下载
nohup huggingface-cli download Qwen/Qwen3-8B \
    --local-dir $MODELS_DIR/Qwen3-8B \
    --local-dir-use-symlinks False > logs/download_qwen3_8b.log 2>&1 &

nohup huggingface-cli download Qwen/Qwen3-Embedding-4B \
    --local-dir $MODELS_DIR/Qwen3-Embedding-4B \
    --local-dir-use-symlinks False > logs/download_embedding.log 2>&1 &

nohup huggingface-cli download Qwen/Qwen3-Reranker-4B \
    --local-dir $MODELS_DIR/Qwen3-Reranker-4B \
    --local-dir-use-symlinks False > logs/download_reranker.log 2>&1 &

echo "模型下载已在后台启动，使用 tail -f logs/*.log 查看进度"
```

## 1. 项目现状梳理

### 1.1 项目定位
GoAfar 是一个智能旅游路线推荐系统，集成了语义检索（BGE-M3）、序列推荐（RecBole）、路线规划（OR-Tools VRPTW）和LLM增强（Qwen3-8B）。

### 1.2 当前数据覆盖
- **现有POI数据**：1333个景点，覆盖8个省份（新疆、西藏、云南、四川、甘肃、青海、宁夏、内蒙古）
- **用户行为数据**：38579条用户事件（user_events.csv）

### 1.3 新下载数据（data/external/）
- **shengfen/**：22个省份的shp地理数据文件（需整合）
- **Geolife Trajectories 1.3.zip**：微软GPS轨迹数据，可用于训练/验证规划模型
- **Yelp-JSON.zip + Yelp-Photos.zip**：Yelp数据集，可补充POI属性和偏好信号
- **loc-gowalla_*.txt.gz**：Gowalla签到数据，可补充行为序列
- **homberger_1000_customer_instances.zip**：Solomon VRPTW基准测试数据

### 1.4 当前模型规划
- **主LLM模型**：Qwen3-8B（用于意图理解、重排序、文案生成）
- **Embedding模型**：BGE-M3（语义召回）
- **Reranking模型**：Qwen3-8B-Embedding（待下载）
- **规划模型**：OR-Tools VRPTW + 未来veRL GRPO

## 2. 代码审查结果（✅全部通过）

### 2.1 新增服务化模块 ✅
- `src/service/pipeline.py` - 统一推荐流水线（语法通过）
- `src/service/config.py` - 运行时配置管理
- `src/schemas/recommendation.py` - Pydantic/dataclass兼容的契约定义
- `src/api/server.py` - FastAPI统一入口

### 2.2 召回升级 ✅
- `src/recommendation/candidate_merger.py` - 多路召回（语义/行为/地理）+ RRF融合（语法通过）
- `src/utils/id_mapping.py` - ID归一化（解决poi_id前导零问题）

### 2.3 规划层升级 ✅
- `src/routing/time_matrix_builder.py` - OSRM优先，Haversine回退（语法通过）

### 2.4 LLM模块 ✅
- `src/llm4rec/qwen_recommender.py` - Qwen推荐器（语法通过）
- `src/content_generation/llm_generator.py` - LLM文案生成器（语法通过）
- `src/embedding/bge_m3_encoder.py` - BGE-M3编码器（语法通过）

### 2.5 RL/GRPO骨架 ✅
- `src/rl/dataset_builder.py` - GRPO/SFT样本构建
- `src/rl/reward_manager.py` - 奖励管理器

**审查结论**：所有核心文件语法检查通过，无需紧急修复。

### 2.1 新增服务化模块 ✅
- `src/service/pipeline.py` - 统一推荐流水线
- `src/service/config.py` - 运行时配置管理
- `src/schemas/recommendation.py` - Pydantic/dataclass兼容的契约定义
- `src/api/server.py` - FastAPI统一入口

### 2.2 召回升级 ✅
- `src/recommendation/candidate_merger.py` - 多路召回（语义/行为/地理）+ RRF融合
- `src/utils/id_mapping.py` - ID归一化（解决poi_id前导零问题）

### 2.3 规划层升级 ✅
- `src/routing/time_matrix_builder.py` - OSRM优先，Haversine回退

### 2.4 RL/GRPO骨架 ✅
- `src/rl/dataset_builder.py` - GRPO/SFT样本构建
- `src/rl/reward_manager.py` - 奖励管理器

## 3. 待下载模型清单

基于你的计划（Qwen3-8B作为主模型），建议补充下载以下模型：

| 模型类型 | 推荐模型 | 用途 | 下载链接 |
|---------|---------|------|---------|
| **LLM主模型** | Qwen/Qwen2.5-7B-Instruct 或 Qwen/Qwen3-8B | 意图理解、重排序、文案生成 | HuggingFace |
| **Embedding** | BAAI/bge-m3-v2 或 Qwen/Qwen3-8B-Embedding | 语义编码/重排序 | HuggingFace |
| **Reranker** | BAAI/bge-reranker-v2-m3 | Cross-encoder重排 | HuggingFace |
| **可选：CLIP** | openai/clip-vit-base-patch32 | 多模态图像理解 | HuggingFace |

**注意**：models目录目前为空，需要下载模型文件。

## 4. 改造任务清单

### Phase 1: 数据整合与扩展
1. **处理shengfen省份shp数据**
   - 解析shp文件，提取省份边界、城市信息
   - 将新省份的POI数据补充到现有poi.csv
   - 目标省份优先级：福建、江苏、浙江、广东等热门旅游省份

2. **整合外部数据集**
   - 解析Geolife轨迹数据，学习真实移动模式
   - 整合Yelp数据，补充POI属性和评价
   - 处理Gowalla签到数据，增强行为召回

### Phase 2: 模型配置与集成
1. **配置Qwen3-8B模型路径**
   - 确认模型存放位置
   - 更新`configs/runtime.yaml`中的模型路径配置
   - 测试模型加载

2. **集成Embedding和Reranker模型**
   - 更新`src/embedding/bge_m3_encoder.py`以支持新的embedding模型
   - 添加reranker模块用于精排后重排

### Phase 3: 代码优化与修复
1. **修复candidate_merger.py语法问题**
   - 第15行：多余的右括号 `sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`
   - 修复为：`sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`

2. **完善配置文件**
   - `configs/runtime.yaml`中LLM默认是disabled，需要根据模型可用性调整
   - 添加新模型的配置项

### Phase 4: 测试与验证
1. **运行全链路测试**
   - 验证pipeline.py在配置Qwen3-8B后能否正常工作
   - 测试语义召回、重排序、文案生成功能

2. **数据飞轮测试**
   - 运行`src/data_processing/synthesize_training_data.py`
   - 运行`src/rl/dataset_builder.py`

## 5. 关键文件路径

### 需要修改的文件
- `configs/runtime.yaml` - 更新模型路径和LLM开关
- `src/recommendation/candidate_merger.py` - 修复语法问题
- `src/embedding/bge_m3_encoder.py` - 可能需要更新以支持新embedding模型

### 需要新建的模块
- `src/data_processing/shp_parser.py` - 解析shp省份数据
- `src/reranking/qwen_reranker.py` - Qwen3-8B重排器

## 6. 执行顺序建议

1. **首先**：下载并配置Qwen3-8B模型
2. **其次**：修复candidate_merger.py的语法问题
3. **然后**：整合shengfen省份数据到POI数据库
4. **最后**：测试全链路并优化

## 7. 用户确认的选择 ✅

1. **模型下载**：使用ModelScope（阿里云镜像）下载到models目录
2. **shengfen数据**：既要地理边界过滤，也要补充POI数据
3. **数据结构**：可以重新设计，有更好的结构建议可以调整

## 8. 需要下载的模型（使用HF Mirror）

### 8.1 模型列表与用途
| 模型 | 用途 | 文件大小 | 优先级 |
|------|------|----------|--------|
| Qwen/Qwen3-8B | 意图理解、重排序、文案生成 | ~16GB | P0 |
| Qwen/Qwen3-Embedding-4B | POI语义编码、向量检索 | ~8GB | P1 |
| Qwen/Qwen3-Reranker-4B | 精排后重排（Cross-Encoder） | ~8GB | P2 |

### 8.2 下载命令（HF Mirror）
详见 0.3 节后台下载脚本，单独下载命令：
```bash
export HF_ENDPOINT=https://hf-mirror.com

# 下载Qwen3-8B
huggingface-cli download Qwen/Qwen3-8B --local-dir ./models/Qwen3-8B --local-dir-use-symlinks False

# 下载Qwen3-Embedding-4B
huggingface-cli download Qwen/Qwen3-Embedding-4B --local-dir ./models/Qwen3-Embedding-4B --local-dir-use-symlinks False

# 下载Qwen3-Reranker-4B
huggingface-cli download Qwen/Qwen3-Reranker-4B --local-dir ./models/Qwen3-Reranker-4B --local-dir-use-symlinks False
```

## 9. 详细实施步骤

### Step 1: 模型下载（后台任务）
创建后台下载脚本，使用ModelScope镜像下载模型到models目录

### Step 2: SHP数据处理
创建新模块`src/data_processing/shp_parser.py`：
- 解析shp文件提取省份边界和城市信息
- 从shp属性中提取POI数据（景点名称、位置、类型等）
- 将新数据整合到poi.csv

### Step 3: 数据结构重新设计
优化poi.csv结构：
- 添加更细粒度的category字段
- 添加tags字段用于多标签分类
- 添加source字段标识数据来源（原始/新增/shp/Yelp）

### Step 4: 配置文件更新
更新`configs/runtime.yaml`：
- 添加新模型路径配置（qwen_model_path, qwen_embedding_path）
- 启用LLM功能（llm.enabled: true）
- 添加SHP数据路径配置

### Step 5: 模型微调（SFT + DPO）

#### 5.1 SFT训练（新增）
基于Qwen3-8B进行监督微调，��建`src/content_generation/train_sft.py`：

```python
"""
SFT训练脚本
使用TRL的SFTTrainer对Qwen3-8B进行监督微调

任务目标：
1. 意图理解：从用户查询中提取结构化信息
2. 文案生成：生成高质量旅游路线标题和描述
3. POI推荐：根据用户意图推荐合适的POI列表
"""
```

SFT数据格式：
```json
{
  "prompt": "用户想去新疆看雪山和草原，计划3天",
  "response": "{\"province\": \"新疆\", \"interests\": [\"雪山\", \"草原\"], \"duration_days\": 3, ...}"
}
```

运行SFT训练：
```bash
# 合成SFT数据
python src/data_processing/synthesize_training_data.py --output-dir outputs/datasets --synth-users 300

# 运行SFT训练
python src/content_generation/train_sft.py \
    --model models/Qwen3-8B \
    --data outputs/datasets/sft_data.jsonl \
    --output outputs/sft/qwen3-8b-tourism \
    --use-lora \
    --lora-r 16 \
    --epochs 3
```

#### 5.2 DPO训练（已有）
使用现有的`src/content_generation/train_dpo.py`：

```bash
# 首先构造偏好数据
python src/content_generation/make_prefs.py

# 运行DPO训练
python src/content_generation/train_dpo.py \
    --model models/Qwen3-8B \
    --prefs outputs/dpo/prefs.csv \
    --output outputs/dpo/qwen3-8b-dpo \
    --use-lora \
    --epochs 1
```

#### 5.3 微调模型部署
训练完成后，更新配置使用微调后的模型：

```yaml
# configs/runtime.yaml
llm:
  enabled: true
  qwen_model: outputs/sft/qwen3-8b-tourism  # SFT模型
  qwen_dpo_model: outputs/dpo/qwen3-8b-dpo   # DPO模型（可选）
  use_lora: true  # 使用LoRA适配器
```

### Step 6: 代码优化

#### 6.1 更新模型加载逻辑
- `src/llm4rec/qwen_recommender.py`：支持Qwen3-8B和微调后的模型路径
- `src/embedding/bge_m3_encoder.py`：支持Qwen3-Embedding-4B作为备选
- 添加LoRA适配器加载支持

#### 6.2 新增Reranker模块
- `src/reranking/qwen_reranker.py`：使用Qwen3-Reranker-4B进行精排后重排
```python
"""
Qwen3 Reranker
使用Qwen3-Reranker-4B对Top-K候选进行重排序
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class QwenReranker:
    def __init__(self, model_path="models/Qwen3-Reranker-4B"):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def rerank(self, query, candidates, topk=20):
        # 实现重排序逻辑
        pass
```

#### 6.3 新增SFT训练脚本
- `src/content_generation/train_sft.py`：基于TRL SFTTrainer实现

### Step 7: 测试与验证

#### 7.1 单元测试
```bash
# 测试模型加载
python -c "from src.llm4rec.qwen_recommender import QwenRecommender; QwenRecommender()"

# 测试配置加载
python -c "from src.service.config import load_runtime_config; print(load_runtime_config())"

# 测试Embedding模型
python -c "from src.embedding.bge_m3_encoder import BGEM3Encoder; BGEM3Encoder()"

# 测试Reranker
python -c "from src.reranking.qwen_reranker import QwenReranker; QwenReranker()"
```

#### 7.2 集成测试
```bash
# 全链路测试
python test_full_pipeline.py

# API服务测试
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/healthz
curl -X POST http://localhost:8000/v1/recommend/itinerary \
  -H "Content-Type: application/json" \
  -d '{"query_text": "想去新疆看雪山", "province": "新疆", "use_llm": true}'
```

#### 7.3 模型微调测试
```bash
# SFT训练测试（小数据集）
python src/content_generation/train_sft.py \
    --model models/Qwen3-8B \
    --data outputs/datasets/sft_sample.jsonl \
    --output outputs/sft/test \
    --epochs 1 \
    --batch-size 2

# DPO训练测试
python src/content_generation/train_dpo.py \
    --model models/Qwen3-8B \
    --prefs outputs/dpo/prefs.csv \
    --output outputs/dpo/test \
    --epochs 1
```

#### 7.4 数据飞轮测试
```bash
# 合成训练数据
python src/data_processing/synthesize_training_data.py --output-dir outputs/datasets

# 构建RL样本
python src/rl/dataset_builder.py --max-samples 1000
```

## 10. 验证清单

完成改造后，验证以下功能：

- [ ] Qwen3-8B模型可以正常加载
- [ ] Qwen3-Embedding-4B编码正常工作
- [ ] Qwen3-Reranker-4B重排序正常工作
- [ ] SFT训练可以正常运行
- [ ] DPO训练可以正常运行
- [ ] SHP数据可以解析，新省份POI可以查询
- [ ] API服务可以启动并响应请求
- [ ] 全链路测试通过
- [ ] 配置文件可以动态切换模型和功能开关

## 11. 关键文件清单

### 需要修改的文件
- `configs/runtime.yaml` - 更新模型路径（Qwen3系列）和LLM开关
- `src/llm4rec/qwen_recommender.py` - 支持Qwen3-8B和LoRA加载
- `src/embedding/bge_m3_encoder.py` - 支持Qwen3-Embedding-4B

### 需要新建的文件
- `scripts/download_qwen_models.sh` - 模型下载脚本（HF Mirror）
- `src/data_processing/shp_parser.py` - SHP数据解析器
- `src/reranking/qwen_reranker.py` - Qwen3-Reranker-4B重排器
- `src/content_generation/train_sft.py` - SFT训练脚本

## 12. 总结

### 项目现状
- GoAfar 是一个功能完整的旅游推荐系统，已实现多路召回、RRF融合、VRPTW规划
- 代码质量良好，核心模块语法检查全部通过
- 已有的服务化架构（FastAPI + Pipeline）为后续扩展奠定了基础
- 已有DPO训练脚本，需要补充SFT训练

### 改造重点
1. **模型升级**：使用Qwen3系列（8B + Embedding-4B + Reranker-4B），通过HF Mirror下载
2. **SFT训练**：新增监督微调流程，让模型学会旅游推荐任务
3. **DPO训练**：使用现有脚本进行偏好对齐
4. **数据扩展**：整合SHP省份地图、Yelp、Gowalla等外部数据源

### 预期成果
- 支持全国22+省份的旅游推荐
- 基于Qwen3的智能意图理解和内容生成
- 完整的数据飞轮闭环（数据→SFT→DPO→评估→上线）
