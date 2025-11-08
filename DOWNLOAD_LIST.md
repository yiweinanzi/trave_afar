# 需要下载的开源项目清单

## ✅ 已下载（在 open_resource/ 目录）
- ✓ FlagEmbedding-master (BGE-M3)
- ✓ RecBole-master (序列推荐)
- ✓ or-tools-stable (路线规划)
- ✓ osmnx-examples-main (路网分析)
- ✓ trl-main (DPO训练)

## 🔥 急需下载（LLM4Rec集成）

### 1. LLM4Rec系列（推荐领域的LLM应用）⭐⭐⭐

#### a) LLM4Rec (综合框架)
```bash
cd /root/autodl-tmp/goafar_project/open_resource
git clone https://github.com/WLiK/LLM4Rec.git
```
**用途**: 
- 学习如何将LLM集成到推荐系统
- Prompt工程最佳实践
- 评测方法和指标

**重点查看**:
- `LLM4Rec/prompts/` - 推荐任务的提示词
- `LLM4Rec/models/` - LLM推荐模型
- `LLM4Rec/evaluation/` - 评测代码

#### b) TALLRec (三元组增强LLM推荐)
```bash
git clone https://github.com/SAI990323/TALLRec.git
```
**用途**:
- POI属性理解和结构化
- Few-shot推荐学习
- 属性引导的推荐

**重点查看**:
- `TALLRec/src/model/` - 模型实现
- `TALLRec/prompts/` - 提示词模板

#### c) RecLLM (对话式推荐)
```bash
git clone https://github.com/HKUDS/RecLLM.git
```
**用途**:
- 对话式推荐系统
- 意图理解和澄清
- 推荐解释生成

### 2. Qwen模型代码和文档 ⭐⭐⭐

```bash
git clone https://github.com/QwenLM/Qwen2.5.git
```
**用途**:
- Qwen模型的正确使用方法
- 中文推理和生成
- 微调和部署示例

**重点查看**:
- `Qwen2.5/README.md` - 模型使用说明
- `Qwen2.5/examples/` - 示例代码
- `Qwen2.5/docs/` - 文档

### 3. LangChain (LLM应用框架) ⭐⭐

```bash
git clone https://github.com/langchain-ai/langchain.git
```
**用途**:
- 统一的LLM接口
- Prompt管理
- RAG应用开发

**重点查看**:
- `langchain/libs/langchain/langchain/prompts/` - Prompt模板
- `langchain/docs/docs/tutorials/` - 教程

### 4. LlamaIndex (RAG框架) ⭐

```bash
git clone https://github.com/run-llama/llama_index.git
```
**用途**:
- 文档索引和检索
- POI知识库构建
- 向量数据库集成

## 📦 模型下载（需要的预训练模型）

### 推荐下载的模型（按优先级）

#### 1. Qwen2.5-0.5B-Instruct (轻量级，推荐) 
```bash
# 使用ModelScope下载（国内快）
cd /root/autodl-tmp/goafar_project
python -c "
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen2.5-0.5B-Instruct', 
                              cache_dir='models')
print(f'模型下载完成: {model_dir}')
"
```
**大小**: ~1GB
**用途**: 意图理解、文案生成、轻量级推荐

#### 2. Qwen2.5-1.5B-Instruct (中等，可选)
**大小**: ~3GB  
**用途**: 更好的理解和生成能力

#### 3. Qwen2-VL (多模态，高级)
**大小**: ~10GB
**用途**: 处理景点图片，图文推荐

## 🎯 推荐下载顺序

### 阶段1: 基础集成（现在）
```bash
cd /root/autodl-tmp/goafar_project/open_resource

# 下载这3个
git clone https://github.com/QwenLM/Qwen2.5.git
git clone https://github.com/WLiK/LLM4Rec.git
git clone https://github.com/langchain-ai/langchain.git
```

### 阶段2: 深度学习（之后）
```bash
# 再下载这2个
git clone https://github.com/SAI990323/TALLRec.git
git clone https://github.com/HKUDS/RecLLM.git
```

### 阶段3: 多模态（可选）
```bash
# 如果需要图像处理
git clone https://github.com/haotian-liu/LLaVA.git
git clone https://github.com/run-llama/llama_index.git
```

## 📝 下载后要看的重点文件

### LLM4Rec
- `README.md` - 了解整体架构
- `src/models/llm_recommender.py` - LLM推荐器实现
- `src/prompts/recommendation_prompts.py` - 提示词模板
- `configs/` - 配置文件

### Qwen2.5
- `README_CN.md` - 中文文档
- `examples/demo_chat.py` - 对话示例
- `examples/vllm_wrapper.py` - 推理优化

### LangChain
- `docs/docs/modules/prompts/` - Prompt教程
- `libs/langchain/langchain/llms/` - LLM集成
- `cookbook/` - 实战案例

## ⚡ 当前项目状态

**已完成**:
- ✅ BGE-M3语义检索
- ✅ RecBole序列推荐框架
- ✅ OR-Tools路线规划
- ✅ 模板文案生成
- ✅ LLM集成框架（待填充实现）

**需要LLM4Rec代码来完善**:
1. Intent Understanding - 参考 RecLLM 的意图理解
2. LLM Reranking - 参考 LLM4Rec 的排序方法
3. POI Feature Encoding - 参考 TALLRec 的物品编码
4. Explanation Generation - 参考 RecLLM 的解释生成

---

## 🎬 下载完成后的下一步

1. **学习参考代码**: 查看LLM4Rec等项目的实现
2. **完善我们的模块**: 补充 `src/llm4rec/` 的具体实现
3. **集成Qwen模型**: 实现LLM调用逻辑
4. **测试端到端**: 运行完整的推荐流程

**请告诉我你要下载哪些项目，我来帮你集成！**

