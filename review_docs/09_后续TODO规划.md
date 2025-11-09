# 后续TODO规划

> 检查时间: 2025-01-XX  
> 基于代码审查和系统分析

---

## 📋 总览

**当前状态**: ✅ **所有核心功能已实现，系统完全可用**

本文档规划后续的**优化方向**和**改进点**，分为：
- 🔴 **高优先级** - 影响用户体验或系统性能
- 🟡 **中优先级** - 提升系统质量或扩展性
- 🟢 **低优先级** - 实验性功能或长期优化

---

## ✅ 已完成功能确认

### 核心功能（100%完成）
- [x] BGE-M3语义检索（Dense/Sparse/ColBERT）
- [x] RecBole序列推荐
- [x] VRPTW路线规划
- [x] LLM意图理解（模板+LLM）
- [x] LLM重排序（规则+LLM）
- [x] LLM文案生成（模板+LLM）
- [x] DPO训练脚本
- [x] GPU加速优化
- [x] Web UI界面
- [x] 全链路测试

### 代码质量
- [x] 所有TODO标记已实现
- [x] 降级策略完善
- [x] 错误处理机制
- [x] 文档完整

---

## 🔴 高优先级TODO

### 1. 性能优化

#### 1.1 时间矩阵缓存优化 ⏱️ **预计1-2天**
**问题**: 每次调用都重新计算时间矩阵，耗时8.5秒（占71%延迟）

**方案**:
```python
# 实现持久化缓存
# src/routing/time_matrix_builder.py
class TimeMatrixCache:
    def __init__(self, cache_file='outputs/routing/time_matrix_cache.pkl'):
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def get(self, poi_ids):
        """获取缓存的时间矩阵"""
        key = tuple(sorted(poi_ids))
        return self.cache.get(key)
    
    def set(self, poi_ids, time_matrix):
        """保存时间矩阵到缓存"""
        key = tuple(sorted(poi_ids))
        self.cache[key] = time_matrix
        self._save_cache()
```

**预期效果**:
- 首次查询: 8.5秒（正常）
- 缓存命中: <100ms
- 延迟降低: **85倍**（重复查询）

**优先级**: 🔴 **高** - 直接影响用户体验

---

#### 1.2 LLM批量推理优化 ⏱️ **预计1天**
**问题**: LLM意图理解、重排序、生成都是单条处理，效率低

**方案**:
```python
# src/llm4rec/intent_understanding.py
def batch_understand(self, queries):
    """批量理解用户意图"""
    prompts = [self._build_prompt(q) for q in queries]
    
    # 批量生成
    model_inputs = self.tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(self.model.device)
    
    generated_ids = self.model.generate(
        **model_inputs,
        max_new_tokens=200,
        temperature=0.3
    )
    
    # 批量解码
    responses = self.tokenizer.batch_decode(
        generated_ids, 
        skip_special_tokens=True
    )
    
    return [self._parse_response(r) for r in responses]
```

**预期效果**:
- 单条处理: 0.5-1秒/条
- 批量处理（10条）: 2-3秒（平均0.2-0.3秒/条）
- 效率提升: **2-3倍**

**优先级**: 🔴 **高** - 提升LLM模块效率

---

#### 1.3 向量检索索引优化 ⏱️ **预计2-3天**
**问题**: 当前使用numpy计算，未使用专业向量数据库

**方案**:
```python
# 使用FAISS或Milvus
from faiss import IndexFlatIP

class VectorIndex:
    def __init__(self, embeddings):
        self.index = IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))
    
    def search(self, query_vec, topk=50):
        """快速检索"""
        distances, indices = self.index.search(
            query_vec.reshape(1, -1).astype('float32'), 
            topk
        )
        return indices[0], distances[0]
```

**预期效果**:
- 当前: 35ms（numpy）
- FAISS: <10ms（GPU加速）
- 延迟降低: **3-5倍**

**优先级**: 🔴 **高** - 提升检索性能

---

### 2. 功能完善

#### 2.1 多日行程规划完善 ⏱️ **预计2-3天**
**问题**: `multi_day_planner.py`存在但未完全集成到主流程

**方案**:
```python
# main.py
def recommend_route(..., days=None):
    if days and days > 1:
        # 使用多日规划器
        from src.routing.multi_day_planner import MultiDayPlanner
        planner = MultiDayPlanner()
        return planner.plan_multi_day(
            candidate_pois=candidates,
            days=days,
            hours_per_day=8
        )
    else:
        # 单日规划
        return single_day_plan(...)
```

**需要完善**:
- [ ] 集成到`main.py`和`run_with_llm.py`
- [ ] 支持跨城市多日规划
- [ ] 优化POI分配策略（考虑地理距离）
- [ ] 添加历史路线推荐功能

**优先级**: 🔴 **高** - 用户需求（多日行程）

---

#### 2.2 DPO训练数据准备 ⏱️ **预计1-2天**
**问题**: `make_prefs.py`生成的是示例数据，需要真实偏好数据

**方案**:
```python
# src/content_generation/make_prefs.py
def collect_real_preferences():
    """收集真实偏好数据"""
    # 1. 从用户反馈收集
    # 2. A/B测试对比
    # 3. 人工标注
    # 4. 规则生成（基于质量指标）
    
    preferences = []
    for route in historical_routes:
        # 对比不同标题/描述的质量
        chosen = route['high_quality_title']
        rejected = route['low_quality_title']
        preferences.append({
            'prompt': route['query'],
            'chosen': chosen,
            'rejected': rejected
        })
    
    return preferences
```

**优先级**: 🔴 **高** - 提升DPO训练效果

---

### 3. 错误处理和监控

#### 3.1 完善错误处理 ⏱️ **预计1天**
**问题**: 部分模块错误处理不够完善

**需要改进**:
- [ ] LLM调用超时处理
- [ ] VRPTW无解时的详细错误信息
- [ ] 向量检索失败时的降级策略
- [ ] 数据文件缺失时的友好提示

**优先级**: 🔴 **高** - 提升系统稳定性

---

#### 3.2 添加日志和监控 ⏱️ **预计1-2天**
**问题**: 缺少详细的日志和性能监控

**方案**:
```python
# src/utils/logger.py
import logging
from datetime import datetime

class PerformanceLogger:
    def __init__(self):
        self.logger = logging.getLogger('goafar')
        self.metrics = {}
    
    def log_module_time(self, module_name, duration):
        """记录模块耗时"""
        self.metrics[module_name] = duration
        self.logger.info(f"{module_name}: {duration:.2f}s")
    
    def get_summary(self):
        """获取性能摘要"""
        total = sum(self.metrics.values())
        return {
            'total_time': total,
            'module_times': self.metrics,
            'breakdown': {k: v/total*100 for k, v in self.metrics.items()}
        }
```

**优先级**: 🔴 **高** - 便于问题排查和优化

---

## 🟡 中优先级TODO

### 4. 代码质量改进

#### 4.1 单元测试覆盖 ⏱️ **预计3-5天**
**问题**: 当前只有集成测试，缺少单元测试

**需要添加**:
- [ ] `test_embedding.py` - 测试BGE-M3编码和检索
- [ ] `test_recommendation.py` - 测试RecBole和候选合并
- [ ] `test_routing.py` - 测试VRPTW求解
- [ ] `test_llm4rec.py` - 测试LLM模块
- [ ] `test_content_generation.py` - 测试文案生成

**目标覆盖率**: >80%

**优先级**: 🟡 **中** - 提升代码质量

---

#### 4.2 代码重构和优化 ⏱️ **预计2-3天**
**问题**: 部分代码可以进一步优化

**需要重构**:
- [ ] `llm_generator.py` - 统一LLM调用接口
- [ ] `intent_understanding.py` - 简化LLM调用逻辑
- [ ] `llm_reranker.py` - 统一重排序接口
- [ ] 提取公共工具函数

**优先级**: 🟡 **中** - 提升代码可维护性

---

### 5. 功能扩展

#### 5.1 支持更多检索模式 ⏱️ **预计2-3天**
**问题**: ColBERT已实现但未充分测试和优化

**需要完善**:
- [ ] ColBERT检索性能测试
- [ ] 混合检索（Dense + ColBERT）
- [ ] 稀疏向量检索（Sparse）
- [ ] 检索模式自动选择

**优先级**: 🟡 **中** - 提升检索效果

---

#### 5.2 支持更多LLM模型 ⏱️ **预计1-2天**
**问题**: 当前只支持Qwen3-8B

**需要支持**:
- [ ] Qwen2.5系列
- [ ] ChatGLM3
- [ ] Baichuan2
- [ ] API模式（OpenAI、DeepSeek等）

**优先级**: 🟡 **中** - 提升灵活性

---

#### 5.3 实时推荐优化 ⏱️ **预计2-3天**
**问题**: 当前是离线推荐，缺少实时更新

**需要实现**:
- [ ] 用户行为实时反馈
- [ ] 动态调整推荐权重
- [ ] 实时POI热度更新
- [ ] 在线学习机制

**优先级**: 🟡 **中** - 提升推荐效果

---

### 6. 数据质量提升

#### 6.1 POI数据增强 ⏱️ **预计3-5天**
**问题**: POI描述质量参差不齐

**需要完善**:
- [ ] 批量使用LLM增强POI描述
- [ ] 添加POI图片链接
- [ ] 添加POI标签（季节、天气、人群等）
- [ ] 添加POI评分和评论数

**优先级**: 🟡 **中** - 提升推荐质量

---

#### 6.2 用户行为数据增强 ⏱️ **预计2-3天**
**问题**: 用户事件数据较少（38K条）

**需要完善**:
- [ ] 数据增强（合成用户行为）
- [ ] 添加用户画像数据
- [ ] 添加用户偏好标签
- [ ] 添加时间序列特征

**优先级**: 🟡 **中** - 提升序列推荐效果

---

## 🟢 低优先级TODO

### 7. 实验性功能

#### 7.1 强化学习路线规划 ⏱️ **预计5-7天**
**问题**: VRPTW是静态规划，无法动态调整

**实验方向**:
- [ ] 使用RL优化路线规划
- [ ] 考虑用户实时反馈
- [ ] 动态调整停留时长

**优先级**: 🟢 **低** - 实验性功能

---

#### 7.2 多模态推荐 ⏱️ **预计5-7天**
**问题**: 当前只使用文本，未使用图片

**实验方向**:
- [ ] 使用CLIP进行图片-文本匹配
- [ ] POI图片检索
- [ ] 视觉相似度计算

**优先级**: 🟢 **低** - 实验性功能

---

#### 7.3 知识图谱增强 ⏱️ **预计5-7天**
**问题**: 当前未利用POI之间的关联关系

**实验方向**:
- [ ] 构建POI知识图谱
- [ ] 基于图谱的推荐
- [ ] 关系推理

**优先级**: 🟢 **低** - 实验性功能

---

### 8. 文档和部署

#### 8.1 API文档完善 ⏱️ **预计1-2天**
**问题**: 缺少API接口文档

**需要添加**:
- [ ] FastAPI接口文档
- [ ] 请求/响应示例
- [ ] 错误码说明

**优先级**: 🟢 **低** - 提升易用性

---

#### 8.2 Docker部署 ⏱️ **预计2-3天**
**问题**: 当前需要手动配置环境

**需要实现**:
- [ ] Dockerfile
- [ ] docker-compose.yml
- [ ] 模型文件管理
- [ ] 环境变量配置

**优先级**: 🟢 **低** - 提升部署便利性

---

#### 8.3 性能基准测试 ⏱️ **预计2-3天**
**问题**: 缺少标准化的性能测试

**需要实现**:
- [ ] 标准测试数据集
- [ ] 性能基准报告
- [ ] 对比实验（不同配置）

**优先级**: 🟢 **低** - 便于性能对比

---

## 📊 TODO优先级总结

| 优先级 | 数量 | 预计总时间 | 核心价值 |
|--------|------|------------|----------|
| 🔴 高 | 6项 | 10-15天 | 性能优化、功能完善 |
| 🟡 中 | 6项 | 15-20天 | 代码质量、功能扩展 |
| 🟢 低 | 6项 | 20-30天 | 实验性功能、部署优化 |

**总计**: 18项TODO，预计45-65天工作量

---

## 🎯 推荐实施顺序

### 第一阶段（1-2周）：性能优化
1. ✅ 时间矩阵缓存优化（1-2天）
2. ✅ LLM批量推理优化（1天）
3. ✅ 向量检索索引优化（2-3天）
4. ✅ 添加日志和监控（1-2天）

**预期效果**: 端到端延迟降低50%+

---

### 第二阶段（2-3周）：功能完善
5. ✅ 多日行程规划完善（2-3天）
6. ✅ DPO训练数据准备（1-2天）
7. ✅ 完善错误处理（1天）
8. ✅ 支持更多检索模式（2-3天）

**预期效果**: 功能完整性提升，用户体验改善

---

### 第三阶段（3-4周）：代码质量
9. ✅ 单元测试覆盖（3-5天）
10. ✅ 代码重构和优化（2-3天）
11. ✅ 支持更多LLM模型（1-2天）
12. ✅ POI数据增强（3-5天）

**预期效果**: 代码质量提升，可维护性增强

---

## 📝 注意事项

### 1. 保持向后兼容
- 所有优化不应破坏现有功能
- 添加新功能时保留降级策略
- 确保测试通过

### 2. 性能优先
- 优先优化影响用户体验的部分
- 关注端到端延迟
- 考虑GPU资源利用

### 3. 文档同步
- 代码更新时同步更新文档
- 记录性能提升数据
- 更新README和review_docs

### 4. 测试驱动
- 先写测试，再实现功能
- 确保测试覆盖率>80%
- 定期运行性能基准测试

---

## ✅ 检查清单

### 每次提交前检查
- [ ] 代码通过所有测试
- [ ] 性能未退化（或已优化）
- [ ] 文档已更新
- [ ] 错误处理完善
- [ ] 日志记录清晰

### 每周检查
- [ ] TODO进度更新
- [ ] 性能指标监控
- [ ] 用户反馈收集
- [ ] 代码质量检查

---

**最后更新**: 2025-01-XX  
**文档版本**: 1.0  
**状态**: 📋 **规划中** - 待实施

