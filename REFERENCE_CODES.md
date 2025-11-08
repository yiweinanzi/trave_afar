# 建议下载的参考代码和资源

## 核心参考代码

### 1. RecBole 官方示例 ⭐⭐⭐ (最重要)
**仓库**: https://github.com/RUCAIBox/RecBole
**用途**: 
- 学习如何正确配置 RecBole
- 了解 SASRec 模型的参数调优
- 参考序列推荐的最佳实践

**建议下载路径**:
```bash
git clone https://github.com/RUCAIBox/RecBole.git
# 重点查看:
# - examples/  (示例代码)
# - recbole/config/  (配置文件示例)
# - docs/  (文档)
```

### 2. OR-Tools 官方示例 ⭐⭐⭐ (最重要)
**仓库**: https://github.com/google/or-tools
**用途**:
- VRPTW (Vehicle Routing Problem with Time Windows) 完整实现
- 时间窗约束的正确设置方法
- 求解参数调优技巧

**建议下载路径**:
```bash
git clone https://github.com/google/or-tools.git
# 重点查看:
# - examples/python/vrptw.py
# - examples/python/cvrptw.py
# - ortools/constraint_solver/samples/
```

**或直接查看在线文档**:
- https://developers.google.com/optimization/routing/vrptw

### 3. FlagEmbedding 示例 ⭐⭐
**仓库**: https://github.com/FlagOpen/FlagEmbedding
**用途**:
- BGE-M3 的正确使用方法
- dense + sparse + colbert 多向量检索
- 批量编码优化

**建议下载路径**:
```bash
git clone https://github.com/FlagOpen/FlagEmbedding.git
# 重点查看:
# - examples/inference/embedder/  (推理示例)
# - FlagEmbedding/BGE_M3/  (模型实现)
```

### 4. TRL DPO 示例 ⭐
**仓库**: https://github.com/huggingface/trl
**用途**:
- DPO训练的完整流程
- LoRA配置参数
- 偏好数据格式

**建议下载路径**:
```bash
git clone https://github.com/huggingface/trl.git
# 重点查看:
# - examples/scripts/dpo.py
# - docs/dpo_trainer.md
```

### 5. OSMnx 实战案例 ⭐
**仓库**: https://github.com/gboeing/osmnx-examples
**用途**:
- 下载和处理路网数据
- 计算最短路径和行驶时间
- 处理大规模路网的优化技巧

**建议下载路径**:
```bash
git clone https://github.com/gboeing/osmnx-examples.git
# 重点查看:
# - notebooks/  (Jupyter示例)
# - 特别关注: 03-graph-place-queries.ipynb, 13-travel-times-speeds.ipynb
```

## 可选参考项目

### 6. 旅游推荐系统实战项目
**搜索关键词**: 
- "tourism recommendation system github"
- "travel itinerary planning github"
- "POI recommendation deep learning"

**推荐仓库**:
```
https://github.com/LibCity/Bigscity-LibCity  # 城市计算库，包含POI推荐
https://github.com/RUCAIBox/RecBole-GNN     # 图神经网络推荐
```

### 7. 路线规划相关
**推荐仓库**:
```
https://github.com/pgRouting/pgrouting     # PostgreSQL路线规划扩展
https://github.com/valhalla/valhalla        # 开源路线规划引擎
```

## 当前最需要的（优先级排序）

### 🔥 优先级 1（必需）
1. **OR-Tools VRPTW示例** - 确保路线规划算法正确实现
2. **RecBole官方文档和示例** - 学习正确的数据格式和配置

### ⚡ 优先级 2（重要）
3. **FlagEmbedding示例** - 优化向量生成和检索效率
4. **OSMnx实战案例** - 如果需要真实路网数据

### 💡 优先级 3（可选）
5. **TRL DPO示例** - 如果决定训练文案模型（否则用API）
6. **其他旅游推荐项目** - 学习业界最佳实践

## 具体建议

基于当前项目进度，我建议你：

1. **立即下载**: OR-Tools 和 RecBole 的官方仓库
2. **重点学习**: 
   - `or-tools/examples/python/vrptw.py`
   - `RecBole/examples/run_*.py`
3. **可选下载**: 如果想优化检索，下载 FlagEmbedding
4. **暂不需要**: TRL（我们可以用提示词工程代替）

是否需要我帮你：
- 下载这些参考代码？
- 或者直接基于现有代码继续完善项目？

