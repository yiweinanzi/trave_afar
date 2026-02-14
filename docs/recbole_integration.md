# RecBole 在线召回集成使用指南

## 概述

本次集成将 RecBole 序列推荐模型引入到 GoAfar 推荐系统中，替代了原有的简单流行度召回，提供了更个性化的行为召回能力。

## 主要特性

### 1. RecBoleProvider 类

位置：`src/recommendation/recbole_trainer.py`

**功能：**
- 加载训练好的 RecBole 模型（SASRec）
- 为用户生成个性化推荐
- 自动降级到流行度召回（当模型不可用时）
- 支持用户冷启动处理

**关键方法：**
```python
# 初始化
provider = RecBoleProvider(
    model_path="outputs/recbole/saved",
    config_file="configs/recbole.yaml",
    use_gpu=True,
    fallback_to_popular=True
)

# 预测
rec_df, metadata = provider.predict(
    user_id="user_123",
    topk=30,
    poi_df=poi_df,
    filter_history=True
)

# 获取用户历史长度
history_len = provider.get_user_history_length("user_123")
```

### 2. 动态权重融合

位置：`src/recommendation/candidate_merger.py`

**策略：**
- **无历史行为**：降低行为召回权重，提升语义和地理召回
- **历史行为较少**（<5）：逐步提升行为召回权重
- **历史行为充足**（≥5）：使用基础权重或提升行为召回

**函数：**
```python
def adaptive_fusion(
    user_history_length: int,
    base_dense_weight: float = 0.55,
    base_behavior_weight: float = 0.30,
    base_geo_weight: float = 0.15,
    min_behavior_weight: float = 0.10,
    max_behavior_weight: float = 0.50,
    history_threshold: int = 5
) -> tuple
```

### 3. 配置更新

位置：`configs/runtime.yaml`

**新增配置：**
```yaml
recall:
  # 原有配置
  semantic_topk: 80
  behavior_topk: 60
  geo_topk: 40
  final_topk: 80
  fusion: rrf
  rrf_k: 60
  dense_weight: 0.55
  behavior_weight: 0.30
  geo_weight: 0.15
  calibrate: minmax

  # RecBole 行为召回配置
  behavior_provider: popularity  # 可选: popularity, recbole
  recbole_model_path: outputs/recbole/saved
  recbole_config: configs/recbole.yaml
  recbole_use_gpu: true

  # 动态权重融合
  adaptive_fusion: false
```

## 使用方式

### 方式 1：使用原有流行度召回（默认）

```python
from recommendation.candidate_merger import merge_candidates

candidates = merge_candidates(
    query_text="想去新疆看雪山和草原",
    user_id="user_123",
    use_recbole=False,  # 不使用 RecBole
    adaptive_fusion_enabled=False
)
```

### 方式 2：使用 RecBole 模型召回

```python
candidates = merge_candidates(
    query_text="想去新疆看雪山和草原",
    user_id="user_123",
    use_recbole=True,  # 使用 RecBole
    recbole_model_path="outputs/recbole/saved",
    recbole_config="configs/recbole.yaml",
    recbole_use_gpu=True
)
```

### 方式 3：启用动态权重融合

```python
candidates = merge_candidates(
    query_text="想去新疆看雪山和草原",
    user_id="user_123",
    use_recbole=True,
    adaptive_fusion_enabled=True  # 启用动态权重
)
```

## 降级策略

RecBole 集成实现了多层降级机制，确保系统稳定性：

1. **RecBole 未安装** → 降级到流行度召回
2. **模型文件不存在** → 降级到流行度召回
3. **模型加载失败** → 降级到流行度召回
4. **用户冷启动**（训练集中无此用户）→ 使用流行度推荐
5. **预测失败** → 降级到流行度召回

## 训练 RecBole 模型

### 1. 导出数据

```bash
cd /root/autodl-tmp/goafar_project_broken
python -c "
from src.recommendation.recbole_trainer import export_recbole_data
export_recbole_data('data/user_events.csv', 'outputs/recbole/custom')
"
```

### 2. 训练模型（需要 GPU）

```python
from src.recommendation.recbole_trainer import train_recbole_model

result = train_recbole_model(
    config_file='configs/recbole.yaml',
    gpu_id=0
)
```

### 3. 使用训练好的模型

训练完成后，模型会保存在 `outputs/recbole/saved/` 目录下，系统会自动加载。

## 测试

运行验证脚本：

```bash
python verify_recbole_integration.py
```

运行单元测试（需要修复依赖）：

```bash
python test_recbole_unit.py
```

## 技术细节

### RecBoleProvider 实现

**初始化流程：**
1. 检查模型文件是否存在
2. 加载 RecBole 配置
3. 创建数据集和模型
4. 加载模型权重
5. 预加载流行度数据（用于降级）

**预测流程：**
1. 将 user_id 转换为内部 ID
2. 获取用户历史序列
3. 处理冷启动（用户不在训练集）
4. 使用模型预测所有物品的得分
5. 过滤历史交互物品
6. 返回 Top-K 推荐

**降级流程：**
1. 检查模型是否可用
2. 不可用时使用预加载的流行度数据
3. 按全局流行度排序
4. 返回 Top-K 推荐

### 动态权重融合算法

```python
if user_history_length == 0:
    # 无历史行为
    behavior_weight = min_behavior_weight
    geo_weight += (base_behavior_weight - min_behavior_weight) * 0.5
    dense_weight = 1.0 - behavior_weight - geo_weight

elif user_history_length < threshold:
    # 历史行为较少
    ratio = user_history_length / threshold
    behavior_weight = min_weight + (max_weight - min_weight) * ratio * 0.5
    dense_weight = base_dense_weight - (behavior_weight - base_behavior_weight) * 0.7
    geo_weight = 1.0 - dense_weight - behavior_weight

else:
    # 历史行为充足
    behavior_weight = min(max_weight, base_behavior_weight * 1.3)
    dense_weight = base_dense_weight - (behavior_weight - base_behavior_weight)
    geo_weight = base_geo_weight
```

## 兼容性

- **向后兼容**：不使用 RecBole 时，系统保持原有行为
- **渐进式升级**：可以先训练模型，再逐步启用
- **配置灵活**：可以随时切换行为召回方式

## 性能考虑

1. **单例模式**：RecBoleProvider 使用全局单例，避免重复加载模型
2. **延迟加载**：只在首次使用时加载模型
3. **GPU 加速**：支持 GPU 推理（推荐）
4. **批量预测**：未来可扩展为批量预测以提高吞吐量

## 未来改进

1. **批量预测**：支持一次预测多个用户
2. **模型版本管理**：支持多个模型版本并存
3. **A/B 测试**：支持不同召回策略的 A/B 测试
4. **实时更新**：支持模型热更新
5. **监控指标**：添加预测延迟、命中率等监控

## 文件清单

- `src/recommendation/recbole_trainer.py`：RecBole 训练器和推理类
- `src/recommendation/candidate_merger.py`：候选合并器（已集成 RecBole）
- `configs/runtime.yaml`：运行时配置（已添加 RecBole 配置）
- `configs/recbole.yaml`：RecBole 模型配置
- `verify_recbole_integration.py`：代码结构验证脚本
- `test_recbole_unit.py`：单元测试脚本

## 总结

本次集成成功实现了：

1. ✓ RecBoleProvider 类：完整的模型加载和预测功能
2. ✓ 动态权重融合：根据用户历史长度自适应调整权重
3. ✓ 降级策略：多层降级机制确保系统稳定性
4. ✓ 冷启动支持：新用户使用流行度推荐
5. ✓ 配置更新：runtime.yaml 支持 RecBole 配置
6. ✓ 向后兼容：不影响现有功能

系统现在可以根据配置灵活选择行为召回方式，在模型可用时提供个性化推荐，在模型不可用时自动降级到流行度召回，确保了系统的稳定性和可用性。
