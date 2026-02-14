# GoAfar 会话上下文日志（2026-02-15 续）

## 会话概述

**日期**: 2026-02-15
**会话主题**: SFT/GRPO/DPO训练数据准备与评测体系建设
**前置状态**: 项目已完成工程收敛，数据已整合到data/all/目录

---

## 一、本次会话目标

用户要求在等待模型下载期间：
1. 准备SFT训练数据（使用LoRA）
2. 准备GRPO训练数据（RL强化学习）
3. 准备DPO训练数据
4. 建立完整的评测体系

---

## 二、关键决策记录

### 2.1 数据策略决策

#### 决策1: 使用data/all/作为数据源
**背景**: 项目中已有127,978个POI的整合数据
**决策内容**:
- 使用`data/all/poi_expanded.csv`（127,978个POI，30省份）
- 使用`data/all/route_templates.json`（174条路线模板）
- 使用`data/all/user_events.csv`（38,580条用户行为）

**影响**:
- POI覆盖从1,333→127,978（96倍增长）
- 省份覆盖从8→30

#### 决策2: 分层构建三类训练数据
**SFT数据**（1,774条）:
- 意图理解: 500条（从路线模板生成）
- 路线生成: 174条（路线模板直接使用）
- 文案生成: 443条（标题+描述生成）
- 现有规划: 657条（从用户行为合成）

**DPO数据**（300对）:
- 路线偏好: 100对（模板vs扰动）
- 标题偏好: 100对（优质vs平淡）
- 轨迹偏好: 100对（高参与vs扰动）

**GRPO数据**（1,754条）:
- 下一步POI预测任务
- 从用户轨迹截断生成

### 2.2 技术实现决策

#### 决策3: 使用TRL框架
**框架选择**:
- SFT: `TRL.SFTTrainer`
- DPO: `TRL.DPOTrainer`
- GRPO: `TRL.GRPOTrainer`

**理由**: HuggingFace官方RLHF框架，与Qwen3兼容性好

#### 决策4: LoRA配置
```yaml
lora_r: 16
lora_alpha: 16
lora_dropout: 0.05
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

#### 决策5: 评测指标体系
| 模型 | 核心指标 | 良好阈值 |
|------|----------|----------|
| SFT | 省份准确率、POI重叠率 | >80%, >60% |
| DPO | 偏好对齐准确率 | >70% |
| GRPO | 下一POI准确率 | >30% |

---

## 三、已完成工作

### 3.1 数据准备脚本

#### 新建文件

**`src/data_processing/enhance_sft_data.py`** (~300行)
- 功能: 从路线模板生成高质量SFT数据
- 输出: `outputs/datasets/sft_data.jsonl`
- 特性:
  - 30省份关键词映射
  - 多样化查询模板
  - 专业旅游文案风格

**`src/data_processing/build_dpo_data.py`** (~350行)
- 功能: 构建DPO偏好对数据
- 输出: `outputs/datasets/dpo_prefs.csv`
- 扰动策略: shuffle/replace/drop/duplicate

#### 修复文件

**`src/data_processing/synthesize_training_data.py`**
- 修复导入路径（添加sys.path.insert）
- 更新默认路径到data/all/
- 输出:
  - planner_trajectories.jsonl: 657条
  - planner_preference_pairs.jsonl: 657条
  - planner_rl_prompts.jsonl: 1,754条

### 3.2 评测脚本

#### 新建评测脚本

**`src/evaluation/evaluate_sft.py`** (~400行)
- 意图理解评测: 省份准确率、天数MAE、兴趣F1
- 路线生成评测: POI重叠率、天数匹配率
- 文案生成评测: 标题/描述质量

**`src/evaluation/evaluate_dpo.py`** (~350行)
- 偏好对齐准确率
- 奖励分数对比
- 按来源分组统计

**`src/evaluation/evaluate_grpo.py`** (~300行)
- 下一POI准确率
- 路线可行性
- 奖励对比

**`src/evaluation/evaluate_pipeline.py`** (~350行)
- 端到端成功率
- 延迟分解（意图/召回/排序/规划）
- 默认5个测试查询

#### 配置文件

**`configs/evaluation.yaml`**
- 统一评测配置
- 指标阈值定义

#### 脚本

**`scripts/evaluate_all.sh`**
- 一键评测所有模型
- 支持参数化配置

### 3.3 文档更新

**`README.md`** - 全面更新
- 添加训练流程说明
- 添加评测指南
- 更新数据统计
- 更新版本日志到v2.2.0

**`docs/EVALUATION.md`** - 新建
- 评测方法说明
- 指标解释
- 示例报告

---

## 四、数据统计汇总

### 4.1 训练数据产出

| 数据类型 | 数量 | 文件 | 状态 |
|----------|------|------|------|
| SFT训练数据 | 1,774条 | sft_data.jsonl | ✅ |
| GRPO训练数据 | 1,754条 | grpo_planner_prompts.jsonl | ✅ |
| DPO偏好数据 | 300对 | dpo_prefs.csv | ✅ |
| 用户轨迹 | 657条 | planner_trajectories.jsonl | ✅ |

### 4.2 SFT数据构成

| 任务类型 | 数量 | 占比 |
|----------|------|------|
| 意图理解 | 500 | 28% |
| 路线生成 | 174 | 10% |
| 文案生成 | 443 | 25% |
| 现有规划 | 657 | 37% |

---

## 五、假设与前提

### 5.1 数据假设

1. **路线模板质量假设**: 174条模板是高质量真实路线
2. **用户行为假设**: click/fav/visit权重关系为1:2:3
3. **POI数据假设**: 坐标、时间窗等信息准确

### 5.2 模型假设

1. **Qwen3-8B假设**: 本地模型路径正确，兼容TRL
2. **LoRA假设**: r=16配置足以获得良好效果

### 5.3 评测假设

1. **指标阈值**: 省份准确率>80%、偏好对齐>70%、下一POI>30%为良好
2. **测试规模**: 100条评测样本足以代表整体性能

---

## 六、未解决的问题

### 6.1 模型下载问题
**状态**: 进行中
**问题**: Qwen3-Embedding-4B和Qwen3-Reranker-4B下载未完成
**解决方案**: 使用`scripts/download_models.sh`后台下载

### 6.2 训练资源问题
**状态**: 未测试
**问题**:
- SFT训练需要多少GPU显存？
- 训练时间预估？

**待验证**: 在实际硬件上运行训练脚本

### 6.3 GRPO训练实现
**状态**: 数据就绪，训练脚本待实现
**问题**: GRPO训练器需要完整实现

### 6.4 评测基准问题
**状态**: 部分解决
**问题**: 没有公开的旅游推荐评测基准

---

## 七、下一步行动

### 7.1 短期（立即可执行）

1. **完成模型下载**
   ```bash
   bash scripts/download_models.sh
   ```

2. **运行端到端评测（无模型基准）**
   ```bash
   python src/evaluation/evaluate_pipeline.py
   ```

### 7.2 中期（等待模型下载后）

1. **运行SFT训练**
2. **运行DPO训练**
3. **运行完整评测**

### 7.3 长期（优化方向）

1. 数据增强：收集更多真实用户轨迹
2. 模型优化：尝试不同LoRA配置
3. 评测完善：添加人类评测、A/B测试

---

## 八、关键文件清单

### 8.1 新建文件

| 文件 | 功能 |
|------|------|
| src/data_processing/enhance_sft_data.py | SFT数据增强 |
| src/data_processing/build_dpo_data.py | DPO偏好数据构建 |
| src/evaluation/evaluate_sft.py | SFT评测 |
| src/evaluation/evaluate_dpo.py | DPO评测 |
| src/evaluation/evaluate_grpo.py | GRPO评测 |
| src/evaluation/evaluate_pipeline.py | 端到端评测 |
| scripts/evaluate_all.sh | 一键评测 |
| configs/evaluation.yaml | 评测配置 |
| docs/EVALUATION.md | 评测文档 |

### 8.2 修改文件

| 文件 | 修改内容 |
|------|----------|
| src/data_processing/synthesize_training_data.py | 修复导入路径 |
| README.md | 添加训练和评测内容 |

---

## 九、技术要点

### 9.1 Qwen聊天格式
```python
text = f"""<|im_start|>system
你是一位专业的旅游规划助手。<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
{assistant_response}<|im_end|>"""
```

### 9.2 评测指标计算

**召回率**:
```python
def recall_at_k(predictions, ground_truth, k):
    pred_k = set(predictions[:k])
    true_set = set(ground_truth)
    return len(pred_k & true_set) / len(true_set)
```

**偏好对齐**:
```python
accuracy = (chosen_logprob > rejected_logprob).mean()
```

---

## 十、与前一会话的衔接

### 10.1 前一会话遗留状态
- 数据已整合到data/all/
- 基础训练数据合成脚本已实现
- RL骨架已搭建

### 10.2 本会话延续工作
- 完善训练数据（路线模板增强）
- 建立完整评测体系
- 准备训练和评测文档

---

**日志结束 - 下次会话可基于此继续推进**
