# SFT/GRPO/DPO训练数据准备计划

## Context

用户需要在等待模型下载期间准备SFT（使用LoRA）、GRPO（RL强化学习）和DPO训练所需的数据。

现有资源：
- `data/all/poi_expanded.csv` - 127,978个POI
- `data/all/route_templates.json` - 174条路线模板
- `data/all/user_events.csv` - 38,580条用户行为
- `data/all/geolife_trajectories.csv` - GPS轨迹
- `data/all/solomon_benchmark.csv` - VRPTW基准

现有代码：
- `src/data_processing/synthesize_training_data.py` - 数据合成脚本
- `src/content_generation/train_sft.py` - SFT训练（支持LoRA）
- `src/content_generation/train_dpo.py` - DPO训练（支持LoRA）
- `src/rl/dataset_builder.py` - GRPO数据构建
- `src/rl/reward_manager.py` - 奖励计算

## 实施方案

### 1. 运行数据合成脚本

首先修复`synthesize_training_data.py`的导入路径问题，然后运行生成基础训练数据：

```bash
python src/data_processing/synthesize_training_data.py \
    --poi-csv data/all/poi_expanded.csv \
    --events-csv data/all/user_events.csv \
    --output-dir outputs/datasets \
    --synth-users 300
```

生成文件：
- `outputs/datasets/planner_trajectories.jsonl` - SFT轨迹数据
- `outputs/datasets/planner_preference_pairs.jsonl` - DPO偏好对
- `outputs/datasets/planner_rl_prompts.jsonl` - GRPO提示数据

### 2. 增强SFT数据（基于路线模板）

创建`src/data_processing/enhance_sft_data.py`，利用174条高质量路线模板生成SFT训练样本：

- **任务1：意图理解** - 从用户需求提取结构化信息
- **任务2：路线生成** - 生成多日POI序列
- **任务3：文案生成** - 生成路线标题和描述

### 3. 构建GRPO训练数据

运行`src/rl/dataset_builder.py`生成LLM格式的GRPO数据：

```bash
python -m src.rl.dataset_builder \
    --rl-prompts outputs/datasets/planner_rl_prompts.jsonl \
    --trajectories outputs/datasets/planner_trajectories.jsonl \
    --out-rl outputs/datasets/grpo_planner_prompts.jsonl \
    --out-sft outputs/datasets/sft_planner_samples.jsonl
```

### 4. 构建DPO偏好数据

创建`src/data_processing/build_dpo_data.py`，利用：
- 路线模板生成高质量chosen样本
- 随机打乱/替换POI生成rejected样本
- 用户行为参与度作为质量信号

## 需要创建/修改的文件

### 新建文件

1. **`src/data_processing/enhance_sft_data.py`**
   - 从route_templates.json生成SFT训练样本
   - 支持意图理解、路线生成、文案生成三类任务

2. **`src/data_processing/build_dpo_data.py`**
   - 从路线模板生成偏好对齐数据
   - 高质量模板=chosen，扰动样本=rejected
   - 输出CSV格式给train_dpo.py使用

### 修改文件

3. **`src/data_processing/synthesize_training_data.py`**
   - 修复导入路径：`from utils.id_mapping import normalize_poi_id` -> 相对导入
   - 更新默认路径指向`data/all/`

## 关键代码复用

- `_resolve_model_path()` - `src/content_generation/train_sft.py:38`
- `RewardManager.score_route()` - `src/rl/reward_manager.py:55`
- `load_base_data()` - `src/data_processing/synthesize_training_data.py:20`

## 执行顺序

1. 修复并运行 `synthesize_training_data.py` 生成基础数据
2. 运行 `enhance_sft_data.py` 从路线模板增强SFT数据
3. 运行 `dataset_builder.py` 构建GRPO数据
4. 运行 `build_dpo_data.py` 构建DPO偏好数据
5. 合并所有SFT数据到单个JSONL文件

## 验证方式

运行完成后检查：
```bash
# 数据统计
wc -l outputs/datasets/*.jsonl

# 查看样本格式
head -n 2 outputs/datasets/sft_data.jsonl | jq
head -n 2 outputs/datasets/grpo_planner_prompts.jsonl | jq
head -n 5 outputs/datasets/dpo_prefs.csv
```
