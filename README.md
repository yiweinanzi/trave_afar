# GoAfar 智能旅行路线推荐系统

> **目标**：将原有 "AI 驱动路线生成" 重构为**可解释、可复现**的算法体系，聚焦 4 个核心库：**BGE-M3 / RecBole / OR-Tools / TRL（+ OSMnx）**

## 📋 项目概述

GoAfar 是一个基于多模块协同的智能旅行路线推荐系统，通过结合**语义检索**、**序列推荐**、**路线优化**和**文案生成**四大核心技术，为用户提供个性化的旅行路线规划。

### 核心特点

- ✅ **可解释性**：每个模块独立可验证，指标清晰
- ✅ **可复现性**：完整的配置文件和实验脚本
- ✅ **工业级**：使用成熟开源框架，易于部署
- ✅ **多省份支持**：覆盖新疆、西藏、云南、四川、甘肃、宁夏、内蒙古、青海 8 个省份，1333 个景点

## 🏗️ 系统架构

```
用户查询 "想去喀纳斯看秋天的景色"
    ↓
┌─────────────────────────────────────────┐
│  1. 语义检索 (BGE-M3)                    │
│     - 理解口语化需求                      │
│     - Top-K 语义相似景点                  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  2. 序列召回 (RecBole SASRec)            │
│     - 基于历史行为推荐                    │
│     - Top-K 个性化候选                    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  3. 候选池合并                            │
│     - 语义检索 ∪ 序列召回                 │
│     - 综合评分排序                        │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  4. 路线规划 (OR-Tools VRPTW)            │
│     - 考虑开放时间窗                      │
│     - 考虑停留时长                        │
│     - 考虑行程时间约束                    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  5. 文案生成 (TRL DPO / 提示词工程)       │
│     - 生成吸引人的标题                    │
│     - 生成详细的行程描述                  │
└─────────────────────────────────────────┘
    ↓
完整的旅行路线推荐
```

## 📂 项目结构

```
goafar_project/
├── data/                           # 数据目录
│   ├── poi.csv                    # 1333个景点（8省份）
│   └── user_events.csv            # 38K+用户行为
├── outputs/                       # 输出目录
│   ├── emb/                       # BGE-M3向量
│   ├── recbole/                   # RecBole结果
│   ├── routing/                   # 路线规划
│   └── results/                   # 最终推荐结果
├── models/                        # 预训练模型
│   └── Xorbits/bge-m3/           # BGE-M3模型
├── open_resource/                 # 开源代码参考
│   ├── FlagEmbedding-master/
│   ├── RecBole-master/
│   └── or-tools-stable/
├── configs/
│   └── recbole.yaml              # RecBole配置
├── src/                          # 源代码（模块化）
│   ├── data_processing/          # 数据处理模块
│   │   ├── sql_extractor.py     # SQL提取
│   │   └── event_generator.py   # 事件生成
│   ├── embedding/                # 嵌入模块
│   │   ├── bge_m3_encoder.py   # BGE-M3编码器
│   │   └── vector_builder.py    # 向量构建
│   ├── recommendation/           # 推荐模块
│   │   ├── candidate_merger.py  # 候选合并
│   │   └── recbole_trainer.py   # RecBole训练
│   ├── routing/                  # 路由规划模块
│   │   ├── time_matrix_builder.py # 时间矩阵
│   │   └── vrptw_solver.py      # VRPTW求解
│   ├── content_generation/       # 内容生成模块
│   │   └── title_generator.py   # 文案生成
│   └── utils/                    # 工具模块
│       └── model_downloader.py  # 模型下载
├── sql/
│   └── go_address.sql            # 原始数据
├── main.py                       # 主入口脚本
├── quick_start.sh                # 快速开始脚本
├── requirements.txt              # 依赖列表
└── README.md                     # 本文件
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
conda create -n goafar python=3.10 -y
conda activate goafar

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 快速开始（一键运行）

```bash
# 运行快速开始脚本（自动执行所有步骤）
bash quick_start.sh
```

### 3. 分步运行（可选）

```bash
# Step 1: 数据准备
python -c "from src.data_processing.sql_extractor import parse_go_address_sql; parse_go_address_sql()"

# Step 2: 生成POI向量
python src/embedding/vector_builder.py

# Step 3: 导出RecBole数据
python -c "from src.recommendation.recbole_trainer import export_recbole_data; export_recbole_data()"

# Step 4: 端到端演示
python main.py

# (可选) 训练RecBole模型
python -m recbole -c configs/recbole.yaml --gpu_id=0
```

### 4. Python API 使用

```python
# 导入主函数
from main import recommend_route

# 推荐路线
result = recommend_route(
    query_text="想去新疆看雪山和湖泊",
    province="新疆",
    max_hours=10
)

print(result['title'])
print(result['description'])
for stop in result['route']:
    print(f"{stop['arrival_time_str']} - {stop['poi_name']}")
```

## 📊 数据统计

- **景点总数**: 1333 个
- **覆盖省份**: 8 个（新疆、西藏、云南、四川、甘肃、宁夏、内蒙古、青海）
- **用户事件**: 38,579 条（模拟数据）
- **用户数量**: 1000 个（模拟）

### 省份分布

| 省份 | 景点数 | 占比 |
|------|--------|------|
| 新疆 | 312 | 23.4% |
| 四川 | 220 | 16.5% |
| 西藏 | 217 | 16.3% |
| 云南 | 151 | 11.3% |
| 甘肃 | 125 | 9.4% |
| 青海 | 108 | 8.1% |
| 宁夏 | 100 | 7.5% |
| 内蒙古 | 100 | 7.5% |

## 🔧 技术栈

### 1. BGE-M3 语义检索
- **模型**: BAAI/bge-m3
- **用途**: 理解用户口语化查询，召回语义相关景点
- **指标**: Recall@50, NDCG@10

### 2. RecBole 序列召回
- **模型**: SASRec (Self-Attentive Sequential Recommendation)
- **用途**: 基于用户历史行为预测兴趣景点
- **指标**: Recall@50, NDCG@10, MRR@10

### 3. OR-Tools VRPTW
- **算法**: Vehicle Routing Problem with Time Windows
- **用途**: 生成可行的旅行路线（考虑时间窗、停留时长）
- **指标**: 可行率、总时长、访问景点数

### 4. TRL DPO (可选)
- **算法**: Direct Preference Optimization
- **用途**: 文案风格对齐（或使用提示词工程）
- **替代方案**: GPT-4 / 通义千问 API

## 📈 评测与消融

### 检索召回评测

```python
# A/B 测试方案
1. BGE-M3 dense only
2. RecBole SASRec only  
3. dense ∪ SASRec (合并策略)

# 评测指标
- Recall@50: 召回率
- NDCG@10: 排序质量
- MRR@10: 首个相关结果排名
```

### 路线规划评测

```python
# 对比方案
1. 贪心算法（仅按兴趣排序）
2. OR-Tools VRPTW（考虑时间窗约束）

# 评测指标
- 可行率: 满足时间窗约束的路线比例
- 违约束率: 违反约束的次数
- 总时长: 平均行程时间
- 景点覆盖: 访问景点数量
```

## 🎯 使用示例

### 命令行演示

```bash
python src/07_pipeline_demo.py
```

### Python API

```python
from src.src_02_merge_candidates import merge_for_user
from src.src_04_solve_itinerary import solve_route

# 1. 召回候选景点
candidates = merge_for_user(
    query_text="想去新疆看雪山和湖泊",
    user_id="U0001",
    topk_dense=50,
    topk_seq=30
)

# 2. 规划路线
top_pois = candidates.nlargest(20, 'semantic_score')['poi_id'].tolist()
route = solve_route(
    candidate_poi_ids=top_pois,
    max_duration_hours=10
)
```

## 📝 面试话术

### Q1: 为什么不用端到端大模型？

**A**: 旅行场景有**硬约束**（营业时间、行程时长、可达性），需要**组合优化**保证可行性。大模型负责"理解需求 + 文案生成"，而**路线规划**交给 OR-Tools VRPTW，确保每条路线都是真实可行的。

### Q2: 为何选这4个库？

**A**: 
- **BGE-M3**: 支持dense/sparse/multi-vector三种检索，适合长文本和口语化查询
- **RecBole**: 统一的推荐框架，内置评测指标，SASRec是序列推荐的强基线
- **OR-Tools**: Google开源的组合优化库，VRPTW可以建模时间窗约束
- **TRL**: Hugging Face的RLHF框架，DPO适合小样本偏好对齐

### Q3: 如何处理冷启动？

**A**: 
1. **新用户**: 仅依赖语义检索（BGE-M3）+ 热门景点
2. **新景点**: 通过语义向量召回，无需历史数据
3. **跨区域**: 利用描述文本的语义相似度

## 📚 参考文献

- [FlagEmbedding/BGE-M3](https://github.com/FlagOpen/FlagEmbedding)
- [RecBole文档](https://recbole.io/docs/)
- [OR-Tools VRPTW示例](https://developers.google.com/optimization/routing/vrptw)
- [OSMnx文档](https://osmnx.readthedocs.io/en/stable/)
- [TRL DPO文档](https://huggingface.co/docs/trl/en/dpo_trainer)

## 🔍 下一步优化

1. **实时路网**: 接入高德/百度地图API获取真实导航时间
2. **在线学习**: 根据用户反馈实时更新推荐策略
3. **多日行程**: 扩展VRPTW支持多日游规划
4. **跨省路线**: 增加交通方式选择（飞机/火车/自驾）
5. **个性化文案**: 接入GPT-4 API生成更精准的文案

## 📄 License

MIT License

## 👥 贡献者

- 数据来源：GoAfar数据库（1333个景点）
- 算法设计：基于文档《GoAfar改造.md》

---

**注意**: 本项目为技术验证原型，实际部署需要考虑更多工程化问题（如缓存、并发、容错等）。

