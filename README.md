# GoAfar 智能旅行路线推荐系统

> 基于 **BGE-M3** / **RecBole** / **OR-Tools** / **Qwen3** 的多模型协同推荐系统

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 项目简介

GoAfar 是一个完整的智能旅行路线推荐系统，实现了从用户查询到路线规划的全流程自动化。

**核心特点**：
- ⚡ **GPU加速600倍** - 向量生成1.99秒处理1333个POI
- 🎯 **召回率提升30%** - 多模型协同召回策略
- ✅ **可行率92%** - VRPTW保证时间窗约束
- 🤖 **LLM4Rec增强** - Qwen3全链路应用

**数据规模**：1333个景点，覆盖8省份（新疆、西藏、云南、四川、甘肃、青海、宁夏、内蒙古）

## ⚡ 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/yiweinanzi/trave_afar.git
cd trave_afar

# 2. 安装依赖
conda create -n goafar python=3.10 -y
conda activate goafar
pip install -r requirements.txt

# 3. 运行测试
python test_pipeline.py

# 4. 运行推荐
python main.py
```

## 📊 性能数据

| 任务 | CPU | GPU (RTX 4090) | 加速 |
|------|-----|----------------|------|
| 向量生成（1333个POI） | 20分钟 | **1.99秒** | **600x** |
| 端到端推荐 | 60分钟 | 10分钟 | 6x |

## 🏗️ 技术架构

```
用户查询 → LLM意图理解 → 多路召回(BGE-M3+RecBole) → LLM重排序 → VRPTW路线规划 → LLM文案生成 → 完整推荐
```

## 📁 项目结构

```
trave_afar/
├── src/                    # 源代码（6大模块）
│   ├── data_processing/   # 数据处理
│   ├── embedding/         # BGE-M3语义检索
│   ├── recommendation/    # RecBole序列推荐
│   ├── routing/           # OR-Tools路线规划
│   ├── content_generation/# 文案生成
│   ├── llm4rec/          # LLM4Rec增强
│   ├── evaluation/       # 评测系统
│   └── utils/            # 工具函数
├── data/                  # 数据（1333景点）
├── configs/               # 配置文件
├── main.py               # 主入口
└── test_pipeline.py      # 测试脚本
```

## 🔧 核心技术

- **语义检索**: BGE-M3 (FlagEmbedding) - 669.7 POI/秒
- **序列推荐**: SASRec (RecBole) - Recall@50提升30%
- **路线规划**: VRPTW (OR-Tools) - 可行率92%
- **LLM增强**: Qwen3-8B - 意图识别85%+

## 💡 核心功能

### 1. 语义检索
```python
from src.embedding.vector_builder import search_similar_pois

results = search_similar_pois("想去新疆看雪山", topk=10)
```

### 2. 路线推荐
```python
from main import recommend_route

result = recommend_route(
    query_text="想去喀纳斯看秋天的景色",
    province="新疆",
    max_hours=10
)
```

### 3. 多日规划
```python
from src.routing.multi_day_planner import MultiDayPlanner

planner = MultiDayPlanner()
result = planner.plan_multi_day(candidates, days=3)
```

## 📈 性能指标

- **GPU加速**: 600倍（向量生成）
- **召回率**: +30%
- **NDCG@10**: 0.82
- **可行率**: 92%
- **端到端延迟**: <30秒

## 📖 文档

- [START_HERE.md](START_HERE.md) - 快速开始指南
- [GPU优化说明.md](GPU优化说明.md) - GPU优化方案
- [LLM4REC_INTEGRATION.md](LLM4REC_INTEGRATION.md) - LLM4Rec集成
- [最终交付报告.md](最终交付报告.md) - 完整报告
- [outputs/简历-项目描述.md](outputs/简历-项目描述.md) - 简历材料

## 🎓 适用场景

- 算法面试（推荐系统/LLM应用/路线规划）
- 项目展示（完整的端到端系统）
- 学习参考（多框架集成实战）

## 📞 联系方式

- Email: 2268867257@qq.com
- GitHub: [@yiweinanzi](https://github.com/yiweinanzi)

## 📄 License

MIT License

---

**更新**: 2025-11-08  
**状态**: ✅ Production Ready
