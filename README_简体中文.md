# GoAfar 智能旅行路线推荐系统

> 基于 BGE-M3 / RecBole / OR-Tools / Qwen3 的多模块协同推荐系统

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 项目概述

GoAfar 是一个完整的智能旅行路线推荐系统，集成了：
- **BGE-M3** 语义检索（GPU加速600倍）
- **RecBole** 序列推荐
- **OR-Tools** VRPTW路线规划
- **Qwen3-8B** LLM增强推荐

**数据规模**: 1333个景点，覆盖8个省份（新疆、西藏、云南、四川、甘肃、青海、宁夏、内蒙古）

## ⚡ 快速开始

```bash
# 1. 激活环境
cd /root/autodl-tmp/goafar_project
source /root/miniconda3/bin/activate goafar

# 2. 运行测试
python test_pipeline.py
# ✓ 所有测试通过！

# 3. 运行推荐
python main.py
```

## 📊 性能数据

| 任务 | CPU | GPU (RTX 4090) | 加速 |
|------|-----|----------------|------|
| 向量生成（1333个） | 20分钟 | **1.99秒** | **600x** |
| 端到端推荐 | 60分钟 | 10分钟 | 6x |

## 🎮 使用示例

### Python API
```python
from main import recommend_route

result = recommend_route(
    query_text="想去新疆看喀纳斯秋天的景色",
    province="新疆",
    max_hours=10
)

print(result['title'])
# 输出: 北疆秘境｜喀纳斯湖-禾木村-白哈巴村，秋日童话
```

## 📁 项目结构

```
goafar_project/
├── src/                    # 源代码（模块化）
│   ├── data_processing/   # 数据处理
│   ├── embedding/         # BGE-M3向量
│   ├── recommendation/    # RecBole推荐
│   ├── routing/           # OR-Tools路线
│   ├── content_generation/# 文案生成
│   └── llm4rec/          # LLM4Rec增强
├── data/                  # 数据（1333景点）
├── models/                # 模型（BGE-M3 + Qwen3）
├── outputs/               # 输出结果
├── open_resource/         # 开源代码参考
├── main.py               # 基础版入口
├── run_with_llm.py       # LLM增强版
└── test_pipeline.py      # 测试脚本
```

## 🔧 技术栈

- **语义检索**: BGE-M3 (FlagEmbedding)
- **序列推荐**: SASRec (RecBole)  
- **路线规划**: VRPTW (OR-Tools)
- **LLM增强**: Qwen3-8B + TALLRec
- **GPU加速**: PyTorch CUDA

## 📖 文档

- [START_HERE.md](START_HERE.md) - 立即开始
- [GPU优化说明.md](GPU优化说明.md) - GPU优化
- [LLM4REC_INTEGRATION.md](LLM4REC_INTEGRATION.md) - LLM4Rec集成
- [✅项目完成总结.md](✅项目完成总结.md) - 项目总结

## 🏆 项目亮点

1. ✅ GPU向量生成 **600倍加速**（1.99秒/1333个POI）
2. ✅ 多模块协同（语义+序列+路线+LLM）
3. ✅ VRPTW硬约束保证可行性
4. ✅ LLM4Rec全链路增强
5. ✅ 完整的工程化实现

## 🎓 适用场景

- **算法面试**: 推荐系统、LLM应用、路线规划
- **项目展示**: 完整的端到端系统
- **学习参考**: 7个顶级开源项目集成

## 📞 联系

项目完全开源，欢迎使用和改进！

---

**状态**: ✅ Production Ready  
**更新**: 2025-11-08
