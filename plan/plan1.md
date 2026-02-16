下面给你一份**“全链路、可落地、面试官会追问的”《GoAfar 大厂级项目改造文档（LLM + RecSys + RL/GRPO with veRL）》**。它严格基于你当前项目的真实链路与技术债（你给的基线文档里已经梳理得非常清晰）并把它升级为**工业级推荐漏斗 + 约束决策（规划）+ LLM 对齐训练（GRPO/DPO）+ 可观测可回放的数据飞轮**的完整系统。

------

# 0. 项目定位升级（让面试官“想聊下去”）

## 0.1 一句话定位（简历/开场白）

**GoAfar：面向旅游/本地生活的“个性化多目标行程决策引擎”，融合多路召回与精排、约束优化（VRPTW）与强化学习（GRPO），并通过偏好对齐（DPO/GRPO）形成数据飞轮闭环。**

## 0.2 你要“冲大厂算法岗”的核心卖点（面试官最吃这套）

你现在的系统是“端到端 Demo 闭环”，但**大厂面试**最看重的是：

1. **算法深度**：为什么这样建模？为什么这套能提升指标？
2. **Trade-off**：速度 vs 体验；硬约束 vs 个性化；召回覆盖 vs 精排精度。
3. **严谨评测**：离线指标、消融、成本、延迟、稳定性。
4. **工业化工程**：统一配置、服务化、容错降级、可观测、可回放、可复现。

本改造文档会让你具备可讲述的“王炸链路”：

> **Intent Agent（工具调用） → Multi-Recall（FAISS/Milvus + RecBole） → Multi-Objective Rank（MMoE/CTR+到访） → RL/GRPO Planner（veRL）+ OR-Tools Repair → DPO/GRPO 文案对齐 → 评测与数据飞轮闭环**

------

# 1. 现状基线复盘（基于你当前项目真实链路）

你当前 GoAfar 的**实际运行链路**（不是宣传口径）如下：

- 语义召回：BGE-M3 向量检索（`src/embedding`）
- 候选融合：语义 + 流行度（`candidate_merger.py`）
- 意图理解与重排序：模板为主，可切换 LLM（`src/llm4rec`）
- 路线规划：OR-Tools VRPTW（`src/routing`）
- 文案生成：模板优先，LLM 可选（`src/content_generation`）
  三入口分叉：`main.py` / `run_with_llm.py` / `app.py`。

### 当前“卡大厂门槛”的关键问题（你文档里点得很准）

- **P0：硬编码路径 + 三入口流程分叉 + 强依赖离线向量产物**（缺少统一初始化/自愈）
- **P1：RecBole 训练存在但未真实进入在线召回**（在线仍用“流行度近似”）
- **P2：时间矩阵 Haversine 假设固定速度**，与真实路网偏差大

这些问题不解决，面试官会认为你是“脚本拼接的 Demo”。所以我们先做工程收敛，再做算法拔高，然后把评测和数据飞轮补齐。

------

# 2. 目标架构（全链路重构后的系统蓝图）

## 2.1 目标：从“脚本流水线”升级为“可训练、可评测、可上线的决策系统”

最终架构建议采用**“模块化单体 + 可选微服务拆分”**两用形态：

- 本地开发：单体（一个 FastAPI 服务），跑通全链路
- 线上/展示：拆成 4~6 个服务（更像大厂真实架构）

FastAPI 是面向生产的高性能 Python API 框架，基于类型提示并自动生成 OpenAPI 文档，适合你做“统一入口服务编排”。([FastAPI](https://fastapi.tiangolo.com/))

### 推荐的逻辑服务拆分

1. **API Gateway（FastAPI）**

- 统一入口：`/recommend/itinerary`
- 鉴权、限流、Tracing、AB 参数透传
- 编排召回/排序/规划/生成

1. **Recall Service（召回服务）**

- FAISS（单机）/ Milvus（分布式）向量检索
- RecBole 行为召回（SASRec/LightGCN）
- Geo 召回（GeoHash/半径过滤）
- RRF/加权融合，输出候选集 TopN

FAISS 是用于**大规模向量相似检索与聚类**的经典库，支持 HNSW 等索引结构，适合替换你现在的 `.npy` 暴搜。([GitHub](https://github.com/facebookresearch/faiss))
Milvus 是面向可扩展 ANN 搜索的向量数据库，支持向量 + 标量过滤与混合检索，适合工业化落地。([GitHub](https://github.com/milvus-io/milvus))

1. **Rank Service（排序服务）**

- 精排：多目标（CTR/到访/停留时长/满意度）
- 可选 LLM Listwise Rerank（Top30→Top10）

1. **Planner Service（规划服务）**

- 路网时间矩阵：OSRM Table Service（批量矩阵）+ 缓存
- 约束优化：OR-Tools VRPTW
- 强化学习规划：veRL + GRPO（训练一个“行程规划策略”）
- Hybrid：RL 给初解 / OR-Tools 做修复与局部搜索

OSRM 提供 Routing API（包含 table service 用来构建时间矩阵），更接近真实道路时间。([OSRM](https://project-osrm.org/docs/v5.5.1/api/))
OR-Tools 官方提供 VRPTW（含时间窗）示例与约束建模流程。([Google for Developers](https://developers.google.com/optimization/routing/vrptw))

1. **LLM Service（推理服务）**

- vLLM 启动 OpenAI-compatible server
- 统一给“意图解析 / LLM 重排 / 文案生成 / Judge 打分”提供高并发推理

vLLM 提供 OpenAI 兼容 HTTP Server，可通过 `vllm serve` 启动，适合高并发推理服务化。([vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/))
Ollama 作为本地轻量方案可用于开发/演示环境。([Ollama 文档](https://docs.ollama.com/))

1. **Offline Pipeline（离线训练/评测）**

- 数据清洗、样本构造、训练、离线评测、模型注册
- 重点：把你现有 `train_dpo.py/make_prefs.py` 变成“可持续迭代的训练流水线”

------

# 3. P0 工程收敛：统一配置、统一入口、统一契约（这是“上岸线”）

> 这一章是你“从 Demo → 工业级”的关键。面试官会非常在意这些。

## 3.1 全量配置化：Hydra + OmegaConf（彻底消灭硬编码）

你文档里 P0 的“绝对路径写死”必须彻底清除。
建议引入 Hydra 做配置组合，OmegaConf 做层级配置与合并（文件/CLI/env），并支持 Structured Config 类型安全。([Hydra](https://hydra.cc/docs/intro/))figs/
runtime.yaml
data.yaml
recall.yaml
rank.yaml
planner.yaml
llm.yaml
eval.yaml

```
### runtime.yaml 示例（关键）
- 数据根目录  
- 模型名与权重路径（允许:contentReference[oaicite:18]{index=18}地路径）  
- 服务地址（vLLM、Milvus、OSRM）  
- TopK、Batch、超参  
- AB 实验开关（ranker版本、planner版本、reranker版本）

## 3.2 统一入口：从“三入口分叉”到“一个 Pipeline + 多前端”
你当前 `main.py/run_with_llm.py/app.py` 分叉，容易出现行为不一致。  
改造方式：

- 新建：`src/service/pipeline.py`（唯一业务编排）  
- CLI、WebUI、API 都调用 pipeline  
- `app.py`（Gradio）只负责前端展示，不再包含业务逻辑

## 3.3 统一 I/O 契约：Pydantic Schema + Contract Test
你基线文档已经明确“模块输入输出契约”，但代码层缺少统一 schema。  
做法：

- `src/schemas/` 定义 Request/Response（Pydantic）  
- 对每个服:contentReference[oaicite:21]{index=21}sts）：
  - “同一个输入 → 必须返回同字段同类型”
  - “缺失向量文件 → 必须自愈/降级并返回可解释错误”

## 3.4 自愈机制：启动时检查与自动构建（解决“离线产物缺失即不可用”）
你文档指出：`poi_emb.npy/poi_meta.csv` 缺失会直接失败。  
改造为：

- 启动时（或首次请求时）检测索引是否存在  
- 不存在则:contentReference[oaicite:23]{index=23}）：
  - 构建 embedding
  - 建 FAISS index 或写入 Milvus
- 在构建完成前：
  - 允许降级走“流行度 + Geo 召回”
  - 或返回“系统初始化中”的结构化响应（而不是崩溃）

---

# 4. 召回体系升级：多路召回漏斗（工业级推荐第一道门）

## 4.1 用 FAISS/Milvus 替换 `.npy` 暴力检索（必要升级）
现状：你是 `.npy` + numpy 相似度，规模一大就会崩，且不:contentReference[oaicite:24]{index=24}  
升级：  
- 开发/单机：FAISS（HNSW / IVF-PQ）:contentReference[oaicite:26]{index=26}  
- 生产/分布式：Milvus（向量检索 + metadata filter + hybrid search）:contentReference[oaicite:27]{index=27}  

## 4.2 利用 BGE-M3 的“稠密 + 稀疏”做 Hybrid Retrieval（比纯 embedding 强）
BGE-M3 支持 embedding 检索和稀疏检索（可类比 BM25 token weights），适合做 hybrid retrieval。:contentReference[oaicite:28]{index=28}:contentReference[oaicite:29]{index=29}级后你可以做：

- `dense_score = cos(q_emb, poi_emb)`
- `sparse_score = dot(q_sparse, poi_sparse)`（来自 BGE-M3 的 token weights）
- `hybrid_score = α*dense + (1-α)*sparse`
- 然后再接 reranker（cross-encoder / LLM）

## 4.3 多路召回（Multi-Recall）标准化（你要把 RecBole“扶正”）
你文档指出：RecBole 脚本存在，但在线召回仍用流行度近似。  
这必须改造成工业漏斗的“第二路召回”。

RecBole 是统一的推荐算法框架，覆盖序列推荐/通用推荐等，并提供标准评测协议。:contentReference[oaicite:31]{index=31}  

### 4.3.1 三条召回路（面试可讲“召回漏斗”）
1) **语义召回（Semantic Recall）**  
- 输入：query + 意图结构化信息（城市/时间/主题/人群）  
- 输出：TopK POI（带语义分数）

2) **行为召回（Behavior Recall，RecBole）**  
- 模型：SASRec（序列）/ LightGCN（图）  
- 输入：用户历史点击/收藏/到访序列（`user_events.csv`）  
- 输出：个性化 TopK 候选  
- 关键：要把 user_id 的冷启动策略讲清楚（见后文）

3) **地:contentReference[oaicite:33]{index=33} 用 GeoHash / 半径过滤（先过滤再语义/行为排序）
- 解决“语义相似但太远”的问题

### 4.3.2 候选融合：从“规则相加”升级为 RRF + 分路校准
你现在是语义+流行度简单融合。  
工业实践更常用 **RRF（Reciprocal Rank Fusion）** 或加权融合，优点是：
- 各路召回分数不可比时仍稳定
- 易做消融与 AB

公式（RRF）：
- `score(item)= Σ_r 1/(k + rank_r(item))`

并对不同召回路做 **Calibration**（如温度缩放/分位数归一化），让融合更稳定。

## 4.4 冷启动与召回策略（必须准备面试追问）
- 冷启动用户：用“城市:contentReference[oaicite:35]{index=35}Geo/热门  
- 半冷启动：最近 3~5 个行为加大权重做序列召回  
- 热用户：行为召回为主，语义召回补长尾探索

---

# 5. 排序体系升级：多目标精排 + LLM Listwise Rerank（指标提升主战场）

## 5.1 精排模型：从“规则重排”升级为“可训练多目标排序”
你当前重排以模板规则为主。:contentReference[oaicite:37]{index=37}如何定义 label  
- 如何做负采样  
- 如何处理多目标与约束  
- 如何评测 AUC/NDCG/Calibration

### 5.1.1 建议模型路线（按性价比）
- **MVP**：LightGBM / Logistic Regression（快速建立强 baseline）  
- **深度精排**：DIN / DIEN（兴趣网络）  
- **多任务**：MMoE（CTR + 到访 + 停留）——非常适合讲 Trade-off

### 5.1.2 特征工程（你必须能讲“特征与泄露”）
- 用户侧：  
  - 最近 N 次行为序列（poi_id、category、time gap）  
  - 城市偏好、主题偏好向量  
- POI 侧：  
  - 类别、价格/热度、开放时间窗、平均停留时长、文本 embedding  
- 上下文：  
  - 出行日期、天气、同行人数:contentReference[oaicite:38]{index=38}：严格做时间切分**（用 timestamp 切 train/val/test）避免 label leakage。

## 5.2 LLM Listwise Rerank：把“连贯性/逻辑性”变成可解释优势
在精排输出 Top-30 后，让 LLM 做 listwise rerank（考虑景点间搭配逻辑、节奏）。  
学术上 listwise LLM reranking 已被系统化研究，如 RankZephyr、FIRST 等工作表明 listwise rerank 在效果上有明显优势空间。:contentReference[oaicite:39]{index=39}  

工程落地策略（避免慢）：
- 只对 Top-30 做一次 listwise  
- 严格 token budget（候选以结构化 JSON 压缩）  
- 缓存：相同 query+候选集 hash 命中则复用  
- 降级：LLM 不可用 → 退回精排结果

---

# 6. 规划层升级：真实路网时间 + RL/GRPO 规划策略 + OR-Tools 修复（最“含金量”的部分）

你当前用 Haversine + 固定速度构时间矩阵、然后 OR-Tools VRPTW 求解。  
这是好基线，但要“冲大厂算法岗”，你需要把它升级为：

> **（1）真实路网时间（OSRM）**  
> **（2）学习型规划策略（GRPO/RL）**  
> **（3）混合求解（RL 初解 + OR-Tools 局部搜索修复）**  
> **（4）多目标奖励（推荐满意度 + 行程可行性 + 路程成本）**

## 6.1 时间矩阵：用 OSRM Table Service 替代 Haversine（P2→P0）
OSRM 的 table service 可以批量构建时间矩阵，更贴近实际路况与路网。:contentReference[oaicite:41]{index=41}  
设计：
- `TimeMatrixProvider` 接口：
  - `OSRMTimeMatrixProvider`
  - `HaversineFallbackProvider`
- 缓存策略：
  - key = `(city, date_bucket, candidate_set_hash)`
  - 缓:contentReference[oaicite:42]{index=42}拆分：
  - N>200 时分块请求 table service（避免 URL 太长/超时）

## 6.2 约束优化：保留 OR-Tools VRPTW 作为“硬约束修复器”
OR-Tools VRPTW 的价值：硬约束建模成熟、可保证可行性。:contentReference[oaicite:43]{index=43}  
但它的短板（你可在面试讲）：
- 规模大时求解耗时不稳定  
- 难把“个性化推荐得分”自然融合为目标（通常需要手工加权）  
- 难在线端到端做“体验最优”的多目标权衡

因此我们引入 RL 作为“学到的启发式”，快速给高质量初解，再用 OR-Tools 修复。

---

# 7. 强化学习升级：用 veRL 跑 GRPO（让项目直接具备“大模型算法岗”含金量）

你要求“库用 verl，RL 用 GRPO 或更好的”，这里给你一个非常面试友好的落地方式：  
**把“行程规划”做成一个可学习的策略模型（LLM/Transformer 生成 POI 序列），用 GRPO 在 veRL 框架里训练。**

## 7.1 为什么 veRL（字节系 + 工业级 RLHF 框架，面试官熟）
veRL 是 ByteDance Seed 发起并维护的 LLM 强化学习训练库，强调灵活高效、可接入 vLLM 等推理引擎，并能用少量代码搭建 GRPO/PPO 等 RL 数据流。:contentReference[oaicite:44]{index=44}  
同时 veRL 的 GRPO 文档明确给出了**组采样 n>1、adv_estimator=grpo、KL loss 等关键配置**。:contentReference[oaicite:45]{index=45}  

> 你在面试可以说：我选择 veRL 是因为它的 HybridFlow 编程模型把控制流与计算流解耦，便于快速试验新 RL 算法且具备工程吞吐能力。:contentReference[oaicite:46]{index=46}

## 7.2 为什么 GRPO（对 LLM/生成式策略特别合适）
GRPO（Group Relative Policy Optimization）在 DeepSeekMath 中提出，是 PPO 的变体：**用组内相对奖励构造 advantage，避免训练 critic，从而降低内存/复杂度**，并被用于提升推理能力。:contentReference[oaicite:47]{index=47}  
veRL 也提供 GRPO 的配置与扩展（例如 DrGRPO）。:contentReference[oaicite:48]{index=48}  

---

# 8. 把“旅游规划”建模成 GRPO 可训问题（你项目的核心创新点）

## 8.1 任务形式：Planner Policy 生成“POI 序列 + 时间安排”
输入（Prompt/State）包含：
- 用户意图结构化信息（城市、天数、主题、预算、出行时间）
- Top-N 候选 POI（来自召回+精排）
- 每个 POI 的结构化属性（lat/lon、open/close、stay、category、rank_score）
- 路网时间矩阵（可压缩：只给近邻 topM 或给 distance-to-current）
- 当前时间/剩余时间/已选 POI 集合

输出（Action/Trajectory）包含：
- 下一步选择哪个 POI（可用 poi_id token 或 index token）
- 可选：到达时间/停留时长（也可由规则/OR-Tools后处理）

## 8.2 关键：动作空间与约束（避免 RL 学成“瞎走”）
### 8.2.1 Action Mask（硬约束优先）
在每一步对候选 POI 做可行性筛选：
- 到达后是否在 open-close 时间窗内  
- 加上 stay 是否超出当天结束  
- 地理距离是否超过阈值（可选）

mask 后的动作集合更小，训练更稳，生成更可用。

### 8.2.2 Reward 设计（多目标，且可解释）
建议拆成可解释的加权奖励（面试官会追问每项权重怎么来）：

1) **个性化满意度奖励**（来自精排分数/预测到访率）
- `r_pref = f(rank_score)`（如归一化到 [0,1]）

2) **可行性奖励**（时间窗/预算/天数约束）
- 违反时间窗：`-λ_tw`  
- 超时：`-λ_over`  
- 不可达/冲突：`-λ_infeasible`

3) **行程成本惩罚**
- 路程时间：`-λ_travel * travel_minutes`
- 过度绕路：`-λ_detour`

4) **多样性与覆盖**
- 类别多样性：Shannon entropy 或覆盖率奖励  
- 避免全是“同质景点”

5) **节奏合理性（可选）**
- 上午/下午活动强度均衡  
- “景点 → 餐饮 → 休息”的链路

> 面试表达方式：我把规划问题从单目标的“最短路/最可行”扩展为多目标优化，并通过 reward 分解把业务目标可解释化。

## 8.3 训练数据怎么来（你给的要求：可合成 + 可搜集）
你基线数据有：`poi.csv`（1333 POI）和 `user_events.csv`（38579 行为）。  
我们把它扩展成“可训练”数据集：

### 8.3.1 从现有 `user_events.csv` 合成三类数据
1) **规划轨迹（trajectory）**  
- 对每个 user_id，按 timestamp 排序，取一天或多天窗口形成“真实游玩序列”
- 作为“专家演示”或 SFT 数据（先训一个能输出合理序列的 planner）

2) **偏好对比数据（preference pairs）**  
- Chosen：真实序列/高满意度序列  
- Rejected：扰动序列（shuffle、替换远距离 POI、插入闭门 POI）  
→ 供 DPO 或 RM 训练

3) **RL prompt 数据（state prompts）**  
- 从真实序列中截断生成“中间状态”训练样本，让策略学会在中途也能做决策

### 8.3.2 引入公开数据（让项目更“像大厂真实数据”）
你可以把 GoAfar 作为“可插拔数据源”的引擎，额外支持：
- **Gowalla check-in**：Stanford SNAP 提供包含大量签到数据与社交边。:contentReference[oaicite:50]{index=50}  
- **GeoLife 轨迹数据**：Microsoft Research 发布的 GPS 轨迹数据，包含多年轨迹与活动类型，可用于构造“出行路径真实分布”。:contentReference[oaicite:51]{index=51}  
- **Yelp Dataset Challenge 数据**：包含商家位置、评论等，可构造 POI 语料与偏好信号。:contentReference[oaicite:52]{index=52}  
- **Solomon VRPTW Benchmarks**：用于验证你的规划器在标准 VRPTW 上的泛化（非常加分）。:contentReference[oaicite:53]{index=53}  

> 面试官会喜欢你说：我在“业务数据（旅游 POI）”之外，引入了标准 OR 基准（Solomon VRPTW）做可复现对比，并用公开轨迹数据做分布鲁棒性验证。

---

# 9. veRL + GRPO 的训练落地方案（你可以照着实现）

## 9.1 两阶段训练（非常推荐，稳定且可讲）
### 阶段 1：SFT（让模型先学会“输出格式 + 基本可行性”）
:contentReference[oaicite:54]{index=54}sv` 合成的“真实序列示例”  
- 目标：输出规范 JSON（路线 + 时间安排 + 解释）  
- 这一步避免 RL 从随机策略开始，提升稳定性

### 阶段 2：GRPO（让模型在多目标 reward 下优化）
- 框架：veRL（推荐）
- 算法：GRPO（group sampling n>1、adv_estimator=grpo、KL loss）:contentReference[oaicite:56]{index=56}  
- 重要：veRL 文档提到 GRPO 与 PPO 类似但**无 critic**，且 KL 正则可直接加在 loss 上。:contentReference[oaicite:57]{index=57}  

## 9.2 veRL 训练关键配置（面试官会问你怎么设）
从 veRL 的 GRPO 文档抽象成你项目的配置要点：:contentReference[oaicite:58]{index=58}  
- `actor_rollout.ref.rollout.n`：每个 prompt 采样 n 次（GRPO 必须 >1）  
- `algorithm.adv_estimator=grpo`  
- `actor_rollout_ref.actor.clip_ratio`：PPO/GRPO clip  
- `actor_rollout_ref.actor.use_kl_loss=True`：GRPO 用 KL loss（不是在 reward 里加 KL）  
- `actor_rollout_ref.actor.loss_agg_mode`：长 CoT 场景建议 token-mean 更稳（文档有说明）:contentReference[oaicite:59]{index=59}  

**加分项（高级扩展）**：  
veRL 文档提供 DrGRPO 作为扩展以缓解长度偏置问题。:contentReference[oaicite:60]{index=60}  
你可以把它作为实验分支写进消融里（“GRPO vs DrGRPO”）。

## 9.3 RewardManager 设计（把业务 reward 工程化）
veRL 的 HybridFlow 训练方式强调你可以在 `RewardManager` 里组合规则 reward 与模型 reward。:contentReference[oaicite:61]{index=61}  
在 GoAfar 中建议：
- rule-based：时间窗可行性、超时惩罚、距离惩罚  
- model-based：精排模型给的满意度、LLM Judge 给的“行程合理性/文案质量”分  
- 产出 token-level reward（或 seq-level 再 broadcast）

---

# 10. DPO/GRPO 文案与解释对齐（把你已有 DPO 链路“纳入主干”）

你当前有 `train_dpo.py` 和 `make_prefs.py`，但未与线上策略自动打通。  
这在:contentReference[oaicite:63]{index=63}上线？怎么选择？怎么回滚？

## 10.1 DPO：作为“低成本稳定对齐”的基线
DPO 的核心价值：把 RLHF 的优化转成对偏好数据的分类式训练，稳定且工程简单。:contentReference[oaicite:64]{index=64}  
TRL 提供 DPOTrainer 文档与实现，适合你快速把偏好对齐做成可复现流水线。:contentReference[oaicite:65]{index=65}  

### DPO 数据构造（复用你现有 user_events）
- Chosen：用户最终采纳的行程文案/解释  
- Rejected：用户没有采纳的版本（或扰动版本）  
- 同时用 LLM 合成“困难负样本”（比如看似合理但违反时间窗/太远）

## 10.2 GRPO 用在“带硬约束的生成”上（更能展示算法深度）
如果你希望把“文案必须不胡说 + 必须符合时间窗/路网”做到更强，可以把 GRPO 用在：
- 输出中出现违反约束的内容直接扣分  
- 输出结构错误扣分  
- 解释不一致扣分（“去了 A 却解释 B”）

GRPO 的论文背景与优势可引用 DeepSeekMath 的描述（提升推理并优化 PPO 内存）。:contentReference[oaicite:66]{index=66}  

---

# 11. 评测体系：从“能跑”到“能证明变好”（必做，否则无法面试）

你已经有 `src/evaluation/metrics.py`，但需要扩展成完整指标矩阵。  

## 11.1 推荐侧指标（离线）
- Recall@K（各召回路、融合后）  
- NDCG@K / HitRate@K（用真实行为序列做 label）  
- 多目标：CTR AUC、Visit AUC、Calibration（ECE）

**必须做消融**：
- 仅语义召回 vs +行为召回 vs +Geo vs +精排 vs +LLM rerank

## 11.2 规划侧指标（离线）
- 可行率（满足 time window 的比例）  
- 违约率（时间窗违背、超时）  
- 总路程时间/距离  
- POI 覆盖数  
- 平均满意度（精排分数之和/均值）  
- 求解耗时（P50/P95，线上很关键）

**对比组要齐全**：
- OR-Tools only（基线）:contentReference[oaicite:68]{index=68}  
- RL/GRPO only  
- RL/GRPO → OR-Tools repair（Hybrid）

## 11.3 生成侧指标（离线 + LLM-as-a-Judge）
- Win-rate：DPO/GRPO 后文案胜率  
- 事实一致性：行程 JSON 与文案描述一致率  
- 结构合规率：输出能否被 parser 正确解析  
- token cost、延迟、失败降级率

---

# 12. 在线链路与可观测性（让项目“像线上系统”）

## 12.1 FastAPI 服务规范
- `/healthz`：健康检查  
- `/readyz`：依赖检查（Milvus/FAISS index、vLLM、OSRM）  
- `/v1/recommend/itinerary`：主接口  
- `/v1/eval/offline`：离线评测触发（仅开发）  
- `/v1/admin/rebuild_index`：重建索引（鉴权）

FastAPI 支持 OpenAPI/JSON Schema，接口规范化非常适合做“可演示的工业服务”。:contentReference[oaicite:69]{index=69}:contentReference[oaicite:70]{index=70}vLLM 推理服务
- 统一 LLM 推理通过 vLLM 的 OpenAI-compatible server  
- 线上服务只依赖 HTTP，不把大模型绑死在主进程  
vLLM 官方文档明确提供 OpenAI-compatible server 的启动与调用方式。:contentReference[oaicite:71]{index=71}  

## 12.3 缓存与降级（面试官会问“线上坏了怎么办”）
- LLM 不可用：降级到模板意图解析/规则重排/模板文案  
- OSRM 不可用：降级 Haversine  
- 向量库不可用：降级热门+Geo  
- 每次降级必须打日志并计数（SLA）

---

# 13. 代码结构建议（让你仓库一眼“工业化”）

> 下面这个目录结构是面试官打开仓库时的第一印象。
```

src/
api/
server.py
routes/
itinerary.py
service/
pipeline.py
dependencies.py
recall/
providers/
semantic_faiss.py
semantic_milvus.py
recbole_provider.py
geo_provider.py
fusion/
rrf.py
calibrate.py
rank/
features/
models/
train.py
infer.py
llm_lisnner/
time_matrix/
osrm_provider.py
haversine_provider.py
cache.py
ortools/
vrptw_solver.py
rl/
env.py
reward.py
policy_format.py
grpo_train/
configs/
dataset_builder.py
run.sh
hybrid_repair.py
llm/
clients/
openai_compatible.py
prompts/
data/
schema.py
build_dataset.py
evaluation/
metrics_recall.py
metrics_rank.py
metrics_plan.py
metrics_gen.py
ablation.py
configs/
...
tests/
contract/
integration/
docker/
docker-compose.yml

```
---

# 14. “面试可讲”的亮点清单（建议你直接写进简历）

你要准备 6 条“可追问”的简历 bullet（每条都能展开讲 10 分钟）：

1) **多路召回漏斗**：BGE-M3 hybrid retrieval + RecBole 行为召回 + Geo 召回，RRF 融合并做分路校准，离线对比 Recall/NDCG 提升。:contentReference[oaicite:74]{index=74}  
2) **多目标精排**：构建 CTR/到访/停留多任务模型（MMoE），引入上下文特征与时间切分防泄露。  
3) **LLM Listwise Rerank**：对 Top-30 候选做 listwise 重排，提升行程连贯性与可解释性。:contentReference[oaicite:75]{index=75}  
4) **RL/GRPO 规划策略（veRL）**：将行程规划建模为生成式策略优化，使用 GRPO（组采样、无 critic、KL loss）训练 planner，并用 OR-Tools 修复硬约束，显著改善可行率与求解耗时稳定性。:contentReference[oaicite:76]{index=76}  
5) **偏好对齐闭环**：用 TRL DPOTrainer 对文案/解释做偏好对齐，并把训练产物接入线上策略选择器，实现可回滚 AB。:contentReference[oaicite:77]{index=77}  
6) **工程化与可观测**：FastAPI 统一入口:contentReference[oaicite:78]{index=78}推理服务化 + OSRM 路网时间 + 缓存降级 + 契约/集成测试，系统可复现可演示。:contentReference[oaicite:79]{index=79}  

---

# 15. 落地执行清单（不讲“几天”，只讲“按优先级闭环”）

## Phase 1：工程收敛（先把系统“拉齐到可上线”）
- [ ] Hydra/OmegaConf 全量配置化，清除硬编码  
- [ ] 统一 pipeline（合并三入口逻辑）  
- [ ] 启动自检与自愈（向量/索引缺失自动构建）  
- [ ] 契约测试 + 最小集成测试

## Phase 2：召回与排序（把“RecSys 漏斗”做完整）
- [ ] FAISS/Milvus 向量检索替换 `.npy` 暴搜:contentReference[oaicite:83]{index=83}  
- [ ] RecBole 行为召回真实接入在线融合:contentReference[oaicite:84]{index=84}  
- [ ] 精排模型上线（先轻量、后深度）  
- [ ] LLM listwise rerank 作为可选增强:contentReference[oaicite:85]{index=85}  

## Phase 3：规划与 RL/GRPO（冲刺“算法含金量”）
- [ ] OSRM 路网时间矩阵 + 缓存:contentReference[oaicite:86]{index=86}  
- [ ] veRL + GRPO 训练 planner policy（SFT→GRPO）:contentReference[oaicite:87]{index=87}  
- [ ] Hybrid：RL 初解 + OR-Tools 修复:contentReference[oaicite:88]{index=88}  
- [ ] 规划指标矩阵 + 消融报告

## Phase 4：对齐与数据飞轮（把项目变成“持续迭代的系统”）
- [ ] DPO/GRPO 文案对齐 + 线上策略选择器:contentReference[oaicite:89]{index=89}  
- [ ] 反馈日志 → 偏好数据自动构造 → 定期训练 → 灰度上线

---

## 你接下来拿去就能用的“面试叙事主线”
你可以按这个顺序讲项目（非常像字节/阿里/腾讯的内部文档风格）：

1) **问题定义**：旅游推荐不是“找相似”，而是“多目标约束决策”。  
2) **漏斗架构**：多路召回→精排→规划→生成。  
3) **关键创新**：把规划策略做成可学习（GRPO），并用 OR-Tools 保硬约束。  
4) **工程化保障**：统一服务、配置治理、可观测、可回放、可降级。  
5) **数据飞轮**：日志→偏好→对齐→上线→再收集。

---

如果你愿意，我可以在你这份“改造蓝图”的基础上，继续把**最关键的三块“面试官最爱追问的代码骨架”**（不需要你给我仓库，我可以按你基线模块名写成可直接套进去的骨架）一次性给你：
1) `FastAPI + pipeline.py` 的统一编排层（含 schema、DI、降级与 tracing 入口）  
2) `FAISS/Milvus` 双实现的 `SemanticRecallProvider` + RRF 融合  
3) `veRL GRPO` 的 GoAfar Planner 训练工程（dataset_builder + RewardManager + 评测脚本）:contentReference[oaicite:90]{index=90}:contentReference[oaicite:91]{index=91}:contentReference[oaicite:92]{index=92}
::contentReference[oaicite:93]{index=93}
```