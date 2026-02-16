# GoAfar 项目探索与实验记录

**日期**: 2026-02-15
**环境**: autodl RTX 5090 32GB, conda goafar环境

## 项目概述

GoAfar是一个智能旅行路线推荐系统，采用多模态AI架构：

### 核心组件
1. **API层** (`src/api/`): FastAPI REST服务
2. **服务层** (`src/service/`): RecommendationPipeline主流程
3. **召回模块** (`src/recommendation/`): 多路召回(语义/行为/地理)
4. **排序模块** (`src/ranking/`): MMoE深度排序模型
5. **重排模块** (`src/reranking/`): Qwen3-Reranker精排
6. **路由规划** (`src/routing/`): VRPTW求解器
7. **内容生成** (`src/content_generation/`): SFT/DPO训练
8. **强化学习** (`src/rl/`): GRPO路线规划训练
9. **评估系统** (`src/evaluation/`): 多维度指标

### 数据状况
- **POI数据**: `data/poi.csv` (1333个POI), `data/all/poi_expanded.csv` (12万+POI)
- **用户行为**: `data/user_events.csv` (38579条事件)
- **训练数据**: `outputs/datasets/` 包含SFT/DPO/GRPO训练数据

### 模型文件
- **Qwen3-8B**: LLM模型 (意图理解、内容生成)
- **Qwen3-Embedding-4B**: 语义嵌入模型
- **Qwen3-Reranker-4B**: 重排序模型

## 已完成的工作

### 1. 嵌入向量构建
- ✅ 使用Qwen3-Embedding-4B成功构建POI嵌入
- ✅ 输出: 1333个POI, 每个2560维向量
- ✅ 保存位置: `outputs/emb/poi_emb.npy`, `outputs/emb/poi_meta.csv`
- 📝 脚本: `scripts/build_qwen3_embeddings.py`

### 2. 全链路实验
- ✅ 运行完整pipeline实验
- ✅ 测试数据准备、pipeline初始化、多组件协调、性能测试
- ✅ 生成实验报告: `docs/FULL_PIPELINE_EXPERIMENT_REPORT.md`
- ⚠️ 发现问题:
  - 语义召回不可用(嵌入模型加载问题)
  - 模型路径环境变量解析错误(`${GOAFAR_...}`未展开)
  - OSRM不可用，回退到Haversine距离

### 3. MMoE排序模型训练准备
- ✅ 修复多个代码bug
- ✅ 创建quick preset配置适配小模型
- ⚠️ 训练过程中发现验证集为空问题(已添加fallback)

## 代码修复记录

### 修复的问题

1. **FlagEmbedding导入问题** (`src/embedding/__init__.py`, `vector_builder.py`)
   - 问题: FlagEmbedding与transformers版本不兼容
   - 解决: 添加try-except导入，使其可选

2. **PyTorch导入错误** (`src/ranking/deep_ranker.py:144`)
   - 问题: `import self.torch.nn as nn`语法错误
   - 修复: `import torch.nn as nn`

3. **MMoE内部类变量捕获** (`src/ranking/deep_ranker.py`)
   - 问题: 内部类中使用`self.nn`和`self.torch`导致AttributeError
   - 解决: 使用闭包捕获`nn`和`torch`

4. **配置字段缺失** (`src/service/config.py`)
   - 问题: YAML配置中有`batch_size`等字段但dataclass中缺少
   - 解决: 添加相关字段

5. **维度不匹配** (`src/ranking/deep_ranker.py`)
   - 问题: seq_repr维度与seq_embed_dim不一致
   - 解决: 调整preset使seq_embed_dim=item_embed_dim

## 待完成任务

### 1. MMoE排序模型训练
- 状态: 模型可前向传播，验证集问题已修复
- 待运行: 完整训练

### 2. GNN模型训练
- 状态: 未开始
- 脚本: `scripts/train_gnn_model.py`

### 3. GRPO强化学习训练
- 状态: 未开始
- 脚本: `scripts/train_grpo.sh`

## 未解决的问题

### 高优先级

1. **环境变量展开问题**
   - 症状: 模型路径显示为`${GOAFAR_LLM_MODEL:models/Qwen3-8B}`

2. **语义召回不可用**
   - 症状: `NoneType object is not callable`

### 中优先级

3. **大嵌入向量构建**
   - 需求: 使用`data/all/poi_expanded.csv`(12万POI)
   - 脚本: `scripts/build_qwen3_embeddings_large.py`

4. **依赖版本冲突**
   - FlagEmbedding与transformers版本不兼容

## 配置说明

### 运行环境
```bash
conda activate goafar
```

### GPU
- NVIDIA GeForce RTX 5090
- 32GB显存

## 训练命令

### MMoE排序模型
```bash
python scripts/train_mmoe_ranker.py --preset quick --device cuda
```

### GNN模型
```bash
bash scripts/train_gnn.sh
```

### GRPO强化学习
```bash
bash scripts/train_grpo.sh
```

---

## 2026-02-15 对话追加记录（全面探索 + 代码审查）

### 本次目标与范围
- 按用户要求完成三件事：
  1. 全面探索项目现状（目录、入口、配置、数据资产、测试现状）。
  2. 审查 `context/log.md`、`plan/` 下文档，并对现有代码做全面 review。
  3. 明确核查 `data/all` 和 `outputs/datasets` 是否真正用于当前代码链路（强调“这是最新数据”）。

### 扫描与核查过程（关键事实）
- 仓库结构核查：
  - `src/` 共 84 个 Python 文件，模块完整覆盖 API/召回/排序/重排/规划/RL/评估。
  - 入口存在 `main.py`、`run_with_llm.py`、`app.py`，以及统一编排 `src/service/pipeline.py`。
- 文档核查：
  - 已完整阅读 `context/log.md`、`plan/data_prepare.md`、`plan/improve.md`、`plan/plan1.md`、`plan/plan2.md`。
  - 文档中提出的关键目标（统一配置、RecBole接入、使用12万POI、训练链路完善）与当前代码状态存在落差（见下文问题）。
- 数据资产核查（已使用最新数据进行一致性检查）：
  - `data/all/poi_expanded.csv`: 127,978 行（含表头，实际 127,977 条 POI）。
  - `data/all/user_events.csv`: 38,580 行（含表头）。
  - `outputs/datasets` 存在：`planner_trajectories.jsonl`、`planner_rl_prompts.jsonl`、`grpo_planner_prompts.jsonl`、`sft_data.jsonl`、`dpo_prefs.csv` 等。
  - 关键一致性结论：
    - `outputs/datasets` 中轨迹/RL/SFT样本涉及 2,588 个唯一 POI ID。
    - 其中有 2,134 个不在 `data/poi.csv`（小库 1333 POI）中。
    - 这 2,134 个也不在当前线上 embedding 元数据 `outputs/emb/poi_meta.csv`（1333条）中。
    - 但这些 ID 在 `data/all/poi_expanded.csv` 中是存在的。

### 在 goafar 环境的复核结果
- 用户补充“环境是 goafar 虚拟环境”后，已切换并复查。
- 环境关键依赖状态：
  - 已安装：`ortools`、`FlagEmbedding`、`peft`、`transformers`（版本 5.1.0）。
  - 缺失：`recbole`、`trl`、`mlflow`、`faiss`、`pytest`。
- 模型配置读取验证：
  - `models/Qwen3-8B`、`models/Qwen3-Embedding-4B`、`models/Qwen3-Reranker-4B` 均可被 `AutoConfig(..., trust_remote_code=True)` 识别为 `qwen3`。
- 实际 pipeline 运行（goafar 环境）：
  - 运行成功但出现语义召回降级日志：`NoneType object is not callable`。
  - 同时出现环境变量占位符未展开导致的路径/URL异常：
    - reranker 路径显示为 `${GOAFAR_RERANK_PATH:...}`
    - OSRM URL 被解析成非法字符串 `${goafar_osrm_url...}`，最终回退 Haversine。

### 决策记录（本轮执行策略）
- 决策 1：先“只审查不改代码”。
  - 原因：用户请求是“全面探索 + review”，优先给出完整问题清单与证据。
- 决策 2：在用户明确“goafar环境”后，以 goafar 环境结论为准，重新校验关键问题，避免环境误判。
- 决策 3：将问题按 P0/P1/P2 分级，先聚焦会直接影响线上正确性和数据有效性的缺陷。

### 关键假设（显式记录）
- 假设 1：用户希望 `data/all` 与 `outputs/datasets` 作为当前主数据源，而非仅离线实验用。
- 假设 2：`src/service/pipeline.py` 是线上主链路（CLI/Web/API 统一入口），其数据路径与配置行为需作为最高优先级修复对象。
- 假设 3：当前 `outputs/emb/poi_emb.npy + poi_meta.csv`（1333条）是旧产物，不能代表“最新全量数据可用”。
- 假设 4：由于 goafar 缺失 `recbole/trl/mlflow`，即使代码有框架，也无法完整执行对应训练/召回链路。

### 已确认问题（按优先级）

#### P0（阻断主链路正确性）
1. 配置占位符未展开，类型错误（字符串代替 bool/path）。
   - 位置：`src/service/config.py`（`load_runtime_config` 只 `yaml.safe_load`，未做 `${...}` 展开）。
   - 现象：`cfg.llm.enabled` 等字段为字符串；模型路径和 OSRM URL 解析异常。

2. 当前 pipeline 默认仍指向旧小数据集。
   - 位置：`configs/runtime.yaml` 中 `paths.poi_csv=data/poi.csv`、`paths.user_events_csv=data/user_events.csv`。
   - 影响：线上只覆盖 1333 POI，未使用 `data/all` 最新规模。

3. 最新训练数据与线上召回/向量基座断层。
   - 证据：`outputs/datasets` 中 2,588 个唯一 POI ID，2,134 个不在 `data/poi.csv` 与 `outputs/emb/poi_meta.csv`。
   - 影响：SFT/GRPO/DPO 使用的数据分布无法在当前线上候选集复现。

4. 语义召回在 goafar 实际降级。
   - 位置：`src/embedding/vector_builder.py` 中 `BGEM3Encoder` 可能为 `None`，但后续无保护直接调用。
   - 现象：运行日志报 `NoneType object is not callable`，语义召回失效。

5. 导入体系不统一（`src.*` 与 `sys.path` 方式混用）。
   - 位置：`src/service/pipeline.py` 采用 `from content_generation...` 顶层导入。
   - 现象：`import src.service.pipeline` 会报 `No module named content_generation`。
   - 影响：测试与运行入口表现不一致，易引入环境依赖型故障。

#### P1（功能有效性/可训练性问题）
6. RecBole在线召回未真正接入 pipeline 调用参数。
   - 位置：`src/service/pipeline.py` 调 `merge_candidates` 时未传 `use_recbole/recbole_model_path/...`。
   - 影响：行为召回长期退化为 popularity。

7. `config_loader` 环境变量覆盖策略对下划线键名不兼容。
   - 位置：`src/service/config_loader.py` 使用 `split("_")` 逐级嵌套。
   - 例：`GOAFAR_RERANK_USE_RERANKER_MODEL` 被解析成 `rerank.use.reranker.model`，而非 `rerank.use_reranker_model`。

8. GRPO 实现中 KL 参考策略构造不合理。
   - 位置：`src/rl/grpo_trainer.py`。
   - 问题：`ref_log_probs` 与 `current_log_probs` 由同一模型计算，KL约束形同弱化。

9. 训练脚本默认路径和线上路径不一致。
   - 现象：部分训练脚本仍默认 `data/poi.csv`、`data/user_events.csv`，与“latest data/all”目标冲突。

#### P2（工程化与测试质量问题）
10. A/B 统计模块边界行为存在逻辑缺陷。
   - 位置：`src/evaluation/ab_test.py`。
   - 现象：零方差样本下 t-test 显著性判断异常；浮点精度导致严格相等断言不稳。

11. 测试组织混用 unittest/pytest，部分文件被 pytest 误收集。
   - 位置：`tests/test_model_loading.py`（脚本风格函数被当作 pytest 测试，缺失 fixture）。

12. goafar 环境缺少 `pytest`，当前无法在目标环境复跑全量 pytest。

### 未解决问题（当前状态）
- 未修复配置展开与类型转换问题（P0，需先改配置加载统一入口）。
- 未完成 `data/all` 主链路切换与全量 embedding 产物落地（P0）。
- 未完成 RecBole 真正在线接线（P1，且环境中缺 recbole）。
- 未完成 GRPO/TRL 真实训练验证（P1，环境中缺 trl）。
- 未完成 MLflow 跟踪打通（P1，环境中缺 mlflow）。
- 未完成导入体系统一（P0/P1，影响可维护性和测试可用性）。

### 下一步行动（建议执行顺序）
1. **先修配置系统（P0）**
   - 统一 `src/service/pipeline.py` 使用 `config_loader`，确保 `${...}` 展开与布尔/数值类型正确。
   - 明确一个权威配置入口，避免 `config.py` 与 `config_loader.py` 双轨并存。

2. **切换主数据到 latest（P0）**
   - 将 runtime 默认路径切到 `data/all/poi_expanded.csv` 与 `data/all/user_events.csv`。
   - 对齐 `outputs/emb` 产物命名，保证 pipeline 实际读取的是全量 embedding。

3. **修语义召回降级根因（P0）**
   - 为 `BGEM3Encoder is None` 增加显式保护与可观测降级路径（不要抛 `NoneType`）。
   - 补充模型后备策略（Qwen3Embedding/BGE-M3二选一时的健壮逻辑）。

4. **打通行为召回接线（P1）**
   - 在 pipeline 调用 `merge_candidates` 时透传 `use_recbole` 相关参数。
   - 若环境无 recbole，明确降级到 popularity 并记录统一日志事件。

5. **修导入与测试体系（P1/P2）**
   - 统一包导入规范（支持 `src.*` 与脚本入口一致行为）。
   - 将脚本型测试与 pytest 用例拆分，避免误收集。

6. **环境补齐（P1）**
   - 在 goafar 环境补齐 `pytest`、`recbole`、`trl`、`mlflow`、`faiss`（按优先级）。

### 本轮产出状态
- 已完成：全面探索、计划文档审阅、代码审查、最新数据一致性核查、goafar 环境复核。
- 未进行：代码修改（本轮按 review-only 决策执行）。

---

## 2026-02-15 对话追加记录（review 后按最新问题直接修复）

### 本轮目标（用户要求）
- 用户要求两件事：
  1. 先理清项目现状；
  2. 基于 `context/log.md` 中“最新问题清单”做代码 review 并直接修改。
- 本轮执行策略：先做定点复核，再按 P0/P1 进行最小闭环修复，并补充基础验证。

### 关键决策（Decision Log）
1. **优先修 P0 主链路问题，不先扩展新功能**。
   - 原因：P0（配置展开/类型、数据源、语义召回降级）直接影响线上正确性。
2. **保留 `RuntimeConfig` dataclass API，但内部对齐 `config_loader`**。
   - 原因：避免一次性重构导致入口脚本/调用方大面积破坏，同时解决 `${...}` 未展开与类型错误。
3. **先打通 RecBole 参数透传，不强行绑定必须安装 recbole**。
   - 原因：环境存在依赖不完整风险，先让接线真实生效，再允许 provider 内部降级。
4. **导入兼容采用“低侵入方案”**。
   - 原因：项目里同时存在 `service.*` 与 `src.service.*` 两种使用方式，先让两者都可 import，后续再做全面统一。
5. **`data/all` 作为默认主数据源（latest）**。
   - 原因：与本日志此前“latest data”结论一致，减少默认行为与目标不一致。

### 关键假设（Assumptions）
1. 当前线上/主流程以 `src/service/pipeline.py` 为核心入口（CLI/Web/API 最终汇聚）。
2. 用户希望“默认即 latest”，而不是仅在少量脚本中手工传参切到 `data/all`。
3. 配置项 `${...}` 应在运行时被展开为可用值，并在进入业务逻辑前具有正确类型（bool/int/float）。
4. 语义召回允许在模型缺失时降级，但不能再出现 `NoneType object is not callable` 这类不可观测错误。

### 本轮代码修改（已落地）

#### A. 配置系统修复
1. `src/service/config.py`
- 接入 `load_yaml_with_env`（来自 `config_loader`），统一使用同一套环境变量展开逻辑。
- 新增递归类型转换（字符串 `true/false/数字/null` -> Python 类型），修复 `llm.enabled`、`rerank.enabled`、`planner.osrm_url` 等值被字符串污染的问题。
- `RuntimeConfig.log` 与 `RuntimeConfig.logging` 统一绑定同一配置对象，减少分叉。
- `PathsConfig` 默认值切到：
  - `data/all/poi_expanded.csv`
  - `data/all/user_events.csv`

2. `src/service/config_loader.py`
- 修复环境变量覆盖路径解析：
  - 旧行为：`GOAFAR_RERANK_USE_RERANKER_MODEL` 会被错误拆成 `rerank.use.reranker.model`。
  - 新行为：基于现有配置键构建索引，优先精确映射到 `rerank.use_reranker_model`。
- 保留未知键回退逻辑，避免完全丢失扩展能力。

3. `configs/runtime.yaml`
- `paths` 默认改为 latest 数据：
  - `poi_csv: data/all/poi_expanded.csv`
  - `user_events_csv: data/all/user_events.csv`

#### B. 语义召回稳定性修复
4. `src/embedding/vector_builder.py`
- 增加 Qwen3 embedding 可选导入与后备路径解析。
- 新增统一 encoder 工厂 `_create_encoder(...)`：
  - 按模型路径优先尝试 Qwen 或 BGE；
  - 支持 BGE 不可用时回退 Qwen；
  - 所有候选失败时抛出显式 `RuntimeError`（包含失败原因汇总），避免 `NoneType` 调用错误。
- `build_poi_embeddings`、`search_similar_pois` 统一通过 `_create_encoder(...)` 初始化编码器。
- 默认 `poi_csv` 参数切到 `data/all/poi_expanded.csv`（包括 build/ensure/search）。

#### C. Pipeline 接线修复
5. `src/service/pipeline.py`
- 增加 `SRC_ROOT` 注入，兼容 `service.pipeline` 与 `src.service.pipeline` 两种导入方式。
- 将本模块内部配置导入改为相对导入（`.config`），降低包路径耦合。
- 在 `merge_candidates(...)` 调用中补齐透传参数：
  - `use_recbole`
  - `recbole_model_path`
  - `recbole_config`
  - `recbole_use_gpu`
  - `adaptive_fusion_enabled`

#### D. 候选召回默认数据源对齐
6. `src/recommendation/candidate_merger.py`
- 默认数据路径切到 latest：
  - `poi_csv -> data/all/poi_expanded.csv`
  - `user_events_csv -> data/all/user_events.csv`

### 验证记录（本轮已执行）
1. 静态编译校验
- 命令：`python -m compileall -q ...`
- 结果：本轮修改文件编译通过，无语法错误。

2. 配置行为验证
- 验证项：`load_runtime_config()` 返回类型正确（bool/int/path），不再是字符串占位符。
- 样例结果：
  - `cfg.rerank.use_reranker_model` 为 `bool`
  - `cfg.llm.enabled` 为 `bool`
  - `cfg.planner.osrm_url` 为有效 URL 字符串
  - `cfg.paths.poi_csv` 为 `data/all/poi_expanded.csv`

3. 环境变量覆盖下划线键验证
- 设置 `GOAFAR_RERANK_USE_RERANKER_MODEL=false` 后复核：
  - `rerank.use_reranker_model == False`（已正确命中目标键）。

4. 导入兼容验证
- 在 `goafar` 环境下验证：
  - `import service.pipeline` 成功
  - `import src.service.pipeline` 成功
- 注：在非 goafar 环境首次验证遇到 `ortools` 缺失，后切换 goafar 复核通过。

5. 语义编码器降级路径验证（轻量）
- 通过 monkey patch 方式验证 `_create_encoder` 在 `BGEM3Encoder=None` 时可回退到 Qwen 路径，不再出现 `NoneType` 调用。

### 本轮未解决问题（Remaining Gaps）
1. **未执行完整端到端 pipeline 回归**（只做了静态+关键行为校验）。
2. **未重建全量 embedding 产物**（仍需对 `data/all/poi_expanded.csv` 真正跑向量构建并替换线上产物）。
3. **RecBole/TRL/MLflow/FAISS 等依赖完整性未在当前轮次统一补齐**（只完成接线与降级路径）。
4. **GRPO KL 参考策略问题（同模型计算 ref/current）仍未改**。
5. **训练脚本层面的默认路径与参数一致性尚未全量巡检**（已改核心链路与关键模块默认值，但未覆盖所有脚本）。
6. **测试体系问题（pytest 收集混用、A/B 边界测试稳定性）尚未在本轮修复**。

### 风险与影响评估
1. 配置加载行为现在会应用 `environments.<env>` 覆盖（默认取 `GOAFAR_ENV_MODE`，未设时为 `dev`）。
2. 因 `dev` 配置存在 `llm.use_gpu=false`，若未显式设置环境，行为会偏向开发配置（此为预期，但需团队知晓）。
3. 默认数据源切到 `data/all` 后，内存/时延/构建成本会上升；需配合全量 embedding 与缓存策略。

### 下一步行动（建议执行顺序）
1. **执行全量 embedding 重建并落盘**
- 目标：基于 `data/all/poi_expanded.csv` 生成与主链路匹配的 `outputs/emb/poi_emb.npy` + `poi_meta.csv`（或明确新命名并同步 pipeline）。

2. **做一次 pipeline E2E 回归（goafar 环境）**
- 覆盖：意图->召回->重排->规划->文案。
- 验证点：
  - 不再出现 `${...}` 原样字符串；
  - 语义召回可用或可观测降级；
  - provider/fallback 日志与 debug 字段一致。

3. **补齐训练/评估依赖并做最小可运行验证**
- 优先：`pytest`、`recbole`、`trl`、`mlflow`、`faiss`。
- 目标：至少完成 smoke 级训练/评估任务启动验证。

4. **修复 GRPO KL 参考策略实现**
- 引入冻结参考模型或 checkpoint 参考分支，避免 KL 约束失真。

5. **继续统一导入规范与测试组织**
- 减少 `sys.path` 注入依赖，逐步统一包导入；
- 清理脚本式测试与 pytest 用例混放问题。

### 本轮结论
- 已从“仅 review”推进到“关键问题实修 + 基础验证”。
- P0 中最关键的三项（配置展开/类型、latest 数据默认链路、语义召回 NoneType 降级）已完成代码级修复。
- 仍需一轮以 goafar 环境为基准的完整回归与训练链路验证，才能确认生产可用性。

---

## 2026-02-15 对话追加记录（本地模型集成 + 全量向量重建 + GPU统一 + 并行Review）

### 本轮目标（用户新增要求）
1. 明确“统一使用 `models/` 目录里的本地模型”，检查并完成集成。
2. 使用本地模型重建完整向量（基于 `data/all/poi_expanded.csv`）。
3. 在向量重建后台运行期间，继续做代码review与修复。
4. 统一默认使用GPU（用户明确要求“统一用GPU”）。

### 关键决策（Decision Log）
1. **模型路径策略：本地路径优先，禁用远程默认值**  
   将配置与训练/评估脚本的默认模型路径统一到 `models/Qwen3-*`，避免默认回落到 `Qwen/...` 或旧缓存目录。

2. **向量重建策略：直接覆盖线上产物文件**  
   直接重建并写入 `outputs/emb/poi_emb.npy` 和 `outputs/emb/poi_meta.csv`，使主链路无需改名切换即可生效。

3. **执行策略：长任务与review并行**  
   向量重建长任务在后台会话持续执行，同时前台并行扫描并修复模型路径、GPU默认值和稳定性问题。

4. **GPU策略：配置层 + 函数签名双重统一**  
   不仅改 `runtime.yaml` / dataclass 默认值，还改关键函数签名的 `use_gpu` 默认值，避免脚本直调时意外走CPU。

### 关键假设（Assumptions）
1. `models/Qwen3-8B`、`models/Qwen3-Embedding-4B`、`models/Qwen3-Reranker-4B` 为当前唯一权威模型目录。
2. `outputs/emb/poi_emb.npy + outputs/emb/poi_meta.csv` 是线上语义召回默认读取产物。
3. 当前环境（goafar + RTX 5090）可承载全量POI向量重建，且允许中间缓存写盘。
4. 用户希望“默认行为”即GPU优先，而不是手动传参才启用GPU。

### 本轮核心执行与代码修改

#### A. 本地模型（`models/`）集成统一
- `src/service/config.py`
  - `EmbeddingConfig.model_path/fallback_model` 默认改为 `models/Qwen3-Embedding-4B`
  - `LLMConfig.qwen_model` 默认改为 `models/Qwen3-8B`
- `configs/runtime.yaml`
  - `models.bge_m3`、`embedding.fallback_model` 统一指向本地Qwen embedding目录
- 训练/评估/推理脚本默认值改为本地路径：
  - `src/content_generation/llm_generator.py`
  - `src/content_generation/train_sft.py`
  - `src/content_generation/train_dpo.py`
  - `src/content_generation/test_sft.py`
  - `src/llm4rec/qwen_recommender.py`
  - `src/rl/grpo_trainer.py`
  - `src/rl/grpo_trainer_trl.py`
  - `src/evaluation/evaluate_dpo.py`
  - `src/evaluation/evaluate_grpo.py`
  - `configs/grpo_planner.yaml`
  - `check_qwen.py`

#### B. 全量向量重建（本地模型）
- 执行命令：
  - `python scripts/build_qwen3_embeddings.py`
- 数据源：
  - `data/all/poi_expanded.csv`（127,977条POI）
- 模型：
  - `models/Qwen3-Embedding-4B`（GPU）
- 输出产物：
  - `outputs/emb/poi_emb.npy`
  - `outputs/emb/poi_meta.csv`
- 结果：
  - 向量形状：`(127977, 2560)`
  - 耗时：`219.1秒`（约`3.7分钟`）
  - 平均速度：`584.2 POI/秒`

#### C. GPU默认统一
- 配置层：
  - `configs/runtime.yaml`：dev环境 `llm.use_gpu: true`
  - `src/service/config.py`：
    - `EmbeddingConfig.use_gpu = True`
    - `LLMConfig.use_gpu = True`
- 函数默认值层：
  - `src/embedding/vector_builder.py`
    - `build_poi_embeddings(..., use_gpu=True)`
    - `ensure_embedding_artifacts(..., use_gpu=True)`
    - `search_similar_pois(..., use_gpu=True)`
  - `src/recommendation/candidate_merger.py`
    - `merge_candidates(..., use_gpu=True)`

#### D. 并行review期间追加修复（稳定性/一致性）
1. VRPTW整数类型修复，解决 OR-Tools 崩溃  
   - 文件：`src/routing/vrptw_solver.py`  
   - 问题：`RoutingModel_AddDimension` 接收浮点 `horizon` 报错  
   - 修复：`horizon/time_limit` 统一强制整型，相关 `SetRange` 也做整型化。

2. 数据读取类型警告与隐患修复  
   - 文件：`src/recommendation/candidate_merger.py`、`src/routing/time_matrix_builder.py`、`src/embedding/vector_builder.py`
   - 修复：`pd.read_csv(..., low_memory=False)`；分数字段显式 `to_numeric + astype(float)`。

3. `warmup` 对齐告警可观测性修复  
   - 文件：`src/service/pipeline.py`
   - 修复：即使 embedding 不可用，也先做训练集-embedding对齐检查并记录 mismatch 告警。

4. data processing 默认路径继续统一到 `data/all`  
   - 文件：`src/data_processing/event_generator.py`、`src/data_processing/sql_extractor.py`、`src/data_processing/wikidata_enricher_full.py`
   - 修复：默认输入/输出路径与主数据策略保持一致。

5. GRPO 参考策略与训练步进逻辑修复（延续本次review）  
   - 文件：`src/rl/grpo_trainer.py`
   - 修复：补充 reference policy 路径（LoRA禁用适配器/冻结副本/退化路径），并修复日志与checkpoint触发的步进时机。

### 验证记录（本轮关键结果）
1. embedding产物完整性
   - `emb_shape = (127977, 2560)`
   - `meta_rows = 127977`
   - `meta_unique = 127977`

2. 训练数据与embedding对齐
   - 对齐检查结果：
     - `dataset_unique_poi_ids = 2588`
     - `embedding_unique_poi_ids = 127977`
     - `missing_in_embeddings = 0`
   - 结论：`outputs/datasets`涉及POI已全部被新的embedding覆盖。

3. pipeline warmup状态
   - `embedding_ready = True`
   - `warmup_messages = []`

4. 轻量冒烟（模板+语义+规划）
   - 推荐成功，语义召回正常（`dense > 0`），无 `embedding_missing` 退化事件。

5. 回归测试抽样
   - `python test_mmoe_ranker.py --device cpu --small-data` 通过。

### 未解决问题（Remaining Gaps）
1. **完整E2E回归仍未覆盖“LLM+Reranker全开”的重负载场景**  
   当前主要验证了轻量链路和语义召回恢复；仍需补全全开模式回归。

2. **RecBole/TRL/MLflow/FAISS依赖仍未在goafar环境统一补齐**  
   影响完整训练与全量评测闭环。

3. **语义模型加载开销仍偏高**  
   当前语义检索路径可用，但仍存在“首次/重复加载模型耗时大”的优化空间（建议做进程级单例缓存）。

4. **部分说明文档/注释仍保留远程模型示例文本**  
   不影响运行，但影响团队对“本地模型优先策略”的一致认知。

### 风险与影响评估
1. 默认GPU统一后，显存占用与并发压力会上升，需要配合服务并发限制与模型复用策略。
2. 全量embedding已切换为12.8万规模，语义检索吞吐依赖GPU与向量索引策略（后续建议补FAISS验证）。
3. 若生产环境缺GPU或显存不足，需要明确CPU退化策略与超时阈值配置。

### 下一步行动（建议执行顺序）
1. **执行一次“LLM + QwenReranker + 语义召回 + 路由 + 文案”全开E2E回归**  
   目标：验证真实服务模式下的延迟、稳定性和降级逻辑。

2. **实现语义编码器进程级缓存/单例**  
   目标：避免推荐请求中重复加载Qwen embedding模型，降低首包和P95时延。

3. **补齐并锁定训练评估依赖版本**  
   重点：`recbole`、`trl`、`mlflow`、`faiss`、`pytest`，并输出一份可复现实验环境说明。

4. **补FAISS全量索引构建与检索对比测试**  
   对比 numpy brute-force 与 FAISS 在召回质量/时延上的差异，形成默认推荐配置。

5. **继续清理文档中的远程模型默认示例**  
   保证团队成员按“`models/`本地优先”执行，不再出现路径歧义。

### 本轮结论
- 用户新增要求已完成：  
  1) 本地模型统一集成到 `models/`；  
  2) 已用本地Qwen embedding重建全量向量；  
  3) `outputs/datasets` 与新embedding已完全对齐（缺失=0）；  
  4) GPU默认策略已统一落地。  
- 项目已从“数据与模型割裂”进入“本地模型+全量向量+GPU默认”的可运行状态，下一步应转向全开链路性能与稳定性优化。

---

## 2026-02-16 对话追加记录（修复 + 训练实验执行）

### 本轮目标
- 在已有 review/修复基础上，继续执行训练与全链路实验（SFT、DPO、GRPO、Pipeline Eval）。
- 若遇到阻塞，优先定位是“代码问题”还是“环境问题”，并给出可复现证据。
- 按用户要求尝试切到 GPU 训练。

### 关键决策（Decision Log）
1. **先做最小可运行烟测（smoke）再扩展**
   - 决策原因：8B 模型全量训练成本极高，先验证链路可跑通可快速暴露代码/API兼容问题。
2. **优先修复训练代码兼容性，再继续实验**
   - SFT/DPO 多次失败后确认是 TRL 新版本接口变更，不先修无法推进。
3. **在 GPU 不可用时，维持 CPU smoke，保证进展可交付**
   - 已按用户要求尝试 GPU，但当前会话无法初始化 CUDA/NVML。
4. **GRPO 增加 CLI 降速参数**
   - 给 `grpo_trainer` 增加 `--max-new-tokens` 与 `--logging-steps`，用于控制 smoke 时长和可观测性。

### 本轮假设（Assumptions）
1. 当前 `models/Qwen3-8B` 可在 CPU 加载且内存足够（机器内存 754Gi，成立）。
2. 用户当前重点是“链路可执行与问题收敛”，不是立即跑完整长训（因此优先 smoke）。
3. 当前环境的 GPU 设备映射/驱动在容器侧存在异常，非项目代码可直接修复项。

### 代码修改清单（本轮新增）

1. **实验追踪接口兼容修复**
   - 文件：`src/utils/experiment.py`
   - 修改：在 `ExperimentTracker` 基类补充默认方法
     - `log_dataset()`
     - `log_training_progress()`
   - 目的：修复 JSON backend 下 `train_sft.py / train_dpo.py / grpo_trainer.py` 调用 `log_dataset` 崩溃。

2. **SFT 训练脚本兼容 TRL 0.28.0**
   - 文件：`src/content_generation/train_sft.py`
   - 关键修改：
     - `SFTConfig(max_seq_length=...)` -> `SFTConfig(max_length=...)`
     - `SFTTrainer(tokenizer=...)` -> `SFTTrainer(processing_class=...)`
     - 移除“先手动 `get_peft_model` 再传 `peft_config`”的冲突做法，改为仅传 base model + `peft_config`
     - 仅在真正启用 4-bit 量化时调用 `prepare_model_for_kbit_training`
   - 结果：SFT smoke 可完整训练+保存。

3. **DPO 训练脚本兼容 TRL 0.28.0**
   - 文件：`src/content_generation/train_dpo.py`
   - 关键修改：
     - `DPOTrainer(tokenizer=...)` -> `DPOTrainer(processing_class=...)`
     - 删除不再接受的构造参数（`beta/max_length/max_prompt_length` 直接传 trainer）
     - 与 SFT 同步：避免“已是 PeftModel + peft_config”双重注入冲突
   - 结果：DPO smoke 可完整训练+保存。

4. **GRPO CLI 可控性增强**
   - 文件：`src/rl/grpo_trainer.py`
   - 新增参数：
     - `--max-new-tokens`
     - `--logging-steps`
   - 目的：降低 CPU smoke 时长，提升训练过程可观测性。

### 数据与实验执行记录（本轮）

1. **smoke 数据切分**
   - 新增临时数据：
     - `outputs/datasets/smoke/sft_data_smoke.jsonl`
     - `outputs/datasets/smoke/sft_data_tiny.jsonl`
     - `outputs/datasets/smoke/dpo_prefs_smoke.csv`
     - `outputs/datasets/smoke/dpo_prefs_tiny.csv`
     - `outputs/datasets/smoke/grpo_planner_prompts_smoke.jsonl`
     - `outputs/datasets/smoke/grpo_planner_prompts_tiny.jsonl`

2. **SFT smoke（成功）**
   - 命令核心：`python -m src.content_generation.train_sft ... --no-gpu`
   - 数据：`sft_data_tiny.jsonl`（2 条）
   - 输出：`outputs/sft/qwen3-8b-smoke`
   - 关键指标：
     - `train_runtime ≈ 6.965s`
     - `train_loss ≈ 4.353`

3. **DPO smoke（成功）**
   - 命令核心：`python -m src.content_generation.train_dpo ... --no-gpu`
   - 数据：`dpo_prefs_tiny.csv`（8 条）
   - 输出：`outputs/dpo/qwen3-8b-dpo-smoke`
   - 关键指标：
     - `train_runtime ≈ 20.31s`
     - `train_loss ≈ 0.6948`

4. **GRPO smoke（未闭环）**
   - 多次尝试（32条/2条样本）均进入训练循环但 CPU 耗时明显偏高。
   - 已新增 `--max-new-tokens` 参数用于后续继续压缩时长。
   - 当前状态：GRPO 仍需一轮“短生成长度 + tiny 数据”的完整收敛验证。

### GPU 尝试与结论（用户要求“用 GPU 跑”）

已执行检测：
- `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader`
  - 结果：`Failed to initialize NVML: Unknown Error`
- `torch.cuda.is_available()` in `goafar` env
  - 结果：`False`
  - 警告：`CUDA error 304` + `Can't initialize NVML`
- 设备节点检查：存在 `/dev/nvidia5`、`/dev/nvidiactl`，但 CUDA 仍不可用。
- 尝试 `CUDA_VISIBLE_DEVICES=5` 后依旧 `cuda_available=False`。

结论：
- 当前会话/容器内 GPU 驱动或运行时不可用，问题不在项目代码层。
- 在该状态下无法真正执行 GPU 训练。

### 未解决问题（Open Issues）
1. **GPU 不可用（阻塞 GPU 训练）**
   - 现象：NVML 初始化失败 + CUDA 304。
   - 影响：仅能 CPU 运行。
2. **GRPO 训练完整闭环尚未完成**
   - 已进入训练，但在 CPU 下默认生成长度导致单步耗时长。
3. **bitsandbytes CPU kernel 警告持续存在**
   - `Failed to load CPU gemm_4bit_forward from kernels-community`。
   - 当前不阻断 SFT/DPO smoke，但提示 CPU 量化路径不完整。
4. **工作区仍为大规模 dirty 状态**
   - 本轮只做了必要修复与实验，不应在未确认前做额外清理或回滚。

### 下一步行动（Next Actions）
1. **先完成 GRPO 最终 smoke 闭环**
   - 使用：
     - `--max-new-tokens 16` 或更小
     - tiny 数据（2 条）
     - `--logging-steps 1`
   - 验收标准：训练完成并写出 `outputs/grpo/...` checkpoint。

2. **完成三类模型评估文件产出**
   - SFT：`python -m src.evaluation.evaluate_sft --model outputs/sft/qwen3-8b-smoke ...`
   - DPO：`python -m src.evaluation.evaluate_dpo --model outputs/dpo/qwen3-8b-dpo-smoke ...`
   - GRPO：`python -m src.evaluation.evaluate_grpo --model outputs/grpo/... ...`

3. **汇总一版实验总报告（可直接对比）**
   - 汇总来源：
     - `docs/FULL_PIPELINE_EXPERIMENT_REPORT.json`
     - `outputs/evaluation/pipeline_eval_no_llm.json`
     - `outputs/evaluation/grpo_baseline_eval.json`
     - 新增 SFT/DPO/GRPO eval 输出
   - 输出：统一 markdown + json 对比结论（成功率、时延、奖励、准确率）。

4. **GPU 路线单独排障**
   - 需要宿主机/容器层处理（驱动/NVML/CUDA runtime），非业务代码改动可解。
   - GPU可用后再将 SFT/DPO/GRPO 从 smoke 扩展至 quick/full 训练。

### 本轮产物索引
- 代码修改：
  - `src/utils/experiment.py`
  - `src/content_generation/train_sft.py`
  - `src/content_generation/train_dpo.py`
  - `src/rl/grpo_trainer.py`
- 训练输出：
  - `outputs/sft/qwen3-8b-smoke`
  - `outputs/dpo/qwen3-8b-dpo-smoke`
- smoke 数据：
  - `outputs/datasets/smoke/*`

---

## 2026-02-16 对话追加记录（Implement plan：review + GPU训练核验 + 全链路复跑）

### 本轮目标（用户要求）
1. 执行既定计划：先 review 关键代码，再跑训练/全链路实验。
2. 以 `goafar` 环境与 GPU 实验结果为准，产出可复现实验文件。

### 已确认的训练产物（GPU）
- SFT smoke（GPU）输出：`outputs/sft/qwen3-8b-smoke-gpu`
- DPO smoke（GPU）输出：`outputs/dpo/qwen3-8b-dpo-smoke-gpu`
- GRPO smoke（GPU）输出：`outputs/grpo/qwen3-8b-grpo-smoke-gpu`
- 对应配置文件时间戳（本机）：
  - `outputs/sft/qwen3-8b-smoke-gpu/training_config.json`：2026-02-16 11:39:37
  - `outputs/dpo/qwen3-8b-dpo-smoke-gpu/training_config.json`：2026-02-16 11:40:11
  - `outputs/grpo/qwen3-8b-grpo-smoke-gpu/grpo_config.json`：2026-02-16 11:41:50

### 本轮 review 发现并修复的问题

1. **评测召回贡献统计失真（固定比例模拟）**
- 文件：`src/evaluation/pipeline_evaluator.py`
- 原问题：贡献值按固定 55/30/15 比例估算，不反映真实 `from_dense/from_behavior/from_geo`。
- 修复：改为读取候选结果实际标记列统计，行为召回可真实显示为 0。

2. **CSV 空值导致评测脚本崩溃（NaN 调用 split）**
- 文件：`src/evaluation/evaluate_pipeline.py`
- 原问题：`ground_truth_pois` 或 `interests` 为空时会变成 `NaN(float)`，触发 `AttributeError: 'float' object has no attribute 'split'`。
- 修复：新增 `_split_cell` 与 `days` 安全转换，兼容空值/NaN。

3. **路线规划前时间矩阵构建会因 NaN 经纬度崩溃**
- 文件：`src/routing/time_matrix_builder.py`
- 原问题：`haversine -> int(NaN)` 抛出 `cannot convert float NaN to integer`。
- 修复：在构建前统一对 `lat/lon` 数值化并剔除无效坐标 POI；若全部无效则抛出明确错误信息。

4. **VRPTW 对脏矩阵/脏时间窗鲁棒性不足**
- 文件：`src/routing/vrptw_solver.py`
- 修复：
  - 统一清洗 `stay_min/open_min/close_min`；
  - 新增 `_sanitize_time_matrix`（方阵校验、NaN/Inf 填充、对角置零、整型化）。

### 本轮实验执行（goafar）

1. **配置覆盖回归测试**
- 命令：`conda run -n goafar pytest -q tests/service/test_config_loader_env_overrides.py`
- 结果：`2 passed`。

2. **全链路评测复跑（5 queries, no-llm）**
- 命令：
  - `conda run -n goafar python -m src.evaluation.evaluate_pipeline --output outputs/evaluation/pipeline_eval_no_llm_rerun2_20260216.json`
- 输出：`outputs/evaluation/pipeline_eval_no_llm_rerun2_20260216.json`
- 关键指标：
  - `successful_queries=5/5`
  - `feasible_routes=4/5`
  - `avg_latency=30.93s`
  - `avg_candidates=63.8`
  - `avg_final_pois=30.0`
  - `recall_contributions: semantic=41.6, behavior=6.0, geo=30.0`（已为真实统计）

3. **单查询修复检查（北京）**
- 输出：`outputs/evaluation/pipeline_eval_beijing_fixcheck2_20260216.json`
- 结果：不再出现 `int(NaN)` 崩溃；当前失败原因转为明确数据问题：
  - `route_error: POI数据为空或坐标无效，无法构建时间矩阵`
  - 日志显示 Top20 候选均被坐标清洗剔除（坐标缺失）。

### 当前结论
1. 训练链路（SFT/DPO/GRPO smoke）与全链路评测已可执行并产出文件。
2. 评测统计口径已修正，不再使用固定比例“伪贡献”。
3. 北京样例的不可行主要是候选 POI 坐标缺失导致的**数据质量问题**，不再是未捕获异常。

### 下一步建议
1. 对 `data/all/poi_expanded.csv` 做坐标完整性清洗（至少保障高频城市候选有可用 `lat/lon`）。
2. 在 rerank 后、规划前增加“坐标有效性过滤+补位候选”策略，避免 TopK 全部无效时直接失败。
3. 若要做 full 训练，建议在现有 smoke 基础上扩展为 quick（增大样本与 epoch），并保留本次评测 JSON 作为基线。

---

## 2026-02-16 增量记录（仅使用有坐标 POI）

### 背景决策
- 用户确认先采用“只用有坐标数据”的临时方案，缺失坐标数据待后续补齐。
- 目标：确保训练/全链路实验先稳定可跑，避免坐标缺失导致规划阶段失败。

### 本轮改动
1. 新增坐标过滤脚本  
- 文件：`scripts/filter_poi_with_coords.py`  
- 功能：从 `data/all/poi_expanded.csv` 过滤出 `lat/lon` 有效记录，输出 `data/all/poi_with_coords.csv`。

2. 默认数据源切换到坐标子集  
- 文件：`configs/runtime.yaml`  
- 文件：`src/service/config.py`  
- 变更：`paths.poi_csv` 默认由 `data/all/poi_expanded.csv` 改为 `data/all/poi_with_coords.csv`。

3. 召回阶段强制过滤无坐标候选  
- 文件：`src/recommendation/candidate_merger.py`  
- 变更：`_prepare_poi_df` 中对 `lat/lon` 数值化并过滤无效坐标行，避免脏数据流入 rerank/规划。

4. 向量模块默认对齐坐标子集  
- 文件：`src/embedding/vector_builder.py`  
- 变更：`build_poi_embeddings / ensure_embedding_artifacts / search_similar_pois` 的默认 `poi_csv` 改为 `data/all/poi_with_coords.csv`。

5. 时间矩阵默认数据源对齐  
- 文件：`src/routing/time_matrix_builder.py`  
- 变更：`build_time_matrix` 默认 `poi_csv` 改为 `data/all/poi_with_coords.csv`。

6. 评测器空候选边界修复  
- 文件：`src/evaluation/pipeline_evaluator.py`  
- 问题：当某省份无候选时仍调用 rerank，触发 `KeyError: 'description'`。  
- 修复：`len(candidates)==0` 时直接返回 `error=未找到匹配的候选景点`，并记录延迟分解。

### 数据与产物
1. 坐标子集数据文件  
- 输出：`data/all/poi_with_coords.csv`  
- 统计：127977 -> 1333（保留率 1.04%）
- 覆盖省份：8（新疆/四川/西藏/云南/甘肃/青海/宁夏/内蒙古）

2. embedding 对齐处理  
- 原全量 embedding 备份：
  - `outputs/emb/poi_emb_full.npy`
  - `outputs/emb/poi_meta_full.csv`
  - `outputs/emb/poi_faiss_full.index`
- 当前在线使用（坐标子集）：
  - `outputs/emb/poi_emb.npy`（1333 行）
  - `outputs/emb/poi_meta.csv`（1333 行）

### 评测结果（no-llm，默认5查询）
- 命令：`conda run -n goafar python -m src.evaluation.evaluate_pipeline --output outputs/evaluation/pipeline_eval_coords_only_default_v2_20260216.json`
- 输出：`outputs/evaluation/pipeline_eval_coords_only_default_v2_20260216.json`
- 指标：
  - `successful_queries=4/5`
  - `feasible_routes=4/5`
  - `avg_latency=33.49s`
- 失败样例：
  - 北京 query 返回 `未找到匹配的候选景点`（当前坐标子集不覆盖北京，行为符合预期）。

### 覆盖省份基线（no-llm，5/5）
- 查询集：`outputs/evaluation/queries_coords_supported5_20260216.csv`（新疆/西藏/云南/四川/甘肃）
- 命令：`conda run -n goafar python -m src.evaluation.evaluate_pipeline --queries outputs/evaluation/queries_coords_supported5_20260216.csv --output outputs/evaluation/pipeline_eval_coords_only_supported5_20260216.json`
- 输出：`outputs/evaluation/pipeline_eval_coords_only_supported5_20260216.json`
- 指标：
  - `successful_queries=5/5`
  - `feasible_routes=5/5`
  - `avg_latency=33.46s`

### 12:xx 增量修复（与上段记录相比）

1. **主链路规划前新增坐标有效性过滤**
- 文件：`src/service/pipeline.py`
- 变更：在 Step4 路由规划前对 `candidates` 做 `lat/lon` 数值化有效性筛选；若有效候选不足，返回明确错误 `候选景点坐标缺失，无法进行路线规划`，并记录 fallback 事件。

2. **评测器规划阶段同样新增坐标过滤**
- 文件：`src/evaluation/pipeline_evaluator.py`
- 变更：
  - 规划前过滤无效坐标候选；
  - 记录 `planning_filtered_out`；
  - 当有效候选不足时返回明确 `route_error=候选POI坐标缺失，无法构建路线`，不再走到异常栈。

3. **再次复跑最终全链路评测（rerun3）**
- 输出：`outputs/evaluation/pipeline_eval_no_llm_rerun3_20260216.json`
- 指标：
  - `successful_queries=5/5`
  - `feasible_routes=4/5`
  - `avg_latency=30.96s`
  - `recall_contributions: semantic=41.6, behavior=6.0, geo=30.0`
- 北京样例现象：`planning_filtered_out=30`，`route_error=候选POI坐标缺失，无法构建路线`（数据质量问题被清晰暴露）。

---

## 2026-02-16 增量记录（coords-only 全链路复跑 + 训练补齐）

### 本轮目标
- 在“仅使用有坐标 POI（`data/all/poi_with_coords.csv`）”前提下，补齐全链路实验与训练执行。
- 优先修复会直接阻断实验脚本可读输出的问题，再做训练闭环。

### 代码修复
1. 修复全链路实验脚本对 `RouteStop` 字段读取错误
- 文件：`scripts/full_pipeline_experiment.py`
- 问题：脚本使用 `stop.name` / `stop.duration_minutes`，但实际 schema 为 `poi_name` / `stay_min`，导致报错：`'RouteStop' object has no attribute 'name'`。
- 修复：新增 `_route_stop_to_dict` 兼容不同字段命名，并统一用于控制台打印与 JSON 报告落盘。

2. 调整多组件测试样例以匹配 coords-only 覆盖省份
- 文件：`scripts/full_pipeline_experiment.py`
- 变更：将“北京短途游”替换为“甘肃短途游”；增加按当前 `poi_csv` 省份集合的校验，不在数据集中的省份可跳过并记原因。

### 全链路实验执行
1. 重新运行全链路实验
- 命令：`conda run -n goafar python scripts/full_pipeline_experiment.py`
- 产物：
  - `docs/FULL_PIPELINE_EXPERIMENT_REPORT.md`
  - `docs/FULL_PIPELINE_EXPERIMENT_REPORT.json`
- 结果：3/3 端到端用例成功（新疆、云南、甘肃）；性能压测 3 次平均约 `28.70s`。

2. 关键观测
- 仍出现对齐告警：`training_embedding_mismatch:2134`（训练样本中大量 POI 不在当前 1333 coords-only embedding 集内）。
- 甘肃样例出现精排告警：`qwen_reranker_failed:'float' object has no attribute 'lower'`（主流程回退后仍能成功出路线）。

### 训练补齐（GRPO）
1. 先做 GRPO 训练数据与 coords-only 对齐过滤
- 输入：`outputs/datasets/grpo_planner_prompts.jsonl`（1754）
- 过滤条件：`target_next_poi` 与 `state_prefix` 全部在 `data/all/poi_with_coords.csv` 的 `poi_id` 集中。
- 输出：`outputs/datasets/grpo_planner_prompts_coords311_20260216.jsonl`（311）

2. 第一次训练尝试（失败）
- 命令：
  - `conda run -n goafar python -m src.rl.grpo_trainer --model models/Qwen3-8B --data outputs/datasets/grpo_planner_prompts_coords311_20260216.jsonl --output outputs/grpo/qwen3-8b-grpo-coords311-quick-20260216 --epochs 1 --batch-size 4 --grad-accum 2 --group-size 4 --lr 1e-5 --max-new-tokens 64 --logging-steps 20 --use-lora`
- 结果：CUDA OOM（约 31GB 占满）。

3. 第二次训练尝试（成功）
- 命令：
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True conda run -n goafar python -m src.rl.grpo_trainer --model models/Qwen3-8B --data outputs/datasets/grpo_planner_prompts_coords311_20260216.jsonl --output outputs/grpo/qwen3-8b-grpo-coords311-quick-20260216-v2 --epochs 1 --batch-size 1 --grad-accum 8 --group-size 2 --lr 1e-5 --max-new-tokens 16 --logging-steps 10 --use-lora`
- 结果：完成 1 epoch（311 steps / global_step=38）。
- 产物：`outputs/grpo/qwen3-8b-grpo-coords311-quick-20260216-v2`
- 指标（`grpo_config.json`）：
  - `global_step=38`
  - `best_reward=0.5`

### 本轮训练状态汇总
- SFT（已有）：`outputs/sft/qwen3-8b-sft-coords-quick-20260216`
  - 风险：后半段 `loss=0`、`grad_norm=nan`，需后续稳定性修复。
- DPO（已有）：`outputs/dpo/qwen3-8b-dpo-coords-quick-20260216`
  - 训练过程正常，`loss` 从约 `0.68` 下降到约 `0.49`。
- GRPO（本轮补齐）：`outputs/grpo/qwen3-8b-grpo-coords311-quick-20260216-v2`
  - 已可在 coords-only 过滤数据上完成训练闭环。

### 当前待跟进问题
1. `training_embedding_mismatch:2134` 仍未消除（训练数据与在线 embedding 基座不一致）。
2. `qwen_reranker_failed:'float' object has no attribute 'lower'` 需要在 reranker 输入标准化处继续排查。
3. SFT 数值稳定性问题（`nan`）需要单独调参/数据清洗后复训。

### 补充：端到端完整评测 rerun2（支持省份5条）
- 命令：
  - `conda run -n goafar bash scripts/evaluate_complete.sh --queries outputs/evaluation/queries_coords_supported5_20260216.csv --output-dir outputs/evaluation/complete_coords_supported5_20260216_rerun2`
- 输出：
  - `outputs/evaluation/complete_coords_supported5_20260216_rerun2/pipeline_eval.json`
  - `outputs/evaluation/complete_coords_supported5_20260216_rerun2/pipeline_eval.log`
  - `outputs/evaluation/complete_coords_supported5_20260216_rerun2/evaluation_summary.txt`
- 结果：
  - `successful_queries=5/5`
  - `feasible_routes=5/5`
  - `avg_latency=33.42s`
  - `avg_candidates=80`
  - `avg_final_pois=30`
  - `recall_contributions: semantic=65.8, behavior=6.0, geo=30.0`

### 补充修复：Reranker NaN 文本字段兼容
- 文件：`src/reranking/qwen_reranker.py`
- 问题：规则回退分支对 `candidate['description']` 等字段直接 `.lower()`，当字段为 `NaN(float)` 时触发 `qwen_reranker_failed:'float' object has no attribute 'lower'`。
- 修复：新增 `_safe_text()`，统一把 `None/NaN/非字符串` 转成安全字符串；在 `_compute_scores`、`_rule_based_rerank`、`compute_pairwise_score` 的 fallback 路径全量接入。
- 验证：
  1. 单元级验证：构造 `description=float('nan')` 候选，规则回退可正常返回分数。
  2. 主链路抽样验证（甘肃 query）已不再出现 `'float'.lower` 报错，`fallback_events` 正常为 `['training_embedding_mismatch:2134', 'qwen_reranker_applied']`。

### 2026-02-16 补充：重建 coords-only embedding（Qwen3）
- 命令：
  - `conda run -n goafar python -c "from src.embedding.vector_builder import build_poi_embeddings; build_poi_embeddings(poi_csv='data/all/poi_with_coords.csv', output_dir='outputs/emb', model_path='models/Qwen3-Embedding-4B', use_gpu=True, build_faiss=True, faiss_index_file='outputs/emb/poi_faiss.index')"`
- 产物（已覆盖更新）：
  - `outputs/emb/poi_emb.npy`
  - `outputs/emb/poi_meta.csv`
  - `outputs/emb/poi_faiss.index`
- 校验：
  - `emb_shape=(1333, 2560)`
  - `meta_rows=1333`，`unique_poi=1333`
  - `nan_in_emb=0`
  - 抽样检索返回 `retrieval_backend=faiss`，结果正常。

### 2026-02-16 补充：重跑评测与训练（rerun）

1. 完整评测 rerun3（支持省份5条）
- 命令：
  - `conda run -n goafar bash scripts/evaluate_complete.sh --queries outputs/evaluation/queries_coords_supported5_20260216.csv --output-dir outputs/evaluation/complete_coords_supported5_20260216_rerun3`
- 输出：
  - `outputs/evaluation/complete_coords_supported5_20260216_rerun3/pipeline_eval.json`
  - `outputs/evaluation/complete_coords_supported5_20260216_rerun3/pipeline_eval.log`
  - `outputs/evaluation/complete_coords_supported5_20260216_rerun3/evaluation_summary.txt`
- 结果（`pipeline_eval.json.metrics`）：
  - `successful_queries=5/5`
  - `feasible_routes=5/5`
  - `avg_latency=33.3787s`
  - `avg_candidates=81.0`
  - `avg_final_pois=30.0`
  - `recall_contributions={semantic:64.2, behavior:6.0, geo:30.0}`

2. SFT 重跑（coords-only）
- 命令：
  - `conda run -n goafar python -m src.content_generation.train_sft --data outputs/datasets/sft_data.jsonl --output outputs/sft/qwen3-8b-sft-coords-rerun-20260216 --use-qlora --lora-r 16 --lora-alpha 16 --lora-dropout 0.05 --lr 5e-5 --epochs 1 --batch-size 2 --grad-accum 8 --max-length 256`
- 输出目录：
  - `outputs/sft/qwen3-8b-sft-coords-rerun-20260216`
- 结果：
  - `checkpoint-111/trainer_state.json: global_step=111, max_steps=111`
  - 训练后半段仍出现数值异常：`loss=0`（10 次记录）与 `grad_norm=nan`（11 次记录）
  - 结论：SFT 仍存在稳定性问题，需后续单独处理（学习率/数据清洗/梯度裁剪等）。

3. DPO 重跑（coords-only）
- 命令：
  - `conda run -n goafar python -m src.content_generation.train_dpo --model models/Qwen3-8B --prefs outputs/datasets/dpo_prefs.csv --output outputs/dpo/qwen3-8b-dpo-coords-rerun-20260216 --epochs 1 --batch-size 2 --grad-accum 2 --lr 1e-5 --beta 0.1 --max-length 512 --max-prompt-length 256 --use-lora --use-qlora`
- 输出目录：
  - `outputs/dpo/qwen3-8b-dpo-coords-rerun-20260216`
- 结果：
  - `checkpoint-75/trainer_state.json: global_step=75, max_steps=75`
  - 训练日志显示 loss 从约 `0.6799` 下降到 `0.4896`，`rewards/accuracies` 升至约 `0.70`
  - 终端汇总：`train_runtime=91.62s, train_loss=0.5855`
  - 结论：DPO 本轮训练正常收敛。

4. GRPO 重跑（coords311）
- 命令：
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True conda run -n goafar python -m src.rl.grpo_trainer --model models/Qwen3-8B --data outputs/datasets/grpo_planner_prompts_coords311_20260216.jsonl --output outputs/grpo/qwen3-8b-grpo-coords311-rerun-20260216 --epochs 1 --batch-size 1 --grad-accum 8 --group-size 2 --lr 1e-5 --max-new-tokens 16 --logging-steps 10 --use-lora`
- 输出目录：
  - `outputs/grpo/qwen3-8b-grpo-coords311-rerun-20260216`
- 结果：
  - 训练 1 epoch（311 样本）完成，进度约 `4m54s`
  - 日志关键点：
    - `Step 10: loss=-0.2515, mean_reward=-1.0000`
    - `Step 20: loss=-0.8594, mean_reward=-0.2500`
    - `Step 30: loss=-1.7322, mean_reward=-0.2500`
  - `grpo_config.json: global_step=38, best_reward=0.5`

5. 备注
- DPO/GRPO 均出现 MLflow 模型落盘告警（`task` 未显式指定），不影响本地模型与配置产物保存。
- 当前 coords-only 路线下，评测与 DPO/GRPO 已形成可复现闭环；SFT 稳定性仍是主要遗留问题。

### 2026-02-16 补充：SFT 稳定性修复与复训（最终可用配置）

1. 代码修复（`src/content_generation/train_sft.py`）
- 数据清洗修复：
  - `response` 与 `completion` 混合存在时，按行回填 `response <- completion`（原先仅在缺失整列时回填）。
  - 清理空文本与字符串 `"nan"`，移除无效样本。
- 训练参数增强：
  - 新增 `max_grad_norm`（CLI: `--max-grad-norm`）并写入 `SFTConfig` 与 `training_config.json`。
- 模板一致性：
  - 训练文本改为 `tokenizer.apply_chat_template(...)` 生成，避免手工模板偏差。
- 训练策略调整：
  - LoRA 与 QLoRA 解耦：即使 `--no-qlora`（不做4-bit量化）仍可注入 LoRA 训练。

2. 三轮稳定化复训对比（同一数据集 `outputs/datasets/sft_data.jsonl`，1 epoch）
- v1（QLoRA, lr=1e-5）：
  - 输出：`outputs/sft/qwen3-8b-sft-coords-stable-20260216`
  - 结果：`first_loss≈306878.6`，后续 `loss=0`，`nan_grad_steps=11`（失败）
- v2（QLoRA + chat_template, lr=1e-5）：
  - 输出：`outputs/sft/qwen3-8b-sft-coords-stablev2-20260216`
  - 结果：`first_loss≈318352.2`，后续 `loss=0`，`nan_grad_steps=11`（失败）
- v3（LoRA bf16，关闭4-bit：`--no-qlora`, lr=5e-6, batch=1, grad_accum=16）：
  - 输出：`outputs/sft/qwen3-8b-sft-coords-stablev3-lora-20260216`
  - 结果（成功）：
    - `first_loss=3.6637`, `last_logged_loss=3.2118`
    - `first_grad_norm=1.2468`, `last_grad_norm=1.6406`
    - `zero_loss_steps=0`, `nan_grad_steps=0`
    - 终端汇总：`train_runtime=520.7s`, `train_loss=3.406`, `mean_token_accuracy=0.5327`

3. 结论
- 本环境下 SFT 的核心不稳定来源来自 **4-bit QLoRA 路径**（至少对当前模型/依赖组合如此）。
- 当前建议将 SFT 基线切换为：**LoRA（bf16, no-qlora）**，使用目录：
  - `outputs/sft/qwen3-8b-sft-coords-stablev3-lora-20260216`

4. 生成验收（smoke）
- 命令：
  - `conda run -n goafar python src/content_generation/test_sft.py --model outputs/sft/qwen3-8b-sft-coords-stablev3-lora-20260216 --base-model models/Qwen3-8B`
- 结果：
  - 模型加载与推理均成功，无 NaN/崩溃。
  - 但 4 个测试样例输出均偏向长链路思考文本（`<think>...`），未严格对齐 JSON 结构。
  - 结论：训练稳定性问题已解决；结构化输出约束仍需后续通过数据模板/解码约束进一步强化。

### 2026-02-16 补充：稳定版 SFT 权重接入全链路并重跑完整评测

1. 接入修复（确保 LoRA 真正生效）
- 文件：`src/service/pipeline.py`
  - `_maybe_get_qwen` 新增透传：
    - `use_lora=self.config.llm.use_lora`
    - `lora_path=str(resolve_path(...))`
  - `_check_llm_health` 新增 LoRA 配置校验：
    - 开启 LoRA 时，`lora_path` 缺失或路径不存在直接判定不健康。
  - `get_model_info` 新增 `llm.use_lora/lora_path` 字段，便于运行时确认。

2. 评测链路修复（`--use-llm` 之前实际上未加载模型）
- 文件：`src/evaluation/pipeline_evaluator.py`
  - `initialize(use_llm=True)` 现在会：
    - 读取运行时配置并初始化 `QwenRecommender`；
    - 传入 base model + LoRA 参数；
    - 将同一个 `llm_model` 注入 `IntentUnderstandingModule` 与 `LLMReranker`。
  - 若 LLM 初始化失败，显式打印并回退模板模式（`effective_use_llm=False`）。

3. 完整评测执行（LLM + stable SFT LoRA）
- 命令：
  - `GOAFAR_LLM_ENABLED=true GOAFAR_LLM_LORA=true GOAFAR_LLM_LORA_PATH=outputs/sft/qwen3-8b-sft-coords-stablev3-lora-20260216 GOAFAR_LLM_MODEL=models/Qwen3-8B conda run -n goafar bash scripts/evaluate_complete.sh --use-llm --queries outputs/evaluation/queries_coords_supported5_20260216.csv --output-dir outputs/evaluation/complete_coords_supported5_20260216_stablev3_llm_20260216_145022`
- 输出：
  - `outputs/evaluation/complete_coords_supported5_20260216_stablev3_llm_20260216_145022/pipeline_eval.json`
  - `outputs/evaluation/complete_coords_supported5_20260216_stablev3_llm_20260216_145022/pipeline_eval.log`
  - `outputs/evaluation/complete_coords_supported5_20260216_stablev3_llm_20260216_145022/evaluation_summary.txt`

4. 关键结果（`pipeline_eval.json.metrics`）
- `successful_queries=5/5`
- `feasible_routes=5/5`
- `avg_latency=52.14s`
- `avg_candidates=81.0`
- `avg_final_pois=30.0`
- `recall_contributions={semantic:64.2, behavior:6.0, geo:30.0}`
- `latency_breakdown={intent:8.48s, recall:3.12s, rerank:10.51s, plan:30.02s}`

5. 运行观测（来自 `pipeline_eval.log`）
- 已确认加载 stable SFT LoRA：
  - `加载LoRA适配器: .../outputs/sft/qwen3-8b-sft-coords-stablev3-lora-20260216`
  - `✓ LoRA适配器加载完成`
- 但 5 条 query 均出现：
  - `LLM意图理解失败，回退到模板: Expecting value...`
  - `LLM重排序失败 ... 回退到规则模式`（其中 1 条包含 CUDA OOM 报错）
- 结论：本轮已完成“权重接入并可加载”，但在线推理阶段仍大量走回退路径；评测成功率维持 5/5，但延迟显著上升。

### 2026-02-16 补充：在线 LLM 输出稳定性修复（fix1/fix2/fix3）

1. 修复范围
- `src/llm4rec/intent_understanding.py`
  - 优先复用 `QwenRecommender.understand_intent()`，并保留模板回退。
- `src/llm4rec/llm_reranker.py`
  - 优先复用 `QwenRecommender.rerank_pois()`，并增强 JSON 抽取与异常回退。
- `src/llm4rec/qwen_recommender.py`
  - 关键修复：`apply_chat_template(..., enable_thinking=False)`，避免 `<think>` 占满输出导致 JSON 被截断。
  - 增加意图字段归一化（兼容 `destination/duration/type` 等别名字段）。
  - 增加显存门控：低显存时跳过 LLM rerank，直接规则回退，避免 OOM。

2. 评测产物
- fix1:
  - `outputs/evaluation/complete_coords_supported5_20260216_stablev3_llm_fix1_20260216_150005`
- fix2:
  - `outputs/evaluation/complete_coords_supported5_20260216_stablev3_llm_fix2_20260216_150500`
- fix3（本轮）:
  - `outputs/evaluation/complete_coords_supported5_20260216_stablev3_llm_fix3_20260216_151627`

3. 结果对比（`pipeline_eval.json.metrics`）
- baseline（接入后未稳定化）：
  - `avg_latency=52.14s`，`intent=8.48s`，`rerank=10.51s`
- fix1：
  - `avg_latency=52.77s`，`intent=12.65s`，`rerank=6.83s`
- fix2：
  - `avg_latency=42.12s`，`intent=6.31s`，`rerank=2.58s`
- fix3：
  - `avg_latency=34.99s`，`intent=1.56s`，`rerank=0.29s`
  - `successful_queries=5/5`，`feasible_routes=5/5`

4. fix3 日志计数（`pipeline_eval.log`）
- `LLM意图理解失败`: `0`
- `LLM重排序失败`: `0`
- `CUDA out of memory`: `0`
- `LLM重排序跳过(低显存保护)`: `4`

5. 结论
- “接入成功但在线不稳定”问题已收敛：意图理解失败、rerank失败、OOM 均清零。
- 当前保守策略会在显存余量不足时主动回退规则重排，以换取稳定性与可复现性。

6. 阈值对照（fix3b）
- 命令（仅调整重排显存阈值）：
  - `GOAFAR_LLM_RERANK_MIN_FREE_MB=192 ... bash scripts/evaluate_complete.sh --use-llm --queries outputs/evaluation/queries_coords_supported5_20260216.csv --output-dir outputs/evaluation/complete_coords_supported5_20260216_stablev3_llm_fix3b_20260216_152135`
- 结果（`fix3b`）：
  - `LLM意图理解失败=0`
  - `LLM重排序失败=1`
  - `CUDA OOM=1`
  - `LLM重排序跳过=3`
- 结论：阈值下调会重新引入 OOM，当前保留 `rerank_min_free_mb=768` 默认更稳。
