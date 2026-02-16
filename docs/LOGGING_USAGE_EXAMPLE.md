# GoAfar 统一���志系统使用示例

## 概述

GoAfar 项目使用统一的日志系统，替代了原有的 `print` 语句。日志系统提供：
- 控制台彩色输出
- 文件日志轮转
- 多级别日志（DEBUG/INFO/WARNING/ERROR/CRITICAL）
- 从配置文件加载日志设置
- 环境变量覆盖

## 基本使用

### 在模块中使用

```python
from src.utils import get_logger

# 获取当前模块的 logger
logger = get_logger(__name__)

# 记录不同级别的日志
logger.debug("调试信息：变量值 = %s", value)
logger.info("模块初始化完成")
logger.warning("配置文件未找到，使用默认值")
logger.error("处理失败: %s", str(e))
logger.critical("系统无法继续运行")
```

### 从配置文件加载

```python
from src.service.config import load_runtime_config
from src.utils import configure_logging_from_config

# 加载完整配置
cfg = load_runtime_config()

# 使用 log 配置初始化日志
logger = configure_logging_from_config(cfg.log)
logger.info("日志系统从配置初始化")
```

### 自定义日志配置

```python
from src.utils import setup_logger

# 创建自定义日志记录器
logger = setup_logger(
    name="my_module",
    level="DEBUG",
    log_dir="logs/my_app",
    console_output=True,
    file_output=True,
    use_colors=True
)
```

## 配置文件

### configs/runtime.yaml

```yaml
# 日志配置
logging:
  level: INFO                    # 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  log_dir: logs                 # 日志输出目录
  log_file: null                # 自定义日志文件路径
  console_output: true            # 是否输出到控制台
  file_output: true              # 是否输出到文件
  max_bytes: 10485760            # 单文件最大字节数 (10MB)
  backup_count: 5                 # 保留的备份文件数量
  use_colors: true               # 控制台是否使用颜色
  format: "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"  # 日志格式
```

### 环境变量

可以通过环境变量覆盖配置：

```bash
# 设置日志级别
export GOAFAR_LOG_LEVEL=DEBUG

# 设置日志目录
export GOAFAR_LOG_DIR=logs/app

# 禁用彩色输出
export GOAFAR_LOG_COLORS=false
```

## 替换 print 语句示例

### 之前（使用 print）

```python
print("开始加载数据...")
print(f"加载了 {len(data)} 条记录")
print("警告：配置文件缺失")
```

### 之后（使用 logger）

```python
from src.utils import get_logger

logger = get_logger(__name__)
logger.info("开始加载数据...")
logger.info("加载了 %d 条记录", len(data))
logger.warning("配置文件缺失")
```

## 关键模块中的使用示例

### 1. 训练脚本 (training)

```python
# src/content_generation/train_sft.py
from src.utils import get_logger

logger = get_logger(__name__, level="INFO")

def train_sft(model, data, config):
    logger.info("=" * 60)
    logger.info("SFT训练 - 旅游推荐任务监督微调")
    logger.info("=" * 60)

    logger.info("设备: %s", device)
    if device == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    logger.info("加载数据集: %s", data_path)
    dataset = load_sft_data(data_path)
    logger.info("训练样本数: %d", len(dataset))

    logger.info("开始训练...")
    for epoch in range(config.epochs):
        logger.info("Epoch %d/%d", epoch + 1, config.epochs)
        train_epoch()

    logger.info("训练完成！模型保存至: %s", output_dir)
```

### 2. 排序模块 (ranking)

```python
# src/ranking/deep_ranker.py
from src.utils import get_logger

logger = get_logger(__name__)

class MMoEDeepRanker:
    def __init__(self, config: MMoEConfig = None):
        self.config = config or MMoEConfig()
        logger = get_logger(__name__)
        logger.info("初始化 MMoE 深度排序模型")
        self._check_torch()

    def _check_torch(self):
        try:
            import torch
            self.torch = torch
            self.torch_available = True
            logger.info("PyTorch 可用，使用 GPU 加速")
        except ImportError:
            self.torch = None
            self.torch_available = False
            logger.warning("PyTorch 未安装，使用 sklearn 回退")

    def fit(self, train_data, valid_data):
        logger.info("开始训练...")
        logger.info("训练集大小: %d", len(train_data))
        logger.info("验证集大小: %d", len(valid_data))

        for epoch in range(self.config.epochs):
            train_loss = self._train_epoch(epoch)
            val_metrics = self._validate()

            logger.info(
                "Epoch %d - Loss: %.4f | Val Loss: %.4f | Val AUC: %.4f",
                epoch + 1, train_loss, val_metrics["loss"], val_metrics["auc"]
            )

        logger.info("训练完成")
        return self.training_history
```

### 3. 评测模块 (evaluation)

```python
# src/evaluation/metrics_advanced.py
from src.utils import get_logger

logger = get_logger(__name__)

def recall_at_k(predictions, ground_truth, k):
    if not ground_truth:
        logger.debug("ground truth 为空，返回 recall=0")
        return 0.0

    pred_k = set(predictions[:k])
    true_set = set(ground_truth)
    recall = len(pred_k & true_set) / len(true_set)

    logger.debug("Recall@%d: %.4f (%d/%d)", k, recall, len(pred_k & true_set), len(true_set))
    return recall

def batch_recall_at_k(predictions_list, ground_truth_list, k):
    logger.info("计算批量 Recall@%d，样本数: %d", k, len(predictions_list))

    recalls = []
    for i, (pred, truth) in enumerate(zip(predictions_list, ground_truth_list)):
        if i % 1000 == 0:
            logger.debug("处理进度: %d/%d", i, len(predictions_list))
        recalls.append(recall_at_k(pred, truth, k))

    avg_recall = sum(recalls) / len(recalls)
    logger.info("平均 Recall@%d: %.4f", k, avg_recall)
    return recalls
```

### 4. AB 测试框架 (ab_test)

```python
# src/evaluation/ab_test.py
from src.utils import get_logger

logger = get_logger(__name__)

class ABTest:
    def __init__(self, config: ABTestConfig = None):
        self.config = config or ABTestConfig()
        logger.info("初始化 AB 测试")
        logger.debug("配置: split_ratio=%.2f", self.config.split_ratio)

    def run_experiment(self, control_fn, treatment_fn, queries):
        logger.info("运行 AB 实验，查询数: %d", len(queries))

        groups = self.splitter.split_traffic([q["id"] for q in queries])
        logger.info("流量分配: control=%d, treatment=%d",
                  len(groups["control"]), len(groups["treatment"]))

        # 执行实验
        control_results = []
        for query_id in groups["control"]:
            result = control_fn(queries[query_id])
            control_results.append(result)

        treatment_results = []
        for query_id in groups["treatment"]:
            result = treatment_fn(queries[query_id])
            treatment_results.append(result)

        # 分析结果
        logger.info("分析实验结果...")
        analysis = self._analyze(control_results, treatment_results)

        if analysis["significant"]:
            logger.info("结果显著！p-value: %.4f", analysis["p_value"])
        else:
            logger.info("结果不显著，p-value: %.4f", analysis["p_value"])

        return analysis
```

## 日志级别指南

| 级别 | 用途 | 示例 |
|------|------|------|
| DEBUG | 详细的调试信息，通常只在诊断问题时使用 | 函数入口/出口、变量值、中间计算结果 |
| INFO | 一般信息，确认事情按预期工作 | 模块初始化、处理进度、配置信息 |
| WARNING | 意外但可以处理的情况 | 缺失配置（使用默认值）、降级到备用方法 |
| ERROR | 由于更严重的问题，软件无法执行某些功能 | API 请求失败、文件损坏、模型加载失败 |
| CRITICAL | 严重错误，程序本身可能无法继续 | 内存不足、关键服务不可用 |

## 控制台输出示例

```
2026-02-15 10:30:45 | INFO     | src.ranking.deep_ranker | 初始化 MMoE 深度排序模型
2026-02-15 10:30:45 | INFO     | src.ranking.deep_ranker | PyTorch 可用，使用 GPU 加速
2026-02-15 10:30:46 | INFO     | src.ranking.deep_ranker | 开始训练...
2026-02-15 10:30:46 | INFO     | src.ranking.deep_ranker | 训练集大小: 8000
2026-02-15 10:30:46 | INFO     | src.ranking.deep_ranker | 验证集大小: 2000
2026-02-15 10:30:50 | INFO     | src.ranking.deep_ranker | Epoch 1/20 - Loss: 0.5234 | Val Loss: 0.4821 | Val AUC: 0.7123
2026-02-15 10:30:55 | ERROR    | src.ranking.deep_ranker | 训练中断: CUDA 内存不足
```

## 日志文件

日志文件保存在 `logs/` 目录下：
- `logs/goafar.log` - 主日志
- `logs/ranking.log` - 排序模块日志
- `logs/evaluation.log` - 评测模块日志

当日志文件达到 `max_bytes`（默认 10MB）时，会自动轮转：
```
logs/
├── goafar.log           # 当前日志
├── goafar.log.1       # 备份 1
├── goafar.log.2       # 备份 2
└── goafar.log.3       # 备份 3
```

## 最佳实践

1. **在模块初始化时创建 logger**
   ```python
   logger = get_logger(__name__)
   ```

2. **使用合适的日志级别**
   - 开发/调试：DEBUG
   - 正常运行：INFO
   - 可恢复的问题：WARNING
   - 需要关注的问题：ERROR
   - 无法继续：CRITICAL

3. **使用格式化字符串而非 % f-string**
   ```python
   # 好 - 延迟格式化，仅在需要时计算
   logger.info("处理 %d 条记录，耗时 %.2f 秒", count, duration)

   # 避免 - 立即格式化
   logger.info(f"处理 {count} 条记录，耗时 {duration:.2f} 秒")
   ```

4. **记录异常堆栈**
   ```python
   try:
       risky_operation()
   except Exception as e:
       logger.exception("操作失败")  # 自动包含异常堆栈
       raise
   ```

5. **在长时间操作中记录进度**
   ```python
   total = len(items)
   for i, item in enumerate(items):
       if i % 1000 == 0:
           logger.info("处理进度: %d/%d", i, total)
       process(item)
   ```

6. **避免在循环中使用 print 调试**
   ```python
   # 之前
   for i in range(100):
       print(i)

   # 之后
   logger.debug("索引: %d", i)
   ```
