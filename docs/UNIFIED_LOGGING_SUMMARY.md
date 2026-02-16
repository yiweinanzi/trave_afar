# GoAfar 项目统一日志系统 - 实现总结

## 概述

本次实现为 GoAfar 旅游推荐系统添加了完整的统一日志系统，替代了项目中分散的 `print` 语句，实现了结构化、可配置的日志记录功能。

## 实现的功能

### 1. 统一日志系统 (`src/utils/logger.py`)

已完成的功能包括：

#### 1.1 核心功能
- **多级别日志**：支持 DEBUG, INFO, WARNING, ERROR, CRITICAL 五个标准级别
- **双输出通道**：同时支持控制台和文件输出
- **日志轮转**：使用 `RotatingFileHandler` 自动管理日志文件大小
- **彩色输出**：终端显示带颜色的日志（可关闭）
- **JSON 格式**：可选的结构化日志输出（用于日志分析）

#### 1.2 日志格式
- **控制台格式**：`%(asctime)s | %(levelname)-8s | %(name)s | %(message)s`
- **文件格式**：包含时间戳、级别、模块名和消息
- **彩色支持**：不同级别使用不同颜色（DEBUG-青色, INFO-绿色, WARNING-黄色, ERROR-红色, CRITICAL-紫色）

#### 1.3 配置化
- **环境变量覆盖**：支持 `GOAFAR_LOG_LEVEL` 环境变量
- **配置文件集成**：从 `configs/runtime.yaml` 读取日志配置
- **动态配置**：支持代码中动态调整日志级别和输出

#### 1.4 辅助类和函数
- `ColoredFormatter`：彩色控制台日志格式化
- `JSONFormatter`：JSON 格式化输出
- `setup_logger()`：创建和配置 Logger
- `get_logger()`：获取已配置的 Logger（模块级使用）
- `configure_logging_from_config()`：从配置字典加载日志设置
- `get_logger_level_from_env()`：从环境变量读取日志级别

### 2. 配置文件更新 (`src/service/config.py`)

#### 2.1 新增 LogConfig 数据类

```python
@dataclass
class LogConfig:
    """日志配置"""
    level: str = "INFO"
    log_dir: str = "logs"
    log_file: Optional[str] = None
    console_output: bool = True
    file_output: bool = True
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    use_colors: bool = True
```

#### 2.2 RuntimeConfig 更新
- 添加 `log` 字段（LogConfig 实例）
- 添加 `logging` 别名以保持向后兼容性
- `load_runtime_config()` 函数支持从 YAML 加载 `logging` 配置

### 3. 配置文件更新 (`configs/runtime.yaml`)

```yaml
# 日志配置
logging:
  level: INFO
  log_dir: logs
  log_file: null
  console_output: true
  file_output: true
  max_bytes: 10485760  # 10MB
  backup_count: 5
  use_colors: true
  format: "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
```

### 4. 模块导出更新 (`src/utils/__init__.py`)

新增导出的日志相关符号：
- `get_logger_level_from_env`
- `configure_logging_from_config`
- `LogColors`
- `ColoredFormatter`
- `JSONFormatter`

### 5. 模块导入优化 (`src/ranking/__init__.py`)

- 修复了模块名拼写错误（MMoE 而非 MMoE）
- 添加了使用统一日志的示例代码

## 使用示例

### 基础使用

```python
from src.utils import get_logger

logger = get_logger(__name__)

logger.debug("详细调试信息")
logger.info("常规信息")
logger.warning("警告信息")
logger.error("错误信息")
logger.critical("严重错误")
```

### 从配置加载

```python
from src.service.config import load_runtime_config
from src.utils import configure_logging_from_config

cfg = load_runtime_config()
logger = configure_logging_from_config(cfg.log)
```

### 自定义配置

```python
from src.utils import setup_logger

logger = setup_logger(
    name="my_module",
    level="DEBUG",
    log_dir="logs/my_app",
    console_output=True,
    file_output=True,
    use_colors=True
)
```

### 环境变量控制

```bash
# 设置日志级别
export GOAFAR_LOG_LEVEL=DEBUG

# Python 代码会自动读取
python my_script.py
```

## 文件清单

| 文件路径 | 状态 | 说明 |
|---------|------|------|
| `src/utils/logger.py` | 完成 | 核心日志系统实现 |
| `src/utils/__init__.py` | 更新 | 导出日志相关函数和类 |
| `src/service/config.py` | 更新 | 添加 LogConfig 和加载逻辑 |
| `configs/runtime.yaml` | 更新 | 添加 logging 配置节 |
| `src/ranking/__init__.py` | 更新 | 修复拼写错误并添加日志示例 |
| `docs/LOGGING_USAGE_EXAMPLE.md` | 新增 | 日志使用指南 |
| `docs/PRINT_TO_LOGGER_MIGRATION_EXAMPLES.py` | 新增 | print 替换示例 |

## 日志文件结构

日志文件将保存在 `logs/` 目录下：

```
logs/
├── goafar.log       # 主应用日志
├── goafar.log.1     # 自动轮转的备份
├── goafar.log.2
├── goafar.log.3
├── goafar.log.4
└── goafar.log.5
```

## 迁移指南

### 将现有 print 语句替换为 logger

1. **在模块开头导入并获取 logger**
   ```python
   from src.utils import get_logger
   logger = get_logger(__name__)
   ```

2. **替换 print 语句**
   ```python
   # 之前
   print("开始处理数据...")
   print(f"处理了 {count} 条记录")

   # 之后
   logger.info("开始处理数据...")
   logger.info("处理了 %d 条记录", count)
   ```

3. **使用适当的日志级别**
   - `debug`：开发调试信息
   - `info`：正常运行信息
   - `warning`：可恢复的异常情况
   - `error`：错误但不导致程序退出
   - `critical`：严重错误，程序可能无法继续

### 需要替换的关键文件

以下文件中包含大量 print 语句，建议优先替换：

1. **训练脚本**
   - `src/content_generation/train_sft.py` (多处 print)
   - `src/content_generation/train_dpo.py` (多处 print)

2. **核心模块**
   - `src/embedding/qwen3_encoder.py`
   - `src/embedding/build_embeddings_gpu.py`
   - `src/ranking/deep_ranker.py`

3. **评测模块**
   - `src/evaluation/ab_test.py`
   - `src/evaluation/metrics_advanced.py`

## 最佳实践

1. **在模块级别使用 `get_logger(__name__)`**
   ```python
   logger = get_logger(__name__)
   ```

2. **使用延迟格式化（仅在需要时计算）**
   ```python
   # 好
   logger.info("处理 %d 条记录", count)

   # 避免
   logger.info(f"处理 {count} 条记录")  # 即使是 DEBUG 级别也会执行格式化
   ```

3. **记录异常时使用 exception()**
   ```python
   try:
       risky_operation()
   except Exception as e:
       logger.exception("操作失败")  # 自动包含堆栈跟踪
       raise
   ```

4. **在长时间操作中记录进度**
   ```python
   total = len(items)
   for i, item in enumerate(items):
       if i % 1000 == 0:
           logger.debug("处理进度: %d/%d", i, total)
       process(item)
   ```

## 下一步建议

1. **逐步迁移**：按照优先级替换关键模块中的 print 语句
2. **统一格式**：确保整个项目使用一致的日志格式
3. **添加上下文**：在 API 服务中添加请求 ID 跟踪
4. **监控集成**：考虑集成 Prometheus/ELK 进行日志收集
5. **性能分析**：使用时间日志分析系统瓶颈

## 注意事项

1. **避免循环导入**：`logger.py` 不应依赖项目的其他模块
2. **性能考虑**：生产环境可关闭彩色输出
3. **敏感信息**：避免记录密码、API 密钥等敏感数据
4. **线程安全**：Python logging 模块是线程安全的，无需额外处理
5. **日志清理**：配置适当的 `max_bytes` 和 `backup_count` 避免磁盘占满

## 总结

通过本次实现，GoAfar 项目现在拥有：

- 统一的日志接口
- 灵活的配置方式
- 完善的文档和示例
- 向后兼容的迁移路径

开发团队可以基于此基础继续推进项目的日志标准化工作。
