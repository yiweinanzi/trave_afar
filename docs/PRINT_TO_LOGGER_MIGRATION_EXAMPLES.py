"""
GoAfar 项目中 print 语句替换为 logger 的示例

这些示例展示了如何将项目中的 print 语句替换为结构化日志。
"""

# =============================================================================
# 示例 1: 简单的 print 替换为 logger.info
# =============================================================================

# === 之前 ===
# print("开始加载数据...")
# print(f"加载了 {count} 条记录")

# === 之后 ===
from src.utils import get_logger

logger = get_logger(__name__)

logger.info("开始加载数据...")
logger.info(f"加载了 {count} 条记录")


# =============================================================================
# 示例 2: 警告信息替换为 logger.warning
# =============================================================================

# === 之前 ===
# print("警告: 配置文件缺失，使用默认值")
# print(f"⚠️ GPU 不可用，将使用 CPU")

# === 之后 ===
from src.utils import get_logger

logger = get_logger(__name__)

logger.warning("配置文件缺失，使用默认值")
logger.warning("GPU 不可用，将使用 CPU")


# =============================================================================
# 示例 3: 错误信息替换为 logger.error
# =============================================================================

# === 之前 ===
# print(f"错误: 无法加载模型 {model_path}")
# print("模型加载失败，将使用规则回退")

# === 之后 ===
from src.utils import get_logger

logger = get_logger(__name__)

logger.error(f"无法加载模型 {model_path}")
logger.error("模型加载失败，将使用规则回退")


# =============================================================================
# 示例 4: 调试信息替换为 logger.debug
# =============================================================================

# === 之前 ===
# print(f"处理第 {i}/{total} 条记录")

# === 之后 ===
from src.utils import get_logger

logger = get_logger(__name__)

logger.debug(f"处理第 {i}/{total} 条记录")


# =============================================================================
# 示例 5: 使用异常记录 (logger.exception)
# =============================================================================

# === 之前 ===
# try:
#     load_data()
# except Exception as e:
#     print(f"加载数据失败: {e}")

# === 之后 ===
from src.utils import get_logger

logger = get_logger(__name__)

try:
    load_data()
except Exception as e:
    logger.exception("加载数据失败")  # 自动包含堆栈跟踪


# =============================================================================
# 示例 6: 格式化日志的正确方式
# =============================================================================

# === 之前 ===
# print(f"Epoch {epoch+1}/{epochs}: Loss {loss:.4f}")

# === 之后 ===
from src.utils import get_logger

logger = get_logger(__name__)

logger.info("Epoch %d/%d: Loss %.4f", epoch + 1, epochs, loss)
# 或者
logger.info("Epoch %d/%d: Loss %.4f" % (epoch + 1, epochs, loss))


# =============================================================================
# 示例 7: 在类中使用 logger
# =============================================================================

# === 之前 ===
# class MyModel:
#     def __init__(self):
#         print("初始化模型")
#         self.ready = True
#
#     def train(self, data):
#         print(f"训练 {len(data)} 条记录")
#         return "done"

# === 之后 ===
from src.utils import get_logger

class MyModel:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.logger.info("初始化模型")
        self.ready = True

    def train(self, data):
        self.logger.info("训练 %d 条记录", len(data))
        return "done"


# =============================================================================
# 示例 8: 带进度的日志
# =============================================================================

# === 之前 ===
# for i, item in enumerate(items):
#     if i % 100 == 0:
#         print(f"处理进度: {i}/{len(items)}")

# === 之后 ===
from src.utils import get_logger

logger = get_logger(__name__)

total = len(items)
for i, item in enumerate(items):
    if i % 100 == 0:
        logger.debug("处理进度: %d/%d", i, total)
logger.info("处理完成: %d 条记录", total)


# =============================================================================
# 示例 9: 在 API 服务中使用
# =============================================================================

# === 之前 ===
# print("API 服务器启动")
# print(f"监听地址: {host}:{port}")
# print("警告: CORS 已启用")

# === 之后 ===
from src.utils import get_logger

logger = get_logger(__name__)

def start_server(host: str, port: int, cors_enabled: bool):
    logger.info("API 服务器启动")
    logger.info("监听地址: %s:%d", host, port)
    if cors_enabled:
        logger.warning("CORS 已启用")


# =============================================================================
# 示例 10: 在脚本入口点使用
# =============================================================================

# === 之前 ===
# def main():
#     print("=" * 60)
#     print("训练脚本")
#     print("=" * 60)
#     print("开始训练...")
#     print("训练完成！")

# === 之后 ===
from src.utils import get_logger, configure_logging_from_config
from src.service.config import load_runtime_config

logger = get_logger(__name__)

def main():
    cfg = load_runtime_config()
    configure_logging_from_config(cfg.log)

    logger.info("=" * 60)
    logger.info("训练脚本")
    logger.info("=" * 60)

    logger.info("开始训练...")
    # ... 训练代码 ...
    logger.info("训练完成！")


# =============================================================================
# 示例 11: 从配置文件加载日志设置
# =============================================================================

# === 使用配置文件 ===
from src.utils import configure_logging_from_config
from src.service.config import load_runtime_config

# 加载完整配置
cfg = load_runtime_config()

# 使用 log 配置初始化日志
logger = configure_logging_from_config(cfg.log)
logger.info("日志系统从配置文件初始化")


# =============================================================================
# 示例 12: 环境变量控制日志级别
# =============================================================================

# === 通过环境变量控制 ===
# 在终端执行:
# export GOAFAR_LOG_LEVEL=DEBUG
# export GOAFAR_LOG_DIR=logs/debug
# python my_script.py

# 在代码中:
from src.utils import get_logger

logger = get_logger(__name__)  # 自动读取 GOAFAR_LOG_LEVEL
logger.debug("这条调试信息只在 DEBUG 级别显示")


# =============================================================================
# 示例 13: 不同模块使用不同的 logger
# =============================================================================

# === 模块特定日志 ===
from src.utils import get_logger

api_logger = get_logger("src.service.api")
ranking_logger = get_logger("src.ranking.matcher")
training_logger = get_logger("src.training.sft")

api_logger.info("API 请求: GET /pois")
ranking_logger.debug("相似度计算: 0.85")
training_logger.info("Epoch 1/20: Loss 2.3456")


# =============================================================================
# 示例 14: 结构化日志（JSON 格式）
# =============================================================================

# === 启用 JSON 日志 ===
from src.utils import setup_logger, JSONFormatter
import logging
import sys

# 创建 JSON 格式化日志器
json_logger = setup_logger(
    "my_app",
    level="INFO",
    console_output=True,
    use_colors=False,  # JSON 不使用颜色
)

# 使用 JSON 格式化器
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
json_logger.handlers = [handler]

# 记录日志
json_logger.info({"event": "user_login", "user_id": 12345})
json_logger.error({"event": "api_error", "code": 500, "message": "Internal error"})


# =============================================================================
# 示例 15: 使用 Extra 字段添加上下文
# =============================================================================

# === 添加额外上下文 ===
from src.utils import get_logger

logger = get_logger(__name__)

# 使用 extra_fields 添加结构化数据
class LogAdapter:
    """添加额外字段的日志适配器"""
    def __init__(self, logger, extra):
        self.logger = logger
        self.extra = extra

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, extra=self.extra, **kwargs)
    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, extra=self.extra, **kwargs)

# 为每个请求创建带上下文的 logger
def handle_request(request_id, user_id):
    log_adapter = LogAdapter(logger, {
        "request_id": request_id,
        "user_id": user_id
    })
    log_adapter.info("处理请求")
    log_adapter.error("请求处理失败")


# =============================================================================
# 示例 16: 在异步代码中使用 logger
# =============================================================================

# === 异步上下文 ===
import asyncio
from src.utils import get_logger

logger = get_logger(__name__)

async def async_task():
    logger.info("异步任务开始")
    await asyncio.sleep(1)
    logger.info("异步任务完成")

async def main():
    logger.info("启动异步任务")
    await async_task()
    logger.info("所有任务完成")


# =============================================================================
# 示例 17: 关键操作使用适当的日志级别
# =============================================================================

# === 关键操作日志指南 ===
from src.utils import get_logger

logger = get_logger(__name__)

def critical_operation(data):
    # DEBUG: 详细诊断信息
    logger.debug("输入数据: %s", data)

    # INFO: 正常流程
    logger.info("开始处理关键操作")

    # WARNING: 可恢复的问题
    if not data:
        logger.warning("输入数据为空，使用默认配置")

    # ERROR: 操作失败
    try:
        result = process(data)
    except Exception as e:
        logger.error("处理失败: %s", str(e))
        raise

    # CRITICAL: 无法继续运行
    if not result:
        logger.critical("关键操作失败，系统可能无法正常运行")

    return result


# =============================================================================
# 示例 18: 避免日志循环引用
# =============================================================================

# === 错误方式 ===
# from src.utils import get_logger
# logger = get_logger(__name__)  # 如果在 __init__.py 中导入，可能导致循环

# === 正确方式 ===
# 方法 1: 在函数内部获取 logger
def my_function():
    from src.utils import get_logger
    logger = get_logger(__name__)  # 在运行时获取
    logger.info("函数执行")

# 方法 2: 使用 __name__ 作为字符串
from src.utils import get_logger
logger = get_logger("my.module.name")  # 使用字符串名称，避免循环


if __name__ == "__main__":
    # 运行所有示例
    print("\n=== 运行日志替换示例 ===\n")
