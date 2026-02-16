"""
统一日志系统

提供结构化日志功能，支持：
- 控制台和文件双输出
- 日志轮转（RotatingFileHandler）
- 日志级别配置
- 彩色控制台输出
- 从配置文件加载
- JSON格式输出（可选）
"""
from __future__ import annotations

import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional, Union


# ANSI 颜色代码
class LogColors:
    """日志颜色配置"""
    DEBUG = "\033[36m"     # 青色
    INFO = "\033[32m"      # 绿色
    WARNING = "\033[33m"   # 黄色
    ERROR = "\033[31m"     # 红色
    CRITICAL = "\033[35m"  # 紫色
    RESET = "\033[0m"      # 重置
    BOLD = "\033[1m"       # 加粗


class JSONFormatter(logging.Formatter):
    """JSON格式化日志输出，用于结构化日志分析"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        # 添加异常信息（如果有）
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        # 添加额外字段
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        return json.dumps(log_entry, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """彩色控制台日志格式化器"""

    COLORS = {
        logging.DEBUG: LogColors.DEBUG,
        logging.INFO: LogColors.INFO,
        logging.WARNING: LogColors.WARNING,
        logging.ERROR: LogColors.ERROR,
        logging.CRITICAL: LogColors.CRITICAL,
    }

    def __init__(self, fmt: str | None = None, datefmt: str | None = None, use_colors: bool = True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors:
            color = self.COLORS.get(record.levelno, "")
            record.levelname = f"{color}{record.levelname}{LogColors.RESET}"
            # 为记录名称添加颜色
            record.name = f"{LogColors.BOLD}{record.name}{LogColors.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "goafar",
    level: str | int = "INFO",
    log_file: Optional[str | Path] = None,
    log_dir: Optional[str | Path] = None,
    console_output: bool = True,
    file_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    use_colors: bool = True,
    fmt: Optional[str] = None,
) -> logging.Logger:
    """
    设置并返回一个配置好的日志记录器

    Args:
        name: 日志记录器名称
        level: 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
        log_file: 日志文件路径（如果指定，优先使用此路径）
        log_dir: 日志目录（如果未指定log_file，则在此目录下创建 {name}.log）
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        max_bytes: 单个日志文件最大字节数
        backup_count: 保留的日志备份文件数量
        use_colors: 是否使用彩色输出
        fmt: 自定义日志格式

    Returns:
        配置好的 Logger 实例

    Examples:
        >>> logger = setup_logger("my_module", level="DEBUG")
        >>> logger.info("这是一条信息")
        >>> logger.debug("调试信息")

        >>> # 使用自定义日志目录
        >>> logger = setup_logger("api", log_dir="logs/app")
    """
    # 解析日志级别
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # 获取或创建日志记录器
    logger = logging.getLogger(name)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False  # 不传播到父日志记录器

    # 默认日志格式
    if fmt is None:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = ColoredFormatter(
            fmt=fmt,
            datefmt="%Y-%m-%d %H:%M:%S",
            use_colors=use_colors
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # 文件处理器
    if file_output:
        # 确定日志文件路径
        if log_file is None:
            if log_dir is None:
                # 默认使用项目根目录下的 logs 目录
                project_root = Path(__file__).resolve().parents[2]
                log_dir = project_root / "logs"
            else:
                log_dir = Path(log_dir)

            # 确保日志目录存在
            log_dir.mkdir(parents=True, exist_ok=True)

            log_file = log_dir / f"{name}.log"
        else:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)

        # 创建轮转文件处理器
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(level)

        # 文件日志不使用颜色
        file_formatter = logging.Formatter(
            fmt=fmt,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None, level: str | int | None = None) -> logging.Logger:
    """
    获取一个配置好的日志记录器

    这是 setup_logger 的简化接口，适合在模块中使用。
    如果指定了 level，会更新日志记录器的级别。

    Args:
        name: 日志记录器名称（默认使用调用者的模块名）
        level: 日志级别

    Returns:
        Logger 实例

    Examples:
        >>> from src.utils import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("模块初始化完成")

        >>> # 指定日志级别
        >>> logger = get_logger("api", level="DEBUG")
    """
    if name is None:
        # 获取调用者的模块名
        frame = sys._getframe(1)
        name = frame.f_globals.get("__name__", "goafar")

    # 从环境变量读取日志级别
    if level is None:
        env_level = os.getenv("GOAFAR_LOG_LEVEL", "INFO")
        level = env_level

    logger = setup_logger(name, level=level)
    return logger


def get_logger_level_from_env() -> str:
    """
    从环境变量获取日志级别

    支持 GOAFAR_LOG_LEVEL 或 LOG_LEVEL 环境变量

    Returns:
        日志级别字符串（DEBUG, INFO, WARNING, ERROR, CRITICAL）
    """
    return os.getenv("GOAFAR_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO"))


def configure_logging_from_config(config: Dict[str, Any] = None, **kwargs) -> logging.Logger:
    """
    从配��字典或关键字参数配置日志

    Args:
        config: 配置字典（来自 config.py）
        **kwargs: 直接配置参数（覆盖 config）

    Returns:
        配置好的根日志记录器

    Examples:
        >>> # 从配置对象
        >>> cfg = load_runtime_config()
        >>> logger = configure_logging_from_config(cfg.log)

        >>> # 从字典
        >>> logger = configure_logging_from_config({
        ...     "level": "DEBUG",
        ...     "log_dir": "logs/app"
        ... })

        >>> # 使用关键字参数
        >>> logger = configure_logging_from_config(level="INFO", console_output=False)
    """
    if config is None:
        config = {}

    # 合并配置：kwargs > config
    log_config = {
        "level": config.get("level", "INFO"),
        "log_dir": config.get("log_dir", None),
        "log_file": config.get("log_file", None),
        "console_output": config.get("console_output", True),
        "file_output": config.get("file_output", True),
        "max_bytes": config.get("max_bytes", 10 * 1024 * 1024),
        "backup_count": config.get("backup_count", 5),
        "use_colors": config.get("use_colors", True),
        "fmt": config.get("format", None),
    }
    log_config.update(kwargs)

    return setup_logger("goafar", **log_config)


# 默认日志记录器（向后兼容）
_default_logger: Optional[logging.Logger] = None


def log(level: str, message: str, *args, **kwargs):
    """
    使用默认日志记录器记录日志

    Args:
        level: 日志级别（debug, info, warning, error, critical）
        message: 日志消息
        *args: 格式化参数
        **kwargs: 额外的日志参数
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = get_logger()

    log_func = getattr(_default_logger, level.lower(), _default_logger.info)
    log_func(message, *args, **kwargs)


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("测试统一日志系统")
    print("=" * 60)

    # 测试各种日志级别
    test_logger = setup_logger(
        "test",
        level="DEBUG",
        log_dir="logs"
    )

    test_logger.debug("这是一条调试信息")
    test_logger.info("这是一条普通信息")
    test_logger.warning("这是一条警告信息")
    test_logger.error("这是一条错误信息")
    test_logger.critical("这是一条严重错误信息")

    # 测试日志格式化
    test_logger.info("处理 %d 条记录，耗时 %.2f 秒", 1000, 3.14)

    # 测试多模块日志
    api_logger = get_logger("api.server")
    api_logger.info("API 服务器启动")

    pipeline_logger = get_logger("service.pipeline")
    pipeline_logger.info("推荐管道初始化完成")

    print("\n日志文件保存在: logs/test.log")
