"""
GoAfar 统一日志系统测试

验证日志系统是否正常工作：
- 控制台彩色输出
- 文件输出和轮转
- 不同级别日志
- 从配置加载
- JSON 格式输出
"""
from pathlib import Path
import sys

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import (
    get_logger,
    setup_logger,
    get_logger_level_from_env,
    configure_logging_from_config,
    LogColors,
    ColoredFormatter,
    JSONFormatter,
)


def test_basic_logging():
    """测试基础日志功能"""
    print("\n" + "=" * 60)
    print("测试 1: 基础日志功能")
    print("=" * 60)

    logger = setup_logger(
        "test_basic",
        level="DEBUG",
        console_output=True,
        file_output=False,  # 测试时不写文件
        use_colors=True
    )

    # 测试各级别日志
    logger.debug("这是一条调试信息 - 应该显示")
    logger.info("这是一条普通信息 - 应该显示")
    logger.warning("这是一条警告信息 - 应该显示")
    logger.error("这是一条错误信息 - 应该显示")
    logger.critical("这是一条严重错误信息 - 应该显示")

    print("\n预期输出：")
    print("- 彩色：DEBUG(青色), INFO(绿色), WARNING(黄色), ERROR(红色), CRITICAL(紫色)")
    print("- 格式：时间戳 | 级别 | 模块名 | 消息")


def test_file_logging():
    """测试文件日志功能"""
    print("\n" + "=" * 60)
    print("测试 2: 文件日志和轮转")
    print("=" * 60)

    log_dir = Path("logs/test")
    logger = setup_logger(
        "test_file",
        level="INFO",
        log_dir=log_dir,
        console_output=True,
        file_output=True,
        max_bytes=1024,  # 1KB 用于快速测试
        backup_count=3,
        use_colors=True
    )

    # 写入足够多的日志以触发轮转
    for i in range(10):
        logger.info(f"测试日志条目 # {i+1} - 这应该写入文件")

    # 检查文件
    log_file = log_dir / "test_file.log"
    if log_file.exists():
        file_size = log_file.stat().st_size
        print(f"\n日志文件已创建: {log_file}")
        print(f"文件大小: {file_size} 字节")
        print(f"配置的最大大小: 1024 字节")

        # 列出备份文件
        backup_files = sorted(log_dir.glob("test_file.log.*"))
        if backup_files:
            print(f"备份文件: {[f.name for f in backup_files]}")


def test_json_logging():
    """测试 JSON 格式日志"""
    print("\n" + "=" * 60)
    print("测试 3: JSON 格式输出")
    print("=" * 60)

    logger = setup_logger(
        "test_json",
        level="INFO",
        console_output=True,
        file_output=False,
        use_colors=False  # JSON 不使用颜色
    )

    # 替换为 JSON formatter
    import logging
    json_handler = logging.StreamHandler(sys.stdout)
    json_handler.setFormatter(JSONFormatter())
    logger.handlers = [json_handler]

    # 记录日志
    logger.info({"event": "user_login", "user_id": 12345})
    logger.warning({"event": "api_rate_limit", "limit": 100})
    logger.error({"event": "payment_failed", "amount": 99.99})


def test_color_formatter():
    """测试彩色格式化器"""
    print("\n" + "=" * 60)
    print("测试 4: 彩色格式化")
    print("=" * 60)

    # 创建不同颜色的测试消息
    levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    colors = [LogColors.DEBUG, LogColors.INFO, LogColors.WARNING,
              LogColors.ERROR, LogColors.CRITICAL]

    for level, color in zip(levels, colors):
        colored = f"{color}{level}{LogColors.RESET}"
        print(f"  {level} 颜色: {colored} -> 实际效果")

    print("\n在日志中使用这些颜色:")
    logger = get_logger("test_colors", level="DEBUG")
    logger.debug("DEBUG 消息 - 青色")
    logger.info("INFO 消息 - 绿色")
    logger.warning("WARNING 消息 - 黄色")
    logger.error("ERROR 消息 - 红色")
    logger.critical("CRITICAL 消息 - 紫色")


def test_get_logger():
    """测试 get_logger 简化接口"""
    print("\n" + "=" * 60)
    print("测试 5: get_logger() 简化接口")
    print("=" * 60)

    # 使用模块名
    logger1 = get_logger(__name__)
    logger1.info("从模块名获取的 logger")

    # 使用自定义名称
    logger2 = get_logger("my.module", level="DEBUG")
    logger2.debug("自定义命名的 logger - DEBUG 消息")


def test_env_level():
    """测试从环境变量读取级别"""
    print("\n" + "=" * 60)
    print("测试 6: 环境变量配置")
    print("=" * 60)

    import os

    # 设置环境变量
    os.environ["GOAFAR_LOG_LEVEL"] = "DEBUG"
    level = get_logger_level_from_env()
    print(f"从环境变量读取的级别: {level}")

    # 使用该级别
    logger = get_logger("test_env", level=level)
    logger.debug("DEBUG 消息应该显示")


def test_config_based():
    """测试从配置字典加载"""
    print("\n" + "=" * 60)
    print("测试 7: 从配置字典加载")
    print("=" * 60)

    config = {
        "level": "DEBUG",
        "console_output": True,
        "file_output": False,
        "use_colors": True,
    }

    logger = configure_logging_from_config(config)
    logger.info("从配置字典初始化的 logger")
    logger.debug("配置的 DEBUG 消息应该显示")


def test_module_usage():
    """测试在类/模块中的使用方式"""
    print("\n" + "=" * 60)
    print("测试 8: 模块/类中的使用示例")
    print("=" * 60)

    # 模拟类使用
    class TestService:
        def __init__(self):
            self.logger = get_logger(__name__)
            self.logger.info("TestService 初始化")

        def process(self, data):
            self.logger.debug(f"处理数据: {data}")
            self.logger.info("处理完成")
            return "result"

        def critical_operation(self):
            self.logger.warning("尝试关键操作...")
            try:
                # 模拟失败
                raise ValueError("操作失败")
            except Exception as e:
                self.logger.error(f"操作失败: {e}")
                # 使用 exception 记录堆栈
                self.logger.exception("完整异常堆栈:")

    service = TestService()
    result = service.process("test_data")
    service.critical_operation()


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("GoAfar 统一日志系统测试套件")
    print("=" * 60)

    test_basic_logging()
    test_file_logging()
    test_json_logging()
    test_color_formatter()
    test_get_logger()
    test_env_level()
    test_config_based()
    test_module_usage()

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
    print("\n检查日志目录: logs/")
    print("查看文件日志以验证输出和轮转")


if __name__ == "__main__":
    run_all_tests()
