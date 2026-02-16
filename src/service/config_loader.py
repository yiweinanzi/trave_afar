"""
增强的配置加载器，支持环境变量覆盖和配置验证。

用法:
    from service.config_loader import load_config, get_env_config

    # 基础加载
    config = load_config()

    # 加载特定环境配置
    config = load_config(env="prod")

    # 使用环境变量覆盖
    # GOAFAR_LLM_USE_GPU=false python app.py
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union

import yaml


T = TypeVar("T")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)

# 兼容历史环境变量命名（与 runtime.yaml 中 ${GOAFAR_...} 占位符一致）
ENV_KEY_ALIASES: Dict[str, List[str]] = {
    "embedding_model": ["embedding", "model_path"],
    "embedding_fallback": ["embedding", "fallback_model"],
    "embedding_gpu": ["embedding", "use_gpu"],
    "embedding_auto_build": ["embedding", "auto_build_if_missing"],
    "embedding_quant": ["embedding", "quantization"],
    "rerank_enabled": ["rerank", "enabled"],
    "rerank_template": ["rerank", "use_template"],
    "rerank_model": ["rerank", "use_reranker_model"],
    "rerank_path": ["rerank", "qwen_reranker_path"],
    "osrm_url": ["planner", "osrm_url"],
    "llm_enabled": ["llm", "enabled"],
    "llm_model": ["llm", "qwen_model"],
    "llm_gpu": ["llm", "use_gpu"],
    "llm_lora": ["llm", "use_lora"],
    "llm_lora_path": ["llm", "lora_path"],
    "llm_max_tokens": ["llm", "max_new_tokens"],
    "llm_8bit": ["llm", "load_in_8bit"],
    "llm_4bit": ["llm", "load_in_4bit"],
    "log_max_size": ["logging", "max_file_size_mb"],
}


def substitute_env_vars(value: str) -> str:
    """
    替换字符串中的环境变量引用。

    支持格式:
        ${VAR_NAME}
        ${VAR_NAME:default_value}

    Args:
        value: 可能包含环境变量引用的字符串

    Returns:
        替换后的字符串
    """
    if not isinstance(value, str):
        return value

    pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        default_value = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(var_name, default_value)

    return re.sub(pattern, replacer, value)


def substitute_env_recursive(data: Any) -> Any:
    """
    递归地替换数据结构中的所有环境变量引用。

    Args:
        data: 要处理的数据（dict, list, 或原始值）

    Returns:
        处理后的数据
    """
    if isinstance(data, dict):
        return {k: substitute_env_recursive(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [substitute_env_recursive(item) for item in data]
    elif isinstance(data, str):
        return substitute_env_vars(data)
    else:
        return data


def apply_env_overrides(data: Dict[str, Any], prefix: str = "GOAFAR_") -> Dict[str, Any]:
    """
    从环境变量应用配置覆盖。

    环境变量命名规则: PREFIX_SECTION_KEY
    例如: GOAFAR_LLM_USE_GPU=false

    Args:
        data: 原始配置字典
        prefix: 环境变量前缀

    Returns:
        应用覆盖后的配置字典
    """
    result = dict(data)
    key_index = _build_env_key_index(result)
    ignored_keys: List[str] = []

    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue

        # 例: GOAFAR_RERANK_USE_RERANKER_MODEL -> rerank_use_reranker_model
        normalized = env_key[len(prefix):].lower()

        # 转换值类型
        parsed_value = parse_env_value(env_value)

        # 优先按现有配置键精确匹配，避免 use_reranker_model 被拆成 use.reranker.model
        key_path = key_index.get(normalized)

        # 兼容历史命名（例如 GOAFAR_LLM_MODEL -> llm.qwen_model）
        if key_path is None:
            alias_path = ENV_KEY_ALIASES.get(normalized)
            if alias_path is not None and _path_exists(result, alias_path):
                key_path = alias_path

        if key_path is not None:
            _set_nested_value(result, key_path, parsed_value)
            continue

        # 回退到旧逻辑（仅在路径存在时覆盖，避免创建无效键污染配置）
        fallback_path = normalized.split("_")
        if _path_exists(result, fallback_path):
            _set_nested_value(result, fallback_path, parsed_value)
            continue

        # 忽略未知键：这类变量通常已通过 ${GOAFAR_...} 在 YAML 中替换生效
        ignored_keys.append(env_key)

    if ignored_keys:
        LOGGER.debug(
            "Ignored %d unknown environment override keys: %s",
            len(ignored_keys),
            ", ".join(sorted(ignored_keys)),
        )

    return result


def _build_env_key_index(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    构建环境变量键到配置路径的索引。

    例如:
      rerank.use_reranker_model -> rerank_use_reranker_model
    """
    index: Dict[str, List[str]] = {}

    def walk(node: Dict[str, Any], prefix: List[str]) -> None:
        for key, value in node.items():
            path = prefix + [str(key)]
            normalized = "_".join(p.lower() for p in path)
            index[normalized] = path
            if isinstance(value, dict):
                walk(value, path)

    walk(data, [])
    return index


def _set_nested_value(data: Dict[str, Any], key_path: List[str], value: Any) -> None:
    """按路径写入嵌套字典值，不存在的中间节点自动创建。"""
    current = data
    for part in key_path[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[key_path[-1]] = value


def _path_exists(data: Dict[str, Any], key_path: List[str]) -> bool:
    """检查嵌套路径是否存在。"""
    current: Any = data
    for part in key_path:
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def parse_env_value(value: str) -> Any:
    """
    将环境变量字符串值解析为适当的Python类型。

    Args:
        value: 环境变量值字符串

    Returns:
        解析后的值
    """
    # 布尔值
    if value.lower() in ("true", "yes", "1", "on"):
        return True
    if value.lower() in ("false", "no", "0", "off"):
        return False

    # None/null
    if value.lower() in ("null", "none", ""):
        return None

    # 尝试数字
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    return value


def _coerce_scalar(value: Any) -> Any:
    """将字符串配置值转换为基础类型。"""
    if not isinstance(value, str):
        return value

    raw = value.strip()
    lowered = raw.lower()

    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "none"):
        return None

    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return value


def _coerce_recursive(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _coerce_recursive(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_coerce_recursive(v) for v in data]
    return _coerce_scalar(data)


def load_yaml_with_env(
    config_path: Union[str, Path],
    env: Optional[str] = None,
    apply_overrides: bool = True
) -> Dict[str, Any]:
    """
    加载YAML配置文件，支持环境变量替换和覆盖。

    Args:
        config_path: 配置文件路径
        env: 环境名称 (dev/staging/prod)，会应用environments.{env}的覆盖
        apply_overrides: 是否应用环境变量覆盖

    Returns:
        配置字典
    """
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / config_path

    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # 替换环境变量引用
    data = substitute_env_recursive(data)

    # 应用环境特定配置
    if env:
        env_config = data.get("environments", {}).get(env, {})
        if env_config:
            data = deep_merge(data, env_config)

    # 移除 environments 节点以避免污染
    data.pop("environments", None)

    # 应用环境变量覆盖（优先级最高）
    if apply_overrides:
        data = apply_env_overrides(data)

    # 统一类型转换（例如 ${...} 展开后得到的字符串数字/布尔）
    data = _coerce_recursive(data)

    return data


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合并两个字典。

    Args:
        base: 基础字典
        override: 覆盖字典

    Returns:
        合并后的字典
    """
    result = dict(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


# =============================================================================
# 配置验证
# =============================================================================

@dataclass
class ValidationRule:
    """配置验证规则"""
    field_path: str  # 例如: "llm.max_new_tokens"
    validator: Callable[[Any], bool]
    error_message: str


class ConfigValidator:
    """配置验证器"""

    def __init__(self):
        self.rules: List[ValidationRule] = []
        self._setup_default_rules()

    def _setup_default_rules(self):
        """设置默认验证规则"""
        default_rules = [
            ValidationRule(
                "llm.max_new_tokens",
                lambda x: isinstance(x, int) and 1 <= x <= 4096,
                "llm.max_new_tokens must be between 1 and 4096"
            ),
            ValidationRule(
                "llm.temperature",
                lambda x: isinstance(x, (int, float)) and 0.0 <= x <= 2.0,
                "llm.temperature must be between 0.0 and 2.0"
            ),
            ValidationRule(
                "api.port",
                lambda x: isinstance(x, int) and 1 <= x <= 65535,
                "api.port must be between 1 and 65535"
            ),
            ValidationRule(
                "recall.semantic_topk",
                lambda x: isinstance(x, int) and x > 0,
                "recall.semantic_topk must be positive"
            ),
            ValidationRule(
                "embedding.batch_size",
                lambda x: isinstance(x, int) and 1 <= x <= 256,
                "embedding.batch_size must be between 1 and 256"
            ),
        ]
        for rule in default_rules:
            self.add_rule(rule)

    def add_rule(self, rule: ValidationRule) -> None:
        """添加验证规则"""
        self.rules.append(rule)

    def validate(self, config: Dict[str, Any]) -> List[str]:
        """
        验证配置。

        Args:
            config: 配置字典

        Returns:
            错误消息列表，空列表表示验证通过
        """
        errors = []

        for rule in self.rules:
            value = self._get_nested_value(config, rule.field_path)
            if value is not None:
                try:
                    if not rule.validator(value):
                        errors.append(f"{rule.field_path}: {rule.error_message}")
                except Exception:
                    errors.append(f"{rule.field_path}: validation error")

        return errors

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """获取嵌套值"""
        parts = path.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class APIConfig:
    """API服务配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "*"
    cors_methods: str = "GET,POST,OPTIONS"
    cors_headers: str = "*"
    rate_limit_enabled: bool = False
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    api_key_required: bool = False
    api_key_header: str = "X-API-Key"


@dataclass
class HealthConfig:
    """健康检查配置"""
    enabled: bool = True
    model_load_timeout: int = 300
    service_check_timeout: int = 10
    readyz_deep_check: bool = True


@dataclass
class CacheConfig:
    """缓存配置"""
    enabled: bool = True
    embedding_ttl: int = 86400
    routing_ttl: int = 3600
    max_items: int = 10000


@dataclass
class MonitoringConfig:
    """监控配置"""
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    prometheus_path: str = "/metrics"
    tracing_enabled: bool = False
    tracing_endpoint: Optional[str] = None
    tracing_sample_rate: float = 0.1


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    log_dir: str = "logs"
    max_file_size_mb: int = 10
    backup_count: int = 5
    format: str = "text"  # text or json
    sanitize_secrets: bool = True


@dataclass
class EnvironmentConfig:
    """环境配置"""
    mode: str = "dev"
    instance_id: str = "goafar-1"
    region: str = "default"


# =============================================================================
# 统一配置加载函数
# =============================================================================

def load_config(
    config_path: str = "configs/runtime.yaml",
    env: Optional[str] = None,
    validate: bool = True
) -> Dict[str, Any]:
    """
    加载统一配置。

    Args:
        config_path: 配置文件路径
        env: 环境名称 (dev/staging/prod)，默认从环境变量GOAFAR_ENV_MODE读取
        validate: 是否验证配置

    Returns:
        配置字典
    """
    # 从环境变量获取环境
    if env is None:
        env = os.environ.get("GOAFAR_ENV_MODE", "dev")

    # 加载配置
    config = load_yaml_with_env(config_path, env=env)

    # 验证配置
    if validate:
        validator = ConfigValidator()
        errors = validator.validate(config)
        if errors:
            raise ValueError(f"配置验证失败:\n" + "\n".join(f"  - {e}" for e in errors))

    return config


def get_env_config() -> str:
    """获取当前环境名称"""
    return os.environ.get("GOAFAR_ENV_MODE", "dev")


def is_production() -> bool:
    """检查是否为生产环境"""
    return get_env_config() == "prod"


def is_development() -> bool:
    """检查是否为开发环境"""
    return get_env_config() == "dev"


def config_section(section: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    获取配置的特定部分。

    Args:
        section: 部分名称
        config: 配置字典，如果为None则自动加载

    Returns:
        该部分的配置字典
    """
    if config is None:
        config = load_config()

    return config.get(section, {})


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("测试配置加载器")
    print("=" * 60)

    # 测试加载
    config = load_config()
    print(f"\n环境: {config.get('environment', {}).get('mode', 'unknown')}")
    print(f"API端口: {config.get('api', {}).get('port', 8000)}")
    print(f"日志级别: {config.get('runtime', {}).get('log_level', 'INFO')}")

    # 测试环境变量覆盖
    os.environ["GOAFAR_API_PORT"] = "9000"
    os.environ["GOAFAR_LLM_TEMPERATURE"] = "0.5"
    config_override = load_config()
    print(f"\n应用覆盖后API端口: {config_override.get('api', {}).get('port', 8000)}")
    print(f"应用覆盖后温度: {config_override.get('llm', {}).get('temperature', 0.7)}")

    # 测试特定环境配置
    config_prod = load_config(env="prod")
    print(f"\n生产环境日志级别: {config_prod.get('runtime', {}).get('log_level', 'INFO')}")
    print(f"生产环境工作进程: {config_prod.get('runtime', {}).get('workers', 1)}")
