"""
Prometheus 指标导出模块。

提供结构化的指标收集和导出功能。
"""
from __future__ import annotations

import time
from functools import wraps
from typing import Callable, Dict, List, Optional

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Summary,
        generate_latest,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    # 如果prometheus_client不可用，使用简单的实现
    PROMETHEUS_AVAILABLE = False

    class Counter:
        def __init__(self, name, documentation, labelnames=()):
            self.name = name
            self.documentation = documentation
            self.labelnames = labelnames
            self._value = 0.0
            self._labels: Dict = {}

        def labels(self, **kwargs):
            key = tuple(sorted(kwargs.items()))
            if key not in self._labels:
                self._labels[key] = 0.0
            return _LabelWrapper(self._labels, key)

        def inc(self, amount=1):
            self._value += amount

        def collect(self):
            pass

    class _LabelWrapper:
        def __init__(self, store, key):
            self._store = store
            self._key = key

        def inc(self, amount=1):
            self._store[self._key] = self._store.get(self._key, 0.0) + amount

    class Gauge:
        def __init__(self, name, documentation, labelnames=()):
            self.name = name
            self.documentation = documentation
            self.labelnames = labelnames
            self._value = 0.0
            self._labels: Dict = {}

        def labels(self, **kwargs):
            key = tuple(sorted(kwargs.items()))
            if key not in self._labels:
                self._labels[key] = 0.0
            return _GaugeLabelWrapper(self._labels, key)

        def set(self, value):
            self._value = value

        def inc(self, amount=1):
            self._value += amount

        def dec(self, amount=1):
            self._value -= amount

        def collect(self):
            pass

    class _GaugeLabelWrapper:
        def __init__(self, store, key):
            self._store = store
            self._key = key

        def set(self, value):
            self._store[self._key] = value

        def inc(self, amount=1):
            self._store[self._key] = self._store.get(self._key, 0.0) + amount

        def dec(self, amount=1):
            self._store[self._key] = self._store.get(self._key, 0.0) - amount

    class Histogram:
        def __init__(self, name, documentation, labelnames=(), buckets=()):
            self.name = name
            self.documentation = documentation
            self.labelnames = labelnames
            self._values: List = []

        def labels(self, **kwargs):
            return self

        def observe(self, value):
            self._values.append(value)

        def collect(self):
            pass

    class Summary:
        def __init__(self, name, documentation, labelnames=()):
            self.name = name
            self.documentation = documentation
            self.labelnames = labelnames
            self._values: List = []

        def labels(self, **kwargs):
            return self

        def observe(self, value):
            self._values.append(value)

        def collect(self):
            pass

    def generate_latest(registry):
        return b""

    class CollectorRegistry:
        pass


# =============================================================================
# GoAfar 指标定义
# =============================================================================

registry = CollectorRegistry()

# 请求计数器
request_total = Counter(
    "goafar_requests_total",
    "Total number of requests",
    ["endpoint", "method", "status"],
)

request_success = Counter(
    "goafar_requests_success_total",
    "Number of successful requests",
    ["endpoint"],
)

request_error = Counter(
    "goafar_requests_error_total",
    "Number of failed requests",
    ["endpoint", "error_type"],
)

# 请求延迟
request_duration = Histogram(
    "goafar_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# Pipeline 指标
pipeline_recommendation_duration = Histogram(
    "goafar_pipeline_recommendation_duration_seconds",
    "Recommendation pipeline duration",
    ["stage"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

pipeline_candidates_gauge = Gauge(
    "goafar_pipeline_candidates",
    "Number of candidates in pipeline",
    ["stage"],
)

# 模型指标
model_load_time = Histogram(
    "goafar_model_load_time_seconds",
    "Model load time in seconds",
    ["model_type"],
)

model_inference_duration = Histogram(
    "goafar_model_inference_duration_seconds",
    "Model inference duration",
    ["model_type"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

# 缓存指标
cache_hits = Counter(
    "goafar_cache_hits_total",
    "Number of cache hits",
    ["cache_type"],
)

cache_misses = Counter(
    "goafar_cache_misses_total",
    "Number of cache misses",
    ["cache_type"],
)

cache_size = Gauge(
    "goafar_cache_size",
    "Current cache size",
    ["cache_type"],
)

# 系统指标
system_info = Gauge(
    "goafar_system_info",
    "System information",
    ["info"],
)

gpu_memory_usage = Gauge(
    "goafar_gpu_memory_usage_bytes",
    "GPU memory usage in bytes",
    ["gpu_id"],
)

gpu_memory_total = Gauge(
    "goafar_gpu_memory_total_bytes",
    "Total GPU memory in bytes",
    ["gpu_id"],
)


# =============================================================================
# 装饰器
# =============================================================================

def track_request(endpoint: str):
    """追踪请求指标"""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            status = "success"
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                request_error.labels(endpoint=endpoint, error_type=type(e).__name__).inc()
                raise
            finally:
                duration = time.time() - start
                request_total.labels(endpoint=endpoint, method="POST", status=status).inc()
                request_duration.labels(endpoint=endpoint).observe(duration)
                if status == "success":
                    request_success.labels(endpoint=endpoint).inc()

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            status = "success"
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                request_error.labels(endpoint=endpoint, error_type=type(e).__name__).inc()
                raise
            finally:
                duration = time.time() - start
                request_total.labels(endpoint=endpoint, method="POST", status=status).inc()
                request_duration.labels(endpoint=endpoint).observe(duration)
                if status == "success":
                    request_success.labels(endpoint=endpoint).inc()

        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def track_pipeline_stage(stage: str):
    """追踪Pipeline阶段指标"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)

                # 记录候选数量（如果返回值有相关信息）
                if hasattr(result, "__len__"):
                    pipeline_candidates_gauge.labels(stage=stage).set(len(result))

                return result
            finally:
                duration = time.time() - start
                pipeline_recommendation_duration.labels(stage=stage).observe(duration)

        return wrapper

    return decorator


# =============================================================================
# 指标导出
# =============================================================================

def generate_metrics() -> bytes:
    """生成Prometheus格式的指标"""
    return generate_latest(registry)


def get_metrics_text() -> str:
    """获取指标的文本格式"""
    data = generate_metrics()
    return data.decode("utf-8") if isinstance(data, bytes) else data


# =============================================================================
# 系统信息收集
# =============================================================================

def collect_system_info() -> None:
    """收集系统信息"""
    import os
    import platform

    system_info.labels(info="version").set(2.0)
    system_info.labels(info="python_version").set(float(f"{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}"))
    system_info.labels(info="instance_id").set(hash(os.environ.get("GOAFAR_INSTANCE_ID", "unknown")))


def collect_gpu_info() -> None:
    """收集GPU信息"""
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_memory_total.labels(gpu_id=str(i)).set(props.total_memory)
                gpu_memory_usage.labels(gpu_id=str(i)).set(torch.cuda.memory_allocated(i))
    except Exception:
        pass


# =============================================================================
# 初始化
# =============================================================================

def init_metrics() -> None:
    """初始化指标收集"""
    collect_system_info()
    collect_gpu_info()


if __name__ == "__main__":
    # 测试代码
    print("Testing metrics module...")

    # 模拟请求
    for i in range(10):
        request_total.labels(endpoint="/test", method="GET", status="success").inc()
        request_duration.labels(endpoint="/test").observe(0.1 + i * 0.01)

    print(generate_metrics_text())
