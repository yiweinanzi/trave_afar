"""
健康检查模块。

提供 /healthz �� /readyz 端点实现，检查:
- 模型加载状态
- 向量索引状态
- 外部服务连接
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    name: str
    status: str  # healthy, degraded, unhealthy
    message: str = ""
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthStatus:
    """总体健康状态"""
    status: str  # healthy, degraded, unhealthy
    checks: List[HealthCheckResult] = field(default_factory=list)
    version: str = "2.0.0"
    instance_id: str = "goafar-1"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "status": self.status,
            "version": self.version,
            "instance_id": self.instance_id,
            "timestamp": self.timestamp,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "message": c.message,
                    "duration_ms": c.duration_ms,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }


class HealthChecker:
    """健康检查器"""

    def __init__(
        self,
        instance_id: str = "goafar-1",
        version: str = "2.0.0",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.instance_id = instance_id or os.environ.get("GOAFAR_INSTANCE_ID", "goafar-1")
        self.version = version
        self.config = config or {}
        self._checkers: List[Callable[[], HealthCheckResult]] = []
        self._setup_default_checks()

    def _setup_default_checks(self):
        """设置默认健康检查"""
        self._checkers = [
            self._check_disk_space,
            self._check_memory,
        ]

    def add_checker(self, checker: Callable[[], HealthCheckResult]) -> None:
        """添加自定义检查器"""
        self._checkers.append(checker)

    def check_healthz(self) -> HealthStatus:
        """
        执行基本健康检查 (healthz)。

        只检查服务是否运行，不检查依赖。
        """
        checks = []

        for checker in self._checkers:
            try:
                checks.append(checker())
            except Exception as e:
                checks.append(HealthCheckResult(
                    name=checker.__name__,
                    status="unhealthy",
                    message=str(e)
                ))

        # 确定总体状态
        if any(c.status == "unhealthy" for c in checks):
            status = "unhealthy"
        elif any(c.status == "degraded" for c in checks):
            status = "degraded"
        else:
            status = "healthy"

        return HealthStatus(
            status=status,
            checks=checks,
            version=self.version,
            instance_id=self.instance_id,
        )

    def check_readyz(self, pipeline: Any = None) -> HealthStatus:
        """
        执行就绪检查 (readyz)。

        检查所有依赖是否可用。
        """
        checks = []

        # 1. 基本检查
        for checker in self._checkers:
            try:
                checks.append(checker())
            except Exception as e:
                checks.append(HealthCheckResult(
                    name=checker.__name__,
                    status="unhealthy",
                    message=str(e)
                ))

        # 2. 数据文件检查
        checks.append(self._check_data_files())

        # 3. 向量索引检查
        checks.append(self._check_vector_index())

        # 4. 模型检查 (可选，深度检查)
        if self.config.get("health", {}).get("readyz_deep_check", True):
            if pipeline is not None:
                checks.append(self._check_pipeline(pipeline))
            else:
                checks.append(self._check_models())

        # 5. 外部服务检查
        checks.append(self._check_external_services())

        # 确定总体状态
        if any(c.status == "unhealthy" for c in checks):
            status = "unhealthy"
        elif any(c.status == "degraded" for c in checks):
            status = "degraded"
        else:
            status = "healthy"

        return HealthStatus(
            status=status,
            checks=checks,
            version=self.version,
            instance_id=self.instance_id,
        )

    def _check_disk_space(self) -> HealthCheckResult:
        """检查磁盘空间"""
        start = time.time()
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")

            free_gb = free / (1024**3)
            used_percent = (used / total) * 100

            if free_gb < 1:
                return HealthCheckResult(
                    name="disk_space",
                    status="unhealthy",
                    message=f"Low disk space: {free_gb:.2f}GB free",
                    duration_ms=(time.time() - start) * 1000,
                    details={"free_gb": free_gb, "used_percent": used_percent}
                )
            elif free_gb < 5:
                return HealthCheckResult(
                    name="disk_space",
                    status="degraded",
                    message=f"Low disk space: {free_gb:.2f}GB free",
                    duration_ms=(time.time() - start) * 1000,
                    details={"free_gb": free_gb, "used_percent": used_percent}
                )

            return HealthCheckResult(
                name="disk_space",
                status="healthy",
                duration_ms=(time.time() - start) * 1000,
                details={"free_gb": free_gb, "used_percent": used_percent}
            )
        except Exception as e:
            return HealthCheckResult(
                name="disk_space",
                status="unhealthy",
                message=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def _check_memory(self) -> HealthCheckResult:
        """检查内存使用"""
        start = time.time()
        try:
            import psutil
            mem = psutil.virtual_memory()

            if mem.percent > 95:
                return HealthCheckResult(
                    name="memory",
                    status="unhealthy",
                    message=f"High memory usage: {mem.percent:.1f}%",
                    duration_ms=(time.time() - start) * 1000,
                    details={"percent": mem.percent, "available_gb": mem.available / (1024**3)}
                )
            elif mem.percent > 85:
                return HealthCheckResult(
                    name="memory",
                    status="degraded",
                    message=f"High memory usage: {mem.percent:.1f}%",
                    duration_ms=(time.time() - start) * 1000,
                    details={"percent": mem.percent, "available_gb": mem.available / (1024**3)}
                )

            return HealthCheckResult(
                name="memory",
                status="healthy",
                duration_ms=(time.time() - start) * 1000,
                details={"percent": mem.percent, "available_gb": mem.available / (1024**3)}
            )
        except ImportError:
            # psutil 不可用，跳过检查
            return HealthCheckResult(
                name="memory",
                status="healthy",
                message="psutil not available, skipping",
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return HealthCheckResult(
                name="memory",
                status="unhealthy",
                message=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def _check_data_files(self) -> HealthCheckResult:
        """检查数据文件是否存在"""
        start = time.time()
        try:
            project_root = Path(__file__).resolve().parents[2]
            paths = self.config.get("paths", {})

            poi_csv = project_root / paths.get("poi_csv", "data/all/poi_expanded.csv")
            user_events_csv = project_root / paths.get("user_events_csv", "data/all/user_events.csv")

            missing = []
            if not poi_csv.exists():
                missing.append("poi_csv")
            if not user_events_csv.exists():
                missing.append("user_events_csv")

            if missing:
                return HealthCheckResult(
                    name="data_files",
                    status="degraded" if len(missing) == 1 else "unhealthy",
                    message=f"Missing files: {', '.join(missing)}",
                    duration_ms=(time.time() - start) * 1000,
                )

            return HealthCheckResult(
                name="data_files",
                status="healthy",
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return HealthCheckResult(
                name="data_files",
                status="degraded",
                message=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def _check_vector_index(self) -> HealthCheckResult:
        """检查向量索引"""
        start = time.time()
        try:
            project_root = Path(__file__).resolve().parents[2]
            paths = self.config.get("paths", {})
            embedding = self.config.get("embedding", {})

            emb_dir = project_root / paths.get("emb_dir", "outputs/emb")
            emb_file = emb_dir / "poi_emb.npy"
            meta_file = emb_dir / "poi_meta.csv"
            faiss_file = project_root / embedding.get("faiss_index_file", "outputs/emb/poi_faiss.index")

            if emb_file.exists() and meta_file.exists():
                backend = embedding.get("backend", "auto")
                if backend in ("auto", "faiss") and faiss_file.exists():
                    return HealthCheckResult(
                        name="vector_index",
                        status="healthy",
                        duration_ms=(time.time() - start) * 1000,
                        details={"has_faiss": True},
                    )
                return HealthCheckResult(
                    name="vector_index",
                    status="healthy",
                    duration_ms=(time.time() - start) * 1000,
                    details={"has_faiss": False},
                )
            else:
                return HealthCheckResult(
                    name="vector_index",
                    status="degraded",
                    message="Vector embeddings not found (will be built on demand)",
                    duration_ms=(time.time() - start) * 1000,
                )
        except Exception as e:
            return HealthCheckResult(
                name="vector_index",
                status="degraded",
                message=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def _check_models(self) -> HealthCheckResult:
        """检查模型文件"""
        start = time.time()
        try:
            project_root = Path(__file__).resolve().parents[2]
            models = self.config.get("models", {})
            llm = self.config.get("llm", {})

            model_paths = [
                (project_root / models.get("qwen_8b", "models/Qwen3-8B"), "qwen_8b"),
                (project_root / models.get("qwen_embedding", "models/Qwen3-Embedding-4B"), "embedding"),
            ]

            missing = []
            for path, name in model_paths:
                if not path.exists():
                    missing.append(name)

            if missing:
                return HealthCheckResult(
                    name="models",
                    status="degraded",
                    message=f"Missing models: {', '.join(missing)}",
                    duration_ms=(time.time() - start) * 1000,
                    details={"missing": missing},
                )

            return HealthCheckResult(
                name="models",
                status="healthy",
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return HealthCheckResult(
                name="models",
                status="degraded",
                message=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def _check_pipeline(self, pipeline: Any) -> HealthCheckResult:
        """检查Pipeline状态"""
        start = time.time()
        try:
            checks = {
                "embedding_ready": getattr(pipeline, "_embedding_ready", False),
                "qwen_loaded": pipeline._qwen is not None,
                "reranker_loaded": pipeline._qwen_reranker is not None,
            }

            if not checks["embedding_ready"]:
                return HealthCheckResult(
                    name="pipeline",
                    status="degraded",
                    message="Embedding not ready",
                    duration_ms=(time.time() - start) * 1000,
                    details=checks,
                )

            return HealthCheckResult(
                name="pipeline",
                status="healthy",
                duration_ms=(time.time() - start) * 1000,
                details=checks,
            )
        except Exception as e:
            return HealthCheckResult(
                name="pipeline",
                status="degraded",
                message=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def _check_external_services(self) -> HealthCheckResult:
        """检查外部服务"""
        start = time.time()
        try:
            planner = self.config.get("planner", {})
            osrm_url = planner.get("osrm_url", "http://router.project-osrm.org")
            timeout = self.config.get("health", {}).get("service_check_timeout", 10)

            # 尝试连接 OSRM
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.get(f"{osrm_url}/health")
                    if response.status_code == 200:
                        return HealthCheckResult(
                            name="external_services",
                            status="healthy",
                            duration_ms=(time.time() - start) * 1000,
                            details={"osrm": "ok"},
                        )
            except Exception:
                # OSRM 不可用，但可以回退到哈夫斯距离计算
                pass

            return HealthCheckResult(
                name="external_services",
                status="degraded",
                message="OSRM unavailable (will fallback to distance calculation)",
                duration_ms=(time.time() - start) * 1000,
                details={"osrm": "unavailable"},
            )
        except Exception as e:
            return HealthCheckResult(
                name="external_services",
                status="degraded",
                message=str(e),
                duration_ms=(time.time() - start) * 1000,
            )


# 全局健康检查器实例
_health_checker: Optional[HealthChecker] = None


def get_health_checker(config: Optional[Dict[str, Any]] = None) -> HealthChecker:
    """获取全局健康检查器实例"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker(config=config)
    return _health_checker


def init_health_checker(config: Dict[str, Any]) -> HealthChecker:
    """初始化全局健康检查器"""
    global _health_checker
    _health_checker = HealthChecker(config=config)
    return _health_checker
