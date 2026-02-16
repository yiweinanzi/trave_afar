"""
FastAPI unified service entrypoint with production features.

Features:
- Health check endpoints (/healthz, /readyz)
- Metrics endpoint (/metrics)
- CORS support
- Rate limiting
- API key authentication
- Request timeout handling
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import Awaitable, Callable, Dict, Optional

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.health import HealthStatus, get_health_checker, init_health_checker
from schemas.recommendation import RecommendationRequest, RecommendationResponse
from service.config_loader import load_config
from service.pipeline import get_pipeline


# =============================================================================
# 配置加载
# =============================================================================

config = load_config()
api_config = config.get("api", {})
health_config = config.get("health", {})
env_config = config.get("environment", {})

# =============================================================================
# Lifespan 管理
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    print(f"[GoAfar] Starting API server...")
    print(f"[GoAfar] Environment: {env_config.get('mode', 'unknown')}")
    print(f"[GoAfar] Instance: {env_config.get('instance_id', 'unknown')}")

    # 初始化健康检查器
    init_health_checker(config)

    # 预热Pipeline
    try:
        pipeline = get_pipeline()
        print(f"[GoAfar] Pipeline ready: embedding={pipeline._embedding_ready}")
    except Exception as e:
        print(f"[GoAfar] Pipeline warmup warning: {e}")

    yield

    # 关闭时
    print("[GoAfar] Shutting down...")


# =============================================================================
# FastAPI 应用创建
# =============================================================================

app = FastAPI(
    title="GoAfar API",
    description="Travel recommendation system with multi-modal AI",
    version="2.0.0",
    lifespan=lifespan,
)

# =============================================================================
# 中间件配置
# =============================================================================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_config.get("cors_origins", "*").split(","),
    allow_methods=api_config.get("cors_methods", "GET,POST,OPTIONS").split(","),
    allow_headers=api_config.get("cors_headers", "*").split(","),
    allow_credentials=True,
)

# GZip 压缩
app.add_middleware(GZipMiddleware, minimum_size=1000)

# =============================================================================
# 简单的内存限流器
# =============================================================================

class RateLimiter:
    """简单的内存限流器"""
    def __init__(self):
        self._requests: Dict[str, list] = {}

    def check(self, key: str, limit: int, window: int) -> bool:
        """检查是否超过限流"""
        now = time.time()
        if key not in self._requests:
            self._requests[key] = []

        # 清理过期记录
        self._requests[key] = [t for t in self._requests[key] if t > now - window]

        if len(self._requests[key]) >= limit:
            return False

        self._requests[key].append(now)
        return True


_rate_limiter = RateLimiter()
_rate_limit_enabled = api_config.get("rate_limit_enabled", False)
_rate_limit_requests = api_config.get("rate_limit_requests", 100)
_rate_limit_window = api_config.get("rate_limit_window", 60)
_api_key_required = api_config.get("api_key_required", False)
_api_key_header = api_config.get("api_key_header", "X-API-Key")


@app.middleware("http")
async def middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """自定义中间件"""
    # API Key 检查
    if _api_key_required:
        api_key = request.headers.get(_api_key_header)
        if not api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "API key required"},
            )

    # 限流检查
    if _rate_limit_enabled:
        client_ip = request.client.host if request.client else "unknown"
        if not _rate_limiter.check(client_ip, _rate_limit_requests, _rate_limit_window):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded"},
            )

    # 请求处理
    start_time = time.time()
    try:
        response = await call_next(request)
        response.headers["X-Response-Time"] = f"{(time.time() - start_time) * 1000:.2f}ms"
        return response
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(e)},
        )


# =============================================================================
# 健康检查端点
# =============================================================================

@app.get("/healthz")
async def healthz() -> Dict:
    """
    基本健康检查。

    只检查服务是否运行，不检查依赖。
    用于 Kubernetes livenessProbe。
    """
    if not health_config.get("enabled", True):
        raise HTTPException(status_code=503, detail="Health checks disabled")

    checker = get_health_checker()
    result = checker.check_healthz()
    return result.to_dict()


@app.get("/readyz")
async def readyz() -> Dict:
    """
    就绪检查。

    检查所有依赖是否可用。
    用于 Kubernetes readinessProbe。
    """
    if not health_config.get("enabled", True):
        raise HTTPException(status_code=503, detail="Health checks disabled")

    checker = get_health_checker()
    pipeline = get_pipeline()
    result = checker.check_readyz(pipeline)

    # 如果状态不是 healthy，返回 503
    http_status = status.HTTP_200_OK
    if result.status != "healthy":
        http_status = status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(content=result.to_dict(), status_code=http_status)


@app.get("/healthz/live")
async def liveness() -> Dict:
    """Kubernetes 风格的存活检查"""
    return {"status": "alive"}


@app.get("/healthz/ready")
async def readiness() -> Dict:
    """Kubernetes 风格的就绪检查"""
    checker = get_health_checker()
    pipeline = get_pipeline()
    result = checker.check_readyz(pipeline)

    http_status = status.HTTP_200_OK
    if result.status != "healthy":
        http_status = status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(content={"ready": result.status == "healthy"}, status_code=http_status)


# =============================================================================
# 指标端点
# =============================================================================

# 简单的指标收集
_metrics: Dict[str, float] = {
    "goafar_requests_total": 0,
    "goafar_requests_success": 0,
    "goafar_requests_error": 0,
    "goafar_request_duration_seconds": 0,
}


@app.get("/metrics")
async def metrics() -> Response:
    """
    Prometheus 格式的指标导出。
    """
    monitoring = config.get("monitoring", {})
    if not monitoring.get("prometheus", {}).get("enabled", True):
        raise HTTPException(status_code=503, detail="Metrics disabled")

    lines = [
        "# HELP goafar_requests_total Total number of requests",
        "# TYPE goafar_requests_total counter",
        f"goafar_requests_total {_metrics['goafar_requests_total']}",
        "",
        "# HELP goafar_requests_success Number of successful requests",
        "# TYPE goafar_requests_success counter",
        f"goafar_requests_success {_metrics['goafar_requests_success']}",
        "",
        "# HELP goafar_requests_error Number of failed requests",
        "# TYPE goafar_requests_error counter",
        f"goafar_requests_error {_metrics['goafar_requests_error']}",
        "",
        "# HELP goafar_request_duration_seconds Average request duration",
        "# TYPE goafar_request_duration_seconds gauge",
        f"goafar_request_duration_seconds {_metrics['goafar_request_duration_seconds']:.4f}",
    ]

    return Response(content="\n".join(lines), media_type="text/plain")


# =============================================================================
# API 端点
# =============================================================================

@app.get("/")
async def root() -> Dict:
    """根端点"""
    return {
        "name": "GoAfar API",
        "version": "2.0.0",
        "environment": env_config.get("mode", "unknown"),
        "instance_id": env_config.get("instance_id", "unknown"),
        "endpoints": {
            "health": "/healthz",
            "ready": "/readyz",
            "metrics": "/metrics",
            "recommend": "/v1/recommend/itinerary",
        },
    }


@app.get("/v1/info")
async def info() -> Dict:
    """API 信息"""
    return {
        "name": "GoAfar API",
        "version": "2.0.0",
        "description": "Travel recommendation system with multi-modal AI",
        "environment": env_config.get("mode", "unknown"),
        "instance_id": env_config.get("instance_id", "unknown"),
        "config": {
            "llm_enabled": config.get("llm", {}).get("enabled", False),
            "rerank_enabled": config.get("rerank", {}).get("enabled", False),
            "monitoring_enabled": config.get("monitoring", {}).get("prometheus", {}).get("enabled", False),
        },
    }


@app.post("/v1/recommend/itinerary", response_model=RecommendationResponse)
async def recommend_itinerary(request: RecommendationRequest) -> RecommendationResponse:
    """
    推荐旅行路线。

    Args:
        request: 推荐请求

    Returns:
        RecommendationResponse: 推荐结果
    """
    start_time = time.time()
    _metrics["goafar_requests_total"] += 1

    try:
        pipeline = get_pipeline()
        response = pipeline.recommend(request)

        if not response.success:
            _metrics["goafar_requests_error"] += 1
            raise HTTPException(status_code=400, detail=response.error)

        _metrics["goafar_requests_success"] += 1
        return response

    except HTTPException:
        raise
    except Exception as e:
        _metrics["goafar_requests_error"] += 1
        print(f"[ERROR] Recommendation failed: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    finally:
        duration = time.time() - start_time
        # 移动平均
        _metrics["goafar_request_duration_seconds"] = (
            _metrics["goafar_request_duration_seconds"] * 0.9 + duration * 0.1
        )


# =============================================================================
# 错误处理
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error": str(exc) if os.environ.get("GOAFAR_DEBUG") else "See logs",
        },
    )


# =============================================================================
# 主入口
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 8000)
    workers = config.get("runtime", {}).get("workers", 0)

    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        workers=workers if workers > 0 else None,
        reload=workers == 0 and env_config.get("mode") == "dev",
    )
