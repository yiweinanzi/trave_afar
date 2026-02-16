# =============================================================================
# GoAfar Multi-Stage Dockerfile
# =============================================================================
# 构建阶段：Python 依赖和模型准备
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    ca-certificates \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# 依赖安装阶段
# =============================================================================
FROM base AS dependencies

WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install -r requirements.txt

# =============================================================================
# 生产镜像阶段
# =============================================================================
FROM base AS production

# 创建非root用户
RUN useradd -m -u 1000 goafar && \
    mkdir -p /app /data /models /logs /outputs && \
    chown -R goafar:goafar /app /data /models /logs /outputs

WORKDIR /app

# 从依赖阶段复制已安装的包
COPY --from=dependencies /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# 复制应用代码
COPY --chown=goafar:goafar src/ /app/src/
COPY --chown=goafar:goafar configs/ /app/configs/
COPY --chown=goafar:goafar requirements.txt .

# 设置Python路径
ENV PYTHONPATH=/app:/app/src:$PYTHONPATH \
    PATH=/app:$PATH

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# 暴露端口
EXPOSE 8000

# 切换到非root用户
USER goafar

# 启动命令
CMD ["python3", "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# 开发镜像阶段（用于本地开发和调试）
# =============================================================================
FROM base AS development

# 安装开发工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    net-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# 创建用户
RUN useradd -m -u 1000 goafar && \
    mkdir -p /app /data /models /logs /outputs && \
    chown -R goafar:goafar /app /data /models /logs /outputs

WORKDIR /app

# 复制并安装依赖
COPY --chown=goafar:goafar requirements.txt .
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    pip3 install pytest pytest-cov black flake8 mypy

# 复制应用代码
COPY --chown=goafar:goafar src/ /app/src/
COPY --chown=goafar:goafar configs/ /app/configs/

# 开发环境变量
ENV PYTHONPATH=/app:/app/src:$PYTHONPATH \
    GOAFAR_ENV_MODE=dev \
    GOAFAR_DEBUG=true

EXPOSE 8000

USER goafar

# 开发模式启动（热重载）
CMD ["python3", "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
