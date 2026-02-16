#!/bin/bash
# =============================================================================
# GoAfar 启动脚本
# =============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查环境变量
check_env() {
    log_info "Checking environment..."

    if [ -z "$GOAFAR_ENV_MODE" ]; then
        export GOAFAR_ENV_MODE=dev
        log_warn "GOAFAR_ENV_MODE not set, using 'dev'"
    fi

    if [ -z "$GOAFAR_LOG_LEVEL" ]; then
        export GOAFAR_LOG_LEVEL=INFO
    fi

    log_info "Environment: $GOAFAR_ENV_MODE"
    log_info "Log Level: $GOAFAR_LOG_LEVEL"
}

# 检查必要的目录
check_dirs() {
    log_info "Checking directories..."

    mkdir -p logs
    mkdir -p outputs/emb
    mkdir -p outputs/cache
    mkdir -p outputs/routing
    mkdir -p data

    log_info "Directories OK"
}

# 检查数据文件
check_data() {
    log_info "Checking data files..."

    if [ ! -f "data/all/poi_expanded.csv" ]; then
        log_warn "data/all/poi_expanded.csv not found. The service may not work properly."
    fi

    if [ ! -f "data/all/user_events.csv" ]; then
        log_warn "data/all/user_events.csv not found. Some features may not work."
    fi
}

# 启动服务
start_service() {
    log_info "Starting GoAfar API server..."

    # 设置 Python 路径
    export PYTHONPATH="/app:/app/src:$PYTHONPATH"

    # 启动命令
    if [ "$GOAFAR_ENV_MODE" = "dev" ]; then
        log_info "Starting in development mode with hot reload..."
        exec python3 -m uvicorn src.api.server:app \
            --host 0.0.0.0 \
            --port ${GOAFAR_API_PORT:-8000} \
            --reload
    else
        log_info "Starting in production mode..."
        exec python3 -m uvicorn src.api.server:app \
            --host 0.0.0.0 \
            --port ${GOAFAR_API_PORT:-8000} \
            --workers ${GOAFAR_WORKERS:-1} \
            --access-log \
            --log-level $GOAFAR_LOG_LEVEL
    fi
}

# 主函数
main() {
    log_info "GoAfar API Server"
    log_info "=================="

    check_env
    check_dirs
    check_data

    log_info "Starting..."
    start_service
}

# 运行
main "$@"
