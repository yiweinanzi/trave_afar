# GoAfar 生产部署文档

## 目录

- [环境准备](#环境准备)
- [本地开发](#本地开发)
- [Docker 部署](#docker-部署)
- [Kubernetes 部署](#kubernetes-部署)
- [监控与运维](#监控与运维)
- [故障排查](#故障排查)

---

## 环境准备

### 系统要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | 4 核 | 8 核+ |
| 内存 | 16 GB | 32 GB+ |
| GPU | - | NVIDIA GPU (8GB+ VRAM) |
| 存储 | 50 GB | 100 GB+ |
| Python | 3.10 | 3.10 |

### 软件依赖

- Docker 24.0+
- Docker Compose 2.20+
- (可选) Kubernetes 1.25+
- (可选) NVIDIA Docker Runtime

---

## 本地开发

### 环境变量配置

创建 `.env` 文件：

```bash
# 环境模式
GOAFAR_ENV_MODE=dev

# API 配置
GOAFAR_API_HOST=0.0.0.0
GOAFAR_API_PORT=8000
GOAFAR_CORS_ORIGINS=*

# LLM 配置
GOAFAR_LLM_ENABLED=true
GOAFAR_LLM_GPU=true
GOAFAR_LLM_MODEL=models/Qwen3-8B

# 日志
GOAFAR_LOG_LEVEL=INFO
GOAFAR_LOG_DIR=logs
```

### 启动开发服务器

```bash
# 方式1: 直接运行
python -m uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000

# 方式2: 使用 Docker
docker-compose up
```

### 验证服务

```bash
# 健康检查
curl http://localhost:8000/healthz

# API 信息
curl http://localhost:8000/v1/info
```

---

## Docker 部署

### 构建镜像

```bash
# 开发镜像
docker build --target development -t goafar:dev .

# 生产镜像
docker build --target production -t goafar:latest .
```

### 运行容器

```bash
# 基础运行
docker run -d \
  --name goafar \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  goafar:latest

# 带环境变量
docker run -d \
  --name goafar \
  -p 8000:8000 \
  -e GOAFAR_ENV_MODE=prod \
  -e GOAFAR_LLM_GPU=true \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  --gpus all \
  goafar:latest
```

### Docker Compose

#### 开发环境

```bash
docker-compose up
```

#### 生产环境

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

#### 启用可选服务

```bash
# 启用缓存 (Redis)
docker-compose --profile cache up -d

# 启用 MLOps (MLflow + PostgreSQL)
docker-compose --profile mlops up -d

# 启用追踪 (Jaeger)
docker-compose --profile tracing up -d

# 启用所有可选服务
docker-compose --profile cache --profile mlops --profile tracing up -d
```

---

## Kubernetes 部署

### 创建 Namespace

```bash
kubectl create namespace goafar
```

### 部署配置

```yaml
# deployments/goafar.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: goafar
  namespace: goafar
spec:
  replicas: 2
  selector:
    matchLabels:
      app: goafar
  template:
    metadata:
      labels:
        app: goafar
        version: v2.0.0
    spec:
      containers:
      - name: goafar
        image: goafar:latest
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: GOAFAR_ENV_MODE
          value: "prod"
        - name: GOAFAR_API_PORT
          value: "8000"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        volumeMounts:
        - name: data
          mountPath: /app/data
          readOnly: true
        - name: models
          mountPath: /app/models
          readOnly: true
        livenessProbe:
          httpGet:
            path: /healthz
            port: http
          initialDelaySeconds: 300
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /readyz
            port: http
          initialDelaySeconds: 60
          periodSeconds: 10
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: goafar-data
      - name: models
        persistentVolumeClaim:
          claimName: goafar-models
---
apiVersion: v1
kind: Service
metadata:
  name: goafar
  namespace: goafar
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: http
    protocol: TCP
    name: http
  selector:
    app: goafar
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: goafar
  namespace: goafar
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - api.goafar.example.com
    secretName: goafar-tls
  rules:
  - host: api.goafar.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: goafar
            port:
              number: 80
```

### 部署

```bash
kubectl apply -f deployments/
```

### 水平自动扩缩容

```yaml
# deployments/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: goafar
  namespace: goafar
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: goafar
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## 监控与运维

### Prometheus 指标

访问 Prometheus: http://localhost:9091

#### 关键指标

| 指标 | 类型 | 描述 |
|------|------|------|
| `goafar_requests_total` | Counter | 请求总数 |
| `goafar_request_duration_seconds` | Histogram | 请求延迟分布 |
| `goafar_pipeline_recommendation_duration_seconds` | Histogram | 推荐Pipeline延迟 |
| `goafar_cache_hits_total` | Counter | 缓存命中数 |
| `goafar_gpu_memory_usage_bytes` | Gauge | GPU内存使用 |

#### PromQL 查询示例

```promql
# 请求成功率
rate(goafar_requests_success_total[5m]) / rate(goafar_requests_total[5m])

# P95 延迟
histogram_quantile(0.95, rate(goafar_request_duration_seconds_bucket[5m]))

# GPU 内存使用率
goafar_gpu_memory_usage_bytes / goafar_gpu_memory_total_bytes

# 缓存命中率
rate(goafar_cache_hits_total[5m]) / (rate(goafar_cache_hits_total[5m]) + rate(goafar_cache_misses_total[5m]))
```

### Grafana 仪表板

访问 Grafana: http://localhost:3000

默认凭据: `admin / admin`

导入预配置仪表板:

1. 进入 Configuration > Data Sources
2. 确认 Prometheus 数据源已配置
3. 进入 Create > Import
4. 上传仪表板配置文件或输入 ID

### 日志管理

```bash
# 查看日志
docker-compose logs -f goafar

# 查看 Kubernetes 日志
kubectl logs -f deployment/goafar -n goafar

# 查看特定时间段日志
kubectl logs --since-time=$(date -d '1 hour ago' +%s) -f deployment/goafar -n goafar
```

### 日志配置

在 `configs/runtime.yaml` 中配置日志:

```yaml
logging:
  level: INFO
  log_dir: logs
  max_file_size_mb: 100
  backup_count: 10
  format: json  # 生产环境使用 JSON 格式
  sanitize_secrets: true
```

---

## 故障排查

### 常见问题

#### 1. 模型加载失败

**症状**: `/readyz` 返回 `models: degraded`

**解决**:
```bash
# 检查模型文件
ls -la models/Qwen3-8B/

# 检查磁盘空间
df -h

# 重新下载模型
python -m src.utils.model_downloader
```

#### 2. GPU 不可用

**症状**: `CUDA out of memory` 或模型加载到 CPU

**解决**:
```bash
# 检查 GPU
nvidia-smi

# 安装 NVIDIA Docker Runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update && apt-get install -y nvidia-docker2
systemctl restart docker

# 运行时添加 GPU 参数
docker run --gpus all ...
```

#### 3. 健康检查失败

**症状**: 容器不断重启

**解决**:
```bash
# 查看日志
docker-compose logs goafar

# 检查健康端点
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz

# 增加启动时间
# docker-compose.yml
healthcheck:
  start_period: 600s  # 10分钟
```

#### 4. 内存不足

**症状**: `OOMKilled`

**解决**:
```yaml
# 增加内存限制
resources:
  limits:
    memory: 16Gi

# 或使用模型量化
# runtime.yaml
llm:
  load_in_8bit: true
```

#### 5. 请求超时

**症状**: 客户端 504 超时

**解决**:
```yaml
# 增加超时时间
runtime:
  request_timeout: 600  # 10分钟

# 或优化配置
llm:
  max_new_tokens: 256  # 减少生成长度
```

### 性能优化

1. **启用缓存**
```yaml
cache:
  enabled: true
  max_items: 10000
```

2. **批处理**
```yaml
embedding:
  batch_size: 64
```

3. **模型量化**
```yaml
llm:
  load_in_8bit: true
```

4. **多实例部署**
```yaml
# docker-compose.yml
deploy:
  replicas: 4
```

---

## 安全建议

1. **API 密钥验证**
```bash
export GOAFAR_API_KEY_REQUIRED=true
```

2. **HTTPS 配置**
```yaml
# 使用 Nginx 或 Traefik 作为反向代理
```

3. **敏感数据保护**
```yaml
logging:
  sanitize_secrets: true
```

4. **网络隔离**
```yaml
# 使用 Docker 网络隔离
networks:
  goafar-internal:
    internal: true
```

---

## 备份与恢复

### 数据备份

```bash
# 备份输出目录
tar -czf outputs-$(date +%Y%m%d).tar.gz outputs/

# 备份到云存储
aws s3 sync outputs/ s3://goafar-backup/outputs/
```

### 恢复

```bash
# 解压备份
tar -xzf outputs-20240101.tar.gz

# 从云存储恢复
aws s3 sync s3://goafar-backup/outputs/ outputs/
```
