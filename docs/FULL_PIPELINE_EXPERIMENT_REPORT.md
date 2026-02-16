# GoAfar 全链路实验报告

**生成时间**: 2026-02-16T12:54:19.570394

## 1. 数据准备检查

- **poi_emb**: ✓
  - shape: [1333, 2560]
  - size_mb: 13.0177001953125
- **poi_meta**: ✓
  - rows: 1333
  - columns: ['poi_id', 'name', 'lat', 'lon', 'open_min', 'close_min', 'stay_min', 'province', 'city', 'description', 'time_str', 'airport', 'osm_id', 'code', 'fclass', 'population']
- **time_matrix**: ✓
  - size_mb: 0.00164794921875
- **poi_data**: ✓
  - rows: 1333
  - provinces: 8

## 2. Pipeline初始化

- ✓ 初始化成功
- 初始化耗时: 0.08s

### 模型健康状态

- ✓ **embedding**: True
- ✓ **reranker**: True
- ✓ **llm**: True
- ✓ **all**: True

## 3. 多组件协调测试

### 基础模板模式

**请求参数**:
- query: 想去新疆看7天雪山和草原，拍照
- city: 新疆
- days: 7
- budget: 5000
- group_type: 朋友
- interests: ['自然', '摄影']
- use_llm: False

**推荐结果**:
- 标题: 天山南北｜东天山-哈密市区-巴里坤草原，牧歌悠扬
- 描述: 根据您的需求「想去新疆看7天雪山和草原，拍照」为您精心规划了这条新疆深度游路线。行程涵盖2个精选景点，预计用时4.9小时。沿途将游览巴里坤草原等知名景点。让您在有限的时间内，领略新疆最精华的风景，体验最地道的文化。...
- 总时长: 4.9小时
- 景点数: 2
- 省份: 新疆

**Debug信息**:
- 召回来源: {'dense': 48, 'behavior': 47, 'geo': 40}
- 候选数（召回后）: 80
- 候选数（排序后）: 20
- 时间矩阵来源: osrm
- 降级事件: ['training_embedding_mismatch:2134', 'qwen_reranker_applied']

**推荐路线**:
1. 东天山 (停留150分钟)
2. 哈密市区 (停留60分钟)
3. 巴里坤草原 (停留180分钟)
4. 东天山 (停留0分钟)

### LLM增强模式

**请求参数**:
- query: 想去云南看古镇和自然风光，5天行程
- city: 云南
- days: 5
- budget: 3000
- group_type: 情侣
- interests: ['文化', '自然']
- use_llm: True

**推荐结果**:
- 标题: 彩云之南｜双廊古镇-大理古城-才村码头，寻古探今
- 描述: 根据您的需求「想去云南看古镇和自然风光，5天行程」为您精心规划了这条云南深度游路线。行程涵盖4个精选景点，预计用时6.8小时。让您在有限的时间内，领略云南最精华的风景，体验最地道的文化。...
- 总时长: 6.8小时
- 景点数: 4
- 省份: 云南

**Debug信息**:
- 召回来源: {'dense': 62, 'behavior': 0, 'geo': 40}
- 候选数（召回后）: 80
- 候选数（排序后）: 20
- 时间矩阵来源: osrm
- 降级事件: ['training_embedding_mismatch:2134', 'qwen_reranker_applied']

**推荐路线**:
1. 双廊古镇 (停留120分钟)
2. 大理古城 (停留90分钟)
3. 才村码头 (停留120分钟)
4. 磻溪村s弯 (停留120分钟)
5. 廊桥 (停留150分钟)
6. 双廊古镇 (停留0分钟)

### 甘肃短途游

**请求参数**:
- query: 甘肃周末两日游，历史文化景点
- city: 甘肃
- days: 2
- budget: 2000
- group_type: 家庭
- interests: ['历史', '文化']
- use_llm: False

**推荐结果**:
- 标题: 河西走廊｜敦煌文化遗址-敦煌博物馆-敦煌市区，探索未知之美
- 描述: 根据您的需求「甘肃周末两日游，历史文化景点」为您精心规划了这条甘肃深度游路线。行程涵盖2个精选景点，预计用时4.4小时。让您在有限的时间内，领略甘肃最精华的风景，体验最地道的文化。...
- 总时长: 4.4小时
- 景点数: 2
- 省份: 甘肃

**Debug信息**:
- 召回来源: {'dense': 60, 'behavior': 0, 'geo': 40}
- 候选数（召回后）: 80
- 候选数（排序后）: 30
- 时间矩阵来源: osrm
- 降级事件: ['training_embedding_mismatch:2134', "qwen_reranker_failed:'float' object has no attribute 'lower'"]

**推荐路线**:
1. 敦煌文化遗址 (停留150分钟)
2. 敦煌博物馆 (停留90分钟)
3. 敦煌市区 (停留60分钟)
4. 敦煌文化遗址 (停留0分钟)

## 4. 性能测试

### 内存使用

- RSS: 5337.77 MB
- VMS: 77346.16 MB

### 延迟统计

- 平均: 28.70s
- 标准差: 0.06s
- 最小: 28.62s
- 最大: 28.77s

### 样本数据

- 运行 1: 28.77s
- 运行 2: 28.71s
- 运行 3: 28.62s

## 5. 问题诊断

### 降级事件
- qwen_reranker_applied
- qwen_reranker_failed:'float' object has no attribute 'lower'
- training_embedding_mismatch:2134

✓ 未发现明显问题，系统运行正常