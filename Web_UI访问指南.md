# 🌐 GoAfar Web UI 访问指南

## 📍 当前状态

**Web UI进程**: ✅ 正在运行  
**本地地址**: http://localhost:7860  
**公网地址**: Gradio会自动生成 https://xxxxx.gradio.live 链接

## 🚀 查看公网链接

```bash
# 查看gradio输出日志
cat demo_ui.log | grep "gradio.live"

# 或查看进程输出
ps aux | grep demo_ui.py
```

公网链接格式示例：`https://abc123def456.gradio.live`

## 💻 如果需要重启

```bash
# 1. 停止当前进程
pkill -f demo_ui.py

# 2. 重新启动
cd /root/autodl-tmp/goafar_project
source /root/miniconda3/bin/activate goafar
python demo_ui.py

# 等待30秒，会显示公网链接
```

## 🎯 Web UI功能

### ✅ 当前可用功能
1. **语义检索** - 关键词搜索景点
2. **意图理解** - AI分析旅游需求
3. **系统信息** - 查看数据统计

### ⏳ 完整功能（需要向量文件）
- 语义相似度检索
- 完整路线规划
- LLM增强推荐

## 📊 数据准备

如果Web UI提示"向量未生成"，请运行：

```bash
# 生成POI向量（GPU加速，约2秒）
python src/embedding/build_embeddings_gpu.py

# 或CPU版本（约20分钟）
python src/embedding/vector_builder.py
```

## 🌐 访问Web UI

1. **本地访问**: 
   - 在服务器上：http://localhost:7860
   
2. **公网访问**:
   - Gradio自动生成的链接（72小时有效）
   - 可以在任何地方访问
   - 无需VPN或端口映射

## 🎬 使用演示

### 场景1: 搜索景点
```
Tab: 语义检索
输入: "湖"
输出: 显示所有包含"湖"的景点
```

### 场景2: 理解意图
```
Tab: 意图理解  
输入: "想去新疆看3天雪山，拍照"
输出: 
  - 省份: 新疆 ✓
  - 天数: 3 ✓
  - 兴趣: 雪山 ✓
  - 活动: 拍照 ✓
```

---

**注意**: Web UI当前正在后台运行，稍等片刻即可通过浏览器访问！

**GitHub**: https://github.com/yiweinanzi/trave_afar

