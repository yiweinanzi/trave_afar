# SFT训练使用指南

使用QLoRA (4-bit量化 + LoRA) 对Qwen3-8B进行监督微调。

## 功能特性

- **QLoRA高效微调**: 使用4-bit量化 (NF4) + LoRA适配器
- **显存优化**: 双数量化进一步减少显存占用
- **多任务支持**:
  - 意图理解: 从用户查询提取结构化信息
  - 文案生成: 生成旅游路线标题和描述
  - POI推荐: 根据意图推荐POI序列

## 文件说明

### 核心文件

- `src/content_generation/train_sft.py` - SFT训练主脚本
- `src/content_generation/test_sft.py` - 模型测试脚本
- `scripts/train_sft.sh` - 训练启动脚本
- `scripts/train_sft_quick.sh` - 快速测试脚本

### 数据文件

- `outputs/datasets/sft_data.jsonl` - SFT训练数据

### 输出目录

- `outputs/sft/qwen3-8b-tourism/` - 训练完成的模型

## 快速开始

### 1. 快速测试（验证配置）

```bash
bash scripts/train_sft_quick.sh
```

这个命令会:
- 使用1个epoch进行快速测试
- 使用较小的序列长度(256)
- 较大的批次大小(4)

### 2. 完整训练

```bash
bash scripts/train_sft.sh
```

默认配置:
- Epochs: 3
- Max Length: 512
- Batch Size: 2
- Gradient Accumulation: 4
- Learning Rate: 2e-4
- LoRA r: 16
- LoRA alpha: 16

### 3. 自定义参数训练

```bash
# 指定训练轮数
bash scripts/train_sft.sh --epochs 5

# 指定批次大小
bash scripts/train_sft.sh --batch-size 4

# 指定最大序列长度
bash scripts/train_sft.sh --max-length 1024

# 组合多个参数
bash scripts/train_sft.sh --epochs 5 --batch-size 4 --max-length 1024

# 使用自定义数据文件
bash scripts/train_sft.sh --data outputs/datasets/my_sft_data.jsonl

# 指定输出目录
bash scripts/train_sft.sh --output outputs/sft/my-model
```

## 训练配置详解

### QLoRA配置

```python
# 4-bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # 启用4-bit量化
    bnb_4bit_quant_type="nf4",            # NF4量化类型
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算数据类型
    bnb_4bit_use_double_quant=True,       # 双数量化
)

# LoRA配置
lora_config = LoraConfig(
    r=16,                                  # LoRA rank
    lora_alpha=16,                         # LoRA alpha
    lora_dropout=0.05,                     # Dropout率
    target_modules=[                       # 目标模块
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type="CAUSAL_LM"
)
```

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 3 | 训练轮数 |
| `--batch-size` | 2 | 每设备批次大小 |
| `--grad-accum` | 4 | 梯度累积步数 |
| `--max-length` | 512 | 最大序列长度 |
| `--lr` | 2e-4 | 学习率 |
| `--lora-r` | 16 | LoRA rank |
| `--lora-alpha` | 16 | LoRA alpha |
| `--lora-dropout` | 0.05 | LoRA dropout |

## 数据格式

训练数据使用JSONL格式，每行一个样本：

```jsonl
{"prompt": "推荐一条四川4日游，要包含亚丁、熊猫、火锅", "response": "{\"province\": \"四川\", \"duration_days\": 4, ...}"}
{"prompt": "推荐西藏4天行程，偏好朝圣、珠峰", "response": "{\"province\": \"西藏\", \"days\": 4, ...}"}
```

支持两种响应字段：
- `response`: 标准响应字段
- `completion`: 备用响应字段（会自动转换为response）

## 测试训练好的模型

```bash
# 使用默认模型路径
python src/content_generation/test_sft.py

# 指定模型路径
python src/content_generation/test_sft.py --model outputs/sft/qwen3-8b-tourism

# 指定base模型（用于LoRA模型）
python src/content_generation/test_sft.py \
  --model outputs/sft/qwen3-8b-tourism \
  --base-model Qwen/Qwen3-8B
```

测试脚本会验证以下任务：
1. 意图理解 - 提取结构化信息
2. 路线生成 - 生成每日POI序列
3. 文案生成 - 生成标题和描述

## 显存需求

| 配置 | 显存占用 |
|------|----------|
| QLoRA (4-bit + LoRA) | ~8-10 GB |
| 完整微调 (bf16) | ~16-20 GB |

## 训练时长参考

基于单卡A100 (40GB):
- 快速测试 (--quick): ~10分钟
- 完整训练 (默认): ~30-45分钟
- 大规模训练 (5 epochs): ~60-75分钟

## 常见问题

### Q: 训练时显存不足怎么办？
A: 尝试以下方法：
1. 减小 `--batch-size`
2. 减小 `--max-length`
3. 增加 `--grad-accum` (保持有效batch size)

### Q: 如何选择LoRA的r值？
A:
- r=8: 更少参数，训练更快，可能欠拟合
- r=16: 平衡选择（默认）
- r=32/64: 更多参数，可能更好效果，需要更多显存

### Q: 训练后的模型如何部署？
A:
1. 更新 `config/runtime.yaml` 中的模型路径
2. 重启服务加载新模型
3. 运行测试验证效果

## 下一步

1. **DPO训练**: 使用偏好数据进一步优化
   ```bash
   bash scripts/train_dpo.sh
   ```

2. **模型评估**: 使用评估脚本测试模型性能
   ```bash
   python src/content_generation/evaluate_sft.py
   ```

3. **在线部署**: 集成到推荐系统中
   ```bash
   # 更新配置文件
   vim config/runtime.yaml
   ```

## 参考

- [QLoRA论文](https://arxiv.org/abs/2305.14314)
- [TRL文档](https://huggingface.co/docs/trl)
- [PEFT文档](https://huggingface.co/docs/peft)
