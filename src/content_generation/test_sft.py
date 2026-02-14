"""
测试SFT模型
用于验证训练后的模型效果
"""
import os
import sys
import json
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 添加src到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CACHE_DIR = Path(os.getenv("GOAFAR_MODEL_CACHE", str(PROJECT_ROOT / "models")))


def load_model(model_path: str, base_model: str = None):
    """加载SFT模型"""
    print(f"加载模型: {model_path}")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        cache_dir=str(MODEL_CACHE_DIR)
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    if Path(model_path / "adapter_config.json").exists():
        # LoRA模型
        print("检测到LoRA适配器，加载base模型...")
        base_model_path = base_model or "Qwen/Qwen3-8B"
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=str(MODEL_CACHE_DIR)
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # 合并权重
    else:
        # 完整模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=str(MODEL_CACHE_DIR)
        )

    model.eval()
    print("✓ 模型加载完成")
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_length: int = 512):
    """生成响应"""
    # Qwen聊天格式
    messages = [
        {"role": "system", "content": "你是一位专业的旅游规划助手。"},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response


def test_model(model_path: str, base_model: str = None):
    """测试模型"""
    print("=" * 80)
    print("SFT模型测试")
    print("=" * 80)

    # 加载模型
    model, tokenizer = load_model(model_path, base_model)

    # 测试用例
    test_cases = [
        {
            "task": "意图理解",
            "prompt": "推荐一条四川4日游，要包含亚丁、熊猫、火锅",
            "expected_fields": ["province", "duration_days", "interests"]
        },
        {
            "task": "路线生成",
            "prompt": "推荐西藏4天行程，偏好朝圣、珠峰",
            "expected_fields": ["province", "days", "daily_pois"]
        },
        {
            "task": "文案生成",
            "prompt": "给'喀什机场-喀什古城-奥依塔格红峡谷-白沙湖'这条新疆3日游写个吸引人的标题和描述",
            "expected_fields": ["title", "description"]
        },
        {
            "task": "意图理解",
            "prompt": "用户想去新疆看雪山和草原，计划3天",
            "expected_fields": ["province", "cities", "interests", "duration_days"]
        }
    ]

    print("\n开始测试...\n")

    for i, test_case in enumerate(test_cases, 1):
        print(f"测试 {i}: {test_case['task']}")
        print(f"Prompt: {test_case['prompt']}")
        print("-" * 80)

        try:
            response = generate_response(model, tokenizer, test_case['prompt'])
            print(f"Response: {response}")

            # 尝试解析JSON
            try:
                response_json = json.loads(response)
                print(f"✓ JSON格式正确")

                # 检查期望字段
                missing_fields = []
                for field in test_case['expected_fields']:
                    if field not in response_json:
                        missing_fields.append(field)

                if missing_fields:
                    print(f"⚠ 缺少字段: {missing_fields}")
                else:
                    print(f"✓ 包含所有期望字段")

            except json.JSONDecodeError:
                print(f"⚠ 响应不是有效的JSON格式")

        except Exception as e:
            print(f"✗ 生成失败: {e}")

        print()

    print("=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='测试SFT模型')
    parser.add_argument('--model', type=str, default='outputs/sft/qwen3-8b-tourism', help='模型路径')
    parser.add_argument('--base-model', type=str, default=None, help='基础模型路径（用于LoRA模型）')

    args = parser.parse_args()

    test_model(args.model, args.base_model)
