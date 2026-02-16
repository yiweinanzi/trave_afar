#!/usr/bin/env python
"""
模型加载测试脚本
验证Qwen3系列模型是否正确下载并可以加载

使用方法:
    python tests/test_model_loading.py
    python tests/test_model_loading.py --test-embedding
    python tests/test_model_loading.py --test-reranker
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
import logging

# 该文件是手工脚本，不作为 pytest 用例收集
__test__ = False

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_model_files(model_path: Path) -> dict:
    """
    检查模型文件完整性

    Returns:
        检查结果字典
    """
    result = {
        'path': str(model_path),
        'exists': model_path.exists(),
        'config_exists': False,
        'tokenizer_exists': False,
        'model_files': [],
        'total_size_mb': 0,
        'status': 'missing'
    }

    if not model_path.exists():
        return result

    # 检查config.json
    config_file = model_path / "config.json"
    result['config_exists'] = config_file.exists()

    # 检查tokenizer
    tokenizer_files = ['tokenizer.json', 'vocab.json', 'merges.txt']
    result['tokenizer_exists'] = all((model_path / f).exists() for f in tokenizer_files)

    # 检查模型文件
    for ext in ['.safetensors', '.bin', '.pth']:
        result['model_files'] = list(model_path.glob(f"*{ext}"))
        if result['model_files']:
            break

    # 计算总大小
    for f in model_path.glob("*"):
        if f.is_file():
            result['total_size_mb'] += f.stat().st_size / 1024 / 1024

    # 判断状态
    if result['config_exists'] and result['model_files'] and result['tokenizer_exists']:
        result['status'] = 'complete'
    elif result['model_files']:
        result['status'] = 'partial'

    return result


def test_llm_loading(model_path: Path, test_prompt: str = "你好"):
    """
    测试LLM模型加载

    Args:
        model_path: 模型路径
        test_prompt: 测试提示词
    """
    logger.info(f"\n测试LLM模型加载: {model_path}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        # 检查GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {device}")

        # 加载tokenizer
        logger.info("加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # 加载模型
        logger.info("加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )

        # 测试推理
        logger.info("测试推理...")
        inputs = tokenizer(test_prompt, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"模型输出: {response}")

        logger.info("✓ LLM模型加载成功！")
        return True

    except Exception as e:
        logger.error(f"✗ LLM模型加载失败: {e}")
        return False


def test_embedding_loading(model_path: Path, test_text: str = "新疆天山天池"):
    """
    测试Embedding模型加载

    Args:
        model_path: 模型路径
        test_text: 测试文本
    """
    logger.info(f"\n测试Embedding模型加载: {model_path}")

    try:
        from transformers import AutoModel, AutoTokenizer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {device}")

        # 加载模型
        logger.info("加载模型...")
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        model = model.to(device)

        # 测试编码
        logger.info("测试编码...")
        inputs = tokenizer(test_text, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)

        logger.info(f"embedding形状: {embedding.shape}")
        logger.info("✓ Embedding模型加载成功！")
        return True

    except Exception as e:
        logger.error(f"✗ Embedding模型加载失败: {e}")
        return False


def test_reranker_loading(model_path: Path):
    """
    测试Reranker模型加载

    Args:
        model_path: 模型路径
    """
    logger.info(f"\n测试Reranker模型加载: {model_path}")

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {device}")

        # 加载模型
        logger.info("加载模型...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        model = model.to(device)

        # 测试排序
        logger.info("测试排序...")
        query = "新疆旅游景点"
        candidates = ["天山天池", "喀纳斯湖", "国际大巴扎"]

        pairs = [[query, candidate] for candidate in candidates]
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            scores = outputs.logits.squeeze(-1)

        logger.info(f"排序分数: {scores.tolist()}")
        logger.info("✓ Reranker模型加载成功！")
        return True

    except Exception as e:
        logger.error(f"✗ Reranker模型加载失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='模型加载测试')
    parser.add_argument('--model-path', type=str, default=None,
                        help='模型路径（默认为models/Qwen3-8B）')
    parser.add_argument('--test-llm', action='store_true', help='测试LLM模型')
    parser.add_argument('--test-embedding', action='store_true', help='测试Embedding模型')
    parser.add_argument('--test-reranker', action='store_true', help='测试Reranker模型')
    parser.add_argument('--check-only', action='store_true', help='只检查文件不加载')

    args = parser.parse_args()

    # 默认模型路径
    models_dir = PROJECT_ROOT / "models"
    llm_path = Path(args.model_path) if args.model_path else models_dir / "Qwen3-8B"
    embedding_path = models_dir / "Qwen3-Embedding-4B"
    reranker_path = models_dir / "Qwen3-Reranker-4B"

    print("="*60)
    print("GoAfar 模型检查")
    print("="*60)

    # 检查LLM模型
    llm_status = check_model_files(llm_path)
    print(f"\n[LLM] Qwen3-8B:")
    print(f"  路径: {llm_status['path']}")
    print(f"  状态: {llm_status['status']}")
    print(f"  大小: {llm_status['total_size_mb']:.1f} MB")
    print(f"  配置文件: {'✓' if llm_status['config_exists'] else '✗'}")
    print(f"  Tokenizer: {'✓' if llm_status['tokenizer_exists'] else '✗'}")
    print(f"  模型文件: {len(llm_status['model_files'])} 个")

    # 检查Embedding模型
    embedding_status = check_model_files(embedding_path)
    print(f"\n[Embedding] Qwen3-Embedding-4B:")
    print(f"  路径: {embedding_status['path']}")
    print(f"  状态: {embedding_status['status']}")
    print(f"  大小: {embedding_status['total_size_mb']:.1f} MB")

    # 检查Reranker模型
    reranker_status = check_model_files(reranker_path)
    print(f"\n[Reranker] Qwen3-Reranker-4B:")
    print(f"  路径: {reranker_status['path']}")
    print(f"  状态: {reranker_status['status']}")
    print(f"  大小: {reranker_status['total_size_mb']:.1f} MB")

    # 如果只是检查，到此结束
    if args.check_only:
        return

    # 测试加载
    test_results = {}

    if args.test_llm or (args.test_embedding and args.test_reranker) or not any([args.test_llm, args.test_embedding, args.test_reranker]):
        if llm_status['status'] == 'complete':
            test_results['llm'] = test_llm_loading(llm_path)
        else:
            logger.warning("LLM模型不完整，跳过测试")

    if args.test_embedding:
        if embedding_status['status'] == 'complete':
            test_results['embedding'] = test_embedding_loading(embedding_path)
        else:
            logger.warning("Embedding模型不完整，跳过测试")

    if args.test_reranker:
        if reranker_status['status'] == 'complete':
            test_results['reranker'] = test_reranker_loading(reranker_path)
        else:
            logger.warning("Reranker模型不完整，跳过测试")

    # 总结
    print("\n" + "="*60)
    print("测试总结:")
    for model, result in test_results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {model}: {status}")


if __name__ == "__main__":
    main()
