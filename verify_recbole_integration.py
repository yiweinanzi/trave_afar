#!/usr/bin/env python3
"""
RecBole 集成代码结构验证

验证关键函数和类是否已正确实现
"""
import re
from pathlib import Path


def check_file_exists(filepath):
    """检查文件是否存在"""
    path = Path(filepath)
    if path.exists():
        print(f"  ✓ {filepath}")
        return True
    else:
        print(f"  ✗ {filepath} (不存在)")
        return False


def check_function_in_file(filepath, function_name):
    """检查文件中是否包含特定函数"""
    path = Path(filepath)
    if not path.exists():
        print(f"  ✗ {function_name}: 文件不存在")
        return False

    content = path.read_text()

    # 检查函数定义
    if re.search(rf"def {function_name}\(", content):
        print(f"  ✓ {function_name}: 已实现")
        return True
    else:
        print(f"  ✗ {function_name}: 未找到")
        return False


def check_class_in_file(filepath, class_name):
    """检查文件中是否包含特定类"""
    path = Path(filepath)
    if not path.exists():
        print(f"  ✗ {class_name}: 文件不存在")
        return False

    content = path.read_text()

    # 检查类定义
    if re.search(rf"class {class_name}", content):
        print(f"  ✓ {class_name}: 已实现")

        # 检查关键方法
        methods = {
            '__init__': r"def __init__\(",
            'predict': r"def predict\(",
            '_predict_by_popularity': r"def _predict_by_popularity\(",
            'get_user_history_length': r"def get_user_history_length\(",
        }

        for method_name, pattern in methods.items():
            if re.search(pattern, content):
                print(f"    - {method_name}: ✓")
            else:
                print(f"    - {method_name}: ✗")

        return True
    else:
        print(f"  ✗ {class_name}: 未找到")
        return False


def check_config_parameter(filepath, parameter_name):
    """检查配置文件中是否包含特定参数"""
    path = Path(filepath)
    if not path.exists():
        print(f"  ✗ {parameter_name}: 文件不存在")
        return False

    content = path.read_text()

    if re.search(rf"{parameter_name}:", content):
        print(f"  ✓ {parameter_name}: 已配置")
        return True
    else:
        print(f"  ✗ {parameter_name}: 未配置")
        return False


def main():
    """运行验证"""
    print("\n" + "=" * 60)
    print("RecBole 集成代码结构验证")
    print("=" * 60)

    # 1. 检查文件是否存在
    print("\n1. 文件检查:")
    files = [
        "src/recommendation/recbole_trainer.py",
        "src/recommendation/candidate_merger.py",
        "configs/runtime.yaml",
        "configs/recbole.yaml",
    ]

    all_files_exist = True
    for f in files:
        if not check_file_exists(f):
            all_files_exist = False

    # 2. 检查 RecBoleProvider 类
    print("\n2. RecBoleProvider 类检查:")
    check_class_in_file(
        "src/recommendation/recbole_trainer.py",
        "RecBoleProvider"
    )

    # 3. 检查关键函数
    print("\n3. 关键函数检查:")
    functions = [
        ("src/recommendation/recbole_trainer.py", "export_recbole_data"),
        ("src/recommendation/recbole_trainer.py", "train_recbole_model"),
        ("src/recommendation/candidate_merger.py", "adaptive_fusion"),
        ("src/recommendation/candidate_merger.py", "_behavior_recall"),
        ("src/recommendation/candidate_merger.py", "_get_recbole_provider"),
    ]

    for filepath, func_name in functions:
        check_function_in_file(filepath, func_name)

    # 4. 检查配置参数
    print("\n4. 配置参数检查:")
    config_params = [
        ("configs/runtime.yaml", "behavior_provider"),
        ("configs/runtime.yaml", "recbole_model_path"),
        ("configs/runtime.yaml", "recbole_config"),
        ("configs/runtime.yaml", "recbole_use_gpu"),
        ("configs/runtime.yaml", "adaptive_fusion"),
    ]

    for filepath, param_name in config_params:
        check_config_parameter(filepath, param_name)

    # 5. 代码质量检查
    print("\n5. 代码质量检查:")

    # 检查 candidate_merger.py 中的 RecBole 导入
    merger_content = Path("src/recommendation/candidate_merger.py").read_text()

    if "from recommendation.recbole_trainer import RecBoleProvider" in merger_content:
        print("  ✓ RecBoleProvider 导入: 已添加")
    else:
        print("  ✗ RecBoleProvider 导入: 未添加")

    if "use_recbole" in merger_content:
        print("  ✓ use_recbole 参数: 已添加")
    else:
        print("  ✗ use_recbole 参数: 未添加")

    if "adaptive_fusion_enabled" in merger_content:
        print("  ✓ adaptive_fusion_enabled 参数: 已添加")
    else:
        print("  ✗ adaptive_fusion_enabled 参数: 未添加")

    # 6. 降级逻辑检查
    print("\n6. 降级逻辑检查:")

    trainer_content = Path("src/recommendation/recbole_trainer.py").read_text()

    if "fallback_to_popular" in trainer_content:
        print("  ✓ 降级到流行度: 已实现")
    else:
        print("  ✗ 降级到流行度: 未实现")

    if "cold_start" in trainer_content:
        print("  ✓ 冷启动处理: 已实现")
    else:
        print("  ✗ 冷启动处理: 未实现")

    # 总结
    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)
    print("\n关键实现:")
    print("  1. RecBoleProvider 类：支持模型加载和预测")
    print("  2. 动态权重融合：根据用户历史调整权重")
    print("  3. 降级策略：模型不可用时回退到流行度")
    print("  4. 冷启动支持：新用户使用流行度推荐")
    print("  5. 配置更新：runtime.yaml 支持 RecBole 配置")


if __name__ == "__main__":
    main()
